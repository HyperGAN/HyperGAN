#!/usr/bin/env python3
"""One supplied G/D rate pair, measured in a disposable native rollout.

Example (select the free device explicitly):
  PYTHONPATH=src python reports/joint_rate_probe.py CONFIG --g-lr 1e-4 \
      --d-lr 1e-4 --steps 32 --device cuda:1 --output /path/joint-rate.json

This is a research observation, not an autotune acceptance decision. It never
changes source configuration, prior learning rates or pretrained state, and
never saves a training checkpoint. The original seed and lazy schedule apply.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from hypergan.checkpoints import capture_rng, restore_rng, trainer_state
from hypergan.config import fingerprint, load_config
from hypergan.signal_structure import _hash
from hypergan.startup_dynamics import (
    _cpu_clone, _eligibility, _optimizer_motion, _protected, _restore, _same_state, _snapshot,
)
from hypergan.startup_response_probe import (
    UpdateObserver, _activation_scalars, _final_owned_affine, _image_response,
    _parameters, _registered_parameters, _values, displacement,
)
from hypergan.training import ReferenceTrainer, source_info
from hypergan.update_response import tensor_change


def _stats(value):
    report = _activation_scalars(value)
    if report['status'] != 'finite':
        return report
    value = value.detach().cpu().double()
    variance = value.var(dim=0, unbiased=False).mean()
    report['sample_diversity_rms'] = float(variance.sqrt())
    if value.ndim == 4:
        means = value.mean((-2, -1), keepdim=True)
        centered = value - means
        color_variance = means.var(dim=0, unbiased=False).mean()
        report.update(
            spatial_sample_diversity_rms=float(centered.var(dim=0, unbiased=False).mean().sqrt()),
            per_image_spatial_rms=float(centered.square().mean().sqrt()),
            sample_mean_color_diversity_rms=float(color_variance.sqrt()),
            mean_color_fraction_of_sample_variance=float(color_variance / variance) if float(variance) > 0 else None,
            mean_absolute_horizontal_difference=float(value.diff(dim=-1).abs().mean()) if value.shape[-1] > 1 else None,
            mean_absolute_vertical_difference=float(value.diff(dim=-2).abs().mean()) if value.shape[-2] > 1 else None,
        )
        if min(value.shape[-2:]) >= 4:
            pooled = torch.nn.functional.adaptive_avg_pool2d(value, (4, 4))
            report['pooled_4x4_sample_diversity_rms'] = float(pooled.var(dim=0, unbiased=False).mean().sqrt())
    return report


def _sample_prior(trainer, prior_rng):
    local = torch.Generator(device=trainer.device)
    local.set_state(prior_rng)
    with torch.no_grad():
        return _cpu_clone(trainer.prior.sample(trainer.config['training']['batch_size'], generator=local))


def _observe(trainer, batch, fixed_latent, prior_rng, measurement_rng, protected, protected_hash, budget, *, features=False):
    """Matched fixed/current-prior forwards with an exact caller-state fence."""
    saved = _snapshot(trainer)
    expected = _hash(_registered_parameters(trainer))
    affine, description = _final_owned_affine(trainer)
    outputs, observations = {}, {}
    try:
        current_latent = _sample_prior(trainer, prior_rng)
        if not _same_state(fixed_latent[1], current_latent[1]):
            raise ValueError('Replayed prior RNG did not reproduce the same particle IDs')
        for name, latent in (('fixed_latent', fixed_latent), ('evolving_prior', current_latent)):
            _restore(trainer, saved)
            restore_rng(measurement_rng)
            captured, handle = [], None
            try:
                if affine is not None:
                    def capture(module, args, output):
                        captured.append(output.detach().cpu().clone() if isinstance(output, torch.Tensor) else None)
                    handle = affine.register_forward_hook(capture)
                with torch.no_grad():
                    _, _, context = trainer._draw(batch, latent)
                budget['rollout_observation_generator_forwards'] += 1
                output = context['generated'].detach().cpu().clone()
                if _hash(_registered_parameters(trainer)) != expected:
                    raise ValueError('Rollout observation changed registered model parameters')
                if _hash(protected) != protected_hash:
                    raise ValueError('Rollout observation changed protected pretrained state')
                outputs[name] = output
                observation = {'output': _stats(output), 'final_affine': dict(description)}
                if affine is not None:
                    if len(captured) == 1 and captured[0] is not None:
                        observation['final_affine']['activation'] = _activation_scalars(
                            captured[0], pre_tanh=description['relationship_to_output'] == 'pre_tanh')
                    else:
                        observation['final_affine'].update(status='skipped', reason='Expected one tensor activation')
                observations[name] = observation
            finally:
                if handle is not None:
                    handle.remove()
        observations['prior_induced_output_difference_at_fixed_generator'] = _image_response(
            outputs['fixed_latent'], outputs['evolving_prior'])
        observations['latent_coordinate_change'] = tensor_change(fixed_latent[0], current_latent[0])
        if features:
            from frozen_feature_probe import measure_frozen_features
            _restore(trainer, saved)
            restore_rng(measurement_rng)
            observations['frozen_features_evolving_prior'] = measure_frozen_features(trainer, (batch, current_latent))
            budget['frozen_feature_observations'] += 1
            for name in ('generator_forwards', 'critic_forwards', 'pretrained_forwards', 'pretrained_images_including_gray_context'):
                budget['feature_' + name] += observations['frozen_features_evolving_prior'].get(name, 0)
        return observations, outputs
    finally:
        _restore(trainer, saved)
        if not _same_state(saved['state'], trainer_state(trainer, None)):
            raise ValueError('Rollout observation did not restore caller state')


def _directional(trainer, observer, budget):
    # Import only for the explicitly requested, separately budgeted research
    # audit. These routines do not select or apply a learning rate.
    from function_space_probe import measure_function_space, measure_gen_stencil
    final = _snapshot(trainer)
    results = []
    try:
        batches = [_cpu_clone(trainer.batch()) for _ in range(2)]
        prior_rng = trainer.streams['prior'].get_state().clone()
        for role in ('generator', 'discriminator'):
            anchor = observer.anchors.get(role)
            if anchor is None:
                results.append({'player': role, 'status': 'skipped', 'reason': 'Requested rollout did not reach the phase anchor'})
                continue
            _restore(trainer, anchor['snapshot'])
            local = torch.Generator(device=trainer.device)
            local.set_state(prior_rng)
            with torch.no_grad():
                banks = [(batch, _cpu_clone(trainer.prior.sample(
                    trainer.config['training']['batch_size'], generator=local))) for batch in batches]
            _restore(trainer, final)
            for index, bank in enumerate(banks):
                print(f'direction player={role} anchor={anchor["step"]} bank={index}', flush=True)
                functional = measure_function_space(trainer, anchor, role, bank, projections=4)
                stencil = measure_gen_stencil(trainer, anchor, role, bank)
                results.append({'player': role, 'step': anchor['step'], 'bank': index,
                                'function_space': functional, 'gen_stencil': stencil})
                budget['directional_player_bank_audits'] += 1
                for name in ('output_forwards', 'projection_backwards'):
                    budget['function_space_' + name] += functional.get(name, 0)
                for name in ('phase_loss_evaluations', 'player_gradient_evaluations'):
                    budget['gen_' + name] += stencil.get(name, 0)
        return results
    finally:
        _restore(trainer, final)


def run_probe(config_path, *, g_lr, d_lr, steps=32, device=None, direction=False, features=False):
    """Return a finite report for one pair; restore the disposable trainer."""
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError('Research rollout must contain 1 to 64 updates')
    if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in (g_lr, d_lr)):
        raise ValueError('Explicit G and D rates must be finite and positive')
    source = Path(config_path).expanduser().resolve()
    original_bytes = source.read_bytes()
    config = load_config(source)
    original_fingerprint = fingerprint(config)
    if steps > config['training']['steps']:
        raise ValueError('Requested rollout exceeds the original configured training horizon')
    if device is not None:
        config['training']['device'] = device
    started = time.monotonic()
    trainer = ReferenceTrainer(config)
    exclusion = _eligibility(trainer)
    if exclusion:
        raise ValueError('Disposable research rollout is unsupported: ' + exclusion)
    initial = _snapshot(trainer)
    protected = _protected(trainer)
    protected_hash = _hash(protected)
    budget = defaultdict(int)
    observer = UpdateObserver(trainer, _snapshot, protected, protected_hash, budget, capture_anchors=direction)
    observation_steps = sorted({0, steps} | {step for step in (1, 8, 16, 32) if step <= steps})
    report = {'schema_version': 1, 'kind': 'disposable-explicit-joint-rate-rollout',
              'purpose': 'Measurement of one supplied rate pair; no acceptance rule or automatic calibration',
              'source': source_info(), 'config': str(source), 'original_config_fingerprint': original_fingerprint,
              'effective_config_fingerprint': fingerprint(config), 'config_sha256': hashlib.sha256(original_bytes).hexdigest(),
              'seed': config['training']['seed'], 'device': str(trainer.device), 'requested_updates': steps,
              'g_lr': g_lr, 'd_lr': d_lr, 'prior_base_lrs': list(trainer.base_lrs[0][1:]),
              'original_base_lrs': _cpu_clone(trainer.base_lrs), 'gradient_penalty': _cpu_clone(config['gradient_penalty']),
              'schedule': 'Original training horizon, annealing, D-then-G draws and lazy penalty schedule; prior base rates unchanged',
              'observation_steps': observation_steps, 'observations': [], 'per_step': [],
              'interpretation': ['Fixed latent values isolate G drift; replayed prior RNG fixes IDs/noise while allowing learned coordinates to evolve.',
                                 'Observation bank starts at initial data/prior draw positions and may overlap training; it is not independent quality validation.',
                                 'Saturation, color variance and spatial statistics are descriptive, not sample quality or an acceptance decision.']}
    error = None
    try:
        trainer.base_lrs[0][0] = trainer.opt_g.param_groups[0]['lr'] = float(g_lr)
        trainer.base_lrs[1][0] = trainer.opt_d.param_groups[0]['lr'] = float(d_lr)
        rate_start = _snapshot(trainer)
        batch = _cpu_clone(trainer.batch())
        prior_rng = trainer.streams['prior'].get_state().clone()
        fixed_latent = _sample_prior(trainer, prior_rng)
        _restore(trainer, rate_start)
        measurement_rng = capture_rng()
        baseline_outputs = None
        context = torch.cuda.device(trainer.device) if trainer.device.type == 'cuda' else nullcontext()
        with context:
            for step in range(steps + 1):
                per_step_motion = None
                if step:
                    before = {role: _values(_parameters(trainer, role)) for role in ('generator', 'discriminator', 'prior')} if step in observation_steps else None
                    trainer._update_response_observer = observer
                    row, _ = trainer.update()
                    trainer._update_response_observer = None
                    budget['completed_native_training_updates'] += 1
                    if _hash(protected) != protected_hash:
                        raise ValueError('Native rollout changed protected pretrained state')
                    report['per_step'].append({key: row[key] for key in
                                              ('step', 'g_loss', 'd_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'lr_scale')})
                    report['per_step'][-1]['actual_lrs'] = [[group['lr'] for group in optimizer.param_groups]
                                                           for optimizer in (trainer.opt_g, trainer.opt_d)]
                    if before is not None:
                        per_step_motion = {role: displacement(_parameters(trainer, role), values)[1]
                                           for role, values in before.items()}
                    print(f'update {step}/{steps} g_loss={row["g_loss"]:.7g} d_loss={row["d_loss"]:.7g} penalty={row["gradient_penalty"]:.7g}', flush=True)
                if step in observation_steps:
                    measured, outputs = _observe(trainer, batch, fixed_latent, prior_rng, measurement_rng,
                                                 protected, protected_hash, budget, features=features)
                    if baseline_outputs is None:
                        baseline_outputs = outputs
                    measured['step'] = step
                    measured['change_from_initial'] = {name: _image_response(baseline_outputs[name], value)
                                                        for name, value in outputs.items()}
                    measured['cumulative_optimizer_motion'] = _optimizer_motion(trainer, initial)
                    for motion in measured['cumulative_optimizer_motion'].values():
                        motion['measurement'] = f'cumulative_{step}_update_parameter_displacement'
                    if per_step_motion is not None:
                        measured['last_update_optimizer_motion'] = per_step_motion
                    report['observations'].append(measured)
            report['native_first_and_eighth_update_response'] = observer.observations
            report['rollout_final_parameters_sha256'] = _hash(_registered_parameters(trainer))
            from function_space_probe import crossed_progress
            report['crossed_progress'] = crossed_progress(trainer, initial, _snapshot(trainer), (batch, fixed_latent))
            budget['crossed_progress_phase_loss_evaluations'] += report['crossed_progress']['phase_loss_evaluations']
            if direction:
                report['directional_probes'] = _directional(trainer, observer, budget)
    except BaseException as failure:
        error = failure
        report['failure'] = {'type': type(failure).__name__, 'message': str(failure)}
    finally:
        report['protected_before_sha256'] = protected_hash
        report['protected_after_sha256'] = _hash(protected)
        _restore(trainer, initial)
        if hasattr(trainer, '_update_response_observer'):
            del trainer._update_response_observer
        report['restored'] = _same_state(initial['state'], trainer_state(trainer, None))
        report['protected_after_restore_sha256'] = _hash(protected)
        report['source_config_unchanged'] = source.read_bytes() == original_bytes
        report['budget'] = dict(budget)
        report['elapsed_seconds'] = time.monotonic() - started
    if not report['restored'] or not report['source_config_unchanged'] or report['protected_before_sha256'] != report['protected_after_sha256']:
        raise ValueError('Research rollout failed its restoration or protected-state audit') from error
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--g-lr', type=float, required=True)
    parser.add_argument('--d-lr', type=float, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--device')
    parser.add_argument('--direction', action='store_true', help='Also run bounded function-space and symmetric-loss audits at G1/D8')
    parser.add_argument('--features', action='store_true', help='Observe frozen DINO feature distributions on the evolving-prior samples')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise ValueError('Research report destination already exists')
    report = run_probe(args.config, g_lr=args.g_lr, d_lr=args.d_lr, steps=args.steps,
                       device=args.device, direction=args.direction, features=args.features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if 'failure' in report:
        raise RuntimeError('Disposable rollout failed; partial report saved at ' + str(destination))
    print(str(destination), flush=True)


if __name__ == '__main__':
    main()

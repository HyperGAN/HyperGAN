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
from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import platform
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


DEFAULT_OBSERVATION_STEPS = (0, 1, 8, 16, 32, 64, 128, 256, 512)


def _observation_steps(steps, supplied=None):
    milestones = DEFAULT_OBSERVATION_STEPS if supplied is None else supplied
    if any(type(value) is not int or not 0 <= value <= 512 for value in milestones):
        raise ValueError('Observation milestones must be integers from 0 to 512')
    return sorted({0, steps} | {value for value in milestones if value <= steps})


def _identity_hash(value):
    """Content identity independent of tensor storage addresses and devices."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(('tensor:' + str(tensor.dtype) + ':' + str(tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update((type(item).__name__ + ':' + str(len(item))).encode())
            for child in item:
                visit(child)
        elif hasattr(item, 'tobytes') and hasattr(item, 'dtype'):
            digest.update(('array:' + str(item.dtype) + ':' + str(item.shape)).encode())
            digest.update(item.tobytes())
        else:
            digest.update((type(item).__name__ + ':' + json.dumps(item, allow_nan=False)).encode())
        digest.update(b';')
    visit(value)
    return digest.hexdigest()


def _atomic_report(destination, report):
    """Readers see a complete previous or next report, never a partial JSON."""
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent,
                                         prefix=destination.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _synchronize(trainer):
    if trainer.device.type == 'cuda':
        torch.cuda.synchronize(trainer.device)


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


def _stage_stats(value):
    """Compact stage measurements; axis zero must represent original images."""
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite generator stage activation')
    rms = float(value.square().mean().sqrt())
    diversity = float(value.var(dim=0, unbiased=False).mean().sqrt())
    return {'shape': list(value.shape), 'rms': rms,
            'sample_diversity_rms': diversity,
            'diversity_to_rms': diversity / rms if rms else None}


def _observe(trainer, batch, fixed_latent, prior_rng, measurement_rng, protected, protected_hash, budget, *, features=False, observe_modules=()):
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
            stage_values, stage_handles = {}, []
            try:
                modules = dict(trainer.graph.named_modules())
                for path in observe_modules:
                    def capture_stage(module, args, output, path=path):
                        if path in stage_values:
                            raise ValueError('Stage observation expected one invocation: ' + path)
                        stage_values[path] = _stage_stats(output)
                    stage_handles.append(modules[path].register_forward_hook(capture_stage))
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
                if observe_modules:
                    if set(stage_values) != set(observe_modules):
                        raise ValueError('Requested stage hooks did not all execute')
                    observation['stages'] = stage_values
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
                for stage_handle in stage_handles:
                    stage_handle.remove()
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


def run_probe(config_path, *, g_lr, d_lr, steps=32, device=None, direction=False, features=False,
              crossed=False, observe_steps=None, progress_path=None, prepare=None, observe_modules=()):
    """Measure one pair and restore state; optional prepare is a context factory.

    Its context spans the rollout and closes before restoring the original
    optimizer layout. Progress snapshots are observations, never checkpoints.
    """
    if type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError('Research rollout must contain 1 to 512 updates')
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
    observation_steps = _observation_steps(steps, observe_steps)
    destination = Path(progress_path).expanduser().resolve() if progress_path is not None else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            stream.write(json.dumps({'status': 'initializing', 'completed_updates': 0}) + '\n')
    started = time.monotonic()
    trainer = ReferenceTrainer(config)
    exclusion = _eligibility(trainer)
    if exclusion:
        raise ValueError('Disposable research rollout is unsupported: ' + exclusion)
    initial = _snapshot(trainer)
    protected = _protected(trainer)
    protected_hash = _hash(protected)
    budget = defaultdict(int)
    observer = UpdateObserver(trainer, _snapshot, protected, protected_hash, budget) if direction else None
    timings = {'training_update_seconds': 0., 'diagnostic_seconds': 0.,
               'observer_seconds_inside_update': 0., 'preparation_seconds': 0.}

    def timed_observer(*args, **kwargs):
        _synchronize(trainer)
        began = time.monotonic()
        try:
            return observer(*args, **kwargs)
        finally:
            _synchronize(trainer)
            elapsed = time.monotonic() - began
            timings['observer_seconds_inside_update'] += elapsed
            timings['diagnostic_seconds'] += elapsed

    report = {'schema_version': 2, 'status': 'running', 'completed_updates': 0, 'kind': 'disposable-explicit-joint-rate-rollout',
              'purpose': 'Measurement of one supplied rate pair; no acceptance rule or automatic calibration',
              'source': source_info(), 'config': str(source),
              'initial_parameters_sha256': _hash(_registered_parameters(trainer)), 'original_config_fingerprint': original_fingerprint,
              'effective_config_fingerprint': fingerprint(config), 'config_sha256': hashlib.sha256(original_bytes).hexdigest(),
              'seed': config['training']['seed'], 'device': str(trainer.device), 'requested_updates': steps,
              'g_lr': g_lr, 'd_lr': d_lr, 'prior_base_lrs': list(trainer.base_lrs[0][1:]),
              'original_base_lrs': _cpu_clone(trainer.base_lrs), 'gradient_penalty': _cpu_clone(config['gradient_penalty']),
              'schedule': 'Original training horizon, annealing, D-then-G draws and lazy penalty schedule; prior base rates unchanged',
              'observation_steps': observation_steps, 'observations': [], 'per_step': [],
              'diagnostics_enabled': {'direction': direction, 'features': features, 'crossed': crossed},
              'evaluation': {'protocol': 'joint-rate-fixed-bank', 'protocol_version': 2,
                             'observation_steps': observation_steps, 'requested_updates': steps,
                             'configured_training_horizon': config['training']['steps'],
                             'batch_size': config['training']['batch_size'],
                             'execution_backend': 'native', 'training_backend': _cpu_clone(config['training']['backend']),
                             'torch_version': str(torch.__version__),
                             'device_type': trainer.device.type,
                             'device_hardware': (torch.cuda.get_device_name(trainer.device) if trainer.device.type == 'cuda'
                                                 else platform.processor() or platform.machine()),
                             'diagnostics': {'direction': direction, 'features': features, 'crossed': crossed,
                                             'observe_modules': list(observe_modules)}},
              'timing_definition': 'Synchronized wall time; training updates exclude observer callbacks. Diagnostics include observations and requested probes; setup, state audits, JSON writes and final restoration remain in elapsed time only.',
              'interpretation': ['Fixed latent values isolate G drift; replayed prior RNG fixes IDs/noise while allowing learned coordinates to evolve.',
                                 'Observation bank starts at initial data/prior draw positions and may overlap training; it is not independent quality validation.',
                                 'Protocol v2 disables implicit first/eighth response and crossed probes unless requested; historical total costs are not directly comparable.',
                                 'Saturation, color variance and spatial statistics are descriptive, not sample quality or an acceptance decision.']}
    def publish():
        report['budget'] = dict(budget)
        report['timings'] = dict(timings)
        report['elapsed_seconds'] = time.monotonic() - started
        if destination is not None:
            _atomic_report(destination, report)

    error = None
    preparation = ExitStack()
    try:
        trainer.base_lrs[0][0] = trainer.opt_g.param_groups[0]['lr'] = float(g_lr)
        trainer.base_lrs[1][0] = trainer.opt_d.param_groups[0]['lr'] = float(d_lr)
        if prepare is not None:
            _synchronize(trainer)
            preparation_started = time.monotonic()
            prior_before_preparation = _cpu_clone(trainer.prior.state_dict())
            try:
                report['proposal'] = preparation.enter_context(prepare(trainer))
                json.dumps(report['proposal'], allow_nan=False)
                if not _same_state(prior_before_preparation, trainer.prior.state_dict()):
                    raise ValueError('Research preparation changed prior state')
                if _hash(protected) != protected_hash:
                    raise ValueError('Research preparation changed protected pretrained state')
            finally:
                _synchronize(trainer)
                timings['preparation_seconds'] += time.monotonic() - preparation_started
        report['prepared_parameters_sha256'] = _hash(_registered_parameters(trainer))
        rate_start = _snapshot(trainer)
        batch = _cpu_clone(trainer.batch())
        report['real_bank_output_stats'] = (_stats(batch['real'])
            if isinstance(batch, Mapping) and isinstance(batch.get('real'), torch.Tensor)
            else {'status': 'skipped', 'reason': 'Batch has no real tensor'})
        prior_rng = trainer.streams['prior'].get_state().clone()
        fixed_latent = _sample_prior(trainer, prior_rng)
        _restore(trainer, rate_start)
        measurement_rng = capture_rng()
        report['evaluation'].update(bank_sha256=_identity_hash((batch, fixed_latent)),
                                    measurement_rng_sha256=_identity_hash(measurement_rng),
                                    prior_rng_sha256=_identity_hash(prior_rng))
        baseline_outputs = None
        context = torch.cuda.device(trainer.device) if trainer.device.type == 'cuda' else nullcontext()
        with context:
            for step in range(steps + 1):
                per_step_motion = None
                if step:
                    before = {role: _values(_parameters(trainer, role)) for role in ('generator', 'discriminator', 'prior')} if step in observation_steps else None
                    trainer._update_response_observer = timed_observer if observer is not None else None
                    observer_before = timings['observer_seconds_inside_update']
                    _synchronize(trainer)
                    update_started = time.monotonic()
                    try:
                        row, _ = trainer.update()
                    finally:
                        _synchronize(trainer)
                        timings['training_update_seconds'] += max(0., time.monotonic() - update_started -
                            (timings['observer_seconds_inside_update'] - observer_before))
                        trainer._update_response_observer = None
                    report['completed_updates'] = step
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
                    _synchronize(trainer)
                    diagnostic_started = time.monotonic()
                    measured, outputs = _observe(trainer, batch, fixed_latent, prior_rng, measurement_rng,
                                                 protected, protected_hash, budget, features=features,
                                                 observe_modules=observe_modules)
                    if baseline_outputs is None:
                        baseline_outputs = outputs
                    measured['step'] = step
                    measured['change_from_initial'] = {name: _image_response(baseline_outputs[name], value)
                                                        for name, value in outputs.items()}
                    measured['cumulative_optimizer_motion'] = _optimizer_motion(trainer, rate_start)
                    for motion in measured['cumulative_optimizer_motion'].values():
                        motion['measurement'] = f'cumulative_{step}_update_parameter_displacement'
                    if per_step_motion is not None:
                        measured['last_update_optimizer_motion'] = per_step_motion
                    report['observations'].append(measured)
                    _synchronize(trainer)
                    timings['diagnostic_seconds'] += time.monotonic() - diagnostic_started
                    measured['timing_at_observation'] = {**timings, 'elapsed_seconds': time.monotonic() - started}
                    publish()
            if observer is not None:
                report['native_first_and_eighth_update_response'] = observer.observations
            report['rollout_final_parameters_sha256'] = _hash(_registered_parameters(trainer))
            if crossed or direction:
                _synchronize(trainer)
                diagnostic_started = time.monotonic()
                if crossed:
                    from function_space_probe import crossed_progress
                    report['crossed_progress'] = crossed_progress(trainer, rate_start, _snapshot(trainer), (batch, fixed_latent))
                    budget['crossed_progress_phase_loss_evaluations'] += report['crossed_progress']['phase_loss_evaluations']
                if direction:
                    report['directional_probes'] = _directional(trainer, observer, budget)
                _synchronize(trainer)
                timings['diagnostic_seconds'] += time.monotonic() - diagnostic_started
    except BaseException as failure:
        error = failure
        report['failure'] = {'type': type(failure).__name__, 'message': str(failure)}
    finally:
        report['protected_before_sha256'] = protected_hash
        report['protected_after_sha256'] = _hash(protected)
        try:
            preparation.close()
        except BaseException as cleanup_failure:
            error = cleanup_failure
            report['cleanup_failure'] = {'type': type(cleanup_failure).__name__, 'message': str(cleanup_failure)}
            report.setdefault('failure', report['cleanup_failure'])
        finally:
            _restore(trainer, initial)
        if hasattr(trainer, '_update_response_observer'):
            del trainer._update_response_observer
        report['restored'] = _same_state(initial['state'], trainer_state(trainer, None))
        report['protected_after_restore_sha256'] = _hash(protected)
        report['source_config_unchanged'] = source.read_bytes() == original_bytes
        report['budget'] = dict(budget)
        report['elapsed_seconds'] = time.monotonic() - started
        report['status'] = 'failed' if error is not None else 'complete'
        report['timings'] = dict(timings)
    if not report['restored'] or not report['source_config_unchanged'] or report['protected_before_sha256'] != report['protected_after_sha256']:
        report['status'] = 'failed'
        report['audit_failure'] = 'Research rollout failed its restoration or protected-state audit'
        publish()
        raise ValueError(report['audit_failure']) from error
    publish()
    json.dumps(report, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--g-lr', type=float, required=True)
    parser.add_argument('--d-lr', type=float, required=True)
    parser.add_argument('--steps', type=int, default=32, help='Disposable updates, from 1 to 512; original annealing horizon is preserved')
    parser.add_argument('--observe-steps', type=lambda value: [int(item) for item in value.split(',')],
                        help='Comma-separated milestones (default: 0,1,8,16,32,64,128,256,512); initial/final always included')
    parser.add_argument('--crossed', action='store_true', help='Also evaluate four final crossed G/D objectives')
    parser.add_argument('--device')
    parser.add_argument('--direction', action='store_true', help='Also run bounded function-space and symmetric-loss audits at G1/D8')
    parser.add_argument('--features', action='store_true', help='Observe frozen DINO feature distributions on the evolving-prior samples')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise ValueError('Research report destination already exists')
    report = run_probe(args.config, g_lr=args.g_lr, d_lr=args.d_lr, steps=args.steps,
                       device=args.device, direction=args.direction, features=args.features, crossed=args.crossed,
                       observe_steps=args.observe_steps, progress_path=destination)
    if 'failure' in report:
        raise RuntimeError('Disposable rollout failed; partial report saved at ' + str(destination))
    print(str(destination), flush=True)


if __name__ == '__main__':
    main()

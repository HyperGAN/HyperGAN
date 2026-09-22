#!/usr/bin/env python3
"""Read-only matched-input online checkpoint audit; never resumes training.

Example (after the diagnostic run stops; pick the free GPU in the environment):
  PYTHONPATH=src python reports/checkpoint_signal_probe.py RUN_DIR \
      --steps 0 20 100 --output-dir /mnt/ml7tb/hypergan-signal-research/probes

Primary comparison fixes real images, particle IDs and Gaussian noise while
allowing the learned prior to evolve. The fixed-tensor control also fixes latent
coordinates, isolating prior drift from G/D drift. Neither identifies causal
training improvement or represents the next post-D-update training gradient.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from hypergan.checkpoints import capture_rng, read_checkpoint, restore_rng
from hypergan.config import resolve_config
from hypergan.initialization_tuning import _inventory, _structural
from hypergan.signal_diagnostic import _digest_state, _probe
from hypergan.training import ReferenceTrainer


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps((list(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoints(run_dir, steps):
    found = {}
    for path in (run_dir / 'checkpoints').glob('*/manifest.json'):
        if path.parent.name.startswith('.'):
            continue
        info = json.loads(path.read_text())
        if info['step'] in steps:
            previous = found.get(info['step'])
            if previous is None or path.stat().st_mtime_ns > previous.stat().st_mtime_ns:
                found[info['step']] = path
    missing = set(steps) - found.keys()
    if missing:
        raise ValueError('Missing completed checkpoints for steps ' + repr(sorted(missing)))
    return [(step, found[step].parent) for step in sorted(set(steps))]


def load_online(trainer, state):
    """Load online tensors/modes only; EMA and optimizer states are not used."""
    for name in ('graph', 'prior'):
        root = getattr(trainer, name)
        root.load_state_dict(state[name], strict=True)
        with torch.no_grad():
            for key, buffer in root.named_buffers():
                buffer.copy_(state['buffers'][name][key])
        for key, parameter in root.named_parameters():
            parameter.requires_grad_(state['trainable'][name][key])
        for key, module in root.named_modules():
            module.training = state['modes'][name][key]
    trainer.step = state['step']


def bank_from_anchor(trainer, state):
    if state['data'] is not None:
        trainer.data.load_state_dict(state['data'])
    trainer.streams['data'].set_state(state['streams']['data'])
    restore_rng(state['rng'])
    batch = trainer.batch()
    prior = trainer.prior
    kind = trainer.config['prior']['kind']
    generator = torch.Generator(device=trainer.device)
    generator.set_state(state['streams']['prior'])
    count = len(batch['real'])
    # Native MoG sample consumes uniform IDs followed by a Gaussian draw.
    if kind == 'mog':
        ids = prior.sample_indices(count, generator=generator)
        means = prior.means()[ids]
        eps = (torch.randn(means.shape, dtype=means.dtype, device=means.device, generator=generator)
               if prior._noise_enabled else torch.zeros_like(means))
        latent = prior(ids, eps=eps)
    elif kind == 'particles':
        ids = prior.sample_indices(count, generator=generator)
        eps, latent = None, prior(ids)
    elif kind == 'gaussian':
        latent, ids = prior.sample(count, generator=generator)
        eps = None
    else:
        raise ValueError('Research probe supports native mog, particles or gaussian priors only')
    return batch, {'kind': kind, 'ids': ids, 'eps': eps, 'initial_latent': latent.detach().clone()}, capture_rng()


def draw(trainer, bank, mode):
    ids = bank['ids']
    if mode == 'fixed_latent_tensor' or bank['kind'] == 'gaussian':
        return bank['initial_latent'].detach().clone(), ids
    if bank['kind'] == 'mog':
        return trainer.prior(ids, eps=bank['eps']), ids
    return trainer.prior(ids), ids


def frozen_digest(trainer):
    # Maximal parameter-owning frozen subtrees include native pretrained modules
    # and the legacy CIFAR ResNet feature container, without hard-coded paths.
    roots = []
    for name, module in trainer.graph.named_modules():
        parameters = list(module.parameters())
        if parameters and all(not p.requires_grad for p in parameters):
            if not any(name == parent or name.startswith(parent + '.') for parent, _ in roots):
                roots.append((name, module))
    return _digest_state(roots)


def output_statistics(module, args, output, destination):
    value = output.detach().float()
    per_sample = value.flatten(1)
    if value.ndim == 4:
        coarse = torch.nn.functional.adaptive_avg_pool2d(value, (4, 4))
        destination['coarse_4x4_sample_diversity_rms'] = float(coarse.var(dim=0, unbiased=False).mean().sqrt())
    destination.update(output_rms=float(value.square().mean().sqrt()),
                       sample_diversity_rms=float(value.var(dim=0, unbiased=False).mean().sqrt()),
                       mean_within_sample_std=float(per_sample.std(dim=1, unbiased=False).mean()),
                       min_within_sample_std=float(per_sample.std(dim=1, unbiased=False).min()),
                       absolute_output_above_0_99_fraction=float((value.abs() > .99).float().mean()))


def evaluate(trainer, batch, bank, rng, mode, branches=False):
    state_before = _digest_state((('graph', trainer.graph), ('prior', trainer.prior)))
    buffers = [(p, p.detach().clone()) for root in (trainer.graph, trainer.prior) for p in root.buffers()]
    layers, _ = _inventory(trainer)
    frozen_before = frozen_digest(trainer)
    try:
        restore_rng(rng)
        structural = _structural(trainer, batch, draw(trainer, bank, mode), layers)
        if frozen_digest(trainer) != frozen_before:
            raise ValueError('Structural probe changed frozen state before restoration')
    finally:
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)
    output = {}
    handle = trainer.graph.models['generator'].register_forward_hook(
        lambda module, args, generated: output_statistics(module, args, generated, output))
    try:
        restore_rng(rng)
        signal = _probe(trainer, 'adversarial', batch=batch, latent_draw=draw(trainer, bank, mode))
    finally:
        handle.remove()
    branch_report = None
    if branches:
        from signal_branch_probe import probe_branches
        restore_rng(rng)
        branch_report = probe_branches(trainer, batch, draw(trainer, bank, mode))
    state_after = _digest_state((('graph', trainer.graph), ('prior', trainer.prior)))
    if state_after != state_before:
        raise ValueError('Research probe changed online model state')
    latent, _ = draw(trainer, bank, mode)
    drift = latent.detach() - bank['initial_latent']
    prior_statistics = {'latent_batch_std_rms': float(latent.detach().float().var(dim=0, unbiased=False).mean().sqrt()),
                        'sigma': float(trainer.prior.sigma) if hasattr(trainer.prior, 'sigma') else None}
    if hasattr(trainer.prior, 'z'):
        table = trainer.prior.z.detach().float()
        prior_statistics['raw_table_std_rms'] = float(table.var(dim=0, unbiased=False).mean().sqrt())
    if callable(getattr(trainer.prior, 'means', None)):
        means = trainer.prior.means().detach().float()
        prior_statistics['read_means_std_rms'] = float(means.var(dim=0, unbiased=False).mean().sqrt())
    return {'input_mode': mode, 'signal': signal, 'transmission': structural,
            'generated_statistics': output, 'branch_attribution': branch_report, 'prior_statistics': prior_statistics,
            'latent_sha256': tensor_sha256(latent),
            'latent_rms_change_from_anchor': float(drift.float().square().mean().sqrt()),
            'online_state_unchanged': True, 'online_state_sha256': state_before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--steps', nargs='+', type=int, default=[0, 20, 100])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', help='Override execution device, keeping the configured batch size')
    parser.add_argument('--latent-mode', choices=('both', 'fixed_ids_noise', 'fixed_latent_tensor'), default='both')
    parser.add_argument('--branches', action='store_true', help='Also run the optional multidepth-D branch research probe')
    args = parser.parse_args()
    selected = checkpoints(args.run_dir, args.steps)
    if selected[0][0] != 0:
        raise ValueError('Include step 0 to anchor the same startup input bank')
    _, initial_info, state = read_checkpoint(args.run_dir, selected[0][1])
    config = copy.deepcopy(initial_info['config'])
    if args.device:
        if torch.device(args.device).type != torch.device(config['training']['device']).type:
            raise ValueError('Matched RNG replay requires the same CPU/CUDA backend as the checkpoint')
        config['training']['device'] = args.device
    trainer = ReferenceTrainer(resolve_config(config))
    del trainer.ema_graph, trainer.ema_prior
    load_online(trainer, state)
    batch, bank, rng = bank_from_anchor(trainer, state)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # A small exact-input artifact, never a modified training checkpoint.
    portable_batch = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    portable_bank = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in bank.items()}
    torch.save({'batch': portable_batch, 'bank': portable_bank, 'rng': rng,
                'anchor_checkpoint_sha256': initial_info['state_sha256']}, args.output_dir / 'matched-input-bank.pt')
    del state
    modes = ['fixed_ids_noise', 'fixed_latent_tensor'] if args.latent_mode == 'both' else [args.latent_mode]
    rows, initial_frozen = [], frozen_digest(trainer)
    for step, checkpoint in selected:
        started = time.monotonic()
        _, info, state = read_checkpoint(args.run_dir, checkpoint)
        if info['run_id'] != initial_info['run_id'] or info['config_sha256'] != initial_info['config_sha256']:
            raise ValueError('Checkpoint run/config identity differs from anchor')
        load_online(trainer, state)
        del state
        if frozen_digest(trainer) != initial_frozen:
            raise ValueError('Frozen model tensors differ across saved checkpoints')
        for mode in modes:
            print(f'Probing completed step {step}: {mode}', flush=True)
            report = evaluate(trainer, batch, bank, rng, mode, args.branches)
            report.update(step=step, phase=('saved-online-initialization-before-updates' if step == 0
                                           else 'saved-online-models-after-complete-training-update'),
                          checkpoint=str(checkpoint), checkpoint_sha256=info['state_sha256'],
                          pretrained_and_frozen_sha256=initial_frozen,
                          batch_real_sha256=tensor_sha256(batch['real']))
            destination = args.output_dir / f'step-{step:08d}-{mode}.json'
            destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            rows.append({'step': step, 'input_mode': mode, 'report': str(destination),
                         'summary': report['signal']['summary'],
                         'transmission_score': report['transmission']['score'],
                         'generated_statistics': report['generated_statistics'],
                         'latent_rms_change_from_anchor': report['latent_rms_change_from_anchor'],
                         'prior_statistics': report['prior_statistics']})
        print(f'Completed step {step} in {time.monotonic() - started:.1f}s', flush=True)
    summary = {'run_dir': str(args.run_dir.resolve()), 'configured_batch_size': config['training']['batch_size'],
               'device': str(trainer.device), 'rows': rows,
               'interpretation': [
                   'Online saved G and D, not EMA, and no optimizer updates; all checkpoints use the same real batch.',
                   'fixed_ids_noise tracks the same learned particles and Gaussian epsilon; latent tensors may evolve.',
                   'fixed_latent_tensor holds startup latent coordinates fixed; prior gradients are intentionally disconnected in this control.',
                   'Both G and D evolve across checkpoints; changes are not attributable to either network alone.',
                   'Saved post-update probes differ from the next training G signal after the next D update.',
                   'The startup bank may overlap calibration or training inputs; it is a drift control, not an independent quality evaluation.',
                   'One matched bank and startup gradient statistics do not establish image quality or long-term stability.',
                   'Exact RNG replay requires the same CPU/CUDA backend and visible CUDA device inventory as the saved run.',
                   'Existing checkpoint directories are retained by the writer; this script never writes into the run.']}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(str(args.output_dir / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()

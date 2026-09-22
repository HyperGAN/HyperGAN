#!/usr/bin/env python3
"""One first-generator grouped probe for the 64px startup contract.

Measures the actual source-rate Adam step. Does not apply a learning rate,
save a checkpoint, or change the training configuration.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from hypergan.checkpoints import trainer_state
from hypergan.config import load_config
from hypergan.signal_structure import _hash, _storage
from hypergan.startup_dynamics import _cpu_clone, _eligibility, _protected, _restore, _same_state, _snapshot
from hypergan.startup_response_probe import UpdateObserver, _copy, _parameters, _registered_parameters, phase_loss
from hypergan.training import ReferenceTrainer, source_info
from joint_rate_probe import _identity_hash, _sample_prior


GROUP_A = (
    'graph.models.generator.network.nodes.n_stage8_block1_ffn.down.weight',
    'graph.models.generator.network.nodes.n_stage8_block0_ffn.down.weight',
    'graph.models.generator.network.nodes.n_stage16_block0_ffn.down.weight',
    'graph.models.generator.network.nodes.n_stage16_block1_ffn.down.weight',
)
_POINTS = {
    'origin': (0., 0.), 'A_minus': (-1., 0.), 'A_plus': (1., 0.),
    'B_minus': (0., -1.), 'B_plus': (0., 1.), 'full_step': (1., 1.),
    'opposite_mix': (1., -1.),
}
_OUTPUT_POINTS = ('origin', 'A_plus', 'B_plus', 'full_step')


def _decision_module():
    path = Path(__file__).resolve().parents[1] / 'research/startup_tuning/algorithms/grouped_first_g.py'
    spec = importlib.util.spec_from_file_location('grouped_first_g', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adam_group_record(pairs, *, lr, eps, step, weight_decay, amsgrad):
    """Compare one group's recorded step with -lr * g / (|g| + eps)."""
    squared_delta = squared_residual = mass = contaminated = 0.
    elements = 0
    finite = True
    for gradient, delta in pairs:
        gradient = gradient.detach().cpu().double().reshape(-1)
        delta = delta.detach().cpu().double().reshape(-1)
        if gradient.shape != delta.shape or not bool(torch.isfinite(gradient).all() and torch.isfinite(delta).all()):
            finite = False
            break
        predicted = -lr * gradient / (gradient.abs() + eps)
        residual = delta - predicted
        squared_delta += float(delta.square().sum())
        squared_residual += float(residual.square().sum())
        slope = gradient.abs() * delta.abs()
        mass += float(slope.sum())
        contaminated += float(slope[gradient.abs() <= 1000 * eps].sum())
        elements += int(delta.numel())
    relative = fraction = None
    if finite and squared_delta > 0 and mass > 0 and elements:
        relative = math.sqrt(squared_residual / squared_delta)
        fraction = contaminated / mass
        finite = math.isfinite(relative) and math.isfinite(fraction)
    return {'relative_rms': relative if finite else None,
            'contaminated_slope_fraction': fraction if finite else None,
            'elements': elements, 'step': step, 'weight_decay': weight_decay,
            'amsgrad': amsgrad, 'eps': eps, 'lr': lr,
            'status': 'measured' if finite else 'nonfinite'}


def _owned_names(trainer):
    by_storage = {}
    for name, parameter in _registered_parameters(trainer):
        by_storage.setdefault(_storage(parameter), []).append(name)
    names = []
    for parameter in _parameters(trainer, 'generator'):
        matched = by_storage.get(_storage(parameter), [])
        if len(matched) != 1:
            return None, 'alias'
        names.append(matched[0])
    if len(set(names)) != len(names):
        return None, 'duplicate'
    if any(path not in names for path in GROUP_A):
        return None, 'missing'
    if len(names) == len(GROUP_A):
        return None, 'empty_complement'
    return names, None


def _draw_banks(trainer):
    initial = _snapshot(trainer)
    monitor_batch = _cpu_clone(trainer.batch())
    prior_state = trainer.streams['prior'].get_state().clone()
    monitor_latent = _sample_prior(trainer, prior_state)
    local = torch.Generator(device=trainer.streams['prior'].device)
    local.set_state(prior_state)
    replay = _cpu_clone(trainer.prior.sample(trainer.config['training']['batch_size'], generator=local))
    if _identity_hash(replay) != _identity_hash(monitor_latent):
        raise RuntimeError('Advancing the prior generator did not replay the monitor latent')
    fitting = []
    for _ in range(2):
        fitting.append((_cpu_clone(trainer.batch()), _cpu_clone(trainer.prior.sample(
            trainer.config['training']['batch_size'], generator=local))))
    _restore(trainer, initial)
    return (monitor_batch, monitor_latent), fitting


def _capture_loss(trainer, anchor, names, factors, bank):
    _restore(trainer, anchor['snapshot'])
    owned = _parameters(trainer, 'generator')
    group_a = set(GROUP_A)
    scale = {name: factors[0] if name in group_a else factors[1] for name in names}
    _copy(owned, [before + scale[name] * delta for name, before, delta in zip(names, anchor['before'], anchor['delta'])])
    captured = []
    original = trainer._draw

    def draw(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result[2]['generated'].detach().to(device='cpu', dtype=torch.float64).clone())
        return result

    trainer._draw = draw
    try:
        loss = phase_loss(trainer, 'generator', *bank, step=anchor['step'])
        slopes = None
        if factors == (0., 0.):
            slopes = torch.autograd.grad(loss, owned, allow_unused=True)
        value = float(loss.detach())
        del loss
        if len(captured) != 1:
            raise RuntimeError('Expected one generated output inside the phase loss')
        return value, captured[0], slopes
    finally:
        trainer._draw = original


def _group_slope(names, gradients, deltas, group):
    selected = set(group)
    total = 0.
    for name, gradient, delta in zip(names, gradients, deltas):
        if name not in selected:
            continue
        if gradient is None:
            return None, True
        product = gradient.detach().cpu().double() * delta.double()
        if not bool(torch.isfinite(product).all()):
            return None, True
        total += float(product.sum())
    if not math.isfinite(total):
        return None, True
    return total, False


def _output_summary(images):
    origin, alone_a, alone_b, both = (images[name] for name in _OUTPUT_POINTS)
    change_a, change_b, change_both = alone_a - origin, alone_b - origin, both - origin
    residual = change_both - change_a - change_b

    def rms(value):
        return float(value.square().mean().sqrt())

    def ms(value):
        return float(value.square().mean())

    return {'rms_dA': rms(change_a), 'rms_dB': rms(change_b), 'rms_dAB': rms(change_both),
            'rms_residual': rms(residual),
            'ms_cross': ms(change_both) - ms(change_a) - ms(change_b)}


def _adam_from_update(trainer, names, anchor):
    optimizer = trainer.opt_g
    groups = {id(parameter): group for group in optimizer.param_groups for parameter in group['params']}
    owned = list(zip(names, _parameters(trainer, 'generator'), anchor['delta']))
    records = {}
    for label, selected in (('A', set(GROUP_A)), ('B', set(names) - set(GROUP_A))):
        pairs, meta = [], []
        unsupported = False
        for name, parameter, delta in owned:
            if name not in selected:
                continue
            group, state = groups.get(id(parameter)), optimizer.state.get(parameter)
            if (group is None or not state or 'exp_avg' not in state or 'step' not in state
                    or state['exp_avg'].shape != parameter.shape):
                unsupported = True
                break
            step = float(state['step'])
            gradient = state['exp_avg'].detach().cpu().double() / (1. - float(group['betas'][0]))
            pairs.append((gradient, delta))
            meta.append((step, float(group.get('weight_decay', 0.)), bool(group.get('amsgrad', False)), float(group['eps'])))
        if unsupported or not pairs or any(item != meta[0] for item in meta):
            records[label] = {'status': 'unsupported'}
            continue
        step, decay, amsgrad, eps = meta[0]
        whole = int(step) if float(step).is_integer() else step
        records[label] = adam_group_record(pairs, lr=2e-4, eps=eps, step=whole, weight_decay=decay, amsgrad=amsgrad)
    return records


def _write(destination, report, started):
    report['elapsed_seconds'] = time.monotonic() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def run_probe(config_path, *, device, output):
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    source = Path(config_path).expanduser().resolve()
    original = source.read_bytes()
    config = load_config(source)
    learning_rate = config['optimizer']['lr']
    discriminator_rate = learning_rate * config['optimizer']['d_lr_mult']
    report = {'schema_version': 1, 'kind': 'grouped-first-g-probe', 'status': 'running',
              'source': source_info(), 'config': str(source), 'group_a': list(GROUP_A),
              'rates': {'g_lr': learning_rate, 'd_lr': discriminator_rate},
              'budget': {'native_updates': None, 'loss_evaluations': 0,
                         'projection_backwards': 0, 'output_forwards': 0}}
    if learning_rate != 2e-4 or discriminator_rate != 2e-4:
        report.update(status='failed', failure={'stage': 'rates', 'message': 'Contract requires source rates 2e-4'})
        _write(destination, report, started)
        raise ValueError(report['failure']['message'])
    if device is not None:
        config['training']['device'] = device
    trainer = ReferenceTrainer(config)
    exclusion = _eligibility(trainer)
    if exclusion:
        raise ValueError('Grouped probe is unsupported: ' + exclusion)
    initial = _snapshot(trainer)
    protected = _protected(trainer)
    protected_hash = _hash(protected)
    try:
        names, identity = _owned_names(trainer)
        if identity:
            report.update(status='failed', failure={'stage': 'group_identity', 'message': identity})
            return report
        report['group_b'] = [name for name in names if name not in GROUP_A]
        monitor, fitting = _draw_banks(trainer)
        hashes = {'monitor': _identity_hash(monitor),
                  'fitting_0': _identity_hash(fitting[0]),
                  'fitting_1': _identity_hash(fitting[1])}
        report['bank_hashes'] = hashes
        if hashes['monitor'] in (hashes['fitting_0'], hashes['fitting_1']):
            report.update(status='failed', failure={'stage': 'bank_identity',
                                                    'message': 'A fitting bank matches the monitor bank'})
            return report
        trainer.base_lrs[0][0] = trainer.opt_g.param_groups[0]['lr'] = 2e-4
        trainer.base_lrs[1][0] = trainer.opt_d.param_groups[0]['lr'] = 2e-4
        observer = UpdateObserver(trainer, _snapshot, protected, protected_hash, defaultdict(int))
        trainer._update_response_observer = observer
        with torch.cuda.device(trainer.device) if trainer.device.type == 'cuda' else nullcontext():
            trainer.update()
        trainer._update_response_observer = None
        report['budget']['native_updates'] = 1
        anchor = observer.anchors.get('generator')
        if anchor is None or anchor.get('step') != 1 or len(anchor['delta']) != len(names):
            raise RuntimeError('The first generator anchor was not captured')
        if _hash(protected) != protected_hash:
            raise RuntimeError('The anchor update changed protected state')
        report['adam'] = _adam_from_update(trainer, names, anchor)
        from function_space_probe import measure_function_space
        banks = []
        for batch, latent in fitting:
            measured = measure_function_space(trainer, anchor, 'generator', (batch, latent), projections=4)
            report['budget']['projection_backwards'] += measured['projection_backwards']
            report['budget']['output_forwards'] += measured['output_forwards']
            banks.append({'parameters': [{'path': item['path'], 'projections': item['projections']}
                                         for item in measured['parameters']]})
        report['banks'] = banks
        if not _decision_module().stencil_allowed(report):
            report['stencil_ran'] = False
            report['status'] = 'complete'
            return report
        for bank, (batch, latent) in zip(banks, fitting):
            losses, images = {}, {}
            origin_slopes = None
            for name, factors in _POINTS.items():
                value, image, slopes = _capture_loss(trainer, anchor, names, factors, (batch, latent))
                report['budget']['loss_evaluations'] += 1
                losses[name] = value
                if name in _OUTPUT_POINTS:
                    images[name] = image
                if factors == (0., 0.):
                    origin_slopes = slopes
                del slopes
            slope_a, missing_a = _group_slope(names, origin_slopes, anchor['delta'], GROUP_A)
            slope_b, missing_b = _group_slope(names, origin_slopes, anchor['delta'], report['group_b'])
            bank.update(losses=losses, exact_slopes={'A': slope_a, 'B': slope_b,
                                                     'missing_gradient': missing_a or missing_b},
                        output=_output_summary(images))
            del images, origin_slopes
        report['stencil_ran'] = True
        report['status'] = 'complete'
        return report
    except BaseException as failure:
        report['status'] = 'failed'
        report['failure'] = {'type': type(failure).__name__, 'message': str(failure)}
        raise
    finally:
        trainer._update_response_observer = None
        _restore(trainer, initial)
        report['restored'] = _same_state(initial['state'], trainer_state(trainer, None))
        report['protected_before_sha256'] = protected_hash
        report['protected_after_sha256'] = _hash(protected)
        report['source_config_unchanged'] = source.read_bytes() == original
        _write(destination, report, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--device')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run_probe(args.config, device=args.device, output=args.output)
    if report.get('status') != 'complete' or not report.get('restored') or not report.get('source_config_unchanged'):
        raise SystemExit('Grouped probe failed: ' + str(report.get('failure')))
    print(args.output, flush=True)


if __name__ == '__main__':
    main()

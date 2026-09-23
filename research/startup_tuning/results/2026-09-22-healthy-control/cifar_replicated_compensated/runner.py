#!/usr/bin/env python3
"""Causal parameterization diagnostic: widen FFNs without changing initial G.

Duplicate each hidden unit k times, repeat its down-projection column / k.
The unscaled Adam update adds k copies of each down-weight update. The
compensated control divides that weight's LR by k and the up-layer Adam eps
by k, preserving the original functional updates in exact arithmetic.
This is a deliberately tied initialization, not an independent-capacity model
or a proposed general training fix. No source recipe or seed is changed.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports'))
import torch
from hypergan.checkpoints import capture_rng, restore_rng
from hypergan.config import load_config
from hypergan.training import ReferenceTrainer
from joint_rate_probe import _identity_hash, _sample_prior, run_probe
from healthy_control_screen import CONFIGS, stages

SOURCE = CONFIGS['cifar']
WIDE = SOURCE.parent.parent / 'cifar-transgan32-replicated-ffn/cifar-transgan.toml'
FACTORS = {f'models.generator.network.nodes.n_stage{s}_block{b}_ffn': k
           for s, k in ((8, 4), (16, 16)) for b in (0, 1)}


def replicate_state(target, source, factors):
    assert set(target) == set(source)
    exact, transformed = {}, {}
    with torch.no_grad():
        for name, dst in target.items():
            src, value = source[name], source[name]
            factor = next((k for prefix, k in factors.items()
                           if name.startswith(prefix + '.')), 1)
            if factor != 1:
                if name.endswith('up.weight'):
                    value = src.repeat(factor, 1)
                elif name.endswith('up.bias'):
                    value = src.repeat(factor)
                elif name.endswith('down.weight'):
                    value = src.repeat(1, factor) / factor
                else:
                    assert name.endswith('down.bias'), name
            assert dst.shape == value.shape, name
            dst.copy_(value)
            assert torch.equal(dst.cpu(), value.cpu()), name
            if dst.shape == src.shape:
                exact[name] = src
            else:
                transformed[name] = {'factor': factor, 'source_shape': list(src.shape),
                                     'target_shape': list(dst.shape)}
    return {'exact_tensors': len(exact), 'transformed_tensors': transformed,
            'exact_source_sha256': _identity_hash(exact),
            'exact_target_sha256': _identity_hash({k: target[k] for k in exact})}


def prepare(compensated):
    @contextmanager
    def context(trainer):
        saved_rng = capture_rng()
        groups, bases = list(trainer.opt_g.param_groups), list(trainer.base_lrs[0])
        try:
            config = load_config(SOURCE)
            config['training']['device'] = 'cpu'
            donor = ReferenceTrainer(config)
            source_rng = capture_rng()
            assert _identity_hash(trainer.prior.state_dict()) == _identity_hash(donor.prior.state_dict())
            alignment = replicate_state(trainer.graph.state_dict(), donor.graph.state_dict(), FACTORS)
            trainer.ema_graph.load_state_dict(trainer.graph.state_dict())
            latent = _sample_prior(trainer, trainer.streams['prior'].get_state())[0]
            # Check mathematical function equality using float64 on CPU; GPU
            # TF32 summation order can differ when changing matrix dimensions.
            source_g = donor.graph.models['generator'].double()
            target_g = copy.deepcopy(trainer.graph.models['generator']).cpu().double()
            with torch.no_grad():
                x = latent[:2].cpu().double()
                old, new = source_g(x), target_g(x)
                torch.testing.assert_close(old, new, rtol=1e-9, atol=1e-9)
                alignment['cpu_float64_initial_output_max_abs_error'] = float((old-new).abs().max())
            del donor, source_g, target_g
            if compensated:
                original = groups[0]
                names = {id(p): n for n, p in trainer.graph.named_parameters()}
                ordinary, changed = [], []
                for parameter in original['params']:
                    name = names[id(parameter)]
                    factor = next((k for prefix, k in FACTORS.items()
                                   if name.startswith(prefix + '.')), 1)
                    if factor != 1 and name.endswith('down.weight'):
                        changed.append({**original, 'params': [parameter], 'lr': original['lr'] / factor})
                    elif factor != 1 and (name.endswith('up.weight') or name.endswith('up.bias')):
                        changed.append({**original, 'params': [parameter], 'eps': original['eps'] / factor})
                    else:
                        ordinary.append(parameter)
                trainer.opt_g.param_groups = [{**original, 'params': ordinary}, *changed, *groups[1:]]
                trainer.base_lrs[0] = [group['lr'] for group in trainer.opt_g.param_groups]
            restore_rng(source_rng)
            yield {'kind': 'function-preserving-FFN-replication', 'compensated': compensated,
                   'factors': FACTORS, 'alignment': alignment,
                   'optimizer_rule': ('down.weight LR / k; up.weight/up.bias eps / k' if compensated
                                      else 'unchanged source Adam rates and epsilon')}
        finally:
            trainer.opt_g.param_groups, trainer.base_lrs[0] = groups, bases
            restore_rng(saved_rng)
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compensated', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    case = 'cifar_replicated_compensated' if args.compensated else 'cifar_replicated'
    destination = args.output_root / case
    destination.mkdir(parents=True, exist_ok=False)
    for path in (WIDE, WIDE.parent / 'generator.hndl', WIDE.parent / 'discriminator.hndl'):
        (destination / path.name).write_bytes(path.read_bytes())
    (destination / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (destination / 'resolved-training-config.json').write_text(json.dumps(load_config(WIDE), indent=2)+'\n')
    report = run_probe(WIDE, g_lr=3e-4, d_lr=4.5e-4, steps=512, device=args.device,
                       observe_modules=stages('cifar'), prepare=prepare(args.compensated),
                       progress_path=destination / 'report.json')
    if report['status'] != 'complete':
        raise RuntimeError(report.get('failure', report.get('audit_failure')))


if __name__ == '__main__':
    main()

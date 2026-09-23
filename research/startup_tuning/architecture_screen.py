#!/usr/bin/env python3
"""Paired generator architecture screens with stage observations, at source rates.

Unchanged tensors are copied from the original seeded CPU initialization.
Narrow FFNs use the first 1024 hidden units; down weights and down biases
are multiplied by two to preserve the declared fan-in initialization law.
The pixelshuffle case slices reduced channel axes and rescales Linear tensors
for their new initialization bounds. This aligns random draws, not initial
functions. No seed search or tuning.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports'))
import torch
from hypergan.checkpoints import capture_rng, restore_rng
from hypergan.config import load_config
from hypergan.training import ReferenceTrainer
from joint_rate_probe import run_probe, _identity_hash

SOURCE = ROOT / 'research/startup_tuning/testbeds/transgan-resnet128/transgan-resnet.toml'
NARROW = SOURCE.parent.parent / 'transgan-resnet128-narrow-ffn/transgan-resnet.toml'
SHUFFLE = SOURCE.parent.parent / 'transgan-resnet128-pixelshuffle/transgan-resnet.toml'
EARLY = tuple(f'models.generator.network.nodes.n_stage{s}_block{b}_ffn.'
              for s in (8, 16) for b in (0, 1))
STAGES = tuple('models.generator.network.nodes.n_' + name for name in (
    'input_projection', 'stage8_block0_ffn', 'stage8_block0',
    'stage8_block1_ffn', 'stage8_block1', 'stage16_upsample',
    'stage16_block0_ffn', 'stage16_block0', 'stage16_block1_ffn',
    'stage16_block1', 'stage32_block1', 'stage64_unwindows',
    'stage128_unwindows', 'output_projection'))


def align_state(target, source):
    """Copy matched draws and fail closed on any unexpected architecture change."""
    if set(target) != set(source):
        raise ValueError('State names differ')
    copied, resized = {}, {}
    with torch.no_grad():
        for name, dst in target.items():
            src = source[name]
            early = any(name.startswith(prefix) for prefix in EARLY)
            rule = 'exact'
            if early and name.endswith('up.weight'):
                assert src.shape == (4096, 1024) and dst.shape == (1024, 1024)
                value, rule = src[:1024], 'first 1024 rows'
            elif early and name.endswith('up.bias'):
                assert src.shape == (4096,) and dst.shape == (1024,)
                value, rule = src[:1024], 'first 1024 entries'
            elif early and name.endswith('down.weight'):
                assert src.shape == (1024, 4096) and dst.shape == (1024, 1024)
                value, rule = src[:, :1024] * 2, 'first 1024 columns times 2 for fan-in'
            elif early and name.endswith('down.bias'):
                assert src.shape == dst.shape == (1024,)
                value, rule = src * 2, 'times 2 for fan-in'
            else:
                assert src.shape == dst.shape, name
                value = src
            dst.copy_(value)
            assert torch.equal(dst.cpu(), value.cpu()), name
            (copied if rule == 'exact' else resized)[name] = rule
    return {'exact_tensors': len(copied), 'transformed_tensors': resized,
            'exact_source_sha256': _identity_hash({k: source[k] for k in copied}),
            'exact_target_sha256': _identity_hash({k: target[k] for k in copied})}


def align_shuffle(target_graph, source_graph):
    """Slice reduced axes and preserve Linear fan-in/Xavier initialization laws."""
    target, source = target_graph.state_dict(), source_graph.state_dict()
    assert set(target) == set(source)
    target_modules, source_modules = dict(target_graph.named_modules()), dict(source_graph.named_modules())
    exact, transformed = {}, {}
    with torch.no_grad():
        for name, dst in target.items():
            src = source[name]
            assert src.ndim == dst.ndim and all(a <= b for a, b in zip(dst.shape, src.shape)), name
            value = src[tuple(slice(0, n) for n in dst.shape)] if src.ndim else src
            parent, leaf = name.rsplit('.', 1)
            module, donor = target_modules[parent], source_modules[parent]
            scale = 1.
            if isinstance(module, torch.nn.Linear):
                if leaf == 'weight' and parent.endswith('n_output_projection'):
                    scale = math.sqrt((donor.in_features + donor.out_features) /
                                      (module.in_features + module.out_features))
                elif leaf in ('weight', 'bias'):
                    scale = math.sqrt(donor.in_features / module.in_features)
            elif src.shape != dst.shape:
                assert 'stage' in parent and 'position' in parent and leaf == 'weight', name
            value = value * scale if scale != 1 else value
            dst.copy_(value)
            assert torch.equal(dst.cpu(), value.cpu()), name
            if src.shape == dst.shape and scale == 1:
                exact[name] = src
            else:
                transformed[name] = {'source_shape': list(src.shape), 'target_shape': list(dst.shape),
                                     'rule': 'leading slice on each axis', 'scale': scale}
    return {'exact_tensors': len(exact), 'transformed_tensors': transformed,
            'exact_source_sha256': _identity_hash(exact),
            'exact_target_sha256': _identity_hash({k: target[k] for k in exact})}


@contextmanager
def prepare_variant(trainer):
    saved_rng = capture_rng()
    config = load_config(SOURCE)
    config['training']['device'] = 'cpu'
    donor = ReferenceTrainer(config)
    # Model initialization occurs on CPU before transfer in both recipes.
    # All explicit data/prior/penalty streams are seeded independently.
    source_rng = capture_rng()
    try:
        shuffle = trainer.config['name'].endswith('-pixelshuffle')
        alignment = (align_shuffle(trainer.graph, donor.graph) if shuffle else
                     align_state(trainer.graph.state_dict(), donor.graph.state_dict()))
        trainer.ema_graph.load_state_dict(trainer.graph.state_dict())
        alignment['source_config_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        alignment['source_trainable_parameters_sha256'] = _identity_hash(
            {k: v for k, v in donor.graph.named_parameters() if v.requires_grad})
        del donor
        restore_rng(source_rng)
        yield {'kind': 'paired-architecture-initialization', 'alignment': alignment}
    finally:
        restore_rng(saved_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('source', 'narrow', 'pixelshuffle'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    config = {'source': SOURCE, 'narrow': NARROW, 'pixelshuffle': SHUFFLE}[args.case]
    destination = args.output_root / args.case
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('generator.hndl', 'discriminator.hndl', 'transgan-resnet.toml'):
        (destination / name).write_bytes((config.parent / name).read_bytes())
    (destination / 'resolved-training-config.json').write_text(json.dumps(load_config(config), indent=2)+'\n')
    (destination / 'runner.py').write_bytes(Path(__file__).read_bytes())
    report = run_probe(config, g_lr=2e-4, d_lr=2e-4, steps=32, device=args.device,
        observe_steps=[0, 1, 8, 16, 32], observe_modules=STAGES,
        prepare=prepare_variant if args.case != 'source' else None,
        progress_path=destination / 'report.json')
    if report['status'] != 'complete':
        raise RuntimeError(report.get('failure', report.get('audit_failure')))


if __name__ == '__main__':
    main()

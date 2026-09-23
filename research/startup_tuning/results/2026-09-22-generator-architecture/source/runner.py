#!/usr/bin/env python3
"""Paired early-FFN width ablation with stage observations, at source rates.

Unchanged tensors are copied from the original seeded CPU initialization.
Narrow FFNs use the first 1024 hidden units; down weights and down biases
are multiplied by two to preserve the declared fan-in initialization law.
This aligns random draws, not initial functions. No seed search or tuning.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
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


@contextmanager
def prepare_narrow(trainer):
    saved_rng = capture_rng()
    config = load_config(SOURCE)
    config['training']['device'] = 'cpu'
    donor = ReferenceTrainer(config)
    # Model initialization occurs on CPU before transfer in both recipes.
    # All explicit data/prior/penalty streams are seeded independently.
    source_rng = capture_rng()
    try:
        alignment = align_state(trainer.graph.state_dict(), donor.graph.state_dict())
        trainer.ema_graph.load_state_dict(trainer.graph.state_dict())
        alignment['source_config_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        alignment['source_trainable_parameters_sha256'] = _identity_hash(
            {k: v for k, v in donor.graph.named_parameters() if v.requires_grad})
        del donor
        restore_rng(source_rng)
        yield {'kind': 'paired-width-initialization', 'alignment': alignment}
    finally:
        restore_rng(saved_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('source', 'narrow'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    config = SOURCE if args.case == 'source' else NARROW
    destination = args.output_root / args.case
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('generator.hndl', 'discriminator.hndl', 'transgan-resnet.toml'):
        (destination / name).write_bytes((config.parent / name).read_bytes())
    (destination / 'resolved-training-config.json').write_text(json.dumps(load_config(config), indent=2)+'\n')
    (destination / 'runner.py').write_bytes(Path(__file__).read_bytes())
    report = run_probe(config, g_lr=2e-4, d_lr=2e-4, steps=32, device=args.device,
        observe_steps=[0, 1, 8, 16, 32], observe_modules=STAGES,
        prepare=prepare_narrow if args.case == 'narrow' else None,
        progress_path=destination / 'report.json')
    if report['status'] != 'complete':
        raise RuntimeError(report.get('failure', report.get('audit_failure')))


if __name__ == '__main__':
    main()

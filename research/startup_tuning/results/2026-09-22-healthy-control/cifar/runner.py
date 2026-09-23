#!/usr/bin/env python3
"""Observe the healthy CIFAR recipe and failing logos recipe without tuning.

Uses each recipe's original seed, rates, prior and training horizon. All hooks
observe original-image batch axes (never the folded window batch). This is a
diagnostic replay, not an independent quality evaluation or a seed experiment.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports'))
from hypergan.config import load_config
from joint_rate_probe import run_probe

TESTBEDS = ROOT / 'research/startup_tuning/testbeds'
CONFIGS = {
    'cifar': TESTBEDS / 'cifar-transgan32-adversarial/cifar-transgan.toml',
    'logos': TESTBEDS / 'transgan-resnet128/transgan-resnet.toml',
}


def stages(case):
    names = ['input_projection', 'stage8_position']
    for size in (8, 16, 32):
        if size != 8:
            names.extend([f'stage{size}_upsample', f'stage{size}_position'])
        for block in (0, 1):
            base = f'stage{size}_block{block}'
            names.extend(base + suffix for suffix in
                         ('_attention', '_attention_residual', '_ffn', ''))
    if case == 'logos':
        names.extend(['stage64_unwindows', 'stage128_unwindows'])
    names.append('output_projection')
    return tuple('models.generator.network.nodes.n_' + name for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CONFIGS), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source = CONFIGS[args.case]
    config = load_config(source)
    destination = args.output_root / args.case
    destination.mkdir(parents=True, exist_ok=False)
    for path in (source, source.parent / 'generator.hndl', source.parent / 'discriminator.hndl'):
        (destination / path.name).write_bytes(path.read_bytes())
    (destination / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (destination / 'resolved-training-config.json').write_text(json.dumps(config, indent=2) + '\n')
    opt = config['optimizer']
    report = run_probe(source, g_lr=opt['lr'], d_lr=opt['lr'] * opt['d_lr_mult'],
                       steps=512 if args.case == 'cifar' else 32, device=args.device,
                       observe_steps=[0, 1, 8, 16, 32, 64, 128, 256, 512],
                       observe_modules=stages(args.case), progress_path=destination / 'report.json')
    if report['status'] != 'complete':
        raise RuntimeError(report.get('failure', report.get('audit_failure')))


if __name__ == '__main__':
    main()

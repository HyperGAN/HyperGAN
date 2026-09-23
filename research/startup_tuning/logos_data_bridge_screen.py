#!/usr/bin/env python3
"""Change only training/evaluation data bindings in the working CIFAR recipe."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports'))
from hypergan.config import load_config
from hypergan.signal_structure import _hash
from hypergan.startup_response_probe import _registered_parameters
from joint_rate_probe import run_probe
from healthy_control_screen import CONFIGS, stages

SOURCE = CONFIGS['cifar']
BRIDGE = SOURCE.parent.parent / 'cifar-transgan32-logos-data/cifar-transgan.toml'
BASELINE = ROOT / 'research/startup_tuning/results/2026-09-22-healthy-control/cifar/report.json'


def verify_data_only_config():
    source, bridge = load_config(SOURCE), load_config(BRIDGE)
    source['name'], source['data'] = bridge['name'], bridge['data']
    for name in source['metrics']['custom']:
        source['metrics']['custom'][name]['evaluation']['data'] = bridge['metrics']['custom'][name]['evaluation']['data']
    assert source == bridge, 'Bridge changed configuration outside data bindings/name'
    return bridge


@contextmanager
def audit_initialization(trainer):
    baseline = json.loads(BASELINE.read_text())
    initial = _hash(_registered_parameters(trainer))
    assert initial == baseline['initial_parameters_sha256'], 'Data bridge changed initial parameters'
    yield {'kind': 'data-only-CIFAR-to-logos32-bridge',
           'source_initial_parameters_sha256': baseline['initial_parameters_sha256'],
           'bridge_initial_parameters_sha256': initial,
           'data_identity': trainer.data.resume_identity()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    config = verify_data_only_config()
    destination = args.output_root / 'logos32_data'
    destination.mkdir(parents=True, exist_ok=False)
    for path in (BRIDGE, BRIDGE.parent / 'generator.hndl', BRIDGE.parent / 'discriminator.hndl',
                 Path(__file__).with_name('bridge_data.py')):
        (destination / path.name).write_bytes(path.read_bytes())
    (destination / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (destination / 'resolved-training-config.json').write_text(json.dumps(config, indent=2)+'\n')
    report = run_probe(BRIDGE, g_lr=3e-4, d_lr=4.5e-4, steps=512, device=args.device,
                       observe_modules=stages('cifar'), prepare=audit_initialization,
                       progress_path=destination / 'report.json')
    if report['status'] != 'complete':
        raise RuntimeError(report.get('failure', report.get('audit_failure')))
    baseline = json.loads(BASELINE.read_text())
    for key in ('measurement_rng_sha256', 'prior_rng_sha256'):
        assert report['evaluation'][key] == baseline['evaluation'][key], key


if __name__ == '__main__':
    main()

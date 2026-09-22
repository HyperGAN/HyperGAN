"""Disposable first-G directional audit; no rate selection or retained updates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch
from hypergan.config import load_config
from hypergan.training import ReferenceTrainer, source_info
from hypergan.startup_dynamics import _snapshot, _restore, _protected, _cpu_clone, _same_state
from hypergan.checkpoints import trainer_state
from hypergan.signal_structure import _hash
from hypergan.startup_response_probe import UpdateObserver, PhaseProbes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Audit destination already exists')
    started = time.monotonic()
    trainer = ReferenceTrainer(load_config(args.config))
    initial = _snapshot(trainer)
    protected = _protected(trainer)
    protected_hash = _hash(protected)
    budget = defaultdict(int)
    observer = UpdateObserver(trainer, _snapshot, protected, protected_hash, budget)
    probes = PhaseProbes(trainer, _restore, protected, protected_hash, budget)
    report = {'source': source_info(), 'config': args.config,
              'purpose': 'first-update model validity, not a candidate search', 'banks': []}
    try:
        trainer._update_response_observer = observer
        trainer.update()
        trainer._update_response_observer = None
        anchor = observer.anchors['generator']
        # Reserve two fresh draws after the first training update, but materialize
        # latent coordinates under the actual first-G prior.
        batches = [_cpu_clone(trainer.batch()) for _ in range(2)]
        sampling_rng = trainer.streams['prior'].get_state().clone()
        _restore(trainer, anchor['snapshot'])
        trainer.streams['prior'].set_state(sampling_rng)
        with torch.no_grad():
            banks = [(batch, _cpu_clone(trainer.prior.sample(
                trainer.config['training']['batch_size'], generator=trainer.streams['prior']))) for batch in batches]
        for index, bank in enumerate(banks):
            rows = []
            # Fixed diagnostic locations resolve local derivative versus the
            # existing full-update loss stencil. No best-point selection.
            for factor in (0., .01, .1, .5, 1.):
                print(f'bank {index}, factor {factor}', flush=True)
                rows.append(probes.measure_direction(anchor, 'generator', bank, factor,
                                                     gradient=factor in (0., .01, .1)))
            report['banks'].append(rows)
        report['observations'] = observer.observations
    finally:
        report['protected_after'] = _hash(protected)
        _restore(trainer, initial)
        del trainer._update_response_observer
        report['protected_before'] = protected_hash
        report['restored'] = _same_state(initial['state'], trainer_state(trainer, None))
        report['budget'] = dict(budget)
        report['elapsed_seconds'] = time.monotonic() - started
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert report['restored'] and report['protected_before'] == report['protected_after']


if __name__ == '__main__':
    main()

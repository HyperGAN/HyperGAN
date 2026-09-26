"""Demo 2: encoder + generator + two critics + reconstruction.

    PYTHONPATH=src python examples/api/config_first/demo_2_multi_network.py [RUN_DIR]

Shows the lowered engine recipe (what the run records), validates it, trains a
few CPU steps and prints every loss term the file lists.
"""
import json
from pathlib import Path
import sys
import tempfile

import hypergan.api as hg

HERE = Path(__file__).resolve().parent


def main(run_dir):
    model = hg.load(HERE / 'two_critics.toml')
    recipe = hg.lower(model)
    print('components:', {name: spec['inputs'] for name, spec in recipe['components'].items()})
    print('adversarial_terms:', json.dumps(recipe['adversarial_terms']))
    print('objectives:', json.dumps(recipe['objectives']))
    for warning in hg.validate(model):
        print('warning:', warning)
    run = hg.train(model, run_dir)
    print(f'status={run.status} steps={run.step} fingerprint={run.fingerprint[:12]}')
    series = hg.metrics(run)
    for name in sorted(series):
        if name.startswith('loss/'):
            step, value = series[name][-1]
            print(f'{name:34s} step {step:3d}  {value:.4f}')
    return run


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='hg-demo2-')) / 'run')

"""Demo 1: train the simplest model file and read its metrics back.

    PYTHONPATH=src python examples/api/config_first/demo_1_simple.py [RUN_DIR]

The model is simple.toml: your nn.Module generator (toy_project.py), an HNDL
critic (critic.hndl), the built-in 2-D Gaussian grid. The same file trains with
``hypergan train examples/api/config_first/simple.toml --run-dir RUN``.
"""
from pathlib import Path
import sys
import tempfile

import hypergan.api as hg

HERE = Path(__file__).resolve().parent


def main(run_dir):
    model = hg.load(HERE / 'simple.toml')
    for warning in hg.validate(model):
        print('warning:', warning)
    run = hg.train(model, run_dir, previews=5)
    print(f'status={run.status} steps={run.step} fingerprint={run.fingerprint[:12]}')
    series = hg.metrics(run)
    for name in ('loss/d_total', 'loss/g_total', 'loss/gradient_penalty', 'loss/prior_regularizer', 'diversity/ratio'):
        step, value = series[name][-1]
        print(f'{name:28s} step {step:3d}  {value:.4f}')
    return run


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='hg-demo1-')) / 'run')

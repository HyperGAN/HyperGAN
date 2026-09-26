"""Multi-process check: one model file, two CPU ranks (Gloo), item-level data.

    PYTHONPATH=src python examples/api/config_first/demo_replicated.py [RUN_DIR]

Every rank rebuilds the model from the lowered recipe plus importable code
(toy_project); HyperGAN, not the dataset, decides which items each step uses.
The replicated path rejects extra adversarial terms today, so this uses the
one-critic model with the item dataset swapped in by override.
"""
from pathlib import Path
import sys
import tempfile

import hypergan.api as hg

HERE = Path(__file__).resolve().parent


def main(run_dir):
    model = hg.override(hg.load(HERE / 'simple.toml'), {
        'data': {'dataset': 'toy_project:GridPoints', 'args': {'side': 5, 'size': 2048}},
        'train.steps': 6,
    })
    run = hg.train(model, run_dir, profile='cpu-replicated-gloo')
    print(f'status={run.status} steps={run.step} execution={run.manifest.get("execution")}')
    series = hg.metrics(run)
    for name in ('loss/d_total', 'loss/g_total'):
        print(f'{name:16s}', [round(value, 4) for _, value in series[name]])
    return run


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='hg-replicated-')) / 'run')

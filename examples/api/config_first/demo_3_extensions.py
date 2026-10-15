"""Demo 3: item-level data loader, sampler, metric and evaluation.

    PYTHONPATH=src python examples/api/config_first/demo_3_extensions.py [RUN_DIR]

The metric and the interval evaluation run in HyperGAN's bounded workers while
training; the manual evaluation runs afterwards; the sampler turns recorded 2-D
samples into images when read.
"""
from pathlib import Path
import sys
import tempfile

import hypergan.api as hg

HERE = Path(__file__).resolve().parent


def main(run_dir):
    run_dir = Path(run_dir)
    model = hg.load(HERE / 'extensions.toml')
    run = hg.train(model, run_dir, previews=5)
    print(f'status={run.status} steps={run.step}')

    print('metric d_over_g:', [(s, round(v, 4)) for s, v in hg.metrics(run)['d_over_g']])
    hg.evaluate(run, 'nearest')                       # manual evaluation, after training
    for metric, rows in hg.evaluations(run).items():
        print(f'evaluation {metric}:', [(row['step'], row['status'], row['value']) for row in rows])

    previews = hg.samples(run)
    print('previews at steps', [item.step for item in previews], 'shape', tuple(previews[-1].data.shape))
    picture = hg.view(previews[-1], 'scatter', run=run)          # declared sampler, by name
    path = picture.save(run_dir.parent / 'scatter.png')
    print(f'sampler scatter -> {picture.width}x{picture.height} image at {path}')
    fresh = hg.sample(run, count=64, seed=1)
    print('fresh sample', tuple(fresh.data.shape), 'modes covered:',
          __import__('toy_project').modes_covered(fresh.data, None))
    return run


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='hg-demo3-')) / 'run')

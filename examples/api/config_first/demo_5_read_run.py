"""Demo 5: read a run back - metrics, evaluations, samples and previews.

    PYTHONPATH=src python examples/api/config_first/demo_5_read_run.py [RUN_DIR]

Reads an existing run (for example demo 3's), or trains extensions.toml first.
Reading needs only the run directory; the sampler named in the model file is
recorded with the run, so ``view(sample, "scatter", run=run)`` works without it.
"""
from pathlib import Path
import sys
import tempfile

import hypergan.api as hg

HERE = Path(__file__).resolve().parent


def main(run_dir):
    run_dir = Path(run_dir)
    model = hg.load(HERE / 'extensions.toml')
    if not (run_dir / 'manifest.json').exists():
        hg.train(model, run_dir, previews=5)
    run = hg.open_run(run_dir)
    print(f'run {run.path.name}: status={run.status} steps={run.step}')
    print('fingerprint matches model file:', run.fingerprint == hg.fingerprint(model))

    series = hg.metrics(run)
    print(f'{len(series)} metric series; last values:')
    for name in ('loss/d_total', 'loss/g_total', 'd_over_g', 'throughput/steps_per_second'):
        if name in series:
            step, value = series[name][-1]
            print(f'  {name:30s} step {step:3d}  {value:.4f}')
    for metric, rows in hg.evaluations(run).items():
        print(f'  evaluation {metric:18s}', ', '.join(f'step {r["step"]}: {r["value"]:.4f} ({r["status"]})' for r in rows))

    previews = hg.samples(run)
    for item in previews:
        print(f'  preview step {item.step} name={item.name} data={tuple(item.data.shape)} '
              f'unique particles={len(set(item.particle_ids))}')
    if previews:
        out = hg.view(previews[-1], 'scatter', run=run).save(run_dir.parent / 'preview-scatter.png')
        print('  scatter of last preview ->', out.name)
    fresh = hg.sample(run, count=32, seed=5)
    print(f'  fresh EMA sample step {fresh.step}: {tuple(fresh.data.shape)} mean={fresh.data.mean(0).tolist()}')
    out = hg.view(fresh, 'toy_project:scatter').save(run_dir.parent / 'fresh-scatter.png')
    print('  scatter of fresh sample ->', out.name)
    return run


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='hg-demo5-')) / 'run')

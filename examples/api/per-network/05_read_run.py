"""Demo 5: read a run back: metric definitions, metrics, evaluations, previews and samples.

    python examples/api/per-network/05_read_run.py [RUNS_ROOT]   (reads RUNS_ROOT/extensions)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hypergan import api_per_network as hg  # noqa: E402
from toy_project.recipes import extensions  # noqa: E402


def main(root):
    run_dir = Path(root) / 'extensions'
    recipe = extensions(steps=20)
    if not (run_dir / 'manifest.json').is_file():
        hg.train(recipe, run_dir, preview_every=10, checkpoint_every=10)
    run = hg.run(run_dir)
    print(run)
    catalog = hg.catalog(run)
    metrics = hg.metrics(run)
    print(f'\n{len(metrics)} published metrics:')
    for name, rows in sorted(metrics.items()):
        definition = catalog.get(name, {})
        print(f'  {name:32s} {definition.get("label", ""):34s} n={len(rows):3d} last={rows[-1][1]:.4f}')
    print('\nevaluations:')
    for name, rows in hg.evaluations(run).items():
        for row in rows:
            print(f'  {name} step {row["step"]}: {row["status"]} {row["value"]}')
    receipt = hg.evaluate(run, 'radius_gap')  # one more, on demand, on the final snapshot
    print(f'  on demand: {receipt["status"]} step {receipt["result"]["step"]}: {receipt["result"]["value"]}')
    print('\npreviews (raw tensors):')
    for preview in hg.previews(run):
        print(f'  step {preview.step} {preview.name}: shape {preview.shape}')
    scatter = next(item for item in recipe.observe if isinstance(item, hg.Sampler))
    view = hg.samples(run, 64, seed=1, sampler=scatter)
    print(f'\nsamples through the scatter sampler: {view["count"]} points, mean radius {view["radius"]}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/api-per-network')

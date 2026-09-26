"""Demo 3: your own item-level dataset, metric, evaluation and sampler.

    python examples/api/per-network/03_extensions.py [RUNS_ROOT]
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hypergan import api_per_network as hg  # noqa: E402
from toy_project.recipes import extensions  # noqa: E402


def main(root):
    recipe = extensions(steps=20)
    config = hg.lower(recipe)
    print('data:', config['data'])
    for name, spec in config['metrics']['custom'].items():
        print(f'observation {name}: {spec["mode"]} via {spec["factory"]} -> {spec["args"]["function"]}')
    run_dir = Path(root) / 'extensions'
    shutil.rmtree(run_dir, ignore_errors=True)
    run = hg.train(recipe, run_dir, preview_every=10, checkpoint_every=10)
    print(f'\n{run.status}: {run.steps} steps')
    print('g_over_d (custom metric):', hg.metrics(run, {'g_over_d'}).get('g_over_d'))
    for name, rows in hg.evaluations(run).items():
        print(f'{name} (custom evaluation):', [(row['step'], row['status'], row['value']) for row in rows])
    # Observation is lossy by design: a busy worker drops or skips instead of stalling training.
    for step, name, status, reason in hg.observations(run):
        print(f'  step {step:3d} {name:10s} {status}' + (f' ({reason})' if reason else ''))
    scatter = next(item for item in recipe.observe if isinstance(item, hg.Sampler))
    for preview in hg.previews(run, sampler=scatter):
        view = preview.value
        print(f'preview step {preview.step}: {view["count"]} points, mean radius {view["radius"]}')
    view = hg.samples(run, 256, seed=0, sampler=scatter)
    print(f'fresh samples (step {view["step"]}): mean {view["mean"]}, radius {view["radius"]}')
    print('\n'.join(view['ascii']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/api-per-network')

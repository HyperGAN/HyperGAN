"""Demo 1: the simplest model. Your own nn.Module generator, an HNDL critic, 2-D toy data.

    python examples/api/per-network/01_simple.py [RUNS_ROOT]
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # makes toy_project importable (workers inherit it)

from hypergan import api_per_network as hg  # noqa: E402
from toy_project.recipes import simple  # noqa: E402


def main(root):
    run_dir = Path(root) / 'simple'
    shutil.rmtree(run_dir, ignore_errors=True)
    recipe = simple(steps=20)
    print(hg.explain(recipe))
    run = hg.train(recipe, run_dir)
    print(f'\n{run.status}: {run.steps} steps, fingerprint {run.config_sha256[:12]}')
    metrics = hg.metrics(run)
    for name in ('loss/d_total', 'loss/g_total', 'loss/gradient_penalty', 'loss/prior_regularizer'):
        step, value = metrics[name][-1]
        print(f'{name:28s} {len(metrics[name]):3d} values, step {step}: {value:.4f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/api-per-network')

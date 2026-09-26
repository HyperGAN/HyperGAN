"""Demo 2: encoder + generator + two critics + reconstruction; each network owns its losses.

    python examples/api/per-network/02_multi_network.py [RUNS_ROOT]
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hypergan import api_per_network as hg  # noqa: E402
from toy_project.recipes import multi_network  # noqa: E402


def main(root):
    recipe = multi_network(steps=10)
    print(hg.explain(recipe))
    lowered = hg.lower(recipe)  # the raw engine config: no Torch, no user code imported
    print('\nlowered objectives:', json.dumps([(t.get('id'), t['factory']) for t in lowered['objectives']]))
    print('lowered adversarial_terms:', json.dumps(lowered['adversarial_terms']))
    print('lowered optimizer:', json.dumps(lowered['optimizer']))
    resolved = hg.validate(recipe)
    print('validated; warnings:', resolved['warnings'])
    run_dir = Path(root) / 'multi-network'
    shutil.rmtree(run_dir, ignore_errors=True)
    run = hg.train(recipe, run_dir)
    print(f'\n{run.status}: {run.steps} steps')
    metrics = hg.metrics(run)
    for name in ('loss/d_total', 'loss/g_total', 'loss/d_adversarial', 'loss/gradient_penalty',
                 'loss/objectives/generator.reconstruction', 'loss/objectives/encoder.code_norm'):
        step, value = metrics[name][-1]
        print(f'{name:42s} step {step}: {value:.4f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/api-per-network')

"""Demo 4: save a recipe to a plain config file, load it back, same fingerprint.

    python examples/api/per-network/04_round_trip.py [RUNS_ROOT]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hypergan  # noqa: E402
from hypergan import api_per_network as hg  # noqa: E402
from toy_project.recipes import multi_network  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def main(root):
    root = Path(root)
    recipe = multi_network()
    path = hg.save(recipe, root / 'multi-network.toml')
    print(path.read_text())
    loaded = hg.load(path)
    print('in memory :', hg.fingerprint(recipe))
    print('from file :', hg.fingerprint(path))
    print('reloaded  :', hg.fingerprint(loaded))
    assert hg.fingerprint(recipe) == hg.fingerprint(path) == hg.fingerprint(loaded)
    assert hg.lower(loaded) == hg.lower(recipe)
    # The same file through the CLI (torch-free validation, no user code imported).
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(hypergan.__file__).parents[1]), str(Path(__file__).resolve().parent)]))
    result = subprocess.run([sys.executable, '-m', 'hypergan', 'validate', str(path)],
                            capture_output=True, text=True, env=environment)
    shown = json.loads(result.stdout) if result.returncode == 0 else result.stderr
    print('hypergan validate exit', result.returncode, '->',
          sorted(shown['components']) if isinstance(shown, dict) and 'components' in shown else shown)
    # Any existing config reads as per-network declarations.
    paired = hg.load(REPO / 'examples' / 'paired-linear.toml')
    print('\nexamples/paired-linear.toml as declarations:\n' + hg.explain(paired))
    print('same fingerprint after lift:',
          hg.fingerprint(paired) == hg.fingerprint(REPO / 'examples' / 'paired-linear.toml'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/api-per-network')

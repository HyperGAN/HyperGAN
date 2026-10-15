"""Demo 4: save a model file, load it back, same fingerprint.

    PYTHONPATH=src python examples/api/config_first/demo_4_round_trip.py [OUT_DIR]

Three routes to the same model must give one fingerprint: the hand-written
file, that file saved elsewhere and reloaded, and the model built in Python
with the helpers and saved. The CLI's loader (hypergan.config.load_config)
agrees, so ``hypergan train FILE`` trains the same recipe.
"""
from pathlib import Path
import sys
import tempfile

import hypergan.api as hg
from hypergan.config import fingerprint, load_config

HERE = Path(__file__).resolve().parent
DECODER = 'concat(x, condition)\nlinear(16)\nleaky_relu(0.2)\nlinear(16)\nleaky_relu(0.2)\nlinear()\n'
PAIR = 'concat(x, condition)\nlinear(16)\nleaky_relu(0.2)\nlinear()\n'


def python_model():
    """two_critics.toml, written in Python."""
    return hg.from_dict({
        'name': 'demo/encoder-two-critics',
        'data': {'dataset': 'toy_project:PairedVectors', 'args': {'size': 512, 'dims': 2}},
        'prior': {'kind': 'particles', 'z_dim': 4, 'num_particles': 64},
        'networks': {
            'encoder': hg.hndl('linear(8)\nleaky_relu(0.2)\nlinear()', role='encoder',
                               input_shape=['B', 2], output_shape=['B', 2], inputs={'input': 'batch.condition'}),
            'decoder': hg.hndl(DECODER, role='generator', input_shape={'x': ['B', 4], 'condition': ['B', 2]},
                               output_shape=['B', 2], inputs={'x': 'latent', 'condition': 'encoder'}),
            'pair_critic': hg.hndl(PAIR, role='critic', input_shape={'x': ['B', 2], 'condition': ['B', 2]},
                                   output_shape=['B', 1], inputs={'x': 'candidate', 'condition': 'batch.condition'}),
            'marginal_critic': hg.hndl(file='critic.hndl', role='critic', input_shape=['B', 2],
                                       output_shape=['B', 1], inputs={'x': 'candidate'}),
        },
        'losses': [
            hg.adversarial('pair_critic', penalty=1.0),
            hg.adversarial('marginal_critic', weight=0.5, penalty=0.5),
            hg.reconstruction(fn='l1'),
            hg.prior_loss(1.0),
        ],
        'penalty': {'kappa': 1.0, 'lazy_k': 1},
        'train': {'steps': 12, 'batch_size': 16, 'seed': 42, 'device': 'cpu'},
        'sampling': {'count': 8, 'seed': 123},
    }, base=HERE)


def main(out_dir):
    out_dir = Path(out_dir)
    original = hg.load(HERE / 'two_critics.toml')
    saved = hg.save(original, out_dir / 'copy' / 'model.toml')   # hndl path rewritten, file not copied
    built = hg.save(python_model(), out_dir / 'built.toml')
    prints = {
        'file': hg.fingerprint(original),
        'file saved + reloaded': hg.fingerprint(hg.load(saved)),
        'python-built saved + reloaded': hg.fingerprint(hg.load(built)),
        'CLI loader (load_config)': fingerprint(load_config(saved)),
    }
    for route, value in prints.items():
        print(f'{route:32s} {value}')
    assert len(set(prints.values())) == 1, prints
    changed = hg.override(original, {'losses.reconstruction.weight': 0.25})
    print('override losses.reconstruction.weight=0.25 ->', hg.fingerprint(changed)[:16], '(differs)')
    print('\n--- saved model file ---\n' + built.read_text())
    return prints


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='hg-demo4-'))

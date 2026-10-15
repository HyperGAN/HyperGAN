"""Tests for hypergan-model/1 format: parsing, lowering, validation, and role checks."""
from pathlib import Path
import sys
import pytest

import hypergan.api as hg
from hypergan.config import fingerprint, resolve_config
from hypergan.model_file import ModelFileError, is_model_file, lower, upstream

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples' / 'api' / 'config_first'


import subprocess

def test_torch_free_lowering():
    """Verify that lower() and is_model_file() execute without importing torch."""
    code = """
import sys
import hypergan.model_file as mf
assert 'torch' not in sys.modules, "torch should not be imported prior to lowering"
res_minimal = mf.lower({'format': 'hypergan-model/1', 'name': 'test', 'checkpoints': {'keep_last': 2}})
assert 'torch' not in sys.modules, "torch must remain unimported during minimal lowering"
assert res_minimal['checkpoints']['keep_last'] == 2

raw = {
    'format': 'hypergan-model/1',
    'data': {'builtin': 'gaussian_grid'},
    'networks': {
        'g': {'role': 'generator', 'module': 'my_module:Generator', 'inputs': {'z': 'latent'}},
        'd': {'role': 'critic', 'module': 'my_module:Critic', 'inputs': {'x': 'candidate'}},
    },
    'losses': [{'type': 'adversarial', 'critic': 'd'}],
    'checkpoints': {'keep_last': 3, 'keep_best': False},
}
assert mf.is_model_file(raw) is True
lowered = mf.lower(raw)
assert 'components' in lowered
assert 'generator' in lowered['components']
assert 'discriminator' in lowered['components']
assert lowered['checkpoints']['keep_last'] == 3
assert lowered['checkpoints']['keep_best'] is None
assert 'torch' not in sys.modules, "torch must remain unimported during model lowering"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_model_file_lowers_roles_and_loss_list():
    raw = hg.load(EXAMPLES / 'two_critics.toml')
    recipe = hg.lower(raw)
    components = recipe['components']
    assert set(components) == {'encoder', 'generator', 'discriminator', 'marginal_critic'}
    assert components['generator']['inputs'] == {'x': 'latent', 'condition': 'components.encoder'}
    assert components['discriminator']['inputs'] == {'x': 'candidate', 'condition': 'batch.condition'}
    assert 'linear()' in components['marginal_critic']['args']['source']
    assert recipe['adversarial'] == {'weight': 1.0}
    assert recipe['adversarial_terms'] == [{
        'id': 'marginal_critic', 'component': 'marginal_critic', 'weight': 0.5,
        'penalty': True, 'penalty_coeff': 0.5, 'real': 'batch.real', 'fake': 'generated'
    }]
    assert recipe['objectives'] == [{
        'id': 'reconstruction', 'factory': 'l1', 'inputs': {'input': 'generated', 'target': 'batch.real'},
        'weight': 1.0, 'args': {}
    }]
    assert recipe['format'] == 'hypergan-model/1'
    assert recipe['name_map']['decoder'] == 'generator'
    assert recipe['name_map']['pair_critic'] == 'discriminator'


def test_packaged_model_is_the_reference_recipe():
    model = hg.packaged('gaussian-grid')
    assert hg.fingerprint(model) == fingerprint(resolve_config({}))
    assert hg.resolve(model)['qualification']['recipe_match'] is True


def test_role_validation():
    # Missing generator
    spec_no_g = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
    }
    with pytest.raises(ModelFileError, match='Exactly one network may have role = "generator"'):
        lower(spec_no_g)

    # Two generators
    spec_two_g = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g1': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'g2': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
    }
    with pytest.raises(ModelFileError, match='Exactly one network may have role = "generator"'):
        lower(spec_two_g)

    # Missing critic network
    spec_no_d = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
    }
    with pytest.raises(ModelFileError, match='Adversarial loss 0 critic must name a network with role = "critic"'):
        lower(spec_no_d)


def test_reserved_optimizer_slot_rejected():
    spec = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}, 'optimizer': {'lr': 1e-4}},
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
    }
    with pytest.raises(ModelFileError, match="Engine limit: networks.g is trained by generator-side Adam"):
        lower(spec)


def test_two_way_loss_graph_validation():
    # Generator not reached by candidate/generated in discriminator
    spec_unreachable_g = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'latent'}},
        },
        'losses': [
            {'type': 'adversarial', 'critic': 'd'},
        ],
    }
    with pytest.raises(ModelFileError, match="Network 'g' has role = 'generator' but no adversarial or objective loss trains it"):
        lower(spec_unreachable_g)

    # Objective loss claimed by owner it doesn't reach
    spec_wrong_owner = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'e': {'role': 'encoder', 'module': 'mod:Enc', 'inputs': {'x': 'batch.real'}},
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [
            {'type': 'adversarial', 'critic': 'd'},
            {'type': 'objective', 'id': 'rec', 'fn': 'mse', 'inputs': {'input': 'g', 'target': 'batch.real'}, 'owner': 'e'},
        ],
    }
    with pytest.raises(ModelFileError, match="do not depend on e"):
        lower(spec_wrong_owner)


def test_upstream_reachability():
    components = {
        'enc': {'inputs': {'x': 'batch.real'}},
        'gen': {'inputs': {'z': 'latent', 'c': 'components.enc'}},
        'disc': {'inputs': {'x': 'components.gen'}},
    }
    assert upstream('components.gen', components) == {'gen', 'enc'}
    assert upstream('components.enc', components) == {'enc'}
    assert upstream('latent', components) == set()

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


def test_hndl_extra_fields_and_file_alias(tmp_path):
    sub_hndl = tmp_path / 'sub_block.hndl'
    sub_hndl.write_text('linear()\n', encoding='utf-8')
    main_hndl = tmp_path / 'gen.hndl'
    main_hndl.write_text('linear()\n', encoding='utf-8')

    spec = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {
                'role': 'generator',
                'file': 'gen.hndl',
                'inputs': {'z': 'latent'},
                'input_shape': ['B', 4],
                'output_shape': ['B', 2],
                'parameters': {'width': 64},
                'input_dtype': {'z': 'float32'},
                'networks': {'inline_blk': 'linear()'},
                'network_files': {'file_blk': 'sub_block.hndl'},
                'pretrained_providers': {'prov': {'factory': 'pkg:fn'}},
            },
            'd': {
                'role': 'critic',
                'source': 'linear()',
                'inputs': {'x': 'candidate'},
                'input_shape': ['B', 2],
                'output_shape': ['B', 1],
            },
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
    }
    lowered = lower(spec, base=tmp_path)
    g_args = lowered['components']['generator']['args']
    assert g_args['source'].strip() == 'linear()'
    assert g_args['parameters'] == {'width': 64}
    assert g_args['input_dtype'] == {'z': 'float32'}
    assert g_args['networks']['inline_blk'] == 'linear()'
    assert g_args['networks']['file_blk'].strip() == 'linear()'
    assert g_args['pretrained_providers'] == {'prov': {'factory': 'pkg:fn'}}


def test_flat_data_keys_and_batch_size_promotion():
    spec = {
        'format': 'hypergan-model/1',
        'data': {
            'builtin': 'image_folder',
            'root': '/tmp/images',
            'height': 64,
            'width': 64,
            'channels': 1,
            'batch_size': 32,
        },
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
    }
    lowered = lower(spec)
    assert lowered['data'] == {
        'factory': 'image_folder',
        'args': {'root': '/tmp/images', 'height': 64, 'width': 64, 'mode': 'L'},
    }
    assert lowered['training']['batch_size'] == 32


def test_adversarial_inline_penalty_and_reconstruction_defaults():
    spec = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'g_ema': {'role': 'generator', 'reuse': 'g', 'inputs': {'z': 'latent'}},
            'critic_a': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
            'discriminator': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [
            {
                'type': 'adversarial',
                'id': 'main_adv',
                'critic': 'critic_a',
                'penalty': {'weight': 2.0, 'lazy_k': 4, 'kappa': 0.5},
            },
            {
                'type': 'adversarial',
                'critic': 'discriminator',
                'weight': 0.5,
            },
            {
                'type': 'reconstruction',
                'fn': 'l1',
                'weight': 0.25,
            },
        ],
    }
    lowered = lower(spec)
    assert lowered['adversarial']['id'] == 'main_adv'
    assert lowered['gradient_penalty'] == {'coeff': 2.0, 'lazy_k': 4, 'kappa': 0.5}
    assert lowered['objectives'][0]['id'] == 'reconstruction'
    assert lowered['objectives'][0]['inputs'] == {'input': 'generated', 'target': 'batch.real'}
    assert 'critic_discriminator' in lowered['components']


def test_train_cadence_and_run_validation():
    spec = {
        'format': 'hypergan-model/1',
        'data': {'builtin': 'gaussian_grid'},
        'networks': {
            'g': {'role': 'generator', 'module': 'mod:Gen', 'inputs': {'z': 'latent'}},
            'd': {'role': 'critic', 'module': 'mod:Critic', 'inputs': {'x': 'candidate'}},
        },
        'losses': [{'type': 'adversarial', 'critic': 'd'}],
        'train': {'steps': 100, 'checkpoint_every': 50, 'preview_every': 25},
        'metrics': {'every': 5},
        'evaluations': {
            'samples': 128,
            'batch_size': 32,
            'seed': 7,
            'data': {'builtin': 'gaussian_grid'},
            'fid': {
                'factory': 'hypergan.image_metrics:InceptionFID',
                'every_steps': 50,
                'target': 'generated',
                'on_busy': 'skip',
            },
        },
    }
    lowered = lower(spec)
    assert lowered['run']['checkpoint_every'] == 50
    assert lowered['run']['preview_every'] == 25
    assert lowered['metrics']['every_steps'] == 5
    fid_eval = lowered['metrics']['custom']['fid']
    assert fid_eval['every_steps'] == 50
    assert fid_eval['on_busy'] == 'skip'
    assert fid_eval['evaluation']['sample_count'] == 128

    bad_run = dict(spec, run={'unknown_run_key': 123})
    with pytest.raises(ModelFileError, match='Unknown run field'):
        lower(bad_run)


def test_override_by_loss_type_and_optimizer_prefix():
    m = hg.packaged('gaussian-grid')
    m2 = hg.override(m, {'losses.prior.weight': 2.5, 'optimizer.lr': 5e-4})
    lowered = hg.lower(m2)
    assert lowered['prior_regularizer']['weight'] == 2.5
    assert lowered['optimizer']['lr'] == 5e-4
    explained = hg.explain(m2)
    assert '0.0005' in explained

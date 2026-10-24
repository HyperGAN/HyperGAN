"""Runtime description checks and Torch-free released-preset metadata checks."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

from hypergan.config import load_config, resolve_config
from hypergan.metrics import objective_id
from hypergan.model_description import describe_config, describe_run

ALL_EXAMPLES = sorted(Path(__file__).resolve().parents[2].joinpath('examples').glob('*.toml'))
RELEASED_EXAMPLES = [path for path in ALL_EXAMPLES
                     if tomllib.loads(path.read_text()).get('defaults') == 'particlegan']
LOCAL_PATH = re.compile(r'"(?:/home|/mnt|/root|/Users|~/)')


@pytest.mark.parametrize('path', RELEASED_EXAMPLES, ids=lambda p: p.stem)
def test_every_example_describes(path):
    config = load_config(path)
    result = describe_run({'config': config, 'config_sha256': 'c' * 64, 'run_id': 'run'})
    text = json.dumps(result, allow_nan=False)
    assert not LOCAL_PATH.search(text), 'absolute local path leaked'
    assert len(text) < 256 * 1024
    assert result['formulation']['family'] == 'particlegan'
    assert result['formulation']['parameters']['coeff'] == config['gradient_penalty']['coeff']
    names = [entry['name'] for entry in result['networks']]
    assert names == list(config['components'])
    assert {'generator', 'discriminator'} <= set(names)
    roles = {entry['name']: entry['role'] for entry in result['networks']}
    assert roles['generator'] == 'generator' and roles['discriminator'] == 'critic'
    for entry in result['networks']:
        assert entry['graph']['status'] == 'unavailable' and entry['graph']['reason']
        spec = config['components'][entry['name']]
        if spec.get('factory') == 'hndl':
            assert entry['source']['text'].strip()
            assert entry['input_shape'] == spec['args']['input_shape']
    # Every loss term names its series, or says why it has none.
    for side in ('discriminator', 'generator'):
        for term in result['losses'][side]:
            assert term['metric'] or term['metric_note']
    objectives = {term['metric'] for term in result['losses']['generator'] if term['kind'] not in (
        'adversarial_g', 'prior_regularizer')}
    assert objectives == {'loss/objectives/' + objective_id(t) for t in config['objectives']}
    assert result['optimizers']['critic']['lr'] == pytest.approx(
        config['optimizer']['lr'] * config['optimizer']['d_lr_mult'])


def test_description_never_imports_torch(tmp_path):
    code = ("import sys, json; from hypergan.config import load_config; "
            "from hypergan.model_description import describe_run; "
            "[json.dumps(describe_run({'config': load_config(p)})) for p in sys.argv[1:]]; "
            "assert 'torch' not in sys.modules and 'hndl' not in sys.modules and 'particlegan' not in sys.modules")
    subprocess.run([sys.executable, '-c', code, *map(str, RELEASED_EXAMPLES)], cwd=tmp_path, check=True)


def _training_view(config):
    """Host parameter ownership after the dependency applies its prior policy."""
    from hypergan.recipes import make_prior
    prior = make_prior({**config['prior'], 'args': {**config['prior']['args'], 'num_particles': 8}
                        if config['prior']['kind'] != 'gaussian' else config['prior']['args']},
                       device='cpu', recipe=config.get('recipe'))
    return any(p.requires_grad for p in prior.parameters())


def _check_against_training(config):
    prior_group = _training_view(config)
    result = describe_config(config)
    assert result['formulation']['family'] == 'particlegan'
    assert result['formulation']['equations'] == {}
    assert result['formulation']['parameters']['coeff'] == config['gradient_penalty']['coeff']
    assert result['optimizers']['generator']['type'] == 'ParticleGAN'
    assert (result['optimizers']['prior'].get('status') != 'unavailable') == prior_group
    spread = [t for t in result['losses']['generator'] if t['id'] == 'prior_regularizer']
    if spread:
        assert spread[0]['active'] == (prior_group and config['prior_regularizer']['weight'] > 0)
    return result


@pytest.mark.parametrize('path', ALL_EXAMPLES, ids=lambda p: p.stem)
def test_example_description_matches_what_training_builds(path):
    _check_against_training(load_config(path))


def test_constant_learning_rate_is_reported_as_configured():
    config = resolve_config({'training': {'lr_floor': 1.0, 'network_lr_floor': None}})
    result = _check_against_training(config)
    assert result['optimizers']['schedule']['lr_floor'] == 1.0


def test_gaussian_and_frozen_priors_have_no_prior_group_or_damping():
    gaussian = resolve_config({'prior': {'kind': 'gaussian', 'args': {'z_dim': 4}}, 'prior_regularizer': {'weight': 0}})
    result = _check_against_training(gaussian)
    assert result['optimizers']['prior'] == {'status': 'unavailable', 'reason': 'Gaussian prior has no trainable table'}
    frozen = resolve_config({'prior': {'args': {'num_particles': 64, 'z_dim': 4, 'learnable': False}},
                             'optimizer': {'latent_damping_max_rate': 0.5}})
    result = _check_against_training(frozen)
    assert 'frozen' in result['optimizers']['prior']['reason']
    spread = next(t for t in result['losses']['generator'] if t['id'] == 'prior_regularizer')
    assert spread['active'] is False and 'frozen' in spread['inactive_reason']

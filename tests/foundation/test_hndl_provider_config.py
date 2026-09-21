"""Provider pins are validated and fingerprinted without model-library imports."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hypergan.config import DEFAULT, config_values, fingerprint, load_config, resolve_config
from hypergan.network_config import validate_network_args


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples' / 'sagan-adain-dinov3-multidepth-128.toml'
PROVIDER = {'source_path': '/unavailable/dinov3-source', 'source_commit': 'a' * 40}


def network_args(providers):
    return {'source': 'linear(1)', 'input_shape': ['B', 2], 'output_shape': ['B', 1],
            'pretrained_providers': deepcopy(providers)}


def resolved_with_providers(providers):
    raw = deepcopy(DEFAULT)
    raw['components']['discriminator'] = {
        'factory': 'hndl', 'inputs': {'x': 'candidate'}, 'args': network_args(providers)}
    return resolve_config(raw)


@pytest.mark.parametrize('providers', [{}, {'dinov3_vits16': PROVIDER}])
def test_provider_options_survive_resolution_without_reading_source(providers):
    args = network_args(providers)
    validate_network_args(args, 'components.discriminator.args')
    resolved = resolved_with_providers(providers)
    assert resolved['components']['discriminator']['args']['pretrained_providers'] == providers


@pytest.mark.parametrize('providers, message', [
    (None, 'must be a table'),
    ([], 'must be a table'),
    ({'unknown': PROVIDER}, 'Unknown configurable pretrained provider'),
    ({'torchvision_resnet18': PROVIDER}, 'Unknown configurable pretrained provider'),
    ({'dinov3_vits16': None}, 'requires source_path and source_commit'),
    ({'dinov3_vits16': {}}, 'requires source_path and source_commit'),
    ({'dinov3_vits16': {'source_path': '/source'}}, 'requires source_path and source_commit'),
    ({'dinov3_vits16': {'source_commit': 'a' * 40}}, 'requires source_path and source_commit'),
    ({'dinov3_vits16': {**PROVIDER, 'pretrained': True}}, 'requires source_path and source_commit'),
    *[({'dinov3_vits16': {**PROVIDER, 'source_path': value}}, 'source_path must be nonempty text')
      for value in ('', '  ', 12, None)],
    *[({'dinov3_vits16': {**PROVIDER, 'source_commit': value}}, 'full lowercase Git SHA')
      for value in ('main', 'a' * 39, 'a' * 41, 'A' * 40, 'g' * 40, 12, None)],
])
def test_invalid_provider_options_fail_direct_and_recipe_validation(providers, message):
    with pytest.raises(ValueError, match=message):
        validate_network_args(network_args(providers), 'components.discriminator.args')
    with pytest.raises(ValueError, match=message):
        resolved_with_providers(providers)


def test_example_snapshots_both_network_sources_and_provider_pins(tmp_path):
    recipe = tmp_path / EXAMPLE.name
    shutil.copyfile(EXAMPLE, recipe)
    networks = tmp_path / 'networks'
    networks.mkdir()
    names = {'generator': 'sagan-adain-generator-128.hndl',
             'discriminator': 'dinov3-multidepth-discriminator-128.hndl'}
    for name in names.values():
        shutil.copyfile(EXAMPLE.parent / 'networks' / name, networks / name)
    resolved = load_config(recipe)
    for component, name in names.items():
        args = resolved['components'][component]['args']
        assert args['source'] == (networks / name).read_text()
        assert 'file' not in args
        (networks / name).unlink()
    providers = resolved['components']['discriminator']['args']['pretrained_providers']
    assert providers == {'dinov3_vits16': {
        'source_path': '/path/to/dinov3-source',
        'source_commit': '6876159a11b4df116f30f667f8c9888617df0751'}}
    assert fingerprint(resolve_config(config_values(resolved))) == fingerprint(resolved)


@pytest.mark.parametrize('key,value', [('source_commit', 'b' * 40),
                                       ('source_path', '/different/dinov3-source')])
def test_provider_pin_changes_numerical_fingerprint(key, value):
    original = load_config(EXAMPLE)
    changed = deepcopy(config_values(original))
    options = changed['components']['discriminator']['args']['pretrained_providers']['dinov3_vits16']
    options[key] = value
    assert fingerprint(resolve_config(changed)) != fingerprint(original)


def test_example_validation_never_imports_model_libraries(tmp_path):
    code = '''
import importlib.abc
import sys

blocked = {'torch', 'torchvision', 'hndl', 'dinov3'}
class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Configuration imported model library: ' + fullname)
sys.meta_path.insert(0, RejectModelImports())
from hypergan.config import load_config, config_values, resolve_config
from hypergan.network_config import validate_network_args
config = load_config(sys.argv[1])
validate_network_args(config['components']['discriminator']['args'], 'discriminator')
resolve_config(config_values(config))
assert not blocked.intersection(sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, str(EXAMPLE)], cwd=tmp_path, check=True)

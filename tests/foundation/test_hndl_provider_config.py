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
IMAJEV_PROVIDER = {
    'base_model_path': '/unavailable/qwen3.5-4b',
    'adapter_path': '/unavailable/imajev-4b',
    'question': 'Which digit is shown in the image?',
    'options': ['0 — zero', '1 — one', '2 — two'],
}


def network_args(providers):
    return {'source': 'linear(1)', 'input_shape': ['B', 2], 'output_shape': ['B', 1],
            'pretrained_providers': deepcopy(providers)}


def resolved_with_providers(providers):
    raw = deepcopy(DEFAULT)
    raw['components']['discriminator'] = {
        'factory': 'hndl', 'inputs': {'x': 'candidate'}, 'args': network_args(providers)}
    return resolve_config(raw)


@pytest.mark.parametrize('providers', [
    {},
    {'dinov3_vits16': PROVIDER},
    {'imajev_4b': IMAJEV_PROVIDER},
    {'imajev_4b': {**IMAJEV_PROVIDER, 'state': {'domain': 'mnist'}, 'temperature': 1.5}},
    {'imajev_4b': {**IMAJEV_PROVIDER, 'state': 'raw-context', 'temperature': 2}},
    {'dinov3_vits16': PROVIDER, 'imajev_4b': IMAJEV_PROVIDER},
])
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
    ({'imajev_4b': None}, 'imajev_4b requires base_model_path, adapter_path, question and options'),
    ({'imajev_4b': {}}, 'imajev_4b requires base_model_path, adapter_path, question and options'),
    *[({'imajev_4b': {k: v for k, v in IMAJEV_PROVIDER.items() if k != missing}},
       'imajev_4b requires base_model_path, adapter_path, question and options')
      for missing in ('base_model_path', 'adapter_path', 'question', 'options')],
    ({'imajev_4b': {**IMAJEV_PROVIDER, 'extra': True}},
     'imajev_4b requires base_model_path, adapter_path, question and options'),
    *[({'imajev_4b': {**IMAJEV_PROVIDER, field: bad}}, f'imajev_4b {field} must be nonempty text')
      for field in ('base_model_path', 'adapter_path', 'question')
      for bad in ('', '   ', 42, None)],
    *[({'imajev_4b': {**IMAJEV_PROVIDER, 'options': bad}}, '2 to 63 nonempty strings')
      for bad in (None, 'ab', [], ['only_one'], [f'opt_{i}' for i in range(64)], ['a', ''], ['a', '  '], ['a', 1])],
    ({'imajev_4b': {**IMAJEV_PROVIDER, 'options': ['yes', '__unknown__']}}, '__unknown__'),
    ({'imajev_4b': {**IMAJEV_PROVIDER, 'options': ['dup', 'dup']}}, 'unique'),
    *[({'imajev_4b': {**IMAJEV_PROVIDER, 'state': bad}}, 'state must be')
      for bad in (123, ['list'], {'nan': float('nan')}, {'inf': float('inf')})],
    *[({'imajev_4b': {**IMAJEV_PROVIDER, 'temperature': bad}}, 'temperature must be a finite positive number')
      for bad in (0, -1.0, False, True, '1.0', None, float('nan'), float('inf'))],
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


@pytest.mark.parametrize('key,value', [
    ('question', 'Different question?'),
    ('options', ['0 — zero', '1 — one']),
    ('state', {'mode': 'strict'}),
    ('temperature', 2.0),
])
def test_imajev_provider_changes_numerical_fingerprint(key, value):
    base = resolved_with_providers({'imajev_4b': IMAJEV_PROVIDER})
    changed_cfg = deepcopy(IMAJEV_PROVIDER)
    changed_cfg[key] = value
    changed = resolved_with_providers({'imajev_4b': changed_cfg})
    assert fingerprint(changed) != fingerprint(base)


def test_example_validation_never_imports_model_libraries(tmp_path):
    code = '''
import importlib.abc
import sys

blocked = {'torch', 'torchvision', 'hndl', 'dinov3', 'transformers'}
class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Configuration imported model library: ' + fullname)
sys.meta_path.insert(0, RejectModelImports())
from hypergan.config import load_config, config_values, resolve_config
from hypergan.network_config import validate_network_args
config = load_config(sys.argv[1])
validate_network_args(config['components']['discriminator']['args'], 'discriminator')
validate_network_args({
    'source': 'linear(1)',
    'input_shape': ['B', 2],
    'output_shape': ['B', 1],
    'pretrained_providers': {
        'imajev_4b': {
            'base_model_path': '/path/to/qwen',
            'adapter_path': '/path/to/imajev',
            'question': 'Is this valid?',
            'options': ['yes', 'no'],
            'state': {'k': 'v'},
            'temperature': 1.0,
        }
    },
}, 'discriminator')
resolve_config(config_values(config))
assert not blocked.intersection(sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, str(EXAMPLE)], cwd=tmp_path, check=True)

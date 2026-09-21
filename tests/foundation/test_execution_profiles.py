"""Execution-profile structure stays available without numerical dependencies."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from hypergan.config import resolve_config
from hypergan.execution_profiles import (
    MAX_PROFILE_BYTES, load_execution_profile, resolve_execution_profile, validate_checkpoint_kind,
)


def raw(name='cpu-single', **execution):
    return {'schema_version': 1, 'execution': {'name': name, **execution}}


def resolve(value):
    return resolve_execution_profile(value, resolve_config({}))


def test_replicated_recipe_rejects_extra_adversarial_terms():
    from hypergan.execution_profiles import validate_replicated_recipe
    legacy = resolve_config({})
    validate_replicated_recipe(legacy)
    extra = resolve_config({"adversarial_terms": [{
        "id": "extra", "component": "discriminator", "real": "batch.real", "fake": "generated"}]})
    with pytest.raises(ValueError, match="adversarial_terms"):
        validate_replicated_recipe(extra)


def test_single_defaults_and_replicated_effective_batch():
    single = resolve(raw())
    assert single == {'schema_version': 1, 'execution': {
        'name': 'cpu-single', 'world_size': 1, 'accumulation_steps': 1,
        'global_batch_size': 16, 'local_batch_size': 16, 'microbatch_size': 16,
        'accumulation_algorithm': 'retained-local-graph-v1'},
        'preflight': {'timeout': 60.0, 'collective_timeout': 15.0}}
    replicated = resolve(raw('cpu-replicated-gloo', world_size=2, accumulation_steps=4))
    assert replicated['execution'] == {
        'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 4,
        'global_batch_size': 16, 'local_batch_size': 8, 'microbatch_size': 2,
        'accumulation_algorithm': 'detached-logit-vjp-replay-v1'}
    assert resolve(raw('cpu-replicated-gloo'))['execution']['world_size'] == 2
    assert resolve(raw('cpu-replicated-gloo'))['execution']['accumulation_algorithm'] == 'retained-local-graph-v1'


def test_policy_changes_do_not_change_identity_or_mutate_inputs():
    values = raw('cpu-replicated-gloo', accumulation_steps=2)
    config = resolve_config({})
    original_values, original_config = deepcopy(values), deepcopy(config)
    baseline = resolve_execution_profile(values, config)
    values['preflight'] = {'timeout': 100, 'collective_timeout': 25}
    changed = resolve_execution_profile(values, config)
    assert baseline['execution'] == changed['execution']
    assert baseline['preflight'] != changed['preflight']
    assert config == original_config
    assert values == dict(original_values, preflight={'timeout': 100, 'collective_timeout': 25})
    assert json.loads(json.dumps(changed)) == changed
    assert resolve(raw('cpu-replicated-gloo', accumulation_steps=4))['execution'] != baseline['execution']


@pytest.mark.parametrize('value,match', [
    ([], 'table'), ({}, 'Missing profile'),
    ({'schema_version': True, 'execution': {'name': 'cpu-single'}}, 'schema_version'),
    ({'schema_version': 2, 'execution': {'name': 'cpu-single'}}, 'schema_version'),
    ({'schema_version': 1, 'execution': {}}, 'Missing execution'),
    ({'schema_version': 1, 'execution': []}, 'table'),
    ({**raw(), 'unexpected': 1}, 'Unknown profile'),
    ({**raw(), 'preflight': []}, 'table'),
    ({**raw(), 'preflight': {'retry': 3}}, 'Unknown preflight'),
    (raw('cuda'), 'execution.name'), (raw(None), 'execution.name'),
    (raw(device='cpu'), 'Unknown execution'), (raw(global_batch_size=16), 'Unknown execution'),
    (raw(world_size=True), 'positive integer'), (raw(world_size=1.0), 'positive integer'),
    (raw(world_size=0), 'positive integer'), (raw(world_size=2), 'cpu-single requires'),
    (raw(accumulation_steps=2), 'cpu-single requires'),
    (raw('cpu-replicated-gloo', world_size=1), 'between 2 and 64'),
    (raw('cpu-replicated-gloo', world_size=65), 'at most 64'),
    (raw('cpu-replicated-gloo', world_size=3), 'Global training.batch_size'),
    (raw('cpu-replicated-gloo', accumulation_steps=3), 'local_batch_size'),
    (raw('cpu-replicated-gloo', accumulation_steps=16), 'local_batch_size'),
    (raw('cpu-replicated-gloo', accumulation_steps=False), 'positive integer'),
    (raw('cpu-replicated-gloo', accumulation_steps=-1), 'positive integer'),
])
def test_invalid_profile_structure_and_options(value, match):
    with pytest.raises(ValueError, match=match):
        resolve(value)


@pytest.mark.parametrize('field', ['timeout', 'collective_timeout'])
@pytest.mark.parametrize('value', [True, '15', 0, -1, float('nan'), float('inf'), 10 ** 1000])
def test_limits_are_finite_positive_seconds(field, value):
    with pytest.raises(ValueError, match='finite positive seconds'):
        resolve({**raw(), 'preflight': {field: value}})


def test_timeout_order_and_boundary_world_size():
    with pytest.raises(ValueError, match='must not exceed'):
        resolve({**raw(), 'preflight': {'timeout': 10}})
    assert resolve({**raw(), 'preflight': {'timeout': 0.5, 'collective_timeout': 0.5}})['preflight']['timeout'] == 0.5
    config = resolve_config({'training': {'batch_size': 128}})
    assert resolve_execution_profile(raw('cpu-replicated-gloo', world_size=64, accumulation_steps=2), config)['execution']['microbatch_size'] == 1


@pytest.mark.parametrize('config', [{}, {'training': []}, {'training': {'device': 'cuda', 'batch_size': 16}},
                                  {'training': {'device': 'cpu', 'batch_size': True}},
                                  {'training': {'device': 'cpu', 'batch_size': 0}}])
def test_required_recipe_execution_fields(config):
    with pytest.raises(ValueError):
        resolve_execution_profile(raw(), config)


def test_toml_loading_and_clean_parse_errors(tmp_path):
    path = tmp_path / 'execution.toml'
    path.write_text('schema_version=1\n[execution]\nname="cpu-replicated-gloo"\naccumulation_steps=2\n', encoding='utf-8')
    assert load_execution_profile(path, resolve_config({})) == resolve(raw('cpu-replicated-gloo', accumulation_steps=2))
    for content in (b'not TOML', b'\xff', b'[execution]\nname="cpu-single"\nname="cpu-single"'):
        path.write_bytes(content)
        with pytest.raises(ValueError, match='Invalid execution profile TOML'):
            load_execution_profile(path, resolve_config({}))
    path.write_bytes(b'#' * (MAX_PROFILE_BYTES + 1))
    with pytest.raises(ValueError, match='exceeds'):
        load_execution_profile(path, resolve_config({}))


def test_checkpoint_kind_routing_is_explicit():
    single, replicated = resolve(raw()), resolve(raw('cpu-replicated-gloo'))
    native = 'hypergan-training-checkpoint'
    distributed = 'hypergan-distributed-training-checkpoint'
    assert validate_checkpoint_kind(native, single) == native
    assert validate_checkpoint_kind(distributed, replicated) == distributed
    for kind, profile in ((native, replicated), (distributed, single)):
        with pytest.raises(ValueError, match='no implicit format conversion'):
            validate_checkpoint_kind(kind, profile)
    for profile in (single, replicated):
        with pytest.raises(ValueError, match='cannot resume training'):
            validate_checkpoint_kind('ema-inference', profile)
    for kind in (None, [], 'pytorch', 'native'):
        with pytest.raises(ValueError, match='Unknown checkpoint kind'):
            validate_checkpoint_kind(kind, single)


def test_structural_profile_imports_no_runtime_or_custom_factories(tmp_path):
    code = '''
import importlib.abc, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'particlegan', 'custom_factory'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from copy import deepcopy
from hypergan.config import DEFAULT, resolve_config
from hypergan.execution_profiles import resolve_execution_profile
values = deepcopy(DEFAULT)
values['components']['generator']['factory'] = 'custom_factory:Generator'
config = resolve_config(values)
profile = resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}}, config)
assert profile['execution']['global_batch_size'] == 16
'''
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True, timeout=15)


def test_cuda_replicated_identity_and_explicit_device_ownership():
    config = resolve_config({'training': {'device': 'cuda', 'batch_size': 32}})
    profile = resolve_execution_profile(raw('cuda-replicated-nccl', accumulation_steps=4), config)
    assert profile['execution'] == {
        'name': 'cuda-replicated-nccl', 'world_size': 2, 'accumulation_steps': 4,
        'global_batch_size': 32, 'local_batch_size': 16, 'microbatch_size': 4,
        'accumulation_algorithm': 'detached-logit-vjp-replay-v1'}
    assert validate_checkpoint_kind('hypergan-distributed-training-checkpoint', profile)
    for device in ('cpu', 'cuda:0', 'cuda:1'):
        with pytest.raises(ValueError, match='training.device=cuda'):
            resolve_execution_profile(raw('cuda-replicated-nccl'),
                resolve_config({'training': {'device': device}}))
    with pytest.raises(ValueError, match='training.device=cpu'):
        resolve_execution_profile(raw('cpu-replicated-gloo'), config)
    with pytest.raises(ValueError, match='between 2 and 64'):
        resolve_execution_profile(raw('cuda-replicated-nccl', world_size=1), config)
    with pytest.raises(ValueError, match='local_batch_size'):
        resolve_execution_profile(raw('cuda-replicated-nccl', accumulation_steps=3), config)


def test_cuda_structural_profile_and_adapter_import_no_torch(tmp_path):
    code = '''
import importlib.abc, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'particlegan'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from hypergan.config import resolve_config
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.execution_preflight import _resolve_profile
from hypergan.replicated_execution import ReplicatedExecutionFactory
config = resolve_config({'training': {'device': 'cuda'}})
profile = resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cuda-replicated-nccl'}}, config)
assert _resolve_profile(profile, config) == profile
adapter = ReplicatedExecutionFactory(profile)(config)
assert adapter.environment()['runtime'] == {'device': 'cuda', 'backend': 'nccl', 'runtime_checked': False}
adapter.shutdown()
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', code],
                   cwd=tmp_path, check=True, timeout=15)

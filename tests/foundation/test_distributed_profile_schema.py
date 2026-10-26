"""Distributed execution-profile schema: topology, rendezvous, aggregation and re-projection.

The base profile contract (defaults, structure errors, canonical aggregation
resolution, TOML parsing, torch-free imports) is in test_execution_profiles.py.
This file holds the multi-node, rendezvous and cross-feature boundaries.
"""
import copy

import pytest

from hypergan.config import resolve_config
from hypergan.execution_preflight import _resolve_profile
from hypergan.execution_profiles import _validate_rendezvous, load_execution_profile, resolve_execution_profile


def _raw(name='cpu-replicated-gloo', aggregation=None, **execution):
    if execution.get('nodes', 1) > 1:
        # Multi-node is experimental and requires an explicit tcp:// rendezvous.
        execution.setdefault('rendezvous', 'tcp://node0.cluster:29500')
    profile = {'schema_version': 1, 'execution': {'name': name, **execution}}
    if aggregation is not None:
        profile['aggregation'] = aggregation
    return profile


def _config(batch_size=64, device='cpu'):
    return resolve_config({'training': {'batch_size': batch_size, 'device': device}})


# ==============================================================================
# Topology
# ==============================================================================

@pytest.mark.parametrize('world_size,nodes,node_rank,ranks_per_node', [
    (8, 4, 2, None), (64, 64, 0, None), (64, 64, 63, None), (64, 64, 0, 1), (8, 2, 0, 4),
])
def test_multinode_topology_derives_ranks_per_node(world_size, nodes, node_rank, ranks_per_node):
    extra = {} if ranks_per_node is None else {'ranks_per_node': ranks_per_node}
    execution = resolve_execution_profile(_raw(world_size=world_size, nodes=nodes, node_rank=node_rank, **extra),
                                          _config())['execution']
    assert (execution['nodes'], execution['node_rank']) == (nodes, node_rank)
    assert execution['ranks_per_node'] == world_size // nodes
    assert execution['local_batch_size'] == 64 // world_size
    assert execution['rendezvous'] == 'tcp://node0.cluster:29500'


@pytest.mark.parametrize('profile,batch_size,match', [
    (_raw(world_size=64, nodes=64, node_rank=64), 64, 'execution.node_rank must be between 0 and nodes - 1'),
    (_raw(world_size=64, nodes=64, node_rank=-1), 64, 'execution.node_rank must be a non-negative integer'),
    (_raw(world_size=64, nodes=64, ranks_per_node=2), 64, 'must equal world_size // nodes'),
    (_raw(world_size=32, nodes=64), 64, 'execution.nodes cannot exceed execution.world_size'),
    (_raw(world_size=65, nodes=65), 128, 'execution.world_size must be at most 64'),
    (_raw(world_size=64, nodes=64), 32, 'Global training.batch_size must divide evenly'),
    (_raw(world_size=3, nodes=2), 12, 'divide evenly across execution.nodes'),
    (_raw(world_size=2), 7, 'divide evenly'),
    (_raw('cpu-single', accumulation_steps=0), 8, 'positive integer'),
])
def test_topology_rejections(profile, batch_size, match):
    with pytest.raises(ValueError, match=match):
        resolve_execution_profile(profile, _config(batch_size))


# ==============================================================================
# Rendezvous
# ==============================================================================

@pytest.mark.parametrize('uri', ['tcp://127.0.0.1:1', 'tcp://127.0.0.1:65535', 'tcp://127.0.0.1:80',
                                 'tcp://127.0.0.1:443', 'tcp://my-host.internal:12345', 'file://', 'file:///tmp/rdzv'])
def test_rendezvous_boundaries_accepted(uri):
    assert _validate_rendezvous(uri) == uri
    assert resolve_execution_profile(_raw(world_size=2, rendezvous=uri), _config(16))['execution']['rendezvous'] == uri


@pytest.mark.parametrize('uri,match', [
    ('tcp://127.0.0.1:0', 'port must be between 1 and 65535: 0'),
    ('tcp://127.0.0.1:65536', 'port must be between 1 and 65535: 65536'),
    ('tcp://localhost:70000', 'port must be between 1 and 65535: 70000'),
    ('tcp://127.0.0.1:-1', 'port must be between 1 and 65535: -1'),
    ('tcp://127.0.0.1:-65535', 'port must be between 1 and 65535: -65535'),
    ('tcp://127.0.0.1:abc', 'port must be an integer: abc'),
    ('tcp://127.0.0.1:', 'port must be an integer: '),
    ('tcp://127.0.0.1', 'must be formatted as tcp://<host>:<port>'),
    ('tcp://:29500', 'must have a non-empty host'),
    ('tcp://', 'must be formatted as tcp://<host>:<port>'),
    ('udp://127.0.0.1:29500', 'must be a file:// or tcp:// URI'),
    ('http://127.0.0.1:29500', 'must be a file:// or tcp:// URI'),
    ('redis://localhost:6379', 'must be a file:// or tcp:// URI'),
    ('', 'non-empty string'),
    (12345, 'non-empty string'),
])
def test_invalid_rendezvous_rejected(uri, match):
    with pytest.raises(ValueError, match=match):
        _validate_rendezvous(uri)
    with pytest.raises(ValueError, match=match):
        resolve_execution_profile(_raw(world_size=2, rendezvous=uri), _config(16))


def test_single_device_profile_rejects_tcp_rendezvous():
    with pytest.raises(ValueError, match='cpu-single requires file:// rendezvous; tcp:// is unsupported for single-device'):
        resolve_execution_profile(_raw('cpu-single', rendezvous='tcp://127.0.0.1:29500'), _config(4))


# ==============================================================================
# Aggregation and cross-feature combinations
# ==============================================================================

@pytest.mark.parametrize('aggregation,match', [
    ({'strategy': 'pseudo_gradient', 'outer_lr': float('inf')}, 'outer_lr'),
    ({'strategy': 'pseudo_gradient', 'outer_lr': float('-inf')}, 'outer_lr'),
    ({'strategy': 'pseudo_gradient', 'outer_lr': float('nan')}, 'outer_lr'),
    ({'strategy': 'pseudo_gradient', 'outer_momentum': 1.05}, 'outer_momentum'),
])
def test_nonfinite_or_out_of_range_outer_settings_rejected(aggregation, match):
    with pytest.raises(ValueError, match=match):
        resolve_execution_profile(_raw(world_size=2, aggregation=aggregation), _config(8))


@pytest.mark.parametrize('bad_version', [0, '1', None])
def test_schema_version_must_be_one(bad_version):
    with pytest.raises(ValueError, match='schema_version'):
        resolve_execution_profile({'schema_version': bad_version, 'execution': {'name': 'cpu-single'}}, _config(8))


@pytest.mark.parametrize('profile,device,expected', [
    (_raw(world_size=2, accumulation_steps=2, aggregation={'strategy': 'parameter_average', 'sync_every_steps': 2}), 'cpu',
     {'microbatch_size': 4, 'accumulation_algorithm': 'detached-logit-vjp-replay-v1', 'strategy': 'parameter_average'}),
    (_raw(world_size=2, accumulation_steps=2,
          aggregation={'strategy': 'pseudo_gradient', 'sync_every_steps': 4, 'outer_lr': 0.8, 'outer_momentum': 0.5}), 'cpu',
     {'accumulation_steps': 2, 'strategy': 'pseudo_gradient', 'sync_every_steps': 4}),
    (_raw(world_size=2, nodes=2, rendezvous='tcp://127.0.0.1:29500',
          aggregation={'strategy': 'pseudo_gradient', 'sync_every_steps': 4, 'outer_lr': 0.9, 'outer_momentum': 0.9}), 'cpu',
     {'ranks_per_node': 1, 'rendezvous': 'tcp://127.0.0.1:29500', 'strategy': 'pseudo_gradient'}),
    (_raw(world_size=4, nodes=2, node_rank=1, rendezvous='tcp://127.0.0.1:29500',
          aggregation={'strategy': 'gradient_allreduce', 'sync_every_steps': 1}), 'cpu',
     {'ranks_per_node': 2, 'local_batch_size': 4, 'strategy': 'gradient_allreduce'}),
    (_raw(world_size=2, aggregation={'strategy': 'pseudo_gradient', 'sync_every_steps': 100000, 'outer_momentum': 0.0}), 'cpu',
     {'sync_every_steps': 100000, 'outer_momentum': 0.0}),
    (_raw('cuda-replicated-nccl', world_size=2, accumulation_steps=4), 'cuda',
     {'local_batch_size': 8, 'microbatch_size': 2, 'accumulation_algorithm': 'detached-logit-vjp-replay-v1'}),
])
def test_feature_combinations_resolve_together(profile, device, expected):
    resolved = resolve_execution_profile(profile, _config(16, device))
    execution = resolved['execution']
    assert resolved['aggregation'] == execution['aggregation']
    for key, value in expected.items():
        assert (execution['aggregation'] if key in execution['aggregation'] else execution)[key] == value, key


def test_released_presets_resolve_on_both_replicated_backends():
    for name, device in (('cpu-replicated-gloo', 'cpu'), ('cuda-replicated-nccl', 'cuda')):
        config = resolve_config({'defaults': 'particlegan', 'training': {'device': device, 'steps': 2, 'batch_size': 8}})
        execution = resolve_execution_profile(_raw(name, world_size=2), config)['execution']
        assert (execution['name'], execution['local_batch_size']) == (name, 4)


# ==============================================================================
# Re-projection through execution_preflight and TOML loading
# ==============================================================================

@pytest.mark.parametrize('profile', [
    {'schema_version': 1, 'execution': {'name': 'cpu-single'}},
    _raw(world_size=4),
    _raw(world_size=64, nodes=64, node_rank=63, rendezvous='tcp://127.0.0.1:65535'),
    _raw(world_size=4, aggregation={'strategy': 'parameter_average', 'sync_every_steps': 8}),
    _raw(world_size=4, aggregation={'strategy': 'pseudo_gradient', 'sync_every_steps': 4, 'outer_lr': 0.7, 'outer_momentum': 0.85}),
])
def test_preflight_re_projection_of_a_resolved_profile_is_idempotent(profile):
    config = _config()
    resolved = resolve_execution_profile(profile, config)
    assert _resolve_profile(resolved, config) == resolved


@pytest.mark.parametrize('field,value', [('ranks_per_node', 999), ('global_batch_size', 999),
                                         ('accumulation_algorithm', 'forged-algorithm')])
def test_preflight_rejects_tampered_derived_fields(field, value):
    config = _config(32)
    tampered = copy.deepcopy(resolve_execution_profile(_raw(world_size=4), config))
    tampered['execution'][field] = value
    with pytest.raises(ValueError, match='Resolved profile differs from configuration'):
        _resolve_profile(tampered, config)


def test_multinode_aggregation_profile_loads_from_toml_and_re_projects(tmp_path):
    path = tmp_path / 'profile.toml'
    path.write_text('schema_version = 1\n\n[execution]\nname = "cpu-replicated-gloo"\nworld_size = 8\nnodes = 4\n'
                    'node_rank = 2\nrendezvous = "tcp://10.0.0.1:29500"\n\n[aggregation]\n'
                    'strategy = "parameter_average"\nsync_every_steps = 4\n', encoding='utf-8')
    config = _config()
    loaded = load_execution_profile(path, config)
    assert (loaded['execution']['nodes'], loaded['execution']['ranks_per_node']) == (4, 2)
    assert loaded['execution']['aggregation']['strategy'] == 'parameter_average'
    assert _resolve_profile(loaded, config) == loaded

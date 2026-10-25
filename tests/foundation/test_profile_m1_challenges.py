"""Empirical challenge probes for Milestone 1 profile parsing, schemas, and preflight drift."""
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from hypergan.config import DEFAULT, resolve_config
from hypergan.execution_profiles import load_execution_profile, resolve_execution_profile
from hypergan.execution_preflight import _resolve_profile, preflight


def _raw(name='cpu-replicated-gloo', **kwargs):
    if kwargs.get('nodes', 1) > 1:
        # Multi-node is experimental and requires an explicit tcp:// rendezvous.
        kwargs.setdefault('rendezvous', 'tcp://node0.cluster:29500')
    return {'schema_version': 1, 'execution': {'name': name, **kwargs}}


# ==============================================================================
# Challenge 1: Extreme Topologies
# ==============================================================================

def test_extreme_topology_max_nodes_and_world_size():
    """nodes = 64, world_size = 64, ranks_per_node = 1."""
    config = resolve_config({'training': {'batch_size': 64, 'device': 'cpu'}})
    profile = resolve_execution_profile(
        _raw('cpu-replicated-gloo', world_size=64, nodes=64, node_rank=0),
        config
    )
    assert profile['execution']['world_size'] == 64
    assert profile['execution']['nodes'] == 64
    assert profile['execution']['node_rank'] == 0
    assert profile['execution']['ranks_per_node'] == 1
    assert profile['execution']['local_batch_size'] == 1
    assert profile['execution']['microbatch_size'] == 1
    assert profile['execution']['rendezvous'] == 'tcp://node0.cluster:29500'


def test_extreme_topology_last_node_rank():
    """Verify max valid node_rank = 63 when nodes = 64."""
    config = resolve_config({'training': {'batch_size': 64, 'device': 'cpu'}})
    profile = resolve_execution_profile(
        _raw('cpu-replicated-gloo', world_size=64, nodes=64, node_rank=63),
        config
    )
    assert profile['execution']['node_rank'] == 63


def test_extreme_topology_explicit_ranks_per_node():
    """Explicit ranks_per_node matching world_size // nodes must be accepted."""
    config = resolve_config({'training': {'batch_size': 64, 'device': 'cpu'}})
    profile = resolve_execution_profile(
        _raw('cpu-replicated-gloo', world_size=64, nodes=64, node_rank=0, ranks_per_node=1),
        config
    )
    assert profile['execution']['ranks_per_node'] == 1


def test_extreme_topology_rejects_out_of_bounds_and_mismatches():
    """Test boundary violations for extreme topology configurations."""
    config = resolve_config({'training': {'batch_size': 64, 'device': 'cpu'}})

    # node_rank == nodes (boundary violation: must be < nodes)
    with pytest.raises(ValueError, match='execution.node_rank must be between 0 and nodes - 1'):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=64, nodes=64, node_rank=64), config)

    # Negative node_rank
    with pytest.raises(ValueError, match='execution.node_rank must be a non-negative integer'):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=64, nodes=64, node_rank=-1), config)

    # Explicit ranks_per_node mismatch
    with pytest.raises(ValueError, match='must equal world_size // nodes'):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=64, nodes=64, ranks_per_node=2), config)

    # nodes > world_size
    with pytest.raises(ValueError, match='execution.nodes cannot exceed execution.world_size'):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=32, nodes=64), config)

    # world_size > 64 boundary
    with pytest.raises(ValueError, match='execution.world_size must be at most 64'):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=65, nodes=65), config)

    # Batch size not divisible by world_size = 64
    config_small_batch = resolve_config({'training': {'batch_size': 32, 'device': 'cpu'}})
    with pytest.raises(ValueError, match='Global training.batch_size must divide evenly'):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=64, nodes=64), config_small_batch)


# ==============================================================================
# Challenge 2: Boundary Port Numbers
# ==============================================================================

@pytest.mark.parametrize('port', [1, 65535, 80, 443, 29500])
def test_boundary_port_numbers_accepted(port):
    """Port 1 and 65535 are exact boundaries of valid port numbers."""
    config = resolve_config({'training': {'batch_size': 16, 'device': 'cpu'}})
    uri = f'tcp://127.0.0.1:{port}'
    profile = resolve_execution_profile(
        _raw('cpu-replicated-gloo', world_size=2, rendezvous=uri),
        config
    )
    assert profile['execution']['rendezvous'] == uri


@pytest.mark.parametrize('bad_uri,match', [
    ('tcp://127.0.0.1:0', 'port must be between 1 and 65535: 0'),
    ('tcp://127.0.0.1:65536', 'port must be between 1 and 65535: 65536'),
    ('tcp://127.0.0.1:-1', 'port must be between 1 and 65535: -1'),
    ('tcp://127.0.0.1:-65535', 'port must be between 1 and 65535: -65535'),
    ('tcp://127.0.0.1:abc', 'port must be an integer: abc'),
    ('tcp://127.0.0.1:', 'port must be an integer: '),
    ('tcp://127.0.0.1', 'must be formatted as tcp://<host>:<port>'),
    ('tcp://:29500', 'must have a non-empty host'),
    ('tcp://', 'must be formatted as tcp://<host>:<port>'),
    ('udp://127.0.0.1:29500', 'must be a file:// or tcp:// URI'),
    ('http://127.0.0.1:29500', 'must be a file:// or tcp:// URI'),
    ('', 'non-empty string'),
    (12345, 'non-empty string'),
])
def test_invalid_rendezvous_ports_and_schemes_rejected(bad_uri, match):
    """Invalid ports (0, 65536, negative, malformed) must raise ValueError."""
    config = resolve_config({'training': {'batch_size': 16, 'device': 'cpu'}})
    with pytest.raises(ValueError, match=match):
        resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=2, rendezvous=bad_uri), config)


# ==============================================================================
# Challenge 3: Re-projection Drift through execution_preflight.py
# ==============================================================================

def test_preflight_resolve_profile_idempotency_all_strategies():
    """Verify that re-projecting an already-resolved profile produces zero configuration drift."""
    config = resolve_config({'training': {'batch_size': 64, 'device': 'cpu'}})
    cases = [
        # Single profile
        {'schema_version': 1, 'execution': {'name': 'cpu-single'}},
        # Replicated with default gradient_allreduce
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 4}},
        # Extreme topology
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 64, 'nodes': 64,
                                            'node_rank': 63, 'rendezvous': 'tcp://127.0.0.1:65535'}},
        # Parameter average
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 4},
         'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 8}},
        # Pseudo gradient
        {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 4},
         'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 4, 'outer_lr': 0.7, 'outer_momentum': 0.85}},
    ]

    for raw_profile in cases:
        resolved = resolve_execution_profile(raw_profile, config)
        re_projected = _resolve_profile(resolved, config)
        assert re_projected == resolved, f"Drift detected for {raw_profile}!\nResolved: {resolved}\nRe-projected: {re_projected}"


def test_preflight_resolve_profile_detects_derived_field_tampering():
    """If derived fields in a resolved profile were tampered with, _resolve_profile must reject them."""
    config = resolve_config({'training': {'batch_size': 32, 'device': 'cpu'}})
    resolved = resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=4), config)

    # Tamper with ranks_per_node
    tampered_rpn = copy.deepcopy(resolved)
    tampered_rpn['execution']['ranks_per_node'] = 999
    with pytest.raises(ValueError, match='Resolved profile differs from configuration'):
        _resolve_profile(tampered_rpn, config)

    # Tamper with global_batch_size
    tampered_batch = copy.deepcopy(resolved)
    tampered_batch['execution']['global_batch_size'] = 999
    with pytest.raises(ValueError, match='Resolved profile differs from configuration'):
        _resolve_profile(tampered_batch, config)

    # Tamper with accumulation_algorithm
    tampered_algo = copy.deepcopy(resolved)
    tampered_algo['execution']['accumulation_algorithm'] = 'forged-algorithm'
    with pytest.raises(ValueError, match='Resolved profile differs from configuration'):
        _resolve_profile(tampered_algo, config)


# ==============================================================================
# Challenge 4: Structural Profile Imports Zero Runtime or Custom Factories
# ==============================================================================

def test_structural_profile_imports_no_runtime_or_custom_factories(tmp_path):
    """Confirm zero torch/numpy/PIL/particlegan/custom_factory imports during profile loading and resolution."""
    code = '''
import importlib.abc, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'particlegan', 'custom_factory'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())

from copy import deepcopy
from pathlib import Path
from hypergan.config import DEFAULT, resolve_config
from hypergan.execution_profiles import load_execution_profile, resolve_execution_profile
from hypergan.execution_preflight import _resolve_profile

values = deepcopy(DEFAULT)
values['components']['generator']['factory'] = 'custom_factory:Generator'
config = resolve_config(values)

# Test resolve_execution_profile with extreme multi-node and aggregation settings
profile_dict = {
    'schema_version': 1,
    'execution': {
        'name': 'cpu-replicated-gloo',
        'world_size': 16,
        'nodes': 16,
        'node_rank': 15,
        'rendezvous': 'tcp://127.0.0.1:65535',
        'accumulation_steps': 1
    },
    'aggregation': {
        'strategy': 'pseudo_gradient',
        'sync_every_steps': 5,
        'outer_lr': 0.5,
        'outer_momentum': 0.9
    }
}
profile = resolve_execution_profile(profile_dict, config)
assert profile['execution']['nodes'] == 16
assert profile['execution']['ranks_per_node'] == 1
assert profile['execution']['aggregation']['strategy'] == 'pseudo_gradient'

# Test load_execution_profile from TOML file
toml_path = Path("test_profile.toml")
toml_path.write_text("""
schema_version = 1

[execution]
name = "cpu-replicated-gloo"
world_size = 8
nodes = 4
node_rank = 2
rendezvous = "tcp://10.0.0.1:29500"

[aggregation]
strategy = "parameter_average"
sync_every_steps = 4
""")

loaded_profile = load_execution_profile(toml_path, config)
assert loaded_profile['execution']['nodes'] == 4
assert loaded_profile['execution']['ranks_per_node'] == 2
assert loaded_profile['execution']['aggregation']['strategy'] == 'parameter_average'

# Test _resolve_profile re-projection
re_projected = _resolve_profile(loaded_profile, config)
assert re_projected == loaded_profile

print("STRUCTURAL_IMPORT_TEST_PASSED")
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STRUCTURAL_IMPORT_TEST_PASSED" in result.stdout


# ==============================================================================
# Challenge 5: Strategy Info and Preflight Drift Regression Probe
# ==============================================================================

def test_preflight_and_worker_strategy_info_keys_divergence():
    """Empirical probe demonstrating drift between profile['execution'] and trainer.strategy_info.

    When canonical keys ('nodes', 'node_rank', 'ranks_per_node', 'rendezvous', 'aggregation')
    were added to execution profiles, trainer.strategy_info was not synchronized.
    This causes _runtime_worker in execution_preflight.py and create_worker in
    replicated_worker.py to fail on all replicated execution profiles.
    """
    from hypergan.execution_preflight import _differences

    config = resolve_config({'training': {'batch_size': 16, 'device': 'cpu'}})
    profile = resolve_execution_profile(_raw('cpu-replicated-gloo', world_size=2), config)
    requested = profile['execution']

    # As set in ReplicatedTrainer.__init__ (distributed_training.py:65):
    strategy_info_sample = {
        'name': 'cpu-replicated-gloo', 'world_size': 2,
        'global_batch_size': 16, 'local_batch_size': 8,
        'accumulation_steps': 1, 'microbatch_size': 8,
        'accumulation_algorithm': 'retained-local-graph-v1',
        'gradient_reduction': 'post-backward-mean',
        'buffers': 'require-replica-equality',
        'data': 'replicated-global-draw-rank-slice',
        'qualification': 'unqualified'
    }
    observed = {key: strategy_info_sample.get(key) for key in requested}
    diffs = _differences(requested, observed, 'execution')

    # The missing keys in strategy_info directly cause preflight failure:
    assert 'execution.aggregation (type differs)' in diffs
    assert 'execution.node_rank (type differs)' in diffs
    assert 'execution.nodes (type differs)' in diffs
    assert 'execution.ranks_per_node (type differs)' in diffs
    assert 'execution.rendezvous (type differs)' in diffs

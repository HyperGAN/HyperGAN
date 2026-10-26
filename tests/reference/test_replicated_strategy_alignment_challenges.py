"""Empirical challenge suite for ReplicatedTrainer strategy_info and replicated_worker.

Verifies:
1. ReplicatedTrainer instantiation with no kwargs (legacy defaults).
2. ReplicatedTrainer instantiation with explicit nodes/node_rank/ranks_per_node/rendezvous kwargs.
3. ReplicatedTrainer instantiation with execution dictionary.
4. ReplicatedCPUTrainer instantiation with kwargs forwarded to super().__init__.
5. Exact strategy_info dictionary alignment against resolved execution profiles.
6. replicated_worker.create_worker execution with synthetic profiles without false rejection.
7. replicated_worker.create_worker strict rejection of tampered execution profiles.
"""
import copy
from pathlib import Path
import pytest
import torch

from hypergan.config import config_values, resolve_config
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_training import ReplicatedCPUTrainer, ReplicatedTrainer
from hypergan.execution_preflight import _differences
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.replicated_worker import create_worker


def _make_base_config():
    return {
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 20, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': 16, 'device': 'cpu'},
    }


# ---------------------------------------------------------------------------
# Multi-Process Worker Callbacks
# ---------------------------------------------------------------------------

def _worker_legacy_instantiation(rank, world_size, config_dict, out_dir):
    """Test 1: Legacy instantiation with no kwargs defaults correctly."""
    trainer = ReplicatedTrainer(config_dict, world_size=world_size)

    # Verify attributes
    assert trainer.nodes == 1
    assert trainer.node_rank == 0
    assert trainer.ranks_per_node == world_size
    assert trainer.rendezvous == 'file://'
    assert trainer.aggregation == {
        'strategy': 'gradient_allreduce', 'sync_every_steps': 1,
        'outer_lr': 1.0, 'outer_momentum': 0.0,
    }

    # Verify strategy_info keys & values
    info = trainer.strategy_info
    assert info['nodes'] == 1
    assert info['node_rank'] == 0
    assert info['ranks_per_node'] == world_size
    assert info['rendezvous'] == 'file://'
    assert info['aggregation'] == {
        'strategy': 'gradient_allreduce', 'sync_every_steps': 1,
        'outer_lr': 1.0, 'outer_momentum': 0.0,
    }
    assert info['world_size'] == world_size
    assert info['accumulation_steps'] == 1
    assert info['global_batch_size'] == 16
    assert info['local_batch_size'] == 8
    assert info['microbatch_size'] == 8
    assert info['accumulation_algorithm'] == 'retained-local-graph-v1'


def _worker_explicit_kwargs_instantiation(rank, world_size, config_dict, out_dir):
    """Test 2: Instantiation with explicit multi-node kwargs."""
    rdzv = 'tcp://127.0.0.1:29500'
    custom_agg = {
        'strategy': 'parameter_average', 'sync_every_steps': 2,
        'outer_lr': 0.8, 'outer_momentum': 0.5,
    }
    trainer = ReplicatedTrainer(
        config_dict,
        world_size=world_size,
        nodes=2,
        node_rank=rank,
        ranks_per_node=1,
        rendezvous=rdzv,
        aggregation=custom_agg,
    )

    assert trainer.nodes == 2
    assert trainer.node_rank == rank
    assert trainer.ranks_per_node == 1
    assert trainer.rendezvous == rdzv
    assert trainer.aggregation == custom_agg

    info = trainer.strategy_info
    assert info['nodes'] == 2
    assert info['node_rank'] == rank
    assert info['ranks_per_node'] == 1
    assert info['rendezvous'] == rdzv
    assert info['aggregation'] == custom_agg


def _worker_execution_dict_instantiation(rank, world_size, config_dict, profile_execution, out_dir):
    """Test 3: Instantiation passing execution dictionary from resolved profile."""
    trainer = ReplicatedTrainer(
        config_dict,
        world_size=world_size,
        execution=profile_execution,
    )

    # Check that trainer attributes and strategy_info match execution dict exactly
    for key, expected in profile_execution.items():
        assert trainer.strategy_info[key] == expected, f"Mismatch for strategy_info key {key}"

    # Also check preflight _differences
    diffs = _differences(profile_execution, {k: trainer.strategy_info.get(k) for k in profile_execution}, 'execution')
    assert not diffs, f"Unexpected differences: {diffs}"


def _worker_cpu_trainer_kwargs(rank, world_size, config_dict, profile_execution, out_dir):
    """Test 4: ReplicatedCPUTrainer receives and forwards kwargs to super().__init__."""
    # 4a: legacy with no kwargs
    cpu_legacy = ReplicatedCPUTrainer(config_dict, world_size=world_size)
    assert cpu_legacy.nodes == 1
    assert cpu_legacy.node_rank == 0
    assert cpu_legacy.strategy_info['nodes'] == 1
    assert cpu_legacy.strategy_info['rendezvous'] == 'file://'

    # 4b: explicit kwargs
    cpu_explicit = ReplicatedCPUTrainer(
        config_dict,
        world_size=world_size,
        nodes=2,
        node_rank=rank,
        ranks_per_node=1,
        rendezvous='tcp://127.0.0.1:29500',
    )
    assert cpu_explicit.nodes == 2
    assert cpu_explicit.node_rank == rank
    assert cpu_explicit.ranks_per_node == 1
    assert cpu_explicit.strategy_info['nodes'] == 2
    assert cpu_explicit.strategy_info['rendezvous'] == 'tcp://127.0.0.1:29500'

    # 4c: with execution dict
    cpu_exec = ReplicatedCPUTrainer(
        config_dict,
        world_size=world_size,
        execution=profile_execution,
    )
    for key, val in profile_execution.items():
        assert cpu_exec.strategy_info[key] == val, f"CPU trainer strategy_info[{key}] mismatch"


def _worker_create_worker_synthetic_profiles(rank, world_size, config_dict, profiles, out_dir):
    """Test 5 & 6: replicated_worker.create_worker works for all valid synthetic profiles and rejects tampered ones."""
    context = {'test_context': True}

    for idx, profile in enumerate(profiles):
        execution = profile['execution']
        # Valid profile must never raise:
        worker_state = create_worker(rank, world_size, config_dict, execution, context)
        assert worker_state['rank'] == rank
        assert worker_state['batch'] is None
        assert worker_state['context'] == context
        assert worker_state['trainer'] is not None

        # Verify all execution keys match strategy_info
        for k, v in execution.items():
            assert worker_state['trainer'].strategy_info[k] == v

        # Adversarial check: tampering strategy name, batch size or adding unknown keys MUST raise ValueError
        tampered = copy.deepcopy(execution)
        tampered['name'] = 'tampered-device-strategy'
        with pytest.raises(ValueError, match='Actual replicated strategy differs from resolved execution profile'):
            create_worker(rank, world_size, config_dict, tampered, context)

        tampered2 = copy.deepcopy(execution)
        tampered2['local_batch_size'] = 999
        with pytest.raises(ValueError, match='Actual replicated strategy differs from resolved execution profile'):
            create_worker(rank, world_size, config_dict, tampered2, context)

        tampered3 = copy.deepcopy(execution)
        tampered3['extra_drift_key'] = 'drift'
        with pytest.raises(ValueError, match='Actual replicated strategy differs from resolved execution profile'):
            create_worker(rank, world_size, config_dict, tampered3, context)


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------

@pytest.mark.heavy
def test_replicated_trainer_legacy_instantiation(tmp_path):
    """Verify ReplicatedTrainer legacy instantiation defaults correctly across ranks."""
    config = _make_base_config()
    launch_cpu_workers(
        _worker_legacy_instantiation,
        args=(config, tmp_path),
        world_size=2,
        timeout=15,
    )


@pytest.mark.heavy
def test_replicated_trainer_explicit_kwargs(tmp_path):
    """Verify ReplicatedTrainer explicit multi-node kwargs."""
    config = _make_base_config()
    launch_cpu_workers(
        _worker_explicit_kwargs_instantiation,
        args=(config, tmp_path),
        world_size=2,
        timeout=15,
    )


@pytest.mark.heavy
def test_replicated_trainer_execution_dict(tmp_path):
    """Verify ReplicatedTrainer with resolved execution profile dictionary."""
    config = _make_base_config()
    raw_profile = {
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'nodes': 2,
            'node_rank': 0,
            'ranks_per_node': 1,
            'rendezvous': 'tcp://127.0.0.1:29500',
            'accumulation_steps': 2,
            'aggregation': {
                'strategy': 'pseudo_gradient',
                'sync_every_steps': 4,
                'outer_lr': 0.5,
                'outer_momentum': 0.9,
            },
        }
    }
    profile = resolve_execution_profile(raw_profile, resolve_config(config))
    launch_cpu_workers(
        _worker_execution_dict_instantiation,
        args=(config, profile['execution'], tmp_path),
        world_size=2,
        timeout=15,
    )


@pytest.mark.heavy
def test_replicated_cpu_trainer_kwargs(tmp_path):
    """Verify ReplicatedCPUTrainer kwarg forwarding and execution profile compatibility."""
    config = _make_base_config()
    raw_profile = {
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'nodes': 1,
            'node_rank': 0,
            'ranks_per_node': 2,
            'rendezvous': 'file://',
            'accumulation_steps': 1,
            'aggregation': {
                'strategy': 'parameter_average',
                'sync_every_steps': 3,
                'outer_lr': 1.0,
                'outer_momentum': 0.0,
            },
        }
    }
    profile = resolve_execution_profile(raw_profile, resolve_config(config))
    launch_cpu_workers(
        _worker_cpu_trainer_kwargs,
        args=(config, profile['execution'], tmp_path),
        world_size=2,
        timeout=15,
    )


@pytest.mark.heavy
def test_replicated_worker_create_worker_synthetic_battery(tmp_path):
    """Verify replicated_worker.create_worker with a battery of valid profiles and adversarial tampering."""
    config = _make_base_config()
    res_cfg = resolve_config(config)

    profiles_to_test = []

    # 1. Standard Gloo defaults
    p1 = resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}, res_cfg)
    profiles_to_test.append(p1)

    # 2. Accumulation steps = 2
    p2 = resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 2}}, res_cfg)
    profiles_to_test.append(p2)

    # 3. Multi-node TCP profile
    p3 = resolve_execution_profile({
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'nodes': 2,
            'node_rank': 0,
            'rendezvous': 'tcp://127.0.0.1:29500',
        }
    }, res_cfg)
    profiles_to_test.append(p3)

    # 4. Parameter average aggregation
    p4 = resolve_execution_profile({
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'aggregation': {
                'strategy': 'parameter_average',
                'sync_every_steps': 5,
            }
        }
    }, res_cfg)
    profiles_to_test.append(p4)

    # 5. Pseudo gradient aggregation
    p5 = resolve_execution_profile({
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'aggregation': {
                'strategy': 'pseudo_gradient',
                'sync_every_steps': 3,
                'outer_lr': 0.25,
                'outer_momentum': 0.85,
            }
        }
    }, res_cfg)
    profiles_to_test.append(p5)

    launch_cpu_workers(
        _worker_create_worker_synthetic_profiles,
        args=(config, profiles_to_test, tmp_path),
        world_size=2,
        timeout=20,
    )


# ---------------------------------------------------------------------------
# Direct Matrix & Edge-Case Probes (Mocked Collective)
# ---------------------------------------------------------------------------

class MockCollectives:
    def __init__(self, world_size, *, device=None):
        self.world_size = world_size
        self.device = torch.device('cpu')
        self.backend = 'gloo'


def _setup_dist_mocks(monkeypatch, world_size=2, rank=0):
    import hypergan.distributed_training as dt
    import torch.distributed as dist

    def mock_all_gather_object(peers, payload, group=None):
        for i in range(len(peers)):
            peers[i] = copy.deepcopy(payload)

    monkeypatch.setattr(dt, 'Collectives', MockCollectives)
    monkeypatch.setattr(dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_world_size', lambda: world_size)
    monkeypatch.setattr(dist, 'all_gather_object', mock_all_gather_object)


def test_rendezvous_default_switching(monkeypatch):
    """Default rendezvous is 'file://' when nodes=1; nodes > 1 requires an explicit rendezvous."""
    import hypergan.distributed_training as dt

    _setup_dist_mocks(monkeypatch, world_size=2)
    config = _make_base_config()

    # nodes = 1 -> file://
    t1 = dt.ReplicatedTrainer(config, world_size=2, nodes=1)
    assert t1.rendezvous == 'file://'
    assert t1.strategy_info['rendezvous'] == 'file://'

    # nodes = 2: a loopback default hangs on separate hosts (review 2026-10-25 L1)
    with pytest.raises(ValueError, match='requires an explicit rendezvous'):
        dt.ReplicatedTrainer(config, world_size=2, nodes=2)
    t2 = dt.ReplicatedTrainer(config, world_size=2, nodes=2, rendezvous='tcp://node0:29500')
    assert t2.strategy_info['rendezvous'] == 'tcp://node0:29500'


def test_derived_ranks_per_node_defaulting(monkeypatch):
    """Verify ranks_per_node = world_size // nodes when omitted."""
    import hypergan.distributed_training as dt

    _setup_dist_mocks(monkeypatch, world_size=4)

    config = _make_base_config()
    config['training']['batch_size'] = 64

    # world_size = 4, nodes = 2 -> ranks_per_node = 2
    t = dt.ReplicatedTrainer(config, world_size=4, nodes=2, rendezvous='tcp://node0:29500')
    assert t.ranks_per_node == 2
    assert t.strategy_info['ranks_per_node'] == 2

    # explicit override ranks_per_node = 1
    t_explicit = dt.ReplicatedTrainer(config, world_size=4, nodes=2, ranks_per_node=1, rendezvous='tcp://node0:29500')
    assert t_explicit.ranks_per_node == 1
    assert t_explicit.strategy_info['ranks_per_node'] == 1


def test_exhaustive_profile_strategy_info_exact_match(monkeypatch):
    """Verify exact key/value alignment between resolve_execution_profile and strategy_info across a wide parameter matrix."""
    import hypergan.distributed_training as dt

    base_cfg = _make_base_config()
    base_cfg['training']['batch_size'] = 64
    res_cfg = resolve_config(base_cfg)

    matrix = [
        # (world_size, nodes, node_rank, accumulation_steps, rendezvous, aggregation)
        (2, 1, 0, 1, None, None),
        (2, 1, 0, 2, 'file:///tmp/custom-rdzv', None),
        (4, 2, 0, 1, 'tcp://coordinator:29500', None),
        (4, 2, 1, 2, 'tcp://10.0.0.1:12345', None),
        (4, 4, 3, 1, 'tcp://cluster.internal:65535', None),
        (2, 1, 0, 1, None, {'strategy': 'parameter_average', 'sync_every_steps': 10}),
        (2, 1, 0, 2, None, {'strategy': 'pseudo_gradient', 'sync_every_steps': 5, 'outer_lr': 0.1, 'outer_momentum': 0.95}),
    ]

    for ws, nodes, n_rank, accum, rdzv, agg in matrix:
        _setup_dist_mocks(monkeypatch, world_size=ws, rank=n_rank)
        raw_exec = {
            'name': 'cpu-replicated-gloo',
            'world_size': ws,
            'nodes': nodes,
            'node_rank': n_rank,
            'accumulation_steps': accum,
        }
        if rdzv:
            raw_exec['rendezvous'] = rdzv
        if agg:
            raw_exec['aggregation'] = agg

        raw_profile = {'schema_version': 1, 'execution': raw_exec}
        profile = resolve_execution_profile(raw_profile, res_cfg)
        execution = profile['execution']

        trainer = dt.ReplicatedTrainer(base_cfg, world_size=ws, execution=execution)

        # 1. Check every execution profile key exists and matches in strategy_info
        for key, val in execution.items():
            assert key in trainer.strategy_info, f"Missing key '{key}' in trainer.strategy_info"
            assert trainer.strategy_info[key] == val, f"Value mismatch for key '{key}': {trainer.strategy_info[key]} vs {val}"

        # 2. Check execution_preflight._differences returns zero diffs
        diffs = _differences(execution, {k: trainer.strategy_info.get(k) for k in execution}, 'execution')
        assert diffs == [], f"Preflight differences detected for profile {raw_exec}: {diffs}"

        # 3. Check create_worker accepts it cleanly
        worker = create_worker(n_rank, ws, base_cfg, execution, {})
        assert worker['trainer'].strategy_info == trainer.strategy_info


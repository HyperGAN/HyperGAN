"""Replicated execution features: recipe policy, strategy alignment, error agreement and smoke runs.

This is the one test file for the replicated trainer's feature surface (Issue #186):
which recipes replicate, how execution profiles become ``strategy_info``, how
rank-local failures become one agreed error per update, and two-rank smoke runs
of each recipe feature. The ``heavy`` tests start one worker process per rank,
this file re-executing itself as the worker.
"""
from __future__ import annotations

import collections
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
from torch import nn

if __package__ in (None, ""):
    # Rank workers run this file directly and need the shared fixtures beside tests/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from tests.distributed_fixtures import make_toy_config
except ImportError:
    from distributed_fixtures import make_toy_config

from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedCPUTrainer, ReplicatedTrainer
from hypergan.execution_preflight import _differences
from hypergan.execution_profiles import resolve_execution_profile, validate_replicated_recipe
from hypergan.replicated_worker import create_worker

HAVE_2_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
requires_2_gpus = pytest.mark.skipif(not HAVE_2_GPUS, reason="needs two CUDA GPUs")
REPO_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Recipes
# ==============================================================================

def _base_config():
    return {
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                              'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 20, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': 16, 'device': 'cpu'},
    }


def _prior_bound_config():
    return resolve_config({
        'prior': {'kind': 'mog', 'fixed_sigma': 0.05, 'args': {'num_particles': 8, 'z_dim': 4}},
        'components': {
            'generator': {'factory': 'linear', 'args': {'in_features': 4, 'out_features': 2, 'bias': False}, 'inputs': {'input': 'latent'}},
            'encoder': {'factory': 'linear', 'args': {'in_features': 4, 'out_features': 4, 'bias': False},
                        'inputs': {'input': 'prior.means'}, 'trainable': True},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'objectives': [{'factory': 'mse', 'inputs': {'input': 'components.encoder', 'target': 'prior.means'}, 'weight': 0.5}],
        'training': {'steps': 2, 'batch_size': 8, 'device': 'cpu', 'seed': 123},
    })


def _reuse_config():
    return resolve_config({
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'encoder_a': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [4]}, 'inputs': {'x': 'latent'}},
            'encoder_b': {'reuse': 'encoder_a', 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'objectives': [{'factory': 'mse', 'inputs': {'input': 'components.encoder_b', 'target': 'batch.real'}, 'weight': 1.0}],
        'prior': {'kind': 'particles', 'args': {'num_particles': 16, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': 8, 'device': 'cpu', 'seed': 123},
    })


def _adversarial_terms_config(terms=1):
    extra = [{'id': f'extra_term_{index}', 'component': 'discriminator', 'real': 'batch.real', 'fake': 'generated'}
             for index in range(terms)]
    return resolve_config({
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'adversarial_terms': extra,
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': 8, 'device': 'cpu', 'seed': 123},
    })


def _preset(**values):
    return resolve_config({**values, 'training': {'device': 'cpu', 'steps': 1, 'batch_size': 8}})


def _reuse_with_adversarial_terms():
    config = make_toy_config(device="cpu", batch_size=8)
    config['components']['encoder_a'] = {'factory': 'mlp', 'args': {'input_dim': 2, 'output_dim': 2, 'hidden': [4]},
                                         'inputs': {'x': 'candidate'}}
    config['components']['encoder_b'] = {'reuse': 'encoder_a', 'inputs': {'x': 'candidate'}}
    config['adversarial_terms'] = [
        {'id': 'adv_term_1', 'component': 'discriminator', 'real': 'batch.real', 'fake': 'generated'},
        {'id': 'adv_term_2', 'component': 'discriminator', 'real': 'batch.real', 'fake': 'generated'},
    ]
    return config


def _preset_with_reuse_and_adversarial_terms():
    config = _preset(defaults='particlegan')
    config['components']['critic_shared'] = {'reuse': 'discriminator', 'inputs': {'input': 'candidate'}}
    config['adversarial_terms'] = [{'id': 'extra_adv', 'component': 'critic_shared', 'real': 'batch.real', 'fake': 'generated'}]
    return config


def _prior_sample_config():
    config = make_toy_config(device="cpu", batch_size=8)
    config['components']['generator']['inputs'] = {'x': 'prior.sample'}
    return config


@pytest.mark.parametrize('build', [
    _prior_sample_config, lambda: _adversarial_terms_config(terms=2), _reuse_with_adversarial_terms,
    lambda: _preset(defaults='particlegan', recipe={'bispf_optimized': True}),
    lambda: _preset(defaults='particlegan', recipe={'dualnorm': True}),
    _preset_with_reuse_and_adversarial_terms,
], ids=['prior-sample-input', 'two-extra-adversarial-terms', 'reuse-in-adversarial-terms', 'bispf-preset',
        'dualnorm-preset', 'preset-reuse-adversarial-terms'])
def test_replicated_recipe_policy_accepts_combined_features(build):
    """The single-feature acceptances are in tests/foundation/test_execution_profiles.py."""
    config = build()
    validate_replicated_recipe(config)
    assert {'generator', 'discriminator'} <= set(config['components'])


def test_reused_component_shares_one_module_in_the_graph():
    from hypergan.recipes import ComponentGraph
    graph = ComponentGraph(_reuse_config()['components'])
    assert 'encoder_b' not in graph.models and 'encoder_a' in graph.models
    assert graph.models[graph.specs['encoder_b']['reuse']] is graph.models['encoder_a']


# ==============================================================================
# An in-process stand-in for a process group of identical ranks
# ==============================================================================

class FakeGroup:
    """One process standing in for ``world_size`` identical ranks, counting collectives.

    Every peer contributes the local value: object gathers replicate the payload
    and a SUM all-reduce multiplies by world_size, so means are unchanged.
    """

    def __init__(self, monkeypatch, world_size=2, rank=0):
        self.world_size, self.calls = world_size, collections.Counter()
        for name in ('all_gather_object', 'all_reduce', 'all_gather', 'broadcast', 'broadcast_object_list'):
            monkeypatch.setattr(dist, name, getattr(self, name))
        monkeypatch.setattr(dist, 'is_available', lambda: True)
        monkeypatch.setattr(dist, 'is_initialized', lambda: True)
        monkeypatch.setattr(dist, 'get_rank', lambda *args, **kwargs: rank)
        monkeypatch.setattr(dist, 'get_world_size', lambda *args, **kwargs: world_size)
        monkeypatch.setattr(dist, 'get_backend', lambda *args, **kwargs: 'gloo')

    def all_gather_object(self, peers, payload, group=None):
        self.calls['all_gather_object'] += 1
        peers[:] = [copy.deepcopy(payload) for _ in peers]

    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        self.calls['all_reduce'] += 1
        tensor.mul_(self.world_size)

    def all_gather(self, tensors, tensor, group=None, async_op=False):
        self.calls['all_gather'] += 1
        for item in tensors:
            item.copy_(tensor)

    def broadcast(self, tensor, src=0, group=None, async_op=False):
        self.calls['broadcast'] += 1

    def broadcast_object_list(self, values, src=0, group=None):
        self.calls['broadcast_object_list'] += 1


# ==============================================================================
# Collectives and error agreement per update
# ==============================================================================

@pytest.mark.parametrize('accumulation_steps', [1, 2, 4])
def test_an_update_agrees_on_rank_errors_with_a_fixed_number_of_object_collectives(monkeypatch, accumulation_steps):
    """Review 2026-10-25 M5: every phase exchanged its own pickled error agreement,
    about 23 all_gather_object calls per update and more with each microbatch. An
    update now settles once between consecutive collectives: the update boundary,
    the inputs, the D backward, the D optimizer, the G backward, the G optimizer
    and the completion, plus the metadata exchange of the prior-row union."""
    group = FakeGroup(monkeypatch)
    trainer = ReplicatedTrainer(make_toy_config(device='cpu', batch_size=8, steps=4), world_size=2,
                                accumulation_steps=accumulation_steps)
    ids = torch.arange(4)
    for batch, latent in (({'real': torch.linspace(-1, 1, 8).reshape(4, 2)}, (trainer.prior.z[ids], ids)), (None, None)):
        group.calls.clear()
        row, _ = trainer.update(batch, latent)
        assert group.calls['all_gather_object'] == 8, group.calls
        assert math.isfinite(row['d_loss']) and trainer.checkpoint_ready


def _bare_trainer(world_size=2):
    trainer = object.__new__(ReplicatedTrainer)
    trainer.world_size = world_size
    exchanged = []

    def exchange(operation, value):
        exchanged.append(operation)
        return [copy.deepcopy(value) for _ in range(world_size)]
    trainer._exchange = exchange
    return trainer, exchanged


def test_a_folded_block_settles_once_and_skips_phases_after_a_local_failure():
    trainer, exchanged = _bare_trainer()
    calls = []

    def boom():
        raise ValueError('boom')
    with pytest.raises(RuntimeError, match=r'^second failed: rank 0: ValueError: boom; rank 1: ValueError: boom$'):
        with trainer._folded('block'):
            trainer._phase('first', lambda: calls.append('first'))
            trainer._phase('second', boom)
            trainer._phase('third', lambda: calls.append('third'))
            trainer._agree('skipped agreement', object())
    assert calls == ['first'] and exchanged == ['fold:block'] and trainer._fold is None
    # Code between phases that trips over a failed rank's missing result still
    # reaches the one exchange instead of leaving the peers waiting.
    with pytest.raises(RuntimeError, match=r'^block failed: rank 0: TypeError'):
        with trainer._folded('block'):
            value = trainer._phase('number', lambda: 1)
            value + None
    with trainer._folded('gathers') as fold:
        trainer._agree('shape', (4, 2))
        assert trainer._phase('values', lambda: [1.0, 2.0], gather=True) == [1.0, 2.0]
    assert fold.gathered == {'values': [[1.0, 2.0], [1.0, 2.0]]}


@pytest.mark.parametrize('peers,error,match', [
    # The earliest failing entry is reported, with every rank that failed there.
    ([[('phase', 'a', None), ('error', 'b', 'ValueError: x')],
      [('phase', 'a', None), ('phase', 'b', None), ('error', 'c', 'KeyError: y')]],
     RuntimeError, r'^b failed: rank 0: ValueError: x$'),
    ([[('error', 'a', 'ValueError: x')], [('error', 'a', 'OSError: y')]],
     RuntimeError, r'^a failed: rank 0: ValueError: x; rank 1: OSError: y$'),
    ([[('phase', 'a', None), ('error', 'b', 'ValueError: x')], [('phase', 'a', None), ('error', 'unwound', 'TypeError: z')]],
     RuntimeError, r'^b / unwound failed: rank 0: ValueError: x; rank 1: TypeError: z$'),
    # A disagreement before a later failure is what happened first.
    ([[('agreement', 'input source', 1), ('error', 'b', 'ValueError: x')], [('agreement', 'input source', 2)]],
     ValueError, r'^Ranks disagree on input source$'),
    ([[('phase', 'a', None)], [('phase', 'other', None)]], ValueError, 'Ranks entered different trainer operations'),
    ([[('phase', 'a', None), ('phase', 'b', None)], [('phase', 'a', None)]], ValueError, 'Ranks entered different trainer operations'),
])
def test_settlement_resolves_rank_reports_identically(peers, error, match):
    from hypergan.distributed_training import _Fold
    trainer = object.__new__(ReplicatedTrainer)
    trainer._exchange = lambda operation, value: peers
    with pytest.raises(error, match=match):
        trainer._settle(_Fold('block'))


# ==============================================================================
# Execution profiles and strategy_info
# ==============================================================================

def test_rendezvous_default_switching(monkeypatch):
    """Default rendezvous is 'file://' when nodes=1; nodes > 1 requires an explicit rendezvous."""
    FakeGroup(monkeypatch)
    t1 = ReplicatedTrainer(_base_config(), world_size=2, nodes=1)
    assert t1.rendezvous == t1.strategy_info['rendezvous'] == 'file://'
    # A loopback default hangs on separate hosts (review 2026-10-25 L1).
    with pytest.raises(ValueError, match='requires an explicit rendezvous'):
        ReplicatedTrainer(_base_config(), world_size=2, nodes=2)
    t2 = ReplicatedTrainer(_base_config(), world_size=2, nodes=2, rendezvous='tcp://node0:29500')
    assert t2.strategy_info['rendezvous'] == 'tcp://node0:29500'


def test_ranks_per_node_defaults_to_world_size_over_nodes(monkeypatch):
    FakeGroup(monkeypatch, world_size=4)
    config = _base_config()
    config['training']['batch_size'] = 64
    derived = ReplicatedTrainer(config, world_size=4, nodes=2, rendezvous='tcp://node0:29500')
    assert derived.ranks_per_node == derived.strategy_info['ranks_per_node'] == 2
    explicit = ReplicatedTrainer(config, world_size=4, nodes=2, ranks_per_node=1, rendezvous='tcp://node0:29500')
    assert explicit.ranks_per_node == explicit.strategy_info['ranks_per_node'] == 1


@pytest.mark.parametrize('world_size,nodes,node_rank,accumulation,rendezvous,aggregation', [
    (2, 1, 0, 1, None, None),
    (2, 1, 0, 2, 'file:///tmp/custom-rdzv', None),
    (4, 2, 0, 1, 'tcp://coordinator:29500', None),
    (4, 2, 1, 2, 'tcp://10.0.0.1:12345', None),
    (4, 4, 3, 1, 'tcp://cluster.internal:65535', None),
    (2, 1, 0, 1, None, {'strategy': 'parameter_average', 'sync_every_steps': 10}),
    (2, 1, 0, 2, None, {'strategy': 'pseudo_gradient', 'sync_every_steps': 5, 'outer_lr': 0.1, 'outer_momentum': 0.95}),
])
def test_resolved_profiles_match_strategy_info_exactly(monkeypatch, world_size, nodes, node_rank, accumulation,
                                                        rendezvous, aggregation):
    FakeGroup(monkeypatch, world_size=world_size, rank=node_rank)
    config = _base_config()
    config['training']['batch_size'] = 64
    raw = {'name': 'cpu-replicated-gloo', 'world_size': world_size, 'nodes': nodes, 'node_rank': node_rank,
           'accumulation_steps': accumulation}
    if rendezvous:
        raw['rendezvous'] = rendezvous
    if aggregation:
        raw['aggregation'] = aggregation
    execution = resolve_execution_profile({'schema_version': 1, 'execution': raw}, resolve_config(config))['execution']
    trainer = ReplicatedTrainer(config, world_size=world_size, execution=execution)
    assert {key: trainer.strategy_info.get(key) for key in execution} == execution
    assert _differences(execution, {key: trainer.strategy_info.get(key) for key in execution}, 'execution') == []
    assert create_worker(node_rank, world_size, config, execution, {})['trainer'].strategy_info == trainer.strategy_info


# ==============================================================================
# Real Gloo ranks (heavy: each case starts one worker process per rank)
# ==============================================================================

INJECTED = {'critic': None, 'objective': None}


def _injected(kind):
    target = INJECTED[kind]
    return target == 'all' or target == dist.get_rank()


class FlakyCritic(nn.Module):
    """A linear critic that fails on the ranks named in INJECTED['critic']."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, input):
        if _injected('critic'):
            raise RuntimeError('injected critic failure')
        return self.linear(input)


class FlakyObjective(nn.Module):
    accumulation_reduction = 'mean'

    def forward(self, input):
        if _injected('objective'):
            raise RuntimeError('injected objective failure')
        return input.square().mean() * 0.1


def _flaky_config():
    return resolve_config({
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': f'{__name__}:FlakyCritic', 'inputs': {'input': 'candidate'}},
        },
        'objectives': [{'factory': f'{__name__}:FlakyObjective', 'inputs': {'input': 'generated'}, 'weight': 0.1}],
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'training': {'steps': 4, 'batch_size': 8, 'device': 'cpu', 'seed': 42},
    })


def _errors_worker(rank, world_size):
    """Each scenario injects a failure on some ranks; every rank must raise the same error."""
    scenarios = [
        ('critic on rank 1', {'critic': 1}, 1),
        ('objective on rank 0', {'objective': 0}, 1),
        ('critic on every rank', {'critic': 'all'}, 1),
        ('accumulated critic on rank 1', {'critic': 1}, 2),
    ]
    data, ids = torch.linspace(-1, 1, 16).reshape(8, 2), torch.arange(8)
    results = {}
    for name, injected, accumulation in scenarios:
        INJECTED.update(critic=None, objective=None)
        trainer = ReplicatedCPUTrainer(_flaky_config(), world_size=world_size, accumulation_steps=accumulation)
        local_ids = ids.chunk(world_size)[rank]
        trainer.update({'real': data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
        INJECTED.update(injected)
        rows = []
        for _ in range(2):  # the failure, then the refused update after it
            try:
                trainer.update({'real': data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
                rows.append(None)
            except Exception as exc:
                rows.append([type(exc).__name__, str(exc)])
        results[name] = rows
    return results


def _smoke_config(name):
    return {
        'toy': lambda: make_toy_config(device="cpu", batch_size=8, steps=2),
        'toy-accumulation': lambda: make_toy_config(device="cpu", batch_size=8, steps=1),
        'prior-bound': _prior_bound_config,
        'reuse': _reuse_config,
        'adversarial-terms': _adversarial_terms_config,
        'particlegan-preset': lambda: _preset(defaults='particlegan'),
        'bispf-preset': lambda: _preset(recipe={'name': 'particlegan-bispf', 'model': 'gan'}),
    }[name]()


def _smoke_worker(rank, world_size, name, steps, accumulation_steps):
    trainer = ReplicatedCPUTrainer(_smoke_config(name), world_size=world_size, accumulation_steps=int(accumulation_steps))
    data, ids = torch.linspace(-1, 1, 16).reshape(8, 2), torch.arange(8)
    local_ids = ids.chunk(world_size)[rank]
    history = []
    for _ in range(int(steps)):
        row, _ = trainer.update({'real': data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
        history.append({key: row[key] for key in ('step', 'd_loss', 'g_loss')})
    return {'history': history, 'weights': {key: value.tolist() for key, value in trainer.graph.state_dict().items()}}


def _construction_worker(rank, world_size):
    """Trainer construction from kwargs and execution tables, and create_worker's alignment check."""
    config = _base_config()
    default_aggregation = {'strategy': 'gradient_allreduce', 'sync_every_steps': 1, 'outer_lr': 1.0, 'outer_momentum': 0.0}
    for cls in (ReplicatedTrainer, ReplicatedCPUTrainer):
        legacy = cls(config, world_size=world_size)
        assert (legacy.nodes, legacy.node_rank, legacy.ranks_per_node, legacy.rendezvous) == (1, 0, world_size, 'file://')
        info = legacy.strategy_info
        assert legacy.aggregation == info['aggregation'] == default_aggregation
        assert (info['nodes'], info['node_rank'], info['ranks_per_node'], info['rendezvous'], info['world_size']) == (
            1, 0, world_size, 'file://', world_size)
        assert (info['accumulation_steps'], info['global_batch_size'], info['local_batch_size'], info['microbatch_size'],
                info['accumulation_algorithm']) == (1, 16, 8, 8, 'retained-local-graph-v1')

        aggregation = {'strategy': 'parameter_average', 'sync_every_steps': 2, 'outer_lr': 0.8, 'outer_momentum': 0.5}
        explicit = cls(config, world_size=world_size, nodes=2, node_rank=rank, ranks_per_node=1,
                       rendezvous='tcp://127.0.0.1:29500', aggregation=aggregation)
        assert (explicit.nodes, explicit.node_rank, explicit.ranks_per_node, explicit.rendezvous) == (
            2, rank, 1, 'tcp://127.0.0.1:29500')
        assert explicit.aggregation == explicit.strategy_info['aggregation'] == aggregation
        assert (explicit.strategy_info['nodes'], explicit.strategy_info['node_rank']) == (2, rank)

    resolved = resolve_config(config)
    tables = [
        {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 2, 'node_rank': 0, 'ranks_per_node': 1,
         'rendezvous': 'tcp://127.0.0.1:29500', 'accumulation_steps': 2,
         'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 4, 'outer_lr': 0.5, 'outer_momentum': 0.9}},
        {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 1, 'node_rank': 0, 'ranks_per_node': 2, 'rendezvous': 'file://',
         'accumulation_steps': 1,
         'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 3, 'outer_lr': 1.0, 'outer_momentum': 0.0}},
        {'name': 'cpu-replicated-gloo', 'world_size': 2},
        {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 2},
        {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 2, 'node_rank': 0, 'rendezvous': 'tcp://127.0.0.1:29500'},
        {'name': 'cpu-replicated-gloo', 'world_size': 2, 'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 5}},
        {'name': 'cpu-replicated-gloo', 'world_size': 2,
         'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 3, 'outer_lr': 0.25, 'outer_momentum': 0.85}},
    ]
    for table in tables:
        execution = resolve_execution_profile({'schema_version': 1, 'execution': table}, resolved)['execution']
        for cls in (ReplicatedTrainer, ReplicatedCPUTrainer):
            trainer = cls(config, world_size=world_size, execution=execution)
            assert {key: trainer.strategy_info[key] for key in execution} == execution
            assert not _differences(execution, {key: trainer.strategy_info.get(key) for key in execution}, 'execution')
        worker = create_worker(rank, world_size, config, execution, {'test_context': True})
        assert (worker['rank'], worker['batch'], worker['context']) == (rank, None, {'test_context': True})
        assert {key: worker['trainer'].strategy_info[key] for key in execution} == execution
        for field, value in (('name', 'tampered-device-strategy'), ('local_batch_size', 999), ('extra_drift_key', 'drift')):
            with pytest.raises(ValueError, match='Actual replicated strategy differs from resolved execution profile'):
                create_worker(rank, world_size, config, dict(copy.deepcopy(execution), **{field: value}), {})
    return {'passed': True}


def _launch(tmp_path, mode, world_size, *args, timeout=90):
    stamp = time.monotonic_ns()
    rendezvous = (tmp_path / f'rendezvous-{mode}-{stamp}').as_uri()
    outputs = [tmp_path / f'{mode}-{stamp}-rank{rank}.json' for rank in range(world_size)]
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve())]
    processes = [subprocess.Popen([*command, mode, str(rank), str(world_size), rendezvous, str(outputs[rank]), *map(str, args)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for rank in range(world_size)]
    failures, deadline = [], time.monotonic() + timeout
    try:
        for rank, process in enumerate(processes):
            _, stderr = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
            if process.returncode:
                failures.append(f'rank {rank} exited {process.returncode}:\n{stderr}')
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert not failures, '\n'.join(failures)
    return [json.loads(path.read_text()) for path in outputs]


def _worker(mode, rank, world_size, rendezvous, output, *args):
    rank, world_size = int(rank), int(world_size)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=30))
    try:
        result = {'errors': _errors_worker, 'smoke': _smoke_worker, 'construct': _construction_worker}[mode](
            rank, world_size, *args)
        Path(output).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.heavy
def test_rank_local_failures_raise_the_same_error_on_every_rank(tmp_path):
    ranks = _launch(tmp_path, 'errors', 2)
    assert ranks[0] == ranks[1], 'ranks reported different errors'
    refused = ['RuntimeError', 'Trainer is not at a complete boundary; restart the whole worker group from a coordinated checkpoint']
    expected = {
        'critic on rank 1': 'discriminator forward and penalty failed: rank 1: RuntimeError: injected critic failure',
        'objective on rank 0': 'generator regularizers and objectives failed: rank 0: RuntimeError: injected objective failure',
        'critic on every rank': ('discriminator forward and penalty failed: rank 0: RuntimeError: injected critic failure; '
                                 'rank 1: RuntimeError: injected critic failure'),
        'accumulated critic on rank 1': 'D micro 0 discovery failed: rank 1: RuntimeError: injected critic failure',
    }
    for name, message in expected.items():
        assert ranks[0][name] == [['RuntimeError', message], refused], name


SMOKE_CASES = [  # name, steps, accumulation_steps, compare losses with the single-device reference trainer
    ('toy', 2, 1, True), ('toy-accumulation', 1, 2, False), ('prior-bound', 1, 1, True), ('reuse', 1, 1, True),
    ('adversarial-terms', 1, 1, True), ('particlegan-preset', 1, 1, False), ('bispf-preset', 1, 2, False),
]


@pytest.mark.heavy
@pytest.mark.parametrize('name,steps,accumulation_steps,reference', SMOKE_CASES, ids=[case[0] for case in SMOKE_CASES])
def test_two_rank_smoke_run_matches_the_reference_and_keeps_replicas_identical(tmp_path, name, steps, accumulation_steps, reference):
    ranks = _launch(tmp_path, 'smoke', 2, name, steps, accumulation_steps)
    for rank in ranks:
        assert [row['step'] for row in rank['history']] == list(range(1, steps + 1))
        assert all(math.isfinite(row['d_loss']) and math.isfinite(row['g_loss']) for row in rank['history'])
    assert ranks[0]['weights'] == ranks[1]['weights'], 'replica parameters differ after a synchronous step'
    if reference:
        from hypergan.training import ReferenceTrainer
        trainer = ReferenceTrainer(_smoke_config(name))
        data, ids = torch.linspace(-1, 1, 16).reshape(8, 2), torch.arange(8)
        for row in ranks[0]['history']:
            expected, _ = trainer.update({'real': data}, (trainer.prior.z[ids], ids))
            assert row['d_loss'] == pytest.approx(expected['d_loss'], rel=3e-5, abs=3e-6)
            assert row['g_loss'] == pytest.approx(expected['g_loss'], rel=3e-5, abs=3e-6)


@pytest.mark.heavy
def test_trainer_construction_and_worker_creation_follow_the_resolved_profile(tmp_path):
    assert _launch(tmp_path, 'construct', 2) == [{'passed': True}, {'passed': True}]


def _driver_env():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')])
    return env


@pytest.mark.heavy
def test_worker_service_creates_and_cleans_a_bare_file_rendezvous(tmp_path):
    driver = tmp_path / 'file_rendezvous_driver.py'
    driver.write_text('''
from hypergan.cpu_worker_service import CPUWorkerService

def factory(rank, world_size, *args):
    return {"rank": rank}

def handler(state, operation, payload):
    return {"rank": state["rank"], "echo": operation}

if __name__ == "__main__":
    for index, rendezvous in enumerate(("file://", None)):
        service = CPUWorkerService(factory, handler, run_id=f"bare-file-{index}", attempt_id="attempt-1", world_size=2,
                                   rendezvous=rendezvous, startup_timeout=15, command_timeout=10, collective_timeout=5)
        with service:
            assert len(service.command("ping", {})["results"]) == 2
    print("PASSED_BARE_FILE_RENDEZVOUS")
''')
    process = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, env=_driver_env(), timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'PASSED_BARE_FILE_RENDEZVOUS' in process.stdout


@pytest.mark.heavy
def test_worker_service_fails_and_reaps_ranks_on_an_unreachable_tcp_rendezvous(tmp_path):
    driver = tmp_path / 'tcp_timeout_driver.py'
    driver.write_text('''
from hypergan.cpu_worker_service import CPUWorkerService

def factory(rank, world_size, *args):
    return {"rank": rank}

def handler(state, operation, payload):
    return {"rank": state["rank"]}

if __name__ == "__main__":
    # Node 1 of 2 connecting to a coordinator that never starts.
    service = CPUWorkerService(factory, handler, run_id="tcp-timeout", attempt_id="attempt-1", world_size=2, nodes=2,
                               node_rank=1, ranks_per_node=1, rendezvous="tcp://127.0.0.1:65534",
                               startup_timeout=2, command_timeout=2, collective_timeout=1)
    failed = False
    try:
        service.start()
    except Exception as exc:
        failed = True
    finally:
        try:
            service.abort()
        except Exception:
            pass
    assert failed, "CPUWorkerService should fail on an unreachable TCP rendezvous"
    assert service._closed is True
    print("PASSED_TCP_TIMEOUT_CLEANUP")
''')
    process = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, env=_driver_env(), timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'PASSED_TCP_TIMEOUT_CLEANUP' in process.stdout


@pytest.mark.heavy
def test_qualify_cli_qualifies_this_host_and_writes_a_valid_receipt(tmp_path):
    from hypergan.qualify import validate_qualification_receipt
    receipt_path = tmp_path / 'qualification-receipt.json'
    process = subprocess.run([sys.executable, '-m', 'hypergan', 'qualify', '--output', str(receipt_path)],
                             capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, f'hypergan qualify failed:\n{process.stdout}\n{process.stderr}'
    receipt = json.loads(receipt_path.read_text())
    validate_qualification_receipt(receipt)
    assert receipt['status'] == 'passed' and receipt['duration_seconds'] <= 180.0


@pytest.mark.heavy
@requires_2_gpus
def test_two_gpu_nccl_worker_service_and_training(tmp_path):
    """The NCCL worker service reduces across both GPUs, and 2-rank CUDA Adam training completes."""
    service_driver = tmp_path / 'nccl_service.py'
    service_driver.write_text('''
import torch, torch.distributed as dist
from hypergan.cpu_worker_service import CPUWorkerService

def factory(rank, world):
    assert dist.get_backend() == 'nccl' and torch.cuda.current_device() == rank
    return rank

def handler(rank, operation, payload):
    tensor = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(tensor)
    torch.cuda.synchronize(rank)
    return {'rank': rank, 'sum': tensor.item()}

if __name__ == '__main__':
    service = CPUWorkerService(factory, handler, backend='nccl', run_id='e2e-gpu', attempt_id='att1',
                               startup_timeout=20, command_timeout=10, collective_timeout=5, total_timeout=30)
    with service:
        assert service.command('ping')['results'] == [{'rank': 0, 'sum': 3.0}, {'rank': 1, 'sum': 3.0}]
''')
    env = dict(_driver_env(), NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
    process = subprocess.run([sys.executable, str(service_driver)], capture_output=True, text=True, timeout=60, env=env)
    assert process.returncode == 0, process.stdout + process.stderr

    training_driver = tmp_path / 'nccl_training.py'
    training_driver.write_text('''
import json, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedTrainer

rank, rendezvous, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20),
                        device_id=torch.device("cuda", rank))
try:
    config = resolve_config({
        "components": {
            "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [4]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "mlp", "args": {"input_dim": 2, "output_dim": 1, "hidden": [4]}, "inputs": {"x": "candidate"}},
        },
        "prior": {"args": {"num_particles": 4, "z_dim": 4}},
        "training": {"steps": 2, "batch_size": 4, "seed": 77, "device": "cuda"},
    })
    trainer = ReplicatedTrainer(config, world_size=2)
    data, ids = torch.randn(4, 2, device=f"cuda:{rank}"), torch.arange(4, device=f"cuda:{rank}")
    local_ids = ids.chunk(2)[rank]
    for _ in range(2):
        trainer.update({"real": data.chunk(2)[rank]}, (trainer.prior.z[local_ids], local_ids))
    output.write_text(json.dumps({"step": trainer.step}))
finally:
    dist.destroy_process_group()
''')
    rendezvous = (tmp_path / 'rendezvous-nccl').as_uri()
    outputs = [tmp_path / f'nccl-{rank}.json' for rank in range(2)]
    processes = [subprocess.Popen([sys.executable, str(training_driver), str(rank), rendezvous, str(outputs[rank])],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for rank in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stdout + stderr
    assert [json.loads(path.read_text())['step'] for path in outputs] == [2, 2]


if __name__ == '__main__':
    _worker(*sys.argv[1:])

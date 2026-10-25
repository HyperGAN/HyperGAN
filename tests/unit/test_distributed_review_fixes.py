"""Regressions for the distributed findings of the 2026-10-25 code review.

Collectives are replaced by deterministic stand-ins: each mocked all_reduce adds a
known peer contribution, so the expected cross-rank mean is exact without spawning
ranks. Real Gloo/NCCL runs of the same fixes live in tests/reference (heavy).
"""
import inspect
import subprocess
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from hypergan import aggregation, distributed_checkpoints, distributed_commit
from hypergan.aggregation import (AggregationStrategy, GradientAllreduceStrategy, ParameterAverageStrategy,
                                  PseudoGradientStrategy, _synchronize_optimizer_regularizers)
from hypergan.config import write_default
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.execution_profiles import experimental_features, resolve_execution_profile, validate_checkpoint_cadence


PEER_OFFSET = 2.0  # The simulated second rank holds every value + PEER_OFFSET.


@pytest.fixture
def two_ranks(monkeypatch):
    """A two-rank group whose all_reduce(SUM) adds a peer equal to local + PEER_OFFSET."""
    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_world_size', lambda *a: 2)
    monkeypatch.setattr(dist, 'get_rank', lambda *a: 0)

    def all_reduce(tensor, op=None, **kwargs):
        tensor.add_(tensor + PEER_OFFSET)
    monkeypatch.setattr(dist, 'all_reduce', all_reduce)
    monkeypatch.setattr(dist, 'broadcast', lambda tensor, src=0, **kwargs: None)


def test_k3p_histories_are_averaged_not_summed(two_ranks):
    """Fix-first #4: latent/direct history used SUM without dividing by world size."""
    optimizer = SimpleNamespace(latent_history=torch.full((3,), 1.0), direct_history=torch.full((2,), 4.0), state={})
    _synchronize_optimizer_regularizers(None, [optimizer])
    # mean(1, 3) == 2 and mean(4, 6) == 5; the bug produced the sums 4 and 10.
    assert torch.equal(optimizer.latent_history, torch.full((3,), 2.0))
    assert torch.equal(optimizer.direct_history, torch.full((2,), 5.0))


class _Buffered(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer('weight_u', torch.ones(2))
        self.register_buffer('calls', torch.tensor(3))


def test_pseudo_gradient_averages_graph_buffers(two_ranks):
    """Review M1: pseudo_gradient synchronized parameters only; spectral-norm weight_u diverged."""
    graph = _Buffered()
    trainer = SimpleNamespace(graph=graph, prior=None, ema_graph=None, ema_prior=None, opt_g=None, opt_d=None,
                              device=torch.device('cpu'))
    strategy = PseudoGradientStrategy(trainer, sync_every_steps=1)
    strategy.on_step_complete(1)
    assert torch.equal(graph.weight_u, torch.full((2,), 1.0 + PEER_OFFSET / 2))
    assert int(graph.calls) == 3  # integer buffers come from rank zero


def test_parameter_average_and_pseudo_gradient_share_buffer_averaging(two_ranks):
    for cls in (ParameterAverageStrategy, PseudoGradientStrategy):
        graph = _Buffered()
        trainer = SimpleNamespace(graph=graph, prior=None, ema_graph=None, ema_prior=None, opt_g=None, opt_d=None,
                                  device=torch.device('cpu'))
        cls(trainer, sync_every_steps=1).on_step_complete(1)
        assert torch.equal(graph.weight_u, torch.full((2,), 2.0)), cls.__name__


def test_local_sgd_forced_synchronization_happens_once_per_step(two_ranks):
    """Fix-first #6: checkpoints between syncs force exactly one synchronization."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    strategy = PseudoGradientStrategy(sync_every_steps=4, outer_lr=1.0, outer_momentum=0.5)
    strategy.initialize(models=model)
    strategy.on_step_complete(3, models=model)
    assert not strategy.is_synchronized(3)
    strategy.synchronize(models=model)
    assert strategy.is_synchronized(3) and strategy.synced_step == 3
    momentum = strategy.momentum_buffer['weight'].clone()
    # A restored strategy remembers the forced sync and does not repeat it.
    restored = PseudoGradientStrategy(sync_every_steps=4)
    restored.load_state_dict(strategy.state_dict())
    assert restored.is_synchronized(3)
    assert torch.equal(restored.momentum_buffer['weight'], momentum)
    # Older checkpoints were only ever written at sync boundaries.
    legacy = {k: v for k, v in strategy.state_dict().items() if k != 'synced_step'}
    restored.load_state_dict(legacy)
    assert restored.synced_step == 3
    assert GradientAllreduceStrategy().is_synchronized(7)


def test_checkpoint_cadence_must_land_on_sync_steps():
    config = {'training': {'device': 'cpu', 'batch_size': 8}}
    profile = resolve_execution_profile({'schema_version': 1, 'execution': {
        'name': 'cpu-replicated-gloo', 'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 3}}}, config)
    with pytest.raises(ValueError, match=r"experimental: checkpoint_every \(1000\) must be a multiple of .*sync_every_steps \(3\), e.g. 999"):
        validate_checkpoint_cadence(profile, 1000)
    validate_checkpoint_cadence(profile, 999)
    plain = resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}}, config)
    validate_checkpoint_cadence(plain, 7)
    assert experimental_features(plain) == []
    assert experimental_features(profile) == ["aggregation.strategy='parameter_average'"]


def test_run_train_refuses_misaligned_local_sgd_checkpoints_before_creating_the_run(tmp_path):
    from hypergan.replicated_execution import run_train
    config = write_default(tmp_path / 'project', device='cpu', defaults='hypergan')
    profile = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo',
                                                  'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 4}}}
    with pytest.raises(ValueError, match='multiple of aggregation.sync_every_steps'):
        run_train(config, tmp_path / 'run', profile=profile, checkpoint_every=10)
    assert not (tmp_path / 'run').exists()


def test_pseudo_gradient_checkpoint_state_has_one_copy_of_each_tensor():
    """Review M6: four aliased full-model clones per rank file."""
    model = torch.nn.Linear(4, 4)
    strategy = PseudoGradientStrategy(sync_every_steps=2)
    strategy.initialize(models=model)
    state = strategy.state_dict()
    tensors = [key for key, value in state.items() if isinstance(value, dict)]
    assert sorted(tensors) == ['anchor', 'momentum_buffer']
    restored = PseudoGradientStrategy(sync_every_steps=2)
    restored.load_state_dict({**state, 'anchor_weights': state.pop('anchor'), 'momentum': state.pop('momentum_buffer')})
    assert torch.equal(restored.anchor['weight'], model.weight.detach())


def test_strategies_declare_gradient_reduction_instead_of_matching_names(monkeypatch):
    """Review: _reduce_gradients compared the strategy name with 'gradient_allreduce'."""
    monkeypatch.setattr(dist, 'all_reduce', lambda tensor, op=None, **kwargs: None)  # one-rank sum
    trainer = object.__new__(ReplicatedTrainer)
    trainer.device, trainer.world_size = torch.device('cpu'), 2
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    class Custom:
        name = 'gradient_allreduce'  # a name alone no longer opts in
        reduces_gradients = False
        def on_backward(self, optimizer, name):
            pass

    for strategy, expected in ((Custom(), 4.0), (type('Opted', (Custom,), {'name': 'mine', 'reduces_gradients': True})(), 2.0)):
        trainer.aggregation_strategy = strategy
        parameter.grad = torch.full((2,), 4.0)
        trainer._reduce_gradients(optimizer, 'test')
        assert torch.equal(parameter.grad, torch.full((2,), expected)), type(strategy).__name__


def test_dead_pre_and_post_step_hooks_are_gone():
    """Review: pre_step/post_step were declared but never called by the trainer."""
    for owner in (AggregationStrategy, GradientAllreduceStrategy, ParameterAverageStrategy, PseudoGradientStrategy):
        assert not hasattr(owner, 'pre_step') and not hasattr(owner, 'post_step'), owner


def _draw_trainer(source):
    trainer = object.__new__(ReplicatedTrainer)
    trainer.data, trainer.rank, trainer.world_size = source, 1, 2
    trainer.global_batch_size, trainer.local_batch_size = 4, 2
    trainer.device = torch.device('cpu')
    trainer.streams = {'data': torch.Generator().manual_seed(0)}
    trainer._phase = lambda name, function: function()
    return trainer


def test_data_type_errors_are_not_retried_without_shards():
    """Review L3: an internal TypeError was retried without rank=, silently unsharding."""
    calls = []

    def sharded(batch_size, *, generator, rank=0, world_size=1):
        calls.append((rank, world_size))
        raise TypeError('bug inside the data source')

    with pytest.raises(TypeError, match='bug inside the data source'):
        _draw_trainer(sharded).batch()
    assert calls == [(1, 2)]

    def unsharded(batch_size, *, generator):
        return {'real': torch.arange(8.0).reshape(4, 2)}

    batch = _draw_trainer(unsharded).batch()
    assert torch.equal(batch['real'], torch.tensor([[4.0, 5.0], [6.0, 7.0]]))


def test_checkpoint_agreement_uses_the_local_cuda_device(monkeypatch):
    """Review M4: set_device(global_rank) when a node had >= world_size GPUs."""
    selected = []
    monkeypatch.setattr(dist, 'get_backend', lambda *a: 'nccl')
    monkeypatch.setattr(dist, 'get_rank', lambda *a: 5)
    monkeypatch.setattr(dist, 'get_world_size', lambda *a: 8)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    monkeypatch.setattr(dist, 'all_gather_object', lambda peers, value: peers.__setitem__(slice(None), [value] * len(peers)))
    trainer = SimpleNamespace(ranks_per_node=4)  # node 1 of 2, local rank 1
    distributed_checkpoints._agree('prepare:collect', {'step': 1}, trainer=trainer)
    assert selected == [torch.device('cuda', 1)]
    selected.clear()
    distributed_checkpoints._agree('prepare:collect', trainer=SimpleNamespace(device=torch.device('cuda', 3)))
    assert selected == [torch.device('cuda', 3)]


def test_one_rank_payload_bound_without_import_time_patching():
    """Review: 256 MiB in two modules, patched to 1 GiB from three others at import."""
    assert distributed_commit.MAX_RANK_BYTES == 1024 ** 3
    assert distributed_checkpoints.MAX_RANK_BYTES is distributed_commit.MAX_RANK_BYTES
    import hypergan.distributed_training
    import hypergan.replicated_execution
    import hypergan.replicated_worker
    for module in (hypergan.distributed_training, hypergan.replicated_execution, hypergan.replicated_worker,
                   distributed_checkpoints):
        assert 'MAX_RANK_BYTES =' not in inspect.getsource(module), module.__name__


def test_trainer_requires_explicit_multi_node_rendezvous():
    """Review L1: nodes > 1 defaulted to tcp://127.0.0.1:29500."""
    source = inspect.getsource(ReplicatedTrainer.__init__)
    assert '127.0.0.1' not in source
    assert 'requires an explicit rendezvous' in source


def test_followers_adopt_node_0_run_identity():
    """Review L6: a follower invented run-<hash> when the leader's manifest was absent."""
    from hypergan.replicated_worker import _node_0_context
    leader = {'run_id': 'leader-run', 'run_dir': '/shared/run'}
    phases = []
    trainer = SimpleNamespace(_exchange=lambda name, value: [leader, value],
                              _phase=lambda name, function: phases.append(name) or function())
    follower = {'run_id': 'follower', 'run_dir': '/shared/run', 'attempt_id': 'a'}
    assert _node_0_context(trainer, follower) == dict(follower, run_id='leader-run')
    with pytest.raises(ValueError, match='same run directory path on every node'):
        _node_0_context(trainer, dict(follower, run_dir='/local/run'))


def test_follower_execution_only_mirrors_node_0(tmp_path):
    from hypergan.config import resolve_config
    from hypergan.replicated_execution import ReplicatedExecution, run_follower
    from hypergan.run_controller import AttemptContext
    config = resolve_config({'training': {'device': 'cpu', 'batch_size': 8}})
    profile = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'nodes': 2, 'node_rank': 1,
                                                  'rendezvous': 'tcp://node0:29500'}}
    execution = ReplicatedExecution(config, profile)
    execution.configure_attempt(AttemptContext('follower', 'a', 1, tmp_path, tmp_path / '.follower'),
                                preview_every=0, on_event=None)
    with pytest.raises(ValueError, match='Follower nodes only mirror node 0'):
        execution.start()
    with pytest.raises(ValueError, match='follower node profile'):
        run_follower(config, tmp_path, profile=dict(profile, execution=dict(profile['execution'], node_rank=0)))


def test_gpu_inventory_tools_have_a_timeout(monkeypatch):
    """Review M7: nvidia-smi ran with no timeout from queue and status paths."""
    from hypergan.gpu_inventory import AmdSmiProvider, NvidiaSmiProvider, RocmSmiProvider, SMI_TIMEOUT_SECONDS
    calls = []

    def hung(command, **kwargs):
        calls.append(kwargs.get('timeout'))
        raise subprocess.TimeoutExpired(command, kwargs.get('timeout'))
    monkeypatch.setattr(subprocess, 'run', hung)
    for provider in (NvidiaSmiProvider, AmdSmiProvider, RocmSmiProvider):
        monkeypatch.setattr(provider, 'is_available', classmethod(lambda cls: True))
        assert provider().get_hardware() == []
        assert provider().get_processes() == []
    assert calls and all(timeout == SMI_TIMEOUT_SECONDS for timeout in calls)

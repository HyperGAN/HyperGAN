"""Learning aggregation strategies: validation, cadence, outer-step math, state and registry.

This is the one test file for the aggregation feature (Issue #186). The fast tests
drive the product strategies in process. The ``heavy`` tests run them across real
Gloo ranks, this file re-executing itself as each rank's worker.
"""
from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist

if __package__ in (None, ""):
    # Rank workers run this file directly and need the shared fixtures beside tests/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from tests.distributed_fixtures import assert_states_close, make_toy_config, oracle_pseudo_gradient_step
except ImportError:
    from distributed_fixtures import assert_states_close, make_toy_config, oracle_pseudo_gradient_step

import hypergan.aggregation as aggregation
from hypergan.aggregation import (
    GradientAllreduceStrategy,
    ParameterAverageStrategy,
    PseudoGradientStrategy,
    _synchronize_optimizer_moments,
    _synchronize_optimizer_regularizers,
    create_aggregation_strategy,
    get_aggregation_strategy,
    register_aggregation_strategy,
    resolve_aggregation_strategy,
)
from hypergan.execution_profiles import resolve_execution_profile


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2, bias=True)
        self.register_buffer("running_mean", torch.tensor([1.0, 2.0]))
        self.register_buffer("int_counter", torch.tensor(10, dtype=torch.int64))


class MockTrainer:
    def __init__(self, device="cpu"):
        self.graph = DummyModel().to(device)
        self.world_size = 2
        self.step = 0
        self.device = torch.device(device)


LOCAL_SGD = (ParameterAverageStrategy, PseudoGradientStrategy)


# ==============================================================================
# Construction and validation
# ==============================================================================

def test_defaults_and_gradient_reduction_capability():
    for cls, name in ((GradientAllreduceStrategy, "gradient_allreduce"), (ParameterAverageStrategy, "parameter_average"),
                      (PseudoGradientStrategy, "pseudo_gradient")):
        strategy = cls()
        assert (strategy.name, strategy.sync_every_steps, strategy.step_count) == (name, 1, 0)
    assert GradientAllreduceStrategy.reduces_gradients is True
    assert ParameterAverageStrategy.reduces_gradients is False
    assert PseudoGradientStrategy.reduces_gradients is False
    strategy = PseudoGradientStrategy(sync_every_steps=2, outer_lr=0.5, outer_momentum=0.9)
    assert (strategy.sync_every_steps, strategy.outer_lr, strategy.outer_momentum) == (2, 0.5, 0.9)


@pytest.mark.parametrize("invalid_sync", [0, 2, 5, -1, 1.5, False, "1"])
def test_gradient_allreduce_requires_sync_every_one(invalid_sync: Any):
    with pytest.raises(ValueError, match="gradient_allreduce requires sync_every_steps = 1"):
        GradientAllreduceStrategy(sync_every_steps=invalid_sync)


def test_gradient_allreduce_accepts_true_because_it_compares_by_value():
    """Unlike the local-SGD strategies, gradient_allreduce checks ``!= 1``, so ``True`` passes."""
    assert GradientAllreduceStrategy(sync_every_steps=True).sync_every_steps == 1  # type: ignore
    with pytest.raises(ValueError, match="positive integer"):
        ParameterAverageStrategy(sync_every_steps=True)  # type: ignore


@pytest.mark.parametrize("cls", LOCAL_SGD)
@pytest.mark.parametrize("invalid_sync", [0, -1, -10, 1.5, "1", None, True, False])
def test_local_sgd_strategies_reject_invalid_sync_cadence(cls, invalid_sync: Any):
    with pytest.raises(ValueError, match="positive integer"):
        cls(sync_every_steps=invalid_sync)


@pytest.mark.parametrize("invalid_outer_lr", [0.0, -0.01, -1.0, float("inf"), float("-inf"), float("nan"), True, False, "1.0", None])
def test_pseudo_gradient_rejects_invalid_outer_lr(invalid_outer_lr: Any):
    with pytest.raises(ValueError, match="outer_lr must be a finite positive float"):
        PseudoGradientStrategy(sync_every_steps=2, outer_lr=invalid_outer_lr)


@pytest.mark.parametrize("invalid_momentum", [-0.01, -1.0, 1.0, 1.001, 2.0, float("inf"), float("-inf"), float("nan"),
                                              True, False, "0.9", None])
def test_pseudo_gradient_rejects_invalid_outer_momentum(invalid_momentum: Any):
    with pytest.raises(ValueError, match=r"outer_momentum must be a finite float in \[0\.0, 1\.0\)"):
        PseudoGradientStrategy(sync_every_steps=2, outer_momentum=invalid_momentum)


@pytest.mark.parametrize("outer_lr,outer_momentum", [(1e-6, 0.0), (100.0, 0.999), (2, 0), (0.5, 0.9999)])
def test_pseudo_gradient_accepts_extreme_outer_settings_as_floats(outer_lr, outer_momentum):
    strategy = PseudoGradientStrategy(sync_every_steps=1, outer_lr=outer_lr, outer_momentum=outer_momentum)
    assert type(strategy.outer_lr) is float and type(strategy.outer_momentum) is float
    assert (strategy.outer_lr, strategy.outer_momentum) == (float(outer_lr), float(outer_momentum))


# ==============================================================================
# Replica-assertion cadence
# ==============================================================================

def test_gradient_allreduce_replica_digest_runs_on_an_interval(monkeypatch):
    # Review 2026-10-25 M5: a full host-side SHA-256 of model and optimizer
    # state after every update dominated step time.
    monkeypatch.delenv("HYPERGAN_REPLICA_CHECK_EVERY", raising=False)
    strategy = GradientAllreduceStrategy()
    assert strategy.replica_check_every == 100
    assert [s for s in range(0, 301) if strategy.should_assert_replicas(s)] == [0, 1, 100, 200, 300]
    explicit = GradientAllreduceStrategy(sync_every_steps=1, replica_check_every=5)
    assert [s for s in range(0, 25) if explicit.should_assert_replicas(s)] == [0, 1, 5, 10, 15, 20]
    monkeypatch.setenv("HYPERGAN_REPLICA_CHECK_EVERY", "1")
    assert all(GradientAllreduceStrategy().should_assert_replicas(s) for s in (0, 1, 2, 5, 100))
    for invalid in ("0", "often"):
        monkeypatch.setenv("HYPERGAN_REPLICA_CHECK_EVERY", invalid)
        with pytest.raises(ValueError, match="replica_check_every|HYPERGAN_REPLICA_CHECK_EVERY"):
            GradientAllreduceStrategy()


@pytest.mark.parametrize("cadence", [1, 2, 3, 5, 10, 100000])
def test_local_sgd_replicas_are_asserted_exactly_at_synchronizations(cadence: int):
    for strategy in (ParameterAverageStrategy(sync_every_steps=cadence),
                     PseudoGradientStrategy(sync_every_steps=cadence, outer_lr=1.0, outer_momentum=0.5)):
        for step in (*range(0, 31), cadence - 1, cadence, 2 * cadence):
            assert strategy.should_assert_replicas(step) is (step % cadence == 0), (strategy.name, cadence, step)


# ==============================================================================
# Outer-step mathematics (single replica, no process group)
# ==============================================================================

def test_pseudo_gradient_two_round_heavy_ball_update():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[10.0, 20.0]]))
    strategy = PseudoGradientStrategy(sync_every_steps=2, outer_lr=0.5, outer_momentum=0.9)
    strategy._init_anchor_and_momentum(models=model)
    assert torch.equal(strategy.anchor["weight"], torch.tensor([[10.0, 20.0]]))
    assert torch.equal(strategy.momentum_buffer["weight"], torch.tensor([[0.0, 0.0]]))

    # Local steps move the replica to [8, 17]: delta = anchor - local = [2, 3].
    model.weight.data.copy_(torch.tensor([[8.0, 17.0]]))
    strategy.on_step_complete(step=2, models=model)
    # v1 = 0.9 * 0 + [2, 3]; anchor1 = [10, 20] - 0.5 * v1 = [9, 18.5]
    torch.testing.assert_close(strategy.momentum_buffer["weight"], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(strategy.anchor["weight"], torch.tensor([[9.0, 18.5]]))
    torch.testing.assert_close(model.weight.data, torch.tensor([[9.0, 18.5]]))

    model.weight.data.copy_(torch.tensor([[8.5, 17.5]]))
    strategy.on_step_complete(step=4, models=model)
    # v2 = 0.9 * [2, 3] + [0.5, 1] = [2.3, 3.7]; anchor2 = [9, 18.5] - 0.5 * v2 = [7.85, 16.65]
    torch.testing.assert_close(strategy.momentum_buffer["weight"], torch.tensor([[2.3, 3.7]]))
    torch.testing.assert_close(strategy.anchor["weight"], torch.tensor([[7.85, 16.65]]))
    torch.testing.assert_close(model.weight.data, torch.tensor([[7.85, 16.65]]))


@pytest.mark.parametrize("outer_momentum", [0.9, 0.5, 0.0])
@pytest.mark.parametrize("outer_lr", [0.5, 1.0, 1.5])
def test_pseudo_gradient_matches_the_diloco_oracle_over_five_rounds(outer_momentum: float, outer_lr: float):
    torch.manual_seed(999)
    model = torch.nn.Linear(3, 2, bias=True)
    strategy = PseudoGradientStrategy(sync_every_steps=2, outer_lr=outer_lr, outer_momentum=outer_momentum)
    strategy.initialize(models=model)
    anchor = {'weight': model.weight.data.clone(), 'bias': model.bias.data.clone()}
    momentum = {key: torch.zeros_like(value) for key, value in anchor.items()}

    for round_index in range(1, 6):
        torch.manual_seed(100 * round_index)
        drift = {'weight': torch.randn_like(model.weight) * 0.2, 'bias': torch.randn_like(model.bias) * 0.2}
        model.weight.data.add_(drift['weight'])
        model.bias.data.add_(drift['bias'])
        local = {'weight': model.weight.data.clone(), 'bias': model.bias.data.clone()}
        anchor, momentum = oracle_pseudo_gradient_step(anchor, [local], outer_lr=outer_lr,
                                                       outer_momentum=outer_momentum, momentum_buffer=momentum)
        strategy.on_step_complete(step=2 * round_index, models=model)
        for key in ('weight', 'bias'):
            assert_states_close(strategy.anchor[key], anchor[key])
            assert_states_close(strategy.momentum_buffer[key], momentum[key])
            assert_states_close(getattr(model, key).data, anchor[key])
            if outer_momentum == 0.0:
                # Without momentum the outer velocity is exactly this round's pseudo-gradient.
                assert_states_close(strategy.momentum_buffer[key], -drift[key])


def test_unit_outer_lr_without_momentum_adopts_the_replica_mean_like_parameter_average():
    torch.manual_seed(12345)
    model = torch.nn.Linear(3, 2, bias=True)
    strategy = PseudoGradientStrategy(sync_every_steps=2, outer_lr=1.0, outer_momentum=0.0)
    strategy.initialize(models=model)
    model.weight.data.add_(torch.randn_like(model.weight) * 0.5)
    model.bias.data.add_(torch.randn_like(model.bias) * 0.5)
    local = {'weight': model.weight.data.clone(), 'bias': model.bias.data.clone()}
    strategy.on_step_complete(step=2, models=model)
    # anchor - 1.0 * (anchor - mean(local)) == mean(local): FedAvg with one replica.
    assert_states_close(model.weight.data, local['weight'], atol=1e-7, rtol=0)
    assert_states_close(model.bias.data, local['bias'], atol=1e-7, rtol=0)


@pytest.mark.parametrize("cadence", [4, 100000])
def test_off_cadence_steps_touch_neither_anchor_momentum_nor_weights(cadence):
    model = torch.nn.Linear(2, 2, bias=False)
    initial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model.weight.data.copy_(initial)
    strategy = PseudoGradientStrategy(sync_every_steps=cadence, outer_lr=1.0, outer_momentum=0.5)
    strategy.initialize(models=model)
    for step in range(1, min(cadence, 11)):
        model.weight.data.add_(0.5 if step == 1 else 0.0)
        strategy.on_step_complete(step=step, models=model)
        assert_states_close(strategy.anchor['weight'], initial)
        assert torch.count_nonzero(strategy.momentum_buffer['weight']) == 0
        assert_states_close(model.weight.data, initial + 0.5)
    if cadence == 4:
        # delta = initial - (initial + 0.5); new anchor = initial - 1.0 * delta = the local weights.
        strategy.on_step_complete(step=4, models=model)
        assert_states_close(strategy.anchor['weight'], initial + 0.5)
        assert_states_close(model.weight.data, initial + 0.5)


# ==============================================================================
# Placement of the pseudo-gradient outer state
# ==============================================================================

def test_pseudo_gradient_keeps_outer_state_on_the_parameter_device():
    """Review 2026-10-25 §4: anchor and momentum lived on the CPU, so every
    synchronization copied each parameter host->device and back. A meta-device
    model has no host copy at all, so any .cpu() round trip raises."""
    model = torch.nn.Linear(3, 2, device='meta')
    strategy = PseudoGradientStrategy(sync_every_steps=1, outer_lr=0.5, outer_momentum=0.9)
    strategy.initialize(models=model)
    for step in (1, 2):
        strategy.on_step_complete(step=step, models=model)
    for state in (strategy.anchor, strategy.momentum_buffer):
        assert {value.device.type for value in state.values()} == {'meta'}
    trainer_strategy = PseudoGradientStrategy(MockTrainer(device='meta'), sync_every_steps=1)
    assert {value.device.type for value in trainer_strategy.anchor.values()} == {'meta'}


def test_checkpointed_cpu_outer_state_loads_onto_the_parameter_device():
    """Distributed checkpoints store the aggregation state on the CPU; resume moves it
    to the trainer device. Older checkpoints with the aliased keys still load."""
    source = PseudoGradientStrategy(MockTrainer(), sync_every_steps=2, outer_lr=0.75, outer_momentum=0.8)
    source.on_step_complete(2)
    state = source.state_dict()
    legacy = {key: value for key, value in state.items() if key not in ('anchor', 'momentum_buffer')}
    legacy.update(anchor_weights=state['anchor'], momentum=state['momentum_buffer'])
    for saved in (state, legacy):
        restored = PseudoGradientStrategy(MockTrainer(device='meta'), sync_every_steps=1)
        restored.load_state_dict(saved)
        assert set(restored.anchor) == set(state['anchor']) and set(restored.momentum_buffer) == set(state['anchor'])
        for value in (*restored.anchor.values(), *restored.momentum_buffer.values()):
            assert value.device.type == 'meta'
        cpu = PseudoGradientStrategy(MockTrainer(), sync_every_steps=1)
        cpu.load_state_dict(saved)
        for key in state['anchor']:
            assert torch.equal(cpu.anchor[key], state['anchor'][key])
            assert torch.equal(cpu.momentum_buffer[key], state['momentum_buffer'][key])


# ==============================================================================
# State, manifest metadata and resume
# ==============================================================================

@pytest.mark.parametrize("strategy,expected", [
    (GradientAllreduceStrategy(), {"strategy": "gradient_allreduce", "sync_every_steps": 1}),
    (ParameterAverageStrategy(sync_every_steps=5), {"strategy": "parameter_average", "sync_every_steps": 5}),
    (PseudoGradientStrategy(sync_every_steps=100000, outer_lr=0.8, outer_momentum=0.25),
     {"strategy": "pseudo_gradient", "sync_every_steps": 100000, "outer_lr": 0.8, "outer_momentum": 0.25}),
])
def test_manifest_metadata(strategy, expected):
    assert strategy.manifest_metadata() == expected


@pytest.mark.parametrize("cls,cadence,step", [(GradientAllreduceStrategy, 1, 15), (ParameterAverageStrategy, 3, 6),
                                              (ParameterAverageStrategy, 5, 3)])
def test_cadence_state_round_trip(cls, cadence, step):
    strategy = cls(sync_every_steps=cadence)
    strategy.on_step_complete(step)
    state = strategy.state_dict()
    assert (state["strategy"], state["step_count"], state["sync_every_steps"]) == (strategy.name, step, cadence)
    restored = cls(sync_every_steps=1) if cls is not GradientAllreduceStrategy else cls()
    restored.load_state_dict(state)
    assert (restored.step_count, restored.sync_every_steps) == (step, cadence)
    assert restored.state_dict() == state


def test_pseudo_gradient_state_round_trip_uses_one_canonical_key_each():
    strategy = PseudoGradientStrategy(MockTrainer(), sync_every_steps=2, outer_lr=0.75, outer_momentum=0.8)
    strategy.on_step_complete(2)
    state = strategy.state_dict()
    # Review 2026-10-25 M6: one canonical key each, not four aliased clones.
    assert "anchor" in state and "momentum_buffer" in state
    assert "anchor_weights" not in state and "momentum" not in state
    assert (state["outer_lr"], state["outer_momentum"], state["sync_every_steps"], state["step_count"]) == (0.75, 0.8, 2, 2)
    restored = PseudoGradientStrategy(MockTrainer(), sync_every_steps=1)
    restored.load_state_dict(state)
    assert (restored.step_count, restored.sync_every_steps, restored.outer_lr, restored.outer_momentum) == (2, 2, 0.75, 0.8)
    for key in strategy.anchor:
        torch.testing.assert_close(restored.anchor[key], strategy.anchor[key])
        torch.testing.assert_close(restored.momentum_buffer[key], strategy.momentum_buffer[key])


def test_pseudo_gradient_resume_continues_the_uninterrupted_trajectory():
    """Save after round 2, load into a fresh strategy, and run round 3 on both."""
    continuous_model, resumed_model = torch.nn.Linear(2, 2, bias=True), torch.nn.Linear(2, 2, bias=True)
    for model in (continuous_model, resumed_model):
        model.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.bias.data.copy_(torch.tensor([0.5, -0.5]))
    continuous = PseudoGradientStrategy(sync_every_steps=2, outer_lr=0.8, outer_momentum=0.9)
    continuous.initialize(models=continuous_model)
    for step, drift in ((2, [[0.2, -0.1], [0.1, 0.3]]), (4, [[-0.1, 0.2], [-0.3, 0.1]])):
        continuous_model.weight.data.add_(torch.tensor(drift))
        continuous.on_step_complete(step=step, models=continuous_model)

    saved = continuous.state_dict()
    resumed = PseudoGradientStrategy(sync_every_steps=1, outer_lr=1.0, outer_momentum=0.0)
    resumed.load_state_dict(saved)
    assert (resumed.step_count, resumed.sync_every_steps, resumed.outer_lr, resumed.outer_momentum) == (4, 2, 0.8, 0.9)
    resumed_model.weight.data.copy_(resumed.anchor['weight'])
    resumed_model.bias.data.copy_(resumed.anchor['bias'])

    drift = torch.tensor([[0.3, 0.1], [-0.2, 0.4]])
    for model, strategy in ((continuous_model, continuous), (resumed_model, resumed)):
        model.weight.data.add_(drift)
        strategy.on_step_complete(step=6, models=model)
    assert_states_close(resumed.anchor, continuous.anchor, atol=1e-7, rtol=0)
    assert_states_close(resumed.momentum_buffer, continuous.momentum_buffer, atol=1e-7, rtol=0)
    assert_states_close(resumed_model.state_dict(), continuous_model.state_dict(), atol=1e-7, rtol=0)


# ==============================================================================
# Optimizer-state synchronization helpers (one-rank process group)
# ==============================================================================

@pytest.fixture
def one_rank_group(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_optimizer_synchronization_tolerates_missing_and_partial_moments(one_rank_group):
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    fresh = torch.optim.Adam([parameter], lr=0.01)
    assert len(fresh.state) == 0
    _synchronize_optimizer_moments([fresh], world_size=1, comm_device=torch.device("cpu"))
    _synchronize_optimizer_regularizers(optimizers=[fresh])

    sgd = torch.optim.SGD([parameter], lr=0.01)
    parameter.sum().backward()
    sgd.step()
    assert len(sgd.state) == 0
    _synchronize_optimizer_moments([sgd], world_size=1, comm_device=torch.device("cpu"))

    used, unused = torch.nn.Parameter(torch.randn(2, 2)), torch.nn.Parameter(torch.randn(2, 2))
    partial = torch.optim.Adam([used, unused], lr=0.01)
    used.sum().backward()
    partial.step()
    _synchronize_optimizer_moments([partial], world_size=1, comm_device=torch.device("cpu"))
    assert used in partial.state and unused not in partial.state
    # Moments on another device than the communication device are left alone.
    before = partial.state[used]["exp_avg"].clone()
    _synchronize_optimizer_moments([partial], world_size=2, comm_device=torch.device("meta"))
    assert torch.equal(partial.state[used]["exp_avg"], before)


def test_optimizer_synchronization_averages_moments_but_not_the_adam_step(one_rank_group):
    parameter = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    parameter.sum().backward()
    optimizer.step()
    step = float(optimizer.state[parameter]["step"])
    exp_avg = optimizer.state[parameter]["exp_avg"].clone()
    # One rank all-reduces to itself, so world_size=2 halves every averaged moment.
    _synchronize_optimizer_moments([optimizer], world_size=2, comm_device=torch.device("cpu"))
    assert float(optimizer.state[parameter]["step"]) == step
    assert_states_close(optimizer.state[parameter]["exp_avg"], exp_avg / 2)


# ==============================================================================
# Registry
# ==============================================================================

def test_canonical_strategies_are_registered_and_unknown_names_fail():
    assert get_aggregation_strategy("gradient_allreduce") is GradientAllreduceStrategy
    assert get_aggregation_strategy("parameter_average") is ParameterAverageStrategy
    assert get_aggregation_strategy("pseudo_gradient") is PseudoGradientStrategy
    with pytest.raises(ValueError, match="Unknown aggregation strategy: 'totally_nonexistent_strategy'"):
        resolve_aggregation_strategy({"strategy": "totally_nonexistent_strategy"})


def test_the_dead_aggregation_aliases_are_gone():
    """Review 2026-10-25 §4: the *Aggregation aliases were used only by these tests."""
    for name in ("GradientAllreduceAggregation", "ParameterAverageAggregation", "PseudoGradientAggregation"):
        assert not hasattr(aggregation, name)
    assert all(hasattr(aggregation, name) for name in aggregation.__all__)


@pytest.mark.parametrize("name", ["", None, 12345])
def test_registration_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="non-empty string"):
        register_aggregation_strategy(name, ParameterAverageStrategy)  # type: ignore


def test_custom_strategy_registration_resolution_and_profile_validation():
    class CustomStrategy:
        name = "unit_custom_strategy"

        def __init__(self, trainer: Any = None, custom_factor: float = 2.0, sync_every_steps: int = 1, **kwargs: Any):
            self.trainer, self.custom_factor, self.sync_every_steps, self.step_count = trainer, float(custom_factor), sync_every_steps, 0

        def on_backward(self, optimizer: Any, name: str) -> None:
            pass

        def on_step_complete(self, step: int) -> None:
            self.step_count = step

        def should_assert_replicas(self, step: int) -> bool:
            return step % self.sync_every_steps == 0

        def state_dict(self) -> Dict[str, Any]:
            return {"step_count": self.step_count, "custom_factor": self.custom_factor}

        def load_state_dict(self, state: Dict[str, Any]) -> None:
            self.step_count, self.custom_factor = state["step_count"], state["custom_factor"]

        def manifest_metadata(self) -> Dict[str, Any]:
            return {"strategy": self.name, "sync_every_steps": self.sync_every_steps}

    register_aggregation_strategy("unit_custom_strategy", CustomStrategy)
    assert get_aggregation_strategy("unit_custom_strategy") is CustomStrategy
    flat = create_aggregation_strategy({"strategy": "unit_custom_strategy", "custom_factor": 3.5, "sync_every_steps": 3})
    assert isinstance(flat, CustomStrategy) and flat.custom_factor == 3.5
    flat.on_step_complete(3)
    assert flat.should_assert_replicas(3) is True and flat.should_assert_replicas(4) is False
    assert resolve_aggregation_strategy({"aggregation": {"strategy": "unit_custom_strategy", "custom_factor": 4.5}}).custom_factor == 4.5
    nested = resolve_aggregation_strategy({"execution": {"aggregation": {"strategy": "unit_custom_strategy", "custom_factor": 5.5}}},
                                          trainer=None)
    assert nested.custom_factor == 5.5

    # The execution-profile validator accepts the registered name but only the canonical options.
    config = make_toy_config(device="cpu", batch_size=8)
    execution = {'name': 'cpu-replicated-gloo', 'world_size': 2}
    resolved = resolve_execution_profile({'schema_version': 1, 'execution': execution,
                                          'aggregation': {'strategy': 'unit_custom_strategy', 'sync_every_steps': 1}}, config)
    assert resolved['execution']['aggregation']['strategy'] == 'unit_custom_strategy'
    with pytest.raises(ValueError, match="Unknown aggregation option"):
        resolve_execution_profile({'schema_version': 1, 'execution': execution,
                                   'aggregation': {'strategy': 'unit_custom_strategy', 'custom_factor': 6.0}}, config)


# ==============================================================================
# Real Gloo ranks (heavy: each case starts one worker process per rank)
# ==============================================================================

def _launch(tmp_path, mode, world_size, *args, timeout=60):
    stamp = time.monotonic_ns()
    rendezvous = (tmp_path / f"rendezvous-{mode}-{stamp}").as_uri()
    outputs = [tmp_path / f"{mode}-{stamp}-rank{rank}.json" for rank in range(world_size)]
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve())]
    processes = [subprocess.Popen([*command, mode, str(rank), str(world_size), rendezvous, str(outputs[rank]), *map(str, args)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for rank in range(world_size)]
    failures, deadline = [], time.monotonic() + timeout
    try:
        for rank, process in enumerate(processes):
            _, stderr = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
            if process.returncode:
                failures.append(f"rank {rank} exited {process.returncode}:\n{stderr}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert not failures, "\n".join(failures)
    return [json.loads(path.read_text()) for path in outputs]


def _mean_worker(rank, world_size):
    """Rank-specific gradients through GradientAllreduceStrategy.on_backward outside a trainer."""
    torch.manual_seed(100 + rank)
    first, second = torch.nn.Parameter(torch.zeros(4, 4)), torch.nn.Parameter(torch.zeros(3))
    first.grad = torch.randn(4, 4) * (rank + 1.5)
    second.grad = torch.tensor([-2.0, 0.0, 5.0]) * (rank - 1.0)
    local = [first.grad.tolist(), second.grad.tolist()]

    class Optimizer:
        param_groups = [{'params': [first, second]}]

    GradientAllreduceStrategy().on_backward(Optimizer(), "test")
    return {"local": local, "reduced": [first.grad.tolist(), second.grad.tolist()]}


def _drift_worker(rank, world_size, strategy, cadence, steps):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    cadence, steps = int(cadence), int(steps)
    spec = {"strategy": strategy, "sync_every_steps": cadence}
    if strategy == "pseudo_gradient":
        spec.update(outer_lr=1.0, outer_momentum=0.9)
    trainer = ReplicatedCPUTrainer(make_toy_config(device="cpu", batch_size=8, steps=steps), world_size=world_size,
                                   aggregation=spec)
    # Distinct rank shards make the replicas drift between synchronizations. Each
    # synchronization step also asserts complete replica agreement inside update().
    data, ids = torch.linspace(-1, 1, 16).reshape(8, 2), torch.arange(8)
    local_ids = ids.chunk(world_size)[rank]
    weights, flags = [], []
    for step in range(1, steps + 1):
        trainer.update({'real': data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
        weights.append([p.detach().tolist() for p in trainer.graph.models['generator'].parameters()])
        flags.append(trainer.aggregation_strategy.should_assert_replicas(step))
    return {"weights": weights, "flags": flags}


def _worker(mode, rank, world_size, rendezvous, output, *args):
    rank, world_size = int(rank), int(world_size)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=30))
    try:
        result = {"mean": _mean_worker, "drift": _drift_worker}[mode](rank, world_size, *args)
        Path(output).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.heavy
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_gradient_allreduce_on_backward_means_gradients_across_ranks(tmp_path, world_size: int):
    ranks = _launch(tmp_path, "mean", world_size)
    for index in range(2):
        mean = torch.stack([torch.tensor(rank["local"][index]) for rank in ranks]).mean(dim=0)
        for rank in ranks:
            torch.testing.assert_close(torch.tensor(rank["reduced"][index]), mean, rtol=1e-6, atol=1e-7)
            assert rank["reduced"][index] == ranks[0]["reduced"][index]


def _difference(left, right):
    return sum(torch.norm(torch.tensor(a) - torch.tensor(b)).item() for a, b in zip(left, right))


@pytest.mark.heavy
@pytest.mark.parametrize("strategy,cadence,steps", [
    ("parameter_average", 1, 3), ("parameter_average", 4, 8), ("pseudo_gradient", 2, 4),
])
def test_local_sgd_replicas_drift_between_synchronizations_and_agree_at_them(tmp_path, strategy, cadence, steps):
    ranks = _launch(tmp_path, "drift", 2, strategy, cadence, steps, timeout=90)
    for step in range(1, steps + 1):
        synchronized = step % cadence == 0
        assert [rank["flags"][step - 1] for rank in ranks] == [synchronized, synchronized]
        difference = _difference(ranks[0]["weights"][step - 1], ranks[1]["weights"][step - 1])
        if synchronized:
            assert difference == 0.0, f"replicas differ after synchronization step {step}: {difference}"
        else:
            assert difference > 1e-4, f"replicas did not drift on local step {step}: {difference}"


if __name__ == "__main__":
    _worker(*sys.argv[1:])

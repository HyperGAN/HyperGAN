"""Unit tests for pluggable learning aggregation strategies, registry, and recovery state (Issue #186)."""
from __future__ import annotations

import math
from typing import Any, Dict
import pytest
import torch

from hypergan.aggregation import (
    AggregationStrategy,
    GradientAllreduceStrategy,
    ParameterAverageStrategy,
    PseudoGradientStrategy,
    GradientAllreduceAggregation,
    ParameterAverageAggregation,
    PseudoGradientAggregation,
    register_aggregation_strategy,
    get_aggregation_strategy,
    create_aggregation_strategy,
    resolve_aggregation_strategy,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2, bias=True)
        self.register_buffer("running_mean", torch.tensor([1.0, 2.0]))
        self.register_buffer("int_counter", torch.tensor(10, dtype=torch.int64))


class MockTrainer:
    def __init__(self):
        self.graph = DummyModel()
        self.world_size = 2
        self.step = 0
        self.device = torch.device("cpu")


# ==============================================================================
# 1. GradientAllreduceStrategy Tests
# ==============================================================================

class TestGradientAllreduceStrategyUnit:
    def test_init_defaults(self):
        strat = GradientAllreduceStrategy()
        assert strat.name == "gradient_allreduce"
        assert strat.sync_every_steps == 1
        assert strat.step_count == 0

    def test_rejects_sync_every_steps_not_equal_to_one(self):
        with pytest.raises(ValueError, match="gradient_allreduce requires sync_every_steps = 1"):
            GradientAllreduceStrategy(sync_every_steps=2)
        with pytest.raises(ValueError, match="gradient_allreduce requires sync_every_steps = 1"):
            GradientAllreduceStrategy(sync_every_steps=0)

    def test_replica_digest_runs_on_an_interval_not_every_step(self, monkeypatch):
        # Review 2026-10-25 M5: a full host-side SHA-256 of model and optimizer
        # state after every update dominated step time.
        monkeypatch.delenv("HYPERGAN_REPLICA_CHECK_EVERY", raising=False)
        strat = GradientAllreduceStrategy()
        assert strat.replica_check_every == 100
        assert [s for s in range(0, 301) if strat.should_assert_replicas(s)] == [0, 1, 100, 200, 300]
        monkeypatch.setenv("HYPERGAN_REPLICA_CHECK_EVERY", "1")
        debug = GradientAllreduceStrategy()
        assert all(debug.should_assert_replicas(s) for s in (0, 1, 2, 5, 100))
        monkeypatch.setenv("HYPERGAN_REPLICA_CHECK_EVERY", "0")
        with pytest.raises(ValueError, match="replica_check_every"):
            GradientAllreduceStrategy()

    def test_reduces_gradients_capability(self):
        assert GradientAllreduceStrategy.reduces_gradients is True
        assert ParameterAverageStrategy.reduces_gradients is False
        assert PseudoGradientStrategy.reduces_gradients is False

    def test_manifest_metadata(self):
        strat = GradientAllreduceStrategy()
        meta = strat.manifest_metadata()
        assert meta == {"strategy": "gradient_allreduce", "sync_every_steps": 1}

    def test_state_dict_and_load(self):
        strat = GradientAllreduceStrategy()
        strat.on_step_complete(15)
        state = strat.state_dict()
        assert state["strategy"] == "gradient_allreduce"
        assert state["step_count"] == 15
        assert state["sync_every_steps"] == 1

        new_strat = GradientAllreduceStrategy()
        new_strat.load_state_dict(state)
        assert new_strat.step_count == 15
        assert new_strat.sync_every_steps == 1


# ==============================================================================
# 2. ParameterAverageStrategy Tests
# ==============================================================================

class TestParameterAverageStrategyUnit:
    def test_init_defaults(self):
        strat = ParameterAverageStrategy(sync_every_steps=4)
        assert strat.name == "parameter_average"
        assert strat.sync_every_steps == 4
        assert strat.step_count == 0

    def test_rejects_invalid_sync_every_steps(self):
        with pytest.raises(ValueError, match="positive integer"):
            ParameterAverageStrategy(sync_every_steps=0)
        with pytest.raises(ValueError, match="positive integer"):
            ParameterAverageStrategy(sync_every_steps=-2)
        with pytest.raises(ValueError, match="positive integer"):
            ParameterAverageStrategy(sync_every_steps=1.5)  # type: ignore

    def test_should_assert_replicas_cadence(self):
        strat = ParameterAverageStrategy(sync_every_steps=4)
        assert strat.should_assert_replicas(0) is True
        assert strat.should_assert_replicas(1) is False
        assert strat.should_assert_replicas(2) is False
        assert strat.should_assert_replicas(3) is False
        assert strat.should_assert_replicas(4) is True
        assert strat.should_assert_replicas(8) is True
        assert strat.should_assert_replicas(9) is False

    def test_manifest_metadata(self):
        strat = ParameterAverageStrategy(sync_every_steps=5)
        meta = strat.manifest_metadata()
        assert meta == {"strategy": "parameter_average", "sync_every_steps": 5}

    def test_state_dict_and_load(self):
        strat = ParameterAverageStrategy(sync_every_steps=3)
        strat.on_step_complete(6)
        state = strat.state_dict()
        assert state["strategy"] == "parameter_average"
        assert state["step_count"] == 6
        assert state["sync_every_steps"] == 3

        restored = ParameterAverageStrategy(sync_every_steps=1)
        restored.load_state_dict(state)
        assert restored.step_count == 6
        assert restored.sync_every_steps == 3
        assert restored.state_dict() == state


# ==============================================================================
# 3. PseudoGradientStrategy Tests
# ==============================================================================

class TestPseudoGradientStrategyUnit:
    def test_init_validation(self):
        strat = PseudoGradientStrategy(sync_every_steps=2, outer_lr=0.5, outer_momentum=0.9)
        assert strat.name == "pseudo_gradient"
        assert strat.sync_every_steps == 2
        assert strat.outer_lr == 0.5
        assert strat.outer_momentum == 0.9

        # sync_every_steps validation
        with pytest.raises(ValueError, match="positive integer"):
            PseudoGradientStrategy(sync_every_steps=0)
        with pytest.raises(ValueError, match="positive integer"):
            PseudoGradientStrategy(sync_every_steps=-1)

        # outer_lr validation
        with pytest.raises(ValueError, match="outer_lr must be a finite positive float"):
            PseudoGradientStrategy(outer_lr=0.0)
        with pytest.raises(ValueError, match="outer_lr must be a finite positive float"):
            PseudoGradientStrategy(outer_lr=-0.5)
        with pytest.raises(ValueError, match="outer_lr must be a finite positive float"):
            PseudoGradientStrategy(outer_lr=float("inf"))

        # outer_momentum validation
        with pytest.raises(ValueError, match="outer_momentum must be a finite float in \\[0.0, 1.0\\)"):
            PseudoGradientStrategy(outer_momentum=-0.1)
        with pytest.raises(ValueError, match="outer_momentum must be a finite float in \\[0.0, 1.0\\)"):
            PseudoGradientStrategy(outer_momentum=1.0)
        with pytest.raises(ValueError, match="outer_momentum must be a finite float in \\[0.0, 1.0\\)"):
            PseudoGradientStrategy(outer_momentum=1.5)
        with pytest.raises(ValueError, match="outer_momentum must be a finite float in \\[0.0, 1.0\\)"):
            PseudoGradientStrategy(outer_momentum=float("nan"))

    def test_should_assert_replicas_cadence(self):
        strat = PseudoGradientStrategy(sync_every_steps=3, outer_lr=1.0, outer_momentum=0.5)
        assert strat.should_assert_replicas(1) is False
        assert strat.should_assert_replicas(2) is False
        assert strat.should_assert_replicas(3) is True
        assert strat.should_assert_replicas(6) is True

    def test_manifest_metadata(self):
        strat = PseudoGradientStrategy(sync_every_steps=4, outer_lr=0.8, outer_momentum=0.25)
        meta = strat.manifest_metadata()
        assert meta == {
            "strategy": "pseudo_gradient",
            "sync_every_steps": 4,
            "outer_lr": 0.8,
            "outer_momentum": 0.25,
        }

    def test_diloco_mathematical_update_sequence(self):
        """Verify mathematical formula for 2-round momentum pseudo-gradient sequence."""
        model = torch.nn.Linear(2, 1, bias=False)
        # Fix initial weights to [10.0, 20.0]
        model.weight.data.copy_(torch.tensor([[10.0, 20.0]]))

        outer_lr = 0.5
        outer_momentum = 0.9
        strat = PseudoGradientStrategy(
            sync_every_steps=2,
            outer_lr=outer_lr,
            outer_momentum=outer_momentum,
        )
        strat._init_anchor_and_momentum(models=model)
        assert torch.equal(strat.anchor["weight"], torch.tensor([[10.0, 20.0]]))
        assert torch.equal(strat.momentum_buffer["weight"], torch.tensor([[0.0, 0.0]]))

        # Simulate local steps: model weights move to [8.0, 17.0]
        # Delta = Anchor - Local = [10.0, 20.0] - [8.0, 17.0] = [2.0, 3.0]
        model.weight.data.copy_(torch.tensor([[8.0, 17.0]]))

        # Round 1 synchronization (step 2)
        strat.on_step_complete(step=2, models=model)
        # v1 = 0.9 * 0 + [2.0, 3.0] = [2.0, 3.0]
        # anchor1 = [10.0, 20.0] - 0.5 * [2.0, 3.0] = [9.0, 18.5]
        expected_v1 = torch.tensor([[2.0, 3.0]])
        expected_anchor1 = torch.tensor([[9.0, 18.5]])
        torch.testing.assert_close(strat.momentum_buffer["weight"], expected_v1)
        torch.testing.assert_close(strat.anchor["weight"], expected_anchor1)
        torch.testing.assert_close(model.weight.data, expected_anchor1)

        # Simulate round 2 local steps from anchor1:
        # Local moves to [8.5, 17.5]
        # Delta2 = Anchor1 - Local = [9.0, 18.5] - [8.5, 17.5] = [0.5, 1.0]
        model.weight.data.copy_(torch.tensor([[8.5, 17.5]]))

        # Round 2 synchronization (step 4)
        strat.on_step_complete(step=4, models=model)
        # v2 = 0.9 * [2.0, 3.0] + [0.5, 1.0] = [1.8 + 0.5, 2.7 + 1.0] = [2.3, 3.7]
        # anchor2 = [9.0, 18.5] - 0.5 * [2.3, 3.7] = [9.0 - 1.15, 18.5 - 1.85] = [7.85, 16.65]
        expected_v2 = torch.tensor([[2.3, 3.7]])
        expected_anchor2 = torch.tensor([[7.85, 16.65]])
        torch.testing.assert_close(strat.momentum_buffer["weight"], expected_v2)
        torch.testing.assert_close(strat.anchor["weight"], expected_anchor2)
        torch.testing.assert_close(model.weight.data, expected_anchor2)

    def test_state_dict_roundtrip_and_persistence(self):
        trainer = MockTrainer()
        strat = PseudoGradientStrategy(trainer, sync_every_steps=2, outer_lr=0.75, outer_momentum=0.8)
        strat.on_step_complete(2)

        state = strat.state_dict()
        # Review 2026-10-25 M6: one canonical key each, not four aliased clones.
        assert "anchor" in state
        assert "momentum_buffer" in state
        assert "anchor_weights" not in state
        assert "momentum" not in state
        assert state["outer_lr"] == 0.75
        assert state["outer_momentum"] == 0.8
        assert state["sync_every_steps"] == 2
        assert state["step_count"] == 2

        restored = PseudoGradientStrategy(MockTrainer(), sync_every_steps=1)
        restored.load_state_dict(state)
        assert restored.step_count == 2
        assert restored.sync_every_steps == 2
        assert restored.outer_lr == 0.75
        assert restored.outer_momentum == 0.8
        for k in strat.anchor:
            torch.testing.assert_close(restored.anchor[k], strat.anchor[k])
            torch.testing.assert_close(restored.momentum_buffer[k], strat.momentum_buffer[k])


# ==============================================================================
# 4. Registry & Resolution Tests
# ==============================================================================

class TestAggregationRegistry:
    def test_canonical_strategies_registered(self):
        assert get_aggregation_strategy("gradient_allreduce") is GradientAllreduceStrategy
        assert get_aggregation_strategy("parameter_average") is ParameterAverageStrategy
        assert get_aggregation_strategy("pseudo_gradient") is PseudoGradientStrategy

    def test_aliases_match(self):
        assert GradientAllreduceAggregation is GradientAllreduceStrategy
        assert ParameterAverageAggregation is ParameterAverageStrategy
        assert PseudoGradientAggregation is PseudoGradientStrategy

    def test_unknown_strategy_raises_error(self):
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            get_aggregation_strategy("nonexistent_strat")

    def test_register_and_create_custom_strategy(self):
        class CustomStrategy:
            name = "unit_custom_test"
            def __init__(self, trainer: Any = None, custom_param: int = 42, **kwargs: Any):
                self.trainer = trainer
                self.custom_param = custom_param

            def on_backward(self, optimizer: Any, name: str) -> None: pass
            def on_step_complete(self, step: int) -> None: pass
            def should_assert_replicas(self, step: int) -> bool: return True
            def state_dict(self) -> Dict[str, Any]: return {"custom_param": self.custom_param}
            def load_state_dict(self, state: Dict[str, Any]) -> None: pass
            def manifest_metadata(self) -> Dict[str, Any]: return {"strategy": "unit_custom_test"}

        register_aggregation_strategy("unit_custom_test", CustomStrategy)
        assert get_aggregation_strategy("unit_custom_test") is CustomStrategy

        # Resolve from flat dict
        strat1 = create_aggregation_strategy({"strategy": "unit_custom_test", "custom_param": 99})
        assert strat1.name == "unit_custom_test"
        assert getattr(strat1, "custom_param") == 99

        # Resolve from nested profile dict
        strat2 = resolve_aggregation_strategy({"aggregation": {"strategy": "unit_custom_test", "custom_param": 100}})
        assert getattr(strat2, "custom_param") == 100

        # Resolve from execution table dict
        strat3 = resolve_aggregation_strategy({"execution": {"aggregation": {"strategy": "unit_custom_test", "custom_param": 101}}})
        assert getattr(strat3, "custom_param") == 101

    def test_register_invalid_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_aggregation_strategy("", GradientAllreduceStrategy)
        with pytest.raises(ValueError, match="non-empty string"):
            register_aggregation_strategy(None, GradientAllreduceStrategy)  # type: ignore

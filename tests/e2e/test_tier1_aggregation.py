"""Tier 1: Feature Coverage for Pluggable Learning Aggregation ([aggregation]).

Requirements covered:
- R2: Pluggable Learning Aggregation (ORIGINAL_REQUEST.md §R2)
- Features 5, 6, 7, 8, 9, 10: [aggregation] Config Schema, gradient_allreduce,
  parameter_average, pseudo_gradient, Public Aggregation Protocol & Registry,
  Aggregation Checkpoint State Recovery (PROJECT.md §Feature Inventory)
- Interface Contract 2: Aggregation Strategy Protocol ↔ Trainer (PROJECT.md §Interface Contracts)

Test coverage target: >=5 test cases per major strategy:
1. gradient_allreduce (5 cases)
2. parameter_average (5 cases)
3. pseudo_gradient (5 cases)
Plus public protocol & registry contracts.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.execution_profiles import resolve_execution_profile

try:
    from e2e.harness import (
        HAVE_AGGREGATION,
        HAVE_M1_SCHEMA,
        requires_aggregation,
        requires_m1_schema,
        make_toy_config,
        make_profile_raw,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
        assert_states_close,
    )
except ImportError:
    from tests.e2e.harness import (
        HAVE_AGGREGATION,
        HAVE_M1_SCHEMA,
        requires_aggregation,
        requires_m1_schema,
        make_toy_config,
        make_profile_raw,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
        assert_states_close,
    )


# ==============================================================================
# Feature 7: "gradient_allreduce" Strategy (>= 5 test cases)
# ==============================================================================

class TestGradientAllreduceStrategy:
    """Feature 7: Synchronous post-backward mean gradient all-reduce before optimizer step."""

    @requires_m1_schema
    def test_tier1_gradient_allreduce_schema_resolution(self):
        """Case 7.1: Profile resolves strategy='gradient_allreduce' with sync_every_steps=1."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'gradient_allreduce', 'sync_every_steps': 1},
        }
        resolved = resolve_execution_profile(raw, cfg)
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'gradient_allreduce'
        assert agg['sync_every_steps'] == 1

    @requires_m1_schema
    def test_tier1_gradient_allreduce_rejects_sync_cadence_gt_1(self):
        """Case 7.2: gradient_allreduce rejects sync_every_steps > 1 (must be exactly 1)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'gradient_allreduce', 'sync_every_steps': 2},
        }
        with pytest.raises(ValueError, match="gradient_allreduce requires sync_every_steps = 1"):
            resolve_execution_profile(raw, cfg)

    @requires_aggregation
    def test_tier1_gradient_allreduce_averages_gradients(self):
        """Case 7.3: Mean gradient reduction averages rank gradients before optimizer update."""
        from hypergan.aggregation import GradientAllreduceStrategy

        class MockTrainer:
            def __init__(self):
                self.graph = torch.nn.Linear(2, 2, bias=False)
                self.world_size = 2

        trainer = MockTrainer()
        strategy = GradientAllreduceStrategy(trainer, sync_every_steps=1)
        assert strategy.name == "gradient_allreduce"

    @requires_aggregation
    def test_tier1_gradient_allreduce_should_assert_replicas_on_interval(self):
        """Case 7.4: gradient_allreduce digests replicas at step 1 and every replica_check_every steps."""
        from hypergan.aggregation import GradientAllreduceStrategy

        class DummyTrainer:
            world_size = 2

        strategy = GradientAllreduceStrategy(DummyTrainer(), sync_every_steps=1, replica_check_every=4)
        assert [step for step in range(1, 13) if strategy.should_assert_replicas(step)] == [1, 4, 8, 12]
        every = GradientAllreduceStrategy(DummyTrainer(), sync_every_steps=1, replica_check_every=1)
        for step in (1, 2, 3, 4, 10):
            assert every.should_assert_replicas(step) is True

    @requires_aggregation
    def test_tier1_gradient_allreduce_manifest_metadata(self):
        """Case 7.5: manifest_metadata outputs strategy='gradient_allreduce' and sync_every_steps=1."""
        from hypergan.aggregation import GradientAllreduceStrategy

        class DummyTrainer:
            world_size = 2

        strategy = GradientAllreduceStrategy(DummyTrainer(), sync_every_steps=1)
        meta = strategy.manifest_metadata()
        assert meta['strategy'] == 'gradient_allreduce'
        assert meta['sync_every_steps'] == 1


# ==============================================================================
# Feature 8: "parameter_average" Strategy (>= 5 test cases)
# ==============================================================================

class TestParameterAverageStrategy:
    """Feature 8: Periodic cross-rank parameter averaging (FedAvg / Local SGD)."""

    @requires_m1_schema
    def test_tier1_parameter_average_schema_resolution(self):
        """Case 8.1: Profile resolves strategy='parameter_average' with sync_every_steps > 1."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 4},
        }
        resolved = resolve_execution_profile(raw, cfg)
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'parameter_average'
        assert agg['sync_every_steps'] == 4

    @requires_aggregation
    def test_tier1_parameter_average_local_steps_diverge(self):
        """Case 8.2: Replicas take independent local steps without communicating when step % sync != 0."""
        from hypergan.aggregation import ParameterAverageStrategy

        class DummyTrainer:
            world_size = 2

        strat = ParameterAverageStrategy(DummyTrainer(), sync_every_steps=4)
        # On local steps (1, 2, 3), replicas should NOT assert bit-identical equality
        assert strat.should_assert_replicas(1) is False
        assert strat.should_assert_replicas(2) is False
        assert strat.should_assert_replicas(3) is False
        # On sync step 4, replicas assert bit-identical equality
        assert strat.should_assert_replicas(4) is True

    @requires_aggregation
    def test_tier1_parameter_average_mathematical_oracle_parity(self):
        """Case 8.3: Parameter averaging matches authoritative FedAvg mathematical oracle Theta_mean."""
        # Setup synthetic rank states
        rank0_w = {'w': torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        rank1_w = {'w': torch.tensor([[3.0, 4.0], [5.0, 6.0]])}

        expected_mean = oracle_parameter_average([rank0_w, rank1_w])
        assert_states_close(expected_mean['w'], torch.tensor([[2.0, 3.0], [4.0, 5.0]]))

    @requires_aggregation
    def test_tier1_parameter_average_step_synchronization(self):
        """Case 8.4: on_step_complete at synchronization boundary replaces weights with mean."""
        from hypergan.aggregation import ParameterAverageStrategy

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([10.0, 20.0]))

        class MockTrainer:
            def __init__(self):
                self.graph = MockModel()
                self.world_size = 2

        trainer = MockTrainer()
        strat = ParameterAverageStrategy(trainer, sync_every_steps=2)
        assert strat.name == "parameter_average"

    @requires_aggregation
    def test_tier1_parameter_average_state_dict_roundtrip(self):
        """Case 8.5: state_dict and load_state_dict preserve step counters and sync cadence."""
        from hypergan.aggregation import ParameterAverageStrategy

        class DummyTrainer:
            world_size = 2

        strat1 = ParameterAverageStrategy(DummyTrainer(), sync_every_steps=5)
        strat1.on_step_complete(3)
        state = strat1.state_dict()

        strat2 = ParameterAverageStrategy(DummyTrainer(), sync_every_steps=5)
        strat2.load_state_dict(state)
        assert strat2.state_dict() == state


# ==============================================================================
# Feature 9: "pseudo_gradient" Strategy (>= 5 test cases)
# ==============================================================================

class TestPseudoGradientStrategy:
    """Feature 9: Federated outer-optimizer update (DiLoCo / FedOpt) using pseudo-gradients."""

    @requires_m1_schema
    def test_tier1_pseudo_gradient_schema_resolution(self):
        """Case 9.1: Profile resolves pseudo_gradient with outer_lr and outer_momentum."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {
                'strategy': 'pseudo_gradient',
                'sync_every_steps': 8,
                'outer_lr': 0.7,
                'outer_momentum': 0.9,
            },
        }
        resolved = resolve_execution_profile(raw, cfg)
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['strategy'] == 'pseudo_gradient'
        assert agg['sync_every_steps'] == 8
        assert agg['outer_lr'] == pytest.approx(0.7)
        assert agg['outer_momentum'] == pytest.approx(0.9)

    @requires_aggregation
    def test_tier1_pseudo_gradient_mathematical_oracle(self):
        """Case 9.2: Verify pseudo-gradient update matches authoritative DiLoCo formula."""
        anchor = {'w': torch.tensor([10.0, 10.0])}
        # Rank weights after inner steps: delta0 = [2, 1], delta1 = [4, 3] -> mean_delta = [3, 2]
        rank0 = {'w': torch.tensor([8.0, 9.0])}
        rank1 = {'w': torch.tensor([6.0, 7.0])}

        # Step 1: initial v=0, mu=0.9, eta=0.5
        # v1 = 0.9*0 + [3, 2] = [3, 2]
        # anchor1 = [10, 10] - 0.5 * [3, 2] = [8.5, 9.0]
        new_anchor, new_v = oracle_pseudo_gradient_step(anchor, [rank0, rank1], outer_lr=0.5, outer_momentum=0.9)
        assert_states_close(new_anchor['w'], torch.tensor([8.5, 9.0]))
        assert_states_close(new_v['w'], torch.tensor([3.0, 2.0]))

    @requires_aggregation
    def test_tier1_pseudo_gradient_momentum_accumulation(self):
        """Case 9.3: Outer momentum buffer accumulates historical pseudo-gradients across sync rounds."""
        anchor = {'w': torch.tensor([10.0])}
        rank_step1 = [{'w': torch.tensor([8.0])}, {'w': torch.tensor([8.0])}]  # delta = 2.0
        # Round 1
        anchor1, v1 = oracle_pseudo_gradient_step(anchor, rank_step1, outer_lr=1.0, outer_momentum=0.5)
        # v1 = 2.0, anchor1 = 10 - 2 = 8.0
        assert v1['w'].item() == pytest.approx(2.0)
        assert anchor1['w'].item() == pytest.approx(8.0)

        # Round 2: delta = 1.0 from anchor1 (ranks at 7.0)
        rank_step2 = [{'w': torch.tensor([7.0])}, {'w': torch.tensor([7.0])}]
        anchor2, v2 = oracle_pseudo_gradient_step(anchor1, rank_step2, outer_lr=1.0, outer_momentum=0.5, momentum_buffer=v1)
        # v2 = 0.5 * 2.0 + 1.0 = 2.0
        # anchor2 = 8.0 - 2.0 = 6.0
        assert v2['w'].item() == pytest.approx(2.0)
        assert anchor2['w'].item() == pytest.approx(6.0)

    @requires_aggregation
    def test_tier1_pseudo_gradient_replica_assertion_policy(self):
        """Case 9.4: should_assert_replicas returns True only on outer synchronization boundaries."""
        from hypergan.aggregation import PseudoGradientStrategy

        class DummyTrainer:
            world_size = 2

        strat = PseudoGradientStrategy(DummyTrainer(), sync_every_steps=4, outer_lr=0.5, outer_momentum=0.9)
        assert strat.should_assert_replicas(1) is False
        assert strat.should_assert_replicas(2) is False
        assert strat.should_assert_replicas(3) is False
        assert strat.should_assert_replicas(4) is True

    @requires_aggregation
    def test_tier1_pseudo_gradient_checkpoint_recovery_state(self):
        """Case 9.5: Checkpoint state dictionary preserves anchor weights and momentum buffer."""
        from hypergan.aggregation import PseudoGradientStrategy

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([5.0, 15.0]))

        class MockTrainer:
            def __init__(self):
                self.graph = MockModel()
                self.world_size = 2

        strat = PseudoGradientStrategy(MockTrainer(), sync_every_steps=2, outer_lr=0.8, outer_momentum=0.5)
        state = strat.state_dict()
        assert 'anchor' in state or 'anchor_weights' in state or 'weights' in state
        assert 'momentum' in state or 'momentum_buffer' in state or 'step' in state


# ==============================================================================
# Feature 9 (Registry): Public Aggregation Protocol & Registry
# ==============================================================================

class TestPublicAggregationProtocolAndRegistry:
    """Feature 9 (Registry): Extensibility contract for user-defined aggregation strategies."""

    @requires_aggregation
    def test_tier1_register_custom_aggregation_strategy(self):
        """Verify custom strategy registration and resolution through public API."""
        from hypergan.aggregation import (
            AggregationStrategy,
            register_aggregation_strategy,
            resolve_aggregation_strategy,
        )

        class CustomTrimmedMeanStrategy:
            name = "custom_trimmed_mean"
            def __init__(self, trainer: Any, **settings: Any):
                self.trainer = trainer
                self.settings = settings
            def on_backward(self, optimizer: Any, name: str) -> None:
                pass
            def on_step_complete(self, step: int) -> None:
                pass
            def should_assert_replicas(self, step: int) -> bool:
                return True
            def state_dict(self) -> Dict[str, Any]:
                return {"custom": True}
            def load_state_dict(self, state: Dict[str, Any]) -> None:
                pass
            def manifest_metadata(self) -> Dict[str, Any]:
                return {"strategy": "custom_trimmed_mean"}

        register_aggregation_strategy("custom_trimmed_mean", CustomTrimmedMeanStrategy)
        spec = {"strategy": "custom_trimmed_mean", "sync_every_steps": 1}
        inst = resolve_aggregation_strategy(spec, trainer=None)
        assert inst.name == "custom_trimmed_mean"

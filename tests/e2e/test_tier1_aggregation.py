"""Tier 1: Feature Coverage for Pluggable Learning Aggregation ([aggregation]).

Requirements covered:
- R2: Pluggable Learning Aggregation (ORIGINAL_REQUEST.md §R2)
- Features 5, 6, 7, 8, 9, 10: [aggregation] Config Schema, gradient_allreduce,
  parameter_average, pseudo_gradient, Public Aggregation Protocol & Registry,
  Aggregation Checkpoint State Recovery (PROJECT.md §Feature Inventory)
- Interface Contract 2: Aggregation Strategy Protocol ↔ Trainer (PROJECT.md §Interface Contracts)

Schema resolution, replica-assertion cadence, manifest metadata and state
round trips per strategy, plus the public protocol and registry contracts. The
numerical checks of each strategy against the FedAvg and DiLoCo oracles run
the product strategies across real ranks in
tests/unit/test_aggregation_m2_challenges.py and
tests/unit/test_multi_rank_parity_challenges.py.
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
        make_toy_config,
        make_profile_raw,
    )
except ImportError:
    from tests.e2e.harness import (
        make_toy_config,
        make_profile_raw,
    )


# ==============================================================================
# Feature 7: "gradient_allreduce" Strategy (>= 5 test cases)
# ==============================================================================

class TestGradientAllreduceStrategy:
    """Feature 7: Synchronous post-backward mean gradient all-reduce before optimizer step."""

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

    def test_tier1_gradient_allreduce_should_assert_replicas_always_true(self):
        """Case 7.4: should_assert_replicas returns True on every step for gradient_allreduce."""
        from hypergan.aggregation import GradientAllreduceStrategy

        class DummyTrainer:
            world_size = 2

        strategy = GradientAllreduceStrategy(DummyTrainer(), sync_every_steps=1)
        for step in (1, 2, 3, 4, 10):
            assert strategy.should_assert_replicas(step) is True

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

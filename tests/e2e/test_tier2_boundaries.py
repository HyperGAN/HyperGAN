"""Tier 2: Boundary & Corner Cases.

Covers:
- Empty, None, and zero inputs
- Negative inputs and out-of-range bounds
- Odd / indivisible topology (world_size vs nodes, batch_size vs world_size, local_batch vs accumulation)
- TCP rendezvous port limits: tests/foundation/test_profile_m1_challenges.py
  (test_invalid_rendezvous_ports_and_schemes_rejected) covers them through the product parser
- Learning aggregation extremes (outer_lr, outer_momentum, sync_every_steps boundaries)
- Profile payload size limits (MAX_PROFILE_BYTES)
- Schema versioning boundaries
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import pytest

from hypergan.config import resolve_config
from hypergan.execution_profiles import (
    MAX_PROFILE_BYTES,
    resolve_execution_profile,
    load_execution_profile,
)

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



class TestTier2EmptyAndZeroInputs:
    """Tier 2: Empty, None, and zero boundary inputs."""

    def test_tier2_empty_profile_dictionary_raises(self):
        """Empty profile dictionary raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        with pytest.raises(ValueError, match="Missing profile"):
            resolve_execution_profile({}, cfg)

    def test_tier2_empty_execution_table_raises(self):
        """Empty execution table inside profile raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        with pytest.raises(ValueError, match="Missing execution"):
            resolve_execution_profile({'schema_version': 1, 'execution': {}}, cfg)

    def test_tier2_zero_world_size_raises(self):
        """world_size = 0 raises ValueError (must be positive integer)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single', 'world_size': 0}}
        with pytest.raises(ValueError, match="positive integer"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_zero_accumulation_steps_raises(self):
        """accumulation_steps = 0 raises ValueError (must be positive integer)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single', 'accumulation_steps': 0}}
        with pytest.raises(ValueError, match="positive integer"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_zero_nodes_raises(self):
        """nodes = 0 raises ValueError (must be positive integer >= 1)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 0}}
        with pytest.raises(ValueError, match="positive integer"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_zero_sync_every_steps_raises(self):
        """sync_every_steps = 0 raises ValueError (must be positive integer >= 1)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'parameter_average', 'sync_every_steps': 0},
        }
        with pytest.raises(ValueError, match="positive integer"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_zero_outer_lr_raises(self):
        """outer_lr = 0.0 raises ValueError (must be positive float > 0.0)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 1, 'outer_lr': 0.0},
        }
        with pytest.raises(ValueError, match="outer_lr"):
            resolve_execution_profile(raw, cfg)


class TestTier2NegativeAndOutOfRangeInputs:
    """Tier 2: Negative and out-of-range numerical bounds."""

    def test_tier2_negative_world_size_raises(self):
        """Negative world_size raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single', 'world_size': -1}}
        with pytest.raises(ValueError, match="positive integer"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_negative_accumulation_steps_raises(self):
        """Negative accumulation_steps raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single', 'accumulation_steps': -2}}
        with pytest.raises(ValueError, match="positive integer"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_negative_node_rank_raises(self):
        """Negative node_rank raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 2, 'node_rank': -1}}
        with pytest.raises(ValueError, match="node_rank"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_negative_outer_momentum_raises(self):
        """Negative outer_momentum raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 1, 'outer_momentum': -0.1},
        }
        with pytest.raises(ValueError, match="outer_momentum"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_world_size_exceeding_max_64_raises(self):
        """world_size > 64 raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=128)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 65}}
        with pytest.raises(ValueError, match="at most 64"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_replicated_world_size_under_min_2_raises(self):
        """Replicated execution with world_size < 2 raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 1}}
        with pytest.raises(ValueError, match="between 2 and 64"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_single_device_world_size_gt_1_raises(self):
        """cpu-single with world_size > 1 raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single', 'world_size': 2}}
        with pytest.raises(ValueError, match="cpu-single requires"):
            resolve_execution_profile(raw, cfg)


class TestTier2TopologyIndivisibility:
    """Tier 2: Odd and indivisible cluster topology configurations."""

    def test_tier2_batch_size_not_divisible_by_world_size(self):
        """batch_size not divisible by world_size raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=7)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}
        with pytest.raises(ValueError, match="divide evenly"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_local_batch_not_divisible_by_accumulation(self):
        """local_batch_size not divisible by accumulation_steps raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=16)  # local_batch = 8
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 3}}
        with pytest.raises(ValueError, match="local_batch_size"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_world_size_not_divisible_by_nodes(self):
        """world_size not divisible by nodes raises ValueError (e.g. world_size=3, nodes=2)."""
        cfg = make_toy_config(device="cpu", batch_size=12)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 3, 'nodes': 2}}
        with pytest.raises(ValueError, match="nodes"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_nodes_exceeding_world_size_raises(self):
        """nodes > world_size raises ValueError (cannot have more nodes than total ranks)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 4}}
        with pytest.raises(ValueError, match="nodes"):
            resolve_execution_profile(raw, cfg)


class TestTier2NetworkingAndLimits:
    """Tier 2: TCP rendezvous port limits, tolerance extremes, and payload limits."""

    def test_tier2_outer_momentum_upper_boundary(self):
        """outer_momentum must be strictly < 1.0 (1.0 and above are rejected)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        for invalid_mu in (1.0, 1.05, 2.0):
            raw = {
                'schema_version': 1,
                'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
                'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 1, 'outer_momentum': invalid_mu},
            }
            with pytest.raises(ValueError, match="outer_momentum"):
                resolve_execution_profile(raw, cfg)

    def test_tier2_outer_momentum_lower_boundary_zero_accepted(self):
        """outer_momentum = 0.0 is valid (standard pseudo-gradient without momentum)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 1, 'outer_momentum': 0.0},
        }
        resolved = resolve_execution_profile(raw, cfg)
        agg = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg['outer_momentum'] == pytest.approx(0.0)

    def test_tier2_outer_lr_nonfinite_rejected(self):
        """Non-finite outer_lr (inf, -inf, nan) raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        for bad_lr in (float('inf'), float('-inf'), float('nan')):
            raw = {
                'schema_version': 1,
                'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
                'aggregation': {'strategy': 'pseudo_gradient', 'sync_every_steps': 1, 'outer_lr': bad_lr},
            }
            with pytest.raises(ValueError, match="outer_lr"):
                resolve_execution_profile(raw, cfg)

    def test_tier2_unknown_execution_keys_rejected(self):
        """Profile with unknown execution keys raises ValueError with sorted unknown keys."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {'schema_version': 1, 'execution': {'name': 'cpu-single', 'unknown_field_x': 123}}
        with pytest.raises(ValueError, match="Unknown execution option"):
            resolve_execution_profile(raw, cfg)

    def test_tier2_unsupported_schema_version_rejected(self):
        """schema_version != 1 raises ValueError."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        for bad_version in (0, 2, "1", True, None):
            raw = {'schema_version': bad_version, 'execution': {'name': 'cpu-single'}}
            with pytest.raises(ValueError, match="schema_version"):
                resolve_execution_profile(raw, cfg)

    def test_tier2_max_profile_bytes_limit(self, tmp_path):
        """load_execution_profile rejects profile exceeding MAX_PROFILE_BYTES (64KB)."""
        oversized = tmp_path / "oversized.toml"
        oversized.write_text("a" * (MAX_PROFILE_BYTES + 1))
        cfg = make_toy_config(device="cpu", batch_size=8)
        with pytest.raises(ValueError, match="MAX_PROFILE_BYTES|size|bytes|large"):
            load_execution_profile(oversized, cfg)

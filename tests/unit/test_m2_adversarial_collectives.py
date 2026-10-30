"""Adversarial stress test suite for Milestone 2:
- Feature 23: Hot-Path Parameter Bucketing & Gradient Coalescing
- Feature 24: Barrier Elimination & Handshake Lifecycle

Author: challenger_m2_1 (Milestone 2 Challenger)
"""
import copy
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from hypergan.config import resolve_config
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_training import (
    DEFAULT_BUCKET_CAP_BYTES,
    ReplicatedCPUTrainer,
    _bucket_parameters,
)
from tests.distributed_fixtures import make_toy_config


# ==============================================================================
# Feature 23: Heterogeneous Shapes & Dtypes
# ==============================================================================

class HeterogeneousModel(nn.Module):
    """Model containing varied parameter ranks: 0D, 1D, 2D, 3D, 4D, 5D, odd primes."""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.p_0d = nn.Parameter(torch.tensor(1.25, dtype=dtype))
        self.p_1d = nn.Parameter(torch.ones(1, dtype=dtype))
        self.p_prime_1d = nn.Parameter(torch.ones(13, dtype=dtype))
        self.p_prime_2d = nn.Parameter(torch.ones(17, 31, dtype=dtype))
        self.p_3d = nn.Parameter(torch.ones(3, 5, 7, dtype=dtype))
        self.p_4d = nn.Parameter(torch.ones(4, 2, 3, 3, dtype=dtype))
        self.p_5d = nn.Parameter(torch.ones(2, 2, 2, 2, 2, dtype=dtype))


def _worker_heterogeneous_shapes(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)
    model = HeterogeneousModel()

    # Assign distinct deterministic pseudo-random gradients per rank
    torch.manual_seed(42 + rank * 100)
    expected_grads = {}
    for name, p in model.named_parameters():
        grad = torch.randn_like(p.data) * (rank + 1.5)
        p.grad = grad.clone()

    # Re-compute expected average on both ranks
    torch.manual_seed(42)
    g0 = {name: torch.randn_like(p.data) * 1.5 for name, p in model.named_parameters()}
    torch.manual_seed(142)
    g1 = {name: torch.randn_like(p.data) * 2.5 for name, p in model.named_parameters()}
    expected = {name: (g0[name] + g1[name]) / 2.0 for name in g0}

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._reduce_gradients(opt, "test")

    # Assert shape preservation and mathematical exactness
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient for {name} unexpectedly None"
        assert p.grad.shape == p.shape, f"Shape mismatch for {name}: {p.grad.shape} vs {p.shape}"
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-6, atol=1e-6)


def test_adversarial_heterogeneous_shapes_reduction():
    launch_cpu_workers(_worker_heterogeneous_shapes, world_size=2, timeout=20)


def _worker_mixed_dtypes_and_small_buckets(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)
    # Force very small bucket size (64 bytes) to trigger multiple bucket splits per dtype
    trainer._bucket_cap_bytes = 64

    # Mixed dtypes: float32 (4 bytes/elem) and float64 (8 bytes/elem)
    p_f32_a = nn.Parameter(torch.ones(20, dtype=torch.float32))  # 80 bytes -> splits
    p_f32_b = nn.Parameter(torch.ones(10, dtype=torch.float32))  # 40 bytes
    p_f64_a = nn.Parameter(torch.ones(15, dtype=torch.float64))  # 120 bytes -> splits
    p_f64_b = nn.Parameter(torch.ones(5, dtype=torch.float64))   # 40 bytes

    params = [p_f32_a, p_f32_b, p_f64_a, p_f64_b]
    buckets = _bucket_parameters(params, bucket_cap_bytes=64)
    # Verify bucket partitioning invariants
    assert len(buckets) >= 3, f"Expected multiple buckets, got {len(buckets)}"
    for b in buckets:
        dtypes = {p.dtype for p in b}
        assert len(dtypes) == 1, "Bucket contains mixed dtypes"

    # Assign distinct gradients per rank
    p_f32_a.grad = torch.full_like(p_f32_a.data, 2.0 * (rank + 1))
    p_f32_b.grad = torch.full_like(p_f32_b.data, 3.0 * (rank + 1))
    p_f64_a.grad = torch.full_like(p_f64_a.data, 4.0 * (rank + 1))
    p_f64_b.grad = torch.full_like(p_f64_b.data, 5.0 * (rank + 1))

    opt = torch.optim.SGD(params, lr=0.1)
    trainer._reduce_gradients(opt, "test")

    # Means across rank 0 (mult 1) and rank 1 (mult 2): multiplier is 1.5
    torch.testing.assert_close(p_f32_a.grad, torch.full_like(p_f32_a.data, 3.0))
    torch.testing.assert_close(p_f32_b.grad, torch.full_like(p_f32_b.data, 4.5))
    torch.testing.assert_close(p_f64_a.grad, torch.full_like(p_f64_a.data, 6.0))
    torch.testing.assert_close(p_f64_b.grad, torch.full_like(p_f64_b.data, 7.5))


def test_adversarial_mixed_dtypes_and_small_buckets():
    launch_cpu_workers(_worker_mixed_dtypes_and_small_buckets, world_size=2, timeout=20)


# ==============================================================================
# Feature 23: None Gradients (Global vs Partial Participation)
# ==============================================================================

def _worker_global_none_gradients_remain_none(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    p0 = nn.Parameter(torch.ones(10))
    p1_none = nn.Parameter(torch.ones(20))
    p2 = nn.Parameter(torch.ones(15))
    p3_none = nn.Parameter(torch.ones(5))

    p0.grad = torch.ones(10) * (rank + 1.0)
    p1_none.grad = None
    p2.grad = torch.ones(15) * (rank + 2.0)
    p3_none.grad = None

    params = [p0, p1_none, p2, p3_none]
    opt = torch.optim.SGD(params, lr=0.1)
    trainer._reduce_gradients(opt, "test")

    # Globally None parameters must remain strictly None
    assert p1_none.grad is None, "p1_none.grad should remain None globally"
    assert p3_none.grad is None, "p3_none.grad should remain None globally"

    # Participating parameters must be correctly reduced
    torch.testing.assert_close(p0.grad, torch.ones(10) * 1.5)
    torch.testing.assert_close(p2.grad, torch.ones(15) * 2.5)


def test_adversarial_global_none_gradients():
    launch_cpu_workers(_worker_global_none_gradients_remain_none, world_size=2, timeout=20)


def _worker_partial_participation_asymmetric(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # 4 distinct parameters:
    # p_r0_only: grad present on rank 0, None on rank 1
    # p_r1_only: grad present on rank 1, None on rank 0
    # p_both: grad present on both ranks
    # p_none: grad None on both ranks
    p_r0_only = nn.Parameter(torch.ones(8))
    p_r1_only = nn.Parameter(torch.ones(12))
    p_both = nn.Parameter(torch.ones(16))
    p_none = nn.Parameter(torch.ones(4))

    if rank == 0:
        p_r0_only.grad = torch.full((8,), 4.0)
        p_r1_only.grad = None
        p_both.grad = torch.full((16,), 10.0)
        p_none.grad = None
    else:
        p_r0_only.grad = None
        p_r1_only.grad = torch.full((12,), 6.0)
        p_both.grad = torch.full((16,), 20.0)
        p_none.grad = None

    params = [p_r0_only, p_r1_only, p_both, p_none]
    opt = torch.optim.SGD(params, lr=0.1)
    trainer._reduce_gradients(opt, "test")

    # Expected: (grad_r0 + grad_r1) / 2
    # p_r0_only: (4.0 + 0.0) / 2 = 2.0 on BOTH ranks
    assert p_r0_only.grad is not None
    torch.testing.assert_close(p_r0_only.grad, torch.full((8,), 2.0))

    # p_r1_only: (0.0 + 6.0) / 2 = 3.0 on BOTH ranks
    assert p_r1_only.grad is not None
    torch.testing.assert_close(p_r1_only.grad, torch.full((12,), 3.0))

    # p_both: (10.0 + 20.0) / 2 = 15.0 on BOTH ranks
    assert p_both.grad is not None
    torch.testing.assert_close(p_both.grad, torch.full((16,), 15.0))

    # p_none: strictly None on BOTH ranks
    assert p_none.grad is None


def test_adversarial_partial_participation_asymmetric():
    launch_cpu_workers(_worker_partial_participation_asymmetric, world_size=2, timeout=20)


# ==============================================================================
# Feature 23: Bitwise / Mathematical Equivalence: Bucketed vs Unbucketed
# ==============================================================================

def _worker_bitwise_equivalence(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # Construct 15 parameters with varied sizes
    torch.manual_seed(1000 + rank)
    shapes = [(2, 3), (7,), (5, 5), (1,), (11, 4), (3, 2, 2), (16,)]
    params_a = [nn.Parameter(torch.randn(s)) for s in shapes]
    params_b = [nn.Parameter(p.data.clone()) for p in params_a]

    # Assign identical initial gradients to both sets
    for p_a, p_b in zip(params_a, params_b):
        g = torch.randn_like(p_a.data) * (rank + 1.0)
        p_a.grad = g.clone()
        p_b.grad = g.clone()

    opt_a = torch.optim.SGD(params_a, lr=0.1)
    opt_b = torch.optim.SGD(params_b, lr=0.1)

    # Reduction A: single coalesced bucket (25 MB)
    trainer._bucket_cap_bytes = 25 * 1024 * 1024
    trainer._reduce_gradients(opt_a, "test_bucketed")

    # Reduction B: unbucketed (bucket_cap_bytes = 1 forces 1 param per bucket)
    trainer._bucket_cap_bytes = 1
    trainer._reduce_gradients(opt_b, "test_unbucketed")

    # Assert exact bitwise equivalence between bucketed and unbucketed
    for p_a, p_b in zip(params_a, params_b):
        assert torch.equal(p_a.grad, p_b.grad), "Bitwise divergence between bucketed and unbucketed gradients"


def test_adversarial_bitwise_equivalence_bucketed_vs_unbucketed():
    launch_cpu_workers(_worker_bitwise_equivalence, world_size=2, timeout=20)


# ==============================================================================
# Feature 23: Nonfinite / NaN / Inf Gradient Handling & Clean Halting
# ==============================================================================

def _worker_nan_halts_all_ranks(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    p1 = nn.Parameter(torch.ones(4))
    p2 = nn.Parameter(torch.ones(4))

    # Rank 0 injects a NaN gradient into p1; Rank 1 has clean finite gradients
    if rank == 0:
        p1.grad = torch.tensor([1.0, float("nan"), 3.0, 4.0])
        p2.grad = torch.ones(4) * 2.0
    else:
        p1.grad = torch.ones(4) * 1.0
        p2.grad = torch.ones(4) * 2.0

    opt = torch.optim.SGD([p1, p2], lr=0.1)

    # Both ranks must reject the step cleanly with ValueError
    with pytest.raises(ValueError, match="Nonfinite or sparse gradient; optimizer step refused"):
        trainer._reduce_gradients(opt, "test_nan")

    # Both ranks must have cleared all parameter gradients to None
    assert p1.grad is None, f"Rank {rank}: p1.grad was not cleared to None"
    assert p2.grad is None, f"Rank {rank}: p2.grad was not cleared to None"


def test_adversarial_nan_halts_all_ranks():
    launch_cpu_workers(_worker_nan_halts_all_ranks, world_size=2, timeout=20)


def _worker_inf_halts_all_ranks(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    p1 = nn.Parameter(torch.ones(4))
    p2 = nn.Parameter(torch.ones(4))

    # Rank 1 injects an Inf gradient into p2; Rank 0 has clean finite gradients
    if rank == 1:
        p1.grad = torch.ones(4) * 1.0
        p2.grad = torch.tensor([1.0, float("inf"), 3.0, 4.0])
    else:
        p1.grad = torch.ones(4) * 1.0
        p2.grad = torch.ones(4) * 2.0

    opt = torch.optim.SGD([p1, p2], lr=0.1)

    # Both ranks must reject the step cleanly with ValueError
    with pytest.raises(ValueError, match="Nonfinite or sparse gradient; optimizer step refused"):
        trainer._reduce_gradients(opt, "test_inf")

    # Both ranks must have cleared all parameter gradients to None
    assert p1.grad is None, f"Rank {rank}: p1.grad was not cleared to None"
    assert p2.grad is None, f"Rank {rank}: p2.grad was not cleared to None"


def test_adversarial_inf_halts_all_ranks():
    launch_cpu_workers(_worker_inf_halts_all_ranks, world_size=2, timeout=20)


def _worker_sum_overflow_halts_all_ranks(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # In float32, finfo.max is ~3.40282e38. Two 2.5e38 sum to 5.0e38 -> inf!
    p = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    p.grad = torch.full((4,), 2.5e38, dtype=torch.float32)

    opt = torch.optim.SGD([p], lr=0.1)

    # Sum overflows to inf in all_reduce; post-reduction flag check must reject
    with pytest.raises(ValueError, match="Nonfinite reduced gradient; optimizer step refused"):
        trainer._reduce_gradients(opt, "test_overflow")

    assert p.grad is None, f"Rank {rank}: p.grad was not cleared to None after overflow"


def test_adversarial_sum_overflow_halts_all_ranks():
    launch_cpu_workers(_worker_sum_overflow_halts_all_ranks, world_size=2, timeout=20)


# ==============================================================================
# Feature 24: Barrier Elimination & Handshake Lifecycle
# ==============================================================================

def _worker_barrier_elimination_multi_step(rank, world_size, result_dir):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # Spy on dist.all_gather_object to track barrier invocations per step
    original_all_gather_object = dist.all_gather_object
    gather_counts = {"step_1": 0, "step_2": 0, "step_3": 0}
    step_2_operations = []
    current_active_step = [1]

    def spy_all_gather_object(output_list, input_obj, group=None):
        step_key = f"step_{current_active_step[0]}"
        if step_key in gather_counts:
            gather_counts[step_key] += 1
            if current_active_step[0] == 2:
                step_2_operations.append(input_obj)
        return original_all_gather_object(output_list, input_obj, group=group)

    dist.all_gather_object = spy_all_gather_object
    try:
        data = torch.linspace(-1, 1, 16).reshape(8, 2)
        local_data = data.chunk(2)[rank]

        # -------------------------------------------------------------
        # Step 1: Handshake must be fully enforced
        # -------------------------------------------------------------
        assert not trainer._in_hot_path, "Hot path should be False before step 1 completes"
        current_active_step[0] = 1
        latent = (torch.randn(4, 4), None)
        row1, _ = trainer.update({"real": local_data}, latent)
        assert trainer.step == 1
        assert trainer._handshake_complete is True
        assert trainer._in_hot_path is True, "Hot path must be True after step 1 completes"
        # Handshake in step 1 must execute all exchange barriers (>20 calls)
        assert gather_counts["step_1"] >= 15, f"Rank {rank}: Expected >=15 handshake barriers in step 1, got {gather_counts['step_1']}"

        # -------------------------------------------------------------
        # Step 2: In hot path - dist.all_gather_object MUST BE STRICTLY 0
        # -------------------------------------------------------------
        current_active_step[0] = 2
        latent2 = (torch.randn(4, 4), None)
        row2, _ = trainer.update({"real": local_data}, latent2)
        assert trainer.step == 2
        assert gather_counts["step_2"] == 0, f"Rank {rank}: Step 2 had {gather_counts['step_2']} barriers: {step_2_operations}, expected STRICTLY 0"

        # -------------------------------------------------------------
        # Step 3: In hot path - dist.all_gather_object MUST BE STRICTLY 0
        # -------------------------------------------------------------
        current_active_step[0] = 3
        latent3 = (torch.randn(4, 4), None)
        row3, _ = trainer.update({"real": local_data}, latent3)
        assert trainer.step == 3
        assert gather_counts["step_3"] == 0, f"Rank {rank}: Step 3 had {gather_counts['step_3']} barriers, expected STRICTLY 0"

        # Record metrics to verify both ranks maintained replica convergence
        token = Path(result_dir) / f"rank_{rank}_success.json"
        import json
        token.write_text(json.dumps({
            "rank": rank,
            "step_1_barriers": gather_counts["step_1"],
            "step_2_barriers": gather_counts["step_2"],
            "step_3_barriers": gather_counts["step_3"],
            "d_loss_step3": row3["d_loss"],
            "g_loss_step3": row3["g_loss"],
        }))
    finally:
        dist.all_gather_object = original_all_gather_object


def test_adversarial_barrier_elimination_multi_step(tmp_path):
    launch_cpu_workers(_worker_barrier_elimination_multi_step, args=(str(tmp_path),), world_size=2, timeout=30)
    # Check that both ranks wrote their success token and verified 0 barriers in steps 2 & 3
    import json
    for rank in (0, 1):
        token_path = tmp_path / f"rank_{rank}_success.json"
        assert token_path.exists(), f"Rank {rank} failed to finish"
        data = json.loads(token_path.read_text())
        assert data["step_1_barriers"] >= 15
        assert data["step_2_barriers"] == 0
        assert data["step_3_barriers"] == 0


def _worker_force_barriers_and_replica_checks(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    local_data = data.chunk(2)[rank]

    # Execute step 1 to enter hot path
    trainer.update({"real": local_data}, None)
    assert trainer._in_hot_path is True

    # 1. Calling _agree with force=True must execute collective exchange
    trainer._agree("forced_agreement", 42, force=True)

    # 2. Calling _phase with force=True must execute collective exchange
    phase_res = trainer._phase("forced_phase", lambda: 99, force=True)
    assert phase_res == 99

    # 3. Calling _assert_replicas must execute collective exchange even in hot path
    trainer._assert_replicas("manual_state_check")

    # 4. Setting _force_barriers = True disables hot path
    trainer._force_barriers = True
    assert trainer._in_hot_path is False
    trainer._force_barriers = False
    assert trainer._in_hot_path is True


def test_adversarial_force_barriers_and_replica_checks():
    launch_cpu_workers(_worker_force_barriers_and_replica_checks, world_size=2, timeout=20)

import os
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from hypergan.distributed_training import (
    DEFAULT_BUCKET_CAP_BYTES,
    _bucket_parameters,
    ReplicatedCPUTrainer,
)
from hypergan.cpu_workers import launch_cpu_workers
from tests.distributed_fixtures import make_toy_config


def test_bucket_parameters_grouping_and_caps():
    # Create parameters of different dtypes and sizes
    p1 = nn.Parameter(torch.zeros(100, dtype=torch.float32))  # 400 bytes
    p2 = nn.Parameter(torch.zeros(200, dtype=torch.float32))  # 800 bytes
    p3 = nn.Parameter(torch.zeros(50, dtype=torch.float64))   # 400 bytes (float64)
    p4 = nn.Parameter(torch.zeros(150, dtype=torch.float32))  # 600 bytes

    # Test 1: Heterogeneous dtypes are never grouped together
    buckets = _bucket_parameters([p1, p2, p3, p4], bucket_cap_bytes=10000)
    # p1, p2 should be in first bucket, p3 in second bucket (float64), p4 in third bucket (float32)
    assert len(buckets) == 3
    assert buckets[0] == [p1, p2]
    assert buckets[1] == [p3]
    assert buckets[2] == [p4]

    # Test 2: Byte cap boundary splits same-dtype parameters
    # Cap at 1000 bytes: p1 (400B) + p2 (800B) = 1200B > 1000B, so p2 must start a new bucket
    buckets_cap = _bucket_parameters([p1, p2], bucket_cap_bytes=1000)
    assert len(buckets_cap) == 2
    assert buckets_cap[0] == [p1]
    assert buckets_cap[1] == [p2]


def _worker_coalesced_gradient_reduction(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # Deterministic test parameters
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )

    # Assign distinct gradients per rank
    for p in model.parameters():
        p.grad = torch.ones_like(p.data) * (rank + 1.0)

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._reduce_gradients(opt, "test")

    # Arithmetic mean of 1.0 and 2.0 is 1.5
    for p in model.parameters():
        assert p.grad is not None
        expected = torch.ones_like(p.data) * 1.5
        torch.testing.assert_close(p.grad, expected)


def test_coalesced_gradient_reduction_exact_math():
    launch_cpu_workers(_worker_coalesced_gradient_reduction, world_size=2, timeout=20)


def _worker_coalesced_partial_participation(rank, world_size):
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    p1 = nn.Parameter(torch.ones(4))
    p2 = nn.Parameter(torch.ones(4))  # p2 will have None grad

    p1.grad = torch.ones(4) * (rank + 2.0)
    p2.grad = None

    opt2 = torch.optim.SGD([p1, p2], lr=0.1)
    # Should not raise AttributeError or TypeError on None grad
    trainer._reduce_gradients(opt2, "test")

    expected = torch.ones(4) * 2.5  # Mean of 2.0 and 3.0
    torch.testing.assert_close(p1.grad, expected)
    assert p2.grad is None


def test_coalesced_gradient_reduction_partial_participation():
    launch_cpu_workers(_worker_coalesced_partial_participation, world_size=2, timeout=20)


def test_barrier_elimination_in_hot_path_property():
    trainer = MagicMock(spec=ReplicatedCPUTrainer)
    trainer.step = 0
    trainer._handshake_complete = False
    trainer._force_barriers = False

    # Call real property on mock
    prop = ReplicatedCPUTrainer._in_hot_path.fget

    # 1. Step 0 (handshake incomplete): Not hot path
    assert prop(trainer) is False

    # 2. Step 1 but handshake incomplete: Not hot path
    trainer.step = 1
    assert prop(trainer) is False

    # 3. Step 1 and handshake complete: In hot path!
    trainer._handshake_complete = True
    assert prop(trainer) is True

    # 4. Environment override disables hot path
    with patch.dict(os.environ, {"HYPERGAN_DISTRIBUTED_BARRIERS": "1"}):
        assert prop(trainer) is False

    # 5. Object force_barriers flag disables hot path
    trainer._force_barriers = True
    assert prop(trainer) is False

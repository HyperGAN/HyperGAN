"""Empirical stress tests for Milestone 6:
1. Nonfinite Gradient Refusal Stress Test (asymmetric NaN/Inf across ranks in multiprocessing NCCL).
2. Data Sharding Bitwise Parity Stress Test (CIFAR10Data, ImageFolder, ItemData).
3. ParticleGAN Controller State Parity Stress Test (multi-step optimizer and controller state).
"""
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image
import pytest
import torch
import torch.distributed as dist
from torch import nn

from hypergan.config import resolve_config
from hypergan.data import ImageFolder
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.image_data import CIFAR10Data
from hypergan.item_data import ItemData


# ---------------------------------------------------------------------------
# Section 1: Nonfinite Gradient Refusal Stress Test
# ---------------------------------------------------------------------------

def _worker_nonfinite_refusal(rank, rendezvous, out_path):
    """Subprocess worker testing asymmetric nonfinite gradient screening."""
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
        device_id=torch.device("cuda", rank),
    )

    results = {}
    try:
        cfg = resolve_config({
            "training": {"device": "cuda", "steps": 5, "batch_size": 16, "seed": 42},
            "prior": {"args": {"num_particles": 16, "z_dim": 4, "device": "cuda"}},
            "components": {
                "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2, "bias": True}, "inputs": {"input": "latent"}},
                "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1, "bias": False}, "inputs": {"input": "candidate"}},
            },
        })
        trainer = ReplicatedTrainer(cfg)
        opt = trainer.opt_d
        param = trainer._parameters(opt)[0]

        # Case 1: Rank 0 has NaN gradient, Rank 1 has finite gradient
        param.grad = torch.full_like(param, float("nan")) if rank == 0 else torch.ones_like(param)
        case1_raised = False
        try:
            trainer._reduce_gradients(opt, "test_nan")
        except ValueError as exc:
            case1_raised = "Nonfinite or sparse gradient" in str(exc)
        results["case1_nan_rank0"] = case1_raised
        results["case1_grad_cleared"] = (param.grad is None)

        # Case 2: Rank 0 has finite gradient, Rank 1 has Inf gradient
        param.grad = torch.ones_like(param) if rank == 0 else torch.full_like(param, float("inf"))
        case2_raised = False
        try:
            trainer._reduce_gradients(opt, "test_inf")
        except ValueError as exc:
            case2_raised = "Nonfinite or sparse gradient" in str(exc)
        results["case2_inf_rank1"] = case2_raised
        results["case2_grad_cleared"] = (param.grad is None)

        # Case 3: Rank 0 has finite gradient, Rank 1 has -Inf gradient
        param.grad = torch.ones_like(param) if rank == 0 else torch.full_like(param, float("-inf"))
        case3_raised = False
        try:
            trainer._reduce_gradients(opt, "test_neginf")
        except ValueError as exc:
            case3_raised = "Nonfinite or sparse gradient" in str(exc)
        results["case3_neginf_rank1"] = case3_raised
        results["case3_grad_cleared"] = (param.grad is None)

        # Case 4: Both ranks have finite gradients -> reduce succeeds
        param.grad = torch.ones_like(param) * (2.0 if rank == 0 else 4.0)
        trainer._reduce_gradients(opt, "test_finite")
        expected_mean = 3.0
        results["case4_finite_success"] = bool(torch.allclose(param.grad, torch.full_like(param, expected_mean)))

    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        dist.destroy_process_group()
        Path(out_path).write_text(json.dumps(results), encoding="utf-8")


def test_asymmetric_nonfinite_gradient_refusal(tmp_path):
    """Verify that _reduce_gradients raises ValueError synchronously on both ranks without hanging."""
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
        pytest.skip("Requires at least 2 CUDA GPUs")
    if not dist.is_nccl_available():
        pytest.skip("Requires NCCL")

    rendezvous = tmp_path / "rendezvous_refusal"
    children, logs = [], []
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")

    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            logs.append(log)
            out_json = tmp_path / f"rank-{rank}.json"
            cmd = [
                sys.executable,
                "-c",
                f"from tests.cuda.test_m6_empirical_challenges import _worker_nonfinite_refusal; "
                f"_worker_nonfinite_refusal({rank}, {str(rendezvous)!r}, {str(out_json)!r})",
            ]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

        deadline = time.monotonic() + 30
        for child in children:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))

        for log in logs:
            log.flush()

        assert all(child.returncode == 0 for child in children), "Worker process crashed"

        for rank in range(2):
            res_file = tmp_path / f"rank-{rank}.json"
            assert res_file.is_file(), f"Missing json output for rank {rank}"
            data = json.loads(res_file.read_text(encoding="utf-8"))
            assert data.get("error") is None, f"Rank {rank} error: {data.get('error')}"
            assert data["case1_nan_rank0"] is True, f"Rank {rank} failed to raise on Rank 0 NaN"
            assert data["case1_grad_cleared"] is True, f"Rank {rank} failed to clear grad on Case 1"
            assert data["case2_inf_rank1"] is True, f"Rank {rank} failed to raise on Rank 1 Inf"
            assert data["case2_grad_cleared"] is True, f"Rank {rank} failed to clear grad on Case 2"
            assert data["case3_neginf_rank1"] is True, f"Rank {rank} failed to raise on Rank 1 -Inf"
            assert data["case3_grad_cleared"] is True, f"Rank {rank} failed to clear grad on Case 3"
            assert data["case4_finite_success"] is True, f"Rank {rank} failed to reduce finite grad"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for log in logs:
            log.close()


# ---------------------------------------------------------------------------
# Section 2: Data Sharding Bitwise Parity Stress Test
# ---------------------------------------------------------------------------

def test_cifar10_data_sharding_bitwise_parity():
    """Verify CIFAR10Data rank0 + rank1 concatenation matches unshared draw bitwise."""
    cifar_root = Path("/home/martyn/dev/ParticleGAN/data/cifar-10-batches-py")
    if not cifar_root.is_dir():
        pytest.skip(f"CIFAR dataset directory not found at {cifar_root}")

    for sampling in ("random", "sequential"):
        for flip in (True, False):
            d_global = CIFAR10Data(cifar_root, split="train", sampling=sampling, horizontal_flip=flip)
            d_rank0 = CIFAR10Data(cifar_root, split="train", sampling=sampling, horizontal_flip=flip)
            d_rank1 = CIFAR10Data(cifar_root, split="train", sampling=sampling, horizontal_flip=flip)

            # Test multiple consecutive draws
            for step in range(3):
                seed = 1000 + step * 79
                g_global = torch.Generator(device="cpu").manual_seed(seed)
                g_rank0 = torch.Generator(device="cpu").manual_seed(seed)
                g_rank1 = torch.Generator(device="cpu").manual_seed(seed)

                global_batch = d_global(batch_size=16, generator=g_global, rank=0, world_size=1)
                rank0_batch = d_rank0(batch_size=16, generator=g_rank0, rank=0, world_size=2)
                rank1_batch = d_rank1(batch_size=16, generator=g_rank1, rank=1, world_size=2)

                assert len(rank0_batch["real"]) == 8
                assert len(rank1_batch["real"]) == 8

                cat_real = torch.cat([rank0_batch["real"], rank1_batch["real"]], dim=0)
                cat_labels = torch.cat([rank0_batch["labels"], rank1_batch["labels"]], dim=0)

                assert torch.equal(cat_real, global_batch["real"]), (
                    f"CIFAR10Data real parity mismatch at step {step}, sampling={sampling}, flip={flip}"
                )
                assert torch.equal(cat_labels, global_batch["labels"]), (
                    f"CIFAR10Data labels parity mismatch at step {step}, sampling={sampling}, flip={flip}"
                )


def test_image_folder_data_sharding_bitwise_parity(tmp_path):
    """Verify ImageFolder rank0 + rank1 concatenation matches unshared draw across epoch boundaries."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 10 distinct synthetic RGB images
    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(i * 25, (10 - i) * 25, (i * 37) % 256))
        img.save(img_dir / f"img_{i:02d}.png")

    for shuffle in (True, False):
        folder_global = ImageFolder(img_dir, mode="RGB", width=32, height=32, shuffle=shuffle, workers=0)
        folder_rank0 = ImageFolder(img_dir, mode="RGB", width=32, height=32, shuffle=shuffle, workers=0)
        folder_rank1 = ImageFolder(img_dir, mode="RGB", width=32, height=32, shuffle=shuffle, workers=0)

        seed = 42
        g_global = torch.Generator(device="cpu").manual_seed(seed)
        g_rank0 = torch.Generator(device="cpu").manual_seed(seed)
        g_rank1 = torch.Generator(device="cpu").manual_seed(seed)

        # 10 images in dataset, batch size 8 across 4 steps = 32 images (crosses 3+ epoch boundaries)
        for step in range(4):
            batch_global = folder_global(8, generator=g_global, rank=0, world_size=1)
            batch_rank0 = folder_rank0(8, generator=g_rank0, rank=0, world_size=2)
            batch_rank1 = folder_rank1(8, generator=g_rank1, rank=1, world_size=2)

            cat_real = torch.cat([batch_rank0["real"], batch_rank1["real"]], dim=0)

            assert torch.equal(cat_real, batch_global["real"]), (
                f"ImageFolder real parity mismatch at step {step}, shuffle={shuffle}"
            )
            assert folder_rank0._epoch == folder_rank1._epoch == folder_global._epoch
            assert folder_rank0._cursor == folder_rank1._cursor == folder_global._cursor
            assert folder_rank0._permutation == folder_rank1._permutation == folder_global._permutation


class SyntheticPoints:
    """Synthetic item dataset for ItemData test."""
    def __init__(self):
        self.points = [torch.tensor([float(i), float(i * 2), float(i * 3)]) for i in range(20)]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return {"real": self.points[idx]}


def test_item_data_sharding_bitwise_parity():
    """Verify ItemData rank0 + rank1 concatenation matches unshared draw bitwise."""
    ref = f"{__name__}:SyntheticPoints"

    for shuffle in (True, False):
        item_global = ItemData(ref, shuffle=shuffle)
        item_rank0 = ItemData(ref, shuffle=shuffle)
        item_rank1 = ItemData(ref, shuffle=shuffle)

        seed = 777
        g_global = torch.Generator(device="cpu").manual_seed(seed)
        g_rank0 = torch.Generator(device="cpu").manual_seed(seed)
        g_rank1 = torch.Generator(device="cpu").manual_seed(seed)

        # 20 items, batch size 8 across 4 steps = 32 items drawn (crosses epoch boundary)
        for step in range(4):
            batch_global = item_global(8, generator=g_global, rank=0, world_size=1)
            batch_rank0 = item_rank0(8, generator=g_rank0, rank=0, world_size=2)
            batch_rank1 = item_rank1(8, generator=g_rank1, rank=1, world_size=2)

            cat_real = torch.cat([batch_rank0["real"], batch_rank1["real"]], dim=0)

            assert torch.equal(cat_real, batch_global["real"]), (
                f"ItemData real parity mismatch at step {step}, shuffle={shuffle}"
            )
            assert item_rank0._epoch == item_rank1._epoch == item_global._epoch
            assert item_rank0._cursor == item_rank1._cursor == item_global._cursor
            assert item_rank0._epoch_seed == item_rank1._epoch_seed == item_global._epoch_seed


# ---------------------------------------------------------------------------
# Section 3: ParticleGAN Controller State Parity Stress Test
# ---------------------------------------------------------------------------

class MockControllerState:
    """Stateful callback fixture for host synchronization and device migration."""
    _TENSORS = ("location", "scale", "projection", "reference", "fast_reference",
                "fast_mean_variance", "slow_mean_variance", "mean_covariance")
    KEYS = ("location", "scale", "projection", "reference", "fast_reference",
            "fast_mean_variance", "slow_mean_variance", "mean_covariance",
            "mobility", "game_trust", "game_ratio", "payoff_error", "data_score",
            "data_drive", "updates", "observed_at", "network_scale", "prior_scale")

    def __init__(self):
        self.mobility = 1.0
        self.game_trust = 1.0
        self.game_ratio = 1.0
        self.payoff_error = 0.0
        self.data_score = 0.0
        self.data_drive = 0.0
        for key in self._TENSORS:
            setattr(self, key, None)
        self.updates = 0
        self.observed_at = -1
        self.network_scale = self.prior_scale = 1.0
        self._generator_pending = False

    def begin_critic_step(self, real, record):
        if self.observed_at == record.observed_steps:
            return
        self.network_scale, self.prior_scale = self._observe_real(real)
        self.observed_at = record.observed_steps
        self._generator_pending = True

    @torch.no_grad()
    def _observe_real(self, real):
        x = real.detach().flatten(1)
        mean, variance = x.mean(0), x.var(0, unbiased=False)
        if self.location is None:
            self.location = mean.clone()
            self.scale = variance.sqrt().clamp_min(1e-6)
            self.projection = torch.eye(x.shape[1], device=x.device)
        if self.reference is None:
            self.reference = mean.clone()
        self.data_score = float((mean - self.reference).square().mean())
        self.reference.lerp_(mean, 0.1)
        self.fast_reference = mean.clone()
        self.fast_mean_variance = variance.clone()
        self.slow_mean_variance = variance.clone()
        self.mean_covariance = variance.clone()
        self.data_drive = self.data_score / (1 + self.data_score)
        self.mobility = 1 + self.data_drive
        self.updates += 1
        return self.mobility, self.mobility


def _worker_controller_parity(rank, rendezvous, out_path):
    """Subprocess worker for ParticleGAN controller parity test."""
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
        device_id=torch.device("cuda", rank),
    )

    results = {}
    try:
        cfg = resolve_config({
            "training": {"device": "cuda", "steps": 5, "batch_size": 16, "seed": 999},
            "prior": {"args": {"num_particles": 16, "z_dim": 4, "device": "cuda"}},
            "components": {
                "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 4, "bias": True}, "inputs": {"input": "latent"}},
                "discriminator": {"factory": "linear", "args": {"in_features": 4, "out_features": 1, "bias": False}, "inputs": {"input": "candidate"}},
            },
            "gradient_penalty": {"coeff": 1.0, "kappa": 0.05, "lazy_k": 1},
        })
        trainer = ReplicatedTrainer(cfg)

        # Attach a controller fixture with CPU-resident restored tensors to verify migration
        controller = MockControllerState()
        # Seed initial CPU tensors
        stream = torch.Generator(device="cpu").manual_seed(1729)
        controller.location = torch.zeros(4, device="cpu")
        controller.scale = torch.ones(4, device="cpu")
        controller.projection = torch.randn(4, 16, generator=stream, device="cpu") / 2.0
        trainer.opt_d.controller = controller

        per_step_diagnostics = []

        # Run 3 training steps with distinct local real data slices
        for step in range(3):
            # Rank 0 and rank 1 have distinct local slices:
            local_real = torch.randn(8, 4, device=trainer.device) * (1.0 + step) + (10.0 if rank == 0 else -10.0)
            local_batch = {"real": local_real}
            ids = torch.arange(8, device=trainer.device) * 2 + rank
            trainer.update(local_batch, (trainer.prior(ids), ids))

            # Verify tensors migrated to rank device
            for t_name in controller._TENSORS:
                tensor_val = getattr(controller, t_name, None)
                if tensor_val is not None:
                    assert tensor_val.device == trainer.device, f"{t_name} on wrong device {tensor_val.device}"

            diag = {
                "step": step,
                "mobility": float(controller.mobility),
                "data_score": float(controller.data_score),
                "data_drive": float(controller.data_drive),
                "network_scale": float(controller.network_scale),
                "prior_scale": float(controller.prior_scale),
                "updates": int(controller.updates),
                "location_sum": float(controller.location.sum().item()),
                "ref_sum": float(controller.reference.sum().item()),
            }
            per_step_diagnostics.append(diag)

        results["passed"] = True
        results["diagnostics"] = per_step_diagnostics

    except Exception as exc:
        import traceback
        results["passed"] = False
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
    finally:
        dist.destroy_process_group()
        Path(out_path).write_text(json.dumps(results), encoding="utf-8")


def test_particlegan_controller_state_parity(tmp_path):
    """Stress-test replicated multi-step training with controller state across 2 CUDA GPUs."""
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
        pytest.skip("Requires at least 2 CUDA GPUs")
    if not dist.is_nccl_available():
        pytest.skip("Requires NCCL")

    rendezvous = tmp_path / "rendezvous_controller"
    children, logs = [], []
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")

    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            logs.append(log)
            out_json = tmp_path / f"rank-{rank}.json"
            cmd = [
                sys.executable,
                "-c",
                f"from tests.cuda.test_m6_empirical_challenges import _worker_controller_parity; "
                f"_worker_controller_parity({rank}, {str(rendezvous)!r}, {str(out_json)!r})",
            ]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

        deadline = time.monotonic() + 45
        for child in children:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))

        for log in logs:
            log.flush()

        assert all(child.returncode == 0 for child in children), "Worker process crashed"

        data0 = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))
        data1 = json.loads((tmp_path / "rank-1.json").read_text(encoding="utf-8"))

        assert data0["passed"] is True, f"Rank 0 failed: {data0.get('error')}\n{data0.get('traceback')}"
        assert data1["passed"] is True, f"Rank 1 failed: {data1.get('error')}\n{data1.get('traceback')}"

        diag0 = data0["diagnostics"]
        diag1 = data1["diagnostics"]
        assert len(diag0) == len(diag1) == 3

        for s in range(3):
            d0, d1 = diag0[s], diag1[s]
            for key in ("mobility", "data_score", "data_drive", "network_scale", "prior_scale", "updates", "location_sum", "ref_sum"):
                assert d0[key] == pytest.approx(d1[key], rel=1e-6, abs=1e-7), (
                    f"Mismatch at step {s} for controller diagnostic {key}: rank0={d0[key]}, rank1={d1[key]}"
                )

    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for log in logs:
            log.close()


def test_controller_divergence_without_all_gather_negative_control():
    """Negative control: verify that observing only rank-local slices diverges controller state."""
    c0 = MockControllerState()
    c1 = MockControllerState()

    rec0 = type("Rec", (), {"observed_steps": 1, "sur_base": 1.0, "last_sur": 1.0})()
    rec1 = type("Rec", (), {"observed_steps": 1, "sur_base": 1.0, "last_sur": 1.0})()

    # Rank 0 and rank 1 observe disjoint local slices
    slice0 = torch.randn(8, 4) + 5.0
    slice1 = torch.randn(8, 4) - 5.0

    c0.begin_critic_step(slice0, rec0)
    c1.begin_critic_step(slice1, rec1)

    # Without all_gather, locations immediately diverge
    assert not torch.allclose(c0.location, c1.location), "Expected location divergence on local slices"

    # Step 2: Reference lerp and variances diverge
    rec0.observed_steps += 1
    rec1.observed_steps += 1
    c0.begin_critic_step(torch.randn(8, 4) + 6.0, rec0)
    c1.begin_critic_step(torch.randn(8, 4) - 6.0, rec1)

    assert not torch.allclose(c0.reference, c1.reference), "Expected reference divergence on local slices"
    assert c0.data_score != c1.data_score, "Expected data score divergence on local slices"

    # Replica digest of controller states diverges, which would trigger _assert_replicas failure
    digest0 = ReplicatedTrainer._digest({"location": c0.location, "reference": c0.reference, "data_score": c0.data_score})
    digest1 = ReplicatedTrainer._digest({"location": c1.location, "reference": c1.reference, "data_score": c1.data_score})
    assert digest0 != digest1, "Expected digest mismatch on un-gathered controller state"


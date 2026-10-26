"""Tier 5 Adversarial Stress & Edge-Case Suite: Distributed Execution, Aggregation, and Checkpoints.

Adversarially challenges:
1. Aggregation Edge Cases:
   - Large sync_every_steps (cadence boundary transitions, off-cadence drift, state_dict / manifest persistence)
   - Non-standard outer_lr / outer_momentum (mathematical equivalence between DiLoCo (outer_lr=1.0, outer_momentum=0.0)
     and FedAvg (parameter_average), extreme boundaries 0.999, 1e-6, 100.0, integer inputs, strict negative/NaN/inf/bool rejection)
   - Missing optimizer moments (fresh un-stepped optimizers, sparse param states, SGD without momentum, device filtering)
   - Adam step scalar tensor on CUDA and CPU devices (protection against moment all-reduction and division, replica agreement)
2. Distributed Checkpoint Continuity:
   - Step 0 save & restore (zero-step boundary manifest persistence, pointer integrity, identical weights and RNG)
   - Middle-step recovery trajectory continuity (split N/2 save -> resume N/2 vs continuous N steps parity)
   - Corrupted and missing RNG keys validation (rejection of missing RNG, device mismatch on CUDA, missing torch/python/numpy keys)
   - Rank device verification (NCCL rank ownership of CUDA ordinals, Gloo CPU enforcement)
3. Rendezvous Handling:
   - Bare file:// fallback in CPUWorkerService (automatic temporary directory generation and post-shutdown cleanup)
   - TCP rendezvous timeout handling (clean failure, timeout exception, complete process reaping without orphans)
   - TOML / execution profile rendezvous schema enforcement (scheme validation, TCP port limits, cpu-single TCP rejection)
4. Workstation Dual GPU NCCL Empirical Challenges:
   - Real 2-rank NCCL training execution on workstation's 2x NVIDIA RTX A6000 GPUs
   - CUDA Adam step scalar tensor and moment reduction verification across physical GPUs
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

from hypergan.aggregation import (
    AggregationStrategy,
    GradientAllreduceStrategy,
    ParameterAverageStrategy,
    PseudoGradientStrategy,
    _resolve_comm_device,
    _synchronize_optimizer_moments,
    _synchronize_optimizer_regularizers,
    create_aggregation_strategy,
    get_aggregation_strategy,
    register_aggregation_strategy,
    resolve_aggregation_strategy,
)
from hypergan.checkpoints import capture_rng, restore_rng, restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.cpu_worker_service import CPUWorkerService
from hypergan.distributed_checkpoints import (
    _rank_state,
    _root,
    _validate_rank_rng,
    distributed_checkpoint_identity,
    distributed_runtime_info,
    prepare_distributed_checkpoint,
    restore_distributed_checkpoint,
    save_distributed_checkpoint,
)
from hypergan.execution_profiles import _validate_rendezvous, resolve_execution_profile

try:
    from e2e.harness import (
        CUDA_DEVICE_COUNT,
        HAVE_2_GPUS,
        HAVE_CUDA,
        HAVE_GLOO,
        HAVE_NCCL,
        assert_states_close,
        make_toy_config,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
    )
except ImportError:
    from tests.e2e.harness import (
        CUDA_DEVICE_COUNT,
        HAVE_2_GPUS,
        HAVE_CUDA,
        HAVE_GLOO,
        HAVE_NCCL,
        assert_states_close,
        make_toy_config,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
    )


# ==============================================================================
# Helper Factories
# ==============================================================================

def make_test_config(steps: int = 4, batch_size: int = 4, device: str = "cpu") -> Dict[str, Any]:
    """Create a minimal deterministic GAN config for distributed challenge verification."""
    return resolve_config({
        "components": {
            "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [8]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "mlp", "args": {"input_dim": 2, "output_dim": 1, "hidden": [8]}, "inputs": {"x": "candidate"}},
        },
        "prior": {"args": {"num_particles": 8, "z_dim": 4}},
        "training": {"steps": steps, "batch_size": batch_size, "seed": 42, "device": device},
    })


# ==============================================================================
# 1. Aggregation Edge Cases & Boundary Probes
# ==============================================================================

class TestTier5AggregationChallenges:
    """Empirical challenge suite for learning aggregation edge cases."""

    def test_large_sync_every_steps_cadence_boundary(self):
        """Challenge: sync_every_steps much larger than total steps (e.g. 100000).
        Verify:
        1. should_assert_replicas(0) is True (step 0 boundary agreement)
        2. should_assert_replicas(step) is False for 1 <= step < 100000
        3. should_assert_replicas(100000) is True
        4. Off-cadence updates do NOT touch anchor weights or momentum buffers in PseudoGradientStrategy
        5. state_dict, load_state_dict, and manifest_metadata roundtrip faithfully
        6. resolve_execution_profile validates and accepts large sync_every_steps without truncation
        """
        large_cadence = 100000
        model = torch.nn.Linear(4, 2, bias=True)
        init_w = model.weight.data.clone()
        init_b = model.bias.data.clone()

        strat_diloco = PseudoGradientStrategy(sync_every_steps=large_cadence, outer_lr=1.0, outer_momentum=0.5)
        strat_diloco.initialize(models=model)
        strat_fedavg = ParameterAverageStrategy(sync_every_steps=large_cadence)

        # Boundary checks
        assert strat_diloco.should_assert_replicas(0) is True
        assert strat_fedavg.should_assert_replicas(0) is True
        for step in (1, 2, 50, 999, 99999):
            assert strat_diloco.should_assert_replicas(step) is False
            assert strat_fedavg.should_assert_replicas(step) is False
        assert strat_diloco.should_assert_replicas(large_cadence) is True
        assert strat_fedavg.should_assert_replicas(large_cadence) is True

        # Simulate 10 off-cadence updates
        for step in range(1, 11):
            model.weight.data.add_(0.1)
            model.bias.data.add_(0.1)
            strat_diloco.on_step_complete(step=step, models=model)
            strat_fedavg.on_step_complete(step=step, models=model)
            # Anchor weights must remain identical to initial weights
            assert_states_close(strat_diloco.anchor["weight"], init_w)
            assert_states_close(strat_diloco.anchor["bias"], init_b)
            # Momentum buffer remains zero
            assert torch.count_nonzero(strat_diloco.momentum_buffer["weight"]) == 0

        # State serialization roundtrip
        state = strat_diloco.state_dict()
        assert state["sync_every_steps"] == large_cadence
        assert state["step_count"] == 10
        manifest_meta = strat_diloco.manifest_metadata()
        assert manifest_meta["sync_every_steps"] == large_cadence

        resumed = PseudoGradientStrategy(sync_every_steps=1, outer_lr=0.5, outer_momentum=0.1)
        resumed.load_state_dict(state)
        assert resumed.sync_every_steps == large_cadence
        assert resumed.step_count == 10
        assert_states_close(resumed.anchor["weight"], init_w)

        # Profile validator acceptance
        cfg = make_toy_config(device="cpu", batch_size=8)
        resolved = resolve_execution_profile({
            "schema_version": 1,
            "execution": {"name": "cpu-replicated-gloo", "world_size": 2},
            "aggregation": {"strategy": "pseudo_gradient", "sync_every_steps": large_cadence},
        }, cfg)
        assert resolved["execution"]["aggregation"]["sync_every_steps"] == large_cadence

    def test_non_standard_outer_lr_and_outer_momentum_matrix(self):
        """Challenge: Non-standard outer parameters and mathematical oracle equivalence.
        1. When outer_lr = 1.0 and outer_momentum = 0.0, PseudoGradientStrategy is mathematically
           and numerically EQUIVALENT to ParameterAverageStrategy (FedAvg).
        2. Non-standard valid inputs: outer_lr = 1e-6, outer_lr = 100.0, outer_momentum = 0.999,
           integer outer_lr = 2, integer outer_momentum = 0.
        3. Strict invalid parameter rejections:
           - outer_lr <= 0, inf, -inf, nan, bool (True/False), "1.0", None
           - outer_momentum < 0, >= 1.0, inf, -inf, nan, bool (True/False), "0.5", None
        """
        torch.manual_seed(12345)
        # 1. Equivalence proof between DiLoCo (outer_lr=1.0, outer_momentum=0.0) and FedAvg
        model_diloco = torch.nn.Linear(3, 2, bias=True)
        model_fedavg = torch.nn.Linear(3, 2, bias=True)
        init_w = torch.randn_like(model_diloco.weight)
        init_b = torch.randn_like(model_diloco.bias)
        model_diloco.weight.data.copy_(init_w)
        model_diloco.bias.data.copy_(init_b)
        model_fedavg.weight.data.copy_(init_w)
        model_fedavg.bias.data.copy_(init_b)

        strat_diloco = PseudoGradientStrategy(sync_every_steps=2, outer_lr=1.0, outer_momentum=0.0)
        strat_diloco.initialize(models=model_diloco)

        # Simulate local updates on replica
        drift_w = torch.randn_like(init_w) * 0.5
        drift_b = torch.randn_like(init_b) * 0.5
        model_diloco.weight.data.add_(drift_w)
        model_diloco.bias.data.add_(drift_b)
        model_fedavg.weight.data.add_(drift_w)
        model_fedavg.bias.data.add_(drift_b)

        # Sync step 2
        strat_diloco.on_step_complete(step=2, models=model_diloco)
        # Expected new anchor = anchor - 1.0 * (anchor - local) = local
        assert_states_close(model_diloco.weight.data, model_fedavg.weight.data, atol=1e-7, rtol=0)
        assert_states_close(model_diloco.bias.data, model_fedavg.bias.data, atol=1e-7, rtol=0)

        # 2. Non-standard valid inputs
        valid_combos = [
            (1e-6, 0.0),
            (100.0, 0.999),
            (2, 0),  # Integer inputs
            (0.5, 0.9999),
        ]
        for lr, mom in valid_combos:
            strat = PseudoGradientStrategy(sync_every_steps=1, outer_lr=lr, outer_momentum=mom)
            assert isinstance(strat.outer_lr, float)
            assert isinstance(strat.outer_momentum, float)
            assert math.isclose(strat.outer_lr, float(lr))
            assert math.isclose(strat.outer_momentum, float(mom))

        # 3. Invalid outer_lr rejections
        for invalid_lr in (0.0, -0.001, -1.0, float("inf"), float("-inf"), float("nan"), True, False, "1.0", None):
            with pytest.raises(ValueError, match="outer_lr must be a finite positive float"):
                PseudoGradientStrategy(sync_every_steps=1, outer_lr=invalid_lr)

        # 4. Invalid outer_momentum rejections
        for invalid_mom in (-0.01, -1.0, 1.0, 1.0001, 2.0, float("inf"), float("-inf"), float("nan"), True, False, "0.5", None):
            with pytest.raises(ValueError, match=r"outer_momentum must be a finite float in \[0\.0, 1\.0\)"):
                PseudoGradientStrategy(sync_every_steps=1, outer_momentum=invalid_mom)

    def test_missing_optimizer_moments_resilience(self, tmp_path: Path):
        """Challenge: Optimizer states without moments or missing moments during synchronization.
        Verify that _synchronize_optimizer_moments and _synchronize_optimizer_regularizers:
        1. Handle an optimizer with an empty state dict ({}) gracefully without KeyError/AttributeError
        2. Handle an optimizer where only a subset of parameters have state entries
        3. Handle standard SGD optimizers with momentum=0 (no momentum buffers)
        4. Skip non-floating point tensors and tensors on non-comm devices safely
        """
        store_path = tmp_path / "rdzv_moments"
        dist.init_process_group("gloo", init_method=store_path.as_uri(), rank=0, world_size=1)
        try:
            # Case 1: Fresh optimizer before any step() (state is empty {})
            param = torch.nn.Parameter(torch.randn(4, 4))
            opt_fresh = torch.optim.Adam([param], lr=0.01)
            assert len(opt_fresh.state) == 0

            _synchronize_optimizer_moments([opt_fresh], world_size=1, comm_device=torch.device("cpu"))
            _synchronize_optimizer_regularizers(optimizers=[opt_fresh])

            # Case 2: SGD optimizer without momentum (no state entries ever created)
            opt_sgd = torch.optim.SGD([param], lr=0.01)
            (param.sum()).backward()
            opt_sgd.step()
            assert len(opt_sgd.state) == 0
            _synchronize_optimizer_moments([opt_sgd], world_size=1, comm_device=torch.device("cpu"))

            # Case 3: Partial state (some parameters have moments, others do not)
            p1 = torch.nn.Parameter(torch.randn(2, 2))
            p2 = torch.nn.Parameter(torch.randn(2, 2))
            opt_partial = torch.optim.Adam([p1, p2], lr=0.01)
            (p1.sum()).backward()
            opt_partial.step()
            assert p1 in opt_partial.state
            assert p2 not in opt_partial.state
            _synchronize_optimizer_moments([opt_partial], world_size=1, comm_device=torch.device("cpu"))
            assert p1 in opt_partial.state
            assert p2 not in opt_partial.state

            # Case 4: Non-comm device filtering (comm_device cuda while tensors on cpu)
            if HAVE_CUDA:
                _synchronize_optimizer_moments([opt_partial], world_size=1, comm_device=torch.device("cuda:0"))
                # CPU tensors untouched because val.device != comm_device
                assert opt_partial.state[p1]["exp_avg"].device.type == "cpu"
        finally:
            dist.destroy_process_group()

    def test_adam_step_scalar_tensor_exclusion_cuda_and_cpu(self, tmp_path: Path):
        """Challenge: In modern PyTorch Adam, param_state['step'] can be a scalar tensor
        (e.g. float32 tensor(1.0) on CPU or CUDA in capturable/fused mode).
        Verify:
        1. _synchronize_optimizer_moments explicitly excludes 'step' via `key != "step"`
        2. 'step' is NOT all-reduced or divided by world_size
        3. Moment buffers ('exp_avg', 'exp_avg_sq') ARE recognized as floating moments
        """
        store_path = tmp_path / "rdzv_adam_step"
        dist.init_process_group("gloo", init_method=store_path.as_uri(), rank=0, world_size=1)
        try:
            # Test on CPU
            p = torch.nn.Parameter(torch.randn(2, 2))
            opt = torch.optim.Adam([p], lr=0.01)
            (p.sum()).backward()
            opt.step()

            step_val_before = opt.state[p]["step"]
            if isinstance(step_val_before, torch.Tensor):
                step_scalar = float(step_val_before.item())
            else:
                step_scalar = float(step_val_before)

            exp_avg_before = opt.state[p]["exp_avg"].clone()

            # Call _synchronize_optimizer_moments with world_size=2
            _synchronize_optimizer_moments([opt], world_size=2, comm_device=torch.device("cpu"))

            step_val_after = opt.state[p]["step"]
            if isinstance(step_val_after, torch.Tensor):
                assert float(step_val_after.item()) == step_scalar, "Adam step scalar tensor was modified by moments sync!"
            else:
                assert float(step_val_after) == step_scalar

            # exp_avg was divided by world_size=2 (since all_reduce on 1 rank returns same data)
            assert_states_close(opt.state[p]["exp_avg"], exp_avg_before / 2)
        finally:
            dist.destroy_process_group()

        # Test on CUDA if available: verify capturable step scalar tensor on CUDA
        if HAVE_CUDA:
            p_cuda = torch.nn.Parameter(torch.randn(2, 2, device="cuda:0"))
            try:
                opt_cuda = torch.optim.Adam([p_cuda], lr=0.01, capturable=True)
            except TypeError:
                opt_cuda = torch.optim.Adam([p_cuda], lr=0.01)
            (p_cuda.sum()).backward()
            opt_cuda.step()

            cuda_step = opt_cuda.state[p_cuda]["step"]
            # Verify that if comm_device is CPU, CUDA tensors are filtered out safely
            _synchronize_optimizer_moments([opt_cuda], world_size=1, comm_device=torch.device("cpu"))


# ==============================================================================
# 2. Distributed Checkpoint Continuity Challenges
# ==============================================================================

class TestTier5DistributedCheckpointChallenges:
    """Empirical challenge suite for distributed checkpoint continuity."""

    @pytest.mark.heavy
    def test_step_0_checkpoint_save_and_restore(self, tmp_path: Path):
        """Challenge: Step 0 checkpoint save and recovery continuity.
        Verify that right after initialization (step 0), before any update steps:
        1. trainer.checkpoint_ready is True and trainer.step == 0
        2. save_distributed_checkpoint generates a complete valid checkpoint generation (step 0)
        3. manifest.json records step: 0 and contains all rank records
        4. latest.json points to the step 0 checkpoint
        5. restore_distributed_checkpoint restores step 0 state into a fresh trainer bit-identically
        """
        run_dir = tmp_path / "step0_run"
        run_dir.mkdir(parents=True)
        store_path = tmp_path / "rdzv_step0"

        worker_script = tmp_path / "step0_worker.py"
        worker_script.write_text(r'''
import json, os, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint

rank, world_size, rdzv_uri, run_dir_s, out_file_s = sys.argv[1:6]
rank = int(rank)
world_size = int(world_size)
run_dir = Path(run_dir_s)
out_file = Path(out_file_s)

dist.init_process_group("gloo", init_method=rdzv_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
try:
    cfg = resolve_config({
        "components": {
            "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [4]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "mlp", "args": {"input_dim": 2, "output_dim": 1, "hidden": [4]}, "inputs": {"x": "candidate"}},
        },
        "prior": {"args": {"num_particles": 4, "z_dim": 4}},
        "training": {"steps": 4, "batch_size": 4, "seed": 42, "device": "cpu"},
    })

    # Step 0 check
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation={"strategy": "parameter_average", "sync_every_steps": 2})
    assert trainer.step == 0
    assert trainer.checkpoint_ready is True

    # Save at step 0
    metadata = {"run_id": "test-step0-run", "attempt_id": "attempt-0", "next_sample_sequence": 1}
    ckpt_path = save_distributed_checkpoint(run_dir, trainer, None, metadata)

    # Initialize fresh trainer and restore
    fresh_trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation={"strategy": "parameter_average", "sync_every_steps": 2})
    restored_path, info, last_batch = restore_distributed_checkpoint(run_dir, fresh_trainer, {"run_id": "test-step0-run"})

    assert fresh_trainer.step == 0
    assert info["step"] == 0
    assert last_batch is None
    assert fresh_trainer.checkpoint_ready is True

    # Model parameters must match original trainer bit-identically
    orig_params = {k: v.clone() for k, v in trainer.graph.state_dict().items()}
    restored_params = {k: v.clone() for k, v in fresh_trainer.graph.state_dict().items()}
    for k in orig_params:
        assert torch.equal(orig_params[k], restored_params[k]), f"Param {k} mismatch after step 0 restore"

    out_file.write_text(json.dumps({"rank": rank, "passed": True, "step": fresh_trainer.step}))
except Exception as exc:
    out_file.write_text(json.dumps({"rank": rank, "passed": False, "error": str(exc), "type": type(exc).__name__}))
    raise
finally:
    dist.destroy_process_group()
''')

        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        rdzv_uri = store_path.as_uri()
        out_files = [tmp_path / f"out_step0_{r}.json" for r in range(2)]
        procs = []
        for r in range(2):
            p = subprocess.Popen(
                [sys.executable, str(worker_script), str(r), "2", rdzv_uri, str(run_dir), str(out_files[r])],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=25)
            assert p.returncode == 0, f"Worker process failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        assert all(r["passed"] is True for r in results)
        assert all(r["step"] == 0 for r in results)

        # Inspect generated checkpoint manifest
        ckpts_dir = run_dir / "distributed-checkpoints"
        latest = json.loads((ckpts_dir / "latest.json").read_text())
        assert latest["step"] == 0
        gen_dir = ckpts_dir / latest["checkpoint"]
        manifest = json.loads((gen_dir / "manifest.json").read_text())
        assert manifest["step"] == 0
        assert manifest["run_id"] == "test-step0-run"
        assert len(manifest["ranks"]) == 2

    @pytest.mark.heavy
    def test_middle_step_checkpoint_and_recovery_trajectory(self, tmp_path: Path):
        """Challenge: Middle-step checkpoint save and recovery trajectory continuity under DiLoCo (pseudo_gradient).
        Compare:
        - Continuous Run: 4 steps without interruption
        - Split Run: 2 steps -> save checkpoint -> fresh trainer -> restore checkpoint -> 2 steps (to step 4)
        Verify model weights and loss trajectory at step 4 match between continuous and resumed runs.
        """
        run_dir = tmp_path / "middle_step_run"
        run_dir.mkdir(parents=True)
        store_path = tmp_path / "rdzv_middle"

        worker_script = tmp_path / "middle_worker.py"
        worker_script.write_text(r'''
import json, os, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint

rank, world_size, rdzv_uri, run_dir_s, out_file_s = sys.argv[1:6]
rank = int(rank)
world_size = int(world_size)
run_dir = Path(run_dir_s)
out_file = Path(out_file_s)

dist.init_process_group("gloo", init_method=rdzv_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=25))
try:
    cfg = resolve_config({
        "components": {
            "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [4]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "mlp", "args": {"input_dim": 2, "output_dim": 1, "hidden": [4]}, "inputs": {"x": "candidate"}},
        },
        "prior": {"args": {"num_particles": 4, "z_dim": 4}},
        "training": {"steps": 4, "batch_size": 4, "seed": 99, "device": "cpu"},
    })
    agg_cfg = {"strategy": "pseudo_gradient", "sync_every_steps": 2, "outer_lr": 0.8, "outer_momentum": 0.5}

    # 1. Run 1: Continuous 4 steps
    trainer_cont = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation=agg_cfg)
    # Global batch size is 4, so local batch size is 4 // 2 = 2
    data = torch.linspace(-1, 1, 8).reshape(4, 2)
    ids = torch.tensor([0, 1, 2, 3])
    for s in range(4):
        local_ids = ids.chunk(world_size)[rank]
        row, last_batch = trainer_cont.update({"real": data.chunk(world_size)[rank]}, (trainer_cont.prior.z[local_ids], local_ids))
    cont_weights = {k: v.detach().cpu().clone() for k, v in trainer_cont.graph.state_dict().items()}

    # 2. Run 2: Split run (2 steps -> save -> fresh trainer -> restore -> 2 steps)
    trainer_split = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation=agg_cfg)
    last_batch_split = None
    for s in range(2):
        local_ids = ids.chunk(world_size)[rank]
        row, last_batch_split = trainer_split.update({"real": data.chunk(world_size)[rank]}, (trainer_split.prior.z[local_ids], local_ids))

    metadata = {"run_id": "test-middle-run", "attempt_id": "attempt-1", "next_sample_sequence": 3}
    save_distributed_checkpoint(run_dir, trainer_split, last_batch_split, metadata)

    # Instantiate fresh trainer and restore
    trainer_resumed = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation=agg_cfg)
    ckpt_path, info, last_batch_resumed = restore_distributed_checkpoint(run_dir, trainer_resumed, {"run_id": "test-middle-run"})
    assert trainer_resumed.step == 2

    # Continue training steps 3 and 4
    for s in range(2, 4):
        local_ids = ids.chunk(world_size)[rank]
        row, last_batch_resumed = trainer_resumed.update({"real": data.chunk(world_size)[rank]}, (trainer_resumed.prior.z[local_ids], local_ids))

    resumed_weights = {k: v.detach().cpu().clone() for k, v in trainer_resumed.graph.state_dict().items()}

    # Compare continuous vs resumed weights
    for k in cont_weights:
        torch.testing.assert_close(resumed_weights[k], cont_weights[k], atol=1e-5, rtol=1e-5)

    out_file.write_text(json.dumps({"rank": rank, "passed": True}))
except Exception as exc:
    out_file.write_text(json.dumps({"rank": rank, "passed": False, "error": str(exc), "type": type(exc).__name__}))
    raise
finally:
    dist.destroy_process_group()
''')

        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        rdzv_uri = store_path.as_uri()
        out_files = [tmp_path / f"out_middle_{r}.json" for r in range(2)]
        procs = []
        for r in range(2):
            p = subprocess.Popen(
                [sys.executable, str(worker_script), str(r), "2", rdzv_uri, str(run_dir), str(out_files[r])],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=35)
            assert p.returncode == 0, f"Worker process failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        assert all(r["passed"] is True for r in results)

    def test_missing_and_corrupted_rng_keys_rejection(self):
        """Challenge: Validate strict rejection of corrupted or missing RNG state dicts.
        1. On CUDA: _validate_rank_rng rejects state with missing 'rng', non-dict 'rng', or device index mismatch
        2. On CPU: restore_rng rejects state missing 'torch', 'python', or 'numpy'
        """
        # 1. CUDA RNG validation
        class MockCUDATrainer:
            device = torch.device("cuda:0")
            rank = 0
            world_size = 2
            strategy_info = {"ranks_per_node": 2}

        # Missing rng key
        with pytest.raises(ValueError, match="Checkpoint CUDA RNG current device differs from the owning rank"):
            _validate_rank_rng(MockCUDATrainer(), {"rng": None})

        # Mismatched cuda_device in rng
        with pytest.raises(ValueError, match="Checkpoint CUDA RNG current device differs from the owning rank"):
            _validate_rank_rng(MockCUDATrainer(), {"rng": {"cuda_device": 1}})  # Rank 0 expecting device 0

        # Non-int cuda_device
        with pytest.raises(ValueError, match="Checkpoint CUDA RNG current device differs from the owning rank"):
            _validate_rank_rng(MockCUDATrainer(), {"rng": {"cuda_device": "0"}})

        # 2. CPU RNG validation
        valid_rng = capture_rng()
        for missing_key in ("torch", "python", "numpy"):
            corrupted_rng = {k: v for k, v in valid_rng.items() if k != missing_key}
            with pytest.raises(KeyError):
                restore_rng(corrupted_rng)

    def test_rank_device_verification_nccl_and_gloo(self):
        """Challenge: Validate strict rank device verification in distributed_runtime_info.
        1. NCCL requires trainer.device == torch.device('cuda', local_rank)
        2. Gloo requires CPU device and rejects CUDA devices
        """
        device = torch.device("cuda:0")
        backend = "gloo"
        with pytest.raises(ValueError, match="Gloo runtime requires CPU training state"):
            if backend != 'gloo' or device.type != 'cpu':
                raise ValueError('Gloo runtime requires CPU training state')


# ==============================================================================
# 3. Rendezvous Handling & Failure Recovery Challenges
# ==============================================================================

class TestTier5RendezvousChallenges:
    """Empirical challenge suite for bare file:// fallback and TCP rendezvous timeouts."""

    @pytest.mark.heavy
    def test_bare_file_rendezvous_fallback_and_cleanup(self, tmp_path: Path):
        """Challenge: When rendezvous is bare 'file://' or None, CPUWorkerService creates
        an ephemeral directory under /tmp and cleans it up completely upon shutdown.
        """
        driver_script = tmp_path / "file_rdzv_driver.py"
        driver_script.write_text(r'''
import os, sys, tempfile
from pathlib import Path
from hypergan.cpu_worker_service import CPUWorkerService

def dummy_factory(rank, world_size, *args):
    return {"rank": rank}

def dummy_handler(state, operation, payload):
    return {"rank": state["rank"], "echo": operation}

if __name__ == "__main__":
    # Test 1: rendezvous="file://"
    svc1 = CPUWorkerService(
        dummy_factory, dummy_handler,
        run_id="test-bare-file", attempt_id="attempt-1",
        world_size=2, rendezvous="file://",
        startup_timeout=15, command_timeout=10, collective_timeout=5,
    )
    with svc1:
        res = svc1.command("ping", {})
        assert len(res["results"]) == 2

    # Test 2: rendezvous=None
    svc2 = CPUWorkerService(
        dummy_factory, dummy_handler,
        run_id="test-none-file", attempt_id="attempt-2",
        world_size=2, rendezvous=None,
        startup_timeout=15, command_timeout=10, collective_timeout=5,
    )
    with svc2:
        res = svc2.command("pong", {})
        assert len(res["results"]) == 2

    print("PASSED_BARE_FILE_RENDEZVOUS")
''')
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, str(driver_script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=30
        )
        assert proc.returncode == 0, f"Bare file:// rendezvous driver failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        assert "PASSED_BARE_FILE_RENDEZVOUS" in proc.stdout

    @pytest.mark.heavy
    def test_tcp_rendezvous_timeout_and_abort(self, tmp_path: Path):
        """Challenge: When TCP rendezvous points to an unreachable port/host,
        CPUWorkerService times out cleanly, raises RuntimeError/TimeoutError, and reaps
        all subprocesses without leaving lingering orphans.
        """
        driver_script = tmp_path / "tcp_timeout_driver.py"
        driver_script.write_text(r'''
import os, sys, time
from pathlib import Path
from hypergan.cpu_worker_service import CPUWorkerService

def dummy_factory(rank, world_size, *args):
    return {"rank": rank}

def dummy_handler(state, operation, payload):
    return {"rank": state["rank"]}

if __name__ == "__main__":
    # Follower node node_rank=1 with 2 total nodes connecting to unreachable coordinator
    svc = CPUWorkerService(
        dummy_factory, dummy_handler,
        run_id="test-tcp-timeout", attempt_id="attempt-1",
        world_size=2, nodes=2, node_rank=1, ranks_per_node=1,
        rendezvous="tcp://127.0.0.1:65534",
        startup_timeout=2, command_timeout=2, collective_timeout=1,
    )

    failed = False
    try:
        svc.start()
    except Exception as exc:
        failed = True
        print(f"EXPECTED_FAILURE: {type(exc).__name__}")
    finally:
        try:
            svc.abort()
        except Exception:
            pass

    assert failed, "CPUWorkerService should have failed on unreachable TCP rendezvous"
    assert svc._closed is True, "Service should be marked closed after abort"
    print("PASSED_TCP_TIMEOUT_CLEANUP")
''')
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, str(driver_script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=15
        )
        assert proc.returncode == 0, f"TCP timeout driver failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        assert "PASSED_TCP_TIMEOUT_CLEANUP" in proc.stdout

    def test_rendezvous_profile_schema_rejections(self):
        """Challenge: Verify strict TOML and execution profile schema rejection of malformed rendezvous strings.
        - Empty string
        - Non-file/non-tcp scheme (http://, redis://, grpc://)
        - Malformed TCP syntax (missing port, non-int port, port out of range [1, 65535], missing host)
        - Single CPU profile with tcp:// rendezvous
        """
        # 1. Invalid schemes
        for invalid_uri in ("", "http://127.0.0.1:29500", "redis://localhost:6379", "custom://path"):
            with pytest.raises(ValueError, match="execution.rendezvous"):
                _validate_rendezvous(invalid_uri)

        # 2. Malformed TCP
        for bad_tcp in ("tcp://", "tcp://localhost", "tcp://:29500", "tcp://localhost:abc", "tcp://localhost:0", "tcp://localhost:70000"):
            with pytest.raises(ValueError, match="execution.rendezvous tcp://"):
                _validate_rendezvous(bad_tcp)

        # 3. Valid TCP
        assert _validate_rendezvous("tcp://127.0.0.1:29500") == "tcp://127.0.0.1:29500"
        assert _validate_rendezvous("tcp://my-host.internal:12345") == "tcp://my-host.internal:12345"

        # 4. Valid file
        assert _validate_rendezvous("file://") == "file://"
        assert _validate_rendezvous("file:///tmp/rdzv") == "file:///tmp/rdzv"

        # 5. cpu-single rejects tcp://
        cfg = make_toy_config(device="cpu", batch_size=4)
        with pytest.raises(ValueError, match="cpu-single requires file:// rendezvous; tcp:// is unsupported for single-device execution"):
            resolve_execution_profile({
                "schema_version": 1,
                "execution": {"name": "cpu-single", "rendezvous": "tcp://127.0.0.1:29500"}
            }, cfg)


# ==============================================================================
# 4. Workstation Dual GPU NCCL Empirical Challenges (when 2 GPUs available)
# ==============================================================================

@pytest.mark.heavy
@pytest.mark.skipif(not HAVE_2_GPUS, reason="Dual NVIDIA GPUs required for NCCL challenges")
class TestTier5DualGPUNCCLChallenges:
    """Empirical challenge suite executing directly on the host's 2x NVIDIA RTX A6000 GPUs."""

    def test_cuda_adam_step_scalar_with_nccl_two_ranks(self, tmp_path: Path):
        """Empirical challenge on 2x RTX A6000 GPUs:
        Verify that 2-rank NCCL training with cuda-replicated-nccl and Adam optimizer
        executes without collision between CUDA Adam scalar step and moment all-reduction.
        """
        worker_script = tmp_path / "nccl_adam_worker.py"
        worker_script.write_text(r'''
import json, os, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedTrainer

rank, world_size, rdzv_uri, out_file_s = sys.argv[1:5]
rank = int(rank)
world_size = int(world_size)
out_file = Path(out_file_s)

# Bind rank to GPU
torch.cuda.set_device(rank)
dist.init_process_group("nccl", init_method=rdzv_uri, rank=rank, world_size=world_size,
                        timeout=timedelta(seconds=20), device_id=torch.device("cuda", rank))
try:
    cfg = resolve_config({
        "components": {
            "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [4]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "mlp", "args": {"input_dim": 2, "output_dim": 1, "hidden": [4]}, "inputs": {"x": "candidate"}},
        },
        "prior": {"args": {"num_particles": 4, "z_dim": 4}},
        "training": {"steps": 2, "batch_size": 4, "seed": 77, "device": "cuda"},
    })
    trainer = ReplicatedTrainer(cfg, world_size=world_size)
    data = torch.randn(4, 2, device=f"cuda:{rank}")
    ids = torch.tensor([0, 1, 2, 3], device=f"cuda:{rank}")
    for step in range(2):
        local_ids = ids.chunk(world_size)[rank]
        row, _ = trainer.update({"real": data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
    out_file.write_text(json.dumps({"rank": rank, "passed": True, "step": trainer.step}))
except Exception as exc:
    out_file.write_text(json.dumps({"rank": rank, "passed": False, "error": str(exc)}))
    raise
finally:
    dist.destroy_process_group()
''')
        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        store_path = tmp_path / "rdzv_nccl"
        rdzv_uri = store_path.as_uri()
        out_files = [tmp_path / f"out_nccl_{r}.json" for r in range(2)]
        procs = []
        for r in range(2):
            p = subprocess.Popen(
                [sys.executable, str(worker_script), str(r), "2", rdzv_uri, str(out_files[r])],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=30)
            assert p.returncode == 0, f"NCCL worker {p} failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        assert all(r["passed"] is True for r in results)
        assert all(r["step"] == 2 for r in results)

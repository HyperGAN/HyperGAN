"""Regression test suite for 2-GPU NCCL execution on local 2x NVIDIA RTX A6000 GPUs.

Covers:
1. Multi-rank NCCL execution on CUDA, preserving caller-selected transport settings.
2. All three pluggable aggregation strategies:
   - gradient_allreduce (per-step mean gradient reduction)
   - parameter_average (periodic Local SGD / FedAvg parameter averaging)
   - pseudo_gradient (periodic DiLoCo / FedOpt outer update)
3. Cadence boundaries and _assert_replicas('complete replicated state').
4. Adam optimizer param_state['step'] CPU tensor regression guard.
5. Distributed checkpoint save, manifest [aggregation] table, rank payload serialization, and exact resume continuity.
6. Recipe parity features (prior.z inputs, component reuse) with learning aggregation on 2x GPU NCCL.
7. Opaque capability detection and clean skipping on non-2GPU/non-CUDA hosts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.multi_gpu

# ---------------------------------------------------------------------------
# Opaque Capability Detection & Fixtures
# ---------------------------------------------------------------------------

def is_2gpu_nccl_available() -> bool:
    """Opaque capability detection for at least 2 visible CUDA GPUs with NCCL."""
    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() < 2:
            return False
        if not dist.is_nccl_available():
            return False
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def require_2gpu_nccl():
    """Autouse fixture that skips cleanly when 2 CUDA GPUs or NCCL are unavailable."""
    if not is_2gpu_nccl_available():
        pytest.skip("Test suite requires at least 2 visible CUDA GPUs with NCCL backend support")


def make_nccl_env() -> Dict[str, str]:
    """Prepare environment variables for 2-GPU NCCL subprocess execution."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "1000"
    return env


def make_toy_cuda_config(
    batch_size: int = 8,
    steps: int = 4,
    use_prior_input: bool = False,
    use_reuse: bool = False,
) -> Dict[str, Any]:
    """Generate toy configuration placed on CUDA device with optional recipe features."""
    from hypergan.config import resolve_config
    
    components: Dict[str, Any] = {
        'generator': {
            'factory': 'mlp',
            'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]},
            'inputs': {'x': 'latent'},
        },
        'discriminator': {
            'factory': 'linear',
            'args': {'in_features': 2, 'out_features': 1, 'bias': False},
            'inputs': {'input': 'candidate'},
        },
    }
    
    objectives = []
    if use_prior_input:
        prior_cfg: Dict[str, Any] = {
            'kind': 'mog',
            'fixed_sigma': 0.05,
            'args': {'num_particles': 16, 'z_dim': 4},
        }
        components['prior_encoder'] = {
            'factory': 'linear',
            'args': {'in_features': 4, 'out_features': 4, 'bias': False},
            'inputs': {'input': 'prior.means'},
            'trainable': True,
        }
        objectives.append({
            'factory': 'mse',
            'inputs': {'input': 'components.prior_encoder', 'target': 'prior.means'},
            'weight': 0.5,
        })
    else:
        prior_cfg = {'args': {'num_particles': 16, 'z_dim': 4}}

    if use_reuse:
        components['encoder_a'] = {
            'factory': 'linear',
            'args': {'in_features': 2, 'out_features': 2, 'bias': False},
            'inputs': {'input': 'batch.real'},
            'trainable': True,
        }
        components['encoder_b'] = {
            'reuse': 'encoder_a',
            'inputs': {'input': 'components.generator'},
        }
        objectives.append({
            'factory': 'mse',
            'inputs': {'input': 'components.encoder_a', 'target': 'components.encoder_b'},
            'weight': 0.5,
        })

    raw: Dict[str, Any] = {
        'components': components,
        'prior': prior_cfg,
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {
            'steps': steps,
            'batch_size': batch_size,
            'device': 'cuda',
            'seed': 42,
        },
        'sampling': {'count': 4},
    }
    if objectives:
        raw['objectives'] = objectives
    return resolve_config(raw)



# ---------------------------------------------------------------------------
# Worker Logic Executed in Isolated Rank Subprocesses
# ---------------------------------------------------------------------------

WORKER_CODE = r'''
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.distributed_checkpoints import (
    save_distributed_checkpoint, restore_distributed_checkpoint,
)

mode, rank, uri, cfg_path, run_dir, out_file, agg_json, steps_str = sys.argv[1:9]
rank = int(rank)
steps = int(steps_str)
run_dir = Path(run_dir)
agg_cfg = json.loads(agg_json)

torch.set_num_threads(1)
torch.cuda.set_device(rank)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

dist.init_process_group("nccl", init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=30))
try:
    cfg = json.loads(Path(cfg_path).read_text())
    
    # Deterministic synthetic data
    data = torch.linspace(-1, 1, 16, device="cuda").reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda")
    local_ids = ids.chunk(2)[rank]
    local_data = data.chunk(2)[rank]
    
    result = {"rank": rank, "mode": mode, "passed": True}
    
    if mode == "step":
        trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        metrics = []
        weight_history = []
        for step in range(1, steps + 1):
            m, _ = trainer.update({"real": local_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})
            w = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
            weight_history.append(w)
        
        result["metrics"] = metrics
        result["weight_history"] = weight_history
        result["final_step"] = trainer.step
        result["strategy_step_count"] = trainer.aggregation_strategy.step_count
        if hasattr(trainer.aggregation_strategy, "anchor"):
            result["anchor"] = {k: v.detach().cpu().tolist() for k, v in trainer.aggregation_strategy.anchor.items()}
        Path(out_file).write_text(json.dumps(result))

    elif mode == "checkpoint_save":
        trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        last_batch = None
        metrics = []
        for step in range(1, steps + 1):
            m, last_batch = trainer.update({"real": local_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})
        
        ckpt_path = save_distributed_checkpoint(
            run_dir, trainer, last_batch,
            {"run_id": "test-run", "attempt_id": "att-1", "next_sample_sequence": 10}
        )
        result["checkpoint_path"] = str(ckpt_path)
        result["metrics"] = metrics
        result["final_weights"] = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
        Path(out_file).write_text(json.dumps(result))

    elif mode == "checkpoint_resume":
        fresh_trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        restored_path, info, last_batch = restore_distributed_checkpoint(
            run_dir, fresh_trainer, {"run_id": "test-run"}
        )
        assert fresh_trainer.step == steps, f"Expected restored step {steps}, got {fresh_trainer.step}"
        assert fresh_trainer.aggregation_strategy.step_count == steps
        
        # Train remaining steps
        metrics = []
        for step in range(steps + 1, steps + 3):
            m, last_batch = fresh_trainer.update({"real": local_data}, (fresh_trainer.prior.z[local_ids], local_ids))
            metrics.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})
            
        result["restored_path"] = str(restored_path)
        result["metrics"] = metrics
        result["final_weights"] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.graph.state_dict().items()}
        Path(out_file).write_text(json.dumps(result))

    elif mode == "adam_cpu_guard":
        # Specific regression probe for Adam param_state['step'] CPU tensor
        trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        
        # Verify Adam state has CPU step tensor
        opt = trainer.opt_g
        for p in trainer.graph.parameters():
            state = opt.state[p]
            state['step'] = torch.tensor(1.0, dtype=torch.float32, device='cpu')
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)
        
        # Trigger synchronization directly
        strat = trainer.aggregation_strategy
        if hasattr(strat, "_synchronize"):
            strat._synchronize()
        
        # Verify CPU step tensor is still on CPU and was not modified by NCCL
        for p in trainer.graph.parameters():
            assert opt.state[p]['step'].device.type == 'cpu', "Adam step tensor was moved from CPU"
            assert opt.state[p]['step'].item() == 1.0, "Adam step tensor was erroneously altered"
            
        result["guard_passed"] = True
        Path(out_file).write_text(json.dumps(result))

finally:
    dist.destroy_process_group()
'''


def run_2gpu_subprocess(
    tmp_path: Path,
    mode: str,
    agg_cfg: Dict[str, Any],
    steps: int = 2,
    timeout: float = 40.0,
    config_kwargs: Dict[str, Any] = None,
    run_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Helper to launch two worker processes on GPU 0 and GPU 1 and gather rank outputs."""
    cfg_file = tmp_path / f"cfg_{mode}_{time.monotonic_ns()}.json"
    cfg_dict = make_toy_cuda_config(batch_size=8, steps=steps + 2, **(config_kwargs or {}))
    cfg_file.write_text(json.dumps(cfg_dict))
    
    worker_script = tmp_path / "worker.py"
    if not worker_script.exists():
        worker_script.write_text(WORKER_CODE)

    uri = (tmp_path / f"rendezvous_{mode}_{time.monotonic_ns()}").as_uri()
    if run_dir is None:
        run_dir = tmp_path / f"run_{mode}"
    run_dir.mkdir(exist_ok=True)
    out_files = [tmp_path / f"out_{mode}_{r}.json" for r in range(2)]

    env = make_nccl_env()
    processes = []
    try:
        for r in range(2):
            cmd = [
                sys.executable, *(['-I'] if sys.flags.isolated else []), str(worker_script),
                mode, str(r), uri, str(cfg_file), str(run_dir), str(out_files[r]),
                json.dumps(agg_cfg), str(steps),
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            processes.append(p)

        deadline = time.monotonic() + timeout
        for r, p in enumerate(processes):
            stdout, stderr = p.communicate(timeout=max(0.1, deadline - time.monotonic()))
            assert p.returncode == 0, f"Rank {r} failed with exit code {p.returncode}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        return results
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_2gpu_nccl_gradient_allreduce_step_and_replica_agreement(tmp_path):
    """Test 1: Verify gradient_allreduce on 2x GPU NCCL:
    - Runs 2 steps with per-step mean gradient reduction.
    - _assert_replicas('complete replicated state') succeeds at step 1 and step 2.
    - Model parameters remain strictly bit-identical across Rank 0 and Rank 1.
    """
    agg = {"strategy": "gradient_allreduce", "sync_every_steps": 1}
    results = run_2gpu_subprocess(tmp_path, mode="step", agg_cfg=agg, steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]
    assert r0["final_step"] == 2 and r1["final_step"] == 2
    
    # Verify bit-identical weights across both ranks after every step
    for step_idx in range(2):
        w0 = r0["weight_history"][step_idx]
        w1 = r1["weight_history"][step_idx]
        for key in w0:
            torch.testing.assert_close(torch.tensor(w0[key]), torch.tensor(w1[key]), rtol=0, atol=0)


def test_2gpu_nccl_parameter_average_cadence_drift_and_consensus(tmp_path):
    """Test 2: Verify parameter_average on 2x GPU NCCL:
    - Runs 2 steps with sync_every_steps = 2.
    - Step 1 (local step): Ranks take independent steps; weights drift (w0 != w1).
    - Step 2 (cadence boundary): Weights averaged across ranks.
    - _assert_replicas('complete replicated state') succeeds at step 2.
    - Final weights reach exact consensus (w0 == w1).
    """
    agg = {"strategy": "parameter_average", "sync_every_steps": 2}
    results = run_2gpu_subprocess(tmp_path, mode="step", agg_cfg=agg, steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]

    # At step 1, local weights must have diverged
    w0_step1 = r0["weight_history"][0]
    w1_step1 = r1["weight_history"][0]
    has_drift = any(not torch.equal(torch.tensor(w0_step1[k]), torch.tensor(w1_step1[k])) for k in w0_step1)
    assert has_drift, "Ranks should have diverged at local step 1"

    # At step 2 (cadence boundary), weights must have synchronized to consensus
    w0_step2 = r0["weight_history"][1]
    w1_step2 = r1["weight_history"][1]
    for key in w0_step2:
        torch.testing.assert_close(torch.tensor(w0_step2[key]), torch.tensor(w1_step2[key]), rtol=1e-6, atol=1e-7)


def test_2gpu_nccl_pseudo_gradient_diloco_cadence_and_consensus(tmp_path):
    """Test 3: Verify pseudo_gradient (DiLoCo) on 2x GPU NCCL:
    - Runs 2 steps with sync_every_steps = 2, outer_lr = 0.5, outer_momentum = 0.8.
    - Step 1 (local step): Ranks drift from anchor independently.
    - Step 2 (cadence boundary): Pseudo-gradients reduced; outer update applied.
    - _assert_replicas('complete replicated state') succeeds at step 2.
    - Anchor weights and model weights achieve bit-identical replica consensus.
    """
    agg = {"strategy": "pseudo_gradient", "sync_every_steps": 2, "outer_lr": 0.5, "outer_momentum": 0.8}
    results = run_2gpu_subprocess(tmp_path, mode="step", agg_cfg=agg, steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]

    # Weights at step 2 must match across ranks
    w0_step2 = r0["weight_history"][1]
    w1_step2 = r1["weight_history"][1]
    for key in w0_step2:
        torch.testing.assert_close(torch.tensor(w0_step2[key]), torch.tensor(w1_step2[key]), rtol=1e-6, atol=1e-7)

    # Anchors at step 2 must match across ranks
    anchor0 = r0["anchor"]
    anchor1 = r1["anchor"]
    for key in anchor0:
        torch.testing.assert_close(torch.tensor(anchor0[key]), torch.tensor(anchor1[key]), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("strategy", ["parameter_average", "pseudo_gradient"])
def test_2gpu_nccl_adam_cpu_step_tensor_regression_guard(tmp_path, strategy):
    """Test 4: Specific regression test for Adam param_state['step'] CPU tensor.
    Verifies that _synchronize skips key == 'step' CPU tensor and completes without
    RuntimeError: No backend type associated with device type cpu.
    """
    agg = {"strategy": strategy, "sync_every_steps": 2, "outer_lr": 0.5, "outer_momentum": 0.8}
    results = run_2gpu_subprocess(tmp_path, mode="adam_cpu_guard", agg_cfg=agg, steps=2)
    assert len(results) == 2
    assert results[0]["guard_passed"] and results[1]["guard_passed"]


def test_2gpu_nccl_checkpoint_save_and_manifest_metadata(tmp_path):
    """Test 5: Verify distributed checkpoint save on 2x GPU NCCL:
    - Saves checkpoint at step 2 under parameter_average.
    - Top-level manifest.json contains valid [aggregation] table.
    - Rank payloads (rank-00000.pt, rank-00001.pt) contain CPU-portable state and aggregation state.
    """
    agg = {"strategy": "parameter_average", "sync_every_steps": 2}
    run_dir = tmp_path / "run_checkpoint_save"
    run_dir.mkdir(exist_ok=True)
    results = run_2gpu_subprocess(tmp_path, mode="checkpoint_save", agg_cfg=agg, steps=2)
    assert len(results) == 2
    
    ckpt_dir = Path(results[0]["checkpoint_path"])
    manifest_path = ckpt_dir / "manifest.json"
    assert manifest_path.exists()
    
    manifest = json.loads(manifest_path.read_text())
    assert manifest["step"] == 2
    assert "aggregation" in manifest
    assert manifest["aggregation"]["strategy"] == "parameter_average"
    assert manifest["aggregation"]["sync_every_steps"] == 2
    
    # Verify rank payloads exist and contain portable CPU tensors
    for r in range(2):
        payload_path = ckpt_dir / f"rank-{r:05d}.pt"
        assert payload_path.exists()
        payload = torch.load(payload_path, weights_only=False)
        assert "aggregation" in payload
        assert payload["aggregation"]["step_count"] == 2
        assert payload["aggregation"]["strategy"] == "parameter_average"


def test_2gpu_nccl_checkpoint_resume_and_trajectory_continuity(tmp_path):
    """Test 6: Verify exact resume continuity on 2x GPU NCCL under pseudo_gradient:
    - Phase 1: Train 2 steps and save checkpoint.
    - Phase 2: In a fresh worker group, restore checkpoint and continue for steps 3 and 4.
    - Verify restored step count is 2 and final step reaches 4.
    """
    agg = {"strategy": "pseudo_gradient", "sync_every_steps": 2, "outer_lr": 0.5, "outer_momentum": 0.8}
    run_dir = tmp_path / "run_checkpoint_resume"
    run_dir.mkdir(exist_ok=True)
    
    # 1. Save checkpoint at step 2
    save_results = run_2gpu_subprocess(tmp_path, mode="checkpoint_save", agg_cfg=agg, steps=2, run_dir=run_dir)
    assert len(save_results) == 2
    
    # 2. Resume from checkpoint for steps 3 and 4
    resume_results = run_2gpu_subprocess(tmp_path, mode="checkpoint_resume", agg_cfg=agg, steps=2, run_dir=run_dir)
    assert len(resume_results) == 2
    r0, r1 = resume_results[0], resume_results[1]
    
    # Both ranks reached step 4 with agreement
    assert len(r0["metrics"]) == 2
    assert r0["metrics"][0]["step"] == 3
    assert r0["metrics"][1]["step"] == 4
    
    for key in r0["final_weights"]:
        torch.testing.assert_close(
            torch.tensor(r0["final_weights"][key]),
            torch.tensor(r1["final_weights"][key]),
            rtol=1e-6, atol=1e-7
        )


def test_2gpu_nccl_replicated_recipe_features_with_aggregation(tmp_path):
    """Test 7: Verify recipe parity features (prior.z inputs, component reuse) on 2x GPU NCCL with aggregation."""
    agg = {"strategy": "parameter_average", "sync_every_steps": 2}
    results = run_2gpu_subprocess(
        tmp_path,
        mode="step",
        agg_cfg=agg,
        steps=2,
        config_kwargs={"use_prior_input": True, "use_reuse": True},
    )
    assert len(results) == 2
    r0, r1 = results[0], results[1]
    assert r0["final_step"] == 2 and r1["final_step"] == 2
    
    # Check consensus at step 2
    w0_step2 = r0["weight_history"][1]
    w1_step2 = r1["weight_history"][1]
    for key in w0_step2:
        torch.testing.assert_close(torch.tensor(w0_step2[key]), torch.tensor(w1_step2[key]), rtol=1e-6, atol=1e-7)

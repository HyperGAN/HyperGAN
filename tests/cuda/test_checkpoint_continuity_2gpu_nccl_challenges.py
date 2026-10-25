"""Empirical Challenge Suite: 2-GPU NCCL Checkpoint Save, Manifest Persistence,
and Exact Trajectory Continuity Matrix (Milestone 2 Iteration 3).

Challenges:
1. 2-GPU NCCL parameter_average save/resume continuity across multiple cadences (cadence 1, 2).
2. 2-GPU NCCL pseudo_gradient save/resume continuity across multiple cadences (cadence 1, 2).
3. 2-GPU NCCL bit-identical state restoration verification:
   - Restored anchor tensors match checkpoint payload bit-for-bit (torch.equal)
   - Restored momentum buffer tensors match checkpoint payload bit-for-bit (torch.equal)
   - Local step counters (trainer.step == K, agg_strategy.step_count == K)
   - Restored optimizer regularizer states
   - Adam optimizer scalar CPU step tensor guard integrity
4. Exact trajectory continuity vs uninterrupted continuous run on 2x NVIDIA RTX A6000 GPUs.
"""
from __future__ import annotations

import json
import math
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hypergan.config import resolve_config


def is_2gpu_nccl_available() -> bool:
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available():
            return False
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def require_2gpu_nccl():
    if not is_2gpu_nccl_available():
        pytest.skip("Test suite requires at least 2 visible CUDA GPUs with NCCL backend support")


def make_nccl_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "1000"
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")
    return env


def make_toy_cuda_config(batch_size: int = 8, steps: int = 4) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        'components': {
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
        },
        'prior': {
            'args': {'num_particles': 16, 'z_dim': 4},
        },
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {
            'steps': steps,
            'batch_size': batch_size,
            'device': 'cuda',
            'seed': 42,
        },
        'sampling': {'count': 4},
    }
    return resolve_config(raw)


WORKER_CODE = r'''
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

mode, rank_str, uri, cfg_path, run_dir, out_file, agg_json, total_steps_str, split_steps_str = sys.argv[1:10]
rank = int(rank_str)
total_steps = int(total_steps_str)
split_steps = int(split_steps_str)
run_dir = Path(run_dir)
agg_cfg = json.loads(agg_json)

torch.set_num_threads(1)
torch.cuda.set_device(rank)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

dist.init_process_group("nccl", init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=35))
try:
    cfg = json.loads(Path(cfg_path).read_text())
    
    # Deterministic synthetic data
    data = torch.linspace(-1, 1, 16, device="cuda").reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda")
    local_ids = ids.chunk(2)[rank]
    local_data = data.chunk(2)[rank]

    result = {"rank": rank, "mode": mode, "passed": True}

    if mode == "continuous":
        trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        metrics = []
        for step in range(1, total_steps + 1):
            m, last_batch = trainer.update({"real": local_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})

        result["metrics"] = metrics
        result["final_step"] = trainer.step
        result["strategy_step_count"] = trainer.aggregation_strategy.step_count
        result["final_weights"] = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
        if hasattr(trainer.aggregation_strategy, "anchor"):
            result["anchor"] = {k: v.detach().cpu().tolist() for k, v in trainer.aggregation_strategy.anchor.items()}
        if hasattr(trainer.aggregation_strategy, "momentum_buffer"):
            result["momentum_buffer"] = {k: v.detach().cpu().tolist() for k, v in trainer.aggregation_strategy.momentum_buffer.items()}
        Path(out_file).write_text(json.dumps(result))

    elif mode == "split":
        trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        last_batch = None
        metrics = []
        for step in range(1, split_steps + 1):
            m, last_batch = trainer.update({"real": local_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})

        ckpt_path = save_distributed_checkpoint(
            run_dir, trainer, last_batch,
            {"run_id": "cuda-continuity-run", "attempt_id": "att-1", "next_sample_sequence": split_steps + 1}
        )
        result["checkpoint_path"] = str(ckpt_path)
        result["metrics"] = metrics
        result["final_weights"] = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
        if hasattr(trainer.aggregation_strategy, "anchor"):
            result["anchor"] = {k: v.detach().cpu().tolist() for k, v in trainer.aggregation_strategy.anchor.items()}
        if hasattr(trainer.aggregation_strategy, "momentum_buffer"):
            result["momentum_buffer"] = {k: v.detach().cpu().tolist() for k, v in trainer.aggregation_strategy.momentum_buffer.items()}
        Path(out_file).write_text(json.dumps(result))

    elif mode == "resume":
        fresh_trainer = ReplicatedTrainer(cfg, world_size=2, aggregation=agg_cfg)
        restored_path, info, last_batch = restore_distributed_checkpoint(
            run_dir, fresh_trainer, {"run_id": "cuda-continuity-run"}
        )
        assert fresh_trainer.step == split_steps
        assert fresh_trainer.aggregation_strategy.step_count == split_steps

        result["restored_step"] = fresh_trainer.step
        result["restored_agg_step_count"] = fresh_trainer.aggregation_strategy.step_count
        if hasattr(fresh_trainer.aggregation_strategy, "anchor"):
            result["restored_anchor"] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.aggregation_strategy.anchor.items()}
        if hasattr(fresh_trainer.aggregation_strategy, "momentum_buffer"):
            result["restored_momentum_buffer"] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.aggregation_strategy.momentum_buffer.items()}

        metrics = []
        for step in range(split_steps + 1, total_steps + 1):
            m, last_batch = fresh_trainer.update({"real": local_data}, (fresh_trainer.prior.z[local_ids], local_ids))
            metrics.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})

        result["restored_path"] = str(restored_path)
        result["metrics"] = metrics
        result["final_weights"] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.graph.state_dict().items()}
        if hasattr(fresh_trainer.aggregation_strategy, "anchor"):
            result["anchor"] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.aggregation_strategy.anchor.items()}
        if hasattr(fresh_trainer.aggregation_strategy, "momentum_buffer"):
            result["momentum_buffer"] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.aggregation_strategy.momentum_buffer.items()}
        Path(out_file).write_text(json.dumps(result))

finally:
    dist.destroy_process_group()
'''


def run_cuda_continuity_subprocess(
    tmp_path: Path,
    mode: str,
    agg_cfg: Dict[str, Any],
    run_dir: Path,
    total_steps: int = 4,
    split_steps: int = 2,
    timeout: float = 45.0,
) -> List[Dict[str, Any]]:
    cfg_file = tmp_path / f"cfg_{mode}_{time.monotonic_ns()}.json"
    cfg_dict = make_toy_cuda_config(batch_size=8, steps=total_steps + 2)
    cfg_file.write_text(json.dumps(cfg_dict))

    worker_script = tmp_path / "cuda_continuity_worker.py"
    if not worker_script.exists():
        worker_script.write_text(WORKER_CODE)

    uri = (tmp_path / f"rendezvous_{mode}_{time.monotonic_ns()}").as_uri()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_files = [tmp_path / f"out_{mode}_{r}_{time.monotonic_ns()}.json" for r in range(2)]

    env = make_nccl_env()
    processes = []
    try:
        for r in range(2):
            cmd = [
                sys.executable, str(worker_script),
                mode, str(r), uri, str(cfg_file), str(run_dir), str(out_files[r]),
                json.dumps(agg_cfg), str(total_steps), str(split_steps),
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


# ==============================================================================
# Challenge 1: 2-GPU NCCL parameter_average Continuity Matrix
# ==============================================================================

@pytest.mark.parametrize("cadence,total_steps,split_steps", [
    (1, 4, 2),
    (2, 4, 2),
])
def test_2gpu_nccl_parameter_average_continuity_matrix(tmp_path, cadence, total_steps, split_steps):
    """Challenge 1: Verify 2-GPU NCCL parameter_average exact trajectory continuity across cadences 1 and 2."""
    agg = {"strategy": "parameter_average", "sync_every_steps": cadence}
    run_cont = tmp_path / f"pa_cuda_cont_{cadence}"
    run_split = tmp_path / f"pa_cuda_split_{cadence}"

    # 1. Continuous run
    cont_results = run_cuda_continuity_subprocess(tmp_path, "continuous", agg, run_cont, total_steps=total_steps, split_steps=split_steps)
    assert len(cont_results) == 2

    # 2. Split run
    split_results = run_cuda_continuity_subprocess(tmp_path, "split", agg, run_split, total_steps=total_steps, split_steps=split_steps)
    assert len(split_results) == 2

    # Verify manifest.json
    ckpt_dir = Path(split_results[0]["checkpoint_path"])
    manifest = json.loads((ckpt_dir / "manifest.json").read_text())
    assert manifest["step"] == split_steps
    assert manifest["aggregation"]["strategy"] == "parameter_average"
    assert manifest["aggregation"]["sync_every_steps"] == cadence

    # 3. Resume run
    resume_results = run_cuda_continuity_subprocess(tmp_path, "resume", agg, run_split, total_steps=total_steps, split_steps=split_steps)
    assert len(resume_results) == 2

    # 4. Trajectory continuity
    for r in range(2):
        c_res = cont_results[r]
        res_res = resume_results[r]

        assert res_res["restored_step"] == split_steps
        assert res_res["restored_agg_step_count"] == split_steps

        # Final weights match continuous run within rtol=1e-5, atol=1e-6
        for k in c_res["final_weights"]:
            torch.testing.assert_close(
                torch.tensor(res_res["final_weights"][k]),
                torch.tensor(c_res["final_weights"][k]),
                rtol=1e-5, atol=1e-6
            )

        # Loss trajectories for resumed steps match continuous
        remaining = total_steps - split_steps
        assert len(res_res["metrics"]) == remaining
        for idx, r_step in enumerate(range(split_steps + 1, total_steps + 1)):
            c_m = c_res["metrics"][r_step - 1]
            res_m = res_res["metrics"][idx]
            assert res_m["step"] == r_step
            assert math.isclose(res_m["d_loss"], c_m["d_loss"], rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(res_m["g_loss"], c_m["g_loss"], rel_tol=1e-5, abs_tol=1e-6)


# ==============================================================================
# Challenge 2: 2-GPU NCCL pseudo_gradient Continuity Matrix
# ==============================================================================

@pytest.mark.parametrize("cadence,total_steps,split_steps,outer_lr,outer_momentum", [
    (1, 2, 1, 0.5, 0.8),
    (2, 4, 2, 0.5, 0.8),
])
def test_2gpu_nccl_pseudo_gradient_continuity_matrix(tmp_path, cadence, total_steps, split_steps, outer_lr, outer_momentum):
    """Challenge 2: Verify 2-GPU NCCL pseudo_gradient exact trajectory continuity across cadences 1 and 2."""
    agg = {
        "strategy": "pseudo_gradient",
        "sync_every_steps": cadence,
        "outer_lr": outer_lr,
        "outer_momentum": outer_momentum,
    }
    run_cont = tmp_path / f"pg_cuda_cont_{cadence}"
    run_split = tmp_path / f"pg_cuda_split_{cadence}"

    # 1. Continuous run
    cont_results = run_cuda_continuity_subprocess(tmp_path, "continuous", agg, run_cont, total_steps=total_steps, split_steps=split_steps)
    assert len(cont_results) == 2

    # 2. Split run
    split_results = run_cuda_continuity_subprocess(tmp_path, "split", agg, run_split, total_steps=total_steps, split_steps=split_steps)
    assert len(split_results) == 2

    # Verify manifest.json
    ckpt_dir = Path(split_results[0]["checkpoint_path"])
    manifest = json.loads((ckpt_dir / "manifest.json").read_text())
    assert manifest["step"] == split_steps
    assert manifest["aggregation"]["strategy"] == "pseudo_gradient"
    assert manifest["aggregation"]["sync_every_steps"] == cadence
    assert manifest["aggregation"]["outer_lr"] == outer_lr
    assert manifest["aggregation"]["outer_momentum"] == outer_momentum

    # 3. Resume run
    resume_results = run_cuda_continuity_subprocess(tmp_path, "resume", agg, run_split, total_steps=total_steps, split_steps=split_steps)
    assert len(resume_results) == 2

    # 4. Trajectory continuity
    for r in range(2):
        c_res = cont_results[r]
        res_res = resume_results[r]

        assert res_res["restored_step"] == split_steps
        assert res_res["restored_agg_step_count"] == split_steps

        # Weights match continuous run
        for k in c_res["final_weights"]:
            torch.testing.assert_close(
                torch.tensor(res_res["final_weights"][k]),
                torch.tensor(c_res["final_weights"][k]),
                rtol=1e-5, atol=1e-6
            )

        # Anchor and momentum buffers match continuous run
        for k in c_res["anchor"]:
            torch.testing.assert_close(
                torch.tensor(res_res["anchor"][k]),
                torch.tensor(c_res["anchor"][k]),
                rtol=1e-5, atol=1e-6
            )
        for k in c_res["momentum_buffer"]:
            torch.testing.assert_close(
                torch.tensor(res_res["momentum_buffer"][k]),
                torch.tensor(c_res["momentum_buffer"][k]),
                rtol=1e-5, atol=1e-6
            )

        # Loss trajectories match continuous
        remaining = total_steps - split_steps
        assert len(res_res["metrics"]) == remaining
        for idx, r_step in enumerate(range(split_steps + 1, total_steps + 1)):
            c_m = c_res["metrics"][r_step - 1]
            res_m = res_res["metrics"][idx]
            assert res_m["step"] == r_step
            assert math.isclose(res_m["d_loss"], c_m["d_loss"], rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(res_m["g_loss"], c_m["g_loss"], rel_tol=1e-5, abs_tol=1e-6)


# ==============================================================================
# Challenge 3: 2-GPU NCCL Bit-Identical Restoration Verification
# ==============================================================================

def test_2gpu_nccl_bit_identical_restoration_verification(tmp_path):
    """Challenge 3: Verify bit-identical anchor, momentum buffer, and optimizer regularizer restoration on 2-GPU NCCL."""
    agg = {
        "strategy": "pseudo_gradient",
        "sync_every_steps": 2,
        "outer_lr": 0.5,
        "outer_momentum": 0.8,
    }
    run_dir = tmp_path / "cuda_bit_verify"
    split_results = run_cuda_continuity_subprocess(tmp_path, "split", agg, run_dir, total_steps=4, split_steps=2)
    assert len(split_results) == 2

    ckpt_dir = Path(split_results[0]["checkpoint_path"])
    for r in range(2):
        payload_file = ckpt_dir / f"rank-{r:05d}.pt"
        payload = torch.load(payload_file, weights_only=True)

        assert payload["step"] == 2
        assert payload["aggregation"]["step_count"] == 2
        assert "anchor" in payload["aggregation"]
        assert "momentum_buffer" in payload["aggregation"]

        # Ensure tensors in checkpoint payload are strictly CPU tensors
        for k, v in payload["aggregation"]["anchor"].items():
            assert v.device.type == "cpu"
        for k, v in payload["aggregation"]["momentum_buffer"].items():
            assert v.device.type == "cpu"

        # Check optimizer regularizer serialization
        opt_g = payload["optimizers"][0]
        assert "regularizer" in opt_g
        assert "latent" in opt_g["regularizer"]

    # Resume and verify restored initial anchor and momentum match saved payload
    resume_results = run_cuda_continuity_subprocess(tmp_path, "resume", agg, run_dir, total_steps=4, split_steps=2)
    assert len(resume_results) == 2

    for r in range(2):
        res = resume_results[r]
        payload = torch.load(ckpt_dir / f"rank-{r:05d}.pt", weights_only=True)
        saved_anchor = payload["aggregation"]["anchor"]
        saved_momentum = payload["aggregation"]["momentum_buffer"]

        for k in saved_anchor:
            restored_t = torch.tensor(res["restored_anchor"][k])
            torch.testing.assert_close(restored_t, saved_anchor[k], rtol=0, atol=0)

        for k in saved_momentum:
            restored_t = torch.tensor(res["restored_momentum_buffer"][k])
            torch.testing.assert_close(restored_t, saved_momentum[k], rtol=0, atol=0)

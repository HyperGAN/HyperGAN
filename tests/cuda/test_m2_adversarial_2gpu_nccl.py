"""Adversarial stress test suite for Milestone 2 on dual NVIDIA RTX A6000 GPUs with NCCL:
- Feature 23: Hot-Path Parameter Bucketing & Gradient Coalescing on CUDA
- Feature 24: Barrier Elimination & Handshake Lifecycle on CUDA

Author: challenger_m2_1 (Milestone 2 Challenger)
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

pytestmark = pytest.mark.multi_gpu


def is_2gpu_nccl_available() -> bool:
    """Opaque capability detection for at least 2 visible CUDA GPUs with NCCL."""
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return False
        if not dist.is_nccl_available():
            return False
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def require_2gpu_nccl():
    if not is_2gpu_nccl_available():
        pytest.skip("Test suite requires at least 2 visible CUDA GPUs with NCCL backend support")


def make_nccl_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "1000"
    return env


CUDA_WORKER_SCRIPT = r'''
import json
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from datetime import timedelta

from hypergan.distributed_training import ReplicatedTrainer, _bucket_parameters
from hypergan.config import resolve_config

mode, rank, uri, out_path = sys.argv[1:5]
rank = int(rank)
torch.cuda.set_device(rank)
dist.init_process_group("nccl", init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=30))

try:
    if mode == "heterogeneous_cuda":
        # 1. Test Feature 23 on CUDA with heterogeneous shapes and partial participation
        device = torch.device(f"cuda:{rank}")
        
        # Heterogeneous model on CUDA
        class CudaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p_0d = nn.Parameter(torch.tensor(1.5, device=device))
                self.p_1d = nn.Parameter(torch.ones(13, device=device))
                self.p_2d = nn.Parameter(torch.ones(17, 31, device=device))
                self.p_4d = nn.Parameter(torch.ones(4, 2, 3, 3, device=device))
                self.p_none = nn.Parameter(torch.ones(8, device=device))
                self.p_partial_r0 = nn.Parameter(torch.ones(6, device=device))
                self.p_partial_r1 = nn.Parameter(torch.ones(6, device=device))
        
        model = CudaModel()
        
        # Assign gradients
        model.p_0d.grad = torch.tensor(1.0 * (rank + 1), device=device)
        model.p_1d.grad = torch.full_like(model.p_1d.data, 2.0 * (rank + 1))
        model.p_2d.grad = torch.full_like(model.p_2d.data, 3.0 * (rank + 1))
        model.p_4d.grad = torch.full_like(model.p_4d.data, 4.0 * (rank + 1))
        model.p_none.grad = None  # Global None
        
        if rank == 0:
            model.p_partial_r0.grad = torch.full_like(model.p_partial_r0.data, 10.0)
            model.p_partial_r1.grad = None
        else:
            model.p_partial_r0.grad = None
            model.p_partial_r1.grad = torch.full_like(model.p_partial_r1.data, 20.0)

        # Mock trainer environment with CUDA device
        from hypergan.distributed import Collectives
        class MockTrainer:
            def __init__(self, dev):
                self.device = dev
                self.world_size = 2
                self.collectives = Collectives(2, device=dev)
                self.aggregation_strategy = type('Agg', (), {'on_backward': lambda *a: None, 'reduces_gradients': True})()
                self._bucket_cap_bytes = 25 * 1024 * 1024
            def _parameters(self, opt):
                return [p for g in opt.param_groups for p in g['params']]
            def _unscale(self, dev):
                return torch.ones((), dtype=torch.float32, device=dev)
            def _gradient_flags(self, params):
                flags = []
                for p in params:
                    if p.grad is not None:
                        flags.append(torch.isfinite(p.grad).all().logical_not().to(device=self.device, dtype=torch.int32))
                return flags

        mock_trainer = MockTrainer(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Call reduction method
        ReplicatedTrainer._reduce_gradients(mock_trainer, opt, "test_cuda")

        # Verify results on CUDA
        torch.testing.assert_close(model.p_0d.grad, torch.tensor(1.5, device=device))
        torch.testing.assert_close(model.p_1d.grad, torch.full_like(model.p_1d.data, 3.0))
        torch.testing.assert_close(model.p_2d.grad, torch.full_like(model.p_2d.data, 4.5))
        torch.testing.assert_close(model.p_4d.grad, torch.full_like(model.p_4d.data, 6.0))
        assert model.p_none.grad is None, "Global None parameter gradient must remain None on CUDA"
        torch.testing.assert_close(model.p_partial_r0.grad, torch.full_like(model.p_partial_r0.data, 5.0))
        torch.testing.assert_close(model.p_partial_r1.grad, torch.full_like(model.p_partial_r1.data, 10.0))

        Path(out_path).write_text(json.dumps({"rank": rank, "passed": True}))

    elif mode == "nan_cuda":
        device = torch.device(f"cuda:{rank}")
        p1 = nn.Parameter(torch.ones(4, device=device))
        p2 = nn.Parameter(torch.ones(4, device=device))

        if rank == 0:
            p1.grad = torch.tensor([1.0, float("nan"), 3.0, 4.0], device=device)
            p2.grad = torch.ones(4, device=device) * 2.0
        else:
            p1.grad = torch.ones(4, device=device) * 1.0
            p2.grad = torch.ones(4, device=device) * 2.0

        from hypergan.distributed import Collectives
        class MockTrainer:
            def __init__(self, dev):
                self.device = dev
                self.world_size = 2
                self.collectives = Collectives(2, device=dev)
                self.aggregation_strategy = type('Agg', (), {'on_backward': lambda *a: None, 'reduces_gradients': True})()
                self._bucket_cap_bytes = 25 * 1024 * 1024
            def _parameters(self, opt):
                return [p for g in opt.param_groups for p in g['params']]
            def _unscale(self, dev):
                return torch.ones((), dtype=torch.float32, device=dev)
            def _gradient_flags(self, params):
                flags = []
                for p in params:
                    if p.grad is not None:
                        flags.append(torch.isfinite(p.grad).all().logical_not().to(device=self.device, dtype=torch.int32))
                return flags

        mock_trainer = MockTrainer(device)
        opt = torch.optim.SGD([p1, p2], lr=0.1)

        raised = False
        try:
            ReplicatedTrainer._reduce_gradients(mock_trainer, opt, "test_cuda_nan")
        except ValueError as exc:
            if "Nonfinite or sparse gradient; optimizer step refused" in str(exc):
                raised = True
        
        assert raised, f"Rank {rank}: Expected ValueError was not raised on CUDA"
        assert p1.grad is None, f"Rank {rank}: p1.grad was not reset to None"
        assert p2.grad is None, f"Rank {rank}: p2.grad was not reset to None"

        Path(out_path).write_text(json.dumps({"rank": rank, "passed": True}))

    elif mode == "barrier_elimination_cuda":
        # Full ReplicatedTrainer 3-step execution on CUDA
        cfg = resolve_config({
            'components': {
                'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
                'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
            },
            'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
            'training': {'steps': 3, 'batch_size': 8, 'device': 'cuda'},
        })
        
        trainer = ReplicatedTrainer(cfg, world_size=2)
        
        # Track dist.all_gather_object calls
        original_gather = dist.all_gather_object
        counts = {"step_1": 0, "step_2": 0, "step_3": 0}
        active_step = [1]
        
        def spy_gather(output_list, input_obj, group=None):
            counts[f"step_{active_step[0]}"] += 1
            return original_gather(output_list, input_obj, group=group)
            
        dist.all_gather_object = spy_gather
        try:
            data = torch.linspace(-1, 1, 16, device="cuda").reshape(8, 2)
            local_data = data.chunk(2)[rank]
            latent = (torch.randn(4, 4, device="cuda"), None)
            
            # Step 1: Handshake
            active_step[0] = 1
            trainer.update({"real": local_data}, latent)
            assert counts["step_1"] >= 15
            assert trainer._in_hot_path is True
            
            # Step 2: In hot path, 0 barriers
            active_step[0] = 2
            trainer.update({"real": local_data}, latent)
            assert counts["step_2"] == 0, f"Step 2 had {counts['step_2']} barriers, expected 0"
            
            # Step 3: In hot path, 0 barriers
            active_step[0] = 3
            trainer.update({"real": local_data}, latent)
            assert counts["step_3"] == 0, f"Step 3 had {counts['step_3']} barriers, expected 0"
            
            Path(out_path).write_text(json.dumps({
                "rank": rank,
                "passed": True,
                "step_1": counts["step_1"],
                "step_2": counts["step_2"],
                "step_3": counts["step_3"],
            }))
        finally:
            dist.all_gather_object = original_gather

finally:
    dist.barrier()
    dist.destroy_process_group()
'''


def run_cuda_subprocesses(mode: str, tmp_path: Path):
    rendezvous_file = tmp_path / "rendezvous"
    uri = rendezvous_file.as_uri()
    env = make_nccl_env()
    
    script_path = tmp_path / "worker.py"
    script_path.write_text(CUDA_WORKER_SCRIPT)

    processes = []
    out_files = []
    for rank in (0, 1):
        out_file = tmp_path / f"out_rank_{rank}.json"
        out_files.append(out_file)
        cmd = [
            sys.executable,
            str(script_path),
            mode,
            str(rank),
            uri,
            str(out_file),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(p)

    for rank, p in enumerate(processes):
        stdout, stderr = p.communicate(timeout=30)
        assert p.returncode == 0, f"Rank {rank} failed with exit code {p.returncode}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        assert out_files[rank].exists(), f"Rank {rank} output file missing"
        res = json.loads(out_files[rank].read_text())
        assert res.get("passed") is True, f"Rank {rank} reported failure: {res}"


def test_cuda_adversarial_heterogeneous_shapes_and_partial_participation(tmp_path):
    run_cuda_subprocesses("heterogeneous_cuda", tmp_path)


def test_cuda_adversarial_nan_halts_all_ranks(tmp_path):
    run_cuda_subprocesses("nan_cuda", tmp_path)


def test_cuda_adversarial_barrier_elimination_multi_step(tmp_path):
    run_cuda_subprocesses("barrier_elimination_cuda", tmp_path)

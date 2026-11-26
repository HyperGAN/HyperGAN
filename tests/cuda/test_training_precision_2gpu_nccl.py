"""Two-GPU NCCL DDP test for bf16_channels_last training_precision."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.multi_gpu]

WORKER = '''
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import torch
import torch.distributed as dist

from hypergan.config import load_config
from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint
from hypergan.distributed_training import ReplicatedTrainer

rank, root = int(sys.argv[1]), Path(sys.argv[2])
torch.set_num_threads(1)
device = torch.device("cuda", rank)
torch.cuda.set_device(device)
assert torch.cuda.is_bf16_supported(including_emulation=False), "Requires native BF16 GPU"
dist.init_process_group(
    "nccl",
    init_method=(root / "store").as_uri(),
    rank=rank,
    world_size=2,
    timeout=timedelta(seconds=45),
    device_id=device,
)
try:
    config = load_config(
        {
            "format": "hypergan-model/1",
            "name": "test/ddp-bf16-channels-last",
            "defaults": "particlegan",
            "data": {"builtin": "gaussian_grid"},
            "prior": {"kind": "particles", "z_dim": 16, "num_particles": 64, "initialization_seed": 2},
            "networks": {
                "generator": {
                    "role": "generator",
                    "inputs": {"x": "latent"},
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 3, 16, 16],
                    "source": (
                        "linear(1024)\\nreshape(64, 4, 4)\\ngroup_norm(8)\\nsilu()\\n"
                        "upsample(2)\\nconv(64, kernel_size=3, padding=1)\\ngroup_norm(8)\\nsilu()\\n"
                        "upsample(2)\\nconv(3, kernel_size=3, padding=1)\\ntanh()"
                    ),
                },
                "discriminator": {
                    "role": "critic",
                    "inputs": {"x": "candidate"},
                    "input_shape": ["B", 3, 16, 16],
                    "output_shape": ["B", 1],
                    "source": (
                        "conv(64, kernel_size=3, stride=2, padding=1)\\nsilu()\\n"
                        "conv(64, kernel_size=3, stride=2, padding=1)\\nsilu()\\n"
                        "flatten()\\nlinear(1)"
                    ),
                },
            },
            "losses": [
                {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
                {"type": "prior"},
            ],
            "train": {
                "steps": 3,
                "batch_size": 8,
                "seed": 17,
                "device": "cuda",
                "initialization": "deterministic_orthogonal",
                "precision": "bf16_channels_last",
            },
        }
    )
    trainer = ReplicatedTrainer(config, world_size=2, accumulation_steps=1)
    assert trainer.graph._bf16_components == frozenset({"generator", "discriminator"})
    assert trainer.graph._channels_last_components == frozenset({"generator", "discriminator"})
    form_attr = "_axiom_bf16_autocast" if getattr(trainer.recipe, "effective_critic_formulation", None) == "axiom" else "_unified_v4_bf16_autocast"
    assert getattr(trainer._generator_probe, "_particlegan_bf16_autocast", False) is True
    assert getattr(trainer._generator_probe, form_attr, False) is True
    for p in trainer.graph.parameters():
        assert p.dtype == torch.float32 and p.layout == torch.strided
    conv_w = [
        m.weight
        for m in trainer.graph.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and m.weight.ndim == 4
    ]
    assert conv_w and all(w.is_contiguous(memory_format=torch.channels_last) for w in conv_w)

    real_all = torch.randn(8, 3, 16, 16, generator=torch.Generator().manual_seed(123)).to(trainer.device)
    batch = {"real": real_all.chunk(2)[rank].clone()}
    ids = torch.arange(8, device=trainer.device).chunk(2)[rank]
    row1, last1 = trainer.update(batch, (trainer.prior.z[ids], ids))
    assert all(math.isfinite(row1[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))
    trainer._assert_replicas("after bf16 step 1")

    run_dir = root / "ddp_run"
    if rank == 0:
        run_dir.mkdir(exist_ok=True)
    dist.barrier()
    save_distributed_checkpoint(
        run_dir,
        trainer,
        last1,
        {"run_id": "ddp-bf16", "attempt_id": "a1", "next_sample_sequence": 1},
    )
    restored = ReplicatedTrainer(config, world_size=2, accumulation_steps=1)
    restore_distributed_checkpoint(run_dir, restored, {"run_id": "ddp-bf16"})
    assert restored.step == 1
    row2, _ = restored.update(batch, (restored.prior.z[ids], ids))
    assert all(math.isfinite(row2[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))
    restored._assert_replicas("after bf16 restored step 2")
    (root / f"rank-{rank}.json").write_text(json.dumps({"ok": True, "step": restored.step}))
finally:
    dist.destroy_process_group()
'''


def test_two_gpu_nccl_bf16_channels_last_update_and_resume(tmp_path: Path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Requires 2 CUDA GPUs"
    assert torch.distributed.is_nccl_available(), "Requires NCCL"
    script = tmp_path / "ddp_bf16_worker.py"
    script.write_text(WORKER, encoding="utf-8")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] = "1000"
    children, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [sys.executable, str(script), str(rank), str(tmp_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            )
        deadline = time.monotonic() + 120
        for child in children:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        for log in logs:
            log.flush()
        errors = "\n".join(path.read_text() for path in sorted(tmp_path.glob("rank-*.log")))
        assert all(child.returncode == 0 for child in children), errors
        for rank in range(2):
            assert json.loads((tmp_path / f"rank-{rank}.json").read_text()) == {"ok": True, "step": 2}
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for log in logs:
            log.close()

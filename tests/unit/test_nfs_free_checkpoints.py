from datetime import timedelta
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist

from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.distributed_checkpoints import (
    prepare_distributed_checkpoint,
    restore_distributed_checkpoint,
    save_distributed_checkpoint,
)
from tests.distributed_fixtures import make_toy_config


def _worker_nfs_free_save_and_restore(rank, world_size, dir_rank0, dir_rank1):
    # rank 0 uses dir_rank0 (exists), rank 1 uses dir_rank1 (does NOT exist on disk)
    local_run_dir = Path(dir_rank0) if rank == 0 else Path(dir_rank1)

    cfg = make_toy_config(device="cpu", batch_size=8, steps=5)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # Perform 1 step so checkpoint_ready is True
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    local_ids = ids.chunk(world_size)[rank]
    local_batch = {"real": data.chunk(world_size)[rank]}
    trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))

    metadata = {
        "run_id": "nfs_free_test_run",
        "attempt_id": "first",
        "next_sample_sequence": 2,
    }

    # Save checkpoint
    checkpoint_path = save_distributed_checkpoint(local_run_dir, trainer, local_batch, metadata)

    if rank == 0:
        assert checkpoint_path.exists()
        assert (Path(dir_rank0) / "distributed-checkpoints" / "latest.json").exists()
    else:
        # Rank 1 must NEVER have touched or created dir_rank1 / distributed-checkpoints
        assert not Path(dir_rank1).exists()

    # Create a fresh trainer and restore
    fresh_trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)
    restored_batch = restore_distributed_checkpoint(
        local_run_dir,
        fresh_trainer,
        {"run_id": "nfs_free_test_run"},
        checkpoint=None,
    )

    # Check restored parameters match
    p_orig = [p.clone() for p in trainer.graph.models["generator"].parameters()]
    p_rest = [p.clone() for p in fresh_trainer.graph.models["generator"].parameters()]
    for p0, p1 in zip(p_orig, p_rest):
        torch.testing.assert_close(p0, p1, rtol=0, atol=0)


def test_nfs_free_distributed_checkpoints(tmp_path):
    dir_rank0 = tmp_path / "node_0_run"
    dir_rank0.mkdir()
    (dir_rank0 / "manifest.json").write_text(json.dumps({"run_id": "nfs_free_test_run"}))

    # dir_rank1 intentionally does not exist on disk
    dir_rank1 = tmp_path / "node_1_nonexistent_run"

    launch_cpu_workers(
        _worker_nfs_free_save_and_restore,
        args=(str(dir_rank0), str(dir_rank1)),
        world_size=2,
        timeout=30,
    )


def test_prepare_checkpoint_descriptor_omits_root():
    import inspect
    from hypergan import distributed_checkpoints
    src = inspect.getsource(distributed_checkpoints.prepare_distributed_checkpoint)
    # The descriptor dictionary in prepare_distributed_checkpoint must not include 'root'
    assert "'root'" not in src.split("descriptor = {")[1].split("}")[0]

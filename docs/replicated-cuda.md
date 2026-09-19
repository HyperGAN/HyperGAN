# Internal replicated CUDA training

The internal replicated run service supports fixed local CUDA ranks with NCCL. Each rank owns one visible GPU: rank zero uses `cuda:0`, rank one uses `cuda:1`, and so on. The recipe must use `training.device="cuda"`; an indexed device is a native single-GPU choice and is rejected for this profile. `CUDA_VISIBLE_DEVICES` may select and order the devices before launch. The fixed mapping and GPU UUIDs become part of strict recovery identity.

Public `hypergan train` and `hypergan resume` still use native single-process execution. Use this internal API from an importable Python script with a guarded entry point:

```python
from pathlib import Path

from hypergan.config import write_default
from hypergan.replicated_execution import run_resume, run_train


if __name__ == "__main__":
    config = write_default(Path("gpu-project"))
    profile = {
        "schema_version": 1,
        "execution": {
            "name": "cuda-replicated-nccl",
            "world_size": 2,
            "accumulation_steps": 2,
        },
    }
    policy = {
        "startup_timeout": 60,
        "command_timeout": 60,
        "collective_timeout": 15,
        "total_timeout": 300,
    }
    stopped = run_train(
        config, "gpu-run", profile=profile, service_policy=policy,
        stop_after_steps=2, checkpoint_every=1, preview_every=1,
    )
    print(stopped["status"], stopped["last_durable_step"])
    complete = run_resume("gpu-run", service_policy=policy)
    print(complete["status"], complete["last_durable_step"])
```

This is the same [shared lifecycle](replicated-run-service.md), [checkpoint publication](distributed-recovery.md) and [bounded observation](replicated-observation.md) contract as the CPU correctness path. The parent and independent broker remain Torch-free. Only supervised numerical ranks bind GPUs and initialize the NCCL group. The broker monitors coordinator death, worker exits and deadlines, including while a rank is stuck in native code. A collective timeout alone is not a bound on NCCL process exit; the broker terminates and reaps its own workers independently.

Numerical tensors, differentiable gathers, gradient-presence reductions and metric reductions use each rank's GPU. Metadata uses the same NCCL group and its current rank device. This follows [PyTorch's object-collective device requirements](https://docs.pytorch.org/docs/stable/distributed); it introduces no second process group whose collective ordering must be coordinated. GPU work completes before an update or restored state is acknowledged.

Global batch and accumulation semantics match the [existing numerical contract](distributed.md) and [replay algorithm](accumulation.md). Data construction and the global sampler stay on CPU, followed by rank slicing and transfer. Each rank has its own CUDA prior/penalty streams and global RNG state. This implementation duplicates global data decoding and checks replica equality; its purpose is correctness and recovery qualification, not maximum throughput.

Checkpoints stage complete rank state as CPU tensors, including models, Adam, EMA, named streams, global CPU/Python/NumPy/CUDA RNG, sampler position and last batch. Recovery loads each rank's own state onto its assigned device, validates a copied candidate, then agrees on complete live restore. A wrong rank's CUDA RNG device is rejected before candidate model loading. Runtime identity records CUDA/cuDNN/NCCL versions, precision/determinism settings, visible device inventory and ordered rank-to-GPU UUID assignment. Recovery requires the same source, runtime, recipe, data and fixed topology; no native/distributed conversion or old-format migration is provided.

Checkpoint transport still gathers bounded serialized payloads at rank zero. NCCL stages those bytes through GPUs; the 256 MiB per-rank and metadata caps are rejection limits, not reserved memory. This is not a sharded checkpoint system or a multi-host storage protocol. Preview snapshots are copied to CPU and rendered without a process group; supported inference bundles remain CPU-loadable. Custom CUDA-only operators need their own compatible inference path.

The explicit `tests/cuda` gate requires two GPUs and never substitutes CPU or skips missing hardware. See the [execution ledger](../reports/resurrection-status.md) for the exact qualified fixtures and validation results. Successful synthetic GAN updates and recovery do not establish image quality, throughput, arbitrary component determinism, real clusters or release readiness. Public profile routing and bounded CLI output are the next integration gate; actual two-host testing requires a concrete agreed allocation.

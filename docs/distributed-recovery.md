# Fixed-topology CPU recovery

`hypergan.distributed_checkpoints` saves and restores the internal replicated CPU trainer. Every rank calls the same API in the same order on a default Gloo group with a finite timeout. The [local worker supervisor](cpu-workers.md) can start a fresh group and clean up failed workers. These APIs do not enable distributed `hypergan train` or `resume`; integration with the shared run/event/preview/request service remains a separate gate.

The caller creates the run directory before workers start, supplies the same run and attempt IDs to all ranks, and holds one `run_lock(run_dir)` in rank zero for the job lifetime. Other ranks must not acquire that same writer lock. Custom factories and worker code are trusted Python, as in the single-process runtime.

Inside an initialized, coordinated worker:

```python
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.distributed_checkpoints import (
    restore_distributed_checkpoint,
    save_distributed_checkpoint,
)

trainer = ReplicatedCPUTrainer(config, world_size=2)
last_batch = None

# On a fresh whole-group restart, all ranks execute this branch:
if resuming:
    path, info, last_batch = restore_distributed_checkpoint(
        run_dir, trainer, {"run_id": run_id},
    )

while trainer.step < config["training"]["steps"]:
    metrics, last_batch = trainer.update()
    path = save_distributed_checkpoint(
        run_dir, trainer, last_batch,
        {"run_id": run_id, "attempt_id": attempt_id},
    )
```

Both APIs return only after their required rank agreements. Save returns the same completed directory path on every rank. Restore returns `(path, metadata, local_last_batch)` and loads the rank's own state. An explicit `checkpoint` argument selects a completed generation by its name or absolute path inside this run. The default is the latest committed pointer. The new attempt ID belongs to the caller; it must not be substituted for the checkpoint's original attempt identity during validation.

## What a complete checkpoint means

The format uses `distributed-checkpoints/<attempt>-step-<step>-<unique>/` with one tensor payload per rank, a versioned manifest, and an atomic `latest.json` pointer. It is distinct from single-process training checkpoints and inference bundles.

Each rank snapshot includes models, learned prior, both optimizers and their base rates, EMA, modes/trainability, persistent and nonpersistent buffers, registered extra state, named RNG streams, Torch/Python/NumPy global RNG, local sampler state, step and last local batch. Replicated numerical state must agree; rank-specific RNG and data ownership are retained separately. The automatic data path currently advances the same global sampler on every rank before slicing, preserving one image permutation/cursor order while repeating decoding.

Compatibility checks include full recipe and schedule, actual runtime and numerical source identity, data/content/preprocessing/class-map identity, strategy, global/local batch and fixed world size. Topology changes, new recipes and transfer initialization are not resume. Source identity covers the recorded modules; it is not proof that arbitrary hidden transitive Python dependencies or external state are reproducible.

Saving requires a complete boundary on every rank. Rank zero receives all payloads, verifies them against the agreed replicated state, writes and syncs a temporary generation, and waits for the final readiness agreement before publishing the generation and latest pointer. A missing rank, snapshot error, divergent state or staging failure cannot publish an incomplete checkpoint. Failed half-updates remain unavailable for saving.

Filesystem publication and continued worker liveness cannot be one atomic operation. If a worker disconnects after the complete commit, the job may report failure while retaining a valid checkpoint. Likewise, a completed generation can remain unreferenced if publishing the latest pointer fails. Recovery validates the selected generation; it never assumes that the last observed step was durable.

## Restore, failure and limits

Rank zero validates every rank file and digest before distributing states. Every worker validates loading on an isolated trainer copy before mutating live state. Imported Python modules remain shared dependencies in that copy; mutable owned state is copied. Canonical expected state is protected from custom load-hook mutation. A live load failure poisons the affected worker group; stop the job and construct fresh workers rather than retrying on partly mutated trainers. RNG consumed by metadata hooks is isolated from the saved and continuing numerical streams.

Payloads are limited to 256 MiB per rank and metadata to 16 MiB. This correctness implementation uses trusted-worker object collectives and holds all rank payloads on rank zero. It duplicates replicated state and validation copies; it is not a scalable checkpoint backend or a disk quota. Immutable generations are retained until the caller deliberately retires them. A hard-killed writer may leave an unpublished `.pending-` directory, which is never a valid restore target.

This module supplies complete snapshots and fixed-topology recovery. It does not yet own persisted job status, attempt/sample counters, periodic save scheduling, cancellation, observation requests, remote artifact transport, elastic workers or cloud retries. Optional lineage fields record `next_sample_sequence` and checkpoint request IDs; the caller remains responsible for the run-wide monotonic sequence and durable request acknowledgements.

Tests compare uninterrupted and fresh-group resumed state exactly, including a shuffled image-folder fixture with labels and stochastic components. Deliberate failures exercise missing workers, half-updates, rank snapshot and disk errors, divergent optimizer state, corrupt/incomplete files, incompatible identities, and custom load hooks. The [checkpoint report](../reports/core-distributed-2026-09-18.md) records installed-package and CI evidence. GPU/NCCL and real multi-node recovery remain unqualified.

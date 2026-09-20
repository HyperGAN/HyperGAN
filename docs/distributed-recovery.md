# Fixed-topology CPU recovery

The [local CUDA/NCCL extension](replicated-cuda.md) now applies this contract to rank-owned GPUs. CPU-specific examples and original qualification evidence below remain explicit correctness fixtures; consult the [execution ledger](../reports/resurrection-status.md) for the current combined scope.

`hypergan.distributed_checkpoints` saves and restores the internal replicated CPU trainer. Every rank calls the same API in the same order on a default Gloo group with a finite timeout. The [local worker supervisor](cpu-workers.md) can start a fresh group and clean up failed workers. These APIs do not enable distributed `hypergan train` or `resume`; integration with the shared run/event/preview/request service remains a separate gate.

The caller creates the run directory before workers start and supplies the same run and attempt IDs to all ranks. The compatibility `save_distributed_checkpoint` API still uses a rank-zero `run_lock(run_dir)` for the job lifetime. A parent-supervised service instead holds that lock in its parent and uses the preparation/publication split below. Do not acquire the same writer lock in both parent and workers. Custom factories and worker code are trusted Python, as in the single-process runtime.

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

## Parent-controlled publication

`prepare_distributed_checkpoint(run_dir, trainer, last_batch, metadata, *, command_sequence, controller_id)` is an all-rank operation. It reuses the snapshot serializer, CPU state checks, identity agreement and final readiness collective. Rank zero writes only `distributed-checkpoints/.prepared/<attempt>/<controller>/command-<sequence>-<unique>/`. It never renames a canonical generation or changes `latest.json`.

Successful preparation returns the same bounded JSON receipt to every worker. The receipt binds the run, attempt, controller token and command sequence to a managed path, manifest digest, identity digest and complete rank-file inventory. **Possessing a receipt does not prove that every supervised worker completed its command.** The parent must accept the whole-group command result and check current worker health before calling commit.

The parent can import `CheckpointCommitAuthority` from `hypergan.distributed_commit` without torch, NumPy or ParticleGAN:

```python
from hypergan.distributed_commit import CheckpointCommitAuthority
from hypergan.run_state import run_lock

with run_lock(run_dir):
    with CheckpointCommitAuthority(
        run_dir, run_id=run_id, attempt_id=attempt_id,
        identity=expected_checkpoint_identity,
    ) as authority:
        # Send only authority.controller_id and the parent-issued sequence to
        # the workers. Obtain expected_checkpoint_identity from the agreed
        # distributed_checkpoint_identity(trainer), not from an unchecked receipt.
        # ... supervise one all-rank prepare command and check worker health ...
        path = authority.commit(receipt, expected_command_sequence=command_sequence)
```

The authority captures its creating PID, a fresh controller token and the expected run/attempt/identity. It rejects cross-process use, closed authorities, stale controller/attempt fences and repeated or older command sequences. It does **not** acquire or verify the OS lock: retaining the caller's run lock and checking supervised command success are explicit caller obligations. This is a cooperative protocol, not a sandbox against arbitrary trusted Python writing directly to the filesystem. The legacy rank-zero save API remains available for its standalone caller-owned lifecycle; it must not be used as a new parent-service worker command.

Commit checks ordinary managed directories, bounded ordinary files, the exact manifest and rank inventory, identities, sizes and hashes. It never loads tensor payloads: their numerical validation happens during all-rank preparation and again during restore. Once full validation passes, the authority consumes the sequence **before** renaming. A rename or pointer error cannot retry the same receipt; use a new supervised command or attempt. Invalid receipts rejected before publication do not consume the sequence.

Both source and destination parent directories are synced around the generation rename. If publication fails before pointer replacement, the prior pointer stays selected; a complete but unselected generation may remain. Failure after pointer replacement, including a directory-sync error, can leave the new valid pointer visible despite the reported failure. A later worker disconnect can likewise follow a valid commit. Treat these as uncertain operation outcomes and revalidate durable state, not as proof that nothing was written.

## What a complete checkpoint means

The format uses `distributed-checkpoints/<attempt>-step-<step>-<unique>/` with one tensor payload per rank, a versioned manifest, and an atomic `latest.json` pointer. It is distinct from single-process training checkpoints and inference bundles.

Each rank snapshot includes models, learned prior, both optimizers and their base rates, EMA, modes/trainability, persistent and nonpersistent buffers, registered extra state, named RNG streams, Torch/Python/NumPy global RNG, local sampler state, step and last local batch. Replicated numerical state must agree; rank-specific RNG and data ownership are retained separately. The automatic data path currently advances the same global sampler on every rank before slicing, preserving one image permutation/cursor order while repeating decoding.

Compatibility checks include the explicit HyperGAN checkpoint compatibility version, full recipe and schedule, numerical runtime, external component/dependency source, data/content/preprocessing/class-map identity, strategy, global/local batch and fixed world size. HyperGAN source hashes, release SHA and package version are recorded as provenance and may differ from an earlier checkpoint. Ranks participating in one job must still agree on the complete current implementation identity; mixed worker builds are rejected. Version 1 supports existing schema-1 checkpoints without an explicit compatibility field; known incompatible changes require a version bump. See [recovery and release provenance](recovery.md). Topology changes, new recipes and transfer initialization are not resume. Recorded identities cannot infer hidden transitive dependencies or external state.

Saving requires a complete boundary on every rank. Rank zero receives all payloads, verifies them against the agreed replicated state, writes and syncs a prepared generation, and waits for the final readiness agreement. The parent path publishes only after successful supervised completion; the compatibility save wrapper performs preparation followed by the same byte/identity validation and publication on rank zero. A missing rank, snapshot error, divergent state or staging failure cannot publish an incomplete checkpoint. Failed half-updates remain unavailable for saving.

Filesystem publication and continued worker liveness cannot be one atomic operation. If a worker disconnects after the complete commit, the job may report failure while retaining a valid checkpoint. Likewise, a completed generation can remain unreferenced if publishing the latest pointer fails. Recovery validates the selected generation; it never assumes that the last observed step was durable.

## Restore, failure and limits

Rank zero validates every rank file and digest before distributing states. Every worker validates loading on an isolated trainer copy before mutating live state. Imported Python modules remain shared dependencies in that copy; mutable owned state is copied. Canonical expected state is protected from custom load-hook mutation. A live load failure poisons the affected worker group; stop the job and construct fresh workers rather than retrying on partly mutated trainers. RNG consumed by metadata hooks is isolated from the saved and continuing numerical streams.

Payloads are limited to 256 MiB per rank, metadata to 16 MiB and receipts to 64 KiB, with at most 64 fixed ranks. This correctness implementation uses trusted-worker object collectives and holds all rank payloads on rank zero. It duplicates replicated state and validation copies; it is not a scalable checkpoint backend or a disk quota. Immutable generations are retained until the caller deliberately retires them. A hard-killed writer may leave managed `.prepared` staging (or historical `.pending-` directories); neither is a valid default or explicit completed-generation restore target. Cleanup must stay inside the abandoned attempt's managed staging namespace.

This module supplies complete snapshots and fixed-topology recovery. The internal [replicated run service](replicated-run-service.md) connects its parent publication path to persisted job status, attempt/sample counters, periodic saves and checkpoint requests through the shared controller. Standalone callers remain responsible for those policies. Optional lineage fields record `next_sample_sequence` and checkpoint request IDs. Remote artifact transport, elastic workers and cloud retries remain separate work.

Tests compare uninterrupted and fresh-group resumed state exactly, including a shuffled image-folder fixture with labels and stochastic components. Deliberate failures exercise missing workers, half-updates, rank snapshot and disk errors, divergent optimizer state, corrupt/incomplete files, incompatible identities, and custom load hooks. The [checkpoint report](../reports/core-distributed-2026-09-18.md) records installed-package and CI evidence. GPU/NCCL and real multi-node recovery remain unqualified.

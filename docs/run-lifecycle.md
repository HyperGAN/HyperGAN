# Shared training lifecycle

The public `hypergan train` and `hypergan resume` commands use one internal run controller. The controller owns the run lock, attempts, manifests, events, stop budgets, checkpoint requests, sample reservations and terminal status. A single-process execution adapter owns the numerical trainer and its last completed batch.

The internal [replicated run service](replicated-run-service.md) now uses that controller with persistent CPU workers and parent checkpoint publication. The public commands still run the CPU reference in one process; bounded distributed observation and public integration remain ahead.

## Ownership and completion

The controller receives completed steps and metrics from the execution adapter. It makes checkpoint, preview and stop decisions only between complete logical updates. A failed update can leave numerical state partially changed; failure handling does not save that state.

The single-process adapter performs construction and strict recovery, numerical updates, checkpoint serialization, preview rendering and inference export. It also isolates callback RNG and restores the caller's CPU thread setting. The replicated adapter keeps numerical state and thread settings in its supervised workers. The controller does not access model tensors, optimizers or batches and can be imported without the optional training dependencies.

Successful adapter shutdown precedes terminal success. Failure cleanup occurs before the terminal failure event while the controller still owns the run lock. Required persistence or shutdown failures fail the attempt; an observer failure remains isolated only where the existing observation contract allows it.

The public signatures, event and artifact formats, checkpoint intervals, request receipts and preview counters are unchanged. See [recovery](recovery.md) and [observation](observation.md) for the supported user workflow.

## Source compatibility

Full recovery compares implementation source hashes, including the extracted controller and execution adapter. Checkpoints from before this source change cannot resume under this implementation. This applies to both native single-process checkpoints and the internal distributed format, which shares implementation identity. Use the original installation for those runs; this refactor does not provide checkpoint migration or weaken strict validation.

Behavioral parity is measured by independently running the old and new implementations and comparing complete numerical state and lifecycle results. It does not imply cross-version checkpoint compatibility.

## Internal replicated integration

The [CPU execution profile and preflight](execution-profiles.md), [persistent worker command service](cpu-worker-service.md) and [parent checkpoint publication](distributed-recovery.md) are implemented as internal prerequisites. The command broker monitors and reaps numerical workers independently of the coordinator. A parent commit authority validates staged state after successful all-rank command completion and a fresh health check; the caller still owns the run lock.

The controller creates an immutable candidate attempt identity in memory before strict restore and persists it only after restore succeeds. Numerical execution identity is recorded separately from mutable service deadlines. Native manifests without an explicit profile retain their native route; changing to a different execution strategy cannot bypass strict checkpoint validation.

The replicated adapter validates agreed complete steps and global metrics, prepares checkpoints in workers, and publishes through the parent authority. Group and publication failures remain fatal during optional saves. It reuses filesystem events, checkpoint receipts and sample reservations; final artifacts when a completed/restored batch is available and worker shutdown precede terminal success. Replicated preview and callback options are explicitly rejected until bounded execution is implemented.

See the [shared-service design](../reports/distributed-run-service-design-2026-09-19.md) and [current checkpoint](../reports/core-replicated-service-2026-09-19.md). GPU/NCCL and actual multi-node execution remain separate qualifications.

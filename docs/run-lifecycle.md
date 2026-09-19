# Shared training lifecycle

The public `hypergan train` and `hypergan resume` commands use one internal run controller. The controller owns the run lock, attempts, manifests, events, stop budgets, checkpoint requests, sample reservations and terminal status. A single-process execution adapter owns the numerical trainer and its last completed batch.

This separation prepares the same lifecycle for distributed execution. The public commands still run the CPU reference in one process. Internal replicated training and recovery remain available through their [developer APIs](distributed.md).

## Ownership and completion

The controller receives completed steps and metrics from the execution adapter. It makes checkpoint, preview and stop decisions only between complete logical updates. A failed update can leave numerical state partially changed; failure handling does not save that state.

The adapter performs construction and strict recovery, numerical updates, checkpoint serialization, preview rendering and inference export. It also isolates callback RNG and restores the caller's CPU thread setting. The controller does not access model tensors, optimizers or batches and can be imported without the optional training dependencies.

Successful adapter shutdown precedes terminal success. Failure cleanup occurs before the terminal failure event while the controller still owns the run lock. Required persistence or shutdown failures fail the attempt; an observer failure remains isolated only where the existing observation contract allows it.

The public signatures, event and artifact formats, checkpoint intervals, request receipts and preview counters are unchanged. See [recovery](recovery.md) and [observation](observation.md) for the supported user workflow.

## Source compatibility

Full recovery compares implementation source hashes, including the extracted controller and execution adapter. Checkpoints from before this source change cannot resume under this implementation. This applies to both native single-process checkpoints and the internal distributed format, which shares implementation identity. Use the original installation for those runs; this refactor does not provide checkpoint migration or weaken strict validation.

Behavioral parity is measured by independently running the old and new implementations and comparing complete numerical state and lifecycle results. It does not imply cross-version checkpoint compatibility.

## Distributed integration still ahead

The [CPU execution profile and preflight](execution-profiles.md), [persistent worker command service](cpu-worker-service.md) and [parent checkpoint publication](distributed-recovery.md) are implemented as internal prerequisites. The command broker monitors and reaps numerical workers independently of the coordinator. A parent commit authority validates staged state after successful all-rank command completion and a fresh health check; the caller still owns the run lock.

The replicated adapter is the next integration step. It must receive a fenced worker-session identity during strict restore, which currently happens before a new attempt is persisted, then bind the reserved attempt without accepting stale results. Persist numerical execution identity separately from mutable attempt policy. Failed collective groups must remain fatal even during optional saves. Bounded previews, progress delivery, final artifacts and whole-job lifecycle acceptance remain gates before public distributed train/resume.

See the [shared-service design](../reports/distributed-run-service-design-2026-09-19.md) and [current checkpoint](../reports/core-worker-service-2026-09-19.md). GPU/NCCL and actual multi-node execution remain separate qualifications.

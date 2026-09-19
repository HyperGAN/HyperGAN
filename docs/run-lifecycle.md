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

The [CPU execution profile and preflight](execution-profiles.md) are implemented. The next work connects supervised worker commands and distributed train/resume. Workers must wait outside training collectives between operations. The controller must hold sole canonical checkpoint commit authority, and abrupt parent death must be tested for worker cleanup and safe takeover before exposing a distributed CLI.

The current adapter treats an ordinary manual-save serialization failure as an observer error. A replicated adapter must distinguish that rejection from a failed collective or poisoned worker group, which must fail the entire attempt. Runtime/RNG descriptions must also come from the resolved execution profile before multi-process use.

Preview rendering and Python observers are still synchronous in the single-process adapter. Copying model state and isolating RNG do not impose a deadline on custom Python. A bounded snapshot renderer and supervised progress delivery remain gates for distributed integration. `max_seconds` remains a cooperative stop budget.

The [integration design](../reports/distributed-run-service-design-2026-09-19.md) records the remaining sequence. GPU/NCCL, real clusters, the optional browser server and deployment require their own acceptance.

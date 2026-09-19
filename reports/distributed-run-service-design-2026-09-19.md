# Shared CPU run service: integration design

Date: 2026-09-19. **Proposal, not implemented behavior.** Reviewed against develop `96222fc6` after PR #309. The coordinator owns the status ledger and implementation sequencing. This report does not qualify GPU, NCCL, multi-node execution, image quality or a release.

## Starting point

The public [training service](../src/hypergan/training.py) already owns attempts, stop budgets, events, required checkpoints, optional previews, manual-save receipts and final inference artifacts. Its `_execute` function couples those policies to `ReferenceTrainer` and the single-process checkpoint format. Preserve that behavior while separating lifecycle decisions from numerical execution.

The internal [replicated trainer](../src/hypergan/distributed_training.py) owns complete CPU Gloo updates, global data draws followed by rank slicing, replica agreement and readiness/poison state. The [checkpoint helper](../src/hypergan/distributed_checkpoints.py) preserves every rank's state, requires all-rank readiness before rank-zero publication and restores fixed topology strictly. The [supervisor](../src/hypergan/cpu_workers.py) launches and reaps a finite group, but currently offers only a blocking callback interface. It explicitly does not handle abrupt parent death. None of these internal APIs supplies the public distributed run lifecycle.

The current [observation contract](../docs/observation.md) and [preview implementation](../src/hypergan/previews.py) are reusable. Preview rendering copies numerical modules and isolates RNG, but runs synchronously: it does not bound custom-forward execution time or make collective-dependent models safe to render in one rank.

## Small shared controller, two execution adapters

Extract one controller from `_execute`; do not introduce a second distributed training loop with copied attempt, receipt and preview policies. Keep its interface internal until both adapters pass the same lifecycle tests.

| Responsibility | Owner |
| --- | --- |
| Run lock; run/attempt IDs; manifests; event sequence; durable/observed counters; request receipts; sample reservations | Parent controller, sole writer |
| D/G/prior/auxiliary/optimizer/EMA operations; RNG and sampler state; complete logical update | Execution adapter and numerical trainers |
| Replicated readiness and state validation; rank payload preparation | Every worker, existing distributed checkpoint implementation |
| Canonical checkpoint publication and latest selection | Parent commit authority after validated all-rank preparation |
| Numerical worker launch, bounded command delivery, monitoring, termination and reaping | Local CPU supervisor |
| Preview computation from an immutable snapshot | Isolated renderer with a deadline; controller publishes artifacts/index |
| Public progress output and observer delivery | Controller, outside numerical collectives |

The internal adapter needs operations equivalent to `start`, `restore`, `update`, `prepare_checkpoint`, `preview_snapshot`, `inference_snapshot`, and `shutdown`. These are operation names, not a commitment to a general scheduler framework. Use small typed results: completed step/global metrics, validated snapshot references and bounded rank diagnostics. Keep trainer objects, optimizer state and local batches inside the execution side.

The single-process adapter can initially execute in the controller process. Its policy behavior must remain unchanged; RNG isolation still applies to callbacks. The replicated adapter owns one fixed worker group for an attempt. A worker failure ends that group; retries require a fresh attempt and explicit checkpoint selection. No automatic restart or elastic world-size change in this slice.

## Profile and preflight

Keep execution profile separate from recipe architecture/objectives. The first public distributed profile is CPU, Gloo, fixed world size, replicated state and explicit gradient averaging. Record resolved global batch size, world size, local batch size, accumulation count, microbatch size and accumulation algorithm. Accumulation counts complete logical updates, not independently optimized microbatches. Never adjust the recipe's batch size or learning rate silently.

Use two preflight stages:

1. Lightweight structural validation, available without importing torch: profile names/types, positive limits, batch divisibility, incompatible options and checkpoint kind.
2. Runtime preflight in the supervised workers: numerical dependencies, component construction, supported CPU/layout/buffer behavior, source/runtime/data identity, replica agreement and recovery capability. Return actionable rank-tagged errors before an update starts. Custom factory execution is trusted Python and is subject to the startup deadline.

Persist resolved numerical execution identity separately from mutable attempt policy such as checkpoint/preview frequency and cooperative stop budgets. Fixed-topology restore compares the former strictly. Distinguish single-process native checkpoints, distributed generations and inference bundles explicitly; no implicit conversion. Existing manifests lacking a profile can mean the historical single-process profile, but that compatibility rule must not bypass checkpoint source/runtime validation. Moving source during extraction may invalidate strict historical checkpoint identity; document that honestly instead of disabling source checks.

## Command boundaries and failure ordering

Workers wait for controller commands **outside Gloo collectives**. Give each rank a bounded parent control channel. Every command carries run ID, attempt ID, monotonically increasing command sequence and operation. Deliver the same operation to all ranks; perform operation/identity agreement before numerical work. Missing delivery, stale identity or sequence disagreement fails the group under a finite deadline.

Do not make nonzero ranks block in a Gloo broadcast while rank zero renders a preview, publishes files or waits for a callback: slow observation would consume the collective timeout. Distinguish idle worker liveness deadlines from collective-operation deadlines and total job deadline. Parent-side progress delivery also needs bounded execution; a stuck observer must not prevent monitoring/reaping workers. Arbitrary recipe stdout must not be parsed as the structured control protocol.

At each boundary:

1. Accept completion only after every rank acknowledges the same logical step and the trainer is checkpoint-ready.
2. Persist the observed step and one global training event. Rank diagnostic messages do not increment logical progress.
3. Evaluate stop budgets and checkpoint/manual-save/preview policy once in the controller.
4. Dispatch required work in one agreed order. A checkpoint operation involves every rank. Optional preview work uses a captured snapshot, never a live trainer mutation.
5. Dispatch the next update only after required persistence succeeds and all workers remain healthy.

An update exception, rank loss or failed collective poisons the attempt. Terminate and reap the group; never checkpoint a half D/G update or continue a partially restored trainer. A manual save's ordinary serialization/filesystem rejection may be observer-only **only if** every rank receives a coherent rejection and the group is still usable. Collective failure during that optional operation is a fatal numerical-group failure.

`max_seconds` remains a cooperative stop evaluated at complete update boundaries. A hard job deadline includes startup, loading and I/O and may kill an incomplete update. Neither timeout promises a checkpoint of that update. Store observed step, last durable step and possible lost updates independently.

## Checkpoint commit authority and parent death: first integration gate

The existing helper assumes a caller-owned lifecycle lock and publishes from rank zero. Simply moving that lock into a supervising parent is insufficient: after an abrupt parent death, an orphan rank zero could still publish `latest.json` after a new controller acquires the released lock.

Before exposing the distributed CLI, separate the existing checkpoint implementation's preparation from canonical publication. Reuse its serializer, strict identities, payload hashes and all-rank readiness checks. Rank zero may prepare a generation in its attempt namespace, but only the parent holding the run lock may rename/publish the accepted generation and update the canonical latest pointer. A new attempt never accepts an old attempt's completion message. Workers must have no alternative path that changes canonical pointers, manifests or receipts.

The controller commits only after the all-rank readiness phase has succeeded and the supervisor has not reported a pre-commit worker failure. A death after the commit can still yield a failed job with a valid complete checkpoint; retain the existing distinction. Unreferenced prepared artifacts are not checkpoints selected for resume and may be cleaned only within their managed namespace.

Add parent-liveness detection and bounded worker cleanup, including while a worker is in a long operation. EOF observed only at the next idle boundary is insufficient. Test real parent termination and takeover: old workers cannot commit, all managed children are eventually reaped, and a new attempt can safely resume. Do not claim safe takeover merely because `run_lock` became available. This is a gate for the first public distributed integration, not a later hardening task. The design need not promise survival of host failure or management of arbitrary subprocesses spawned by user components.

## Resume, requests, previews and terminal status

Resume starts fresh workers and restores each rank's own named/global RNG, sampler/order state and local last batch; never broadcast rank zero's RNG as a shortcut. Select and validate one generation consistently. Preserve the original total schedule. Reserve a new attempt and run-wide sample sequence above existing reservations. Replaying an older generation must republish the selected state in the new attempt before a zero-update stop so default resume follows that selection without overwriting older artifacts.

Reuse the filesystem [request protocol](../src/hypergan/run_requests.py) and [event cursor](../src/hypergan/run_events.py) formats. Read requests once in the controller, with existing queue bounds. Reject stale run/attempt requests explicitly. Coalesce matching checkpoint requests at a completed boundary, include their IDs in checkpoint metadata, and acknowledge only after canonical durable publication. Reconcile a lost acknowledgement using committed metadata. Preserve immutable receipts and the documented at-least-once behavior before acknowledgement; pending does not mean execution. Do not add cancellation or server endpoints in this slice.

Reserve preview sequence before dispatch. Capture bounded conditioning and EMA/prior state only from a completed step. Render with fixed isolated RNG in a process without a training process group, with a deadline and bounded queue; permit at most one pending render or document deliberate coalescing. A custom sampler requiring collectives must fail as an observer error, not hang numerical workers. Share the existing retention/index/publication implementation through a snapshot-based entry point. Observer failure must not change numerical state, rewind counters or remove recovery/final inference artifacts. Numeric previews remain acceptable; image-grid quality is a separate feature.

On completion or cooperative stop, require the final checkpoint and inference artifacts, worker shutdown agreement and successful group exit before publishing terminal success. Rank zero finishing first does not complete a job. On failure, first stop/reap the group, then finalize bounded diagnostics and status while retaining the lock. Preserve failing rank, operation, attempt and traceback before temporary supervisor files disappear. An observer or reader sees one whole-job status plus rank health diagnostics, not competing per-rank manifests. A browser can consume this same protocol later; no browser/server or cloud dependency is needed here.

## Accumulation review contract

The in-progress accumulation design uses detached full-batch logits to compute nonlinear GAN derivatives, then replays microbatch vector-Jacobian products. It retains a complete latent graph and applies the global prior regularizer once. Its strategy identity must include accumulation count, microbatch size and algorithm revision; a changed accumulation strategy is a strict resume mismatch unless separately qualified.

Checkpoint/control/preview boundaries remain outside the complete logical update. Replay must restore each microbatch's RNG entry state temporarily and preserve one canonical post-discovery continuation for Torch, Python, NumPy and every named stream. Data sampling, lazy penalty timing, learning-rate schedule, optimizer steps and EMA advance once per logical update. A failure after the D optimizer or during any G/prior replay leaves readiness false and poisons the group.

Custom modules remain configurable but unqualified unless their forward/backward and mutable-state behavior meet the replay contract. Registered state and replay-output checks provide diagnostics, not a proof for arbitrary Python or unregistered state. In particular, regenerating the fake under a different grad mode from discovery must either produce the same value or be rejected explicitly. These review requirements supplement numerical parity and fresh-group recovery tests; this report does not mark the evolving accumulation implementation complete.

## Reviewable implementation sequence and acceptance

| PR | Bounded change | Required evidence |
| --- | --- | --- |
| 1 | Extract lifecycle/controller and single-process adapter; preserve current public behavior | Existing installed-wheel single-process recovery/observation suite; same attempts, stop policy, receipts, preview counters and observer-on/off numerical state |
| 2 | Explicit CPU profile, resolved identity and structural/runtime preflight | Base-only import checks; invalid profile/divisibility/options; fresh-worker construction failures; strict profile/runtime/source/data mismatch diagnostics |
| 3 | Supervised all-rank command channels, replicated adapter, checkpoint prepare/commit authority and parent-death handling | Complete train/resume/zero-update replay; command disagreement/missing delivery; rank crashes/timeouts; abrupt parent death and safe takeover; no stale canonical commit |
| 4 | Shared events, checkpoint receipts, isolated bounded previews and final artifacts | Reconnect across attempts/partial tails; live save coalescing/stale IDs/lost acknowledgements; slow/failing observer; hung preview; exact observer-on/off full checkpoint state |
| 5 | Whole-job installed-package acceptance and user documentation | Fresh spawned groups with shuffled image-folder fixtures; accumulation greater than one; full-state uninterrupted/resumed equality within fixed strategy; complete/stopped/interrupted/failed status and group reaping |

Fault fixtures must cover rank failure after D but before G; after payload gather but before checkpoint readiness; after a complete commit but before result delivery; disk-full staging/publication; malformed/corrupt state; partial live restore; interrupted coordinator and rank-zero premature exit. Assert prior latest remains selected when no complete new commit exists, and valid post-commit generations remain recoverable when a later job failure occurs. Required persistence failures are fatal; optional failures are isolated only while execution remains healthy.

The shared lifecycle is the next CPU usability gate, not completion of distributed issue #186. Actual NCCL/two-GPU and real multi-node qualification, license resolution, selected image-experiment admission, dataset acquisition, deployment and the optional viewer remain the separate gates recorded in the [status ledger](resurrection-status.md) and [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md).

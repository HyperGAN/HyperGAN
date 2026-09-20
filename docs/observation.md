# Observe a run and request a checkpoint

Event reads and checkpoint requests work from a base installation without importing PyTorch or loading model weights. Native CUDA and local replicated runs share these records. Periodic previews are opt-in; the [optional local viewer](local-web.md) can start with training or reconnect independently.

```sh
hypergan train demo --run-dir runs/demo --preview-every 10 --preview-keep 3
hypergan events runs/demo --limit 20
hypergan checkpoint runs/demo --request-id my-save
hypergan checkpoint runs/demo --status my-save
```

Run the observation and checkpoint commands in another terminal while training is active. The default demo only runs five updates; choose a longer total schedule for a live walkthrough. `--steps` sets that schedule when creating a run.

## Reconnecting to events

`events` returns a JSON page with `events`, an opaque `cursor`, `has_more`, and `partial_tail`. Start without a cursor to read from the beginning, then pass the returned value with `--cursor` for the next page. Persist the cursor only after consuming the page; retrying the same cursor can deliver the same events again. Identify events by run, attempt and sequence. The reader returns only complete JSONL records and does not treat an unfinished trailing write as a completed update.

`--limit` accepts 1–10,000 rows and `--max-bytes` accepts 1–16,777,216 bytes; the reader also checks a fixed 256-byte cursor boundary anchor. A record larger than the selected byte budget requires a larger budget. No command watches indefinitely or loads the entire log. An invalid or stale cursor produces an actionable error; reconnect deliberately from the beginning and deduplicate previously consumed events. An append or normal resume preserves existing complete records. The cursor binds to the local directory and file generation and checks the consumed boundary. It detects replacement, truncation below its offset and boundary edits; it is not a checksum of the whole history or a portable remote cursor.

The root manifest remains the source of current status and durable progress. An event showing update N does not prove that update N can be restored. Inspect `last_durable_step` and `checkpoint_path`. Absence of new events does not establish completion or worker health.

## Safe checkpoint requests

`checkpoint` submits an atomic request addressed to a specific run and attempt. The default attempt comes from the current manifest; `--attempt-id` lets a caller pin it explicitly. Use a stable `--request-id` for retries. Reusing an ID with different target identity is an error. After resume changes the current attempt, use `--attempt-id ORIGINAL` to retry that original request, or `--status ID` to read its receipt. Without an ID, the command creates one and returns it in the receipt.

Submission returns `pending`; it does **not** mean that a checkpoint was saved. `checkpoint --status ID` reads the receipt without submitting another request. The trainer serializes requests at complete D/G/EMA update boundaries. A `succeeded` receipt identifies the durable checkpoint and update. Unsupported recovery and requests for another attempt are rejected explicitly. Commands never mutate live tensors from the caller's process. A durable acknowledgement is immutable. Before acknowledgement, recovery may repeat a save; this is not an exactly-once distributed transaction.

The queue allows at most 256 pending requests and processes at most 32 per boundary. Individual protocol records are capped at 8 KiB; completed receipts are retained for future idempotent retries. Queue-lock contention reports a retryable error to producers and defers trainer polling.

This is a local filesystem protocol for a trusted run directory, not a remote authenticated service. A request racing the last polling boundary, or sent to a process that has died, can remain pending. It is not automatically retried against a new attempt: a later attempt rejects stale requests. Resume the run deliberately and submit a new request for that attempt if needed. Polling a receipt has no timeout or execution guarantee; supervisors must also inspect attempt status and liveness. Cancellation and browser controls are still planned.

## Periodic previews

`--preview-every N` samples an isolated EMA snapshot after every N complete updates. `--preview-keep N` (1–100) bounds the retained periodic history across attempts. Resume inherits those settings; `--no-previews` explicitly disables periodic previews. These controls are operational settings and do not change the numerical recipe or total schedule.

Previews are immutable numeric JSON artifacts, with run, attempt, update and monotonic sample identities. They are not image grids or training checkpoints. Their run-wide counter never rewinds when an older checkpoint is replayed; gaps are allowed after failed publication. Retention runs after successful publication and applies only to periodic preview artifacts. Cleanup errors are visible and may leave extra files until a later successful cleanup; the retention setting is not a disk quota. Complete checkpoints and final attempt inference bundles remain separate.

Sampling uses a copied EMA graph and prior, copied conditioning inputs, evaluation mode and isolated random state. The supported native and replicated execution contracts protect model buffers, optimizer state and data/prior/penalty RNG streams. Tests exercise stochastic modules and nonpersistent buffers. Arbitrary side effects in trusted custom Python components remain the author's responsibility.

Previews cap sample count at 16 and combined output/conditioning tensors at 65,536 elements, with a 2 MiB serialized artifact limit. Large samples can reduce the effective count or fail preview publication. Rendering and preview-artifact errors are reported separately and do not stop optimization while mandatory run manifest/event writes remain available. Failure of those core writes can still fail the run. Forward execution and model copying still cost time and memory; these bounds are not a sandbox or a hard latency deadline for custom code.

The [local viewer](local-web.md) serves these records through the public API and browser. Use `--no-server` for headless training. [Execution profiles](execution.md) select native CUDA or supervised local CUDA/NCCL and explicit CPU fixtures; observation does not change their numerical identity.

### Bounded training command output

Public `train` and `resume` use the bounded output transport below for native and
replicated execution.

Training CLI output is best effort: unread, slow or closed stdout/stderr does not
hold training or terminal process cleanup. Independent drain processes preserve
Python and native diagnostics with a healthy consumer. Each stream has at most
16 queued lines per stage (parent and drain), capped at 64 KiB per line; full
queues drop oldest lines and oversized progress/diagnostic lines are omitted.
Shutdown allows one second for delivery before killing and reaping blocked
drains. Reconnect through `hypergan events RUN` or `hypergan inspect RUN` for the
durable result and complete event history.

Routine CLI training progress prints every **100 updates** by default. Set
`--progress-every N` on `train` or `resume`, or use **CLI progress every N steps**
in the browser. The setting is saved as `console.json` inside the run directory
and survives resume; an explicit CLI flag replaces the saved value. Active CLI
processes check at complete update boundaries, at most four times per second.
Long updates delay when a change takes effect. This controls console reporting only: collected metric events,
checkpoint cadence, previews and numerical configuration remain independent.
Lifecycle events, errors and final status bypass this interval. The same cadence
applies to `--progress-json`; use `--progress-every 1` for every console update.

A normal command result is compact single-line JSON. With `--progress-json`, the
final envelope remains `{"event":"result","manifest":...}`. Results over 64 KiB
produce a small `output_omitted` record with reason `result_exceeds_output_limit`
and a durable manifest location, plus a warning. Backpressure can drop even the
final record; the run manifest remains authoritative. This transport changes no
checkpoint or numerical completion semantics.

Cached Python standard streams and the owning C runtime's stdout/stderr are
flushed through the active drains before descriptor restoration, so buffered
stdio cannot hold interpreter exit against a full destination. This covers libc
on Linux/macOS and UCRT on Windows. Arbitrary custom streams, separate CRTs,
separately cached OS handles and externally held native stream locks remain
outside the descriptor transport. On POSIX, drain children use separate sessions
so a terminal process-group SIGINT/SIGTERM reaches the training coordinator while
the drains remain available for its final status. Parent-death monitoring still
bounds cleanup if the coordinator is killed.

Live event and status persistence runs on a bounded background I/O worker. An
ordinary update queues an event without waiting for disk writes. The event queue
holds at most 256 pending rows and 8 MiB of conservative JSON size estimates, plus
one active row (each row is limited to 1 MiB and bounded nesting/node count).
When storage cannot keep up, optional observation events are omitted rather than
stalling updates. The next accepted event carries `observation_gap` with the omitted
counts by event kind and first/last step, and the run manifest records cumulative
`dropped_observation_events` and `dropped_train_events`. Accepted events retain contiguous sequence numbers;
missing measurements are never fabricated or averaged. Checkpoint boundaries
record any outstanding gap before committing the event prefix.

Live manifests coalesce to the newest snapshot and are scheduled at most four
times per second. Checkpoint requests are polled at most four times per second,
plus startup, periodic checkpoint and terminal boundaries. A slow update can
extend request latency until its next safe boundary. Initial status, checkpoint
publication, artifact identity reservation and final status still wait for
durable metadata; checkpoint commits drain all accepted events before fsync.
These explicit recovery boundaries can wait for storage. Background storage
failures fail the run visibly rather than publishing an invalid durable frontier.

# Observe a run and request a checkpoint

Event reads and checkpoint requests work from a base installation without importing PyTorch or loading model weights. Native CUDA and local replicated runs share these records. Periodic previews are opt-in; the [optional local viewer](local-web.md) can start with training or reconnect independently.

```sh
hypergan train demo --run-dir runs/demo --preview-every 10
hypergan events runs/demo --limit 20
hypergan checkpoint runs/demo --request-id my-save
hypergan checkpoint runs/demo --status my-save
```

Run the observation and checkpoint commands in another terminal while training is active. The default demo only runs five updates; choose a longer total schedule for a live walkthrough. `--steps` sets that schedule when creating a run.

## Reconnecting to events

`events` returns a JSON page with `events`, an opaque `cursor`, `has_more`, and `partial_tail`. Start without a cursor to read from the beginning, then pass the returned value with `--cursor` for the next page. Persist the cursor only after consuming the page; retrying the same cursor can deliver the same events again. Identify events by run, attempt and sequence. The reader returns only complete JSONL records and does not treat an unfinished trailing write as a completed update.

`--limit` accepts 1–10,000 rows and `--max-bytes` accepts 1–16,777,216 bytes; the reader also checks a fixed 256-byte cursor boundary anchor. A record larger than the selected byte budget requires a larger budget. No command watches indefinitely or loads the entire log. An invalid or stale cursor produces an actionable error; reconnect deliberately from the beginning and deduplicate previously consumed events. An append or normal resume preserves existing complete records. The cursor binds to the local directory and file generation and checks the consumed boundary. It detects replacement, truncation below its offset and boundary edits; it is not a checksum of the whole history or a portable remote cursor.

The root manifest remains the source of current status and durable progress. An event showing update N does not prove that update N can be restored. Inspect `last_durable_step` and `checkpoint_path`. Absence of new events does not establish completion or worker health.

## Training progress metrics

Every complete update publishes three built-in progress scalars alongside the
losses, in the run manifest and on the `train` event:

| Metric ID | Manifest field | Meaning |
| --- | --- | --- |
| `throughput/steps_per_second` | `steps_per_second` | Completed updates per second, averaged over the last 20 updates. |
| `timing/training_seconds` | `training_seconds` | Cumulative wall clock spent inside training attempts. |
| `progress/samples_seen` | `samples_seen` | Real examples drawn from the data stream. |

Samples seen is `completed updates × training.batch_size`: one completed update
consumes exactly one global batch, which gradient accumulation and a world size
larger than one split into microbatches and shards but never change.

Throughput is a trailing average so it charts cleanly; the window is
attempt-local and restarts on resume, and an update boundary too fast for the
clock to separate publishes no rate rather than an infinite one. Cumulative
training time continues across attempts — a resumed run adds its own attempt's
wall clock to the stored total — and time between attempts is never counted.
Both are observations of this machine, not part of the numerical recipe: they do
not change the configuration fingerprint, and a replayed run reproduces the same
losses with different durations. They are selectable in the browser's learning
curves like any loss, and `metrics.disable` removes any of them.

## Safe checkpoint requests

`checkpoint` submits an atomic request addressed to a specific run and attempt. The default attempt comes from the current manifest; `--attempt-id` lets a caller pin it explicitly. Use a stable `--request-id` for retries. Reusing an ID with different target identity is an error. After resume changes the current attempt, use `--attempt-id ORIGINAL` to retry that original request, or `--status ID` to read its receipt. Without an ID, the command creates one and returns it in the receipt.

Submission returns `pending`; it does **not** mean that a checkpoint was saved. `checkpoint --status ID` reads the receipt without submitting another request. The trainer serializes requests at complete D/G/EMA update boundaries. A `succeeded` receipt identifies the durable checkpoint and update. Unsupported recovery and requests for another attempt are rejected explicitly. Commands never mutate live tensors from the caller's process. A durable acknowledgement is immutable. Before acknowledgement, recovery may repeat a save; this is not an exactly-once distributed transaction.

The queue allows at most 256 pending requests and processes at most 32 per boundary. Individual protocol records are capped at 8 KiB; completed receipts are retained for future idempotent retries. Queue-lock contention reports a retryable error to producers and defers trainer polling.

This is a local filesystem protocol for a trusted run directory, not a remote authenticated service. A request racing the last polling boundary, or sent to a process that has died, can remain pending. It is not automatically retried against a new attempt: a later attempt rejects stale requests. Resume the run deliberately and submit a new request for that attempt if needed. Polling a receipt has no timeout or execution guarantee; supervisors must also inspect attempt status and liveness. Cancellation and browser controls are still planned.

## Periodic previews

`--preview-every N` requests an isolated EMA snapshot after every N complete updates when the preview worker is available. `--preview-keep N` bounds the retained periodic history across attempts; the default is 128. Retention thins rather than truncates: when a run outgrows the bound the spacing between the older samples doubles (every 500 steps becomes every 1,000, then every 2,000) in one chunk, the first sample of the run and the newest 16 sequences are never pruned, and the browser's per-sample history slider therefore still spans the whole run. `--preview-keep all` keeps every published preview for the life of the run. A resume inherits a stored bound only when it was requested explicitly; a bound a manifest recorded because it was the default at the time resumes into the current default. `--preview-name NAME` sets the short stable name indexing this run's generated samples (default `g`; the comparable real batch is published as `x`). Resume inherits those settings; `--no-previews` explicitly disables periodic previews. These controls are operational settings and do not change the numerical recipe or total schedule.

Previews are immutable numeric JSON artifacts with bounded PNG grids for image output, carrying run, attempt, update and monotonic sample identities plus their sample name. They are separate from training checkpoints. Their run-wide counter never rewinds when an older checkpoint is replayed; gaps are allowed after failed publication. Retention runs after successful publication and applies only to periodic preview artifacts. The index is rewritten before anything is removed and the expired generations are deleted by a background worker, so a prune never delays the next publication; cleanup errors are visible and may leave extra files until a later successful cleanup, and the retention setting is not a disk quota. Complete checkpoints and final attempt inference bundles remain separate.

Sampling uses a copied EMA graph and prior, copied conditioning inputs, evaluation mode and isolated random state. The supported native and replicated execution contracts protect model buffers, optimizer state and data/prior/penalty RNG streams. Tests exercise stochastic modules and nonpersistent buffers. Arbitrary side effects in trusted custom Python components remain the author's responsibility.

Periodic preview rendering and artifact publication run in an isolated CPU process
with no visible CUDA devices. Training captures one immutable snapshot at a
completed update boundary, then continues without waiting for rendering. At most
one snapshot is in flight; scheduled previews while it is busy are skipped before
copying state or reserving a sample sequence. `preview_skipped` events and
`skipped_previews_busy` in the run manifest expose these omissions. Completed
preview events retain the snapshot's original step. Terminal cleanup drains the
one outstanding job under the renderer deadline (60 seconds for native execution,
the configured `preview_timeout` for replicated execution). A signal or failed
training update cancels pending rendering and reaps its process; a publication
that already finished remains an immutable artifact. CPU worker thread counts
are limited to one and POSIX workers lower their scheduling priority. Snapshot copying and the durable sequence reservation still pause the training
boundary; replicated file handoff also serializes at that boundary. Asynchronous
rendering does not remove those capture costs.
Python scripts that enable periodic previews must use the usual
`if __name__ == '__main__':` guard for process creation and importable component
factories; the CLI already supplies that guard.

For native training, the update boundary freezes owned CPU tensors and plain
containers under the training RNG fence. Snapshot directory creation,
serialization, fsync and content hashing run on the preview supervisor thread;
that thread never reads a live trainer or executes custom serialization hooks.
Only one snapshot is retained, capped at 256 MiB of captured payload and 256 MiB
on disk. The same preview deadline covers persistence and CPU rendering, with
bounded cleanup grace. Cancellation is checked between storage operations;
uninterruptible OS writes cannot be force-cancelled inside a thread and surface
a cleanup failure if they outlive that grace. Device transfer, copying and
custom state hooks still run at the safe numerical boundary.

Replicated training retains its rank-zero snapshot file handoff: that capture
command copies, serializes, fsyncs and hashes the snapshot before ranks resume.
Rendering and final preview publication are asynchronous. Removing this remaining
transport barrier requires a separate rank/coordinator handoff protocol; native
training does not require it.

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
durable result and accepted event history, including recorded observation gaps.

A printed progress line names the update, the published losses and the progress
metrics above:

```
step 400: D=0.683355 G=0.805652 | 123.4 steps/s | 1h 23m | 12,800 samples
```

Fields that an update has not measured are omitted rather than guessed. With
`--progress-json`, the same values appear as the `steps_per_second`,
`training_seconds` and `samples_seen` keys of each `train` row.

Routine CLI training progress prints every **100 updates** by default. Set
`--progress-every N` on `train` or `resume`. The setting is saved as
`console.json` inside the run directory and survives resume; an explicit CLI
flag replaces the saved value. Each attempt resolves the cadence once, at its
first delivery; nothing changes it while the attempt runs. This controls console
reporting only: collected metric events,
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
`dropped_observation_events` and `dropped_train_events`. Accepted events retain
contiguous sequence numbers;
missing measurements are never fabricated or averaged. Checkpoint boundaries
record any outstanding gap before committing the event prefix.

Live manifests coalesce to the newest snapshot and are scheduled at most four
times per second. Checkpoint request discovery uses a single background
read slot, at most four reads per second. Ordinary updates only consume
cached results. Startup, periodic checkpoint and terminal control boundaries
may perform fresh synchronous scans; terminal processing retries one lost
acknowledgement against the saved request IDs without repeating the checkpoint.
A slow update can
extend request latency until its next safe boundary. Initial status, checkpoint
publication, artifact identity reservation and final status still wait for
durable metadata; checkpoint commits drain all accepted events before fsync.
These explicit recovery boundaries can wait for storage. Optional live read
workers are daemons and their cancellation does not wait for a stalled OS read;
they exit when that read returns. The terminal checkpoint-request scan waits for
its in-flight scan before checking the latest requests. Initial console policy
and CLI override persistence happen at the start/resume boundary. Background
storage failures fail the run visibly rather than publishing an invalid durable frontier.

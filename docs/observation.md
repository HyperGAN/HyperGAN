# Observe a CPU run and request a checkpoint

Run observation works from a base installation, without importing PyTorch or loading model weights. Periodic previews are opt-in while the browser server remains planned.

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

`checkpoint` submits an atomic request addressed to a specific run and attempt. The default attempt comes from the current manifest; `--attempt-id` lets a caller pin it explicitly. Use a stable `--request-id` for retries. Reusing an ID with different target identity is an error. Without an ID, the command creates one and returns it in the receipt.

Submission returns `pending`; it does **not** mean that a checkpoint was saved. `checkpoint --status ID` reads the receipt without submitting another request. The trainer serializes requests at complete D/G/EMA update boundaries. A `succeeded` receipt identifies the durable checkpoint and update. Unsupported recovery and requests for another attempt are rejected explicitly. Commands never mutate live tensors from the caller's process.

This is a local filesystem protocol for a trusted run directory, not a remote authenticated service. A request racing the last polling boundary, or sent to a process that has died, can remain pending. It is not automatically retried against a new attempt: a later attempt rejects stale requests. Resume the run deliberately and submit a new request for that attempt if needed. Polling a receipt has no timeout or execution guarantee; supervisors must also inspect attempt status and liveness. Cancellation and browser controls are still planned.

## Periodic previews

`--preview-every N` samples an isolated EMA snapshot after every N complete updates. `--preview-keep N` (1–100) bounds the retained periodic history across attempts. Resume inherits those settings; `--no-previews` explicitly disables periodic previews. These controls are operational settings and do not change the numerical recipe or total schedule.

Previews are immutable numeric JSON artifacts, with run, attempt, update and monotonic sample identities. They are not image grids or training checkpoints. Their run-wide counter never rewinds when an older checkpoint is replayed; gaps are allowed after failed publication. Retention applies only to periodic preview artifacts. Complete checkpoints and final attempt inference bundles remain separate.

Sampling uses a copied EMA graph and prior, copied conditioning inputs, evaluation mode and isolated random state. The supported CPU contract protects model buffers, optimizer state and data/prior/penalty RNG streams. Tests exercise stochastic modules and nonpersistent buffers. Arbitrary side effects in trusted custom Python components remain the author's responsibility.

Previews cap sample count at 16 and combined output/conditioning tensors at 65,536 elements, with a 2 MiB serialized artifact limit. Large samples can reduce the effective count or fail preview publication. A preview error is reported separately and does not stop optimization. Forward execution and model copying still cost time and memory; these bounds are not a sandbox or a hard latency deadline for custom code.

No HTTP server, browser startup or GPU execution is part of this interface. The [viewer plan](../reports/local-web-view-plan-2026-09-18.md) builds on these records after their core contracts settle.

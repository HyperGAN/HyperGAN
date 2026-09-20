# Browser / public consumer contract

All API paths are under `/api/v1`. JSON bodies, same-origin only. Static `/` and
`/assets/*` are public login shell resources; run endpoints require the HttpOnly
session cookie created by `POST /session` with `{ "token": "private session token" }`.
The token never enters a URL, storage, telemetry, or browser logs.

- `GET /capabilities` -> `{ "run_id":"r", "reducer":{"sha256":"..."} }`.
- `GET /runs/r` -> run manifest, including status, steps, last_durable_step,
  total_steps, metrics_catalog and current attempt_id; optional name/config.name.
  Optional `evaluation_schedule` maps configured metric IDs to scheduler status,
  `source_step`, `next_step`, and optional `evaluation_id`, `reason`, `skipped_busy`
  and `last_skipped_step`. Ready/heartbeat run payloads carry the same state.
  A busy skip can retain `status: "running"` for the active snapshot while its
  skip count advances. Configured snapshot metrics remain visible before any
  result; cadence and device come from their catalog `specification`.
  Without scheduler state, the viewer infers the next interval boundary from
  observed training steps; stopped runs label that boundary as awaiting training.
  No new field carries the "nothing is scheduled" case: when every snapshot
  metric in the catalog has `specification.trigger != "interval"` and
  `evaluation_schedule` is absent or empty, the evaluation panel renders a
  persistent notice naming those metrics and the configuration edit that
  schedules them. Any scheduled metric, or any scheduler entry, hides it.
- `GET /runs/r/metrics/catalog[?revision=<sha256>]` -> the immutable metric catalog shape.
  Omit revision for the active training catalog; independent evaluation documents
  always supply their own catalog revision.
- `GET /runs/r/artifacts` -> `{ "schema_version":1, "artifacts":{ "id":{ "name":"g", "role":"sample", "modality":"tensor", "media_type":"application/json", "bytes":54, "shape":[2,2], "provenance":{ "step":2, "name":"g" } } } }`.
  `GET /runs/r/artifacts/{id}` downloads a bounded, digest-checked indexed artifact.
  Artifact IDs remain the opaque digest keys; `name` is an added short stable
  label for the source being sampled (`g` generated, `x` real), repeated in
  `provenance`. Periodic previews publish `preview-<digest>` (tensor),
  `preview-<digest>-grid` (generated PNG) and `preview-<digest>-real-grid`
  (real PNG). Explicit `artifacts/index.json` entries may carry their own
  `name`; otherwise the artifact ID is its own name. Consumers group versions of
  one source by `(name, modality)` and order them by `provenance.step`.
- `GET /runs/r/views` -> `{ "map_revision":"...", "streams":[{"stream_id":"evaluation:<id>", "cursor":"...", "caught_up":true, "error":null}], "discovery_error":null }`.
  Training, projection and independent evaluation sources share this inventory.
  At most 64 streams are registered. Inventory overflow is visible without
  stopping existing streams or run metadata updates.
- `GET /runs/r/views/{map_revision}/bootstrap?series=loss%2Fd_total,...&bucket_steps=8`
  returns the following object, or 202 while background historical reduction runs.
  Optional `step_from` / `step_to` apply inclusive integer step bounds.

```json
{
  "schema_version":1,"run_id":"r","view_revision":"view digest",
  "map_revision":"map digest","module_sha256":"module digest",
  "bucket_steps":8,"lineage_revision":"lineage digest",
  "lineage":[{"attempt_id":"a","through_step":null}],
  "groups":[{"key":["loss/d_total","definition digest","a",0],
             "state":{"reducer":"envelope/v1","version":1,"count":1,
                      "first":{"value":1,"position":[1,"emission digest"]},
                      "min":{"value":1,"position":[1,"emission digest"]},
                      "max":{"value":1,"position":[1,"emission digest"]},
                      "last":{"value":1,"position":[1,"emission digest"]}}}],
  "cursor":"opaque committed projection cursor","projection_sequence":1,
  "coverage":{"complete":true}
}
```

The backend owns current recovery lineage. Each allowed attempt has an inclusive
`through_step` or null for unbounded current attempt. UI reboots its view on
lineage/catalog change, never appending abandoned future steps to active lineage.
A selected metric ID is partitioned by definition hash, attempt and aligned step
bucket; different definitions/attempts are separate chart series.

`GET /runs/r/stream?stream_id=projection:{map_revision}&cursor={opaque}` is SSE:

- `frame`: `{ "stream_id":"projection:...", "cursor":"...", "frame":{ "map_revision":"...", "projection_sequence":2, "source":{"run_id":"r","attempt_id":"a",...}, "emissions":[{"id":"emission digest","key":["loss/d_total","a",2],"definition_hash":"definition digest","value":1.2}] } }`
- `ready`, `heartbeat`: liveness only; optional `run` provides latest manifest.
- `artifacts`: refresh the indexed artifact shelf; sampling remains independent
  of selected metrics.
- `metadata`: run/catalog/lineage changed; refresh metadata and request a bootstrap.
- `bootstrap_ready`: refetch the pending bootstrap once. No HTTP polling.
- `reset_required`, `gap`: visible interruption; discard/rebootstrap incompatible
  coverage. A slow browser has bounded queue and reconnects from its last ACK.

SSE received IDs are not commits. Browser closes native EventSource on errors,
waits for already queued complete frames to finish, and explicitly reconnects
from the last fully acknowledged cursor. The worker applies the same bundled
WASM kernel served at `/reducers/host.js`, `/reducers/reducer.wasm` and
`/reducers/reducer.json`. No live server re-reduction is needed. The frontend
request timeout terminates a stuck worker; failed state is never acknowledged.


Snapshot evaluations are independent immutable measurement streams. On load,
after reconnect, and on `stream_added`, the browser reads the stream inventory.
For each evaluation it reads `GET /runs/r/events?stream_id=evaluation:<id>&limit=2`
and the event's explicit `catalog` revision. The current protocol is exactly one
terminal document per evaluation. No live/history reducer runs for these reads.
The browser serializes result loads, retains at most the 64 registered streams,
and allocates plots/tables only on expansion. There is no recurring HTTP poll.

A complete evaluation has `event="evaluation"`, `status="complete"`,
`source_position_known=true`, and the evaluated source `attempt_id` and `step`.
Finite scalar values appear in `metrics`; histograms appear in `distributions`
as `{edges:[...],counts:[...]}` with at most 512 ordered bins. Each result remains
separate by evaluation ID and definition hash, with snapshot/protocol digests and
full recorded protocol available for inspection and raw export. Snapshot metrics
are excluded from the training curve selector. Failed evaluations expose their
`measurement_status` reason; `source_position_known=false` must never be plotted
as a step-zero observation. Sampling artifacts keep their separate shelf.

`discovery_error` is a control event with a visible `reason`; unlike a cursor
reset it does not invalidate current training coverage. Existing sources continue
when optional stream discovery reaches its bounded capacity or finds corruption.

Public source cursors identify run, stream and generation plus a byte offset,
consumed-boundary hash and last event identity. They carry no local inode/path
identity and replay after faithfully copying a run. Readers validate the first
source document's generation and consumed boundary on every page; changed
sources require restarting without a cursor. Private projector file cursors
remain local. Projection cursors already use portable logical source identity.

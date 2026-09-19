# Browser / public consumer contract

All API paths are under `/api/v1`. JSON bodies, same-origin only. Static `/` and
`/assets/*` are public login shell resources; run endpoints require the HttpOnly
session cookie created by `POST /session` with `{ "token": "private session token" }`.
The token never enters a URL, storage, telemetry, or browser logs.

- `GET /capabilities` -> `{ "run_id":"r", "reducer":{"sha256":"..."} }`.
- `GET /runs/r` -> run manifest, including status, steps, last_durable_step,
  total_steps, metrics_catalog and current attempt_id; optional name/config.name.
- `GET /runs/r/metrics/catalog` -> the immutable metric catalog shape.
- `GET /runs/r/views` -> `{ "map_revision":"..." }` for the built-in scalar map.
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

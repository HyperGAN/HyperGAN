# Local browser and streaming API

Install `hypergan[web]`, then run the projection service and viewer independently:

```sh
hypergan project runs/example --follow
# In another terminal:
hypergan serve runs/example --open
```

`serve` binds `127.0.0.1` on an available port. Its startup JSON identifies the
origin and a new private credential file outside the run. Paste that file's token
into the browser login. Tokens never enter URLs or run artifacts. An agent can
send `Authorization: Bearer <token>` instead of using the browser session cookie.
Cookies are scoped by local port so multiple experiments do not overwrite each
other's credentials. The first UI is read-only; these APIs neither start training
nor execute custom Python maps or numerical metric factories.

`hypergan metrics RUN` reads the current catalog. `hypergan contributions RUN`
reads bounded mapped frames without installing the web or reducer runtime. A
missing projection is visible: run `project`, rather than expecting an HTTP
request to create it. Custom maps use the [event-view contract](event-views.md).

The authenticated `/api/v1/openapi.json` contains OpenAPI 3.1 routes, query
parameters and JSON schemas for source events, projection frames, catalogs,
bootstrap state and SSE envelopes. The browser uses this same public interface:

- `/api/v1/capabilities`: server incarnation, selected run, reducer digest and limits.
- `/api/v1/runs/{run_id}`: public observed/durable progress summary.
- `/api/v1/runs/{run_id}/metrics/catalog?revision=...`: current or historical catalog.
- `/api/v1/runs/{run_id}/events?stream_id=...&cursor=...&limit=100`: bounded raw pages.
- `/api/v1/runs/{run_id}/views`: built-in view and registered streams.
- `/api/v1/runs/{run_id}/views/{map_revision}/bootstrap?series=loss%2Fg_total&bucket_steps=100`: historical envelope states.
- `/api/v1/runs/{run_id}/stream?stream_id=projection:{map_revision}&cursor=...`: SSE continuation.
- `/api/v1/runs/{run_id}/artifacts`: bounded artifact index; append `/{artifact_id}` to download a digest-verified entry.

The global `/api/v1/stream` also reports run creation and future stream
registrations. This lets an automatically started viewer wait for the first run
manifest without browser polling. Standalone `serve` requires an existing run.
Selectors are `training`, `projection:<map_revision>`, `evaluation:<id>`, or `*` for
future matching streams. A replay cursor requires one explicit stream. Without a
cursor, an SSE subscriber starts at the current committed watermark; source
history remains available through the paginated API.

A bootstrap returns HTTP 202 while the source lineage index catches up or a
historical job runs. Keep the SSE subscription open and refetch on
`bootstrap_ready`. There is no polling fallback. One bounded background job
reduces through a captured committed projection watermark H using the bundled
WASM module. Its result contains mathematical state, module/view/lineage identity,
bucket alignment, the exact cursor H and projection sequence. The browser's
worker continues those states with frames strictly after H using the same module.

Completed historical jobs are shared by compatible queries. A result has a
five-second completion lease, and its first fetch is guaranteed that result.
Afterward an explicit query can replace it when more than 4096 projection frames
behind, or when over 30 seconds old with a changed head. Pending work remains one
fixed-H job even while training advances. Unchanged history is reused indefinitely.
Live updates never refresh or reduce the cache; changing buckets, selected series,
range or recovery lineage requests a separate bounded job. `step_from` and
`step_to` are inclusive optional query limits.

SSE sends unchanged `frame` envelopes and control events: `stream_added`,
`metadata`, `heartbeat`, `bootstrap_ready`, `artifacts`, `gap` and
`reset_required`. One tail per file performs live fanout for all subscribers.
Registration and replay watermarks are captured without yielding; bounded replay
through H is followed by queued live frames, with overlap discarded before
consumer reduction. A client records its cursor only after applying the complete
frame. SSE received IDs are deliberately not treated as application commits.

Resource bounds are explicit: 32 subscribers, 1 MiB/256 queued messages per
subscriber, 4096 frames per reconnect replay, 64 registered streams, 4096 lineage
attempts, eight cached/pending history jobs with one reducer job active, 32 selected
metrics and 2048 grouped states per bootstrap. Responses cap bootstrap state at
1 MiB; history jobs have a 60-second runtime budget. A slow subscriber receives a
visible gap and disconnects instead of delaying training. Increase bucket size or
narrow step bounds if a requested view exceeds its state budget. Excessive replay
requires a refreshed bootstrap or explicit raw-page reads. These finite limits are
v1 policy, not claims of arbitrary experiment scale.

Current selected lineage includes ancestor attempts only through the checkpoint
used by their child, even when the new attempt performs zero updates. Abandoned
future measurements remain in raw logs. Late evaluation source streams retain the
evaluated attempt and step; registration does not change them to publish time.
The initial default chart maps training scalar events; numerical evaluation stream
production and any additional projection are independent services.

Preview publication records a SHA256 in the existing retained preview index. The
server exposes these sample descriptors without rereading every sample payload.
A final JSON sample is indexed once in a background file task, capped at 16 MiB;
oversized/unavailable samples receive an explicit unavailable descriptor. Model
checkpoints are never deserialized by serving. Generic explicit
`artifacts/index.json` entries may also describe sample, measurement or diagnostic
outputs. Downloads traverse only indexed paths inside the run, reject links and
verify size/hash. The browser's artifact role does not imply that every modality
has a built-in renderer.

The local server rejects unexpected Host/Origin, uses no permissive CORS, keeps
sessions HttpOnly/SameSite, serves its chart/reducer assets offline, and does not
support remote binding. The API returns a public run summary instead of paths and
full recipe arguments. Trusted owner-controlled run files remain the data source;
this is not a multi-tenant service. A corrupt or replaced stream fails visibly and
requires restoring/rebuilding valid history and reconnecting the server; the server
never edits a source to repair a chart.

Run the optional large-log proof explicitly:

```sh
python scripts/metrics_server_proof.py --events 1000000 --output /tmp/server-proof.json
```

It generates and removes a synthetic run, measures shared source/projection
indexing, historical bootstrap, warm query serialization and five-viewer live
continuation, and asserts no live reducer calls. It measures standalone serving,
not training overhead or image quality.

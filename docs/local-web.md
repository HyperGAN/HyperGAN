# Local browser and streaming API

Install `hypergan[web]`, then run the projection service and viewer independently:

```sh
hypergan project runs/example --follow
# In another terminal:
hypergan serve runs/example --open
```

`serve` listens on `0.0.0.0` on an available port by default. Select another
address with `--host 127.0.0.1` and a fixed port with `--port 8123`. The printed
browser URL uses `127.0.0.1` for a wildcard bind; remote browsers use this machine's
address with that same port. Browser requests remain same-origin.

### Remote browsers over plain HTTP

Opening the viewer from another machine, for example `http://mlserver:8123` over a
Tailscale or LAN address, needs no TLS. Browsers expose the Web Crypto API
(`crypto.subtle`) only to secure contexts, so the reducer host verifies the bundled
WebAssembly module with `crypto.subtle` where it exists and with an equivalent
portable SHA-256 otherwise. The integrity check therefore still runs, and still
refuses a modified module, on the insecure origin a plain-HTTP host other than
`localhost` receives. The viewer uses no other secure-context-only browser API.

HTTPS remains the recommendation outside a trusted private network; put the port
behind TLS, for example with `tailscale serve`. Such a proxy must present the
viewer's own `Host` authority including its port, and a matching `Origin`, because
the local server rejects a `Host` or `Origin` it does not recognize.

Authentication defaults to `none`. Select `--auth token` to require a private
bearer token. Startup JSON identifies the browser origin, server incarnation and
private session file outside the run. In token mode, paste that file's token into
the browser login, or send `Authorization: Bearer <token>` from an external client.
Tokens never enter URLs or run artifacts. Credentials last for that server
incarnation and rotate on restart. Browser cookies are scoped by port so multiple
experiments do not overwrite each other's credentials. These APIs neither start
training nor execute custom Python maps or numerical metric factories.

`hypergan metrics RUN` reads the current catalog. `hypergan contributions RUN`
reads bounded mapped frames without installing the web or reducer runtime. A
missing projection is visible: run `project`, rather than expecting an HTTP
request to create it. Custom maps use the [event-view contract](event-views.md).

The `/api/v1/openapi.json` endpoint contains OpenAPI 3.1 routes, query
parameters and JSON schemas for source events, projection frames, catalogs,
bootstrap state and SSE envelopes. The browser uses this same public interface:

- `/api/v1/capabilities`: server incarnation, selected run, reducer digest and limits.
- `/api/v1/runs/{run_id}`: public observed/durable progress summary.
- `/api/v1/runs/{run_id}/console`: GET/PUT the persisted `progress_every` interval; the browser control changes terminal output at complete update boundaries without changing stored metrics.
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
1 MiB; history jobs have a 180-second runtime budget, reported in capabilities.
The limit can be set explicitly when constructing the server. A slow subscriber receives a
visible gap and disconnects instead of delaying training. A five-second ASGI send
deadline also closes a socket consumer that stops reading before it can receive
the gap; cleanup removes its subscription. Increase bucket size or
narrow step bounds if a requested view exceeds its state budget. Replay-budget gaps preserve the browser reducer and its acknowledged cursor.
The browser drains queued work and reconnects promptly, making progress through
bounded replay segments even when cold history has a large live suffix. Changed
stream generation or history still requires a fresh bootstrap. These finite limits are
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

## Automatic viewer during CLI training

With the `web` extra installed, `hypergan train` and `hypergan resume` automatically
reserve an available port on `0.0.0.0` and launch a supervised local viewer. Startup
runs concurrently with training; there is no browser launch or readiness wait by
default. Without the extra, these commands remain silently headless. The Python
`train()` and `resume()` APIs never start a viewer.

```sh
hypergan train project/config.toml --run-dir runs/example --open
hypergan resume runs/example --server --server-port 8123
hypergan train project/config.toml --run-dir runs/headless --no-server
```

`--server` requires dependencies, an available port and a ready
server before numerical imports or training begin. `--server-port`, `--server-host`, `--auth` and `--open`
also imply that requirement. `--server-host` selects the bind address; `--auth token`
enables token authentication with the same behavior as standalone `serve`. Only `--open` launches a browser. `--no-server`
performs no web imports or socket setup and conflicts with these explicit viewer options.
Viewer diagnostics go to stderr; stdout and `--progress-json` remain machine
readable. Token mode prints the private credential path on stderr for signing in.

The supervised HTTP server and built-in scalar projection producer run in
separate spawned processes. The producer reads at most 100 documents per page;
HTTP requests never run Python maps. Custom maps and expensive metric evaluators
remain explicit independent commands. The viewer can wait for a run manifest
without creating the trainer's run directory. Once a manifest exists, a nonsecret
`observations/viewer-<instance>.json` receipt records launch mode, process IDs and
health, independently of numerical configuration/checkpoint identity.

Optional startup errors and failures after startup produce warnings without
failing training. Required startup failure prevents numerical work. The detached
supervisor keeps the server and projection producer running after completion,
bounded stops, numerical failures, SIGINT, SIGTERM and even abrupt training-process
death. You can continue inspecting the same URL and existing browser session.

`resume` reconnects to the same per-run viewer, without starting another server
or projection writer. Private registry state lives outside the experiment under
the operating system's temporary directory. An OS lifetime lock and a checked
server incarnation identify the running service; recorded PIDs are diagnostic,
never used to signal a process. Omitted bind/auth options inherit the active
server settings. Explicit conflicting options fail with an instruction to stop
the existing viewer before choosing new settings.

Startup immediately prints a discovery command even when a short training
attempt ends before optional HTTP startup has finished. Read the connection URL,
private session path, health and log location, or stop the automatic viewer:

```sh
hypergan server-status runs/example
hypergan stop-server runs/example
```

This command can run from another terminal and is safe to repeat. It stops only
the registered incarnation, removes its private credentials, drains one final
bounded projection page and joins children within bounded deadlines. A large
backlog may remain resumable with `hypergan project RUN`. If the supervisor itself
dies, its children exit independently; the next launch safely replaces stale
registry state and rotates credentials. Standalone `serve` remains a foreground
command; stop that command with Ctrl-C.

### Checkpoint metrics and projection progress

Run responses and stream heartbeats expose `durable_event_boundary` when the
training controller recorded one. `metric_consistency` compares its event byte
boundary with the built-in metric projection's source cursor. `pending` means
the projection has not reached that committed boundary; `caught_up` means the
server has observed projection frames through it. `unavailable` means the run
has no recorded boundary or its projection cannot be read. The response includes
`committed_step`, `committed_offset`, and `projected_offset`.

This readout does not revalidate checkpoint durability or acknowledge that a
particular browser has rendered the frames. Projection progress is independent
of the training event tail, and uses byte offsets so restoring an earlier step
does not accidentally look caught up. Projection-only changes emit a heartbeat
even after training and its manifest have stopped changing.

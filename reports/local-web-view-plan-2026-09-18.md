# Local browser view for HyperGAN

Decision recorded 2026-09-18: replace the archived Electron/Tk/Pygame viewers with an optional local HTTP server and browser client. HyperGAN owns training and run state; the browser observes that state and, later, requests supported job actions. The same project must work headlessly and on a cluster.

This is the implementation contract for the next workflow slice. **The current CPU foundation does not implement a web server or the flags below.** Electron was already removed in PR #300; this decision selects its replacement without restoring desktop dependencies.

The [2026-09-19 metrics research](metrics-first-research-2026-09-19.md) refines the proposed observation/API design: a file-based metric catalog, distinct modality-neutral sampling and measurement roles, Python event maps and shared reducers for CouchDB-style views without a database, removable defaults and opt-in expensive evaluation. The browser must use the same versioned public read API as agents and external consumers. The first viewer streams mapped contributions without server re-reduction; a shared reducer handles historical bootstrap and live client views. SSE is the proposed first transport. Future WebSocket adapters and multi-server collectors reuse its identity/replay contracts; those additions are not required in the first server. W2/W3 remain unimplemented.

## User experience

The recommended local training install will include an optional `web` extra, documented as `hypergan[train,web]`. Base installation and `hypergan[train]` remain usable without it. Serve bundled assets offline; no account, frontend build step, external CDN or telemetry is needed to view a run.

Planned command behavior:

| Invocation | Contract |
| --- | --- |
| `hypergan train …` | For a local CLI run, attempt the viewer when the web extra is installed; print its loopback URL. Missing web dependencies produce one informative message and training proceeds |
| `hypergan train … --no-server` | Disable server startup, socket binding and viewer dependencies completely |
| `hypergan train … --server` | Explicitly request the viewer; validate dependencies and bind availability before starting training, with an actionable failure if unavailable |
| `hypergan train … --open` | Explicitly request a browser tab as well as the server; never open a browser automatically |
| `hypergan serve RUN_DIR` | Attach an independent viewer to an existing running or completed run without constructing a model or importing training |
| Scheduled/cluster worker | Headless by default. No server per rank; expose artifacts/status through the launcher connection and view from the user's machine |

`--no-server` combined with `--server` or `--open` is a preflight usage error. CLI overrides take precedence over the execution profile. Record the effective viewer mode separately from numerical recipe identity. The Python training API starts no listener unless requested. Automatic startup failure warns once and leaves the training result unchanged; failure after a successful explicit startup also leaves training running and reports viewer health separately. An explicit occupied port fails preflight; the default chooses an available loopback port and prints the actual address.

The first screen shows run identity, qualified/unqualified status, latest durable update, losses, elapsed time, current sample previews and terminal errors. Distinguish the latest observed update from the last recoverable checkpoint: a flushed event is not proof of saved numerical state. Show that live throughput and completion estimates are estimates. Do not label adversarial losses as image quality. Image grids arrive with the image recipe; the current synthetic reference can display its numeric samples without pretending they are images.

## Process and state boundaries

Use a separate server process reading the run's versioned manifest, append-only events and published preview artifacts. The trainer is the writer. It never waits for browser responses. An attached viewer can be stopped, refreshed or restarted without stopping training. The automatically started child is cleaned up when the CLI exits; `serve` provides persistent inspection afterward.

Following the 2026-09-19 owner clarification, start with a live event/contribution stream (proposed SSE), plus bounded HTTP bootstrap/history and replay cursors. The browser and external consumers use the same versioned public API. Live fanout does not re-reduce mapped contributions; the browser uses the shared reducer, also used for bounded historical bootstrap. A WebSocket adapter can reuse the protocol later. Handle partial trailing JSONL records, reconnects, duplicate deliveries, attempt transitions and bounded pagination. Surface stale or unreachable state instead of guessing that a job completed. Durable update counters are authoritative; browser timestamps are not training state. Streaming and finite replay pages preserve the same identity/cursor contract.

Publish previews atomically with immutable run/attempt/update/sample identifiers. A preview request must not consume the data/prior/penalty RNG streams or change model buffers, gradients, optimization or EMA. Generate previews at configured safe update boundaries using an isolated EMA snapshot or explicitly preserved evaluation-mode state, inference mode, a separate sampling stream and declared conditioning. Inference mode alone does not prevent BatchNorm buffer updates or dropout RNG use; restore module modes and test stateful/stochastic custom modules. Bound preview frequency, artifact size and retention. Do not let a slow client or growing event log add unbounded memory or training latency.

For distributed runs, only the designated writer publishes global progress and preview metadata after coordinated updates. The viewer does not join process groups or collectives. For real clusters, use a local client with the launcher's authenticated artifact/status transport, or an explicit SSH tunnel to a single service on the allocated host. Private worker addresses must not be presented as directly reachable local URLs. Viewer disconnection must not cancel a submitted job.

The first version is **read-only**. Later cancellation and sampling requests use the same authenticated job-command interface as the CLI, with acknowledgements, idempotency and safe update-boundary handling. Do not claim `cancel`, pause, resume or live configuration editing until the loop implements the corresponding state transition. HTTP handlers never mutate training tensors or optimizer state.

## Local access contract

Bind loopback only in the first version. Use a per-session access token and validate Host/Origin for browser requests; keep credentials out of run manifests, committed files and shared logs. Specify a local authentication bootstrap before W2 lands: store tokens with owner-only permissions, keep raw tokens out of printed URLs/access logs/shared logs, use a browser session exchange, and define rotation/expiry. Test rejected unauthenticated requests and invalid Host/Origin values. Do not enable permissive CORS. Any later state-changing endpoint needs explicit request authentication and CSRF protection. Remote public binding is a separately reviewed feature; an optional local viewer is not a hosted multi-user service.

Only serve explicitly indexed artifacts within the selected run root. Reject traversal and symlink escapes; never expose a general filesystem browser, arbitrary Python/config execution or a model-unpickling endpoint. Inspect metadata without loading weights. Escape user-controlled run names, errors and event text in HTML. Keep image/data previews local and make shared access an explicit user action.

## Implementation checklist

- [x] **W1 — stabilize observable run state alongside recovery.** The [core observation slice](core-observation-2026-09-18.md) completes bounded reconnect cursors, periodic immutable previews/retention, and serialized attempt-bound manual checkpoint requests on top of versioned manifests/events, durable progress and recovery. Installed tests cover partial writes, attempts/reconnect, exact preview-on/off numerical state, older-checkpoint replay, filename collisions, acknowledgement failure and live CLI submission. The local request protocol is at-least-once until durable acknowledgement; pending does not prove execution. Payload/retention limits and custom-code boundaries are documented in the [observation guide](../docs/observation.md).
- [ ] **W2 — standalone read-only server.** Implement the optional `web` extra, `serve`, loopback/session access, indexed artifacts, bounded historical bootstrap/replay, live streaming and an offline browser page using the public API. Exit when a live fixture and completed run can be inspected without torch/model imports, with traversal and stale-state tests passing.
- [ ] **W3 — local CLI integration.** Integrate the standalone server with the planned server/disable/open flags. Exit when no-server binds no socket; startup and mid-run server failure do not corrupt training; viewer-on/off runs have identical controlled training state; subprocess cleanup passes on the supported platforms.
- [ ] **W4 — image and cluster observation.** Add fixed sample grids and attach through one qualified launcher transport. Exit when reconnect works across a real job restart, only the designated writer publishes global progress, and disconnect leaves training alive.
- [ ] **W5 — job controls after recovery.** Add authenticated, acknowledged cancellation and bounded sampling through the common CLI/service command contract. Exit when duplicate requests, worker failure, safe stopping and accessible error presentation are tested.

W1 is implemented by the core observation checkpoint. W2–W3 follow the staged metrics/view backbone and server PRs targeting `develop`, alongside the next public distributed integration work. W4 uses the actual GPU/cluster qualification stages. This ordering delivers a useful viewer without making frontend work a prerequisite for numerical correctness.

Implementation tracker: [#303](https://github.com/HyperGAN/HyperGAN/issues/303). The coordinator owns sequencing and review; implementation PRs must link these gates. The [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md) remains authoritative for release requirements, and [the status ledger](resurrection-status.md) records what has actually shipped on `develop`.

# Human feedback checklist

Running list of feedback from the project owner, gathered while using HyperGAN.
This is a living document: new items are appended as they come up, and each
item is updated as it is addressed. Earlier, already-implemented feedback lives
in [feedback-2026-09-20.md](feedback-2026-09-20.md).

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (link the PR).

## Open

### 8. Sample slider should cover the whole run, not the last N (raised 2026-09-20)

Owner: "the viewer shows the last N images but it should really show all of them. if it's on the last one and a new sample comes in it can update and stay on the last one. but i want someone to be able to slide from the beginning of their training to the end."

- [ ] Stop pruning image history by default: keep every published preview (or at least every image grid) for the life of the run, so the slider reaches back to step 0. Keep a bound only for disk-heavy tensor payloads if one is needed, and make any retention an explicit opt-in.
- [ ] Lift the viewer/API caps that assume a small preview count (`previews/index.json` is currently rejected above 100 entries; the artifact list is rebuilt from it) so a long run with thousands of samples still loads quickly.
- [ ] Slider behavior: when positioned on the latest sample and a new one arrives, advance to the new one; when positioned on an earlier sample, stay put and do not jump.
- [ ] Tests for retention-off, index size, and the follow-latest / stay-put slider behavior.

Where this lives today:
- `src/hypergan/previews.py` `DEFAULT_KEEP = 20`, `MAX_KEEP = 100`; `_publish_preview` deletes expired generations on every publish.
- `src/hypergan/cli.py` `--preview-keep`; `src/hypergan/web_service.py` rejects a preview index longer than 100.
- `frontend/src/app.js` groups artifacts by name and renders the slider.

### 9. FID (snapshot evaluations) should be a chart, not a wall of text (raised 2026-09-20)

Owner: "on snapshot evaluations FID should be a graph like the metrics, different x tho ofc. right now it's a wall of text. it may be a graph eventually, maybe it's just a graph with one point atm."

- [ ] Plot each scalar snapshot metric (FID and friends) as a line chart with source step on the x axis, one point per completed evaluation, using the same chart look as the training metrics.
- [ ] A single result is a chart with one point, not a text card; failed/cancelled evaluations show as status, not as text walls.
- [ ] Keep the per-result details (duration, device, sample count, status) reachable but collapsed.
- [ ] Tests for the chart with one point and with several points across steps.

Where this lives today:
- `frontend/src/evaluations.js` renders one card per evaluation stream, with an SVG single-point `plot` and a text list of fields.
- Results come from `metrics/evaluations/<id>/stream.json` streams, one metric per evaluation, surfaced by `src/hypergan/web_service.py`.
- The training metrics chart uses echarts in `frontend/src/app.js`.

### 10. Clarify or remove the "sample - tensor" artifact (raised 2026-09-20)

Owner: "theres a 'sample - tensor' that i'm not sure what it's supposed to be or how to use it. lets clarify or remove it."

- [ ] Decide whether the raw tensor preview (the JSON payload behind every image grid, plus the final sample) earns a place in the viewer. If kept, label it by what it is (for example "g raw tensor, step N") and say what it is for; if not, hide it from the artifact list by default and keep only the download.
- [ ] Make sure the image grid, not the tensor, is what a user sees first under each sample name.
- [ ] Update docs/image-previews.md and the viewer test that covers the artifact list.

Where this lives today:
- `src/hypergan/web_service.py` emits `role='sample', modality='tensor'` records for every preview payload (`preview-<digest>`) and for the final sample.
- `frontend/src/app.js` renders it with a "Preview numbers" button and a size hint.

### 7. Investigate the Python + Node + Rust stack and its onboarding cost (raised 2026-09-20)

Owner: "it seems odd that we use python and node and rust. i think python and rust is a bit sensible. but node seems like an outlier. is that something our users will need to install? gotta think about the onboarding experience."

What is true today:
- Python is the product. `pip install hypergan[web]` pulls only Python wheels (starlette, uvicorn, wasmtime).
- Rust lives in `reducers/core` and compiles to the metrics reducer WASM, which is committed as `src/hypergan/metrics_reducer/assets/reducer.wasm` and executed by wasmtime (Python side) and the browser. Users do not need cargo.
- Node is used only to bundle the viewer frontend (`frontend/`, esbuild + echarts) into the committed `src/hypergan/web_assets/app.js`. Users do not need Node; contributors who edit the UI do.

- [ ] Confirm and document the split clearly: end users need Python only; Rust and Node are contributor-only build tools, with the built artifacts committed. State this in README and a contributor guide.
Owner note: an acceptable outcome of this investigation is "it's fine as is", provided the user-facing install stays Python-only.

- [ ] Decide whether Node is worth keeping. Options to evaluate: keep esbuild with committed output (status quo), drop the bundler and ship plain ES modules plus a vendored chart library, or move the bundling step into a Python-invoked tool so there is one contributor toolchain. Record the trade-offs (echarts size, minification, dev-mode watch from item 4).
- [ ] Add a CI check that the committed `app.js` and `reducer.wasm` match their sources, so a contributor without Node or Rust can still trust the artifacts they ship.
- [ ] Verify the onboarding path end to end on a clean machine: `pip install`, `hypergan train`, open the viewer, without Node or cargo present.

## Done

### 6. FID should evaluate on an interval by default (raised 2026-09-20)

Owner ran `training-runs/start.sh` and saw no FID because both FID metrics in the run's `cifar10.toml` are `trigger = "manual"`, so the manifest records an empty evaluation schedule and nothing ever fires.

- [x] Make snapshot metrics such as FID default to `trigger = "interval"` with a sensible `every_steps` (the public example uses 10,000 for FID50k) and `on_busy = "skip"`, so a recipe that declares an FID metric gets periodic results without extra fields. `trigger = "manual"` stays available as an explicit opt-out.
- [x] Decide the default cadence and evaluation device behavior when a metric omits them; document contention when the evaluation device is the training GPU.
- [x] Update the CIFAR example, the generated/pretrained recipe used by the owner's run, and the docs to reflect the new default.
- [x] Have the viewer and CLI progress output make it obvious when a configured FID metric has no schedule, so a manual-only setup is not silent.
- [x] Add tests for the default schedule and for explicit manual opt-out.

**Status:** Implemented in commit 7d937f94, merged to develop.

Follow-up (2026-09-20): the owner reached step 10k on a fresh run with no FID because their `cifar10.toml` still says `trigger = "manual"`, and the stderr warning at launch was missed among other config warnings.

- [x] Show the "no automatic evaluation scheduled" notice in the viewer's evaluation panel and in the periodic CLI progress line, not only at launch.
- [x] Consider whether an explicit `trigger = "manual"` on an FID metric in a training run should print a louder, single-line hint naming the exact edit. It should: one is printed.

The notices appear in three places when every enabled snapshot metric is manual:
a persistent notice above the schedule cards in the viewer's **Snapshot evaluations**
panel (`frontend/src/evaluations.js`, derived from the catalog `specification` and
the run's empty `evaluation_schedule`, with no new API field), a one-line `reminder:`
on the first periodic CLI progress line and every tenth one after it (an added
`evaluation_reminder` field on `--progress-json` train rows), and a single loud
`hint:` line after the launch warnings block naming the exact `[metrics.custom.NAME]`
edit and the `hypergan resume RUN --config CONFIG` that applies it
(`hypergan.metrics.manual_evaluation_hint` / `manual_evaluation_reminder`).

- `src/hypergan/metric_plugins.py` defines `DEFAULT_SNAPSHOT_TRIGGER = "interval"`
  and `DEFAULT_EVALUATION_EVERY_STEPS = 10000` in one place. A snapshot metric that
  omits `trigger` resolves to interval evaluation, and an explicit
  `trigger = "interval"` without `every_steps` takes the same default; both resolve
  `on_busy = "skip"`. Manual metrics keep no cadence fields, so a run recorded before
  this change resolves to exactly the specification it already stored.
- Interval evaluation still has no device fallback. A metric that resolves to
  interval and names no `evaluation.device` is rejected during configuration
  resolution, with a message naming the metric and both remedies (an explicit
  device, or `trigger = "manual"`). Failing is the right side of the
  "warn on unqualified, fail on incompatible" rule here: a silent fallback to
  manual would reproduce exactly the bug this item reports.
- `hypergan.metrics.evaluation_warnings` adds two warnings, printed by
  `hypergan train`, `resume`, `validate` and `preflight` through the existing
  `_warnings` path and recorded in the run manifest: one when every enabled
  snapshot metric is manual (naming them, and the empty schedule that results),
  and one when an interval metric's evaluation device may be the training device.
- Resume is unaffected for existing runs, which store their resolved trigger in
  the manifest. Adding a schedule to an existing run is refused by
  `hypergan train` with a message naming `metrics` as the differing section and
  pointing at `hypergan resume RUN --config CONFIG`, which accepts it.
- `examples/cifar-pretrained-sagan.toml`, [configuration](../docs/configuration.md),
  [image FID](../docs/image-fid.md) and the [CIFAR recipe](../docs/cifar-recipe.md)
  document the default, the device rule and the GPU contention cost.
- The viewer already labels a manual snapshot metric `manual` with no next step
  from the catalog specification, so no frontend change was needed.
- No recipe generator in this repository emits `fid_smoke`/`fid50k_train`;
  `hypergan new` writes only the numerical reference recipe. The owner's
  `cifar10.toml` lives outside the repository and still needs its two
  `trigger = "manual"` lines removed (and a separate `evaluation.device`, e.g.
  `cuda:1`, to avoid sharing the training GPU) on a new run directory.

### 1. Named samples with history slider (raised 2026-09-20)

**Status:** Implemented in commit a10611c3, merged to develop in e315ebb6. Previews are named `g` (EMA generator output, override with `--preview-name`) and `x` (the matching real batch, published in the same generation). The viewer groups artifacts by name, shows the newest image, and offers a slider with keyboard support plus a Latest button. Retention default moved from 3 to 20 (`--preview-keep`, max 100). Names live in the per-run manifest, not the numerical config, so existing runs resume unchanged.

- [x] Index every sample by a short stable name, e.g. `x` (real input) and `g` (generator output), instead of the preview digest key. The UI and API should present samples by that name.
- [x] For image samples, show only the most recent image per name by default.
- [x] Add a slider (or equivalent scrubber) per named sample to step back through earlier versions by step.
- [x] Decide how much history to retain; previews are currently pruned to a small fixed count, so the slider needs either a larger retention window or a configurable one.

Where this lives today:
- `src/hypergan/previews.py` publishes previews keyed by an `identity` dict and keeps only a few (`keep=3`).
- `src/hypergan/web_service.py` builds the artifact list with keys `preview-<digest>` and `-grid` suffixes.
- `frontend/src/app.js` renders every artifact as a flat list under "Samples & artifacts".

### 5. HTTPS fronting of the viewer (e.g. `tailscale serve`) is rejected by the origin check (found 2026-09-20 while fixing item 3)

**Status:** Implemented in commit 3acfd240, merged to develop (fast-forward). Not yet verified against a live `tailscale serve`; the design accepts both a preserved and a rewritten Host. `train`, `resume` and `serve` take `--public-origin URL`; it is the only thing that makes a non-local origin acceptable, and it neither widens nor weakens the direct `http://<host>:<port>` check.

- [x] Accept an `https://` origin (and a configurable public origin, e.g. `--public-origin https://mlserver.tailnet.ts.net`) so a TLS proxy can front the viewer.
- [x] Mark the session cookie `Secure` when the public origin is HTTPS.
- [x] Add a web test that exercises requests carrying a proxied HTTPS origin.
- [x] Document the `tailscale serve` setup once it works end to end.

What the viewer now trusts, and only when `--public-origin` is set:
- A request whose `Host` is that origin's authority and whose `Origin`, when sent, equals that origin. `tailscale serve --bg 8765` forwards the public `Host` unchanged, so this is the normal path.
- A proxy that rewrites `Host` to the local authority: its own `Host` must still pass the direct check, `X-Forwarded-Host` must name the public authority, and `X-Forwarded-Proto`, if present, must match the public scheme. Forwarded headers are ignored entirely without a public origin.
- Nothing else: no wildcard, no other `https` origin, no permissive CORS.

Where this lives now:
- `src/hypergan/web_session.py` `normalize_public_origin` (strict absolute http/https URL, host and optional port only) and `LocalSession.match_request`, which names the channel `public`, `direct` or `None`; `permits_request` delegates to it.
- `src/hypergan/web_server.py` reads `X-Forwarded-Proto`/`X-Forwarded-Host`, records the channel on the ASGI scope, and marks the session cookie `Secure` only on the https channel, so the plain-HTTP loopback login keeps working. CSP gained `form-action 'self'`; every directive stays `'self'` and every page URL is relative, so nothing is pinned to `http://`.
- `src/hypergan/web_launch.py` `serve(..., public_origin=...)` and `src/hypergan/web_autostart.py`, where the value travels in the viewer's private registry state to the detached supervisor, and is reported by `server-status`, the startup JSON, the credential file and the run receipt.
- `tests/web/test_public_origin_proxy.py` and `tests/foundation/test_cli_viewer_options.py`.
- [docs/local-web.md](../docs/local-web.md) "HTTPS through your own TLS proxy" and the README viewer paragraph.

### 2. Stable server port with `--port` and increment-on-conflict (raised 2026-09-20)

**Status:** Implemented in commit 7ff6d719, merged to develop. `train`/`resume`/`serve` take `--port` (alias `--server-port`) with default 8765; when the default is busy the viewer steps upward through 100 ports and reports the bound URL. An explicit port stays strict, and `--port 0` still asks the OS for any free port. The default lives once in `src/hypergan/ports.py`.

- [x] The viewer port changes on every start because the default is an OS-assigned port (0). Add a `--port` option to `train` with a fixed default port.
- [x] If the default port is in use, increment and retry until a free port is found, and report the port actually chosen.
- [x] Keep an explicit `--port N` strict: if the user names a port and it is busy, fail rather than silently move.
- [x] Apply the same default and increment behavior to `hypergan serve`, whose `--port` also defaults to 0.

Where this lives today:
- `src/hypergan/cli.py` defines `--server-port` on `train` (default automatic) and `--port` on `serve` (default 0).
- `src/hypergan/web_launch.py` `bind_server` binds exactly one port and raises on conflict.
- `src/hypergan/web_autostart.py` reuses a recorded port when resuming an existing run's viewer.

### 4. Live UI development without restarting the training server (raised 2026-09-20)

**Status:** Implemented in merge 32bfa04e. Enable with `--dev` on `train`/`resume` (alias `--viewer-dev`), `--dev` on `serve`, or `HYPERGAN_VIEWER_DEV=1`. Run `npm run --prefix frontend watch` to rebuild on source edits; the page auto-reloads in dev mode. Note: static responses already sent `Cache-Control: no-store`; the real causes were no watch mode and assets resolving through the editable install rather than the working checkout.

Owner: "it'd be cool if refreshing the UI changed in development mode without having to update the running server, because it makes me have to stop and resume training to work on the UI."

Where this lives today:
- `src/hypergan/web_server.py` `static` route reads `web_assets/*` from the installed package on every request (the package is installed editable), but sends no `Cache-Control`, so browsers may keep a stale `app.js`.
- Frontend source is `frontend/src/*.js`; `frontend/build.mjs` bundles it once with esbuild into `src/hypergan/web_assets/app.js`. There is no watch mode, so source edits need a manual `npm run build`.

- [x] Add a development mode for the viewer (flag or env var) in which static assets are served with `Cache-Control: no-store` so a browser refresh always picks up the current files.
- [x] Add an esbuild watch (`npm run watch` or similar) that rebuilds `web_assets/app.js` on every `frontend/src` change, so refresh reflects source edits without touching the running server.
- [x] Optionally auto-reload the page when the bundle changes.
- [x] Document the dev workflow: start training once, run the watcher, edit, refresh.

### 3. Viewer fails over Tailscale: "Cannot read properties of undefined (reading 'digest')" (raised 2026-09-20)

Loading the viewer from a laptop at `http://mlserver:<port>` over Tailscale shows this error with no failed network requests. Loading it on the server itself works.

Cause: the metrics reducer verifies the bundled WASM with `crypto.subtle.digest`. Browsers expose `crypto.subtle` only in secure contexts (HTTPS, or `localhost`/`127.0.0.1`). A plain-HTTP origin on any other hostname leaves `crypto.subtle` undefined, so the reducer never loads.

- [x] Fall back to a bundled pure-JS SHA-256 when `crypto.subtle` is unavailable, so the digest check still runs on insecure origins.
- [x] Replace the raw TypeError with a clear message in the UI that names the cause (insecure context) and the remedy.
- [x] Document remote access: either the fallback above, or serve over HTTPS (for example `tailscale serve`) and say so in the viewer docs.
- [x] Audit the rest of the frontend for other secure-context-only APIs so remote HTTP viewing works end to end.

Fixed in e8f2b646:
- `src/hypergan/metrics_reducer/assets/host.js` verifies the module with `crypto.subtle`
  where the browser exposes it and with an inline portable SHA-256 otherwise. A digest
  mismatch is still refused on either path.
- `frontend/src/app.js` names the missing Web Crypto API and the remedy instead of
  relaying a raw `TypeError`.
- [Remote browsers over plain HTTP](../docs/local-web.md) documents plain-HTTP remote
  viewing and still recommends HTTPS.
- The audit found no other secure-context-only API in `frontend/src`,
  `web_assets/view-worker.js` or the reducer assets; the session cookie is
  `HttpOnly`/`SameSite` without `Secure`, so it works over plain HTTP.
- `tests/browser/test_reducer_browser.py` covers the standard SHA-256 vectors, the
  bundled module digest, loading with `crypto.subtle` undefined and rejection of a
  modified module on that path.

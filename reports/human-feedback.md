# Human feedback checklist

Running list of feedback from the project owner, gathered while using HyperGAN.
This is a living document: new items are appended as they come up, and each
item is updated as it is addressed. Earlier, already-implemented feedback lives
in [feedback-2026-09-20.md](feedback-2026-09-20.md).

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (link the PR).

## Open

### 1. Named samples with history slider (raised 2026-09-20)

- [ ] Index every sample by a short stable name, e.g. `x` (real input) and `g` (generator output), instead of the preview digest key. The UI and API should present samples by that name.
- [ ] For image samples, show only the most recent image per name by default.
- [ ] Add a slider (or equivalent scrubber) per named sample to step back through earlier versions by step.
- [ ] Decide how much history to retain; previews are currently pruned to a small fixed count, so the slider needs either a larger retention window or a configurable one.

Where this lives today:
- `src/hypergan/previews.py` publishes previews keyed by an `identity` dict and keeps only a few (`keep=3`).
- `src/hypergan/web_service.py` builds the artifact list with keys `preview-<digest>` and `-grid` suffixes.
- `frontend/src/app.js` renders every artifact as a flat list under "Samples & artifacts".

### 2. Stable server port with `--port` and increment-on-conflict (raised 2026-09-20)

- [ ] The viewer port changes on every start because the default is an OS-assigned port (0). Add a `--port` option to `train` with a fixed default port.
- [ ] If the default port is in use, increment and retry until a free port is found, and report the port actually chosen.
- [ ] Keep an explicit `--port N` strict: if the user names a port and it is busy, fail rather than silently move.
- [ ] Apply the same default and increment behavior to `hypergan serve`, whose `--port` also defaults to 0.

Where this lives today:
- `src/hypergan/cli.py` defines `--server-port` on `train` (default automatic) and `--port` on `serve` (default 0).
- `src/hypergan/web_launch.py` `bind_server` binds exactly one port and raises on conflict.
- `src/hypergan/web_autostart.py` reuses a recorded port when resuming an existing run's viewer.

### 5. HTTPS fronting of the viewer (e.g. `tailscale serve`) is rejected by the origin check (found 2026-09-20 while fixing item 3)

`web_session.permits_request` requires `Origin` to equal `http://<host>` and the Host port to match the bound port. A TLS reverse proxy in front of the viewer sends an `https://` origin and usually a different port, so token login and the console `PUT` are rejected. Plain HTTP remote viewing works after item 3; HTTPS remote viewing does not yet.

- [ ] Accept an `https://` origin (and a configurable public origin, e.g. `--public-origin https://mlserver.tailnet.ts.net`) so a TLS proxy can front the viewer.
- [ ] Mark the session cookie `Secure` when the public origin is HTTPS.
- [ ] Add a web test that exercises requests carrying a proxied HTTPS origin.
- [ ] Document the `tailscale serve` setup once it works end to end.

### 6. FID should evaluate on an interval by default (raised 2026-09-20)

Owner ran `training-runs/start.sh` and saw no FID because both FID metrics in the run's `cifar10.toml` are `trigger = "manual"`, so the manifest records an empty evaluation schedule and nothing ever fires.

- [ ] Make snapshot metrics such as FID default to `trigger = "interval"` with a sensible `every_steps` (the public example uses 10,000 for FID50k) and `on_busy = "skip"`, so a recipe that declares an FID metric gets periodic results without extra fields. `trigger = "manual"` stays available as an explicit opt-out.
- [ ] Decide the default cadence and evaluation device behavior when a metric omits them; document contention when the evaluation device is the training GPU.
- [ ] Update the CIFAR example, the generated/pretrained recipe used by the owner's run, and the docs to reflect the new default.
- [ ] Have the viewer and CLI progress output make it obvious when a configured FID metric has no schedule, so a manual-only setup is not silent.
- [ ] Add tests for the default schedule and for explicit manual opt-out.

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

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

### 4. Live UI development without restarting the training server (raised 2026-09-20)

Owner: "it'd be cool if refreshing the UI changed in development mode without having to update the running server, because it makes me have to stop and resume training to work on the UI."

Where this lives today:
- `src/hypergan/web_server.py` `static` route reads `web_assets/*` from the installed package on every request (the package is installed editable), but sends no `Cache-Control`, so browsers may keep a stale `app.js`.
- Frontend source is `frontend/src/*.js`; `frontend/build.mjs` bundles it once with esbuild into `src/hypergan/web_assets/app.js`. There is no watch mode, so source edits need a manual `npm run build`.

- [ ] Add a development mode for the viewer (flag or env var) in which static assets are served with `Cache-Control: no-store` so a browser refresh always picks up the current files.
- [ ] Add an esbuild watch (`npm run watch` or similar) that rebuilds `web_assets/app.js` on every `frontend/src` change, so refresh reflects source edits without touching the running server.
- [ ] Optionally auto-reload the page when the bundle changes.
- [ ] Document the dev workflow: start training once, run the watcher, edit, refresh.

### 5. HTTPS fronting of the viewer (e.g. `tailscale serve`) is rejected by the origin check (found 2026-09-20 while fixing item 3)

`web_session.permits_request` requires `Origin` to equal `http://<host>` and the Host port to match the bound port. A TLS reverse proxy in front of the viewer sends an `https://` origin and usually a different port, so token login and the console `PUT` are rejected. Plain HTTP remote viewing works after item 3; HTTPS remote viewing does not yet.

- [ ] Accept an `https://` origin (and a configurable public origin, e.g. `--public-origin https://mlserver.tailnet.ts.net`) so a TLS proxy can front the viewer.
- [ ] Mark the session cookie `Secure` when the public origin is HTTPS.
- [ ] Add a web test that exercises requests carrying a proxied HTTPS origin.
- [ ] Document the `tailscale serve` setup once it works end to end.

## Done

### 6. FID should evaluate on an interval by default (raised 2026-09-20)

Owner ran `training-runs/start.sh` and saw no FID because both FID metrics in the run's `cifar10.toml` are `trigger = "manual"`, so the manifest records an empty evaluation schedule and nothing ever fires.

- [x] Make snapshot metrics such as FID default to `trigger = "interval"` with a sensible `every_steps` (the public example uses 10,000 for FID50k) and `on_busy = "skip"`, so a recipe that declares an FID metric gets periodic results without extra fields. `trigger = "manual"` stays available as an explicit opt-out.
- [x] Decide the default cadence and evaluation device behavior when a metric omits them; document contention when the evaluation device is the training GPU.
- [x] Update the CIFAR example, the generated/pretrained recipe used by the owner's run, and the docs to reflect the new default.
- [x] Have the viewer and CLI progress output make it obvious when a configured FID metric has no schedule, so a manual-only setup is not silent.
- [x] Add tests for the default schedule and for explicit manual opt-out.

Implemented on branch `worktree-agent-aee640f816c3f60b9` (not yet merged to develop):

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

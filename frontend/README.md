# Local observation UI

A small JavaScript application consumes the same authenticated public JSON/SSE
API as external clients. The browser owns live view reduction in a dedicated
worker using HyperGAN's bundled Rust/WASM kernel. The server supplies bounded
historical bootstrap state and forwards live contributions unchanged.

No CDN, remote fonts, runtime npm installation, React, or alternate numerical
implementation is required. The shipped module is a modular ECharts build with
line charts, axes, rich-text tooltips and Canvas rendering. Explicit bounded PNG
grids render as images; tensor JSON is not interpreted as an image. The artifact
shelf uses the indexed artifact API and stream notifications,
with role, modality, media type, shape, step and safe downloads. An explicitly
requested JSON tensor preview is limited to 64 KiB, eight axes and 4,096 finite
values; it validates the declared shape and shows at most 128 numbers. Unknown
media remain downloadable without decoding or executing them.

## Rebuild

```sh
npm ci --prefix frontend
npm run --prefix frontend build
npm run --prefix frontend check
```

Node 22 was used for local validation. `package-lock.json` pins ECharts 6.1.0,
esbuild 0.28.2 and transitive dependencies. `build.mjs` produces the bundled
`app.js` plus upstream license/notice texts in `src/hypergan/web_assets`;
`--check` compares bytes without changing them. Browser runtime has no npm
requirement. HTML, CSS and the small worker remain readable static resources.

## Develop the UI while training runs

Editing the UI never requires stopping or resuming training. Start the run once
with viewer development mode, leave the watcher running, and refresh the browser:

```sh
hypergan train config.toml --run-dir runs/example --dev
# In another terminal, rebuild on every change under frontend/src:
npm run --prefix frontend watch
```

`npm run watch` uses esbuild's context/watch API and writes exactly the bytes
`npm run build` writes, so `npm run check` still passes afterwards. It rebuilds
`src/hypergan/web_assets/app.js` and the license notices, printing the bundle
size on each rebuild.

In development mode the server reads every browser asset from disk per request
and answers with `Cache-Control: no-store` and no cache validator, so a plain
refresh always shows the current files. It also prints the resolved asset
directory once at startup, and serves the checkout containing `frontend/` in
preference to the installed package, which matters when an editable install
points at a different checkout or worktree. `index.html`, `style.css` and
`view-worker.js` are served directly from `src/hypergan/web_assets` and need no
build step at all; only `app.js` comes from `frontend/src`.

A small same-origin `/dev/reload.js` module polls `/dev/version` once a second
and reloads the page when the served assets change. Both routes exist only in
development mode. See [the viewer documentation](../docs/local-web.md#viewer-development-mode)
for the flags and environment variables.

The [public API contract](API.md) describes paths and envelopes. Data selection,
lineage changes and step ranges request a new bootstrap. Pending history waits
for an SSE notification, with no HTTP polling. Source events and mapped
contributions remain independently readable through the server API.

## Correctness and limits

The worker stages an entire projection frame before committing state and its
cursor. Replay does not double count; gaps require a new bootstrap. Reconnect
uses only the acknowledged application bookmark. Native EventSource's received
message ID is not used as a commit. The browser closes failed EventSources and
reopens explicitly after queued complete frames finish.

There are at most eight selected chart metrics, 2,048 reduced groups, and 128
queued frames / 1 MiB of queued text. Worker requests have a ten-second deadline.
Increasing aligned step buckets can free capacity; four automatic coarsening
attempts are the limit, then the user must narrow the range or selection.
Different metric definitions and attempts stay separate chart series. Earlier
snapshot recovery obeys backend-provided attempt/through-step lineage.

An envelope contains actual first/min/max/last observations. Optional EMA is a
presentation of those visible points, labeled separately from raw values; it is
not a newly published metric or an average of omitted history. Log scale omits
nonpositive points visibly and preserves gaps. Tooltips and the keyboard
accessible latest-value table retain exact numeric values, while summary cards
use compact formatting.

```sh
python -m pytest tests/browser/test_viewer_ui.py -q
```

This dedicated Playwright/Chromium gate uses a real HTTP/SSE fixture: login,
plots, live updates, ACK-based reconnect, pending history without polling,
pretraining server activation, range/selection/search controls, responsive
layout, and atomic worker recovery. Missing Chromium is a failure. Integration
with the real ASGI server is additionally qualified in the server workstream.

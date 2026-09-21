# Human feedback checklist

Running list of feedback from the project owner, gathered while using HyperGAN.
This is a living document: new items are appended as they come up, and each
item is updated as it is addressed. Earlier, already-implemented feedback lives
in [feedback-2026-09-20.md](feedback-2026-09-20.md).

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (link the PR).

## Open

### 19. Training throughput: use the GPU fully, less host blocking if needed (raised 2026-09-20)

Owner: "i'm running hypergan on card 0, it's like 10-12 steps/s. particlegan its
semi-adapted from is 14-15 steps per second ... i want to optimize this to utilize
the gpu as much as possible." Later: "we can do less blocking if we need to, idk
where the cost is coming from."

Where the cost was (measured 2026-09-20 on GPU 0, CIFAR recipe, batch 64):
- ~103 blocking device reads per step: `torch.isfinite(p.grad).all()` per parameter
  tensor plus loss checks before each backward (`src/hypergan/training.py`). Worth
  about 22 ms of a 100 ms step under the shipped backend.
- Whole-graph EMA: 193 separate `lerp_`/`copy_` launches per step (8 ms).
- The recipe's `[training.backend]` determinism/no-TF32 block: +32% throughput when
  matched to the ParticleGAN source flags (TF32, cudnn benchmark, nondeterministic).
- Not the cost: `--server`, progress cadence, `metrics.every_steps`, previews,
  checkpoints, data loading, the prior regularizer (all ≤0.5 ms/step). Models match
  ParticleGAN parameter for parameter.

- [x] Fuse the nonfinite screening into one device-side check per phase and fuse the
  EMA — PR #355 (`2bc0a9d3`, merged locally as `82f9df11`). 69.8 → 61.4 ms/step under
  the shipped backend, 47.0 → 41.2 ms/step under the source backend flags; metrics
  bitwise identical.
- [x] Owner decision: "i dont care about deterministic tbh it should resume fine
  without." PR #356 (merged locally as `a23dfbfd`) ships the example with the source
  backend flags; the strict policy stays documented as an opt-in. Back to back on
  GPU 0: ParticleGAN 19.9 steps/s, HyperGAN 21.9 steps/s. The live run keeps its
  recorded config, so start a new run directory (`training-runs/start-fast.sh`).
- [ ] Apply the same fused screening to `distributed_training._reduce_gradients`
  (per-parameter host reads on every rank) with two-GPU qualification.
- [ ] Beyond parity: both loops are launch-bound (~5,000 launches/step, 38% GPU
  utilization for ParticleGAN itself). Raising utilization means fewer launches
  (CUDA graphs / `torch.compile`) or a larger batch, which changes the recipe.

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

### 18. Resume rejected because CUDA enumerated the other identical GPU; say what differs and warn instead of failing (raised 2026-09-20)

Owner: "I think that should probably be a warning, and more specific on what is failing to load."

Context: restarting a single-GPU run (`hypergan train ... --run-dir train-develop` with
`CUDA_VISIBLE_DEVICES=0`) failed with the bare `ValueError: Resume runtime/topology
differs from checkpoint`. The machine holds two identical NVIDIA RTX A6000s. The
checkpoint was written on `548116b7-9dbe-de58-b3d9-a6e27b0f74ce`; after the restart CUDA
device 0 enumerated as `ed080e41-3193-3755-6756-f3d46c433331` (`CUDA_DEVICE_ORDER` is
unset, so the order is not stable). Only `cuda.uuid` and `cuda.visible_devices` differed
— the model, capability, CUDA/cuDNN/torch versions, dtype and every determinism and TF32
setting matched. `validate_runtime` compared the whole `runtime` dict as one JSON blob,
so it could neither report which field differed nor tell a physical card swap apart from
a real incompatibility.

**Status:** Merged into develop (`e8ada8b0`, merge `9c4985dc`). Implemented on branch `worktree-agent-a71d0612e78de539e` (the
coordinator merges). `src/hypergan/checkpoint_compatibility.py` now flattens both runtime
dicts to dotted key paths and classifies the differences against one explicit set,
`DEVICE_IDENTITY_KEYS = {'cuda.uuid', 'cuda.visible_devices'}`. A difference confined to
that set is which card ran, not what it computes, so it warns and resumes; because any
difference outside the set still rejects, an accepted swap has already proved the model,
the capability and every numerical setting still match. Both call sites
(`single_execution.restore` and `distributed_checkpoints._validate_restore_identity`)
share the behavior. No checkpoint compatibility version bump: the metadata shape is
unchanged and old checkpoints resume as before. Verified with the CPU suite
(`pytest tests/foundation tests/reference`): 980 passed, 2 failed in 18:55, both
environment-only and pre-existing — `test_console_entrypoint` (the known one) and
`test_distribution_contains_only_supported_package`, which both require the installed
wheel that CI builds rather than this machine's editable checkout. Not run: `tests/cuda/`.

Where this lives today:
- `src/hypergan/checkpoint_compatibility.py`: `_flatten`, `_differences`, `_describe`,
  `DEVICE_IDENTITY_KEYS` and `validate_runtime(saved, current, *, warn=None)`, which
  returns the warning messages and sends each to `warn` or to `warnings.warn(...,
  RuntimeWarning)`.
- `src/hypergan/single_execution.py` `restore` returns them on `Restored.warnings`.
- `src/hypergan/run_controller.py`: `Restored` gained a `warnings` field; `run_resume`
  appends them to `manifest['warnings']` (the existing durable channel, already used for
  `recovery_reasons`), records `manifest['resume_warnings']`, and the `resume` event in
  `events.jsonl` carries them as `warnings`. The console path is the existing
  `warnings.warn(..., RuntimeWarning)` on stderr, raised at the moment of detection;
  there is no separate console warning channel in the CLI for run-time (as opposed to
  config-time) warnings.

- [x] A rejection names every differing key path with the saved and the current value,
  e.g. `cuda.name: saved "NVIDIA RTX A6000", current "NVIDIA GeForce RTX 4090"`. Nested
  metadata is flattened to dotted paths, an absent key reads `absent`, and a long value
  is truncated so one field cannot flood the message. The message ends with the action:
  resume with the runtime that wrote the checkpoint, or start a new run directory.
- [x] A different physical GPU of the same model and capability warns and continues. The
  warning names the model, both UUIDs and the remedy (`CUDA_DEVICE_ORDER=PCI_BUS_ID` plus
  `CUDA_VISIBLE_DEVICES`) for pinning one card across restarts.
- [x] Everything else stays a hard failure: device type, dtype, world size,
  torch/CUDA/cuDNN versions, the determinism, TF32 and matmul settings, the GPU model and
  capability, python/numpy and platform/machine.
- [x] The warning reaches the owner in three places: stderr at resume time, the run
  manifest (`warnings` and `resume_warnings`) and the `resume` event in `events.jsonl`.
- [x] Tests: `tests/foundation/test_runtime_compatibility.py` (36 torch-free unit tests:
  identical dicts pass silently, a UUID-only swap warns and names both UUIDs, the warning
  sink replaces `warnings.warn`, each runtime setting and each top-level field rejects
  with its key path and both values, absent keys are named, long values truncate) and two
  end-to-end cases in `tests/reference/test_checkpoint_compatibility.py` (a resume onto
  the other identical card warns, completes, matches the baseline numerics and is
  recorded on the run; a different card model rejects naming `cuda.name` and
  `cuda.capability` and leaves the run untouched).
- [x] [docs/recovery.md](../docs/recovery.md) describes the specific rejection message and
  the one qualified difference, with the `CUDA_DEVICE_ORDER` remedy.

### 17. Preview retention: default 128 with thinning instead of keep-all, and fix inherited `preview_keep` (raised 2026-09-20)

Owner: "ok lets fix the preview_keep bug. i think it should be set to idk, 128(?) by default. i also think we'll want to prune the middle when we prune, like go from once every 500 to once every 1000. stuff like that. so it's at most N but prunes when it gets too big in chunks."

Supersedes the keep-everything default shipped for item 8. Item 8 Follow-up 2 (the manifest that inherits the old default of 20) is fixed here as well.

**Status:** Implemented in commit e02fbf58, merged to develop in 015a0473. Default
`--preview-keep` is now 128 (`all` still keeps everything, `N` is an explicit bound).
Retention is thinning, not drop-the-oldest: `previews.thin` keeps the first sample of
the run and the latest unconditionally, keeps the newest samples dense inside a window
that advances in whole 16-sample steps, and retains older samples at a spacing that
doubles each time the run outgrows the bound (every 500 steps becomes every 1,000, then
every 2,000). Both the doubling and the window advance only ever remove, so the retained
set is nested and deterministic from what is on disk; over 1,200 simulated publications
at 128 there were 76 prune events in chunks of 8 to 64. Pruning is off the training path:
the index is rewritten first, expired directories are renamed to `.expired-<name>` (the
reindex scan skips the dot prefix) and deleted by a daemon worker; `drain_pruning()` and
an atexit hook drain it, and leftovers from a crash are swept on the next publication.
The manifest records `preview_keep_source: "explicit" | "default"`, and the shared
`run_controller.resolve_preview_keep` (used by `run_train`/`run_resume` and
`prepare_train`/`prepare_resume`) inherits a stored bound only when it was explicit; a
manifest without the marker resumes into the current default. Verified on merged develop:
703 passed across previews, foundation, core CLI, observation, replicated observer
acceptance, web and browser suites, with the one known environment failure
(`test_console_entrypoint`). The viewer bundle is unchanged. Not run: `tests/cuda/`.
The owner's live run still needs a restart to pick up the new default; generations
already pruned are gone.

- [x] Default `--preview-keep` becomes 128. `--preview-keep all` stays available for keep-everything; `--preview-keep N` stays an explicit bound.
- [x] Pruning thins the middle instead of dropping the oldest: when a run holds more than N generations, older samples are thinned by halving their density (a run publishing every 500 steps ends up with every 1,000, then every 2,000, ...), in a chunk, so the next prune only happens after many more publications. The first sample of the run is never pruned, and the most recent samples stay dense. Thinning is monotonic: a later prune only ever removes generations, never reorders or "un-thins", and the result is deterministic from the index.
- [x] A manifest that recorded `preview_keep` only because it was the default at the time must not pin a resumed run to that stale value. Record whether the bound was explicit; `hypergan train` on an existing run directory and `hypergan resume` behave the same way; an explicit `--preview-keep N` still carries across resume.
- [x] Tests: thinning keeps the first and latest samples and halves spacing; repeated publications never exceed N; an old manifest with `preview_keep: 20` and no explicitness marker resumes into the new default; an explicit bound still resumes as explicit.
- [x] Docs (`docs/image-previews.md`, README, `docs/recovery.md` if it mentions retention) describe the thinning policy and the default in plain words.

Owner addendum (2026-09-20): "it should be smooth and not interrupt the main workflow, happening in the background with the ui updating after."

- [x] Pruning never stalls training: the index is rewritten first (readers only ever see retained generations), then the expired directories are deleted by a background worker owned by the run and drained at shutdown. The next publication does not wait on an in-flight prune and never re-indexes a directory pending deletion; a directory left behind by a crash mid-prune is dropped again by the same deterministic policy on the next publication.
- [x] The viewer picks up the new retained set after the prune: `preview_count` and the index reflect the post-prune count.
- [x] Test: the trainer-side publish call returns without waiting on directory removal, the directories are gone once the worker drains, and the index never references a directory the worker deletes.

### 11. Steps per second as a training metric (raised 2026-09-20)

Owner: "we should have steps/s during training as a metric."

**Status:** Implemented in commit 0ec7ce79, merged to develop in ed9632c5. `throughput/steps_per_second` is a
built-in training-scope scalar published on every complete update, smoothed over
the last 20 updates. It is listed in the catalog, part of the viewer's default
Learning curves selection, shown in the headline stats next to Completed step,
and printed on the CLI progress line as `123.4 steps/s`.

- [x] Publish a `throughput/steps_per_second` (name to taste) scalar on the training stream, smoothed over a short window so it charts cleanly, and list it in the catalog so it appears in Learning curves and can be selected like any loss.
- [x] Show the current value in the headline stats next to Step.
- [x] Include it in the CLI progress line.
- [x] Tests for the value across a few updates and for resume (the window restarts, no negative or infinite values).

Where this lives now:
- `src/hypergan/metrics.py` holds `Throughput` (a `THROUGHPUT_WINDOW = 20`
  trailing window over measured update durations) and the catalog entry. A
  window whose durations sum to zero — a clock too coarse to separate two
  boundaries, which is what the first update can look like — returns `None`, so
  the metric reports `unavailable` instead of dividing by zero or publishing an
  infinity. `select_metrics(..., progress)` accepts the controller-measured
  scalars for the same boundary and never fabricates a missing one.
- `src/hypergan/run_controller.py` owns one `Throughput` per attempt, so resume
  restarts the window rather than averaging across the idle gap, and mirrors the
  current rate into `manifest['steps_per_second']`.
- `frontend/src/app.js` `defaults()` selects it after the losses; the
  **Steps per second** tile in `web_assets/index.html` reads the run payload, so
  it updates from stream heartbeats without selecting the metric.
- Tests: `tests/foundation/test_metrics_config.py`
  (`test_throughput_window_averages_recent_updates_without_dividing_by_zero`,
  `test_progress_scalars_publish_throughput_time_and_samples_seen`),
  `tests/foundation/test_bounded_cli_output.py`,
  `tests/reference/test_core_cli.py::test_cli_stop_resume_and_json_progress`,
  `tests/browser/test_viewer_ui.py::test_headline_stats_report_throughput_training_time_and_samples`.

### 12. Time spent training (raised 2026-09-20)

Owner: "we should have time spent training."

**Status:** Implemented in commit 0ec7ce79, merged to develop in ed9632c5. The run manifest carries
`training_seconds`, the cumulative wall clock spent inside training attempts.
Each attempt reads it as its baseline and adds only its own elapsed time, so a
resumed run continues the total and the gap between two commands is never
counted. It is charted as `timing/training_seconds` and displayed as `1h 23m` in
the headline stats and on the CLI progress line.

- [x] Track cumulative wall-clock training time across attempts (resume adds to it, idle time between attempts does not count) and store it in the run manifest.
- [x] Show it in the headline stats and the CLI progress line as a human duration.
- [x] Publish it on the training stream so it can be charted against step if useful.
- [x] Tests, including that resume continues the total rather than resetting it.

Where this lives now:
- `src/hypergan/run_controller.py` validates the stored baseline at attempt
  start (a finite, nonnegative number or the attempt refuses to run) and
  refreshes `manifest['training_seconds']` on every manifest publication and
  every update, beside the existing per-attempt `seconds`.
- `src/hypergan/bounded_cli_output.py` `human_duration()` formats `45s`,
  `12m 05s` and `1h 23m`; `frontend/src/app.js` `duration()` matches it exactly
  for the **Time training** tile.
- `src/hypergan/web_service.py` publishes `training_seconds` (with
  `samples_seen`, `steps_per_second` and `global_batch_size`) in the run payload,
  described in `src/hypergan/web_schema.py` and `frontend/API.md`.
- `tests/reference/test_core_cli.py::test_cli_stop_resume_and_json_progress`
  asserts that the resumed total equals the first attempt's total plus the second
  attempt's own `seconds`, which is exactly the idle time being excluded.

### 13. Number of samples seen (raised 2026-09-20)

Owner: "we should have number of samples seen."

**Status:** Implemented in commit 0ec7ce79, merged to develop in ed9632c5. `samples_seen` is now in the run
manifest and the headline stats, on the CLI progress line (`12,800 samples`) and
chartable as `progress/samples_seen`. `step * batch_size` was already correct
under gradient accumulation and a world size larger than one: `training.batch_size`
is the global batch, which those executions split into per-rank shards and
microbatches without changing how many real examples a completed update draws
(`distributed_training.py` derives `local_batch_size` and `microbatch_size` from
it, and `replicated_execution.py` checks every update against that profile). The
definition is now stated in the catalog entry and in `docs/observation.md`.

- [x] Surface `samples_seen` (already emitted on every `train` event as `step * batch_size`) in the headline stats and the CLI progress line, and make it correct when gradient accumulation or a world size larger than one changes the effective batch.
- [x] Tests.

Where this lives now:
- `src/hypergan/run_controller.py` computes it from the execution's reported
  `global_batch_size` when the update carries one, falling back to
  `training.batch_size`, and stores it in the manifest with the batch size it
  used.
- The **Samples seen** tile shows exact counts below 100,000 and a compact
  `1.2M` above it, with the exact value in its tooltip, so a long run cannot
  widen the headline row.

### 14. Header: unexplained hash next to the status, and "RUNNING" should read "TRAINING" (raised 2026-09-20)

Owner: "theres a hash at the top of the page idk what it is, next to 'RUNNING' which should say 'TRAINING'."

- [x] The hash is the run id (`<code id="run-id">`). Either label it ("Run id …", with a copy affordance) or move it out of the heading into the run details; it should not be a bare hash.
- [x] Map the manifest status to user wording in the badge: `running` displays as "Training"; check the other statuses (`complete`, `failed`, `stopped`, …) read well too.
- [x] Update the browser UI test that asserts the badge text.

**Status:** Implemented in commit b81823fc, merged to develop (fast-forward).

The status badge now stands alone under the run name and reads in the words the
owner uses: `running` and `training` both show **Training**, `initializing`,
`pending` and `starting` show **Starting**, `complete`, `failed`,
`stopped`/`cancelled` and `interrupted` get their own plain word, and any status
the map does not know falls back to the raw value capitalised. The raw manifest
status stays on the badge as `data-status` for styling and tests.

The run id is no longer a bare hash glued to the badge. It sits on its own line
below it, captioned "Run id", with a **Copy** button: a clipboard write when the
browser allows one, otherwise the id is selected so Ctrl+C takes it, and either
way a short status says which happened. The message is cleared only when the id
itself changes, so a manifest refresh does not wipe it.

Where this lives now:
- `src/hypergan/web_assets/index.html` run heading (`.run-subtitle` badge,
  `.run-identity` caption, `#copy-run-id`, `#run-id-status`) and the matching
  `.run-identity` rules in `src/hypergan/web_assets/style.css`.
- `frontend/src/app.js` `RUN_STATUS_LABELS` / `statusLabel`, `updateRun`, and the
  `#copy-run-id` click handler.
- Tests: `tests/browser/test_viewer_ui.py`
  (`test_status_badge_reads_in_plain_words_beside_a_labelled_run_id`, which
  parametrises the mapped and the fallback wording and exercises the copy
  affordance).

### 15. Remove the CLI interval controller from the viewer (raised 2026-09-20)

Owner: "we don't need the cli interval controller."

- [x] Remove the "CLI progress every N steps / Save CLI interval" form from the viewer.
- [x] Decide whether the `/runs/{id}/console` API, the `console` control capability and `console_settings.py` stay for scripted use or go with it; keep the `--progress-every` CLI flag either way. Remove dead code and tests, update `frontend/API.md` and docs.

**Status:** Implemented in commit b81823fc, merged to develop (fast-forward).

The form is gone from the viewer, and so is the API behind it: `GET`/`PUT
/api/v1/runs/{run_id}/console`, its OpenAPI entry and the `console` control
capability (`controls` is now an empty list, kept so consumers still find the
field). Nothing but that endpoint ever changed the cadence while a run was live,
so the mid-run polling went with it: `ConsolePolicy` reads `console.json` once,
at the attempt's first delivery, instead of taking a background read slot four
times a second.

`console_settings.py` stays, because the trainer — not the viewer — is its other
user: `--progress-every N` is persisted to `console.json` at the start/resume
boundary, so a later attempt without the flag inherits the interval, the same
way `--preview-keep` is inherited. The flag is unchanged. Hand-editing
`console.json` mid-run no longer reaches the running attempt; it takes effect at
the next resume.

Where this lives now:
- `src/hypergan/web_server.py`: the `console_settings` handler, its route, its
  OpenAPI path and the `console` capability are removed.
- `src/hypergan/console_settings.py` `ConsolePolicy` resolves once
  (`resolved`), no longer importing `BackgroundPoll`; `background_poll.py` stays
  for checkpoint request discovery in `run_controller.py`.
- `src/hypergan/web_assets/index.html` (`#console-settings` form removed) and
  `frontend/src/app.js` (`consoleSupported`, the `/console` fetch and the submit
  handler removed).
- Tests: `tests/web/test_web_service.py`
  (`test_console_control_is_no_longer_served`),
  `tests/web/test_public_origin_proxy.py` (the origin gate is checked on a run
  read now that there is no write route), `tests/foundation/test_console_settings.py`
  (`test_cli_interval_persists_across_attempts_and_a_flag_replaces_it`,
  `test_unreadable_settings_warn_once_and_retain_the_default`);
  the viewer control tests in `tests/browser/test_viewer_integration.py` and the
  live-refresh test in `tests/foundation/test_background_poll.py` are deleted.
- Docs: [observation](../docs/observation.md), [local web](../docs/local-web.md),
  [recovery](../docs/recovery.md), README. `frontend/API.md` never listed the
  endpoint.

### 16. Snapshot evaluations panel: FID out of Learning curves, one tile per metric, better empty state (raised 2026-09-20)

Owner: "We don't want the FID score in learning curves section. It should only be in snapshot evaluations. Also there are tiles in snapshot evaluation that are just really wordy. Like there are two fid_smoke tiles. I like it otherwise. It does need a better initial state when no readings are available."

- [x] Stop merging evaluation metrics into the Learning curves chart, metric list and default selection; they belong only in Snapshot evaluations.
- [x] One tile per metric: fold the schedule card (status, cadence, next step, device, skips) and the result card (chart, collapsed details) into a single card per metric id, with much shorter wording.
- [x] Design the empty state: a metric with a schedule but no result yet shows its chart area with "No evaluations yet · next at step N" (or "manual"), not a wall of text; a run with no snapshot metrics keeps hiding the panel.
- [x] Update the browser tests for the panel.

**Status:** Implemented in commit 48b5e919, merged to develop in 3f2edebd.

Where this lives now:

Learning curves is the training stream and nothing else. `state.evaluationMetrics`
and `state.evaluationResults` are gone from `frontend/src/app.js` along with the
callback that filled them, so an evaluation id can no longer reach
`metricDefinitions()`, `defaults()`, the metric list and its count, the selection
filter, the chart series, the data table or the coverage line. `evaluationShelf`
is constructed with no `changed` callback; the evaluations module already loads
its own results, so nothing was lost by cutting the bridge. The data table's
summary and caption say "training values" now that no evaluation row reaches it.

Snapshot evaluations is one tile per metric id. `#evaluation-schedules` is gone
from `src/hypergan/web_assets/index.html`; `frontend/src/evaluations.js` renders a
single `#evaluation-items` list from the union of the catalog's snapshot
definitions and the loaded results, so a metric that is configured, published, or
both is one `li` either way. A tile is its label, the metric id only when it
differs from the label, one status line, the chart, the compact failure and
cancellation lines, and one collapsed **Details**. The status line is the cadence
and the schedule state and nothing more -- `Every 10000 steps · next at 20000`,
`Manual`, `Every 10000 steps · running since step 10000`, `Manual · last at step
6`, `Every 10000 steps · disabled`. Device, the busy policy, skipped counts, the
last recorded reason and the per-result rows moved into Details, which is where
the export links and protocol documents already were.

A metric with no result yet replaces the chart with one quiet line:
`No evaluations yet · first at step 10000`, `No evaluations yet · next at step
30000` once a schedule is live, `... when training resumes` when the run is not
training, or `No evaluations yet · run hypergan evaluate` for a manual metric.
The next step is printed in exactly one place: the status line once a metric has
a history, the empty line before that. A run with no snapshot metrics and no
evaluation streams keeps the panel hidden, and the all-manual notice above the
tiles is unchanged.

Tests in `tests/browser/test_viewer_integration.py` assert one `li` per metric,
the status-line text through running/failed/stopped/disabled, both empty-state
lines, that device and busy policy are only behind Details, and that "Snapshot
quality" appears in neither `#metric-list` nor `#charts` while `#metric-count`
counts training scalars only. The item 9 chart assertions are unchanged.


### 8. Sample slider should cover the whole run, not the last N (raised 2026-09-20)

Owner: "the viewer shows the last N images but it should really show all of them. if it's on the last one and a new sample comes in it can update and stay on the last one. but i want someone to be able to slide from the beginning of their training to the end."

- [x] Stop pruning image history by default: keep every published preview (or at least every image grid) for the life of the run, so the slider reaches back to step 0. Keep a bound only for disk-heavy tensor payloads if one is needed, and make any retention an explicit opt-in.
- [x] Lift the viewer/API caps that assume a small preview count (`previews/index.json` is currently rejected above 100 entries; the artifact list is rebuilt from it) so a long run with thousands of samples still loads quickly.
- [x] Slider behavior: when positioned on the latest sample and a new one arrives, advance to the new one; when positioned on an earlier sample, stay put and do not jump.
- [x] Tests for retention-off, index size, and the follow-latest / stay-put slider behavior.

**Status:** Implemented in commit 2b601cf1, merged to develop in d99d2164.

Retention is now opt-in. `hypergan.previews.KEEP_ALL = 0` is the default `keep`,
so a run accumulates every published generation (tensor payload and both PNG
grids together) for its whole life, and the viewer's slider spans the run from
its first sample to its latest. `--preview-keep N` still deletes the oldest
generations when a disk is small, and `--preview-keep all` returns a run to
keeping everything; resume inherits whichever value the run recorded, so a run
started before this change keeps its stored bound until a resume overrides it.
The tensor payload was kept under the same rule as the grids rather than pruned
separately: a generation is published as one atomic directory whose index entry
points at a readable `preview.json`, and splitting that would have given
"retained" two meanings for no gain the owner asked for. Each generation is
bounded at 2 MiB of JSON plus its PNGs, which a bound can still cap.

Nothing assumes a short history any more. `previews/index.json` is rebuilt from
the records it already publishes and only reads a manifest for a generation the
index does not name, so a publication costs one directory listing instead of one
manifest read per retained sample. The run manifest, which is rewritten and
fsynced on every publication, now repeats only the most recent 16 records plus a
`preview_count`; the index remains the whole history. The service accepts an
index of up to `web_service.MAX_PREVIEWS = 4096` generations (was 100) read under
a 16 MiB budget, and the viewer's per-group version cap matches it. A 100k-step
run at `--preview-every 100` publishes 1,000 generations, well inside that.

The slider now pins a sample by the step it was taken at, not by its slider
position or its artifact ID. With no pin it follows the latest, so a new sample
advances it; once moved to an earlier sample it holds that sample even as newer
ones arrive and pruning shifts every position, and **Latest** resumes following.

Where this lives now:
- `src/hypergan/previews.py` `KEEP_ALL`/`DEFAULT_KEEP`, `_indexed_generations`
  and the retention branch in `_publish_preview`.
- `src/hypergan/cli.py` `_preview_keep` (a count or `all`);
  `src/hypergan/run_controller.py` `MANIFEST_PREVIEWS` and `_controls`;
  `src/hypergan/web_service.py` `MAX_PREVIEWS` / `PREVIEW_INDEX_BYTES`.
- `frontend/src/app.js` `versionKey` and the pin lookup in `renderSampleGroup`.
- Tests: `tests/reference/test_image_previews.py`
  (`test_retention_keeps_every_generation_until_a_bound_is_requested`,
  `test_index_reuses_published_records_instead_of_rereading_manifests`),
  `tests/web/test_web_service.py` (long index accepted, bounded above
  `MAX_PREVIEWS`), `tests/browser/test_viewer_ui.py`
  (`test_sample_slider_follows_latest_and_holds_an_earlier_pick`),
  `tests/reference/test_core_cli.py`
  (`test_preview_keep_accepts_a_count_or_the_whole_run`).
- Docs: [image previews](../docs/image-previews.md), [observation](../docs/observation.md),
  [replicated observation](../docs/replicated-observation.md), README.

Follow-up (2026-09-20): after the merge the owner moved the `g` slider to the start and still saw only the 20 most recent samples. "i want to be able to slide all the way back to the beginning of time. it may need refactoring to be on-demand depending if the browser is caching."

- [x] Investigate against the owner's live run: whether the run is still on the pre-change retention (a run started before this change keeps `preview_keep = 20` in its manifest until resumed with `--preview-keep all`), whether already-pruned generations are simply gone, and whether the viewer or API still caps the history it returns.
- [x] If the viewer loads the whole history up front, consider loading version metadata eagerly but images on demand as the slider moves, so a long run does not fetch thousands of PNGs.

**Follow-up status (2026-09-20):** the live run is still pruning; the shipped
code is not. `train-develop` was started at 17:41:32 from commit `aa227704`, and
the retention change was committed at 17:49:35 and merged at 17:50:41 — eight
minutes later. The package is an editable install, but a running Python process
does not reload its code, so that process is still the old build and still
prunes. Its manifest records `preview_keep: 20` with a full 20-entry `previews`
list and no `preview_count`, which is the pre-change shape, and
`previews/index.json` says `"keep": 20, "retention": "bounded"` over exactly 20
entries: sequences 31–50, steps 15500–25000. Twenty generation directories exist
under `previews/`. Sequences 1–30 (steps 500–15000) were deleted by the old code
as each new sample was published and cannot be recovered; the slider is showing
everything that is left on disk. Restarting is not enough on its own: `run_resume`
reads `preview_keep` back out of the manifest whenever the flag is absent, so a
plain resume would inherit the stored `20` forever.

The owner should stop the current process and resume with retention turned off,
from `/home/martyn/dev/hypergan/training-runs`:

```
hypergan resume train-develop --config ./cifar10-pretrained-20260920/cifar10.toml \
  --preview-keep all --server --dev --checkpoint-every 1000 --preview-every 500 --progress-every 100
```

`--preview-keep all` rewrites the manifest to `0`, so every sample from that
point on is kept and the slider grows to span the rest of the run. The steps
before 15500 stay gone. `--server` and `--dev` are per-attempt and must be
restated; the intervals are inherited and are repeated here only to match
`start.sh`.

No code defect was found, so nothing in `src/` or `frontend/` changed. The whole
path was checked end to end and carries no residual cap at 20 or 100:
`previews.py` keeps every generation at the default `keep`, the service accepts
`MAX_PREVIEWS = 4096` entries under a 16 MiB budget (the owner's index measures
1,985 bytes per entry, so 4,096 entries is about 8.1 MiB), the `artifacts` SSE
event carries only a revision digest, `/artifacts` is unpaginated and returns
every record, and the viewer's slider is `min = 0`, `max = versions - 1` up to
`MAX_SAMPLE_VERSIONS = 4096`. Two tests now pin that end to end:
`tests/web/test_web_service.py`
(`test_default_retention_publishes_a_whole_run_history_to_the_viewer`) publishes
60 generations through the real `publish_preview_payload` with no `keep`, and
asserts 60 directories survive and the service lists all 180 records; and
`tests/browser/test_viewer_integration.py`
(`test_slider_reaches_the_first_sample_of_a_whole_run_history`) drives the real
browser over the same 60 and asserts the slider reads `max="59"` and that
**Home** reaches "Version 1 of 60 · step 500".

Nothing is cached and nothing is prefetched. Every viewer response gets
`Cache-Control: no-store` — the ASGI wrapper in `web_server.py` adds it to any
response that did not state its own policy, which covers `/artifacts` and every
PNG, and `--dev` sets it again on the static assets. The bundle the running
server hands the browser is the current one: `--dev` was launched from
`training-runs`, which is not a checkout, so assets fall back to the packaged
`hypergan.web_assets` — which the editable install resolves to the checkout's
`src/hypergan/web_assets/app.js`, byte-identical to the committed bundle and
already carrying the 4096 cap. On-demand loading is already how the viewer
works: `renderSampleGroup` builds one `<img>` per group and `show()` swaps the
body as the slider moves, so a 60-version history costs 4 artifact requests, not
120. The browser test asserts that count, so a future prefetch would fail it.
The only cost of the blanket `no-store` is that scrubbing back over a sample
refetches its PNG (about 48 KB each over loopback in this run); making immutable
digest-addressed artifacts privately cacheable would smooth that, but it is a
deliberate change to the viewer's cache posture and nobody has asked for it.

Follow-up 2 (2026-09-20 18:41): the owner restarted training at 18:39 on the new code (manifest `source.hypergan_commit` = aeabd95f) using `start.sh`, i.e. `hypergan train ... --run-dir train-develop` on the existing run directory, and the `g` slider still reads "Version 1 of 20 · step 26,000". Read from the run: `manifest.preview_keep = 20`, `preview_count = 20`, index `keep = 20` with 20 entries (steps 26500–36000), 20 generation directories on disk. So the new code inherits the `20` that the old default wrote into the manifest, and keeps pruning. Owner: "item 0 is supposed to be the beginning of time."

Owner decision (2026-09-20, after compaction): the default is no longer keep-everything but 128 with thinning; the inheritance fix moved into item 17 with that change. The three boxes below are tracked there.

- [x] Superseded by item 17: only inherit an explicit `preview_keep`; treat a manifest without an explicitness marker as default.
- [x] Superseded by item 17: `hypergan train` on an existing run directory and `hypergan resume` behave the same way.
- [x] Superseded by item 17: tests for old-manifest resume and explicit-bound resume.

### 9. FID (snapshot evaluations) should be a chart, not a wall of text (raised 2026-09-20)

Owner: "on snapshot evaluations FID should be a graph like the metrics, different x tho ofc. right now it's a wall of text. it may be a graph eventually, maybe it's just a graph with one point atm."

**Status:** Implemented in commit 6f364c38, merged to develop in aa227704 (frontend only,
no API change). The **Snapshot evaluations** panel now groups results by metric
instead of by stream: one echarts line chart per scalar metric, evaluated source
step on the x axis, one visible point per evaluation. Failed and cancelled
evaluations are single status lines under their metric's chart, and every
per-result field moved into one collapsed list per metric.

- [x] Plot each scalar snapshot metric (FID and friends) as a line chart with source step on the x axis, one point per completed evaluation, using the same chart look as the training metrics.
- [x] A single result is a chart with one point, not a text card; failed/cancelled evaluations show as status, not as text walls.
- [x] Keep the per-result details (duration, device, sample count, status) reachable but collapsed.
- [x] Tests for the chart with one point and with several points across steps.

Where this lives now:
- `frontend/src/chart.js` holds the one echarts registration, the shared colour
  palette and `chartStyle()` (grid, axes, tooltip). `frontend/src/app.js` and
  `frontend/src/evaluations.js` both draw with it, so the evaluation charts and
  the training curves are read the same way. That is the only change to
  `app.js`: its chart option literal became `chartStyle(log)`.
- `frontend/src/evaluations.js` groups completed results by metric ID across
  streams, keeps one persistent card, chart instance and results list per metric,
  and redraws a chart only when that metric gained a result. Series stay separate
  per definition hash and protocol digest, markers are always drawn (one result is
  one visible point), and a new stream extends the chart in place without
  reducing or reordering history. Histogram metrics keep their SVG bar plot and
  bin table, now inside the same per-metric card.
- Failed/cancelled results render as one `.evaluation-status` line each (source
  step or "Source position unknown", status, recorded reason clamped to one line).
  Streams that fail to load keep their own "Evaluation unavailable" entry.
- The collapsed `details.evaluation-results` per metric holds the results table
  (source step, value, status, seconds, device, samples, attempt, evaluation) plus
  one `details.evaluation-result` per result with its raw export link, histogram
  plot and, behind one more expansion, its protocol document.
- The 64-stream bound is unchanged and deliberate: the server admits at most
  `MAX_STREAMS = 64` observation streams (`src/hypergan/web_service.py`), so
  raising the client slice would not show more results.
- Tests: `tests/browser/test_viewer_integration.py`
  `test_snapshot_chart_draws_one_point_and_extends_across_steps` (one point, its
  painted marker, then three steps plus a cancellation arriving live) and the
  updated `test_snapshot_scalar_histogram_failure_discovery_and_export`,
  `test_evaluation_metrics_sort_snapshots_preserve_repeats_and_protocols` and
  `test_cancelled_evaluation_retains_source_and_export_without_failure_or_value`.
- Docs: [local web viewer](../docs/local-web.md), [image FID](../docs/image-fid.md)
  and `frontend/API.md`.

### 10. Clarify or remove the "sample - tensor" artifact (raised 2026-09-20)

Owner: "theres a 'sample - tensor' that i'm not sure what it's supposed to be or how to use it. lets clarify or remove it."

- [x] Decide whether the raw tensor preview (the JSON payload behind every image grid, plus the final sample) earns a place in the viewer. If kept, label it by what it is (for example "g raw tensor, step N") and say what it is for; if not, hide it from the artifact list by default and keep only the download.
- [x] Make sure the image grid, not the tensor, is what a user sees first under each sample name.
- [x] Update docs/image-previews.md and the viewer test that covers the artifact list.

**Status:** Implemented in commit 6b8bc372, merged to develop (fast-forward). Kept, renamed and folded into the picture it belongs to. The API is
unchanged apart from one additive marker: the run's finished sample now carries
`provenance.final: true`, so the viewer can name it instead of showing one more
anonymous tensor. When a grid exists for the same name and step, its tensor is no
longer a card of its own: it is a secondary **Download raw tensor (JSON, shape
...)** action on the image card with one line saying it is the same sample as
numbers. A recipe with no images (the numerical path) keeps its tensor card,
labelled "Raw generator output (JSON numbers) · Step N" and explaining that
"Preview numbers" shows the first values; the final sample reads "Final sample ·
raw generator output (JSON numbers)". The generic "sample · tensor" wording is
gone from sample cards.

Where this lives now:
- `src/hypergan/web_service.py` marks the final sample with `provenance.final`; preview and grid records are otherwise untouched.
- `frontend/src/app.js` `foldRawTensors` attaches a generation's tensor to the image version of the same name and step, `sampleKind` names each card, and `rawTensorLink`/`note` render the download and its explanation.
- Copy lives in `src/hypergan/web_assets/index.html` (shelf intro) and `docs/image-previews.md` ("The raw tensor behind a picture"); `frontend/API.md` documents the pairing and the `final` marker.
- Tests: `tests/browser/test_viewer_ui.py` (image run folds the tensor; numerical run names its tensors and final sample), `tests/browser/test_viewer_integration.py` (real published generation, tensor downloads from the image card), `tests/web/test_web_service.py` (`provenance.final`).


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

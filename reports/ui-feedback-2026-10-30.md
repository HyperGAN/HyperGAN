# Web Observatory UI Feedback — 2026-10-30

Captured during live 2× RTX A6000 CIFAR-10 (`cifar10-transgan`) validation on `0.0.0.0:8765`.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done.

## Checklist

- [ ] **1. Index page: only show `steps/s` for active runs**
  - Currently Stopped / Completed / Failed rows still display their last recorded `it/s` (`src/hypergan/web_assets/runs.js`, `src/hypergan/web_server.py`).
  - Show `—` unless the run is actively running.

- [ ] **2. Index page: remove `Checkpoint`, `Sample`, and `Stop` row buttons**
  - Remove `[data-action]` buttons (`Checkpoint`, `Sample`, `Stop`) and action-handler clutter from the index table (`src/hypergan/web_server.py`, `src/hypergan/web_assets/runs.js`, `src/hypergan/web_assets/runs.css`). Keep run controls on the individual run show page.

- [ ] **3. Show GPU / node topology count**
  - Surface how many GPUs and nodes a run is executing on (e.g. `2 GPUs`, `8 GPUs · 2 nodes`, `1 GPU`, `CPU`) from `manifest['execution']` / `manifest['runtime']` in `read_run_summary()` (`src/hypergan/web_server.py`), displayed on both the index table and the run show page.

- [ ] **4. Fix 3-second refresh blink on index page**
  - Every 3s background poll in `src/hypergan/web_assets/runs.js` currently:
    1. Rewrites `thumbCell.innerHTML = '<a ...><img .../></a>'`, destroying and re-fetching the `no-store` thumbnail `<img>` element every 3 seconds.
    2. Toggles `byId('refresh').disabled = true` (`.btn:disabled { opacity: .6 }`) and flashes `'Refreshing…'` text in `#refresh-status`.
  - Update DOM nodes in place only when values/step change, preserve existing `<img>` elements unless thumbnail revision changes, and keep background polling silent without disabling the manual `Refresh` button.

- [ ] **5. Show page: display `samples/sec`**
  - Add `Samples / sec` (`steps_per_second × global_batch_size`, active runs) to the top summary stats on the run show page (`src/hypergan/web_assets/index.html`, `frontend/src/app.js`).

- [ ] **6. Surface `batch_size` clearly**
  - Batch size is currently nowhere visible (`#samples-batch` only says literal `"Updates × global batch"`).
  - Show global batch size (and per-GPU local batch size on multi-GPU runs, e.g. `Batch 256 (128 / GPU)`) in the run show page summary stats (and index metadata).

- [ ] **7. Replace `Model` tab with an inline page section**
  - Remove the `Metrics` / `Model` tab bar (`nav.tabs` in `src/hypergan/web_assets/index.html` and tab-switching state in `frontend/src/app.js`).
  - Render the Model summary (architecture, HNDL layers, formulation, losses, hyperparameters) directly as a regular section on the show page.

- [ ] **8. Drop `steps/sec` (`throughput/steps_per_second`) as a default chart**
  - Remove `throughput/steps_per_second` from the default selected learning-curve charts in `frontend/src/app.js` (throughput is already in the summary stat cards; keep default charts focused on training/health metrics).

- [ ] **9. Remove `← All Runs` header link on show page**
  - Remove `<a class="nav-all-runs text-link" href="/">← All Runs</a>` in `src/hypergan/web_assets/index.html` since the `HYPERGAN / EXPERIMENTS` masthead brand link already navigates back to `/`.

- [ ] **10. Unify background & visual styling between index and show pages**
  - `src/hypergan/web_assets/runs.css` (`#0b0d13` / `#131722` / `#38bdf8`) and `src/hypergan/web_assets/style.css` use mismatched backgrounds, mastheads, card surfaces, badges, and typography.
  - Align both pages onto one shared design system (background, masthead, surfaces, borders, badges, and typography).

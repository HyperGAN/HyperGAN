# Usability plan: one experiment file, runs that report themselves, a viewer that stops (2026-10-11)

Owner request: "there's an http server that starts but doesn't stop. it's kinda weird
usability. also whenever we start an experiment in ~/dev/hypergan/training-runs we
usually have a folder, config file, and shell file per experiment ... the shell and
config is sorta annoying to need both. the original didn't have this problem but it had
other usability problems. ... the goal of the project is to easily train and monitor GAN
training across clusters, multiple gpus, or single gpus. the core aspects are
evaluation, metrics, checkpoints, config files."

This report comes from five read-only reviews:
1. Viewer lifecycle.
2. How experiments are defined, surveying `../training-runs` and comparing with v1 at `291ddccd`.
3. The run directory, evaluation and checkpoints.
4. Single-GPU, multi-GPU and cluster execution.
5. The CLI surface and the open owner feedback.

Nothing was changed while reviewing. File references are `src/hypergan/...` on develop
`54959592`.

## Summary

HyperGAN v2 records each run carefully. The manifest, events, fingerprints and
recovery are solid. The friction is in everything around a run:

- **The experiment is split across two files.** In 69 of 77 launchers, the shell script
  adds nothing specific to that experiment apart from a GPU, a Python env and some
  comments. Everything else is the same boilerplate copied over and over.
- **The viewer outlives training on purpose.** Every `train` leaves three background
  processes behind, and nothing stops them unless you run `stop-server`.
- **Nothing works across runs.** There is no list or comparison of runs, and no way to
  evaluate or sample a checkpoint taken mid-run. `queue-status.sh` fills that gap by
  scraping shell scripts and JSON.
- **Checkpoints are never deleted.** `/mnt/ml7tb` is **100% full (23 GB free of 7 TB)**.
  One 200k CIFAR run is 80 GB, 79 GB of it checkpoints.

The plan rests on one model. Every command and file below fits one of these five things:

| Thing | What it is | File / command |
|---|---|---|
| **Workspace** | Settings shared by every experiment in a folder: where runs go, path aliases, GPU names, Python envs, default cadence | `training-runs/hypergan.toml` (found by walking up from the config) |
| **Experiment** | One TOML file that fully defines a run, inheriting from a base | `cifar-…-pg217-lr4e-4.toml` with `extends = "…"` and an optional `[run]` table |
| **Run** | The directory the experiment produces; the single source of truth | `RUN/` with `manifest.json`, `summary.json`, `launch/`, `config-source/`, checkpoints, evaluations |
| **Dashboard** | One long-lived viewer for many runs on one fixed port | `hypergan view ROOT` |
| **Queue** | Ordered experiments per GPU, which survives reboot | `hypergan queue add/ls/run` |

Moving from one GPU to several, and later to a cluster, changes only `[run].devices` or
`--devices`. The experiment file stays the same.

## 1. The viewer that doesn't stop

### What happens today

- **`train` and `resume` start a viewer automatically** when the `web` extra is installed
  (`cli.py:203-222`, `web_autostart.py:467-470`). It starts a broker with
  `start_new_session=True` (`web_autostart.py:394-400`). The broker starts the HTTP
  server and a projector.
- **On every exit, `training_viewer` only detaches** (`web_autostart.py:487-489`).
  `close()`, which would stop the broker, runs only when startup fails (`:480`). The
  broker is in its own session, so Ctrl-C never reaches it. Once training finishes, the
  projector keeps checking the run 10 times a second until something kills it
  (`:97-102`).
- **Each viewer serves one run** (`web_server.py:46`), on the first free port from 8765
  (`ports.py:8-9`). Ports are not remembered after the viewer dies. Nothing lists every
  viewer that is running.
- **The URL appears once, in stderr.** For queued runs that means inside
  `logs/<launcher>.log`. The URL printed is a loopback address even though the viewer
  listens on `0.0.0.0`.
- **Staying up after training was something you asked for.** In
  `feedback-2026-09-20.md` item 5 ("Keep the server running after training ends", PR #339)
  you asked to inspect metrics and samples after a run finishes, to reuse the viewer on
  resume, and to avoid duplicate servers. Commit a94f6059 implemented that. The problem
  is not that the viewer persists. The problem is that there is **one persistent viewer
  per run**, with no index and nothing that ever cleans them up.
- **The evidence is in the saved viewer records.** The records in
  `/mnt/ml7tb/hypergan-training-runs/*/observations/viewer-*.json` show ports climbing to
  8801, so about 36 lower ports were taken at that moment. Once about 100 viewers have
  leaked, the port search runs out, and `--server` refuses to start training.
- **A resumed run reconnects to the old viewer** without checking its version, so an
  upgraded frontend doesn't appear until you run `stop-server`.

### What it should be

Keep item 5's behaviour: results stay viewable after training, a resumed run reuses the
viewer, and there are no duplicates. Change what persists from one viewer per run to
**one dashboard per runs root**.

1. **`train` attaches to the workspace dashboard and starts it only if it isn't already
   running.**
   - There is at most one dashboard process per root (`runs_root` from the workspace
     file, or the run directory's parent when there is none). It uses one fixed port
     (8765), so there is one bookmark.
   - It finds runs by scanning `*/manifest.json`. The index page shows each run's status,
     step, it/s, latest and best FID, and disk use. Clicking a run opens today's run view
     at `/runs/<id>`.
   - The server dynamically routes `/api/v1/runs/{run_id}/*` to per-run `ObservationService`
     instances and only runs a `Projector` for runs whose status is `running` (stopping the
     projector as soon as a run reaches a terminal status).
   - A queue of 50 runs leaves one process behind, not 150.
2. **The dashboard is visible and can be stopped.**
   - `hypergan view ROOT` runs it in the foreground, for tmux or a systemd user unit.
   - `hypergan view --status` lists the dashboard and its URL.
   - `hypergan view --stop` stops it.
   - `train` prints the dashboard URL in its header every time.
   - `stop-server --all` clears the per-run viewers already left running from before
     this change.
3. **The dashboard never serves stale code.** It records the HyperGAN version and
   frontend hash it started with. If `train` finds a dashboard from a different version,
   it restarts it instead of reusing it. Item 4's `--dev` mode is unchanged.
4. **Cluster nodes don't start one.** `[run].viewer = false` skips the dashboard; local
   non-TTY runs (such as the queue) still start or attach to the local root dashboard (D1).
   A single `hypergan view` on a login node or workstation reads the shared or synced run
   directories.

The `--server` flag disappears from the launchers. A single run without a workspace still
gets a viewer automatically. Its root is simply that run's parent directory.

## 2. Shell file plus config: one experiment file

### What the 77 launchers carry

| Item | Scripts | Same across scripts, or per experiment? |
|---|---|---|
| `--server --checkpoint-every 1000 --preview-every 100 --progress-every 20` | 69 identical | Same everywhere. The built-in defaults (checkpoint every 100, previews off, `execution.py:198-199`) don't match real use. |
| `--run-dir` | 77 | 66 are `train-<script stem>` under one of 3 roots, so the name can be derived. |
| `CUDA_VISIBLE_DEVICES=<uuid>` | 76 | Only 2 distinct UUIDs: really just a choice of GPU 0 or GPU 1. |
| `OMP_NUM_THREADS`/`MKL_NUM_THREADS=1` | 74 | Same everywhere. |
| Venv Python plus `PYTHONPATH` worktree | 74 / 60 | 15 venvs and 7 source trees. About 25 launchers exist only to test ParticleGAN PRs. The rest pin frozen code for long runs that get resumed, which is a real ongoing need. |
| Other flags | 10 | `--no-tune`, `--stop-after-steps`, `--steps`, `--port`, `--dev` |
| Intent comments | 76 | 4–7 lines, often repeated in the TOML header. Never recorded in the run. |

The configs are copied and edited by hand. The 36 CIFAR TOMLs are each about 105
non-comment lines and differ from `pg08-defaults` by 1–8 lines. One of those lines is
always `name`, which has to be edited by hand and is part of the numerical fingerprint
(`config.py:593-595`). Four files describe their parent in prose ("Same as X except
`lazy_k = 1`"). All 46 hard-code `/home/martyn/...` paths, so none of them can be moved
to a cluster as they are. The queue runs `.sh` files, and `queue-status.sh` finds each
run directory by regex-scraping `--run-dir` out of the launcher.

**v1 for comparison:**
- **How it worked.** `hypergan new NAME -c template` copied a packaged JSON file. Running
  it was `hypergan train folder/ -s 32x32x3 -c NAME`, and output went to
  `saves/<config>/`.
- **What it got right.** The run location came from the config name, so there was no
  launcher.
- **What it got wrong.**
  - Data path, size, batch, devices and cadence were CLI-only, so half the experiment
    still lived on the command line.
  - Saves were keyed only by name with no fingerprint, so editing the JSON quietly
    resumed onto weights that no longer matched.
  - It recorded no provenance.
  - Config lookup failed in confusing ways (#285/#288).

v2 fixed v1's correctness problems. The plan below brings back v1's convenience without
undoing those fixes.

### Design

PR #388 (`hypergan-model/1`) lands first, and Phase 1 builds directly on `hypergan-model/1`
so `training-runs` is only ported once.

**Workspace file**, found by walking up from the experiment, the way `pyproject.toml` is
found:

```toml
# training-runs/hypergan.toml
[workspace]
runs_root = "/mnt/ml7tb/hypergan-training-runs"
run_dir = "{runs_root}/train-{stem}"

[paths]                       # ${data}, ${networks}, ... usable in any string
data = "/home/martyn/dev/ParticleGAN/data"
networks = "/home/martyn/dev/hypergan/HyperGAN/examples/networks"

[gpus]
gpu0 = "GPU-ed080e41-3193-3755-6756-f3d46c433331"
gpu1 = "GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce"

[envs]                        # named code + interpreter pins, not a venv per experiment
develop = { python = "particlegan08-env/bin/python", src = "../HyperGAN/src" }
pg217   = { python = "particlegan-pr217-env/bin/python",
            src = "../worktrees/particlegan-dv12/src", note = "ParticleGAN#217 @5852f380" }

[run]                         # defaults for every experiment here
devices = ["gpu0"]
env = "develop"
cpu_threads = 1
checkpoint_every = 1000
preview_every = 100
progress_every = 20

[checkpoints]
keep_last = 3
keep_best = { metric = "fid50k_train", mode = "min", count = 2 }
keep_every = 50000
```

**Experiment file (`hypergan-model/1`).** It contains only what is specific to this
experiment, and the comment explaining intent travels with it:

```toml
# cifar-tiny-transformer-resnet-features-pg217-lr4e-4-noise.toml
# #217 PR defaults except peak lr 4e-4, lazy_k = 1, plus 0.8.0 critic input noise.
format = "hypergan-model/1"
extends = "cifar-tiny-pg-base.toml"

[run]
env = "pg217"
devices = ["gpu1"]

[train]
lr = 0.0004
lazy_k = 1
```

**Command.** `hypergan train cifar-…-pg217-lr4e-4-noise.toml`. Running it again resumes,
as `train` already does. One-off changes use `--set train.lr=3e-4`, which is recorded
in the manifest. A change worth keeping should become a small child file instead.

**Rules:**
1. **`extends`.** Chains are allowed and cycles are rejected. Tables (including
   `[networks.*]`, `[train]`, `[evaluations.*]`) deep-merge by name and arrays replace. A
   relative path resolves against the file that declares it. Inheritance and `${paths}`
   interpolation are resolved before lowering (`model_file.lower` / `resolve_config`), so
   the resolved config and fingerprint come out identically to a flat file. A test checks
   fingerprint equality across all 46 existing TOMLs so runs already in progress resume
   unchanged.
2. **`[run]` and `[checkpoints]`** hold run-level settings: run directory, cadence,
   devices, threads, env, stop conditions, tuning, viewer and retention. Precedence is
   workspace, then the `extends` chain, then the file, then the CLI. These tables are
   left out of the numerical fingerprint, just as the CLI controls are today. The
   existing flags become overrides.
3. **`name`** defaults to the file stem. Today it silently becomes `reference/100gaussians`
   (`config.py:29`). It is removed from the numerical fingerprint (D3) so that renaming a
   file doesn't block resume.
4. **Env re-exec.** If `[run].env` names a different interpreter or source tree, the outer
   `hypergan train` resolves the workspace, `extends`, `${paths}` and `[run]` first,
   materializes the lowered config and provenance into `RUN/launch/<attempt>.json` and
   `RUN/config-source/`, and re-execs into the target interpreter before importing torch.
   Materializing in the outer CLI ensures older worktrees (`pg217`, `pg07`, etc.) work
   without needing Phase 1 backported.
5. **Provenance.** Every attempt writes `RUN/launch/<attempt>.json` containing argv, the
   resolved `[run]`, GPU UUIDs, `sys.executable`, `PYTHONPATH`, and the commit and dirty
   state of every source tree, including the ParticleGAN PR pin. It also copies the
   source TOMLs, comments included, into `RUN/config-source/`. Today the PR pin and the
   experiment's intent exist only in shell comments.

`start-*.sh` launchers keep working while existing runs finish, and are deleted as those
runs complete.

## 3. Runs, checkpoints and evaluation

### Gaps, most important first

1. **Checkpoints are never deleted, and the default is every 100 steps.**
   `write_checkpoint` (`checkpoints.py:178-201`) only ever adds files. A run has 203
   checkpoints of about 400 MB each, and the final step is written twice. At the
   100-step default, a 200k run would need about 800 GB. The disk is full now.
2. **Comparing runs means writing a script.** The manifest stores only the *next*
   evaluation, not results or the best value so far. FID results live in uuid-named
   `metrics/evaluations/<uuid>/` directories. `queue-status.sh` reimplements "best FID at
   step".
3. **A checkpoint taken mid-run can't be evaluated or sampled.** `evaluate` and `sample`
   only accept `attempts/*/inference/model.pt`, and those exist only at the end of an
   attempt (`metric_evaluation.py:83-90`, `artifacts.py:104-113`). `evaluate` also takes
   the run lock (`metric_evaluation.py:65`).
4. **No quick way to see how a run is doing.** `inspect` dumps 60 KB of JSON.
5. **Monitoring data on disk is never trimmed.** Per-step metrics add 263 MB to
   `events.jsonl`, and the projection in `views/` adds 1 GB.
6. **Resuming on different hardware is refused** unless the only difference is the GPU
   uuid (`docs/recovery.md:77`). That blocks moving a run between machines or GPU models.

### Design

- **`summary.json`**, rewritten at every checkpoint and evaluation, records:
  - status, step and total, it/s and ETA;
  - the latest metrics;
  - `best: {metric, value, step, checkpoint}`;
  - every checkpoint, with its evaluation values.

  The dashboard, `runs ls` and the queue all read this one file. Evaluation directories
  get step-based names.
- **Retention policy in `[checkpoints]`** (`keep_last = 3`, `keep_best = 2`,
  `keep_every = 50000`, `export_bundle = "on_eval"`). Retention never deletes `latest`,
  any checkpoint at a step whose interval evaluation is still in flight or pending
  (`IntervalEvaluations`), or any checkpoint that doesn't yet have a durable newer one.
  `hypergan checkpoints prune RUN [--dry-run]` applies the policy to existing runs, and
  only when you run it explicitly.
- **Commands for one run**, each taking `--checkpoint best|latest|STEP|PATH`:
  - `hypergan status RUN`: one screen covering status, best FID, recent checkpoints and
    warnings. It replaces `inspect`, `metrics` and `server-status` for everyday use.
  - `hypergan eval RUN --checkpoint … --metric … [--devices …]`: read-only. It extracts
    an EMA snapshot from `checkpoints/*/state.pt` when needed, does not take
    `run_lock(root)` or create a new attempt, and can run alongside the trainer.
  - `hypergan sample RUN --checkpoint … -o grid.png`.
  - `hypergan export RUN --checkpoint … -o model.pt`.
  - *(Note: per owner decision D8, `hypergan branch` is omitted — changing LR or schedule
    remains a new experiment.)*
- **Commands across runs:** `hypergan runs ls [ROOT] [--sort fid50k_train] [--status
  running]`, and `hypergan runs compare R1 R2 … --metric fid50k_train --at 50k,100k,200k
  [--csv]`.
- **Coarser default metric cadence (`every_steps = 10`) and `hypergan gc RUN`** (D12):
  log training metrics every 10 steps by default (cutting `events.jsonl` and `views/` by
  ~10x while keeping smooth curves), and `hypergan gc RUN` removes rebuildable `views/`
  and compacts the event log on finished runs.
- **Hardware-tolerant resume by default** (D9): differences in runtime hardware (GPU
  model, capability, world size, `cuda.rank_devices`) emit a structured warning in stderr,
  `manifest['warnings']`, `manifest['resume_warnings']` and the `resume` event rather
  than refusing to resume.

## 4. Single GPU, multi-GPU, cluster

| Scale | Today | Gap |
|---|---|---|
| 1 GPU | Works, and it is what every run uses. Device is `training.device` in the recipe; the actual card is chosen by a `CUDA_VISIBLE_DEVICES` UUID in the launcher. Stopping cleanly and resuming are solid (`run_signals.py`, `run_state.py:37-52`). | No `--devices` option, so the card can only be chosen in the shell. |
| N GPUs, one host | `--profile cuda-replicated-nccl` is built and tested on a synthetic toy. Monitoring, evaluation and checkpoints go through the same controller. | The CIFAR recipes are **rejected**: input noise, phase draws and fused Adam are all refused (`execution_profiles.py:18-32`). Workers are **killed after 1 h** (`total_timeout=3600`, `replicated_execution.py:50`). Every rank decodes the whole global batch. No launcher has ever used it. |
| M hosts | Nothing. Workers find each other through a local temp file (`cpu_worker_service.py:255`). | No rendezvous, no node rank, no Modal/cluster support. Distributed resume refuses moved GPUs (`cuda.rank_devices` isn't on the warn-only list). |

**Scope (D10, D11).** Get **local 2-GPU CIFAR training working first**, together with
`--devices` and `hypergan queue`, and **then run multi-node on Modal** using the reserved
Modal credit.

- **`[run].devices` / `--devices`** accepts `0`, `0,1`, a UUID, a workspace alias, or
  `auto`. HyperGAN sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` itself
  and records the choice. One device runs on a single GPU; more than one selects
  replicated execution automatically. The profile TOML is still where expert settings
  such as accumulation and timeouts live.
- **Local 2-GPU CIFAR support:**
  - Lift the replicated-execution restrictions for the CIFAR K3P recipe: support critic
    input noise, generator output noise, independent phase draws, and recipe-built
    optimizers in `distributed_training.py` and `execution_profiles.py`.
  - Apply fused device-side nonfinite gradient screening to
    `distributed_training._reduce_gradients` (item 19 open sub-task).
  - Default replicated `total_timeout` to no limit (or `--max-seconds`).
  - Pass `rank` and `world_size` to the data loader so each rank decodes only its shard.
  - Qualify a 2-GPU CIFAR run on `gpu0,gpu1`.
- **`hypergan queue add|ls|status|stop|run --devices gpu0`** replaces `run-queue.sh`.
  - Queue entries are experiment TOMLs, not shell scripts.
  - `wait RUN` and the `.stop` file keep working.
  - `queue status` reads `summary.json`.
  - `queue install-service` writes a systemd user unit so the queue survives a reboot.
    Today it has to be restarted by hand.
- **Preflight checks the hardware.** It reports the visible GPUs and their UUIDs, free
  memory, and any other processes on each card, so a busy GPU is caught before launch.
- **Modal multi-node after local 2-GPU works (D11):**
  - Accept torchrun-style environment variables (`MASTER_ADDR`, `MASTER_PORT`,
    `NODE_RANK`, `NNODES`) for rendezvous across hosts instead of a local temp file.
  - Assume a shared/mounted volume for the run directory first.
  - Warn (per D9) when `cuda.rank_devices` or host hardware changes across resume.
  - Run the two-node test on Modal using the reserved Modal credit.

## 5. CLI surface and console

- **The CLI is large.** It has 20 subcommands and about 85 flags; `train` and `resume`
  each take 26. Six replicated-execution timeouts and `--dev` sit on `train`. `project`
  and `contributions` are internal plumbing exposed at the top level.
- **`train` already resumes**, so `resume` is a duplicate with a slightly different set
  of flags. Make `resume` an alias of `train RUN_DIR`.
- **Viewer flags are named differently** on `train` and `serve` (`--server-host` vs
  `--host`, and `--auth` with different defaults). Use one set of names.
- **Run arguments are inconsistent.** `--run-dir` is a required flag on `train`, but the
  run is a positional argument everywhere else. With a workspace, the run directory
  becomes optional.
- **The console needs a header and a footer.**
  - Header: run directory, experiment file, devices, env, and a URL that works from
    another machine (hostname or `--public-origin`).
  - Progress: `step N/T (x%) · it/s · ETA`.
  - Show FID results when they arrive, and checkpoint lines with their paths.
  - Remove the 2,000 `metric at step N` / `preview at step N` lines that a 200k log
    contains.
  - Collapse the four trust/qualification warnings into one line.
  - End with a short human summary in place of the manifest JSON on stdout. Machine
    output stays under `--progress-json`.
- **There is no path from a folder of images to a run.** `hypergan new` writes a 2-D toy;
  the only recipe listed is `reference/100gaussians`; no example uses `image_folder`. A
  new user needs 10+ steps and about 12 concepts. The target is three steps:
  `hypergan new mygan --dataset ./images` (a weight-free image recipe sized from the
  data, with cadence and retention defaults filled in), then `hypergan train mygan.toml`,
  then open the URL it prints.

PR #388 (the Python API proposal) section 7 already maps every CLI command onto an
`hg.*` call, adds `--set k=v`, and states that "the CLI must become a thin layer over the
same API". This plan uses that mapping. The new commands here (`status`, `runs`,
`checkpoints`, `queue`, `view`, `gc`, `export`) are added to that table rather than
designed separately.

## Plan & Team Workflow Workstreams

Each slice below is scoped for an independent PR into `develop` so a multi-agent team
workflow can execute them with minimal file contention.

### Prerequisite: Unblock CI & Merge PR #388
1. **Fix `develop` CI packaging check** (`pyproject.toml` `particlegan` dependency spec)
   and **merge PR #388 (`hypergan-model/1` & Python API)** into `develop` so Phase 1
   builds directly on `hypergan-model/1`.

### Workstream A: Checkpoints, Retention & Mid-Run Evaluation (Urgent — Disk Full)
*Primary files: `src/hypergan/checkpoints.py`, `src/hypergan/interval_evaluation.py`, `src/hypergan/metric_evaluation.py`, `src/hypergan/artifacts.py`*
1. **Checkpoint retention & `prune` (Phase 0a):**
   - Change default `checkpoint_every` to `1000`.
   - Implement `[checkpoints]` retention (`keep_last = 3`, `keep_best = { metric = "fid50k_train", mode = "min", count = 2 }`, `keep_every = 50000`, `export_bundle = "on_eval"`).
   - Coordinate with `IntervalEvaluations` so checkpoints at steps with an in-flight or
     pending snapshot evaluation are pinned until the evaluation finishes.
   - Add `hypergan checkpoints prune RUN [--dry-run]` and run it to free space on
     `/mnt/ml7tb`.
2. **Mid-run & checkpoint-targeted `eval`, `sample`, `export` (Phase 3.1):**
   - Accept `--checkpoint best|latest|STEP|PATH` on `eval`, `sample`, and `export`.
   - Extract EMA weights/prior from `checkpoints/*/state.pt` when an inference bundle is
     not already materialized.
   - Make `hypergan eval RUN --checkpoint … [--devices …]` read-only and lock-free (no
     `run_lock(root)` or new attempt) so it can run on a second GPU while training runs.

### Workstream B: Single Root Dashboard & Multi-Run Viewer
*Primary files: `src/hypergan/web_autostart.py`, `src/hypergan/web_server.py`, `src/hypergan/web_service.py`, `frontend/src/`*
1. **Multi-run server routing & projector lifecycle (Phase 0b):**
   - Route `/runs/<run_id>` and `/api/v1/runs/{run_id}/*` dynamically across `*/manifest.json`
     under the dashboard root.
   - Run `Projector` only for runs with `status == "running"`; stop a run's projector as
     soon as its manifest reaches a terminal status.
   - One dashboard process per `runs_root` on fixed port `8765`, auto-restarted if the
     HyperGAN version or frontend asset hash differs.
   - Add `hypergan view [ROOT] [--status|--stop]` and `stop-server --all`.
   - Test that a queue of N runs leaves at most one viewer process behind.
2. **Dashboard run index page (Phase 2.3):**
   - Serve the root index table at `/` reading `summary.json` (status, step, it/s, latest
     and best FID, disk usage) with sorting and filtering.

### Workstream C: Workspace, `extends`, `[run]` on `hypergan-model/1` & Hardware Resume
*Primary files: `src/hypergan/config.py`, `src/hypergan/model_file.py` (from #388), `src/hypergan/checkpoint_compatibility.py`, `src/hypergan/execution.py`*
1. **Workspace & single-file experiments (Phase 1):**
   - Discover `hypergan.toml` by walking up from the config file; resolve `${paths}`,
     `extends` chains (deep-merging `[networks.*]`, `[train]`, `[evaluations.*]`), `[run]`
     and `[checkpoints]`.
   - Default `name` to the file stem and remove `name` from the numerical fingerprint (D3).
   - Verify fingerprint equality across all 46 existing TOMLs in `training-runs`.
   - Implement `[run].devices` (GPU index, UUID, workspace alias, `auto`), `cpu_threads`,
     and outer-CLI materialization into `RUN/launch/<attempt>.json` + `RUN/config-source/`
     followed by `[run].env` re-exec before importing torch.
   - Port `../training-runs` to `hypergan.toml` + `cifar-tiny-pg-base.toml` + child TOMLs.
2. **Hardware-tolerant resume by default (D9):**
   - Expand `checkpoint_compatibility.py` so runtime hardware differences (GPU model,
     capability, world size, `cuda.rank_devices`) emit structured warnings instead of
     raising `ValueError`.

### Workstream D: Local 2-GPU CIFAR Training, Queue & Modal Multi-Node
*Primary files: `src/hypergan/distributed_training.py`, `src/hypergan/replicated_execution.py`, `src/hypergan/execution_profiles.py`, `src/hypergan/cpu_worker_service.py`, `src/hypergan/queue.py` (new)*
1. **Local 2-GPU CIFAR support (D10):**
   - Default `replicated_execution.py` `total_timeout` to no limit (or `--max-seconds`).
   - Support critic input noise, generator output noise, independent phase draws, and
     recipe-built K3P optimizers in `distributed_training.py` and `execution_profiles.py`.
   - Fuse nonfinite gradient screening in `distributed_training._reduce_gradients`.
   - Pass `rank` and `world_size` to data loading so each rank decodes only its shard.
   - Qualify 2-GPU CIFAR training on `gpu0,gpu1`.
2. **Queue & hardware preflight (Phase 4.1):**
   - Implement `hypergan queue add|ls|status|stop|run --devices …` and
     `hypergan queue install-service` (systemd user unit), plus GPU busy/memory preflight.
3. **Modal multi-node (D11, after local 2-GPU is green):**
   - Add torchrun-style TCP rendezvous (`MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`,
     `NNODES`) to `cpu_worker_service.py` and run the 2-node Modal qualification test over
     a shared/mounted run volume.

### Workstream E: `summary.json`, Run Inspection CLI, Console & Metric Cadence
*Primary files: `src/hypergan/run_controller.py`, `src/hypergan/bounded_cli_output.py`, `src/hypergan/cli.py`, `src/hypergan/event_views.py`*
1. **`summary.json`, `status`, `runs ls|compare` (Phase 2.1–2.2):**
   - Write `summary.json` at every checkpoint and evaluation; name evaluation directories
     by step.
   - Add `hypergan status RUN`, `hypergan runs ls [ROOT]`, and `hypergan runs compare`.
2. **Metric cadence (`every_steps = 10`) & `hypergan gc` (D12 / Phase 3.3):**
   - Change default training metric cadence to `every_steps = 10` (~10x smaller event logs
     and projections).
   - Add `hypergan gc RUN` to remove rebuildable `views/` and compact `events.jsonl`.
3. **Console header/footer & `hypergan new --dataset DIR` (Phase 2.4 & 4.2):**
   - Clean console header (run dir, experiment, devices, env, reachable dashboard URL),
     progress with ETA, inline FID/checkpoint notifications, single-line warning summary,
     and human footer.
   - Make `--run-dir` optional on `train`, alias `resume` to `train RUN_DIR`, unify
     viewer flags, and add `hypergan new NAME --dataset DIR`.

## Resolved owner decisions (2026-10-14)

| # | Decision | Resolution |
|---|---|---|
| PR #388 | Config format for Phase 1 (`extends`, `${paths}`, `[run]`, `[checkpoints]`) | **Land PR #388 (`hypergan-model/1`) first** and build Phase 1 directly on `hypergan-model/1` |
| D1 | Should `train` start the root dashboard when it isn't attached to a terminal (queue, cluster node), or only print its URL? | **Start it locally** (including the queue); skip it when `[run].viewer = false` |
| D2 | Should `train CONFIG` with the run directory optional replace `train --run-dir`, or should there be a new `hypergan run`? | **Make `--run-dir` optional on `train`**; no new command |
| D3 | Remove `name` from the numerical fingerprint, so renaming an experiment doesn't block resume? | **Yes** |
| D4 | May `[run].env` choose the Python interpreter and source tree (re-exec)? | **Yes**, as named envs in the workspace file (outer CLI materializes resolved config before re-exec) |
| D5 | Under `extends`, should tables (`networks`, `train`, `evaluations`) deep-merge, or be replaced whole? | **Deep-merge by name** |
| D6 | Default retention, and whether to prune existing runs | **`keep_last = 3` + `keep_best = 2` + `keep_every = 50000`** (`checkpoint_every = 1000`); existing runs pruned only by explicit `hypergan checkpoints prune` |
| D7 | Should `eval`/`sample` on a checkpoint be allowed while training runs (second GPU, or sharing the training GPU)? | **Yes**, read-only and lock-free, with `--devices` |
| D8 | `branch` with a changed LR or schedule, given today's rule of "new experiment, no transfer" | **No `branch` command** — do not allow changing LR/schedule/steps mid-run |
| D9 | Allow resuming on different hardware? | **Yes, warn by default** (no `--allow-runtime-change` flag required) |
| D10 | Is multi-GPU in scope now or deferred? | **In scope now**: get local 2-GPU CIFAR training working first |
| D11 | Which cluster comes first? | **Local 2-GPU first, then Modal** (assuming a shared/mounted volume for the run dir initially) |
| D12 | Default training metric logging cadence | **Every 10 steps by default** (`every_steps = 10`), plus `hypergan gc` |

## Housekeeping found along the way

- `/mnt/ml7tb` is at 100% (19 GB free as of 2026-10-14). The GPU 0 queue has been empty
  since 2026-09-29, and at today's rate the next 200k run will run out of disk space after
  about 45–50 checkpoints unless Phase 0a (`checkpoints prune`) runs first.
- In `human-feedback.md`, item 22 (Model tab) was merged in `820cede5` (now updated in
  `human-feedback.md`). Item 24 (Python API) is on PR #388 and will be merged first.
- The audit's question about the viewer's default bind address was already settled by
  owner item 4 / PR #337 (`0.0.0.0`).

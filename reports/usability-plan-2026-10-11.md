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
   - A queue of 50 runs leaves one process behind, not 150.
   - The projector stops checking a run once its manifest shows it has finished (today
     it checks 10 times a second for as long as the viewer lives).
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
4. **Cluster nodes don't start one.** `[run].viewer = false` (or not being attached to a
   terminal, D1) skips the dashboard. A single `hypergan view` on a login node or
   workstation reads the shared or synced run directories.

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

**Experiment file.** It contains only what is specific to this experiment, and the
comment explaining intent travels with it:

```toml
# cifar-tiny-transformer-resnet-features-pg217-lr4e-4-noise.toml
# #217 PR defaults except peak lr 4e-4, lazy_k = 1, plus 0.8.0 critic input noise.
extends = "cifar-tiny-pg-base.toml"

[run]
env = "pg217"
devices = ["gpu1"]

[optimizer]
lr = 0.0004

[gradient_penalty]
lazy_k = 1
```

**Command.** `hypergan train cifar-…-pg217-lr4e-4-noise.toml`. Running it again resumes,
as `train` already does. One-off changes use `--set optimizer.lr=3e-4`, which is recorded
in the manifest. A change worth keeping should become a small child file instead.

**Rules:**
1. **`extends`.** Chains are allowed and cycles are rejected. Tables deep-merge and arrays
   replace. A relative path resolves against the file that declares it. Inheritance is
   resolved before `resolve_config`, so the resolved config and fingerprint come out
   exactly as the copy-pasted files produce them today. A test checks this for all 46
   existing TOMLs, so runs already in progress resume unchanged.
2. **`[run]` and `[checkpoints]`** hold run-level settings: run directory, cadence,
   devices, threads, env, stop conditions, tuning, viewer and retention. Precedence is
   workspace, then the `extends` chain, then the file, then the CLI. These tables are
   left out of the numerical fingerprint, just as the CLI controls are today. The
   existing flags become overrides.
3. **`name`** defaults to the file stem. Today it silently becomes `reference/100gaussians`
   (`config.py:29`). It should come out of the numerical fingerprint so that renaming a
   file doesn't block resume. (This needs owner sign-off; see decision D3.)
4. **Env re-exec.** If `[run].env` names a different interpreter or source tree,
   `hypergan train` re-execs into it before importing torch. This is about 30 lines.
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
4. **A run can't be branched.** Lowering the LR at the best checkpoint means starting a
   new run from scratch (`docs/recovery.md:15-19`).
5. **No quick way to see how a run is doing.** `inspect` dumps 60 KB of JSON.
6. **Monitoring data on disk is never trimmed.** Per-step metrics add 263 MB to
   `events.jsonl`, and the projection in `views/` adds 1 GB.
7. **Resuming on different hardware is refused** unless the only difference is the GPU
   uuid (`docs/recovery.md:77`). That is right for exact reproduction, but it blocks
   requeueing on a cluster.

### Design

- **`summary.json`**, rewritten at every checkpoint and evaluation, records:
  - status, step and total, it/s and ETA;
  - the latest metrics;
  - `best: {metric, value, step, checkpoint}`;
  - every checkpoint, with its evaluation values.

  The dashboard, `runs ls` and the queue all read this one file. Evaluation directories
  get step-based names.
- **Retention policy in `[checkpoints]`** (`keep_last`, `keep_best`, `keep_every`,
  `export_bundle = "on_eval"`). Retention never deletes `latest`, a checkpoint that a
  running evaluation depends on, or any checkpoint that doesn't yet have a durable newer
  one. `hypergan checkpoints prune RUN --dry-run` applies the policy to existing runs,
  and only when you run it explicitly.
- **Commands for one run**, each taking `--checkpoint best|latest|STEP|PATH`:
  - `hypergan status RUN`: one screen covering status, best FID, recent checkpoints and
    warnings. It replaces `inspect`, `metrics` and `server-status` for everyday use.
  - `hypergan eval RUN --checkpoint … --metric …`: read-only. It creates no new attempt
    and can run alongside the trainer.
  - `hypergan sample RUN --checkpoint … -o grid.png`.
  - `hypergan export RUN --checkpoint … -o model.pt`.
  - `hypergan branch RUN --checkpoint best --set optimizer.lr=1e-4 [--run-dir R2]`:
    copies weights and optimizer state into a new run with `parent: {run, checkpoint}`.
    Only an explicit list of fields may change: LR, schedule, steps, evaluation and
    observation settings.
- **Commands across runs:** `hypergan runs ls [ROOT] [--sort fid50k_train] [--status
  running]`, and `hypergan runs compare R1 R2 … --metric fid50k_train --at 50k,100k,200k
  [--csv]`.
- **`hypergan gc RUN`** removes `views/`, which can be rebuilt, and compacts the event log.

## 4. Single GPU, multi-GPU, cluster

| Scale | Today | Gap |
|---|---|---|
| 1 GPU | Works, and it is what every run uses. Device is `training.device` in the recipe; the actual card is chosen by a `CUDA_VISIBLE_DEVICES` UUID in the launcher. Stopping cleanly and resuming are solid (`run_signals.py`, `run_state.py:37-52`). | No `--devices` option, so the card can only be chosen in the shell. |
| N GPUs, one host | `--profile cuda-replicated-nccl` is built and tested on a synthetic toy. Monitoring, evaluation and checkpoints go through the same controller. | The CIFAR recipes are **rejected**: input noise, phase draws and fused Adam are all refused (`execution_profiles.py:18-32`). Workers are **killed after 1 h** (`total_timeout=3600`, `replicated_execution.py:50`). Every rank decodes the whole global batch. No launcher has ever used it. |
| M hosts | Nothing. Workers find each other through a local temp file (`cpu_worker_service.py:255`). | No rendezvous, no node rank, no Slurm or SkyPilot support. Distributed resume probably refuses moved GPUs (`cuda.rank_devices` isn't on the warn-only list). |

**Scope.** Two things decide it. The 2.0.0b1 single-GPU scope still holds
(`claude-audit-2026-09-20.md:40-44`). The owner said of the Python API: "multi-host is
later, but the API must not rule it out". So 2.0 should get the right *shape*: devices
are a run setting, there is a queue, and many runs are monitored together. Training
across hosts can wait for 2.1.

- **`[run].devices` / `--devices`** accepts `0`, `0,1`, a UUID, a workspace alias, or
  `auto`. HyperGAN sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` itself
  and records the choice. One device runs on a single GPU; more than one selects
  replicated execution automatically. The profile TOML is still where expert settings
  such as accumulation and timeouts live.
- **`hypergan queue add|ls|status|stop|run --devices gpu0`** replaces `run-queue.sh`.
  - Queue entries are experiment TOMLs, not shell scripts.
  - `wait RUN` and the `.stop` file keep working.
  - `queue status` reads `summary.json`.
  - `queue install-service` writes a systemd user unit so the queue survives a reboot.
    Today it has to be restarted by hand.
- **Preflight checks the hardware.** It reports the visible GPUs and their UUIDs, free
  memory, and any other processes on each card, so a busy GPU is caught before launch.
- **Clusters (2.1): ship templates, not a scheduler.**
  - Write `examples/slurm/` (`--signal=TERM@120 --requeue`) and then `examples/skypilot/`.
    Both work because "SIGTERM, then checkpoint, then rerun `train` with the same
    experiment" is already resumable.
  - Accept torchrun-style environment variables for rendezvous across hosts.
  - Require a shared filesystem for the run directory first; syncing to object storage
    can come later.
  - Treat `cuda.rank_devices` the same as single-GPU device changes: warn instead of
    refusing.
  - Then use the reserved Modal credit for the two-node test.

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
`checkpoints`, `branch`, `queue`, `view`) should be added to that table rather than
designed separately.

## Plan

The phases are ordered by how much each one hurts today. Each phase is a separate PR
into develop.

**Phase 0: stop the bleeding.** These are small changes, and some are urgent.
1. Checkpoint retention (`[checkpoints]` plus `--keep-last` and friends) and
   `hypergan checkpoints prune --dry-run`. Change the default `checkpoint_every` to 1000.
   *The disk is full.*
2. Change the auto viewer from one per run to one per root: `train` reuses the root's
   dashboard, and restarts it when its version differs. Add `hypergan view
   [--status|--stop]` and `stop-server --all`. Stop the projector checking finished runs.
   Add a test that a queue of N runs leaves at most one viewer behind.
3. Replicated `total_timeout` defaults to no limit, or to `--max-seconds`.

**Phase 1: one experiment file.**
1. The workspace `hypergan.toml`, `extends`, `[run]`, `${paths}`, the default run
   directory, `name` from the file stem, and `--set`. Include the fingerprint-equality
   test over all existing TOMLs.
2. `[run].devices` with GPU aliases, `cpu_threads`, and env re-exec.
3. `RUN/launch/<attempt>.json` and `RUN/config-source/`.
4. Port `training-runs` over: a CIFAR base TOML plus small child files. Delete the
   launchers as their runs finish.

**Phase 2: see everything.**
1. `summary.json` and step-named evaluation directories.
2. `hypergan status RUN`, `hypergan runs ls|compare`.
3. The dashboard's run index page, with sorting and filtering on `summary.json` fields.
4. The console header, ETA, FID lines, end summary and reduced noise.

**Phase 3: work with checkpoints.**
1. `eval|sample|export --checkpoint best|latest|STEP` without a new attempt, and
   allowed while training runs.
2. `hypergan branch`.
3. `hypergan gc`, and an optional coarser metric cadence.

**Phase 4: scheduling and scale.**
1. `hypergan queue …` plus `install-service`. Preflight checks the hardware.
2. `hypergan new --dataset DIR` and an image-folder recipe, so new users have a path.
3. (2.1) Lift the replicated-execution restrictions for the CIFAR recipe, then run a
   2-GPU image test. Add `examples/slurm`, rendezvous across hosts, and the Modal
   two-node run.

Phases 0 and 1 together remove both complaints in the request. Every later phase only
adds commands that read the run directory those phases define.

## Decisions for the owner

Where a default is suggested, it is what the plan above assumes.

| # | Decision | Suggested |
|---|---|---|
| D1 | Should `train` start the root dashboard when it isn't attached to a terminal (queue, cluster node), or only print its URL? | Start it locally (the queue case); skip it when `[run].viewer = false` (cluster nodes) |
| D2 | Should `train CONFIG` with the run directory optional replace `train --run-dir`, or should there be a new `hypergan run`? | Make `--run-dir` optional on `train`; no new command |
| D3 | Remove `name` from the numerical fingerprint, so renaming an experiment doesn't block resume? | Yes |
| D4 | May `[run].env` choose the Python interpreter and source tree (re-exec)? | Yes, as named envs in the workspace file |
| D5 | Under `extends`, should `components` deep-merge, or be replaced whole as it is today? | Deep-merge by component name |
| D6 | Default retention, and whether to prune existing runs | keep_last 3 + best 2 + every 50k; existing runs pruned only by an explicit `prune` |
| D7 | Should `eval`/`sample` on a checkpoint be allowed while training runs (second GPU, or sharing the training GPU)? | Yes, read-only, with `--devices` |
| D8 | `branch` with a changed LR or schedule, given today's rule of "new experiment, no transfer" | Allow it, with a fixed list of fields that may change and a `parent` record |
| D9 | Allow resuming on different hardware with an opt-in "not bit-exact" mark (`--allow-runtime-change`)? | Yes, opt-in only |
| D10 | Is multi-GPU in 2.0, or deferred to 2.1 (and hidden from the 2.0 README)? | 2.1; 2.0 ships `--devices` and the queue |
| D11 | Which cluster comes first: Slurm, SkyPilot, Modal or plain ssh? Can a shared filesystem be assumed? | Need your input |
| D12 | Should metrics be logged every step (263 MB events + 1 GB projection per run), or less often by default? | Need your input |

These connect to PR #388's 8 pending decisions in section 8. The config-first base, the
single documented model format, and resume identity all overlap with D2, D3 and D5. They
should be answered together.

## Housekeeping found along the way

- `/mnt/ml7tb` is at 100% (23 GB free). The GPU 0 queue has been empty since 2026-09-29,
  but at today's rate the next 200k run will run out of disk space after about 60
  checkpoints.
- In `human-feedback.md`, item 22 (Model tab) still says "not merged", but it was merged
  in `820cede5`. Item 24 (Python API) exists only on the PR #388 branch.
- The audit's question about the viewer's default bind address was already settled by
  owner item 4 / PR #337 (`0.0.0.0`).

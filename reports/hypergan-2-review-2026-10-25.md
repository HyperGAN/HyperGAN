# HyperGAN 2 code review (2026-10-25)

Owner request: "review this project for clarity, readability, bugs, implementation
quality, etc."

Scope: `develop` at `95f37d83` (just after PR #410, multi-device). Seven read-only
reviewers each took one slice:

1. CLI, Python API and config.
2. Training core, run lifecycle and checkpoints.
3. Distributed/replicated execution, workers, `qualify` and RunPod.
4. Web viewer and frontend.
5. Data, datasets and metrics.
6. Registry, inspection, queue and retention.
7. Tests, docs, packaging, CI and repository hygiene.

Reviewers confirmed suspected bugs with small CPU snippets in scratch directories. No GPU
training, no network calls to RunPod, no dataset downloads. I ran the fast test suite
myself and re-checked the top findings against the source. The per-slice notes, with
more detail and the reproduction steps, are at `/nvme/dev/hypergan/review-notes-2026-10-25/`
(outside the repo).

"Confirmed" means reproduced or fully traced. "Likely" means read but not reproduced.

## Summary

The numerical and storage core is good, better than most research codebases. On CPU,
stop/resume is bit-identical to an uninterrupted run (graph, prior, EMA, optimizer,
every RNG stream). Checkpoint writes are atomic and restores check everything before
changing any state. FID statistics are pinned and merged exactly in float64. Dataset
extraction is properly safe. File serving in the viewer uses `O_NOFOLLOW` dir-fd
traversal. Preview retention is the best-designed destructive path in the codebase.

The problems sit in the layers around that core, and they share a few causes:

- **Destructive commands trust a fuzzy resolver.** `hypergan rm` will `rmtree` any
  directory, including the runs root or the workspace. Run names resolve by prefix and
  by substring, and `ps` and `rm` can disagree about which run a slug names.
- **Silent fallbacks change behaviour.** An explicit `-t dcgan` becomes the 2-D Gaussian
  toy. A missing `reference.hndl` becomes the packaged MLP. `--devices` is applied to a
  copy of the config that the trainer never reads. "Best checkpoint" mixes unrelated
  metrics. The RunPod live path writes an all-"passed" receipt without running anything.
  Hundreds of `except Exception: pass` blocks make all of this invisible.
- **The same logic is implemented two to five times, and the copies disagree.** Run
  metadata (×3), run-directory naming (×3), resume-config check (×5), checkpoint
  resolution (×3), "is this FID?" (×3), streaming moments (×3), file locks (×3), model
  loaders (×2), diff (×2). Many of the bugs below are one copy drifting from another.
- **Some functions and modules are too long to review.** `cli._dispatch` is about 1,070
  lines and `run_controller._execute_run` about 600. `inspection.py` (2.1k lines),
  `web_server.py` (1.6k) and `web_autostart.py` (1.3k) each mix five or more concerns.
- **The multi-device work from PR #410 holds up only on the paths its tests exercise.**
  There are four High bugs in that slice, and its "100% pass" and qualification-receipt
  claims are not backed by what the tests check.
- **Production code knows about tests.** It detects pytest by walking caller stack
  frames, branches on whether functions were monkeypatched, and ships `MockRunPodClient`.

**`develop` CI is red.** Foundation CI run 40145484106 on the #410 merge fails two tests.
Locally the fast suite gives 3 failed, 2,302 passed and 1 skipped in 6 min. It used to
take about 25 s; part of that slowdown is the reviewers running at the same time.

## Fix first

The list is ranked by harm: data loss, then wrong results, then security, then broken
first-run experience. All items are confirmed unless marked otherwise.

| # | Issue | Where | Slice |
|---|---|---|---|
| 1 | `hypergan rm X` deletes any existing directory. With no registry match, the resolver returns the raw path and `remove_run` calls `rmtree` on it. `rm runs`, `rm .` and `rm ~/dataset` all delete. There is no run-identity check and no confirmation. | `registry.py:1312`, `inspection.py:1966` | 6, 7 |
| 2 | `rm`/`stop`/`tag` resolve fuzzily. With several exact name matches they silently pick the first. A single prefix or substring match (even one character) also resolves. `ps` and the resolver deduplicate shared slugs in different orders, so `rm <slug from ps>` deleted a different, inactive run. | `registry.py:1255-1296`, `:264-273`, `inspection.py:772-777,973` | 6 |
| 3 | `gc`/`prune --all` outside a workspace lossily compacts **every registered run on the host**, including ones found by `/proc` and sibling `training-runs` discovery. `gc DIR` deletes `views/` in any directory. Compaction runs without the run lock, so a trainer that starts mid-compaction writes into an orphaned inode. | `inspection.py:1284-1401,1408,1457,1496-1524` | 6, 7 |
| 4 | K3P `latent_history`/`direct_history` are all-reduced with `SUM` and never divided by world size. Every sync multiplies them by about W, which silently corrupts latent damping. The replica-equality check still passes. | `aggregation.py:213-215` | 3 |
| 5 | The RunPod live path creates a paid pod, runs nothing on it, and writes a hard-coded receipt with every suite "passed" (NCCL included, on a 1-GPU pod). `RunPodHttpClient` has no `execute_qualification`, so the code always falls through to `_build_mock_receipt`. | `runpod_runner.py:598-644,516-572` | 3 |
| 6 | With `parameter_average`/`pseudo_gradient` and `sync_every_steps > 1`, a checkpoint at a non-sync step requires replica agreement, which local SGD does not have between syncs, so the run dies. This includes the final checkpoint after all training is done. | `distributed_checkpoints.py:358-365` | 3 |
| 7 | Multi-node runs die on any leader-only command (preview, evaluation, on-demand checkpoint, `max_seconds`, final publication checkpoint). The follower replays a fixed script, so the command digests disagree. | `cpu_worker_service.py:193-207`, `replicated_execution.py:511-559` | 3 |
| 8 | Interval evaluations starve: rotation resets to just after the last metric that ran. In `examples/logos-tiny-transformer-dinov3-projected-128.toml`, `structure` and `fid10k_heldout` **never run**; they are logged as "skipped (worker_busy)". | `interval_evaluation.py:194,229` | 5 |
| 9 | "Best" is chosen by pooling unrelated metrics and guessing direction from the metric name. The catalog's declared `direction` is ignored. `--checkpoint best` can pick the most collapsed generator, and `keep_best` can keep the worst checkpoints and prune the best ones irreversibly. Separately, `summary.latest`/`latest_fid` freezes at the first evaluation. | `summary.py:164-187,273-284`, `checkpoints.py:392-396,799,823-827` | 2, 5 |
| 10 | In `--auth token` mode, every GET under `/api/runs/*` and `/api/v1/runs` is public: configs, samples, events, SSE. With the default `0.0.0.0` bind, any Host header is accepted, so DNS rebinding lets any website the owner visits read runs and POST stop/checkpoint. The dashboard also serves every run in the user-global registry, not just its root. | `web_server.py:1455-1459,431-479,633-645`, `web_session.py:119` | 4 |
| 11 | `hypergan new --project X -t dcgan`, and the README's `hg.new(project=..., template="dcgan", dataset=...)`, silently produce the gaussian-grid toy. Explicit `dcgan` is rewritten when no data or resolution is given, and `dataset=` is silently dropped (the parameter is `data_dir`). | `recipe_generator.py:633`, `cli.py:1564`, `api.py:401-412`, `README.md:225` | 1 |
| 12 | The `fid` extra allows torch-fidelity 0.4.0 (current PyPI latest), but FID refuses anything except 0.3.0. A fresh install following `docs/cifar-recipe.md` cannot compute the headline metric. | `pyproject.toml:19`, `image_metrics.py:79-80` | 7 |
| 13 | Two queue runners can launch the same job, and a job cancelled during preflight still launches: pick and claim happen in separate lock sections. `queue stop --all` re-sends SIGINT every 0.5 s, which turns the graceful stop into `os._exit` with no final checkpoint. The stop file is never removed, so every later `queue run` exits immediately. | `queue.py:449-457,540-546,606-619,317-318` | 6 |
| 14 | When the documented API example runs as a script without a `__main__` guard, every `spawn` preview worker re-runs the whole script. Previews fail silently and side effects run twice; in one repro the script's own `rmtree` deleted the run mid-training. | `single_execution.py:134-144`, `api.py:1-12` | 2 |
| 15 | `devices=` / `--devices` only changes the prepared copy of the config. `PreparedExecution.run` reloads from the original path, so training silently runs on the configured device (e.g. CPU). | `execution.py:196-207,249-253` | 2 |
| 16 | `develop` is red: `test_native_only_recipe_rejected_by_replicated_profile_before_runtime` is stale after the #410 parity change, and `test_partition_cuda_tests_exact_counts` hard-codes 31. The ROCm partitioner also treats any test missing from its hard-coded multi-GPU list as single-GPU, so all the new 2-GPU NCCL tests land in the single-GPU set. | `tests/reference/test_image_phase_policy.py:188`, `scripts/rocm_preflight_helper.py:138-142` | me |

## Cross-cutting recommendations

These remove whole classes of the bugs above. They are ordered by payoff.

1. **Strict resolver for destructive operations.** `rm`, `gc`, `prune`, `stop` and `tag`
   should accept only an exact run id or short id, an exact slug, or a path that contains
   a `manifest.json` with a `run_id`. Any ambiguity is an error that lists the candidates.
   Refuse workspace roots, runs roots and `$HOME`. Keep fuzzy matching for read-only
   commands. Assign the slug once at creation, store it in the manifest, and never
   re-deduplicate it on read.
2. **One `RunInfo` reader.** `registry._extract_run_entry`, `inspection._collect_ps_row`,
   `inspection.run_status` and `web_server.read_run_summary` each re-derive name, slug,
   status, step, speed and FID through `a.get(x) or b.get(y)` chains over keys that may
   not exist. `summary.py` is the only writer, so read its schema once, return a
   dataclass, and delete the alias keys (`steps_per_sec`/`steps_per_second`/`speed_its`,
   `deleted`/`pruned`, …). This also fixes the queue status reading keys that don't exist
   (`queue.py:797-799`).
3. **Use the metric catalog's `direction`.** Replace the three "contains `fid`"
   heuristics and the name-suffix direction guesses with the declared descriptor. Share
   one "best" implementation between `summary`, `resolve_checkpoint('best')` and
   `keep_best`.
4. **Fail loudly instead of falling back.** Audit `except Exception: pass` (registry,
   inspection, controller summary writes, web launch, gpu inventory). Remove the
   `reference.hndl` fallback (`config.py:887-899`) and the `except TypeError` retry
   without `rank=` (`distributed_training.py:240-244`), which hides dataset bugs and
   silently unshards. Make the CLI catch-all (`cli.py:2270`) show a traceback for
   `TypeError`/`KeyError`, or offer `--debug`.
5. **Take test detection out of production code.**
   - `_is_isolated_test_env` walks 15 caller frames looking for pytest tmp paths
     (`registry.py:178-221`).
   - `get_registry_dir` changes location under `PYTEST_CURRENT_TEST`.
   - `web_server.py` has 16 branches on `_ORIG_*` monkeypatch sentinels, so the code that
     runs under test is not the code that runs in production.
   - `MockRunPodClient` and `_build_mock_receipt` ship in the package.
   - Owner-specific discovery (`../training-runs`, `/tmp/hypergan-viewers-*`, a `/proc`
     sweep) runs inside `ps`.

   Inject collaborators and configuration instead.
6. **Collapse duplicates into single owners.** One `default_run_dir`, one model loader
   (`api.load` and `config.load_config` already disagree), one resume-compatibility
   check, one checkpoint resolver, one file-lock helper, one atomic-JSON writer, one
   streaming-moments class, one colorspace helper, one `MAX_RANK_BYTES`. That last one is
   currently 256 MiB in two modules and patched to 1 GiB from three others at import
   time.
7. **Split the oversized units.**
   - `cli._dispatch`: one function per command, plus one `_maybe_reexec` helper. It has
     two verbatim copies today, and `resume` re-exec under `python -m hypergan` is broken
     (`cli.py:1890-1898`).
   - `_execute_run`: a small `Attempt` class whose request protocol is a separate
     method. The stale-preview-acknowledged-as-success bug (`run_controller.py:784-803`)
     is invisible in the current closure soup.
   - `inspection.py`: `run_liveness`, `run_info`, `run_maintenance`, `run_control` and
     formatting.
   - `web_server.py`: run index, service registry, routes and auth. Generate the twice-
     written 47-route table from one list, so auth and routes cannot drift (that drift is
     #10).
   - `web_autostart.py`: port-owner discovery to `ports.py`, root finding to
     `workspace.py`, one projector, one daemon launcher.
8. **Treat multi-device as experimental until its High bugs are fixed.** The owner's
   stated goal includes clusters and multiple GPUs, so this is about honesty, not
   deletion. Keep the careful single-node core (broker, prepare/commit checkpoints,
   `gradient_allreduce`). Then:
   - Gate multi-node, the local-SGD strategies and RunPod behind explicit "experimental"
     errors until items 4–7 are fixed.
   - Refuse the RunPod live mode until it actually runs `hypergan qualify` on the pod.
   - Stop shipping a committed receipt.
   - Run `gradient_allreduce`'s per-step full host-side SHA-256 of model and optimizer
     state every N steps or behind a debug flag (`aggregation.py:109-110`). With about 23
     object collectives per update, it would dominate step time on a 24 steps/s recipe.

## Findings by area

Only items not already in the table above are listed. The per-slice notes carry the full
reproductions.

### 1. CLI, Python API, config

Assessment: the low-level pieces (`toml_writer`, `network_config`, `console_settings`,
`model_file` lowering errors, loss-graph checks) are small and careful. The lightweight
commands really are torch-free. The user-facing layer above them substitutes or rewrites
configuration silently, and much of its behaviour has two or three implementations.

- **M** `resume` re-exec drops `-m hypergan` under `python -m hypergan`. Python then tries
  to run a file named `resume` (`cli.py:1890-1898`).
- **M** A missing relative `reference.hndl` is silently replaced by the packaged network
  in `config.load_config`, which the training paths use. `hg.validate` raises an error
  for the same file (`config.py:887-902`).
- **M** `hg.train(model)` with no `run_dir` uses the model name as a nested,
  deterministic directory. A second call with changed hyperparameters fails "differs from
  the original run", while the CLI makes a fresh slugged run (`api.py:798-825`).
- **M** Workspace interpolation rewrites every string in the spec. HNDL `${param}`
  templates raise `KeyError`, bare `{data}` in labels is substituted, and `os.environ` is
  consulted (`workspace.py:198-202`).
- **M** `docs/configuration.md:163-176` documents `architecture`, `outputs` and `frozen`
  network fields, but the validator rejects all three (`model_file.py:71-72`).
- **L**
  - `hg.logs(raw=True, follow=True)` raises `NameError` because `sys` is not imported
    (`api.py:1393`).
  - `doctor` shows the wrong state dir (`cli.py:994`).
  - `sample --sampler` overwrites its own output (`cli.py:2165-2177`).
  - `queue install-service --user` is `store_true, default=True` and so cannot be turned
    off.
  - `model_file.lower()` mutates the caller's spec, so `hg.save` writes both `[data]`
    and `[dataset]` (`model_file.py:760`).
  - `-s name=none` becomes `None`.
  - The hard-coded `metavar` (`cli.py:131`) hides 13 subcommands from `--help`,
    including ones the docs use.
  - The gaussian-grid starter embeds an absolute site-packages path.
- Readability: run-directory naming ×3, diff ×2 (`cli.py:840-958` vs
  `api.py:2169-2275`), role inference ×3. About 10 API aliases are all in `__all__`, with
  differing defaults (`runs_ls(sort="fid")` vs `list_runs(sort="updated")`).
  `hg.metrics` swaps its module `__class__` so it can be called.
- Dead code: `config.write_default`/`DEFAULT_TOML`/`LEGACY_REFERENCE_TOML` (about 130
  lines, used only by tests), `workspace.resolve_run_dir`, and the
  `hasattr(inspection, …)` fallbacks.

### 2. Training core, lifecycle, checkpoints

Assessment: the numerical core is the strongest part of the project. Resume is
bit-exact on CPU with and without previews. `GracefulStop` does no work in the signal
handler. Non-finite screening reads the host once per phase and leaves state untouched
on refusal. Most of the risk sits in the orchestration around the trainer.

- **M** Retention keys on step number, not on which branch of the run a checkpoint
  belongs to. After resuming from an older checkpoint, the live branch keeps only its
  newest checkpoint, and the abandoned branch's higher steps are kept indefinitely
  (`checkpoints.py:748-751,851`).
- **M** A `sample` control request is acknowledged `succeeded` with the *previous*
  preview's path when the new preview fails or is cancelled (`run_controller.py:784-803`).
- **L** `checkpoint_compatibility.py:34-46` tolerates a different GPU model, capability,
  platform and device, while its own comment (`:55-60`) calls these hard failures. Either
  fix the comment or tighten the check.
- **L**
  - Crashed checkpoint writes leave `.pending-*` directories that are never swept
    (`checkpoints.py:207-225`).
  - A run directory without `manifest.json` is routed to resume and fails with a raw
    `FileNotFoundError` (`execution.py:264,301`).
  - `run_train` checks the manifest before taking the lock (`run_controller.py:263,279`).
  - `checkpoints prune` runs without the run lock and reports failed `rmtree`s as
    reclaimed.
  - Component `i` is seeded `seed+i` and the prior `seed+2`, so they collide when there
    are three or more networks (`training.py:228,305`).
- Quality:
  - The resume-config check exists ×5, the checkpoint resolver ×3, receipt parsing ×2.
  - Every resume writes a redundant identical checkpoint, so a fixture ended with three
    identical step-3 checkpoints.
  - `write_checkpoint` re-reads all of `state.pt` to hash it.
  - `DeviceAdam` and per-instance lambda patching are two mechanisms for one guard.
  - Docstrings describe history ("bitwise identical to the previous loop") instead of
    behaviour.
  - The "checkpoint requests" directory also carries `stop` and `sample` actions; call
    them control requests.

### 3. Distributed, workers, qualification

Assessment: the older core is careful work, especially the broker, the prepare/commit
checkpoint split with fencing, and the `_phase` error agreement. The #186 additions on
top of it (multi-node followers, local-SGD strategies, `qualify`, RunPod) hold up only
on the narrow cases their tests exercise. The `pseudo_gradient` maths itself is correct.

- **M** `pseudo_gradient` never synchronizes graph or prior buffers. Spectral-norm
  `weight_u` diverges, and the next sync's replica assertion poisons the run
  (`aggregation.py:469-496`; `parameter_average` does sync buffers at `:324-330`).
- **M** RunPod cleanup is not guaranteed:
  - `terminate_pod` swallows errors and the pod is unregistered anyway (`:359-364`,
    `:438-443`).
  - A `create_pod` timeout leaks the pod.
  - `stopAfter` with `volumeInGb: 20` keeps billing.
  - **The API key is sent in the URL query string** (`:273`) and via `--api-key` in `ps`.
- **M** `qualify` clamps `duration_seconds` to 180, so the 3-minute check can never fire.
  `--dry-run` and skipped suites report `status: passed`. The "multinode" suite runs both
  ranks on one host with `node_rank=0`. The "SHA-256 tamper check" is an unkeyed
  self-hash, and `TEST_INFRA.md` calls the receipts "signed". Validation uses `assert`,
  which disappears under `python -O` (`qualify.py:48-79,407,519-537,617`).
- **M** Multi-node NCCL checkpoint agreement picks `set_device(global_rank)` when a node
  has at least `world_size` GPUs (`distributed_checkpoints.py:150`). Likely.
- **M** `pseudo_gradient` checkpoint state stores four aliased full-model clones per rank
  (`aggregation.py:540-543`). `gpu_inventory` runs `nvidia-smi` with no timeout from the
  queue and status paths.
- **L**
  - The default multi-node rendezvous is `tcp://127.0.0.1:29500`, which hangs on real
    separate hosts.
  - `qualify._run_subprocess_ranks` leaks rank processes on timeout.
  - `RunPodRunner()` replaces the SIGINT/SIGTERM handlers and drops the previous ones.
  - The follower invents its own `run_id`.
  - Strategy dispatch is by name string, so custom strategies cannot opt into gradient
    reduction.
  - `pre_step`/`post_step` are declared but never called.
- **L** The committed `qualification-receipt.json` was overwritten with a dry-run mock
  during this review (I restored it). `--output` defaults to the current directory, so
  running `qualify` or the runner from the repo root dirties the tree. Gitignore receipts.

### 4. Web viewer and frontend

Assessment: the low layers are strong:
- `web_files._open` uses dir-fd and `O_NOFOLLOW`.
- Artifacts are served only from digest-checked indexes.
- The CSP is strict.
- The frontend has no HTML sinks at all, only `textContent`.
- Epoch guards protect every async continuation.
- The built `app.js` matches its sources byte for byte (`npm run check`).

Access control and lifecycle are where it fails.

- **H** Runs that were moved, renamed or rsynced back from RunPod cannot be opened:
  absolute paths in `previews/index.json` make every run API return 400, and the page
  reloads in a loop (`web_service.py:230-234`).
- **M** A `/api/v1/capabilities` call during run start prevents the training stream from
  ever registering, so bootstrap stays "Indexing history" forever
  (`web_server.py:886-895`).
- **M** Live views freeze silently when a service is evicted (`max_services=16`) or its
  tail fails: subscribers keep receiving heartbeats and no frames
  (`web_server.py:648-691`, `web_service.py:483-490`). Likely.
- **M** Blocking projection catch-up and `_discover_runs` run on the asyncio loop
  (`web_server.py:701,711,932,943`). The cold first open of a long run stalls every
  stream.
- **M** `GET /contributions` reads the whole projection log into memory, about 265 MB of
  text at 200k steps (`web_server.py:1217-1230`). Nothing in the frontend uses it.
- **M** Every bootstrap query leaves a permanent `.fold`/`.cache`/`.lock` set (about
  700 KB each) inside the run directory, and the viewer needs write access to runs
  (`web_service.py:621-626`).
- **M** `ensure_dashboard_running` probes without credentials, so token mode restarts
  the dashboard on every `train`. On a version mismatch it stops the viewer registered
  for *this* root, not the port owner, and returns another root's URL
  (`web_autostart.py:942-1004`).
- **M** `stop-server --all` sends SIGTERM and then SIGKILL to PIDs read from *stopped*
  state files, so after PID reuse it can kill a training job (`web_autostart.py:1048-1141`).
- **M** Binding 127.0.0.1 and then opening the printed `http://localhost:PORT` returns 403
  (`web_session.py:119`, `web_autostart.py:1001`).
- **M** A run page left open stops charting partway through a long run, because
  `coarsenAttempts` is never reset (`frontend/src/app.js:913-916,1118-1124`). Likely.
- **L**
  - Dashboard buttons cannot work in token mode.
  - `hypergan view` fails if training already started a dashboard on 8765.
  - The `--port` help claims a free-port search that doesn't exist.
  - The broker re-opens a projector for every active run every 0.1 s, even with no
    browser connected.
- Readability:
  - Three control endpoints are near copies (about 200 lines).
  - SSE depends on uvicorn private internals (`transport._conn_lost`, `send.__self__`).
  - There are three projection paths and two viewer lifecycles.
  - `frontend/src/app.js` is 1,415 lines with a 30-field global `state`.

### 5. Data, datasets, metrics

Assessment: the maths is careful. FID pins weights and the torch-fidelity version,
disables TF32, merges float64 moments exactly and enforces equal counts. The `[-1,1]` to
uint8 conversion is an exact inverse of the loaders. Download/extract is exemplary: no
`extractall`, link and `..` rejection, size caps, SHA pins and staged swaps. The
problems are in scheduling and summarisation.

- **M** A manual `hypergan evaluate` takes no lock. When training starts or closes,
  `_recover_abandoned` marks the evaluation failed and deletes its snapshot, and the
  manual worker then crashes with `FileExistsError` (`metric_evaluation.py:104,292,319-413`).
  The module docstring claims it holds the run lock, and `run_lock` is imported but
  unused.
- **M** `evaluate --bundle typo.pt` silently evaluates the latest checkpoint instead
  (`metric_evaluation.py:170-182`).
- **M** `ImageFolder` serially decodes, preprocesses and SHA-256s every image in its
  constructor: on every training start, every rank, and every evaluation worker, under a
  120 s snapshot timeout. A CelebA-sized folder times out before generating a sample
  (`data.py:74-93`). `ColorizationData`'s pinned manifest is the fix pattern.
- **L**
  - Sharded `ImageFolder` prefetches rank 0's slice on every rank (`data.py:186,412`).
  - The sharded path with `bad_image_policy='skip'` crashes on reshape and would diverge
    across ranks; it is latent today.
  - `ColorizationData` and `ColorWords` don't accept `rank=`, so every rank decodes the
    full batch.
  - Evaluation data exhaustion surfaces as PEP 479 `RuntimeError: generator raised
    StopIteration`.
  - `ItemData` rebuilds `randperm` every batch (4.9 ms at 1M items).
  - The symlink checks in `download_dataset` call `is_symlink()` after `resolve()`, so
    they never trigger (`dataset_download.py:374,455`).
- Quality:
  - Streaming moments ×3, the luma formula ×4, `_digest` ×3, `CIFAR10_SHA256` ×2 (the
    copies can drift), atomic-JSON writers ×4.
  - FID recomputes 50k reference features on every evaluation; cache them keyed by the
    protocol hash.
  - `color_word_metrics` says "four measurements" and returns six.

### 6. Registry, inspection, queue

Assessment: the storage primitives are right: atomic fsync'd writes and kernel `flock`
liveness, so a dead trainer never leaves a stale lock. `ObservationIO` is well bounded,
and `ps` never reads full event logs. Resolution, the queue state machine and the
module structure are not right.

- **M** A runner crash or reboot leaves jobs `running` forever. `queue stop` later sends
  SIGINT to the recorded PID with no identity check, and a PID now owned by another user
  raises `PermissionError` after the queue has already been saved (`queue.py:315-356`).
- **M** `find_live_pid_for_run(runs/cifar)` matches `hypergan train runs/cifar-v2` by
  substring. That PID feeds `stop`, `rm --force` (SIGKILL) and `ps` VRAM attribution
  (`registry.py:322`, `inspection.py:1597,1810`).
- **M** `tag` on an active run is reverted within about 0.25 s by the trainer's manifest
  republish, and it can briefly replace the manifest with a stale copy
  (`inspection.py:1887-1898`).
- **M** `logs -f` drops any event that is mid-write when it polls, because the offset
  advances past the partial line (`inspection.py:2051-2099`).
- **M** `ps` and every resolver call hold the global registry lock for the whole scan
  (`/proc`, all roots, liveness), and the trainer blocks on that lock from
  `write_summary_json` at checkpoints. Slug deduplication makes `ps` O(N²): 2.6 s for
  1,000 runs, 1.38 s of it in `_ensure_unique_slug` (`registry.py:973`).
- **L**
  - Liveness probes take the trainer's exclusive lock, so a probe that lands during a
    trainer start makes the start fail.
  - Queue `wait` jobs hang forever on a SIGKILLed run.
  - A corrupt `runs.json` or `queue.json` is silently replaced with an empty one, so job
    ids restart at 1 and append to old logs.
  - Roots on a temporarily unmounted drive are dropped permanently.
  - `rm --force` deletes without waiting for SIGKILL to take effect.
  - `gc --dry-run` reports `before_size * 0.85` as if it were a measurement.
  - A string-prefix path test lets `/data2` match `/data` (`registry.py:1022`).
  - Queue devices are compared as strings (`-d 0` vs `cuda:0`).
- Readability:
  - `inspection.py` says "operates strictly on run metadata", yet it sends SIGKILL and
    calls `rmtree`.
  - `"/tmp"` is special-cased about 15 times in `registry.py`.
  - `_read_registry_unlocked` is imported by four other modules and rewrites slugs on
    read.

### 7. Tests, docs, packaging, CI

Assessment: the core tests are behavioural and strong:
- bit-exact resume vs uninterrupted runs (`tests/reference/test_recovery.py`,
  `test_image_recovery.py`);
- checkpoint failure injection;
- metric maths against explicit oracles.

The wheel is correct: every `.hndl`, `.toml` and web asset is included, and nothing
extra. CI tests the installed wheel with `python -I` and rebuilds from the sdist. The
weaknesses arrived recently.

- **H** `tests/e2e` (120 tests) is never run by CI or by the default `pytest`. Its
  `harness.py` probes wrap everything in `except Exception: return False`, so a broken
  import turns tests into skips. One probe even string-matches
  `inspect.getsource(ReplicatedTrainer.__init__)` (`tests/e2e/harness.py:44-160`). Today
  it passes on CPU (116 passed, 4 skipped), but nothing would notice if it stopped.
- **M** Several "oracle" tests never call product code. `test_tier2_boundaries.py:208`
  tests a helper defined in the test file, and `test_tier1_aggregation.py:169,243`
  compares harness helpers to constants.
- **M** About 25 receipt-tamper tests across 7 files and about 20 pod-cleanup tests
  across 5 files exercise the shipped mock. The real `RunPodHttpClient` has one test.
- **M** Roughly 11k lines of "tier/challenge/m4" tests were added on 2026-10-15 and
  2026-10-24. Consolidate them to one file per feature.
- **M** The README claims Python 3.11–3.14, but the CI matrix is 3.11–3.13 and there are
  no classifiers. The everyday environment (3.14.8) is untested.
- **M** The sdist ships unit tests that import the excluded `tests/e2e` harness, giving
  2 collection errors.
- **M** There are two changelogs that disagree. `docs/changelog.md` cites a constant that
  doesn't exist. `__version__` is still `2.0.0a1` after #410, and there is no
  "Unreleased" section.
- **M** `GLOSSARY.md` documents the internal resolved schema (`[training]`,
  `[components.*]`) rather than the model-file schema users write (`[train]`,
  `[networks.*]`, `[[losses]]`). `validate` accepts a file that has only those internal
  tables as "Valid".
- **L** `SECURITY.md` asks reporters to open a public issue.
- **L** There is no `docs/README.md`, and 12 docs are orphaned, including the Docker and
  FID guides.
- **L** `hndl>=0.10.0` has no upper bound while it is pre-1.0.
- **L** `.dockerignore` drops `.git`, so image provenance always records `commit: null`.
- **L** README images use relative paths, which break on PyPI.
- **L** The heavy-marker comment ("the same 180 tests") is stale; 222 tests are
  deselected now.
- **L** The first-run gaussian-grid recipe warns "custom … unqualified".
- **Order-dependent failure** (found while running the suite):
  `test_colorization_128_trainer_step_and_comparison_preview` passes alone but fails in
  the full suite on a CUDA machine, because an earlier test initialises CUDA in-process
  and `apply_backend_policy` then refuses the CUBLAS setting. CI has no GPU and doesn't
  see it.
- **Gaps**:
  - no negative tests for `rm`/`gc`/`checkpoints prune` against non-run directories,
    symlinks or ambiguous names;
  - no test that the extras satisfy runtime version checks (which would have caught #12);
  - no doc-command smoke test (all of about 140 documented commands parse today, so this
    is cheap);
  - no end-to-end FID with fake features;
  - no ROCm hardware run behind the "first-class ROCm" changelog claim.

## Repository clarity

Newcomers first see `PROJECT.md`, `TEST_READY.md`, `TEST_INFRA.md` and
`qualification-receipt.json` at the root. All four are Issue #186 agent work products,
and they reference "ORIGINAL_REQUEST §R1", an agent worktree rootdir, and a bug
"escalation" that has since been fixed. `CONTRIBUTING.md` points at a 91 KB execution
ledger. `reports/` holds 72 files.

Suggestions:
- Move `PROJECT.md`, `TEST_READY.md` and `TEST_INFRA.md` to `reports/issue-186/`.
- Delete and gitignore `qualification-receipt.json`.
- Archive the September `reports/` files into `reports/archive/`. Keep
  `human-feedback.md` and the durations file the tests reference.
- Add `docs/README.md` grouped as user guide, recipes and internals.
- Add a short `tests/README.md` covering layout, markers and how to run each tier.
- Rewrite `GLOSSARY.md` in model-file terms.
- Keep one changelog.

## Done well

- Bit-exact CPU resume, including previews, data, prior, penalty and noise RNG streams,
  EMA and optimizer state. RNG isolation around observers is thorough.
- Checkpoint write order: pending dir and fsync, then manifest, then rename, then fsync
  of the parent, then the atomic `latest.json` pointer. Loads use
  `torch.load(weights_only=True)`. `_restore_trainer` validates on throwaway generators
  before changing anything.
- The distributed prepare/commit split with controller-id fencing, and a broker that
  reaps ranks on coordinator death.
- FID protocol hygiene and safe dataset extraction.
- Web file serving, CSP, a frontend with no HTML sinks, bounded queues, and a
  reproducible bundle.
- Preview retention (`previews.py`) and `ObservationIO` bounds.
- Lightweight CLI commands that really are torch-free, and a correct, minimal wheel with
  CI testing the installed artifact.

## Suggested order of work

1. **Turn CI green.** Fix or delete the two stale tests, and invert the ROCm partition
   default to use markers.
2. **Data-loss fixes.** The strict destructive resolver (#1–#3), the queue claim and stop
   fixes (#13), and negative tests for each.
3. **Wrong-results fixes.** K3P history (#4), evaluation starvation (#8), "best" and
   "latest" (#9), retention by branch, the `--devices` override (#15), and the
   `pseudo_gradient` buffers.
4. **First-run fixes.** `-t dcgan` and the README `hg.new` example (#11), the
   torch-fidelity pin (#12), the main-guard detection (#14), and the docs that disagree
   with the validator.
5. **Viewer access control.** Authenticate all `/api/**` routes in token mode, allow only
   known Host values in wildcard mode, and limit the dashboard to its own root (#10).
   Store relative preview-index paths.
6. **Mark multi-device experimental.** Refuse the RunPod live mode, refuse unsupported
   multi-node features up front, and validate `checkpoint_every` against
   `sync_every_steps`.
7. **Structural cleanup**, in the order of the cross-cutting list: the `RunInfo` reader,
   catalog `direction`, de-duplication, test-detection removal, then the large-module
   splits. Do it as separate small PRs, each with behaviour-preserving tests.

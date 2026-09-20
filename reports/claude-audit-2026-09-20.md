# HyperGAN revival audit — 2026-09-20

Author: Claude (Fable 5.1) at the owner's request. Four Opus research agents
gathered evidence (architecture review, claims-versus-evidence verification,
ParticleGAN relationship, legacy/product comparison); the coordinator
spot-checked the sharpest findings against the source and wrote this report.
Repository state audited: local `develop` at `a9a8bebf`, with `3ea1974c`
(checkpoint compatibility and release provenance, another agent's work) landing
during the audit. Both commits are ahead of `origin/develop` `f8f5ae15` and
unpushed. `master` is at `291ddccd` (2021-01-24).
Nothing in the repository was modified other than adding this file. No GPU
training was started.

## 1. Verdict in one page

**The three-day rewrite on `develop` is real, honest and well built. It is not
yet a product, and it is not yet "version 2".** What exists is a
~13k-line training runtime with unusually good durability, observability and
test discipline, wrapped around ~1,350 lines of actual GAN numerics that call
ParticleGAN 0.5.0 through nine symbols. What does not exist yet is the thing
1,183 stargazers will judge a "2.0" by: training a picture from a folder of
images in one command, a recipe catalog, samplers, export, a Python API, a
published package and a tutorial.

The ledger (`reports/resurrection-status.md`) is accurate claim by claim. Every
PR it cites exists and is merged; every FID number matches its receipt; failed
gates were preserved rather than deleted. Its problem is aggregate impression:
34 sections titled "checkpoint" or "acceptance" describe a system whose product
default (CUDA) has no automated regression test, whose best image run reached
20% of its own 200k-step target, and whose newest entry describes a commit that
exists only on this machine.

The scientific claim behind the revival holds up as far as it has been tested.
The HyperGAN port of the ParticleGAN SAGAN/frozen-ResNet18/MoG recipe reached
EMA FID50k 19.38 at 40k updates on CIFAR-10 against the historical 19.64 at
the same point, with exact bitwise recovery across interruption. That is the
single most valuable asset in the repository and the launch should be built
around it.

**Recommended path:** stop widening the runtime. Declare a `2.0.0b1` whose
scope is "train an image GAN from your own folder, on one GPU, with the
ParticleGAN recipe, and get PNGs, FID and a resumable run". Ship the
distributed stack, export and cluster work as 2.1+ gates. Section 8 sequences
this; section 9 lists the decisions only the owner can make.

## 2. Repository state (verified today)

| Item | Finding |
|---|---|
| Public default branch | Still `master`. GitHub visitors see the 2021 "HyperGAN 1.0" README, `pip install hypergan`, dead Discord/Twitter badges. |
| `develop` | 423 commits since 2026-09-18, all authored by Martyn Garcia with no co-author trailers, although the reports describe subagent authorship throughout. |
| Unpushed work | Two local-only commits: `a9a8bebf` (repeatable `train`) and `3ea1974c` (checkpoint compatibility version and release provenance). No PR, no GitHub CI, no external review. The ledger records owner authorisation for direct commits to `develop`. Push them or open a PR so protected CI runs. |
| Open PRs / issues | 0 open PRs. 5 open issues (#166, #186, #213, #224, #303), untouched since 2026-09-19 while the ledger claims much of their work shipped. #303 (viewer) has zero comments and all five checkboxes unchecked. |
| CI on develop | Last 20 runs: 19 green, 1 red. Run `35521061583` (Foundation CI after PR #334, Windows viewer API job) failed and is not recorded anywhere. |
| Branch protection | Required checks on `develop`, but `enforce_admins` is off. |
| Local hygiene | 57 branches, 54 registered worktrees, 50 branches already merged into `develop` and never deleted. Seven worktrees have uncommitted changes. |
| Scope change today | Four worktrees (`checkpoint-compatibility*`, `replicated-compatibility`, `release-provenance`) produced `3ea1974c`. It records an owner clarification that reverses the earlier "no cross-version compatibility" rule: HyperGAN source hashes and package versions are now provenance, not resume rejection keys, and an explicit `hypergan_checkpoint_version` gates known-incompatible state. `AGENTS.md` and the ledger were updated in the same commit. The four worktrees and branches are merged and can be pruned. |
| Archive tags | 15 `archive/resurrection-2026-09-18/*` tags exist locally and on origin. Preservation claims verified. |
| Disk | 1.1 GB `resurrection-backups/`, 9.1 GB `training-runs/`. |
| GPUs | Neither GPU is training right now. A viewer process for `train-develop` has been up since 11:46. |

## 3. What has actually been proven

| Claim | Evidence | Verdict |
|---|---|---|
| Historical preservation (tags, bundle, restore) | PRs #299–#301; `resurrection-backups/2026-09-18-032714-preservation` | Verified |
| CPU foundation and reference tests | 504 foundation tests reproduced by the audit in 41 s; 838 tests collect | Verified, in CI |
| Native CUDA train/resume | PR #317; `2026-09-19-gpu-core/installed-cuda-tests.log` "10 passed" | Verified once by hand; never in CI |
| Two-GPU NCCL replicated training and recovery | PR #318; `2026-09-19-nccl-core` "19 passed" | Verified for the synthetic toy only; the image recipe is rejected by the replicated path |
| CIFAR-10 FID trajectory 33.47 → 25.72 → 21.60 → 19.38 at 10k–40k | Four `cli-fid50k_train-*.json` receipts | Verified exactly |
| 200k reproduction (historical 12.53) | No run has passed 57,287 steps; the 57k run measured no FID | Not started |
| Interval FID evaluation | PR #354 merged at `f8f5ae15`; CPU and CUDA tests | Verified |
| Feedback fixes (durable stop, persistent viewer, ordered FID) | PRs #336–#343; newer manifest shows `SIGINT` with zero lost steps versus the older run's `KeyboardInterrupt` with 492 lost steps | Verified, and the improvement is visible in real run data |
| Repeatable `train` resumes matching config | Local commit `a9a8bebf`; logs match claims | Partly verified, not on GitHub |
| Checkpoint compatibility version and release provenance | Local commit `3ea1974c`; 694 added lines incl. 5 test files; ledger claims subagent review | Read, not independently run; not on GitHub |
| Web viewer and browser UI | PRs #320–#328; Playwright/Chromium in CI | Verified, in CI |

Two report statements are now wrong and should be corrected. The image
execution report says the 40k run's manifest reads `stopped` at step 40,000;
the file reads `interrupted`, `KeyboardInterrupt`, step 41,492 with 492
possibly lost updates. The ledger's "all required checks passed" refrain omits
the red Windows job above.

Plan milestone status (`reports/resurrecting-hypergan-plan-2026-09-18.md`):
1–4a done; 5 partial (I1–I3 landed, I4 at 40k/200k with no diversity or
nearest-neighbour evidence); 6, 6a, 7, 9, 10 not started; 8 partial. Seven of
ten milestones are open and the three hardest have not begun.

## 4. Architecture assessment

Line budget of `src/hypergan` (13,316 lines, 58 files):

| Cluster | Lines | Share |
|---|---|---|
| Numerical loop (`training.py`, `distributed_training.py`, `recipes.py`, `numerical_policy.py`, `distributed.py`) | 1,345 | 10% |
| Checkpoints and run state | 1,428 | 11% |
| Run orchestration (controller, executions, workers, preflight, profiles) | 3,461 | 26% |
| Metrics and observation (incl. WASM reducer host) | 3,077 | 23% |
| Web viewer | 1,807 | 14% |
| Previews, images, data | 1,477 | 11% |
| Config and CLI | 720 | 5% |

Plus 17,288 lines of tests, 1,234 lines of frontend JS bundled to 540 KB, and
a 178-line Rust crate compiled to a 153 KB WASM reducer.

**The actual G/D/prior/EMA step is `ReferenceTrainer._update` at
`src/hypergan/training.py:212-288`, about 77 lines.** Everything else exists to
call it safely and observably. That ratio is defensible for a product, but the
orchestration was built ahead of the product it serves.

What is genuinely good and should be protected:

- Durability primitives are textbook: fsync-then-rename atomic JSON, digested
  append-only event journal with torn-tail repair, cross-platform run locks
  (`run_state.py`). Checkpoints never come from an exception handler.
- Every `torch.load` is `weights_only=True` behind a SHA-256 check.
- The `Execution` protocol seam in `run_controller.py` cleanly separates the
  torch-free lifecycle from the numerical adapters.
- Tests are behaviour tests. No `conftest.py`, no `Mock()`, 237 `monkeypatch`
  fault injections on real production symbols, real Gloo processes, real
  Chromium. CUDA and browser tests assert rather than skip. This honours the
  "no blanket skips" rule in `AGENTS.md`.
- Zero `TODO`/`FIXME` markers; no silent CPU fallback; unqualified recipes warn.

Where it is over-built for a 2.0:

- **The distributed stack cannot run the flagship recipe.**
  `execution_profiles.py:18-29` rejects independent phase draws, backend
  policies, fused Adam, reused components and prior-bound inputs. The CIFAR
  recipe uses all five. About 2,660 lines of replicated/accumulation/worker
  machinery currently serve only the 2-D Gaussian toy.
- **The numerical loop is written three times** (`training.py:212-288`,
  `distributed_training.py:265-358`, `distributed_training.py:435-624`) and
  has already drifted: the distributed variants call `graph.generate` without
  the `prior=` argument the native loop passes (`training.py:207` versus
  `distributed_training.py:287,470`).
- A Rust→WASM reducer, fuel-limited wasmtime host, digest-checked loader and a
  pinned Rust toolchain compute two reducers (`mean/v1`, `envelope/v1`).
- `event_views.py` (529 lines) is a general revisioned map/projection engine
  with one production caller; `ArtifactDescriptor` is never instantiated
  outside its test. The `project` and `contributions` CLI verbs exist for it.
- Two parallel HTTP supervisors (`web_launch.serve` and `web_autostart.Viewer`).
- 19 CLI commands, 13 of them observability/ops, serving a catalog with one
  recipe. `hypergan train --help` shows six distributed timeout flags to a
  user who has trained nothing.

## 5. Relationship to ParticleGAN

ParticleGAN 0.5.0 (released 2026-09-18, the newest version; four releases in
three days) is 1,565 lines across nine modules. Its stabilising idea is the
learnable prior: a table of optimizable latent particles, with the MoG variant
giving each a Gaussian neighbourhood, kept from collapsing by a VICReg row
regularizer, trained against a relativistic-paired logistic loss with the
re-centred b-cap gradient penalty. That is the "embarrassingly simple" fix and
it is entirely inside the library.

HyperGAN's coupling is thin and clean: `GANLoss`, `GradientPenalty`,
`ParticleRegularizer`, `learning_rate_scale`, three prior classes and
`ucd_scores`, at four call sites, no subclassing, exact pin. Two issues:

- **The science boundary and the package boundary do not coincide.** The
  architectures that produced FID 12.53 (SAGAN generator, frozen ResNet18
  feature critic, routing encoder) are repo-only in ParticleGAN, not in the
  wheel. `src/hypergan/image_components.py` is a 289-line hand port bound to
  upstream's module construction order via explicit attention seeds. That is
  the one unhealthy part of the coupling. Either push those modules upstream
  into a `particlegan.models` namespace and import them, or declare
  architectures HyperGAN's surface and drop the bit-identity constraint once the
  200k comparison is done. `AGENTS.md` already says source identity is not a
  requirement.
- `ucd_scores(..., num_classes=1, target='time_class', num_steps=1)` at
  `image_components.py:222` is a conditioning helper used as a degenerate
  scalar reduction purely for bit identity. Drop it after qualification.
- `recipes.make_prior` (`recipes.py:73-86`) hand-rolls the prior dispatch and
  already lags upstream (`FreshGaussianPrior`, `fresh_gaussian`). Use
  ParticleGAN's own factory.
- Upstream still has no root `LICENSE` file and PyPI metadata has a null
  license. The owner authorised MIT distribution of the port on 2026-09-20, but
  a one-line upstream fix closes a real distribution gate.

The right framing for 2.0 is "HyperGAN is the product layer over ParticleGAN":
the TOML `[prior]`, `[adversarial]`, `[gradient_penalty]`,
`[prior_regularizer]` sections map 1:1 to ParticleGAN constructor kwargs, and
that mapping should be a tested contract with defaults pulled from
`get_recipe()` rather than retyped. Keep the exact pin; upgrade only through a
qualification PR with a numerical comparison.

## 6. Product gap: 1.x versus develop

| Capability | 1.x (master, 1.0.6) | develop (2.0.0a1) |
|---|---|---|
| Train from an image folder | `hypergan train folder/ -s 32x32x3` | Gone. `train CONFIG --run-dir`; data is a config section; image components exist only for CIFAR |
| Presets | 32, listable with `-l` | 1 (`reference/100gaussians`, a 2-D toy) plus 2 example files |
| Loss / trainer / hook menu | 10 losses, 4 trainers, ~11 hooks | 1 loss, 1 loop, penalties as config sections |
| Layer DSL | JSON string layers | Dropped for ordinary Python factories (deliberate, and right) |
| Samplers | 8 incl. walks and grids | One `sample` path; JSON for the toy, PNG grid for images |
| Export / Python API | `build` to ONNX, `hg.GAN(...)` documented | Absent / undocumented |
| Checkpoint and resume | optimistic | Complete, durable, exact, attempt lineage |
| Multi-GPU | 7 nominal backends | Real NCCL replication, coordinator-death recovery, toy-only today |
| Metrics / FID | none | Pinned Inception FID, interval scheduling, viewer charts |
| Viewer | Tk window, Electron app elsewhere | In-tree streaming web UI, autostarts with train |
| Package | published `hypergan` | not published; 8 extras (`train, image, cifar, fid, reducers, web, test, dev`) |
| Docs | GitBook, tutorials, 16 examples, Discord | 24 contract docs, 0 tutorials, 3 examples |

First-run experience on develop today, verified by actually running it: every
lightweight README command works first try, a 5-step CPU run completes in
under 3 seconds, `serve` starts, and error messages are clear. But
`hypergan new demo` writes `device = "cuda"` into a 2-D toy that never needs a
GPU, the happy path ends with a 4 KB JSON manifest and four 2-D points, and
training a real image model requires a manual CIFAR download, two hash-pinned
weight files placed in the torch hub cache, four extras and a separately
indexed torch wheel. The runtime is honest about all of this; every run
manifest says `qualification: unqualified`.

## 7. Concrete defects to fix before any tag

Numbered by priority. File references verified by the coordinator where marked.

1. **Viewer binds `0.0.0.0` with no authentication by default** and
   autostarts on every `train` when `[web]` is installed (`cli.py:36-37,105-106`,
   `web_autostart.py:369`; verified). `web_session.py:64` disables the
   Host-header check precisely for that bind, and `web_server.py:209` exposes a
   mutable console endpoint. PR #337 delivered owner intent but nobody wrote
   down the trade-off. Default to `127.0.0.1`; make `0.0.0.0` opt-in.
2. **RNG seed collision** (`training.py:182`; verified). The penalty stream is
   hardcoded `seed + 3` while prior/data offsets are configurable and
   unchecked. The CIFAR recipe (`seed 24002`, `prior_seed_offset 3`) gives the
   prior and penalty generators the same seed 24005. Harmless for b-cap, live
   for interpolating penalty arms. Add a uniqueness check and a configurable
   penalty offset.
3. **Distributed loop drift** (`distributed_training.py:287,470`; verified).
   Missing `prior=` argument, masked only by the profile validator. Collapse the
   three loops into one implementation with injected all-reduce hooks.
4. **Resume still dies on any dependency patch release.** `3ea1974c` fixed
   half of this: HyperGAN's own hashes and version are now provenance
   (`checkpoint_compatibility.py:29-49`), so a hypergan 2.0.0 to 2.0.1 upgrade
   no longer un-resumes runs. But `validate_runtime` still requires exact
   equality of every other runtime field, so torch 2.14.0 to 2.14.1 or a NumPy
   patch still refuses to resume. Gate on the fields that affect numerics and
   warn on the rest.
5. **Three divergent checkpoint-path validators** (`execution.py:92-100`,
   `checkpoints.py:210-217`, `distributed_checkpoints.py:393-400`). The
   weakest is reachable from the public `hypergan.training.resume()`.
6. **Dependency pins that will rot within weeks**: `wasmtime>=48,<49`,
   `starlette>=1.6,<1.7`, `uvicorn>=0.53,<0.54`, `torch-fidelity==0.3.0`
   (last released ~2021, with a hard runtime assertion at
   `image_metrics.py:76-80`).
7. **Torch private-API override** in `DeviceAdam` (`training.py:121-133`)
   silently stops applying if torch renames the graph-capture health check;
   the pin `torch>=2.6,<3` allows that.
8. **Process-global side effects**: `numerical_policy.apply_backend_policy`
   mutates `os.environ` and global torch flags with no restore;
   `run_signals.py:26,28` calls `os._exit()`, which kills an embedding host.
9. **`on_event` contract changes with the profile**: native accepts any
   callable, replicated rejects lambdas/closures (`bounded_observer.py:34-38`).
10. **CUDA tests are not in CI** (`.github/workflows/ci.yml` gates on
    lightweight, reference, reducers, web). The product default has no
    regression safety net beyond a maintainer remembering to run
    `tests/cuda`. A self-hosted runner on this machine fixes it.

Also: a foundation test spawns `python -I` and fails on this machine because
`hypergan` is only importable via a user-site editable `.pth`; CI covers
Python 3.10–3.12 while the dev machine runs 3.14; `image_data.py:42` uses
`pickle.load` on CIFAR batches (standard, but arbitrary code from a data dir).

## 8. Path to version 2

**Define the release first.** The plan's ten milestones make cluster training,
external benchmark submission and ONNX deployment release gates. That is a
2.x roadmap, not a 2.0 gate list. Reserving the "2.0" number until two-node
training is proven means shipping nothing for months while the 1.0 README
stays public. Proposed split:

**2.0.0b1 — "new foundation, one recipe that works" (target: 2–3 weeks of
focused work).** Scope:

1. Push `a9a8bebf` and `3ea1974c` (or open one PR for both) so protected CI
   runs on them. Nothing else lands on `develop` without CI again.
2. Fix defects 1, 2, 3, 4 and 10 above. Everything else in section 7 can be a
   tracked issue.
3. **Ship an image recipe that works from a folder of images.** A builtin
   SAGAN-class G/D pair sized from `image_folder` arguments, using the
   ParticleGAN MoG/b-cap/VICReg defaults, no pinned third-party weights
   required. `hypergan new mymodel --data ~/pics --size 64`, `train`, PNG grids
   every N steps, `sample` writes PNGs. This single item closes most of the
   perceived regression from 1.x. Pretrained-discriminator CIFAR remains the
   measured reference recipe alongside it.
4. Make `hypergan recipes` list at least three entries (toy, image-folder,
   CIFAR-pretrained) and make `new` default to the image-folder recipe.
5. Collapse extras to `hypergan[train]`, `hypergan[web]` (and `[dev]`). Fold
   `image`, `cifar`, `fid`, `reducers` into `train`.
6. Cut for 2.0 (archive, do not delete): the Rust/WASM reducer and its
   toolchain, the generic `event_views` engine and the `project`/`contributions`
   verbs, the OpenAPI generator, one of the two HTTP supervisors. Roughly
   1,200–1,500 lines and a Rust toolchain removed with no user-visible loss.
7. Finish the 200k CIFAR run on GPU 1 with interval FID, publish the FID vs
   steps curve, fixed-latent grids, fresh samples and reproduction commands.
   Report whatever the number is.
8. Write one tutorial and one human-facing migration note ("your 1.x JSON and
   checkpoints do not load; here is the 2.0 equivalent of the dcgan preset").
   Switch the GitHub default branch to `develop` or merge develop to master.
9. Set up a self-hosted CUDA CI job on this machine for `tests/cuda`.
10. Publish `2.0.0b1` to PyPI with a README that leads with the ParticleGAN
    story and the CIFAR curve.

**2.0.0 (stable).** Add a documented Python API (`hypergan.load(run_dir).sample()`),
a native inference bundle and ONNX export of the generator with the prior as
bundled data, grid and interpolation samplers, and a second measured recipe
(e.g. a 64×64 face or texture set from a folder). Collect beta feedback first.

**2.1+.** Make the replicated trainer support independent phase draws, reused
components and prior bindings so the image recipe runs on two GPUs; then the
real two-host run with a concrete paid allocation; then the external
benchmark submission. The existing distributed code is a head start, not a
blocker; it should stop being a gate.

**Process changes that cost nothing:**

- Delete the 50 merged branches and their worktrees; keep only active ones.
- Reconcile issues #186, #213, #303 with what shipped, or close them.
- Add co-author trailers to future agent-written commits; the current history
  attributes 423 AI-assisted commits to one human with no record, which the
  project's own preservation rules would object to in anyone else's history.
- Correct the two wrong statements noted in section 3 in their reports.
- Shorten the ledger. Each milestone should be a five-line entry with a link;
  the current 526 lines make it hard to see that seven of ten milestones are
  open.

## 9. Decisions only the owner can make

1. Is 2.0 "one GPU, one great image recipe, resumable, with FID" (ship in
   weeks) or "cluster-qualified and benchmark-submitted" (ship in months)?
   This report recommends the former with the latter as 2.1.
2. Do the image architectures move upstream into ParticleGAN, or does HyperGAN
   own them and drop bit-identity with upstream after the 200k comparison?
3. Viewer default: loopback with opt-in `0.0.0.0`, or keep open-by-default and
   document the exposure?
4. Is the WASM reducer worth keeping for a reason not written down anywhere?

## 10. Immediate next actions

1. Push or PR the two local commits, let CI run.
2. Fix the viewer bind default and the seed collision in one small PR each.
3. Start the 200k CIFAR continuation on GPU 1 (the run is stopped, nothing is
   training) so evidence accumulates while product work proceeds.
4. Prune merged branches and stale worktrees; record the prune in the ledger.
5. Write the `2.0.0b1` scope into the plan and mark milestones 6, 6a, 7 and 9
   as post-2.0 gates.

## Sources

Ledger and plan: `reports/resurrection-status.md`,
`reports/resurrecting-hypergan-plan-2026-09-18.md`,
`reports/image-training-plan-2026-09-19.md`,
`reports/image-training-execution-2026-09-20.md`,
`reports/image-next-steps-2026-09-20.md`, `reports/feedback-2026-09-20.md`,
`reports/feedback-implementation-2026-09-20.md`. Evidence directories under
`/home/martyn/dev/hypergan/resurrection-backups/` and run data under
`/home/martyn/dev/hypergan/training-runs/`. GitHub state via `gh` on
2026-09-20. ParticleGAN: https://pypi.org/project/particlegan/ ,
https://github.com/255BITS/ParticleGAN , local checkout
`/home/martyn/dev/ParticleGAN`, installed 0.5.0 in
`/home/martyn/dev/hypergan/image-training-env`.

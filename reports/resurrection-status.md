# Resurrection execution ledger

Authoritative design: [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md). Updated 2026-09-20 (America/Denver).

Colorization discriminator simplification (2026-09-20): owner authorized a single
unconditional `D(X)` / `D(G(z))` and PR/merge. The new projected DINOv3 path uses
frozen pretrained features and random channel/local-spatial projections, then
trainable attention/head; E/G, 4,096 fixed-sigma particles and encoder-only L2
remain unchanged. Original discriminator/config/environment are preserved.
[Implementation and validation report](colorization-projected-2026-09-20.md).
Clean installed source `de11816d` matches all 65 Python files; 858 fast tests pass,
185 heavy tests deselected. Actual GPU-1 DINO image double backward/frozen masks
and eight full-manifest CLI updates pass. Resume reached ten; exact full restore
matched 948 tensors / 407,302 values, and all three 512-output metrics plus g/x/gray
viewer PNGs pass. Independent review found no blockers. The verification run and
viewer are stopped. start-color.sh selects the fresh, unstarted projected run.
[PR #360](https://github.com/HyperGAN/HyperGAN/pull/360) is open. Owner clarified that the full heavy suite is for prereleases; the
outdated AGENTS.md instruction is corrected. The active suite was stopped at
121 passing tests / zero failures, with 64 selected tests unfinished; this is
not a complete heavy pass. The develop gates are fast, focused GPU and GitHub
checks. Original two-path checkpoint restore also matched 1,050 tensors / 407,414
values. Normal PR CI at `588a5cc4` passed all required checks; the earlier manual
run's unchanged browser test timed out, with evidence preserved. Next: merge after
GitHub checks for the final policy/docs update.
Evidence: `../resurrection-backups/2026-09-20-colorization-projected/` outside repo.


Colorization demo acceptance (2026-09-20): the owner requested a 256x256
logo colorization demo with learned grayscale E, 4,096 hard-routed fixed-sigma
particles, frozen DINOv3 plus discriminator attention, independent metrics, and
GPU-1 verification followed by a stopped launcher for direct control. The
[acceptance report](colorization-2026-09-20.md) records the formulation,
provenance, complete inventory, failure handling and evidence. Owner authorized
PRs/merges and pushing the existing local develop history; local/GitHub develop
at `a83d7072` preserved all earlier work and closed prior PRs #355/#356.

[PR #357](https://github.com/HyperGAN/HyperGAN/pull/357) adds bounded 256px PNG
previews with g/x/gray shelves and explicit routed particle IDs. [PR #358](https://github.com/HyperGAN/HyperGAN/pull/358)
adds the colorization recipe/data/models/metrics and includes that preview head.
The combined head `5a34cc76` passed Foundation CI and repository integrity
([CI run](https://github.com/HyperGAN/HyperGAN/actions/runs/35561553137)).
PR #358 merged into develop as `48e0b34648d732d57bca1cdf5feeb3430515f352`;
PRs #357 and [#359](https://github.com/HyperGAN/HyperGAN/pull/359) are also marked
merged through preserved ancestry. #359 implements the owner's subsequent
request: GitHub heavy jobs run only for master pushes and master-targeting PRs;
develop keeps fast, viewer, reducer and integrity checks. All 36 gate-result
combinations passed validation, and GitHub confirmed the two heavy jobs were
intentionally omitted. The standalone CI-policy PR encountered the known Windows
CRLF fixture failure; the combined passing head includes its fix from #357.

CI initially caught a preview captured-default byte-limit regression;
`377974f8` / `49f206b5` fixes it and all 42 focused preview checks passed.
The combined installed fast suite passed 851 tests with 185 heavy tests explicitly
deselected in 24.46s. Older superseded CI runs that still included heavy tests
were canceled after the owner changed the policy.

Full preparation attempted 426,445 image paths and accepted 426,343: 404,757
train / 21,586 held-out. Exactly 102 exclusions (99 truncated, two unreadable,
one animated) are explicitly hash-pinned. Three valid one-bit PNGs were
revalidated and retained after adding mode support. Original files are unchanged;
initial failure and final exclusion reports are preserved. The manifest SHA256 is
`e336ddec33364aa9300108fa9566f51a1e321d632067f553ffb97f410e3fbdb0`.

Acceptance: 58 focused local tests, 34 installed focused checks, and 12 final
installed data/metric checks passed. Combined preview/API/browser checks passed
67 initially; the sole isolated-server environment failure passed after installing
the declared web extras. Actual DINO CPU/GPU-1 double backward and frozen masks
passed. The fixture CLI crossed lazy b-cap step 8, resumed to 10, and recovered
from earlier step 4 to 5. The actual launcher/full-inventory run also trained to
8 and resumed to 10; complete restore matched 1,050 tensors and 407,414 values.
All three full 512-output held-out snapshot metrics passed; actual Chromium
confirmed all g/x/gray grids. Source->sdist->wheel packaging matched all 65 runtime
Python files. The validated run records clean source `165dea8e`; the launcher
was subsequently updated to clean `90339371` with the preview guard fix, again
matching all 65 runtime files. No trained quality or distributed
qualification is claimed.

Ready launcher: `/home/martyn/dev/hypergan/training-runs/start-color.sh`, physical
GPU 1 pinned by UUID, dedicated installed environment and `train-color` run.
The verification run and its viewer are stopped; the owner run was left fresh.
Recipe defaults are batch 16, 200k total updates, 1k checkpoints and 100-update
previews; the owner controls startup and stopping. Evidence/builds/commands and
preexisting-ledger backup: `/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-colorization/`.
Main-checkout HNDL/research edits are preserved separately. No paid compute or
release. All task branches are preserved in develop. Owner next runs
start-color.sh and assesses color diversity and logo structure over training.
Unrelated preexisting HNDL/research edits remain outside these commits.

Resume on the other identical GPU, and the heavy-test gate (2026-09-20): the
owner restarted `training-runs/start.sh` and resume failed with the bare
`Resume runtime/topology differs from checkpoint`. Diagnosis: two identical RTX
A6000s enumerate in an unstable order with `CUDA_DEVICE_ORDER` unset, so
`CUDA_VISIBLE_DEVICES=0` landed on the other card and only `cuda.uuid` /
`cuda.visible_devices` differed from the step-6359 checkpoint. The owner asked for
a warning that names what differs. Two owner-authorized Opus subagents worked in
Agent worktrees. Item 18 (`e8ada8b0`, merged `9c4985dc`): `validate_runtime`
flattens both runtime dicts to dotted paths, rejects with every differing field
and both values, and warns-and-continues when the difference is confined to
`DEVICE_IDENTITY_KEYS`; the warning goes to stderr, the run manifest and the
`resume` event. Replaying the owner's checkpoint against the current card warns
and returns. Item 20 (`f236af15`, merged `f3b21832`; 19 was taken by the
throughput work in flight): the owner asked that heavy tests be gated and run
intentionally. A measured `heavy` marker (>= 1 s, which is exactly the set that
spawns subprocesses or multi-rank jobs; 185 of 1000 tests) is deselected by
`addopts`, run with `-m heavy`, registered with `--strict-markers`, never skipped;
CI gained `heavy-lightweight` and `heavy-reference` jobs required by the
foundation gate, and AGENTS.md states the rule. Fast suite on merged develop:
814 passed, 1 failed (`test_distribution_contains_only_supported_package`,
wheel-only, known) in 25 s; heavy suite on the merged tree: 185 passed in
19 min 40 s. Both subagents reported that the
Agent tool created their worktree off `291ddccd` (pre-resurrection history) and
fast-forwarded to `develop` before working; coordinators should verify the base.
Also: tests that re-invoke `python -I -m hypergan` cannot see this machine's
user-site editable install, so a plain `python3 -m pytest` fails 17 CLI tests
spuriously; run the suite from a venv (`--system-site-packages` plus an editable
install of the checkout). Neither branch was pushed; `develop` stays local. The
owner's run can restart; `CUDA_DEVICE_ORDER=PCI_BUS_ID` in `start.sh` pins the
card. Blockers: none. Next: continue the plan's CUDA save/resume and two-GPU
qualification.

Training throughput (2026-09-20): the owner reported the CIFAR run on one A6000
at 10-12 steps/s against 14-15 for the ParticleGAN source and asked for the GPU
to be utilized as fully as possible, with less host blocking if needed. Two
owner-authorized Opus subagents profiled `hypergan train` on GPU 0 (ablation runs
`training-runs/perf-*`, in-process harness, dispatch census) and diffed one step
against ParticleGAN `9e9ce96` `experiments/train_cifar_ae_sagan.py`. Models are
at exact parameter parity (G 930883 / D 3588324 / E 428320); the gap was host
synchronization and backend policy: ~103 blocking device reads per step from
per-parameter `isfinite` gradient checks and pre-backward loss checks, a whole-
graph EMA of 193 launches, and the recipe's deterministic/no-TF32 backend block
(+32% throughput when matched to the source). Server, progress cadence, metrics
`every_steps`, previews and data loading measured at or below 0.5 ms/step.
ParticleGAN reference on GPU 0 under the same box load: 16.5 steps/s at 38% GPU
utilization, so both loops are launch-bound at batch 64.
PR #355 (`perf/trainer-host-syncs`, `2bc0a9d3`) fuses each phase's loss and
gradient screening into one device-side check with one host read, and fuses the
EMA with `_foreach_lerp_`/`_foreach_copy_`; messages, refusal-before-step
semantics, checkpoint contents and numerics are unchanged (20 metric rows bitwise
identical). Measured on a quiet GPU 0: 69.8 -> 61.4 ms/step (deterministic
backend) and 47.0 -> 41.2 ms/step (source backend flags). New tests
`tests/reference/test_nonfinite_refusal_and_ema_fusion.py` (18 passed) and
`tests/cuda/test_trainer_host_syncs.py` (10 passed); the existing suites were not
run at the owner's request (heavy tests being removed). GitHub CI: all Linux/macOS
checks pass; the Windows viewer failure is a pre-existing CRLF assertion in
`tests/web/test_viewer_dev_mode.py`, unrelated. Merged locally as `82f9df11`;
`develop` is not pushed, so the PR stays open until the owner pushes. A derived
recipe `training-runs/cifar10-pretrained-20260920/cifar10-fast.toml` with
`start-fast.sh` applies the source backend flags for a new run directory; adopting
them in the shipped recipe is an owner decision (exact within-run resume is lost).
Owner decision (same day): determinism is not needed and resume must simply
work. Back-to-back on GPU 0 under the same load, ParticleGAN `9e9ce96` ran 19.9
steps/s and HyperGAN with the merged fix plus source flags 21.9 steps/s (45.7
ms/step), so the port now matches the source. PR #356
(`perf/recipe-source-backend`, merged locally as `a23dfbfd`) ships
`examples/cifar-pretrained-sagan.toml` with the source backend (TF32, cuDNN
benchmark, nondeterministic kernels, `deterministic_features = false`) and keeps
the strict policy as a documented opt-in in `docs/cifar-recipe.md` and
`docs/image-core.md`. The example resolves on CPU; suites not run at the owner's
request. The owner's live run directory keeps its recorded configuration, so
the faster policy needs a new run directory (`training-runs/start-fast.sh`).
Blockers: none. Next: apply the same fused screening to
`distributed_training._reduce_gradients` with two-GPU qualification; beyond
parity, throughput needs fewer launches (CUDA graphs or larger batch).

Checkpoint compatibility and release provenance (2026-09-20): the owner
clarified that HyperGAN release/source SHAs must be recorded, not used as resume
rejection keys. This supersedes the earlier no-cross-version-compatibility policy
below; AGENTS.md and recovery guides now carry the updated contract. Work starts
from local `develop` at `a9a8bebf` (GitHub remains at merged PR #354, `f8f5ae15`).
The owner's direct-develop authorization remains in effect. Three subagents
implemented provenance/distributed recovery and independently tested/reviewed the
combined change in external worktrees.

Native and replicated recovery now use `hypergan_checkpoint_version=1`. Existing
schema-1 checkpoints without that field use version 1; malformed, unknown or
known-incompatible versions fail explicitly before CLI viewing and before state
loading. HyperGAN module hashes and package versions remain provenance; external
component/dependency hashes, numerical configuration/runtime, data, fixed topology,
payload integrity and durable event boundaries remain enforced. Ranks within a
single job must still agree on their complete current build identity. This is
supported state recovery across HyperGAN releases, not a claim that arbitrary
algorithm changes reproduce identical learning trajectories.

Git SHA/dirty state is stamped into wheels and source distributions; rebuilt
wheels retain it without Git. Editable checkouts report their current SHA/dirty
state; unavailable provenance is explicit. Runs keep original `initial_source`
and current `source`, with each attempt and new checkpoint retaining provenance.
Existing runs recover original provenance from their earliest saved attempt when
available; fallback provenance is labelled. Failed started attempts identify the
attempted current SHA, never silently inherit the preceding release's identity.

Read-only owner-run diagnosis found stopped/durable step **57,287**, recorded
source `74aab6c6ba591af30e5d1583cccad5956e75d364`, and changes in three HyperGAN
module hashes. All saved HyperGAN hashes match that historical commit. The revised
native adapter successfully restored that actual checkpoint on physical GPU 1
under the owner's Python 3.14.7 runtime, without executing updates, creating an
attempt or modifying run/checkpoint metadata. The first diagnostic used `-I`,
which hid this installation's user-site torch; rerunning with the actual user-site
runtime resolved that environment setup issue. No package reinstall or run restart
was performed for the owner.

Validation: source → sdist → wheel builds succeeded; all 60 installed Python
runtime files byte-match the candidate. Base-only installation passed **536**
foundation tests (39.73s); integrated installed CPU acceptance passed **608** tests
(273.09s), including actual fresh-group distributed checkpoints and public CLI.
After the two provenance review fixes, **124** focused installed checks passed
(22.61s). Native GPU-0 exact repeated/uninterrupted state acceptance passed. Final
actual two-GPU CUDA/NCCL public train/stop/resume/earlier-snapshot/observation
acceptance passed (1 complete workflow, 35.54s), including exact complete-state
comparisons and worker cleanup. This remains generic-runtime qualification, not
image-workload or multi-host qualification.

Subagent acceptance additionally passed 29 real Gloo checkpoint tests, 61
native/foundation compatibility checks and 42 provenance/commit-authority tests.
Independent final provenance review passed 24 focused checks. Actual clean-Git
sdist→wheel and isolated installed/editable proofs preserve the SHA and correct
dirty state; the dirty candidate's sdist/wheel stamps match. No failures/skips were
hidden. Whitespace and local documentation-link checks passed. The original live
run and earlier attempt/checkpoint artifacts remain untouched.
Evidence/build/test logs and exact commands:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-checkpoint-compatibility/`.
No paid compute, release, checkpoint rewrite or automatic long-running training.
Next: rerun the owner's existing `bash start.sh` against updated editable develop,
then continue the image plan and its distinct multi-host/image-quality gates.

Repeatable training CLI (2026-09-20): owner requested subagent implementation
and a direct local commit to `develop` for this change, overriding the usual PR
workflow. Verified clean local/fetched/GitHub `develop` at
`f8f5ae15c61ff48a88bedbe6a6818ebd9e91cde0` and PR #354 merged before starting.
Three subagents implemented, tested and independently reviewed in external
`repeatable-train*` worktrees.

`hypergan train CONFIG --run-dir RUN_DIR` now creates a new run or resumes the
latest complete checkpoint in an existing run. The full resolved configuration,
including metrics and the effective `--steps` total, must match. Comments and
formatting do not matter. Omitted profile/checkpoint/preview options inherit the
saved values. Config rejection occurs before viewer startup and is rechecked
under the writer lock before custom factories. Explicit `resume` retains earlier
snapshot selection and observation-only changes. Completed repeats add no
training updates but still record a restored attempt and final artifacts.
Invalid directories/checkpoints never become fresh runs. See the updated
[recovery guide](../docs/recovery.md).

Validation: source → sdist → wheel build; all 58 installed Python runtime files
byte-match candidate source. Base-only installed foundation tests passed 504
checks in 39.51 seconds with no torch/ParticleGAN installed. Focused installed
under-lock config mutation regression passed (1 test, 2.05 seconds), including
custom metric argument `true` versus `1`. Native CUDA on physical GPU 0 passed
exact uninterrupted/repeated complete-state equality, completed-repeat zero
updates, immutable earlier artifacts and config rejection without run mutation.
The GPU proof used Python 3.12.13 and torch 2.14.0+cu130. Integrated installed CPU
acceptance passed 533 tests in 157.04 seconds, covering foundation, native recovery,
real two-process replicated CLI execution and core CLI. The final under-lock
regression above was added after that suite started and passed separately. All
checks completed without failures or skips; whitespace and local documentation
links passed.

Commands: `git fetch origin develop`, GitHub branch/PR queries, `python -m build`,
installed `python -I -m pytest` over foundation/recovery/public execution/core CLI,
and the evidence directory's `cuda-acceptance.py` with GPU 0's UUID as the sole
visible GPU. Evidence/build/test logs and source hashes:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-repeatable-train/`.
The initial isolated validation environment lacked its inherited build dependency
path; that setup error was fixed before a successful package build. Existing
installations, owner training runs and GPU 1 were untouched. No PR, push, release,
paid compute, older-checkpoint migration or new two-GPU qualification is included.
Next: use the repeatable command for a supported run in an updated installation,
then continue the image plan and its remaining qualification gates.

Interval FID implementation (2026-09-20): [PR #354](https://github.com/HyperGAN/HyperGAN/pull/354)
adds configured step intervals, one asynchronous snapshot evaluator, explicit busy
skips, current-run recovery and visible schedules/cancellation before any result.
The [implementation report](interval-evaluation-2026-09-20.md) records the
configuration, native/replicated execution and UI contract. Final installed
acceptance passed 48 integration/API checks, 22 browser checks and native CUDA
complete-state/recovery acceptance. All 58 installed runtime files match source.
A fresh three-update CIFAR CLI run on GPU 0 completed automatic FID128 at source
step 2; this is plumbing evidence only. Protected checks and the linked PR/merge
receipt establish final integration state. Evidence and full CPU logs:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-interval-evaluation/`.
The public CIFAR example schedules FID50k every 10,000 steps. The owner's live
`train-develop` run on GPU 1 and its installation were untouched. Next: test the
new schedule in an updated installation/new supported run, then continue the
image plan. No paid compute, release or new two-GPU CUDA qualification occurred.

Metric optimization acceptance (2026-09-20): the owner-requested audit and worker
fixes are recorded in [metric optimization](metric-optimization-2026-09-20.md).
Scalar transport [#344](https://github.com/HyperGAN/HyperGAN/pull/344) passed
protected CI and merged into `develop` as `895f9c7681f81569627f07d243712f75ed408b20`.
The combined implementation [#347](https://github.com/HyperGAN/HyperGAN/pull/347)
preserves the reviewed heads of #345–#346 and #348–#353, including asynchronous
metrics/I/O/previews/callbacks/control reads and cancellation/recovery fixes.
Its protected checks and merge receipt are the authoritative final integration
state; do not merge the superseded slice states independently.

Measured source `2c749af9`: 20 synthetic and 12 actual CIFAR GPU trials retained
identical complete numerical state across all conditions and the baseline.
CIFAR controller/observation overhead was 1.16% (95% interval 0.41–1.92%); the
custom-worker condition added 0.66% with 75% explicit busy sampling drops. A 1%
upper bound is **not established**. Native CUDA save/resume, worker isolation and
scalar transport passed 11 checks; final source `0955fa6f` passed two additional
strict deterministic CUDA snapshot/histogram checks. All experiments masked only
physical GPU 0's UUID; GPU 1 and the owner's training environment were untouched.
Final installed source `0955fa6f` passed 518 CPU acceptance tests in 361.87 seconds,
including real signal/fault/recovery checks, with all 56 runtime files matching
the archived source. Exact commands and identities are in the report and receipts. Benchmark windows exclude final durability and
observer drains; replicated snapshot file transport and native Python control
callbacks remain explicit synchronous boundaries.

Evidence and command/result receipts:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-metric-optimization/`.
No release, paid compute, old-checkpoint migration or new two-GPU qualification
is included. Next: continue the owner workflow/image plan below from the merged
`develop`; use this report's overload/cadence measurements when selecting passive
metrics. A future distributed performance qualification must use an explicitly
available second GPU and then real multi-host execution.

Current cutpoint: the [feedback implementation report](feedback-implementation-2026-09-20.md) records all seven workflow fixes in PRs #336–#342 and their combined acceptance. The coordinator integrates the reviewed branch heads through `feat/feedback-integration`; consult its protected PR and receipts for final merge state. Planning PR #335 merged at `17a3cb307c8ff7b12a47600e855ca2d0c52409eb`. The actual run was stopped at step 41,000 on inspection, and its frozen environment remains untouched. Native CUDA and two-GPU generic runtime recovery passed; the reference's historical 40k EMA FID50k/train remains **19.37753221446735**. **Next:** finish protected feedback integration if still open, then continue the [five-step plan](image-next-steps-2026-09-20.md): owner workflow testing, the preserved 200k reference continuation and evidence, actual two-GPU image qualification, and external benchmarking. Check live processes before training; paid compute, real multi-host execution and release remain separate gates.

Metrics implementation checkpoint: the [implementation report](metrics-implementation-2026-09-19.md) links the file-backed metrics, shared WASM reducers, public streaming API/browser, ordinary Python factories, manual snapshot evaluation and supervised CLI startup slices. Native CUDA, full two-GPU numerical/recovery checks, actual browser and installed-package proofs accompany the protected PRs. Samples remain separate modality-neutral artifacts. The performance report records measured costs and explicitly unestablished throughput targets; no database, federation or paid compute was introduced. Final integrated checks and merge receipts are retained in the durable metrics evidence directory. The separate numerical/distributed core cutpoint above remains unchanged.

## First checkpoint

The clean CPU foundation is merged in PRs [#301](https://github.com/HyperGAN/HyperGAN/pull/301) and [#300](https://github.com/HyperGAN/HyperGAN/pull/300). The integrated baseline is `86c7a3cf360a8975d48f53589b018d48719cad49`; its [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35419513982) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35419513976) both passed. Continue from the current develop and the next checkpoint below; do not restart the historical audit.

The next release integrates on `develop`. The coordinator reviews and merges passing PRs; bounded subagent work uses external worktrees. Master remains the historical stable line until a separately qualified release.

## Accepted decisions

- On 2026-09-20 the owner confirmed authorship of both projects and explicitly permitted MIT distribution of the ParticleGAN port, with no additional attribution requested. Normal source revision and experiment provenance remain recorded. This resolves the earlier source-distribution hold; it does not change the recorded numerical or quality acceptance gates.

- The first image workflow retains ParticleGAN MoG + exact b-cap and uses pretrained discriminator features immediately; G starts from scratch. The owner's `feat/cifar-ae-gan-pretrained-encoder` run is the new source target (original FID50k12.5345 at200k, batch64), superseding the earlier residual candidate. Ship qualified defaults, image grids and complete recovery, then comparable CIFAR-10 evidence. See the [image plan](image-training-plan-2026-09-19.md) for protocol and PR gates. Ordinary Python configurability remains; unsupported combinations cannot pass as successful no-ops.

- GPU execution is the product default (owner direction, 2026-09-19). `new` targets CUDA; explicit CPU runs remain available for small correctness fixtures. This box has two RTX A6000 48 GB GPUs authorized for local validation. Native CUDA and real two-GPU work proceed alongside public distributed integration; no paid allocation is implied.

- Superseded by the 2026-09-20 explicit compatibility-version policy above: older-checkpoint compatibility was not required (owner clarification, 2026-09-19). Checkpoint formats and source identities may break across implementations without migration or compatibility work. Reliable save/resume, corruption checks and recovery from earlier snapshots within supported current runs remain requirements. Historical compatibility notes below record behavior, not a commitment to maintain old runtimes.
- Recipe configuration supports custom generator/discriminator/encoder/auxiliary components, explicit I/O, losses and regularizers. Warn on unqualified combinations; reject actual incompatibilities. Default b-cap and VICReg follow the pinned upstream reference.
- Colorization and super-resolution influence the implemented conditional I/O contract. The paired example is a synthetic fixture, not a qualified image recipe.
- Development version: 2.0.0a1. ParticleGAN dependency: 0.5.0. Tested CPU tuple: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3. The package records actual runtime/source identity and does not certify arbitrary installations.
- Replace archived desktop viewers with the [optional local browser server](local-web-view-plan-2026-09-18.md). Planned local training defaults to attempting the server when its optional extra is installed; `--no-server` disables it. Cluster workers stay headless, standalone `serve` reconnects to run state, and the first UI is read-only. Standalone `serve`, the public API/browser and supervised automatic train/resume startup are implemented in the metrics slices below. `--no-server` keeps local CLI execution headless; browser opening remains explicit.
- No paid compute or release publishing occurred. Reserve $30 Modal credit for later cluster qualification; additional spending needs a concrete agreed allocation.

## Completed implementation and evidence

| ID | Deliverable | Evidence |
| --- | --- | --- |
| F1 | Historical preservation and branch backlog | 15 published archive tags; 59 named refs restored; full fsck passed. Six legacy remote branches and five PRs retired; clean historical worktrees removed after recheck. [Preservation report](resurrection-preservation-2026-09-18.md) |
| F2 | Develop bootstrap and protection | PR #299 merged with master ancestry preserved. Strict required checks now Repository integrity and Foundation CI, bound to GitHub Actions. No force-push or admin bypass used |
| F3 | Approved plan and continuity | PR #298 merged; AGENTS.md and this ledger guide future sessions |
| F4 | Modern package, lightweight CLI and CI | PR #301. Fresh wheel/sdist installation; Python 3.10–3.12 lightweight checks across Linux, Windows and macOS |
| F5 | Flexible configuration and CPU reference | PR #301. Controlled upstream-based update parity, paired encoder/conditioning/reconstruction, finite reference, EMA inference reload, actionable errors and interrupted-run status |
| F6 | Legacy retirement and current documentation | PR #300. Superseded runtime/research/tests/viewers removed; current quickstart, config guide, migration note and extraction ledger retained |
| F7 | Integrated local acceptance | Cleaned wheel/sdist build; 26 installed-package tests passed outside checkout; ten-command default/paired walkthrough passed. PR #300 and final develop required CI passed |

PR #301's complete [Foundation CI run](https://github.com/HyperGAN/HyperGAN/actions/runs/35419292962) passed all nine lightweight platform/Python jobs and the Linux CPU reference. Initial Windows failures exposed a test's incorrect console-script search location; the test now uses Python's installation scheme and all Windows jobs passed. No failures were hidden or skipped.

Local acceptance used `/tmp/hypergan-foundation-verify/bin/python` and a built wheel, not imports from the checkout:

```sh
python -I -m pytest /path/to/checkout/tests --import-mode=importlib -q
# 26 passed
```

The walkthrough exercised help/version/recipes/new/validate/train/inspect/sample plus the paired example. Default commands emitted no stderr; the paired recipe emitted its expected unqualified warning. Default and paired samples had finite shapes [16,2] and [8,2]. Durable local evidence is under the preservation directory's `foundation-acceptance/`; temporary environments can be recreated from README/CI.

## Complete issue audit and viewer decision

[PR #302](https://github.com/HyperGAN/HyperGAN/pull/302) publishes the [complete audit](issue-audit-2026-09-18.md) and [local web contract](local-web-view-plan-2026-09-18.md). Three subagents reviewed all 112 original issues and 299 comments. The coordinator posted 17 individual explanations, closed 13 obsolete/completed/deferred issues, and kept four concrete requirements open. All 95 historical closed issues were left untouched. GitHub state and posted comments were re-read and verified; the report links every action.

The five remaining open issues are [#166](https://github.com/HyperGAN/HyperGAN/issues/166) dataset acquisition, [#186](https://github.com/HyperGAN/HyperGAN/issues/186) distributed training, [#213](https://github.com/HyperGAN/HyperGAN/issues/213) resume-safe sample numbering, [#224](https://github.com/HyperGAN/HyperGAN/issues/224) containers, and newly created [#303](https://github.com/HyperGAN/HyperGAN/issues/303) optional local viewer. Their comments/body define acceptance gates. Closure of installation issues describes unreleased develop behavior; historical PyPI packages were not repaired or republished.

The audit added explicit requirements for wall-time stopping, serialized save requests, last durable checkpoint visibility, data/class-map compatibility, preprocessing/image diagnostics, beginner command context and supervisor-friendly live logs. These are incorporated into the workstream table and next checkpoint. The server remains planned: no viewer code, flags or new dependency was added in this documentation slice.

Audit evidence is preserved outside the repository at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-18-issue-audit/`, including the snapshot, reviewed decisions, action receipt and final inventory. Reports have exact 112/112 unique issue coverage and resolving local file links; PR #302 merged after the full required CI checks passed. No GPU execution, paid compute or release occurred.

## Core recovery and image data checkpoint

[PR #304](https://github.com/HyperGAN/HyperGAN/pull/304) merged at `b0aec260cf757d9d4acb9d7da5fb1cd12fb5ddf3`: optional Pillow extra, deterministic `image_folder` batches, content/preprocessing/class-map identity, sampler state, and torch-free `data-check`. Its installed wheel passed **57 tests**; [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35420997367) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35420997272) passed, including all nine lightweight platform/Python jobs.

[PR #305](https://github.com/HyperGAN/HyperGAN/pull/305) adds complete CPU training checkpoints, explicit `resume`, cooperative stop limits, isolated inference/progress RNG, immutable attempt artifacts and versioned observed/durable progress. Implementation through `6f33d64e06e9dfd45385ef5b59d69427f4649940` passed **81 installed-package tests**, using a new wheel built through the source distribution and tested outside the checkout. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3, Pillow 12.3.0. A separate base-only environment passed 20 lightweight tests after CI exposed and we fixed the missing-NumPy installation message; the regressions cover each optional numerical dependency. PR #305 merged at `19a3be70bf8e6b7f784f8430ec8bb23439202d83` with all required checks passing. Its post-merge [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35421655270) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35421655276) also passed.

Three subagents implemented and independently reviewed data, recovery and upstream/image integration. Tests compare complete uninterrupted/resumed checkpoint state across stochastic image components and shuffled data, including BatchNorm and nonpersistent buffers, trainability/modes, Adam/prior/EMA, named/global RNG and sampler position. Regressions cover half-D/G failure, failed checkpoint writes, malformed checkpoints, runtime/config/data/class changes, custom factory source hashes, older-checkpoint replay (including zero updates), lock conflicts, partial event tails, sample collisions and live subprocess progress. [The core report](core-recovery-2026-09-18.md) records details and upstream provenance; [recovery commands](../docs/recovery.md) and [image data](../docs/image-data.md) define the supported contracts.

ParticleGAN still lacks an explicit root license file at the pinned revision. No image architecture was copied and no upstream changes were made. The selected direct image experiment also uses different D/G draws and schedule/optimizer behavior from the synthetic CPU reference; simply plugging its networks into the existing toy loop would not reproduce that experiment. Keep its extraction and numerical qualification gated.

No GPU execution, dataset downloads, cloud allocations or release publishing occurred. The five open issues remain open: #166 acquisition, #186 distribution, #213 the complete periodic-preview/counter workflow, #224 containers, and #303 viewer. Current artifacts have no-overwrite regression coverage, but no periodic preview scheduler or browser exists.

## Core observation and CPU collective checkpoint

[PR #306](https://github.com/HyperGAN/HyperGAN/pull/306) merged at `6991acf325f109e503c953376cad968c6df0fc15`. Small internal Gloo primitives and eight real two-process fixtures cover differentiable global gathers/means, unique prior IDs, standardized MoG dense derivatives, lazy b-cap derivatives and bounded missing-rank failures. Its sdist-built wheel passed 89 installed-package tests; all required [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35422519344) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35422519277) checks passed. The merged tree equals the reviewed head. [Distributed numerical documentation](../docs/distributed.md) explains scaling and the limits: this is not a DDP trainer, optimizer/EMA parity result, launcher or distributed recovery.

[PR #307](https://github.com/HyperGAN/HyperGAN/pull/307) merged at `c1162869240abef2030142a53b73dddde6ba7dda` and completes W1, documented in the [core observation checkpoint](core-observation-2026-09-18.md): bounded event cursors, immutable periodic EMA previews with retention, and an attempt-bound manual checkpoint request/receipt protocol. The CLI exposes `events`, `checkpoint`, `--preview-every`, `--preview-keep` and `--no-previews`. These stay separate from browser/server work; local observation commands import no training runtime. [The guide](../docs/observation.md) documents exact bounds, acknowledgements, reconnect and failure behavior.

The combined sdist-built wheel at `c234ca76fa9172b39d9d686c07ad81139a8afde2` passed **140 installed-package tests** outside the checkout in 31.84 seconds, including all eight distributed fixtures. A separate base-only installation passed **59 lightweight tests** in 1.31 seconds, with torch, ParticleGAN, NumPy and Pillow absent. Three subagents implemented and cross-reviewed the core pieces; the coordinator integrated CLI workflows and checked full state equality with stochastic components, partial writes, preview failures, request retries and live subprocess submission. All required PR CI passed. The merged tree equals the reviewed head; post-merge [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35422875484) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35422875491) also passed.

No GPU execution, paid compute, dataset downloads, upstream copying or release occurred. All five tracked issues remain open; #213's periodic counter/replay regression contract is now implemented. Remote/browser authentication, image grids and worker supervision are not supplied by the local files protocol.

## Complete replicated CPU update checkpoint

[PR #308](https://github.com/HyperGAN/HyperGAN/pull/308) merged at `79573c5e10647fcea674c03660dd245d4be63457`. The [core distributed report](core-distributed-2026-09-18.md) records the internal execution path: fixed-world-size CPU Gloo training with explicit post-backward gradient averaging, complete D/G/prior/auxiliary/Adam/EMA updates, differentiable global RA, global unique VICReg, exact lazy penalties and deterministic global-data rank slicing. A bounded [worker supervisor](../docs/cpu-workers.md) stops and reaps the group on failure. [Numerical contracts](../docs/distributed.md) describe custom-objective, buffer and floating-point limits. The public train/resume service remains single-process.

The sdist-built wheel at implementation head `9e7c4732` passed **162 installed-package tests in 67.36 seconds** outside the checkout; a separate base-only installation passed **59 tests in 1.40 seconds**. Three subagents supplied implementation and independent review. An initial CI failure exposed a degenerate zero-gradient penalty fixture; it was corrected without widening tolerances. Final [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35424327926) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35424327931) passed. The merged tree equals the reviewed head. No GPU or paid compute occurred.

## Fixed-topology CPU recovery checkpoint

[PR #309](https://github.com/HyperGAN/HyperGAN/pull/309) merged at `96222fc65b5b25d01df32e694a929e826dada95b` and adds complete rank snapshots, strict runtime/source/data/topology agreement, staged all-rank readiness before atomic rank-zero publication, and fresh-group restore of each rank's own state. The [recovery guide](../docs/distributed-recovery.md) defines the distinct checkpoint format and caller-owned locking/lifecycle. A crash after a complete commit may leave a valid checkpoint even if the job reports failure; a missing payload or readiness agreement cannot publish an incomplete checkpoint.

The sdist-built combined wheel at `a9716764` passed **185 installed-package tests in 140.10 seconds** outside the checkout, including all 21 checkpoint cases and the production supervisor/lock/train/save/restart workflow. A separate base-only installation passed **59 tests in 1.36 seconds**, with all optional numerical/image dependencies absent. Three subagents implemented and cross-reviewed the code. Tests compare full state exactly across fresh worker groups with shuffled image-folder data, labels and stochastic components; failures cover half-updates, missing ranks before/after payload transfer, corrupt/incompatible state, custom load-hook mutation and partial live restore. Required PR CI passed; the merged tree equals the reviewed head. Post-merge [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35424764223) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35424764247) also passed.

This is an internal CPU execution/recovery contract. Public `train`/`resume`, live events, previews and checkpoint requests still use the single-process run service. At that checkpoint accumulation was explicitly one; the next checkpoint below adds replay. No DDP hooks, GPU/NCCL or actual cluster is qualified. No GPU training, paid compute, dataset download, upstream architecture copy or release occurred. The five tracked issues remain open, and the optional browser server remains planned. See the report for exact source/test evidence and the durable receipt location.

## CPU accumulation checkpoint

[PR #310](https://github.com/HyperGAN/HyperGAN/pull/310) merged at `628a406cb2efa1a377b1ed03b7528356f8e8904d` after required PR CI passed. The [accumulation report](core-accumulation-2026-09-19.md) and [usage contract](../docs/accumulation.md) document complete CPU updates with accumulation greater than one. Detached full-batch logits preserve RA derivatives, and global unique/full-table VICReg is evaluated once. Microbatch replay retains one network graph at a time; the full input/prior/logit state still exists. Custom sample-independent components remain configurable with an unqualified warning, and custom separable objectives declare mean/sum aggregation. Registered forward mutation, replay mismatch, nonfinite combined losses and invalid controls fail explicitly.

Three subagents handled implementation, independent numerical/memory/recovery acceptance and shared-service design review. Complete Adam/EMA updates match accumulation one across all 12 adversarial combinations at unchanged tolerances. A controlled network measured 395,380 bytes versus 98,784 bytes of peak saved activation storage at factors one and four; this is not total memory or a GPU benchmark. Fresh-group accumulated recovery is exact for the tested stochastic/sampler contract, changed accumulation is rejected, and a rank failure on the second G microbatch preserves the previous durable checkpoint.

The source-distribution-built wheel at implementation/test head `07f99694` passed **194 installed-package tests in 185.19 seconds** outside the checkout. A separate base-only installation passed **59 tests in 1.30 seconds**, with torch, ParticleGAN, NumPy and Pillow absent. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0. Required [PR Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35425978102) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35425978123) passed. The later [post-merge Foundation run](https://github.com/HyperGAN/HyperGAN/actions/runs/35426214214) failed in the Windows/Python 3.12 checkpoint-request lock test; its CPU reference job passed. The next session diagnosed a pre-lock initialization write race, documented in the [lifecycle checkpoint](core-lifecycle-2026-09-19.md). The durable receipt records the reviewed head, merge and branch preservation.

The [shared run-service design](distributed-run-service-design-2026-09-19.md) is a proposal for the next milestone. Workers must wait outside Gloo between updates; one controller owns run state and checkpoint commit authority. Parent-death fencing and bounded preview execution are public-integration gates. No distributed CLI, browser server, GPU execution, paid compute, dataset download, copied upstream image architecture or release was added. The five tracked issues remain open.

## Shared lifecycle and Windows lock checkpoint

[PR #311](https://github.com/HyperGAN/HyperGAN/pull/311), the [lifecycle report](core-lifecycle-2026-09-19.md) and [developer guide](../docs/run-lifecycle.md) record the extraction. One torch-free controller owns locks, attempts, stop policy, events, checkpoint requests and sample counters. The single-process adapter owns trainer/batch state, strict restore, numerical updates, snapshots, inference and callback RNG/thread isolation. Successful shutdown precedes terminal success; failure cleanup retains the run lock and never snapshots a partial update. Public command signatures and numerical code remain unchanged.

Three subagents handled extraction, independent acceptance and the Windows CI failure. Review fixed warning handlers escaping RNG isolation and stale cleanup diagnostics across attempts. Both run and request-queue locks now acquire their OS lock without an unsafe initialization write. Regressions cover actual process contention and release, including empty lock files; required platform CI confirms native Windows behavior before integration.

The sdist-built wheel at source/test head `4a8c1703` passed **214 installed-package tests in 188.23 seconds** outside the checkout, plus **74 tests in 1.62 seconds** in a separate base-only installation. Default, paired and stochastic old/new runs also matched full checkpoint state and normalized lifecycle results exactly through stop/resume, previews and a manual save. Source hashes now include the extracted files, deliberately rejecting old-source native and internal distributed checkpoints; those runs require their original installation. PR #311 merged at `e615afabd62d2cf3ba9e57a1b22e184add118b83`. Required PR CI and both post-merge workflows passed: [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35427372368) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35427372272). The durable session receipt preserves the reviewed head, merge and branch attribution.

No distributed CLI, server, GPU execution, paid compute, dataset download, upstream architecture copy or release was added. All five tracked issues remain open. Durable evidence is under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-lifecycle/`.

## CPU execution profile and preflight checkpoint

[PR #312](https://github.com/HyperGAN/HyperGAN/pull/312) and the [preflight report](core-preflight-2026-09-19.md) record separate CPU profile TOML, torch-free structural checks and construction-only runtime checks in bounded workers. `hypergan preflight CONFIG --profile FILE` validates execution settings; `--runtime` additionally checks CPU state, actual runtime/source/data identity, initialized replicas and recovery declarations. Global batch remains recipe-owned, and preflight timeouts stay outside numerical identity. Public `train`/`resume` remain single-process.

Three subagents implemented the profile and runtime and independently tested rank failures, data-identity disagreement, hung startup, strict comparisons and worker cleanup. Review corrected single-process preflight inadvertently initializing a Gloo group; it now matches native single-process construction. Worker output stays on stderr while the CLI emits JSON. Unsupported recovery declarations are reported with reasons rather than mistaken for a failed construction or proven restore.

The sdist-built wheel at source/test head `8dc0a42e` passed **286 installed-package tests in 225.67 seconds**, plus **126 base-only tests in 1.83 seconds** with optional numerical/image dependencies absent. The installed CLI walkthrough passed both profiles without creating a run. Existing numerical training, lifecycle, checkpoint, recipe and data implementations are unchanged. See the [profile guide](../docs/execution-profiles.md) for report limits and the checks preflight does not perform. PR #312 merged at `ea824231d015dc4ea814c3f08c2b9d16dd07fd22`. Required PR CI and both post-merge workflows passed: [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35428879433) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35428879431). Preserved branch attribution is recorded in the durable integration receipt.

No GPU execution, paid compute, dataset download, copied upstream architecture, server or release occurred. All five tracked issues remain open. Evidence: `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-preflight/`.

## CPU worker commands and parent publication checkpoint

[PR #313](https://github.com/HyperGAN/HyperGAN/pull/313), the [worker-service report](core-worker-service-2026-09-19.md), [command guide](../docs/cpu-worker-service.md) and [checkpoint guide](../docs/distributed-recovery.md) record the internal APIs. Persistent workers agree run/attempt/sequence/operation before each command and wait outside Gloo between commands. An independent broker owns the numerical processes and monitors exits, coordinator death and deadlines even while the caller is idle or a worker is stuck in native code.

Checkpoint preparation now returns a bounded descriptor after complete rank staging and readiness. It leaves canonical generations and latest untouched. The parent retains the run lock, checks successful all-rank command completion and current health, and uses a process-bound authority to validate expected identity, staged bytes and command lineage before publication. The authority does not itself acquire or verify the OS lock. New supervised code must use this split path; the legacy standalone helper retains its rank-zero publication contract.

Three subagents implemented the service, checkpoint split and independent acceptance. The sdist-built wheel at source/test head `e18395c8` passed **356 installed-package tests in 279.85 seconds**, plus **176 base-only tests in 2.08 seconds** with all optional numerical/image dependencies absent. Tests prove exact accumulated shuffled/stochastic recovery, coordinator-death rank reaping, stale receipt rejection, zero-update checkpoint republication and rejection when a rank fails after staging but before its command acknowledgement. One older assertion was corrected to distinguish hidden preparation metadata from completed generations; the failed initial run remains preserved. PR #313 merged at `d65c338cf9dfbd89ab5e3abeb1a21167fe7bc47f`. Required PR CI and both post-merge workflows passed: [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35431096223) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35431096213). Full evidence is recorded in the checkpoint report and durable receipt at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-worker-service/`. Distributed source identity changes intentionally reject older-source distributed checkpoints. Native single-process numerical/checkpoint source remains unchanged.

Public train/resume, browser serving, GPU/NCCL, real clusters and paid compute are outside this slice. All five tracked issues remain open.

## Internal replicated run-service checkpoint

[PR #314](https://github.com/HyperGAN/HyperGAN/pull/314), the [replicated service report](core-replicated-service-2026-09-19.md) and [developer guide](../docs/replicated-run-service.md) record the shared-controller integration. A candidate attempt identity is fixed before strict restore and persisted only after success. Resolved numerical execution identity is separate from mutable service deadlines. Workers acknowledge complete steps and global metrics; parent-only checkpoint publication remains fenced and fatal group errors cannot be swallowed by optional saves.

Filesystem events, coalesced save requests, lost-acknowledgement reconciliation and monotonic sample reservations use the existing controller. Final copied-state inference artifacts, when a completed/restored batch is available, and successful group shutdown precede terminal success. The internal adapter rejects periodic preview and callback options before run mutation while isolated bounded observation remains unimplemented. Public train/resume remain single-process.

The sdist-built wheel at source/test head `f1f6c550` passed **392 installed-package tests in 337.14 seconds**, plus **205 base-only tests in 2.12 seconds**. Three subagents implemented and cross-reviewed the controller, adapter and independent acceptance. Complete accumulated state matches across shuffled synthetic/image data recovery; tests cover older zero-update selection, strict mismatches, lost save acknowledgements, fatal save/export failures and actual coordinator death/takeover. Native default and paired workflows also match the prior installed version exactly. PR/merge checks and verified branch preservation are recorded in the report and durable receipt at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-replicated-service/`. This controller source change deliberately rejects older-source native and distributed checkpoints; their original installation remains required. Numerical update code is unchanged. No GPU or paid compute was used; all five tracked issues remain open.

## Bounded replicated observation checkpoint

[PR #316](https://github.com/HyperGAN/HyperGAN/pull/316), the [bounded observation report](core-bounded-observation-2026-09-19.md) and [developer guide](../docs/replicated-observation.md) record copied-state preview capture, rendering without a training process group, shared parent preview publication and bounded progress delivery. The existing independent broker now has an explicit one-worker group-free mode; parent orchestration remains Torch-free. Preview execution is synchronous with at most one pending render. Callback delivery uses a fresh process for every event and disables itself after a runtime failure while filesystem progress continues.

Three subagents implemented and cross-reviewed the renderer, callback service and independent whole-job acceptance. The source-distribution-built wheel at source/test head `39611e2c` passed **434 installed-package tests in 465.60 seconds**, plus **239 base-only tests in 7.89 seconds**. Final review narrowed optional progress handling to an explicit error type so native RNG/thread-restoration failures remain fatal; that correction at `cda591d7` passed 55 focused tests before the final package/CI gates. The report records strict observation-on/off recovery, bounded faults, coordinator-death takeover and the installed walkthrough; the durable receipt records the final rebuilt package results, PR checks, integration and preserved branches. Current-run recovery and corruption checks remain required; PR [#315](https://github.com/HyperGAN/HyperGAN/pull/315) records that older-checkpoint compatibility and migrations are outside scope. No GPU execution or paid compute is part of this slice.

## GPU-first native execution and remaining fault gates

[PR #317](https://github.com/HyperGAN/HyperGAN/pull/317) and the [GPU core report](core-gpu-2026-09-19.md) record GPU-first project creation, one-device native CUDA training and complete recovery, plus all six remaining shared-controller persistence/partial-update fault cases. Three subagents implemented and independently reviewed the changes. Actual two-GPU communication, autograd and linear Adam parity passed; the separate [NCCL readiness report](gpu-readiness-2026-09-19.md) records the timeout-diagnostic delay and verified worker cleanup. This is not yet full replicated CUDA GAN training.

Focused acceptance passed 37 CPU tests, 9 CUDA tests and the required two-GPU diagnostic. Final installed-package/PR integration results and exact source identity are recorded in the GPU core report's durable receipt directory. New projects select `cuda`; `--device cuda:1` and `--device cpu` are explicit alternatives. Existing lightweight and CPU CI remain required. No paid compute, dataset download, upstream architecture copy or release occurred.

## Supervised local CUDA replication checkpoint

[PR #318](https://github.com/HyperGAN/HyperGAN/pull/318), the [replicated CUDA report](core-replicated-cuda-2026-09-19.md) and [internal workflow](../docs/replicated-cuda.md) extend the existing independent worker broker and shared run controller to rank-owned GPUs. The `cuda-replicated-nccl` profile keeps recipe device selection unindexed, binds rank *r* to visible GPU *r*, and uses device-aware numerical and metadata collectives. CUDA completion precedes acknowledgements. Coordinated checkpoints stage portable CPU tensors and validate common ordered GPU/runtime identity plus each rank's complete CUDA RNG before fresh-group continuation. CPU APIs and CI remain available for explicit correctness fixtures.

Three subagents implemented and independently reviewed numerical execution, profile/supervisor integration and whole-job acceptance; the coordinator handled checkpoint/runtime identity, preview portability and integration. Real two-GPU tests cover complete D/G/prior/auxiliary/Adam/EMA parity against a controlled global reference (RP/RA, accumulation one/two, unique VICReg and lazy higher derivatives), exact stochastic/shuffled recovery with observations and earlier-snapshot replay, partial G backward failure, abrupt rank loss, coordinator death with blocked workers and corrupted CUDA RNG. Test-only fixture corrections and the corrected CPU preflight checker-shape regression are preserved in the report; no numerical tolerance was widened. Final installed-package results, exact source/artifact identity and PR/post-merge checks are recorded under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-nccl-core/`.

Both owner-authorized RTX A6000s were used; unrelated GPU jobs were left running. No paid compute, dataset download, upstream architecture copy or release occurred. Public `train`/`resume` still select native single-process execution. The five retained issues remain open, and actual clusters, image quality and deployment are not qualified by this local synthetic milestone.

## Metrics and composable observation research checkpoint

[PR #319](https://github.com/HyperGAN/HyperGAN/pull/319) publishes the [2026-09-19 metrics report](metrics-first-research-2026-09-19.md), recording three subagent research tracks and coordinator synthesis: the pinned mikkel/sliders-conceptmod dashboard, current runtime/config/recovery seams, and bounded server/chart options. Local/fetched/GitHub develop agreed at `477e63d07a425acfbdb8ea921e1e702e7e700f8a`; PR #318 and both post-merge workflows passed. Commands: `git status --short --branch`, `git log`, `git worktree list`, `git fetch origin develop`, `gh pr list`, branch/protection APIs and `gh run list`. Existing JSONL events and readers are implemented; the owner clarified that browser/server work remains planned.

Accepted owner directions: expensive metrics such as FID are explicitly opt-in; defaults must be removable; ordinary Python metric factories belong in recipe configuration; standalone serving exposes the same API to its UI and external consumers, anticipating WebSocket and future multi-server collection; sampling and measurement have distinct roles in a shared modality-neutral observation system; establish CouchDB-style event views with Python-native maps and shared reducers from the beginning, without a database; stream live contributions without re-reduction on the server. Future modalities, collectors and parallel evaluators need not ship in v1.

Research changes only reports. No runtime, GPU training, paid compute, dependency installation or release occurred. Validation: `git diff --check` passed, all 58 relative report links resolve, and the proposed TOML/JSON examples parse. Two subagents reviewed the final contracts; fixes cover reducer recovery, stream discovery, map/view identity, bootstrap state and numerical definition partitions. PR integration is recorded in the durable receipt at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-research/`. No design blocker remains. Next concrete metrics action: M0 proves one small Rust/core-WASM reducer can bootstrap historical state in Python and continue it in a browser worker, with bounded state, packaging and performance evidence. M1a/M1b then establish event documents, Python maps, derived contributions, reducer/view descriptors and removable metric defaults. Coordinate controller/schema edits with the existing public distributed integration cutpoint.

## Shared metrics reducer checkpoint

[PR #320](https://github.com/HyperGAN/HyperGAN/pull/320), M0 of the [metrics plan](metrics-first-research-2026-09-19.md) implements one bundled Rust/core-WASM module for bounded mean and first/min/max/last reductions, optional Python/Wasmtime hosting and a dedicated browser worker. The [proof report](metrics-shared-reducer-2026-09-19.md) records the ABI, delivery coverage rules, source/build provenance and measured costs. Bootstrap returns real state; live continuation uses the same module, with replay deduplicated before reduction. No database or live server reduction was introduced.

The coordinator verified current develop against Git/GitHub before starting from PR #319. Three subagents implement the reducer, event projections and configurable publication in external worktrees; the coordinator owns review, packaging and integration. Local M0 checks: reproducible pinned rebuild and 27 Python/actual Chromium cases passed. New required CI jobs cover optional reducer installation on Linux/macOS/Windows and the browser proof; installed-package/PR results and exact hashes are recorded in the durable receipt directory `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/`. No paid compute or release. Next metrics action: integrate M1a Python-map projections, then M1b/M2 publication and recovery before standalone serving. The production server remains unimplemented at this checkpoint.

## Next bounded checkpoint: public distributed integration


- [x] Complete fixed-global-batch two-process CPU D/G/prior/auxiliary/Adam/EMA updates and worker cleanup.
- [x] Implement coordinated fixed-topology snapshots and exact fresh-group continuation, including shuffled image data and rank failure.
- [x] Add CPU activation-memory-bounded accumulation preserving full-global-batch RA and VICReg, complete updates and exact fixed-strategy recovery. See the accumulation checkpoint above.
- [x] Extract the common lifecycle controller and single-process adapter from `training.py`; preserve current commands, attempts, checkpoints, previews and request semantics under installed-package and old/new behavior comparisons.
- [x] Add the CPU execution profile and structural/runtime preflight, with strict numerical identity and actionable rank errors. Keep recipe architecture separate from execution policy; validate structure without training dependencies and runtime behavior in bounded workers. See the [preflight checkpoint](core-preflight-2026-09-19.md).
- [x] Implement persistent supervised CPU worker commands, idle outside Gloo, with abrupt coordinator-death cleanup. Split checkpoint preparation from parent-controlled publication, reject stale receipts, and test fresh-group recovery and takeover at the internal protocol boundary. See the [worker-service checkpoint](core-worker-service-2026-09-19.md).
- [x] Connect an internal replicated execution adapter to the shared controller: fenced pre-attempt restore, separate numerical identity and timeout policy, fatal save/group errors, shared requests and final artifacts. See the [replicated service checkpoint](core-replicated-service-2026-09-19.md).
- [x] Reuse one event/request/counter service for the internal replicated lifecycle; require final inference artifacts when a batch is available and successful group shutdown before terminal success.
- [x] Add isolated snapshot preview execution without a training process group and bound parent progress delivery. Keep worker health monitoring independent of slow or hung observers; compare complete state with observation on/off. See the [bounded observation checkpoint](core-bounded-observation-2026-09-19.md).
- [x] Finish whole-job staging/publication/partial-update fault acceptance: staging disk failure, rename/pointer failure before selection, exception after a complete selected commit, second G microbatch backward failure and EMA failure after the G optimizer. Fresh groups recover complete state exactly; failed attempts do not acknowledge partial updates or successful saves. See the GPU core checkpoint below.
- [x] Add GPU-first project defaults and native CUDA training/recovery, with separate installed-package CPU and explicit local CUDA acceptance gates. Two-GPU NCCL diagnostics are infrastructure evidence; they do not close full replicated GPU training.
- [x] Port the supervised replicated strategy to rank-owned CUDA devices and NCCL-aware numerical/control paths. Qualify complete two-GPU updates, accumulated replay, CUDA RNG and coordinated recovery against controlled single-process results, including rank failure and independent cleanup. See the [local CUDA checkpoint](core-replicated-cuda-2026-09-19.md).
- [x] Expose execution profiles through public `train`/`resume`, with GPU-first workflows and explicit CPU fixtures. Preserve headless operation, persisted numerical identity, mutable deadlines, early conflict rejection and bounded progress/result streams. Installed CPU/native CUDA/two-GPU and disconnected-output acceptance are recorded in the [public execution checkpoint](core-public-execution-2026-09-19.md).
- [ ] **Next core cutpoint:** follow I1–I4 of the [image plan](image-training-plan-2026-09-19.md): partial-freezing correctness, pinned pretrained weights and source terms, the selected MoG/b-cap SAGAN-style recipe, real image grids and CIFAR reproduction/evaluation. The original200k experiment supersedes the earlier residual candidate; synthetic recovery remains distinct from image-quality qualification.
- [ ] After full local two-GPU training/recovery gates pass, prepare a concrete, separately agreed real two-node allocation using the reserved Modal credit or another provider. Communication-only diagnostics do not authorize this transition.

The optional standalone viewer and automatic local serving are implemented; image-grid artifacts/rendering are the next observation work. ONNX/container deployment and release promotion remain later gates. Continue from current `develop`; do not restart branch or issue audits. Keep master historical. Use the authorized local GPUs; paid compute still requires a concrete agreed allocation.

## Configurable metric publication checkpoint

The [publication report](metrics-publication-2026-09-19.md) records removable scalar defaults, immutable definition catalogs, schema-2 completed-update events and native/internal replicated integration. Numerical completion and finite checks remain mandatory with metrics disabled. Observations have a separate configuration identity, so changing selected metrics or cadence on resume preserves numerical state. Attempts record explicit checkpoint ancestry, including earlier and zero-update recovery. GP/prior contributions are labeled weighted; unavailable raw upstream values are not reconstructed by division.

Subagent source checks cover strict configuration, exact on/off/cadence checkpoint equality, native and supervised CPU/CUDA recovery, and actual accumulated two-GPU comparisons. The coordinator's independent review and installed-package/CI evidence are retained under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/`; integration receipts identify the exact PR/head and results. The report preserves environment corrections and complete commands. Both authorized local GPUs were used; no paid compute, dataset download or release. Next: merge bounded Python-map projections and standalone serving. Custom/snapshot factories are explicitly rejected until their execution slice lands.

## Standalone metric viewer checkpoint

The [viewer report](metrics-web-2026-09-19.md) records the optional `web` extra, authenticated loopback `serve`, transport-independent public API/OpenAPI, bounded SSE fanout and an offline browser using the same API. Historical bootstrap and live browser updates execute the exact same WASM reducer. Live server fanout performs zero reductions. Samples/artifacts have a separate modality-neutral shelf and remain visible with metrics disabled.

The source-matched million-update proof measured **2.32 ms warm query p95**, **202.68 MiB peak RSS**, **19.4 ms five-subscriber live fanout** and zero live reductions. Cold lineage indexing took 86.43 seconds and historical reduction 75.46 seconds; these are real costs, not a training-throughput or browser-render latency claim. Historical work defaults to a bounded 180-second budget. The receipt records exact measured source hashes and fixture size. The coordinator also verifies an installed wheel rebuilt from the source distribution, offline asset reproducibility and required platform CI. Next: custom evaluation presentation and CLI automatic startup, followed by integrated training/viewer overhead qualification. No paid compute or release.

## Custom metric and manual evaluation checkpoint

The [custom metric report](metrics-custom-evaluation-2026-09-19.md) records configured ordinary Python factories with explicit detached scalar inputs, cadence, bounded workers and visible failure policy. Manual snapshot evaluation pins immutable EMA bytes and an independent dataset/seed/protocol; RGB moments and histogram differences demonstrate task-specific scalar/distribution output. Evaluation events have independent registered streams, historical definition catalogs and visible scalar/histogram/failure results in the browser. Training samples remain separate artifacts.

Source acceptance covers exact CPU/native CUDA/internal two-GPU state equality, factory/source changes, failure recovery, timeout cleanup, snapshot provenance, repeated evaluation, complete declared iterator consumption and immutable registration recovery. Actual CPU producer-to-ASGI-to-Chromium checks show earlier-snapshot scalar results, histogram results and failed evaluations through the same public API. The coordinator retains installed-package/CI receipts in the metrics implementation evidence directory. Automatic snapshot schedules and built-in FID are explicitly unsupported in this first version; there are no implicit weight downloads. Next: integrate automatic CLI supervision and final installed end-to-end qualification. No paid compute or release.

## File-backed event-view checkpoint

The [event-view report](metrics-event-views-2026-09-19.md) records versioned Python maps, bounded append-only contribution frames, replay cursors and the `metrics`, `project` and `contributions` commands. Custom maps are supervised independently of HTTP and training state. Coordinator review fixed append continuity and source-boundary validation; a fresh sdist-built base-only installation passed **309 foundation tests in 11.01 seconds**. Durable acceptance evidence remains in the metrics implementation receipt directory. Standalone API/browser integration is the next slice; no database, paid compute or release was introduced.

## Automatic CLI viewing and final metrics integration

The [implementation report](metrics-implementation-2026-09-19.md) connects PRs #320–#325 and documents the usable commands, independent sampling/evaluation contracts and v1 limits. The [startup proof](metrics-autostart-proof-2026-09-19.md) qualifies optional automatic CLI serving, explicit preflight, no-server isolation, independent projector/server processes, parent-death cleanup and exact installed CPU/CUDA stop/resume state. The [performance report](metrics-observation-performance-2026-09-19.md) preserves sixteen complete-state CUDA comparisons and source-matched million-event server measurements. Candidate 1%/2% throughput budgets remain unestablished because measured intervals are wide; they are not advertised as guarantees.

Coordinator acceptance rebuilt source distributions and wheels: **423 integrated installed tests** and the end-to-end train/resume/evaluate/serve walkthrough passed, followed by **15 final browser tests** after the frontend-only visual fix. Offline JavaScript/WASM rebuilds matched. Protected PRs still merge only after exact-head required checks pass. Receipts under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/` record those checks and final merge identities. The next core action remains public execution-profile routing and bounded CLI output; future metrics work can add sparse historical indexing, a pinned explicit FID adapter and scheduled evaluation without replacing the event/map/shared-reducer contracts. No paid compute or release.


## Public execution and bounded CLI checkpoint

[PR #327](https://github.com/HyperGAN/HyperGAN/pull/327), [PR #328](https://github.com/HyperGAN/HyperGAN/pull/328) and the [public execution report](core-public-execution-2026-09-19.md) complete local public profile routing. New projects still target CUDA; named/TOML profiles select supervised local CUDA/NCCL or explicit CPU fixtures. Resume infers fixed numerical identity, preserves earlier-snapshot recovery and accepts separate deadline policy. Preparation rejects structural/configuration/checkpoint conflicts before viewer startup; full restore remains locked and strict. Output is explicitly best effort and bounded, with complete events/results recoverable from durable run files.

Three subagents implemented and cross-reviewed the slices in external worktrees. Coordinator acceptance at `70d9e92e` passed **641 installed tests**; corrected source at `3a519111` passed **402 installed foundation/native-CLI/viewer tests** and **38 focused installed CPU tests**, and the separate output slice passed **358 installed lightweight tests**. Actual corrected native CUDA and public two-GPU viewer/recovery state comparisons are exact; the live API observed running progress and projected metric frames. The report preserves initial environment corrections, the fixed partial-JSON/buffered-stdio issues and malformed-metadata regression, as well as the distinct baseline macOS CI exit stall and post-test Wasmtime finalizer diagnostics. A narrow cached-WASM resource cleanup addresses the latter and passed **44 installed reducer/viewer tests** with a deterministic shutdown regression; it does not claim to explain the macOS stall. PR #326 adds a bounded viewer CI deadline and diagnostics. No failures were skipped or converted to success.

PR #328 merged at `ddebd5a33ce5ff5395f4cbe8d79bdccebd6068ff` after all eighteen exact-head checks passed. Its deterministic shutdown regression reproduced a shared Event lock-owner death hazard; the fix uses a shared stop byte and independent parent-death monitoring. Final source passed **21 installed viewer tests** and repeated the actual two-GPU live-viewer/WASM-bootstrap comparison exactly across **144 tensors and 2,740 state values**, with clean process/credential cleanup. `viewer-shutdown-acceptance.json` preserves that final proof. PR #327 includes this merged base and remains subject to final exact-head required checks and the protected merge receipt; the report also preserves its earlier browser CI failure, ten matching-source repeats, a passing complete browser suite, and **227 passing CPU CI tests in 985.55 seconds**.

Durable source/package hashes, commands/results, review findings, required CI and PR merge identities are under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-public-execution/`; `final-public-acceptance.json` and `integration-current.json` identify the final accepted tree and integration. Local/fetched/GitHub develop and strict required checks were verified before work. Both authorized local GPUs were used; unrelated jobs remained running. No paid compute or release. Next concrete action: settle explicit upstream licensing and freeze the selected image recipe/protocol/provenance before actual image-workload qualification; agree a separate real two-node allocation afterward.

## Agreed next-version image plan

[PR #329](https://github.com/HyperGAN/HyperGAN/pull/329) records the owner
clarification replacing the proposed conventional DCGAN baseline with
**ParticleGAN MoG + b-cap and a pretrained discriminator from the first image
recipe**. The [focused image plan](image-training-plan-2026-09-19.md) updates the
version checklist without restarting completed runtime work. It sequences partial
freezing and source fidelity, image components, PNG previews and the usable local
workflow, pinned FID reproduction, real two-GPU qualification and externally
comparable CIFAR-10 evidence. Full-data CIFAR-10 is first; 10%-data CIFAR-10 is a
proposed follow-up. Multi-host, deployment and release remain later gates.

Verified clean local/fetched/GitHub develop at
`11b4ccc2663beef185709e3d5dcbd59b0bc593f8`. PR #327 is merged; its post-merge
[Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35474619748)
and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35474619657)
both passed. The earlier checkpoint text records its pre-merge state.

Read-only inspection found ParticleGAN branch
`feat/cifar-ae-gan-pretrained-encoder` at
`9e9ce96c96948197e21e1171c8394e3819bb0013`, matching its remote. The actual
saved run records FID50k12.5344567 at200k, batch64. Its archived trainer, image
components and pretrained critic match committed branch files. The later
12.2464 result is explicitly a different attention-depth intervention. An
independent subagent reviewed architecture, E-only reconstruction, distinct D/G
draws, optimizer/initialization differences and required acceptance gates.

Commands: `git status --short --branch`, `git fetch origin develop`, GitHub branch
and workflow queries, `git show`, `git ls-remote`, read-only JSON/ZIP/SHA256
inspection, `git diff --check` and local Markdown-link validation. Source/config
receipts and PR integration evidence are under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-image-plan/`.
This is a documentation milestone; no new runtime, experiment, GPU job, dataset
download, upstream edit, paid compute or release occurred.

Next concrete implementation: I1's preservation of factory-defined frozen
parameter masks with native/replicated gradient and full-recovery checks, then
the pinned image recipe. Source/weight terms and exact port/evaluation protocol
remain implementation gates; the existing score is not yet HyperGAN reproduction
or leaderboard placement.

## Image workflow execution in progress (2026-09-20)

Historical cutpoint, superseded by the completed allocation and authorization below.

Owner requested execution of the image plan with GPU 1, the public CLI, configured
metrics, a working smoke test and ongoing training. PR [#330](https://github.com/HyperGAN/HyperGAN/pull/330)
merged at `0cdefc96666d198f97adb38a1c2129247042963b` after all required checks
passed, preserving factory-defined frozen parameters. Independent subagents own
native image policies, the pinned CIFAR components/data, and PNG/browser flow.

The integrated installed wheel passes 35 focused CPU tests; the PNG slice passed
461 installed foundation/reference/viewer/browser checks. The port matches all
50,000 cached CIFAR images/labels and passed 77 source architecture/state/RNG/gradient
checks on CPU and GPU 1. Pinned Inception FID plumbing passed an identical-image
16-sample GPU fixture (raw FID -0.0001325, numerical tolerance 0.001); this is not
a GAN quality measurement. `hypergan validate` and native `preflight --runtime`
passed against the real batch-64/16,384-component recipe. The CLI preflight gap
was fixed: profile omission now checks the configured native CPU/CUDA device.

The complete source-update comparison did **not** pass its declared tolerances:
four generator weights differed after the first Adam update. A diagnostic
confirmed nonexact same-port CUDA replay, with preoptimizer gradient differences
on the order of 1e-9 amplified by Adam. Enabling strict deterministic algorithms
failed on `adaptive_avg_pool2d_backward_cuda`. Preserve these failed gates; do not
widen tolerances or use warn-only determinism. Next concrete action: explicit
backend policy and derivative-qualified deterministic feature operations, then
repeat complete-update and actual CLI recovery checks before the longer run.

Source licensing remains pending owner confirmation. The copied CIFAR components
and example are committed locally in the external integration worktree, not
published. Independent core/FID/PNG PRs #331–333 are in required CI. No GPU 0,
paid compute or release publication was used.

Worktree: `/home/martyn/dev/hypergan/image-integration/`. Dedicated environment:
`/home/martyn/dev/hypergan/image-training-env/`. Run configuration and live setup
log: `/home/martyn/dev/hypergan/training-runs/cifar10-pretrained-20260920/`.
Evidence, failed/passing receipts, package builds and merge receipts:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-image-workflow/`;
subagent source and preview evidence are in neighboring `2026-09-20-image-recipe`
and `2026-09-20-image-previews` directories. This is an in-progress cutpoint,
not closure of image quality, distributed image training or leaderboard gates.

## Image CLI acceptance, completed allocation and distribution authorization (2026-09-20)

The [execution report](image-training-execution-2026-09-20.md) supersedes the
in-progress next action above. Strict deterministic backend settings plus an
equivalent derivative-qualified feature-pool backward resolve current-run
recovery for an explicitly different execution variant; original failures remain
preserved. Eight controlled complete source/port updates pass 6,854 checks at
the original tolerances. Installed CLI validate/preflight/train/resume/sample/
evaluate all pass on GPU 1, including exact 802-tensor/1,984-value recovery across
the lazy b-cap boundary and replay from an older snapshot. FID128 is plumbing
evidence only. The 40k allocation completed on GPU 1 with scalar metrics,
1,000-update checkpoints, 500-update PNG previews and four serial FID50k/train
evaluations. Automatic resume succeeded between its 10k segments. The supervisor
finished at 08:21:19 UTC after 5,152.7077005680185 seconds including evaluation;
the saved 200k schedule remains stopped cleanly at observed/durable step 40,000.

PR #330 merged at `0cdefc96666d198f97adb38a1c2129247042963b`; PR #331 merged at
`e7595ead4645160594ed1e9015d0b8634e6c278a`, each after all exact-head checks
passed. PR #332 merged at `d838be6336d41ca6216b25d2ff43360de424eab1` after all
18 exact-head checks passed (244 installed CPU reference tests in 1,051.87 seconds).
PR #333 merged at `a5919d6572ad62f494f4a1b7d8b31bafac49b374` after all 18 checks
passed, including 250 CPU reference tests in 874.17 seconds. Protected merge
receipts are retained with the execution evidence. The core and PNG
scopes each passed 461 installed CPU tests, including actual browser checks in
the PNG scope. The owner has now authorized MIT distribution of the copied port,
with no additional attribution requested because the projects share authorship.
Source revision and experiment provenance remain recorded. Public-port review and
integration are tracked on `feat/cifar-recipe-public`; consult its GitHub PR and
protected receipts for final state. No GPU 0, paid compute or release was used.

Public-port packaging passed source → sdist → wheel and 88 focused installed CPU
tests in 14.64 seconds, without failures or skips. All 51 Python runtime files
byte-match the frozen accepted wheel, and the public recipe differs only in its
six local paths; the example is present in the sdist. The training environment
was untouched. Receipt:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-cifar-public/installed-verification.json`.

EMA FID50k/train improved from **33.46801440699488 at 10k** to
**25.71653952561263 at 20k**, **21.595076630349126 at 30k**, and
**19.37753221446735 at 40k**. The final evaluation completed in 171.04115945997182
seconds. The execution report names all four complete CLI receipts and their
common pinned protocol. Next concrete action: complete public recipe integration
and assess the saved grids, quality-versus-time and diversity evidence before
further reproduction work. This 40k allocation does not complete historical 200k
quality reproduction, two-GPU image training/recovery or external ranking gates.


## User feedback and five-step continuation handoff (2026-09-20)

The owner requested committing the [feedback report](feedback-2026-09-20.md)
and the [five-step plan](image-next-steps-2026-09-20.md) before the next work
session. Feedback covers CLI cadence (default 100, configurable through the UI),
the tensor-sample issue with image display confirmed working, termination,
server binding/authentication and lifetime, durable checkpoint/metric boundaries,
and step-ordered evaluation metrics. These are recorded requests and design
proposals, not implemented changes.

The plan sequences the 200k GPU-1 continuation, evidence publication, fresh-user
workflow, actual two-GPU recipe qualification and external benchmark submission.
It preserves the current formulation and existing qualification gates. Immediate
engineering work starts with the feedback; termination and metric/checkpoint
consistency should be reviewed together. Inspect the owner's live run before
using GPU 1. Keep its frozen environment unchanged.

Verified clean develop (apart from the new feedback draft), GitHub develop and
PR #334's merged identity at `261ff9f3d34aa74475c57b79fadb53554680c84c`.
Publication receipts remain under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-cifar-public/`.
This documentation slice uses external worktree `image-feedback-next-steps` and
branch `docs/image-feedback-next-steps`. Verification includes `git diff --check`,
relative Markdown-link checks and document review. Its protected PR/check state
is available on GitHub by branch. No runtime changes, GPU jobs, paid compute or
release publication are part of this handoff.


## Feedback implementation and integrated acceptance (2026-09-20)

Owner authorized subagents to fix every `reports/feedback*` item in PRs and merge
into develop. Three agents implemented bounded slices; the coordinator supplied
projection/durable-boundary visibility, cross-reviewed the code and resolved
shared API/CLI/documentation changes. The [implementation report](feedback-implementation-2026-09-20.md)
links PRs #336–#342 and defines the resulting user flow and durability limits.
The integration branch retains every reviewed branch head so the combined tree
can pass strict required checks without bypassing branch protection.

Planning PR #335 passed all required checks and merged at
`17a3cb307c8ff7b12a47600e855ca2d0c52409eb`. Its previously untracked feedback draft
was verified byte-identical and backed up before updating the main checkout.
The owner run was stopped at observed/durable step 41,000. No owner processes,
training files or frozen package environment were changed.

Installed acceptance exercised the actual CLI/UI, default and live console
cadence, persistent viewer reuse/explicit stop, read-only real tensor and FID
artifacts, event durability and failure injection. Native CUDA passed eight
checks; actual two-GPU public CLI/NCCL recovery passed seven, including exact
state, rank failure and coordinator death. Full installed package and protected
CI receipts record the final accepted source; initial failed assertions and their
fixes remain in the evidence. This is generic runtime qualification, not closure
of the actual two-GPU image-recipe gate.

Commands include source → sdist → wheel builds, isolated installed pytest,
`hypergan new/train/resume/server-status/stop-server`, actual Chromium interaction,
Git/GitHub state checks and exact-head protected merges. Evidence lives under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-feedback/`, with additional
publication/signal tests under sibling `2026-09-20-feedback-recovery/`.
No older-checkpoint migration, paid compute or release was added.

Next concrete action: verify the integration PR is merged, then have the owner
exercise these changes in a new run under the updated installation while retaining
the existing reference run's original environment. Continue the five-step plan;
do not silently reinstall or restart the preserved reference experiment.

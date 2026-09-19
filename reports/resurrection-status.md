# Resurrection execution ledger

Authoritative design: [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md). Updated 2026-09-19 (America/Denver).

Current cutpoint: CPU accumulation now preserves full-global-batch GAN objectives while replaying one microbatch at a time. Complete replicated updates, worker supervision and fixed-topology recovery are implemented. Next, extract the shared run lifecycle while preserving current single-process behavior, then connect distributed launch/resume and observation. GPU and real cluster qualification remain ahead.

## First checkpoint

The clean CPU foundation is merged in PRs [#301](https://github.com/HyperGAN/HyperGAN/pull/301) and [#300](https://github.com/HyperGAN/HyperGAN/pull/300). The integrated baseline is `86c7a3cf360a8975d48f53589b018d48719cad49`; its [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35419513982) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35419513976) both passed. Continue from the current develop and the next checkpoint below; do not restart the historical audit.

The next release integrates on `develop`. The coordinator reviews and merges passing PRs; bounded subagent work uses external worktrees. Master remains the historical stable line until a separately qualified release.

## Accepted decisions

- Recipe configuration supports custom generator/discriminator/encoder/auxiliary components, explicit I/O, losses and regularizers. Warn on unqualified combinations; reject actual incompatibilities. Default b-cap and VICReg follow the pinned upstream reference.
- Colorization and super-resolution influence the implemented conditional I/O contract. The paired example is a synthetic fixture, not a qualified image recipe.
- Development version: 2.0.0a1. ParticleGAN dependency: 0.5.0. Tested CPU tuple: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3. The package records actual runtime/source identity and does not certify arbitrary installations.
- Replace archived desktop viewers with the [optional local browser server](local-web-view-plan-2026-09-18.md). Planned local training defaults to attempting the server when its optional extra is installed; `--no-server` disables it. Cluster workers stay headless, standalone `serve` reconnects to run state, and the first UI is read-only. These flags/server are not implemented yet.
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

[PR #310](https://github.com/HyperGAN/HyperGAN/pull/310) delivers this checkpoint. The [accumulation report](core-accumulation-2026-09-19.md) and [usage contract](../docs/accumulation.md) document complete CPU updates with accumulation greater than one. Detached full-batch logits preserve RA derivatives, and global unique/full-table VICReg is evaluated once. Microbatch replay retains one network graph at a time; the full input/prior/logit state still exists. Custom sample-independent components remain configurable with an unqualified warning, and custom separable objectives declare mean/sum aggregation. Registered forward mutation, replay mismatch, nonfinite combined losses and invalid controls fail explicitly.

Three subagents handled implementation, independent numerical/memory/recovery acceptance and shared-service design review. Complete Adam/EMA updates match accumulation one across all 12 adversarial combinations at unchanged tolerances. A controlled network measured 395,380 bytes versus 98,784 bytes of peak saved activation storage at factors one and four; this is not total memory or a GPU benchmark. Fresh-group accumulated recovery is exact for the tested stochastic/sampler contract, changed accumulation is rejected, and a rank failure on the second G microbatch preserves the previous durable checkpoint.

The source-distribution-built wheel at implementation/test head `07f99694` passed **194 installed-package tests in 185.19 seconds** outside the checkout. A separate base-only installation passed **59 tests in 1.30 seconds**, with torch, ParticleGAN, NumPy and Pillow absent. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0. Required PR CI is the integration gate; the durable receipt records the final reviewed head, checks, merge and branch preservation.

The [shared run-service design](distributed-run-service-design-2026-09-19.md) is a proposal for the next milestone. Workers must wait outside Gloo between updates; one controller owns run state and checkpoint commit authority. Parent-death fencing and bounded preview execution are public-integration gates. No distributed CLI, browser server, GPU execution, paid compute, dataset download, copied upstream image architecture or release was added. The five tracked issues remain open.

## Next bounded checkpoint: shared run lifecycle


- [x] Complete fixed-global-batch two-process CPU D/G/prior/auxiliary/Adam/EMA updates and worker cleanup.
- [x] Implement coordinated fixed-topology snapshots and exact fresh-group continuation, including shuffled image data and rank failure.
- [x] Add CPU activation-memory-bounded accumulation preserving full-global-batch RA and VICReg, complete updates and exact fixed-strategy recovery. See the accumulation checkpoint above.
- [ ] **First next-session cutpoint:** extract the common lifecycle controller and single-process adapter from `training.py`; preserve the current commands, attempts, checkpoints, previews and request semantics under the installed-package suite.
- [ ] Add the CPU execution profile and structural/runtime preflight, with strict numerical identity and actionable rank errors.
- [ ] Connect supervised worker commands, distributed train/resume and checkpoint prepare/commit; prove parent-death cleanup and safe takeover before exposing the distributed CLI.
- [ ] Reuse one event/request/counter service and add bounded snapshot preview execution outside training collectives.
- [ ] Complete installed-package whole-job acceptance with accumulated shuffled data, fresh-group recovery and fault injection; preserve headless behavior.
- [ ] Resolve upstream explicit licensing and freeze the selected image experiment's architecture, preprocessing/augmentation/evaluation and pretrained-weight identity. A synthetic image recovery fixture is not image-quality qualification.
- [ ] After CPU gates pass, qualify actual two-GPU NCCL locally; then prepare a concrete, separately agreed real two-node allocation using the reserved Modal credit or another provider.

The optional standalone viewer W2/W3 can follow the implemented W1 read contract alongside core work. ONNX/container deployment and release promotion remain later gates. Continue from current `develop`; do not restart branch or issue audits. Keep master historical and paid compute out of the next CPU slice.

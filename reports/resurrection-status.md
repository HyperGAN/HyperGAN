# Resurrection execution ledger

Authoritative design: [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md). Updated 2026-09-18 (America/Denver).

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

## Next bounded checkpoint: CPU distributed correctness

1. Finish the remaining core observer contract alongside strategy preparation: reconnect cursors, periodic atomic previews/retention and a common serialized manual save-request API. Current W1 is partial: versioned manifests/events, unique attempts, durable progress and a bounded tail reader exist; the cursor and command protocol do not.
2. Add fixed-world-size two-process CPU execution and compare single-process versus distributed objectives and complete updates. Cover global batch/accumulation semantics, repeated prior IDs/global VICReg statistics, lazy penalties/double backward, synchronized EMA, named per-rank RNG/data state and whole-job failure/recovery. No GPU claim follows from Gloo fixtures.
3. Resolve upstream explicit licensing before copying the selected image graph. Freeze preprocessing/augmentation/held-out evaluation, pretrained weight identity and the actual direct-GAN update contract. Use CPU shape/gradient/recovery fixtures, then qualify image quality in the bounded GPU stage. Keep custom components runnable but unqualified.
4. After numerical and recovery gates pass, test actual two-GPU NCCL locally. Only then prepare a concrete, separately budgeted real two-node allocation/provider profile using the reserved Modal credit or another suitable provider.
5. Implement the optional standalone viewer after W1 settles; it can proceed alongside distributed work. ONNX/container deployment and release promotion remain later gates.

Continue from current `develop`; PR #305 is the merged recovery integration point. Do not restart branch or issue audits. Keep master historical until a separately qualified release; keep paid compute out of the next CPU core slice.

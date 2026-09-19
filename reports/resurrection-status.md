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

Audit evidence is preserved outside the repository at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-18-issue-audit/`, including the snapshot, reviewed decisions, action receipt and final inventory. Reports have exact 112/112 unique issue coverage and resolving local file links; PR #302 requires the existing full CI checks before merge. No GPU execution, paid compute or release occurred.

## Next bounded checkpoint: ready for GPU qualification

Image data portion implemented on `resurrection/image-data`: optional Pillow extra, deterministic `image_folder` batches, complete content/preprocessing/class-map identity and sampler state, and torch-free `data-check`. See [the data contract](../docs/image-data.md). A fresh wheel built through the source distribution passed **57 installed-package tests**, including 30 image fixtures and a subprocess preflight with training imports blocked (Python 3.12.13, torch 2.14.0+cpu, Pillow 12.3.0). This establishes data preparation and sampler continuation; complete trainer recovery is the next integration slice. No upstream image architecture was copied, no dataset downloaded and no GPU execution occurred.

1. Resolve the upstream explicit license declaration before copying image architectures or publishing a release. The dependency integration is provisional; no root-license provenance was invented.
2. Select and freeze one small image architecture and its data/preprocessing contract; verify CPU shape/update and preprocessing fixtures now. Reproduce and qualify image quality later in the bounded GPU stage. Keep custom components configurable; qualify exact profiles separately.
3. Implement full atomic training checkpoints: optimizers, counters, EMA/prior, named RNG streams, data/sampler position, schema and topology. Include monotonic sample/attempt identifiers, save requests at safe update boundaries, explicit config/data/class-map compatibility checks, bounded wall-time stopping and complete failure status. Expose the last durable step, checkpoint path, save interval and possible lost-work window. Test uninterrupted versus interrupted/resumed runs. Current model.pt is inference-only and cannot resume training. Stabilize the viewer read contract here (web plan W1).
4. Add a fixed-size distributed execution strategy and two-process CPU tests for G/D/prior updates, global batch/statistics, repeated prior IDs, accumulation, lazy penalties and recovery. Preserve the objective across ranks.
5. Add the standalone optional read-only server and CLI integration (web plan W2–W3) after the event/artifact contract stabilizes; it can proceed alongside CPU distributed work.
6. After the numerical/recovery gates pass, test actual two-GPU NCCL locally. Only then prepare an explicitly bounded real two-node allocation and provider profile.

Do not claim image quality, complete resume, multi-GPU, cluster, ONNX/container deployment or stable release support from this first checkpoint. No CUDA execution was used to validate it. Keep these release gates visible rather than expanding the default recipe catalog prematurely.

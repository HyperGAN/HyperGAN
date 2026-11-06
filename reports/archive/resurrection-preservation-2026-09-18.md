# HyperGAN preservation audit — 2026-09-18

Backup directory: `/home/martyn/dev/hypergan/resurrection-backups/2026-09-18-032714-preservation`. The preservation agent performed the local archive/restore audit. The coordinator subsequently pushed all 15 archive tags to origin. Branch retirement is recorded below as it completes.

## Verified backup

- Bundle: `HyperGAN-all.bundle` (40,055,518 bytes).
- SHA-256: `e34b456e2bf2217cb43136ad6063e27d8cf8aa2c145af9e5f6f34d6646621f3f`.
- `git bundle verify` passed; mirror restoration and `git fsck --full` passed.
- All 59 named refs restored identically; all 15 annotated archive tags resolve to their recorded commit. HEAD/worktree pseudo-ref commit objects also verified.
- `restore.git/` retains the mirror restore; `restore-checkout/` contains an actual checkout of the accepted plan at 7b7232fc.
- See verification.json and bundle/restore logs for exact evidence.

## Local files and concurrent work

Root and five historical worktrees were checked for tracked modifications, untracked and ignored paths. There were no local files requiring a tar backup. Root only lists the five nested Git worktree entries, deliberately excluded from recursive duplication. Empty staged/unstaged patch files document clean states. Existing worktrees were left untouched.

New external implementation worktrees were active concurrently; their registration is recorded in worktrees.txt and reachable commits are in the bundle, but their changing uncommitted files are outside this historical preservation snapshot. New develop/bootstrap/plan progress does not invalidate historical archive tags. Do not delete develop or master.

## Archive tags

| Tag | Full commit | Source |
| --- | --- | --- |
| `archive/resurrection-2026-09-18/branches/develop` | `36363df19b8713c321f83c5c242af5dad3426711` | `refs/heads/develop` |
| `archive/resurrection-2026-09-18/branches/electron-train` | `e1fdaa1abc7853b05a969b3aafdc7ba39bbedbd1` | `refs/heads/electron-train` |
| `archive/resurrection-2026-09-18/branches/fastgan` | `69434a19bd12c047d1d4489d0513c67b94bba955` | `refs/heads/fastgan` |
| `archive/resurrection-2026-09-18/branches/feature/omnigan` | `03a5813c4afb5319476f38c91d65052a1644ee1c` | `refs/heads/feature/omnigan` |
| `archive/resurrection-2026-09-18/branches/fix/examples` | `2c0011408f9cd52564235cb227659002b41b9a7e` | `refs/heads/fix/examples` |
| `archive/resurrection-2026-09-18/branches/master` | `291ddccda847e4f4ccb273bb26121a0a0d738164` | `refs/heads/master` |
| `archive/resurrection-2026-09-18/branches/nd` | `880e78f6b0f22ee7dd60ca075690a37795dacc5d` | `refs/heads/nd` |
| `archive/resurrection-2026-09-18/branches/stylegan` | `9d557a54e6e4abfec02ee2849bc204395b4a7ee7` | `refs/heads/stylegan` |
| `archive/resurrection-2026-09-18/prs/264` | `9d557a54e6e4abfec02ee2849bc204395b4a7ee7` | `refs/pull/264/head` |
| `archive/resurrection-2026-09-18/prs/280` | `e1fdaa1abc7853b05a969b3aafdc7ba39bbedbd1` | `refs/pull/280/head` |
| `archive/resurrection-2026-09-18/prs/292` | `b074e74abf0ed9b81bd52084706e3707a47e0fe2` | `refs/pull/292/head` |
| `archive/resurrection-2026-09-18/prs/295` | `c54a05d8d17275765ea30c1b1ba4c750a28e9292` | `refs/pull/295/head` |
| `archive/resurrection-2026-09-18/prs/296` | `265428bb46eb97611b7fb5bab1c5e5b58caf5096` | `refs/pull/296/head` |
| `archive/resurrection-2026-09-18/prs/298-before-execution` | `9b174b665574f30ff362861686738622df68f1c0` | `refs/pull/298/head` |
| `archive/resurrection-2026-09-18/prs/298` | `7b7232fcd587c390e2c0de51aeda7b2d73bd8458` | `refs/pull/298/head` |

## Legacy cleanup ledger

Remote deletion must recheck the current expected tip immediately before action. Preserve archive tags remotely before retiring remote heads; the external bundle already preserves them locally. Do not delete GitHub-owned pull refs or contributor fork branches. Keep master, develop, PR298 and new implementation branches.

| Branch | Expected tip | Disposition |
| --- | --- | --- |
| `fix/examples` | `2c0011408f9cd52564235cb227659002b41b9a7e` | Fully merged in master; archive, then delete branch after reference checks. |
| `electron-train` | `e1fdaa1abc7853b05a969b3aafdc7ba39bbedbd1` | Archive implementation; retain run control, worker separation, discovery and preview requirements. Close PR280 as superseded. |
| `stylegan` | `9d557a54e6e4abfec02ee2849bc204395b4a7ee7` | Archive unfinished configuration/pretrained integration. Close PR264 as superseded; mature pretrained recipes require separate acceptance. |
| `feature/omnigan` | `03a5813c4afb5319476f38c91d65052a1644ee1c` | Archive unfinished hardcoded-class experiment; conditional generation remains a future validated recipe. |
| `fastgan` | `69434a19bd12c047d1d4489d0513c67b94bba955` | Archive numerical experiments; retain documented input/device, fixed-sample and interactive-preview extraction candidates. |
| `nd` | `880e78f6b0f22ee7dd60ca075690a37795dacc5d` | Archive numerical experiments and unqualified vendored code; retain explicit checkpoint transfer, dataset and mature recipe candidates. |

Legacy PR closures: 264, 280, 292, 295, 296. Exact heads and archive tags are in cleanup-expected-tips.json. Remove corresponding clean linked worktrees through git worktree remove before removing checked-out local branches; never recursively delete worktrees/.

## Source extraction ledger

The following records concrete candidates and acceptance gates; no implementation is represented as already ported.

| Source / commits | Capability | Decision / acceptance gate |
| --- | --- | --- |
| fastgan / `756da0f3` | Interactive latent feature sliders and resample | Retain UX requirement; replace old Tk/Pygame integration. Capability-aware browser controls; repeatable inputs; exploration labelled separately from prior sampling |
| fastgan / `c0843996`, `929daa42` | Fixed-latent galleries with source/reconstruction/mask previews | Retain product requirement. Stable evaluation inputs and uniquely named checkpoint comparisons without training RNG mutation |
| fastgan / `63f0e2d4`, `a2bf5316`, `f628c0a1`, `8a2eb7ad` | Data/device handling and crop dimensions | Selective behavior port into new data pipeline. CPU/CUDA device tests, integer crop dimensions, explicit bounded corrupt-file handling |
| nd / `1be85cda`, `283e43ad`, `28d29e93` | Selective load, shape-matched transfer initialization, optimizer restore | Retain strict resume and explicit transfer distinction; rewrite format. Exact resume never silently accepts partial state; transfer emits loaded/skipped tensor report |
| nd / `7123cf97` | Image/CSV conditioning input | Retain dataset contract candidate; rewrite parsing/retries. Quoted CSV, missing labels, image conversion, sharded deterministic iteration |
| nd / `e7c9b20f`, `dc40f1c8` | Pretrained model adapters | Consider only mature, licensed model recipes. Named workload, source attribution/license, held-out quality and target parity |
| master / `291ddccd` | Multi-GPU backend lifecycle and explicit device selection | Retain user outcome; replace local parameter averaging/shared-memory algorithms with owned distributed loop. Worker ready/failure/stop, coherent checkpoint acknowledgement, G/D/prior synchronization, two-node acceptance |
| electron-train / `e1fdaa1a` | Process-separated preview/control service | Archive broken implementation; retain run service requirements. Reconnectable progress, cancellation, independent training process, valid image serialization |
| PR296 / `265428bb` | Lightweight new/help/template commands | Reimplement in package foundation; credit contribution. No torch dependency for help/new/list/inspect; installed-wheel tests |
| PR295 / `c54a05d8` | Missing optimizer diagnostics | Reject dummy-module and identity-layer shims; retain meaningful config errors. Unsupported legacy TensorFlow configurations explicitly diagnosed |
| PR292 / `b074e74a` | Loss visualization dependencies | Supersede global chart_studio/plotly requirements with optional local reporting. Core CLI independent of plotting/service dependencies |
| fastgan / nd / stylegan / feature/omnigan / `69434a19`, `880e78f6`, `9d557a54`, `03a5813c` | Numerical research and additional architectures | Archived by default; only proven mature methods with explicit recipe justification are candidates. Controlled upstream/reference comparison, quality/resource budget, distributed behavior and licensing provenance |

## Research preservation notes

fastgan is 182 commits ahead of historical master; nd is 281 ahead. Neither contains the other: 9 fastgan-only and 108 nd-only commits, with no patch-equivalence matches between those differences. Preserve both archived tips. nd contains NVIDIA restricted-rights headers in vendored StyleGAN/CUDA files; top-level MIT is not evidence those files can be repackaged. The historical master checkpoints already save optimizer state; replacement is needed for coherent counters/RNG/data-order/atomic distributed state, not because optimizer serialization is absent.

A historical develop tag points to 36363df1. Develop is now the retained integration branch and may advance; that historical tag is preservation, never authorization to reset or delete current develop.

## Execution status

All 15 archive tags are published on origin. PR #299 advanced develop with preserved master ancestry; PR #298 landed the approved plan and continuity files. The new reference passed installed-wheel acceptance and landed through PR #301. PR #300 removes the archived runtime and updates current documentation under the same required CI gates. Legacy PRs #264, #280, #292, #295 and #296 are closed with source-preservation and replacement dispositions. Remote branches fix/examples, electron-train, stylegan, feature/omnigan, fastgan and nd were deleted only after rechecking their exact tips and using expected-tip leases. The five historical linked worktrees were rechecked for tracked, untracked and ignored files, removed with git worktree remove, and their archived local branches retired. Cached PR tracking refs were pruned; GitHub-owned pull refs and contributor forks were untouched. Evidence is recorded externally in retirement-results.json and worktree-retirement.json.

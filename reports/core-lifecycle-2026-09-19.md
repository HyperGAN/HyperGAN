# Shared run lifecycle checkpoint — 2026-09-19

[PR #311](https://github.com/HyperGAN/HyperGAN/pull/311) separates run policy from numerical execution so distributed training can reuse the current recovery and observation workflow. Public `train` and `resume` continue to execute the single-process CPU reference. The [lifecycle guide](../docs/run-lifecycle.md) describes ownership and remaining distributed gates.

## Implementation and review

One controller owns locking, attempts, manifests, events, cooperative stop budgets, manual checkpoint requests and sample reservations. A single-process adapter owns the trainer, completed batch, numerical construction/restore/update, checkpoint serialization, preview rendering, inference artifacts and callback RNG isolation. Shutdown must complete before terminal success; failure cleanup precedes the terminal failure event while the run remains locked.

Three subagents split production extraction, independent lifecycle/source-identity acceptance, and diagnosis of the preceding post-merge CI failure. The coordinator reviews integration, compares old/new installed behavior, builds and tests the package, and manages the develop PR.

Production extraction `988f65fa` is integrated as `b3eb6737`, independent acceptance `3709456a` as `4a8c1703`, and the Windows lock fix `9adb9d56` as `df9a7931`. An AST comparison against `628a406c` confirms unchanged `ReferenceTrainer`, EMA updates, runtime/source reporting and recovery-contract logic. Only the lifecycle boundary and source-identity inventory change around those numerical operations.

Review caught and fixed warning delivery escaping callback RNG isolation and stale shutdown diagnostics leaking into a later attempt. The callback and its failure warning now share RNG/thread isolation. Tests verify cleanup before terminal events, lock ownership through cleanup, failure after restore without a new attempt, unchanged durable state after partial updates, preservation of a primary error when cleanup also fails, and exact numerical state when a warning handler consumes randomness and changes CPU threads.

The extraction changes strict implementation hashes. Both native and internal distributed checkpoints from the prior implementation require their original installation. Existing checkpoint validation is retained; exact pre/post-refactor run comparisons do not claim cross-version checkpoint migration.

## Windows lock race found during preflight

PR #310 merged at `628a406cb2efa1a377b1ed03b7528356f8e8904d` after its required PR checks passed. Its later [post-merge Foundation run](https://github.com/HyperGAN/HyperGAN/actions/runs/35426214214) failed in the Windows/Python 3.12 concurrent checkpoint-request test. The CPU reference job passed. This corrects the previous handoff, which observed that run while it was still in progress.

Both queue and run locks initialized an empty file before acquiring its OS lock. Two contenders could observe an empty file, then one could acquire a mandatory Windows byte-range lock before the other's buffered initialization write flushed. That second write raised `PermissionError` outside the intended busy-lock handling.

The fix removes the unnecessary initialization write from both helpers. Windows `_locking` explicitly permits ranges beyond EOF ([Microsoft documentation](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking?view=msvc-170)). Tests cover denied initialization writes, lock contention in another process, empty and existing lock files, and release on process exit. The denied-write regression fails on the previous code. Native Windows behavior is checked in the normal required CI matrix.

## Acceptance evidence

The source-distribution-built wheel at implementation/test head `4a8c170339c4061eae7533b2b17832430cc595dd` passed **214 installed-package tests in 188.23 seconds**, run outside the checkout with `python -I -m pytest /path/to/checkout/tests --import-mode=importlib -q`. A separate base-only installation passed **74 tests in 1.62 seconds**, with Torch, ParticleGAN, NumPy and Pillow absent. Runtime: Python 3.12.13, Torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0. Required PR checks remain the integration gate; their exact results and merge are preserved in the durable receipt. Durable commands, logs, baseline comparison state and preservation receipts are under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-lifecycle/`.

The coordinator's behavioral comparison independently runs default, paired and stochastic recipes through a two-update stop and resume, periodic previews and an acknowledged manual save. All three comparisons passed exactly for complete training state, event order/steps/sequences, attempt stop outcomes, durable counters, sample reservations, receipt outcomes and retained preview steps. The comparison uses independent old/new runs, never loads an old checkpoint into the new implementation, and does not compare nondeterministic IDs, wall times or filesystem paths.

## Next cutpoint

Add an explicit CPU execution profile with lightweight structural validation, resolved global/local/microbatch identity, and supervised runtime preflight. Then connect worker commands and distributed recovery to this controller, with parent-owned canonical checkpoint publication and actual parent-death/takeover tests before exposing distributed commands.

Single-process previews and Python callbacks remain synchronous. Bounded snapshot rendering, worker command channels, distributed CLI, actual GPU/NCCL and real clusters remain separate gates in the [integration design](distributed-run-service-design-2026-09-19.md). No GPU execution, paid compute, dataset download, upstream image architecture copy or release publishing is included. The optional server remains planned and all five tracked issues remain open.

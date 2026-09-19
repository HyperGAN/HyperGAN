# CPU worker commands and parent checkpoint publication

Date: 2026-09-19. [PR #313](https://github.com/HyperGAN/HyperGAN/pull/313). Baseline: develop `ea824231d015dc4ea814c3f08c2b9d16dd07fd22`, after PR #312.

This checkpoint implements two prerequisites for the shared distributed run service: [persistent supervised worker commands](../docs/cpu-worker-service.md) and [checkpoint preparation separated from canonical publication](../docs/distributed-recovery.md). Public `train` and `resume` remain single-process. The replicated execution adapter, shared observation integration and bounded renderer are the next cutpoint.

## Ownership and failure boundaries

The new internal CPU command service has a coordinator, an independent broker and a fixed group of spawned Gloo workers. Numerical state stays in the workers. The broker owns their process handles, monitors the coordinator's process sentinel, and terminates and reaps ranks on failure or coordinator death. It continues monitoring while a worker is inside a long operation or the coordinator is idle. The coordinator and broker modules do not import the numerical runtime.

Commands carry a run ID, attempt ID, increasing sequence and operation. Every rank agrees the command before invoking its handler. Workers wait for the next command outside Gloo, so an idle interval does not consume the collective timeout. Startup, operation and total deadlines have separate purposes; the total deadline includes idle time. Control messages use bounded JSON frames, separately from recipe stdout. Successful shutdown requires all ranks to agree and exit successfully.

Checkpoint preparation reuses complete rank serialization, identity checks, replicated-state agreement and all-rank staging readiness. Workers write only a managed hidden preparation directory and return a descriptor. That descriptor is not a committed checkpoint or evidence that the supervisor received every rank's successful result.

The parent retains `run_lock` and a process-bound commit authority. After successful all-rank completion and a fresh health check, the authority checks the expected run, attempt, controller token, command sequence and numerical identity, validates staged file sizes and hashes, then publishes the immutable generation and latest pointer. A new authority rejects an old controller's receipt. A validated sequence is consumed before publication, so uncertain write failures cannot be retried with the same receipt.

The authority does not acquire or verify the caller's OS lock. This is a cooperative internal protocol for trusted worker code, not filesystem isolation from arbitrary custom Python. The old standalone `save_distributed_checkpoint` API continues to publish from rank zero under its caller-owned rank-zero lock; future supervised adapters must use preparation and parent commit instead.

A complete commit can survive a later worker or job failure. A rename followed by a failed pointer write can leave a valid unselected generation; a failure syncing an already replaced pointer can leave latest changed. Prepared directories remain outside normal restore selection. These distinctions are retained rather than treating every reported write failure as an unchanged filesystem.

## Coordination and acceptance

Three subagents implement the worker service, checkpoint split and independent process-level acceptance. The coordinator reviews protocol boundaries, integrates changes, validates the installed distribution and owns the develop PR and preservation receipts.

The first integrated wheel passed 355 tests and failed one legacy accumulation fault assertion that counted the new hidden preparation root as a completed generation. The assertion now checks that only the prior visible generation exists and separately requires no abandoned command staging. A related metadata-limit fixture also checks the new staging namespace. No numerical tolerance or production behavior was changed for this correction. The failed build/test evidence is retained under `initial-local-attempt/`.

The corrected source-distribution-built wheel at source/test head `e18395c867245c1b7eb2a6714b564c425610448f` passed **356 installed-package tests in 279.85 seconds**, including 70 new cases. A separate base-only installation passed **176 tests in 2.08 seconds**, with torch, ParticleGAN, NumPy and Pillow absent. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0. Both runs used `python -I -m pytest /path/to/checkout/tests --import-mode=importlib` from outside the checkout; the base-only run selected `tests/foundation`.

Independent process fixtures compare exact accumulated full-rank state with shuffled data and Torch/Python/NumPy stochastic components across fresh groups. They kill the coordinator while ranks hold their GIL inside native calls, verify the broker reaps both ranks, reject the old receipt during takeover, republish the selected checkpoint before any new update, and continue exactly. A separate fault leaves a staged receipt but kills a rank before its command response; no aggregate succeeds, health refuses and the prior checkpoint remains recoverable. Startup death, missing/conflicting commands, idle intervals beyond the collective timeout, deadline failures and malformed replies have direct process tests. A Linux subreaper harness reaps the adopted broker without masking whether the broker reaped its own ranks.

The guide's exact example also passed from the installed wheel with no stderr output. Review fixed buffered failure diagnostics, malformed-response cleanup, potentially blocking special-file opens, typed JSON identity comparison and preservation of primary errors during cleanup. All five original subagent commits retain identical stable patch IDs after integration.

Wheel SHA-256: `87609a02200cdfa35133302e3a507a79bc15288cfc783f1f0c006e34e1666f57`. Source distribution: `0394d1e8883ddac7bdbc038e4f43f164832be059d67bdbabf47969d685622b51`. Subsequent changes are documentation/reporting only. Required CI, the reviewed head, final merge, tree equality and verified branch preservation are recorded in the durable integration receipt. Durable build, test, GitHub and branch-preservation evidence is under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-worker-service/`.

Distributed checkpoint source identity now includes the parent publication implementation, and replicated preflight reports it too. Strict recovery rejects checkpoints from before this distributed source change; use their original installation. Ten existing numerical, lifecycle, configuration and blocking-supervisor modules are byte-for-byte unchanged from the baseline, including native single-process numerical/checkpoint source. No checkpoint conversion or relaxed source comparison is introduced.

## Next satisfying cutpoint

- [ ] Connect one replicated execution adapter to the existing shared controller using the supervised commands and parent commit authority. Return the full prepared receipt from rank zero with compact peer acknowledgements so the aggregate response remains within its 64 KiB wire bound.
- [ ] Supply a fenced worker-session identity during strict restore before a new attempt is persisted; then bind the reserved attempt without accepting stale commands or weakening failed-resume behavior.
- [ ] Persist resolved numerical execution identity separately from mutable timeout, checkpoint, preview and stopping policy.
- [ ] Keep failed groups fatal. Treat an optional save rejection as an observer error only after coherent rejection and verified group health.
- [ ] Reuse events, checkpoint receipts and monotonic sample reservations; publish final artifacts and finish worker shutdown before terminal success.
- [ ] Add isolated snapshot rendering and bounded progress delivery so observers cannot hold up worker health monitoring.
- [ ] Pass installed-package lifecycle acceptance: accumulated shuffled data, fresh-group recovery, older-checkpoint zero-update replay, save requests and injected failures.
- [ ] Expose distributed train/resume only after those gates; qualify actual two-GPU NCCL and then a separately agreed two-node allocation.

No GPU execution, paid compute, dataset download, upstream architecture copying, browser server or release is part of this checkpoint. The five tracked issues stay open. Image licensing/qualification and deployment remain separate release gates.

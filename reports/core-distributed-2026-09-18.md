# Core distributed execution checkpoint — 2026-09-18

PRs [#308](https://github.com/HyperGAN/HyperGAN/pull/308) and [#309](https://github.com/HyperGAN/HyperGAN/pull/309) advance the CPU correctness gate on `develop`. It does not qualify a GPU, cluster provider or image recipe. The implementation remains an internal Python execution path; the existing `train`, `resume` and observer commands still use the single-process run service.

## Complete replicated updates

`ReplicatedCPUTrainer` owns complete D → G/prior/auxiliary → EMA updates on a fixed CPU Gloo group. It reduces parameter gradients explicitly after backward, preserving the exact input-gradient penalty calculation without introducing DDP hooks. The configured batch size is global, divided evenly across ranks; optimizer rates and the total update schedule keep their configured values.

Rp keeps matched real/fake pairs. RA evaluates the pinned ParticleGAN kernel on differentiably gathered global logits. Sampled-row VICReg uses the global unique union of particle IDs; full-table regularization remains full-table, and standardized MoG gradients can reach unsampled rows. Missing local gradients contribute zero to a globally used parameter; a parameter unused everywhere retains `grad=None`. Nonfinite losses or gradients require rank agreement before the relevant optimizer step. A failed half-update poisons the trainer and cannot become a complete checkpoint boundary.

The default data path draws the same deterministic global batch on every rank, verifies equality and takes contiguous rank slices. This preserves one global sampler order, including image-folder permutation/cursor state, at the cost of duplicate CPU decoding. Prior, penalty and global stochastic RNG remain rank-specific. Automatic data factories must use the supplied named generator or otherwise produce the same agreed global batch.

Registered model state, buffers, extra state, optimizer state, base rates and EMA must agree across replicas. Training-mode BatchNorm has rank-local statistics and is rejected in this strategy; frozen evaluation-mode components remain possible. Custom scalar objectives are averaged across ranks, so a custom global statistic needs its own collective semantics. These checks do not certify hidden Python state, arbitrary hooks or arbitrary custom batch-dependent computation.

The independent global oracle uses a bias-free critic and explicit matched draws, comparing complete parameter/Adam/EMA state at tight floating-point tolerances. Review found that a relativistic critic's mathematically invariant additive bias can receive different near-zero gradients from different reduction orders, which Adam can amplify. The fixtures do not establish bitwise equivalence between arbitrary single-process and distributed trajectories. Within-strategy recovery has its own state-equality gate.

## Worker lifecycle

`launch_cpu_workers` starts a fresh local spawn/Gloo group with finite collective and whole-job deadlines. A worker exception, abrupt exit, stall, partial spawn failure or parent interruption terminates and reaps the remaining direct children. Every worker must finish its callback and the final barrier before success. Restart is explicit; there are no automatic paid retries or external resource allocations. See [the worker guide](../docs/cpu-workers.md) and [numerical contracts](../docs/distributed.md).

## Coordinated recovery

A separate distributed checkpoint format stores complete per-rank snapshots and publishes one latest pointer after all-rank readiness. Strict identity includes the configured recipe/schedule, runtime and actual thread settings, source modules, data/class map and fixed topology. Both named and global RNG, sampler ownership, last local batches, optimizers and all registered numerical state survive a fresh worker-group restart. The [recovery guide](../docs/distributed-recovery.md) defines caller locking, compatibility and atomic-publication limits.

The recovery implementation is `3f55dcb0` (integrated as `54f5a958`), independent RNG/metadata acceptance is `7a21918b` (integrated as `0b9e3d69`), and coordinator integration is `831e2d52`. Review fixed RNG-consuming identity hooks, metadata bounds, latest-pointer consistency, Pillow module handling during validation copies, load-hook mutation of expected state, and poisoning after partial live restore. A staged generation cannot be published merely because all payloads arrived: a worker death before final readiness leaves the prior latest pointer intact.

The image recovery fixture uses seven generated grayscale images, labels, shuffled global data and an original tiny stochastic generator. It is shape/recovery evidence, with no copied upstream image architecture, external dataset or image-quality claim. A separate custom sampler checks rank ownership explicitly. The coordinator's integration test exercises the actual spawn supervisor, designated writer lock, updates, complete publication and fresh-group resume together.

## Validation and review

The numerical implementation is `70091f6b` (integrated as `9e7c4732`), the independent acceptance suite is `67c4d6fa` (integrated as `8e8153a6`), and the coordinator's worker lifecycle is `5310e641` plus `1d517f7d`. Three subagents split numerical implementation, recovery implementation and independent adversarial review; the coordinator reviewed and integrated the pieces.

A fresh wheel built through the source distribution at implementation head `9e7c4732` passed **162 installed-package tests in 67.36 seconds** outside the checkout. A separate base-only wheel installation passed **59 lightweight tests in 1.40 seconds**. Runtime: Python 3.12.13, PyTorch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3, Pillow 12.3.0. Commands:

```sh
python -m build --outdir /tmp/hypergan-cpu-updates-build
uv pip install --python /tmp/hypergan-distributed-core-verify/bin/python --no-deps /tmp/hypergan-cpu-updates-build/hypergan-2.0.0a1-py3-none-any.whl
# Run from /tmp, outside the checkout:
/tmp/hypergan-distributed-core-verify/bin/python -I -m pytest /path/to/checkout/tests --import-mode=importlib -q
/tmp/hypergan-distributed-core-lightweight/bin/python -I -m pytest /path/to/checkout/tests/foundation --import-mode=importlib -q
```

The three trainer tests include 12 adversarial combinations over complete updates; five independent acceptance tests cover the global oracle and failure boundaries. Fourteen worker tests cover controls, successful collectives, abrupt exit, exceptions, stalls, parent interruption and failure while starting the second process. Review added replica extra-state/optimizer checks, mixed automatic/explicit input rejection, explicit operation-order agreement and the Gaussian full-row regularizer guard. Remaining limitations are stated above; no failures were skipped.

The combined recovery wheel built through the source distribution at `a9716764` passed **185 installed-package tests in 140.10 seconds**, including all 21 checkpoint cases, the independent checkpoint acceptance test and the production supervisor/lock/restart integration. A separate base-only wheel installation passed **59 tests in 1.36 seconds**, with torch, ParticleGAN, NumPy and Pillow absent. Build output: `/tmp/hypergan-cpu-recovery-build`; verification environments remain the two separate environments above. The subsequent merge of current develop (`40f87d3f`) changes no source, tests, package configuration or CI files from this tested implementation.

Initial PR #308 CI exposed a degenerate nonlinear penalty fixture: identical real/fake inputs produced a zero adversarial D gradient, making Adam sensitive to reduction roundoff. Fix `167f6107` (integrated as `d3b79732` and `a9716764`) uses distinct real data, retains `rtol=3e-5`/`atol=3e-6`, verifies active/skipped lazy steps and improves assertion context. The original failure did not reproduce locally under default/AVX2 probes; the corrected fixture passed both probes and the complete [Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35424327926) plus [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35424327931). PR #308 merged at `79573c5e10647fcea674c03660dd245d4be63457`; its merged tree equals the reviewed head. PR #309 subsequently merged at `96222fc65b5b25d01df32e694a929e826dada95b` after all required checks passed; its merged tree equals the reviewed head. The [post-merge Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/runs/35424764223) and [Repository integrity](https://github.com/HyperGAN/HyperGAN/actions/runs/35424764247) passed as well.

Durable local build/test logs, the initial CI failure and integration receipts are under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-18-distributed-core/`. The final receipt records reviewed heads, CI results, tree equality and branch preservation/cleanup.

## Session cutpoint and remaining gates

- [x] Implement, validate and merge complete fixed-batch CPU updates and bounded worker lifecycle: PR #308.
- [x] Implement and validate coordinated fixed-topology checkpoint publication and fresh whole-group recovery with per-rank RNG/data state and failure coverage: PR #309, with required CI as its merge gate.
- [ ] At this checkpoint, accumulation was limited to one. See the later [accumulation checkpoint](core-accumulation-2026-09-19.md) and current status ledger for subsequent implementation and evidence.
- [ ] Connect distributed execution to the shared run/attempt/event/preview/request service and a usable launch/resume workflow with one designated writer.
- [ ] Resolve image extraction licensing and freeze the direct image experiment, data/evaluation protocol and pretrained-weight identity.
- [ ] Qualify actual local two-GPU NCCL only after CPU numerical/recovery gates; then prepare a separately agreed real two-node allocation.

No GPU training, paid compute, dataset download, upstream architecture copy or release publishing is part of this checkpoint. The existing five issues remain open. The optional browser server remains a separate follow-on to the already implemented observation contract.

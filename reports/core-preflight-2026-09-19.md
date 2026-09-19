# CPU execution profiles and preflight

Date: 2026-09-19. [PR #312](https://github.com/HyperGAN/HyperGAN/pull/312). Baseline: develop `e615afabd62d2cf3ba9e57a1b22e184add118b83`, after PR #311. This checkpoint adds a usable way to check execution settings and worker startup before the distributed run service is connected.

## Workflow and scope

A separate execution-profile TOML selects the single-process CPU reference or the internal replicated CPU Gloo strategy. The recipe continues to own architecture, objectives and global batch size. Profile resolution derives world/local/microbatch sizes and the accumulation algorithm without changing the global batch or learning rate. Startup and collective timeouts are separate operational policy.

`hypergan preflight CONFIG --profile FILE` checks structure using the base installation. Adding `--runtime` constructs the recipe in supervised workers, checks runtime/source/data and replica agreement, and reports recovery capability. It runs no training updates and creates no run or checkpoint. Worker diagnostics go to stderr; the result is JSON on stdout. The [profile guide](../docs/execution-profiles.md) and [example](../examples/execution/cpu-replicated.toml) document the commands.

Preflight checks known construction and state constraints. It cannot prove arbitrary custom forward/backward behavior, serialize or restore every possible custom state, or qualify a recipe's quality. Recovery declarations and initialization agreement are evidence about startup, not an executed restart. Existing strict checkpoint validation remains authoritative. Single-process `train` and `resume` retain their existing behavior and checkpoint source identity.

## Coordination and validation

Three subagents own the torch-free profile schema, supervised runtime implementation, and independent acceptance tests. The coordinator owns CLI integration, review, installed-package validation, the develop PR and preservation receipts. The sdist-built wheel at source/test head `8dc0a42e0f074e5ff5213f396fbec588968449ad` passed **286 installed-package tests in 225.67 seconds** outside the checkout. A separate base-only installation passed **126 tests in 1.83 seconds**, with torch, ParticleGAN, NumPy and Pillow absent. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0.

The installed module-entrypoint walkthrough exercised project creation and both structural/runtime profiles, producing clean JSON and no run artifacts. New regressions cover strict profile parsing, timeout-only identity invariance, actual fresh-group expected-identity comparison, typed mismatches, changed data identity, forged derived fields, CPU tensor/thread constraints, missing optional dependencies, group-sensitive single-process constructors, native/Python output routing, rank-specific failures and hung-worker reaping. Generator/data forwards are forbidden in the independent constructor fixtures.

Review caught and fixed single-process construction accidentally running under a world-size-one Gloo group; it now matches the native process-group environment. Recovery declarations remain separate from construction success. Ten existing numerical/lifecycle/configuration/data modules are byte-for-byte unchanged from the baseline. Existing checkpoint source checks are preserved; this preflight-only change does not provide checkpoint migration.

Validation commands used the installed wheel from `/tmp`:

```sh
/tmp/hypergan-distributed-core-verify/bin/python -I -m pytest /path/to/checkout/tests --import-mode=importlib -q
# 286 passed
/tmp/hypergan-distributed-core-lightweight/bin/python -I -m pytest /path/to/checkout/tests/foundation --import-mode=importlib -q
# 126 passed
```

The wheel SHA-256 is `74378e870db82e2647feca7439909223b39102c315079133f8240f95c07f57e7`; the source distribution is `93e4b1b5428526f2d83211b6e80cf70c66bd75afc8f459a20e137eb0431d89b7`. Later edits are documentation/reporting only. Durable build/test logs, command receipts, walkthrough output, source comparison and branch preservation are under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-preflight/`. The final integration receipt records required CI, the reviewed head, merge and branch cleanup.

## Next cutpoint

- [x] Extract one shared run controller and single-process execution adapter (PR #311).
- [x] Implement profile/preflight and pass installed-package acceptance; required CI and merge evidence are recorded with the PR and durable integration receipt.
- [ ] Connect supervised worker commands and a replicated execution adapter to the common controller.
- [ ] Separate distributed checkpoint preparation from parent-only canonical publication; prove abrupt parent-death cleanup and safe takeover before exposing distributed train/resume.
- [ ] Add bounded snapshot previews and progress delivery, then whole-job accumulated training/recovery acceptance with shuffled data and injected failures.
- [ ] Qualify actual two-GPU NCCL, then a separately agreed real two-node allocation.

No GPU execution, paid compute, dataset download, copied upstream architecture or release is part of this checkpoint. The five tracked issues remain open; image licensing/qualification, the browser server and deployment remain later work.

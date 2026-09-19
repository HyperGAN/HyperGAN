# CPU accumulation checkpoint — 2026-09-19

[PR #310](https://github.com/HyperGAN/HyperGAN/pull/310) adds microbatch activation replay to the internal fixed-world-size CPU/Gloo trainer. It advances the numerical preparation for multi-GPU training; public `train`/`resume` still use the single-process service. The next cutpoint is the first shared lifecycle extraction in the [distributed run-service design](distributed-run-service-design-2026-09-19.md).

## Numerical contract

The global batch and logical D → G/prior/auxiliary → EMA update remain unchanged. `accumulation_steps` is a positive integer dividing each rank's local batch. Each logical adversarial loss is evaluated once on detached full-batch logit leaves, with global RA gathers through the pinned ParticleGAN kernels. Replay backpropagates their cotangents through one microbatch at a time. It does not average independently computed RA objectives.

The original full latent/prior graph receives the accumulated latent gradient once; the global unique sampled-row or full-table VICReg population is evaluated once. Standardized MoG retains dense derivatives through unsampled rows. Explicit latent graphs can remain unused without creating zero gradients or advancing otherwise absent Adam state. Penalty scheduling, optimizer rates and EMA frequency remain per logical update.

[The guide](../docs/accumulation.md) states the custom component/objective contract. Built-in MSE/L1 mean and sum reductions are understood; custom separable objectives declare their aggregation. Arbitrary hidden state or batch coupling is not certified. Registered forward-state mutation and changed replay outputs fail explicitly. Replay preserves Torch/Python/NumPy/named RNG; state-descriptor hooks are observed without consuming their RNG. A failed microbatch poisons the trainer, and strict checkpoint identity includes accumulation factor, microbatch size and algorithm.

## Independent review and evidence

Three subagents split numerical implementation, independent numerical/memory/recovery acceptance, and run-service/failure design review. The coordinator reviewed source, added independent upstream stochastic-loss/prior-control tests and integrated the results. Numerical source `dcb5442c` is integrated as `c462c707`; independent acceptance `adcba05d` and `e23e6479` is integrated as `7d52ef45` and `07f99694`. Coordinator acceptance is `f0f10107` plus `6cf24bb2`. The run-service design is `581b736c`, integrated as `c1025c63`.

The acceptance compares accumulation one and two across all 12 Rp/RA/vanilla × logistic/hinge/Wasserstein/LSGAN combinations for three complete updates, including nonlinear active/skipped lazy b-cap, a conditional encoder, declared custom mean objectives and both MoG regularizer populations. Parameters, both Adam states, EMA, base learning rates and metrics use the existing `rtol=3e-5`, `atol=3e-6`; arbitrary cross-strategy bitwise equality is not claimed.

A deep synthetic network's peak live saved-autograd activation storage measured **395,380 bytes with accumulation one and 98,784 bytes with accumulation four**. Its forwards obeyed the microbatch bound, and no saved activations remained after the update. The tracker counts unique retained storage and excludes shared inputs/parameters/buffers and the pre-existing prior graph; it does not measure RSS, GPU allocation or total job memory. Full input batches, latent/prior state, logits/cotangents and replicated optimizer/checkpoint storage remain. Repeated forwards trade time for activation memory; this is not a GPU throughput benchmark.

Fresh worker groups reproduce complete accumulated checkpoint state exactly with a stateful shuffled sampler and a custom stochastic generator consuming Torch, Python and NumPy RNG. A changed accumulation factor is rejected before live mutation. A rank-one failure during the second generator microbatch backward stops both ranks after D has stepped: G/prior/EMA and the logical step remain unchanged, checkpointing is refused and the prior durable pointer/generation survives. Other regressions cover canonical replay RNG, custom sum objectives, impure forwards, nonpersistent buffer mutation, missing reduction contracts, rank-mismatched scalar types and nonfinite combined losses. Coordinator tests instrument the upstream vanilla loss's smoothing/flipping fields directly to check one logical loss invocation; those optional upstream fields are not added to HyperGAN's recipe schema.

Review fixed grad-mode-sensitive generation, D/G fake replay equality, RNG-consuming state descriptors, absent latent-gradient participation, and finite-total-loss checks before optimizer updates. Mean terms are scaled before summing to avoid overflow caused only by an intermediate unscaled sum. No tolerance widening, skipped failure or GPU run is part of this evidence.

Run from outside the checkout after installing the wheel built through the source distribution:

```sh
python -m build --outdir /tmp/hypergan-cpu-accumulation-build
# Install the wheel into the numerical and separate base-only environments.
python -I -m pytest /path/to/checkout/tests --import-mode=importlib -q
# Base-only environment:
python -I -m pytest /path/to/checkout/tests/foundation --import-mode=importlib -q
```

## Run-service handoff

The new design proposes one controller shared by single-process and replicated adapters. Workers wait for supervised commands outside Gloo between logical updates, so slow artifact/preview work does not consume collective timeouts. The controller owns run/attempt state, events, receipts and sample reservations. Public distributed integration must also prevent orphan workers from publishing canonical checkpoints after parent death; parent-owned commit authority and cleanup are explicit gates.

The five reviewable steps are lifecycle extraction preserving current behavior; CPU profile/preflight; supervised commands and checkpoint publication; shared events/requests/bounded previews; installed-package whole-job acceptance. The report is a proposal, not shipped run-service code.

## Session acceptance and remaining work

The source-distribution-built wheel at implementation/test head `07f99694` passed **194 installed-package tests in 185.19 seconds** outside the checkout. A separate base-only installation passed **59 tests in 1.30 seconds**, with torch, ParticleGAN, NumPy and Pillow absent. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0. Required PR CI is the integration gate; the durable receipt records the final reviewed head, checks, merge and branch preservation.

Installed-package build/test evidence and the reviewed PR are recorded in the [status ledger](resurrection-status.md). Durable session logs and Git preservation receipts are under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-accumulation/`.

- [x] Implement and validate CPU accumulation with installed-package numerical, memory, recovery and failure evidence; require passing PR CI before coordinator integration.
- [ ] Extract the shared lifecycle with unchanged single-process recovery/observation behavior, then follow the run-service integration sequence.
- [ ] Resolve upstream image extraction licensing and freeze the selected image experiment and evaluation/weight identity.
- [ ] Qualify actual two-GPU NCCL, then prepare a separately agreed real two-node allocation.

No GPU execution, paid compute, dataset download, upstream architecture copying or release publishing occurred. The five tracked issues remain open; optional browser-server implementation and deployment remain later work.

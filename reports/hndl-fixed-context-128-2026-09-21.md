# HNDL 0.4 fixed-context ResNet discriminator

PR: https://github.com/HyperGAN/HyperGAN/pull/370

The complete discriminator is defined in
[`examples/networks/resnet18-multiscale-discriminator-128.hndl`](../examples/networks/resnet18-multiscale-discriminator-128.hndl).
HNDL 0.4.0 comes from the published PyPI wheel. No custom network operators,
Python topology, or runtime patches were added.

The graph adapts the successful CIFAR design to 128px: five residual pixel
blocks, SAGAN at 16px, and three trainable heads on frozen ResNet18 layers 1–3.
It concatenates raw candidate and zero-context images at batch axis 0, normalizes
both, runs one backbone prefix at `2*B`, then chunks every output back to `B`
and concatenates candidate/context maps at the channel axis. Context is midgray
RGB after mapping from `[-1,1]`, and frozen BN preserves sample independence.
The context is recomputed rather than cached. There are 1,445,284 trainable
parameters and 11,689,512 frozen parameters, in 140 native HNDL nodes.

## Local launch

```sh
~/dev/hypergan/training-runs/start-dcgan-resnet-multiscale-128.sh
```

- Configuration: `~/dev/hypergan/training-runs/logos-dcgan-resnet-multiscale-128/dcgan-resnet-multiscale.toml`
- Editable discriminator: same directory, `discriminator.hndl`.
- Environment: `~/dev/hypergan/training-runs/dcgan-multiscale-128-env`, HNDL 0.4.0.
- New run directory: `~/dev/hypergan/training-runs/train-dcgan-resnet-multiscale-128`.
- GPU: physical card 0, UUID `GPU-ed080e41-3193-3755-6756-f3d46c433331`.
- 128×128 RGB, batch64, 200,000 target updates; checkpoint every1,000,
  preview every100, progress every20; local viewer enabled.

TOML comparison with the original logos DCGAN launch recipe found every setting
identical except discriminator and run name: generator, dataset and manifest,
prior, optimizer, losses, backend settings, sampling, batch size and seeds.
The old launchers and running jobs were not modified. The new long run has not
been started; validation used a separate temporary directory.

## Validation

- Full installed-wheel CPU suite: **1014 passed**, 187 heavy tests deselected.
- Provider tests: **10 passed**, including the actual new HNDL file at batches1
  and3. Assertions cover exact ImageNet normalization and feature split order,
  one shared prefix, first/second derivatives, all four score-head updates,
  D/G trainability restoration, and unchanged backbone/BN state.
- Mapped-weight CPU comparison with the existing CIFAR-style discriminator
  adapted to128px: maximum score error `4.47e-8`, input-gradient error `8.15e-10`,
  second-derivative error `6.65e-12`.
- Actual CUDA training with the pinned pretrained checkpoint and logos dataset:
  batch64, eight updates, then checkpoint resume for eight more. All losses
  finite; lazy b-cap execution at steps8 and16 completed. All four score heads
  changed; all122 backbone state tensors remained bitwise identical. Frozen
  parameters were excluded from D's optimizer and remained frozen after G.
- Peak CUDA allocated memory17.69GiB; reserved20.05GiB on RTX A6000.
- Resumed step16 published an EMA preview and all six diversity measurements.
  Full-resolution generated/reference RMS ratio0.06068; pooled4 ratio0.002580.
  These are an early EMA observation, not a trained quality result or evidence
  that collapse has been solved.

The first preview attempt failed because the temporary instrumentation script
lacked a multiprocessing main guard, causing its child to retry the locked CLI
run. Adding the guard to that temporary script fixed observation on resume;
no product code or architecture change was needed. Both training attempts
completed and preserved their checkpoints.

Local evidence: `/tmp/hypergan-hndl040-fast.log`,
`/tmp/hypergan-hndl040-smoke.log`, `/tmp/hypergan-hndl040-resume.log`, and
`/tmp/hypergan-hndl040-batch64-smoke/`. The temporary training wrapper only
observed state and gradients; it did not implement any network operations.

This is the CIFAR discriminator design adapted to a different resolution and
DCGAN training recipe. No FID or absence-of-collapse claim is made. Track both
`diversity/ratio` and `diversity/pooled4_ratio` during the actual experiment;
variance alone cannot establish semantic coverage or sample quality.

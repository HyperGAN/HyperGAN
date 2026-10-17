New projects use the current public ParticleGAN preset through
`defaults = "particlegan"`. ParticleGAN 0.9 selects optimized BISPF: non-saturating
loss, Born–Infeld critic penalty, SBNP prior regularization, DualNorm steps and
quintic convolution updates. Architecture and input/output bindings remain HNDL
definitions. The resolved configuration records the full recipe; later package
releases cannot change a saved run's trainer silently.

Use `examples/cifar-tiny-transformer-bispf.toml` for the two-block transformer,
frozen ResNet18 feature critic and 65,536-particle prior used by the integration
benchmark. The learning rates come from the released preset. This example uses
two visible GPUs: training on `cuda:0`, interval FID evaluation on `cuda:1`.
Change the evaluation device explicitly when using one GPU. FID and previews
are separate from the full-update timing benchmark.

Existing concrete configurations without a `recipe` snapshot keep the legacy
K3P mapping. `write_default(..., defaults="hypergan")` emits that explicit
reference fixture. Historical checkpoints still require their recorded runtime
and source compatibility; this release does not claim cross-version resume.

The `recipe` table holds ParticleGAN settings not expressed by HyperGAN's
`optimizer`, `gradient_penalty`, `prior_regularizer` and `training` tables.
Those existing tables own their corresponding scalar overrides. Unsupported
controller mechanisms fail before training. The new BISPF path currently
supports one critic network, including any number of heads inside its HNDL
graph. Continuous ParticleGAN policies and strict-progress finite probes need
their own host execution bindings.

Optimizers receive both parameter groups and module ownership, so convolution
updates work in HNDL generators, critics and trainable auxiliary networks.
The prior keeps a separate sampled-row group and rate. The loop binds the
protected adversarial objective, updates only sampled prior rows, and checkpoints
the complete optimizer law, controller state, initial prior moments and RNGs.
The SBNP law can be inactive when a sampled batch has too few rows for its latent
dimension; the CIFAR batch of 64 at latent dimension 128 is such a case.

`training.initialization = "deterministic_orthogonal"` uses ParticleGAN's public
initializer on fresh parameters. Loaded HNDL pretrained subtrees keep their
weights whether `trainable` is true or false. Effective `requires_grad` controls
optimizer membership. Frozen networks remain differentiable with respect to
their inputs; changing a component's trainability after optimizer construction
requires rebuilding its groups and an explicit state migration.

Run the bounded performance harness with one visible GPU and the same environment
for both rules. Its output directory must be new; stdout and all bulk artifacts
belong outside Git:

```sh
python scripts/benchmark_particlegan_cifar.py \
  --config /path/to/the/original-cifar-model.toml \
  --rule quintic --output /artifact-volume/cifar-quintic --profile --resume
```

It fixes seed 0, public initialization, original architecture/data/prior/batches
and the released global recipe, then measures 20 complete updates (5 warmup),
profiles 3 updates and verifies an exact 3-update checkpoint continuation.
The numerical contract disables autograd multithreading across forward and
backward. Frozen parameters and backbone buffers must remain unchanged.

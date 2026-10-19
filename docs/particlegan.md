# ParticleGAN integration

HyperGAN consumes ParticleGAN's released GAN recipe through
`defaults = "particlegan"` (ParticleGAN 0.9.2 or later). ParticleGAN owns the formulation and its selected
loss, learning rates, hyperparameters, normalization, regularizers and optimizer
implementations. HyperGAN owns networks and bindings, datasets, task objectives,
execution, checkpoints, sampling and metrics. Formulation research and technique
selection belong in ParticleGAN.

Generated projects inherit the installed public preset. Explicit configuration
values override the corresponding defaults. The resolved `recipe` table records
the dependency's complete settings, including settings HyperGAN does not
interpret. Preserve it as dependency-owned state when exporting a configuration;
do not hand-edit it to select internal techniques. Saved runs restore that
snapshot rather than selecting the latest preset again. Exact checkpoint
continuation still requires compatible runtime and source fingerprints.

`hypergan.particlegan_integration` is the adapter for preset resolution, scalar
configuration overrides and host feature checks. Configuration inspection reads
ParticleGAN's public recipe fields without importing Torch. Training delegates
loss, penalty, prior policy, regularization, schedules and optimizer construction
to the recipe's public factories. The Model tab identifies ParticleGAN as the
provider and shows host settings, loss metrics and networks.

Use `examples/cifar-tiny-transformer.toml` for the two-block transformer,
frozen ResNet18 feature critic and 65,536-particle prior. It uses two visible
GPUs: training on `cuda:0`, interval FID evaluation on `cuda:1`. Set the evaluation
device explicitly when using one GPU. FID and previews run separately from the
update timing benchmark.

The current integration supports one critic network per resolved recipe;
combine feature heads inside that network. Replicated execution and additional
host controller callbacks remain restricted until their execution bindings are
implemented. These are HyperGAN host limits. The integration does not reproduce
or validate ParticleGAN's internal numerical techniques.

Existing concrete configurations can still be loaded with their explicit
settings. Without a recipe snapshot, construction uses ParticleGAN's public
`Recipe` restoration defaults; HyperGAN no longer selects a named historical
formulation. Historical checkpoints require their recorded dependency and source
compatibility. `write_default(..., defaults="hypergan")` emits the concrete
reference fixture; fresh projects use the released ParticleGAN preset.

Fresh parameter initialization uses ParticleGAN's public initializer when
`training.initialization = "deterministic_orthogonal"`. Loaded pretrained
subtrees retain their weights. Effective `requires_grad` determines optimizer
membership, and frozen networks remain differentiable with respect to inputs.

The bounded performance harness measures the released preset without selecting
optimizer internals. Output directories must be new and bulk artifacts stay
outside Git:

```sh
python scripts/benchmark_particlegan_cifar.py \
  --config /path/to/cifar-model.toml \
  --output /artifact-volume/cifar-particlegan --profile --resume
```

It fixes seed 0, the public initializer, architecture, data, prior and batches;
measures 20 complete updates with 5 warmups; profiles 3 updates; and verifies
an exact 3-update checkpoint continuation. Frozen parameters and backbone buffers
must remain unchanged. This is integration and performance evidence; image
quality is measured separately.

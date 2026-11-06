# ParticleGAN integration

HyperGAN consumes ParticleGAN's released GAN recipe through
`defaults = "particlegan"`. The `train` and `all` extras require ParticleGAN
`>=0.9.5,<0.10`; 0.9.5 is the current released compatibility baseline.
ParticleGAN owns the formulation and its selected
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
compatibility. The packaged `gaussian-grid` model is the concrete reference
fixture; fresh image projects use the released ParticleGAN preset.

The packaged `cifar10-transgan` model also inherits that complete preset. It
previously omitted `defaults = "particlegan"`, so the Round 1 CIFAR runs
(`amber-heron`, `marble-anchor`, `coral-crest`) used KA2 despite installing
ParticleGAN 0.9.3/0.9.4. Their saved critic optimizer records explicitly contain
`formulation = "ka2"`; their results do not evaluate BISPF. See the
[QA report](../reports/qa-2026-10-30.md). Fresh runs now inherit the dependency's
loss, optimizer, rates, penalty and prior regularization instead of the old
model-specific overrides. Architecture, reconstruction objective and prior
dimensions are still declared by the model. Use a new run directory to test
the corrected preset; installing a newer library does not migrate a saved run.
The learned CLS token uses a one-row position table on a zero token, preserving
its forward value and gradients while giving DualNorm a supported matrix
parameter instead of the `cls_token` operator's rank-three broadcast tensor.

ParticleGAN 0.9.5 adds `Recipe.from_dict(saved_fields)` and rejects partial
`Recipe(...)` construction. HyperGAN uses the restoration API when available
and retains the legacy restoration fallback for 0.9.4. Install the current
release from PyPI:

```sh
python -m pip install --upgrade 'particlegan>=0.9.5,<0.10'
```

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

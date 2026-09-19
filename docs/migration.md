# Migrating from historical HyperGAN

The next release is being rebuilt on `develop`. It replaces the legacy TensorFlow/PyTorch experiment catalog with a small configurable runtime using ParticleGAN primitives and a HyperGAN-owned training loop.

Historical source, examples, documentation and tests remain available at [the preserved master commit](https://github.com/HyperGAN/HyperGAN/tree/291ddccda847e4f4ccb273bb26121a0a0d738164) and the `archive/resurrection-2026-09-18/branches/master` tag. Research branches and historical PR heads have their own archive tags. See [preservation and disposition evidence](../reports/resurrection-preservation-2026-09-18.md).

Legacy JSON configurations, checkpoints, dynamic layer strings, viewers and backends are not compatible with the new runtime. There is no automatic conversion or exact-resume path from legacy weights. Keep an archived checkout and its original environment when accessing old experiments. Future transfer initialization must be explicit and separately tested.

The new recipe configuration describes ordinary Python components, explicit input/output bindings, losses and regularizers. Custom visual generation is the first image-product direction; conditional I/O preserves a path for colorization and super-resolution. Old examples are archived, not evidence that these image tasks are already qualified.

The initial CPU reference is a numerical integration fixture. It does not establish image quality, complete training resume, GPU/cluster support or deployment support. These are subsequent acceptance gates in [the resurrection plan](../reports/resurrecting-hypergan-plan-2026-09-18.md).

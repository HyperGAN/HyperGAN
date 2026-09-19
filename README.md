# HyperGAN

HyperGAN is being rebuilt to make proven GANs configurable and practical: prepare data, train, inspect samples, recover runs, and build generators for applications. The next release integrates on `develop`, using [ParticleGAN](https://github.com/255BITS/ParticleGAN) primitives inside a HyperGAN-owned training loop.

The current foundation is a **CPU numerical reference**, with flexible component configuration and reloadable inference artifacts. It is not yet a qualified image-training product. Image recipes, complete training resume, multi-GPU and real cluster training remain release gates. Custom visual generators are the first product direction; conditional input/output contracts also support development of colorization and super-resolution recipes.

## Install the development foundation

Use Python 3.12 for the tested CPU training profile. Lightweight package checks cover Python 3.10–3.12 on Linux, macOS and Windows; the CPU reference profile is tested on Linux.

```sh
git clone --branch develop https://github.com/HyperGAN/HyperGAN.git
cd HyperGAN
python -m venv .venv
. .venv/bin/activate
python -m pip install .
hypergan --help
python -m hypergan --version
hypergan recipes
hypergan new demo
hypergan validate demo
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Lightweight commands do not need PyTorch or ParticleGAN. The development package version is `2.0.0a1`; these instructions install this checkout, not a promised published release.

For the Linux CPU reference, install the tested runtime and training extra:

```sh
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[train]'
hypergan train demo --steps 5 --run-dir runs/demo
hypergan inspect runs/demo
hypergan sample runs/demo --count 16 --seed 42 --output samples.json
```

A five-step run checks the integration; it is not a convergence benchmark. The run records its resolved configuration, runtime, counters and events, and saves a generator/prior inference artifact. Inference artifacts are not resumable training checkpoints. Use a new run directory for a new experiment.

## Configure the recipe

`hypergan new` writes `config.toml`. Configuration selects generator, discriminator, optional encoder/auxiliary components, constructor arguments, explicit input bindings, adversarial losses, gradient penalties, prior regularization and additional task objectives. Built-in identifiers and importable `module:object` constructors support ordinary Python implementations without a layer language.

The reference defaults to ParticleGAN's relativistic-paired objective, b-cap discriminator regularization and VICReg prior regularization. Custom configurations remain runnable with an explicit qualification warning. An unknown combination is different from an invalid binding or incompatible tensor shape: actual incompatibilities fail with an error. No custom configuration inherits quality, distributed or deployment approval merely by completing a run.

See [configuration and component contracts](docs/configuration.md). Custom factories execute Python code from your environment; use implementations you trust. Lightweight validation checks configuration structure without importing those factories; training validates runtime bindings and tensors.

## Development and migration

Install `.[dev,train]` in the tested runtime environment, then run `python -m pytest`. [Foundation CI](.github/workflows/ci.yml) additionally builds wheel/sdist artifacts, checks clean installations outside the checkout and tests lightweight commands without the training stack.

Historical HyperGAN code and experiments are preserved in archive tags. Legacy configurations and checkpoints require their archived runtime; see [migration notes](docs/migration.md) and [preservation evidence](reports/resurrection-preservation-2026-09-18.md).

Work is coordinated through [the execution ledger](reports/resurrection-status.md), [resurrection plan](reports/resurrecting-hypergan-plan-2026-09-18.md), and PRs targeting `develop`. No paid compute is needed for the foundation checkpoint.

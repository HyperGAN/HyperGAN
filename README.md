# HyperGAN

HyperGAN is being rebuilt to make proven GANs configurable and practical: prepare data, train, inspect samples, recover runs, and build generators for applications. The next release integrates on `develop`, using [ParticleGAN](https://github.com/255BITS/ParticleGAN) primitives inside a HyperGAN-owned training loop.

The current foundation is a **numerical reference with native CUDA training**, flexible component configuration, complete training checkpoints, reloadable inference artifacts and an optional image-folder data contract. New projects target CUDA by default; CPU execution is an explicit option for small correctness fixtures. It is not yet a qualified image-training product. Image recipes, multi-GPU and real cluster training remain release gates. Custom visual generators are the first product direction; conditional input/output contracts also support development of colorization and super-resolution recipes.

## Install the development foundation

Use Python 3.12 for the tested training runtime. Lightweight package checks cover Python 3.10–3.12 on Linux, macOS and Windows; the CPU reference profile is tested on Linux.

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

Run these commands in a terminal (PowerShell on Windows), not at the Python `>>>` prompt. Use the activated environment so `python`, `pip` and `hypergan` refer to the same installation.

On Windows, activate with `.venv\Scripts\Activate.ps1`. Lightweight commands do not need PyTorch or ParticleGAN. The development package version is `2.0.0a1`; these instructions install this checkout, not a promised published release.

For local NVIDIA GPU training, install the CUDA runtime and training extra:

```sh
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip install '.[train]'
hypergan train demo --steps 5 --run-dir runs/demo
hypergan inspect runs/demo
hypergan sample runs/demo --count 16 --seed 42 --output samples.json
```

`hypergan new demo` writes `training.device="cuda"`. Select another visible GPU with `hypergan new demo-gpu1 --device cuda:1`. Training requires that device and fails with installation/device guidance if it is unavailable; it does not silently fall back to CPU. The native CLI path uses one GPU per run. Developers can use the [internal replicated CUDA service](docs/replicated-cuda.md) for fixed local GPU training and recovery; public distributed commands and real clusters remain upcoming gates.

For a CPU correctness fixture, install the CPU PyTorch wheel instead (`--index-url https://download.pytorch.org/whl/cpu`) and create the project with `hypergan new cpu-demo --device cpu`.

A five-step run checks the integration; it is not a convergence benchmark. The run records its resolved configuration, runtime, counters and events, and saves separate complete training checkpoints and generator/prior inference artifacts. Use a new run directory for a new experiment.

To stop and continue the same numerical schedule:

```sh
hypergan train demo --run-dir runs/recoverable --checkpoint-every 1 --stop-after-steps 2
hypergan resume runs/recoverable
```

Resume creates a new attempt and preserves existing samples. `--max-seconds` provides a cooperative wall-time limit; `--progress-json` emits live JSONL for process managers. See [recovery and compatibility](docs/recovery.md). Image-folder preflight uses the optional `image` extra and `hypergan data-check CONFIG`; see [image data and preprocessing](docs/image-data.md). The default generator remains a 2D numerical fixture and cannot consume image batches.

Use `--preview-every N` for bounded periodic EMA previews. `hypergan events RUN_DIR` reads reconnectable event pages, and `hypergan checkpoint RUN_DIR` requests a save at the next complete update boundary. These commands share the [run observation contract](docs/observation.md); the optional [local browser and streaming API](docs/local-web.md) now plots file-backed metrics and exposes indexed samples. Install `.[web]` to start the local viewer and projection worker automatically with `train`/`resume`. Use `--open` to open its sign-in page, `--server` to require startup, or `--no-server` for headless execution. After training, use `hypergan serve RUN_DIR --open` to inspect the run; `hypergan project RUN_DIR --follow` independently catches up or follows file contributions. The browser and external API clients share the same authenticated interface. Metric defaults are removable in [configuration](docs/configuration.md); `hypergan metrics RUN_DIR` and `hypergan contributions RUN_DIR` support headless agents.

Developers can exercise the internal [replicated run service](docs/replicated-run-service.md) with CPU/Gloo fixtures or [CUDA/NCCL workers](docs/replicated-cuda.md). The shared controller manages attempts, recovery, save requests, [bounded previews and progress](docs/replicated-observation.md), and final artifacts. The independent broker retains parent-controlled checkpoint publication and cleanup after coordinator death. The public training commands remain single-process; public distributed integration and real cluster qualification are still ahead.

Check a proposed CPU execution profile separately from an explicitly created CPU recipe:

```sh
hypergan preflight cpu-demo --profile examples/execution/cpu-replicated.toml
hypergan preflight cpu-demo --profile examples/execution/cpu-replicated.toml --runtime
```

For GPU construction checks, use `demo` with `--profile examples/execution/cuda-replicated-nccl.toml`. The first command checks structure without training dependencies. `--runtime` constructs the recipe in supervised local workers with a deadline and reports runtime/source/data identity and declared recovery capability. It performs no training updates. See [execution profiles and preflight](docs/execution-profiles.md) for the checks and limits; profile selection for `train` and `resume` is a later integration step.

## Configure the recipe

`hypergan new` writes a GPU-first `config.toml`; `--device cpu` explicitly selects CPU. Configuration selects generator, discriminator, optional encoder/auxiliary components, constructor arguments, explicit input bindings, adversarial losses, gradient penalties, prior regularization and additional task objectives. Built-in identifiers and importable `module:object` constructors support ordinary Python implementations without a layer language.

The reference defaults to ParticleGAN's relativistic-paired objective, b-cap discriminator regularization and VICReg prior regularization. Custom configurations remain runnable with an explicit qualification warning. An unknown combination is different from an invalid binding or incompatible tensor shape: actual incompatibilities fail with an error. No custom configuration inherits quality, distributed or deployment approval merely by completing a run.

See [configuration and component contracts](docs/configuration.md) and the [paired synthetic example](examples/paired-linear.toml). Custom factories execute Python code from your environment; use implementations you trust. Lightweight validation checks configuration structure without importing those factories; training validates runtime bindings and tensors.

## Development and migration

Install `.[dev,train,image]` in the tested runtime environment, then run `python -m pytest` for CPU and lightweight correctness tests. With two visible NVIDIA GPUs and the CUDA runtime, run `python -m pytest tests/cuda`; this separate hardware gate fails if its required GPUs are unavailable. See [CUDA execution and validation](docs/cuda.md). [Foundation CI](.github/workflows/ci.yml) additionally builds wheel/sdist artifacts, checks clean installations outside the checkout and tests lightweight commands without the training stack.

Historical HyperGAN code and experiments are preserved in archive tags. Legacy configurations and checkpoints require their archived runtime; see [migration notes](docs/migration.md) and [preservation evidence](reports/resurrection-preservation-2026-09-18.md).

Work is coordinated through [the execution ledger](reports/resurrection-status.md), [resurrection plan](reports/resurrecting-hypergan-plan-2026-09-18.md), and PRs targeting `develop`. No paid compute is needed for the foundation checkpoint.

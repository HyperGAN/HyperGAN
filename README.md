# HyperGAN

HyperGAN is being rebuilt to make proven GANs configurable and practical: prepare data, train, inspect samples, recover runs, and build generators for applications. The next release integrates on `develop`, using [ParticleGAN](https://github.com/255BITS/ParticleGAN) primitives inside a HyperGAN-owned training loop.

The current foundation supports native CUDA training, flexible Python components,
complete training checkpoints and a local viewer. The first measured image recipe
is [CIFAR-10 with ParticleGAN MoG, b-cap and a pretrained discriminator](docs/cifar-recipe.md),
with a scratch generator, PNG previews and explicit FID evaluation. New projects
target CUDA; CPU execution is explicit for small correctness fixtures. Full image
reproduction, actual two-GPU image qualification and real cluster training remain
release gates. Custom configurations still require their own qualification.

The experimental [256×256 logo colorization demo](docs/colorization.md) pairs a
grayscale particle encoder with a DINOv3 attention discriminator, held-out color
and structure metrics, and multiple sampled colorizations.

## Install the development foundation

Use Python 3.12 for the tested training runtime. Lightweight package checks cover Python 3.11–3.13 on Linux, macOS and Windows; the CPU reference profile is tested on Linux.

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

`hypergan new demo` writes `training.device="cuda"`. Select another visible GPU with `hypergan new demo-gpu1 --device cuda:1`. Training requires that device and fails with installation/device guidance if it is unavailable; it does not silently fall back to CPU. The native CLI path uses one GPU per run. Use `--profile cuda-replicated-nccl` for supervised training on the two visible local GPUs; `resume` infers that saved profile. See [public execution profiles](docs/execution.md) for accumulation, mutable service deadlines and explicit CPU fixtures. Actual multi-host qualification remains a separate gate.

For a CPU correctness fixture, install the CPU PyTorch wheel instead (`--index-url https://download.pytorch.org/whl/cpu`) and create the project with `hypergan new cpu-demo --device cpu`.

A five-step run checks the integration; it is not a convergence benchmark. The run records its resolved configuration, runtime, counters and events, and saves separate complete training checkpoints and generator/prior inference artifacts. Use a new run directory for a new experiment.

To stop and continue the same numerical schedule:

```sh
hypergan train demo --run-dir runs/recoverable --checkpoint-every 1 --stop-after-steps 2
hypergan train demo --run-dir runs/recoverable --checkpoint-every 1 --stop-after-steps 2
```

Repeat `train` with the same project and run directory to resume the latest complete checkpoint. Each invocation above advances at most two updates toward the saved total; omit `--stop-after-steps` to finish it. The resolved configuration must match, except that you can increase `training.steps` or `--steps` when both saved and requested `training.lr_floor` are `1.0` (constant learning rate). The target is a total update count; decreases and extensions of annealed runs are rejected. Other changed training settings require a new run directory. Checkpoint and preview intervals and the execution profile are inherited when omitted. `hypergan resume RUN_DIR` continues toward the saved target, including an accepted extension, and supports `--checkpoint PATH` for recovery from an earlier snapshot. Resume creates a new attempt and preserves existing samples. `--max-seconds` provides a cooperative wall-time limit; `--progress-json` emits live JSONL for process managers. See [recovery and compatibility](docs/recovery.md). Image-folder preflight uses the optional `image` extra and `hypergan data-check CONFIG`; see [image data and preprocessing](docs/image-data.md). The default generator remains a 2D numerical fixture and cannot consume image batches.

Use `--preview-every N` for bounded periodic EMA previews, indexed by a short stable sample name (`g` generated, `x` real; rename the generated source with `--preview-name`) and retained across the whole run, so the viewer's per-sample history slider scrubs from the first sample to the latest. At most 128 previews are kept by default: when a run outgrows the bound the older samples are thinned by doubling their spacing (every 500 steps becomes every 1,000) rather than dropping the beginning of the run, and `--preview-keep N` or `--preview-keep all` chooses a different bound. `hypergan events RUN_DIR` reads reconnectable event pages, and `hypergan checkpoint RUN_DIR` requests a save at the next complete update boundary. These commands share the [run observation contract](docs/observation.md). Routine CLI progress defaults to every 100 updates; change it with `--progress-every N`, independently of stored metrics.

Install `.[web]` to start the [browser and streaming API](docs/local-web.md) automatically with `train`/`resume`. It displays training and evaluation metrics, image grids and tensor previews, grouped by sample name with the newest image shown and a slider for earlier versions. Use `--open` to open the browser, `--server` to require startup, or `--no-server` for headless execution. The viewer stays available after training and is reused on resume: `hypergan server-status RUN_DIR` finds it, and `hypergan stop-server RUN_DIR` stops it. Binding defaults to `0.0.0.0:8765`, or the next free port above it, so the viewer URL stays the same between runs; `--port` requires an exact port and `--server-host` selects another address. Enable authentication with `--auth token`. To reach the viewer over HTTPS, put it behind your own TLS proxy and name that proxy's origin: `tailscale serve --bg 8765` then `--public-origin https://<machine>.<tailnet>.ts.net`, which is the only non-local origin the viewer accepts. Standalone `hypergan serve RUN_DIR --open` and `hypergan project RUN_DIR --follow` remain available. Metric defaults are removable in [configuration](docs/configuration.md); `hypergan metrics RUN_DIR` and `hypergan contributions RUN_DIR` support headless agents.

Developers can exercise the [replicated run service](docs/replicated-run-service.md) with CPU/Gloo fixtures or [CUDA/NCCL workers](docs/replicated-cuda.md). The shared controller manages attempts, recovery, save requests, [bounded previews and progress](docs/replicated-observation.md), and final artifacts. The independent broker retains parent-controlled checkpoint publication and cleanup after coordinator death. Public `train`/`resume` select native or replicated execution through profiles; actual distributed image-recipe and real multi-host qualification remain open.

Check a proposed CPU execution profile separately from an explicitly created CPU recipe:

```sh
hypergan preflight cpu-demo --profile examples/execution/cpu-replicated.toml
hypergan preflight cpu-demo --profile examples/execution/cpu-replicated.toml --runtime
```

For GPU construction checks, use `demo` with `--profile examples/execution/cuda-replicated-nccl.toml`. The first command checks structure without training dependencies. `--runtime` constructs the recipe in supervised local workers with a deadline and reports runtime/source/data identity and declared recovery capability. It performs no training updates. See [execution profiles and preflight](docs/execution-profiles.md) for the checks and limits; profile selection for `train` and `resume` is a later integration step.

## Configure the recipe

`hypergan new` writes a GPU-first `config.toml`; `--device cpu` explicitly selects CPU. Configuration selects generator, discriminator, optional encoder/auxiliary components, constructor arguments, explicit input bindings, adversarial losses, gradient penalties, prior regularization and additional task objectives. Built-in identifiers and importable `module:object` constructors support ordinary Python implementations without a layer language.

The reference trains with ParticleGAN 0.8's formulation (K3P): the relativistic-paired logistic objective, its learning-rate-scheduled critic penalty with an EMA-critic anchor, and VICReg prior regularization. Custom configurations remain runnable with an explicit qualification warning. An unknown combination is different from an invalid binding or incompatible tensor shape: actual incompatibilities fail with an error. No custom configuration inherits quality, distributed or deployment approval merely by completing a run.

Networks are defined in HNDL configuration, with source recorded in every resolved run. Edit the TOML or `.hndl` file to change an architecture without rebuilding HyperGAN.

See [configuration and component contracts](docs/configuration.md) and the [paired synthetic example](examples/paired-linear.toml). Custom factories execute Python code from your environment; use implementations you trust. Lightweight validation checks configuration structure without importing those factories; training validates runtime bindings and tensors.

## Development and migration

Install `.[dev,train,image]` in the tested runtime environment, then run `python -m pytest` for CPU and lightweight correctness tests. That default selection deselects the `heavy` marker, which gates the tests that start real subprocesses, multi-rank jobs and worker services; run those deliberately with `python -m pytest -m heavy`. GitHub runs the heavy jobs only on pushes to `master` and PRs targeting `master`; `develop` uses the fast selection. With two visible NVIDIA GPUs and the CUDA runtime, run `python -m pytest tests/cuda`; this separate hardware gate fails if its required GPUs are unavailable. See [CUDA execution and validation](docs/cuda.md). [Foundation CI](.github/workflows/ci.yml) additionally builds wheel/sdist artifacts, checks clean installations outside the checkout and tests lightweight commands without the training stack.

Historical HyperGAN code and experiments are preserved in archive tags. Legacy configurations and checkpoints require their archived runtime; see [migration notes](docs/migration.md) and [preservation evidence](reports/resurrection-preservation-2026-09-18.md).

Work is coordinated through [the execution ledger](reports/resurrection-status.md), [resurrection plan](reports/resurrecting-hypergan-plan-2026-09-18.md), and PRs targeting `develop`. No paid compute is needed for the foundation checkpoint.

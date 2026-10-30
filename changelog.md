# Changelog

This is the project's only changelog. Entries under **Unreleased** are on `develop` and not yet on PyPI; the package version (`hypergan.__version__`) changes only when a release is cut.

## Unreleased

## 2.0.0 — 2026-10-30

HyperGAN 2.0.0 is the official General Availability (GA) production release. It is a ground-up architectural rebuild centering on configurable ParticleGAN (`>=0.9.4,<0.10`) primitives, declarative `hypergan-model/1` TOML model files, HNDL (`>=0.10.0,<1`) networks, reproducible training lifecycles, an asynchronous Web Observatory, distributed multi-GPU & multi-node training, standalone privacy-preserving model export, and a fully typed Python API (`import hypergan as hg`).

### Added
- **Core Engine & Dependency Modernization**:
  - Python 3.11–3.14 support with updated stable dependencies: `particlegan>=0.9.4,<0.10`, `torch-fidelity>=0.4.0,<0.5`, `safetensors>=0.8,<1`, `torch>=2.6,<3`, and `Pillow>=13.0.0,<14`.
  - Added optional dependency extras: `[fid]`, `[onnx]`, `[all]`, `[safetensors]`.
  - On-demand external URL cache engine (`~/.hypergan/cache/`) with SHA-256 integrity verification, multi-rank file locking (`fcntl.flock`), and `HYPERGAN_OFFLINE=1` support without caching local paths.
  - Automatic download and caching of pretrained backbones (ResNet-18 `f37072fd` and Inception-v3 `6726825d`).
- **Distributed Multi-GPU & Multi-Node Training**:
  - Out-of-the-box multi-node & multi-GPU training promoted to GA (`gradient_allreduce`, `parameter_average`, `pseudo_gradient`).
  - Zero-TOML CLI arguments: `--nodes`, `--node-rank`, `--rendezvous`, `--strategy`, `--sync-every`.
  - Cluster environment auto-detection for Slurm and torchrun (`SLURM_NNODES`, `SLURM_NODEID`, `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `GROUP_RANK`, `NNODES`).
  - NFS-free distributed checkpoints where Rank 0 owns `run_dir` locally and coordinates checkpoint distribution over collective scatter/gather.
  - Hot-path collective optimization: memory-contiguous parameter gradient bucketing in `_reduce_gradients` and elimination of per-step Python `all_gather_object` barriers (`_agree`/`_phase`) after step 1.
- **AI Agent Ergonomics & Telemetry**:
  - Static AST and symbolic shape verification via `hndl.resolve()` inside `validate_network_args()` before allocating GPU weights.
  - Symbolic `shape_fn` hooks on pretrained providers to bypass meta-device weight tracing during validation.
  - `difflib` field suggestions on unknown keys in TOML model files.
  - New introspection CLI commands and Python APIs: `hypergan schema` (`hg.schema()`) and `hypergan ops` (`hg.operators()`).
  - Machine-readable root `llms.txt` and complete `docs/hndl-reference.md` for AI coding agents.
  - Universal `--json` support across all CLI subcommands with predictable stdout output and structured error envelopes `{"ok": false, "error": {...}}`.
  - Global `--quiet` / `-q` flag suppressing banners, headers, and spinners.
  - Real-time `D=` and `G=` loss metrics rendered on normal CLI progress lines.
  - 8 quantitative GAN health telemetry scalars emitted every 10 steps: `critic/real_mean`, `critic/fake_mean`, `critic/margin`, `critic/accuracy`, `grad_norm/critic`, `grad_norm/generator`, `grad_norm/prior`, `optimizer/d_guard_ratio`, `prior/particle_rms`, and batch `diversity/ratio`.
  - Rolling EMA metrics and automated `health_alerts` (e.g. `mode_collapse_suspected`, `critic_saturated`) in `summary.json` and `hg.status()`.
- **Web Observatory**:
  - Network binding flags `-b` / `--bind` (aliasing `--host`, including `-b 0.0.0.0`) across `train`, `resume`, `view`, and `serve`.
  - Loopback default listener (`127.0.0.1`) skipping external UDP socket probes on loopback.
  - TTY-gated autostart respecting `HYPERGAN_NO_SERVER=1` and `CI=1`.
  - Rebuilt Web UI frontend asset (`app.js`) surfacing latest/best evaluation metrics (FID), preview thumbnails, and single-run header action controls (`Sample` and `Checkpoint`).
  - Informative empty-state guidance when `preview_every = 0` or no samples exist.
- **Production Model Export & Inference**:
  - Privacy-preserving export stripping raw `last_batch` image tensors from exported artifacts unless `--include-example-inputs` is specified.
  - Standalone generator baking with learned particle tables (`prior.z`, `prior.sigma`) embedded directly into generator modules.
  - Multi-format standalone export across `bundle`, `safetensors`, `pt2`, and `onnx` with zero ParticleGAN/HNDL runtime dependencies.
  - Callable `LoadedGenerator` (`nn.Module`) from `hg.load_bundle(..., device="cuda")` with `.sample(...)` and `.interpolate(...)`.
  - GPU-accelerated batch sampling CLI: `hypergan sample --device cuda --output-dir <dir>`.
  - Non-image conditioning support for JSON and tensor strings in `hypergan sample` and `hg.sample()`.
- **Data & Starter Templates**:
  - `ImageFolder` RGBA and palette alpha compositing options (`alpha = "composite_white" | "composite_black" | "error"`).
  - Opt-in symlink traversal (`follow_symlinks = true`) in `ImageFolder`.
  - Multi-GPU-safe starter templates supporting 32..1024px resolutions using `group_norm` and `rms_norm` (never `batch_norm`).
  - Extended starter template catalog: `transgan`, `colorization`, `paired-image`, and `--conditional`.
  - Packaged canonical recipes in `src/hypergan/models/`: `gaussian-grid.toml`, `cifar10-tiny-transformer.toml`, `image-folder-128.toml`, `colorization-128.toml`, and `cifar10-transgan.toml`.
- **Benchmarks**:
  - Multi-Scale TransGAN Discriminator authored in pure HNDL (`transgan-discriminator-32.hndl`).
  - Head-to-head CIFAR-10 TransGAN Critic benchmark on 2× RTX A6000 evaluating FID-50k curves, packaging the winning recipe (`cifar10-transgan.toml`).
  - Bounded multi-GPU scaling benchmark re-measured on 2× RTX A6000 demonstrating 2.20× speedup at `batch = 256`.
- **Developer APIs & Typing**:
  - Static typing compliance: PEP 561 `src/hypergan/py.typed` marker, `if TYPE_CHECKING:` imports in `__init__.py`, and trimmed `__all__` public symbol surface (~35 canonical symbols).
  - Typed dataclasses: `RunStatus`, `PsSnapshot`, `ValidationResult`, `LoadedGenerator`.
  - Non-blocking training launch `hg.launch()` and `on_event` early-stop mechanism via `hg.StopTraining`.
  - In-process CUDA device targeting (`cuda:N`) without mutating global `CUDA_VISIBLE_DEVICES`.
  - Five new comprehensive user guides: `custom-images.md`, `deployment.md`, `custom-pytorch.md`, `python-api.md`, `hndl-reference.md`.

### Changed
- Promoted distributed multi-GPU and multi-node training out of `--experimental` into first-class GA.
- Restored default metric logging cadence to 10 steps (`metrics.every_steps = 10`).
- Replaced 6 divergent metadata readers with a single unified `RunInfo` dataclass reader.
- `compare_runs` now dispatches `min` vs `max` based on metric catalog declared `direction`.
- Cleaned `examples/` directory of broken scratchpads, ablations, and `REPLACE_WITH_...` placeholders, switching unconditional examples to `builtin = "image_folder"`.
- Reconciled 8 documentation contradictions across `configuration.md`, `image-fid.md`, `README.md`, `cifar-recipe.md`, and `GLOSSARY.md`.

### Fixed
- Fixed InceptionFID crash on SciPy 1.18+ by adopting `np.linalg.eigvals` via `torch-fidelity>=0.4.0`.
- Fixed child SIGTERM/SIGINT process forwarding in `hypergan queue run` with bounded shutdown escalation.
- Fixed `checkpoints prune` to acquire `run_lock`, restrict deletion to `run_dir / "checkpoints"`, and account reclaimed bytes accurately.
- Fixed 5 silent `except Exception: pass` swallows during summary JSON writes.
- Fixed Pytest state leakage polluting user runs registry by introducing autouse `HYPERGAN_STATE_DIR` temp fixture.
- Fixed CUDA initialization order race in `test_execution_launch.py`.
- Fixed 512px template shape mismatch bug where 512px datasets generated 256px network shapes.
- Fixed `hypergan doctor` incorrectly reporting active runs due to dictionary length evaluation.
- Fixed `hypergan report` missing `summary.json` keys rendering Loss G as `--`.
- Fixed tag visibility across `hypergan status`, `Run.name`, and Web Observatory.

### Removed
- Removed dead 806-line `src/hypergan/runpod_runner.py` stub and associated mock tests.
- Removed legacy unmaintained scripts and broken ablation configs from `examples/`.

## 2.0.0a1 — 2026-10-23

HyperGAN 2.0.0a1 is a ground-up architectural rebuild centered on configurable ParticleGAN (`>=0.9.2,<0.10`) primitives, declarative `hypergan-model/1` TOML model files, HNDL (`>=0.10.0`) networks, reproducible training lifecycles, a functional Python API (`import hypergan as hg`), and real-time web observation (`hypergan view`) backed by Foldlite DB (`>=0.1.2,<0.2`).

- **Unified `format = "hypergan-model/1"` Configuration Standard**:
  - Single, human-editable TOML model schema (`[data]`, `[prior]`, `[networks.<name>]`, `[[losses]]`, `[penalty]`, `[train]`, `[evaluations.<id>]`, `[metrics]`, `[sampling]`, `[checkpoints]`, `[run]`).
  - Recursive configuration inheritance via `extends` with relative path anchoring, workspace variable interpolation (`hypergan.toml`), and on-the-fly CLI overrides (`-s / --set KEY=VALUE`).
  - Automatic dataset header inspection and starter project scaffolding via `hypergan new --project <dir> --dataset <path>` across `dcgan`, `tiny-transformer`, `resnet`, and `gaussian-grid` templates.
- **HNDL Declarative Network DSL**:
  - Networks are authored inline (`source = """..."""`) or in standalone `.hndl` files (`hndl = "path/to/net.hndl"`) with static shape contract verification (`input_shape`, `output_shape`) before training begins.
  - Native support for residual blocks, self-attention, 2D relative-position bias, split-latent AdaIN (`adaptive_norm`), equalized linear/conv layers, coordinate warps, Fourier features, differentiable augmentation (`diff_augment`), and SHA-256-pinned frozen pretrained backbones (`torchvision_resnet18`, `dinov3_vits16`).
  - Resolved HNDL sources are embedded into run manifests and version-2 checkpoints (`CURRENT_VERSION = 2`) for self-contained provenance.
- **Config-First Python Developer API (`import hypergan as hg`)**:
  - Lazy-loaded functional Python API (`hg.new`, `hg.load`, `hg.Model`, `hg.override`, `hg.validate`, `hg.train`, `hg.resume`, `hg.sample`, `hg.evaluate`, `hg.export`, `hg.load_bundle`, `hg.ps`, `hg.status`, `hg.logs`, `hg.compare`, `hg.diff`).
  - Importing `hypergan` and validating or inspecting configurations remains 100% Torch-free until numerical execution starts.
- **Multi-Network & Auxiliary Graph Routing**:
  - Explicit role-based network graphs (`role = "generator" | "critic" | "encoder" | "auxiliary"`) with weight sharing (`reuse`, `freeze_parameters`), multi-critic adversarial terms, particle-routing autoencoders (`ae_gan`, `vae_gan`), and paired conditional reconstruction objectives (`d-then-g-v1` update program).
- **FoldLiteDB Embedded Map/Reduce Storage & WASM Reducers**:
  - Backed by `foldlitedb>=0.1.2,<0.2` for atomic append-only event journaling, bounded cursor pagination, durable `.fold` historical view caches, and shared Rust/WebAssembly streaming reducers across server and browser.
- **Web Observatory (`hypergan view`) & Docker-Style Run Controller**:
  - Global Runs Dashboard (`GET /` and `GET /api/runs`) and per-run Web Observatory (`GET /runs/<id_or_slug_or_tag>`) with real-time SSE streaming, retained EMA preview history sliders, snapshot evaluation charts, and an interactive **Model** tab.
  - Live browser and CLI run controls (`RunController` start, graceful stop, resume, on-demand durable checkpointing, sampling, evaluation, and multi-GPU sequential job queueing via `hypergan queue`).
  - Daemonless global run registry addressing runs from any directory by Docker-style slug (`fleet-hawk`), short ID, custom tag (`hypergan tag`), or path.
- **CUDA, Multi-GPU & AMD ROCm Support**:
  - Single-GPU and multi-GPU replicated training (`--devices 0,1` / `cuda-replicated-nccl`) with coordinated rank recovery and automatic CPU fallback warnings when CUDA is unavailable.
  - First-class AMD ROCm / HIP runtime detection, determinism guards, and GPU inventory support alongside NVIDIA CUDA.
- **Security & Packaging**:
  - Require `Pillow>=13.0.0,<14` for image extras, allowlist `Image.open` formats, and enforce `weights_only=True` in EMA bundle extraction.
  - Automated CI distribution verification (`twine check --strict`, stamped `_build_provenance.json`) and PyPI Trusted Publishing via `.github/workflows/release.yml`.

The entries below describe historical versions; see [migration notes](docs/migration.md) before using legacy configurations or checkpoints.

## 1.0 - Pytorch!

* Switched to pytorch
* Almost a complete rewrite
* Faster, cleaner, more stable
* Model definition language formalized
* New custom train hooks not available in other projects
* More...

## 0.10 ~ Community Project release

* HyperGAN 0.10 released as a community project
* Configurable network architectures using a simple DSL
* Lots of new regularizers and losses
* Evolution based GangTrainer
* Curriculum trainer with progressive growing
* Optimistic loading
* New next-frame example
* More...

## 0.9 ~ Refactorings and optimizations

* API Documentation - [https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html](https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html)
* Prepackaged configurations
* New viewer front-end!
* Examples, including the ability to randomly search for good configurations

See more here [https://github.com/255BITS/HyperGAN/pull/66](https://github.com/255BITS/HyperGAN/pull/66)

## 0.8 ~ Least Squares GAN and API examples

* Tensorflow 1.0 support
* New configuration format and refactored api.
* New loss based on least squared GAN.  See [least squares GAN implementation](changelog.md#Least-Squares-GAN).
* API example `2d-test` - tests a trainer/encoder/loss combination against a known distribution.
* API example `2d-measure` - measure and report the above test by randomly combining options.
* Updated default configuration.
* More

## 0.7 ~ WGAN & API

* New loss `wgan`
* Initial Public API Release
* API example: `colorizer` - re-colorize an image!
* API example: `inpainter` - remove a section of an image and have your GAN repaint it
* API example: `super-resolution` - zoom in and enhance.  We've caught the bad guy!
* 4 _new_ samplers.  `--sampler` flag.  Valid options are: `batch`,`progressive`,`static_batch`,`grid`. 

## 0.5 / 0.6

* pip package released

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold-0-5-6.png) ![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold.png)

### 0.1-0.4

* Initial private release

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1.png) ![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1-2.png)


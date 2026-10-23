# Changelog

## 2.0.0a1 — 2026-10-23

HyperGAN 2.0.0a1 is a ground-up architectural rebuild centered on configurable ParticleGAN (`>=0.9.2,<0.10`) primitives, declarative `hypergan-model/1` TOML model files, HNDL (`>=0.9.0`) networks, reproducible training lifecycles, a functional Python API (`import hypergan as hg`), and real-time web observation (`hypergan view`) backed by Foldlite DB (`>=0.1.2,<0.2`).

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


# Changelog

## 0.2.0 (`develop` Release Highlights)

HyperGAN 0.2.0 is a ground-up architectural rebuild centered on declarative model files, reproducible training lifecycles, and real-time web observation.

### Highlights

- **Unified `format = "hypergan-model/1"` Configuration Standard**:
  - Single, human-editable TOML model schema (`[data]`, `[prior]`, `[networks.<name>]`, `[[losses]]`, `[penalty]`, `[train]`, `[evaluations.<id>]`, `[metrics]`, `[sampling]`, `[checkpoints]`, `[run]`).
  - Recursive configuration inheritance via `extends` with relative path anchoring, workspace variable interpolation (`hypergan.toml`), and on-the-fly CLI overrides (`-s / --set KEY=VALUE`).
  - Automatic dataset header inspection and starter project scaffolding via `hypergan new --project <dir> --dataset <path>` across `dcgan`, `tiny-transformer`, `resnet`, and `gaussian-grid` templates.
- **HNDL Declarative Network DSL**:
  - Networks are authored inline (`source = """..."""`) or in standalone `.hndl` files (`hndl = "path/to/net.hndl"`) with static shape contract verification (`input_shape`, `output_shape`) before training begins.
  - Native support for residual blocks, self-attention, 2D relative-position bias, split-latent AdaIN (`adaptive_norm`), equalized linear/conv layers, coordinate warps, Fourier features, differentiable augmentation (`diff_augment`), and SHA-256-pinned frozen pretrained backbones (`torchvision_resnet18`, `dinov3_vits16`).
  -Resolved HNDL sources are embedded into run manifests and version-2 checkpoints (`CHECKPOINT_SCHEMA_VERSION = 2`) for self-contained provenance.
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

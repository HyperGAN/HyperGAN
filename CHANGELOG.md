# Changelog

This is the project's only changelog. Entries under **Unreleased** are on `develop` and not yet on PyPI; the package version (`hypergan.__version__`) changes only when a release is cut.

## Unreleased

- **Reproducible whitepaper benchmark suite & deterministic 64px dataset preparer (`benchmarks/whitepaper/`, `scripts/prepare_benchmark_dataset.py`, `tests/unit/test_whitepaper_benchmark_configs.py`).**
  - Added 8 pure declarative `hypergan-model/1` + native HNDL benchmark configurations (`batch_size = 256`, `steps = 100000`, interval `evaluations.fid50k_train` plus live manual `evaluations.inception_suite` and `evaluations.dinov2_suite` evaluators) pairing ParticleGAN `unified_v12` (`defaults = "particlegan"`, zero recipe/optimizer overrides) against an honest non-saturating Adam + R1 baseline (`lr = 0.0002`, `penalty = 1.0`, `ema = 0.999`, Gaussian prior) across 4 distinct architecture/dataset pairs: CIFAR-10 Radford BN-DCGAN (`1.18M` G / `0.66M` D), CIFAR-10 scratch ResNet-GAN (`6.03M` G / `7.43M` D), CIFAR-10 pure Transformer TransGAN (`2.84M` G / `10.20M` D), and FFHQ-64 64×64 scratch ResNet-GAN (`6.22M` G / `7.84M` D).
  - Added `scripts/prepare_benchmark_dataset.py` (stdlib + Pillow only, HTTP Range resume, SHA-256 verification, forward-order archive extraction, center-crop + Lanczos resizing, atomic `image_folder` publishing with `dataset-manifest.json`) supporting pinned `ffhq64` (70,000 aligned 64×64 RGB faces for unbiased FID-50k / FD_DINOv2-50k) and `flowers102` (8,189 Oxford VGG images).
- **Publication-grade image evaluation metrics: DINOv2 FD & single-pass Inception/DINOv2 PRDC suites (`src/hypergan/image_metrics.py`, `src/hypergan/cache.py`, `src/hypergan/schema_info.py`, `docs/image-fid.md`).**
  - Pinned Meta's official DINOv2 ViT-L/14 (`dinov2_vitl14_pretrain.pth`, LVD-142M) URL, byte size, and SHA-256 (`d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`) in `cache.py` with on-demand locked caching and strict offline enforcement (`HYPERGAN_OFFLINE=1`).
  - Added a clean, self-contained PyTorch DINOv2 ViT-L/14 implementation (zero `torch.hub` git-clone or external registry dependencies) and `hypergan.image_metrics:DINOv2FD` implementing the Stein et al. (NeurIPS 2023) `FD_DINOv2` protocol (uint8 RGB quantization, antialiased bicubic 224×224 resize, ImageNet normalization, FP32 without TF32, float64 unbiased covariance, eigenvalue Fréchet distance without `scipy.linalg.sqrtm`, and integrity-checked reference moment caching under `$HYPERGAN_CACHE_DIR/fid-reference`).
  - Added multi-scalar single-pass evaluators `hypergan.image_metrics:InceptionSuite` (returning `fid`, `precision`, `recall`, `density`, `coverage`) and `hypergan.image_metrics:DINOv2Suite` (returning `fd_dinov2`, `precision`, `recall`, `density`, `coverage`), combining Fréchet Distance over the full sample count with Kynkäänniemi et al. (2019) k-NN Precision/Recall (`pr_k=3`) and Naeem et al. (2020) Density/Coverage (`dc_k=5`) over a deterministic `manifold_samples=10000` subset via chunked exact L2 distances and cached reference manifold features/radii (`.suiteref`). Preserved `InceptionFID` 100% backwards-compatibly.
  - Added standalone EMA bundle evaluation helper `hypergan.image_metrics:evaluate_bundle` and module CLI (`python -m hypergan.image_metrics`) supporting JSON receipts (`--output`) and idempotent run history publication via `evaluation_results.import_scores` (`--publish-run`).
- **Google Colab GPU benchmark & evaluation orchestrator (`scripts/colab_bench.py`, `docs/colab.md`, `tests/unit/test_colab_bench.py`).**
  - Added `scripts/colab_bench.py` with `run`, `eval-bundles`, `status`, and `cleanup` subcommands for parallel HyperGAN benchmarks across up to 3 warm Google Colab GPU VMs (`T4`, `L4`, `G4`, `A100`, `H100`, with ordered `--fallback-gpus` on 503 allocation bursts and CU/hr cost tracking), isolated per-worker `--config` files, staggered VM launches (`--stagger-seconds 20`), OS process-group hard timeouts (`SIGTERM -> SIGKILL`), two-layer teardown (local `try/finally`/`SIGINT`/`SIGTERM`/`atexit` + detached VM-side `/content/watchdog.py` `/unassign`), automatic CIFAR-10 / FFHQ-64 / Flowers-102 dataset caching (`scripts/prepare_benchmark_dataset.py`), detached VM training with short-poll SSH monitoring (`--detached`, `--poll-seconds`) and serialized periodic/chunked checkpoint pull-back with automatic `hypergan resume`, collapse/NaN telemetry detection, `python -m hypergan.image_metrics` (`InceptionSuite`, `DINOv2Suite`, `InceptionFID`, `DINOv2FD`) bundle evaluation, and seamless Observatory artifact unpacking into `runs/<run_name>/`.
  - Documented workflows and safety guarantees in `docs/colab.md` and added CPU unit tests mocking `colab` CLI subprocesses in `tests/unit/test_colab_bench.py`.

## 2.2.0 — 2026-11-23

HyperGAN 2.2.0 upgrades the default training formulation to **ParticleGAN `0.12.0`** (`>=0.12.0,<0.13`, `DEFAULT_RECIPE = "unified_v12"`) and upgrades **Foldlite DB** to **`0.5.0`** (`>=0.5.0,<0.6`), alongside the new 255labs-branded GitHub Pages documentation site and Observatory UI metrics rail & model configuration clarity overhaul. Install with `python -m pip install "hypergan[all]==2.2.0"` after installing the matching Torch wheels for your hardware. [Installation](https://github.com/HyperGAN/HyperGAN/blob/v2.2.0/README.md#install)

- **ParticleGAN `0.12.0` (`unified_v12`) upgrade (`pyproject.toml`, `.github/scripts/install_particlegan.py`, `aggregation.py`, `schema_info.py`, `docs/particlegan.md`, `README.md`, `llms.txt`).**
  - Upgraded the `train` and `all` extras to `particlegan>=0.12.0,<0.13`, adopting ParticleGAN's energy-optimal 4-Law default `unified_v12` (`UnifiedV12Optimizer`, `optimizer_family = "unified_v12_equil"` — zero-state polar-matched 4D Conv2d Frobenius-MP, Law 4.4 asymmetric 0-NS 2D critic equilibration with generator-free 2D NS fallback, prime-67 CountSketch period-2 flip-bifurcation control, and zero-sync pullback/EMA fast-paths while preserving explicit `defaults = "particlegan:unified_v11"` byte-identically).
  - Extended `sync_unified_v11_record` and `_synchronize_optimizer_regularizers` in `src/hypergan/aggregation.py` to synchronize `UnifiedV11StepRecord._g_func_gain` and `UnifiedV12Optimizer`'s `_ema_flip_cos` / `_prev_flip_sketch` state across distributed ranks, and added `particlegan:unified_v12` to `MODEL_SCHEMA` defaults choices. [ParticleGAN integration](https://github.com/HyperGAN/HyperGAN/blob/v2.2.0/docs/particlegan.md)
- **Upgrade Foldlite DB to `0.5.0` (`pyproject.toml`, `.github/scripts/install_foldlitedb.py`, `README.md`, `docs/foldlitedb.md`, `llms.txt`, `scripts/modal_multinode_smoke.py`, `scripts/modal_scaling.py`).** Upgraded Foldlite DB to `foldlitedb>=0.5.0,<0.6` across base dependencies, optional `reducers`/`web`/`all` extras, CI installation, documentation, and Modal smoke/scaling scripts (grouped portable view reduction, single-pass SDK JSON/number parser, native Rust/storage/PyO3 hot-path optimizations, and browser worker improvements; `FOLDLOG2` format and ABI v1 remain unchanged and existing disposable `.cache` files reconstruct automatically on first open).
- **GitHub Pages documentation site, auto-generated references, and README/metadata polish (`site/`, `.github/workflows/pages.yml`, `README.md`, `pyproject.toml`, `MANIFEST.in`, `.gitignore`, `llms.txt`, `examples/README.md`, `docs/local-web.md`, `docs/hndl-reference.md`, `GLOSSARY.md`).**
  - Added the 255labs-branded GitHub Pages documentation site (`https://hypergan.github.io/HyperGAN/`) built from `site/` and deployed via `.github/workflows/pages.yml`, with auto-generated Python API, CLI, `hypergan-model/1` TOML schema, HNDL operator, and Examples Gallery references alongside the existing `docs/` guides, `GLOSSARY.md`, and `CHANGELOG.md`.
  - Polished `README.md` for GitHub and PyPI with top-of-fold badges, GitHub Pages navigation links, colorizer visual showcase, and verified highlights table; updated `[project.urls]` in `pyproject.toml`; excluded `site/` from sdists (`MANIFEST.in`) and ignored `/_site/` (`.gitignore`).
  - Updated `llms.txt` and `examples/README.md` (including `cifar-transgan-resnet18.toml` and `cifar-transgan-scratch.toml`) and fixed factual inaccuracies in `docs/local-web.md` (default `127.0.0.1` viewer bind), `docs/hndl-reference.md` (12 operator categories and duplicate `#65` numbering), and `GLOSSARY.md` (HNDL bundle serving dependencies).
- **Observatory UI metrics rail & model configuration clarity (`frontend/src/presets.js`, `frontend/src/app.js`, `frontend/src/model.js`, `src/hypergan/web_assets/index.html`, `src/hypergan/web_assets/style.css`, `src/hypergan/web_assets/app.js`).**
  - Bounded and grouped the left metrics catalog (`Core` open by default; `Losses`, `Critic & Gradients`, `Diversity`, `Per-critic`, and `Runtime & Progress` collapsible with search auto-expand and selected-count badges) inside a sticky viewport-bounded rail so large catalogs never elongate the page.
  - Tightened default Learning Curves selection from 8 to 4 core metrics (`loss/g_total`, `loss/d_total`, `critic/accuracy` → `critic/margin`, `diversity/ratio` → `loss/gradient_penalty`) for a 2×2 initial grid while preserving `8 / 8` capacity, FoldliteDB pinned headline losses, and preset buttons; added `#metric-count` (`selected / 8`), chart-card `×` removal buttons, and cleaner single-partition tooltip labels.
  - Added an at-a-glance `#model-overview` architecture strip near the top of the run page (prior, primary networks with shapes and trainable/frozen parameter splits, optimizer LRs, and smooth jump link) and reordered `#model-section` to lead with a 3-stage **Architecture & Dataflow** diagram built from `model.edges` and enhanced **Networks** cards (`display_name`, trainable/frozen parameter bars, leading HNDL summary, parameterized/pretrained op-breakdown chips, explicit layer `Key args`, and a `Key layers only` / `Show all N nodes` filter for graphs with >16 nodes).

## 2.1.1 — 2026-11-21

HyperGAN 2.1.1 upgrades the training dependency baseline to **ParticleGAN `0.11.2`** (`>=0.11.2,<0.12`), **HNDL `0.12.0`** (`>=0.12.0,<1`), and **Foldlite DB `0.4.2`** (`>=0.4.2,<0.5`), adding independent multi-critic training with coupled thermal coordination, dual-safe `fwAD` softmax critic penalties, full-graph Unified-v11 composite generator probing, per-critic telemetry, multi-output snapshot evaluations, and `checkpoints.latest_only`. Install with `python -m pip install "hypergan[all]==2.1.1"` after installing the matching Torch wheels for your hardware. [Installation](https://github.com/HyperGAN/HyperGAN/blob/v2.1.1/README.md#install)

- **ParticleGAN `0.11.2` (`unified_v11`) upgrade (`pyproject.toml`, `docs/particlegan.md`, `.github/scripts/install_particlegan.py`).** Upgraded the `train` and `all` extras to `particlegan>=0.11.2,<0.12` (fixing high-dimensional 2D vector critic silencing, premature thermal freeze, and critic probe lockout in `unified_v11`, and adding the public composite generator probe API and RNG-safe replay). [ParticleGAN integration](https://github.com/HyperGAN/HyperGAN/blob/v2.1.1/docs/particlegan.md)

- **Multi-critic coupled thermal coordination & dual-safe `fwAD` softmax critic penalties (`particlegan_runtime.py`, `training.py`, `objective_program.py`, `distributed_training.py`, `aggregation.py`, `checkpoints.py`, `metrics.py`, `docs/particlegan.md`).**
  - Enforce the conservative `coupled_max` thermal policy (`MultiCriticThermalCoordinator`) across multi-critic Unified-v11 runs so each critic keeps independent per-critic residual EMAs while sharing one synchronized `_thermal_scale = max_i s_i` held constant within each step and committed at the step boundary (preserving configured `generator_weight`/`critic_weight` ratios without silent weight drift, keeping `opt_g`/`prior`/`spread` aligned with the hottest critic, and preserving bit-exact checkpoints and replicated ranks). Publish `critic/thermal_scale`, `critic/<term_id>/thermal_scale`, and `critic/<term_id>/raw_thermal_scale` telemetry.
  - Wrap critic penalty evaluation in `invoke_critic_penalty` / `DualSafeSoftmaxMode` (a thread-local `TorchFunctionMode` active only for forward-AD dual tensors during critic penalties) to compute exact analytic reverse-mode-safe JVPs for `torch.softmax`, `F.softmax`, `Tensor.softmax`, `log_softmax`, `softmin`, `scaled_dot_product_attention`, and `nn.MultiheadAttention` without calling `torch.autograd.grad`, fixing the PyTorch `fwAD` + `torch.softmax` backward `Exp` version-counter crash (`pytorch#162350`) across PyTorch 2.8–2.15.

- **Unified-v11 composite generator probing (`recipes.py`, `training.py`, `distributed_training.py`).** Attach a full-graph generator probe adapter (`_GeneratorProbeModule`) that replays the complete generator computation from the original sampled latent and conditioning inputs through every upstream component to the configured generated output using current weights, while keeping `ComponentGraph.state_dict()` and checkpoint/resume contracts unchanged. Associate the probe with the unique trainable generator parameters affecting that output (including shared upstream components once and keeping critic/prior parameters in their existing roles), preserve RNG streams, module modes, and BatchNorm running statistics during replay, and synchronize composite/conditional probes across distributed ranks.

- **Per-critic metrics for multi-critic training (`metrics.py`, `objective_program.py`, `distributed_training.py`, `model_description.py`, viewer).** Publish per-term discriminator metrics (`loss/critics/<term_id>/d_adversarial`, `loss/critics/<term_id>/gradient_penalty`, `loss/critics/<term_id>/d_total`, `critic/<term_id>/real_mean`, `critic/<term_id>/fake_mean`, `critic/<term_id>/margin`) directly from the training forward pass on single-device and distributed runs (including gradient accumulation) with stable term IDs so terms sharing a critic remain distinguishable, while preserving existing aggregate series. Fix top-level `critic/real_mean`, `critic/fake_mean`, `critic/margin`, and `critic/accuracy` to report the primary adversarial term instead of the last term, identify the term on margin charts, link per-term discriminator series in the viewer's Model tab, and surface multi-critic series in the viewer's Losses and Health presets.

- **Independent multi-critic training & checkpoint retention.** Train multiple ParticleGAN critics from one `hypergan-model/1` file with independent training and checkpoint state, `critic_weight` and `generator_weight` overrides on adversarial losses, and `checkpoints.latest_only = true` to retain one published training checkpoint across saves and resumes while preserving metric and event histories.

- **Multi-output snapshot evaluations & flexible batch sizes.** Snapshot evaluators can score a shared population of named EMA outputs in one call with per-metric charts, source-step history, `hypergan.import_evaluation` backfill, and any positive integer `batch_size` (including `batch_size > 1024`). [Model configuration and Python examples](https://github.com/HyperGAN/HyperGAN/blob/v2.1.1/docs/configuration.md#several-scores-from-one-ema-sample-population)

- **Dependency & release verification updates.** Upgrade HNDL to `hndl>=0.12.0,<1` in the `train` and `all` extras, upgrade Foldlite DB to `foldlitedb>=0.4.2,<0.5` across base dependencies, optional `reducers`/`web`/`all` extras, and CI installation (disposable viewer caches created under `foldlitedb 0.1.x` `FOLDLOG1` can be removed from `~/.cache/hypergan/viewer` to regenerate under `FOLDLOG2`), and ignore `*.publish.attestation` sidecar files when verifying published `.whl` and `.tar.gz` digests on PyPI in `.github/workflows/release.yml`.

## 2.1.0 — 2026-11-20

HyperGAN 2.1.0 upgrades the default training formulation to **ParticleGAN `0.11.1`** (`DEFAULT_RECIPE = "unified_v11"`, Unified 4-Law) with stateful prior-regularizer checkpoints and full multi-GPU replica synchronization. Install with `python -m pip install "hypergan[all]==2.1.0"` after installing the matching Torch wheels for your hardware. [Installation](https://github.com/HyperGAN/HyperGAN/blob/v2.1.0/README.md#install)

- **ParticleGAN `0.11.1` (`unified_v11`) upgrade.** Upgraded the `train` and `all` extras to `particlegan>=0.11.1,<0.12` (universal 1st-order `fwAD` Law 2 Born-Infeld critic penalty with zero `torch.autograd.grad`, rank-agnostic `ema_margin` / `ema_data_scale` / `ema_push_res` and scalar logit relativistic Law 3 coupling, element-budgeted generator secant probe + envelope-tracked `_ema_func_gain` / `peak_rms` Law 4 equilibrated optimizer with analytic `direction_blend`, `UnifiedV11PriorRegularizer`, and upstream `_live_captures` autograd-hook VRAM leak fix). [ParticleGAN integration](https://github.com/HyperGAN/HyperGAN/blob/v2.1.0/docs/particlegan.md)
- **Stateful prior checkpoints & live-endpoint dispatch.** Added stateful `prior_regularizer` checkpoint save/validate/restore across single-device and distributed checkpoints (`checkpoints.py`, `distributed_checkpoints.py`, `_replica_digest`), plus positional single-input HNDL/MLP dispatch in `recipes.py` and `objective_program.py` so ParticleGAN forward hooks reuse live critic endpoints cleanly.
- **Multi-GPU `unified_v11` synchronization.** Added `sync_unified_v11_record`, `_gather_t1c_pair`, element-budgeted generator/critic secant probe synchronization, `_BroadcastProbeProxy` bitwise post-step probe broadcast across CUDA replicas, and optimizer `_ema_func_gain` / `last_step_gain` synchronization in `ReplicatedTrainer` and `aggregation.py`.
- **CI, docs & release automation.** Streamlined `Foundation CI` (sharded CPU runs, wheel reuse for heavy checks, integration-branch scoping, gated `Shared reducer` browser matrix), preserved Git tags on GitHub Release updates, added GitBook preview publishing, and added append-only S3 release archive verification from `master`.

## 2.0.0 — 2026-11-18

HyperGAN 2.0.0 is the first stable release of the TOML and HNDL training runtime. Install with `python -m pip install "hypergan[all]==2.0.0"` after installing the matching Torch wheels for your hardware. [Installation](https://github.com/HyperGAN/HyperGAN/blob/v2.0.0/README.md#install)

- **Training and recipes.** Define models in `hypergan-model/1` TOML, inspect them with the typed Python API, and follow training in the Web Observatory. Since a2, the training runtime has moved to ParticleGAN 0.10.1 with the EMPF default and explicit named presets. The wheel includes two CIFAR-10 models and a 128×128 colorizer. The 256×256 DINOv3 colorization example ships in the source checkout. [CIFAR results by preset](https://github.com/HyperGAN/HyperGAN/blob/v2.0.0/docs/cifar-recipe.md) · [Colorization setup](https://github.com/HyperGAN/HyperGAN/blob/v2.0.0/docs/colorization.md)
- **Distributed checkpoints.** Replicated workers store one copy of the shared training state plus each rank's sampler and RNG state. Sample or evaluate directly from a checkpoint and optionally stage an EMA bundle at evaluation or checkpoint time. Single-host NCCL checkpoint and resume were exercised on two RTX A6000s. Multi-host execution remains experimental. [Recovery](https://github.com/HyperGAN/HyperGAN/blob/v2.0.0/docs/recovery.md)
- **Serving.** Python bundles carry the learned prior and rebuild HNDL networks on load. PT2 accepts the exported batch of one, ONNX exposes a batch dimension, and SafeTensors provides named weights and configuration for a consumer that constructs the graph. Training example pixels are omitted by default. [Serving commands](https://github.com/HyperGAN/HyperGAN/blob/v2.0.0/docs/deployment.md)
- **Upgrading.** Preserve the original alpha environment to resume an existing training run. To move a trained generator, export its bundle there and load it in GA. Start a new run directory to train under GA defaults. [Alpha and 1.x migration](https://github.com/HyperGAN/HyperGAN/blob/v2.0.0/docs/migration.md)

## 2.0.0a4 — 2026-11-17

HyperGAN 2.0.0a4 is the fourth alpha release of HyperGAN 2: declarative `format = "hypergan-model/1"` TOML model files, HNDL (`>=0.11.0,<1`) neural graphs, configurable ParticleGAN (`>=0.10.1,<0.11`) e-means (`empf`) primitives, unified multi-GPU and multi-node training and checkpoints (`schema_version = 2`), a real-time Web Observatory backed by Foldlite DB (`>=0.1.2,<0.2`), standalone privacy-preserving model export across `bundle`, `safetensors`, `pt2`, and `onnx`, and a fully typed config-first Python API (`import hypergan as hg`).

### Added
- **Named ParticleGAN Preset Selection (`defaults = "particlegan:<preset>"`)**:
  - Added support for selecting explicit ParticleGAN presets via `defaults = "particlegan:<preset>"` (e.g., `particlegan:empf`, `particlegan:bispf_optimized`, `particlegan:rhbi_optimized`, including `-s defaults=particlegan:empf` CLI overrides) without overwriting preset learning rates or prior regularization weights, alongside full `_empf_record` binding and multi-GPU e-means prototype/thermal synchronization (`sync_empf_record`).
- **Automatic Standalone Bundle Staging (`checkpoints.export_bundle`)**:
  - Wired `checkpoints.export_bundle = "on_eval" | "on_checkpoint" | "never"` end-to-end across single-GPU and distributed multi-GPU runs, atomically staging `<run_dir>/model.pt` and `<run_dir>/model.sha256` (`.tmp` + `fsync` + atomic rename) and recording `manifest.exported_bundle_path`.
  - `"on_eval"` stages the pinned privacy-clean evaluation snapshot immediately upon completed interval evaluations without importing `torch` on the coordinator or waiting for `checkpoint_every` boundaries, and immediately runs `keep_best` retention pruning.
  - Added a validation warning in `resolve_config()` when `checkpoints.keep_best` tracks an interval evaluation whose `every` cadence is not a multiple of `checkpoint_every`.
- **Conditional & High-Dimensional `pt2` / `onnx` Export**:
  - Added conditional input export support (`_ConditionalExportWrapper` and ONNX dynamic batch axes) to `hg.export(..., format="pt2" | "onnx")` and JSON-safe tensor spec metadata serialization in `safetensors` when `include_example_inputs=True`.
- **256×256 Declarative Colorization & Multidepth DINOv3 Critic HNDL Architectures**:
  - Added `examples/networks/colorization-generator-256.hndl` (6-stage GroupNorm residual U-Net + RoPE/SwiGLU Transformer palette cross-attention + bilinear particle palette synthesis + exact 7-face polytope luminance lock at 256×256), `examples/networks/dinov3-projected-discriminator-256.hndl` (frozen ViT-S/16 `16×16` multidepth projected heads + full-resolution multiscale spatial/color pyramid), and `examples/archive/logos-colorization-dinov3-256.toml`.
- **Automated GitHub Releases in Release CI (`.github/workflows/release.yml`)**:
  - Tagging `v<version>` (or dispatching `release.yml`) now validates `CHANGELOG.md`, runs full `Foundation CI`, publishes distributions to PyPI via Trusted Publishing, and automatically creates/updates the GitHub Release with extracted changelog notes, attached `.whl` / `.tar.gz` assets, and an up-to-date `Latest` pointer on the `2.x` line.

### Changed
- **ParticleGAN `0.10.1` Upgrade (`empf` Default Preset)**:
  - Upgraded the `train` and `all` dependency extras to `particlegan>=0.10.1,<0.11` (`DEFAULT_RECIPE = "empf"`, Exponential Moving Prototype Flow / e-means formulation with `newton_schulz_hybrid_v1` DualNorm updates, retaining explicit `Recipe.from_dict` restoration and `particlegan:<preset>` selection for `rhbi_optimized`, `rhbi`, `bispf_optimized`, and `ka2` runs), with matching recipe settings to the ParticleGAN 0.10.0 EMPF runs that reached `FID-50k = 15.16` on `cifar10-tiny-transformer` (`3.32 steps/s`) and `FID-50k = 51.30` on scratch pure-Transformer `cifar10-transgan` (`0.47 steps/s`).
  - Bound critic transport/thermal step records (`bind_empf_record`, `bind_rhbi_record`, `bind_thermal_source`) in `ReferenceTrainer._initialize`, wired unconditional discrete `ParticlePrior` simplicial manifold regularization (`simplicial_manifold_loss` and `bind_discrete_prior_step`) into `objective_program.py` and `distributed_training.py`, and added global-batch coupled Sinkhorn/RHBI score and capture gathering across multi-GPU ranks so `ReplicatedTrainer` preserves global transport coupling and bitwise replica agreement.
- **Promoted `cifar10-tiny-transformer` to Frozen Pretrained ResNet-18 Feature Critic**:
  - Upgraded the packaged `cifar10-tiny-transformer` recipe (`src/hypergan/models/cifar10-tiny-transformer.toml` and `examples/cifar10-tiny-transformer.toml`) to pair the 64-token 2-block `tiny-transformer-generator-32` generator (`z_dim = 64`, `16,384` MoG particles, `rows = "full"`) with the SHA-256-pinned frozen ImageNet ResNet-18 multi-scale feature critic (`resnet18-features-discriminator-32`) under `defaults = "particlegan"`.
- **`hypergan-model/1` Configuration Readability, Customizability & Schema Alignment**:
  - Synchronized `MODEL_SCHEMA` in `src/hypergan/schema_info.py` (`hypergan schema` / `hg.schema()`) with `src/hypergan/model_file.py` (`data.builtin` including `"cifar10"`, `data.width`/`height`/`mode`/`resize`, `prior.fixed_sigma` as float, `losses.type` choices `["adversarial", "prior", "reconstruction", "objective"]`, `metrics.preset` choices `["standard", "none"]`, removed unsupported `networks.<name>.optimizer`, and added `penalty`, `sampling`, `samplers`, `defaults`, and `aggregation` schema sections).
  - Added support for keyed `[losses.<id_or_critic_or_type>]` overrides in child configurations using `extends`, allowing selective patching or appending of loss entries without repeating the entire `[[losses]]` array.
  - Added `data = "train"` (and `data = { extends = "train", ... }`) shorthand in `[evaluations.<id>]` to explicitly reuse `[data]` with deterministic evaluation defaults (`shuffle = false` / `sampling = "sequential"`, `horizontal_flip = false`).
  - Added support for an optional `[train.optimizer]` subtable inside `[train]` and improved error hints when a top-level `[optimizer]` table is used.
  - Updated `_anchor_relative_paths()` during `extends` resolution to anchor relative path fields (`_PATH_KEYS` and `./` / `../`) to the declaring file's directory even before dataset/weights files are created on disk, while preserving `http(s)://`, `~`, and `${...}` references.
  - Cleaned up internal `components.<net>` binding prefixes and detached `[losses.inputs]` subtables across `hg.lift()`, `recipe_generator.py`, `examples/*.toml`, and `src/hypergan/models/*.toml`.
- **Packaging, Typing & Documentation**:
  - Synchronized `src/hypergan/__init__.py` `if TYPE_CHECKING:` re-exports with 100% of `hypergan.__all__` (`128` symbols) so `py.typed` static type checkers (`mypy`, `pyright`) resolve the full public API without importing `torch`.
  - Updated `README.md` for PyPI rendering (`master` GitHub doc links, `2.0.0a4` installation commands, `cu130` CUDA wheel index for `torch==2.14.0` on Python 3.11–3.14, and `hypergan gc`), `docs/cifar-recipe.md`, `docs/configuration.md`, and `docs/releasing.md`.

### Fixed
- Deferred `torch` imports in `extract_ema_bundle()` until checkpoint tensor loading and removed `torch` from `tests/foundation/test_checkpoints_prune_safety.py` so the Torch-free lightweight foundation suite runs without PyTorch installed.
- Updated `QualificationRunner._run_suite_cpu_replicated_gloo` (`hypergan qualify`) and heavy replicated acceptance tests (`test_replicated_fault_acceptance.py`, `test_metrics_replicated.py`) to verify `schema_version = 2` unified distributed checkpoints (`state.pt` + `ranks/rank-RRRRR.pt`) alongside legacy v1 rank files.
- Resolved HTTP/HTTPS `weights_path` template parameters inside `build_network()` (`src/hypergan/hndl_networks.py`) via `resolve_or_cache()`, enabling native HNDL `pretrained()` nodes (including `src/hypergan/models/cifar10-tiny-transformer.toml` and `examples/image-folder-tiny-transformer.toml`) to verify and cache remote SHA-256-pinned weights in `~/.hypergan/cache/`.
- Prioritized bare packaged model names in `hg.load()`, `hg.describe()`, `hg.diff()`, and `hypergan model` / `validate` / `explain` / `preflight` / `diff` ahead of fuzzy global-registry (`runs.json`) name/prefix matches from historical runs, while preserving local file/directory precedence.
- Cached startup package version strings (`particlegan`, `hypergan`, `hndl`) inside `trainer._checkpoint_provenance` in `_distributed_checkpoint_identity()` (`src/hypergan/distributed_checkpoints.py`) so mid-run editable package metadata updates on disk never drift a running worker's checkpoint identity.
- Raised `MAX_SHARD_BYTES` in `src/hypergan/distributed_commit.py` from `16 MiB` to match `MAX_RANK_BYTES` (`1 GiB`) so high-resolution (`256×256`–`1024×1024`) multi-GPU rank shards (`last_batch` + large `image_folder` / `ColorizationData` epoch permutations) prepare, commit, and restore without hitting a premature shard size limit.
- Fixed `export_generator(..., format="pt2" | "onnx")` and `LoadedGenerator.sample_latents()` latent dimension resolution (`_resolve_z_dim`) to read baked `particle_z.shape[-1]`, `prior.args.z_dim`, `prior.z_dim`, and generator `input_shape` instead of defaulting to `4`, and wrapped unconditional PT2 exports so `torch.export.load(...).module()` unlifts cleanly without colliding with `torch.fx.GraphModule.graph`.
- Fixed `hypergan ps` (`hg.ps()`, `hg.list_runs()`, `hg.status()`) on multi-GPU runs to aggregate live VRAM across descendant `replicated_worker` CUDA processes and report multi-GPU device topologies (`cuda:0,cuda:1`).
- Prevented `pytest-of-*` temporary viewer states and run directories from leaking into the host user's global run registry (`~/.local/share/hypergan/runs.json`) during host discovery.
- Included `tests/conftest.py` and `tests/README.md` in `MANIFEST.in` and `.github/workflows/ci.yml` so source distributions (`sdist`) preserve test isolation fixtures (`HOME`, `XDG_*`, `HYPERGAN_NO_SERVER`).
- Updated `source_info()` provenance metadata in `src/hypergan/training.py` to point `integration_reference.repository` to `https://github.com/HyperGAN/ParticleGAN`.

## 2.0.0a3 — 2026-11-09

HyperGAN 2.0.0a3 is the third alpha of HG 2. It combines declarative `format = "hypergan-model/1"` TOML model files, HNDL (`>=0.11.0,<1`) neural graphs, configurable ParticleGAN (`>=0.9.7,<0.10`) primitives, reproducible training lifecycles, unified multi-GPU and multi-node checkpoints (`schema_version = 2`), a real-time Web Observatory backed by Foldlite DB (`>=0.1.2,<0.2`), standalone privacy-preserving model export, and a fully typed config-first Python API (`import hypergan as hg`).

### Highlights Since 2.0.0a2
- **ParticleGAN `0.9.7` Upgrade**:
  - Upgraded the `train` and `all` dependency extras to `particlegan>=0.9.7,<0.10` (`bispf_optimized` default preset and `generator_gradient_response` host capability guard).
- **Unified Multi-GPU Checkpoints (`schema_version = 2`)**:
  - Consolidated distributed multi-GPU checkpoints into the canonical `<run_dir>/checkpoints/` directory alongside single-GPU checkpoints, storing shared replicated weights, EMA parameters, prior states, optimizer states, and aggregation state once in `state.pt` and lightweight per-rank RNG/stream/sampler state in `ranks/rank-RRRRR.pt` (`~35 KB`), eliminating `N×` redundant model weight storage and wire transfer across ranks.
  - Preserved transparent read-only backward compatibility for legacy `distributed-checkpoints/` (`schema_version = 1`) across `resume`, `sample`, `export`, `evaluate`, `status`, and `checkpoints prune`.
  - Enabled `hypergan evaluate` (`hg.evaluate()`) to directly evaluate distributed training checkpoints without requiring a separate exported bundle, and allowed `hypergan sample <bundle_dir>` (`hg.sample(<bundle_dir>)`) to resolve `<bundle_dir>/model.pt` automatically.
  - Recorded `<run_dir>/model.json` on Rank 0 at startup during replicated multi-GPU training so the Web Observatory renders layer-by-layer network topology for active multi-GPU runs.
- **Distributed & CUDA Runtime Hardening**:
  - Fixed `ReplicatedTrainer._settle` / `_phase` / `_agree` hot-path fast-path barrier handling so forced replica agreement checks (`_assert_replicas`) and `synchronize_replicas()` inside `_folded` blocks are never skipped on `step > 1`.
  - Staged all 9 single-GPU health telemetry scalars in `run_native_program` (`src/hypergan/objective_program.py`) through `trainer._metric_transfer(...)` pinned memory buffers, eliminating 9 blocking pageable device-to-host copies per step.
  - Fixed checkpoint retention symlink vulnerability where premature path resolution in `_checkpoint_roots()` bypassed symlink rejection guards for `distributed-checkpoints/` directories during checkpoint pruning.
- **255labs Brand Identity & Viewer Polish**:
  - Aligned HyperGAN README and Web Observatory visual identity with 255labs (`255labs-lockup.svg`, `255labs-lockup-ink.svg`, `255labs-tokens.css`, `brand.css`).
  - Bundled offline self-hosted `Manrope` and `IBM Plex Mono` `.woff2` fonts (with OFL license texts) in `src/hypergan/web_assets/` for reproducible offline rendering.
  - Aligned run status badge tints, chart color tokens, and strengthened viewer control boundaries across the Index dashboard and per-run Observatory.
- **Repository & Upstream References**:
  - Updated all ParticleGAN upstream repository references and error guidance to `HyperGAN/ParticleGAN`.
  - Standardized the repository changelog at root `CHANGELOG.md`.

### Added
- **Unified `format = "hypergan-model/1"` Configuration Standard & Starter Templates**:
  - Single, human-editable TOML model schema (`[data]`, `[prior]`, `[networks.<name>]`, `[[losses]]`, `[penalty]`, `[train]`, `[evaluations.<id>]`, `[metrics]`, `[sampling]`, `[checkpoints]`, `[run]`).
  - Recursive configuration inheritance via `extends` with relative path anchoring, workspace variable interpolation (`hypergan.toml`), `difflib` field suggestions on unknown keys, and on-the-fly CLI overrides (`-s / --set KEY=VALUE`).
  - Automatic dataset header inspection (`--dataset <path>`) and starter project scaffolding via `hypergan new` across `dcgan`, `tiny-transformer`, `resnet`, `transgan`, `colorization`, `paired-image`, `--conditional`, and `gaussian-grid` templates supporting 32..1024px resolutions with multi-GPU-safe `group_norm` and `rms_norm`.
  - Packaged canonical recipes in `src/hypergan/models/` addressable by bare name (`hypergan train cifar10-transgan`): `gaussian-grid.toml`, `cifar10-tiny-transformer.toml`, `cifar10-transgan.toml`, `image-folder-128.toml`, and `colorization-128.toml`.
- **HNDL Declarative Network DSL (`hndl>=0.11.0,<1`)**:
  - Inline (`source = """..."""`) or standalone `.hndl` network definitions (`hndl = "path/to/net.hndl"`) with static AST and symbolic shape contract verification (`hndl.resolve()`, `input_shape`, `output_shape`) before allocating GPU weights.
  - Native support for residual blocks, self-attention, 2D relative-position bias, one-row learned token position tables, split-latent AdaIN (`adaptive_norm`), equalized linear/conv layers, coordinate warps, Fourier features, differentiable augmentation (`diff_augment`), and SHA-256-pinned frozen pretrained backbones (`torchvision_resnet18`, `dinov3_vits16`) with symbolic `shape_fn` hooks.
  - Multi-network and auxiliary graph routing (`role = "generator" | "critic" | "encoder" | "auxiliary"`) with weight sharing (`reuse`, `freeze_parameters`), multi-critic adversarial terms, particle-routing autoencoders (`ae_gan`, `vae_gan`), paired conditional reconstruction (`d-then-g-v1`), and [routed particle LoRA host integration](docs/particle-sliders.md).
  - Resolved HNDL sources embedded into run manifests and version-2 checkpoints (`CURRENT_VERSION = 2`) for self-contained provenance.
- **ParticleGAN (`particlegan>=0.9.7,<0.10`) Integration & Core Engine**:
  - Native integration with ParticleGAN's released `bispf_optimized` preset (`defaults = "particlegan"`), DualNorm optimizer family, SBNP prior regularization, and explicit `Recipe.from_dict` snapshot restoration (with backwards-compatible fallback for 0.9.4 checkpoints).
  - Python 3.11–3.14 support with modern dependency bounds (`torch>=2.6,<3`, `safetensors>=0.8,<1`, `Pillow>=13.0.0,<14`, `torch-fidelity>=0.4.0,<0.5`) and optional extras (`[train]`, `[image]`, `[cifar]`, `[fid]`, `[reducers]`, `[web]`, `[safetensors]`, `[onnx]`, `[all]`).
  - On-demand external URL and dataset cache (`~/.hypergan/cache/`) with SHA-256 integrity verification, multi-rank file locking (`fcntl.flock`), automatic ResNet-18 (`f37072fd`) / Inception-v3 (`6726825d`) backbone caching, opt-in CIFAR-10 dataset preparation, and `HYPERGAN_OFFLINE=1` support.
  - `ImageFolder` RGBA and palette alpha compositing (`alpha = "composite_white" | "composite_black" | "error"`) and opt-in symlink traversal (`follow_symlinks = true`).
- **Distributed Multi-GPU & Multi-Node Training**:
  - First-class multi-GPU and multi-node training (`gradient_allreduce`, `parameter_average`, `pseudo_gradient`) with zero-TOML CLI flags (`--devices`, `--nodes`, `--node-rank`, `--rendezvous`, `--strategy`, `--sync-every`) and Slurm / `torchrun` environment auto-detection.
  - Elastic cross-topology resume (`1 <-> N` and `M <-> N` GPUs) across `checkpoints/` and `distributed-checkpoints/`, plus startup provenance caching on workers so mid-run `git` commits do not disrupt periodic checkpoints.
  - NFS-free distributed checkpoints coordinated by Rank 0 over collective scatter/gather, with coordinated post-step phase failure handling (optimizer, backward, EMA, and evaluation snapshot failures) so all ranks reject partial checkpoints and restore rank-owned CUDA device contexts.
  - Memory-contiguous parameter gradient bucketing in `_reduce_gradients`, elimination of per-step Python `all_gather_object` barriers after step 1, and documented **2.20× scaling speedup** on dual NVIDIA RTX A6000 at `batch_size = 256` ([Scaling Benchmark](docs/scaling-benchmark.md)).
  - Hardware qualification suite (`hypergan qualify`) and AMD ROCm / HIP runtime detection and preflight support ([ROCm Readiness](docs/rocm.md)).
- **Web Observatory (`hypergan view`) & Foldlite DB Storage**:
  - Backed by `foldlitedb>=0.1.2,<0.2` for atomic append-only event journaling, durable `.fold` view caches, and shared Rust/WebAssembly streaming reducers across server and browser with pinned headline losses (`loss/g_total`, `loss/d_total`).
  - Overhauled Global Runs Dashboard (`GET /`) and Per-Run Observatory (`GET /runs/<id_or_slug_or_tag>`) featuring the 255labs design system, zero-blink live DOM polling, clickable run rows, hardware topology and batch columns (`Batch 256 (128 / GPU)`), active throughput (`steps/s` and `samples/s`), live `ETA`, 5×2 summary stat grid, inline **Model** architecture section, 4×2 default chart grid with quick presets and URL-persisted state, dismissible health/contention alerts, 2-click confirm `Stop` control with live receipt tracking, and a side-by-side **Samples** comparison workbench with synced step scrubbing, Crisp/Smooth rendering toggle, and zoom lightbox.
  - Loopback default listener (`127.0.0.1:8765`), `-b / --bind` network binding, automatic Tailscale MagicDNS host allowlisting via local `tailscaled.sock`, and TTY-gated autostart respecting `HYPERGAN_NO_SERVER=1` and `CI=1`.
- **Docker-Style CLI, AI Agent Ergonomics & Telemetry**:
  - Daemonless global run registry addressing runs from any directory by Docker-style slug (`fleet-hawk`), short ID, custom tag (`hypergan tag`), or path across `ps`, `status`, `logs`, `checkpoint`, `stop`, `resume`, `sample`, `eval`, `export`, `runs`, `queue`, `prune`, `gc`, and `rm`.
  - Universal `--json` output with structured error envelopes, global `--quiet` / `-q` flag, `hypergan schema` (`hg.schema()`), `hypergan ops` (`hg.operators()`), machine-readable root `llms.txt`, and complete [HNDL Reference](docs/hndl-reference.md).
  - Coalesced 10-scalar GAN health telemetry emitted every 10 steps across single-GPU and multi-GPU runs (`critic/real_mean`, `critic/fake_mean`, `critic/margin`, `critic/accuracy`, `grad_norm/critic`, `grad_norm/generator`, `grad_norm/prior`, `optimizer/d_guard_ratio`, `prior/particle_rms`, and batch `diversity/ratio`), plus rolling EMA metrics and automated `health_alerts` in `summary.json` and `hg.status()`.
- **Standalone Production Export & Typed Python API (`import hypergan as hg`)**:
  - Privacy-preserving standalone export stripping raw `last_batch` training tensors by default (`--include-example-inputs` to opt in) and baking learned particle tables (`prior.z`, `prior.sigma`) into generator modules across `bundle`, `safetensors`, `pt2`, and `onnx` formats with zero ParticleGAN/HNDL runtime dependencies.
  - Callable `LoadedGenerator` (`nn.Module`) via `hg.load_bundle(..., device="cuda")` with `.sample(...)` and `.interpolate(..., mode="slerp")`, plus GPU-accelerated CLI batch sampling (`hypergan sample --device cuda --output-dir <dir>`).
  - Lazy-loaded, Torch-free top-level Python API with PEP 561 `src/hypergan/py.typed` marker, typed dataclasses (`RunStatus`, `PsSnapshot`, `RunInfo`, `ModelDescription`, `SampleResult`, `TrainResult`, `ValidationWarning`, `LoadedGenerator`), non-blocking `hg.launch()`, and `on_event` early stopping via `hg.StopTraining`.

### Changed
- Standardized global training step reporting across single-GPU and multi-GPU runs so one complete global discriminator/generator update counts as one step across progress bars, throughput, checkpoints, preview sliders, FID, and metric charts.
- Promoted distributed multi-GPU and multi-node execution to first-class CLI and API flags without `--experimental`.
- Unified run metadata reading through `RunInfo` and dispatched `compare_runs` `min`/`max` ordering from the metric catalog's declared `direction`.
- Sanitized `examples/` to canonical runnable recipes with `builtin = "image_folder"`, archiving historical research ablations under `examples/archive/` and `reports/archive/`.

### Fixed
- Placed `ema_critic` in `.eval()` mode during anchor evaluations so stochastic `DiffAugment` or dropout cannot perturb critic anchors, and wired live post-guard / pre-guard gradient scaling into `optimizer/d_guard_ratio`.
- Prioritized packaged recipe resolution in `hypergan train <recipe>` before matching older local run directories sharing the same experiment name.
- Fixed InceptionFID eigenvalue computation on SciPy 1.18+ via `torch-fidelity>=0.4.0`.
- Fixed `hypergan queue run` signal forwarding, atomic job claiming, and cross-platform process identity checks on Linux, macOS, and Windows.
- Fixed `checkpoints prune` locking, branch-aware retention of abandoned attempts, and reclaimed byte accounting.

### Removed
- Removed legacy 1.x TensorFlow/PyTorch 1.x catalog components in favor of the `hypergan-model/1` + HNDL + ParticleGAN architecture (see [migration notes](docs/migration.md)).
- Removed the unimplemented `src/hypergan/runpod_runner.py` stub and broken scratchpad configs from `examples/`.

## 2.0.0a2 — 2026-11-05

HyperGAN 2.0.0a2 is the second alpha of HG 2. It combines declarative `hypergan-model/1` TOML model files, HNDL (`>=0.11.0,<1`) networks, ParticleGAN (`>=0.9.5,<0.10`) primitives, distributed training, a typed Python API, the Web Observatory, and standalone model export.

### Release updates

- Coordinate replicated phase failures after the first update, including optimizer, backward, EMA, and evaluation snapshot failures, so every rank rejects partial checkpoints.
- Restore the rank-owned CUDA device after custom modules change the ambient device, including after the initial handshake.

- Refresh the Web Observatory with live run polling, hardware topology and batch details, health alerts, chart presets, and side-by-side sample comparison.
- Repair installed-package CI dependencies, test imports, and Windows lifecycle checks.

- Report one complete global discriminator/generator update as one training
  step throughout the viewer. Multi-GPU progress, throughput and saved-state
  counters now use the same steps as preview cadence, FID and metric charts,
  rather than multiplying them by the number of ranks.

- Require the published HNDL 0.11.0 release in the `train` and `all` extras,
  and use it in the Modal smoke and scaling runtimes.

- Require the published ParticleGAN 0.9.5 release in the `train` and `all` extras.

- Fresh `cifar10-transgan` runs inherit ParticleGAN's released `bispf_optimized`
  preset, including its optimizer, rates and prior regularization. The packaged
  model previously omitted `defaults = "particlegan"` and silently used KA2;
  the Round 1 CIFAR QA runs therefore did not evaluate BISPF. Existing saved
  runs keep their recorded formulation. Represent the same learned CLS vector
  as a one-row HNDL position table so the released DualNorm optimizer accepts
  its parameter shape.

- Restore saved ParticleGAN recipe fields through `Recipe.from_dict` when available,
  retaining the constructor fallback for ParticleGAN 0.9.4. This keeps compact
  snapshots, legacy host configurations and strict prior restoration compatible
  with ParticleGAN 0.9.5 without changing their saved training settings.

### Added
- **Core Engine & Dependency Modernization**:
  - Python 3.11–3.14 support with updated dependencies: `particlegan>=0.9.5,<0.10`, `torch-fidelity>=0.4.0,<0.5`, `safetensors>=0.8,<1`, `torch>=2.6,<3`, and `Pillow>=13.0.0,<14`.
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
- Expose distributed multi-GPU and multi-node training directly without `--experimental`.
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
* New loss based on least squared GAN.  See [least squares GAN implementation](CHANGELOG.md#Least-Squares-GAN).
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


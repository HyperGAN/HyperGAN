# HyperGAN Whitepaper Reproducible Benchmark Suite

This directory contains **8 self-contained, pure declarative `hypergan-model/1` + native HNDL** benchmark configurations designed to compare **ParticleGAN `unified_v12`** (`defaults = "particlegan"`, **zero recipe or optimizer overrides**) head-to-head against an **honest, well-tuned Adam + R1 baseline** across 4 distinct architecture/dataset pairs (no seed sweeps).

All generators and discriminators are pure declarative HNDL graphs (`factory = "hndl"`) with **no custom PyTorch `nn.Module` shortcuts, no pretrained backbones, no ADA, and no spectral normalization**.

---

## Benchmark Matrix

| # | Config | Dataset | Resolution | Architecture | Formulation | Generator Params | Critic Params | Total Params |
|:-:|:---|:---|:---:|:---|:---|---:|---:|---:|
| 1 | [`cifar10-dcgan-bn-v12.toml`](cifar10-dcgan-bn-v12.toml) | CIFAR-10 | 32×32 | Radford (2015) BN-DCGAN | ParticleGAN `unified_v12` (zero overrides) | 1,183,619 | 663,361 | 1,846,980 |
| 2 | [`cifar10-dcgan-bn-adam.toml`](cifar10-dcgan-bn-adam.toml) | CIFAR-10 | 32×32 | Radford (2015) BN-DCGAN | Non-saturating + R1 (`γ=1.0`) + Adam (`2e-4`, `β=(0.5, 0.999)`) | 1,183,619 | 663,361 | 1,846,980 |
| 3 | [`cifar10-resnet-scratch-v12.toml`](cifar10-resnet-scratch-v12.toml) | CIFAR-10 | 32×32 | Scratch ResNet-GAN (GN+SiLU G / unnorm SiLU+AvgPool D) | ParticleGAN `unified_v12` (zero overrides) | 6,028,675 | 7,428,993 | 13,457,668 |
| 4 | [`cifar10-resnet-scratch-adam.toml`](cifar10-resnet-scratch-adam.toml) | CIFAR-10 | 32×32 | Scratch ResNet-GAN (identical G & D) | Non-saturating + R1 (`γ=1.0`) + Adam (`2e-4`, `β=(0.0, 0.999)`) | 6,028,675 | 7,428,993 | 13,457,668 |
| 5 | [`cifar10-transgan-scratch-v12.toml`](cifar10-transgan-scratch-v12.toml) | CIFAR-10 | 32×32 | Pure Transformer TransGAN (G & D) | ParticleGAN `unified_v12` (zero overrides) | 2,836,747 | 10,200,025 | 13,036,772 |
| 6 | [`cifar10-transgan-scratch-adam.toml`](cifar10-transgan-scratch-adam.toml) | CIFAR-10 | 32×32 | Pure Transformer TransGAN (identical G & D) | Non-saturating + R1 (`γ=1.0`) + Adam (`2e-4`, `β=(0.0, 0.999)`) | 2,836,747 | 10,200,025 | 13,036,772 |
| 7 | [`ffhq64-resnet-v12.toml`](ffhq64-resnet-v12.toml) | FFHQ-64 | 64×64 | Scratch ResNet-GAN 64px (GN+SiLU G / unnorm SiLU+AvgPool D) | ParticleGAN `unified_v12` (zero overrides) | 6,220,227 | 7,841,345 | 14,061,572 |
| 8 | [`ffhq64-resnet-adam.toml`](ffhq64-resnet-adam.toml) | FFHQ-64 | 64×64 | Scratch ResNet-GAN 64px (identical G & D) | Non-saturating + R1 (`γ=1.0`) + Adam (`2e-4`, `β=(0.0, 0.999)`) | 6,220,227 | 7,841,345 | 14,061,572 |

All 8 configurations share `batch_size = 256`, `steps = 100000` (with interval evaluations every `10,000` steps and checkpoints every `5,000` steps for preemption resilience), `seed = 0`, and `device = "cuda"` targeting single-GPU Google Colab A100 / L4 runtimes.

---

## Fairness & Formulation Contract

Every `*-v12.toml` and `*-adam.toml` pair uses **byte-identical Generator and Discriminator HNDL source graphs**, identical initialization policy, identical data pipeline, identical batch size, and identical evaluation protocol:

- **ParticleGAN `unified_v12` (`*-v12.toml`):**
  - Declares `defaults = "particlegan"` with **zero `[recipe]`, `[penalty]`, or optimizer overrides**.
  - Uses ParticleGAN's native learnable particle prior (`kind = "particles"`, `num_particles = 16384`) and `[[losses]] type = "prior"`.
- **Honest Adam + R1 Baseline (`*-adam.toml`):**
  - Declares `defaults = "hypergan"` with explicit pure PyTorch Adam (`optimizer_family = "adam"`, `implementation = "torch_fused_adam"`), standard Goodfellow/Mescheder/StyleGAN2 non-saturating logistic loss (`loss = "non_saturating"`), pure Mescheder et al. (2018) R1 gradient penalty (`reg_arm = "a_r1r2"`, `reg_real_weight = 1.0`, `reg_fake_weight = 0.0`, `penalty = 1.0`, `lazy_k = 1`), isotropic Gaussian prior (`kind = "gaussian"`), and inference EMA (`ema = 0.999`).
  - Explicitly disables all ParticleGAN-specific interventions (`anchor_weight = 0.0`, `d_guard_ratio = 0.0`, `latent_damping_max_rate = 0.0`, `direct_particle_gain = false`, `input_noise_std = 0.0`, `output_noise_std = 0.0`).
  - Uses textbook literature hyperparameters (`lr = 0.0002`, `d_lr_mult = 1.0`, constant LR; `betas = [0.5, 0.999]` for Radford BN-DCGAN and `betas = [0.0, 0.999]` for ResNet-GAN and TransGAN).

---

## Dataset Preparation

### 1. CIFAR-10 (32×32, 50,000 train images)

```bash
hypergan download-data cifar10 --output ~/.hypergan/cache/cifar10
```

### 2. FFHQ-64 (64×64, 70,000 RGB faces — Primary 64px Benchmark)

FFHQ-64 is chosen as the primary 64×64 visual benchmark because it provides **70,000 aligned RGB faces** (supporting full unbiased `FID-50k` and `FD_DINOv2-50k` without wrapping and avoiding small-dataset discriminator memorization without ADA) via a fast, zero-auth-wall, SHA-256-pinned HuggingFace CDN archive (`877.7 MB`, SHA-256 `76af01fc2b10b6631e08713365c7ae2a9d8ee99a7a62ac834ade684a913094ff`):

```bash
python scripts/prepare_benchmark_dataset.py --dataset ffhq64
```

This writes verified 64×64 RGB PNGs and `dataset-manifest.json` to `~/.hypergan/cache/datasets/ffhq64`.

An optional compact 64×64 fallback (**Oxford Flowers-102**, 8,189 images from Oxford VGG, `344.9 MB`, SHA-256 `2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246`, center-cropped and Lanczos-resized to 64×64 RGB PNGs) is also supported by the same script:

```bash
python scripts/prepare_benchmark_dataset.py --dataset flowers102
```

---

## Validation, Smoke Testing & Evaluation

Validate all configs structurally (CPU-only, Torch-free):

```bash
CUDA_VISIBLE_DEVICES="" hypergan validate benchmarks/whitepaper/cifar10-resnet-scratch-v12.toml
```

Run a 2-step CPU smoke test (zero GPU usage):

```bash
CUDA_VISIBLE_DEVICES="" hypergan train benchmarks/whitepaper/cifar10-resnet-scratch-v12.toml \
  --run-dir /tmp/smoke-resnet-v12 -d cpu -s train.batch_size=4 --stop-after-steps 2 --no-previews --no-server
```

### Publication Metric Suites (`InceptionSuite` & `DINOv2Suite`)

Each config runs `evaluations.fid50k_train` (`hypergan.image_metrics:InceptionFID`) every `10,000` steps, exports an EMA bundle (`model.pt`) on evaluation (`export_bundle = "on_eval"`), and declares live manual snapshot evaluators for both `evaluations.inception_suite` (`hypergan.image_metrics:InceptionSuite` -> `fid`, `precision`, `recall`, `density`, `coverage`) and `evaluations.dinov2_suite` (`hypergan.image_metrics:DINOv2Suite` -> `fd_dinov2`, `precision`, `recall`, `density`, `coverage`, pinned DINOv2 ViT-L/14 SHA-256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`):

```bash
hypergan eval <run_dir> --metric inception_suite
hypergan eval <run_dir> --metric dinov2_suite
```

Exported `model.pt` EMA bundles can also be evaluated post-hoc via `python -m hypergan.image_metrics <model.pt>`.

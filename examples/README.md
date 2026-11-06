# HyperGAN Examples

This directory provides canonical, minimal, verified recipes and scripts for training models with HyperGAN 2.0.

## Active Production Examples

| File | Description | Requirements | Typical Device |
|:---|:---|:---|:---:|
| [`gaussian-grid.toml`](gaussian-grid.toml) | 2D 100-mode Gaussian grid reference model | None (synthetic) | CPU (<1s) |
| [`paired-linear.toml`](paired-linear.toml) | 2D paired linear regression / conditional generation | None (synthetic) | CPU (<1s) |
| [`color-words.toml`](color-words.toml) | 1D discrete color-word vocabulary with multi-critic adversarial terms | None (synthetic) | CPU (<2s) |
| [`image-folder-dcgan.toml`](image-folder-dcgan.toml) | 128×128 DCGAN for custom image datasets with GroupNorm and alpha compositing | Local image folder | CPU / GPU |
| [`image-folder-tiny-transformer.toml`](image-folder-tiny-transformer.toml) | 128×128 Tiny Transformer generator with multi-scale ResNet-18 critic | Local image folder | GPU |
| [`image-folder-colorization.toml`](image-folder-colorization.toml) | 128×128 Conditional Colorizer (L → AB) with exact polytope luminance lock | Local image folder | GPU |
| [`cifar10-tiny-transformer.toml`](cifar10-tiny-transformer.toml) | 32×32 CIFAR-10 Tiny Transformer generator with convolutional critic | `hypergan download-data cifar10` | GPU |
| [`cifar10-transgan.toml`](cifar10-transgan.toml) | 32×32 CIFAR-10 Pure Transformer TransGAN (winning benchmark recipe) | `hypergan download-data cifar10` | GPU |

## Directory Structure

- `networks/`: Modular, reusable HNDL architecture definitions (generators and critics).
- `execution/`: Execution profiles (`cpu-single.toml`, `cpu-replicated.toml`, `cuda-replicated-nccl.toml`).
- `api/config_first/`: Standalone Python API tutorials and execution demos.
- `archive/`: Historical research ablations and private dataset recipes maintained for numerical reproducibility.

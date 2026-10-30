# HyperGAN Examples Archive

This directory contains historical research ablations, development scratchpads, and private dataset experiment configurations from the HyperGAN 2.0 research phase.

These files are preserved to maintain numerical reproducibility and backwards compatibility with earlier experiment checkpoints and test assertions.

## Historical Experiment Groups

- **Logos & Private Dataset Experiments**: `logos-*.toml`, `dcgan-*.toml` (referenced private `logos256` dataset and manifests; superseded by `examples/image-folder-*.toml` using `builtin = "image_folder"`).
- **DINOv3 Critic Ablations**: `transgan-dinov3-*.toml`, `sagan-adain-dinov3-*.toml` (explored DINOv3 frozen visual foundation models).
- **TransGAN Architectural Variants**: `transgan-resnet-multiscale-128-e2.toml`, `-e3.toml`, `-equalized.toml`, `-stable.toml` (historical generator depth and learning-rate ablations).
- **CIFAR Historical Scratchpads**: `cifar-pretrained-sagan.toml`, `cifar-tiny-transformer-*.toml` (historical exploratory runs; superseded by canonical `examples/cifar10-*.toml` and packaged recipes in `src/hypergan/models/`).

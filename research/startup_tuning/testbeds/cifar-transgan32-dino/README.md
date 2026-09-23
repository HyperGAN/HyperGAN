# Adversarial-only CIFAR TransGAN, DINOv3 critic

Same 32px TransGAN generator, prior, optimizer, data, and FID schedule as the
adversarial ResNet run. The critic is a new 32px pixel-plus-four-depth DINOv3
discriminator. ViT-S/16 sees a native 2×2 patch grid. There is no encoder and
no reconstruction loss. Tuning is off.

It has not been started. The launcher uses GPU 1. The fresh run directory is
`~/dev/hypergan/training-runs/train-cifar-transgan-32-dinov3`.

```bash
~/dev/hypergan/training-runs/start-cifar-transgan-32-dinov3.sh
```

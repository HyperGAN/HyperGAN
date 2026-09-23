# Adversarial-only CIFAR TransGAN

Same 32px TransGAN generator, ResNet18 critic, fixed prior, optimizer, data,
and FID schedule as the run whose best `fid50k_train` was 10.93 at step
50,000. The encoder and the reconstruction loss are removed. Tuning is off.

It has not been started. The launcher uses GPU 1. The fresh run directory is
`~/dev/hypergan/training-runs/train-cifar-transgan-32-adversarial`. It does
not resume `train-cifar-transgan-32`.

```bash
~/dev/hypergan/training-runs/start-cifar-transgan-32-adversarial.sh
```

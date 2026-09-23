# Adversarial-only CIFAR TransGAN

Same 32px TransGAN generator, ResNet18 critic, fixed-sigma learned prior, optimizer, data,
and FID schedule as the run whose best `fid50k_train` was 10.93 at step
50,000. The encoder and the reconstruction loss are removed. Tuning is off.

The earlier training run was stopped at step 1825 after remaining unsaturated
with useful diversity through step 1700. FID 10.93 belongs to the original
encoder/reconstruction recipe, not this adversarial-only run. A diagnostic
replay now records stage measurements in `results/2026-09-22-healthy-control/`.

The historical launcher uses GPU 1, which is reserved; do not start it there
without checking current authorization. The run directory is
`~/dev/hypergan/training-runs/train-cifar-transgan-32-adversarial`. It does
not resume `train-cifar-transgan-32`.

```bash
~/dev/hypergan/training-runs/start-cifar-transgan-32-adversarial.sh
```

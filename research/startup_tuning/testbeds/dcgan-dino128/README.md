# DCGAN generator, DINOv3 discriminator

This pairs the plain 128px DCGAN generator with the same frozen four-depth
DINOv3 discriminator used by the TransGAN runs. The logos manifest, batch,
seeds, logistic relativistic loss, lazy b-cap penalty, prior, and Adam rates
are unchanged. Tuning is off.

It has not been started. GPU 1 is the device in the launcher, and the plain
DCGAN formulation run may still be using it. The fresh run directory is
`~/dev/hypergan/training-runs/train-dcgan-dinov3-multidepth-128`.

```bash
~/dev/hypergan/training-runs/start-dcgan-dinov3-multidepth-128.sh
```

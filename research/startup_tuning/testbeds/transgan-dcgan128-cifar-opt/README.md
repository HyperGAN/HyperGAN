# TransGAN generator, DCGAN discriminator, CIFAR player optimizer

Same adversarial recipe as the collapsed TransGAN/DCGAN run. The only change
is the player optimizer: generator rate 3e-4, discriminator rate 4.5e-4
(`d_lr_mult = 1.5`), and betas `[0.0, 0.999]`. Prior betas stay `[0.5, 0.999]`.
There is no encoder and no reconstruction loss.

It has not been started. The fresh run directory is
`~/dev/hypergan/training-runs/train-transgan-dcgan-128-cifar-opt`. It does not
resume `train-transgan-dcgan-128`.

```bash
~/dev/hypergan/training-runs/start-transgan-dcgan-128-cifar-opt.sh
```

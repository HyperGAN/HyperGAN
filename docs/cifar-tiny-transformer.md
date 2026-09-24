# CIFAR tiny transformer with a feature-only critic

The [128px logo adaptation](logos-tiny-transformer-features.md) embeds both
networks in one config and retains this recipe's latent, prior, and optimizer.

The [recipe](../examples/cifar-tiny-transformer-resnet-features.toml) pairs the
simplified transformer with a discriminator that uses only pretrained ResNet18
features. This tests whether the pair can learn CIFAR at 32px before another
128px experiment. It changes both networks, so it does not isolate the effect
of removing the pixel branch from the previous TransGAN run.

## Networks

The [generator](../examples/networks/tiny-transformer-generator-32.hndl) adapts
the tiny 128px generator to CIFAR's existing 64-dimensional latent. One linear
projection produces an 8x8 grid of 256-dimensional tokens. Two pre-LayerNorm
transformer blocks have four heads and 512-wide GELU FFNs. Nearest-neighbor
upsampling and 3x3 convolutions produce 128 channels at 16px and 64 at 32px,
followed by a 3x3 RGB convolution and tanh. It has 2,489,731 parameters.

The [discriminator](../examples/networks/resnet18-features-discriminator-32.hndl)
retains CIFAR's bilinear 32-to-64px resize, ImageNet normalization, frozen
ResNet18 layer1/2/3 features, and all three original feature heads. Each head
concatenates candidate and fixed midgray-context features, applies a 1x1
projection, GroupNorm, LeakyReLU, a 3x3 convolution, LeakyReLU, 4x4 average
pooling, and a scalar linear readout. There are 171,779 trainable parameters.
The full checkpoint registers 11,689,512 frozen parameters; only the backbone
through layer3 participates in these readouts.

The entire pixel branch, including its residual blocks and attention, is
removed. The final score is `(head1 + head2 + head3) / sqrt(3)`. The old outer
division by sqrt(2), which combined the pixel and feature branches, is removed
with the pixel branch. ResNet weights and BatchNorm statistics remain frozen,
but gradients with respect to images pass through it to the generator.

## Training and evaluation

This is adversarial-only, matching the existing CIFAR adversarial ablation:
no routing encoder or reconstruction objective. It retains the CIFAR recipe's
16,384-particle prior, fixed sigma 0.212616428732872, logistic relativistic
paired loss, b_cap penalty (coefficient 1, kappa 1, every 8 steps), and prior
regularizer. It preserves batch 64, seeds 24002/24003, EMA 0.995, random CIFAR
sampling with horizontal flips, and the 200,000-step schedule. No DiffAug.

Effective learning rates are G=0.0003, D=0.00045, prior=0.003. G/D Adam betas
remain (0.0, 0.999), with prior betas (0.5, 0.999). These are the CIFAR settings,
not the preceding 128px settings. The existing manual FID smoke protocol and
scheduled 50K-sample training-set FID every 10,000 steps are retained. Both
use local hash-pinned Inception weights; no downloads occur.

Four focused CPU checks passed, covering both tiny generator resolutions,
recipe preservation, discriminator preprocessing/score combination, frozen
backbone state, input gradients, and an active second-derivative penalty.
A separate batch-64 CUDA smoke check with real CIFAR images and pinned ResNet
weights passed a D update and a G update with finite gradients. The check
forced kappa=0 to exercise an active penalty; the recipe retains kappa=1.
Peak allocated memory was 1.262 GiB for that check, excluding prior, EMA, and
evaluation. This does not establish convergence or training quality.

## Local launch

```bash
../training-runs/start-cifar-tiny-transformer-resnet-features.sh
```

The launcher uses physical GPU 0 and the separate run directory
`/mnt/ml7tb/hypergan-training-runs/train-cifar-tiny-transformer-resnet-features`.
Previews are every 100 steps and checkpoints every 1000. It uses the existing
local CIFAR dataset and pinned ResNet/Inception checkpoints. Training is left
for the user to launch. This is a fresh architecture, not a checkpoint resume.

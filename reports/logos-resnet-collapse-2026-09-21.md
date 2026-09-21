# Logos ResNet discriminator collapse investigation

Read-only investigation of `~/dev/hypergan/training-runs/start-dcgan-128.sh`
and `start-dcgan-resnet-128.sh`, 2026-09-21. No training, environment, launcher,
or recipe changes were made. No seed experiments were run.

## Is the discriminator the only difference?

Yes, in the saved resolved training configurations: only the discriminator
source/weight parameters and run name differ. Generator, data manifest,
batch size 64, seeds, prior, losses, b-cap settings, optimizers, EMA, sampling,
and metrics are identical. Launchers differ in config/output paths and physical
GPU selection (both RTX A6000). Both running processes record HNDL 0.2.0 and the
same dependency versions. The source commits differ (34136fe vs 2075114), both
record dirty working trees; the intervening committed runtime change registers
the native pretrained provider. Exact in-memory source identity is not archived.

At matched step 2000, checkpoint data/prior/penalty RNG streams are byte-identical,
as is the last real batch. Both data cursors are 256000. This is a much stronger
comparison than comparing their latest dashboards at different training steps.

## Collapse measurements

CPU inference from both saved step-2000 checkpoints, reusing the first 64 saved
latent particle means without new sampling or optimization:

| Mean per-pixel standard deviation across generated samples, range [-1,1] | DCGAN | Frozen ResNet |
|---|---:|---:|
| Live generator, eval | 0.727014 | 0.014790 |
| Live generator, train | 0.740404 | 0.014725 |
| EMA generator, eval | 0.641923 | 0.011212 |
| EMA generator, train | 0.779540 | 0.012751 |

These probes use particle means rather than the full noisy MoG distribution.
Independent saved preview evidence agrees: the 16-sample EMA grids at step 2000
have mean pixel standard deviations 0.702092 versus 0.016702 (range [-1,1]).
Both previews select the same 16 distinct particle IDs. Collapse affects the live
model as well as EMA and persists in generator train mode.

Across saved previews, pairwise RMS after averaging each image down to 4x4,
normalized by the corresponding real-grid RMS, is:

| Step | DCGAN | Frozen ResNet |
|---|---:|---:|
| 500 | 29.0% | 1.53% |
| 1000 | 109.7% | 1.19% |
| 2000 | 107.4% | 0.79% |
| 2400 | 122.4% | 0.63% |

This is a coarse diversity diagnostic, not a quality score. At step 2400 the
ResNet samples have high adjacent-pixel variation (0.280 on [0,1], versus 0.0406
for real images), despite low variation between samples: repeated texture can
look numerically varied within each image while having little sample diversity.

## Freezing and gradients

At step 2000, all 102 tensors in the original pinned ResNet checkpoint remain
bitwise identical, including BatchNorm running statistics. The backbone is in
eval mode and its parameters are nontrainable. The discriminator optimizer has
only four head tensors, totaling 17,473 trainable parameters; the plain DCGAN
critic has 6,959,553.

A CPU probe of 16 saved particle means gives nonzero discriminator input-gradient
L2 norms: 0.0822 on ResNet-generated samples and 0.4191 on its saved real batch.
Thus freezing has not severed the path back to G. On this probe, its real/fake
mean scores are 0.930/0.826, with fake-score standard deviation 0.00728. These
small probe batches characterize this checkpoint, not a dataset-wide evaluation.

Sampled median adversarial losses at steps 1500–2000 are D/G 0.0864/7.6812 for
plain DCGAN and 0.6573/0.7694 for frozen ResNet. The latter are near the logistic
indistinguishability baseline, log(2). Finite, balanced-looking losses and a healthy
latent prior do not certify image quality or generator use of that prior.

## Why the CIFAR result does not validate this particular discriminator

The [CIFAR reference](image-training-plan-2026-09-19.md) records FID50k 12.5344567
at 200k steps. Its frozen ResNet is part of a substantially different critic:

| Architecture | CIFAR reference | Logos ResNet variant |
|---|---|---|
| Trainable pixel branch | Residual network with SAGAN attention | None |
| Frozen feature stages | layer1, layer2, layer3 | layer3 only |
| Feature heads | Three: 1x1 conv, GroupNorm, 3x3 conv, pool, linear | One: 1x1 conv, pool, linear |
| Trainable parameters | 805,540 | 17,473 |

CIFAR combines pixel and feature scores as
`(pixel + (head1 + head2 + head3)/sqrt(3))/sqrt(2)`.
Its feature heads additionally receive cached features of a fixed zero-image
context. It also has different generator architecture, optimization settings,
prior size, dataset and resolution. Both backbones are frozen.

The concrete architectural suspects are the missing pixel branch, missing
shallow feature heads, and limited head capacity. The evidence does not isolate
which causes collapse or prove that frozen ImageNet features cannot work on logos.
A useful next controlled change is adding a trainable pixel branch to the existing
HNDL discriminator while keeping the current generator/data/training settings.
A separate architecture comparison can reproduce the full CIFAR pixel-plus-three-
feature-head design at 128px. Track cross-sample diversity at coarse and full
resolution alongside losses, and evaluate quality with a fixed held-out protocol.
Neither experiment was started by this investigation.

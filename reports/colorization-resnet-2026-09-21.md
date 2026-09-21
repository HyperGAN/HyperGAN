# ResNet18 colorization discriminator trial

The owner DINO + RGB run stopped at step 3,428 after renewed collapse. The
previous 1,500-step experiment established temporary spread, not lasting
stability or useful colorization. This trial substitutes the working CIFAR
ResNet18 discriminator, adapted to 256px, while retaining the generator,
encoder, particle prior, losses, b-cap, optimizer and learning rates.

## Failure evidence

The saved step-3,400 previews contain eight EMA samples. All conditional
samples select particle 3550. In the random preview, 97.93% of pixels are within
2/255 per channel of magenta or green, versus 0.0668% in its real-image grid.
About 98.65% of random RGB channel values are saturated above absolute 0.98.
These are small, quantized preview diagnostics, not full-dataset estimates.

The 512-sample random chroma distance worsened from 0.03723 at step 2,200 to
0.28402 at 3,300. Meanwhile the 256-sample spread ratio rose from 0.763 at 3,000
to 1.383 at 3,250. Spatial variation in a degenerate palette can therefore make
spread appear healthy. The published b-cap loss reached 26,378 near step 3,400;
this identifies instability, not its root cause.

## Adaptation

`CIFARDiscriminator(image_size=256, feature_size=256)` uses the same pinned
ImageNet ResNet18 weights as CIFAR. Three frozen stages yield native feature
maps at 64, 32 and 16 pixels. The learned feature heads, fixed zero context,
and score combination are the existing CIFAR implementation. The pixel branch
extends from three to six residual downsampling blocks, ending at 4px, with
SAGAN attention at 16px. It has one candidate input and one scalar output.
There is no candidate-dependent grayscale conditioning. The fixed context's
pretrained features are computed once and cached.

Backbone parameters and BatchNorm statistics remain frozen while pixel
first and second derivatives pass through the features. Native resolution
and the deeper pixel branch change capacity and computation; this is not an
assertion that the 256px dynamics match CIFAR. Default 32px behavior remains
compatible. Changing architecture requires a fresh run.

## Owner setup

`~/dev/hypergan/training-runs/start-color.sh` selects physical GPU 1, a fresh
`train-color-resnet` directory, `colorization-resnet-env`, and
`logos-colorization-resnet-256/colorization.toml`. The existing DINO + RGB
run is preserved with `start-color-critic.sh`. A parsed configuration comparison
confirms only the discriminator and display name change from that owner recipe.

Evidence is in
`~/dev/hypergan/resurrection-backups/2026-09-21-colorization-resnet/`.
The owner run is left unstarted. Execution checks do not establish that ResNet
fixes collapse; training must assess palette and conditional routing as well as
random spread. No repeated-seed experiments or prerelease heavy tests are used.

## Validation

The installed wheel passes **939 fast tests**, with **186 heavy tests deselected**.
Eleven new CPU tests cover native feature shapes, frozen BatchNorm and parameters,
first/second input derivatives, checkpoint/cache handling, and legacy defaults.
An independent real-weight comparison against develop commit `a1bdd1b6` found
all 159 default CIFAR state tensors, constructor RNG state, train/eval outputs,
input gradients, cached context features and pretrained metadata exactly equal.

On physical GPU 1, the installed ResNet recipe completed updates 1–8, then
resumed its checkpoint and completed updates 9–16. Both attempts exercised b-cap
and emitted checkpoints plus paired and random previews. This validates execution
and recovery only; no long training or alternate-seed run was performed.

Receipts include `fast-tests.log`, `installed-smoke.log`, `installed-resume.log`,
`cifar-default-parity.json`, and `owner-collapse-audit.json`. The launcher passes
shell syntax and CLI argument checks, and the owner run directory is absent.

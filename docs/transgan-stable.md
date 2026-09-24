# TransGAN 128px with the section 3.4 recipe

[`examples/transgan-projected-dinov3-128-stable.toml`](../examples/transgan-projected-dinov3-128-stable.toml)
combines the three techniques in [TransGAN section 3.4](https://arxiv.org/html/2102.07074v4#S3.SS4):

- Differentiable color, translation and cutout augmentation on real and generated
  RGB images, before ImageNet preprocessing and frozen DINOv3 features.
- Learned 2D relative-position bias added to scaled attention logits, alongside
  the generator's existing learned absolute position embeddings.
- Parameter-free per-token normalization,
  `x / sqrt(mean(x**2, channels) + 1e-8)`, before attention and feed-forward blocks.

The latter two were already present on `develop`. DiffAug adds the missing
section 3.4 technique; the stable recipe also restores the paper’s full generator
depth and latent width and removes the final tanh to match upstream RGB output. It applies during both discriminator and generator updates, including
when discriminator parameters are frozen. Evaluation disables it without drawing
random numbers. Translation and cutout use integer indexing and masks so
gradient penalties can differentiate through the augmentation. The stable recipe follows the basic three-operation policy in the
[official TransGAN implementation](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/diff_aug.py):
translation (maximum 20% per axis), half-side cutout enabled on 30% of batches,
then color. The cutout gate is shared across the batch; cutout centers and other
transform values are per image. Its batch gate uses device PyTorch RNG for
checkpoint replay instead of upstream Python RNG.

The configured HNDL operation is:

```python
x = diff_augment(x, transforms="translation,cutout,color",
    translation_ratio=0.2, cutout_probability=0.3)
```

Omitting these arguments retains generic DiffAug defaults (color first, 12.5%
translation, cutout on every batch).
`transforms` avoids HNDL's reserved `policy` keyword. The operation and selected
transforms are recorded in resolved network source and configuration identity.
The recipe explicitly uses autograd gradient penalties. Finite-difference
penalties would require reusing augmentation draws across perturbed critic calls.

This is an adaptation with a frozen projected DINOv3 critic, not a reproduction
of the paper's transformer discriminator or WGAN-GP optimizer recipe. DINOv3
retains its pretrained normalization and positional encoding. Four frozen
cross-channel projections feed independent spectral convolutional heads, yielding
`[B, 4, 4, 4]` logits. All four feature depths are 8x8, so there is no cross-scale
fusion. The stable generator uses the paper’s 128px depths (5/4/4/4/4), a 512-dimensional
latent and raw RGB output. The existing reduced-depth generator remains available
for other recipes. Logistic relativistic loss, lazy b-cap penalty, MoG prior type,
optimizer settings, update schedule and seeds are retained. Only the prior latent
width changes to match the new generator input.

Set the example's dataset manifest, digest, DINOv3 weights and pinned source
checkout before using it. The prepared local configuration is
`../training-runs/transgan-projected-dinov3-128-stable.toml`; its launcher is:

```bash
../training-runs/start-transgan-projected-dinov3-128-stable.sh
```

The local launcher uses the `HyperGAN` develop checkout, existing `transgan-128-env`,
GPU UUID from the projected-DINO launcher, and a separate
`train-transgan-projected-dinov3-128-stable` run directory. It accepts additional
training CLI arguments. The default duration is the configured 200,000 steps;
`--stop-after-steps 128` bounds an initial run without changing its target duration.

"Stable" names the candidate configuration. Numerical derivative and replay tests
do not establish convergence or improved image quality. Compare published
training/diversity metrics before drawing that conclusion; no seed sweep is
needed to validate this implementation.

See the [pinned upstream comparison](transgan-upstream-comparison.md) for remaining
architecture, optimizer, loss and high-resolution recipe differences.

## Implementation validation

The default suite passed 1,132 tests with 188 heavy tests deselected. Its only
failure was the distribution inventory assertion against an editable installation;
the two distribution tests passed against a wheel installed in a temporary target.
After aligning the policy with upstream, 46 augmentation, HNDL and projected-critic
tests passed, including exact checkpoint continuation and second derivatives.
The generator's normalization/position audit passed all 18 tests.
Five further tests validate the final 151,512,455-parameter generator: all 21
blocks and 42 normalization layers, 512-dimensional stem initialization, finite
128px forward/input gradients, unbounded RGB output and unchanged training
settings. The latter comparison permits only the declared network changes and
matching prior latent width.

A CPU forward/backward check loaded the actual pinned DINOv3 checkpoint and
verified the dataset manifest digest. The final configured augmentation produced
finite `[1, 4, 4, 4]` logits, input gradient L2 norm 0.282507, and finite gradients
for all eight trainable critic parameter tensors, while frozen weights received
none. Evaluation was exactly repeatable and consumed no RNG. These are
implementation checks; the prepared training run has not been launched.

## Matched run without DiffAug

[`transgan-projected-dinov3-128-no-diffaug.toml`](../examples/transgan-projected-dinov3-128-no-diffaug.toml)
uses the same generator, discriminator weights and architecture, optimizer, loss,
batch size and seeds. Its only numerical configuration difference is an empty
DiffAug transform list: the operation becomes identity and consumes no RNG.
This is an augmentation ablation, starting from scratch with the same seed.

The local launcher is
`../training-runs/start-transgan-projected-dinov3-128-no-diffaug.sh`.
It uses physical GPU 0 (`GPU-ed080e41-3193-3755-6756-f3d46c433331`), while the
augmented run uses physical GPU 1. Checkpoints and previews for this run live in
`/mnt/ml7tb/hypergan-training-runs/train-transgan-projected-dinov3-128-no-diffaug`
to avoid filling the home filesystem. Checkpoint, preview and metric cadences
match the augmented launcher.

## Full-depth generator with a ResNet critic

[`transgan-resnet-multiscale-128-stable.toml`](../examples/transgan-resnet-multiscale-128-stable.toml)
replaces the no-DiffAug variant's entire discriminator with the existing
`resnet18-multiscale-discriminator-128.hndl`. It uses a SHA256-pinned frozen
ImageNet ResNet18, native layer1/2/3 feature maps, trainable feature heads and
a trainable pixel branch with fixed gray context. It emits one score per image.
BatchNorm retains its pretrained statistics; input gradients still pass through
the frozen backbone. DiffAug remains disabled.

The full-depth 512-latent TransGAN, prior, optimizer, loss, gradient penalty,
batch size, seeds and update schedule match the no-DiffAug baseline. A resolved
configuration comparison permits only the discriminator component and run name
to differ. This compares critic designs, including their heads and pixel paths,
rather than isolating the pretrained backbone alone.

Run the prepared local launcher manually:

```bash
../training-runs/start-transgan-resnet-multiscale-128-stable.sh
```

It uses physical GPU 0 and a separate run directory on the data volume:
`/mnt/ml7tb/hypergan-training-runs/train-transgan-resnet-multiscale-128-stable`.
The older launcher without the `-stable` suffix uses the earlier reduced-depth
generator and is a different configuration.

Validation: 10 pretrained-provider tests passed. A CPU check with the real
ResNet18 weights produced finite scores and trainable-head gradients through
an autograd gradient penalty. Frozen backbone weights received no gradients,
BatchNorm buffers stayed unchanged, and generator-side input gradients remained
finite and nonzero with discriminator parameters frozen. Preparation did not
start a training run.

## E2 residual scaling with the ResNet critic

[`transgan-resnet-multiscale-128-e2.toml`](../examples/transgan-resnet-multiscale-128-e2.toml)
adapts E2 to TransGAN. Each transformer block already has identity bypasses
around attention and the feed-forward network. This variant scales the learned
branches before adding them to those bypasses:

```text
u      = h + alpha * Attention(PixelNorm(h))
h_next = u + alpha * FFN(PixelNorm(u))
alpha  = 1 / sqrt(42) = 0.1543033499620919
```

There are 21 blocks and 42 residual branches, so the scale uses 42 instead of
the 15 hidden transitions in the proposed MLP experiment. Both global and
windowed attention stages use the same scale. The input projection, position
embeddings, upsampling and raw RGB readout are unchanged. This tests damping
existing residual branches; it does not add new bypasses to TransGAN.

The parameter count remains 151,512,455. All generator state keys, tensor
shapes and initialization draws match the unscaled generator. E2 adds only
fixed elementwise multiplies; E3 latent concatenation would widen learned
projections and is not included. Numerical tests check identical initial
weights, strict generator state loading, all 42 scaled additions, changed
outputs and finite nonzero latent gradients. These checks do not establish
improved conditioning or convergence.

The no-DiffAug ResNet critic, prior, optimizer, losses, batch size, seeds and
schedule match the ResNet stable recipe. The prepared launcher starts a new
run from scratch; generator weight compatibility does not bypass full training
checkpoint recipe checks.

```bash
../training-runs/start-transgan-resnet-multiscale-128-e2.sh
```

It selects physical GPU 0 and writes to
`/mnt/ml7tb/hypergan-training-runs/train-transgan-resnet-multiscale-128-e2`.
Training is left for the user to launch.

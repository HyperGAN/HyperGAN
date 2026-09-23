# TransGAN 128px with the section 3.4 recipe

[`examples/transgan-projected-dinov3-128-stable.toml`](../examples/transgan-projected-dinov3-128-stable.toml)
combines the three techniques in [TransGAN section 3.4](https://arxiv.org/html/2102.07074v4#S3.SS4):

- Differentiable color, translation and cutout augmentation on real and generated
  RGB images, before ImageNet preprocessing and frozen DINOv3 features.
- Learned 2D relative-position bias added to scaled attention logits, alongside
  the generator's existing learned absolute position embeddings.
- Parameter-free per-token normalization,
  `x / sqrt(mean(x**2, channels) + 1e-8)`, before attention and feed-forward blocks.

The latter two were already present on `develop`. DiffAug is the new training
change. It applies during both discriminator and generator updates, including
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
fusion. The existing reduced-depth generator, logistic relativistic loss,
lazy b-cap penalty, prior, optimizer settings and seeds are retained.

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

A CPU forward/backward check loaded the actual pinned DINOv3 checkpoint and
verified the dataset manifest digest. The final configured augmentation produced
finite `[1, 4, 4, 4]` logits, input gradient L2 norm 0.282507, and finite gradients
for all eight trainable critic parameter tensors, while frozen weights received
none. Evaluation was exactly repeatable and consumed no RNG. These are
implementation checks; the prepared training run has not been launched.

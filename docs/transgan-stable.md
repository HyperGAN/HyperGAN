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
gradient penalties can differentiate through the augmentation. The policy follows
the [official DiffAugment reference](https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py).

HNDL exposes it as `diff_augment(x, transforms="color,translation,cutout")`.
`transforms` avoids HNDL's reserved `policy` keyword. The operation and selected
transforms are recorded in resolved network source and configuration identity.

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

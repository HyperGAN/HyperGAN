# Logo colorization at 256×256

`examples/logos-colorization-256.toml` is an experimental conditional particle
VAEGAN. It takes grayscale logos through a learned encoder, hard-routes each
query to one of 4,096 learned 128-dimensional particle centers, adds Gaussian
noise, and decodes an RGB image. There are no spatial skip connections. Changing
the noise gives multiple candidate colorizations of the same input. The narrow
particle bottleneck may also lose logo structure; useful quality and diversity
require training and evaluation, and are not established by the smoke test.

The routing follows ParticleGAN's `vae_gan` / `particle_vae(hard=True)`:
the selected posterior component has the same fixed sigma as the prior and no
learned offset or variance. Sigma is calibrated once at prior initialization and
saved. For the uniform particle prior the joint KL is the constant `log(4096)`,
so there is no optimized KL term. The hard forward route uses a soft
straight-through gradient. This is a surrogate gradient, not an unbiased
categorical estimator. See the original
[ParticleGAN implementation](https://github.com/255BITS/ParticleGAN/blob/d2a6450985282047d2d27c36eb80ff9f5200f8c9/particlegan/autoencoder.py).

GAN training samples the **uniform particle prior**, independently of the encoder.
This keeps all particles eligible for generator training even if encoder routing
collapses. The separate reconstruction objective follows the CIFAR recipe's
encoder-only variant: it updates E through a frozen G and detached particle
means. Particle spread regularization and lazy b-cap remain configured separately.

```text
B = BW(X)
z ~ uniform particle prior + fixed-sigma noise
D(X), D(G(z))                         # adversarial: update D, G, prior
B -> E(B) -> selected center + noise -> G -> Xhat
L2 = mean((BW(Xhat) - B)^2)            # reconstruction: update E only
```

Grayscale reconstruction asks the encoder to recover structure without requiring
the original colors. It does not force each colorization to have plausible colors;
the unconditional GAN learns the overall RGB distribution. RGB reconstruction is
an available comparison: bind the objective's input to `components.reconstruction`
and its target to `batch.real`, and remove the unused `reconstruction_gray`
component. Neither form constrains aggregate encoder particle
usage. Hard routing and a moving decoder can still make the encoder collapse.

The discriminator receives only RGB through the same single path for real and fake:

```text
RGB -> [frozen DINOv3 + frozen projection, learned RGB stem]
    -> concatenate at 16x16 -> shared attention -> convolutional head -> D(RGB)
```

The projection mixes DINOv3's final 16×16 patch features with fixed random 1×1
channel and 3×3 spatial convolutions. A learned four-stage RGB stem supplies
local pixel features at the same resolution. Their concatenated features pass
through one SAGAN attention module and one spectrally normalized convolutional
head. The RGB stem, attention, and output head learn.
This is a minimal single-map adaptation of the frozen feature/projection idea in
[Projected GAN](https://github.com/autonomousvision/projected-gan/blob/main/pg_modules/projector.py).
The 3×3 layer mixes local spatial features; it does not reproduce the paper's
multiscale feature fusion or multiple discriminator heads. There is one backbone
call per candidate batch, with no grayscale input or separate pixel critic.
Normal training still evaluates real and fake candidates and the configured
b-cap regularizer; “one path” does not mean only one D evaluation per update.

The frozen DINOv3 ViT-S/16 LVD-1689M backbone and random projection stay in
evaluation mode while retaining derivatives with respect to candidate pixels,
including the second derivatives required by b-cap. The recipe pins the local
weight file by SHA256 and the external source checkout by commit; it does not
download during training. The upstream code and weights retain their
[DINOv3 terms](https://github.com/facebookresearch/dinov3).

The discriminator comparisons also expose these alternatives (use a fresh run
when changing architecture):

- `DINOv3ProjectedDiscriminator(head="conv")` replaces the linear head with
  spectrally normalized nonlinear convolutions. Its optional `pixel_width=32`
  adds a learned RGB stem, concatenated with projected DINO features before the
  shared attention/head. The example uses this configuration: it avoided the
  earlier near-constant failure through 1,500 controlled updates while retaining
  the original b-cap settings. This is bounded collapse evidence, not a guarantee
  of long-run stability or colorization quality.
- `DINOv3MultiScaleDiscriminator` reads transformer blocks 2, 5, 8, and 11 in
  one backbone pass. Frozen random projections build and fuse a synthetic
  32/16/8/4 pyramid, followed by four attention/convolution heads whose scalar
  outputs are averaged. DINOv3's native maps are all 16×16; this is a multidepth
  adaptation, not an exact reproduction of Projected GAN.
- `DCGANDiscriminator256(width=32)` is an RGB-only convolutional control with
  spectral normalization. It has no pretrained backbone.

The DINOv3 token and normalization audit, controlled comparisons, and their limits
are recorded in [the collapse report](../reports/colorization-collapse-2026-09-21.md).
The working CIFAR critic combines a learned pixel path with multiscale pretrained
ResNet features, so it is not a pretrained-only counterpart to the original
single-map DINO critic.

The earlier two-path `DINOv3Discriminator` remains available for existing
configurations and checkpoints. The new `DINOv3ProjectedDiscriminator` is a
different architecture: use a fresh run directory and its own configuration.

## Data preparation

Use `hypergan.colorization_data.prepare_manifest(root, output)` once to validate
the image inventory and record content hashes. Set `manifest_sha256` in the
recipe to the returned digest. The source images are not modified. Preprocessing
applies EXIF orientation, composites transparency on white, and resizes with
aspect-preserving Lanczos fitting and white padding to 256×256. Grayscale is
`0.299 R + 0.587 G + 0.114 B`; RGB and grayscale are in `[-1,1]`.

The SHA256 of each source image assigns approximately 95% to training and 5% to
held-out evaluation. Byte-identical duplicates stay in the same split; perceptual
near-duplicates can still cross it. Archives and other non-image files are
inventoried but not expanded. Invalid files require explicit, recorded exclusions
or source repair. Training verifies each accessed file and fails on changed or
unreadable bytes. Added files do not silently enter an existing inventory.

## Metrics and samples

Use the held-out split for independent snapshot evaluation. Chroma and diversity are not directly optimized objectives; the edge measurement
is a structural diagnostic related to the grayscale reconstruction loss:

- Chroma distribution distance compares generated and reference color
  distributions. Lower is closer, but a good value cannot prove correct spatial
  placement or useful conditional diversity.
- Grayscale structure distance measures whether generated logos retain the
  source's edge structure. Lower is better.
- Repeated-condition chroma diversity measures variation across several noise
  draws for each identical grayscale input. Interpret it alongside structure;
  arbitrary noise can increase diversity without improving colorization.

Two additional metrics explicitly select uniform-prior outputs with
`evaluation.generated = "generated"`, independently of the conditional preview:

- `random_spread` measures pooled pixel pairwise RMS relative to the held-out
  reference. Near zero detects constant outputs; one matches overall spread.
  Noise or repeated coarse patterns can also produce spread, so this is a
  collapse diagnostic, not a quality target. It runs every 250 updates on 256
  samples to reveal the early failure seen in the original run.
- `random_chroma` compares unconditional color distributions every 1,100 updates
  on 512 samples. It shares the existing chroma metric's spatial limitations.

Keep the evaluation seed, count, held-out inventory and sample multiplicity
fixed when comparing checkpoints. Show grayscale inputs, source RGB and generated
RGB previews together. The `comparison` view has one example per row and labelled
`X | B | X_hat` columns. The separate `random` view shows uniform-prior samples;
`g` remains the conditional reconstruction. `sampling.generated` explicitly binds
sampling and paired evaluation to `components.reconstruction`, so changing the GAN
input does not silently turn conditional metrics into unconditional comparisons.
A short smoke test establishes execution and recovery,
not a quality score. Do not select checkpoints solely by one color statistic.

The example evaluates a fixed, content-hash-ordered held-out subset: 512 images
for color and structure, and 128 conditions with four draws each for diversity.
Hash ordering avoids taking a prefix of one source collection's filenames. Color
statistics include white backgrounds, which can dominate sparse logos. The
intervals are 5,000, 6,000 and 7,000 updates respectively; distinct intervals
avoid making every metric compete for the single evaluator at every boundary.
Colliding intervals or a still-busy evaluator produce explicit skipped events.
Evaluation shares GPU 1, so it temporarily slows training.

## Local owner workflow

The prepared machine-specific configuration, pinned artifacts and isolated
environment live under `~/dev/hypergan/training-runs/`. Run:

```sh
bash ~/dev/hypergan/training-runs/start-color.sh
```

The launcher pins physical GPU 1 by UUID and uses a fresh `train-color-critic` run.
Interrupt it with Ctrl-C to save a recoverable boundary. Repeating the same
command resumes the latest complete checkpoint. It does not touch the CIFAR run.
The default total schedule is 200,000 updates; batch size starts at 16. The fast
backend permits nondeterministic CUDA kernels, so recovery restores complete
state without promising bitwise-identical future learning trajectories.

Configuration and dataset/component dependency checks remain enforced on resume;
HyperGAN release hashes remain provenance, not compatibility rejection keys.
The validation run is separate from `train-color-critic`, leaving it fresh.
The collapsed projected run and its launcher `start-color-projected.sh` remain
available. The updated launcher uses `colorization-critic-env` and
`logos-colorization-critic-256/colorization.toml`. The earlier random-prior run
and its `start-color-random.sh` launcher are preserved, along with
`start-color-original.sh`.

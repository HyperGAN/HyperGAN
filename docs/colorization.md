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

The discriminator receives a single RGB candidate, using the pretrained ResNet18
critic from the CIFAR recipe extended to 256×256:

```text
RGB -> frozen ResNet18 layer1/layer2/layer3 -> three learned feature heads
RGB -> learned residual pixel branch with attention at 16x16
D(RGB) = (pixel_score + sum(feature_scores) / sqrt(3)) / sqrt(2)
```

`CIFARDiscriminator(image_size=256, feature_size=256)` keeps the CIFAR feature
heads and score combination. Six pixel residual blocks reduce 256×256 to 4×4;
SAGAN attention remains at 16×16. The native pretrained feature maps are
64×64, 32×32 and 16×16, each projected and pooled to 4×4 by a learned head.
The existing fixed zero context and its cached pretrained features are retained
for recipe continuity; they contain no grayscale input or other training sample.
After this constant cache is populated, each candidate batch traverses the
backbone once. The pixel branch and three feature heads form one discriminator
with one scalar output and the same adversarial objective.

ResNet parameters and batch-normalization statistics stay frozen, but input
derivatives pass through the backbone, including the second derivatives needed
by b-cap. Inputs are mapped from [-1,1] to [0,1] and ImageNet-normalized.
Training requires the `cifar` extra and a local, SHA256-pinned
`resnet18-f37072fd.pth` file; it does not download weights. The 32×32 CIFAR
architecture, initialization order and checkpoint layout retain their defaults.

This is the next discriminator experiment after the DINO + RGB owner run
collapsed by step 3,428. Its earlier 1,500-step diversity result did not establish
lasting stability or useful samples. ResNet is not yet a demonstrated cure.
Generator, encoder, particles, losses, b-cap and learning rates are unchanged.

The discriminator comparisons also expose these alternatives (use a fresh run
when changing architecture):

- `DINOv3ProjectedDiscriminator(head="conv")` replaces the linear head with
  spectrally normalized nonlinear convolutions. Its optional `pixel_width=32`
  adds a learned RGB stem, concatenated with projected DINO features before the
  shared attention/head. The previous example used this configuration. It retained spread
  through 1,500 controlled updates, but the subsequent owner run collapsed.
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

The launcher pins physical GPU 1 by UUID and uses a fresh `train-color-resnet` run.
Interrupt it with Ctrl-C to save a recoverable boundary. Repeating the same
command resumes the latest complete checkpoint. It does not touch the CIFAR run.
The default total schedule is 200,000 updates; batch size starts at 16. The fast
backend permits nondeterministic CUDA kernels, so recovery restores complete
state without promising bitwise-identical future learning trajectories.

Configuration and dataset/component dependency checks remain enforced on resume;
HyperGAN release hashes remain provenance, not compatibility rejection keys.
The validation run is separate from `train-color-resnet`, leaving it fresh.
The collapsed projected run and its launcher `start-color-projected.sh` remain
available. The updated launcher uses `colorization-resnet-env` and
`logos-colorization-resnet-256/colorization.toml`. The earlier random-prior run
and its `start-color-random.sh` launcher are preserved, along with
`start-color-original.sh`.

The previous DINO + RGB run is preserved under `train-color-critic`, with its
configuration, environment, and `start-color-critic.sh` launcher. Use a fresh run
when changing discriminator architecture; old checkpoints are not compatible.

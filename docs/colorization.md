# Logo colorization in HyperGAN and HNDL

![HyperGAN 2.0 Logo Colorization: Grayscale HG to 4×4 Colorized Grid](assets/hypergan-colorizer-banner.png)

_Assets: [128×128 grayscale HG input (`assets/hg-bw-128.png`)](assets/hg-bw-128.png) · [128×128 single colorized badge (`assets/hypergan-logo.png`)](assets/hypergan-logo.png) · [512×512 4×4 colorized grid (`assets/hypergan-logo-grid.png`)](assets/hypergan-logo-grid.png) · [736×512 workflow banner (`assets/hypergan-colorizer-banner.png`)](assets/hypergan-colorizer-banner.png)._

## 128×128 conditional colorization (`examples/image-folder-colorization.toml`)

`examples/image-folder-colorization.toml` (archived research ablation: `examples/archive/logos-colorization-128.toml`) is a declarative `hypergan-model/1` model
file (`defaults = "particlegan"`) that pairs a multi-input HNDL conditional
colorization generator (`examples/networks/colorization-generator-128.hndl`)
with the 128×128 multidepth DINOv3 projected critic
(`examples/networks/dinov3-projected-discriminator-128.hndl`) and a 4,096-particle
latent prior.

Zero custom Python model classes are required: the entire conditional generator,
bilinear particle palette head, exact polytope luminance projection, and
multidepth ViT-S/16 critic are declared directly in HNDL and `hypergan-model/1`.

### Why the earlier 256×256 particle-routing recipe collapsed

The earlier 256×256 experiment (`examples/archive/logos-colorization-256.toml`) had four
structural bottlenecks:

1. **No spatial conditioning in `G`**: `ColorizationGenerator(z)` took only a
   1D vector `z` with no spatial input or skip connections from `batch.gray`,
   forcing a single hard-routed particle index `k in {0..4095}` to reconstruct
   the entire 2D logo geometry.
2. **Disconnected adversarial and conditional paths**: the adversarial phases
   trained `G(z)` unconditionally on uniform random particles, while `E(B)`
   received only a grayscale L2 loss through a straight-through softmax
   surrogate and collapsed onto a single particle index.
3. **Unconstrained RGB output**: predicting raw RGB wasted generator capacity
   re-learning known pixel luminance `Y` and allowed saturated magenta/green
   tinting on white (`Y = +1`) backgrounds.
4. **Pre-0.9 optimizer and critic penalty**: training used raw Adam and
   unnormalized `b_cap` through frozen backbones at batch size 16 prior to the
   ParticleGAN 0.9.2 `bispf_optimized` + `dualnorm` + `deterministic_orthogonal`
   stabilization.

### First-principles HNDL & `hypergan-model/1` design

`examples/networks/colorization-generator-128.hndl` and
`examples/logos-colorization-128.toml` solve all four issues by construction
while showcasing HNDL and HyperGAN's declarative capabilities:

1. **Multi-port HNDL generator (`input_shape = { z = ["B", 128], gray = ["B", 1, 128, 128] }`)**:
   - **Multi-scale residual grayscale U-Net encoder (`128 -> 64 -> 32 -> 16 -> 8`)**:
     Five declarative `resblock(..., norm="group_norm")` stages extract full-resolution
     spatial edge/region features (`e128`, `e64`, `e32`, `e16`, `e8`).
   - **Bounded loop (`for _ in range(2):`) with RoPE + SwiGLU `transformer_block` and `cross_attention`**:
     The particle latent `z` projects into 8 learned palette tokens `[B, 8, 256]`.
     Inside the HNDL loop (which automatically suffixes node names `0`, `1`),
     the 64 spatial tokens run 2D-aware RoPE + SwiGLU self-attention and
     cross-attend directly to the 8 particle palette tokens.
   - **Sub-pixel (`pixel_shuffle(2)`) + `resblock` U-Net skip decoder with shape-inferred `adaptive_norm`**:
     Each decoder stage upsamples via `conv` + `pixel_shuffle(2)`, concatenates
     the corresponding encoder skip (`e16`, `e32`, `e64`, `e128`), refines via
     `resblock`, and modulates features with `adaptive_norm(h, linear(z, name="style..."))`
     where HNDL's bidirectional shape solver automatically infers `2 * C` (`256`,
     `128`, `64`) for each `linear(z)` projection.
2. **Bilinear particle palette synthesis (anti-mode-collapse by construction)**:
   - At 128×128, `group_norm(h128, 8, affine=False)` produces a 16-channel
     spatial region basis `basis` `[B, 16, 16384]` with zero mean and unit
     variance across spatial positions, while `z` projects through a bias-free
     `palette = reshape(linear(z, 48, bias=False), 3, 16)`.
   - The raw RGB proposal is `tanh(scale(reshape(matmul(palette, basis), 3, 128, 128), 0.25))`.
   - Because `palette_head` has `bias=False`, `z = 0` produces zero chroma
     (pure grayscale), so the generator cannot ignore `z` without being
     penalized by the chroma-sensitive critic.
3. **Exact 7-face polytope luminance lock in pure HNDL**:
   - Luminance in `ColorizationData` is `Y = 0.299 R + 0.587 G + 0.114 B` in `[-1, 1]`.
   - Iterating `x <- clamp(x + g_k * (gray - Y) * [1, 1, 1], -1, 1)` over the 7
     exact reciprocals of the non-empty subsets of `{0.299, 0.587, 0.114}`
     (`1.0`, `1/0.886`, `1/0.701`, `1/0.587`, `1/0.413`, `1/0.299`, `1/0.114`)
     projects any proposal onto the exact intersection of the RGB cube `[-1, 1]^3`
     and the luminance hyperplane `w^T x = gray` to float32 machine precision (`< 1e-5`).
   - White backgrounds (`gray = +1`) map identically to `(1, 1, 1)`, black
     (`gray = -1`) maps identically to `(-1, -1, -1)`, and `GrayscaleStructureDistance`
     is `< 1e-5` from step 0 without any auxiliary L2 loss.
   - Because `L(G(gray, z)) == gray` is an exact bijection between `X_hat` and
     `(gray, C_hat)`, the 3-channel multidepth DINOv3 projected critic `D(X)` vs
     `D(G(gray, z))` on the same batch is automatically a conditional critic over
     `p(C | gray)`.

---

## Historical 256×256 particle-routing experiment (`examples/archive/logos-colorization-256.toml`)

`examples/archive/logos-colorization-256.toml` is an experimental conditional particle
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
means. Particle spread regularization and lazy critic regularization remain configured separately.

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
an available comparison: bind the objective's `input` to `reconstruction`
and its `target` to `batch.real`, and remove the unused `reconstruction_gray`
network. Neither form constrains aggregate encoder particle
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
by critic regularization. Inputs are mapped from [-1,1] to [0,1] and ImageNet-normalized.
Training requires the `cifar` extra and a local, SHA256-pinned
`resnet18-f37072fd.pth` file; it does not download weights. The 32×32 CIFAR
architecture, initialization order and checkpoint layout retain their defaults.

This is the next discriminator experiment after the DINO + RGB owner run
collapsed by step 3,428. Its earlier 1,500-step diversity result did not establish
lasting stability or useful samples. ResNet is not yet a demonstrated cure.
Generator, encoder, particles, losses, critic regularization and learning rates are unchanged.

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
are recorded in [the collapse report](../reports/archive/colorization-collapse-2026-09-21.md).
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
aspect-preserving Lanczos fitting and white padding to the configured resolution
(128×128 or 256×256). Grayscale is `0.299 R + 0.587 G + 0.114 B`; RGB and
grayscale are in `[-1,1]`.

The SHA256 of each source image assigns approximately 95% to training and 5% to
held-out evaluation. Byte-identical duplicates stay in the same split; perceptual
near-duplicates can still cross it. Archives and other non-image files are
inventoried but not expanded. Invalid files require explicit, recorded exclusions
or source repair. Training verifies each accessed file and fails on changed or
unreadable bytes. Added files do not silently enter an existing inventory.

## Metrics and samples

Use the held-out split for independent snapshot evaluation:

- Chroma distribution distance compares generated and reference color
  distributions. Lower is closer, but a good value cannot prove correct spatial
  placement or useful conditional diversity.
- Grayscale structure distance measures whether generated logos retain the
  source's edge structure. Under the 128×128 exact polytope luminance lock,
  structure distance is zero up to floating-point rounding by construction.
- Repeated-condition chroma diversity measures variation across several particle
  draws (`repeats = 4`) for each identical grayscale input.
- `random_spread` measures pooled pixel pairwise RMS relative to the held-out
  reference. Near zero detects constant outputs; one matches overall spread.
- `fid10k_heldout` evaluates 10,000 held-out samples with `InceptionFID`.

Keep the evaluation seed, count, held-out inventory and sample multiplicity
fixed when comparing checkpoints. Show grayscale inputs, source RGB and generated
RGB previews together. The `comparison` view has one example per row and labelled
`X | B | X_hat` columns. The separate `random` view shows uniform-prior samples;
`g` remains the conditional reconstruction. `sampling.output` explicitly binds
sampling and paired evaluation to `reconstruction`, so changing the GAN
input does not silently turn conditional metrics into unconditional comparisons.
A short smoke test establishes execution and recovery,
not a quality score. Do not select checkpoints solely by one color statistic.

## Local workflow

Run the 128×128 colorization recipe with a dedicated run directory:

```sh
hypergan train examples/image-folder-colorization.toml \
    --run-dir ./runs/train-logos-colorization-128 -d 0
```

Interrupt it with Ctrl-C to save a recoverable boundary. Repeating the same
command resumes the latest complete checkpoint. Configuration and dataset/component
dependency checks remain enforced on resume; HyperGAN release hashes remain
provenance, not compatibility rejection keys.

### Post-training single-image colorization (`hypergan sample --input`)

Once trained, sample diverse colorizations of any single grayscale image (such as
the 128×128 grayscale **HG** monogram in `docs/assets/hg-bw-128.png`) from the
CLI or Python API. When a single conditioning image is passed via `--input` (or
`inputs=...` in Python), HyperGAN preprocesses it with the `ColorizationData`
aspect-preserving Lanczos fit and broadcasts it across all `count` particle draws:

```sh
hypergan sample logos-colorization-128 \
    --input docs/assets/hg-bw-128.png \
    --checkpoint 3000 \
    --count 16 \
    --seed 10000 \
    --output docs/assets/hypergan-logo-grid.png
```

Or in Python:

```python
import hypergan as hg

sample = hg.sample(
    "logos-colorization-128",
    inputs="docs/assets/hg-bw-128.png",
    checkpoint=3000,
    count=16,
    seed=10000,
    output="docs/assets/hypergan-logo-grid.png",
)
```

To regenerate the full set of HyperGAN 2.0 logo assets (`docs/assets/hg-bw-128.png`,
`docs/assets/hypergan-logo.png`, `docs/assets/hypergan-logo-grid.png`, and
`docs/assets/hypergan-colorizer-banner.png`) with quantitative checkpoint metrics:

```sh
python scripts/generate_logo.py --run-dir logos-colorization-128
```

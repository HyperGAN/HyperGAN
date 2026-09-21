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

This conditional adaptation makes the generated adversarial sample depend on
the grayscale encoder. Adversarial gradients update the encoder, generator and
particle centers. As in the CIFAR recipe, the separate reconstruction objective
updates only the encoder: it reuses the generator with frozen parameters and
detached particle means, with the same selected center and noise. Particle spread
regularization and lazy b-cap remain configured separately.

The discriminator combines a pixel critic conditioned on grayscale with a frozen
DINOv3 ViT-S/16 LVD-1689M feature critic and trainable SAGAN attention. Its feature
path remains differentiable with respect to candidate pixels, including the
second derivatives required by b-cap. The pretrained backbone stays frozen and
in evaluation mode. The recipe pins the local weight file by SHA256 and the
external DINOv3 source checkout by commit; it does not download during training.
The upstream code and weights retain their [DINOv3 terms](https://github.com/facebookresearch/dinov3).

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

Use the held-out split for independent snapshot evaluation. These measurements
are not training objectives:

- Chroma distribution distance compares generated and reference color
  distributions. Lower is closer, but a good value cannot prove correct spatial
  placement or useful conditional diversity.
- Grayscale structure distance measures whether generated logos retain the
  source's edge structure. Lower is better.
- Repeated-condition chroma diversity measures variation across several noise
  draws for each identical grayscale input. Interpret it alongside structure;
  arbitrary noise can increase diversity without improving colorization.

Keep the evaluation seed, count, held-out inventory and sample multiplicity
fixed when comparing checkpoints. Show grayscale inputs, source RGB and generated
RGB previews together. A short smoke test establishes execution and recovery,
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

The launcher pins physical GPU 1 by UUID and uses a separate `train-color` run.
Interrupt it with Ctrl-C to save a recoverable boundary. Repeating the same
command resumes the latest complete checkpoint. It does not touch the CIFAR run.
The default total schedule is 200,000 updates; batch size starts at 16. The fast
backend permits nondeterministic CUDA kernels, so recovery restores complete
state without promising bitwise-identical future learning trajectories.

Configuration and dataset/component dependency checks remain enforced on resume;
HyperGAN release hashes remain provenance, not compatibility rejection keys.
The actual test run is separate from `train-color`, leaving the owner run fresh.

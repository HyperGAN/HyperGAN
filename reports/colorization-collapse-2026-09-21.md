# Diagnosing 256px colorization collapse

PR: https://github.com/HyperGAN/HyperGAN/pull/362

The random-prior GAN path also collapses, independently of encoder routing. These
experiments keep the latent-only generator, fixed-sigma 4096-particle prior,
encoder-only grayscale reconstruction, data, and GAN update ordering fixed while
changing the discriminator. The earlier eight-image reconstruction and fixed-fake
classification probes established limited capacity, not adversarial stability.

## DINOv3 wiring audit

The real pinned ViT-S/16 backbone passes a CPU audit against its upstream API:
input conversion from [-1,1] to [0,1] and ImageNet normalization agree exactly;
256 patch tokens exclude the CLS token and four storage tokens; the normalized
final patch map agrees exactly with `get_intermediate_layers(...)[-1]`. The frozen
backbone retains finite nonzero derivatives with respect to image pixels.
The four sampled transformer depths all have 16x16 spatial resolution.

There is no detected DINOv2-to-v3 token or normalization bug. The existing
projected discriminator uses only the last map, two frozen linear convolutions,
attention, and a linear output: 6,145 trainable parameters. This is much smaller
than Projected GAN's nonlinear multiscale discriminators.

## Controlled comparisons

The bounded GPU 1 harness uses the same initial G/E/prior and shared discriminator
weights, independent phase RNG streams, and deterministic CUDA policy. Evaluation
uses 64 fixed hash-ordered held-out images and 64 fixed random-prior samples.
Evaluation restores the training RNG and does not update spectral-normalization
buffers. These raw-model panels are diagnostics, not representative quality
estimates or checkpoint selection by a learned perceptual metric.

The convolutional DINO head retains the frozen backbone/projection and attention,
then applies spectrally normalized nonlinear convolutions at 16, 8, and 4 pixels.
The DCGAN control takes RGB through six convolutional stages with spectral
normalization and no batch normalization; it has no pretrained features.
Both accept only the candidate image, D(X) or D(G(z)).

Pairwise RMS measures spread; pooled pairwise RMS suppresses fine pixel noise;
nearest-neighbor RMS detects repeated outputs. None establishes logo quality.
The current D's AUC measures its own discrimination, not independent fidelity.
Spectral-normalization power iteration also occurs on training-mode forwards with
frozen parameters, so penalty/no-penalty comparisons can differ in buffer updates.

Machine evidence, reproducible scripts, JSON results, training logs, and checkpoints:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-21-colorization-collapse/`.

## Results before compaction

| Discriminator | b-cap | Updates at evaluation | Random pairwise RMS | Pooled random RMS | D AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Original linear DINO head | yes | 350 | 0.1035 | 0.1014 | 0.000 |
| Original linear DINO head | no | 350 | 0.2251 | 0.2093 | 0.726 |
| Convolutional DINO head | yes | 350 | 0.0293 | 0.0244 | 0.000 |
| Convolutional DINO head | no | 1000 | 0.4812 | 0.4735 | 0.999 |
| DCGAN control | yes | 1000 | 0.7836 | 0.7310 | 0.407 |
| Multiscale DINO | yes | 300 | 0.0561 | 0.0510 | 0.000 |
| Multiscale DINO | no | 400 | 0.9121 | 0.8423 | 1.000 |

Reference pairwise RMS is 1.0493; pooled reference RMS is 0.9794. The multiscale
runs were interrupted to pause for compaction; their shorter results do not
qualify stability beyond the original failure horizon. Completed arms have full
model/optimizer/EMA/RNG checkpoints. Interrupted arms retain logs and metrics,
but the harness only checkpoints on completion and therefore cannot resume them.

The **DCGAN control changed only D**. Its generator, encoder, prior, losses,
b-cap, learning rates, and reconstruction ownership stayed fixed. It avoided the
near-constant-output failure through 1,000 updates. This implicates the original
D setup; it does not identify a DINOv3 token bug or prove longer-run stability.
A fixed-latent CPU probe also finds preserved generator sensitivity after DCGAN
training, while the collapsed DINO models progressively suppress variation in
later upsampling blocks. No generator modification was justified by this audit.

A separate exploratory joint-L2 arm was stopped after the user clarified that
mode collapse, rather than image quality/reconstruction, is the immediate task.
It is not evidence for the discriminator-only result and is not the selected
formulation. A second DCGAN seed was also stopped before its first 100-update
evaluation so effort could return to DINO. No image-quality metric run was done.

## Chosen next comparison

The working CIFAR discriminator combines a learned pixel path and three
pretrained ResNet feature stages. Its success does not establish that a
pretrained-only DINO critic should work. The next candidate restores local RGB
information while retaining one DINO call and one shared attention/output head:

```text
X or G(z)
  -> [frozen DINOv3 projected features, learned RGB stem]
  -> concatenate at 16x16
  -> shared attention -> shared convolutional head -> scalar D
```

Use `DINOv3ProjectedDiscriminator(head="conv", pixel_width=32)`. The learned
RGB stem uses four stride-two convolutions; there is no additional grayscale
input, separate pixel scalar loss, or generator/encoder change. The default
`pixel_width=0` preserves older architectures and checkpoint keys. This new
candidate has **CPU validation only** and is not yet a demonstrated fix. Next,
compare it against the saved discriminator-only controls at the same initialization
and through at least the observed 700-update failure horizon. Keep b-cap initially
fixed; the no-cap DINO results identify the penalty as a second controlled variable.

## Validation and owner state

The fast suite before adding the RGB stem passed **920 tests**, with all **186
heavy tests deselected**. The final RGB-stem/projected/multiscale targeted CPU
selection passed **45 tests**, including independent gradients through either
feature source, frozen weights, double backward, and strict state reload.
No prerelease heavy suite ran.

A clean wheel from `c0be96f8520e59a2df27e18492b950810823a59b` was installed in
`training-runs/colorization-critic-env`. An isolated eight-update real-DINO
multiscale smoke run completed its lazy b-cap update, checkpoint, and previews.
A full 256-sample `random_spread` snapshot completed and recorded
`generated_binding="generated"`, independent of the conditional sampler.
That installed wheel predates the RGB-stem candidate and must be rebuilt before
using it. The new spread and random-color metrics are in the draft example;
the owner's active configuration and `start-color.sh` were not changed.

All diagnostic training processes were stopped for compaction. The implementation
and evidence are pushed on `fix/color-collapse`, PR #362 remains a draft, and no
merge has been claimed. Finish the selected DINO comparison, update the installed
runtime/launcher with the supported choice, then review and merge to `develop`.

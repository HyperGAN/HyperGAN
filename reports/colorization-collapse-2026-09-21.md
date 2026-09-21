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

Results and final runtime validation will be recorded after the bounded runs.

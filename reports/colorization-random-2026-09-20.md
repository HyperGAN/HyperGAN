# Random-prior GAN and separate logo reconstruction

The stopped `train-color-projected` run collapsed visibly. Its step-1084 checkpoint
routes all 256 sampled training images and all 256 held-out images to particle
2597; both raw and EMA models do this. Step-zero models used 70 and 74 particles
on those respective sets. Particle-table spread remained healthy because it does
not measure encoder usage.

The generator still responds to different centers: random-particle output
pairwise pixel RMS is 0.405 versus 0.00716 for conditional output. These random
outputs are varied textures, not usable logos. A separate eight-image G-only
reconstruction probe reduced MSE from 0.617 to 0.0306 in 200 updates, starting
from the collapsed generator. This establishes small-set capacity, not generalization.
The checkpoint D ranks every collapsed fake above every real in the fixed probe.
Training its existing 6,145-parameter head against saved frozen features separates
the fixed examples: held-out AUC changes from zero to one by 25 updates. This BCE
probe does not reproduce the moving-generator relativistic training dynamics.
Together, these observations localize lost conditioning to E and rule out a
completely constant G or fundamentally inseparable current D features; they do
not establish which live training dynamics initiated collapse.

## Change

```text
z ~ uniform particle prior
D(X), D(G(z))                         # GAN trains D / G / prior
B = BW(X)
B -> E -> hard particle + noise -> frozen G -> X_hat
L2(BW(X_hat), B)                      # trains E only
```

The original adaptation sent only encoder-routed samples to the GAN. Now the GAN
uses independent uniform-prior draws, as ParticleGAN's archived CIFAR
`encoder_only` variant does (`9e9ce96`, `train_cifar_ae_sagan.py`, reconstruction
and training sections). This is a selected reconstruction strategy, not a
requirement of every VAEGAN. Grayscale L2 permits different colors while matching
luminance. The hard posterior retains matching fixed sigma and constant joint
KL `log(4096)`; there is no learned local offset, variance, or optimized KL.

G, D, attention, particle count, optimizer and regularizers are unchanged.
The deterministic condition still passes through a 12-bit hard particle ID.
Neither random GAN coverage nor encoder-only reconstruction guarantees useful
conditional structure or aggregate routing diversity.

Sampling explicitly resolves the conditional reconstruction independently of
GAN generation. The `comparison` image labels aligned `X | B | X_hat` columns,
one example per row; `random` shows uniform-prior images. Existing `g`, `x`, and
`gray` images remain. Preview metadata records particle usage and largest route
share. Inference exports and snapshot metrics retain the encoder/reconstruction
dependencies. All three existing metrics continue to evaluate **conditional**
outputs; edge error is related to grayscale training loss and is not presented
as independent quality evidence.

## Bounded comparison

Three 200-update arms use identical saved step-zero weights, data streams, batch
16 and real DINOv3 on physical GPU1. Each measures the same 64 held-out images
with fixed evaluation draws. CUDA kernels are nondeterministic and the old arm
consumes posterior noise in an additional phase, so trajectories are not bitwise
paired. These are raw-model diagnostics, not EMA checkpoint selection.

| Arm at step 200 | Used particles / 64 | Largest share | Conditional pairwise RMS | Random pairwise RMS | Gray MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoder-routed GAN, RGB L2 | 1 | 100% | 0.0131 | 0.3049 | 0.6074 |
| Random-prior GAN, RGB L2 | 8 | 57.8% | 0.0971 | 0.0491 | 0.6396 |
| Random-prior GAN, grayscale L2 | 4 | 48.4% | 0.2685 | 0.1596 | 0.5092 |

Routing improved transiently (18–25 particles for grayscale at steps50–100), but
by step200 the grayscale previews still show a small family of repetitive
textures. This change must not be described as curing collapse. The separate
random view makes that remaining failure visible.

Machine evidence (saved source checkpoints, scripts, logs, JSON results and PNGs):
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-colorization-collapse/`.
The original run/config/environment are preserved. The fresh experiment lives in
`logos-colorization-random-256/`, uses `colorization-random-env`, and starts through
`training-runs/start-color.sh` on physical card1. An RGB-objective comparison
config is retained alongside it. The owner training run is left unstarted.

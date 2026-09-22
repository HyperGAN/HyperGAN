# TransGAN initialization and collapse review, 2026-09-21

The first production TransGAN/DINOv3 run collapsed by step 400 and stopped at 437.
This review used scalar metrics and immutable step 0/437 checkpoints, without
viewing images, restarting training, or running seed experiments.

| EMA preview step | Full-resolution diversity ratio | Pooled-4 ratio |
| --- | ---: | ---: |
| 100 | 0.316168 | 0.133194 |
| 200 | 0.039966 | 0.009064 |
| 300 | 0.000349 | 0.0000605 |
| 400 | 0.00000922 | 0.00000163 |

The prior was not collapsed: particle-coordinate standard deviation was 1.00123
at step 0 and 1.00323 at 437. Diagnostics used the same first 16 saved particle rows,
without drawing fresh noise. These rows provide a controlled probe rather than
an estimate of the complete MoG sampling distribution.

| Diagnostic | Initial G | Trained G, initial particles | Trained G, current particles |
| --- | ---: | ---: | ---: |
| Output fraction abs(x)>0.99 | 28.077% | 100% | 99.9997% |
| Mean tanh derivative | 0.30805 | 0.000000944 | 0.000001690 |
| Pre-tanh RMS | 2.450 | 24.491 | 24.478 |
| Output pairwise RMS spread | 1.03813 | 0.0000423 | 0.0000747 |

Initial output therefore had substantial pixel variation, despite weak coarse
structure. Training subsequently erased nearly all output variation. Internal
features still varied across latents; huge pre-tanh activations mapped them to
almost identical saturated pixels. Finite losses and gradients alone did not
establish healthy training. The EMA checkpoint was also heavily saturated.

## Initialization mismatch

The earlier HyperGAN HNDL file applied Xavier-uniform to all linear weights.
That was an incorrect reading of the upstream initialization: the official
[training initializer](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/train_derived.py)
applies Xavier to Conv2d, while its Linear branch is commented out. The
[generator-wide initializer call](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/ViT_custom_local544444_256_rp.py)
is also commented out. Thus its stem and transformer linears retain PyTorch
Linear defaults, while the RGB convolution receives Xavier weights.

For HyperGAN's 128->65536 stem, Xavier weight variance 2/(128+65536) is 85.5 times
smaller than default 1/(3*128): approximately 9.25 times smaller standard deviation.
Measured initial stem RMS was 0.06398. Conversely, the attention output and FFN
contraction Xavier weight variances were 3 and 4.8 times their default values.
This is an initialization imbalance, not an HNDL operator defect.

A controlled counterfactual reused exactly the step 0 weights and the same 16
particle rows. Only stem/transformer weight amplitudes were rescaled to default
Linear variances; zero biases, position tables, topology, and RGB weights stayed
fixed. Saturated output fraction fell from 28.077% to 1.566%; mean tanh derivative
rose from 0.30805 to 0.57765. Full-resolution spread changed 1.03813->0.88300;
pooled-4 spread changed 0.19225->0.08443. Reduced saturation is not evidence of
better coarse structure or eventual GAN quality. This isolates an initial-scale
problem; it does not prove that initialization alone caused long-run collapse.

## Correction and validation

The generator HNDL now explicitly uses `kaiming_uniform(a=2.23606797749979)`
for stem and transformer linears, with each bias drawn uniformly from
`[-1/sqrt(fan_in), +1/sqrt(fan_in)]`. This expresses PyTorch's default Linear
initialization in HNDL. The final RGB weights keep Xavier initialization and
its bias uses uniform [-0.25,0.25], matching the upstream 16-channel 1x1 readout.
Position tables remain truncated-normal(std=0.02). Architecture, final tanh,
DINOv3 discriminator, prior, optimizer, and all seed settings are unchanged.

All 15 generator reference tests pass. New regression coverage checks weight
variance, per-layer initialization bounds, default biases, and the initial
stem/residual activation balance on fixed analytical inputs. This catches the
previous undersized stem without imposing an image-quality or seed-dependent
saturation threshold. Existing attention-oracle, gradients, grid ordering,
RNG, checkpoint, and EMA checks still pass. The local recipe validates and
its launcher passes bash syntax checking.

A fresh launch is prepared; it has not been started:

```
~/dev/hypergan/training-runs/start-transgan-dinov3-multidepth-128-init-v2.sh
```

It uses card 0 and batch 64 with the existing editable `transgan-128-env`.
Both HyperGAN and HNDL now load from their local source checkouts. The original
run/config/checkpoints remain intact. Initialization changes require a new run:
resuming the old checkpoint restores its collapsed weights.

Artifacts: `/tmp/hypergan-transgan-collapse-review.py`,
`/tmp/hypergan-transgan-collapse-review.log`,
`/tmp/hypergan-transgan-init-counterfactual.py`,
`/tmp/hypergan-transgan-init-counterfactual.log`.

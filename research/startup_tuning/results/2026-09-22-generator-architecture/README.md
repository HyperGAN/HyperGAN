# Generator architecture screens: both changes still collapse

**Interpretation update:** the later [healthy-control comparison](../2026-09-22-healthy-control/README.md)
shows that CIFAR can recover after severe startup saturation. These measurements
establish poor behavior within 32 updates, not irreversible or persistent failure.
The narrow and pixelshuffle variants have not been extended; only the unchanged
logos/ResNet source receives a 512-update comparison in that follow-up.

Two proposed generator changes were tested with the frozen pretrained ResNet18 critic.
Narrowing the four early FFNs and using pixel shuffle from the first transition both
delay saturation, but neither prevents collapse within 32 updates. Neither was extended.

| Case | Step | Saturation | Diversity RMS | Initial diversity retained | Pre-tanh RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | 0 | 1.010% | 0.551861 | 100.00% | 0.999 |
| source | 1 | 26.310% | 0.333930 | 60.51% | 2.242 |
| source | 8 | 84.090% | 0.107895 | 19.55% | 7.205 |
| source | 16 | 91.356% | 0.082008 | 14.86% | 9.789 |
| source | 32 | 99.653% | 0.005690 | 1.03% | 19.226 |
| narrow | 0 | 1.133% | 0.548784 | 100.00% | 1.011 |
| narrow | 1 | 6.987% | 0.452310 | 82.42% | 1.478 |
| narrow | 8 | 78.764% | 0.137719 | 25.10% | 4.586 |
| narrow | 16 | 91.085% | 0.073488 | 13.39% | 6.485 |
| narrow | 32 | 93.675% | 0.059346 | 10.81% | 7.817 |
| pixelshuffle | 0 | 3.822% | 0.540836 | 100.00% | 1.299 |
| pixelshuffle | 1 | 2.938% | 0.549720 | 101.64% | 1.231 |
| pixelshuffle | 8 | 66.376% | 0.184822 | 34.17% | 3.881 |
| pixelshuffle | 16 | 81.864% | 0.129954 | 24.03% | 5.232 |
| pixelshuffle | 32 | 89.050% | 0.067511 | 12.48% | 6.092 |

Measurements are online-G outputs on the same 64-example monitor bank with evolving
prior coordinates and fixed particle IDs/noise. Saturation means abs(output) > 0.99;
diversity is population RMS spread across images, not pairwise preview diversity.
Retention here is relative to each model at initialization, not relative to real images.
These diagnostics establish collapse, not perceptual sample quality.

## What the stage measurements add

The latent signal is not simply gone before the output. Much of the growth is a
component shared across the monitored images, while absolute between-image variation
stays substantial through the hidden stages. The output projection yields large
pre-tanh values, and tanh greatly reduces output variation.

| Case, step 32 | Stage | Activation RMS | Between-image RMS | Variation / activation |
| --- | --- | ---: | ---: | ---: |
| source | stage8_block1 | 2.8978 | 0.6476 | 0.2235 |
| source | stage16_block1 | 6.4974 | 0.5713 | 0.0879 |
| source | stage128_unwindows | 6.8930 | 0.5721 | 0.0830 |
| source | output_projection | 19.2261 | 0.7441 | 0.0387 |
| narrow | stage8_block1 | 0.8373 | 0.6635 | 0.7925 |
| narrow | stage16_block1 | 2.6512 | 0.5992 | 0.2260 |
| narrow | stage128_unwindows | 3.1269 | 0.6071 | 0.1941 |
| narrow | output_projection | 7.8168 | 0.7699 | 0.0985 |
| pixelshuffle | stage8_block1 | 3.7408 | 0.6432 | 0.1719 |
| pixelshuffle | stage16_block1 | 4.2528 | 0.6454 | 0.1517 |
| pixelshuffle | stage128_unwindows | 4.5917 | 0.6563 | 0.1429 |
| pixelshuffle | output_projection | 6.0917 | 0.7611 | 0.1249 |

The source stage16 RMS grows from 0.671 to 6.497, while its diversity changes from
0.640 to 0.571. Its pre-tanh diversity is still 0.744 at step 32, but output diversity
is only 0.00569. The narrow model reduces stage16 RMS to 2.651 and pre-tanh RMS to
7.817, yet still clips 93.7% of outputs. Pixel shuffle preserves RMS/diversity exactly
at its permutation, but cannot remove the shared component already built at stage8.
These observations do not isolate attention, FFN, output projection, or optimizer
as the sole cause; they identify activation growth and output clipping as a more
specific target for the next intervention.

## Controlled changes and initialization

- Source: original ResNet128 recipe, repeated only to add stage diagnostics.
- Narrow: stage8/stage16 FFNs 4096 -> 1024; corresponding down-bias fan-in bounds updated.
  All channel widths, attention, upsampling and later stages remain identical.
- Pixelshuffle: an independent comparison against source, not combined with narrow.
  Replace 8->16 bicubic with pixel shuffle, with channels 1024/256/64/16/4 and FFNs
  4096/1024/256/64/16. This also changes later capacity, head dimensions and initializer
  scales; it does not isolate interpolation alone.

Both variants copy unchanged tensors from the original seeded CPU initialization,
including every critic tensor. Changed tensors use leading slices and declared
fan-in/Xavier rescaling; positions retain their distribution. All copied values
are checked exactly. Narrow preserves 301 tensors and transforms 16; pixelshuffle
preserves 239 and transforms 78. This pairs random draws, not initial functions.
Every report records transformations and matching source/target hashes.

The data, learned prior, source G/D rates 2e-4, prior rate 0.002, Adam betas
[0.5, 0.999], batch 64, seed 25002, lazy penalty schedule and 200000-step annealing
horizon are unchanged. No learning-rate tuning or seed sweep. Explicit monitor bank
and measurement RNG hashes match across all three cases. Each fixed-latent control
also collapses; this does not exclude prior effects during training.

Configured nondeterministic CUDA/TF32 execution remains enabled. The original
ResNet screen ended at 97.14% saturation, while this instrumented source replay ends
at 99.65%. Initial parameter/bank identities match the prior run, but numerical
trajectories are not asserted identical. Both reproduce the same failure. Do not
interpret small between-run differences as an estimated treatment effect.

## Validation and artifacts

All runs completed 32 updates and restored full trainer state. Frozen/protected
pretrained hashes match before, after and after restoration. Every recorded lazy
penalty application is zero. The CPU stage-hook regression confirms identical
updates and final parameters with/without hooks. Six probe tests plus ten pretrained
provider tests passed. The pixelshuffle network also passed a CPU finite-output
128px shape check and all tensor-alignment assertions.

Runs used GPU 0 (GPU-ed080e41-3193-3755-6756-f3d46c433331) only, with the specified
transgan-128-env Python and worktree PYTHONPATH. No ordinary training job or checkpoint
was created. GPU 1 and training-runs source TOMLs were untouched. PR #382 is unmerged.

Reports: [source](source/report.json), [narrow](narrow/report.json),
[pixelshuffle](pixelshuffle/report.json). Each directory includes its exact runner
snapshot, TOML, and resolved HNDL recipe. The recipes live under
`testbeds/transgan-resnet128`, `testbeds/transgan-resnet128-narrow-ffn`, and
`testbeds/transgan-resnet128-pixelshuffle`.

Runner: `research/startup_tuning/architecture_screen.py --case CASE --device cuda:0
--output-root OUTPUT`, with CUDA_VISIBLE_DEVICES set to the free GPU UUID.
Ordinary TOML training does not perform the paired weight alignment.

Original artifacts: `/mnt/ml7tb/hypergan-signal-research/transgan128-ffn-width-v1`.
Elapsed seconds including diagnostics: source 98.53, narrow 93.98, pixelshuffle 95.66.

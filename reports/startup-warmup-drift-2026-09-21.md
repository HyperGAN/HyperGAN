# Saturation during the 1,000-step G learning-rate ramp

The user launched `transgan-tuned-warmup-1000-live` on GPU 1 and reported
increasing G loss and extreme colored tiles from the TransGAN generator.
They subsequently localized the visible deterioration to approximately steps
350–500.
Startup tuning selected a **0.1 G rate factor**: 0.00002 instead of 0.0002.
The requested warmup then increased that rate linearly back to 0.0002 at
update 1,000. D stayed at 0.0002 and the learned prior at 0.002; the configured
annealing multiplier is one. The run used commit `2a9a8909` and seed 25002.

Read-only CPU measurements support severe output saturation during the ramp.
They do **not** establish that a smaller constant rate would prevent it.
No training or GPU probe was launched, stopped, or modified for this analysis.

## Saved EMA preview measurements

`preview_saturation_probe.py` verifies saved PNG hashes, separates the grid into
its 16 generated images, and measures values mapped back to [-1,1]. Saturation
below counts individual color-channel values with absolute value at least .98.
These are quantized **EMA previews**, not exact online activations or gradients.

| Step | G learning rate | Near-limit channel values | Across-sample RMS spread | Mean-color share of between-sample variance |
| --- | ---: | ---: | ---: | ---: |
| 0 | .00002000 | 0.16% | .4417 | 0.17% |
| 100 | .00003784 | 0.77% | .3870 | 0.31% |
| 200 | .00005586 | 18.37% | .1936 | 1.17% |
| 300 | .00007387 | 4.84% | .4526 | 1.07% |
| 400 | .00009189 | 26.53% | .7644 | 5.77% |
| 500 | .00010991 | 75.40% | .7384 | 18.44% |
| 600 | .00012793 | 96.33% | .9321 | 38.15% |
| 900 | .00018198 | 98.65% | .9919 | 89.92% |
| 1000 | .00020000 | 97.11% | .9875 | 92.78% |
| 1100 | .00020000 | 97.79% | .9279 | 80.79% |

Saturation rises sharply between preview steps 400 and 600. The earlier
step-200 increase recedes at step 300, so this is not a monotonic trajectory.
EMA lag and 100-step preview spacing prevent identifying the exact first online
failure step or a safe learning-rate threshold.

The preview spread grows even as the outputs deteriorate. An exact population
variance decomposition shows that, by step 1,000, 92.78% of between-sample
variance comes from differences in each image's mean RGB color. This is a
concrete reason that output spread alone cannot certify useful diversity.
It does not identify the spatial tile mechanism, and the decomposition is not
a semantic quality metric. The generator source contains windowed attention,
pixel-shuffle stages, and a final tanh, but this observation does not isolate
one of them as the cause.

At step 1,000, logged G adversarial loss is 12.9528 and D adversarial loss is
0.0001875. Large G loss is not itself a measurement of large parameter gradients.
The software remained running without recorded observation errors at the audit
snapshot; the reported failure concerns training behavior.

## Saved parameter checks

`checkpoint_parameter_drift.py` compared complete online checkpoints at
0, 250, 500, 750 and 1,000. All 175 pretrained backbone parameter tensors remain
bitwise unchanged. This audit does not cover backbone buffers. G cumulative
relative L2 displacement rises from 4.08% at step 250 to 20.04% at step 1,000;
discriminator feature-head displacement reaches 20.46%, and prior displacement
14.36%. These are cumulative tensor changes, not update sizes or functional
distances, and do not isolate G from D/prior dynamics.

## Next discriminating comparison

Hold the selected G rate with `--tune --tune-warmup-steps 0`, keep D/prior rates
and the configured seed unchanged, and observe beyond the same failure window.
Use a new run directory. Confirm its selected rate and starting state before
treating it as a matched comparison: same seed alone does not guarantee the
same tensors under nondeterministic execution. The previous constant-rate
evidence only extended to about 100 updates and cannot settle this question.

The current observation makes excessive G step size a leading hypothesis, not
an established complete explanation. Extending warmup automatically or lowering
all optimizer rates together would not distinguish that hypothesis cleanly.
No default or tuning policy was changed by this investigation.

A separate local launcher,
`~/dev/hypergan/training-runs/start-transgan-dinov3-multidepth-128-tuned-fixed-lr.sh`,
is prepared for the user to launch on GPU 1 after the current run stops. It uses
the same source configuration and seed, disables warmup, and stops after 1,200
retained updates. Checkpoints are saved every 100 steps to inspect the reported
350–500 failure window. It has not been launched by this investigation.

The [measurement JSON](startup-warmup-drift-2026-09-21.json) records the run,
artifact hashes, preview statistics, losses, and parameter displacement. Raw
audits are under `/mnt/ml7tb/hypergan-signal-research/` with the run name prefix.

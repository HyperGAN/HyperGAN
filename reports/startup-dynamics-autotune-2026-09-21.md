# Bounded startup dynamics tuning

The initialization-only check missed the early saturation documented in the
[matched startup investigation](startup-signal-drift-2026-09-21.md). `--tune`
now follows initialization calibration with eight disposable configured training
updates. It measures retention of output variation and first-layer structural
transmission on two matched probe batches. A failing finite retention below
0.25 proposes one smaller G learning-rate factor:

```text
factor = clip(minimum_retention, 0.1, 0.5)
```

One eight-update confirmation can accept that proposal. This is a heuristic
feedback rule, not an optimal-rate derivation. It has no retry loop or rate grid.
A passing configured rate needs only eight updates; the maximum is sixteen.
Snapshots, hashes, initialization checks and signal probes add time and memory
beyond that update budget. See [usage and safeguards](../docs/initialization-tuning.md).

## Initial GPU evidence

The original TransGAN/DINOv3 128px configuration, batch size and seed were used
on physical GPU 1. The first successful startup used additional per-tensor hashes
in the diagnostic, preserving its original aggregate hash and strict rejection.
The run is `/mnt/ml7tb/hypergan-signal-research/transgan-auto-debug-state`.
The companion [JSON](startup-dynamics-autotune-2026-09-21.json) records the
decision, retained rates, measurements and protected-state verification.

| Retention after eight updates | Configured rate | Formula-derived rate |
| --- | ---: | ---: |
| Sample diversity, bank 0 | 0.23024 | 0.71386 |
| Sample diversity, bank 1 | 0.23219 | 0.71959 |
| First-layer relative cotangent gain, bank 0 | 0.11776 | 0.50591 |
| First-layer relative cotangent gain, bank 1 | 0.11899 | 0.51327 |

The selected factor was **0.11775735815588224**, changing G's rate from
0.0002 to 0.00002355147163117645. D remained at 0.0002 and the prior at 0.002.
The dynamics phase took **62.815 seconds** and discarded sixteen updates.
The complete tuning event span, including initialization checks, was **93.241 seconds**.
Complete trial state was verified restored to step zero; protected tensor hashes
matched before, during and after the trials. This is one testbed timing, not a
hardware-independent startup-cost guarantee.

The step-zero G/D graph, prior, EMA graph, random streams and global RNG are
bitwise equal to the earlier initialization-only baseline. Normal CLI resume
restored the selected optimizer rates without recalibration or compounding the
factor. Its newly written step-one checkpoint preserves those exact rates. Resume
completed 100 additional updates, stopping at step 101, with checkpoints at
steps 20 and 100 available for the matched drift comparison.

A subsequent **normal CLI** startup on commit `afd03e77`, with the enhanced
audit diagnostics and previews enabled, also completed both phases. It selected
factor **0.11579168882558553**, restored all sixteen trial updates, saved the
step-zero checkpoint and published its step-zero preview and metric. Total
tuning time was **93.925 seconds**, including **62.763 seconds** for dynamics.
This validates the normal command path on this testbed; the earlier intermittent
audit failures below still have no identified cause. This execution check used
the original seed and was not a seed sweep.

## Matched checkpoint outcome

The automatically selected 0.117757x run was probed at steps 0, 20 and 100
using the existing read-only research script. All six probes passed state
verification. Real-image bank hashes match the earlier initialization-only and
manual 0.1x comparisons; frozen/pretrained hashes match across all three runs
and all checkpoints. Initial generated-output statistics also match exactly.

| Online measurement | Init-only, step 20 | Auto, step 20 | Init-only, step 100 | Auto, step 100 |
| --- | ---: | ---: | ---: | ---: |
| Fraction of output values with abs(value) > .99 | 97.41% | 15.82% | 93.31% | 12.31% |
| Across-sample output standard-deviation RMS | .02923 | .23689 | .06931 | .28993 |
| Across-sample spread after 4x4 pooling | .001968 | .01936 | .004954 | .08641 |
| First observed G layer / image gradient RMS | .03584 | .32874 | .07179 | 2.96670 |

These measurements fix particle IDs and Gaussian noise. The fixed-latent control
also shows the improvement: automatic step-100 saturation is 12.32% and output
spread is .28991, so evolving prior coordinates do not explain the result.
The 100-step follow-up supports improvement of the measured early saturation
and attenuation, beyond the eight-update selection window. It is still the same
testbed and matched bank, not independent quality evaluation.

The earlier manual 0.1x run has step-100 saturation 26.85% and output spread
.77892. Automatic tuning lowers saturation further but has less output spread
than that manual trial. Neither statistic determines sample quality, so these
results do not establish that the automatic rate is better than the manual one.
The later normal CLI run validates startup and the preview path only; it was
stopped at step one and is not the source of these 100-step measurements.

Probe artifacts: `/mnt/ml7tb/hypergan-signal-research/transgan-auto-dynamics-probes`.
The companion JSON includes both latent controls, baseline comparisons and
state hashes. No images were used to select the rate or evaluate this result.

## Audit failures and limits

Two preceding ordinary CLI startups stopped before dynamics trials: one detected
a parameter hash change during a structural initialization probe; the second
detected changed registered state during a generator-objective probe. The runs
are `transgan-init-v2-auto-dynamics-100` and
`transgan-init-v2-auto-dynamics-100-v2` under the research artifact root. Startup
refused these probes and rolled back. They are not successful tuning trials.

Additional per-tensor audit diagnostics passed, but change allocation and timing;
that does not identify the original failure's cause. Errors now identify changed
registered tensor paths while retaining the exact aggregate guard. No mismatch
is accepted, retried away, or classified as harmless.

The retention guards cannot certify semantic diversity, useful gradient
directions, image quality, convergence or long-term stability. The controlled
manual 0.1x experiment motivated the implementation; it is not independent
validation of the automatic formula. Tuning remains opt-in. PR #382 remains
unmerged with automatic merging disabled for user testing.

## Automated validation

Sixty focused tests passed for initialization, bounded dynamics, persistence,
override recovery and lifecycle. Coverage includes real optimizer-driven
saturation, baseline abstention, failed single confirmation, complete state
restoration, mutable data-state ownership, pretrained storage aliases, callback
and persistence failures, and validated rate recovery. UI and console tests
cover trial progress and selected, unchanged, unresolved and skipped outcomes.

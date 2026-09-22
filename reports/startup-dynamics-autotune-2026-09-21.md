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
Complete trial state was verified restored to step zero; protected tensor hashes
matched before, during and after the trials. This is one testbed timing, not a
hardware-independent startup-cost guarantee.

The step-zero G/D graph, prior, EMA graph, random streams and global RNG are
bitwise equal to the earlier initialization-only baseline. Normal CLI resume
restored the selected optimizer rates without recalibration or compounding the
factor. Its newly written step-one checkpoint preserves those exact rates.

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

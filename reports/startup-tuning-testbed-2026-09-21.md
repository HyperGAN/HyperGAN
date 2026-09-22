# Startup tuning: implementation and local testbeds

The experimental `hypergan train CONFIG --run-dir NEW_RUN --tune` workflow
searches a small set of owned generator boundary-layer scales before the first
optimizer update. It remains opt-in: the measurements below establish local
behavior and state preservation, not improved GAN convergence or sample quality.
See [usage and limits](../docs/initialization-tuning.md).

## Protocol

Both user-supplied recipes used their original configured batch size of 64,
unchanged seeds, and physical GPU 1. No seed experiments were run. The original
TOML/HNDL files and existing training runs were preserved. Each tuned trial used
a separate run directory, saved its step-zero full checkpoint, and completed one
ordinary training update. The 128px trial then resumed for one additional update
with `--tune` still present to verify that calibration did not repeat.

The fixed output-cotangent heuristic measures relative sensitivity at the first
and final eligible affine boundaries. Its score is the mean absolute base-10
logarithm of the reported relative gains; smaller means closer to the declared
unit-gain target. It is architecture-dependent and is not an objective measure
of the usefulness of the discriminator's signal. The actual configured
adversarial gradient is measured separately.

## Observed startup decisions

| Measurement | TransGAN DINOv3 multidepth 128 init-v2 | Working CIFAR TransGAN 32 |
| --- | --- | --- |
| Decision | Scale output projection weight and bias by 0.70103694 | Keep baseline |
| Search-batch heuristic, before → after | 0.03111725 → 0.01168361 | 0.0246811 → unchanged |
| Held-out heuristic, before → after | 0.0304872 → 0.0113676 | No candidate selected |
| Output RMS, before → after | 0.622485 → 0.515979 | 0.64469 → unchanged |
| Sample diversity RMS, before → after | 0.551861 → 0.455321 | 0.556999 → unchanged |
| Fraction of absolute outputs above 0.99 | 1.01045% → 0.03732% | 1.33108% → unchanged |
| Protected tensor hashes | Exact match | Exact match |

The output rescaling reduced the measured saturation proxy and also reduced
output variation. Both passed the explicitly bounded scale/diversity guards;
neither observation establishes improved visual quality. The selected 128px
change also passed the second batch check. The healthy CIFAR control received
no change, which is a useful abstention check.

Initial adversarial-only diagnostics gave generated-output gradient RMS
2.241878e-6 and first/output gradient RMS ratio 0.927510 for the 128px recipe,
versus 1.370378e-5 and 0.475714 for the working CIFAR recipe. The working recipe's
smaller ratio is direct evidence against treating this raw ratio as a universal
ranking. Neither initial profile showed wholesale disappearance of the
generator signal. Attention query/key subpaths can be much weaker without that
alone diagnosing training failure.

## Artifacts and safety checks

Local run roots under `~/dev/hypergan/training-runs/`:

- `train-transgan-init-v2-tune-proof`
- `train-cifar-transgan-32-tune-proof`

Each contains `tuning/config.base.json`, `tuning/overrides.json`, and
`tuning/report.json`, plus full training checkpoints. The selected generator
weights and matching EMA initialization are saved in the initial checkpoint.
Resume restores that checkpoint rather than replaying JSON scale factors.
The 128px resume reached step 2 with no additional tuning events.

Deterministic regression tests cover saturated toy generators, held-out
confirmation, nonfinite gradients, protected pretrained/frozen/shared tensors,
RNG/data restoration, artifact failure rollback, lifecycle ordering, resume
behavior, console output, and dashboard rendering. An analytic saturated-critic
case has nonzero score gradient and exactly zero image gradient despite healthy
generator transmission. Checkpoint diagnosis tests validate online G/D loading,
source-file invariance, explicit older selection, and integrity rejection.
The final focused packaging/CLI/lifecycle/algorithm/checkpoint/web suite passed
93 tests (5 heavy tests deselected). Separate Chromium rendering tests passed
7 tests, and CPU-to-GPU1/GPU1-to-CPU checkpoint remapping passed 2 tests.
The real 128px step-2 checkpoint diagnostic completed in 15.44 seconds, reported
330 discriminator boundary rows, and verified unchanged registered model state.
Successive checkpoint probes use each checkpoint's saved next draw, so their
differences are not by themselves a matched-input measure of signal drift.

Automatic changes remain generator-only. Discriminator profiles help locate
signal loss; arbitrarily increasing discriminator magnitude would not establish
better guidance. Longer user testing is needed to assess drift and training
outcomes before changing defaults. This implementation PR is intentionally left
unmerged for that testing.

## User training follow-up

The user's separate `train-transgan-dinov3-multidepth-128-init-v2-tuned` run
successfully applied calibration but **has not resolved the reported training
problem**. Reported losses were D=0.00190275/G=6.38079 at step20,
D=0.000084597/G=9.35966 at step100, and D=0.0225799/G=9.81009 at step120.
The user explicitly reported that it was not fixed. Passing the startup
transmission heuristic must not be presented as fixing training behavior.

Next investigation: compare generator and discriminator boundary measurements
at initialization and later checkpoints, using matched probe inputs where
possible, to locate changes in signal transmission and distinguish them from
loss magnitude. Do not assume the losses alone prove vanishing gradients.
The active user run should continue undisturbed; GPU1 is now occupied by it.
Pretrained weights remain protected; no seed experiments, no automatic PR merge.

The user requested a step-zero sample for comparison. New runs with previews
enabled will capture a baseline after calibration and before optimizer updates,
in addition to their regular preview cadence. Existing processes keep the code
they started with and are not retroactively modified.

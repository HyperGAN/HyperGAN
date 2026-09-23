# The startup pass did not validate the selected learning rate

The user reported that the long test did not look promising and questioned the
very small G learning rate. Read-only inspection through step 500 supports that
concern. **The selector remains unsuccessful as a demonstrated training
calibration.** The earlier eight-update pass must not be presented as a useful
rate recommendation. PR #382 remains unmerged.

Run: `/mnt/ml7tb/hypergan-signal-research/transgan-gradient-response-startup-v1`.
[Scalar evidence](gradient-response-long-run-review-2026-09-22.json) includes
saved-preview statistics, checkpoint parameter changes, and hashes/paths of the
underlying audit outputs. The running experiment was not changed or stopped.
No additional GPU work or seed experiments were performed.

## What happened

The selected G learning rate, 9.75455e-7, is **205 times smaller** than the source
2e-4. D remained at 2e-4 because its two proposal banks did not resolve a valid
reduction. The prior stayed at 2e-3. Nominal LR ratios alone do not establish
functional learning balance, particularly with Adam and different parameter groups.

| Saved EMA preview | Step 100 | Step 300 | Step 500 |
| --- | ---: | ---: | ---: |
| Fraction of color channels with absolute value > .99 | .875% | 3.370% | 21.596% |
| Between-sample variation RMS | .54556 | .45009 | .23266 |
| Spatial per-channel variation RMS | .60425 | .49579 | .24736 |

By step 500, between-sample variation had fallen to 42.6% of its step-100 value.
The measurements are from quantized saved EMA previews using the evolving prior,
not online gradients, and do not provide semantic quality scores. They nevertheless
show that the startup pass did not prevent later saturation and loss of variation.
The user also directly reported poor-looking samples.

G total loss was 11.30 at step 100, 13.99 at 300, 16.13 at 400, and 8.63 at 500.
D adversarial loss was 4.22e-5, 3.63e-6, .0104, and .000583 respectively. These
losses are not monotonic quality measurements or causal proof of one failure mechanism.

The generator is not literally frozen: all 109 G parameter tensors changed.
At step 500, cumulative relative parameter L2 movement was .00451 for G,
.07047 for owned pixel/other D parameters, .09739 for feature heads, and .17559
for the prior. Those norms cannot be directly interpreted as relative functional
learning rates, but they underscore why fixed-initial-latent startup probes cannot
certify the evolving full training process. All 175 pretrained backbone parameter
tensors remained byte-equal; this audit did not compare backbone buffers.

## Why this rate is not a credible north star

The selector uses a gradient-norm product to estimate a conservative response
scale. On its limiting G bank, the Adam-metric scale C was 45.30, versus a
signed directional secant of -1.87. C is 24.2 times the magnitude of that signed
projection. These are different quantities: the norm product captures changes
orthogonal to the update too. Its small factor is not evidence that 9.75e-7 is
an optimal or necessary G learning rate. Negative signed curvature does not
supply an alternative positive Newton rate either.

More fundamentally, the acceptance checks rewarded preserving initial variation
and signal transmission over eight updates, with only nonzero parameter motion
and local held-out loss decrease required. Very small updates can satisfy those
conditions without establishing useful progress under a moving discriminator and
learned prior. This is a design limitation, not a reason to loosen the retention
threshold or choose a favorable minimum LR after seeing the result.

The paired startup pass also did not validate a D reduction: D remained unresolved.
Reducing G alone by 205 times while retaining D's configured rate should not be
communicated as established G/D balance.

## Consequence

Treat this as failed longer-run validation of the current selector. Do not promote
it to default tuning or merge the PR on the strength of the startup checks.
The next research step needs a progress-sensitive check of the coupled process,
including current-prior samples, and a controlled comparison against moderate
joint rate changes. A universal safe gradient-norm ratio has not been established.
No replacement rate is claimed by this audit.

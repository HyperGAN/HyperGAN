# Gradient-response calibration: accepted startup, ready for a longer test

Implementation: `d0469c0b`. PR #382 remains open and unmerged; auto-merge is off.
[Condensed evidence](gradient-response-startup-2026-09-22.json) records the full
report path, SHA-256, clean implementation revision, and all proposal decisions.

**This testbed now passes the bounded startup checks.** It selected G factor
**0.004877275373811331**, giving G LR **9.754550747622662e-7**. D's two banks did
not agree on a valid reduction, so D stayed at **0.0002**, and the learned prior
stayed at **0.002**. This is readiness for a longer experiment, not evidence that
the selected rate is optimal or that useful samples will learn quickly.

## Why the loss-quadratic selector was replaced

The [first-update correction](first-update-calibration-2026-09-22.md) fixed the
G anchor and its prior inputs but still chose a large, rejected update. We then
measured exact matching-bank directional derivatives at the first G anchor.
The [fixed diagnostic](first-update-direction-audit-2026-09-22.json) used two
fresh banks after one disposable update, initial-prior latents, and factors
0, .01, .1, .5, 1. It was not a best-point search and made no rate decision.

At bank zero the derivative was -0.46109 initially, -0.46882 at .01 and -0.57754
at .1. Bank one was also increasingly negative. Local directional curvature was
negative, so a positive-curvature Newton step could not justify a smaller rate.
Taking an absolute value of that scalar curvature would not fix its meaning.

The [next diagnostic](first-update-gradient-field-2026-09-22.json), also only one
discarded update, measured the full change of gradient at a prespecified small
interval h=0.04229. In Adam's metric the response scale was 147.29/125.26 across
its banks, while signed directional curvature was only -3.13/-2.26. Much of the
gradient change was not captured by its projection onto the update direction.
These diagnostic banks differ from the production banks reserved after eight
baseline updates; their proposed factors are not expected to be identical.

## One replacement formula

For each player's actual update delta, retain the phase snapshot and the Adam
second-moment diagonal immediately after that update. Let M be
sqrt(bias-corrected second moment) + optimizer epsilon. Hold M, opponent,
prior, batch, buffers, and sampling randomness fixed between probes:

```text
h = min(.1, sqrt(machine_epsilon) * max(norm(theta), norm(delta)) / norm(delta))
g0 = grad L(theta)
gh = grad L(theta + h*delta)
a = g0 dot delta
C = sqrt(sum(delta^2 * M)) * sqrt(sum((gh-g0)^2 / M)) / h
factor = min(1, -a / (2*C))
```

Both banks must resolve a negative a and nonzero vector response. Use their
smaller positive factor; abstain on invalid/unsupported measurements. The
interval h is a coordinate-dependent numerical heuristic; actual representable
parameter movement is checked. Arithmetic guards are not bounds on all TF32,
backpropagation, or stochastic error. There is no floor rounding factors up to
0.1. New schema-three rate metadata supports finite positive factors at most one;
historical schema-one/two checkpoints retain their original validation rules.

Cauchy bounds the magnitude of the *observed directional secant* by C, including
when signed curvature is negative. C is not a certified smoothness bound over a
neighborhood or the proposed step. For ordinary gradient descent the formula
reduces to the curvature term norm(dx)/(2*norm(dg)) in
[Malitsky and Mishchenko, ICML 2020](https://proceedings.mlr.press/v119/malitsky20a.html).
Their algorithm also limits step growth and analyzes convex optimization. Our
startup-only adaptation to held-out Adam directions and GANs does not inherit
that convergence result. The half margin was chosen before the GPU field test,
not adjusted to obtain a passing result.

The old loss-value quadratic selector is removed. There is one proposal method,
two separate validation banks, and at most one coupled replay. No weight-scale
search, rate grid, fallback proposal, new regularizer, or online tuning loop.
G still anchors its first update; D anchors update eight with its configured
lazy penalty. Normal training keeps that lazy schedule.

## Original TransGAN/DINOv3 startup result

Run: `/mnt/ml7tb/hypergan-signal-research/transgan-gradient-response-startup-v1`.
Physical GPU 1: `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`.
Original init-v2 config, source weights, batch 64, seeds 25002/25003, source
rates, learned prior, and backend settings were retained. This was a baseline
versus a derived intervention, not a seed sweep. Source files were not modified.
The configured backend is nondeterministic; compare paired measurements within
this run rather than attributing historical baseline differences to one change.

| Measurement after eight disposable updates | Configured baseline | Selected replay |
| --- | ---: | ---: |
| G factor | 1 | 0.0048773 |
| D factor | 1 | 1, unresolved reduction |
| Sample-diversity retention, two banks | .14158 / .14103 | 1.00235 / 1.00245 |
| First-layer structural transmission retention | .08419 / .08373 | 1.00251 / 1.00223 |
| Pixel-channel fraction with absolute value > .99 | 91.09% / 91.11% | .950% / .913% |
| First-G fixed-latent output displacement RMS | .90824 | .01134 |
| First-G pre-tanh displacement RMS | 2.29293 | .01663 |

G's two bank factors were .0048773 and .0079440. Its held-out losses decreased
by .0021493 and .0021636, over 1,100 times the reported arithmetic resolution.
The first replay update's output change was 1.83% of initial output RMS;
pre-tanh change was 1.67% of initial activation RMS. These are observations,
not target ratios used by the selector. G made nonzero cumulative parameter
movement (relative L2 .0001577); the selected step did not round away.

D's first bank suggested factor .06506, but the second had a positive matching-bank
slope and was unresolved. The policy therefore retained configured D. D did not
pass a held-out reduction test; the unchanged D rate was only included in the
accepted coupled replay. End replay losses were G=5.4818 and D=.007049. Those
losses and an eight-update signal check do not establish balanced long-term
learning or sample quality.

Calibration took **121.41 seconds**: 16 discarded complete updates, eight
phase-loss/gradient evaluations for the response estimates, four held-out G
loss evaluations, eight G observation forwards, two fixed-image generations,
four D input backwards, and six each of structural and objective-signal probes.
No additional candidates were tried.

All disposable training state was restored. Frozen/pretrained parameter and
buffer hashes matched before measurement, through completion, and after rollback:
`01d54beef4f137b8e502c67dc7896e6a619eef9fe2c1e67c0820aa906e49fa9c`.
Only accepted G rate metadata was applied. The CLI then completed one retained
update and saved full checkpoints zero and one. A separate native GPU restore
loaded checkpoint one with the exact G/D/prior rates above, no warmup, and zero
additional updates. There is no retuning or compounding on resume.

The retention banks fix the initial prior's latent coordinates, isolating G's
transmission. They do not certify the distribution generated by the evolving
learned prior. Its relative parameter displacement in the replay was .0126,
versus .0001577 for G. The longer run must evaluate current-prior samples as
well as fixed-bank signal diagnostics. The conservative G rate may be slow.

## Run the longer test

The prepared executable resumes this exact accepted checkpoint on GPU 1:

```sh
~/dev/hypergan/training-runs/start-transgan-dinov3-multidepth-128-gradient-response.sh
```

It enables the viewer, checkpoints/previews every 100 steps, progress every 20,
and stops after **1,200 additional updates** (step 1,201 from the current step-one
checkpoint). The viewer URL is printed on startup. The launcher was checked with
`bash -n` and its real `--help` path; longer training has not been started.
This crosses the previous 350–500-step failure window and deliberately tests
learning speed and sample behavior beyond the startup checks.

For a separate fresh configuration, the normal interface remains
`hypergan train CONFIG --run-dir NEW_RUN --tune`.

## Verification

- 108 focused tuning, probe, numerical, persistence, and lifecycle tests passed.
- 92 rate-recovery and historical-warmup tests passed, including factors below
  .1, exact resume behavior, and rejection of zero/nonfinite/underflowed rates.
- 68 numerical/console checks and the measured-stage Chromium test passed;
  the frontend production bundle was rebuilt and verified.
- GPU startup, protected-state audit, rollback, and native checkpoint restoration
  passed. GPU 1 is released for the user to start the longer test.

These test groups overlap; their counts are not an aggregate total.

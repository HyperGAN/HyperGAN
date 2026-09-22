# First-update calibration: timing fixed, testbed still unresolved

Implementation: `c14faa85`. PR #382 remains open and unmerged, with auto-merge disabled.
[Condensed scalar evidence](first-update-calibration-2026-09-22.json) records the
full report path, SHA-256, clean implementation commit, and measured decisions.

**The G anchor now measures the first actual Adam update correctly. This change
still does not produce an accepted rate pair on the TransGAN/DINOv3 testbed.**
The proposed pair passed both players' held-out loss checks, but failed the
coupled replay's first-layer structural transmission guard. The tuner rejected
it and restored the source rates. No successful long-run calibration is claimed.

## What changed

- Capture G immediately before its first optimizer update, after the first D
  update. Preserve that exact parameter displacement and phase snapshot instead
  of replacing them with update eight.
- Keep D at update eight. Its objective uses the configured lazy schedule; on
  this testbed `lazy_k=8` makes that measurement penalty-active, including the
  multiplier. Normal training remains lazy.
- Reserve four real batches after the baseline. Materialize fixed latent banks
  separately under each phase's prior, using the same tail prior-sampling RNG.
  G uses the initial learned prior; D uses the prior after seven G/prior updates.
  Fit banks and validation banks stay separate. Initial, baseline-end, and
  replay-end guards reuse the G anchor's initial-prior validation latents.
- Record final owned affine activation displacement at updates one and eight
  using the existing matched G forwards. Follow the native output dependency
  through supported layout operations and optional tanh; skip ambiguous paths.
  On this model the measurement is the RGB projection before tanh. It is a
  diagnostic only, with no invented response threshold used to select rates.

The selector still fits one quadratic per player/bank at factors `0, .5, 1`,
forms one bounded pair, checks held-out decrease, and allows one discarded
coupled replay. No additional candidates, backward evaluations, regularizers,
weight rescaling, warmup, or online tuning were added.

## Testbed and controls

Original source:
`~/dev/hypergan/training-runs/logos-transgan-dinov3-multidepth-128-init-v2/transgan-dinov3-multidepth.toml`.

Run: `/mnt/ml7tb/hypergan-signal-research/transgan-first-update-response-smoke-v1`.
Adjacent log: `transgan-first-update-response-smoke-v1.log`.
Physical GPU 1: `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`.

Original batch 64, 128px TransGAN, frozen DINOv3, owned discriminator heads,
G/D rates 0.0002, prior rate 0.002, seeds 25002/25003, and backend settings were
retained. This compares a baseline with a derived rate intervention; it is not
a seed sweep. Source files were unchanged. The normal CLI used `--tune
--no-server --no-previews --stop-after-steps 1`, completed normally, and saved
full checkpoints at zero and one. No long experiment was started after the
startup proposal failed.

Phase-local latent materialization and guard inputs changed relative to the
previous implementation. The configured GPU backend is nondeterministic, and
the baseline trajectory also differs from the previous run. Historical scalar
differences therefore cannot be attributed solely to moving the G anchor.
Use the baseline and replay within this run to assess its proposal.

## Measurements

| Measurement | Configured baseline | Proposed pair replay |
| --- | ---: | ---: |
| G factor | 1 | 0.86545 |
| D factor | 1 | 0.47297 |
| First G output displacement RMS | 0.90819 | 0.80459 |
| First G output displacement / preceding RMS | 1.46274 | 1.29589 |
| First G pre-tanh displacement RMS | 2.29279 | 1.92736 |
| First G pre-tanh displacement / preceding RMS | 2.30853 | 1.94059 |
| First G pre-tanh RMS before → after | 0.99318 → 2.38806 | 0.99318 → 2.11249 |
| End sample-diversity retention, two banks | 0.33523 / 0.33437 | 0.30112 / 0.29847 |
| End first-layer structural cotangent retention | 0.21198 / 0.21015 | 0.18860 / 0.18714 |
| End fraction of pixel channels with absolute value > 0.99 | 66.75% / 66.68% | 70.12% / 70.05% |

Initial saturation on those banks was approximately 1%. The declared minimum
retention is 0.25: both baseline and replay pass sample-diversity retention but
fail structural transmission. The structural cotangent is independent of D's
objective derivative; it describes G's transmission, not semantic image quality.

G's two first-update loss stencils were:

```text
bank 0: 0.970709 → 0.737587 → 0.693856    fitted factor 0.865454
bank 1: 0.971020 → 0.741266 → 0.696604    fitted factor 0.870645
```

D's two unconstrained factors were 1.13765 and 0.47297. The reduction-only rule
and minimum across banks selected 0.47297. Its first bank's losses were
0.05787 / 0.03790 / 0.02917; the second's were 2.52717 / 2.42764 / 2.55131.
This substantial bank dependence is not explained by a measured decomposition
of adversarial versus penalty loss; no causal attribution is made here.

At the selected pair, held-out G loss decreased by 0.26716 / 0.27268 and held-out
D loss by 0.01704 / 0.02368. These are well-resolved local decreases. They did
not predict passing the coupled startup transmission check.

Calibration took **121.85 seconds**, including 16 discarded complete updates,
12 fitting loss evaluations, eight validation evaluations, eight G response
forwards, two fixed-image generations, four D input backwards, and six each of
structural and objective-signal guard evaluations. Final-affine observation
added no forwards. No wall-time guarantee is implied.

All trial training state was restored. Frozen/pretrained parameter and buffer
hashes were identical before, during completion, and after rollback:
`01d54beef4f137b8e502c67dc7896e6a619eef9fe2c1e67c0820aa906e49fa9c`.
Selected factors remained G=1/D=1; persisted effective rates are both 0.0002,
with prior rate 0.002 and no new warmup. Read-only CPU inspection confirmed
checkpoint zero has empty optimizer state, checkpoint one has populated optimizer
state, and both store those same source rates with no warmup field.

## Interpretation and limits

Moving the measurement before the startup transient was necessary for measuring
the first update, but it was insufficient as a rate-selection fix. The first
G update already produces a large pre-tanh change; the selected pair reduces
that change modestly while still failing the replay guard. This supports
investigating actual functional response alongside local loss descent. It does
not establish a universal safe activation-change ratio or prove that a
particular smaller rate will train successfully.

The three-point stencil spans the full configured update. Its coefficients
are not an exact Hessian/Jacobian measurement at the start, and no matching-bank
exact directional slope was measured. A favorable fit and held-out loss decrease
therefore do not establish that the first update is within a useful local
approximation, or that subsequent coupled dynamics will preserve transmission.

Validation: **65** focused tuning/probe/math/persistence tests and **86**
rate-recovery, historical-warmup, lifecycle, and native-update tests passed.
New tests check first-G/eighth-D anchor timing, exact actual Adam displacement,
phase-specific learned-prior coordinates, fixed validation banks, full rollback,
pre-tanh motion hidden by tanh, output-path detection/skips, and hook cleanup on
failure. The GPU result above is an operational success and a rejected numerical
proposal, not evidence of successful GAN training.

# Joint 1e-4 rates fail startup; early generator blocks dominate response

The requested same-seed G=D=1e-4 control failed within 32 native updates.
It does not warrant a longer test at this pair. Layer attribution gives a
specific next research target, but does not yet justify automatic multipliers.
The production tuner is unchanged and remains unvalidated. PR #382 stays open,
unmerged, with automerge disabled.

[Condensed scalar evidence](joint-rate-layer-response-2026-09-22.json) records
the raw report's path/hash, complete rollout observations, numerical model
checks, and the largest parameter contributions. The full per-parameter report
is `/mnt/ml7tb/hypergan-signal-research/joint-rate-1e-4-32-v2.json`.

## Controlled experiment

- Clean implementation `f63d356ab23fcd4281558d585762b270de10b7ad`, GPU 1.
- Original TransGAN/DINO configuration, seed 25002, batch 64, no seed sweep.
- G and owned D learning rates explicitly set to 1e-4; prior stays 0.002.
- Native alternating updates, independent phase draws, lazy penalty every eight
  steps with its native active-step scaling. No warmup, initialization rescale,
  or automatic rate selection.
- Exactly 32 disposable training updates, 177.99 seconds including all probes.
  No training checkpoint or retained update; original source config unchanged.
- Exact trainer-state restoration passed. Protected pretrained/frozen parameter
  and buffer hashes match before, after the experiment, and after restoration:
  `01d54beef4f137b8e502c67dc7896e6a619eef9fe2c1e67c0820aa906e49fa9c`.

The first launch stopped before any training update because the new feature
probe mistakenly rejected the native detached-real-score policy. That no-grad
guard was corrected and regression-tested. Its failure report is preserved;
the second launch is the completed experiment, not another seed or a search.

## The online generating distribution deteriorates

These measurements use the online generator with the evolving prior, replaying
the same particle IDs/noise and measurement RNG. They are not EMA previews.
The fixed real bank may overlap training data and is not independent validation.

| Native step | Output values with abs > .99 | Between-sample variation RMS | DINO polynomial MMD² | Fake / real DINO feature spread |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.01% | .55186 | .33885 | .24594 |
| 1 | 10.49% | .44869 | .37664 | .19848 |
| 8 | 55.03% | .20754 | .44434 | .06991 |
| 16 | 86.83% | .09360 | .39280 | .03919 |
| 32 | 97.04% | .01767 | .53533 | .01061 |

Between-sample variation falls 96.8%. Within-image spatial variation instead
increases from .59366 to .97839: large spatial contrast can coexist with nearly
identical samples. This is why a single output standard deviation is misleading.

Feature distance uses the frozen DINO final-depth spatial-mean representation
(384 dimensions), excluding the constant-gray context images. The kernel is
`(x dot y / 384 + 1)^3`, with the unbiased finite-sample MMD² estimate. This is
not Inception KID, a universal quality score, or a precise population estimate
from 64 samples. The representation also participates in the trained critic,
although its own weights stay fixed. Its deterioration supports the saturation
and sample-variation evidence; it does not independently prove semantic failure.

Fixed-initial-latent samples deteriorate almost identically: final saturation
97.039% versus 97.035% with the evolving prior. At fixed final G, switching from
initial to current prior coordinates changes outputs by only .000976 RMS on
this bank. That isolates a small instantaneous prior effect in this run; it does
not prove the prior had no effect on the preceding training trajectory.

## Where the first generator update acts

Following the measurement idea in
[Function-Space Learning Rates](https://arxiv.org/html/2502.17405v2), four
isotropic output projections estimate each owned parameter tensor's
`RMS(J_l delta_l)`, using the actual Adam displacement. Two reserved banks
agree on the largest four contributions:

| Owned weight tensor | Estimated output-response RMS, bank 0 | Bank 1 |
| --- | ---: | ---: |
| 8px block 1 FFN down projection | .19360 | .16545 |
| 8px block 0 FFN down projection | .18656 | .15950 |
| 16px block 0 FFN down projection | .14381 | .11058 |
| 16px block 1 FFN down projection | .12707 | .10194 |
| Final RGB projection | .000269 | .000339 |

The first four account for **91.54% / 93.47% of the sum of individual squared
response estimates**. This is not their share of the total output change:
cross terms are large and positive. Estimated total squared response is
1.00683 / .58551, versus individual squared-response sums .11920 / .08071.
Four projections give noisy estimates, especially of magnitudes; these are
attribution evidence, not exact norms or a FLeRM matching target.

Those four tensors also supply roughly 65% of the matching-bank first-order
loss decrease. Large contributions are therefore not evidence that the layers
are intrinsically harmful. Suppressing them could also suppress useful descent.

The first complete G update changes generated pixels by about .65 RMS on both
banks. On its native training input, pre-tanh activation movement is 1.3213 RMS,
versus an initial pre-tanh RMS of .9932. The final RGB projection's own parameter
update contributes little to the measured response; large final activations can
result from updates earlier in the generator.

The first-order approximation is also strained: projected linearization-error
RMS is .8553 / .3929. These are low-count projection estimates, not exact errors.
Any layer proposal must be checked with finite, jointly applied updates.

## GeN does not yet produce a credible replacement pair

[GeN](https://arxiv.org/html/2407.02772v3) motivates the symmetric loss stencil
at factors -1, 0, +1 along the actual displacement. We also measure the exact
matching-bank derivative at zero and an unused half-step point. These extra
checks assess approximation error; an interpolating three-point fit alone
cannot establish model validity. We do not apply the resulting stationary
factors, smoothing, or a periodic GeN optimizer.

| Anchor / bank | Fitted curvature | Exact initial slope | Half-point prediction error / actual loss change magnitude | Nominal stationary factor |
| --- | ---: | ---: | ---: | ---: |
| First G / 0 | -.18483 | -.14801 | 41.2% | unresolved |
| First G / 1 | -.17784 | -.13841 | 39.4% | unresolved |
| Lazy D step 8 / 0 | .26954 | -.14207 | 60.4% | .67395 |
| Lazy D step 8 / 1 | .23803 | -.13078 | 60.0% | .68894 |

Both G fits have negative curvature, so they provide no positive Newton
minimizer. The D fits have the requisite signs, but their independent-point
errors are substantial compared with the measured improvement. Neither a
new G rate nor D=6.7e-5 is validated by this experiment. Do not replace negative
curvature with its absolute value, rescue it with the failed norm-product rule,
or choose a rate floor to manufacture a recommendation.

At D's lazy step, the largest measured score-response contributions come from
the owned pixel-critic convolutions. Frozen DINO weights are never part of the
attribution or a proposed tuning group. D score units and G pixel units cannot
be compared as if their raw response RMS defined a universal player balance.

## Local descent still does not establish useful coupled progress

With initial prior, latent values, buffers, and randomness held fixed, crossing
initial/final owned G and D parameters gives these G phase losses:

| | Initial D | Final D |
| --- | ---: | ---: |
| Initial G | .71669 | 5.90671 |
| Final G | .69392 | 5.77700 |

G improves against either fixed critic, while changing D raises its objective
much more. Those comparisons do not imply D is wrong or prescribe a TTUR ratio.
They demonstrate why fixed-opponent descent alone could accept this measured
unhealthy distribution, and why useful-progress checks need a fixed external
representation as well as internal training losses.

## Next bounded research step

Test the dominant early FFN down-projection group separately from the remaining
owned G parameters, using a few predetermined directional evaluations rather
than a per-tensor grid search. This is a candidate grouping suggested by the
measurement, not a validated rate rule. Measure group loss slopes, curvature,
mixed effects, and actual combined functional response. The observed output
cross terms are not loss-Hessian cross terms; both kinds of interaction matter.
With the same saved state and banks, symmetric evaluations of each group and
two unused mixed points would add 12 loss evaluations. The negative measured
whole-direction curvature already rules out a positive-definite joint quadratic
at that scale; even positive diagonal group curvatures would not rescue an
unconstrained joint Newton minimum. A smaller local stencil would need all
terms remeasured at that common scale.
Validate any resulting joint direction on unused points/banks and a coupled
rollout that includes current-prior samples and frozen-feature diagnostics.

Use explicit optimizer-group learning rates if supported by evidence: multiplying
raw gradients can largely cancel inside Adam's normalization. FLeRM matching
still needs a defensible successful reference profile; equal responses from all
layers are not an established optimum. Negative or unresolved group curvature
must remain an honest abstention. No replacement selector or layer multipliers
have been installed in this research pass.

## Reproduction and verification

Research entry point: `reports/joint_rate_probe.py CONFIG --g-lr 1e-4 --d-lr 1e-4
--steps 32 --direction --features --output NEW_JSON`, using this worktree on
`PYTHONPATH=src` and selecting the free GPU explicitly. `--features` is guarded
for this particular 128px DINO multidepth testbed. This is not a long-test launcher
or an endorsement of the rate pair.

Ten CPU function/GeN/crossed-progress tests and eight frozen-feature tests pass,
including native phase detachment, exact state and gradient-object restoration,
protected-buffer mutation detection, failure cleanup, projection cross terms,
and a nonquadratic counterexample to three-point fit validation. A native CPU
eight-update orchestration smoke with all four directional bank audits also
passed. The completed GPU run verifies the actual testbed and audit hashes.

The report itemizes cost: 32 native updates; 16 GeN phase-loss evaluations with
four matching-bank backwards; eight logical function-output evaluations with
16 projection backwards; four crossed phase losses; ten observation G forwards;
five feature G and ten feature critic forwards; and four native-observer G
forwards. D function evaluations include real/fake scoring, and lazy D loss
evaluations include the configured input-gradient penalty, so a logical loss
evaluation is not necessarily a cheap no-grad forward.

# FLeRM / GeN follow-up: measure response before selecting another rate

Research-only review of head `dbb2480a`, the primary papers, and the
[failed longer-run selector](gradient-response-long-run-review-2026-09-22.md).
No code, GPU runs, weights, or configuration changes were made in this review.
The user's requested G=D=1e-4 probe is an explicit intervention, not an
inferred optimum or a newly invented minimum rate.

## What the papers actually supply

[Function-Space Learning Rates, ICML 2025](https://arxiv.org/html/2502.17405v2)
defines each block's response as `r_l = RMS(J_l delta_l)`, where `delta_l`
is an optimizer displacement. Random output projections permit estimates
for all blocks in one reverse pass per projection. Its practical estimator
uses covariance approximations and moving averages; a few projections are
uncertain estimates. FLeRM transfers a measured base model's response profile:
with unit-rate directions, `eta_l = eta_base * r_base,l / r_current,l`.
Section 3.3 permits matching once at startup. It does not identify a universal
ideal response. Depth transfer requires a block mapping; input/output blocks
are treated differently from replicated hidden blocks. A successful CIFAR GAN
with different data and critic is therefore a diagnostic reference, not an
established numerical target for the logo GAN.

[GeN, ICLR 2025](https://arxiv.org/html/2407.02772v3) fits loss along an
optimizer direction, including Adam. Algorithm 1 uses symmetric perturbations
around the current point, positive fitted curvature and descent, smoothing
(example gamma=.9), and periodic adaptation (example period eight). Appendix
B.4 includes DCGAN/CelebA: five epochs, batch 128, qualitative generation
evidence. It does not establish startup-only calibration for this model.
For signed actual displacement delta, fit
`phi(s)=phi(0)+a*s+k*s^2/2`, then `s*=-a/k` when a<0 and k>0.
The prior positive-only stencil and gradient-norm replacement were adaptations.
Three interpolation points exactly determine a quadratic, so their fitting
residual alone cannot establish model validity. Use an extra evaluation or a
matching-bank derivative as an independent check. Negative curvature does
not justify a positive Newton rate by taking its absolute value.

## What the failed selector established

The limiting G bank's norm product C=45.30 exceeded the magnitude of its signed
directional secant, 1.87, by 24.2 times. Orthogonal gradient changes contribute
to C even when they do not oppose the proposed direction. Cauchy controls an
observed projection; it does not establish a necessary safe rate or useful
functional progress. The resulting G rate was 205 times below the source rate;
D and prior rates stayed unchanged. These facts do not alone prove causality.

Startup preserved fixed-latent diversity and transmission, but the evolving-prior
EMA preview's between-sample variation fell from .54556 at step100 to .23266 at
step500, while saturation rose from .875% to 21.596%. The user also reported
poor samples. Nonzero parameter changes and small local loss decreases were
insufficient acceptance conditions. Choosing a floor after observing this failure
would conceal that limitation rather than repair it.

## One bounded, informative diagnostic for the requested reference

This is our diagnostic design, not an algorithm claimed by either paper.

1. Observe the same G=D=1e-4 trajectory at first update and update eight;
   record G, owned D, and prior displacements separately. The eighth D update
   includes the configured lazy penalty. Keep pretrained tensors immutable.
2. At the first G anchor and eighth D anchor, use two fixed fitting banks and
   loss points s=-1,0,+1 along each actual delta. Reserve s=+1/2 as an
   independent model check. This is 16 player-loss evaluations total, not a
   learning-rate grid: none of these states becomes a training candidate.
   Report held-out prediction error, signed curvature, and proposed optimum.
   D evaluations must include the active penalty; split adversarial/penalty
   contributions in the report, preserving the total configured objective.
3. On two fixed banks per anchor, use four shared output projections. For each
   block record `c_l=<grad_theta_l(r^T f/sqrt(m)),delta_l>`; retain `sum_l c_l`
   as well as the separate squares. Compare the projection of the actual finite
   output change with `sum_l c_l`. This tests linearization fidelity and exposes
   block reinforcement/cancellation. The implemented helper reuses each graph
   across four backwards, then evaluates its finite response once. Two banks
   times two players cost sixteen projection backwards and eight logical output
   evaluations (four before, four after). Each D output evaluation draws a G
   sample and scores real and fake separately; these are not eight elementary
   network calls. The GeN stencil also adds four matching-bank slope backwards
   beyond its sixteen phase-loss evaluations. Projection uncertainty
   must remain visible; these are measurement draws, not training seed sweeps.
4. Project generated pixels for G and fixed real/fake score vectors for D.
   Also record finite pre-tanh G displacement through a forward hook, without
   another projected backward. This catches saturation hiding motion;
   it is an additional diagnostic, not a FLeRM-prescribed target. Do not
   equalize blocks or scale rates without a defensible reference profile.
5. Proposed follow-up, not implemented in the current helper: separate model
   validity from usefulness. At saved early/late states, evaluate
   the crossed losses L_G(G_old,D_old), L_G(G_new,D_old), L_G(G_old,D_new), and
   L_G(G_new,D_new), with fixed input values and real context. This distinguishes
   G progress against each fixed opponent from an increase caused by D changing.
   Repeat distribution diagnostics using fixed particle IDs/noise through the
   current prior; fixed latent values intentionally omit learned-prior drift.
   A fixed held-out feature-distribution discrepancy can supplement spatial
   variation/saturation, but is not a universal quality measure. Longer-run
   improvement must be measured before calling a rate useful.

The first four items can reject a bad local model without a new arbitrary
response cap. They cannot certify useful training. If the requested reference
learns, it can supply a same-task measured response profile for a later matching
experiment. If it does not, FLeRM provides attribution rather than a missing
target; GeN may still correctly abstain. No automatic fallback or production
selection change is proposed by this note.

## The user's layerwise calibration idea

Layerwise output-response measurement is directly useful: a few dominant owned
blocks could explain why reducing every G tensor also suppresses useful learning.
Dominance alone is not proof that a block's contribution is harmful; measure
its signed loss contribution and interactions before selecting a multiplier.

[StyleGAN2-ADA's official FullyConnectedLayer implementation](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/training/networks.py)
uses runtime effective weights: raw weights are initialized divided by the LR
multiplier, then multiplied in forward by `lr_multiplier/sqrt(in_features)`;
biases have their own runtime gain. This is a parameterization with lasting
functional and optimizer consequences, not simply a backward gradient hook.
It provides a concrete precedent for scale-aware owned layers, not permission
to rewrite pretrained tensors or a universal calibration formula.

For a positive constant gradient multiplier c applied consistently from the
start, Adam has m'=c*m and v'=c^2*v. Consequently m'/sqrt(v') equals
m/sqrt(v) when epsilon is zero. Epsilon, clipping, decay, changing multipliers,
and preexisting moments qualify that algebra. Control actual updates through
owned parameter-group LRs when the intention is layerwise step scaling; a
raw-gradient multiplier can largely disappear in Adam normalization.

A possible bounded next hypothesis is grouped directional GeN: a small declared
set of architectural groups, with `a_l=g_l^T delta_l` and
`k_ll=delta_l^T H_ll delta_l`, proposes `s_l=-a_l/k_ll` only for resolved
descent/positive curvature. With two fitting banks, K groups require at least
`2*(1+2*K)` phase-loss evaluations for symmetric diagonal fits, before model
validation. A shared backward supplies the block slopes. The true joint model
also contains cross terms `k_lm*s_l*s_m`; separate diagonal optima need not
improve its loss. Verify the composed direction on separate banks and one
coupled replay. These loss-Hessian cross terms are different from the measured
output-response cross terms. Prefer a few coarse, architecturally declared
groups over every tensor, and preserve prior/frozen ownership boundaries.

This is a future hypothesis, not an implemented selector. It does not require
a FLeRM reference target, but also does not inherit FLeRM's transfer result or
GeN's empirical performance. A same-task successful response profile remains
the stronger basis for actual FLeRM matching.

## Helper validation

`reports/test_function_space_probe.py` executes both helpers on native CPU
trainers and checks exact state/RNG/optimizer restoration, existing `.grad`
object identity, protected-buffer mutation detection, exception rollback,
constructive/destructive output cross terms, symmetric sign conventions, and
an independent point detecting a nonquadratic function. Nine tests pass. A
precision fix keeps projected finite responses in float64, matching the other
reported projection arithmetic. No GPU experiments were performed here.

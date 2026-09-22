# Research: problem-dependent G/D rates from measured updates

Date: 2026-09-21. Implementation inspected: `a4221a86`, PR #382.
This report proposes research follow-ups; it does not change training behavior.
No GPU training, seed experiments, checkpoint mutation, or pretrained calibration
was performed. The PR remains unmerged with automatic merging disabled.

**Recommendation:** develop an observation of the actual optimizer update's
effect on the network output, then evaluate a bounded directional-curvature
proposal for both G and D. The most relevant literature is Function-Space
Learning Rates and GeN. The former addresses the measurement and transfer
problem; the latter supplies a local step-size formula. Neither establishes an
optimal startup-only policy for this ParticleGAN/TransGAN configuration.

The working assumption is that the configured game already has useful training
dynamics. The question is how architecture, data, critic features, and optimizer
state change the numerical step sizes it needs. Existing GAN stabilization
papers are useful boundaries on the claims, not the main prescription.

The user subsequently requested **one solution**, with no rush to merge, and
stopped the other experiment. The intended destination is one `--tune` pipeline:
measure update response, derive a bounded proposal, verify a coupled replay.
Once supported by evidence, replace the current D-half-rate and G-retention
proposal rules instead of exposing them as competing user-selectable strategies.
Keep useful read-only diagnostics, state restoration, ownership protection,
persistence and UI infrastructure. Historical experiment reports remain evidence.
The production selector has not been replaced by this research commit.

## What the literature contributes

| Primary source | Relevant result | Boundary for this project |
| --- | --- | --- |
| [Function-Space Learning Rates, Milsom et al., ICML 2025](https://proceedings.mlr.press/v267/milsom25a.html) | Measures each parameter block's contribution to output change; FLeRM transfers measured targets across model scales. | Requires a base model/reference; does not establish transfer between arbitrary datasets or GAN objectives. |
| [GeN, Bu and Xu, ICLR 2025](https://arxiv.org/html/2407.02772v3) | Fits a local loss quadratic along an optimizer direction, including Adam, using additional evaluations. | Periodic adaptation is not startup-only calibration. Appendix B.4 reports DCGAN/CelebA, five epochs, batch 128, with qualitative evidence. Its optional initial-rate grid is outside our scope. |
| [Automatic Gradient Descent, Bernstein et al., 2023](https://arxiv.org/html/2304.05187v1) | Derives updates from architecture-dependent bounds on functional perturbations. | The reported implementation disables biases and affine normalization parameters. It is not a drop-in recipe for our existing TransGAN and pretrained components. |
| [Tensor Programs V, Yang et al., 2022](https://arxiv.org/abs/2203.03466) | Under muP, many tuned hyperparameters transfer as model size changes. | Requires a compatible parameterization and a tuned proxy; does not establish a universal rate across problems. |
| [Painless Stochastic Gradient, Vaswani et al., NeurIPS 2019](https://papers.neurips.cc/paper/8630-painless-stochastic-gradient-interpolation-line-search-and-convergence-rates.pdf) | Same-batch sufficient-decrease tests can adapt step size. | Its interpolation/convergence assumptions do not cover arbitrary GANs; unbounded backtracking conflicts with our budget. |
| [Understanding the Difficulty of Training Transformers, Liu et al., EMNLP 2020](https://arxiv.org/abs/2004.08249) | Small parameter perturbations can be amplified into large output changes through residual branches. | Supports measuring update response; it does not diagnose our particular colored-tile failure. |
| [TTUR, Heusel et al., NeurIPS 2017](https://arxiv.org/abs/1706.08500) | Analyzes separate player rates under stochastic-approximation assumptions. | Does not give a universal finite-run D:G ratio for this configuration. |
| [The Mechanics of n-Player Differentiable Games, Balduzzi et al., ICML 2018](https://arxiv.org/abs/1802.05642) | Separates potential and rotational components of game derivatives. | Explains why each player's scalar curvature alone misses their interaction. No new game optimizer is proposed here. |

Other relevant directions are [gradient noise scale](https://arxiv.org/abs/1812.06162)
for batch-size efficiency, [online hypergradients](https://arxiv.org/abs/1703.04782),
and [DoG](https://arxiv.org/abs/2302.12022). These address different or ongoing
adaptation problems; they are not evidence that eight startup updates identify
all useful hyperparameters. Noise measurements must hold both players fixed:
successive training gradients mix sampling noise with changes in the game.

## Three different Jacobians

Let `x = G_theta(z)` and let `q = dL_G/dx` include the complete configured
adversarial path, including frozen feature extractors. Use the signed **actual**
optimizer displacement `delta_theta = theta_after - theta_before`.

1. **Transmission through G:** derivatives between activations, or `dG/dz`.
   These help locate attenuation. They are not the same as the parameter-to-output
   derivative, and healthy latent sensitivity does not prove useful learning.
2. **Response to an optimizer update:** `J_theta = dx/dtheta`, especially
   `J_theta delta_theta`. This predicts the change G will actually make.
3. **Response of the training system:** derivatives of the joint update map.
   This includes G, D, the learned prior, Adam moments, update order, and the
   penalty schedule. A two-player raw-gradient Jacobian is only an approximation.

For G, the chain rule gives

```text
g_adv = J_theta^T q
predicted image change = J_theta delta_theta
predicted adversarial loss change = q^T J_theta delta_theta
                                 = g_adv^T delta_theta
```

The final scalar is available without a full JVP when the correct adversarial
gradient is already available. A negative value means descent against this
fixed critic, in this local approximation. It does not certify visual progress.
With auxiliary losses, distinguish `g_adv` from the total optimizer gradient.

Raw gradient norm is insufficient because Adam changes the direction/scale and
because the generator can amplify some parameter directions much more than
others. A maximum singular value is also not the desired scalar: it measures
the worst direction, which the optimizer may barely use. Full singular spectra
would cost more while still omitting usefulness of the discriminator's target.

The [dynamical-isometry literature](https://arxiv.org/abs/1711.04735) concerns
singular-value structure and trainability. It is not a prescription that every
trained GAN layer should transmit equal gradient magnitude.

## Measurements worth adding first

| Observation | Interpretation | Important control |
| --- | --- | --- |
| Actual per-step G/D/prior displacement | What the optimizer did, including moments and penalty effects | Separate ownership groups; cumulative eight-step displacement can cancel |
| `G_after(z) - G_before(z)` | Actual finite-step G output response | Fix latent values, stochastic choices, buffers and modes |
| `g_adv^T delta_theta` and `q^T delta_x` | Predicted descent and agreement with the delivered signal | Frozen critic; linear and finite-step values have different meanings |
| Per-block output contributions | Whether early, middle and late parameters affect output | Measure at a common output; hidden activation norms have different units |
| `q_after_D(x) - q_before_D(x)` | D's effect on the instruction delivered to G | Same detached fake images, real context, augmentations and stochastic choices |
| Pre-output activation movement and output derivative | Whether tanh is concealing internal motion | Output saturation is architecture-specific, not a universal GAN failure label |
| Independent held-out quality change over training | Whether calibrated updates are useful | Internal signal health cannot replace a distribution/task metric |

For D-induced signal drift, report absolute RMS change, norm ratio, and cosine
when norms are resolved. A large change can be a useful correction by D; there
is no basis for maximizing agreement with an initially poor discriminator.
Near-zero q makes relative metrics undefined or unreliable. Report that status
instead of dividing by a convenient epsilon and assigning a healthy score.

The same caution applies to diversity. Our [warmup audit](startup-warmup-drift-2026-09-21.md)
found that 92.78% of between-sample variance at step 1000 came from differences
in mean RGB color. A large output spread can describe bad samples. Measure both
full and per-image-mean-removed output changes, plus pooled spatial scales;
these remain diagnostics, not semantic quality scores.

### Cheap per-layer attribution

For block `l`, use `v_l = J_l delta_theta_l`. FLeRM estimates
`RMS(v_l)` through random output projections and VJPs, exposing all parameter
blocks in one backward pass per projection. Its detailed covariance approximation
trades bias against variance; a few probes should not be treated as exact.
See [the method](https://arxiv.org/html/2502.17405v2).

For our proposed diagnostics, also retain the common-projection **sum across
blocks**. In general `||sum_l v_l||^2 != sum_l ||v_l||^2`; layers can reinforce
or cancel. Raising every weak layer to the same contribution can change that
interaction. First observe it; do not automatically equalize layers.

Use an isolated measurement RNG stream if stochastic projections are later
implemented. Multiple projections estimate one measurement; they are not
retraining the experiment under different seeds. This research check uses exact
enumeration on a tiny example and consumes no random stream at all.

## A formula for a bounded candidate, with explicit limits

This is our proposed adaptation of directional quadratic calibration, not a
claim of a new optimizer or a proven GAN tuning rule.

At a fixed player/optimizer state, record a configured-rate displacement
`delta`. Freeze the opponent and all other parameter groups. Define
`phi(s) = L(w + s delta)`. Evaluate at the predetermined stencil `0, 1/2, 1`:

```text
a = 4 phi(1/2) - phi(1) - 3 phi(0)
k = 4 [phi(1) - 2 phi(1/2) + phi(0)]
phi(s) approximately phi(0) + a s + (k/2) s^2
s_curvature = -a/k          only when a < 0 and k > 0 are resolved
new learning rate = configured learning rate * s
```

The identities follow by interpolation. `k` estimates `delta^T H delta`, not
the largest Hessian eigenvalue. Three points always fit a quadratic exactly;
zero fitting residual proves nothing. Compare its slope with an available
`gradient^T delta`, then check a selected point on reserved data and verify the
actual coupled update. If the selected point coincides with a stencil point,
held-out data and the coupled rollout supply independent checks of different
claims, not proof of quadratic accuracy between points.

This formula cancels a constant positive rescaling of the measured loss **for
a fixed delta**. Rescaling the training loss can still change Adam's direction
through epsilon, moments, other terms and clipping. Do not promise full
loss-scale or parameterization invariance of the optimizer.

If curvature is flat, negative, noisy, or the actual direction is non-descent,
abstain from a curvature-derived proposal. A flat monotone discriminator loss
does not establish that a huge D step is desirable. Never force a denominator
positive merely to get a rate.

An optional functional-response cap has the transparent form

```text
s_response = target_allowed_response / measured_response_at_unit_scale
s = min(1, s_curvature, s_response)
```

Apply a response cap only to an already valid curvature proposal; it does not
rescue an unresolved curvature estimate. A response based on an exact JVP scales linearly
in the local model; a finite displacement is a secant and requires verification.
For an initial conservative experiment, retain the existing reduction-only
range. If a computed safety cap falls below 0.1, report unresolved rather than
rounding it upward and violating the cap.

**The response target is still an engineering choice.** Normalizing by image
range, real-data variation, or a related successful run gives interpretable
units; none establishes a universal optimum. Curvature helps propose progress;
response limits and current retention guards can reject disproportionate changes.
They do not prove that a surviving update improves the target distribution.

G and D need separate proposals. An eventual joint proposal must be checked in
the real D-then-G schedule: changing D's rate changes G's subsequent direction.
Do not combine independently measured directions and assume the resulting
training update is their sum. This would extend the current code, whose trials
only reduce one player at a time.

## A finite research protocol

The first implementation should report measurements while leaving selection
unchanged. Observe the first and eighth disposable updates: this distinguishes
fresh Adam behavior from a later startup state and includes this configuration's
first lazy-penalty event. Both are observations along one baseline, not extra
training candidates. At the first update, initially record parameter motion and
G output response using the normal training input. Reserve the extra two-bank
D-signal and layer-attribution probes for the eighth-update anchor.

If those measurements are reliable, a subsequent experimental selector can use:

1. The existing eight-update baseline, recording actual phase-local directions.
2. One predetermined loss stencil for each player at the eighth update's
   respective pre-phase state, on two reserved banks. G is measured with its
   baseline post-D opponent; D is measured with its pre-D context.
3. At most one proposed pair of G/D reductions. Leave a player unchanged if its
   curvature is unresolved; never infer its rate from the other player's slope.
   If neither player has a valid reduction, report unresolved. Protected-state
   failures remain fatal audit errors with rollback. There is no extra search.
4. Evaluate the selected point on two additional validation banks that were not
   used to fit or select it, then perform one eight-update rollout from the
   original state. Each changed player must have finite values and strictly
   lower loss on both validation banks against the same frozen opponent; reject
   ties, increases, or unresolved numerical differences. This is a conservative
   screen, not a statistical guarantee of decrease. The unchanged player is
   checked by the coupled rollout rather than forced to satisfy a curvature
   criterion. Reject the whole proposal if a changed player fails; do not try a
   different combination. Include existing guards and the new observations.
5. Exact restoration of all trial state. Persist only selected rates and the
   report; leave pretrained weights/buffers unchanged.

This proposed protocol permits at most **16 disposable training updates**, but
that is not its full cost. Two banks times two players times three stencil
points means up to 12 phase-loss evaluations; selected-point validation adds up
to eight, including a fresh baseline loss for each validation bank/player.
A two-bank before/after D signal check adds four critic input-gradient
evaluations. An optional four-projection G attribution adds four G backwards
at one anchor/bank. State copies, hashing and existing guards cost more. These
counts describe the eighth-anchor proposal/validation probes. Basic G response
observation at the first and eighth updates adds up to two G forwards when the
normal pre-update outputs are reusable; otherwise record the additional replay
forwards explicitly. They are budget components, not a measured total; the
complete implementation must cap and report actual executions. It may be slower than the
current 24-update maximum and must be timed before adoption. No fallback grid,
repeat-until-success loop, or automatic new experiment belongs in this protocol.

Exact slope verification on the fitting banks, if enabled, additionally needs
up to four player-gradient evaluations: a training-batch gradient cannot be
substituted for a reserved-bank gradient. On a penalty-active D phase this can
involve higher-order differentiation. Probe costs must be itemized in the report
rather than hidden inside the disposable-update count.

The validation above compares measured loss changes; it is not an Armijo test.
An Armijo test would require matching validation-bank directional derivatives,
adding up to four further gradient evaluations. Do not reuse fitting-bank losses
or gradients as validation-bank baselines.

The coefficient/threshold policy, how to aggregate two banks, and whether the
eighth-update direction predicts a good rate at step zero remain research
questions. In the initial experiment, choose the smaller valid proposal across
banks for each player, leave that player unchanged on sign disagreement, and
record raw values; this is a conservative
policy choice, not a theorem. Two banks do not provide a precise uncertainty
estimate. Startup success must be followed beyond the observed 350–500 failure
window before claiming the policy works.

## Integration constraints found in the actual code

The optimizer boundaries are in `src/hypergan/objective_program.py`, D at line
358 and G/prior at line 397 in the inspected commit. Capturing displacement there
includes the configured fused Adam behavior without differentiating Adam.
The G backward at line 392 can expose q cheaply for this configuration, which
has no image-dependent auxiliary loss. A generic diagnostic must distinguish
the total and adversarial gradients.

`startup_dynamics._optimizer_motion` currently measures net displacement over
eight updates. `_measure` evaluates q on each state's freshly generated images;
those comparisons cannot attribute q changes to D alone. A new fixed-image
probe is necessary.

The learned prior shares G's optimizer but has a distinct rate and objective.
Record G-only motion at fixed latent values and full generating-system motion
at fixed particle IDs/noise. Standardized MoG coordinates can change even when
the selected particle alone was not directly updated. Do not tune G to compensate
silently for large prior motion; report prior-dominated drift as unresolved.

The installed ParticleGAN `b_cap` uses `lazy_k=8`, applies on step multiples of
eight, and scales the active coefficient by eight. D's eighth-update direction
therefore includes the penalty event. Compare ordinary and penalty-active
response separately. A D loss stencil must preserve that event, including its
input-gradient calculation and stochastic draws; it is not necessarily a
forward-only evaluation. The frozen feature extractor still participates in
input differentiation.

Exact JVPs are optional. A deterministic CPU primitive check in the installed
PyTorch 2.14 found default SDPA does not support the required forward AD/double
backward, while the math backend does. HNDL generator attention uses unrestricted
SDPA; the DINO provider already requests math attention. CUDA kernels were not
tested here. Changing attention backend can change cost and numerics, so finite
response measurements and ordinary reverse-mode projection probes are better
first integrations. [PyTorch documents operator-coverage limitations](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html).

All evaluations must replay buffers, train/eval modes, random streams,
augmentations, batch relationships and data state. `no_grad()` or
`functional_call()` alone does not provide that isolation. Never perturb or
reinitialize pretrained tensors; differentiating their inputs is allowed.

## Startup calibration, drift, and warmup

An initialization or rate can be well scaled at startup and become poorly scaled
later as the critic, prior and Adam moments evolve. Both reviewed measurement
and rate-adaptation approaches allow observations over training; startup-only
use sacrifices that feedback. A later read-only diagnostic can detect drift
without changing rates or adding an adaptive training controller.

[Ma and Yarats](https://arxiv.org/abs/1910.04209) motivate warmup through actual
update magnitudes and discuss a duration linked to Adam's beta2. That does not
establish our original configured peak rate as a good endpoint, or prove that
1,000 steps is right for this GAN. Calibration of the endpoint and the choice
of ramp duration are separate questions.

For research comparisons, hold the selected rate using the existing
`--tune-warmup-steps 0` so a later rise does not confound the result. This report
does not change the user-requested 1,000-step default. If a new selector is later
adopted, its schedule should treat the selected rate as the calibrated endpoint;
ramping back to an unvalidated source rate would undo that interpretation.

## Evidence produced in this research pass

Run the deterministic CPU check with:

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/hypergan-preview-test-env/bin/python -I reports/update_response_proof.py
```

[The script](update_response_proof.py) and [saved results](update-response-proof-2026-09-21.json)
verify the chain-rule identity using an actual Adam displacement; the JVP has
1.57% relative error against the finite output change, falling to 0.79% at half
the displacement in this example. Exact sign-projection enumeration recovers
per-block and total squared response, including cross terms. A quadratic stencil
recovers its known minimum and abstains on flat/negative curvature examples.

Counterexamples show why a single scalar cannot do everything: a worst-direction
spectral bound overestimates the chosen direction's motion by 1,000x; tanh hides
a unit preactivation change behind approximately 0.0000106 output change; and
independently optimal player steps can break a coupled map that is stable at
smaller rates. The last example uses simultaneous updates for algebraic clarity,
not HyperGAN's alternating implementation. These are mathematical checks, not
empirical GAN validation or measurements of the user's active run.

The next decisive evidence is whether measured functional response predicts
rate sensitivity on the TransGAN/DINO testbed, and whether the same observations
remain sensible on the user's successful CIFAR setup. The CIFAR configuration
has different ownership/objective structure, so it is a diagnostic reference,
not a source of unquestioned numeric targets. Use the configured seed for
distinct interventions; no seed sweeps. Evaluate quality as well as saturation,
transmission, cost and functional response before making tuning a default.

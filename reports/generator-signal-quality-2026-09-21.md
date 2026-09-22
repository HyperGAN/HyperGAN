# Research: is the generator receiving a useful learning signal?

Date: 2026-09-21. Code inspected: `origin/develop` at `90c4dd8c`.
Scope: observation and evaluation only. No loss, regularizer, optimizer, recipe,
or training behavior changes. The accompanying calculation runs no training
experiments and uses no random seeds.

**Recommendation:** measure the signal at the generator output, its passage back
to the first blocks, the resulting optimizer updates, and independent progress.
There is no defensible single gradient magnitude, signal-to-noise ratio, or
layer ratio that certifies good GAN learning. The closest practical north star
is **improvement in a declared independent distribution/task metric after one
generator update**, evaluated on held-out inputs. It is relative to that metric,
and sometimes too noisy to resolve in one step.

The user's distinction between the beginning and end of the generator is
essential. Forward execution starts at the latent/conditioning inputs and ends
at the generated sample. The adversarial gradient travels in the reverse
direction. Measure both **activation gradients** (what passes through a block)
and **parameter gradients/updates** (what that block can learn). Neither replaces
the other. In particular, a small gradient with respect to the latent input alone
does not establish that early trainable parameters cannot learn.

## 1. Separate four questions

| Question | Measurement | What it cannot establish |
| --- | --- | --- |
| Does the critic supply a signal? | Gradient with respect to generated output | Whether the requested change improves the data distribution |
| Does it reach the beginning of G? | Gradients at first/middle/last blocks and before/after output activation | Whether those blocks make useful updates |
| Does the optimizer act on it? | Per-block actual update, update/weight ratio, output displacement | Whether the action is beneficial |
| Is the action useful? | Paired independent quality change; optionally reference-gradient alignment | All aspects of quality, or long-term game convergence |

For a differentiable generator, write the generated batch as
`X = G_theta(Z, C)` and freeze the critic at its generator-phase state. For the
adversarial loss routed solely through X:

```text
q       = d L_adv / d X                  signal delivered to G's output
J_theta = d X / d theta
g_adv   = J_theta^T q                    signal delivered to G's parameters
g_total = g_adv + g_auxiliary            actual optimizer input
delta   = theta_after - theta_before    actual optimizer action
```

If there are several generated outputs or direct auxiliary paths, use the sum
of their chain-rule contributions; do not pretend the final image is the entire
graph. Critic preprocessing and differentiable augmentation are part of q's
path. A frozen feature extractor still needs differentiation with respect to its
input; frozen weights and a detached computation are different things.

Under plain SGD on this adversarial term, the local output motion is
`delta_X ~= -eta J_theta J_theta^T q`. Consequently, a strong q can land in a
direction G can barely realize. Adam, momentum, clipping, other objectives, and
trainable priors change this motion. Observe the actual delta, not just `lr*g`.
These are chain-rule consequences, not new convergence claims.

This framework covers differentiable GAN objectives without depending on BCE,
hinge, Wasserstein, or a particular architecture. Some discrete GANs use
score-function estimators or surrogate backward rules: parameter-gradient
statistics and sample-based progress still apply, but output gradients and
layer transmission may be unavailable or describe only the surrogate. Publish
that limitation explicitly. “Universal” can mean a common measurement contract;
it cannot mean every GAN exposes every derivative.

## 2. Why a perfect internal score cannot work

**Strength has arbitrary units.** Multiplying a loss by a positive constant
multiplies its gradients without changing its preferred direction. Rescaling a
hidden representation and inversely rescaling the following layer preserves
the forward function while changing its activation and parameter gradients.
The toy calculation demonstrates both effects. Optimizers need not preserve
their trajectories under these transformations.

**Consistency does not establish correctness.** A critic can supply the same
wrong instruction on every sample. A useful mean shift can also legitimately
give all examples similar gradients; a high alignment score is not itself mode
collapse. Conversely, different conditions or modes can need opposing changes.

**A quiet signal is ambiguous.** It can mean equilibrium, saturation, a dead
path, or missing coverage. For the original idealized GAN game, distribution
matching and an uninformative optimal discriminator coincide, but this is a
theoretical limit, not a rule that discriminator accuracy should be 50% during
training. Saturation of the minimax generator loss motivated the original
non-saturating alternative. [Goodfellow et al., 2014](https://arxiv.org/html/1406.2661v1)

**Correct classification is insufficient.** The geometry of critic derivatives
on or near generated samples matters; reliable classification need not give
useful generator directions.
[Arjovsky and Bottou, 2017](https://arxiv.org/abs/1701.04862)
Wasserstein objectives offer useful theory under their assumptions, but the
gradient of an arbitrary trained critic is not automatically the gradient of
the exact Wasserstein distance.
[Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875)

**Local descent is insufficient.** An update may reduce today's frozen-critic
loss while harming quality or participating in an oscillating game. Even
WGAN/WGAN-GP do not always converge with finitely many discriminator updates
under the settings analyzed by Mescheder and colleagues.
[Mescheder et al., 2018](https://arxiv.org/abs/1801.04406)

There is also a direct identifiability issue: a metric that observes only G, D,
and their gradients cannot distinguish two target distributions requiring
opposite motion if the observed networks and gradients are identical. It needs
independent information about the target. Even with held-out data, finite
measurements cannot certify all properties of a high-dimensional distribution.

## 3. Beginning-to-end measurements

Start with semantic block boundaries: first trainable block, roughly 25%, 50%,
75% of the main path, final block, pre-output nonlinearity, generated output.
For residual/branched networks include the residual branch separately where
possible: a healthy skip route can hide a branch that receives little signal.
Identify blocks by graph/module path and invocation, not just execution number;
reused modules and multiple outputs need distinct observations.

For activation h at a selected boundary, record:

```text
A_h = RMS(h)
R_h = RMS(d L / d h)
S_h = A_h * R_h
T_h = R_h / R_output
```

Use per-sample RMS then summarize its distribution (for example p10/median/p90),
plus the batch-wide RMS. Do not average gradient vectors over samples first:
opposing gradients would cancel. Record exact-zero and nonfinite fractions;
define any additional “tiny” threshold in the protocol, relative to dtype and
scale. Store numerator, denominator, tensor dimensions, objective, and reduction
convention alongside ratios. Use log-scale plots and compare trends within the
same protocol rather than assigning universal green/red cutoffs.

`T_first` answers the user's beginning-versus-end question in raw coordinates.
The intervening profile locates attenuation or amplification; an endpoint ratio
alone can miss both happening at different places. Also show `S_h / S_output`
when defined: S compensates for a uniform rescaling of an activation and the
inverse rescaling of its gradient. It is a **relative-sensitivity context
measure**, not a coordinate-invariant information score. It is affected by
offsets, channel mixing, dead/zero activations, and architectural choices. A
constant profile is not a universal optimum.

For a batch-mean separable loss, `B * dL/dh_i` removes the explicit 1/B from its
sample's activation gradient. Keep both the actual backward magnitude and this
documented normalization. Patch/token/spatial reductions also matter. With
batch-coupled losses, BatchNorm, relativistic scores, or accumulation, B times
the batch derivative is **not** an isolated per-example loss gradient. Compare
identical full-batch protocols and label it as a batch derivative. Do not use
this B multiplier on the already averaged parameter gradient.

For every selected trainable parameter block W, separately record:

```text
parameter gradient RMS = RMS(dL/dW)
parameter RMS          = RMS(W)
update RMS             = RMS(W_after - W_before)
relative update        = update RMS / parameter RMS
optimizer alignment   = cos(-dL/dW, W_after - W_before)
```

Separate weights, biases, and normalization parameters where aggregation would
hide a problem. Zero-initialized weights make the relative update undefined;
report the absolute update and a status. `grad is None`, exactly zero gradient,
frozen parameters, and nonfinite values are different states. Momentum can
legitimately disagree with today's gradient; alignment diagnoses that behavior
without proving the optimizer is wrong. Ratios remain parameterization-dependent.

The first implementation should observe the total loss. A less frequent probe
should separate adversarial and auxiliary gradients, including each critic
where needed. Otherwise auxiliary supervision can hide a weak adversarial path.
Optional term diagnostics are pairwise gradient cosine and
`norm(sum(g_term))/sum(norm(g_term))`; cancellation can be intended, so this is
not a score to maximize. Tag trainable prior and auxiliary parameters separately
from G.

Depth can also damage directional structure while norms look healthy:
“shattered gradients” are a distinct concern.
[Balduzzi et al., 2017](https://arxiv.org/abs/1702.08591)
Jacobian singular-value structure matters beyond average gain; dynamical
isometry is a useful theoretical reference for deep-network initialization,
not a claim that all trained GAN Jacobians should be identity matrices.
[Pennington et al., 2017](https://arxiv.org/abs/1711.04735)
A recent feedback-alignment preprint also argues for checking reference-gradient
validity and per-layer utility instead of relying on aggregate cosine. This is
adjacent evidence, not GAN validation.
[Hao et al., 2026, preprint](https://arxiv.org/abs/2606.21126)

## 4. Reliability and reachability: occasional probes

### Fixed-state gradient agreement and noise

Freeze G, D, optimizer state, and the model/buffer policy. Draw K independent
batches from one declared sampling stream, all with the same size B and exact
loss semantics. Compute full-batch parameter gradients g_k, globally and by
selected block. Adjacent training steps are not substitutes: both players move.
Report pairwise cosine and the distribution across pairs, omitting pairs with
unresolved norms. Two batches give a cheap but noisy first estimate; K=4–8 is a
proposed starting budget, not an empirically selected setting.

The following is an elementary moment estimator for IID batch gradients:

```text
g_bar = mean_k(g_k)
V     = sum_k ||g_k - g_bar||^2 / (K - 1)  # trace of batch-gradient covariance
S_hat = ||g_bar||^2 - V/K                  # unbiased estimate of ||E[g_k]||^2
single_batch_SNR_hat = S_hat / V           # plug-in ratio, not unbiased
averaged_K_SNR_hat   = K*S_hat / V
```

Keep S_hat and V. If S_hat <= 0 or its uncertainty is large, report
`unresolved_signal`; do not clamp it to epsilon and display a definitive SNR.
If V=0, say `no_observed_variation`, not proven infinite reliability. Streaming
vector moments avoid retaining all K gradients. Do not average per-block SNRs
to manufacture a global SNR.

Only if gradients average IID per-example contributions is `B*V/S_hat` a
simple gradient-noise-scale estimate in sample units. The gradient noise scale
literature motivates this diagnostic and its relation to batch efficiency.
GAN critic drift and batch-coupled objectives limit that interpretation; this
report does not derive an optimal GAN batch size from it.
[McCandlish et al., 2018, §§2.2, 2.4 and Appendix A.1](https://arxiv.org/html/1812.06162v1)

In particular, splitting HyperGAN's globally coupled relativistic objective
into independent microbatch losses changes the question. Use independently
drawn complete objective batches. Snapshot probes can draw multiple batches
from one stream without rerunning a training experiment under another seed.
Condition-wise disagreement and deliberately balanced batches need their own
interpretation; they are not automatically IID noise estimates.

### Does the requested change survive G's geometry?

On matched fixed latent/conditioning inputs, compute the actual output change
`delta_X = G_after(Z,C) - G_before(Z,C)`, output RMS displacement, and
`cos(-q, delta_X)`. This says whether the output follows the critic's request,
not whether the request is good. Show distributions and block contributions
instead of only a flattened cosine. A matrix-free Jacobian-vector product
`J_theta delta` additionally separates predicted output motion from nonlinear
effects, without materializing the enormous Jacobian.

For a deeper geometry audit, several vector-Jacobian/Jacobian-vector products
can estimate directional gain or dominant singular values at suspect blocks.
A few random probes cannot certify the smallest singular value or the absence
of a nullspace. Full spectra and per-example parameter gradients are poor
first defaults. For coherence/shattering, compare nearby-input gradients or
fixed-input gradients under controlled perturbations, with parameter state
held fixed. Natural variation across modes is not itself a defect.

## 5. The nearest practical north star: independent update utility

Choose and freeze a lower-is-better evaluator Q using held-out real samples and
a held-out latent/conditioning bank H. It must not be the current critic's loss.
Evaluate the **actual G update**, keeping evaluation buffers/mode, feature
extractor, samples, and all other state fixed:

```text
progress_Q = Q(theta; H) - Q(theta + delta; H)
```

Positive means this update improved this evaluator; negative means it harmed
it. Raw signed change should be the primary value. Optionally report progress
per measured update second and per declared output/feature displacement, while
retaining their denominators and marking tiny denominators unavailable. Neither
ratio is comparable across arbitrary Q definitions.

If Q is differentiable, first-order predicted progress is `-grad(Q)^T delta`.
The corresponding cosine is
`C_Q = -grad(Q)^T delta / (||grad(Q)|| ||delta||)`.
The ideal local direction under a **specified Euclidean parameter norm** gives
C_Q=1; a different geometry or optimizer changes the relevant notion of best.
This is not parameterization-invariant. Finite progress also depends on step
size and curvature, and can be negative even when C_Q=1. The cosine is undefined
at an optimum with zero reference gradient; the finite change can still detect
a harmful departure from that optimum.

A paired before/after test avoids requiring derivatives of Q and so has the
broadest model coverage. For frozen-state counterfactual evaluation, copy the
optimizer state and apply one shadow update to copied models. Never apply and
undo an update on the live trainer. Specify whether the update includes all G
objectives or adversarial-only: the latter is an attribution probe, not the
actual training step. If learned priors/auxiliary components also affect
generation, report whether they move jointly or stay fixed. To attribute to G
alone, keep their states and latent values fixed. Separate buffer changes from
parameter changes in either case.

Suggested evaluator ladder:

| Setting | Independent evaluation | Main caveat |
| --- | --- | --- |
| Controlled 1-D distribution | Exact Wasserstein-2 squared via sorted equal-weight atoms | An oracle only for this toy distribution |
| Generic continuous outputs | Fixed-kernel multi-scale RBF MMD squared on a declared representation | Kernel bandwidth, finite-sample power, and representation define what is visible |
| Image generation | Fixed-feature MMD/KID plus separate fidelity and coverage measurements | Feature choice can hide defects; a single-update difference may be tiny |
| Conditional generation | Condition-aware distribution/task error plus within-condition diversity | Unconditional quality can improve while G ignores conditions |

Population MMD with a characteristic kernel distinguishes distributions on its
domain. A fixed non-injective feature map can still hide differences in the
original data, and finite sample estimates have uncertainty. Use a predeclared
kernel/representation rather than fitting the evaluator to whichever update
looks best. [Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html)

KID is a polynomial-kernel MMD in Inception features; it is not a universal
characteristic-kernel oracle. Work on MMD GANs also distinguishes fixed-critic
gradient estimation from the bias introduced by learning the critic on data.
[Bińkowski et al., 2018](https://arxiv.org/abs/1801.01401)
Use separate fidelity and coverage reporting because one aggregate quality
number can obscure the tradeoff.
[Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991),
[Naeem et al., 2020](https://arxiv.org/abs/2002.09797)

Do not make a tiny one-step FID change the first diagnostic. Reuse paired
samples and independent held-out blocks; report paired differences and sampling
uncertainty, with resampling methods appropriate to the evaluator (pairwise
kernel terms are not independent samples). If the change is unresolved, increase
the evaluation budget or additionally measure a declared multi-step horizon.
That horizon mixes subsequent critic changes with G learning and must not be
called one-step signal quality. A fixed evaluation bank used repeatedly for
model selection needs a separate audit bank to avoid overfitting the evaluator.

## 6. Deterministic evidence in this PR

Run `python3 scripts/generator_signal_proof.py` to print the complete calculation,
or add `--output reports/generator-signal-proof-2026-09-21.json` to write it.
The committed [JSON](generator-signal-proof-2026-09-21.json) contains 12 cases,
runtime versions, and analytic checks. The
[script](../scripts/generator_signal_proof.py) uses explicit CPU float64 tensors;
it performs no RNG draws, image inspection, dataset loading, or training runs.

| Counterexample | Measured result | Implication |
| --- | --- | --- |
| 16 affine blocks, gain 0.5 vs gain 1 | Output derivative RMS=1 in both; first/output ratio=0.0000305176 vs 1 | End-to-beginning profiling detects attenuation |
| Same gain-0.5 chain | First and last weight gradients are equal, but first/last bias gradients differ by 32768x | Parameter aggregation can conceal an activation-path problem |
| Saturated final tanh | Output derivative RMS=1; pre-tanh RMS about 0.000671 | Tap both sides of the final activation |
| Hidden representation scaled 1000x, following weight divided by 1000 | Identical output; hidden gradient 1000x smaller; RMS(h)*RMS(gradient) unchanged | Raw layer norms alone can misdiagnose a change of coordinates |
| Loss multiplied by 1000, SGD rate divided by 1000 | Gradient 1000x larger; identical 0.1 update and quality improvement | Absolute strength is not intrinsic usefulness |
| Helpful and harmful linear critics | Equal gradient magnitude, batch cosine=1, zero sample-gradient variance; W2 squared changes 4→3.61 and 4→4.41 | Strong and perfectly consistent can still be wrong |
| Helpful direction with step size 5 | Oracle cosine=1, but W2 squared changes 4→9 | Direction alone misses overshoot |
| Critic rewards contraction at an already matched distribution | Critic loss falls; variance falls 0.625→0.549316; W2 squared rises 0→0.00244141 | Critic progress and agreement can accompany loss of coverage |
| Dead ReLU gate | Output derivative RMS=1, parameter gradient=0 | Strong critic signal can be blocked completely |
| Exhaustive IID gradient-pair enumeration | Expected debiased squared signal=1 and covariance trace=1; opposing pair gives negative signal estimate | Moment estimates need unresolved/zero handling |

These are constructed counterexamples and formula checks, not evidence that a
particular HyperGAN run has one of these failures. No production GPU overhead,
predictive correlation, or training-quality improvement has been measured here.

## 7. Cost and fit with HyperGAN

Operation counts below are engineering estimates. Let P be observed parameter
count, A the number of observed activation elements, and K the number of probe
batches. Actual latency depends on launch costs, memory traffic, model kernels,
and execution mode; “no extra backward” does not mean free.

| Measurement | Extra numerical work | Extra storage / limitation |
| --- | --- | --- |
| Existing parameter-gradient RMS/zero counts | O(P) reductions; no extra model pass | A few scalars per block; bandwidth and kernel launches |
| Activation-gradient profile during existing backward | O(A) reductions; no extra model pass | Scalar summaries if hooks reduce immediately; avoid retaining full gradients |
| Exact per-block update | O(P) copy/subtraction/reductions | O(P) old parameter values unless optimizer exposes exact delta |
| Adversarial vs auxiliary separation | Additional differentiated loss pass(es), or recomputation | Retained graph or forward/backward memory; number of objectives matters |
| K independent batch gradients | K frozen-state forward/backward probes | Graph memory plus O(P) streaming vector moments |
| Paired output displacement | Before and after G forwards on the bank | Bounded matched outputs/features, or streaming accumulators |
| Independent Q progress | Those forwards plus evaluator work twice | RBF MMD naive pair work O(N²); streaming blocks can bound storage |
| Reference-gradient alignment | Differentiable evaluator forward/backward | Includes evaluator and G graph memory |
| Jacobian probes | One or more JVP/VJP passes per direction/iteration | No dense Jacobian; operator support must be checked |

Proposed starting policy, to be benchmarked: collect selected training gradient
summaries every 100 updates; do K=4 fixed-state batch probes around every 1000
updates or on demand. Prefer early, middle, and stalled existing checkpoints
for research over starting new training runs. Choose evaluation sample count by
whether paired progress is resolvable, not a universal fixed N. If a probe costs
E seconds and ordinary updates cost T seconds, doing it every I updates adds
roughly `E/(I*T)` compute relative to ordinary training on the same device.
Separate-device execution still has snapshot, transfer, and contention costs.
Measure step latency, throughput, peak memory, and evaluation latency before
selecting defaults. No repeated-seed experiments are proposed.

The relevant code paths at the inspected revision are:

| Existing location | Observation / integration consequence |
| --- | --- |
| [objective_program.py: run_native_program](../src/hypergan/objective_program.py) | D updates before G; G total loss calls backward before `opt_g.step()`. This is the correct live phase for activation hooks and reduced parameter-gradient observation. Capture exact delta around the G optimizer call. |
| [training.py: DeviceAdam and trainer construction](../src/hypergan/training.py) | G optimizer includes generator/auxiliary parameters and optional trainable prior parameters. Record their ownership; do not call the whole optimizer vector “generator layers.” |
| [distributed_training.py: _reduce_gradients and accumulation](../src/hypergan/distributed_training.py) | Measure actual optimizer gradients after reduction and accumulation. Global norm is the norm of the reduced gradient, not the average of rank norms. Accumulation uses discovery/replay and full-logit derivatives; count the effective G backward once, not every discovery/replay forward. |
| [metrics.py](../src/hypergan/metrics.py) and [metric_plugins.py](../src/hypergan/metric_plugins.py) | Current built-ins expose losses/timing/diversity; scalar plugins receive declared finite JSON sources, not live autograd tensors. Layer derivatives require numerical collection, not a JSON callback pretending to access gradients. |
| [evaluation_snapshot.py](../src/hypergan/evaluation_snapshot.py) and [artifacts.py: bundle_state](../src/hypergan/artifacts.py) | Existing snapshots copy EMA inference state and selected generation/evaluation dependencies. They do not supply the required training-state G/D/optimizer phase. |
| [metric_evaluation_worker.py: evaluate_snapshot](../src/hypergan/metric_evaluation_worker.py) | Reconstructs eval-mode models with gradients disabled, and runs metrics in inference mode. Current snapshot plugins cannot perform this gradient audit. |

A future training-signal evaluator needs a separately identified immutable
training snapshot with non-EMA G, D and all required dependencies, objectives,
trainable prior, optimizer/scheduler state when studying delta, buffers, phase,
RNG provenance, numerical policy, and declared batch/conditioning inputs. A
complete-boundary snapshot can answer a frozen-state question; matching the next
actual G step additionally requires replaying the next D phase and exact draws.
An actual in-flight G signal is after the current D update and before the G
update. Do not confuse those snapshots or label EMA gradients as the signal
used for training. Bounded worker/device scheduling and receipt infrastructure
may be reused, but the artifact and numerical contracts must be extended.

For passive training collection, prefer tensor hooks that only reduce to
detached on-device scalars and return no replacement gradient. Retain no
autograd graphs in event/callback workers. Batch scalar transfers with the
existing metric transfer path; avoid `.item()` per layer or accidental
synchronization. Hooks can affect compilation/capture and must be measured in
the supported backend.
[PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd)

For frozen-state probes, restore/reset copied buffers and stochastic state per
the declared protocol, preserving training-mode semantics where required.
Switching all modules to eval mode silently changes BatchNorm/dropout behavior.
No measurements may mutate live gradients, weights, optimizer moments, data
cursors, or RNG streams. Where AMP is used, unscale gradient observations first.
For replicated/accumulated activation derivatives, validate normalization
against the full logical-batch reference, especially globally coupled losses;
local derivative scaling can differ from the post-reduction parameter gradient.

Publish metric definitions with model/critic/step/phase identity, block path,
objective, reduction, tensor shape, dtype, batch/world/microbatch sizes,
parameter ownership, probe/data/feature identity, and normalization version.
Keep statuses such as `frozen`, `disconnected`, `zero_gradient`,
`unresolved_signal`, and `unsupported_derivative` distinct. Existing finite-only
publication must omit undefined scalar values and attach measurement status,
not encode infinity, NaN, or a fabricated epsilon ratio. The proof JSON uses
null for undefined cosine; it is not the production event schema.

## 8. What to build and validate next

First implement an opt-in output/first/middle/last gradient profile with
activation scale, parameter gradients, and actual block updates. Then add a
frozen-state independent-batch agreement probe and paired independent progress
evaluation. Keep the quantities visible separately; do not combine them into
one weighted health score or use them as a regularizer/controller yet.

Before accepting an implementation, require:

1. **Measurement correctness:** reproduce these analytic counterexamples,
   handle disconnected/frozen/zero paths, and distinguish loss scaling from
   function changes. Confirm first-block and pre/post-output-activation taps.
2. **Training transparency:** observer enabled/disabled on the same exact
   inputs and state must preserve the declared numerical semantics, buffers,
   RNG, optimizer state, and update. Check supported accumulation/replicated
   modes against their logical-batch reference. These are equivalence checks,
   not seed sweeps.
3. **Practical expense:** measure overhead on the actual shallow/deep recipes,
   including snapshot capture. Avoid a universal overhead claim from toy CPU
   timings.
4. **Predictive value:** audit existing checkpoints spanning progress and
   stalls. Ask whether weak first-block signals, unstable batch directions, or
   negative independent progress predict subsequent quality/coverage changes.
   Observational correlation does not establish a causal fix.

Interpretation should guide the next controlled investigation:

| Observed pattern | Next hypothesis to examine |
| --- | --- |
| Weak output q | Critic/loss saturation, objective weight, or a break in critic preprocessing |
| Strong output q; sharp drop before output nonlinearity | Output saturation or output scaling |
| Healthy late signal; weak early activation/parameter signal | Block Jacobian gain, residual routing, activation scaling, numerical precision |
| Healthy gradients; nearly absent actual updates | Optimizer moments/rates, parameter ownership/freezing, or numerical update resolution |
| Strong repeatable updates; negative independent progress | Critic/evaluator mismatch, wrong direction, excess step size, or conflicting objectives |
| Good quality/fidelity; declining coverage | Missing-mode signal, contraction, conditioning failure, or evaluator blind spots |
| Low SNR but useful independent progress | Legitimate heterogeneous directions or stochastic learning; do not automatically increase agreement |

This gives a concrete answer to “once the generator gets a signal, is it good?”:
measure **where it arrives, where it survives, what it changes, and whether that
change helps**. Only the last question is a candidate north star, and it needs a
declared external definition of progress.

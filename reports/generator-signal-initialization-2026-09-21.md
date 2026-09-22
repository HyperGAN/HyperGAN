# Research follow-up: calibrating initialization and testing signal drift

Companion to [generator signal quality](generator-signal-quality-2026-09-21.md).
Proposal only: no initialization, config-search agent, or online adaptation is
implemented by this PR.

**Yes, initialization calibration is a plausible first application.** Make the
initial network capable of transmitting a range of signals, then check whether
its actual optimizer updates improve an independent evaluator. The hypothesis
worth testing is that calibration improves the *duration* of useful learning,
not that a good initial gradient profile is automatically preserved.

The distinction matters at startup: an untrained discriminator usually cannot
be treated as a trustworthy teacher. We can calibrate the generator's capacity
to receive instructions before we can establish that this particular critic's
instructions are useful. Even excellent conditioning cannot supply information
the critic has not learned.

## Existing methods that make this plausible

| Method | What it contributes | Applicability limit |
| --- | --- | --- |
| LSUV | Orthogonal initialization followed by data-driven, sequential adjustment of layer output variance | Controls forward scale on calibration data; does not certify useful backward directions or GAN convergence |
| Dynamical isometry | Looks at the spread of Jacobian singular values, beyond gradient magnitude in one direction | Strong results assume particular architectures/initialization settings; not an all-GAN guarantee |
| Fixup | Depth-aware residual initialization designed to control initial updates without normalization | A residual-network recipe with coordinated scaling, not a universal multiplier for every layer |
| DeepNorm | Combines Transformer residual scaling and initialization to control deep-network updates | Architecture-specific; not a drop-in theorem for arbitrary GANs |
| Maximal-update parameterization / muTransfer | Makes width scaling and hyperparameter transfer more principled | Primarily addresses width/parameterization, not a solution to GAN critic drift or arbitrary depth |

Sources: [Mishkin and Matas, LSUV](https://arxiv.org/abs/1511.06422),
[Pennington et al., dynamical isometry](https://arxiv.org/abs/1711.04735),
[Zhang et al., Fixup](https://arxiv.org/abs/1901.09321),
[Wang et al., DeepNet](https://arxiv.org/abs/2203.00555),
[Yang et al., Tensor Programs V](https://arxiv.org/abs/2203.03466).
These are research directions to evaluate for the network family in use. They
do not establish that one initializer solves signal quality for every GAN.

An automatic initialization pass could first apply a suitable known recipe,
then use observed forward/backward statistics to adjust a small set of gains.
It need not add a regularizer or change the training objective. However, it
does change the initialized model, so it belongs to model construction and
must be recorded with the resulting initial checkpoint.

## What “ideal at initialization” should mean

Use a feasibility envelope with several measurements, not a target of maximum
gradient magnitude or exactly equal gradients in every layer:

- Finite, resolvable activation and gradient scales at the selected boundaries.
- No unexplained dead trainable routes or saturation at the final activation.
- Useful gain across several output directions, not only today's critic q.
- Trainable early and late blocks make sensible first-step changes under the
  actual optimizer; record zero-initialized residual branches explicitly.
- Latent/condition sensitivity and generated variation remain present.
- Once the critic has informative feedback, held-out quality and coverage
  improve over a declared short training horizon.

Some valid residual initializations deliberately start with zero branch outputs
or temporarily zero gradients in upstream branch parameters. An “all layers
must be nonzero at step zero” filter would incorrectly reject them. Assess the
prescribed warm-up behavior and branch contribution after initial updates. A
large input-output Jacobian is also no proof of parameter learning: latent
sensitivity and parameter sensitivity are separate checks.

The feasibility ranges need architecture, dtype, output scaling, and optimizer
context. Start with a known functioning recipe as a reference and retain every
raw numerator/denominator. Ratios alone can be manipulated by rescaling hidden
coordinates. Improving transmission while destroying the generator's output
scale or diversity is not a successful calibration.

At startup use two complementary backward probes:

1. **Generator geometry:** inject several declared unit output cotangents into
   G and observe their transmission to blocks and parameters. Balanced fixed
   directions or one fixed bank of isotropic probes reduce dependence on the
   current critic. This tests transport of a signal, not its semantic quality.
   Do not construct a dense Jacobian or infer all singular values from a few
   directions.
2. **Actual task signal:** use the exact G objective and a declared D state.
   At step zero this is mostly a wiring/scale check. At later checkpoints it
   becomes a learning diagnostic, paired with independent progress.

To compare G-only initialization candidates, use the same discriminator state
and protocol at the initial probe. For subsequent short training comparisons,
start both players and optimizers from equivalent declared states and let D
co-evolve with each candidate. A D adapted to the baseline generator can unfairly
rank alternatives, while a random D is not an external quality oracle.

## Why an initially healthy signal can drift

There are several independently changing quantities:

```text
generator signal at time t = J_G(theta_t, inputs_t)^T q(D_t, G_t, batch_t)
actual update at time t    = optimizer(signal_t, moments_t, schedule_t)
```

Weights, biases, activation regimes, normalization statistics, generated
samples, critic direction, and optimizer moments all change. Spectra are not
conserved by ordinary updates. Residual structure or initialization can make
conditioning more robust, but an initial measurement alone provides no bound
on those later changes.

The [deterministic proof](../scripts/generator_signal_proof.py) contains two
explicit examples:

- A scalar chain of 64 unit-gain layers initially transmits a gradient exactly.
  If each gain becomes 0.95, transmission is `0.95^64 ~= 0.0375`; at 1.05 it is
  about 22.7. These are constructed parameter states, not observed training
  trajectories. They disprove automatic persistence, without claiming that
  actual training takes this path.
- A fixed generator with Jacobian `diag(1, 0.001)` transmits one unit critic
  direction strongly and the orthogonal unit direction 1000x less strongly.
  The initial actual gradient can look healthy even though changing only D's
  direction exposes a weak route. No generator-weight drift is necessary.

Even a parameter Jacobian that stays constant would not guarantee useful
learning: q may become wrong, inconsistent, or lose support for missing modes.
That is why both geometry probes and independent update utility remain useful
after initialization.

## A bounded config-search loop

Begin with an outer loop over **initialization configuration**, before trying
online layer adaptation. An agent can propose interpretable configuration edits;
a numerical runner should enforce the budget, comparison protocol, and
acceptance criteria. The same runner could later use grid search, coordinate
search, or another optimizer without changing the measurement definition.

Suggested stages (future work; not commands supported by today's CLI):

1. **Declare the search.** Freeze the base recipe, initial tensor/randomness
   source, calibration bank, separate validation bank, evaluator, compute
   budget, and editable fields. Save each candidate as its own resolved config
   and artifact rather than changing an active run's config in place.
2. **Use a small search space.** Start with generator initialization family,
   grouped block gains, residual-branch scale, and output-head scale only where
   the component supports them. Do not simultaneously tune loss weights to
   inflate the very gradients being measured. Keep optimizer settings fixed
   during this first attribution study.
3. **Screen cheaply.** Construct candidates and evaluate activations, several
   cotangent probes, and actual-objective gradients on the fixed banks. Use
   bounded sequential gain changes and remeasure the whole network; adjusting
   one layer affects those after it. Do not repeatedly rescale every weight by
   inverse gradient norm: that is coupled to the critic and can destabilize
   the forward computation or be canceled by normalization.
4. **Check optimizer response.** On isolated copies, take the actual first few
   optimizer steps and evaluate block updates and output displacement. Raw
   gradient normalization is inadequate because Adam's moment normalization
   changes how scale maps to updates.
5. **Validate finalists.** Only a few surviving candidates receive equal-budget
   short GAN training runs and independent quality/coverage evaluation, from
   the same declared initial source and data sequence. Choose the horizon in
   advance. No same-configuration/different-seed runs; a changed initialization
   method or gain is a substantive candidate, not a seed sweep. Statistical
   claims remain conditional on this protocol.
6. **Choose by useful learning.** Reject numerical failures first, then compare
   independent progress and retained coverage at equal compute. Keep signal
   profiles as explanatory evidence. Retain the baseline if differences are
   unresolved; do not select whichever scalar happens to peak.
7. **Persist and monitor.** Record the chosen config, initialized weights,
   constructor/source versions, calibration transformations and data identity,
   budget, and rejected candidates. Continue passive monitoring during ordinary
   training and audit against data not used for candidate selection.

Reuse the same explicit base tensors for gain-only candidates. Changing
architecture or initialization family may change tensor shapes or RNG draw
order; sharing a seed alone does not make those initializations equivalent.
Record that limitation rather than claiming all candidates are paired exactly.

This is more practical than exposing every weight as an independent search
variable. Layerwise gains can be automated, but their search objective should
remain constrained by output behavior and update utility. Architecture changes
such as adding residual paths are a later, separately attributed intervention.
Do not reinitialize pretrained components during calibration.

For HyperGAN, the initializer options must be inventoried per component/HNDL
definition before any runner writes configurations. Existing factory arguments
are not a universal layer-initialization API. Proposed fields here are search
concepts, not new accepted TOML keys. The gradient-evaluation snapshot extension
described in the main report is another prerequisite.

## Test the persistence hypothesis directly

For baseline and a small number of substantively different initialization
candidates, measure at initialization, after the first update, after a declared
short warm-up, and at logarithmically spaced later checkpoints within equal
budgets. Use one fixed diagnostic bank for longitudinal comparison and a
separate held-out bank for selection/audit. No seed sweep is needed to conduct
this first conditional study; it will not establish robustness across seeds.

Track the time series of first/middle/last transmission, activation scales,
selected Jacobian-direction gains, block update sizes, independent quality
progress, and coverage. For a positive quantity R, `log(R_t/R_0)` summarizes
drift, but retain R_t and R_0 and mark zero/near-zero references unresolved.
Drift away from the initial profile is not inherently bad: some change may be
necessary to learn. Define loss of usable transmission and negative independent
progress separately rather than treating any departure as failure.

To separate causes at a checkpoint, use copied, compatible G/D states and
matched inputs for a small crossed audit: current versus reference G, current
versus reference D. Interpret cross-pairs as counterfactual diagnostics, not
trained GAN quality scores; a critic evaluated on unfamiliar generated samples
may be out of distribution. Fixed output-cotangent probes isolate G's changing
geometry more directly.

Evidence for the user's hypothesis would be an initially calibrated candidate
retaining usable early-layer updates longer *and* achieving better independent
progress/coverage at equal compute. A better initial profile alone is not
sufficient. Existing-checkpoint audits can establish what drifts today; testing
whether a different initialization prevents it needs an actual controlled
initialization comparison. This PR supplies neither empirical claim yet.

## Keep online adaptation a separate decision

Changing initialization after training has started is reinitialization: it can
discard learned representations. Even a function-preserving rescaling can
change optimizer dynamics; optimizer moments need corresponding treatment, and
arbitrary layer rescaling may not preserve the function at all.

If passive monitoring later supports an online controller, start by evaluating
bounded changes to explicitly supported training controls on copied states,
with persistence windows/cooldowns and independent progress checks. Exact
controls should follow observed failure mechanisms. A controller that chases a
constant gradient norm or maximal agreement can fight useful learning or reward
collapse. Online control does not need to be a regularizer, but it is a new
training algorithm that needs its own evidence.

The immediate deliverable to pursue is therefore a **calibration evaluator and
config comparison loop**, followed by a direct drift study. Automatic layer
adjustment becomes an informed engineering choice once those measurements show
which aspect of the signal fails and which changes preserve useful learning.

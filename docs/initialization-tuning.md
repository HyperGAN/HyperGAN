# Startup tuning

For a **new run**, add `--tune` to the normal training command:

```sh
hypergan train config.toml --run-dir runs/tuned --tune
```

Tuning measures how the actual optimizer updates affect G and D, derives a
bounded learning-rate proposal, and checks it before training starts. The
console and dashboard show **Tuning startup**, with stages shown as needed: measuring
optimizer updates, fitting directional curvature, validating held-out response,
and replaying proposed rates. They display the current G/D factors and the
update counter during disposable baseline and replay updates.

The result reports selected G/D rates or **Startup tuning unresolved**.
Unresolved tuning keeps the configured rates
and records why no measured adjustment was accepted. Unsupported ownership or
measurement contracts are skipped with a reason. Neither result means the
startup problem was fixed. Progress remains visible independently of the normal
training print interval.

Tuning is opt-in; `--no-tune` uses configured rates directly. Native CPU and
single-GPU execution are supported. Replicated execution with `--tune` is rejected
before creating a run.

## One measured-update pipeline

The pipeline starts from the configured initialization. It does not rescale
layers or replace weights. It runs at most **eight disposable baseline updates**
with the configured losses, Adam updates, learned prior, and D/G update order.
The actual phase-local optimizer displacement is the measurement direction,
rather than a raw gradient or cumulative movement over the whole trial.
G is anchored at its **first update**, after the first D update and before G or
the learned prior have moved. D is anchored at **update eight**, including the penalty if its configured
schedule activates it (as with `lazy_k=8` on the TransGAN testbed). Normal training retains the configured lazy schedule.
The first G anchor avoids fitting only after an early saturation transient.

After the baseline, tuning measures both players even if the baseline passes
the startup guards. It evaluates each player's loss along its own recorded update while
holding the opponent and other parameter groups fixed. A predetermined stencil
at displacement factors `0`, `0.5`, and `1` on two fitting banks estimates a
local quadratic. The phase loss includes its configured auxiliary terms and
the penalty when that update's lazy schedule activates it. Four real batches
are reserved after the baseline. Their fixed latent tensors are materialized
under each player's phase-local prior using the same reserved sampling RNG.
Startup guards use the G anchor's initial-prior latent values throughout:

```text
phi(s) = loss(parameters + s * actual_optimizer_displacement)
a = 4 * phi(0.5) - phi(1) - 3 * phi(0)
k = 4 * (phi(1) - 2 * phi(0.5) + phi(0))
proposed factor = -a / k
```

A proposal requires resolved descent and positive directional curvature.
Output-variation and transmission guards may reject a proposal; they neither
derive its rate factors nor decide whether to measure curvature.
Flat, negative, inconsistent, or numerically unresolved curvature does not
justify inventing a smaller rate. G and D are measured separately; an unresolved
player keeps its configured factor. If both players have resolved fits calling
for no reduction and the baseline passes its guards, the configured rates are
kept. Otherwise, no resolved reduction means an unresolved result. Passing
baseline guards does not turn an unresolved fit into a successful calibration.
The factors are restricted to the declared reduction-only range of 0.1 to 1.
A computed factor below 0.1 is unresolved; it is not rounded up. These are directional loss measurements, not an estimate
of the largest Hessian eigenvalue or an optimal GAN learning rate.

Two separate validation banks check the proposed changes before a single **eight-update
coupled replay** from the original state. Changing D changes G's subsequent
optimizer direction, so independently favorable player measurements are not
accepted without that replay. The replay checks finite learning signals,
parameter movement, and the declared output-variation/transmission guards.
A proposal that fails validation or replay is rejected as a whole, without
searching another combination. There is no rate grid, fixed D-half fallback,
or G-retention formula selecting an alternative rate.

The report also measures the first and eighth G update at fixed latent values.
When the native output path identifies a final owned affine layer through layout
operations and optional tanh, it records that layer's activation displacement
using the same forwards. On TransGAN this exposes pre-tanh changes that saturated
pixels can hide. Unsupported output paths are explicitly skipped. This is an
observation, not a response threshold used to select learning rates.

The hard training-update budget is **eight baseline plus eight replay updates**.
This excludes the additional loss evaluations, signal probes, model snapshots,
and state verification; it is not a wall-time guarantee. The report records
measurements and decisions. Historical timing from earlier tuners does not
establish the cost of this pipeline.

All disposable weights, optimizer moments, EMA, counters, gradient fields,
RNG streams, data state, and model buffers are restored before real step zero.
Only accepted G/D learning-rate factors are retained. The learned prior's
absolute rate stays configured. Pretrained weights and buffers, frozen tensors,
and protected storage aliases are excluded from mutation; frozen operations
can still transmit gradients. Unknown ownership is not treated as permission
to calibrate external weights. Audit or persistence failures roll back and stop
startup instead of training with partially applied changes.

## Selected rates and normal training

Fresh tuned runs keep the selected G and D base learning rates. The configured
annealing schedule still applies. There is no automatic ramp back to the
original source rates and no warmup option for new tuning runs.

With previews enabled, fresh runs capture a **step 0** preview after all tuning
state has been restored and before the first retained optimizer update. The
initial weights and EMA remain at their configured initialization. Preview
rendering uses the normal worker, so the saved step-zero result may appear
after training starts. Untuned new runs also get this preview; `--no-previews`
disables it, and resume does not repeat it.

Older checkpoints that already recorded a G warmup retain their saved schedule
on resume. The dashboard and console continue to display that historical warmup.
The new pipeline does not rewrite old checkpoint schedules.

## Saved overrides and resume

Source config and `.hndl` files remain unchanged. Each run records its own
baseline, proposed changes, and measurements:

```text
runs/tuned/tuning/config.base.json  resolved baseline recipe
runs/tuned/tuning/overrides.json    selected G/D learning-rate overrides
runs/tuned/tuning/report.json       measurements, decisions, and state verification
```

The initial full checkpoint stores the unchanged initial tensors, selected
optimizer rates, and restored training state. Metadata records the tuning
artifact hashes. Override JSON explains the checkpoint; it is not replayed.

```sh
hypergan resume runs/tuned
```

Resume restores saved rates and continues from the saved step. It never repeats
tuning or multiplies the factors again. Repeating `train` on an existing run
also resumes it; `--tune` prints a reminder that startup tuning is skipped.
Use separate run directories to compare substantive configuration changes.

## What passing means

This is an experimental, local check of numerical update response. Three stencil
points always fit a quadratic; an exact fit is not independent evidence that
its curvature predicts other points. Held-out loss checks and coupled replay
provide separate tests, but do not certify image quality, semantic diversity,
useful gradient directions, long-term stability, or convergence. Both players
can make finite updates and still learn undesirable samples. Two probe batches
do not provide a precise uncertainty estimate.

The [update-response research memo](../reports/update-response-research-2026-09-21.md)
explains the motivation and limits. The [first measured-update testbed](../reports/measured-update-testbed-2026-09-21.md)
completed safely but rejected its proposed pair: held-out losses decreased while
the coupled replay still lost output variation and transmission. It did not fix
that configuration. The [first-update calibration follow-up](../reports/first-update-calibration-2026-09-22.md)
fixes the G anchor and its prior inputs, but its G ×0.86545 / D ×0.47297 pair
also failed the coupled transmission guard. This remains experimental.
Earlier [startup drift](../reports/startup-signal-drift-2026-09-21.md),
[automatic tuning](../reports/startup-dynamics-autotune-2026-09-21.md), and
[warmup drift](../reports/startup-warmup-drift-2026-09-21.md) reports describe older
pipelines. Their measurements remain evidence about those implementations,
not validation of the new selector.

## Inspect without changing initialization

The standalone diagnostic performs one generator backward evaluation without
any optimizer update or calibration. Pass a config for fresh initialization:

```sh
hypergan diagnose-signal config.toml --output signal.json
```

Pass an existing run directory to inspect its latest complete online generator
and discriminator checkpoint:

```sh
hypergan diagnose-signal runs/tuned --output checkpoint-signal.json --device cuda:1
```

Use `--checkpoint CHECKPOINT_NAME` to select an older complete checkpoint within
that run. The diagnostic loads disposable models, verifies checkpoint integrity,
and never writes to the source run. Inference-only generator exports lack the
discriminator and are rejected. Native single-process checkpoints are supported.
The report records the saved and evaluation runtimes and any RNG reset needed
when changing device backends; this is an evaluation, not a training resume.

The report includes generated-output gradients, module activation/gradient RMS,
first-to-output ratios, parameter gradients, shapes, and state hashes. Named
HNDL nodes retain their paths, including attention and residual boundaries.
Discriminator profiles include score gradients, candidate-input gradients, and
intermediate boundaries, with fake and real passes identified separately.
Pretrained interfaces are measured without changing their weights. This can
distinguish a vanishing discriminator derivative from poor generator
transmission. With multiple critics, the candidate gradient is the combined
configured objective; it is not an isolated causal attribution to each critic.
`--objective total` includes auxiliary objectives; the default isolates the
adversarial loss. `--batch-size N` and `--device cuda:1` are explicit probe
overrides, recorded in the report. The default uses the configured batch/device.
An existing output file is never overwritten.

Config-based diagnosis uses the initial critic **before** any D update.
Checkpoint diagnosis uses the saved online pair at the reported step. Neither
substitutes EMA weights. Raw RMS values
depend on batch/shape/reduction and activation coordinates. A lower first-layer
ratio is not automatically worse: interpret it with the full profile and actual
training outcomes. See the [research report](../reports/generator-signal-quality-2026-09-21.md)
for the distinctions between strength, transmission, reliability, and usefulness.

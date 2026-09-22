# Startup tuning

For a **new run**, add `--tune` to the normal training command:

```sh
hypergan train config.toml --run-dir runs/tuned --tune
```

Tuning measures how the actual optimizer updates affect G and D, derives a
bounded learning-rate proposal, and checks it before training starts. The
console and dashboard show **Tuning startup**, with stages shown as needed: measuring
optimizer updates, measuring gradient response, validating held-out response,
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
the startup guards. Each player is measured on two reserved banks with its
opponent, prior, buffers, randomness, and post-update Adam diagonal held fixed.
The complete phase loss includes auxiliary terms and the penalty when that
update's lazy schedule activates it. Four real batches are reserved after the
baseline; phase-local prior values are drawn with matching sampling randomness.
Startup guards reuse the G anchor's initial-prior validation latents.

Instead of fitting a quadratic across a full update, tuning measures two exact
parameter gradients separated by a small displacement along the actual Adam
update. It measures gradient rotation as well as magnitude change:

```text
delta = actual optimizer parameter displacement
M = sqrt(bias-corrected Adam second moment) + optimizer epsilon
h = min(0.1, sqrt(machine epsilon) * max(norm(parameters), norm(delta)) / norm(delta))
g0 = gradient of phase loss at parameters
gh = gradient of phase loss at parameters + h * delta
C = norm_M(delta) * norm_inverse_M(gh - g0) / h
proposed factor = min(1, -(g0 dot delta) / (2 * C))
```

Both banks must resolve descent and a nonzero gradient response. The smaller
factor is proposed. Signed directional curvature may be negative: the measured
vector response still captures rotation that a scalar loss curve misses. The
finite difference interval is a numerical heuristic, not an ideal signal target.
The factor is strictly positive and reduction-only; there is no 0.1 floor that
would round a derived smaller rate upward. Unrepresentable perturbations,
unresolved differences, missing Adam state, or non-descent measurements abstain.
An unresolved player retains its configured factor; the coupled replay must still
accept the whole pair. If both resolved proposals call for no reduction and the
baseline passes its guards, configured rates are kept.

This is an empirical local estimate, not a bound on smoothness elsewhere or a
universal optimal learning rate. The factor of one-half reproduces the gradient
secant term of [Adaptive Gradient Descent without Descent](https://proceedings.mlr.press/v119/malitsky20a.html)
in the ordinary gradient-descent special case. This startup adaptation to Adam
and GANs does not inherit that algorithm's convergence guarantees. Output and
transmission guards can reject a proposal, but do not generate the rate.

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

This is an experimental, local check of numerical update response. A gradient
secant measures only the chosen direction and interval; it does not bound an
entire neighborhood or the full coupled game. Held-out loss checks and coupled replay
provide separate tests, but do not certify image quality, semantic diversity,
useful gradient directions, long-term stability, or convergence. Both players
can make finite updates and still learn undesirable samples. Two probe batches
do not provide a precise uncertainty estimate.

The [gradient-response testbed](../reports/gradient-response-startup-2026-09-22.md)
passed startup retention checks with G LR about 9.75e-7 and configured D LR
0.0002. Its conservative rate still needs longer evaluation for learning speed,
evolving-prior samples, and the previous 350–500-step failure window.

The [update-response research memo](../reports/update-response-research-2026-09-21.md)
explains the earlier motivation and limits. The [first measured-update testbed](../reports/measured-update-testbed-2026-09-21.md)
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

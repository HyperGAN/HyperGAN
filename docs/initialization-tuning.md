# Startup tuning

For a **new run**, add `--tune` to the normal training command:

```sh
hypergan train config.toml --run-dir runs/tuned --tune
```

The dashboard and console show **Tuning startup**, first during initialization
calibration and then during short trial runs. They show the candidate, trial
step out of 8, and generator learning-rate factor. Progress remains visible
even when ordinary training progress is printed less frequently. Trial updates
are discarded; the actual run still starts at step zero.

The final summary distinguishes the initialization decision from the learning
rate decision: a selected factor, a passing configured rate, or **Startup tuning
unresolved** when no candidate passes. An unresolved result retains the configured
G learning rate; it does not certify that the startup problem was fixed. A
skipped dynamics check includes its reason.

With previews enabled, a new run also captures a **step 0** preview after tuning
and before the first retained optimizer update. Its EMA weights match the selected
initialization, after all trial state has been discarded. Rendering runs in the
normal preview worker, so publication can
arrive after training starts while still showing the saved step-zero state.
Untuned new runs also get this baseline preview. `--no-previews` disables it,
and resume does not repeat it.

Tuning is currently opt-in. `--no-tune` uses the configured initialization and
learning rates. Passing the startup checks has not established better training
quality across recipes, so tuning remains off by default.
Native CPU and single-GPU execution are supported; replicated execution with
`--tune` is rejected before creating a run.

## Learning-rate warmup after tuning

New runs with `--tune` default to a 1,000-update G learning-rate ramp:

```sh
hypergan train config.toml --run-dir runs/tuned-warmup \
  --tune
```

The first retained update uses the G rate selected by tuning. A linear ramp
reaches the original config's G rate on update 1,000, then holds that rate.
The existing configured annealing multiplier still applies. D and prior rates
follow their original schedules. Warmup takes place during normal training,
after all tuning trials have been discarded; it adds no search or trial updates.
The console and dashboard show its progress and current G learning rate.

The ramp is experimental: eight startup trial updates do not establish that
returning to the original rate later will remain stable. Pass
`--tune-warmup-steps 0` to keep the selected rate as its base rate. If tuning retains the original
rate, both ramp endpoints are equal and the option does not change that rate.
Use `--tune-warmup-steps N` with an integer of at least two to change its duration.
Untuned runs have no ramp.

The run's tuning artifacts and checkpoints record the ramp. Resume automatically
continues from the saved training step without restarting tuning or warmup.
The new default does not add a ramp to existing runs that saved none.
Use `hypergan resume runs/tuned-warmup` to continue it.

## What it does

First, a bounded initialization search measures the configured adversarial
gradient and the generator's response to a fixed output-gradient probe. It tries at most
three bounded scale changes to eligible first/final affine generator layers.
Candidates must improve the declared transmission heuristic, keep finite
gradients, and pass output-scale and sample-diversity checks. A separate batch
checks the selected candidate before it is accepted. No discriminator or
generator optimizer updates are taken during this initialization phase.

Next, a dynamics check runs **8 configured training updates** at the configured
generator learning rate. If that baseline fails its checks, a formula uses the
measured startup response to propose **one smaller factor between 0.1 and 0.5**.
The experimental formula is `clip(minimum_retention, 0.1, 0.5)`, using the lowest
finite, nonnegative diversity or first-layer transmission retention across both
probe batches when it falls below 0.25. These constants are heuristic guards.
One additional eight-update trial checks that proposal. Each trial starts
from the same selected initialization, optimizer state, data state and RNG state.
D and learned-prior absolute learning rates stay at their configured values.
The trial uses the configured losses and normal D/G/prior update sequence.
Two matched probe batches check output variation and signal transmission after
the trial. A passing configured rate is kept immediately. Otherwise the proposed
rate is selected only if its confirmation passes. If that confirmation fails,
the configured rate is retained and the result is explicitly unresolved.

The hard budget is **two trials and 16 discarded training updates**, plus signal
probes; a passing baseline takes only eight trial updates. Trials
can move owned G, D and prior parameters, but none of those trial weights,
optimizer moments, counters or RNG/data advances become the actual run's
starting state. Only the selected initialization and G learning-rate factor
are retained. Unsupported trial ownership or recovery conditions cause the
dynamics check to be skipped with a recorded reason.

This search does not change architectures, loss weights, or the absolute D/prior
learning rates. Its checks are short-run guards against measured startup
failures, not proof of useful gradient directions, desirable samples, semantic
diversity or convergence. A passing startup can still drift later in training.

In the 128px DINOv3 testbed, startup calibration passed but G saturated during
the first 20 updates. A separate controlled trial with a smaller G learning
rate reduced that failure. The [matched checkpoint investigation](../reports/startup-signal-drift-2026-09-21.md)
documents that controlled comparison and its limits. The dynamics phase derives
and checks one bounded rate adjustment on each new run rather than applying a
fixed testbed factor or searching a grid of rates.
The [automatic startup test](../reports/startup-dynamics-autotune-2026-09-21.md)
records about 94 seconds of total tuning on this testbed, including about
63 seconds for dynamics. Cost depends on the model and hardware; the fixed
update budget also includes snapshot and probe overhead. The report records
successful command validation and the earlier unresolved state-audit failures.

Only eligible newly initialized layers of a native HNDL generator can be rescaled.
Pretrained nodes and their descendants, frozen parameters, shared storage,
critic weights, normalization state, and the prior are excluded from initialization
calibration. A custom generator with uncertain ownership keeps its baseline.
Pretrained weights and buffers also stay protected throughout discarded trials.
Gradients can still flow through frozen pretrained operations. DINOv3 and other pretrained models are
never reinitialized or rescaled.

Search probes replay fixed inputs and restore training RNG streams, data state,
model buffers, and module modes. This compares substantive initialization/rate
changes, not different random seeds. Execution or persistence errors roll back
the search and stop startup; they do not continue with a partially applied
candidate. A completed but unresolved search is reported separately from an error.

## Saved overrides and resume

Your source config and `.hndl` files remain the baseline. Each run records its
applied overrides in its own folder:

```text
runs/tuned/tuning/config.base.json  resolved baseline recipe
runs/tuned/tuning/overrides.json    selected initialization and G learning-rate overrides
runs/tuned/tuning/report.json       measurements, decisions, and state verification
```

The initial full training checkpoint stores the exact selected weights, matching
EMA initialization, selected optimizer rates, and restored training state. Its
tuning metadata refers to the artifact hashes. The JSON override file explains the changes; it is not
replayed on resume.

Repeating `train` on that run directory resumes the saved checkpoint. Keeping
`--tune` on the command prints a reminder that tuning is skipped for an existing
run. The warmup option is only accepted for a new run; omit it when resuming.
Resume keeps its saved weights and selected learning-rate schedule. To compare
baseline and tuned startup, use distinct run directories
with the same config and seed. Do not point a calibration comparison at an
existing training run and expect it to reinitialize the model.

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

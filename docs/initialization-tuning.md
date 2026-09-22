# Startup initialization tuning

For a **new run**, add `--tune` to the normal training command:

```sh
hypergan train config.toml --run-dir runs/tuned --tune
```

The dashboard and console show **Tuning initialization** before training updates
begin, followed by **Applied tuned initialization** or **Kept baseline
initialization**. Candidate progress is shown even when ordinary training
progress is printed less frequently.

With previews enabled, a new run also captures a **step 0** preview after tuning
and before the first optimizer update. Its EMA weights match the selected
initialization. Rendering runs in the normal preview worker, so publication can
arrive after training starts while still showing the saved step-zero state.
Untuned new runs also get this baseline preview. `--no-previews` disables it,
and resume does not repeat it.

Tuning is currently opt-in. `--no-tune` explicitly selects the existing
initialization. A better gradient measurement has not yet established better
training quality across recipes, so the default initialization is unchanged.
Native CPU and single-GPU execution are supported; replicated execution with
`--tune` is rejected before creating a run.

## What it does

At step zero, a bounded search measures the configured adversarial gradient and
the generator's response to a fixed output-gradient probe. It tries at most
three bounded scale changes to eligible first/final affine generator layers.
Candidates must improve the declared transmission heuristic, keep finite
gradients, and pass output-scale and sample-diversity checks. A separate batch
checks the selected candidate before it is accepted. No discriminator or
generator optimizer updates are taken during this search.

This first version calibrates boundary-layer initialization. It does not search
architectures, tune every attention layer, change loss weights or learning
rates, or establish that the critic's gradient points toward better samples.
The report separates the heuristic from the actual adversarial gradient.
Initialization can drift during training. These limits are why `--tune` is an
explicit experiment rather than the default for every recipe.

Only eligible newly initialized layers of a native HNDL generator can change.
Pretrained nodes and their descendants, frozen parameters, shared storage,
critic weights, normalization state, and the prior are excluded. A custom
generator with uncertain ownership keeps its baseline. Gradients can still flow
through frozen pretrained operations. DINOv3 and other pretrained models are
never reinitialized or rescaled.

Search probes replay fixed inputs and restore training RNG streams, data state,
model buffers, and module modes. This compares substantive initialization
changes, not different random seeds. Failed searches restore the baseline and
stop startup; they do not continue with a partially applied candidate.

## Saved overrides and resume

Your source config and `.hndl` files remain the baseline. Each run records its
applied overrides in its own folder:

```text
runs/tuned/tuning/config.base.json  resolved baseline recipe
runs/tuned/tuning/overrides.json    selected parameter paths and scale factors
runs/tuned/tuning/report.json       measurements, decisions, and state verification
```

The initial full training checkpoint stores the exact selected weights, matching
EMA initialization, and restored training state. Its tuning metadata refers to
the artifact hashes. The JSON override file explains the changes; it is not
replayed on resume.

Repeating `train` on that run directory resumes the saved checkpoint. Keeping
`--tune` on the command prints a reminder that tuning is skipped for an existing
run. To compare baseline and tuned initialization, use distinct run directories
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

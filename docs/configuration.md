# Recipe configuration

A project is a directory containing `config.toml`, or a TOML file passed directly to `hypergan validate` and `hypergan train`. Run `hypergan new demo` to obtain the complete default configuration. Defaults are resolved into the run manifest so experiments do not depend on hidden CLI settings.

New projects use native CUDA execution by default (`hypergan new demo --device cuda:1` selects another visible GPU). Use `--device cpu` for a CPU numerical fixture. Supported `training.device` values are `cpu`, `cuda` and `cuda:N`; structural validation does not probe hardware. The historical programmatic `DEFAULT`/empty recipe remains the explicit CPU reference fixture, so hand-written recipes should declare their intended device. The runtime is a bounded numerical reference. It has a configurable component/data interface, not a qualified folder-of-images recipe or a distributed strategy.

## Components and I/O

`components.generator` and `components.discriminator` are required. Other named components can represent an encoder, feature extractor or auxiliary transform. Each component has a `factory`, constructor `args`, explicit forward keyword `inputs`, and a `trainable` flag.

Factories are built-in identifiers such as `mlp`, `linear` and `identity`, or explicit `module:object` paths available in the Python environment. Custom model factories must construct a `torch.nn.Module`. Unknown constructor arguments fail instead of being silently discarded.

Bindings refer to `latent`, `batch.<field>`, `components.<name>` and their nested outputs. Generator output is available as `generated`; discriminator candidate input is `candidate`. Auxiliary components execute when an input or objective requires them. Cyclic or missing bindings fail. Dictionary/list outputs can be selected through dotted keys/indices.

The runnable [paired synthetic example](../examples/paired-linear.toml) exercises a trainable encoder, conditional generator/discriminator and weighted reconstruction loss:

```sh
hypergan train examples/paired-linear.toml --run-dir runs/paired
hypergan sample runs/paired --count 8 --output paired-samples.json
```

CLI sampling reuses saved example conditions and records that provenance in the JSON output. The Python `hypergan.artifacts.sample(..., inputs={...})` interface accepts fresh batched conditions. This is an I/O fixture, not a colorization or super-resolution model.

For conditional generation, bind a generator keyword to `batch.condition` or an encoder output and bind the discriminator to the corresponding condition. The data factory supplies both the condition and real target. This can express paired tasks, but an image architecture, preprocessing and task evaluation still need their own implementation and qualification.

Trainable auxiliary components belong to the generator optimizer. The discriminator has its own optimizer; discriminator conditioning is detached. Frozen components stay in evaluation mode. Separate encoder optimizers or arbitrary alternating schedules are outside this initial runtime and must not be implied by an encoder's presence.

## Numerical recipe

The configuration separates `prior`, `adversarial`, `gradient_penalty`, `prior_regularizer`, `objectives`, `optimizer`, `training` and `sampling`. The generated default uses a particle prior, paired relativistic logistic loss, b-cap and VICReg. The exact resolved parameters are saved with each run.

The discriminator penalty uses ParticleGAN arm names: `b_cap` is the default; `a_r1r2` denotes its paired zero-centered R1/R2 alternative. Changing an arm produces a custom, unqualified configuration. Do not treat these names as interchangeable with other papers' formulations.

Additional objectives select a loss factory, input bindings and weight. Reconstruction objectives such as MSE or L1 can connect generated output and paired targets. Custom task losses can be imported through the same factory mechanism. The runtime's supported update ownership is explicit; it does not infer a new training algorithm from component names.

Changing objective or regularizer settings may change batch/distributed semantics. No configuration in this checkpoint is approved for multi-GPU or cluster execution. Unsupported device/execution settings must fail rather than silently run a different profile.

## Validation and qualification

`hypergan validate demo` checks the configuration without importing torch or executing custom Python factories. Training constructs components and validates their actual interfaces. Configuration errors, unavailable bindings and runtime failures return a nonzero exit status.

The built-in configuration is labelled reference-only, with its runtime recorded as not certified. It does not receive an application approval stamp. Future qualification must apply to an exact resolved configuration, component versions and tested execution profile. Custom values or implementations receive an unqualified warning and may run when compatible. Numerical-reference qualification does not establish image quality or application readiness. Review the recorded qualification and configuration in the run manifest rather than relying on a recipe's name.

Sample artifacts preserve inference state. Separate [complete training checkpoints](recovery.md) restore optimizer/RNG/data-position state with exact configuration and runtime compatibility. Changing a seed or loading an inference artifact is not exact training resume. [Image-folder data](image-data.md) provides content and preprocessing identity for custom image components; it does not change the reference model into an image GAN.

## Scalar metrics

Metrics publish completed update values to `events.jsonl`, with immutable definitions
in `metrics/catalog-<sha256>.json`. No optional training dependencies are needed to
validate these settings or read a catalog.

```toml
[metrics]
preset = "standard" # "none" starts with no published values
# Every ID is independently removable. Unknown IDs fail validation.
disable = ["loss/d_adversarial_raw", "loss/g_adversarial_raw"]
every_steps = 1     # sample this boundary, not an average of skipped steps

[metrics.overrides."loss/total"]
enabled = false
```

The standard preset includes D/G totals, their combined diagnostic sum, weighted
adversarial, gradient penalty, prior regularizer and additional objective
contributions, raw D/G adversarial values, learning-rate multiplier and update
seconds. The combined sum is **not** a GAN quality score or a joint optimization
objective. Catalogs include formula, coefficient, units, phase owner and immutable
definition hashes. Gradient penalties report whether the lazy schedule applied
and its effective coefficient. Upstream gradient/prior regularizers expose their
weighted contribution only: raw values are explicitly unavailable and never
recovered by dividing by a coefficient.

`preset = "none"` emits no metric values, while lifecycle events and mandatory
finite/update validation remain active. `overrides` can explicitly enable one
built-in ID with the empty preset. Disabling D or G publication does not prevent
the combined sum from using those already computed internal values. Metrics
settings may change during resume without changing numerical identity; the new
attempt references a new catalog and retains the previous revision. Changing
objectives, weights or the training schedule still invalidates numerical resume.

Give additional objectives a stable `id`, for example `id = "reconstruction"`,
to publish `loss/objectives/reconstruction`. Without an explicit ID, a deterministic
content hash names the term. Repeated identical terms require distinct explicit
IDs. Custom scalar factories and manual or interval snapshot evaluation are
described below. Unsupported scheduling modes fail validation. The
[CIFAR recipe](cifar-recipe.md) includes a pinned Inception FID adapter.

## Custom metrics and explicit snapshot evaluation

Custom metrics use ordinary importable Python factories. Structural validation
checks configuration without importing a factory. Runtime preflight executes
`describe()` in a fresh bounded worker and records its source hashes and output
metadata. Selected calls execute in fresh workers too; no live trainer or tensor
is passed to a primitive transform. Every custom metric is explicit opt-in.

```toml
[metrics.custom.loss_ratio]
factory = "hypergan.metric_examples:ScalarRatio"
inputs = { numerator = "update.g_loss", denominator = "update.d_loss" }
mode = "scalar"
every_steps = 100
timeout = 10
on_error = "disable"
```

`describe()` returns `kind = "scalar"` plus optional text `label`, `unit`,
`direction` (`none`, `minimize`, `maximize`) and `description`. For this mode,
`evaluate(*, context, **inputs)` returns a finite Python number. Bindings select
the internal D/G totals, raw/weighted adversarial values, gradient penalty, prior
regularizer, learning-rate scale, step or step seconds via `update.<name>`.
They remain available when the corresponding built-in published metric is
removed. Factory construction/description changes create a different definition
hash on resume, independently of numerical recipe identity.

A selected custom call queues a snapshot of primitive scalars without waiting
for the factory. A background dispatcher runs one fresh CPU worker at a time,
with one outstanding observation per metric (at most 32 total, including
completed results). A busy metric drops new observations with an explicit
`dropped` status; accepted observations record `queued`. Results arrive in
`metric` events at their original training step. Choose a cadence suited to
worker startup cost; built-in scalars require no plugin worker.

The per-metric timeout includes queue time, worker startup, evaluation and
shutdown, with broker cleanup grace additional. Scalar workers hide CUDA
devices, cap numerical-library CPU threads and lower process priority where
supported. These defaults apply only to the disposable worker. There are no
implicit retries. `on_error = "disable"` records a visible reason and disables
that metric for the remainder of the attempt; a new attempt preflights it again.
`on_error = "fail"` fails the attempt when its asynchronous result is collected.
Normal completion drains accepted observations before its final checkpoint.
A cooperative signal cancels outstanding observations and reaps their workers,
including when the signal arrives during terminal draining. Optional observations
record `cancelled` at their source step; required observations retain `on_error =
"fail"` and fail the attempt on cancellation. Exceptional shutdown also cancels
outstanding observations and reaps their workers.
A required failure can therefore arrive after subsequent completed updates;
resume replays from the selected complete checkpoint. Disabling optional
publication never disables mandatory numerical validation.

Factories are trusted Python. Input/output bytes and deadlines are bounded;
arbitrary allocations, external state and subprocesses created by plugin code
are not sandboxed. Instances and global RNG state are isolated from training,
but persistent evaluator state across calls is not supported.

Snapshot metrics use a separate evaluation dataset/iterator and RNG, and an
immutable copied EMA inference bundle. A declared snapshot metric is scheduled by
default: an omitted `trigger` resolves to `trigger = "interval"` with
`every_steps = 10000` and `on_busy = "skip"`, and an explicit `trigger =
"interval"` without `every_steps` takes the same 10,000-step default. Choose
`trigger = "manual"` to opt out and evaluate that metric only on request. The
resolved defaults are recorded in the resolved configuration, the metric catalog
and the run manifest, so a run's cadence is always explicit in its provenance.

```toml
[metrics.custom.color_mean]
factory = "hypergan.metric_examples:ColorMomentDistance"
mode = "snapshot"
trigger = "manual" # explicit opt-out; omitting this schedules it every 10,000 steps
inputs = { generated = "evaluation.generated", reference = "evaluation.reference" }
timeout = 120
on_error = "fail"
[metrics.custom.color_mean.args]
statistic = "mean" # "spread" measures population standard deviation
color_space = "rgb"
low = -1.0
high = 1.0
[metrics.custom.color_mean.evaluation]
device = "cuda" # default; CPU correctness fixtures opt in explicitly
sample_count = 256
batch_size = 16
seed = 123
[metrics.custom.color_mean.evaluation.data]
factory = "my_project.data:EvaluationData"
args = { split = "validation" }
```

Call the standalone evaluation API with the run and metric ID:

```python
from hypergan.metric_evaluation import evaluate
receipt = evaluate("runs/experiment", "color_mean", config_path="config.toml")
```

The integrated CLI exposes the same operation as `hypergan evaluate RUN --metric
color_mean --config config.toml`. `--bundle` selects an older `model.pt` under
this run's attempts directory. The selected numerical recipe must match the run;
observation settings may differ. This standalone command holds the run lock and
requires a terminal run. It can also explicitly evaluate an interval-configured
metric. GPU remains the default. The source inference bundle must have saved
run/attempt/step provenance and a matching SHA256; no old-format migration is provided.

To restore automatic evaluation after that opt-out, or to choose a different
cadence, write the scheduling fields out and explicitly select the evaluation
device; keep the factory, inputs, arguments and evaluation protocol:

```toml
# Fields in [metrics.custom.color_mean]
trigger = "interval"
every_steps = 10000
on_busy = "skip"
# Field in [metrics.custom.color_mean.evaluation]
device = "cuda:1"
```

Interval evaluation has no device fallback. A snapshot metric that resolves to
`trigger = "interval"` and names no `evaluation.device` is rejected during
configuration resolution, before training starts; the error names the metric and
both remedies (an explicit device, or `trigger = "manual"`). A manual metric may
still omit the device and resolve to the `cuda` default, because nothing runs it
until an explicit `hypergan evaluate` on a stopped run.

When every enabled snapshot metric is manual, the run's `evaluation_schedule` is
empty and no evaluation ever fires. `hypergan train`, `resume`, `validate` and
`preflight` print a warning naming those metrics so the absent schedule is not
silent, and the viewer shows each of them as `manual` with no next step.

An existing run resumes under its own recorded settings: the manifest stores the
resolved configuration, including the resolved trigger, so a run created with
`trigger = "manual"` keeps that schedule and is unaffected by the default. Adding
a schedule to an existing run is an observation change, not a numerical one:
`hypergan train CONFIG --run-dir RUN` on an existing directory still requires the
whole configuration to match and refuses with a message naming `metrics` as the
differing section, while `hypergan resume RUN --config CONFIG` accepts the new
schedule and continues the same run.

The scheduler evaluates at completed global training steps divisible by
`every_steps`, independently of checkpoint and scalar-metric cadence. Resume
continues that global step cadence: resuming at step 41,000 with an interval of
10,000 schedules step 50,000 next. It does not backfill missed intervals.
Recovery from an earlier snapshot may produce another evaluation at a replayed
step; source attempt identity distinguishes those measurements.

An accepted evaluation uses an immutable inference snapshot taken at its source
step and runs in a separate worker while training continues. Its result is
plotted at that source step even if several more updates have completed. Capturing
the snapshot still has a copy/I/O cost. An evaluation using the training GPU
shares its memory and compute: training steps slow down and peak memory rises
while that evaluation runs, which is most visible for large sample counts.
Resolution warns when an interval metric's device may be the training device;
choose a separate available GPU to avoid that
contention. Device indices refer to the process's visible CUDA devices, including
any `CUDA_VISIBLE_DEVICES` mapping. CPU evaluation must be selected explicitly
for small correctness fixtures.

`on_busy = "skip"` is the only supported busy policy and the default. A due
interval is recorded as skipped when the evaluator is busy; it is not queued
for catch-up. One evaluation worker runs at a time across the run. When several
metrics are due together, selection rotates between them and unselected intervals
are recorded as busy skips. Factory failures and timeouts are visible:
`on_error = "disable"` disables that metric for the remainder of the attempt, while `on_error = "fail"`
fails training when the asynchronous failure is collected. A failure may arrive
after subsequent updates; recovery starts from a complete saved checkpoint.
Normal completion or a step/time budget stop waits for accepted evaluation work
within its bounded deadline. A signal or exceptional shutdown cancels outstanding
snapshot evaluations and reaps their workers; cancellation itself does not fail
training. Cancelled work has an explicit status and no fabricated metric value.
A new attempt starts a fresh scheduler and never resumes a partial evaluation.

The viewer shows configured snapshot metrics before the first result, including
manual scheduling or the next interval step, and exposes evaluation status.

Snapshot `evaluate(*, batches, context)` receives a bounded iterator of dictionaries
with the explicitly bound generated/reference tensors. It must consume the full
declared sample count and return a finite scalar or, with `kind = "histogram"`,
`{"edges": [...], "counts": [...]}` with ordered finite edges and at most 512
nonnegative bins. Data factories follow the normal batched tensor API; custom
evaluation data also provides `resume_identity()` identifying its actual dataset,
split and preprocessing. There is no implicit training-data fallback.

`ColorMomentDistance` compares pixel-weighted RGB mean or population spread.
`ColorHistogramDifference` produces absolute differences of fixed-bin pooled RGB
probabilities. Both require declared RGB ranges and finite NCHW RGB tensors.
These are color-statistic diagnostics, not semantic or spatial-quality claims.
Other modalities can use the generic snapshot protocol without the RGB examples.

Results live in `metrics/evaluations/<id>/`, with a receipt, one immutable JSONL
result and an atomic `stream.json` registration. They retain evaluated snapshot,
source attempt/step, EMA choice, sample count, seed, data identity, factory/runtime
sources and protocol hash. Late results do not inherit the trainer's current
step. Scalar values use `metrics`; histograms use `distributions`. Failures have
explicit statuses; unknown source position is marked `source_position_known =
false` and has no plotted value. An interrupted evaluator restarts only as a new
evaluation ID; the next explicit evaluation reconciles abandoned receipts and
releases temporary snapshot copies. It also recovers a completed result whose
registration was interrupted, without recomputing that result.

Partial evaluation-job resume remains unsupported. No dataset or weight downloads
occur implicitly. The Inception FID adapter requires pinned local weights and an
explicit preprocessing/sample protocol; see the [CIFAR recipe](cifar-recipe.md).

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
IDs. Custom scalar factories and explicit manual snapshot evaluation are described
below. Unsupported scheduling modes fail validation; a built-in FID adapter is
not yet provided.

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

A selected custom call starts a fresh process and has visible startup cost;
choose a cadence suited to that cost. Built-in scalars require no worker. The
per-metric timeout includes worker startup, evaluation and shutdown, with the
broker's cleanup grace additional. There is one synchronous call at a time and
no unbounded queue or implicit retries. `on_error = "disable"` records a visible
reason and disables that metric for the remainder of the attempt; a new attempt
preflights it again. `on_error = "fail"` fails the attempt. Because transforms
run after a complete numerical update, a required failure may leave a completed
but unpublished update beyond the last durable checkpoint; resume replays from
the selected complete checkpoint. Disabling optional publication never disables
mandatory numerical validation.

Factories are trusted Python. Input/output bytes and deadlines are bounded;
arbitrary allocations, external state and subprocesses created by plugin code
are not sandboxed. Instances and global RNG state are isolated from training,
but persistent evaluator state across calls is not supported.

Snapshot metrics use a separate evaluation dataset/iterator and RNG, and an
immutable copied EMA inference bundle. This first implementation is explicit
and standalone; automatic snapshot schedules are rejected.

```toml
[metrics.custom.color_mean]
factory = "hypergan.metric_examples:ColorMomentDistance"
mode = "snapshot"
trigger = "manual"
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
observation settings may differ. Evaluation holds the run lock and requires a
terminal run, so it cannot compete with this run's trainer. GPU remains the
default. The source inference bundle must have saved run/attempt/step provenance
and a matching SHA256; no old-format migration is provided.

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

Automatic snapshot scheduling, asynchronous evaluation, partial-job resume and a
built-in FID adapter remain unsupported. No dataset or weight downloads occur
implicitly. A future FID adapter still needs a pinned implementation, explicit
local weights and a complete preprocessing/sample protocol.

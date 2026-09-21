# Recipe configuration

A project is a directory containing `config.toml`, or a TOML file passed directly to `hypergan validate` and `hypergan train`. Run `hypergan new demo` to obtain the complete default configuration. Defaults are resolved into the run manifest so experiments do not depend on hidden CLI settings.

New projects use native CUDA execution by default (`hypergan new demo --device cuda:1` selects another visible GPU). Use `--device cpu` for a CPU numerical fixture. Supported `training.device` values are `cpu`, `cuda` and `cuda:N`; structural validation does not probe hardware. The historical programmatic `DEFAULT`/empty recipe remains the explicit CPU reference fixture, so hand-written recipes should declare their intended device. The runtime is a bounded numerical reference. It has a configurable component/data interface, not a qualified folder-of-images recipe or a distributed strategy.

## Components and I/O

`components.generator` and `components.discriminator` are required. Other named components can represent an encoder, feature extractor or auxiliary transform. Each component has a `factory`, constructor `args`, explicit forward keyword `inputs`, and a `trainable` flag.

Network components use `factory = "hndl"`. The source, input shapes, and output shapes belong in the recipe. HNDL constructs the network when training starts; editing the configuration needs no package rebuild. Data sources, objectives, and particle-routing adapters retain their explicit `module:object` factories. Unknown constructor arguments fail instead of being silently discarded.

```toml
[components.generator]
factory = "hndl"
inputs = { z = "latent" }
[components.generator.args]
input_shape = { z = ["B", 128] }
output_shape = ["B", 3, 128, 128]
source = """
linear(8192)
reshape(512, 4, 4)
relu()
deconv(256, kernel_size=4, stride=2, padding=1)
relu()
deconv(128, kernel_size=4, stride=2, padding=1)
relu()
deconv(64, kernel_size=4, stride=2, padding=1)
relu()
deconv(32, kernel_size=4, stride=2, padding=1)
relu()
deconv(3, kernel_size=4, stride=2, padding=1)
tanh()
"""
```

Use `file = "generator.hndl"` instead of `source` to load a standalone architecture relative to the recipe. File contents are copied into the resolved configuration and participate in the numerical fingerprint. Checkpoints and inference artifacts retain that source; moving or editing the original file does not change a saved run. Training from an edited file starts a different recipe and cannot silently resume an older architecture.

For conditional graphs, declare named `input_shape` contracts and use HNDL `concat`, branches, and joins in the source. Named `output_shape` contracts expose several outputs to component bindings. Concatenation belongs in HNDL source. Set `input_dtype = "int64"` for embedding inputs, or use a table keyed by input port for mixed float and integer inputs.

The image and particle-routing adapters load their architectures from [`src/hypergan/networks`](../src/hypergan/networks). Their recipes expose each template through `[components.<name>.args.network_files]`; paths are relative to the recipe. `[components.<name>.args.networks]` accepts inline source overrides instead. The resolved configuration records all selected template text. `${name}` template parameters substitute Python literals only; applications provide any trusted reusable source fragments before HNDL parses the resulting declarative graph.

Legacy `mlp` and `linear` factory arguments remain readable through HNDL adapters. They no longer construct handwritten PyTorch networks. New examples and generated projects use explicit HNDL source.

HNDL 0.4.0 or newer is part of the `train` extra. Configuration loading and ordinary CLI validation stay Torch-free. HNDL parses and resolves the architecture at model construction; invalid operators and shape constraints fail before training updates. Custom factories execute trusted Python and must construct a `torch.nn.Module`.

Bindings refer to `latent`, `batch.<field>`, `components.<name>` and their nested outputs. Generator output is available as `generated`; discriminator candidate input is `candidate`. Auxiliary components execute when an input or objective requires them. Cyclic or missing bindings fail. Dictionary/list outputs can be selected through dotted keys/indices.

The runnable [paired synthetic example](../examples/paired-linear.toml) exercises a trainable encoder, conditional generator/discriminator and weighted reconstruction loss:

```sh
hypergan train examples/paired-linear.toml --run-dir runs/paired
hypergan sample runs/paired --count 8 --output paired-samples.json
```

CLI sampling reuses saved example conditions and records that provenance in the JSON output. The Python `hypergan.artifacts.sample(..., inputs={...})` interface accepts fresh batched conditions. This is an I/O fixture, not a colorization or super-resolution model.

For conditional generation, bind a generator keyword to `batch.condition` or an encoder output and bind the discriminator to the corresponding condition. The data factory supplies both the condition and real target. This can express paired tasks, but an image architecture, preprocessing and task evaluation still need their own implementation and qualification.

Trainable auxiliary components belong to the generator optimizer. The discriminator has its own optimizer; discriminator conditioning is detached. Frozen components stay in evaluation mode. Separate encoder optimizers or arbitrary alternating schedules are outside this initial runtime and must not be implied by an encoder's presence.

## Frozen pretrained discriminator

The [128px ResNet variant](../examples/dcgan-resnet-128.toml) keeps the DCGAN generator and uses a frozen ImageNet ResNet18 plus a trainable discriminator head. The [complete discriminator source](../examples/networks/resnet18-discriminator.hndl) includes RGB normalization and declares freezing on the pretrained node:

```python
pretrained(${weights_path}, provider="torchvision_resnet18",
           sha256=${weights_sha256}, layer="layer3",
           trainable=False, name="resnet")
conv(64, kernel_size=1, name="projection")
leaky_relu(0.2)
adaptive_avg_pool(4)
flatten()
linear(1, name="score")
```

`trainable=False` freezes the backbone parameters and keeps its BatchNorm layers in evaluation mode. Autograd still differentiates its output with respect to the input, so the discriminator trains the generator through the frozen features. The head's parameters remain trainable. Keep `components.discriminator.trainable` at its default `true`; freezing the entire component would also freeze the head.

The provider registration uses torchvision's external ResNet18 checkpoint architecture; no ResNet layers are authored in HyperGAN Python. It is registered lazily through HNDL's native provider API and requires the `cifar` extra (torchvision). The local weights path and its SHA256 are in `args.parameters`. HNDL verifies the checkpoint before loading it; this recipe downloads nothing.

Edit the head in `.hndl` to change its width or pooling, or select `layer2`, `layer3`, or `layer4` for features at different depths. HNDL infers the connecting channels. Use a new run directory for an architecture change. The variant preserves the generator, dataset, batch size, and seed of the DCGAN recipe to isolate the discriminator change.

The [128px multiscale variant](../examples/dcgan-resnet-multiscale-128.toml) adapts the CIFAR discriminator design to 128px while retaining the DCGAN generator and training settings. Its [standalone HNDL source](../examples/networks/resnet18-multiscale-discriminator-128.hndl) declares the complete discriminator: a trainable pixel branch with five residual downsampling blocks and SAGAN attention at 16px, plus three trainable feature heads on frozen ResNet18 `layer1`, `layer2`, and `layer3` outputs. Each feature head uses a 1×1 projection, GroupNorm, a 3×3 convolution, and a scalar readout. The final score is `(pixel + (feature1 + feature2 + feature3) / sqrt(3)) / sqrt(2)`.

The fixed context is a zero RGB image in the input range `[-1, 1]`. HNDL concatenates candidate and context along the batch axis, applies ImageNet normalization to both, and runs one shared frozen backbone with `layers=("layer1", "layer2", "layer3")`. Native `chunk(..., 2, dim=0)` restores the two batches; each head concatenates its candidate and context features along channels. The context therefore has the pretrained features of midgray RGB, not zero feature maps. Frozen BatchNorm keeps examples independent. This graph supports variable batch sizes and differentiates through the candidate features, including the second derivatives required by b-cap.

Unlike the historical CIFAR adapter's cached context, the standalone graph recomputes context features in its shared `2*B` pass. It uses native 128px backbone input and 32/16/8px feature maps. Set the dataset manifest path/hash and local checkpoint path/hash in the TOML before training, and choose a new run directory. This is a configurable architecture experiment; no image-quality result is claimed for the 128px variant.

### Frozen DINOv3 with four feature depths

The [DINOv3 128px recipe](../examples/sagan-adain-dinov3-multidepth-128.toml)
keeps the SAGAN/AdaIN generator, pixel discriminator branch, fixed zero-image
context, and batch-64 training settings. Its [discriminator HNDL file](../examples/networks/dinov3-multidepth-discriminator-128.hndl)
replaces the ResNet branch with frozen DINOv3 ViT-S/16 and four trainable heads.

```python
features = pretrained(normalized, ${weights_path}, provider="dinov3_vits16",
    sha256=${weights_sha256}, readout="multidepth", trainable=False, name="backbone")
feature1, rest = split(features, 384)
feature2, rest = split(rest, 384)
feature3, feature4 = split(rest, 384)
```

The trusted provider's `multidepth` readout calls
`get_intermediate_layers(n=(2,5,8,11), reshape=True, norm=True)` once and
concatenates its four outputs. Each is 384×8×8 at 128px: these are different
transformer depths, all at the same spatial resolution. The HNDL graph handles
normalization, the shared candidate/context batch, channel splits, context joins,
all trainable heads, and the final score
`(pixel + (feature1 + feature2 + feature3 + feature4) / 2) / sqrt(2)`.

Point the provider to a clean local upstream checkout, pinned by full Git SHA:

```toml
[components.discriminator.args.pretrained_providers.dinov3_vits16]
source_path = "/absolute/path/to/dinov3-source"
source_commit = "6876159a11b4df116f30f667f8c9888617df0751"
```

Set the local checkpoint path and SHA-256 in `args.parameters`, as in the example.
The source pin and provider settings participate in the recipe fingerprint.
Provider registration only loads the upstream checkpoint architecture and selects
its named readout; no discriminator topology is built in Python. No new HNDL
release is required beyond 0.4.0. The source checkout and weights must remain
available when reconstructing the discriminator, including checkpoint resume.

`trainable=False` holds the backbone in evaluation mode, which also disables
DINOv3's training-time random RoPE coordinate rescaling. Candidate gradients
remain enabled; the readout uses math SDPA to support the second derivatives
needed by gradient penalties. The backbone stays out of the discriminator
optimizer while the pixel branch and four feature heads learn. This is an
architecture experiment, with no FID or long-run quality claim.
See the [upstream DINOv3 implementation](https://github.com/facebookresearch/dinov3)
for the pretrained model and checkpoint access.

### SAGAN with split-latent AdaIN

The [SAGAN/AdaIN variant](../examples/sagan-adain-resnet-multiscale-128.toml)
keeps that fixed-context discriminator and the batch-64 training settings, and
replaces the generator with [one editable HNDL file](../examples/networks/sagan-adain-generator-128.hndl).
A single 128-dimensional prior sample splits into 64 content dimensions and 64
style dimensions. Content projects into a 512-channel 4×4 feature map; every
AdaIN layer has its own affine projection of the same style half. No additional
noise inputs, random noise nodes, or random draws occur inside the generator.

Five residual nearest-neighbor upsampling blocks produce 8/16/32/64/128px maps
with 512/256/128/64/32 channels. Linear and convolution weights use native
spectral normalization. Self-attention at 32px uses pooled 16px keys/values and
a learned residual gain initialized to zero. The gain learns first; attention
projection weights receive gradients once it opens. These choices adapt the
[SAGAN residual and attention design](https://github.com/brain-research/self-attention-gan)
to this unconditional recipe; this is not a reproduction of the paper's
class-conditioned ImageNet experiment.

Native HNDL `adaptive_norm(features, params)` implements AdaIN as
`(1 + delta_gamma) * instance_normalize(features) + beta`, where each style
projection supplies `[delta_gamma, beta]`. Normalization uses per-example,
per-channel spatial statistics and no running batch statistics. Spectral
normalization still updates its power-iteration buffers in training mode;
inference uses frozen buffers. All of this is specified in HNDL 0.4.0 without
custom operators. The style-based conditioning is inspired by
[StyleGAN](https://arxiv.org/abs/1812.04948); there is no additional noise injection
or mapping network here.

This comparison changes the generator architecture and its initialization,
while preserving the latent distribution and seed settings. It does not
isolate attention from AdaIN or promise better quality. Track FID where
configured alongside `diversity/ratio` and `diversity/pooled4_ratio`.

## Numerical recipe

The configuration separates `prior`, `adversarial`, `gradient_penalty`, `prior_regularizer`, `objectives`, `optimizer`, `training` and `sampling`. The generated default uses a particle prior, paired relativistic logistic loss, b-cap and VICReg. The exact resolved parameters are saved with each run.

The discriminator penalty uses ParticleGAN arm names: `b_cap` is the default; `a_r1r2` denotes its paired zero-centered R1/R2 alternative. Changing an arm produces a custom, unqualified configuration. Do not treat these names as interchangeable with other papers' formulations.

Additional objectives select a loss factory, input bindings and weight. Reconstruction objectives such as MSE or L1 can connect generated output and paired targets. Custom task losses can be imported through the same factory mechanism. The runtime's supported update ownership is explicit; it does not infer a new training algorithm from component names.

The native step runs one compiled program, `d-then-g-v1`: a critic step, then a generator step. The program records the adversarial terms, their sample bindings, the per-phase detach policy of each scored sample and each critic input, the generator terms, and the ordered parameter groups. The executor follows those records. It does not choose sample sources, detachment, routing, or optimizer membership by reading component names during the update.

A recipe declares four things. Components and bindings say what each forward reads. Weighted losses include weight 0, which still runs and scales the scalar by zero. Gradient routing records, for each phase, where each scored sample is read, whether that sample is detached before the critic forward, and whether its score is detached after the forward. It also records, for each critic input and phase, whether the resolved value is detached. A frozen module can still pass gradients to its inputs. A detached score or a detached input cannot. The update schedule names which groups step and in what order. `d-then-g-v1` is the schedule these recipes use. A later method can add another schedule when that method needs it.

Legacy recipes compile into this program. The legacy compiler is what reads the discriminator component and its input bindings, and writes today's policy. `[adversarial]` remains the implicit first term, and `[gradient_penalty]` is that term's penalty. `[prior_regularizer]` and each `[[objectives]]` entry become generator terms. Each adversarial term stores its loss and penalty callable at compile time; the program stores the prior-spread callable. The executor does not read `[adversarial]`, `[gradient_penalty]`, `[prior_regularizer]`, `[[objectives]]`, or `[[adversarial_terms]]` during the update. A recipe that omits `[[adversarial_terms]]` keeps today's term, files, and fingerprints. Replicated and accumulated execution keep their own loops until they run this same program, and they reject a recipe that adds extra adversarial terms.

The legacy compiler records this policy, and the executor runs those records:

- The real sample binding is `batch.real`. The fake sample binding is `generated`.
- On the critic step, the fake sample is detached before the forward. Both scores stay attached. Gradients enter the critic. The penalty scores those same samples with that same sample-detach policy. Its coefficient comes from `[gradient_penalty]`, including when `adversarial.weight` is 0. That weight does not scale the penalty.
- On the generator step, the fake sample and its score stay attached. The real sample is not detached before the forward; its score is detached after the forward, because relativistic losses need the value. `adversarial.weight` scales only the adversarial scalar. Critic parameters are frozen for this step.
- The critic input whose path is `candidate` stays attached on both steps. Every other critic input is detached on both steps, whether it comes from the batch or from a component. A detached input is resolved against a detached context, so a component-produced condition sees detached inputs.

These records describe the current recipe. They do not implement another training method.

`[[adversarial_terms]]` adds further terms, in list order, after that implicit first term. Each term names an existing non-reuse component, required `real` and `fake` binding paths, and an `id`. `loss_type` and `mode` are optional and inherit `[adversarial]` when omitted; the allowed values are the same as that table. `weight` defaults to 1 and may be 0. Weight 0 still runs that term's forward and scales only its adversarial scalar by zero. `penalty` defaults to false, so an extra term does not take the legacy gradient penalty. `penalty = true` builds a separate penalty from `[gradient_penalty]` using that term's `penalty_coeff` (the default is `[gradient_penalty].coeff`). Two terms may name one component and keep different penalty coefficients. A term may override `inputs`; otherwise it uses the component's inputs. Exactly one of those paths is `candidate`.

Extra terms use the legacy detach policy above. Conditioning inputs stay detached on both steps. A trainable component whose only use is that detached conditioning still fails validation. A component used as a real or fake sample is generator-reachable and is not reported as disconnected. The schedule stays `d-then-g-v1`. Replicated execution rejects these extra terms.

Objective forwards run on the generator step even at weight 0. Names in an objective's `detach` list contribute no gradient. `freeze_parameters` on a reused module blocks that module's parameter gradients for the forward and still passes gradients to its inputs.

Parameter groups are ordered sequences with duplicates removed by identity. The same parameter in two groups fails validation. The legacy compiler builds the critic group from the trainable parameters of `discriminator`, in parameter order, and the generator group from `generator_parameters()`, so optimizer checkpoints keep their parameter mapping. Extra terms append the trainable parameters of their modules after that discriminator sequence; a parameter already owned by an earlier term is kept at its first occurrence. Modules used by those extra terms are removed from the generator group. A legacy recipe has no extra terms, so both sequences stay the same. The executor then uses those sequences and does not look up the component name again.

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

Snapshot metrics normally evaluate `sampling.generated`, or the adversarial
generator output when that sampling override is absent. Set
`generated = "generated"` inside a metric's `evaluation` table to measure the
adversarial generator output independently of a conditional reconstruction shown
by the sampler. A component binding such as
`generated = "components.reconstruction"` selects another declared output.
Each metric records its resolved `generated_binding` in the evaluation protocol.
New inference bundles retain the selected component and its dependencies; an
older bundle that omitted those components cannot evaluate that override.

`hypergan.colorization_metrics:SampleDiversityRatio` measures generated/reference
sample spread after RGB average pooling (`args.pool_size = 32` by default).
Zero means identical generated outputs; one matches reference spread. This is a
collapse diagnostic, not a quality score: noise can also have substantial spread.
It needs at least two samples in each set and nonzero reference spread.

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

That launch warning is easy to miss among other configuration warnings, so the
same condition is also reported while the run is under way:

- `hypergan train` and `hypergan resume` print one extra `hint:` line after the
  warnings block, naming the exact edit and the command that applies it, for
  example `hint: remove 'trigger = "manual"' from [metrics.custom.fid50k_train]
  to evaluate it every 10000 steps, then apply it with `hypergan resume RUN
  --config CONFIG``.
- The periodic progress output carries a one-line `reminder:` on its first
  printed progress line and every tenth one after that. With `--progress-json`
  those same progress rows gain an `evaluation_reminder` string field; every
  existing field keeps its meaning, and rows without the reminder omit it.
- The viewer's **Snapshot evaluations** panel shows a persistent notice above
  the schedule cards naming the manual metrics and the same edit. Runs with any
  scheduled metric, or any recorded `evaluation_schedule`, never show it.

Neither the hint nor the reminder appears for a recipe with no snapshot metrics
or with any metric on an interval.

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

### Sample diversity and collapse monitoring

The `standard` metrics preset includes diversity observations at **preview cadence**
(`--preview-every`), using the existing CPU-rendered EMA samples. No additional
training forward pass or GPU evaluator is needed. Enable previews to collect these
metrics; `metrics.every_steps` still controls update scalars, not previews.

- `diversity/generated_rms` and `diversity/reference_rms`: RMS distance across
  distinct sample pairs, computed as `sqrt(2 * mean(unbiased sample variance))`.
  These work with real floating-point vectors, images and other batched tensors.
- `diversity/ratio`: generated spread divided by real spread. Zero means identical
  generated outputs, and one matches the reference spread.
- `diversity/pooled4_generated_rms`, `diversity/pooled4_reference_rms`, and
  `diversity/pooled4_ratio`: the same statistics after averaging NCHW images down
  to 4x4, reducing the influence of fine texture on the diversity comparison.

The viewer selects generated spread and both ratios by default. Previews record sample counts and the
selected `sampling.generated` binding. Real spread uses the first available real
samples from the completed local batch (rank zero in distributed runs), without
repeating rows to fill the display grid.
Fewer than two samples, incompatible shapes, unsupported pooling, or zero reference
spread produce explicit unavailable statuses where applicable, not fabricated
zeros or infinities. Disable an individual metric with `metrics.disable`, or all
of them with `metrics.preset = "none"`.

These are collapse diagnostics, not quality scores or counts of semantic modes.
Noise can have large variance; compare coarse and full-resolution spread together.
A preview is a small sample (at most 16 outputs), and conditioning variation can
hide a generator that ignores its latent input. Conditional applications should
also evaluate repeated identical conditions when latent diversity matters.

For larger scheduled or manual evaluations, use the general snapshot factory
`hypergan.diversity_metrics:SampleDiversity` with the existing snapshot protocol:

```toml
[metrics.custom.sample_diversity]
factory = "hypergan.diversity_metrics:SampleDiversity"
mode = "snapshot"
every_steps = 1000
inputs = { generated = "evaluation.generated", reference = "evaluation.reference" }
[metrics.custom.sample_diversity.args]
statistic = "ratio" # alternatively "generated_rms" or "reference_rms"
pool_size = 4       # omit for full-resolution images or non-image tensors
[metrics.custom.sample_diversity.evaluation]
device = "cpu"
sample_count = 256
batch_size = 16
seed = 123
[metrics.custom.sample_diversity.evaluation.data]
factory = "my_project.data:EvaluationData"
args = { split = "validation" }
```

This factory merges centered moments across the whole evaluation, including
between-batch differences; changing evaluation batch size does not average away
collapse or inflate spread. It requires no pretrained metric model. Choose the
evaluation device and sample count to fit the generator's inference cost.

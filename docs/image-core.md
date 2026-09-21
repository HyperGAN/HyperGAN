# Image recipe numerical controls

A component's `trainable = true` preserves the parameter mask and module modes
chosen by its factory. A pretrained feature backbone can remain frozen and in
evaluation mode while its head trains. `trainable = false` freezes the entire
component and sets it to evaluation mode. Frozen parameters still permit input
gradients, including the second backward needed by exact b-cap.

Native and replicated checkpoints retain these masks, nested module modes,
buffers and optimizer ownership. The factory must establish the same parameter
inventory when reconstructing a supported run.

## Native phase and initialization policies

The default numerical reference retains its shared real/latent draw for D and G.
`training.phase_draws = "independent"` draws a fresh batch and prior sample for
each phase; the D fake is generated without a generator autograd graph. The
returned completed batch is the G-phase batch. Controlled Python comparisons
may pass `generator_batch` and `generator_latent_draw` to `ReferenceTrainer.update`
alongside its D-phase `batch` and `latent_draw`. These keywords are rejected for
shared draws.

`optimizer.implementation = "torch_fused_adam"` selects PyTorch's fused Adam
implementation explicitly. The default remains `device_adam`. Both use the
configured learning rates and G/D/prior betas, and save complete optimizer state.

Factories construct graph modules on CPU before execution-device transfer, in
generator, discriminator, then sorted auxiliary-name order. JSON/TOML key order
does not change the initialization stream; aliases do not allocate modules.
`prior.initialization_device = "cpu"` independently constructs/calibrates the
prior there before transfer; its default is `"execution"`. An optional
`prior.initialization_seed` gives prior initialization its own generator.
`prior.fixed_sigma` overrides the calibrated MoG sigma explicitly; it is invalid
for other prior kinds. Saved prior state includes the calibrated reference
distance and the effective sigma.

`training.data_rng_device` is `"cpu"` by default, or `"execution"` for a data
factory that draws on the execution device. `data_seed_offset` and
`prior_seed_offset` default to 1 and 2, added to `training.seed`. The source image
recipe uses execution-device data with offsets 2 and 3 and CPU prior seed+1.
Factories must support the requested generator device; there is no implicit
fallback. The separate penalty RNG remains seed+3.

These policies are part of numerical config identity, so changing them requires
a new run. Native-only phase/RNG/fused/alias policies fail public replicated
profile validation and direct replicated initialization, including accumulated
execution. Existing replicated shared-draw behavior remains supported.

## Shared components and encoder-only reconstruction

An auxiliary call can reuse an existing non-discriminator factory component:

```toml
[components.reconstruction]
reuse = "generator"
inputs = { z = "components.encoder.latent" }
freeze_parameters = true
```

The alias has no second module or optimizer allocation. Its explicit inputs
replace the target's normal bindings for that call. `freeze_parameters` temporarily
disables parameter gradients for the forward, then restores the original mask;
it preserves gradients through the operation to its inputs. It does not change
module modes or suppress forward buffer updates. Use an appropriate stateless
or explicitly mode-controlled factory when sharing a module this way.

MoG bindings `prior.means` and `prior.sigma` expose the current prior to ordinary
components. Means are computed on demand. An encoder-only reconstruction factory
must detach the means while keeping its encoder output differentiable. Bind the
reused generator output to an ordinary reconstruction objective. The core tests
verify that the reconstruction term affects E, with no G/prior parameter gradient.

Inference bundles, periodic previews and evaluation snapshots preserve aliases
and prior bindings, serialize each real module once and omit objective-only
components. These controls do not qualify an image architecture by themselves.
Source/image CUDA parity and quality remain separate acceptance gates.

## Explicit backend policy

An empty `training.backend` preserves the caller's current runtime settings.
An explicit table applies process-wide numerical settings before construction,
runtime identity reporting and resume comparison. The CIFAR example ships the
source's accelerated policy (`deterministic_algorithms = false`,
`cudnn_benchmark = true`, TF32 on); the strict variant is:

```toml
[training.backend]
deterministic_algorithms = true
cudnn_deterministic = true
cudnn_benchmark = false
matmul_allow_tf32 = false
cudnn_allow_tf32 = false
cublas_workspace_config = ":4096:8"
```

These are strict settings: unsupported deterministic kernels fail, never warn
and continue nondeterministically. Workspace values are `:4096:8` or `:16:8` and
must agree with any already initialized CUDA process. Deterministic CUDA requires
a valid workspace policy or the corresponding environment setting before CUDA
initialization. Effective runtime flags and configured policy are both recorded.
Disposable snapshot evaluators apply the saved backend policy before device/model
construction and record the effective flags during generated batches, including
any explicit evaluator context. Changed flags between batches fail evaluation.
Fresh CPU sampling remains independent of CUDA initialization and preserves the
caller's existing backend settings.

The historical source enabled TF32 and cuDNN benchmarking, and the shipped
example matches it. Switching a recipe to the strict policy, or replacing
nondeterministic image operations, is an explicit numerical adaptation that
requires measured source comparisons and exact within-run recovery; it does not
retrospectively reproduce the historical trajectory. Resume does not depend on
the policy: it restores the complete state under either.

# Model & Recipe Configuration (`hypergan-model/1`)

A HyperGAN project is initialized with `hypergan new --project <dir>` (which creates `hypergan.toml`, `runs/`, and `configs/<template>.toml`), or as a standalone `.toml` model file created with `hypergan new <path>.toml`. Pass a `.toml` file or project directory to `hypergan validate` and `hypergan train`. Defaults are resolved into the run manifest so experiments do not depend on hidden CLI settings.

New projects target native CUDA execution by default (`--devices cuda:1` selects another visible GPU; `--devices cpu` or `hypergan train -d cpu` selects CPU execution for small correctness fixtures).

---

## 1. Top-Level Metadata & Inheritance

Every HyperGAN model file declares `format = "hypergan-model/1"` at the top level alongside optional experiment metadata, preset defaults, and parent configuration inheritance:

```toml
format = "hypergan-model/1"
name = "cifar-tiny-transformer"
description = "32x32 CIFAR-10 generator with a frozen ResNet-18 feature critic"
defaults = "particlegan"
extends = "./base-cifar.toml"
```

| Field | Type | Description |
| --- | --- | --- |
| `format` | `string` | Schema identifier. Always `"hypergan-model/1"`. |
| `name` | `string` | Experiment or model identifier used for run naming, slugs, and registry display (defaults to the file stem). |
| `description` | `string` | Optional human-readable summary of the model or experiment. |
| `extends` | `string` | Optional parent `.toml` path or packaged model name (e.g., `"gaussian-grid"` or `"./base.toml"`). Tables are deep-merged recursively; scalars and arrays (`[[losses]]`) are replaced by the child config. Relative paths inside each file remain anchored to the file that declared them. |
| `defaults` | `string` | Numerical formulation preset (`"particlegan"` by default). Supplies ParticleGAN's released defaults for loss, learning rates, hyperparameters, normalization, regularization, and optimizers. |
| `recipe` | `table` | Optional table of resolved ParticleGAN preset overrides saved alongside the run. |

---

## 2. Dataset & Input Pipeline (`[data]`)

The `[data]` section (or alias `[dataset]`) configures the training data source, image resolution, batching, and worker preprocessing. It supports built-in sources (`builtin`), item-level datasets (`dataset`), or batch-level callables (`factory`), and accepts dataset arguments either flat on `[data]` or nested under `[data.args]`:

```toml
[data]
builtin = "image_folder"          # "gaussian_grid" | "cifar10" | "image_folder" | "paired_linear"
root = "./data/images"
width = 64
height = 64
channels = 3
mode = "RGB"
resize = "center_crop"
batch_size = 64
shuffle = true
workers = 4
```

Inside a workspace, path fields may use the workspace's variables: `root = "${paths.data}/cats"` reads `[paths] data` from `hypergan.toml` (`${data}` is the same), and `${workspace.root}` / `${workspace.runs_root}` name the workspace directories. Only path fields are substituted (`root`, `manifest`, `cache_dir`, `hndl`, `file`, `network_files`, `weights_path`, `source_path`, `path`), only the `${...}` form is recognized, and an undefined variable is an error. Environment variables are not read. Other strings, such as HNDL sources with their own `${param}` template parameters or metric labels, are left as written.

### Supported Data Modes

1. **Built-in Datasets (`builtin`)**:
   - `"image_folder"`: Recursively scans an image directory (`root = "./data/images"`), validating extensions and applying `width`, `height`, `channels` / `mode` (`"RGB"` or `"L"`), and `resize` (`"center_crop"`, `"pad"`, `"stretch"`, or `"none"`). For aspect-fitting with letterboxing/pillarboxing, use `resize = "pad"` alongside optional `fill = 0` (or `[0..255]`).
   - `"cifar10"`: CIFAR-10 dataset loader (`root = "./data/cifar10"`, `split = "train"`, `sampling = "random" | "sequential"`, `horizontal_flip = true`).
   - `"gaussian_grid"`: 2D mixture-of-Gaussians synthetic reference grid (`side = 10`, `noise = 0.015`).
   - `"paired_linear"`: Synthetic paired-input fixture yielding `batch.condition` and `batch.real` (`dimensions = 2`, `scale = 2.0`, `offset = 0.25`).

2. **Custom Item-Level Datasets (`dataset`)**:
   - `"pkg.module:DatasetClass"`: Any importable Python map-style or iterable dataset class implementing `__len__` and `__getitem__(index)` (or `load(index)`), returning a dictionary of tensors or primitives (e.g., `{"real": tensor}` or `{"condition": cond, "real": target}`). Wrapped automatically with background multi-threaded worker prefetching and epoch shuffling via `shuffle = true` and `workers = 4`. (Note: string presets like `"image_folder"` and `"cifar10"` belong under `builtin`, not `dataset`).

3. **Batch-Level Factories (`factory`)**:
   - `"pkg.module:BatchCallable"`: Importable callable invoked as `factory(batch_size, *, generator)` returning a dictionary of batched tensors (e.g., `hypergan.colorization_data:ColorizationData` or `hypergan.image_data:CIFAR10Data`).

| Field | Type | Description |
| --- | --- | --- |
| `builtin` | `string` | Built-in dataset name (`"gaussian_grid"`, `"cifar10"`, `"image_folder"`, `"paired_linear"`). |
| `dataset` | `string` | Python import path to custom item dataset class (`"module:Class"`). |
| `factory` | `string` | Batch-level callable import path (`"module:callable"`). |
| `batch_size` | `int` | Global batch size per training step (can be set in `[data]` or `[train]`). |
| `width`, `height` | `int` | Target spatial width and height in pixels. |
| `channels` | `int` | Number of image channels (`3` for RGB, `1` for grayscale). |
| `shuffle` | `bool` | Whether to shuffle dataset items each epoch (`true` by default for training). |
| `workers` | `int` | Number of background data-loading worker threads/processes. |
| `args` | `table` | Optional subtable (`[data.args]`) for dataset-specific constructor arguments (`root`, `manifest`, `manifest_sha256`, `cache_dir`, `bad_image_policy`, etc.). Flat keys in `[data]` are forwarded automatically. |

---

## 3. Latent Prior (`[prior]`)

The `[prior]` section configures the latent distribution sampled by the generator and regularized during training:

```toml
[prior]
kind = "particles"                # or type = "particles" | "gaussian" | "mog"
z_dim = 128
num_particles = 16384
init_std = 1.0                    # for kind = "mog", use fixed_sigma = 0.2126 or sigma_rel = 0.025
learnable = true
initialization_seed = 2
initialization_device = "cpu"
```

| Field | Type | Description |
| --- | --- | --- |
| `kind` / `type` | `string` | Prior distribution family: `"particles"` (learned particle table), `"gaussian"` (fixed standard normal), or `"mog"` (learned mixture of Gaussians). |
| `z_dim` | `int` | Latent vector dimensionality. |
| `num_particles` | `int` | Number of particles or mixture components in `"particles"` / `"mog"` priors (e.g., `10000`, `16384`, `65536`). |
| `init_std` | `float` | Initial standard deviation for particle or Gaussian draws (`1.0` by default). |
| `sigma` / `fixed_sigma` | `float` | Isotropic Gaussian noise scale added to selected particle centers in `"mog"` priors. |
| `sigma_rel` | `float` | Relative nearest-neighbor calibration scale for `"mog"` priors when `sigma` / `fixed_sigma` is omitted (default `0.025`). |
| `standardize` | `bool` | Whether to standardize the learned `"mog"` particle table across rows. |
| `learnable` | `bool` | Whether particle centers are updated by the prior optimizer (`true` by default). |
| `initialization_seed` | `int` | Dedicated deterministic seed for initializing the particle table. |
| `initialization_device` | `string` | Device used for initial particle table draw (`"execution"` or `"cpu"`). |

---

## 4. Networks & HNDL Architectures (`[networks.<name>]`)

Networks are declared under `[networks.<name>]`. Every model declares one network with `role = "generator"`, at least one network with `role = "critic"`, and any number of optional `role = "encoder"` or `role = "auxiliary"` networks.

```toml
[networks.generator]
role = "generator"
inputs = { z = "latent" }
input_shape = ["B", 128]
output_shape = ["B", 3, 64, 64]
source = """
reshape(128, 1, 1)
deconv(512, kernel_size=4, stride=1, padding=0, bias=False)
batch_norm()
relu()
deconv(256, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(128, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(64, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(3, kernel_size=4, stride=2, padding=1)
tanh()
"""

[networks.discriminator]
role = "critic"
inputs = { x = "candidate" }
input_shape = ["B", 3, 64, 64]
output_shape = ["B", 1]
# Or load from a standalone file: hndl = "networks/discriminator-64.hndl"
source = """
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(256, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
"""
```

### Network Specification Fields

| Field | Type | Description |
| --- | --- | --- |
| `role` | `string` | Required role: `"generator"`, `"critic"`, `"encoder"`, or `"auxiliary"`. |
| `source` | `string` | Inline HNDL declarative network source. |
| `hndl` / `file` | `string` | Path to a standalone `.hndl` network file (resolved relative to the `.toml` file; contents are snapshotted into the run's resolved recipe for reproducibility). |
| `module` | `string` | Importable Python `torch.nn.Module` class or factory (`"pkg.mod:Class"`). |
| `inputs` | `table` | Mapping of forward keyword arguments to graph bindings (`"latent"`, `"candidate"`, `"generated"`, `"batch.real"`, `"batch.condition"`, `"prior.means"`, `"prior.sigma"`, or another network name `"encoder"` / `"encoder.<output>"`). |
| `input_shape` | `list` / `table` | Shape contract for HNDL networks, e.g., `["B", 128]` for a single input or `{ z = ["B", 128], cond = ["B", 16] }` for multiple inputs. |
| `output_shape` | `list` / `table` | Expected output shape contract, e.g., `["B", 3, 64, 64]` or `["B", 1]`. |
| `input_dtype` | `string` / `table` | Optional input tensor dtype (e.g., `"int64"` for token/embedding indices, or a per-port table). |
| `trainable` | `bool` | Whether the network's parameters are updated during training (`true` by default; `false` freezes them). |
| `reuse` | `string` | Name of another declared network whose weights should be shared for a second forward pass with different `inputs`. |
| `freeze_parameters` | `bool` | When set to `true` on a `reuse` network, blocks parameter gradients for that forward pass while still propagating gradients with respect to its inputs. |
| `args` | `table` | Additional constructor or HNDL template arguments (`parameters`, `pretrained_providers`, `networks`, `network_files`, etc.). |

Any other field is rejected. There are no named architecture presets: each network is written as HNDL (`source`, `hndl` / `file`) or a Python `module`. `hypergan new --template dcgan|resnet|tiny-transformer` writes HNDL sources sized for your images that you can edit.

### Frozen Pretrained Feature Critics (ResNet-18 & DINOv3)

HNDL networks can embed frozen pretrained backbones alongside trainable discriminator heads:

```toml
[networks.discriminator]
role = "critic"
inputs = { x = "candidate" }
input_shape = ["B", 3, 128, 128]
output_shape = ["B", 1]
hndl = "networks/dinov3-multidepth-discriminator-128.hndl"

[networks.discriminator.args.parameters]
weights_path = "./weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
weights_sha256 = "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d"

[networks.discriminator.args.pretrained_providers.dinov3_vits16]
source_path = "./vendor/dinov3"
source_commit = "6876159a11b4df116f30f667f8c9888617df0751"
```

Inside the `.hndl` file, `pretrained(..., trainable=False)` freezes the backbone weights and keeps BatchNorm/dropout layers in evaluation mode while allowing autograd first and second derivatives to flow back to the candidate image and generator:

```hndl
# ResNet-18 single-depth or multiscale feature readout
pretrained(${weights_path}, provider="torchvision_resnet18",
           sha256=${weights_sha256}, layer="layer3",
           trainable=False, name="resnet")
conv(64, kernel_size=1, name="projection")
leaky_relu(0.2)
adaptive_avg_pool(4)
flatten()
linear(1, name="score")
```

For DINOv3 ViT-S/16 (`provider="dinov3_vits16"`, `readout="multidepth"`), the provider extracts intermediate transformer blocks `(2, 5, 8, 11)` in a single pass and concatenates them so HNDL `split(...)` nodes can route each depth to its own trainable head. Keep `networks.discriminator.trainable = true` so the projection heads train while the `pretrained(..., trainable=False)` node remains frozen.

### SAGAN with Split-Latent AdaIN & TransGAN Generators

- **SAGAN / Split-Latent AdaIN (`examples/networks/sagan-adain-generator-128.hndl`)**: Splits a 128-D latent into 64 content dimensions and 64 style dimensions, applying native HNDL `adaptive_norm(features, params)` across five spectrally normalized residual upsampling stages with self-attention at 32px.
- **TransGAN (`examples/networks/transgan-generator-128.hndl`)**: Projects a latent vector into an 8×8 token grid and progresses through 16×16, 32×32, 64×64, and 128×128 scales using bicubic upsampling, pixel shuffle, learned 2D relative-position bias, `rms_norm(eps=1e-8, affine=False)`, and 16×16 windowed attention at 64px/128px, optionally paired with `diff_augment(x, transforms="translation,cutout,color", translation_ratio=0.2, cutout_probability=0.3)` in the critic.

---

## 5. Losses & Gradient Penalties (`[[losses]]` & `[penalty]`)

All training objectives—adversarial losses, prior spread regularization, paired reconstruction losses, and custom auxiliary objectives—are listed together under `[[losses]]`.

```toml
# Primary adversarial loss and critic gradient penalty
[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

# Learned prior particle-spread regularizer
[[losses]]
type = "prior"
weight = 1.0
rows = "sampled_unique"           # "sampled_unique" | "full"

# Reconstruction or custom objective loss
[[losses]]
type = "reconstruction"           # or type = "objective"
id = "recon_l1"
fn = "l1"                         # "l1" | "mse" | "module:callable"
input = "generated"
target = "batch.real"
weight = 0.5
detach = ["target"]

# Critic regularization schedule and anchor settings
[penalty]
kappa = 1.0
lazy_k = 8
anchor_weight = 1.0
anchor_decay = 0.999
```

### Loss Types (`[[losses]]`)

1. **`type = "adversarial"`**:
   - `critic`: Name of a `[networks.<name>]` entry with `role = "critic"`.
   - `weight`: Scalar multiplier on the adversarial loss (default `1.0`).
   - `penalty`: Critic gradient penalty coefficient (`>= 0`, default `1.0`; `0` disables the gradient penalty for this term).
   - `real`, `fake`: Sample bindings (defaults to `real = "batch.real"` and `fake = "generated"`). Additional adversarial terms can score custom bindings or override `inputs` for multi-critic training.
   - `id`: Unique identifier for additional adversarial terms (the primary adversarial loss publishes as `loss/d_adversarial` and `loss/g_adversarial`).

2. **`type = "prior"`**:
   - Regularizes the learned particle or MoG prior table to maintain target variance and avoid collapse.
   - `weight`: Regularizer weight (default `1.0`; set `0.0` for non-learnable or Gaussian priors).
   - `rows`: `"sampled_unique"` (default, regularizes unique particles drawn in the global batch) or `"full"`.
   - `target_std`, `eps`: Target standard deviation (`1.0`) and numerical stability epsilon (`0.0001`).

3. **`type = "reconstruction"` / `type = "objective"`**:
   - `id`: Required unique identifier (`[a-zA-Z0-9_.-]+`); publishes scalar metric `loss/objectives/<id>`.
   - `fn`: Loss function identifier. Built-in functions are `"mse"` (Mean Squared Error) and `"l1"` (Mean Absolute Error). For perceptual, L2, or custom losses, supply an importable Python callable path: `fn = "pkg.mod:custom_loss_fn"` or `fn = "torch.nn.functional.smooth_l1_loss"`. Note: the key is `fn` (the field `kind` is not accepted).
   - `input` and `target` (or `inputs = { ... }`): Graph bindings passed to the loss function.
   - `weight`: Scalar loss weight (default `1.0`).
   - `detach`: List of input port names to detach before computing gradients (defaults to `["target"]` when `target` is bound).
   - `owner` / `network`: Optional network name verified to receive gradients from this objective.

### Gradient Penalty Settings (`[penalty]`)

While each `type = "adversarial"` loss sets its own penalty coefficient via `penalty = 1.0`, the top-level `[penalty]` table configures shared regularizer dynamics:
- `kappa`: Scale factor for ParticleGAN `b_cap` / zero-centered gradient penalty (`1.0` by default).
- `lazy_k`: Lazy regularization interval in steps (`1` evaluates every step; `8` evaluates every 8 steps and scales the penalty accordingly).
- `anchor_weight` (`1.0`) and `anchor_decay` (`0.999`): Anchor stabilization parameters.

### Update Schedule & Gradient Routing (`d-then-g-v1`)

HyperGAN compiles the declared networks and `[[losses]]` into the `d-then-g-v1` update program:
1. **Critic Step**: Evaluates real (`batch.real`) and detached fake (`generated`) samples through all critic networks, computes adversarial critic losses and lazy gradient penalties, and steps the critic optimizer.
2. **Generator Step**: Freezes critic parameters, evaluates attached `generated` samples (and any encoder/auxiliary paths), computes adversarial generator terms, `type = "prior"` regularization, and `type = "reconstruction"` / `"objective"` losses, and steps the generator/prior optimizers followed by the EMA update.

---

## 6. Training & Optimizer Schedule (`[train]`)

The `[train]` section unifies training schedule settings, optimizer hyperparameters, noise schedules, and backend determinism in one place:

```toml
[train]
steps = 200000
batch_size = 64
seed = 0
device = "cuda"
initialization = "deterministic_orthogonal"

# Optimizer & learning rate settings
lr = 0.0002
d_lr_mult = 1.5
prior_lr_mult = 10.0
betas = [0.0, 0.999]
prior_betas = [0.5, 0.999]
ema = 0.995
lr_anneal_start = 0.6
lr_floor = 0.05

# Optional execution & cadence controls
checkpoint_every = 1000
preview_every = 100
progress_every = 100

[train.backend]
deterministic_algorithms = true
cudnn_deterministic = true
cudnn_benchmark = false
matmul_allow_tf32 = false
cudnn_allow_tf32 = false
cublas_workspace_config = ":4096:8"
```

| Field | Type | Description |
| --- | --- | --- |
| `steps` | `int` | Total logical training updates to execute. |
| `batch_size` | `int` | Global batch size per update. |
| `seed` | `int` | Base deterministic RNG seed. |
| `device` | `string` | Target execution device (`"cuda"`, `"cuda:0"`, `"cuda:1"`, `"cpu"`). |
| `lr` | `float` | Base generator/encoder learning rate (e.g., `0.0002`). |
| `d_lr_mult`, `prior_lr_mult` | `float` | Multipliers relative to `lr` for the critic Adam group (`1.5`) and learnable prior table (`10.0`). |
| `betas` | `[float, float]` | Adam `[beta1, beta2]` coefficients for network parameters (default `[0.0, 0.999]`). |
| `prior_betas` | `[float, float]` | Adam `[beta1, beta2]` coefficients for the learnable prior table. |
| `implementation` | `string` | Optimizer implementation (`"recipe"`, `"device_adam"`, or `"torch_fused_adam"`). |
| `ema` | `float` | Exponential moving average decay for inference generator weights (`[0, 1)`, default `0.995`). |
| `initialization` | `string` | Weight initialization policy: `"deterministic_orthogonal"` or `"constructor"`. |
| `lr_anneal_start`, `lr_floor` | `float` | Fraction of `steps` where learning-rate cosine/linear annealing begins (`0.6`) and final floor multiplier (`0.05`; set `lr_floor = 1.0` for a constant learning rate that allows seamless `--steps` extension on resume). |
| `d_guard_ratio`, `d_guard_min_steps` | `float`, `int` | Critic spike guard threshold (`5.0`) and warmup step count (`200`). |
| `latent_damping_max_rate` | `float` | Maximum latent damping rate (`[0, 1]`). |
| `checkpoint_every`, `preview_every`, `progress_every` | `int` | Default cadences for durable checkpoints (`1000`), EMA preview snapshots (`0` / off unless set), and console progress lines (`100`). |

In `hypergan-model/1`, the `[train]` section unifies all training loop, optimizer, and cadence settings into a single canonical table. Optimizer settings (`lr`, `betas`, `d_lr_mult`, `prior_lr_mult`, `ema`, `implementation`) are defined directly under `[train]` (do not create a top-level `[optimizer]` table, which is rejected by the schema validator). When overriding parameters via CLI flags (`-s / --set`), you may use either `train.<key>` (canonical), `training.<key>`, or `optimizer.<key>` (e.g. `-s train.lr=0.0002` or `-s optimizer.lr=0.0002`).

Global batch size may be specified in either `[train] batch_size = 64` or `[data] batch_size = 64` (both resolve to `training.batch_size`).

Cadence defaults:
- `[metrics] every_steps = 10`: Writes scalar metrics (`events.jsonl`, `summary.json`) every 10 steps.
- `[train] progress_every = 100`: Prints console progress lines (with real-time `D=` and `G=` losses) every 100 steps.
- `[train] checkpoint_every = 1000`: Saves a durable training checkpoint every 1,000 steps.
- `[train] preview_every = 500`: Renders and archives visual sample grids every 500 steps (default `0` / off for non-image templates).

---

## 7. Evaluations & Scalar Metrics (`[evaluations]` & `[metrics]`)

### Snapshot Evaluations (`[evaluations]` and `[evaluations.<id>]`)

Snapshot evaluations run against an isolated copy of the EMA generator and a dedicated evaluation dataset, either periodically during training (when `every` / `every_steps` is set) or on demand via `hypergan eval <run>` / `hypergan evaluate <run> --metric <id>`. Shared evaluation defaults can be placed in `[evaluations]` and specialized per metric in `[evaluations.<id>]`:

```toml
[evaluations.fid50k_train]
factory = "hypergan.image_metrics:InceptionFID"
every = 10000                     # omit `every` for manual-only evaluation (`trigger = "manual"`)
device = "cuda:0"
samples = 50000
batch_size = 128
seed = 10000
on_busy = "skip"
timeout = 3600
on_error = "fail"

[evaluations.fid50k_train.args]
weights_path = "~/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
weights_sha256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"

[evaluations.fid50k_train.data]
factory = "hypergan.image_data:CIFAR10Data"
[evaluations.fid50k_train.data.args]
root = "./data/cifar10"
split = "train"
sampling = "sequential"
horizontal_flip = false
```

| Field | Type | Description |
| --- | --- | --- |
| `factory` / `fn` | `string` | Importable Python evaluator class (`factory = "pkg.mod:Class"`) or function (`fn = "pkg.mod:func"`). |
| `every` / `every_steps` | `int` | Interval in global training steps for automatic background evaluation. Omit to configure the metric as `manual` (run only via `hypergan eval`). |
| `samples` | `int` | Total number of generated and reference examples to evaluate (required). |
| `batch_size` | `int` | Batch size per evaluation forward pass (required). |
| `seed` | `int` | Deterministic RNG seed for evaluation latent draws and dataset iteration (required). |
| `device` | `string` | Execution device for the evaluation worker (`"cuda"`, `"cuda:1"`, `"cpu"`). |
| `data` | `table` | Required evaluation dataset specification (`builtin`, `dataset`, or `factory` + `args`). Evaluations never implicitly fall back to training data. |
| `output` / `target` | `string` | Generator output binding to evaluate (defaults to `sampling.output` or `"generated"`, or bind a specific network such as `"reconstruction"`). |
| `on_busy` | `string` | Behavior when a previous evaluation worker is still running (`"skip"` by default). |
| `timeout`, `on_error` | `int`, `string` | Worker deadline in seconds and failure policy (`"fail"` or `"disable"`). |

### Scalar Training Metrics (`[metrics]`)

Scalar training metrics publish completed update diagnostics to `events.jsonl` and the immutable metric catalog (`metrics/catalog-<sha256>.json`):

```toml
[metrics]
preset = "standard"               # "standard" | "none"
every_steps = 10                  # or every = 10
disable = ["loss/d_adversarial_raw", "loss/g_adversarial_raw"]

[metrics.overrides."loss/total"]
enabled = false

[metrics.loss_ratio]
factory = "hypergan.metric_examples:ScalarRatio"
inputs = { numerator = "g_loss", denominator = "d_loss" }
every_steps = 100
timeout = 10
on_error = "disable"
```

### Sample Diversity & Collapse Monitoring

The `"standard"` metrics preset automatically computes sample diversity diagnostics at preview cadence (`--preview-every`) on CPU-rendered EMA previews:
- `diversity/generated_rms`, `diversity/reference_rms`, and `diversity/ratio`: Full-resolution pairwise RMS spread of generated samples relative to real batch samples (`0.0` indicates mode collapse to a constant output; `1.0` matches reference spread).
- `diversity/pooled4_generated_rms`, `diversity/pooled4_reference_rms`, and `diversity/pooled4_ratio`: Pairwise RMS spread after 4×4 average pooling, measuring structural/color layout diversity independently of high-frequency texture.

For larger scheduled or manual diversity evaluations across hundreds of samples, declare `hypergan.diversity_metrics:SampleDiversity` (or `hypergan.colorization_metrics:SampleDiversityRatio`) under `[evaluations.<id>]`:

```toml
[evaluations.sample_diversity]
factory = "hypergan.diversity_metrics:SampleDiversity"
every = 1000
device = "cpu"
samples = 256
batch_size = 16
seed = 123
args = { statistic = "ratio", pool_size = 4 }

[evaluations.sample_diversity.data]
builtin = "image_folder"
args = { root = "./data/images", height = 64, width = 64, mode = "RGB", resize = "center_crop" }
```

---

## 8. Sampling & Checkpoints (`[sampling]`, `[samplers]`, `[checkpoints]`)

### Periodic Previews & Inference Sampling (`[sampling]` & `[samplers.<name>]`)

```toml
[sampling]
count = 16                        # number of samples in periodic preview grids
seed = 123                        # deterministic preview seed
output = "generator"              # primary generated output binding (default: generator output)
views = { real = "batch.real", random = "generated" }
comparison = [
  { label = "X", binding = "batch.real" },
  { label = "G(z)", binding = "generated" },
]

[samplers.custom_grid]
fn = "my_project.vis:render_grid"
inputs = { images = "generated" }
count = 16
seed = 42
```

### Checkpoint Retention & Bundle Export (`[checkpoints]`)

```toml
[checkpoints]
keep_last = 3
keep_every = 50000
keep_best = { metric = "fid50k_train", mode = "min", count = 2 }
export_bundle = "on_eval"         # "on_eval" | "on_checkpoint" | "never"
```

| Field | Type | Description |
| --- | --- | --- |
| `keep_last` | `int` | Number of most recent complete checkpoints to retain during pruning (default `3`). |
| `keep_every` | `int` | Milestone step interval whose checkpoints are permanently retained (default `50000`). |
| `keep_best` | `int` / `table` | Retain top-performing checkpoints by evaluation metric (`{ metric = "fid50k_train", mode = "min" | "max", count = 2 }`, or an integer `count` shorthand for `fid50k_train`). Set `false` or `0` to disable. |
| `export_bundle` | `string` | Automatic standalone EMA inference bundle (`model.pt`) export policy (`"on_eval"`, `"on_checkpoint"`, or `"never"`). |

---

## 9. Run Execution & Operational Settings (`[run]`)

The optional `[run]` section (supported both in model `.toml` files and in workspace `hypergan.toml` files) sets operational defaults for device selection, Web Observatory autostart, CPU threading, and cadences without altering the numerical recipe fingerprint:

```toml
[run]
devices = "0"                     # "0", "0,1", "gpu0,gpu1", "cuda:0", or "cpu"
viewer = true                     # launch background Web Observatory during CLI training
cpu_threads = 4
checkpoint_every = 1000
preview_every = 100
progress_every = 100

[run.env]
NCCL_DEBUG = "WARN"
```

---

## 10. End-to-End Configuration Examples

### Example A: Basic Unconditional Image GAN

```toml
format = "hypergan-model/1"
name = "dcgan-64"
defaults = "particlegan"

[data]
builtin = "image_folder"
[data.args]
root = "./data/images"
height = 64
width = 64
mode = "RGB"
resize = "center_crop"

[prior]
kind = "particles"
initialization_seed = 2
z_dim = 128
num_particles = 10000

[networks.generator]
role = "generator"
inputs = { z = "latent" }
input_shape = ["B", 128]
output_shape = ["B", 3, 64, 64]
source = """
reshape(128, 1, 1)
deconv(512, kernel_size=4, stride=1, padding=0, bias=False)
batch_norm()
relu()
deconv(256, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(128, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(64, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(3, kernel_size=4, stride=2, padding=1)
tanh()
"""

[networks.discriminator]
role = "critic"
inputs = { x = "candidate" }
input_shape = ["B", 3, 64, 64]
output_shape = ["B", 1]
source = """
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(256, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
"""

[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

[[losses]]
type = "prior"

[train]
steps = 200000
batch_size = 64
seed = 0
device = "cuda"
lr = 0.0002
initialization = "deterministic_orthogonal"

[checkpoints]
keep_last = 3
keep_every = 50000

[sampling]
count = 16
seed = 123
```

### Example B: Multi-Network HNDL Feature Critic (CIFAR-10 Tiny Transformer)

```toml
format = "hypergan-model/1"
name = "images/cifar-tiny-transformer"
defaults = "particlegan"

[data]
factory = "hypergan.image_data:CIFAR10Data"
[data.args]
root = "./data/cifar10"
split = "train"
sampling = "random"
horizontal_flip = true

[prior]
kind = "particles"
initialization_device = "cpu"
initialization_seed = 2
num_particles = 65536
z_dim = 128

[networks.generator]
role = "generator"
inputs = { z = "latent" }
input_shape = ["B", 128]
output_shape = ["B", 3, 32, 32]
hndl = "networks/tiny-transformer-generator-32.hndl"

[networks.discriminator]
role = "critic"
inputs = { x = "candidate" }
input_shape = ["B", 3, 32, 32]
output_shape = ["B", 1]
hndl = "networks/resnet18-features-discriminator-32.hndl"
[networks.discriminator.args.parameters]
weights_path = "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
weights_sha256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

[[losses]]
type = "prior"
rows = "sampled_unique"

[penalty]
kappa = 1.0
lazy_k = 8

[train]
steps = 200000
batch_size = 64
seed = 0
device = "cuda:0"
initialization = "deterministic_orthogonal"
lr = 0.0003
d_lr_mult = 1.5
prior_lr_mult = 10.0
betas = [0.0, 0.999]
prior_betas = [0.5, 0.999]
ema = 0.995

[evaluations.fid50k_train]
factory = "hypergan.image_metrics:InceptionFID"
every = 10000
device = "cuda:0"
samples = 50000
batch_size = 128
seed = 10000
timeout = 3600
on_error = "fail"
[evaluations.fid50k_train.args]
weights_path = "~/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
weights_sha256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
[evaluations.fid50k_train.data]
factory = "hypergan.image_data:CIFAR10Data"
[evaluations.fid50k_train.data.args]
root = "./data/cifar10"
split = "train"
sampling = "sequential"
horizontal_flip = false
```

### Example C: Paired Conditional Translation with Encoder & Reconstruction Loss

```toml
format = "hypergan-model/1"
name = "custom/paired-linear"

[data]
builtin = "paired_linear"
[data.args]
dimensions = 2
scale = 2.0
offset = 0.25

[prior]
kind = "particles"
num_particles = 64
z_dim = 4

[networks.encoder]
role = "encoder"
inputs = { input = "batch.condition" }
input_shape = ["B", 2]
output_shape = ["B", 2]
source = "linear()"

[networks.generator]
role = "generator"
inputs = { x = "latent", condition = "encoder" }
input_shape = { x = ["B", 4], condition = ["B", 2] }
output_shape = ["B", 2]
source = """
concat(x, condition)
linear(16)
leaky_relu(0.2)
linear(16)
leaky_relu(0.2)
linear()
"""

[networks.discriminator]
role = "critic"
inputs = { x = "candidate", condition = "batch.condition" }
input_shape = { x = ["B", 2], condition = ["B", 2] }
output_shape = ["B", 1]
source = """
concat(x, condition)
linear(16)
leaky_relu(0.2)
linear(16)
leaky_relu(0.2)
linear()
"""

[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

[[losses]]
type = "prior"

[[losses]]
type = "reconstruction"
id = "reconstruction"
fn = "mse"
input = "generated"
target = "batch.real"
weight = 1.0
detach = ["target"]

[train]
steps = 100
batch_size = 8
seed = 42
device = "cpu"

[sampling]
count = 8
seed = 123
```

### Example D: Configuration Inheritance (`extends`) & CLI Overrides (`-s / --set`)

Create a child configuration that inherits from a base model and overrides only the learning rate and batch size:

```toml
# configs/dcgan-fast.toml
format = "hypergan-model/1"
name = "dcgan-fast"
extends = "./dcgan-64.toml"

[train]
batch_size = 32
lr = 0.0001
steps = 50000
```

Override any field on the command line during validation or training without editing the TOML file:

```sh
# Override [train] and [[losses]] settings on the fly
hypergan validate configs/dcgan-fast.toml -s train.batch_size=16 -s losses.discriminator.penalty=2.0 --json
hypergan train configs/dcgan-fast.toml -n exp-lr1e4 -s train.lr=0.0001 -s losses.0.weight=1.0
```

- In CLI overrides (`-s / --set`), `[train]` fields can be addressed as `train.<field>` (canonical), `training.<field>`, or `optimizer.<field>`.
- `[[losses]]` entries can be addressed by list index (`losses.0.weight=0.5`), `id` (`losses.reconstruction.weight=0.5`), or `critic` name (`losses.discriminator.penalty=2.0`).

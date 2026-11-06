# HyperGAN Glossary

This glossary defines the terms used across HyperGAN documentation, CLI flags and TOML files. Each term is named the way you write it.

You write three kinds of TOML file:

- a **model file** (`format = "hypergan-model/1"`), which `hypergan new` generates and `hypergan train` reads;
- an optional workspace file, **`hypergan.toml`**, with project defaults;
- an optional **execution profile**, which says how a run is spread over devices and hosts.

HyperGAN lowers a model file into an internal resolved recipe before it trains. Runs record that recipe in `manifest.json`, and some older docs and tests use its names. The model-file tables below list the resolved name in their last column so you can map one to the other. Do not write the resolved tables (`[training]`, `[components.*]`, `[objective]`, `[optimizer]`, `[ema]`) in a model file; they are not part of the model-file format. [Configuration](docs/configuration.md) is the full reference for every field.

---

## 1. Model File (`format = "hypergan-model/1"`)

| Term | Model file | Default | Definition | Resolved recipe |
| :--- | :--- | :--- | :--- | :--- |
| **model file** | `format = "hypergan-model/1"` | *(required)* | The editable description of a model: data, prior, networks, losses and training schedule. | the whole recipe |
| **`name`** | `name` | file stem | Model identifier used in run names and the registry. | `name` |
| **`extends`** | `extends = "base.toml"` | *(none)* | Parent model file (or packaged model name) whose tables this file deep-merges over. Arrays such as `[[losses]]` are replaced, not merged. | merged before lowering |
| **`defaults`** | `defaults = "particlegan"` | `"particlegan"` | Numerical preset that supplies ParticleGAN's released loss, learning-rate and optimizer defaults. | `defaults` |
| **`data`** | `[data]` | *(needed to train)* | Training data source: one of `builtin` (`"gaussian_grid"`, `"image_folder"`, ...), `dataset` (item-level preset or `module:Class`) or `factory` (batch-level `module:callable`). | `data` |
| **`prior`** | `[prior]` | `kind = "particles"` | Latent source fed to the generator: `"particles"` (learned particle table), `"mog"` or `"gaussian"`, with `z_dim` and `num_particles`. | `prior` |
| **network** | `[networks.<name>]` | *(a generator and a critic)* | A named neural network. Declared in HNDL (`hndl = "file.hndl"` or `source = "..."`) or as Python (`module = "pkg.mod:Class"`). | `components.<name>` |
| **`role`** | `networks.<name>.role` | *(required per network)* | What the network does: `"generator"`, `"critic"`, `"encoder"` or `"auxiliary"`. | position in `components` |
| **`inputs`** | `networks.<name>.inputs` | *(required per network)* | Bindings from the network's input ports to values (see *binding*). | `components.<name>.inputs` |
| **`reuse`** | `networks.<name>.reuse` | *(none)* | Name of another network whose parameters this graph position shares. | `components.<name>.reuse` |
| **`freeze_parameters`** / **`trainable`** | `networks.<name>.freeze_parameters`, `.trainable` | trainable | Exclude a network's parameters (or a subset) from optimization. | same keys |
| **binding** | `"latent"`, `"batch.<field>"`, `"generated"`, `"candidate"`, `"prior.means"`, `"<network>"`, `"<network>.<output>"` | — | A named value a network input, loss or sampler reads. `candidate` is whatever a critic is currently scoring (real or generated). | `components.<name>` paths |
| **loss** | `[[losses]]` | *(an adversarial loss)* | Every training objective, listed together. `type` is `"adversarial"`, `"prior"`, `"reconstruction"` or `"objective"`. | see the next rows |
| **adversarial loss** | `[[losses]] type = "adversarial"`, `critic = "<network>"` | `weight = 1.0`, `penalty = 1.0` | The GAN loss between real and generated samples through one critic. `penalty` is the critic gradient-penalty coefficient; `0` disables it. The first adversarial loss is the primary one; later ones add terms. | `adversarial`, `gradient_penalty.coeff`; later ones `adversarial_terms` |
| **prior loss** | `[[losses]] type = "prior"` | `weight = 1.0` | Regularizer that keeps a learned particle table spread out. | `prior_regularizer` |
| **reconstruction / objective loss** | `[[losses]] type = "reconstruction"` or `"objective"`, `id`, `fn`, `input`, `target` | `fn = "mse"` | An extra loss on bound values, published as `loss/objectives/<id>`. | `objectives` |
| **`penalty`** | `[penalty]` | `lazy_k = 1`, `kappa = 1.0` | Shared critic-regularization schedule: lazy interval and scale. | `gradient_penalty` |
| **`train`** | `[train]` | *(optional)* | Training schedule and optimizer settings in one table: `steps`, `batch_size`, `seed`, `device`, `lr`, `d_lr_mult`, `betas`, `ema`, `lr_anneal_start`, `lr_floor`, and the `checkpoint_every`/`preview_every`/`progress_every` cadences. | split into `training`, `optimizer` and `run` |
| **`batch_size`** | `train.batch_size` | `16` | Global number of samples in one training step, across all ranks. | `training.batch_size` |
| **`steps`** | `train.steps` | `5` | Total number of global training steps in a run. | `training.steps` |
| **`device`** | `train.device` | `"cpu"`; `hypergan new` writes `"cuda"` unless `--device cpu` | Device for single-device runs (`"cuda"`, `"cuda:1"`, `"cpu"`). Replicated runs take devices from the execution profile instead. | `training.device` |
| **`seed`** | `train.seed` | `42` | Master seed; data, prior, penalty and noise streams derive from it. | `training.seed` |
| **`ema`** | `train.ema` | `0.995` | Decay of the exponential moving average of generator weights used for previews, sampling and evaluation. | `training.ema` |
| **`metrics`** | `[metrics]` | `preset = "standard"` | Scalar training metrics written to `events.jsonl`, plus custom `[metrics.<name>]` functions. | `metrics` |
| **evaluation** | `[evaluations.<id>]` | *(none)* | A snapshot metric such as FID, computed on an EMA snapshot with its own `[evaluations.<id>.data]`, either every N steps or on demand with `hypergan eval`. | `evaluations` |
| **`sampling`** | `[sampling]`, `[samplers.<name>]` | `count = 256`, `seed = 123` | Preview grids during training and the inputs used by `hypergan sample`. | `sampling`, `samplers` |
| **`checkpoints`** | `[checkpoints]` | `keep_last = 3` | Checkpoint retention (`keep_last`, `keep_every`, `keep_best`) and bundle export policy. | `checkpoints` |
| **`run`** | `[run]` | *(optional)* | Operational settings that do not change the numerical recipe: `devices`, `viewer`, `cpu_threads`, cadences, `[run.env]`. Also allowed in `hypergan.toml`. | `run` |

---

## 2. Execution Profile

An execution profile separates *how* a run is distributed from *what* is trained. Without one, `hypergan train` runs on a single device. Profiles are TOML files with `schema_version = 1`, passed with `--profile` (see [examples/execution/](examples/execution/) and [execution profiles](docs/execution-profiles.md)).

| Term | Profile key / value | Default | Definition |
| :--- | :--- | :--- | :--- |
| **`execution`** | `[execution]` | *(required in a profile)* | The backend, rank topology and batch partitioning. |
| **`name`** | `execution.name` | `"cpu-single"` | Profile identifier: `"cpu-single"`, `"cpu-replicated-gloo"` or `"cuda-replicated-nccl"`. |
| **`world_size`** | `execution.world_size` | `1` (single) / `2` (replicated) | Total number of worker ranks across all nodes (`1..64`). |
| **`rank`** | *(runtime identity)* | `0` | Zero-based global worker index (`0 .. world_size - 1`). |
| **`nodes`** | `execution.nodes` | `1` | Number of host machines (node brokers) in the run. |
| **`node_rank`** | `execution.node_rank` | `0` | Zero-based index (`0 .. nodes - 1`) of this host. |
| **`ranks_per_node`** | `execution.ranks_per_node` | `world_size // nodes` | Derived number of ranks started on each node. |
| **`rendezvous`** | `execution.rendezvous` | local `file://` URI | Process-group meeting point: `file:///...` on one host, `tcp://<host>:<port>` across hosts. |
| **`global_batch_size`** | `execution.global_batch_size` | `train.batch_size` | The model file's `train.batch_size`, recorded in the profile identity. |
| **`local_batch_size`** | `execution.local_batch_size` | `batch_size // world_size` | Derived samples per rank per global step. |
| **`accumulation_steps`** | `execution.accumulation_steps` | `1` | Number of microbatches each rank splits its local batch into before one optimizer step. |
| **`microbatch_size`** | `execution.microbatch_size` | `local_batch_size // accumulation_steps` | Derived samples in one forward/backward pass on one rank. |
| **`accumulation_algorithm`** | `execution.accumulation_algorithm` | `"retained-local-graph-v1"` | Resolved accumulation mechanism: `"retained-local-graph-v1"` (`accumulation_steps = 1`) or `"detached-logit-vjp-replay-v1"` (`accumulation_steps > 1`). |

---

## 3. Learning Aggregation (`[aggregation]` in an execution profile)

Aggregation defines how ranks combine what they learn. Without `[aggregation]`, replicated profiles average gradients every step (`strategy = "gradient_allreduce"`, `sync_every_steps = 1`).

```toml
[aggregation]
strategy = "gradient_allreduce"
sync_every_steps = 1
outer_lr = 1.0
outer_momentum = 0.0
```

| Term | Profile key / value | Default | Definition |
| :--- | :--- | :--- | :--- |
| **`aggregation`** | `[aggregation]` | *(optional)* | How distributed ranks combine learned updates. |
| **`strategy`** | `aggregation.strategy` | `"gradient_allreduce"` | `"gradient_allreduce"`, `"parameter_average"`, `"pseudo_gradient"`, or a registered custom strategy name. |
| **`gradient_allreduce`** | `strategy = "gradient_allreduce"` | *(default)* | Synchronous data parallelism: gradients are averaged across ranks after every backward pass, before each optimizer step. |
| **`parameter_average`** | `strategy = "parameter_average"` | — | Periodic parameter averaging (FedAvg / local SGD): ranks take `sync_every_steps` independent optimizer steps on their shards, then replace their parameters with the cross-rank mean. |
| **`pseudo_gradient`** | `strategy = "pseudo_gradient"` | — | Outer-optimizer aggregation (DiLoCo / FedOpt): ranks take `sync_every_steps` inner steps from a shared anchor, the mean weight delta ($\Delta\theta = \theta_{\text{anchor}} - \theta_{\text{local}}$) becomes a pseudo-gradient, and an outer update (`outer_lr`, `outer_momentum`) moves the anchor. |
| **`sync_every_steps`** | `aggregation.sync_every_steps` | `1` | Local steps between synchronization rounds (exactly `1` for `"gradient_allreduce"`, `>= 1` for the periodic strategies). |
| **`outer_lr`** | `aggregation.outer_lr` | `1.0` | Outer step size applied to the averaged pseudo-gradient. |
| **`outer_momentum`** | `aggregation.outer_momentum` | `0.0` | Heavy-ball outer momentum factor (`0.0 .. <1.0`): $v \leftarrow \mu v + \Delta\theta$, then $\theta_{\text{anchor}} \leftarrow \theta_{\text{anchor}} - \eta v$. There is no Nesterov variant. |

---

## 4. Runs, Checkpoints & Qualification

| Term | Where | Default | Definition |
| :--- | :--- | :--- | :--- |
| **run** | a run directory | — | One training job: its directory holds `manifest.json`, `events.jsonl`, checkpoints and previews. |
| **run id** / **slug** / **tag** | `hypergan ps` | — | Ways to name a run: a unique id (and its short form, e.g. `0608d849`), a generated `<adjective>-<noun>` slug, and an optional name you set with `hypergan tag`. |
| **`manifest`** | `manifest.json` | — | JSON record in a run or checkpoint directory: schema version, recipe fingerprint, execution profile, runtime identity and recovery lineage. |
| **checkpoint** | `checkpoints/<attempt>-step-NNNNNNNN-<nonce>/` (single device), `distributed-checkpoints/<attempt>-step-NNNNNNNN-<nonce>/` (replicated) | — | Hash-verified snapshot of parameters, optimizer state, RNG streams, aggregation state and run metadata at a completed step. |
| **preflight** | `[preflight]` in a profile; `hypergan validate --runtime` | *(optional)* | A structural check, or a bounded worker-construction check, run before training. |
| **`timeout`** | `preflight.timeout` | `60` | Seconds allowed for worker startup and preflight construction before forced cleanup. |
| **`collective_timeout`** | `preflight.collective_timeout` | `15` | Seconds allowed for one distributed collective before the process group is aborted. |
| **qualification** | `hypergan qualify` | — | A bounded smoke suite of single-device, replicated, aggregation and multi-node checks that writes a receipt. |
| **receipt** | `--output` (default `qualification-receipt.json`) | — | Machine-readable JSON result of `hypergan qualify`: environment, suite outcomes, timings and parity numbers, plus `receipt_sha256`. That hash is an unkeyed self-hash: it catches accidental edits, not deliberate ones, and the receipt is not signed. |

---

## 5. HNDL Architecture & Operators

Hyper Network Description Language (HNDL) is HyperGAN's declarative neural network DSL. HNDL graphs compile directly into high-performance `torch.nn.Module` execution graphs without boilerplate.

| Term | Context | Definition |
| :--- | :--- | :--- |
| **`diff_augment`** | Critic regularization | Differentiable augmentation layer applying stochastic spatial transforms (translation, cutout, color jitter) to both real and fake images during discriminator scoring. |
| **`rms_norm`** | Layer normalization | Root Mean Square Normalization across feature dimensions, invariant to scaling without mean centering ($y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$). |
| **`group_norm`** | Normalization layer | Channel grouping normalization dividing channels into $G$ groups, independent of batch size and GPU rank topologies. |
| **`swiglu`** | Activation function | Swish-Gated Linear Unit activation ($x \cdot \text{sigmoid}(\beta x) \otimes W_g x$), delivering enhanced expressivity in transformer and feedforward blocks. |
| **`pixel_shuffle`** | Upsampling operator | Efficient sub-pixel convolution upsampling rearranging depth elements into spatial blocks ($r \times r$). |
| **`rope_2d`** | Positional encoding | 2D Rotary Position Embedding applying sinusoidal frequency rotations across height and width spatial axes in vision transformers. |
| **AST / Shape Inference** | Graph compilation | Abstract Syntax Tree shape propagation tracking symbolic dimensions: `'B'` (batch size), `'H'` (height), `'W'` (width), `'C'` (channels), `'S'` (sequence length), and `'D'` (embedding/feature dimension). |

---

## 6. Run Directory Artifacts

Every training run produces an isolated run directory containing durable metadata, telemetry, and checkpoint state.

| Term | File / Artifact | Definition |
| :--- | :--- | :--- |
| **`manifest.json`** | Run root | Complete immutable record of the lowered recipe, dependency fingerprints, hardware environment, CLI invocation, and recovery lineage. |
| **`events.jsonl`** | Run root | Append-only newline-delimited JSON stream of scalar metric values, step timings, and health indicators recorded every `[metrics] every_steps`. |
| **`summary.json`** | Run root | Machine-readable JSON snapshot containing final step counts, total duration, stop reason, and latest metric measurements. |
| **`tag.json`** | Run root | User-assigned human-readable label or metadata dictionary written by `hypergan tag <run> <name>`. |

---

## 7. GAN Health Telemetry

HyperGAN automatically tracks GAN dynamics and numerical stability scalars during training.

| Metric | Source | Definition |
| :--- | :--- | :--- |
| **`critic/real_mean`** | Adversarial loss | Mean scalar score assigned by the discriminator to real batch samples. |
| **`critic/fake_mean`** | Adversarial loss | Mean scalar score assigned by the discriminator to generated samples. |
| **`critic/margin`** | Adversarial loss | Score separation $D(\text{real}) - D(\text{fake})$; indicates discriminator discriminative confidence. |
| **`critic/accuracy`** | Adversarial loss | Binary classification accuracy of the discriminator across real and fake batch distributions. |
| **`grad_norm/critic`** | Backpropagation | L2 gradient norm across all discriminator/critic trainable parameters. |
| **`grad_norm/generator`** | Backpropagation | L2 gradient norm across all generator trainable parameters. |
| **`grad_norm/prior`** | Backpropagation | L2 gradient norm across latent particle coordinates. |
| **`optimizer/d_guard_ratio`** | Optimizer | Dynamically throttled discriminator step frequency ratio under discriminator guard protection. |
| **`prior/particle_rms`** | Prior | Root-mean-square coordinate magnitude of the active ParticleGAN latent particle table. |
| **`diversity/ratio`** | Metric suite | Pairwise sample-to-latent distance ratio measuring generator diversity and alerting against mode collapse. |
| **`health_alerts`** | Telemetry monitor | Discrete integer counter tracking detected pathological states (exploding gradients, mode collapse, discriminator runaway). |

---

## 8. Multi-Node & Zero-TOML CLI Flags

Distributed runs can be launched across nodes directly from the CLI without authoring an execution profile TOML file.

| Flag | Argument | Definition |
| :--- | :--- | :--- |
| **`--nodes`** | Integer ($N \ge 1$) | Total number of physical host machines (nodes) participating in distributed training. |
| **`--node-rank`** | Integer ($0 \le r < N$) | Zero-based index identifying the current host within the multi-node cluster. |
| **`--rendezvous`** | URL string | Process group rendezvous URI (e.g. `tcp://10.0.0.1:29500` or `file:///shared/sync`). |
| **`--strategy`** | String identifier | Learning aggregation strategy: `"gradient_allreduce"`, `"parameter_average"`, or `"pseudo_gradient"`. |
| **`--sync-every`** | Integer ($k \ge 1$) | Step interval between inter-node parameter or gradient synchronization rounds. |

---

## 9. Model Export & Inference

Trained generators can be exported into standalone formats for zero-dependency deployment.

| Format / Class | Command / File | Definition |
| :--- | :--- | :--- |
| **`bundle`** | `hypergan export --format bundle` | Self-contained PyTorch `.pt` bundle embedding generator weights, baked latent particle buffers, and configuration with zero ParticleGAN or HNDL runtime dependency. |
| **`safetensors`** | `hypergan export --format safetensors` | Memory-mapped, zero-copy tensor archive for high-throughput Rust, WebAssembly, and Hugging Face inference. |
| **`pt2`** | `hypergan export --format pt2` | PyTorch 2.0 / TorchScript export format for embedded and C++ LibTorch production runtimes. |
| **`onnx`** | `hypergan export --format onnx` | Open Neural Network Exchange model format for deployment with ONNX Runtime, TensorRT, or CoreML. |
| **`LoadedGenerator`** | `hg.load_bundle(...)` | Subclass of `torch.nn.Module` wrapping an exported bundle, exposing `.sample(count, seed)` and `.interpolate(seed_a, seed_b, steps, mode)`. |

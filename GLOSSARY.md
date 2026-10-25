# HyperGAN Glossary

This glossary defines the canonical terminology used across HyperGAN documentation, code, CLI flags, and TOML configuration files.

**Design Rule:** Every concept has **one canonical term**, and that term is mirrored verbatim as the TOML table, key, or enum value in `config.toml` (recipe configuration) or `execution.toml` (execution profile). When options are omitted, HyperGAN defaults to single-device (`world_size = 1`, `nodes = 1`, `accumulation_steps = 1`) and standard synchronous training (`strategy = "gradient_allreduce"`, `sync_every_steps = 1`).

---

## 1. Recipe Configuration (`config.toml`)

| Canonical Term | TOML Key / Table | Default | Definition |
| :--- | :--- | :--- | :--- |
| **`recipe`** | `[recipe]` | *(none)* | Optional preset model and objective declaration (e.g., ParticleGAN BISPF / DualNorm presets) resolved into concrete components before training. |
| **`training`** | `[training]` | *(required)* | Core training schedule, batch sizing, device selection, and deterministic seeding table. |
| **`batch_size`** | `training.batch_size` | `8` | Global number of samples processed across all ranks in one complete training step. |
| **`steps`** | `training.steps` | `10` | Total number of complete global training steps to execute in a run. |
| **`device`** | `training.device` | `"cuda"` (CLI) / `"cpu"` | Target compute device family (`"cuda"` or `"cpu"`). In replicated profiles, each rank binds to its own visible device index. |
| **`seed`** | `training.seed` | `0` | Master deterministic random seed from which data, prior, penalty, and noise streams are derived. |
| **`prior`** | `[prior]` | `type = "gaussian"` | Latent input source or learnable particle table (`"gaussian"`, `"mog"`, etc.) sampled to feed the generator. |
| **`components`** | `[components.<name>]` | *(recipe-defined)* | Named neural network modules (such as `generator`, `discriminator`, or auxiliary encoders) and their graph input bindings (`inputs`, `reuse`). |
| **`reuse`** | `components.<name>.reuse` | *(none)* | Reference to another declared component whose parameters and buffers are shared across multiple graph positions. |
| **`objective`** | `[objective]` | *(recipe-defined)* | Primary adversarial loss formulation (`type = "particlegan"`, `"ral"`, `"wgan"`, etc.) and critic gradient penalty settings. |
| **`adversarial_terms`** | `[[adversarial_terms]]` | `[]` | Additional critic/generator adversarial loss pairs evaluated alongside the primary `objective`. |
| **`optimizer`** | `[optimizer]` | `type = "adam"` | Parameter update rule, learning rates (`lr`, `d_lr`, `g_lr`), and numerical implementation (`"device_adam"`, `"torch_fused_adam"`). |
| **`ema`** | `[ema]` | `enabled = true` | Exponential moving average shadow parameters maintained over generator weights for evaluation and sampling. |
| **`data`** | `[data]` | *(optional)* | Dataset source (`type = "image_folder"`, `"synthetic"`, etc.), resolution, and preprocessing contract. |
| **`metrics`** | `[metrics]` | *(optional)* | Periodic evaluation schedule (`every_steps`) and metric definitions (such as `fid` or custom metrics). |

---

## 2. Execution Profile & Topology (`execution.toml`)

An execution profile separates *how* a run is distributed and scheduled from *what* model and objective (`config.toml`) are being trained. By default, `hypergan train` runs on a single device without requiring an explicit `execution.toml`.

| Canonical Term | TOML Key / Value | Default | Definition |
| :--- | :--- | :--- | :--- |
| **`execution`** | `[execution]` | *(required in profile)* | Table defining the execution backend, rank topology, and batch partitioning. |
| **`name`** | `execution.name` | `"cpu-single"` | Execution profile identifier: `"cpu-single"`, `"cpu-replicated-gloo"`, or `"cuda-replicated-nccl"`. |
| **`world_size`** | `execution.world_size` | `1` (single) / `2` (replicated) | Total number of worker ranks participating in the run across all nodes (`1..64`). |
| **`rank`** | *(runtime identity)* | `0` | Zero-based global worker index (`0 .. world_size - 1`) within the distributed process group. |
| **`nodes`** | `execution.nodes` | `1` | Total number of independent host machines (or node brokers) participating in the run. |
| **`node_rank`** | `execution.node_rank` | `0` | Zero-based index (`0 .. nodes - 1`) of the current host node. |
| **`ranks_per_node`** | `execution.ranks_per_node` | `world_size // nodes` | Derived number of worker ranks spawned on each participating node. |
| **`rendezvous`** | `execution.rendezvous` | Local `file://` URI | Process-group coordination endpoint (`file:///...` for single-node local runs or `tcp://<host>:<port>` for multi-node runs). |
| **`global_batch_size`** | `execution.global_batch_size` | `training.batch_size` | Resolved alias of `training.batch_size` recorded in the execution profile identity. |
| **`local_batch_size`** | `execution.local_batch_size` | `batch_size // world_size` | Derived number of samples assigned to each rank per global step. |
| **`accumulation_steps`** | `execution.accumulation_steps` | `1` | Number of sequential microbatches into which each rank's `local_batch_size` is partitioned before stepping the optimizer. |
| **`microbatch_size`** | `execution.microbatch_size` | `local_batch_size // accumulation_steps` | Derived number of samples processed in a single forward/backward pass on one rank. |
| **`accumulation_algorithm`** | `execution.accumulation_algorithm` | `"retained-local-graph-v1"` | Resolved gradient accumulation mechanism: `"retained-local-graph-v1"` (`accumulation_steps = 1`) or `"detached-logit-vjp-replay-v1"` (`accumulation_steps > 1`). |

---

## 3. Learning Aggregation (`[aggregation]` in `execution.toml`)

Learning aggregation defines how parameter updates or gradients are combined across distributed ranks. When `[aggregation]` is omitted, replicated profiles default to standard synchronous per-step gradient averaging (`strategy = "gradient_allreduce"`, `sync_every_steps = 1`).

```toml
[aggregation]
strategy = "gradient_allreduce"
sync_every_steps = 1
outer_lr = 1.0
outer_momentum = 0.0
```

| Canonical Term | TOML Key / Value | Default | Definition |
| :--- | :--- | :--- | :--- |
| **`aggregation`** | `[aggregation]` | *(optional)* | Execution profile section configuring how distributed ranks combine learned updates. |
| **`strategy`** | `aggregation.strategy` | `"gradient_allreduce"` | Active learning aggregation algorithm (`"gradient_allreduce"`, `"parameter_average"`, `"pseudo_gradient"`, or a registered custom strategy name). |
| **`gradient_allreduce`** | `strategy = "gradient_allreduce"` | *(default strategy)* | Synchronous data-parallel training: after backward on every step, parameter gradients are averaged across all ranks before each optimizer step. |
| **`parameter_average`** | `strategy = "parameter_average"` | — | Periodic parameter averaging (FedAvg / Local SGD): ranks take `sync_every_steps` independent local optimizer steps on their local shards and then replace model parameters with the cross-rank parameter mean. |
| **`pseudo_gradient`** | `strategy = "pseudo_gradient"` | — | Federated outer-optimizer aggregation (DiLoCo / FedOpt): ranks take `sync_every_steps` local inner steps from a shared anchor state, average the resulting weight deltas ($\Delta\theta = \theta_{\text{anchor}} - \theta_{\text{local}}$) across ranks as a pseudo-gradient, and apply an outer update (`outer_lr`, `outer_momentum`) to the anchor weights. |
| **`sync_every_steps`** | `aggregation.sync_every_steps` | `1` | Number of local training steps executed between cross-rank synchronization rounds (`1` for `"gradient_allreduce"`, `>= 1` for periodic strategies). |
| **`outer_lr`** | `aggregation.outer_lr` | `1.0` | Outer step size applied to the averaged pseudo-gradient during `"pseudo_gradient"` synchronization rounds. |
| **`outer_momentum`** | `aggregation.outer_momentum` | `0.0` | Outer Nesterov/heavy-ball momentum factor (`0.0 .. <1.0`) maintained across `"pseudo_gradient"` synchronization rounds. |

---

## 4. Preflight, Lifecycle & Qualification

| Canonical Term | TOML Key / CLI / Artifact | Default | Definition |
| :--- | :--- | :--- | :--- |
| **`preflight`** | `[preflight]` / `hypergan preflight` | *(optional)* | Structural or bounded-worker construction check run prior to training. |
| **`timeout`** | `preflight.timeout` | `60` | Maximum wall-clock seconds allowed for worker startup and preflight construction before forced cleanup. |
| **`collective_timeout`** | `preflight.collective_timeout` | `15` | Maximum seconds allowed for any single distributed collective operation before aborting the process group. |
| **`checkpoint`** | `checkpoints/step-NNNNNN/` | — | Immutable, hash-verified snapshot of model parameters, optimizer states, RNG streams, aggregation state, and run metadata at a completed step boundary. |
| **`manifest`** | `manifest.json` | — | JSON descriptor inside a run or checkpoint directory recording schema version, config fingerprint, execution profile, runtime identity, and recovery lineage. |
| **`qualification`** | `hypergan qualify` / receipt JSON | — | Bounded, fast smoke-test verification suite that executes multi-device, multi-node, and aggregation parity checks and emits a machine-readable JSON verification receipt. |
| **`runpod`** | CLI / environment | `RUNPOD_API_KEY` (env) / `--dry-run` | Ephemeral GPU cloud provider integration used to provision and qualify remote hardware instances with hard cost guardrails and guaranteed cleanup. |
| **`receipt`** | `qualification-receipt.json` | — | Standardized machine-readable JSON verification artifact produced by `hypergan qualify` documenting environment metadata, suite outcomes, execution timings, and numerical parity metrics. |

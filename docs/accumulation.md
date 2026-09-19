# CPU gradient accumulation

`ReplicatedCPUTrainer(config, world_size=2, accumulation_steps=2)` divides each rank's effective local batch into equal microbatches. The factor must be a positive integer dividing `training.batch_size / world_size`. Every nonscalar input tensor must retain the local batch dimension; critic outputs must retain the microbatch dimension. Batch size still means the global number of samples in one complete D → G/prior/auxiliary → EMA update. Learning rates, lazy-penalty frequency and update counts retain their configured meaning.

This internal CPU/Gloo strategy is used by callers of the [worker API](cpu-workers.md); public `train` and `resume` remain single-process. GPU/NCCL, mixed precision and actual clusters require later qualification.

## Preserving the objective

Independent microbatch RA losses would use different means, and independent VICReg losses would use different covariances. HyperGAN instead collects detached logits for the full effective batch and evaluates the pinned ParticleGAN adversarial kernel once per logical D/G loss. RA uses globally gathered logits. Their derivatives provide cotangents for replaying one microbatch's network computation and immediately backpropagating it.

The prior draw remains a full local draw. Gradients from replay accumulate at the latent boundary and propagate through its original prior graph once, preserving standardized MoG's dependence on unsampled rows. VICReg evaluates the global unique sampled raw rows, or the configured full table, once per update. Both optimizers step once; EMA advances only after the complete successful update.

## Memory and replay limits

Network activations are retained for one microbatch at a time. The strategy still retains the full input batch, latent/prior graph, detached logits and cotangents; the table, optimizers and checkpoint copies remain replicated. Automatic data loading also still duplicates the global draw on every rank. This is an activation-memory bound, not a constant-memory or sharded-training claim. Repeated forward work trades throughput for lower activation memory.

Replay must reproduce the recorded forward computation. Torch, Python, NumPy and named RNG streams are restored around replay. Registered forward-state mutation is incompatible with this replay path; failures poison the trainer and cannot produce a complete checkpoint. Custom hidden external state and arbitrary batch-dependent models remain unqualified. Discovery consumes the logical RNG trajectory once; replay adds no extra consumption. Partitioning stochastic custom forwards can change that trajectory compared with accumulation one. Exact continuation is qualified within a fixed accumulation strategy.

Components must treat samples independently along the batch dimension for accumulation to preserve their full-batch behavior. Training-mode BatchNorm is already incompatible with the replicated global-batch strategy. Arbitrary custom components remain runnable with an unqualified warning; a declaration or successful smoke test cannot certify hidden batch coupling.

Built-in MSE/L1 objectives with `reduction="mean"` or `"sum"` have known aggregation semantics. Custom scalar objectives must declare `accumulation_reduction = "mean"` or `"sum"`: this promises an additive per-sample objective whose full-local-batch value is the corresponding weighted mean or sum of microbatch values. A covariance, contrastive population loss or other cross-sample objective does not satisfy that promise merely by returning a scalar. Missing or unsupported reduction semantics fail explicitly when accumulation is enabled; accumulation one retains the existing custom-objective behavior.

The [session report](../reports/core-accumulation-2026-09-19.md) records numerical, memory and recovery evidence.

## Recovery

The distributed checkpoint identity includes accumulation factor, microbatch size and algorithm, in addition to recipe/runtime/source/data/world size. A fresh worker group restores only a complete update boundary under the same identity. Changing the factor is a different execution profile and is rejected by strict resume. Failed or partially completed microbatches are not resumable state.

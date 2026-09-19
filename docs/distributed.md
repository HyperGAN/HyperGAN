# CPU distributed numerical groundwork

`hypergan.distributed_training.ReplicatedCPUTrainer` implements bounded, fixed-world-size CPU GAN updates over Gloo. It is an internal strategy with explicit gradient reduction, **not DDP**. The product `train`/`resume` CLI remains single-process CPU; no distributed CLI, GPU execution, cluster launching or distributed checkpoint/recovery is enabled here. The lower-level `GlooCollectives` primitives remain the numerical building blocks described below.

## Complete replicated CPU updates

The caller initializes the default Gloo process group with a finite timeout and invokes the trainer on every rank in the same order:

```python
from hypergan.distributed_training import ReplicatedCPUTrainer

trainer = ReplicatedCPUTrainer(config, world_size=2, accumulation_steps=1)
metrics, local_batch = trainer.update()
```

`config` uses the normal resolved recipe schema. `training.batch_size` is the **global** batch size and must be divisible by the world size. `update(batch=..., latent_draw=(z, ids))` accepts explicit **local** shards for controlled comparisons. No learning rate is scaled automatically. Accumulation other than one fails explicitly: independently averaging microbatch RA means or VICReg covariance would change the objective, and memory-saving accumulation remains a separate gate.

Each update computes D backward (including exact lazy b-cap), averages complete D parameter gradients, steps D, then computes and averages G/prior/auxiliary gradients, steps their shared optimizer, and advances EMA. Rp preserves local real/fake pairing. RA gathers differentiable logits and delegates all four loss kernels to the public ParticleGAN implementation, so its means span the global batch. Sampled-row VICReg uses the globally unique raw prior population once per update; standardized MoG still depends on the full replicated table. Custom scalar objectives are averaged across ranks: built-in elementwise mean MSE/L1 fits that contract, while custom global statistics require explicit collective logic and remain unqualified.

The gradient reducer runs **after backward**, without DDP hooks. A parameter unused on one rank contributes zero; if unused everywhere, its gradient stays `None` so Adam does not advance its state. Ranks agree on gradient participation and reject nonfinite/sparse gradients before each optimizer step; reduced gradients are checked again for overflow. Parameters, optimizer moments/groups, original learning rates, persistent and nonpersistent buffers, registered extra state, module modes and trainability must agree at complete boundaries. Initialization checks equality rather than silently replacing divergent constructors. Training-mode BatchNorm is rejected because rank-local statistics would not preserve the specified global-batch reference; frozen evaluation-mode normalization can be used. Other custom components remain runnable but unqualified, and observed replica-state divergence fails the update rather than being hidden by a rank-zero broadcast.

An error can occur after D has already stepped. `checkpoint_ready` is false throughout an update and becomes true only after G/prior/auxiliary/EMA and complete replica agreement. Failed trainers are poisoned and refuse further updates. This is a boundary contract for future coordinated recovery, not a rollback implementation or permission to save independent rank files as a successful job checkpoint.

### Global data ownership

Automatic data loading performs the same **global draw on every rank**, using the same named data RNG, checks the complete batch digest, then selects contiguous rank slices. For `image_folder`, all ranks advance the same permutation/cursor through one global epoch order; they do not run independent shuffled epochs. This deliberately duplicates CPU decoding and data I/O. It establishes correctness before an efficient rank-zero loader or a qualified sharded sampler is implemented.

A custom data callable must produce the same global draw under the supplied named generator; different outputs fail agreement. Explicit local batch fixtures bypass the data sampler. Prior, penalty and global stochastic-module RNGs use distinct deterministic rank seeds, while the data seed is shared. This does not promise that arbitrary stochastic custom modules follow a bit-identical single-process random trajectory.

### Complete-update evidence and limits

`tests/reference/test_distributed_training.py` runs three bounded subprocess tests. Its parity fixture exercises all 12 Rp/RA/vanilla × logistic/hinge/Wasserstein/LSGAN combinations for three complete updates against `ReferenceTrainer`, comparing parameters, prior, both Adam states, base learning rates, EMA and metrics. It also tests a nonlinear critic's active lazy b-cap with controlled interpolation, global data slicing, Gaussian prior handling, absent gradient contributions, rank-local nonfinite input refusal, poisoned-state reuse and unsupported accumulation/global-batch/config mismatches.

These float32 comparisons use explicit numerical tolerances, not bitwise equality. Reduction order changes floating-point cancellation. In particular, RA's additive critic-bias direction is mathematically null, and tiny residual gradients can be amplified by Adam's epsilon; the controlled RA fixture omits that bias. Arbitrary architectures, data and custom objectives are not promoted to a qualified profile by these tests. Multi-GPU/NCCL, real clusters, efficient accumulation, distributed checkpoint publication and whole-job recovery remain separate gates.

## Collective primitives

The caller initializes the **default Gloo process group**, with a finite operation timeout, and owns the worker lifecycle. Construct `GlooCollectives(world_size=2)` in both workers. Group membership must remain fixed for its lifetime. The wrapper provides:

| Method | Contract |
| --- | --- |
| `gather(local)` | Concatenate equal, nonempty CPU float32/float64 batch tensors in rank order, with autograd across ranks |
| `mean(local)` | Differentiable mean across the global batch, retaining dimension zero for broadcasting; equal local batch shapes |
| `unique_indices(ids, num_rows=N)` | Sorted union of int64 prior indices from every rank; supports unequal lists, duplicates and empty lists |

No function initializes workers or changes the training configuration. Subgroups, variable floating batch sizes, lower precision and GPU backends are outside this contract. `mean` currently gathers the whole input: it is a correctness reference, not an optimized communication implementation.

All workers must execute the same collective calls **and backward passes** in the same order. Before each forward tensor collective, small trusted-worker metadata is exchanged to reject differing operation names, shapes, dtypes, autograd participation or invalid indices on all ranks. This does not make arbitrary divergent control flow safe. A missing worker requires the process-group timeout and supervisor cleanup; the wrapper cannot cancel a remote process or recover a job. Worker membership and communication requirements follow [PyTorch distributed documentation](https://docs.pytorch.org/docs/2.14/distributed.html).

## Numerical meaning and scaling

The implementation uses the public `torch.distributed.nn.functional.all_gather`. Its backward sums every consuming rank's gradient contribution into the originating shard. PyTorch's Gloo path uses an autograd-aware all-to-all operation, allowing the tested second derivatives; ordinary `dist.all_gather` does not provide that same autograd contract. See the [PyTorch 2.14 implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/nn/functional.py). HyperGAN does not replace these derivatives with detached values or hand-coded approximations.

There is no universal world-size division:

- For a local mean adversarial loss on each rank, differentiable global means preserve the global relativistic-average objective. Averaging the resulting **replicated model parameter gradients** matches a single global-batch mean loss. Detached means or rank-local means change the objective.
- If every rank computes an identical full objective on gathered **independent input shards**, gathering sums contributions from all copies of that objective. The input-shard derivative test divides each copy by world size because no DDP parameter-gradient average is involved. That division is not a recommendation to divide every distributed training loss.
- For a replicated particle table, gather only sampled IDs, index the globally unique raw rows on every rank, and evaluate the same regularizer on each rank. Averaging identical parameter gradients preserves that penalty without another world-size factor. Averaging rank-local covariance penalties changes the selected population and objective.
- Standardized MoG centers depend on the entire raw table. Unsampled rows can receive gradients through global centering and scaling. The table remains replicated in this fixture; it is not sharded or reduced to sampled rows before normalization.

For example, inside an already coordinated worker:

```python
collectives = GlooCollectives(world_size=2)
mean_fake = collectives.mean(fake_logits)
rows = collectives.unique_indices(sampled_ids, num_rows=prior.z.shape[0])
prior_loss = regularizer(prior.z[rows])
```

An empty global ID union is returned as an empty tensor. Whether a particular regularizer accepts that population is the caller's responsibility; no fallback population is silently substituted.

## Evidence and next gate

`tests/reference/test_distributed.py` launches two real local Gloo processes with a 15-second normal process-group timeout (three seconds for deliberate missing-worker tests) and a 45-second parent deadline, terminating surviving workers on failure. The numerical fixtures compare against ordinary single-process global-batch autograd:

- float32 and float64 gathered nonlinear objectives, gradients and Hessian-vector products, including cross-rank coupling;
- ParticleGAN relativistic-average logistic loss, replicated parameter gradients and Hessian-vector products with global means;
- globally unique VICReg populations with repeated IDs, empty/unequal ID lists, standardized MoG gradients and Hessian-vector products, including dense unsampled-row gradients;
- the actual lazy b-cap input-gradient/parameter-backward computation, including an inactive step and the active lazy multiplier;
- collective input/operation errors, a rank exiting and a live rank failing to participate until timeout.

The primitive fixtures use explicit averaged parameter reductions. The trainer fixtures above additionally compare complete optimizer and EMA updates. Neither exercises DDP hooks or establishes coordinated checkpoints or recovered worker groups. Passing these tests does not qualify a future DDP implementation of b-cap.

Local validation uses Python 3.12.13, PyTorch 2.14.0+cpu and ParticleGAN 0.5.0. Run the bounded test with the numerical dependencies installed:

```sh
python -m pytest tests/reference/test_distributed.py -q
```

Next, qualify coordinated complete-state publication, rank-specific RNG/data restore, worker failure and whole-job restart, followed by efficient global-objective accumulation. Only then qualify actual two-GPU NCCL and real multi-node behavior. Internal CPU updates do not close the distributed-training issue or establish cluster support.

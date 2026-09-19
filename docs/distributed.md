# CPU distributed numerical groundwork

`hypergan.distributed.GlooCollectives` is an internal numerical building block. It does **not** enable distributed `train`, DDP, `torchrun`, GPU execution, cluster launching or distributed recovery. The existing training runtime remains single-process CPU. These tests are one prerequisite for a future fixed-world-size strategy.

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

Losses and parameter derivatives use explicit averaged reductions in these fixtures. No DDP reducer is exercised, and no synchronized optimizer, EMA, sharded data stream, coordinated checkpoint or recovered worker group is implemented. In particular, passing b-cap and collective double-backward fixtures separately does not qualify their interaction with DDP hooks in a complete GAN update.

Local validation uses Python 3.12.13, PyTorch 2.14.0+cpu and ParticleGAN 0.5.0. Run the bounded test with the numerical dependencies installed:

```sh
python -m pytest tests/reference/test_distributed.py -q
```

Next, integrate these semantics into a fixed-world-size CPU trainer and compare complete D/G/prior updates, optimizer/EMA state, accumulation and recoverable rank state against the single-process reference. Only then qualify actual two-GPU NCCL and real multi-node behavior. These primitives do not close the distributed-training issue or establish cluster support.

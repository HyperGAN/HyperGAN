# Multi-GPU readiness before a paid run

HyperGAN launches one supervised worker per GPU on a single host, using NCCL and a replicated float32 model, prior, optimizer and EMA. Select devices with `hypergan train MODEL --run-dir RUN --devices 0,1 --no-server`, or supply an execution profile. The parent launches the ranks; do not wrap this command in `torchrun`.

## Supported execution and limits

| Capability | Current scope |
| --- | --- |
| Complete D/G/prior/auxiliary/EMA updates | Real two-GPU NCCL numerical tests against a global-batch reference |
| Gradient accumulation | Replay with one microbatch's activations retained; global batch and learning rate remain unchanged |
| Checkpoint, stop/resume, earlier checkpoint selection | Complete per-rank state, optimizer, sampler and RNG recovery on a fixed topology |
| Failed rank or coordinator | Independent broker terminates and reaps the worker group; partial updates cannot become checkpoints |
| Aggregation | Synchronous `gradient_allreduce`; experimental `parameter_average` and `pseudo_gradient` also have two-GPU tests |
| Four/eight GPUs | Profiles and the qualification driver accept these counts; real four/eight-rank CPU training is tested, but GPU scaling needs that hardware |
| Model memory | Replicated on each GPU; adding GPUs does not pool model/optimizer memory. Accumulation reduces activation memory |
| Mixed precision / model sharding | Not provided by this replicated float32 path |
| Multiple hosts | Experimental; separate real-cluster qualification required |

The download-free image gate uses the installed CIFAR generator and pixel critic with synthetic 32×32 tensors. It exercises convolution, normalization, attention and critic double backward on two GPUs, and compares complete state after checkpoint recovery at accumulation factors one and two. This does not qualify CIFAR image quality or a pretrained feature backbone.

The global recipe batch must divide evenly by GPU count, and each local batch must divide by `accumulation_steps`. For a global batch of 64, four GPUs and accumulation two mean 16 samples per rank and eight per microbatch. HyperGAN does not scale learning rates automatically.

Training-mode BatchNorm is rejected. Accumulation greater than one also rejects configured input/output noise and independent D/G draws, which its replay implementation does not implement. Use accumulation one to retain those policies. Custom batch-coupled objectives and mutable forwards require their own numerical qualification.

## Run the local gates

Use an isolated environment containing the intended wheel and training dependencies. Check the import location first: another editable worktree must not supply the training implementation.

```sh
python -c 'import hypergan, torch; print(hypergan.__file__); print(torch.__version__, torch.cuda.device_count())'
hypergan qualify --suite cuda_replicated_nccl --cuda-world-size 2 --output /tmp/hypergan-2gpu.json --json
python -m pytest tests/cuda -q -m ""
```

The filtered qualification command exits unsuccessfully if the requested GPU count cannot run; a skipped suite is not a pass. `--cuda-world-size` changes the CUDA suite only. It verifies complete bounded updates and consensus on every requested rank. On a four/eight-GPU allocation, run the same command with that exact count before a longer job.

Qualification preserves inherited `CUDA_VISIBLE_DEVICES` order, including GPU UUID masks, and respects caller-supplied NCCL transport settings. It no longer forces P2P or InfiniBand off. On the local RTX A6000 host, default NCCL transport stalled in an initial object collective; the gate passes with this explicitly recorded host workaround:

```sh
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest tests/cuda -q -m ""
```

This host workaround is not a cloud default. Test the allocated machine's default transport before applying a workaround there. Preserve the receipt, test output and environment with the run.

For runtime construction checks against the actual model, use a profile file. This example selects two GPUs; set `world_size` to the actual allocation:

```toml
schema_version = 1
[execution]
name = "cuda-replicated-nccl"
world_size = 2
accumulation_steps = 1
```

```sh
hypergan preflight project/model.toml --profile execution.toml --runtime
hypergan train project/model.toml --run-dir runs/multigpu-smoke --profile execution.toml --stop-after-steps 4 --checkpoint-every 2 --no-previews --no-server
hypergan resume runs/multigpu-smoke --stop-after-steps 2 --no-server
```

Preflight constructs modules; the bounded training and resume commands exercise data, forward/backward, reductions and checkpoint I/O. Set a recipe target beyond six steps for this stop/resume check. Keep its initialization fixed. Compare finite losses, sample counts, parameters/optimizer continuity and throughput; an image inspection or a seed sweep does not establish distributed correctness.

## Modal boundary

[Modal supports requesting multiple GPUs for one function](https://modal.com/docs/guide/gpu). A single function containing HyperGAN's local supervisor matches the topology tested here. No Modal job is provisioned by these gates.

Persist the entire run directory and recipe/data artifacts. A successful local checkpoint write alone does not establish cloud durability: [Modal Volumes have explicit commit and reload semantics](https://modal.com/docs/guide/volumes). Validate persistence and an actual restart in the intended container/storage setup before a long run.

Recovery currently requires the recorded CUDA runtime, visible GPU inventory and ordered physical GPU UUID assignment. A replacement cloud container with different GPUs cannot transparently resume that checkpoint under this strict identity contract. See [replicated CUDA recovery](replicated-cuda.md). Portable recovery across replacement GPUs remains a cloud-readiness gap.

Correctness tests do not establish a throughput gain. Measure the intended model at a fixed global batch and fixed initialization on one and two GPUs before increasing the allocation. Current costs include per-parameter reductions, Python/object collective synchronization, full-state CPU hashing at replica checks/checkpoints, and serialized checkpoint gathering at rank zero. The current KA2 optimizer also reads one control scalar per trained critic tensor each step; the GPU profiler gate accounts for these separately from HyperGAN's fused gradient screening. Image/item loaders can decode local rows; custom global loaders still duplicate their global draw. Four/eight-GPU speedup must be measured on the requested hardware.

## Local evidence: 2026-10-26

Final core, public recovery and CUDA validation used the clean installed wheel from `73cc71c4`, outside the checkout, on two RTX A6000s. Runtime: Python 3.14.8, PyTorch 2.14.1+cu130, NCCL 2.30.7, ParticleGAN 0.9.3 and HNDL 0.10.0. The NCCL workaround above was explicit in the GPU environment. Larger-rank and reordered-mask checks also ran earlier in this change.

| Gate | Result |
| --- | --- |
| Default CPU/core selection | 2,644 passed; one version-specific migration skip; 261 heavy tests deselected |
| Public CPU stop/resume, repeated train and checkpoint replay | Three heavy acceptance tests passed |
| CUDA coverage | 62 passed across the main gate and focused follow-up runs; two external-data/artifact skips |
| Hardware qualification receipt | All six suites passed, including two-GPU NCCL and checkpoint compatibility |
| Reordered GPU UUID mask | Two-GPU qualification passed |
| Larger rank counts | Complete four/eight-rank CPU driver tests passed; no four/eight-GPU hardware claim |

The main CUDA run passed 51 cases before stopping at an outdated profiler expectation. Updating its exact accounting for KA2's per-critic control reads preserved the strict gradient-screening assertions; all ten profiler/nonfinite/EMA cases then passed. The added image-component recovery case passed separately. Together these runs cover every currently collected CUDA case. The two skips require the archived CIFAR qualification run and a local CIFAR dataset; neither was counted as passing.

Machine receipts, JUnit XML and logs remain locally under `reports/multigpu-readiness-2026-10-26/` (ignored by Git). These checks launched no Modal resources.

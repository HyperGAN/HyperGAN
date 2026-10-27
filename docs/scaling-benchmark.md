# Bounded training scaling benchmark

The [local harness](../scripts/benchmark_scaling.py) and [Modal runner](../scripts/modal_scaling.py) measure complete native or replicated float32 trainer updates after warmup. They use one initialization, fixed architecture and a fixed 256-particle prior. The default workload uses the installed CIFAR generator and a pixel critic on synthetic resident 32×32 tensors, exercising convolution, attention, normalization and the critic's double backward. No dataset or pretrained weights are downloaded, and no seed sweeps run.

## Local validation

Use an isolated environment containing the intended installed wheel and training dependencies. The driver never adds the checkout to the training import path. One GPU uses `ReferenceTrainer`; larger allocations use `ReplicatedTrainer` with synchronous gradient all-reduce. For the two available local GPUs:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -I scripts/benchmark_scaling.py --topologies 1,2 --output reports/local-scaling
```

On the RTX A6000 host described in [multi-GPU readiness](multigpu-readiness.md), also set the explicitly recorded `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` transport workaround. The Modal runner does not apply it. `--backend gloo --workload mlp` provides a small CPU integration fixture; it does not measure GPU scaling. The local driver refuses multiple physical nodes.

## Modal matrix and budget

The supported A10 allocations are `1`, `2`, `4` GPUs on one node, and `2x4` for two managed nodes with four GPUs each. [Managed clusters require whole GPU nodes](https://modal.com/docs/guide/multi-node-clusters); this runner uses private networking with RDMA disabled. Profiles run sequentially, without retries or warm containers. The eight-GPU profile fits the Starter plan's ten-GPU concurrency allowance.

Inspect the allocation and conservative estimate without importing Modal or provisioning anything:

```sh
python scripts/benchmark_scaling.py --plan-modal --topologies 1,2,4,2x4 --budget-usd 2
```

Authenticate Modal locally, build exactly one current wheel into `dist/`, then run:

```sh
python -m build --wheel --outdir dist
modal run scripts/modal_scaling.py --topologies 1,2,4,2x4 --budget-usd 2 --output reports/modal-scaling
```

The default **strong scaling** cohort keeps the global batch at 64 for every topology, with ten warmup updates and fifty measured updates. To select profiles or shorten the gate, use `--topologies`, `--warmup` and `--steps`. All configurations in a matrix must retain identical initial model/prior/EMA state; strong scaling also requires an identical resolved recipe fingerprint.

**Weak scaling** keeps a fixed local batch and increases the global batch with GPU count:

```sh
modal run scripts/modal_scaling.py --mode weak --batch-size 8 --output reports/modal-weak-scaling
```

Global batches must divide evenly across the world size and remain at most 256. These are benchmark cohorts, not changes to an existing training run's topology or learning-rate policy.

The budget guard uses [Modal's list prices](https://modal.com/pricing), verified on 2026-10-26: $0.000306 per A10-second, $0.0000131 per CPU-core-second and $0.00000222 per GiB-second. Each GPU requests one CPU core and 4 GiB RAM, with matching CPU and memory limits. The estimate conservatively charges the entire 180-second call wait for every requested resource and adds a $0.50 image-build allowance: **$1.386 for the default matrix**, below its $2 budget. It rejects larger estimates before submitting GPU calls. This is a resource-planning estimate, not a workspace billing limit or verified invoice; image builds occur before Modal's local entrypoint and their actual charge is not measured by the guard.

Each function has a 105-second execution timeout and a 45-second startup timeout. Local rank groups have a 90-second worker deadline and 25-second collective timeout. A failed or overdue call cancels submitted calls, terminates containers and writes a failure receipt alongside any complete earlier profiles. Existing output directories are refused to preserve evidence.

## Timing and interpretation

Warmup, imports, initialization, synthetic-input preparation, extra boundary replica digests and receipt serialization are outside the timed window. CUDA work is synchronized before the window and after every update. Distributed ranks enter a common boundary before timing; no extra per-step barriers or debug hashes are added. The trainer's normal gradient screening, metrics, collectives and replica-check cadence stay enabled and are recorded. With the default sixty total updates, the normal step-one digest falls in warmup; longer runs include any periodic digest they reach.

Throughput is global samples consumed divided by the **slowest rank's measured window**. Step latency uses the maximum duration across ranks for each corresponding step; the report gives its median and nearest-rank p95. Peak allocated memory is the maximum across ranks after resetting the peak counter at the measurement boundary. Replica state must match after warmup and at completion; all measured loss scalars must be finite. Slowdowns are valid measurements and do not make the correctness gate fail.

`report.json` retains per-rank timings, topology, resolved-recipe and state digests, GPU/runtime identity, memory, startup/warmup duration and the uploaded wheel checksum. `comparison.md` reports throughput, speedup, scaling efficiency and memory relative to the first selected topology. Modal JSON also includes estimated steady samples per dollar using requested GPU/CPU/RAM resources; startup, queue time, image builds and billing adjustments are excluded from that efficiency estimate. Partial results are preserved if a later profile fails.

The benchmark covers resident synthetic inputs and the internal trainer API. Data decoding/transfers, the public supervisor, previews, checkpoint I/O, restart, image quality, mixed precision and RDMA require separate tests. Results on a small 32px model do not establish scaling for larger recipes; compare throughput and cost on the intended workload before choosing a long-run allocation.

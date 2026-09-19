# Two-GPU NCCL readiness and distributed CUDA blockers

Date: 2026-09-19. Audit baseline: develop `9d93f6de018719c2606f797a488e7293d6db034a` (full identity verified in the external receipt). This is a bounded local hardware diagnostic and source audit. It does not port the replicated trainer or qualify full GAN training/recovery on CUDA. The user authorized the two local GPUs and a GPU-first default; no paid compute or remote allocation was used.

## Actual device and runtime evidence

Both GPUs are NVIDIA RTX A6000 with compute capability 8.6. `nvidia-smi` reports 49,140 MiB per device and driver 610.57.04; topology reports NV4 between them. PyTorch confirms bidirectional peer access. Device UUIDs are:

- GPU 0: `GPU-ed080e41-3193-3755-6756-f3d46c433331`.
- GPU 1: `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`.

The diagnostic ran with Python 3.12.13, Torch `2.14.0+cu130`, CUDA runtime 13.0 and NCCL 2.30.7. The native CUDA implementation agent supplied `/tmp/hypergan-cuda-native-verify/bin/python`: an isolated environment using read-only CUDA packages from `/home/martyn/dev/ParticleGAN/.venv` plus its own pinned ParticleGAN 0.5.0 installation. This diagnostic does not call ParticleGAN. The original CPU verification environment remains Torch `2.14.0+cpu` and was not modified.

GPU 0 already hosted unrelated Python PID 654701 using about 1,060 MiB; GPU 1 hosted desktop applications. They remained running and untouched. Native CUDA testing paused during this two-GPU diagnostic window and resumed after all diagnostic ranks had been reaped.

## Measured results, including unsuccessful probes

[The standalone tool](../tests/cuda/nccl_smoke.py) runs two spawned ranks, each selecting its own CUDA device before NCCL initialization. It checks all-reduce, broadcast and gather; first and second derivatives through differentiable all-gather; and explicit gradient averaging followed by a small linear Adam update. The averaged gradients and Adam state are compared against a full-batch baseline on each rank. This exercises useful primitives, not HyperGAN's complete D/G, prior, regularizer, accumulation or recovery algorithms.

| Probe | Observed result |
| --- | --- |
| Normal collectives and linear Adam | Passed on all three runs; 4.30 seconds in the final run |
| Global differentiable-gather derivatives | First derivative 3.0, second derivative 2.0 on both ranks, exact expected values |
| Rank-averaged gradient | `[8.5, 10.0]` on both ranks; matches full-batch reference |
| Linear Adam state and weight | Exact reference/rank equality; resulting weight `[0.49900001287460327, -0.25099998712539673]` |
| Torch peak allocation/reservation | 17,049,600 / 23,068,672 bytes per rank in the normal probe; excludes driver/context/NCCL allocations outside Torch's allocator |
| Missing peer, default NCCL diagnostics | Five-second collective timeout logged; process did not promptly exit. Outer 45-second supervisor terminated and reaped both ranks at 45.55 seconds. Prompt-abort expectation failed; safety cleanup passed |
| Missing peer, `TORCH_NCCL_DUMP_ON_TIMEOUT=0` | Same delayed-exit behavior; outer supervisor terminated/reaped both at 45.57 seconds. This setting alone did not resolve it |
| Missing peer, `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=1000` | NCCL timeout caused rank 0 exit `-6`; supervisor terminated peer with `-15`, reaping both at 18.86 seconds under a 20-second outer deadline |

All twelve owned PIDs across these probes were absent from `/proc` after completion, including zombies. A final `nvidia-smi` process query showed only the pre-existing unrelated applications. There were no retries of a failed training state and no checkpoint publication.

The five-second collective timeout therefore cannot be treated as a process-lifetime bound. The installed PyTorch `init_process_group` documentation describes asynchronous NCCL failure, and its `ProcessGroupNCCL.hpp` exposes the extra diagnostic-dump wait control. The observed delay is a concrete reason to retain independent supervisor deadlines and whole-group termination. General API guidance is in the [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html); the measured runtime and local installed source, rather than a presumed latest web version, define this receipt.

The installed runtime also emits a deprecation warning for `torch.distributed.nn.functional.all_gather`, which the current CPU collective helper uses. A later replacement needs the same first/second-derivative and full update tests; this diagnostic does not change that algorithm or suppress the warning.

Exact commands, JSON receipts, per-rank logs, tool copy/hash and PID verification are retained at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-gpu-core/gpu-readiness/`. Original working receipts are under `/tmp/hypergan-nccl-readiness-2026-09-19/`. The probe originally ran as `tools/check_nccl.py`; the same numerical diagnostic now lives in `tests/cuda/nccl_smoke.py` with a required `tests/cuda/test_nccl.py` wrapper for the explicit hardware suite. The wrapper uses a 35-second outer deadline to leave margin over observed cleanup. The equivalent passing standalone command was:

```sh
TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=1000 \
/tmp/hypergan-cuda-native-verify/bin/python tests/cuda/nccl_smoke.py \
  --output /tmp/hypergan-nccl-readiness-2026-09-19/bounded-dump-wait.json \
  --fault-timeout --timeout 20
```

The final required-hardware wrapper passed **1 test in 24.07 seconds** with `/tmp/hypergan-cuda-native-verify/bin/python -I -m pytest tests/cuda/test_nccl.py -q`, run outside the checkout using its absolute test path. It repeats the numerical and missing-peer checks with a 35-second outer deadline. All four additional owned rank PIDs were absent from `/proc` afterward; its receipt and per-rank logs are preserved under `required-suite/` in the evidence directory.

The standalone parent supervises its own children but has no independent parent-death broker. This is an explicit tool limitation; production must reuse and qualify the existing broker path. Host/guardian failure and unkillable kernel processes remain outside the local reaping guarantee.

## Concrete production blockers at the audited baseline

| Area | CPU-specific behavior | Required CUDA/NCCL work |
| --- | --- | --- |
| Profiles and selection | `execution_profiles.py` exposes CPU-only names and requires `training.device=cpu` | Explicit CUDA device/rank assignment, backend identity and actionable hardware checks. GPU-first project initialization must never silently claim a CPU run is a GPU run |
| Supervisor | `cpu_worker_service.py` and `cpu_workers.py` initialize Gloo; workers have no GPU ownership contract | Bind each rank to one visible device before NCCL; preserve independent broker monitoring, operation fences, deadlines and cleanup even after NCCL watchdog failures |
| Collective primitives | `GlooCollectives` rejects non-CPU tensors/backend; metadata uses object collectives | Device-aware numerical collectives and an explicit metadata/control-group policy. If NCCL handles objects, establish each rank's current device first and account for GPU staging; a separate Gloo control group must have coherent ordering/timeouts |
| Gradient reduction | `_reduce_gradients` creates its gradient-presence int64 tensor on CPU before `all_reduce` | Place all NCCL tensors on the owned device or explicitly route metadata through the control group; preserve absent-gradient and optimizer-ownership semantics |
| Complete updates | `ReplicatedCPUTrainer` validates CPU modules/batches/latent tensors; state digests copy to CPU | Port actual D/G/prior/auxiliary/Adam/EMA paths and replay, preserving differentiable global RA, unique global prior regularization and strict complete-boundary poisoning. Measure host synchronization costs instead of assuming scalability |
| RNG and replay | `capture_rng` saves CPU Torch, Python and NumPy RNG; named generators are CPU | Per-rank CUDA RNG and device-aware prior/penalty generators; CPU data ordering remains explicit. Include CUDA streams in replay, observation isolation, save and restore |
| Checkpoints | Distributed group/runtime identity is fixed to Gloo/CPU; payload validation requires dense CPU tensors and deserialization uses CPU | Portable device-to-host snapshots, explicit restore to rank-owned devices and optimizer placement, per-rank CUDA RNG and backend/runtime identity. Keep all-rank readiness and parent-only canonical publication |
| Runtime agreement | CPU replicas compare common runtime/config/data/strategy | Record common CUDA/cuDNN/NCCL/determinism settings plus an ordered rank-device assignment; do not incorrectly require different ranks' physical UUIDs to be equal |
| Observation/inference | Preview reconstruction and final inference use CPU assumptions or current rank device | Explicit render/export device policy, portable copied snapshots and GPU RNG isolation; preserve bounded renderer/callback cleanup and optional-versus-fatal error distinctions |

Source anchors: [distributed.py](../src/hypergan/distributed.py), [distributed_training.py](../src/hypergan/distributed_training.py), [distributed_checkpoints.py](../src/hypergan/distributed_checkpoints.py), [checkpoints.py](../src/hypergan/checkpoints.py), [cpu_worker_service.py](../src/hypergan/cpu_worker_service.py), [execution_profiles.py](../src/hypergan/execution_profiles.py). Concurrent native CUDA work may remove shared single-device blockers; its acceptance must be reported separately. This audit deliberately changes none of those production files.

## Small integration sequence

1. Land the separately implemented native CUDA path and GPU-first project creation with explicit CPU override. Require actual device placement, complete CUDA save/resume, named/global RNG continuity, previews and final inference in an installed GPU-enabled package. CPU fixtures remain explicit; unavailable requested CUDA fails clearly.
2. Add an internal fixed two-GPU NCCL strategy using the existing supervised command service and shared controller. Resolve one device per rank, metadata-group policy, complete update/accumulation math and poison-on-failure rules before exposing it publicly.
3. Qualify full D/G/prior/Adam/EMA state against controlled global-batch references for the admitted numerical configurations, including penalties and higher derivatives. Test accumulation greater than one and stochastic replay on the actual two GPUs. The linear smoke above is insufficient.
4. Add full two-GPU checkpoint/recovery acceptance: uninterrupted versus fresh-group continuation, per-rank CUDA RNG/data state, older snapshot replay, timeout/rank loss/half-update, failed staging and failure before/after parent publication. Preserve observed/durable/lost-step accounting and parent-death takeover. Checkpoint identities must reflect actual device/backend/runtime policy without promising cross-device bitwise equivalence.
5. Expose admitted CUDA profiles through public train/resume/preflight only after installed headless lifecycle and fault gates pass. Keep recipe architecture/data settings separate from execution topology and mutable service deadlines.
6. Then specify a real two-node experiment: allocation/provider, device counts, rendezvous and transport/network assumptions, data/artifact access, runtime image, timeout budget, cost cap and cleanup. The local NVLink test supplies no inter-node networking or cluster qualification; no remote allocation was authorized or started by this task.

Image licensing, architecture admission and image-quality evaluation remain separate gates. No dataset download, upstream architecture copy, GPU performance benchmark or release occurred here.

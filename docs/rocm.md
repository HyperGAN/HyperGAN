# AMD ROCm Readiness & Verification in HyperGAN

HyperGAN has experimental, hardware-untested readiness for AMD ROCm GPUs that does not regress NVIDIA CUDA or CPU execution. This document covers current support status, host requirements, execution of the automated preflight qualification script (`scripts/rocm_preflight.sh`), an overview of the collected preflight report metrics, the complete 38-row codebase audit for NVIDIA assumptions, and 10 open hardware questions mapped to preflight report sections.

---

## 1. Current ROCm Support Status in HyperGAN

**Status: experimental, untested on AMD hardware.** No HyperGAN run has executed on an AMD GPU yet; the components below are covered only by CPU tests that patch `torch.cuda`. Running `scripts/rocm_preflight.sh` (section 3) on a real host is the first hardware check.

HyperGAN targets AMD ROCm architectures via PyTorch's HIP abstraction layer (`device = "cuda"`), backed by hardware-independent readiness components landed in `exp/rocm-readiness`:

1. **GPU Inventory Provider Abstraction (`src/hypergan/gpu_inventory.py`)**:
   - Behind `hypergan.queue.inspect_gpu_hardware()` and `hypergan.queue.inspect_gpu_processes()`, a unified provider interface dynamically selects between `NvidiaSmiProvider`, `AmdSmiProvider` (`amd-smi static/metric/process --json`), and fallback `RocmSmiProvider` (`rocm-smi --json`).
   - Normalizes hardware records (`index`, `name`, `uuid`/`serial`, `memory_total`, `memory_free`, `memory_used`, `utilization`) and process records (`gpu_uuid`, `pid`, `process_name`, `used_memory`, `cmdline`).
   - Emits exactly one `WARNING` log across the process lifecycle if no GPU management tool is found, gracefully degrading rather than throwing unhandled exceptions.

2. **ROCm Runtime Identity & Checkpoint Metadata Parity (`src/hypergan/training.py`, `src/hypergan/distributed_checkpoints.py`)**:
   - Captures `torch.version.hip` and GPU architecture `properties.gcnArchName` (e.g. `gfx1100`, `gfx90a`, `gfx942`) in runtime info *only* when running on ROCm (`torch.version.hip is not None`).
   - **Strict Backward Compatibility**: Leaves existing NVIDIA and CPU metadata byte-identical, preserving cryptographic config hashes and checkpoint compatibility guarantees.
   - Missing or non-standard device properties (such as consumer GPU UUIDs) gracefully degrade to `'unavailable'`.

3. **Checkpoint Compatibility Hardware Tolerance (`src/hypergan/checkpoint_compatibility.py`)**:
   - Adds `'cuda.gcnArchName'` to `HARDWARE_TOLERANCE_KEYS` and tolerance matching rules.
   - Resuming ROCm checkpoints across different AMD GPU models (or across architectures) issues structured `HyperGANHardwareWarning` events without halting execution (Decision D9).

4. **ROCm Determinism Error Guards & BLAS Workspace Gating (`src/hypergan/numerical_policy.py`)**:
   - `CUBLAS_WORKSPACE_CONFIG` validation and warning checks are gated to execute only when `torch.version.hip is None`. On ROCm, where `rocBLAS` handles matrix operations, `CUBLAS_WORKSPACE_CONFIG` enforcement is a clean no-op.
   - When non-deterministic operations trigger PyTorch `RuntimeError` under `torch.use_deterministic_algorithms(True)`, HyperGAN enriches the exception with the failing ATen operation and instructions to relax determinism via `training.backend.deterministic_algorithms = false`.

5. **Device Ordering Documentation (`src/hypergan/execution.py`)**:
   - Preserves NVIDIA `CUDA_DEVICE_ORDER=PCI_BUS_ID` behavior without modification, while documenting AMD ROCm ordering considerations and providing side-by-side device index mapping in preflight tooling.

---

## 2. Prerequisites & Host Requirements

Before executing HyperGAN on AMD ROCm hardware, ensure the host system satisfies driver, device node, and permission requirements.

### 2.1 Kernel Modules & Device Nodes
- **`/dev/kfd` (Kernel Fusion Driver)**:
  The AMD KFD character device node is mandatory for user-space HSA compute dispatch.
  - Verification: `test -e /dev/kfd`
  - If missing: install the AMD kernel driver (e.g. `amdgpu-dkms`) and load the module via `sudo modprobe amdgpu`.
- **`/dev/dri` (Direct Rendering Infrastructure)**:
  DRM render nodes (e.g. `/dev/dri/renderD128`) must exist and be accessible.

### 2.2 User Group Permissions
The user account running HyperGAN or the preflight script must belong to the `render` group (and `video` group on certain Linux distributions):
```bash
sudo usermod -a -G render,video $USER
```
After adding groups, log out and back in or launch a new shell with `newgrp render`.

### 2.3 Supported AMD GPU Architectures
Official PyTorch ROCm wheels target the following GFX architectures:
- **RDNA 3**: `gfx1100` (AMD Radeon RX 7900 XTX / 7900 XT / PRO W7900), `gfx1101` (RX 7800 XT), `gfx1102` (RX 7700 XT)
- **CDNA 3**: `gfx942` (AMD Instinct MI300X, MI300A)
- **CDNA 2**: `gfx90a` (AMD Instinct MI250X, MI250, MI210)
- **CDNA 1**: `gfx908` (AMD Instinct MI100)

*Note on RDNA 3 Consumer GPUs*: For consumer RDNA3 cards (e.g. RX 7900 XTX) running ROCm, export `HSA_OVERRIDE_GFX_VERSION=11.0.0` if the driver does not natively recognize the PCI revision ID.

### 2.4 Python Interpreter Compatibility
Official PyTorch ROCm wheels on `download.pytorch.org/whl/rocm6.2` are prebuilt for:
- Python 3.9, 3.10, 3.11, and 3.12 (`cp39` through `cp312`).
- Python 3.14 does not currently host official prebuilt ROCm wheels on PyTorch's index.

---

## 3. Running `scripts/rocm_preflight.sh`

HyperGAN ships an idempotent, automated preflight script (`scripts/rocm_preflight.sh`) that runs top-to-bottom on a fresh single-GPU AMD Linux host or container.

### 3.1 Basic Usage
```bash
# Full preflight run (creates .venv-rocm, runs reference suite, single-GPU tests/cuda tests, and training)
./scripts/rocm_preflight.sh

# Quick smoke mode (smoke reference tests and 10 training steps)
./scripts/rocm_preflight.sh --quick

# Custom output directory and existing venv
./scripts/rocm_preflight.sh --output-dir /tmp/rocm_eval --skip-install
```

### 3.2 CLI Arguments
| Flag | Description | Default |
|---|---|---|
| `--output-dir DIR` | Output directory for `rocm_preflight_report.md` and raw CLI logs | `rocm_preflight_output` |
| `--venv DIR` | Path to virtual environment to create or activate | `.venv-rocm` |
| `--skip-install` | Skip venv creation and wheel installation (use active Python environment) | `false` |
| `--quick` | Quick execution mode: runs fast smoke tests and 10 training steps | `false` |
| `--help`, `-h` | Display help message and exit | - |

### 3.3 Environment Variables
| Variable | Description | Default |
|---|---|---|
| `TORCH_ROCM_WHEEL_URL` | PyTorch ROCm wheel index repository URL | `https://download.pytorch.org/whl/rocm6.2` |
| `TORCH_VERSION` | PyTorch package version string | `2.5.1+rocm6.2` |
| `TORCHVISION_VERSION` | Torchvision package version string | `0.20.1+rocm6.2` |
| `HSA_OVERRIDE_GFX_VERSION` | AMD HSA GFX architecture override (e.g. `11.0.0` for RX 7900 XTX) | Unset |

### 3.4 Expected Runtime
- **Quick Mode (`--quick`)**: ~15–30 seconds.
- **Full Mode**: ~2–4 minutes (depending on CPU speed for `tests/reference` and compilation of first-run MIOpen kernels).

---

## 4. Preflight Output Overview (9 Sections)

The preflight tool generates `rocm_preflight_report.md` inside `--output-dir` containing 9 comprehensive sections:

1. **Section 1: Environment Checks**:
   - Host OS, Linux kernel release, architecture.
   - `/dev/kfd` HSA character device presence and read/write accessibility.
   - `/dev/dri` DRM render nodes and user group memberships (`render`, `video`).
   - Detected GPU GFX architecture and available SMI management tool (`amd-smi` vs `rocm-smi`).
2. **Section 2: Venv Setup & Wheel Installation**:
   - Virtualenv directory, Python version.
   - Installed PyTorch ROCm wheel (`torch==2.5.1+rocm6.2`) and `torchvision`.
   - HyperGAN editable installation with optional extras (`[train,image,cifar,fid,test]`).
3. **Section 3: Sanity Check**:
   - Query results for `torch.version.hip`, device count, device name, and `gcnArchName`.
   - Forward and backward matrix multiplication (`matmul`) numerical parity vs CPU reference.
   - Forward and backward 2D convolution (`conv2d`) numerical parity vs CPU reference.
   - Maximum absolute error metrics compared against tolerance (`rtol=1e-3, atol=1e-3`).
4. **Section 4: Device-Order Table**:
   - Side-by-side comparison table: PyTorch index (`cuda:0`, `cuda:1`) → Name, UUID vs driver SMI index → Name, Serial, PCI Bus ID (BDF).
   - Alignment verification resolving whether device enumeration preserves PCIe bus order.
5. **Section 5: CPU Reference Suite**:
   - Execution results of `pytest tests/reference` verifying baseline stability inside the ROCm environment.
   - Passed, failed, skipped counts and execution duration.
6. **Section 6: Single-GPU `tests/cuda` Execution**:
   - Partitioning of the collected `tests/cuda` tests by the `multi_gpu` pytest marker (see `pyproject.toml`):
     - **Single-GPU tests** (no `multi_gpu` marker): executed directly against the target AMD GPU, recording per-test outcome and failure tracebacks.
     - **Multi-GPU tests** (marked `multi_gpu`): cataloged explicitly in the report as `deferred: needs >=2 GPUs`.
   - The counts come from collection at run time; `pytest tests/cuda --collect-only -q -m multi_gpu` lists the deferred set.
7. **Section 7: Short Real Training Run**:
   - Real training execution of the `gaussian-grid` recipe on `cuda:0` (50 steps, or 10 on quick mode).
   - Measures step throughput (`steps/sec`), peak allocated VRAM (`MB`), execution duration, and final loss metrics.
   - Single seed, metrics-only (no image rendering).
8. **Section 8: GPU Inventory Provider Validation**:
   - Live execution of `hypergan.queue.inspect_gpu_hardware()` and `inspect_gpu_processes()`.
   - Dumps both raw CLI tool output (`amd-smi` / `rocm-smi`) and normalized JSON records into the report.
9. **Section 9: Failure Classification & Summary**:
   - Automatic classification of every error across all steps into 5 canonical categories:
     `crash` | `numerical tolerance` | `determinism error` | `tooling` | `unknown`.
   - Overall qualification verdict for the AMD host.

---

## 5. Summary of Codebase Audit for NVIDIA Assumptions

The Phase 1 audit investigated all occurrences of NVIDIA-specific assumptions and CUDA symbols across `src/`, `scripts/`, and `tests/`. Below is the complete 38-row audit table:

| Location | Assumption | Behavior on ROCm | Fix in this PR or Deferred to Hardware |
|---|---|---|---|
| `src/hypergan/queue.py:95-123` | Queries `nvidia-smi` CLI via CSV output to inspect GPU inventory, name, UUID, VRAM, and utilization | `degrades` | Fix in this PR (Phase 2A) |
| `src/hypergan/queue.py:126-161` | Queries `nvidia-smi --query-compute-apps` to inspect active GPU compute processes | `degrades` | Fix in this PR (Phase 2A) |
| `src/hypergan/queue.py:183-214` | Matches target devices and active processes against NVIDIA GPU UUIDs (`GPU-...`) | `breaks` | Fix in this PR (Phase 2A) |
| `src/hypergan/inspection.py:1469-1508` | Enriches active runs in `ps` with live VRAM and device index via `nvidia-smi` process table | `degrades` | Fix in this PR (Phase 2A) |
| `src/hypergan/workspace.py:245-254` | Runs `nvidia-smi --query-gpu=index` via subprocess when torch is not loaded to resolve `auto` device selection | `breaks` | Fix in this PR (Phase 2A) |
| `src/hypergan/execution.py:175, 194` | Sets `os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"` to enforce bus-order device indexing | `unknown` | Fix in this PR (Phase 2C comments / documented) |
| `src/hypergan/execution.py:176, 190, 195` | Sets `CUDA_VISIBLE_DEVICES` to select physical GPU or isolate to CPU | `works` | Fix in this PR (documented) |
| `src/hypergan/snapshot_renderer.py:13` | Sets `os.environ['CUDA_VISIBLE_DEVICES'] = ''` to force preview renderer to run on CPU | `works` | Fix in this PR |
| `src/hypergan/metric_plugins.py:153` | Sets `os.environ['CUDA_VISIBLE_DEVICES'] = ''` to force observation workers to run on CPU | `works` | Fix in this PR |
| `src/hypergan/provenance.py:100` | Records `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` in attempt launch provenance metadata | `works` | Fix in this PR |
| `src/hypergan/numerical_policy.py:13-20` | Requires `CUBLAS_WORKSPACE_CONFIG` (`:4096:8` or `:16:8`) before CUDA init for deterministic execution | `breaks` | Fix in this PR (Phase 2D no-op on ROCm) |
| `src/hypergan/numerical_policy.py:22` | Assumes all operators have deterministic kernels under `torch.use_deterministic_algorithms(True)` | `breaks` | Fix in this PR (Phase 2D enriched error) |
| `src/hypergan/config.py:565-567` | Validates `training.backend.cublas_workspace_config` values in experiment configs | `works` | Fix in this PR |
| `src/hypergan/numerical_policy.py:26-27, 41-42` | Reads and writes `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` | `works` | Deferred to hardware |
| `src/hypergan/image_metrics.py:96-100, 113-114` | Toggles TF32 flags off during Inception-v3 FID evaluation | `works` | Deferred to hardware |
| `src/hypergan/training.py:116` | Records `result['cuda']['version'] = torch.version.cuda` and cuDNN version; omits HIP version | `degrades` | Fix in this PR (Phase 2B) |
| `src/hypergan/training.py:117` | Assumes `torch.cuda.get_device_capability()` returns CUDA major/minor compute capability | `degrades` | Fix in this PR (Phase 2B) |
| `src/hypergan/training.py:118-119` | Reads `properties.uuid` with fallback to `'unavailable'` | `degrades` | Fix in this PR (Phase 2B) |
| `src/hypergan/distributed_checkpoints.py:74` | Calls `torch.cuda.nccl.version()` to record NCCL version in distributed runtime metadata | `unknown` | Fix in this PR (Phase 2B getattr fallback) |
| `src/hypergan/distributed_checkpoints.py:77-80` | Records per-rank device metadata with CUDA capability, name, UUID, omitting `gcnArchName` and `hip` | `degrades` | Fix in this PR (Phase 2B) |
| `src/hypergan/checkpoint_compatibility.py:34-45` | `HARDWARE_TOLERANCE_KEYS` validates saved vs current runtime keys during checkpoint resume | `breaks` | Fix in this PR (Phase 2B added `cuda.gcnArchName`) |
| `src/hypergan/distributed_training.py:250` | Fused device-side nonfinite gradient check via `torch._amp_foreach_non_finite_check_and_unscale_` | `unknown` | Deferred to hardware |
| `src/hypergan/cpu_worker_service.py:165-167` | Validates `backend == 'nccl'` with message referencing `available NVIDIA GPUs` | `works` | Fix in this PR (modernized error message) |
| `src/hypergan/distributed.py:21-27` | Restricts distributed backends to `gloo` (CPU) and `nccl` (CUDA/ROCm) | `works` | Deferred to hardware |
| `src/hypergan/execution_profiles.py:12, 91` | Uses execution profile name `cuda-replicated-nccl` for multi-GPU replicated training | `works` | Deferred to hardware |
| `src/hypergan/cli.py:891-902` | `hypergan doctor` reads `torch.version.cuda` and omits `torch.version.hip` | `degrades` | Fix in this PR |
| `src/hypergan/cli.py:736-742` | Formats console banner device string as `(CUDA {dev_idx}: {prop.name})` | `degrades` | Fix in this PR |
| `src/hypergan/recipes.py:77` | Raises error recommending to check the `NVIDIA driver` when CUDA is unavailable | `degrades` | Fix in this PR |
| `scripts/metrics_training_proof.py:207-209, 275` | Probes GPU stats via `nvidia-smi` and reads `torch.version.cuda` | `degrades` | Deferred to hardware |
| `scripts/benchmark_particlegan_cifar.py:82-84` | Sets `deterministic_algorithms=True` without `cublas_workspace_config` | `breaks` | Fix in this PR |
| `scripts/start-cifar-tiny-transformer-resnet-features-pg07-lr2e-4-demo.sh:26` | Exports hardcoded NVIDIA device UUID `GPU-ed080e41-3193-3755-6756-f3d46c433331` | `breaks` | Deferred to hardware |
| `tests/cuda/nccl_smoke.py:107, 111, 208-210` | Directly accesses `properties.uuid` without getattr; reads `torch.version.cuda`; executes `nvidia-smi` and `topo -m` | `breaks` | Deferred to hardware |
| `tests/cuda/test_metric_scalar_transfer.py:20-27` | Profiler key averages assert string match on `'cudaStreamSynchronize'` and `'cudaLaunchKernel'` | `unknown` | Deferred to hardware |
| `tests/cuda/test_hndl_images.py:4, 18` | Sets `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms(True)` | `breaks` | Fix in this PR |
| `tests/cuda/test_interval_evaluation.py:16-17` | Sets `deterministic_algorithms = true` on `cuda` without `cublas_workspace_config` | `breaks` | Fix in this PR |
| `tests/cuda/test_native_cuda.py` | Runs single-GPU training, updates, model parameter devices, and exact RNG restoration | `works` | Deferred to hardware |
| `tests/cuda/test_trainer_host_syncs.py` | Single-GPU qualification of fused nonfinite check and EMA lerp/copy | `unknown` | Deferred to hardware |
| `tests/cuda/` multi-GPU tests (marked `multi_gpu`) | Assumes >= 2 CUDA GPUs with NCCL collectives | `unknown` | Deferred to hardware |

---

## 6. 10 Open Questions Requiring AMD Hardware Mapped to Preflight Sections

The following 10 open questions cannot be resolved without physical AMD hardware and live ROCm drivers. Each question is mapped directly to the preflight report section that will answer it:

1. **Host Environment, Kernel Driver & Device Node Accessibility**:
   - *Question*: Are the ROCm kernel modules (`amdgpu`, `amdkfd`) loaded, does `/dev/kfd` exist with user read/write permissions (`video`/`render` group), and what is the exact hardware GFX architecture (`gfx1100`, `gfx90a`, `gfx942`)?
   - *Mapped Preflight Section*: **Section 1: Environment Checks**
2. **PyTorch ROCm Wheel Version & Compatibility**:
   - *Question*: Does the target PyTorch ROCm wheel (`torch==2.5.1+rocm6.2` from `https://download.pytorch.org/whl/rocm6.2`) install cleanly alongside matching `torchvision==0.20.1+rocm6.2`, and do HyperGAN dependencies install without binary symbol collisions?
   - *Mapped Preflight Section*: **Section 2: Venv Setup & Wheel Install**
3. **HIP Runtime Identity & Core Tensor Kernels**:
   - *Question*: What exact values are returned by `torch.version.hip`, `torch.cuda.get_device_name(0)`, and `properties.gcnArchName`? Do standard FP32 `matmul` and `conv2d` forward/backward passes on `cuda:0` match CPU reference tensors within numerical tolerance (`1e-3`)?
   - *Mapped Preflight Section*: **Section 3: Sanity Check**
4. **Device Ordering & Bus ID Alignment**:
   - *Question*: Does PyTorch's default device enumeration index (`cuda:0`, `cuda:1`) match the PCIe bus ID order reported by `amd-smi` / `rocm-smi`? Does setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` alter or preserve this order on ROCm?
   - *Mapped Preflight Section*: **Section 4: Device-Order Table**
5. **CPU Reference Suite Stability in HIP/ROCm Environment**:
   - *Question*: Does `pytest tests/reference` pass with 100% success inside the ROCm environment without triggering unexpected GPU driver initialization or library symbol collisions?
   - *Mapped Preflight Section*: **Section 5: CPU Reference Suite**
6. **Single-GPU CUDA Test Execution & Profiler Activity Events**:
   - *Question*: Do the single-GPU tests in `tests/cuda` (those without the `multi_gpu` marker) pass without modification on the AMD GPU? Specifically, does the PyTorch Kineto profiler in `test_metric_scalar_transfer.py` report `'cudaStreamSynchronize'` or `'hipStreamSynchronize'`?
   - *Mapped Preflight Section*: **Section 6: Single-GPU `tests/cuda` Execution**
7. **Multi-GPU / RCCL Collective Support & Deferral**:
   - *Question*: On a single-GPU AMD box, does the preflight runner cleanly identify and defer every test marked `multi_gpu` as `deferred: needs >=2 GPUs`? When >= 2 AMD GPUs are present, does `backend='nccl'` (RCCL) successfully initialize process groups and reduce gradients?
   - *Mapped Preflight Section*: **Section 6: Single-GPU `tests/cuda` Execution** (deferral partition) and **Section 9: Failure Classification**
8. **Live Training Dynamics, Step Throughput, and Peak VRAM**:
   - *Question*: When executing a short training run of `gaussian-grid` on `cuda:0` for 50 steps, what is the sustained throughput (`steps/s`), peak allocated VRAM (`MB`), and final loss value? Does training proceed without numerical divergence?
   - *Mapped Preflight Section*: **Section 7: Short Real Training Run**
9. **Live GPU Management CLI Output Validation (`amd-smi` / `rocm-smi`)**:
   - *Question*: Does the target system run `amd-smi` or legacy `rocm-smi`? Does raw JSON output conform to the official documentation schema, and does HyperGAN's Phase 2A provider extract normalized records (index, name, memory, utilization, processes) correctly?
   - *Mapped Preflight Section*: **Section 8: GPU Inventory Provider Validation**
10. **Operator Determinism & Failure Classification on AMD Hardware**:
    - *Question*: Which specific PyTorch/MIOpen operators raise `RuntimeError` under `torch.use_deterministic_algorithms(True)`, does HyperGAN's error handler enrich the traceback with the failing operation and recommend setting `training.backend.deterministic_algorithms = false`, and is the failure correctly categorized as `determinism error`?
    - *Mapped Preflight Section*: **Section 9: Failure Classification & Summary**

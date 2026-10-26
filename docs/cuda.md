# Native CUDA training

New projects target one NVIDIA GPU:

```sh
hypergan new demo
hypergan train demo --run-dir runs/demo --checkpoint-every 1 --stop-after-steps 2
hypergan resume runs/demo
hypergan sample runs/demo --count 16
```

`new` writes `train.device = "cuda"`, which resolves to visible CUDA index zero. To select the second visible GPU, use `hypergan new demo-gpu1 --device cuda:1`. To run the small CPU correctness fixture, use `hypergan new cpu-demo --device cpu`. During `hypergan train`, HyperGAN auto-selects CUDA when available and falls back to CPU with a `HyperGANHardwareWarning` if CUDA is unavailable on the host, unless a CUDA device is explicitly forced via `--devices cuda:...` (or hardware verification tests). Validation and project creation remain usable without PyTorch or GPU hardware.

Install a CUDA-enabled PyTorch build and the `train` extra before training. The local validation environment uses Python 3.12 and PyTorch 2.14.0+cu130. The [quickstart](../README.md) has installation commands. GPU indices are relative to `CUDA_VISIBLE_DEVICES`; the checkpoint records the resolved device and visible GPU identity. Changing visible-device mapping, selected hardware, precision policy or runtime requires a new run rather than silent recovery conversion.

The native training loop moves the component graph, prior, objectives and batches to the configured device. Data construction and its named generator remain on CPU; batched tensors are transferred recursively without changing their dtype. Prior and penalty RNG streams live on the selected GPU. An update is acknowledged only after its CUDA work completes. CPU execution remains available for CI and small numerical comparisons.

Complete training checkpoints preserve models, prior, Adam, EMA, buffers, modes, trainability, data position, named streams and global CPU/Python/NumPy/CUDA random state. CUDA recovery requires a complete RNG inventory and the recorded runtime identity. Restore completes device work before the new attempt is published. The tests compare uninterrupted and fresh-process resumed state exactly for a controlled deterministic stochastic fixture; arbitrary custom CUDA kernels are not thereby certified deterministic. Use deterministic kernels and a matching execution policy when exact replay is required.

Periodic previews operate on copied EMA state on CPU. Inference bundles remain portable for CPU sampling with the supported components. These small observation operations preserve training RNG and selected device; they do not move training back to CPU. Custom components that require CUDA-only operators may need a separately supported inference/rendering implementation. Native callbacks still execute synchronously in the training process; supervised bounded callbacks belong to the [internal replicated service](replicated-cuda.md), which supports rank-owned CUDA devices.

# Explicit hardware checks

The regular `python -m pytest` suite selects the fast foundation and CPU numerical tests; the subprocess and multi-rank half is gated behind `-m heavy`. Hardware checks are a separate gate and fail if required hardware is missing:

```sh
python -m pytest tests/cuda -q
# Two GPUs whose NCCL peer-to-peer path hangs (see Troubleshooting below):
NCCL_P2P_DISABLE=1 python -m pytest tests/cuda -q
```

Tests marked `multi_gpu` need two or more visible GPUs; the rest run on one.

Run against an installed wheel outside the checkout for integration evidence, using an absolute tests path and `python -I -m pytest ... --import-mode=importlib`. The CUDA suite includes native updates/recovery, the NCCL diagnostic, and complete replicated numerical/recovery cases. See the [execution ledger](../reports/archive/resurrection-status.md) for exact results and limits. Two GPUs on one host do not establish image quality or a real cluster.

The [internal CUDA/NCCL profile](replicated-cuda.md) connects rank-owned devices, device-aware reductions, accumulated replay and coordinated recovery to the existing independent supervisor. The [public execution commands](execution.md) now select this profile with `--profile cuda-replicated-nccl` and infer it on resume. Actual two-host testing follows with a concrete cloud allocation.

# Troubleshooting: NCCL hangs at the first collective

On some multi-GPU hosts NCCL's direct GPU-to-GPU (peer-to-peer, P2P) transport
does not work, and every NCCL job stalls at its first collective, an all-gather,
until the collective timeout aborts it. This happens on the project's development
machine with two RTX A6000s, for two-rank training and for the NCCL tests in
`tests/cuda` alike. Setting `NCCL_P2P_DISABLE=1` makes NCCL copy through host
memory instead, and the same jobs then complete, at some cost in collective
bandwidth.

Set it in the shell that launches training or the tests; the rank processes
inherit it:

```sh
export NCCL_P2P_DISABLE=1
hypergan train demo --run-dir runs/demo --profile cuda-replicated-nccl --steps 100 --no-server
python -m pytest tests/cuda -q
```

or record it in the `[run.env]` table of the model file or the workspace
`hypergan.toml`; `hypergan train` and `hypergan resume` then re-execute themselves with that environment
(see [configuration](configuration.md)):

```toml
[run.env]
NCCL_P2P_DISABLE = "1"
```

`hypergan qualify` already sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` for
its own NCCL suite. If a job still hangs, rerun it with `NCCL_DEBUG=INFO` to see
which transport NCCL selects, and check `nvidia-smi topo -m` for the link between
the GPUs.

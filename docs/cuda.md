# Native CUDA training

New projects target one NVIDIA GPU:

```sh
hypergan new demo
hypergan train demo --run-dir runs/demo --checkpoint-every 1 --stop-after-steps 2
hypergan resume runs/demo
hypergan sample runs/demo --count 16
```

`new` writes `training.device="cuda"`, which resolves to visible CUDA index zero. To select the second visible GPU, use `hypergan new demo-gpu1 --device cuda:1`. To run the small CPU correctness fixture, use `hypergan new cpu-demo --device cpu`. Explicit device requests fail if unavailable; there is no automatic CPU fallback. Validation and project creation remain usable without PyTorch or GPU hardware.

Install a CUDA-enabled PyTorch build and the `train` extra before training. The local validation environment uses Python 3.12 and PyTorch 2.14.0+cu130. The [quickstart](../README.md) has installation commands. GPU indices are relative to `CUDA_VISIBLE_DEVICES`; the checkpoint records the resolved device and visible GPU identity. Changing visible-device mapping, selected hardware, precision policy or runtime requires a new run rather than silent recovery conversion.

The native training loop moves the component graph, prior, objectives and batches to the configured device. Data construction and its named generator remain on CPU; batched tensors are transferred recursively without changing their dtype. Prior and penalty RNG streams live on the selected GPU. An update is acknowledged only after its CUDA work completes. CPU execution remains available for CI and small numerical comparisons.

Complete training checkpoints preserve models, prior, Adam, EMA, buffers, modes, trainability, data position, named streams and global CPU/Python/NumPy/CUDA random state. CUDA recovery requires a complete RNG inventory and the recorded runtime identity. Restore completes device work before the new attempt is published. The tests compare uninterrupted and fresh-process resumed state exactly for a controlled deterministic stochastic fixture; arbitrary custom CUDA kernels are not thereby certified deterministic. Use deterministic kernels and a matching execution policy when exact replay is required.

Periodic previews operate on copied EMA state on CPU. Inference bundles remain portable for CPU sampling with the supported components. These small observation operations preserve training RNG and selected device; they do not move training back to CPU. Custom components that require CUDA-only operators may need a separately supported inference/rendering implementation. Native callbacks still execute synchronously in the training process; supervised bounded callbacks belong to the [internal replicated service](replicated-cuda.md), which supports rank-owned CUDA devices.

# Explicit hardware checks

The regular `python -m pytest` suite selects foundation and CPU numerical tests. Hardware checks are a separate gate and fail if required hardware is missing:

```sh
python -m pytest tests/cuda -q
```

Run against an installed wheel outside the checkout for integration evidence, using an absolute tests path and `python -I -m pytest ... --import-mode=importlib`. The CUDA suite includes native updates/recovery, the NCCL diagnostic, and complete replicated numerical/recovery cases. See the [execution ledger](../reports/resurrection-status.md) for exact results and limits. Two GPUs on one host do not establish image quality or a real cluster.

The [internal CUDA/NCCL profile](replicated-cuda.md) connects rank-owned devices, device-aware reductions, accumulated replay and coordinated recovery to the existing independent supervisor. The [public execution commands](execution.md) now select this profile with `--profile cuda-replicated-nccl` and infer it on resume. Actual two-host testing follows with a concrete cloud allocation.

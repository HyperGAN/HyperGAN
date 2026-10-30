#!/usr/bin/env bash
# Head-to-Head TransGAN Critic Benchmark Harness (2x NVIDIA RTX A6000)
# GPU 0: Scratch Multi-Scale TransGAN D (examples/cifar-transgan-scratch.toml)
# GPU 1: Pretrained ResNet-18 Feature D (examples/cifar-transgan-resnet18.toml)
#
# Evaluation: FID-50k curve across 200,000 steps (or user-specified steps)
# Single seed: 24002 (no seed sweeps per AGENTS.md rules)
set -euo pipefail

STEPS="${1:-200000}"

echo "================================================================="
echo "HyperGAN 2.0.0 Head-to-Head TransGAN Critic Benchmark"
echo "Host GPUs: 2x NVIDIA RTX A6000 (48GB VRAM each)"
echo "Dataset: CIFAR-10"
echo "Steps: ${STEPS}"
echo "Evaluation: FID-50k curve (every 10,000 steps)"
echo "Single seed: 24002"
echo "================================================================="

# 1. Host environment flags per project specifications
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HYPERGAN_NO_SERVER=1

# 2. Acquire and verify dataset into ./data if not present
echo "[1/4] Ensuring CIFAR-10 dataset is prepared..."
if [ ! -d "./data/cifar-10-batches-py" ]; then
    python3 -m hypergan download-data cifar10 --output ./data --cache-dir ~/.hypergan/cache/ --quiet
fi

# 3. Create logs directory
LOG_DIR="./runs/logs"
mkdir -p "${LOG_DIR}"

RUN_DIR_SCRATCH="./runs/benchmark-transgan-scratch"
RUN_DIR_RESNET="./runs/benchmark-resnet18-feature"

# 4. Launch concurrent training runs on GPU 0 and GPU 1
echo "[2/4] Launching GPU 0 (Scratch TransGAN D) and GPU 1 (Pretrained ResNet-18 Feature D)..."

CUDA_VISIBLE_DEVICES=0 hypergan train examples/cifar-transgan-scratch.toml \
    --run-dir "${RUN_DIR_SCRATCH}" \
    -d 0 \
    --no-server \
    --stop-after-steps "${STEPS}" \
    > "${LOG_DIR}/benchmark-transgan-scratch.log" 2>&1 &
PID_GPU0=$!

CUDA_VISIBLE_DEVICES=1 hypergan train examples/cifar-transgan-resnet18.toml \
    --run-dir "${RUN_DIR_RESNET}" \
    -d 0 \
    --no-server \
    --stop-after-steps "${STEPS}" \
    > "${LOG_DIR}/benchmark-resnet18-feature.log" 2>&1 &
PID_GPU1=$!

echo "GPU 0 PID: ${PID_GPU0} (Scratch TransGAN D -> ${RUN_DIR_SCRATCH})"
echo "GPU 1 PID: ${PID_GPU1} (Pretrained ResNet-18 Feature D -> ${RUN_DIR_RESNET})"

# Wait for both benchmark runs to complete
wait "${PID_GPU0}"
wait "${PID_GPU1}"

echo "[3/4] Both training runs completed. Extracting comparison metrics..."

# 5. Extract and compare curves
mkdir -p ./runs
python3 -m hypergan compare \
    "${RUN_DIR_SCRATCH}" \
    "${RUN_DIR_RESNET}" \
    --metric fid50k_train \
    --json > ./runs/benchmark-critic-comparison.json || true

echo "[4/4] Benchmark comparison results saved to ./runs/benchmark-critic-comparison.json"
python3 -c "
import json
from pathlib import Path
path = Path('./runs/benchmark-critic-comparison.json')
if path.exists() and path.stat().st_size > 0:
    try:
        data = json.loads(path.read_text())
        print('Benchmark Comparison Summary:')
        for r in data.get('runs', []):
            name = r.get('name', 'unknown')
            best = r.get('best', {})
            val = best.get('value')
            step = best.get('step')
            print(f'  Run: {name} | Best FID: {val} at step {step}')
    except Exception as e:
        print('Could not parse JSON summary:', e)
else:
    print('Comparison JSON not written or empty.')
"

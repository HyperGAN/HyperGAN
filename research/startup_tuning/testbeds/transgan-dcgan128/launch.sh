#!/usr/bin/env bash
set -euo pipefail
RUNS_DIR="/home/martyn/dev/hypergan/training-runs"
WORKTREE="/home/martyn/dev/hypergan/generator-signal-diagnostic"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# TransGAN generator, plain DCGAN critic. Tuning stays off.
exec "$RUNS_DIR/transgan-128-env/bin/python" -m hypergan train \
  "$WORKTREE/research/startup_tuning/testbeds/transgan-dcgan128/transgan-dcgan.toml" \
  --run-dir "$RUNS_DIR/train-transgan-dcgan-128" \
  --no-tune --server --checkpoint-every 1000 --preview-every 100 --progress-every 20 "$@"

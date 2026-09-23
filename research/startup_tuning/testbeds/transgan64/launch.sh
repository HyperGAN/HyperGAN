#!/usr/bin/env bash
set -euo pipefail
RUNS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="/home/martyn/dev/hypergan/generator-signal-diagnostic"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Fresh, untuned 64px control with the original 128px training hyperparameters.
exec "$RUNS_DIR/transgan-128-env/bin/python" -m hypergan train \
  "$RUNS_DIR/logos-transgan-dinov3-multidepth-64-init-v2/transgan-dinov3-multidepth.toml" \
  --run-dir "$RUNS_DIR/train-transgan-dinov3-multidepth-64-init-v2" \
  --no-tune --server --checkpoint-every 1000 --preview-every 100 --progress-every 20 "$@"

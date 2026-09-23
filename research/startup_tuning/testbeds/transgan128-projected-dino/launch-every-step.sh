#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd -- "$RECIPE_DIR/../../../.." && pwd)"
RUNS_DIR="/home/martyn/dev/hypergan/training-runs"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=GPU-ed080e41-3193-3755-6756-f3d46c433331
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# GPU 0. Original endpoint cap every step; ordinary 1:1 training, tuning off.
exec "$RUNS_DIR/transgan-128-env/bin/python" -m hypergan train \
  "$RECIPE_DIR/transgan-projected-dino-every-step.toml" \
  --run-dir "$RUNS_DIR/train-transgan-projected-dinov3-128-every-step" \
  --steps 128 --no-tune --server \
  --checkpoint-every 128 --preview-every 16 --progress-every 8 "$@"

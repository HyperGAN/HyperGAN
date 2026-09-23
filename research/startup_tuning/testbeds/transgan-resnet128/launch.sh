#!/usr/bin/env bash
set -euo pipefail
RUNS_DIR="/home/martyn/dev/hypergan/training-runs"
WORKTREE="/home/martyn/dev/hypergan/generator-signal-diagnostic"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=GPU-ed080e41-3193-3755-6756-f3d46c433331
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# TransGAN generator, frozen pretrained ResNet18 multiscale critic. Tuning stays off.
exec "$RUNS_DIR/transgan-128-env/bin/python" -m hypergan train \
  "$WORKTREE/research/startup_tuning/testbeds/transgan-resnet128/transgan-resnet.toml" \
  --run-dir "$RUNS_DIR/train-transgan-resnet-multiscale-128" \
  --no-tune --server --checkpoint-every 1000 --preview-every 100 --progress-every 20 "$@"

#!/usr/bin/env bash
set -euo pipefail

# Resolve the configuration and existing run relative to this launcher.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec hypergan train "$script_dir/cifar10-pretrained-20260920/cifar10.toml" \
  --run-dir "$script_dir/train-develop" --server --checkpoint-every 1000 \
  --preview-every 500 --progress-every 100 --dev "$@"

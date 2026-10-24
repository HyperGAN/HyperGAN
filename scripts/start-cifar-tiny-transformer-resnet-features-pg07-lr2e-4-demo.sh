#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="cifar-tiny-transformer-resnet-features-pg07-lr2e-4-demo.toml"
if [[ -f "$SCRIPT_DIR/$CONFIG_NAME" ]]; then
  RUNS_DIR="${RUNS_DIR:-$SCRIPT_DIR}"
  CONFIG="${CONFIG:-$SCRIPT_DIR/$CONFIG_NAME}"
else
  RUNS_DIR="${RUNS_DIR:-$REPO_ROOT}"
  CONFIG="${CONFIG:-$REPO_ROOT/examples/$CONFIG_NAME}"
fi
RUNS_DIR="$(cd -- "$RUNS_DIR" && pwd)"
WORKTREE="${WORKTREE:-$REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON="${PYTHON:-$PYTHON_BIN}"
RUN_DIR="${RUN_DIR:-$RUNS_DIR/runs/train-cifar-tiny-transformer-resnet-features-pg07-lr2e-4-demo}"

# develop uses ParticleGAN 0.8. A pg07-compatible worktree retains the pg07 training loop
# alongside the current viewer, including sample history and the Model tab.
if ! command -v "$PYTHON" >/dev/null 2>&1 || [[ ! -f "$WORKTREE/src/hypergan/training.py" ]]; then
  printf 'Demo requires a valid Python interpreter (%s) and HyperGAN checkout at %s.\n' "$PYTHON" "$WORKTREE" >&2
  exit 1
fi
WORKTREE="$(cd -- "$WORKTREE" && pwd)"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" - <<'PY'
from importlib.metadata import version

installed = version("particlegan")
if installed != "0.7.0":
    raise SystemExit(f"This demo requires ParticleGAN 0.7.0; found {installed}")
PY

cd "$RUNS_DIR"
# Construction-only validation: no training updates or run directory.
if [[ "${1:-}" == "--check" ]]; then
  shift
  exec "$PYTHON" -m hypergan preflight "$CONFIG" --runtime "$@"
fi

# Same seeds and 200k-step schedule as pg07/lr2e-4, with separate artifacts.
# Repeating this launcher resumes its own run. Extra CLI options go last so
# --port, --public-origin, --stop-after-steps, and interval overrides work.
exec "$PYTHON" -m hypergan train "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --server --open --checkpoint-every 1000 --preview-every 100 \
  --progress-every 20 "$@"

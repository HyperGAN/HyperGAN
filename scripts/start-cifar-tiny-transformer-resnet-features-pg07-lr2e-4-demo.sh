#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="cifar-tiny-transformer-resnet-features-pg07-lr2e-4-demo.toml"
if [[ -f "$SCRIPT_DIR/$CONFIG_NAME" ]]; then
  RUNS_DIR="${RUNS_DIR:-$SCRIPT_DIR}"
  CONFIG="${CONFIG:-$SCRIPT_DIR/$CONFIG_NAME}"
else
  RUNS_DIR="${RUNS_DIR:-$SCRIPT_DIR/../../training-runs}"
  CONFIG="${CONFIG:-$SCRIPT_DIR/../examples/$CONFIG_NAME}"
fi
RUNS_DIR="$(cd -- "$RUNS_DIR" && pwd)"
WORKTREE="${WORKTREE:-$RUNS_DIR/../worktrees/develop-pg07-lazy16}"
PYTHON="$RUNS_DIR/particlegan07-env/bin/python"
RUN_DIR="${RUN_DIR:-/mnt/ml7tb/hypergan-training-runs/train-cifar-tiny-transformer-resnet-features-pg07-lr2e-4-demo}"

# develop uses ParticleGAN 0.8. This worktree retains the pg07 training loop
# alongside the current viewer, including sample history and the Model tab.
if [[ ! -x "$PYTHON" || ! -f "$WORKTREE/src/hypergan/training.py" ]]; then
  printf 'Demo requires particlegan07-env in %s and the pg07-compatible worktree at %s.\n' "$RUNS_DIR" "$WORKTREE" >&2
  exit 1
fi
WORKTREE="$(cd -- "$WORKTREE" && pwd)"
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-ed080e41-3193-3755-6756-f3d46c433331}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" - <<'PY'
from importlib.metadata import version

installed = version("particlegan")
if installed != "0.7.0":
    raise SystemExit(f"This demo requires ParticleGAN 0.7.0; found {installed} in particlegan07-env")
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

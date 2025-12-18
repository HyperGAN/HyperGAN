#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_and_new.sh <model-name> [--install-editable]
# Creates a Python 3 venv (in ./venv), installs minimal dependencies required
# to run `bin/hypergan new` (avoids installing heavy packages like torch),
# and runs the `new` command to create a configuration file.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model-name> [--install-editable]"
  exit 2
fi

MODEL_NAME="$1"
INSTALL_EDITABLE=false
if [ "${2-}" = "--install-editable" ]; then
  INSTALL_EDITABLE=true
fi

# Locate a Python 3 executable
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  # on some systems python -> python3
  PY=python
elif command -v py >/dev/null 2>&1; then
  PY='py -3'
else
  echo "No Python 3 interpreter found. Install Python 3 and try again." >&2
  exit 1
fi

# Ensure we're running from repository root (scripts/..)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Using Python: $PY"

# Create venv
if [ ! -d "venv" ]; then
  echo "Creating virtual environment in ./venv..."
  $PY -m venv venv
fi

# Activate venv (handles both Unix and Windows Git Bash paths)
if [ -f "venv/bin/activate" ]; then
  # Unix-like
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  # Git Bash / Windows
  # shellcheck disable=SC1091
  source venv/Scripts/activate
else
  echo "Could not find the venv activation script. Created venv, but can't activate it." >&2
  exit 1
fi

# Upgrade pip/tools and install minimal deps (avoid heavy GPU packages)
python -m pip install --upgrade pip setuptools wheel
python -m pip install hyperchamber semantic_version

# Run the lightweight `new` command using the script file directly (avoids importing heavy libs)
echo "Creating new configuration: ${MODEL_NAME}.json"
python bin/hypergan new "${MODEL_NAME}"

# Check result
if [ -f "${MODEL_NAME}.json" ] || [ -f "${MODEL_NAME}.toml" ]; then
  echo "Success: created $(ls ${MODEL_NAME}.*)"
else
  echo "Warning: configuration file was not created in the current directory." >&2
  exit 1
fi

if [ "$INSTALL_EDITABLE" = true ]; then
  echo "Installing package in editable mode (this may attempt to install torch and other heavy deps)."
  python -m pip install -r requirements.txt
  python -m pip install -e .
  echo "Editable install complete. You can run 'hypergan <command>' directly once installation finished." 
else
  echo "Done. To run full HyperGAN commands later (train, sample, etc.), run:"
  echo "  source venv/bin/activate  # or venv/Scripts/activate on Windows"
  echo "  python -m pip install -r requirements.txt"
  echo "  python -m pip install -e ."
  echo "Or add --install-editable to this script to perform that step automatically (warning: may install torch)."
fi

exit 0

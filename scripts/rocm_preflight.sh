#!/usr/bin/env bash
# ==============================================================================
# HyperGAN ROCm Preflight Verification Script
# ==============================================================================
# Verifies host environment, driver nodes, PyTorch ROCm packages, numerical sanity,
# device ordering, reference suite, single-GPU tests, live training, and GPU inventory.
#
# Usage:
#   ./scripts/rocm_preflight.sh [OPTIONS]
#
# Options:
#   --output-dir DIR   Directory for report and logs (default: rocm_preflight_output)
#   --venv DIR         Path to virtual environment (default: .venv-rocm or current venv)
#   --skip-install     Skip virtualenv creation and wheel installation
#   --quick            Quick mode (smoke reference tests and 10 training steps)
#   --help, -h         Show this help message and exit
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${REPO_ROOT}/rocm_preflight_output"
VENV_DIR="${REPO_ROOT}/.venv-rocm"
SKIP_INSTALL=false
QUICK_MODE=false

# Default wheel and package versions (overridable via environment)
TORCH_ROCM_WHEEL_URL="${TORCH_ROCM_WHEEL_URL:-https://download.pytorch.org/whl/rocm6.2}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+rocm6.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1+rocm6.2}"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help|-h)
            echo "HyperGAN ROCm Preflight Tool"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR   Directory for report and raw logs (default: rocm_preflight_output)"
            echo "  --venv DIR         Path to virtual environment (default: .venv-rocm)"
            echo "  --skip-install     Skip venv creation and wheel installation"
            echo "  --quick            Quick mode (smoke reference tests and 10 training steps)"
            echo "  --help, -h         Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

echo "======================================================================"
echo " Starting HyperGAN ROCm Preflight Verification"
echo " Working Directory: ${REPO_ROOT}"
echo " Output Directory:  ${OUTPUT_DIR}"
echo " Timestamp:         $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Step 1: Environment Checks
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/9] Environment Checks..."

OS_NAME="$(uname -s)"
KERNEL_VER="$(uname -r)"
ARCH="$(uname -m)"
echo "  Operating System: ${OS_NAME} ${KERNEL_VER} (${ARCH})"

# Check /dev/kfd - Mandatory AMD Kernel Fusion Driver early failure check
if [ ! -e /dev/kfd ]; then
    echo "======================================================================" >&2
    echo "ERROR: /dev/kfd does not exist!" >&2
    echo "The AMD GPU Kernel Fusion Driver (amdkfd) character device is required" >&2
    echo "for ROCm user-space compute dispatch." >&2
    echo "" >&2
    echo "Remediation Steps:" >&2
    echo "  1. Install the amdgpu kernel module package (e.g. amdgpu-dkms):" >&2
    echo "     sudo apt install amdgpu-dkms    # Ubuntu/Debian" >&2
    echo "     sudo dnf install amdgpu-dkms    # RHEL/Fedora" >&2
    echo "  2. Ensure the kernel module is loaded:" >&2
    echo "     sudo modprobe amdgpu" >&2
    echo "  3. Verify /dev/kfd is created with correct permissions:" >&2
    echo "     ls -la /dev/kfd" >&2
    echo "======================================================================" >&2
    exit 1
fi

# Check /dev/kfd read/write permissions
if [ ! -r /dev/kfd ] || [ ! -w /dev/kfd ]; then
    echo "======================================================================" >&2
    echo "ERROR: Current user (${USER}) cannot read/write /dev/kfd!" >&2
    echo "ROCm requires read and write permissions to the /dev/kfd device node." >&2
    echo "" >&2
    echo "Remediation Step:" >&2
    echo "  Add your user account to the 'render' and 'video' groups:" >&2
    echo "     sudo usermod -a -G render,video ${USER}" >&2
    echo "  Log out and log back in, or run 'newgrp render' to refresh group membership." >&2
    echo "======================================================================" >&2
    exit 1
fi
echo "  /dev/kfd: Present and readable/writeable by ${USER} (PASS)"

# Check /dev/dri render nodes
if [ ! -d /dev/dri ]; then
    echo "  WARNING: /dev/dri directory not found." >&2
else
    RENDER_NODES=$(ls /dev/dri/renderD* 2>/dev/null || true)
    if [ -z "${RENDER_NODES}" ]; then
        echo "  WARNING: No /dev/dri/renderD* DRM render nodes found." >&2
    else
        echo "  /dev/dri DRM Nodes: ${RENDER_NODES}"
    fi
fi

# Check user groups
USER_GROUPS=$(id -Gn 2>/dev/null || true)
echo "  User Group Memberships: ${USER_GROUPS}"
if [[ ! " ${USER_GROUPS} " =~ " render " ]] && [[ ! " ${USER_GROUPS} " =~ " video " ]]; then
    echo "  WARNING: User is not explicitly a member of 'render' or 'video' group." >&2
fi

# Probe for GPU GFX architecture via SMI or rocminfo
GFX_ARCH="unknown"
if command -v rocminfo >/dev/null 2>&1; then
    GFX_ARCH=$(rocminfo 2>/dev/null | grep -E "Name:[[:space:]]+gfx" | head -n1 | awk '{print $2}' || echo "unknown")
    rocminfo > "${LOGS_DIR}/rocminfo.raw.txt" 2>&1 || true
fi

if [ "${GFX_ARCH}" = "unknown" ] && command -v rocm-smi >/dev/null 2>&1; then
    GFX_ARCH=$(rocm-smi --showgfxversion --json 2>/dev/null | grep -oE "gfx[0-9a-z]+" | head -n1 || echo "unknown")
    rocm-smi --showall > "${LOGS_DIR}/rocm-smi.raw.txt" 2>&1 || true
fi

if [ "${GFX_ARCH}" = "unknown" ] && command -v amd-smi >/dev/null 2>&1; then
    amd-smi static --json > "${LOGS_DIR}/amd-smi.static.raw.json" 2>&1 || true
    amd-smi metric --json > "${LOGS_DIR}/amd-smi.metric.raw.json" 2>&1 || true
fi

echo "  Detected GPU GFX Architecture: ${GFX_ARCH}"

# ------------------------------------------------------------------------------
# Step 2: Venv Setup & Wheel Installation
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/9] Venv Setup & PyTorch ROCm Wheel Install..."

if [ "${SKIP_INSTALL}" = true ]; then
    echo "  Skipping virtualenv setup and package install (--skip-install)."
    PYTHON_BIN="python3"
    if [ -n "${VIRTUAL_ENV}" ]; then
        PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    elif [ -f "${VENV_DIR}/bin/python" ]; then
        PYTHON_BIN="${VENV_DIR}/bin/python"
    fi
else
    # Find suitable Python interpreter (prefer 3.12 or 3.11 for PyTorch ROCm wheels)
    SYSTEM_PYTHON=""
    for py_candidate in python3.12 python3.11 python3; do
        if command -v "${py_candidate}" >/dev/null 2>&1; then
            PY_VER=$("${py_candidate}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            # PyTorch ROCm wheels are available for 3.9 through 3.12
            if [[ "${PY_VER}" == "3.11" || "${PY_VER}" == "3.12" || "${PY_VER}" == "3.10" || "${PY_VER}" == "3.9" ]]; then
                SYSTEM_PYTHON="${py_candidate}"
                break
            fi
        fi
    done

    if [ -z "${SYSTEM_PYTHON}" ]; then
        SYSTEM_PYTHON="python3"
        echo "  Notice: Defaulting to $(which ${SYSTEM_PYTHON}). Note that PyTorch ROCm wheels officially target Python 3.9–3.12."
    fi

    echo "  Using Python interpreter: ${SYSTEM_PYTHON} ($(${SYSTEM_PYTHON} --version))"

    if [ ! -d "${VENV_DIR}" ]; then
        echo "  Creating virtual environment at ${VENV_DIR}..."
        "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"
    fi

    PYTHON_BIN="${VENV_DIR}/bin/python"
    PIP_BIN="${VENV_DIR}/bin/pip"

    echo "  Upgrading pip, setuptools, wheel..."
    "${PIP_BIN}" install --upgrade pip setuptools wheel > "${LOGS_DIR}/pip_upgrade.log" 2>&1 || true

    echo "  Installing PyTorch ROCm wheels from ${TORCH_ROCM_WHEEL_URL}..."
    echo "    torch==${TORCH_VERSION}"
    echo "    torchvision==${TORCHVISION_VERSION}"
    "${PIP_BIN}" install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_ROCM_WHEEL_URL}" > "${LOGS_DIR}/torch_install.log" 2>&1 || {
            echo "  WARNING: Direct wheel pin install exited non-zero. Attempting unpinned ROCm wheel install..."
            "${PIP_BIN}" install torch torchvision --index-url "${TORCH_ROCM_WHEEL_URL}" >> "${LOGS_DIR}/torch_install.log" 2>&1 || true
        }

    echo "  Installing HyperGAN editable with extras [train,image,cifar,fid,test]..."
    "${PIP_BIN}" install -e "${REPO_ROOT}[train,image,cifar,fid,test]" > "${LOGS_DIR}/hypergan_install.log" 2>&1 || {
        echo "  WARNING: Full extras install encountered dependency warnings; falling back to core install..."
        "${PIP_BIN}" install -e "${REPO_ROOT}" >> "${LOGS_DIR}/hypergan_install.log" 2>&1 || true
    }
fi

# Ensure Python can find repo sources
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------------------
# Steps 3 through 9: Executed via Python Preflight Helper
# ------------------------------------------------------------------------------
echo ""
echo "[Steps 3-9] Executing Python preflight helper..."
HELPER_ARGS=(
    "--output-dir" "${OUTPUT_DIR}"
)
if [ "${QUICK_MODE}" = true ]; then
    HELPER_ARGS+=("--quick")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/rocm_preflight_helper.py" "${HELPER_ARGS[@]}"

echo ""
echo "======================================================================"
echo " ROCm Preflight Verification Completed"
echo " Report written to: ${OUTPUT_DIR}/rocm_preflight_report.md"
echo "======================================================================"

import os
from pathlib import Path
import tempfile
import pytest

# Ensure multi-GPU NCCL runs smoothly across PCIe topology without NVLink
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

try:
    from e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_GLOO,
        HAVE_NCCL,
        make_toy_config,
    )
except ImportError:
    from tests.e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_GLOO,
        HAVE_NCCL,
        make_toy_config,
    )


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """Provide a clean isolated workspace path."""
    return tmp_path


@pytest.fixture
def toy_cpu_config():
    """Return a minimal deterministic CPU training configuration."""
    return make_toy_config(device="cpu", batch_size=8, steps=3)


@pytest.fixture
def toy_cuda_config():
    """Return a minimal deterministic CUDA training configuration if CUDA is available."""
    if not HAVE_CUDA:
        pytest.skip("CUDA device not available")
    return make_toy_config(device="cuda", batch_size=8, steps=3)

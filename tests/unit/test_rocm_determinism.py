"""Unit tests for ROCm determinism handling and CUBLAS_WORKSPACE_CONFIG gating.

Covers:
1. CUBLAS_WORKSPACE_CONFIG is a no-op on ROCm while preserving validation on NVIDIA.
2. ROCm determinism error message enrichment surfaces the failing op and configuration key.
"""
import os
from unittest.mock import patch
import pytest
import torch

from hypergan.numerical_policy import (
    apply_backend_policy,
    enrich_rocm_determinism_error,
    rocm_determinism_guard,
)


# ==============================================================================
# 1. CUBLAS_WORKSPACE_CONFIG Gating
# ==============================================================================

def test_cublas_workspace_config_is_noop_on_rocm():
    config = {
        "training": {
            "device": "cuda:0",
            "backend": {
                "deterministic_algorithms": True,
                # Intentionally omit cublas_workspace_config
            }
        }
    }

    # Ensure CUBLAS_WORKSPACE_CONFIG is unset or invalid
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(torch.version, "hip", "6.2.0"), \
         patch("torch.use_deterministic_algorithms"):

        # On ROCm (torch.version.hip is not None), this must succeed without raising ValueError
        apply_backend_policy(config)


def test_cublas_workspace_config_init_check_is_noop_on_rocm():
    config = {
        "training": {
            "device": "cuda:0",
            "backend": {
                "cublas_workspace_config": ":4096:8"
            }
        }
    }

    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
         patch.object(torch.version, "hip", "6.2.0"), \
         patch("torch.cuda.is_initialized", return_value=True):

        # Even if CUDA is initialized and config differs, ROCm must ignore CUBLAS setting
        apply_backend_policy(config)


def test_cublas_workspace_config_is_enforced_on_nvidia():
    config = {
        "training": {
            "device": "cuda:0",
            "backend": {
                "deterministic_algorithms": True,
            }
        }
    }

    with patch.dict(os.environ, {}, clear=True), \
         patch.object(torch.version, "hip", None):

        # On NVIDIA (hip is None), missing CUBLAS_WORKSPACE_CONFIG must raise ValueError
        with pytest.raises(ValueError, match="Deterministic CUDA execution requires"):
            apply_backend_policy(config)


def test_cublas_workspace_config_init_check_is_enforced_on_nvidia():
    config = {
        "training": {
            "device": "cuda:0",
            "backend": {
                "cublas_workspace_config": ":4096:8"
            }
        }
    }

    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
         patch.object(torch.version, "hip", None), \
         patch("torch.cuda.is_initialized", return_value=True):

        # On NVIDIA (hip is None), changing workspace after init must raise ValueError
        with pytest.raises(ValueError, match="CUBLAS workspace policy must be set before CUDA initialization"):
            apply_backend_policy(config)


# ==============================================================================
# 2. Determinism Error Enrichment on ROCm
# ==============================================================================

def test_rocm_determinism_error_enrichment():
    failing_op_error = RuntimeError(
        "aten::upsample_nearest2d_backward does not have a deterministic implementation, "
        "but you set 'torch.use_deterministic_algorithms(True)'."
    )

    with patch.object(torch.version, "hip", "6.2.0"):
        enriched = enrich_rocm_determinism_error(failing_op_error)

    msg = str(enriched)
    assert "[ROCm Determinism Error]" in msg
    assert "upsample_nearest2d_backward" in msg
    assert "training.backend.deterministic_algorithms = false" in msg
    assert "torch.version.hip=6.2.0" in msg


def test_rocm_determinism_guard_context_manager():
    with patch.object(torch.version, "hip", "6.2.0"):
        with pytest.raises(RuntimeError) as exc_info:
            with rocm_determinism_guard():
                raise RuntimeError(
                    "operation aten::conv2d does not have a deterministic implementation "
                    "when used with use_deterministic_algorithms(True)"
                )

    msg = str(exc_info.value)
    assert "[ROCm Determinism Error]" in msg
    assert "conv2d" in msg
    assert "training.backend.deterministic_algorithms = false" in msg


def test_rocm_determinism_guard_leaves_nvidia_unchanged():
    orig_msg = "aten::conv2d does not have a deterministic implementation"

    with patch.object(torch.version, "hip", None):
        with pytest.raises(RuntimeError) as exc_info:
            with rocm_determinism_guard():
                raise RuntimeError(orig_msg)

    # On NVIDIA, message must be completely unmodified
    assert str(exc_info.value) == orig_msg
    assert "[ROCm Determinism Error]" not in str(exc_info.value)


def test_rocm_determinism_guard_leaves_non_determinism_errors_unmodified():
    oom_error = "CUDA out of memory. Tried to allocate 2.00 GiB"

    with patch.object(torch.version, "hip", "6.2.0"):
        with pytest.raises(RuntimeError) as exc_info:
            with rocm_determinism_guard():
                raise RuntimeError(oom_error)

    # Unrelated errors should not be enriched
    assert str(exc_info.value) == oom_error
    assert "[ROCm Determinism Error]" not in str(exc_info.value)

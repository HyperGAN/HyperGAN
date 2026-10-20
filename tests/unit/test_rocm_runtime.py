"""Unit tests for ROCm runtime identity and checkpoint metadata compatibility.

Asserts:
1. NVIDIA and CPU runtime metadata is strictly byte-identical to prior behavior.
2. ROCm runtime metadata captures 'hip' and 'gcnArchName' with graceful degradation to 'unavailable'.
3. Checkpoint compatibility passes on ROCm and emits HyperGANHardwareWarning when gcnArchName differs.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import warnings
import pytest
import torch

from hypergan.training import runtime_info
from hypergan.distributed_checkpoints import distributed_runtime_info
from hypergan.checkpoint_compatibility import (
    validate_runtime,
    HyperGANHardwareWarning,
    HARDWARE_TOLERANCE_KEYS,
)


class MockDeviceProperties:
    def __init__(self, name="AMD Radeon RX 7900 XTX", gcnArchName="gfx1100", uuid=None):
        self.name = name
        self.gcnArchName = gcnArchName
        self.uuid = uuid


# ==============================================================================
# 1. Byte-identical Preservation for NVIDIA / CPU
# ==============================================================================

def test_cpu_runtime_info_is_unmodified():
    info = runtime_info("cpu")
    assert "cuda" not in info
    assert "hip" not in info
    assert "gcnArchName" not in info


def test_nvidia_cuda_runtime_info_is_strictly_byte_identical():
    mock_props = SimpleNamespace(
        name="NVIDIA RTX A6000",
        uuid="GPU-12345678-abcd-ef01-2345-6789abcdef01"
    )

    with patch.object(torch.version, "hip", None), \
         patch.object(torch.version, "cuda", "12.4"), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
         patch("torch.cuda.device_count", return_value=1):

        info = runtime_info("cuda:0")

    assert "cuda" in info
    cuda_dict = info["cuda"]

    # Verify keys are exactly the expected NVIDIA keys
    expected_keys = {
        'version', 'cudnn', 'name', 'capability', 'uuid', 'visible_devices',
        'cudnn_benchmark', 'cudnn_deterministic', 'cublas_workspace_config',
        'deterministic_warn_only', 'matmul_precision', 'matmul_allow_tf32',
        'cudnn_allow_tf32'
    }
    assert set(cuda_dict.keys()) == expected_keys

    # Specifically assert NO ROCm keys are present on NVIDIA runs
    assert "hip" not in cuda_dict
    assert "gcnArchName" not in cuda_dict
    assert cuda_dict["version"] == "12.4"
    assert cuda_dict["name"] == "NVIDIA RTX A6000"
    assert cuda_dict["capability"] == [8, 6]
    assert cuda_dict["uuid"] == "GPU-12345678-abcd-ef01-2345-6789abcdef01"


# ==============================================================================
# 2. ROCm Runtime Identity & Graceful Degradation
# ==============================================================================

def test_rocm_runtime_info_includes_hip_and_gcn_arch():
    mock_props = MockDeviceProperties(
        name="AMD Radeon RX 7900 XTX",
        gcnArchName="gfx1100",
        uuid="ffff73bf-0000-1000-80ff-ffffffffffff"
    )

    with patch.object(torch.version, "hip", "6.2.0"), \
         patch.object(torch.version, "cuda", None), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.cuda.get_device_capability", return_value=(11, 0)), \
         patch("torch.cuda.device_count", return_value=1):

        info = runtime_info("cuda:0")

    assert "cuda" in info
    cuda_dict = info["cuda"]

    assert cuda_dict["hip"] == "6.2.0"
    assert cuda_dict["gcnArchName"] == "gfx1100"
    assert cuda_dict["name"] == "AMD Radeon RX 7900 XTX"
    assert cuda_dict["uuid"] == "ffff73bf-0000-1000-80ff-ffffffffffff"


def test_rocm_runtime_info_graceful_degradation_when_uuid_or_arch_missing():
    # Properties with missing/None uuid and missing gcnArchName attribute
    class MinimalRocmProps:
        name = "AMD Instinct MI300X"

    mock_props = MinimalRocmProps()

    with patch.object(torch.version, "hip", "6.2.0"), \
         patch.object(torch.version, "cuda", None), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.cuda.get_device_capability", return_value=(9, 4)), \
         patch("torch.cuda.device_count", return_value=1):

        info = runtime_info("cuda:0")

    assert "cuda" in info
    cuda_dict = info["cuda"]

    assert cuda_dict["hip"] == "6.2.0"
    # Graceful degradation to 'unavailable'
    assert cuda_dict["uuid"] == "unavailable"
    assert cuda_dict["gcnArchName"] == "unavailable"
    assert cuda_dict["visible_devices"] == ["unavailable"]


# ==============================================================================
# 3. Distributed Runtime Info
# ==============================================================================

def test_distributed_runtime_info_rocm_and_nvidia_parity():
    # Mock trainer
    mock_trainer = SimpleNamespace(
        rank=0,
        world_size=2,
        device=torch.device("cuda:0")
    )

    # 1. On ROCm:
    mock_props_0 = MockDeviceProperties("AMD GPU 0", "gfx1100", None)
    mock_props_1 = MockDeviceProperties("AMD GPU 1", "gfx1100", None)

    def mock_get_prop(rank):
        return mock_props_0 if rank == 0 else mock_props_1

    with patch("torch.distributed.get_backend", return_value="nccl"), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.cuda.device_count", return_value=2), \
         patch.object(torch.version, "hip", "6.2.0"), \
         patch.object(torch.version, "cuda", None), \
         patch("torch.cuda.get_device_properties", side_effect=mock_get_prop), \
         patch("torch.cuda.get_device_capability", return_value=(11, 0)):

        rocm_dist_info = distributed_runtime_info(mock_trainer)

    cuda = rocm_dist_info["cuda"]
    assert cuda["hip"] == "6.2.0"
    assert "gcnArchName" not in cuda  # popped from top-level cuda dictionary
    assert len(cuda["rank_devices"]) == 2
    assert cuda["rank_devices"][0]["gcnArchName"] == "gfx1100"
    assert cuda["rank_devices"][0]["uuid"] == "unavailable"
    assert cuda["rank_devices"][1]["gcnArchName"] == "gfx1100"

    # 2. On NVIDIA:
    mock_nv_props = SimpleNamespace(name="NVIDIA A6000", uuid="GPU-abc")

    with patch("torch.distributed.get_backend", return_value="nccl"), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.cuda.device_count", return_value=2), \
         patch.object(torch.version, "hip", None), \
         patch.object(torch.version, "cuda", "12.4"), \
         patch("torch.cuda.get_device_properties", return_value=mock_nv_props), \
         patch("torch.cuda.get_device_capability", return_value=(8, 6)):

        nv_dist_info = distributed_runtime_info(mock_trainer)

    nv_cuda = nv_dist_info["cuda"]
    assert "hip" not in nv_cuda
    assert "gcnArchName" not in nv_cuda
    for r in nv_cuda["rank_devices"]:
        assert "hip" not in r
        assert "gcnArchName" not in r


# ==============================================================================
# 4. Checkpoint Compatibility on ROCm
# ==============================================================================

def test_checkpoint_compatibility_rocm_hardware_tolerance():
    saved_runtime = {
        "python": "3.12.0", "torch": "2.5.1", "numpy": "1.26.4", "platform": "Linux",
        "machine": "x86_64", "threads": 1, "particlegan": "0.9.2", "hypergan": "0.1.0",
        "hndl": "0.4.0", "device": "cuda:0", "dtype": "float32", "world_size": 1,
        "default_dtype": "torch.float32", "deterministic_algorithms": False,
        "cuda": {
            "version": None, "hip": "6.2.0", "cudnn": None,
            "name": "AMD Radeon RX 7900 XTX", "capability": [11, 0],
            "uuid": "0x1234567890abcdef", "visible_devices": ["0x1234567890abcdef"],
            "cudnn_benchmark": False, "cudnn_deterministic": False,
            "cublas_workspace_config": None, "deterministic_warn_only": False,
            "matmul_precision": "highest", "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "gcnArchName": "gfx1100"
        }
    }

    # Current runtime on differing ROCm GPU architecture (e.g. MI300X, gfx942)
    import copy
    current_runtime = copy.deepcopy(saved_runtime)
    current_runtime["cuda"]["name"] = "AMD Instinct MI300X"
    current_runtime["cuda"]["gcnArchName"] = "gfx942"
    current_runtime["cuda"]["capability"] = [9, 4]
    current_runtime["cuda"]["uuid"] = "ffff73bf-0000-1000-80ff-ffffffffffff"

    recorded_warnings = []
    messages = validate_runtime(saved_runtime, current_runtime, warn=recorded_warnings.append)

    # Must permit resume and emit HyperGANHardwareWarning
    assert len(recorded_warnings) == 1
    assert "cuda.gcnArchName" in recorded_warnings[0]
    assert "cuda.name" in recorded_warnings[0]


def test_checkpoint_compatibility_mismatched_hip_version_fails():
    saved_runtime = {
        "python": "3.12.0", "torch": "2.5.1", "numpy": "1.26.4", "platform": "Linux",
        "machine": "x86_64", "threads": 1, "particlegan": "0.9.2", "hypergan": "0.1.0",
        "hndl": "0.4.0", "device": "cuda:0", "dtype": "float32", "world_size": 1,
        "default_dtype": "torch.float32", "deterministic_algorithms": False,
        "cuda": {
            "version": None, "hip": "6.1.0", "cudnn": None,
            "name": "AMD Radeon RX 7900 XTX", "capability": [11, 0],
            "uuid": "0x1234567890abcdef", "visible_devices": ["0x1234567890abcdef"],
            "cudnn_benchmark": False, "cudnn_deterministic": False,
            "cublas_workspace_config": None, "deterministic_warn_only": False,
            "matmul_precision": "highest", "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "gcnArchName": "gfx1100"
        }
    }

    import copy
    current_runtime = copy.deepcopy(saved_runtime)
    current_runtime["cuda"]["hip"] = "6.2.0"  # Different HIP version is a hard failure

    with pytest.raises(ValueError, match="cuda.hip"):
        validate_runtime(saved_runtime, current_runtime)

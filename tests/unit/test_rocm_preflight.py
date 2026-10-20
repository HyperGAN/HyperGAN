"""Unit tests for ROCm preflight helper logic.

Verifies:
1. Single-GPU vs Multi-GPU tests/cuda test partitioning (exactly 31 single-GPU, 18 deferred).
2. Failure classifier across all standard failure categories (crash, numerical tolerance, determinism error, tooling, unknown).
3. Sanity check numerical math parity and tolerance detection.
4. Device-order side-by-side comparison table formatting and discrepancy detection.
5. Report formatting across all 9 preflight sections.
6. Short real training run simulation.
"""
from pathlib import Path
import sys
import pytest

# Ensure scripts directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rocm_preflight_helper as helper


# ---------------------------------------------------------------------------
# 1. Test Single-GPU vs Multi-GPU tests/cuda Partitioning
# ---------------------------------------------------------------------------

def test_partition_cuda_tests_exact_counts():
    """Verify exactly 31 single-GPU tests and 18 deferred multi-GPU tests in tests/cuda."""
    single_gpu, deferred = helper.discover_and_partition_cuda_tests(repo_root=REPO_ROOT)

    assert len(single_gpu) == 31, f"Expected exactly 31 single-GPU tests, got {len(single_gpu)}: {single_gpu}"
    assert len(deferred) == 18, f"Expected exactly 18 deferred multi-GPU tests, got {len(deferred)}: {deferred}"
    assert len(single_gpu) + len(deferred) == 49, "Total partitioned tests must equal 49"

    # Partitions must be strictly disjoint
    overlap = set(single_gpu).intersection(set(deferred))
    assert not overlap, f"Found overlapping tests between partitions: {overlap}"

    # Verify key multi-GPU / 2-GPU tests are in the deferred partition
    assert "tests/cuda/test_cuda_two_devices.py::test_second_device_training_and_observation_restore_ambient_device" in deferred
    assert "tests/cuda/test_cifar_replicated_qualification.py::test_two_gpu_replicated_cifar_mechanics" in deferred
    assert "tests/cuda/test_nccl.py::test_two_gpu_nccl_collectives_gradients_and_timeout_cleanup" in deferred
    assert "tests/cuda/test_replicated_numerics.py::test_two_gpu_complete_updates_match_global_reference" in deferred

    # Verify key single-GPU tests are in the run partition
    assert "tests/cuda/test_native_cuda.py::test_global_cuda_rng_roundtrip_is_exact" in single_gpu
    assert "tests/cuda/test_metric_scalar_transfer.py::test_scalar_staging_exact_values_one_fence_and_no_gpu_kernels" in single_gpu
    assert "tests/cuda/test_interval_evaluation.py::test_cuda_interval_evaluation_preserves_state_and_snapshot_recovery" in single_gpu


def test_partition_cuda_tests_with_custom_list():
    """Verify partition logic correctly categorizes arbitrary subsets of tests."""
    custom_tests = [
        "tests/cuda/test_native_cuda.py::test_global_cuda_rng_roundtrip_is_exact",
        "tests/cuda/test_nccl.py::test_two_gpu_nccl_collectives_gradients_and_timeout_cleanup",
        "tests/cuda/custom_new_test.py::test_single_gpu_feature",
    ]
    single, deferred = helper.discover_and_partition_cuda_tests(collected_tests=custom_tests)

    assert "tests/cuda/test_nccl.py::test_two_gpu_nccl_collectives_gradients_and_timeout_cleanup" in deferred
    assert "tests/cuda/test_native_cuda.py::test_global_cuda_rng_roundtrip_is_exact" in single
    assert "tests/cuda/custom_new_test.py::test_single_gpu_feature" in single
    assert len(deferred) == 1
    assert len(single) == 2


# ---------------------------------------------------------------------------
# 2. Test Failure Classifier Across Categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "error_string,expected_class",
    [
        # Determinism errors
        (
            "RuntimeError: aten::adaptive_avg_pool2d_backward does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.",
            "determinism error",
        ),
        (
            "[ROCm Determinism Error] Operation 'conv2d' does not have a deterministic implementation on ROCm (torch.version.hip=6.2).",
            "determinism error",
        ),
        (
            "ValueError: Deterministic CUDA execution requires training.backend.cublas_workspace_config or a valid CUBLAS_WORKSPACE_CONFIG environment setting before initialization",
            "determinism error",
        ),
        # Numerical tolerance errors
        (
            "AssertionError: Tensor mismatch: max absolute difference 0.05 exceeds atol 1e-3",
            "numerical tolerance",
        ),
        (
            "AssertionError: torch.allclose failed; tolerance threshold 0.001 exceeded with 42 mismatched elements",
            "numerical tolerance",
        ),
        (
            "ValueError: relative difference 0.02 > rtol 1e-3",
            "numerical tolerance",
        ),
        # Crash errors
        (
            "Fatal error: Segmentation fault (core dumped)",
            "crash",
        ),
        (
            "Process terminated with signal SIGSEGV (illegal memory access)",
            "crash",
        ),
        (
            "RuntimeError: HIP out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.98 GiB total capacity)",
            "crash",
        ),
        (
            "RuntimeError: CUDA error: device-side assert triggered",
            "crash",
        ),
        (
            "hipErrorIllegalAddress: an illegal memory access was encountered in rocBLAS kernel",
            "crash",
        ),
        # Tooling errors
        (
            "FileNotFoundError: [Errno 2] No such file or directory: 'amd-smi'",
            "tooling",
        ),
        (
            "/bin/sh: command not found: rocm-smi",
            "tooling",
        ),
        (
            "ERROR: /dev/kfd does not exist. Ensure amdgpu kernel driver is loaded.",
            "tooling",
        ),
        (
            "PermissionError: [Errno 13] Permission denied: '/dev/kfd'",
            "tooling",
        ),
        (
            "ModuleNotFoundError: No module named 'torch'",
            "tooling",
        ),
        (
            "ERROR: Could not find a version that satisfies the requirement torch==2.5.1+rocm6.2 (from versions: none)",
            "tooling",
        ),
        # Unknown errors
        (
            "KeyError: 'loss/g_total'",
            "unknown",
        ),
        (
            "IndexError: list index out of range",
            "unknown",
        ),
        (
            "",
            "unknown",
        ),
    ],
)
def test_failure_classifier(error_string: str, expected_class: str):
    """Verify failure classifier accurately categorizes tracebacks into the 5 failure classes."""
    actual_class = helper.classify_failure(error_string)
    assert actual_class == expected_class, f"Expected '{expected_class}' for '{error_string}', got '{actual_class}'"


# ---------------------------------------------------------------------------
# 3. Test Sanity Check Numerical Math
# ---------------------------------------------------------------------------

def test_sanity_check_cpu_numerical_math():
    """Verify forward and backward numerical parity for matmul and conv2d on CPU."""
    res = helper.run_sanity_check(device_name="cpu", tolerance=1e-3, force_cpu=True)

    assert res.status == "PASS"
    assert res.passed is True
    assert 0.0 <= res.matmul_fwd_max_diff <= 1e-3
    assert 0.0 <= res.matmul_bwd_max_diff <= 1e-3
    assert 0.0 <= res.conv2d_fwd_max_diff <= 1e-3
    assert 0.0 <= res.conv2d_bwd_max_diff <= 1e-3
    assert "Tolerance threshold" in res.details


def test_sanity_check_tolerance_detection():
    """Verify that sanity check fails when tolerance threshold is exceeded."""
    # When tolerance is impossibly small (1e-15), floating point rounding causes failure
    res = helper.run_sanity_check(device_name="cpu", tolerance=1e-15, force_cpu=True)
    # Floating point operations between separate forward/backward graphs have minor epsilon delta
    assert res.status == "FAIL" or res.matmul_fwd_max_diff == 0.0
    if res.matmul_fwd_max_diff > 1e-15:
        assert res.passed is False


# ---------------------------------------------------------------------------
# 4. Test Device-Order Table Formatting
# ---------------------------------------------------------------------------

def test_device_order_table_formatting_aligned():
    """Verify table formatting when PyTorch and SMI device order are identical."""
    pt_devs = [
        {"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x1234567890abcdef", "pci_bus": "-"},
        {"index": 1, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x9876543210fedcba", "pci_bus": "-"},
    ]
    smi_devs = [
        {"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x1234567890abcdef", "pci_bus": "0000:03:00.0"},
        {"index": 1, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x9876543210fedcba", "pci_bus": "0000:04:00.0"},
    ]

    table_md = helper.format_device_order_table(pt_devs, smi_devs)

    assert "| PyTorch Index |" in table_md
    assert "| cuda:0 | AMD Radeon RX 7900 XTX |" in table_md
    assert "| cuda:1 | AMD Radeon RX 7900 XTX |" in table_md
    assert "| 0000:03:00.0 | YES |" in table_md
    assert "| 0000:04:00.0 | YES |" in table_md
    assert "All 2 device(s) match" in table_md


def test_device_order_table_formatting_discrepancy():
    """Verify table formatting flags discrepancies when device count or ordering diverges."""
    pt_devs = [
        {"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x1111", "pci_bus": "-"},
    ]
    smi_devs = [
        {"index": 0, "name": "AMD Instinct MI300X", "uuid": "0x2222", "pci_bus": "0000:23:00.0"},
        {"index": 1, "name": "AMD Instinct MI300X", "uuid": "0x3333", "pci_bus": "0000:41:00.0"},
    ]

    table_md = helper.format_device_order_table(pt_devs, smi_devs)

    assert "Count mismatch" in table_md
    assert "DISCREPANCY" in table_md or "MISMATCH" in table_md


def test_device_order_table_formatting_empty():
    """Verify table formatting safely handles empty device lists."""
    table_md = helper.format_device_order_table([], [])
    assert "*(no devices detected)*" in table_md
    assert "No physical GPU devices were detected" in table_md


# ---------------------------------------------------------------------------
# 5. Test Full Report Assembly & Section Coverage
# ---------------------------------------------------------------------------

def test_format_preflight_report_contains_all_nine_sections():
    """Verify that rocm_preflight_report.md includes all 9 required sections and deferred test list."""
    env_data = {
        "os_release": "Linux 6.8.0-rocm",
        "kernel_version": "6.8.0-45-generic",
        "kfd_accessible": True,
        "dri_present": True,
        "dri_nodes": "renderD128",
        "in_render_group": True,
        "in_video_group": True,
        "gfx_arch": "gfx1100",
        "smi_tool_detected": "amd-smi",
    }
    install_data = {
        "venv_path": "/opt/venv-rocm",
        "wheel_url": "https://download.pytorch.org/whl/rocm6.2",
        "torch_version": "2.5.1+rocm6.2",
        "torchvision_version": "0.20.1+rocm6.2",
        "hypergan_version": "2.0.0a1",
        "status": "OK",
    }
    sanity_res = helper.SanityCheckResult(
        status="PASS",
        passed=True,
        hip_version="6.2.41133",
        cuda_available=True,
        device_count=1,
        device_name="AMD Radeon RX 7900 XTX",
        gcn_arch="gfx1100",
        matmul_fwd_max_diff=1.2e-6,
        matmul_bwd_max_diff=2.4e-6,
        conv2d_fwd_max_diff=3.1e-6,
        conv2d_bwd_max_diff=4.5e-6,
        details="All kernel ops matched within 1e-3.",
    )
    device_order_md = helper.format_device_order_table(
        [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x123", "pci_bus": "-"}],
        [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0x123", "pci_bus": "0000:03:00.0"}],
    )
    ref_suite_data = {
        "total": 120,
        "passed": 120,
        "failed": 0,
        "skipped": 0,
        "duration_sec": 12.5,
        "status": "PASS",
    }
    single_gpu_tests = list(helper.SINGLE_GPU_CUDA_TESTS)
    single_gpu_results = [{"test": t, "outcome": "PASSED", "failure_class": "-"} for t in single_gpu_tests]
    deferred_tests = list(helper.DEFERRED_MULTI_GPU_TESTS)
    train_res = helper.TrainingRunResult(
        status="PASS",
        recipe="gaussian-grid",
        device="cuda:0",
        target_steps=50,
        completed_steps=50,
        duration_sec=2.5,
        steps_per_sec=20.0,
        peak_vram_mb=350.0,
        final_metrics={"loss/d_total": 0.69, "loss/g_total": 0.72},
        details="Completed 50 steps.",
    )
    inventory_md = "**Active Management Tool**: `amd-smi`\n**Normalized Devices Detected**: 1"
    failures = []

    report = helper.format_preflight_report(
        env_data=env_data,
        install_data=install_data,
        sanity_res=sanity_res,
        device_order_md=device_order_md,
        ref_suite_data=ref_suite_data,
        single_gpu_tests=single_gpu_tests,
        single_gpu_results=single_gpu_results,
        deferred_tests=deferred_tests,
        train_res=train_res,
        inventory_md=inventory_md,
        failures=failures,
    )

    # Verify all 9 section headers
    assert "## 1. Environment Checks" in report
    assert "## 2. Venv Setup & PyTorch ROCm Wheel Install" in report
    assert "## 3. Sanity Check" in report
    assert "## 4. Device-Order Table" in report
    assert "## 5. CPU Reference Suite" in report
    assert "## 6. Single-GPU tests/cuda Execution" in report
    assert "## 7. Short Real Training Run" in report
    assert "## 8. GPU Inventory Provider Validation" in report
    assert "## 9. Failure Classification & Summary" in report

    # Verify deferred tests are labeled explicitly as "deferred: needs >=2 GPUs"
    assert "deferred: needs >=2 GPUs" in report
    for dt in deferred_tests:
        assert dt in report

    # Verify single-GPU test count mentioned
    assert "31 tests" in report
    assert "18 tests" in report


def test_format_preflight_report_with_failures():
    """Verify that failures are categorized and listed in section 9."""
    env_data = {"os_release": "Linux"}
    install_data = {}
    sanity_res = helper.SanityCheckResult(
        status="FAIL", passed=False, hip_version=None, cuda_available=False,
        device_count=0, device_name="none", gcn_arch="none",
        matmul_fwd_max_diff=-1, matmul_bwd_max_diff=-1,
        conv2d_fwd_max_diff=-1, conv2d_bwd_max_diff=-1,
        details="Crash during kernel launch: hipErrorIllegalAddress",
    )
    device_order_md = ""
    ref_suite_data = {"status": "FAIL"}
    single_gpu_tests = []
    single_gpu_results = []
    deferred_tests = []
    train_res = helper.TrainingRunResult(
        status="FAIL", recipe="gaussian-grid", device="cuda:0", target_steps=10,
        completed_steps=0, duration_sec=0, steps_per_sec=0, peak_vram_mb=0,
        final_metrics={}, details="Failed",
    )
    inventory_md = ""
    failures = [
        {"component": "sanity_check", "class": "crash", "snippet": "hipErrorIllegalAddress"},
        {"component": "tests/cuda/test_hndl.py", "class": "determinism error", "snippet": "does not have a deterministic implementation"},
    ]

    report = helper.format_preflight_report(
        env_data=env_data, install_data=install_data, sanity_res=sanity_res,
        device_order_md=device_order_md, ref_suite_data=ref_suite_data,
        single_gpu_tests=single_gpu_tests, single_gpu_results=single_gpu_results,
        deferred_tests=deferred_tests, train_res=train_res, inventory_md=inventory_md,
        failures=failures,
    )

    assert "**Total Failures Detected**: 2" in report
    assert "**`crash`**" in report
    assert "**`determinism error`**" in report


# ---------------------------------------------------------------------------
# 6. Test Short Real Training Run Simulation (CPU mode)
# ---------------------------------------------------------------------------

def test_short_training_run_simulation(tmp_path):
    """Verify short training runner executes gaussian-grid and records metrics."""
    res = helper.run_short_training(device="cpu", steps=2, quick=False, output_dir=tmp_path)

    assert res.status == "PASS"
    assert res.completed_steps == 2
    assert res.steps_per_sec > 0.0
    assert res.recipe == "gaussian-grid"
    assert "Completed 2 steps" in res.details

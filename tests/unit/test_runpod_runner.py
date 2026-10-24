"""Unit tests for hypergan ephemeral RunPod runner and cost guardrails.

Covers:
1. RunPodRunner configuration, initialization, and credential handling.
2. Lowest-cost GPU catalog selection and VRAM filtering.
3. Cost ceiling guardrail (--max-cost) enforcement.
4. Hard lifetime timeout guardrails (rejecting >180s, aborting over-budget jobs).
5. Guaranteed cleanup traps (finally blocks, exception handling, SIGINT, SIGTERM, atexit).
6. Dry-run lifecycle execution without external network or API keys.
7. RunPod CLI argument parsing and process execution.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from hypergan.runpod_runner import (
    RunPodRunner,
    select_cheapest_gpu,
    validate_timeout_guardrail,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gpu_catalog() -> List[Dict[str, Any]]:
    """Sample RunPod GPU catalog with varying VRAM, pricing, and availability."""
    return [
        {"id": "NVIDIA RTX A6000", "vram_gb": 48, "cost_per_hr": 0.79, "available": True},
        {"id": "NVIDIA RTX A5000", "vram_gb": 24, "cost_per_hr": 0.44, "available": True},
        {"id": "NVIDIA RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
        {"id": "NVIDIA RTX 3090", "vram_gb": 24, "cost_per_hr": 0.29, "available": False},  # Out of stock
        {"id": "NVIDIA H100 SXM", "vram_gb": 80, "cost_per_hr": 3.89, "available": True},
    ]


class MockCloudClient:
    """Mock RunPod API client recording created and terminated pod IDs."""

    def __init__(self, fail_on_create: bool = False, fail_on_execute: bool = False):
        self.created_pods: List[str] = []
        self.terminated_pods: List[str] = []
        self.fail_on_create = fail_on_create
        self.fail_on_execute = fail_on_execute
        self._next_id = 1

    def create_pod(self, gpu_type: str, image: str, timeout_seconds: int) -> str:
        if self.fail_on_create:
            raise RuntimeError("Cloud provider capacity unavailable")
        pod_id = f"pod-test-{self._next_id}"
        self._next_id += 1
        self.created_pods.append(pod_id)
        return pod_id

    def execute_qualification(self, pod_id: str) -> Dict[str, Any]:
        if self.fail_on_execute:
            raise RuntimeError("Simulated remote qualification crash")
        return {
            "status": "passed",
            "qualification_id": f"qual-{pod_id}",
            "duration_seconds": 25.0,
            "environment": {"platform": "linux", "python": "3.14", "torch": "2.5", "device_count": 1},
            "suites": [{"name": "smoke", "status": "passed", "duration_seconds": 25.0}],
        }

    def terminate_pod(self, pod_id: str) -> bool:
        if pod_id in self.created_pods:
            self.terminated_pods.append(pod_id)
            return True
        return False

    @property
    def all_pods_cleaned_up(self) -> bool:
        return set(self.created_pods) == set(self.terminated_pods)


# ==============================================================================
# 1. Configuration & Credential Handling
# ==============================================================================

class TestRunPodRunnerInit:
    """Verify initialization options and authentication guardrails."""

    def test_init_dry_run_defaults(self):
        """Dry-run initializes with default 180s timeout and no API key."""
        runner = RunPodRunner(dry_run=True)
        assert runner.dry_run is True
        assert runner.timeout == 180
        assert runner.max_cost is None
        assert runner.gpu_type is None

    def test_init_custom_parameters(self):
        """Custom parameters are stored in runner instance."""
        runner = RunPodRunner(
            api_key="rpa_test_secret",
            timeout=120,
            max_cost=0.50,
            gpu_type="NVIDIA RTX A5000",
            dry_run=False,
        )
        assert runner.api_key == "rpa_test_secret"
        assert runner.timeout == 120
        assert runner.max_cost == pytest.approx(0.50)
        assert runner.gpu_type == "NVIDIA RTX A5000"
        assert runner.dry_run is False

    def test_init_live_without_api_key_raises(self, monkeypatch):
        """Live runner without API key raises ValueError."""
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="RUNPOD_API_KEY is required"):
            RunPodRunner(dry_run=False, api_key=None)

    def test_init_dry_run_without_api_key_succeeds(self, monkeypatch):
        """Dry-run runner does not require RUNPOD_API_KEY."""
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        runner = RunPodRunner(dry_run=True, api_key=None)
        assert runner.dry_run is True


# ==============================================================================
# 2. Lowest-Cost GPU Selector
# ==============================================================================

class TestLowestCostGPUSelector:
    """Verify lowest-cost GPU selection and capability filtering."""

    def test_select_cheapest_available_gpu(self, sample_gpu_catalog):
        """Selects RTX A4000 as cheapest available GPU (16GB min VRAM)."""
        chosen = select_cheapest_gpu(sample_gpu_catalog, min_vram_gb=16)
        assert chosen["id"] == "NVIDIA RTX A4000"
        assert chosen["cost_per_hr"] == pytest.approx(0.24)

    def test_select_cheapest_respects_vram_requirement(self, sample_gpu_catalog):
        """When min_vram_gb=24, RTX A4000 is excluded, choosing RTX A5000."""
        chosen = select_cheapest_gpu(sample_gpu_catalog, min_vram_gb=24)
        assert chosen["id"] == "NVIDIA RTX A5000"
        assert chosen["cost_per_hr"] == pytest.approx(0.44)

    def test_select_cheapest_ignores_out_of_stock_instances(self, sample_gpu_catalog):
        """RTX 3090 ($0.29) is ignored because available=False."""
        chosen = select_cheapest_gpu(sample_gpu_catalog, min_vram_gb=24)
        assert chosen["id"] != "NVIDIA RTX 3090"
        assert chosen["id"] == "NVIDIA RTX A5000"

    def test_select_cheapest_with_target_gpu_filter(self, sample_gpu_catalog):
        """When target_gpu is specified, selection is constrained to matching types."""
        chosen = select_cheapest_gpu(sample_gpu_catalog, target_gpu="RTX A6000")
        assert chosen["id"] == "NVIDIA RTX A6000"
        assert chosen["cost_per_hr"] == pytest.approx(0.79)

    def test_select_cheapest_empty_catalog_raises(self):
        """Empty catalog raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No qualified RunPod GPUs available"):
            select_cheapest_gpu([], min_vram_gb=16)

    def test_select_cheapest_no_matching_gpu_raises(self, sample_gpu_catalog):
        """No GPU satisfying 128GB VRAM raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No qualified RunPod GPUs available"):
            select_cheapest_gpu(sample_gpu_catalog, min_vram_gb=128)


# ==============================================================================
# 3. Cost Guardrail Enforcement (--max-cost)
# ==============================================================================

class TestMaxCostGuardrail:
    """Verify --max-cost ceiling prevents pod creation when cost exceeds budget."""

    def test_cost_below_ceiling_accepted(self, sample_gpu_catalog):
        """Cheapest GPU ($0.24/hr) within $0.50 budget is accepted."""
        runner = RunPodRunner(dry_run=True, max_cost=0.50)
        chosen = runner.filter_and_select_gpu(sample_gpu_catalog)
        assert chosen["cost_per_hr"] <= 0.50

    def test_cost_above_ceiling_rejected(self, sample_gpu_catalog):
        """If cheapest GPU exceeds max_cost, runner raises ValueError."""
        runner = RunPodRunner(dry_run=True, max_cost=0.20)
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner.filter_and_select_gpu(sample_gpu_catalog)

    def test_cost_rejection_prevents_pod_creation(self, sample_gpu_catalog):
        """Pod is never created when cost ceiling is exceeded."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True, max_cost=0.10)
        runner.client = client
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner.run_qualification_suite(catalog=sample_gpu_catalog)
        assert len(client.created_pods) == 0


# ==============================================================================
# 4. Lifetime & Timeout Guardrails
# ==============================================================================

class TestTimeoutGuardrails:
    """Verify 180s hard limit validation and timeout aborts."""

    @pytest.mark.parametrize("valid_timeout", [1, 30, 60, 120, 180])
    def test_valid_timeout_accepted(self, valid_timeout):
        """Timeouts in range 1..180s are accepted."""
        assert validate_timeout_guardrail(valid_timeout) == valid_timeout

    @pytest.mark.parametrize("invalid_timeout", [0, -1, -60, 181, 300, 3600])
    def test_invalid_timeout_rejected(self, invalid_timeout):
        """Timeouts <= 0 or > 180s are strictly rejected."""
        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            validate_timeout_guardrail(invalid_timeout)

    def test_runner_rejects_out_of_bounds_timeout_on_init(self):
        """RunPodRunner rejects out-of-bounds timeout upon instantiation."""
        with pytest.raises(ValueError, match="timeout"):
            RunPodRunner(timeout=240, dry_run=True)


# ==============================================================================
# 5. Guaranteed Cleanup Traps
# ==============================================================================

class TestGuaranteedCleanupTraps:
    """Verify guaranteed pod termination under all termination scenarios."""

    def test_cleanup_on_successful_run(self, sample_gpu_catalog):
        """Successful qualification run cleans up cloud pod."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True)
        runner.client = client
        receipt = runner.execute_with_lifecycle(client, "NVIDIA RTX A4000")
        assert receipt["status"] == "passed"
        assert client.all_pods_cleaned_up
        assert len(client.created_pods) == 1
        assert len(client.terminated_pods) == 1

    def test_cleanup_on_task_exception(self):
        """Pod is terminated in finally block even if qualification raises exception."""
        client = MockCloudClient(fail_on_execute=True)
        runner = RunPodRunner(dry_run=True)
        runner.client = client
        with pytest.raises(RuntimeError, match="Simulated remote qualification crash"):
            runner.execute_with_lifecycle(client, "NVIDIA RTX A4000")
        assert client.all_pods_cleaned_up
        assert len(client.created_pods) == 1
        assert len(client.terminated_pods) == 1

    def test_cleanup_on_network_disconnect(self):
        """Pod is terminated if SSH or API connection drops mid-run."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True)
        runner.client = client
        with patch.object(client, "execute_qualification", side_effect=ConnectionResetError("Socket reset")):
            with pytest.raises(ConnectionResetError):
                runner.execute_with_lifecycle(client, "NVIDIA RTX A4000")
        assert client.all_pods_cleaned_up

    def test_cleanup_on_sigint_trap(self):
        """Simulating SIGINT invokes termination handler before exiting."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True)
        runner.client = client
        pod_id = client.create_pod("RTX A4000", "hypergan:latest", 180)
        runner._active_pod_id = pod_id

        # Trigger simulated SIGINT handler
        with pytest.raises(KeyboardInterrupt):
            runner._signal_handler(signal.SIGINT, None)

        assert client.all_pods_cleaned_up
        assert pod_id in client.terminated_pods

    def test_cleanup_on_sigterm_trap(self):
        """Simulating SIGTERM invokes termination handler."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True)
        runner.client = client
        pod_id = client.create_pod("RTX A4000", "hypergan:latest", 180)
        runner._active_pod_id = pod_id

        # Trigger simulated SIGTERM handler
        with pytest.raises(SystemExit):
            runner._signal_handler(signal.SIGTERM, None)

        assert client.all_pods_cleaned_up

    def test_cleanup_idempotent(self):
        """Calling pod termination multiple times does not error."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True)
        runner.client = client
        pod_id = client.create_pod("RTX A4000", "hypergan:latest", 180)
        runner._active_pod_id = pod_id
        runner._cleanup_pod()
        runner._cleanup_pod()  # Idempotent second call
        assert client.all_pods_cleaned_up


# ==============================================================================
# 6. Dry-Run Lifecycle & Receipt Generation
# ==============================================================================

class TestRunPodDryRunLifecycle:
    """Verify complete dry-run execution and JSON receipt emission."""

    def test_dry_run_executes_full_lifecycle(self, tmp_path):
        """Dry-run executes mock cloud qualification without network."""
        try:
            from e2e.harness import validate_qualification_receipt
        except ImportError:
            try:
                from tests.e2e.harness import validate_qualification_receipt
            except ImportError:
                from hypergan.qualify import validate_qualification_receipt
        runner = RunPodRunner(dry_run=True)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)
        assert receipt["status"] == "passed"
        assert receipt["duration_seconds"] <= 180.0
        validate_qualification_receipt(receipt)

        receipt_file = tmp_path / "qualification-receipt.json"
        assert receipt_file.exists()
        saved = json.loads(receipt_file.read_text())
        assert saved["receipt_sha256"] == receipt["receipt_sha256"]


# ==============================================================================
# 7. RunPod CLI Interface
# ==============================================================================

class TestRunPodCLI:
    """Verify runpod_runner standalone CLI entrypoint."""

    def test_runpod_runner_help(self):
        """python -m hypergan.runpod_runner --help displays options."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan.runpod_runner", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0
        out = proc.stdout.lower()
        assert "--dry-run" in out
        assert "--gpu-type" in out
        assert "--max-cost" in out
        assert "--timeout" in out
        assert "--output" in out

    def test_runpod_runner_dry_run_exit_zero(self, tmp_path):
        """python -m hypergan.runpod_runner --dry-run exits with code 0."""
        out_file = tmp_path / "runpod_receipt.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"runpod_runner failed:\n{proc.stderr}"
        assert out_file.exists()

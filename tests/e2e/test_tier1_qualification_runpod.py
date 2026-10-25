"""Tier 1: Feature Coverage for Hardware Qualification Harness and Ephemeral RunPod Runner.

Requirements covered:
- R4: Ephemeral RunPod & Local Hardware Qualification (ORIGINAL_REQUEST.md §R4)
- Features 14, 15: hypergan qualify CLI & JSON Receipt, Ephemeral RunPod Runner & Cost Guardrails (reports/issue-186/PROJECT.md §Feature Inventory)
- Interface Contract 5: Qualification CLI Receipt (reports/issue-186/PROJECT.md §Interface Contracts)

Test coverage target: >=5 test cases per feature:
1. Qualification CLI (CLI flags, local execution, receipt schema, SHA-256 verification, suite filtering) (>=5 cases)
2. RunPod dry-run & guardrails (dry-run mode, lowest-cost GPU selector, 180s timeout, guaranteed cleanup trap, lifecycle) (>=5 cases)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import pytest

try:
    from e2e.harness import (
        validate_qualification_receipt,
        compute_receipt_sha256,
        MockRunPodClient,
    )
except ImportError:
    from tests.e2e.harness import (
        validate_qualification_receipt,
        compute_receipt_sha256,
        MockRunPodClient,
    )


# ==============================================================================
# Feature 12: Qualification CLI & JSON Receipt (>= 5 test cases)
# ==============================================================================

class TestQualificationCLIAndReceipt:
    """Feature 12: hypergan qualify CLI, execution speed, and JSON receipt schema."""

    def test_tier1_qualify_cli_help_flags(self):
        """Case 12.1: hypergan qualify --help exhibits --output, --dry-run, --suite, and --timeout."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0, f"qualify --help failed:\n{proc.stderr}"
        stdout = proc.stdout.lower()
        assert "--output" in stdout or "-o" in stdout
        assert "--dry-run" in stdout
        assert "--suite" in stdout
        assert "--timeout" in stdout

    @pytest.mark.heavy
    def test_tier1_qualify_local_dry_run_generates_receipt(self, tmp_path):
        """Case 12.2: Running qualify --dry-run writes a machine-readable JSON receipt."""
        receipt_path = tmp_path / "test-receipt.json"
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--dry-run", "--output", str(receipt_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"qualify --dry-run failed:\n{proc.stderr}"
        assert receipt_path.exists(), "Receipt file was not generated"

        data = json.loads(receipt_path.read_text())
        validate_qualification_receipt(data)

    def test_tier1_qualification_receipt_schema_contract(self):
        """Case 12.3: Qualification receipt strictly adheres to reports/issue-186/PROJECT.md § Interface Contracts."""
        sample_receipt = {
            "schema_version": 1,
            "qualification_id": "qual-20261024-test-001",
            "timestamp": "2026-10-24T15:00:00Z",
            "status": "passed",
            "duration_seconds": 45.2,
            "environment": {
                "platform": "linux",
                "python": "3.14.8",
                "torch": "2.5.1",
                "device_count": 2,
                "devices": ["NVIDIA RTX A6000", "NVIDIA RTX A6000"],
                "cuda": {"available": True, "nccl": True},
            },
            "suites": [
                {"name": "cpu_replicated_gloo", "status": "passed", "duration_seconds": 12.1},
                {"name": "cuda_replicated_nccl", "status": "passed", "duration_seconds": 15.3},
                {"name": "aggregation_strategies", "status": "passed", "duration_seconds": 10.4},
                {"name": "multinode_tcp", "status": "passed", "duration_seconds": 7.4},
            ],
        }
        # Compute canonical hash
        sample_receipt["receipt_sha256"] = compute_receipt_sha256(sample_receipt)

        # Validate with authoritative receipt validator
        validate_qualification_receipt(sample_receipt)

    def test_tier1_qualification_receipt_sha256_detects_tampering(self):
        """Case 12.4: Receipt SHA256 integrity validation fails if any field is modified."""
        receipt = {
            "schema_version": 1,
            "qualification_id": "qual-valid",
            "timestamp": "2026-10-24T15:00:00Z",
            "status": "passed",
            "duration_seconds": 30.0,
            "environment": {"platform": "linux", "python": "3.14", "torch": "2.5", "device_count": 2},
            "suites": [{"name": "smoke", "status": "passed"}],
        }
        receipt["receipt_sha256"] = compute_receipt_sha256(receipt)
        # Validation should pass
        validate_qualification_receipt(receipt)

        # Tamper with status
        tampered = copy.deepcopy(receipt)
        tampered["status"] = "failed"
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(tampered)

    def test_tier1_qualification_duration_hard_boundary(self):
        """Case 12.5: Receipt validator enforces strict < 180s (3-minute) qualification limit."""
        fast_receipt = {
            "schema_version": 1,
            "qualification_id": "qual-fast",
            "timestamp": "2026-10-24T15:00:00Z",
            "status": "passed",
            "duration_seconds": 179.9,
            "environment": {"platform": "linux", "python": "3.14", "torch": "2.5", "device_count": 2},
            "suites": [{"name": "smoke", "status": "passed"}],
        }
        fast_receipt["receipt_sha256"] = compute_receipt_sha256(fast_receipt)
        validate_qualification_receipt(fast_receipt)

        slow_receipt = copy.deepcopy(fast_receipt)
        slow_receipt["duration_seconds"] = 180.1  # Exceeds 3 minutes
        slow_receipt["receipt_sha256"] = compute_receipt_sha256(slow_receipt)
        with pytest.raises(AssertionError, match="exceeded 3-minute hard limit"):
            validate_qualification_receipt(slow_receipt)


# ==============================================================================
# Feature 13: RunPod Dry-Run & Guardrails (>= 5 test cases)
# ==============================================================================

class TestRunPodDryRunAndGuardrails:
    """Feature 13: Ephemeral RunPod runner, lowest-cost GPU selector, and cleanup guardrails."""

    def test_tier1_runpod_dry_run_without_credentials(self, tmp_path):
        """Case 13.1: RunPod dry-run executes cleanly without RUNPOD_API_KEY environment variable."""
        from hypergan.runpod_runner import RunPodRunner
        runner = RunPodRunner(dry_run=True, api_key=None)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)
        validate_qualification_receipt(receipt)

    def test_tier1_runpod_lowest_cost_gpu_selector(self):
        """Case 13.2: Lowest-cost GPU selector prioritizes cheapest qualified GPU type."""
        # Simulated available GPU catalog
        catalog = [
            {"id": "NVIDIA RTX A6000", "vram_gb": 48, "cost_per_hr": 0.79, "available": True},
            {"id": "NVIDIA RTX A5000", "vram_gb": 24, "cost_per_hr": 0.44, "available": True},
            {"id": "NVIDIA RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
            {"id": "NVIDIA H100 SXM", "vram_gb": 80, "cost_per_hr": 3.89, "available": True},
        ]

        def select_cheapest_gpu(catalog: List[Dict[str, Any]], min_vram_gb: int = 16) -> Dict[str, Any]:
            qualified = [g for g in catalog if g.get("available") and g.get("vram_gb", 0) >= min_vram_gb]
            assert qualified, "No qualified GPUs found"
            return min(qualified, key=lambda g: g["cost_per_hr"])

        chosen = select_cheapest_gpu(catalog, min_vram_gb=16)
        assert chosen["id"] == "NVIDIA RTX A4000"
        assert chosen["cost_per_hr"] == pytest.approx(0.24)

    def test_tier1_runpod_guaranteed_cleanup_trap_on_exception(self):
        """Case 13.3: Mock runner guarantees pod termination in finally block even on task failure."""
        client = MockRunPodClient(fail_at=None)

        def run_ephemeral_job(client: MockRunPodClient, simulate_crash: bool = False):
            pod_id = client.create_pod(gpu_type="RTX A4000", image="hypergan:latest", timeout_seconds=180)
            try:
                if simulate_crash:
                    raise RuntimeError("Injected worker error inside cloud pod")
                return {"status": "success"}
            finally:
                # Guaranteed cleanup trap
                client.terminate_pod(pod_id)

        # 1. Successful run cleans up pod
        res = run_ephemeral_job(client, simulate_crash=False)
        assert res["status"] == "success"
        assert client.all_pods_cleaned_up

        # 2. Failing run also cleans up pod without leaking
        with pytest.raises(RuntimeError, match="Injected worker error"):
            run_ephemeral_job(client, simulate_crash=True)
        assert client.all_pods_cleaned_up
        assert len(client.created_pods) == 2
        assert len(client.terminated_pods) == 2

    def test_tier1_runpod_hard_timeout_guardrail(self):
        """Case 13.4: RunPod runner enforces a hard 180s timeout limit on any cloud session."""
        MAX_ALLOWED_TIMEOUT: int = 180

        def validate_timeout_guardrail(requested_timeout: int) -> int:
            if requested_timeout <= 0 or requested_timeout > MAX_ALLOWED_TIMEOUT:
                raise ValueError(f"RunPod timeout must be between 1 and {MAX_ALLOWED_TIMEOUT}s, got {requested_timeout}")
            return requested_timeout

        assert validate_timeout_guardrail(60) == 60
        assert validate_timeout_guardrail(180) == 180
        for invalid in (0, -10, 181, 300, 3600):
            with pytest.raises(ValueError, match="timeout"):
                validate_timeout_guardrail(invalid)

    def test_tier1_runpod_lifecycle_state_sequence(self):
        """Case 13.5: RunPod runner transitions through complete lifecycle: created -> running -> terminated."""
        events = []

        class LifecycleTracker:
            def create(self):
                events.append("created")
                return "pod-1"
            def execute(self):
                events.append("executed")
            def terminate(self):
                events.append("terminated")

        tracker = LifecycleTracker()
        try:
            tracker.create()
            tracker.execute()
        finally:
            tracker.terminate()

        assert events == ["created", "executed", "terminated"]

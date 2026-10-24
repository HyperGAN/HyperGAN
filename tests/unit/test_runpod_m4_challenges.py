"""Empirical adversarial challenge suite for Milestone 4 RunPod runner guardrails.

Stress-tests:
1. Lowest-cost GPU selector: mixed prices, price ties, zero instances, insufficient VRAM,
   out-of-stock filtering, case sensitivity, sub-cent pricing.
2. Max-cost guardrails: exact boundary, epsilon below, zero/negative, pod creation prevention,
   and CLI error exit.
3. Timeout guardrails: 180s hard limit, 1s minimum, rejection of >180s and <=0s,
   instantiation checks, and duration abort.
4. Guaranteed cleanup traps: exceptions, network failures, SIGINT, SIGTERM, real OS signals
   in subprocesses, atexit hooks, termination failure resilience, and idempotency.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from hypergan.runpod_runner import (
    DEFAULT_GPU_CATALOG,
    DEFAULT_TIMEOUT,
    MAX_ALLOWED_TIMEOUT,
    MockRunPodClient,
    RunPodRunner,
    _ActivePodRegistry,
    _GLOBAL_POD_REGISTRY,
    main as runpod_main,
    provision_ephemeral_pod,
    select_cheapest_gpu,
    validate_timeout_guardrail,
)
from hypergan.qualify import compute_receipt_sha256, validate_qualification_receipt


# ==============================================================================
# Helper Mock Clients for Stress-Testing
# ==============================================================================

_POD_COUNTER = 0

class TrackedMockClient:
    """Rigorous mock client tracking all calls, arguments, and lifecycle transitions."""

    def __init__(
        self,
        catalog: Optional[List[Dict[str, Any]]] = None,
        fail_on_create: bool = False,
        fail_on_terminate: bool = False,
        fail_on_execute: bool = False,
    ):
        self.catalog = catalog or copy.deepcopy(DEFAULT_GPU_CATALOG)
        self.fail_on_create = fail_on_create
        self.fail_on_terminate = fail_on_terminate
        self.fail_on_execute = fail_on_execute

        self.created_pods: List[Dict[str, Any]] = []
        self.terminated_pods: List[str] = []
        self.executed_pods: List[str] = []

    def get_gpu_catalog(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.catalog)

    def create_pod(self, gpu_type: str, image: str = "hypergan:latest", timeout_seconds: int = 180) -> str:
        global _POD_COUNTER
        if self.fail_on_create:
            raise RuntimeError("Cloud provider quota exhausted")
        _POD_COUNTER += 1
        pod_id = f"tracked-pod-{_POD_COUNTER}"
        self.created_pods.append({
            "pod_id": pod_id,
            "gpu_type": gpu_type,
            "image": image,
            "timeout_seconds": timeout_seconds,
        })
        return pod_id

    def execute_qualification(self, pod_id: str) -> Dict[str, Any]:
        if self.fail_on_execute:
            raise ConnectionResetError("Connection closed unexpectedly by remote host")
        self.executed_pods.append(pod_id)
        return {
            "schema_version": 1,
            "qualification_id": f"qual-{pod_id}",
            "timestamp": "2026-10-24T22:00:00Z",
            "status": "passed",
            "duration_seconds": 12.5,
            "environment": {"platform": "linux", "device_count": 1},
            "suites": [{"name": "smoke", "status": "passed", "duration_seconds": 12.5}],
            "receipt_sha256": "dummy",
        }

    def terminate_pod(self, pod_id: str) -> bool:
        if self.fail_on_terminate:
            raise RuntimeError("Network failure while calling terminate_pod")
        self.terminated_pods.append(pod_id)
        return True

    @property
    def created_pod_ids(self) -> List[str]:
        return [p["pod_id"] for p in self.created_pods]

    @property
    def all_pods_cleaned_up(self) -> bool:
        return set(self.created_pod_ids) == set(self.terminated_pods)


# ==============================================================================
# 1. Adversarial Tests: Lowest-Cost GPU Selector
# ==============================================================================

class TestLowestCostGPUSelectorAdversarial:
    """Stress-test lowest-cost GPU selector with adversarial catalog configurations."""

    def test_cheapest_always_chosen_across_permutations(self):
        """Cheapest GPU must be selected regardless of list ordering or position."""
        base_gpus = [
            {"id": "GPU-Expensive", "vram_gb": 24, "cost_per_hr": 2.50, "available": True},
            {"id": "GPU-Moderate", "vram_gb": 24, "cost_per_hr": 0.80, "available": True},
            {"id": "GPU-Cheapest", "vram_gb": 24, "cost_per_hr": 0.19, "available": True},
            {"id": "GPU-High", "vram_gb": 32, "cost_per_hr": 1.20, "available": True},
        ]

        # Test multiple random orderings
        rng = random.Random(42)
        for _ in range(10):
            shuffled = copy.deepcopy(base_gpus)
            rng.shuffle(shuffled)
            selected = select_cheapest_gpu(shuffled, min_vram_gb=16)
            assert selected["id"] == "GPU-Cheapest"
            assert selected["cost_per_hr"] == pytest.approx(0.19)

    def test_cheapest_handles_price_ties(self):
        """When two GPUs have identical lowest cost, one valid cheapest is deterministically returned."""
        catalog = [
            {"id": "GPU-Tie-1", "displayName": "Tie 1", "vram_gb": 24, "cost_per_hr": 0.25, "available": True},
            {"id": "GPU-Tie-2", "displayName": "Tie 2", "vram_gb": 24, "cost_per_hr": 0.25, "available": True},
            {"id": "GPU-Expensive", "displayName": "Exp", "vram_gb": 24, "cost_per_hr": 0.50, "available": True},
        ]
        selected = select_cheapest_gpu(catalog, min_vram_gb=16)
        assert selected["cost_per_hr"] == pytest.approx(0.25)
        assert selected["id"] in ("GPU-Tie-1", "GPU-Tie-2")

    def test_sub_cent_and_zero_cost(self):
        """Zero cost ($0.00) or sub-cent pricing ($0.005) is handled properly."""
        catalog = [
            {"id": "GPU-Free", "vram_gb": 16, "cost_per_hr": 0.0, "available": True},
            {"id": "GPU-SubCent", "vram_gb": 16, "cost_per_hr": 0.005, "available": True},
            {"id": "GPU-Standard", "vram_gb": 16, "cost_per_hr": 0.20, "available": True},
        ]
        selected = select_cheapest_gpu(catalog, min_vram_gb=16)
        assert selected["id"] == "GPU-Free"
        assert selected["cost_per_hr"] == 0.0

    def test_empty_catalog_aborts(self):
        """Empty catalog raises RuntimeError immediately."""
        with pytest.raises(RuntimeError, match="No qualified RunPod GPUs available"):
            select_cheapest_gpu([], min_vram_gb=16)

    def test_insufficient_vram_all_gpus_aborts(self):
        """When all GPUs have VRAM below min_vram_gb, RuntimeError is raised."""
        catalog = [
            {"id": "GPU-8GB", "vram_gb": 8, "cost_per_hr": 0.05, "available": True},
            {"id": "GPU-12GB", "vram_gb": 12, "cost_per_hr": 0.09, "available": True},
        ]
        with pytest.raises(RuntimeError, match="min_vram_gb=16"):
            select_cheapest_gpu(catalog, min_vram_gb=16)

    def test_all_gpus_unavailable_aborts(self):
        """When all GPUs are available=False, RuntimeError is raised."""
        catalog = [
            {"id": "GPU-A", "vram_gb": 24, "cost_per_hr": 0.10, "available": False},
            {"id": "GPU-B", "vram_gb": 24, "cost_per_hr": 0.20, "available": False},
        ]
        with pytest.raises(RuntimeError, match="No qualified RunPod GPUs available"):
            select_cheapest_gpu(catalog, min_vram_gb=16)

    def test_unavailable_cheapest_never_selected(self):
        """A cheaper unavailable GPU must never be chosen over an available GPU."""
        catalog = [
            {"id": "GPU-Cheap-Unavailable", "vram_gb": 24, "cost_per_hr": 0.05, "available": False},
            {"id": "GPU-Available-Costlier", "vram_gb": 24, "cost_per_hr": 0.35, "available": True},
        ]
        selected = select_cheapest_gpu(catalog, min_vram_gb=16)
        assert selected["id"] == "GPU-Available-Costlier"
        assert selected["cost_per_hr"] == pytest.approx(0.35)

    def test_target_gpu_filter_case_and_substring_matching(self):
        """Target GPU matches substrings case-insensitively in id and displayName."""
        catalog = [
            {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
            {"id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "vram_gb": 24, "cost_per_hr": 0.44, "available": True},
        ]
        # Match by lowercase substring of displayName
        chosen1 = select_cheapest_gpu(catalog, target_gpu="a5000")
        assert chosen1["id"] == "NVIDIA RTX A5000"

        # Match by uppercase substring
        chosen2 = select_cheapest_gpu(catalog, gpu_type="A4000")
        assert chosen2["id"] == "NVIDIA RTX A4000"

    def test_target_gpu_not_found_raises(self):
        """Target GPU filter matching nothing raises RuntimeError."""
        catalog = [
            {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
        ]
        with pytest.raises(RuntimeError, match="target='nonexistent_gpu'"):
            select_cheapest_gpu(catalog, target_gpu="nonexistent_gpu")


# ==============================================================================
# 2. Adversarial Tests: Max Cost Guardrail Ceilings
# ==============================================================================

class TestMaxCostGuardrailAdversarial:
    """Stress-test max-cost boundary conditions and pod creation prevention."""

    def test_max_cost_exact_boundary_permitted(self):
        """When cheapest GPU cost exactly equals max_cost, selection succeeds."""
        catalog = [
            {"id": "GPU-Exact", "vram_gb": 24, "cost_per_hr": 0.50, "available": True},
            {"id": "GPU-Costly", "vram_gb": 24, "cost_per_hr": 1.00, "available": True},
        ]
        runner = RunPodRunner(dry_run=True, max_cost=0.50)
        chosen = runner.filter_and_select_gpu(catalog)
        assert chosen["cost_per_hr"] == pytest.approx(0.50)

    def test_max_cost_epsilon_below_cheapest_rejected(self):
        """When max_cost is 0.0001 below cheapest, selection raises ValueError."""
        catalog = [
            {"id": "GPU-Exact", "vram_gb": 24, "cost_per_hr": 0.50, "available": True},
        ]
        runner = RunPodRunner(dry_run=True, max_cost=0.4999)
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner.filter_and_select_gpu(catalog)

    def test_max_cost_zero_and_negative_rejected(self):
        """Zero or negative max_cost rejects non-free instances."""
        catalog = [
            {"id": "GPU-Paid", "vram_gb": 24, "cost_per_hr": 0.24, "available": True},
        ]
        runner_zero = RunPodRunner(dry_run=True, max_cost=0.0)
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner_zero.filter_and_select_gpu(catalog)

        runner_neg = RunPodRunner(dry_run=True, max_cost=-0.5)
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner_neg.filter_and_select_gpu(catalog)

    def test_max_cost_violation_prevents_pod_creation(self):
        """Verifies that no pod is EVER created when max_cost is violated."""
        client = TrackedMockClient(
            catalog=[{"id": "GPU-Exp", "vram_gb": 24, "cost_per_hr": 2.50, "available": True}]
        )
        runner = RunPodRunner(dry_run=True, max_cost=1.00, client=client)

        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner.run_qualification_suite()

        # Guarantee zero pod creation
        assert len(client.created_pods) == 0

    def test_max_cost_cli_rejection_exit_code(self):
        """CLI invocation with max_cost lower than catalog exits with code 1."""
        rc = runpod_main(["--dry-run", "--max-cost", "0.05"])
        assert rc == 1


# ==============================================================================
# 3. Adversarial Tests: Hard Timeout Guardrails (180s)
# ==============================================================================

class TestTimeoutGuardrailsAdversarial:
    """Stress-test timeout boundary conditions and enforcement points."""

    def test_exact_timeout_boundaries(self):
        """Timeouts 1 and 180 are accepted; 0 and 181 are rejected."""
        assert validate_timeout_guardrail(1) == 1
        assert validate_timeout_guardrail(180) == 180

        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            validate_timeout_guardrail(0)

        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            validate_timeout_guardrail(181)

    @pytest.mark.parametrize("invalid_timeout", [-1, -100, 181, 182, 300, 3600])
    def test_all_out_of_bound_timeouts_rejected(self, invalid_timeout):
        """All out-of-bound timeouts raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            validate_timeout_guardrail(invalid_timeout)

    def test_runner_instantiation_rejects_out_of_bounds(self):
        """RunPodRunner rejects out-of-bound timeouts upon creation."""
        with pytest.raises(ValueError, match="timeout"):
            RunPodRunner(dry_run=True, timeout=181)

        with pytest.raises(ValueError, match="timeout"):
            RunPodRunner(dry_run=True, timeout=0)

    def test_provision_ephemeral_pod_validates_timeout_before_creation(self):
        """provision_ephemeral_pod must validate timeout BEFORE calling client.create_pod."""
        client = TrackedMockClient()
        with pytest.raises(ValueError, match="timeout"):
            with provision_ephemeral_pod(client, gpu_type="RTX A4000", timeout_seconds=185):
                pass
        # Assert client was never touched
        assert len(client.created_pods) == 0

    def test_timeout_propagates_to_create_pod_request(self):
        """Validated timeout is correctly passed down to client.create_pod."""
        client = TrackedMockClient()
        with provision_ephemeral_pod(client, gpu_type="RTX A4000", timeout_seconds=95):
            pass
        assert len(client.created_pods) == 1
        assert client.created_pods[0]["timeout_seconds"] == 95

    def test_cli_rejects_out_of_bounds_timeout(self):
        """CLI main rejects timeout > 180s and timeout <= 0s."""
        rc_high = runpod_main(["--dry-run", "--timeout", "200"])
        assert rc_high != 0

        rc_low = runpod_main(["--dry-run", "--timeout", "0"])
        assert rc_low != 0

    def test_duration_exceeding_180s_raises_and_cleans_up_pod(self):
        """If qualification duration somehow exceeds 180s, runner aborts and cleans up pod."""
        client = TrackedMockClient()
        runner = RunPodRunner(dry_run=True, client=client)

        # Mock monotonic clock so elapsed appears to be 185s
        start = 100.0
        clock_values = [start, start + 185.0, start + 185.0]

        with patch("time.monotonic", side_effect=clock_values):
            with pytest.raises(RuntimeError, match="exceeded hard 180s limit"):
                runner.run_qualification_suite()

        # Pod MUST still have been terminated!
        assert client.all_pods_cleaned_up
        assert len(client.created_pods) == 1
        assert len(client.terminated_pods) == 1


# ==============================================================================
# 4. Adversarial Tests: Guaranteed Cleanup Traps
# ==============================================================================

class TestGuaranteedCleanupTrapsAdversarial:
    """Stress-test guaranteed pod cleanup across exceptions, signals, and exits."""

    def test_cleanup_trap_on_generic_exception(self):
        """Pod is destroyed when user code inside context manager raises Exception."""
        client = TrackedMockClient()
        runner = RunPodRunner(dry_run=True, client=client)

        with pytest.raises(ZeroDivisionError):
            with runner.ephemeral_pod(gpu_type="NVIDIA RTX A4000"):
                _ = 1 / 0

        assert client.all_pods_cleaned_up
        assert len(client.terminated_pods) == 1

    def test_cleanup_trap_on_base_exceptions(self):
        """Pod is destroyed even on BaseException subclasses (KeyboardInterrupt, SystemExit)."""
        # Test KeyboardInterrupt
        client_ki = TrackedMockClient()
        runner_ki = RunPodRunner(dry_run=True, client=client_ki)
        with pytest.raises(KeyboardInterrupt):
            with runner_ki.ephemeral_pod(gpu_type="NVIDIA RTX A4000"):
                raise KeyboardInterrupt()
        assert client_ki.all_pods_cleaned_up

        # Test SystemExit
        client_se = TrackedMockClient()
        runner_se = RunPodRunner(dry_run=True, client=client_se)
        with pytest.raises(SystemExit):
            with runner_se.ephemeral_pod(gpu_type="NVIDIA RTX A4000"):
                sys.exit(42)
        assert client_se.all_pods_cleaned_up

    def test_cleanup_trap_on_remote_network_disconnect(self):
        """Pod is destroyed when execute_qualification suffers connection failure."""
        client = TrackedMockClient(fail_on_execute=True)
        runner = RunPodRunner(dry_run=True, client=client)

        with pytest.raises(ConnectionResetError):
            runner.execute_with_lifecycle(client, gpu_type="NVIDIA RTX A4000")

        assert client.all_pods_cleaned_up
        assert len(client.terminated_pods) == 1

    def test_cleanup_resilient_to_terminate_pod_failure(self):
        """When client.terminate_pod throws, unregister still occurs in finally."""
        client = TrackedMockClient(fail_on_terminate=True)
        runner = RunPodRunner(dry_run=True, client=client)

        # Context manager propagates terminate exception from finally, but guarantees registry unregister
        with pytest.raises(RuntimeError, match="Network failure while calling terminate_pod"):
            with runner.ephemeral_pod(gpu_type="NVIDIA RTX A4000") as pod_id:
                pass

        # Active pod state should be cleared in registry and runner
        assert runner._active_pod_id is None
        assert pod_id not in _GLOBAL_POD_REGISTRY._active_pods

    def test_cleanup_idempotency(self):
        """Repeated cleanup calls do not raise or corrupt state."""
        client = TrackedMockClient()
        runner = RunPodRunner(dry_run=True, client=client)
        pod_id = client.create_pod("NVIDIA RTX A4000")
        runner._active_pod_id = pod_id

        runner._cleanup_pod()
        runner._cleanup_pod()
        runner._cleanup_pod()

        assert client.all_pods_cleaned_up
        assert runner._active_pod_id is None

    def test_registry_cleanup_all_multiple_active_pods(self):
        """_ActivePodRegistry.cleanup_all cleans up multiple active pods concurrently."""
        registry = _ActivePodRegistry()
        client1 = TrackedMockClient()
        client2 = TrackedMockClient()

        pod1 = client1.create_pod("RTX A4000")
        pod2 = client2.create_pod("RTX A5000")

        registry.register(pod1, client1)
        registry.register(pod2, client2)

        assert len(registry._active_pods) == 2

        registry.cleanup_all()

        assert len(registry._active_pods) == 0
        assert client1.all_pods_cleaned_up
        assert client2.all_pods_cleaned_up

    def test_real_subprocess_sigterm_cleanup(self):
        """A real OS child process receiving SIGTERM must invoke cleanup traps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proof_file = Path(tmpdir) / "cleaned_up.txt"
            ready_file = Path(tmpdir) / "ready.txt"

            child_code = f"""
import os, sys, time, signal
from hypergan.runpod_runner import _GLOBAL_POD_REGISTRY

class ChildMockClient:
    def terminate_pod(self, pod_id):
        with open("{proof_file}", "w") as f:
            f.write("TERMINATED:" + pod_id)

client = ChildMockClient()
_GLOBAL_POD_REGISTRY.register("child-pod-sigterm", client)

with open("{ready_file}", "w") as f:
    f.write("READY")

# Sleep until killed by signal
while True:
    time.sleep(0.1)
"""
            proc = subprocess.Popen([sys.executable, "-c", child_code])
            try:
                # Wait for child to be ready
                for _ in range(50):
                    if ready_file.exists():
                        break
                    time.sleep(0.05)
                assert ready_file.exists(), "Child process did not become ready"

                # Send real OS SIGTERM
                os.kill(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)

                # Verify cleanup occurred
                assert proof_file.exists(), "Proof file was not written by SIGTERM cleanup handler"
                assert "TERMINATED:child-pod-sigterm" in proof_file.read_text()
            finally:
                if proc.poll() is None:
                    proc.kill()

    def test_real_subprocess_sigint_cleanup(self):
        """A real OS child process receiving SIGINT must invoke cleanup traps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proof_file = Path(tmpdir) / "cleaned_up_int.txt"
            ready_file = Path(tmpdir) / "ready_int.txt"

            child_code = f"""
import os, sys, time, signal
from hypergan.runpod_runner import _GLOBAL_POD_REGISTRY

class ChildMockClient:
    def terminate_pod(self, pod_id):
        with open("{proof_file}", "w") as f:
            f.write("TERMINATED:" + pod_id)

client = ChildMockClient()
_GLOBAL_POD_REGISTRY.register("child-pod-sigint", client)

with open("{ready_file}", "w") as f:
    f.write("READY")

while True:
    time.sleep(0.1)
"""
            proc = subprocess.Popen([sys.executable, "-c", child_code])
            try:
                for _ in range(50):
                    if ready_file.exists():
                        break
                    time.sleep(0.05)
                assert ready_file.exists(), "Child process did not become ready"

                # Send real OS SIGINT
                os.kill(proc.pid, signal.SIGINT)
                proc.wait(timeout=5)

                assert proof_file.exists(), "Proof file was not written by SIGINT cleanup handler"
                assert "TERMINATED:child-pod-sigint" in proof_file.read_text()
            finally:
                if proc.poll() is None:
                    proc.kill()

    def test_real_subprocess_atexit_cleanup_on_sys_exit(self):
        """A real OS child process exiting via sys.exit must invoke atexit cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proof_file = Path(tmpdir) / "cleaned_up_atexit.txt"

            child_code = f"""
import sys
from hypergan.runpod_runner import _GLOBAL_POD_REGISTRY

class ChildMockClient:
    def terminate_pod(self, pod_id):
        with open("{proof_file}", "w") as f:
            f.write("TERMINATED:" + pod_id)

client = ChildMockClient()
_GLOBAL_POD_REGISTRY.register("child-pod-atexit", client)
sys.exit(0)
"""
            proc = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True, timeout=10)
            assert proc.returncode == 0
            assert proof_file.exists(), "Proof file was not written by atexit hook"
            assert "TERMINATED:child-pod-atexit" in proof_file.read_text()


# ==============================================================================
# 5. Adversarial Tests: Receipt Schema & SHA-256 Tamper Detection
# ==============================================================================

class TestReceiptIntegrityAdversarial:
    """Stress-test receipt format, schema contract, and SHA-256 tamper-evidence."""

    def test_receipt_full_schema_contract(self, tmp_path):
        """Receipt emitted by dry-run RunPodRunner satisfies all interface contracts."""
        runner = RunPodRunner(dry_run=True)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        validate_qualification_receipt(receipt)
        assert receipt["schema_version"] == 1
        assert receipt["status"] in ("passed", "failed")
        assert receipt["duration_seconds"] <= 180.0
        assert "runpod" in receipt["environment"]
        assert receipt["environment"]["runpod"]["dry_run"] is True
        assert len(receipt["suites"]) >= 4

    def test_tamper_detection_on_every_field(self, tmp_path):
        """Mutating any field in receipt invalidates the computed SHA-256 digest."""
        runner = RunPodRunner(dry_run=True)
        original = runner.run_qualification_suite(output_dir=tmp_path)
        orig_hash = original["receipt_sha256"]

        # Mutate status
        mutated1 = copy.deepcopy(original)
        mutated1["status"] = "failed"
        assert compute_receipt_sha256(mutated1) != orig_hash

        # Mutate duration
        mutated2 = copy.deepcopy(original)
        mutated2["duration_seconds"] = original["duration_seconds"] + 0.01
        assert compute_receipt_sha256(mutated2) != orig_hash

        # Mutate suites
        mutated3 = copy.deepcopy(original)
        mutated3["suites"][0]["status"] = "failed"
        assert compute_receipt_sha256(mutated3) != orig_hash

        # Mutate environment
        mutated4 = copy.deepcopy(original)
        mutated4["environment"]["platform"] = "mutated_os"
        assert compute_receipt_sha256(mutated4) != orig_hash

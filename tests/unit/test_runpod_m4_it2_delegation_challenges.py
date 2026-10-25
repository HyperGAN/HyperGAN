"""Empirical adversarial challenge suite for Milestone 4 Iteration 2 RunPod guardrails and delegation.

Challenges:
1. Ephemeral guardrails in runpod_runner.py:
   - Lowest-cost GPU selector: filtered types, case-insensitivity, max-cost boundaries,
     and RunPodHttpClient resilience to corrupted GraphQL pricing payloads.
   - 180s hard timeout guardrail: rejection of >180s and <=0s across validator, runner init,
     ephemeral context manager, and CLI.
   - Cleanup traps: verification that normal exit and exceptions invoke pod termination
     without leaks.
2. Client qualification delegation in run_qualification_suite():
   - Minimal dictionary normalization into a fully valid receipt envelope (all 6 suites, SHA256).
   - Partial dictionary preservation of custom telemetry alongside normalization.
   - Exception handling during execute_qualification ensuring immediate pod termination.
   - Fallback to _build_mock_receipt when execute_qualification is missing, None, non-callable,
     or returns a non-dict.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from hypergan.qualify import compute_receipt_sha256, validate_qualification_receipt
from hypergan.runpod_runner import (
    DEFAULT_GPU_CATALOG,
    DEFAULT_TIMEOUT,
    MAX_ALLOWED_TIMEOUT,
    RunPodHttpClient,
    RunPodRunner,
    _GLOBAL_POD_REGISTRY,
    main as runpod_main,
    provision_ephemeral_pod,
    select_cheapest_gpu,
    validate_timeout_guardrail,
)


# ==============================================================================
# Helper Mock Clients for Delegation Testing
# ==============================================================================

class DelegationMockClient:
    """Mock client tracking lifecycle calls and allowing custom qualification responses."""

    def __init__(
        self,
        catalog: Optional[List[Dict[str, Any]]] = None,
        qualification_result: Any = None,
        fail_execute_exception: Optional[Exception] = None,
    ):
        self.catalog = catalog or copy.deepcopy(DEFAULT_GPU_CATALOG)
        self.qualification_result = qualification_result
        self.fail_execute_exception = fail_execute_exception
        self.created_pods: List[str] = []
        self.terminated_pods: List[str] = []
        self._next_id = 1

    def get_gpu_catalog(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.catalog)

    def create_pod(self, gpu_type: str, image: str = "hypergan:latest", timeout_seconds: int = 180) -> str:
        pod_id = f"mock-del-{self._next_id}"
        self._next_id += 1
        self.created_pods.append(pod_id)
        return pod_id

    def execute_qualification(self, pod_id: str) -> Any:
        if self.fail_execute_exception is not None:
            raise self.fail_execute_exception
        return self.qualification_result

    def terminate_pod(self, pod_id: str) -> bool:
        if pod_id in self.created_pods:
            self.terminated_pods.append(pod_id)
            return True
        return False

    @property
    def all_pods_cleaned_up(self) -> bool:
        return set(self.created_pods) == set(self.terminated_pods)


class FallbackMockClient:
    """Mock client without execute_qualification method."""

    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None):
        self.catalog = catalog or copy.deepcopy(DEFAULT_GPU_CATALOG)
        self.created_pods: List[str] = []
        self.terminated_pods: List[str] = []
        self._next_id = 1

    def get_gpu_catalog(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.catalog)

    def create_pod(self, gpu_type: str, image: str = "hypergan:latest", timeout_seconds: int = 180) -> str:
        pod_id = f"mock-fallback-{self._next_id}"
        self._next_id += 1
        self.created_pods.append(pod_id)
        return pod_id

    def terminate_pod(self, pod_id: str) -> bool:
        if pod_id in self.created_pods:
            self.terminated_pods.append(pod_id)
            return True
        return False

    @property
    def all_pods_cleaned_up(self) -> bool:
        return set(self.created_pods) == set(self.terminated_pods)



# ==============================================================================
# 1. Ephemeral Guardrails Challenge Suite
# ==============================================================================

class TestRunPodGuardrailsM4It2:
    """Empirical challenge suite for lowest-cost selector, timeout limits, and cleanup traps."""

    def test_select_gpu_with_filtered_types_and_case_insensitivity(self):
        """Selector matches GPU types case-insensitively across id and displayName with whitespace."""
        catalog = [
            {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
            {"id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "vram_gb": 24, "cost_per_hr": 0.44, "available": True},
            {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "vram_gb": 48, "cost_per_hr": 0.79, "available": True},
            {"id": "NVIDIA H100 SXM", "displayName": "H100 SXM", "vram_gb": 80, "cost_per_hr": 3.89, "available": True},
        ]
        # Lowercase substring matching displayName
        chosen1 = select_cheapest_gpu(catalog, gpu_type="  a5000  ")
        assert chosen1["id"] == "NVIDIA RTX A5000"

        # Uppercase substring matching id
        chosen2 = select_cheapest_gpu(catalog, target_gpu="RTX A4000")
        assert chosen2["id"] == "NVIDIA RTX A4000"

        # Non-matching filter raises RuntimeError with diagnostics
        with pytest.raises(RuntimeError, match="target='B200'"):
            select_cheapest_gpu(catalog, gpu_type="B200")

    def test_select_gpu_max_cost_exact_and_epsilon_boundaries(self):
        """Max-cost ceiling permits exact boundary and rejects epsilon below."""
        catalog = [
            {"id": "GPU-Exact", "vram_gb": 24, "cost_per_hr": 0.50, "available": True},
            {"id": "GPU-Expensive", "vram_gb": 24, "cost_per_hr": 1.50, "available": True},
        ]
        # Exact boundary permitted
        runner_exact = RunPodRunner(dry_run=True, max_cost=0.50)
        chosen = runner_exact.filter_and_select_gpu(catalog)
        assert chosen["cost_per_hr"] == pytest.approx(0.50)

        # Epsilon below ($0.4999) rejected
        runner_below = RunPodRunner(dry_run=True, max_cost=0.4999)
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner_below.filter_and_select_gpu(catalog)

        # Zero and negative max_cost rejected
        runner_zero = RunPodRunner(dry_run=True, max_cost=0.0)
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            runner_zero.filter_and_select_gpu(catalog)

    def test_runpod_http_client_handles_corrupted_graphql_pricing(self):
        """RunPodHttpClient drops null or missing prices from live GraphQL and falls back cleanly."""
        client = RunPodHttpClient(api_key="rpa_test_key")

        # Mock payload containing entries with null prices or missing lowestPrice
        corrupted_payload = {
            "gpuTypes": [
                {"id": "GPU-NoPrice", "displayName": "No Price", "memoryInGb": 24, "lowestPrice": None},
                {
                    "id": "GPU-NullPrices",
                    "displayName": "Null Price",
                    "memoryInGb": 24,
                    "lowestPrice": {"minimumBidPrice": None, "uninterruptablePrice": None},
                },
                {
                    "id": "GPU-Valid",
                    "displayName": "Valid Price",
                    "memoryInGb": 24,
                    "lowestPrice": {"minimumBidPrice": 0.35, "uninterruptablePrice": 0.45},
                },
            ]
        }
        with patch.object(client, "_execute", return_value=corrupted_payload):
            parsed = client.get_gpu_catalog()
            assert len(parsed) == 1
            assert parsed[0]["id"] == "GPU-Valid"
            assert parsed[0]["cost_per_hr"] == pytest.approx(0.45)

        # Mock payload completely empty -> falls back to DEFAULT_GPU_CATALOG
        with patch.object(client, "_execute", return_value={"gpuTypes": []}):
            fallback_catalog = client.get_gpu_catalog()
            assert fallback_catalog == DEFAULT_GPU_CATALOG

    @pytest.mark.parametrize("invalid_timeout", [181, 182, 250, 3600, 0, -1, -60])
    def test_timeout_guardrail_rejects_out_of_bounds(self, invalid_timeout):
        """Hard 180s timeout guardrail rejects any timeout <= 0 or > 180."""
        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            validate_timeout_guardrail(invalid_timeout)

        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            RunPodRunner(dry_run=True, timeout=invalid_timeout)

        client = DelegationMockClient()
        with pytest.raises(ValueError, match="timeout must be between 1 and 180 seconds"):
            with provision_ephemeral_pod(client, "RTX A4000", timeout_seconds=invalid_timeout):
                pass
        # No pod was ever created
        assert len(client.created_pods) == 0

    def test_cleanup_traps_invoke_pod_termination_on_normal_exit_and_exceptions(self):
        """Normal exit and task exceptions invoke pod termination exactly once with zero leaks."""
        # 1. Normal exit
        client_ok = DelegationMockClient()
        runner_ok = RunPodRunner(dry_run=True, client=client_ok)
        runner_ok.run_qualification_suite()
        assert len(client_ok.created_pods) == 1
        assert len(client_ok.terminated_pods) == 1
        assert client_ok.all_pods_cleaned_up
        assert runner_ok._active_pod_id is None

        # 2. Exception in task
        client_err = DelegationMockClient(fail_execute_exception=RuntimeError("Subprocess qualification failed"))
        runner_err = RunPodRunner(dry_run=True, client=client_err)
        with pytest.raises(RuntimeError, match="Subprocess qualification failed"):
            runner_err.run_qualification_suite()
        assert len(client_err.created_pods) == 1
        assert len(client_err.terminated_pods) == 1
        assert client_err.all_pods_cleaned_up
        assert runner_err._active_pod_id is None


# ==============================================================================
# 2. Client Qualification Delegation Challenge Suite
# ==============================================================================

class TestRunPodDelegationM4It2:
    """Empirical challenge suite for client qualification delegation and normalization."""

    def test_client_execute_qualification_minimal_dict_envelope_normalization(self, tmp_path):
        """When client returns an empty dict {}, runner normalizes all canonical envelope fields."""
        client = DelegationMockClient(qualification_result={})
        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        # Verify envelope structure
        assert receipt["schema_version"] == 1
        assert receipt["status"] == "passed"
        assert receipt["duration_seconds"] >= 0.0
        assert receipt["qualification_id"].startswith("qual-runpod-")
        assert "timestamp" in receipt

        # Verify all 6 canonical suites are injected
        suite_names = [s["name"] for s in receipt["suites"]]
        assert len(suite_names) == 6
        expected_suites = [
            "single_device",
            "cpu_replicated_gloo",
            "cuda_replicated_nccl",
            "aggregation_strategies",
            "multinode_tcp",
            "checkpoint_compatibility",
        ]
        assert suite_names == expected_suites

        # Verify environment normalization
        env = receipt["environment"]
        assert "platform" in env
        assert "python" in env
        assert "torch" in env
        assert env["device_count"] == 1
        assert len(env["devices"]) >= 1
        assert env["cuda"]["available"] is True
        assert env["runpod"]["dry_run"] is True

        # Verify SHA-256 integrity and receipt validation
        assert "receipt_sha256" in receipt
        assert receipt["receipt_sha256"] == compute_receipt_sha256(receipt)
        validate_qualification_receipt(receipt)

        # Verify pod terminated
        assert client.all_pods_cleaned_up
        assert len(client.terminated_pods) == 1

    def test_client_execute_qualification_preserves_custom_status_and_suites(self, tmp_path):
        """When client returns partial telemetry, custom status and suites are preserved."""
        custom_payload = {
            "status": "passed",
            "custom_telemetry": {"gpus_tested": 2, "link_type": "nvlink"},
            "suites": [
                {"name": "custom_nccl_stress", "status": "passed", "duration_seconds": 1.2},
            ],
        }
        client = DelegationMockClient(qualification_result=custom_payload)
        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        # Custom fields preserved
        assert receipt["custom_telemetry"] == {"gpus_tested": 2, "link_type": "nvlink"}
        assert len(receipt["suites"]) == 1
        assert receipt["suites"][0]["name"] == "custom_nccl_stress"

        # Envelope normalized
        assert receipt["schema_version"] == 1
        assert "environment" in receipt
        assert receipt["receipt_sha256"] == compute_receipt_sha256(receipt)
        assert client.all_pods_cleaned_up

    @pytest.mark.parametrize(
        "exc_type",
        [RuntimeError, ConnectionResetError, ValueError, KeyError, TimeoutError],
    )
    def test_client_execute_qualification_exception_triggers_cleanup_trap(self, exc_type):
        """When execute_qualification raises any exception, cleanup trap destroys the pod."""
        client = DelegationMockClient(fail_execute_exception=exc_type("Cloud qualification failed"))
        runner = RunPodRunner(dry_run=True, client=client)

        with pytest.raises(exc_type, match="Cloud qualification failed"):
            runner.run_qualification_suite()

        # Pod MUST be terminated exactly once
        assert client.all_pods_cleaned_up
        assert len(client.created_pods) == 1
        assert len(client.terminated_pods) == 1
        assert client.created_pods[0] == client.terminated_pods[0]
        assert runner._active_pod_id is None

    def test_client_without_execute_qualification_falls_back_to_mock_receipt(self, tmp_path):
        """When client lacks execute_qualification, runner falls back to _build_mock_receipt."""
        client = FallbackMockClient()
        assert not hasattr(client, "execute_qualification")

        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        assert receipt["status"] == "passed"
        assert len(receipt["suites"]) == 6
        assert receipt["receipt_sha256"] == compute_receipt_sha256(receipt)
        assert client.all_pods_cleaned_up

    def test_client_with_none_or_non_callable_execute_qualification_falls_back(self, tmp_path):
        """When execute_qualification is None or not callable, runner falls back gracefully."""
        # execute_qualification is None
        client_none = FallbackMockClient()
        client_none.execute_qualification = None  # type: ignore[assignment]
        runner_none = RunPodRunner(dry_run=True, client=client_none)
        r_none = runner_none.run_qualification_suite(output_dir=tmp_path)
        assert r_none["status"] == "passed"
        assert len(r_none["suites"]) == 6

        # execute_qualification is a string (non-callable)
        client_str = FallbackMockClient()
        client_str.execute_qualification = "not_callable"  # type: ignore[assignment]
        runner_str = RunPodRunner(dry_run=True, client=client_str)
        r_str = runner_str.run_qualification_suite(output_dir=tmp_path)
        assert r_str["status"] == "passed"
        assert len(r_str["suites"]) == 6

    def test_client_returning_non_dict_falls_back_to_mock_receipt(self, tmp_path):
        """When execute_qualification returns non-dict (e.g. None or str), fallback is used."""
        for non_dict in [None, "completed", [1, 2, 3], True]:
            client = DelegationMockClient(qualification_result=non_dict)
            runner = RunPodRunner(dry_run=True, client=client)
            receipt = runner.run_qualification_suite(output_dir=tmp_path)
            assert isinstance(receipt, dict)
            assert receipt["status"] == "passed"
            assert len(receipt["suites"]) == 6
            assert client.all_pods_cleaned_up

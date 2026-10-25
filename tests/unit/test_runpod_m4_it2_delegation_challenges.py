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
   - Incomplete or partial receipts are rejected, never filled in with invented results.
   - A complete remote receipt, custom telemetry included, is returned unchanged.
   - Exception handling during execute_qualification ensuring immediate pod termination.
   - A client whose execute_qualification is missing, None or non-callable is refused before
     any pod is created; a non-dict result is rejected after the pod is terminated.
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

from hypergan.qualify import ReceiptValidationError, compute_receipt_sha256, validate_qualification_receipt
from tests.runpod_fakes import build_fake_receipt
from hypergan.runpod_runner import (
    DEFAULT_GPU_CATALOG,
    DEFAULT_TIMEOUT,
    MAX_ALLOWED_TIMEOUT,
    LiveModeUnavailable,
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
        """RunPodHttpClient drops null or missing prices, and refuses an empty catalog."""
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

        # Empty payload -> error; never select a GPU from invented prices
        with patch.object(client, "_execute", return_value={"gpuTypes": []}):
            with pytest.raises(RuntimeError, match="no priced GPU types"):
                client.get_gpu_catalog()

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
        client_ok = DelegationMockClient(qualification_result=build_fake_receipt())
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

    @pytest.mark.parametrize("partial", [
        {},
        {"status": "passed"},
        {"status": "passed", "suites": [{"name": "custom_nccl_stress", "status": "passed", "duration_seconds": 1.2}]},
    ])
    def test_incomplete_remote_receipt_is_rejected_not_normalized(self, tmp_path, partial):
        """An incomplete receipt is an error; the runner never invents suites, status or environment."""
        client = DelegationMockClient(qualification_result=partial)
        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(ReceiptValidationError, match="missing required keys"):
            runner.run_qualification_suite(output_dir=tmp_path)
        assert not (tmp_path / "qualification-receipt.json").exists()
        # Verify pod terminated
        assert client.all_pods_cleaned_up
        assert len(client.terminated_pods) == 1

    def test_complete_remote_receipt_is_returned_unchanged(self, tmp_path):
        """A complete receipt, custom telemetry and suites included, passes through untouched."""
        remote = build_fake_receipt(
            suites=[{"name": "custom_nccl_stress", "status": "passed", "duration_seconds": 1.2}],
            custom_telemetry={"gpus_tested": 2, "link_type": "nvlink"},
        )
        client = DelegationMockClient(qualification_result=remote)
        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        assert receipt == remote
        assert receipt["receipt_sha256"] == compute_receipt_sha256(receipt)
        assert client.all_pods_cleaned_up

    def test_remote_receipt_contradicting_its_suites_is_rejected(self, tmp_path):
        """A receipt claiming passed while a suite failed is rejected even with a matching checksum."""
        remote = build_fake_receipt(
            status="passed",
            suites=[{"name": "cuda_replicated_nccl", "status": "failed", "duration_seconds": 1.0}],
        )
        client = DelegationMockClient(qualification_result=remote)
        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(ReceiptValidationError, match="contradicts its suites"):
            runner.run_qualification_suite(output_dir=tmp_path)
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

    def test_client_without_execute_qualification_is_refused_before_pod_creation(self, tmp_path):
        """When client lacks execute_qualification, no pod is created and no receipt is written."""
        client = FallbackMockClient()
        assert not hasattr(client, "execute_qualification")

        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(LiveModeUnavailable, match="cannot run qualification on a pod"):
            runner.run_qualification_suite(output_dir=tmp_path)
        assert client.created_pods == []
        assert not (tmp_path / "qualification-receipt.json").exists()

    @pytest.mark.parametrize("execute", [None, "not_callable"])
    def test_client_with_none_or_non_callable_execute_qualification_is_refused(self, tmp_path, execute):
        """When execute_qualification is None or not callable, the runner refuses before creating a pod."""
        client = FallbackMockClient()
        client.execute_qualification = execute  # type: ignore[assignment]
        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(LiveModeUnavailable):
            runner.run_qualification_suite(output_dir=tmp_path)
        assert client.created_pods == []

    @pytest.mark.parametrize("non_dict", [None, "completed", [1, 2, 3], True])
    def test_client_returning_non_dict_is_rejected(self, tmp_path, non_dict):
        """When execute_qualification returns a non-dict, the run fails after terminating the pod."""
        client = DelegationMockClient(qualification_result=non_dict)
        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(ValueError, match="not a receipt object"):
            runner.run_qualification_suite(output_dir=tmp_path)
        assert len(client.created_pods) == 1
        assert client.all_pods_cleaned_up

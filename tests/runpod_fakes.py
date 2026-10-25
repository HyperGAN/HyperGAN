"""Test doubles for hypergan.runpod_runner: a fake pod client and a fake HTTP transport.

These live in tests, not in the package: the shipped runner has no mock client and never
builds a receipt for suites that did not run.
"""
from __future__ import annotations

import copy
import io
import json
from typing import Any, Dict, List, Optional

from hypergan.qualify import compute_receipt_sha256, summarize_suite_status
from hypergan.runpod_runner import DEFAULT_GPU_CATALOG


def build_fake_receipt(
    qualification_id: str = "qual-fake",
    suites: Optional[List[Dict[str, Any]]] = None,
    status: Optional[str] = None,
    duration_seconds: float = 12.5,
    **extra: Any,
) -> Dict[str, Any]:
    """A complete, self-consistent receipt such as a remote `hypergan qualify --json` would emit."""
    suites = copy.deepcopy(suites) if suites is not None else [
        {"name": "single_device", "status": "passed", "duration_seconds": 0.5},
        {"name": "cuda_replicated_nccl", "status": "skipped", "duration_seconds": 0.0},
    ]
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "qualification_id": qualification_id,
        "timestamp": "2026-10-25T00:00:00+00:00",
        "status": status or summarize_suite_status(suites),
        "duration_seconds": duration_seconds,
        "environment": {"platform": "linux", "python": "3.14.8", "torch": "2.14.1", "device_count": 1,
                        "devices": ["NVIDIA RTX A4000"]},
        "suites": suites,
        **extra,
    }
    receipt["receipt_sha256"] = compute_receipt_sha256(receipt)
    return receipt


class FakeRunPodClient:
    """Pod client recording lifecycle calls; ``fail_at`` names the call that raises."""

    def __init__(
        self,
        catalog: Optional[List[Dict[str, Any]]] = None,
        fail_at: Optional[str] = None,
        qualification_result: Any = None,
    ):
        self.catalog = catalog or copy.deepcopy(DEFAULT_GPU_CATALOG)
        self.fail_at = fail_at
        self.qualification_result = qualification_result
        self.created_pods: List[str] = []
        self.terminated_pods: List[str] = []
        self.executed_pods: List[str] = []
        self.create_calls: List[Dict[str, Any]] = []
        self.lifecycle_history: List[str] = []
        self._next_id = 1

    def get_gpu_catalog(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.catalog)

    def create_pod(self, gpu_type: str, image: str = "hypergan:latest", timeout_seconds: int = 180) -> str:
        if self.fail_at == "create_pod":
            raise RuntimeError("Simulated cloud creation failure")
        pod_id = f"fake-pod-{self._next_id}"
        self._next_id += 1
        self.created_pods.append(pod_id)
        self.create_calls.append({"pod_id": pod_id, "gpu_type": gpu_type, "image": image,
                                  "timeout_seconds": timeout_seconds})
        self.lifecycle_history.append("created")
        return pod_id

    def execute_qualification(self, pod_id: str) -> Any:
        if self.fail_at == "execute":
            raise ConnectionResetError("Simulated remote qualification crash")
        self.executed_pods.append(pod_id)
        self.lifecycle_history.append("executed")
        if self.qualification_result is not None:
            return copy.deepcopy(self.qualification_result)
        return build_fake_receipt(qualification_id=f"qual-{pod_id}")

    def terminate_pod(self, pod_id: str) -> bool:
        if self.fail_at == "terminate_pod":
            raise RuntimeError("Network failure while calling terminate_pod")
        if pod_id in self.created_pods:
            self.terminated_pods.append(pod_id)
            self.lifecycle_history.append("terminated")
            return True
        return False

    @property
    def all_pods_cleaned_up(self) -> bool:
        return set(self.created_pods) == set(self.terminated_pods)


class CreateOnlyClient(FakeRunPodClient):
    """A client that can create pods but cannot run anything on them (like RunPodHttpClient)."""

    execute_qualification = None  # type: ignore[assignment]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeTransport:
    """Stands in for urllib.request.urlopen; replies in order, recording every request.

    Each reply is a JSON-serializable dict (the response body), an exception to raise, or a
    callable taking the transport and returning one of those.
    """

    def __init__(self, *replies: Any):
        self.replies = list(replies)
        self.requests: List[Any] = []

    def __call__(self, request: Any, timeout: float = None):
        self.requests.append(request)
        if not self.replies:
            raise AssertionError(f"Unexpected RunPod request: {request.data!r}")
        reply = self.replies.pop(0)
        if callable(reply) and not isinstance(reply, BaseException):
            reply = reply(self)
        if isinstance(reply, BaseException):
            raise reply
        return FakeResponse(json.dumps(reply).encode("utf-8"))

    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(r.data.decode("utf-8")) for r in self.requests]

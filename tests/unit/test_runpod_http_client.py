"""RunPodHttpClient against a fake HTTP transport: request shape, auth, retries, orphan recovery.

No test here touches the network; every request goes to tests.runpod_fakes.FakeTransport.
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from hypergan.runpod_runner import (
    RUNPOD_GRAPHQL_ENDPOINT,
    LiveModeUnavailable,
    PodCleanupError,
    RunPodApiError,
    RunPodHttpClient,
    RunPodRunner,
    _GLOBAL_POD_REGISTRY,
    provision_ephemeral_pod,
)
from tests.runpod_fakes import FakeTransport

SECRET = "rpa_TEST_SECRET_123"
CREATED = {"data": {"podFindAndDeployOnDemand": {"id": "pod-abc", "desiredStatus": "RUNNING", "imageName": "img"}}}
TERMINATED = {"data": {"podTerminate": None}}


def make_client(*replies, **kwargs):
    transport = FakeTransport(*replies)
    sleeps = []
    client = RunPodHttpClient(api_key=SECRET, urlopen=transport, sleep=sleeps.append, **kwargs)
    return client, transport, sleeps


def http_error(code: int, body: bytes = b"boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(RUNPOD_GRAPHQL_ENDPOINT, code, "error", {}, io.BytesIO(body))


class TestAuthentication:
    def test_api_key_only_in_authorization_header(self):
        client, transport, _ = make_client({"data": {"gpuTypes": [
            {"id": "GPU-A", "displayName": "A", "memoryInGb": 24, "lowestPrice": {"uninterruptablePrice": 0.3}},
        ]}})
        client.get_gpu_catalog()
        (request,) = transport.requests
        assert request.full_url == RUNPOD_GRAPHQL_ENDPOINT
        assert SECRET not in request.full_url
        assert "api_key" not in request.full_url
        assert request.get_header("Authorization") == f"Bearer {SECRET}"
        assert SECRET not in request.data.decode("utf-8")

    def test_api_key_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "rpa_from_env")
        client = RunPodHttpClient(urlopen=FakeTransport())
        assert client.api_key == "rpa_from_env"

    def test_missing_api_key_rejected(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="RUNPOD_API_KEY is required"):
            RunPodHttpClient(urlopen=FakeTransport())


class TestRequests:
    def test_gpu_catalog_parses_prices_and_refuses_an_empty_catalog(self):
        client, _, _ = make_client(
            {"data": {"gpuTypes": [
                {"id": "GPU-A", "displayName": "A", "memoryInGb": 24,
                 "lowestPrice": {"minimumBidPrice": 0.2, "uninterruptablePrice": 0.35}},
                {"id": "GPU-B", "displayName": "B", "memoryInGb": 48, "lowestPrice": None},
            ]}},
            {"data": {"gpuTypes": []}},
        )
        assert client.get_gpu_catalog() == [
            {"id": "GPU-A", "displayName": "A", "vram_gb": 24, "cost_per_hr": 0.35, "available": True},
        ]
        with pytest.raises(RuntimeError, match="no priced GPU types"):
            client.get_gpu_catalog()

    def test_graphql_errors_are_definitive_api_errors(self):
        client, _, _ = make_client({"errors": [{"message": "bad gpu"}]})
        with pytest.raises(RunPodApiError, match="bad gpu") as exc:
            client.get_gpu_catalog()
        assert exc.value.definitive is True

    def test_create_pod_payload_has_no_volume_and_a_unique_name(self):
        client, transport, _ = make_client(CREATED, CREATED)
        assert client.create_pod("NVIDIA RTX A4000", image="img", timeout_seconds=90) == "pod-abc"
        client.create_pod("NVIDIA RTX A4000", image="img", timeout_seconds=90)
        first, second = (body["variables"]["input"] for body in transport.bodies())
        assert first["gpuTypeId"] == "NVIDIA RTX A4000"
        assert first["imageName"] == "img"
        assert first["volumeInGb"] == 0
        assert first["stopAfter"] == 90
        assert first["name"].startswith("hypergan-qualify-")
        assert first["name"] != second["name"]


class TestCreateFailureRecovery:
    def test_definitive_rejection_does_not_look_for_orphans(self):
        client, transport, _ = make_client({"errors": [{"message": "no capacity"}]})
        with pytest.raises(RunPodApiError, match="no capacity"):
            client.create_pod("NVIDIA RTX A4000")
        assert len(transport.requests) == 1

    @pytest.mark.parametrize("failure", [TimeoutError("timed out"), urllib.error.URLError("reset"), http_error(502)])
    def test_ambiguous_failure_terminates_the_orphaned_pod(self, failure):
        """A create that times out may still have made a pod: find it by name and terminate it."""
        def pods_listing(transport):
            name = transport.bodies()[0]["variables"]["input"]["name"]
            return {"data": {"myself": {"pods": [{"id": "pod-orphan", "name": name},
                                                 {"id": "pod-other", "name": "someone-else"}]}}}

        client, transport, _ = make_client(failure, pods_listing, TERMINATED)
        with pytest.raises(RunPodApiError) as exc:
            client.create_pod("NVIDIA RTX A4000")
        assert exc.value.definitive is False
        bodies = transport.bodies()
        assert len(bodies) == 3
        assert "myself" in bodies[1]["query"]
        assert bodies[2]["variables"] == {"input": {"podId": "pod-orphan"}}

    def test_unreadable_pod_list_after_ambiguous_failure_fails_loudly(self):
        client, _, _ = make_client(TimeoutError("timed out"), TimeoutError("still down"))
        with pytest.raises(PodCleanupError, match="hypergan-qualify-") as exc:
            client.create_pod("NVIDIA RTX A4000")
        assert "Terminate it manually" in str(exc.value)


class TestTermination:
    def test_terminate_retries_with_backoff_then_confirms(self):
        client, transport, sleeps = make_client(urllib.error.URLError("down"), http_error(503), TERMINATED)
        assert client.terminate_pod("pod-abc") is True
        assert len(transport.requests) == 3
        assert sleeps == [2.0, 4.0]

    def test_terminate_failure_is_raised_not_swallowed(self):
        client, transport, sleeps = make_client(*(urllib.error.URLError("down") for _ in range(4)))
        with pytest.raises(PodCleanupError, match="pod-abc") as exc:
            client.terminate_pod("pod-abc")
        assert "4 attempts failed" in str(exc.value)
        assert len(transport.requests) == 4
        assert len(sleeps) == 3

    def test_lifecycle_keeps_pod_registered_when_termination_fails(self):
        client, _, _ = make_client(CREATED, *(urllib.error.URLError("down") for _ in range(4)))
        try:
            with pytest.raises(PodCleanupError, match="pod-abc"):
                with provision_ephemeral_pod(client, "NVIDIA RTX A4000"):
                    pass
            assert _GLOBAL_POD_REGISTRY.is_registered("pod-abc")
        finally:
            _GLOBAL_POD_REGISTRY.unregister("pod-abc")

    def test_lifecycle_creates_then_terminates(self):
        client, transport, _ = make_client(CREATED, TERMINATED)
        with provision_ephemeral_pod(client, "NVIDIA RTX A4000") as pod_id:
            assert pod_id == "pod-abc"
            assert _GLOBAL_POD_REGISTRY.is_registered(pod_id)
        assert not _GLOBAL_POD_REGISTRY.is_registered("pod-abc")
        assert transport.bodies()[1]["variables"] == {"input": {"podId": "pod-abc"}}


def test_runner_refuses_http_client_before_any_request(tmp_path):
    """RunPodHttpClient cannot run qualification, so the runner sends nothing and creates nothing."""
    client, transport, _ = make_client()
    runner = RunPodRunner(client=client)
    with pytest.raises(LiveModeUnavailable, match="RunPodHttpClient cannot run qualification"):
        runner.run_qualification_suite(output_dir=tmp_path)
    assert transport.requests == []

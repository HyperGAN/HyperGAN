"""Tests for the ephemeral RunPod runner: guardrails, pod cleanup, delegation and the CLI.

Covers:
1. Construction: dry-run defaults, live mode refused, signal handlers left alone.
2. The 180-second lifetime guardrail at every entry point.
3. Lowest-cost GPU selection and the --max-cost ceiling.
4. Pod cleanup: termination on success, exceptions, BaseExceptions and signals (in process and
   in a real child process), confirmed termination, idempotent release, the cleanup registry.
5. Delegation to a client's execute_qualification: complete receipts pass through untouched,
   incomplete, contradictory or non-dict ones are rejected, incapable clients are refused
   before any pod exists.
6. Dry-run receipts: nothing reported as passed, the inspected environment recorded verbatim.
7. The ``python -m hypergan.runpod_runner`` CLI: flags, stream separation, exit codes, receipt paths.

No test touches the network; the HTTP client's transport tests live in test_runpod_http_client.py.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from hypergan.qualify import ReceiptValidationError, compute_receipt_sha256, validate_qualification_receipt
from hypergan.runpod_runner import (
    LiveModeUnavailable,
    PodCleanupError,
    RunPodHttpClient,
    RunPodRunner,
    _ActivePodRegistry,
    _GLOBAL_POD_REGISTRY,
    _build_parser,
    main as runpod_main,
    provision_ephemeral_pod,
    select_cheapest_gpu,
    validate_timeout_guardrail,
)
from tests.runpod_fakes import CreateOnlyClient, FakeRunPodClient, build_fake_receipt

A4000 = "NVIDIA RTX A4000"


@pytest.fixture
def sample_gpu_catalog() -> List[Dict[str, Any]]:
    """Sample RunPod GPU catalog with varying VRAM, pricing, and availability."""
    return [
        {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "vram_gb": 48, "cost_per_hr": 0.79, "available": True},
        {"id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "vram_gb": 24, "cost_per_hr": 0.44, "available": True},
        {"id": A4000, "displayName": "RTX A4000", "vram_gb": 16, "cost_per_hr": 0.24, "available": True},
        {"id": "NVIDIA RTX 3090", "displayName": "RTX 3090", "vram_gb": 24, "cost_per_hr": 0.29, "available": False},
        {"id": "NVIDIA H100 SXM", "displayName": "H100 SXM", "vram_gb": 80, "cost_per_hr": 3.89, "available": True},
    ]


def passing_receipt(pod_id: str) -> Dict[str, Any]:
    return build_fake_receipt(qualification_id=f"qual-{pod_id}", duration_seconds=25.0,
                              suites=[{"name": "smoke", "status": "passed", "duration_seconds": 25.0}])


class FailingTerminateClient(FakeRunPodClient):
    def __init__(self, confirm: bool):
        super().__init__(fail_at=None if not confirm else "terminate_pod")
        self.confirm = confirm

    def terminate_pod(self, pod_id):
        if self.confirm:
            return super().terminate_pod(pod_id)  # raises: fail_at="terminate_pod"
        return False


@pytest.fixture
def forget_pods():
    """Unregister pods a test deliberately leaves registered."""
    pods: List[str] = []
    yield pods
    for pod_id in pods:
        _GLOBAL_POD_REGISTRY.unregister(pod_id)


# ==============================================================================
# 1. Construction
# ==============================================================================

class TestRunPodRunnerInit:

    def test_dry_run_defaults_need_no_api_key(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        runner = RunPodRunner(dry_run=True)
        assert runner.dry_run is True
        assert runner.timeout == 180
        assert runner.max_cost is None
        assert runner.gpu_type is None
        assert runner.verbose is False
        assert runner.client is None

    def test_custom_parameters_are_stored(self):
        runner = RunPodRunner(timeout=120, max_cost=0.50, gpu_type="NVIDIA RTX A5000", dry_run=True, verbose=True)
        assert runner.timeout == 120
        assert runner.max_cost == pytest.approx(0.50)
        assert runner.gpu_type == "NVIDIA RTX A5000"
        assert runner.verbose is True

    @pytest.mark.parametrize("api_key", [None, "rpa_test_secret"])
    def test_live_mode_refused(self, monkeypatch, api_key):
        """Live mode is refused, with or without a key, because nothing can run on the pod."""
        if api_key:
            monkeypatch.setenv("RUNPOD_API_KEY", api_key)
        else:
            monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(LiveModeUnavailable, match="refuses to create a paid pod"):
            RunPodRunner(dry_run=False)

    def test_construction_leaves_signal_handlers_untouched(self):
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        RunPodRunner(dry_run=True)
        RunPodRunner(dry_run=True, client=FakeRunPodClient())
        assert {sig: signal.getsignal(sig) for sig in before} == before


# ==============================================================================
# 2. Lifetime guardrail (1..180 s)
# ==============================================================================

INVALID_TIMEOUTS = [0, -1, -60, -100, 181, 182, 185, 250, 300, 3600]


class TestTimeoutGuardrail:

    @pytest.mark.parametrize("valid_timeout", [1, 30, 60, 95, 120, 179, 180])
    def test_valid_timeout_accepted(self, valid_timeout):
        assert validate_timeout_guardrail(valid_timeout) == valid_timeout

    @pytest.mark.parametrize("invalid_timeout", INVALID_TIMEOUTS)
    def test_invalid_timeout_rejected_everywhere_before_any_pod(self, invalid_timeout):
        """The validator, the runner and the context manager all refuse, and no pod is created."""
        match = "RunPod timeout must be between 1 and 180 seconds"
        with pytest.raises(ValueError, match=match):
            validate_timeout_guardrail(invalid_timeout)
        with pytest.raises(ValueError, match=match):
            RunPodRunner(dry_run=True, timeout=invalid_timeout)
        client = FakeRunPodClient()
        with pytest.raises(ValueError, match=match):
            with provision_ephemeral_pod(client, A4000, timeout_seconds=invalid_timeout):
                pass
        assert client.created_pods == []

    def test_timeout_propagates_to_create_pod(self):
        client = FakeRunPodClient()
        with provision_ephemeral_pod(client, gpu_type=A4000, timeout_seconds=95):
            pass
        assert [c["timeout_seconds"] for c in client.create_calls] == [95]

    @pytest.mark.parametrize("timeout", ["200", "0"])
    def test_cli_rejects_out_of_bounds_timeout(self, timeout, capsys):
        assert runpod_main(["--dry-run", "--timeout", timeout]) == 1
        assert "timeout must be between 1 and 180" in capsys.readouterr().err

    def test_over_budget_remote_run_raises_and_still_terminates_the_pod(self):
        """A remote qualification that outlives the timeout fails the run; the pod is still terminated."""
        now = [100.0]

        class SlowClient(FakeRunPodClient):
            def execute_qualification(self, pod_id):
                now[0] += 185.0
                return super().execute_qualification(pod_id)

        client = SlowClient()
        with patch("hypergan.runpod_runner.time.monotonic", side_effect=lambda: now[0]):
            with pytest.raises(RuntimeError, match="exceeded hard 180s limit"):
                RunPodRunner(dry_run=True, client=client).run_qualification_suite()
        assert len(client.created_pods) == 1
        assert client.all_pods_cleaned_up


# ==============================================================================
# 3. GPU selection and the cost ceiling
# ==============================================================================

class TestGPUSelector:

    def test_cheapest_available_gpu(self, sample_gpu_catalog):
        chosen = select_cheapest_gpu(sample_gpu_catalog, min_vram_gb=16)
        assert chosen["id"] == A4000
        assert chosen["cost_per_hr"] == pytest.approx(0.24)

    def test_vram_requirement_and_out_of_stock(self, sample_gpu_catalog):
        """min_vram_gb=24 drops the A4000; the cheaper RTX 3090 is out of stock, so the A5000 wins."""
        chosen = select_cheapest_gpu(sample_gpu_catalog, min_vram_gb=24)
        assert chosen["id"] == "NVIDIA RTX A5000"
        assert chosen["cost_per_hr"] == pytest.approx(0.44)

    def test_cheapest_chosen_regardless_of_order(self):
        base = [
            {"id": "GPU-Expensive", "vram_gb": 24, "cost_per_hr": 2.50, "available": True},
            {"id": "GPU-Moderate", "vram_gb": 24, "cost_per_hr": 0.80, "available": True},
            {"id": "GPU-Cheapest", "vram_gb": 24, "cost_per_hr": 0.19, "available": True},
            {"id": "GPU-High", "vram_gb": 32, "cost_per_hr": 1.20, "available": True},
        ]
        rng = random.Random(42)
        for _ in range(10):
            shuffled = copy.deepcopy(base)
            rng.shuffle(shuffled)
            assert select_cheapest_gpu(shuffled, min_vram_gb=16)["id"] == "GPU-Cheapest"

    def test_price_tie_returns_one_of_the_cheapest(self):
        catalog = [
            {"id": "GPU-Tie-1", "vram_gb": 24, "cost_per_hr": 0.25, "available": True},
            {"id": "GPU-Tie-2", "vram_gb": 24, "cost_per_hr": 0.25, "available": True},
            {"id": "GPU-Expensive", "vram_gb": 24, "cost_per_hr": 0.50, "available": True},
        ]
        assert select_cheapest_gpu(catalog, min_vram_gb=16)["id"] in ("GPU-Tie-1", "GPU-Tie-2")

    def test_zero_and_sub_cent_prices(self):
        catalog = [
            {"id": "GPU-SubCent", "vram_gb": 16, "cost_per_hr": 0.005, "available": True},
            {"id": "GPU-Free", "vram_gb": 16, "cost_per_hr": 0.0, "available": True},
            {"id": "GPU-Standard", "vram_gb": 16, "cost_per_hr": 0.20, "available": True},
        ]
        chosen = select_cheapest_gpu(catalog, min_vram_gb=16)
        assert chosen["id"] == "GPU-Free"
        assert chosen["cost_per_hr"] == 0.0

    @pytest.mark.parametrize("filter_kwargs,expected", [
        ({"target_gpu": "RTX A6000"}, "NVIDIA RTX A6000"),
        ({"target_gpu": "a5000"}, "NVIDIA RTX A5000"),
        ({"gpu_type": "A4000"}, A4000),
        ({"gpu_type": "  a5000  "}, "NVIDIA RTX A5000"),
        ({"gpu_type": "h100 sxm"}, "NVIDIA H100 SXM"),
    ])
    def test_type_filter_is_case_insensitive_substring_of_id_or_name(self, sample_gpu_catalog, filter_kwargs, expected):
        assert select_cheapest_gpu(sample_gpu_catalog, **filter_kwargs)["id"] == expected

    @pytest.mark.parametrize("catalog_kwargs,match", [
        ({"catalog": []}, "No qualified RunPod GPUs available"),
        ({"min_vram_gb": 128}, "min_vram_gb=128"),
        ({"target_gpu": "nonexistent_gpu"}, "target='nonexistent_gpu'"),
        ({"gpu_type": "B200"}, "target='B200'"),
        ({"max_cost": 0.10}, r"max_cost=\$0.1/hr"),
        ({"catalog": [{"id": "GPU-A", "vram_gb": 24, "cost_per_hr": 0.10, "available": False}]},
         "No qualified RunPod GPUs available"),
    ])
    def test_no_qualifying_gpu_raises_with_the_constraints(self, sample_gpu_catalog, catalog_kwargs, match):
        kwargs = dict(catalog_kwargs)
        catalog = kwargs.pop("catalog", sample_gpu_catalog)
        with pytest.raises(RuntimeError, match=match):
            select_cheapest_gpu(catalog, **kwargs)

    def test_max_cost_filters_expensive_gpus(self):
        catalog = [
            {"id": "GPU-Cheap", "vram_gb": 16, "cost_per_hr": 0.20, "available": True},
            {"id": "GPU-Expensive", "vram_gb": 16, "cost_per_hr": 1.50, "available": True},
        ]
        assert select_cheapest_gpu(catalog, min_vram_gb=16, max_cost=0.50)["id"] == "GPU-Cheap"

    def test_missing_cost_raises_key_error(self):
        catalog = [{"id": "GPU-Broken", "vram_gb": 24, "available": True},
                   {"id": "GPU-Valid", "vram_gb": 24, "cost_per_hr": 0.50, "available": True}]
        with pytest.raises(KeyError, match="cost_per_hr"):
            select_cheapest_gpu(catalog, min_vram_gb=16)

    def test_none_cost_raises_type_error(self):
        catalog = [{"id": "GPU-None", "vram_gb": 24, "cost_per_hr": None, "available": True},
                   {"id": "GPU-Valid", "vram_gb": 24, "cost_per_hr": 0.50, "available": True}]
        with pytest.raises(TypeError):
            select_cheapest_gpu(catalog, min_vram_gb=16)

    def test_http_client_drops_unpriced_gpu_types(self):
        """GraphQL entries with no price, or only null prices, never enter the catalog."""
        client = RunPodHttpClient(api_key="rpa_test_key")
        payload = {"gpuTypes": [
            {"id": "GPU-NoPrice", "displayName": "No Price", "memoryInGb": 24, "lowestPrice": None},
            {"id": "GPU-NullPrices", "displayName": "Null Price", "memoryInGb": 24,
             "lowestPrice": {"minimumBidPrice": None, "uninterruptablePrice": None}},
            {"id": "GPU-Valid", "displayName": "Valid Price", "memoryInGb": 24,
             "lowestPrice": {"minimumBidPrice": 0.35, "uninterruptablePrice": 0.45}},
        ]}
        with patch.object(client, "_execute", return_value=payload):
            parsed = client.get_gpu_catalog()
        assert [(g["id"], g["cost_per_hr"]) for g in parsed] == [("GPU-Valid", pytest.approx(0.45))]


class TestMaxCostCeiling:

    @pytest.mark.parametrize("max_cost", [0.50, 0.24])
    def test_cheapest_within_or_exactly_at_ceiling_accepted(self, sample_gpu_catalog, max_cost):
        chosen = RunPodRunner(dry_run=True, max_cost=max_cost).filter_and_select_gpu(sample_gpu_catalog)
        assert chosen["id"] == A4000

    @pytest.mark.parametrize("max_cost", [0.2399, 0.20, 0.0, -0.5])
    def test_cheapest_above_ceiling_rejected(self, sample_gpu_catalog, max_cost):
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            RunPodRunner(dry_run=True, max_cost=max_cost).filter_and_select_gpu(sample_gpu_catalog)

    def test_ceiling_violation_creates_no_pod(self):
        client = FakeRunPodClient(catalog=[{"id": "GPU-Exp", "vram_gb": 24, "cost_per_hr": 2.50, "available": True}])
        with pytest.raises(ValueError, match="exceeds --max-cost ceiling"):
            RunPodRunner(dry_run=True, max_cost=1.00, client=client).run_qualification_suite()
        assert client.created_pods == []

    def test_cli_ceiling_violation_exits_1(self, capsys):
        assert runpod_main(["--dry-run", "--max-cost", "0.05"]) == 1
        assert "exceeds --max-cost ceiling" in capsys.readouterr().err


# ==============================================================================
# 4. Pod cleanup
# ==============================================================================

class TestPodCleanup:

    def test_successful_run_terminates_its_pod(self):
        client = FakeRunPodClient(qualification_result=passing_receipt("pod"))
        runner = RunPodRunner(dry_run=True)
        receipt = runner.execute_with_lifecycle(client, A4000)
        assert receipt["status"] == "passed"
        assert len(client.created_pods) == 1
        assert client.terminated_pods == client.created_pods
        assert client.lifecycle_history == ["created", "executed", "terminated"]
        assert runner.lifecycle_history == ["created", "running", "terminated"]

    @pytest.mark.parametrize("exc_type", [RuntimeError, ConnectionResetError, ValueError, KeyError, TimeoutError])
    def test_remote_exception_terminates_the_pod_exactly_once(self, exc_type):
        class Crashing(FakeRunPodClient):
            def execute_qualification(self, pod_id):
                raise exc_type("Cloud qualification failed")

        client = Crashing()
        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(exc_type, match="Cloud qualification failed"):
            runner.run_qualification_suite()
        assert len(client.created_pods) == 1
        assert client.terminated_pods == client.created_pods
        assert runner._active_pod_id is None

    @pytest.mark.parametrize("raised", [ZeroDivisionError, KeyboardInterrupt, SystemExit])
    @pytest.mark.parametrize("via_runner", [True, False])
    def test_exception_inside_the_pod_context_terminates_it(self, raised, via_runner):
        """Exceptions and BaseExceptions (KeyboardInterrupt, SystemExit) all terminate the pod."""
        client = FakeRunPodClient()
        runner = RunPodRunner(dry_run=True, client=client)
        context = runner.ephemeral_pod(A4000) if via_runner else provision_ephemeral_pod(client, A4000)
        with pytest.raises(raised):
            with context as pod_id:
                assert pod_id in client.created_pods
                raise raised()
        assert len(client.created_pods) == 1
        assert client.all_pods_cleaned_up
        if via_runner:
            assert runner.lifecycle_history[-1] == "terminated"

    def test_sigint_during_qualification_terminates_once_and_raises(self):
        """A real SIGINT while the pod runs terminates it exactly once and still raises KeyboardInterrupt."""
        class Interrupted(FakeRunPodClient):
            def execute_qualification(self, pod_id):
                os.kill(os.getpid(), signal.SIGINT)
                raise AssertionError("SIGINT handler did not run")

        client = Interrupted()
        with pytest.raises(KeyboardInterrupt):
            RunPodRunner(dry_run=True).execute_with_lifecycle(client, A4000)
        assert len(client.terminated_pods) == 1
        assert client.terminated_pods == client.created_pods

    def test_signal_handlers_chain_to_and_restore_previous_handler(self):
        """During a lifecycle SIGTERM cleans up then calls the previous handler, which is restored after."""
        client = FakeRunPodClient()
        runner = RunPodRunner(dry_run=True, client=client)
        received = []

        def previous_handler(signum, frame):
            received.append(signum)

        original = signal.signal(signal.SIGTERM, previous_handler)
        try:
            with runner.ephemeral_pod(A4000) as pod_id:
                assert signal.getsignal(signal.SIGTERM) is not previous_handler
                os.kill(os.getpid(), signal.SIGTERM)
                assert pod_id in client.terminated_pods
            assert received == [signal.SIGTERM]
            assert signal.getsignal(signal.SIGTERM) is previous_handler
            assert client.terminated_pods == [pod_id]
        finally:
            signal.signal(signal.SIGTERM, original)

    def test_failed_termination_keeps_pod_registered_and_fails_loudly(self, forget_pods):
        client = FailingTerminateClient(confirm=True)
        runner = RunPodRunner(dry_run=True, client=client)
        with pytest.raises(PodCleanupError, match="Network failure while calling terminate_pod") as exc:
            with runner.ephemeral_pod(A4000) as pod_id:
                forget_pods.append(pod_id)
        assert exc.value.pod_id == pod_id
        assert pod_id in str(exc.value)
        # Nothing may forget a pod whose termination was not confirmed.
        assert runner._active_pod_id == pod_id
        assert _GLOBAL_POD_REGISTRY.is_registered(pod_id)
        assert "terminated" not in runner.lifecycle_history

    def test_unconfirmed_termination_is_a_failure(self, forget_pods):
        """A client that returns False from terminate_pod has not confirmed termination."""
        client = FailingTerminateClient(confirm=False)
        with pytest.raises(PodCleanupError, match="did not confirm"):
            with provision_ephemeral_pod(client, A4000) as pod_id:
                forget_pods.append(pod_id)
        assert _GLOBAL_POD_REGISTRY.is_registered(pod_id)

    def test_release_is_idempotent(self):
        client = FakeRunPodClient()
        pod_id = client.create_pod(A4000)
        _GLOBAL_POD_REGISTRY.register(pod_id, client)
        for _ in range(3):
            _GLOBAL_POD_REGISTRY.release(pod_id)
        assert client.terminated_pods == [pod_id]
        assert not _GLOBAL_POD_REGISTRY.is_registered(pod_id)

    def test_registry_cleans_up_every_active_pod(self):
        registry = _ActivePodRegistry()
        clients = [FakeRunPodClient(), FakeRunPodClient()]
        for client in clients:
            registry.register(client.create_pod(A4000), client)
        assert len(registry._active_pods) == 2
        assert registry.cleanup_all() == []
        assert registry._active_pods == {}
        assert all(client.all_pods_cleaned_up for client in clients)

    def test_registry_keeps_going_past_a_failing_client(self):
        """cleanup_all terminates the other pods, reports the failure and keeps that pod registered."""
        registry = _ActivePodRegistry()
        bad, good1, good2 = FakeRunPodClient(fail_at="terminate_pod"), FakeRunPodClient(), FakeRunPodClient()
        pods = {}
        for name, client in (("bad", bad), ("good1", good1), ("good2", good2)):
            pods[name] = client.create_pod(A4000)
            registry.register(pods[name], client)
        try:
            assert registry.cleanup_all() == [pods["bad"]]
            assert good1.terminated_pods == [pods["good1"]]
            assert good2.terminated_pods == [pods["good2"]]
            assert registry.is_registered(pods["bad"])
            assert not registry.is_registered(pods["good1"])
        finally:
            registry.unregister(pods["bad"])

    @pytest.mark.parametrize("ending", ["SIGTERM", "SIGINT", "sys.exit"])
    def test_child_process_cleans_up_on_signal_or_exit(self, tmp_path, ending):
        """A real child process with a registered pod terminates it on SIGTERM, SIGINT and atexit."""
        proof_file, ready_file = tmp_path / "terminated.txt", tmp_path / "ready.txt"
        child_code = textwrap.dedent(f"""
            import sys, time
            from hypergan.runpod_runner import _GLOBAL_POD_REGISTRY

            class ChildClient:
                def terminate_pod(self, pod_id):
                    with open({str(proof_file)!r}, "w") as f:
                        f.write("TERMINATED:" + pod_id)
                    return True

            _GLOBAL_POD_REGISTRY.register("child-pod", ChildClient())
            if {ending!r} == "sys.exit":
                sys.exit(0)
            open({str(ready_file)!r}, "w").write("READY")
            while True:
                time.sleep(0.1)
        """)
        proc = subprocess.Popen([sys.executable, "-c", child_code])
        try:
            if ending != "sys.exit":
                deadline = time.monotonic() + 10
                while not ready_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert ready_file.exists(), "Child process did not become ready"
                os.kill(proc.pid, getattr(signal, ending))
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proof_file.read_text() == "TERMINATED:child-pod"


# ==============================================================================
# 5. Delegation to execute_qualification
# ==============================================================================

class TestDelegation:

    def test_complete_remote_receipt_is_returned_unchanged(self, tmp_path):
        """A complete receipt, custom telemetry and suites included, passes through untouched and is written."""
        remote = build_fake_receipt(
            qualification_id="qual-remote",
            suites=[{"name": "single_device", "status": "passed", "duration_seconds": 0.05},
                    {"name": "custom_nccl_stress", "status": "passed", "duration_seconds": 1.2}],
            custom_telemetry={"gpus_tested": 2, "link_type": "nvlink"},
        )
        client = FakeRunPodClient(qualification_result=remote)
        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)
        assert receipt == remote
        assert receipt["receipt_sha256"] == compute_receipt_sha256(receipt)
        assert client.executed_pods == client.created_pods == client.terminated_pods
        assert json.loads((tmp_path / "qualification-receipt.json").read_text()) == remote

    def test_receipt_comes_from_the_pod_that_ran(self, tmp_path):
        client = FakeRunPodClient()
        receipt = RunPodRunner(dry_run=True, client=client).run_qualification_suite(output_dir=tmp_path)
        (pod_id,) = client.created_pods
        assert receipt["qualification_id"] == f"qual-{pod_id}"
        assert client.all_pods_cleaned_up

    @pytest.mark.parametrize("partial", [
        {},
        {"status": "passed"},
        {"status": "passed", "suites": [{"name": "custom_nccl_stress", "status": "passed", "duration_seconds": 1.2}]},
    ])
    def test_incomplete_remote_receipt_is_rejected_not_normalized(self, tmp_path, partial):
        """An incomplete receipt is an error; the runner never invents suites, status or environment."""
        client = FakeRunPodClient(qualification_result=partial)
        with pytest.raises(ReceiptValidationError, match="missing required keys"):
            RunPodRunner(dry_run=True, client=client).run_qualification_suite(output_dir=tmp_path)
        assert not (tmp_path / "qualification-receipt.json").exists()
        assert len(client.terminated_pods) == 1
        assert client.all_pods_cleaned_up

    def test_remote_receipt_contradicting_its_suites_is_rejected(self, tmp_path):
        """A receipt claiming passed while a suite failed is rejected even with a matching checksum."""
        remote = build_fake_receipt(status="passed",
                                    suites=[{"name": "cuda_replicated_nccl", "status": "failed", "duration_seconds": 1.0}])
        client = FakeRunPodClient(qualification_result=remote)
        with pytest.raises(ReceiptValidationError, match="contradicts its suites"):
            RunPodRunner(dry_run=True, client=client).run_qualification_suite(output_dir=tmp_path)
        assert client.all_pods_cleaned_up

    @pytest.mark.parametrize("non_dict", ["completed", [1, 2, 3], True])
    def test_non_dict_result_is_rejected_after_terminating_the_pod(self, tmp_path, non_dict):
        client = FakeRunPodClient(qualification_result=non_dict)
        with pytest.raises(ValueError, match="not a receipt object"):
            RunPodRunner(dry_run=True, client=client).run_qualification_suite(output_dir=tmp_path)
        assert len(client.created_pods) == 1
        assert client.all_pods_cleaned_up

    def test_none_result_is_rejected(self, tmp_path):
        class NoneClient(FakeRunPodClient):
            def execute_qualification(self, pod_id):
                super().execute_qualification(pod_id)
                return None

        client = NoneClient()
        with pytest.raises(ValueError, match="not a receipt object"):
            RunPodRunner(dry_run=True, client=client).run_qualification_suite(output_dir=tmp_path)
        assert client.all_pods_cleaned_up

    @pytest.mark.parametrize("execute", ["missing", None, "not_callable"])
    @pytest.mark.parametrize("dry_run", [True, False])
    def test_client_that_cannot_run_qualification_is_refused_before_any_pod(self, tmp_path, execute, dry_run):
        client = CreateOnlyClient()
        if execute != "missing":
            client.execute_qualification = execute  # type: ignore[assignment]
        runner = RunPodRunner(dry_run=dry_run, client=client)
        with pytest.raises(LiveModeUnavailable, match="cannot run qualification on a pod"):
            runner.run_qualification_suite(output_dir=tmp_path)
        assert client.created_pods == []
        assert not (tmp_path / "qualification-receipt.json").exists()


# ==============================================================================
# 6. Dry-run receipts
# ==============================================================================

class TestDryRunReceipt:

    def test_dry_run_plans_without_a_pod_and_reports_nothing_passed(self, tmp_path):
        receipt = RunPodRunner(dry_run=True).run_qualification_suite(output_dir=tmp_path)
        validate_qualification_receipt(receipt)
        assert receipt["schema_version"] == 1
        assert receipt["status"] == "dry_run"
        assert len(receipt["suites"]) >= 4
        assert {s["status"] for s in receipt["suites"]} == {"dry_run"}
        runpod = receipt["environment"]["runpod"]
        assert runpod["dry_run"] is True
        assert runpod["pod_created"] is False
        assert runpod["gpu_type"] == A4000
        assert json.loads((tmp_path / "qualification-receipt.json").read_text()) == receipt

    def test_dry_run_records_the_runtime_torch_version(self, tmp_path):
        import torch
        receipt = RunPodRunner(dry_run=True).run_qualification_suite(output_dir=tmp_path)
        assert receipt["environment"]["torch"] == torch.__version__

    @pytest.mark.parametrize("torch_version,device_count", [
        ("2.14.1+cu130", 1), ("2.5.1", 1), ("2.4.0+rocm6.0", 1), ("2.6.0.dev20241024", 1), ("2.1.0+cpu", 1),
        ("2.9.9+dynamic.test", 1), ("none", 0),
    ])
    def test_dry_run_records_the_inspected_environment_verbatim(self, tmp_path, torch_version, device_count):
        """The dry-run receipt records what inspect_environment saw; it substitutes nothing."""
        seen = {"torch": torch_version, "platform": sys.platform, "python": platform.python_version(),
                "device_count": device_count}
        with patch("hypergan.qualify.inspect_environment", return_value=seen):
            receipt = RunPodRunner(dry_run=True).run_qualification_suite(output_dir=tmp_path)
        assert receipt["status"] == "dry_run"
        assert receipt["environment"]["torch"] == torch_version
        assert receipt["environment"]["device_count"] == device_count
        validate_qualification_receipt(receipt)


# ==============================================================================
# 7. CLI
# ==============================================================================

class TestRunPodCLI:

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_lists_every_flag(self, flag):
        proc = subprocess.run([sys.executable, "-m", "hypergan.runpod_runner", flag],
                              capture_output=True, text=True, timeout=15)
        assert proc.returncode == 0, proc.stderr
        for text in ("--dry-run", "--gpu-type", "--max-cost", "--timeout", "--output", "-v, --verbose", "--json",
                     "RUNPOD_API_KEY"):
            assert text in proc.stdout

    @pytest.mark.parametrize("argv,attr,expected", [
        ([], "verbose", False), (["-v"], "verbose", True), (["--verbose"], "verbose", True),
        ([], "json", False), (["--json"], "json", True),
    ])
    def test_parser_flags(self, argv, attr, expected):
        assert getattr(_build_parser().parse_args(argv), attr) is expected

    def test_unknown_argument_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args(["--dry-run", "--bogus-flag"])
        assert exc.value.code == 2
        assert "unrecognized arguments: --bogus-flag" in capsys.readouterr().err

    def test_api_key_in_argv_rejected(self):
        """The key comes from RUNPOD_API_KEY only; --api-key would leak it into ps."""
        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args(["--dry-run", "--api-key", "rpa_secret"])
        assert exc.value.code == 2

    @pytest.mark.parametrize("flags", [[], ["-v"], ["--verbose"]])
    def test_dry_run_text_output(self, tmp_path, capsys, flags):
        out_file = tmp_path / "receipt.json"
        assert runpod_main(["--dry-run", "--output", str(out_file), *flags]) == 0
        out = capsys.readouterr().out
        assert "Qualification status: dry_run" in out
        assert f"Receipt written to: {out_file}" in out
        assert ("[runpod]" in out) is bool(flags)
        assert ("single_device: dry_run" in out) is bool(flags)
        assert json.loads(out_file.read_text())["status"] == "dry_run"

    @pytest.mark.parametrize("verbose", [False, True])
    def test_json_keeps_stdout_pure_and_logs_on_stderr(self, tmp_path, capsys, verbose):
        out_file = tmp_path / "receipt.json"
        assert runpod_main(["--dry-run", "--json", "--output", str(out_file)] + (["-v"] if verbose else [])) == 0
        captured = capsys.readouterr()
        receipt = json.loads(captured.out)
        validate_qualification_receipt(receipt)
        assert receipt["status"] == "dry_run"
        assert json.loads(out_file.read_text()) == receipt
        assert "[runpod]" not in captured.out
        assert ("Starting ephemeral qualification" in captured.err) is verbose

    def test_default_output_goes_to_state_dir_not_cwd(self, tmp_path, monkeypatch, capsys):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setenv("HYPERGAN_STATE_DIR", str(tmp_path / "state"))
        assert runpod_main(["--dry-run"]) == 0
        assert list(work.iterdir()) == []
        (receipt_file,) = (tmp_path / "state" / "qualification").glob("qualification-receipt-*.json")
        assert str(receipt_file) in capsys.readouterr().out

    def test_live_mode_refused_with_exit_code_2(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test_secret")
        out_file = tmp_path / "live.json"
        assert runpod_main(["--output", str(out_file)]) == 2
        assert "not implemented" in capsys.readouterr().err
        assert not out_file.exists()

    @pytest.mark.heavy
    def test_subprocess_json_verbose_streams_are_separate(self, tmp_path):
        """As a real process piped to a consumer: stdout is one JSON document, the logs go to stderr."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan.runpod_runner", "--dry-run", "--json", "-v"],
            capture_output=True, text=True, timeout=60, cwd=tmp_path,
            env={**os.environ, "HYPERGAN_STATE_DIR": str(tmp_path / "state")},
        )
        assert proc.returncode == 0, proc.stderr
        validate_qualification_receipt(json.loads(proc.stdout))
        assert "[runpod]" in proc.stderr and "[runpod]" not in proc.stdout
        assert not (tmp_path / "qualification-receipt.json").exists()
        assert len(list((tmp_path / "state" / "qualification").glob("qualification-receipt-*.json"))) == 1

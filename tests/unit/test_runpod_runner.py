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

import json
import os
import signal
import subprocess
import sys
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from hypergan.runpod_runner import (
    LiveModeUnavailable,
    RunPodRunner,
    _GLOBAL_POD_REGISTRY,
    _build_parser,
    main as runpod_main,
    select_cheapest_gpu,
    validate_timeout_guardrail,
)
from tests.runpod_fakes import CreateOnlyClient, build_fake_receipt


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
        return build_fake_receipt(qualification_id=f"qual-{pod_id}", duration_seconds=25.0,
                                  suites=[{"name": "smoke", "status": "passed", "duration_seconds": 25.0}])

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
        assert runner.verbose is False

    def test_init_verbose_option(self):
        """Verbose option defaults to False and can be explicitly configured."""
        runner_default = RunPodRunner(dry_run=True)
        assert runner_default.verbose is False

        runner_verbose = RunPodRunner(dry_run=True, verbose=True)
        assert runner_verbose.verbose is True

    def test_init_custom_parameters(self):
        """Custom parameters are stored in runner instance."""
        runner = RunPodRunner(
            timeout=120,
            max_cost=0.50,
            gpu_type="NVIDIA RTX A5000",
            dry_run=True,
        )
        assert runner.timeout == 120
        assert runner.max_cost == pytest.approx(0.50)
        assert runner.gpu_type == "NVIDIA RTX A5000"
        assert runner.dry_run is True
        assert runner.client is None

    @pytest.mark.parametrize("api_key", [None, "rpa_test_secret"])
    def test_init_live_mode_refused(self, monkeypatch, api_key):
        """Live mode is refused, with or without a key, because nothing can run on the pod."""
        if api_key:
            monkeypatch.setenv("RUNPOD_API_KEY", api_key)
        else:
            monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(LiveModeUnavailable, match="refuses to create a paid pod"):
            RunPodRunner(dry_run=False)

    def test_init_leaves_signal_handlers_untouched(self):
        """Constructing a runner must not replace the process's SIGINT/SIGTERM handlers."""
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        RunPodRunner(dry_run=True)
        RunPodRunner(dry_run=True, client=MockCloudClient())
        assert {sig: signal.getsignal(sig) for sig in before} == before

    def test_init_dry_run_without_api_key_succeeds(self, monkeypatch):
        """Dry-run runner does not require RUNPOD_API_KEY."""
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        runner = RunPodRunner(dry_run=True)
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

    def test_cleanup_on_real_sigint_during_qualification(self):
        """A SIGINT while the pod runs terminates it exactly once and still raises KeyboardInterrupt."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True)

        def interrupted(pod_id):
            os.kill(os.getpid(), signal.SIGINT)
            raise AssertionError("SIGINT handler did not run")

        with patch.object(client, "execute_qualification", side_effect=interrupted):
            with pytest.raises(KeyboardInterrupt):
                runner.execute_with_lifecycle(client, "NVIDIA RTX A4000")
        assert client.terminated_pods == client.created_pods
        assert len(client.terminated_pods) == 1

    def test_signal_handlers_chain_to_and_restore_previous_handler(self):
        """During a lifecycle SIGTERM cleans up then calls the previous handler, which is restored after."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True, client=client)
        received = []

        def previous_handler(signum, frame):
            received.append(signum)

        original = signal.signal(signal.SIGTERM, previous_handler)
        try:
            with runner.ephemeral_pod("NVIDIA RTX A4000") as pod_id:
                assert signal.getsignal(signal.SIGTERM) is not previous_handler
                os.kill(os.getpid(), signal.SIGTERM)
                assert pod_id in client.terminated_pods
            assert received == [signal.SIGTERM]
            assert signal.getsignal(signal.SIGTERM) is previous_handler
            assert client.terminated_pods == [pod_id]
        finally:
            signal.signal(signal.SIGTERM, original)

    def test_cleanup_idempotent(self):
        """Releasing a pod twice terminates it once and does not error."""
        client = MockCloudClient()
        pod_id = client.create_pod("RTX A4000", "hypergan:latest", 180)
        _GLOBAL_POD_REGISTRY.register(pod_id, client)
        _GLOBAL_POD_REGISTRY.release(pod_id)
        _GLOBAL_POD_REGISTRY.release(pod_id)  # Idempotent second call
        assert client.terminated_pods == [pod_id]
        assert not _GLOBAL_POD_REGISTRY.is_registered(pod_id)


# ==============================================================================
# 6. Dry-Run Lifecycle & Receipt Generation
# ==============================================================================

class TestRunPodDryRunLifecycle:
    """Verify complete dry-run execution and JSON receipt emission."""

    def test_dry_run_executes_full_lifecycle(self, tmp_path):
        """Dry-run executes mock cloud qualification without network."""
        from hypergan.qualify import validate_qualification_receipt
        runner = RunPodRunner(dry_run=True)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)
        assert receipt["status"] == "dry_run"
        assert {s["status"] for s in receipt["suites"]} == {"dry_run"}
        assert receipt["environment"]["runpod"]["pod_created"] is False
        assert receipt["duration_seconds"] <= 180.0
        validate_qualification_receipt(receipt)

        receipt_file = tmp_path / "qualification-receipt.json"
        assert receipt_file.exists()
        saved = json.loads(receipt_file.read_text())
        assert saved["receipt_sha256"] == receipt["receipt_sha256"]

    def test_dry_run_receipt_dynamically_populates_torch_version(self, tmp_path):
        """Mock receipt environment['torch'] matches runtime torch.__version__."""
        import torch
        runner = RunPodRunner(dry_run=True)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)
        assert receipt["status"] == "dry_run"
        assert receipt["environment"]["torch"] == torch.__version__

    def test_dry_run_receipt_dynamic_torch_version_mocked(self, tmp_path):
        """Receipt torch version dynamically reflects mocked torch.__version__ or inspect_environment."""
        runner = RunPodRunner(dry_run=True)
        with patch("hypergan.qualify.inspect_environment", return_value={"torch": "2.9.9+dynamic.test", "platform": sys.platform, "python": "3.14", "device_count": 1}):
            receipt = runner.run_qualification_suite(output_dir=tmp_path)
            assert receipt["environment"]["torch"] == "2.9.9+dynamic.test"

    def test_run_qualification_suite_delegates_to_client_execute_qualification(self, tmp_path):
        """When client implements execute_qualification, run_qualification_suite delegates to it."""
        client = MockCloudClient()
        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        assert receipt["status"] == "passed"
        assert len(client.created_pods) == 1
        pod_id = client.created_pods[0]
        assert receipt["qualification_id"] == f"qual-{pod_id}"
        assert client.all_pods_cleaned_up
        assert pod_id in client.terminated_pods

    def test_client_without_execute_qualification_is_refused_before_pod_creation(self, tmp_path):
        """A client that cannot run qualification never gets to create a pod or a receipt."""
        client = CreateOnlyClient()
        runner = RunPodRunner(dry_run=False, client=client)
        with pytest.raises(LiveModeUnavailable, match="refusing to create one"):
            runner.run_qualification_suite(output_dir=tmp_path)
        assert client.created_pods == []
        assert not (tmp_path / "qualification-receipt.json").exists()

    def test_dry_run_never_reports_a_suite_passed(self, tmp_path):
        """The dry-run receipt claims nothing about hardware it did not touch."""
        receipt = RunPodRunner(dry_run=True).run_qualification_suite(output_dir=tmp_path)
        assert all(s["status"] == "dry_run" for s in receipt["suites"])
        assert receipt["environment"]["runpod"]["dry_run"] is True


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

    def test_parser_accepts_verbose_short_and_long_flag(self):
        """Parser accepts both -v and --verbose flags, defaulting to False."""
        parser = _build_parser()

        args_default = parser.parse_args([])
        assert args_default.verbose is False

        args_short = parser.parse_args(["-v"])
        assert args_short.verbose is True

        args_long = parser.parse_args(["--verbose"])
        assert args_long.verbose is True

    def test_parser_accepts_json_flag(self):
        """Parser accepts --json flag, defaulting to False."""
        parser = _build_parser()

        args_default = parser.parse_args([])
        assert args_default.json is False

        args_json = parser.parse_args(["--json"])
        assert args_json.json is True

    def test_runpod_runner_cli_dry_run_with_verbose_short_flag(self, tmp_path):
        """CLI command python3 -m hypergan.runpod_runner --dry-run --output <path> -v exits with code 0."""
        out_file = tmp_path / "runpod_receipt_verbose_short.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(out_file),
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"runpod_runner -v failed:\n{proc.stderr}"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["status"] == "dry_run"

    def test_runpod_runner_cli_dry_run_with_verbose_long_flag(self, tmp_path):
        """CLI command with --verbose flag exits with code 0 and emits receipt."""
        out_file = tmp_path / "runpod_receipt_verbose_long.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(out_file),
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"runpod_runner --verbose failed:\n{proc.stderr}"
        assert out_file.exists()

    def test_runpod_runner_cli_dry_run_with_json_flag(self, tmp_path):
        """CLI command with --json flag emits valid receipt JSON to stdout."""
        out_file = tmp_path / "runpod_receipt_json.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(out_file),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"runpod_runner --json failed:\n{proc.stderr}"
        assert out_file.exists()
        stdout_receipt = json.loads(proc.stdout)
        assert stdout_receipt["schema_version"] == 1
        assert stdout_receipt["status"] == "dry_run"
        assert "receipt_sha256" in stdout_receipt
        assert "environment" in stdout_receipt

    def test_runpod_runner_cli_dry_run_with_json_and_verbose_flag(self, tmp_path):
        """CLI command with both --json and -v flags keeps stdout valid JSON and logs to stderr."""
        out_file = tmp_path / "runpod_receipt_json_v.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(out_file),
                "--json",
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"runpod_runner --json -v failed:\n{proc.stderr}"
        assert out_file.exists()
        stdout_receipt = json.loads(proc.stdout)
        assert stdout_receipt["status"] == "dry_run"
        assert "[runpod]" in proc.stderr

    def test_main_direct_invocation_processes_flags(self, tmp_path, capsys):
        """Direct call to main() processes -v and --json correctly."""
        out_file = tmp_path / "main_direct.json"
        rc = runpod_main(["--dry-run", "-v", "--json", "--output", str(out_file)])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "dry_run"
        assert data["schema_version"] == 1

    def test_cli_default_output_goes_to_state_dir_not_cwd(self, tmp_path, monkeypatch, capsys):
        """Without --output the receipt lands under the state directory, never the cwd."""
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setenv("HYPERGAN_STATE_DIR", str(tmp_path / "state"))
        assert runpod_main(["--dry-run"]) == 0
        assert list(work.iterdir()) == []
        receipts = list((tmp_path / "state" / "qualification").glob("qualification-receipt-*.json"))
        assert len(receipts) == 1
        assert str(receipts[0]) in capsys.readouterr().out

    def test_cli_live_mode_refused_with_exit_code_2(self, tmp_path, capsys, monkeypatch):
        """Without --dry-run the CLI refuses (exit 2) and writes nothing."""
        monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test_secret")
        out_file = tmp_path / "live.json"
        rc = runpod_main(["--output", str(out_file)])
        assert rc == 2
        assert "not implemented" in capsys.readouterr().err
        assert not out_file.exists()

    def test_cli_rejects_api_key_in_argv(self, capsys):
        """The key comes from RUNPOD_API_KEY only; --api-key would leak it into ps."""
        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args(["--dry-run", "--api-key", "rpa_secret"])
        assert exc.value.code == 2

"""Empirical adversarial challenge suite for Milestone 4 Iteration 2.

Focuses specifically on:
1. CLI option handling (-v, --verbose, --json, combined --json -v).
2. Pipe behavior to jq and stream isolation (stdout pure JSON, diagnostic logs on stderr).
3. CLI unknown argument rejection and help flag output.
4. Receipt dynamic environment detection under simulated / monkeypatched runtime versions.
5. Receipt SHA-256 cryptographic integrity and tamper detection across all fields.
6. Validation of receipts generated through dry-run and live runner client delegation paths.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from hypergan.runpod_runner import (
    RunPodRunner,
    _build_parser,
    main as runpod_main,
)
from hypergan.qualify import (
    compute_receipt_sha256,
    validate_qualification_receipt,
)


# ==============================================================================
# 1. Adversarial CLI Option Handling & Stream Isolation
# ==============================================================================

class TestRunPodCLIOptionsAdversarial:
    """Stress-test CLI option parsing, stream separation, and pipe behavior."""

    def test_cli_short_flag_v(self, tmp_path):
        """CLI flag -v enables verbose logging and writes receipt successfully."""
        receipt_file = tmp_path / "receipt_v.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(receipt_file),
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Failed with stderr: {proc.stderr}"
        assert receipt_file.exists()
        # Non-JSON verbose output is written to stdout
        assert "[runpod]" in proc.stdout
        assert "Qualification status: passed" in proc.stdout
        assert "single_device" in proc.stdout

    def test_cli_long_flag_verbose(self, tmp_path):
        """CLI flag --verbose enables verbose logging identically to -v."""
        receipt_file = tmp_path / "receipt_verbose.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(receipt_file),
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Failed with stderr: {proc.stderr}"
        assert receipt_file.exists()
        assert "[runpod]" in proc.stdout
        assert "Receipt written to:" in proc.stdout

    def test_cli_json_flag_stdout_purity(self, tmp_path):
        """CLI flag --json outputs ONLY valid JSON on stdout parseable by json.loads."""
        receipt_file = tmp_path / "receipt_json.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(receipt_file),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Failed with stderr: {proc.stderr}"
        # Direct parse of stdout must succeed without stripping or filtering
        parsed = json.loads(proc.stdout)
        assert isinstance(parsed, dict)
        assert parsed["status"] == "passed"
        assert parsed["schema_version"] == 1
        assert "receipt_sha256" in parsed
        validate_qualification_receipt(parsed)

    def test_cli_combined_json_and_verbose_pipe_to_jq(self, tmp_path):
        """Combined --json -v pipe to jq: stdout has zero corrupting logs, stderr has logs."""
        jq_path = shutil.which("jq")
        if not jq_path:
            pytest.skip("jq is not installed on this system")

        receipt_file = tmp_path / "receipt_json_v_pipe.json"
        # Run pipeline: python3 -m hypergan.runpod_runner ... | jq .status
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hypergan.runpod_runner",
                "--dry-run",
                "--output",
                str(receipt_file),
                "--json",
                "-v",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        p2 = subprocess.Popen(
            [jq_path, ".status"],
            stdin=p1.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits

        jq_out, jq_err = p2.communicate(timeout=30)
        p1_out, p1_err = p1.communicate(timeout=30)

        assert p1.returncode == 0, f"p1 failed: {p1_err}"
        assert p2.returncode == 0, f"jq failed: {jq_err}"
        assert jq_out.strip() == '"passed"'

        # Diagnostic logs MUST be present in stderr, NEVER in stdout
        assert "[runpod]" in p1_err
        assert "Starting ephemeral qualification" in p1_err
        assert "[runpod]" not in (p1_out or "")

    def test_cli_unknown_argument_rejection(self):
        """Passing unrecognized arguments exits with code 2 and standard error message."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan.runpod_runner", "--dry-run", "--bogus-flag"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 2
        assert "unrecognized arguments: --bogus-flag" in proc.stderr

    def test_cli_help_flag_output(self):
        """CLI flags -h and --help exit with code 0 and describe -v, --verbose, and --json."""
        for flag in ("-h", "--help"):
            proc = subprocess.run(
                [sys.executable, "-m", "hypergan.runpod_runner", flag],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert proc.returncode == 0
            assert "-v, --verbose" in proc.stdout
            assert "--json" in proc.stdout
            assert "--dry-run" in proc.stdout


# ==============================================================================
# 2. Receipt Dynamic Environment Simulation & Version Probing
# ==============================================================================

class TestReceiptDynamicEnvironmentSimulation:
    """Stress-test receipt dynamic environment detection under simulated / mock environments."""

    @pytest.mark.parametrize("mock_torch_version", [
        "2.14.1+cu130",
        "2.5.1",
        "2.4.0+rocm6.0",
        "2.6.0.dev20241024",
        "2.1.0+cpu",
    ])
    def test_receipt_populates_simulated_torch_version(self, mock_torch_version, tmp_path):
        """Mock receipt environment['torch'] strictly reflects mocked torch version."""
        runner = RunPodRunner(dry_run=True)
        with patch("hypergan.runpod_runner.inspect_environment", return_value={
            "torch": mock_torch_version,
            "platform": sys.platform,
            "python": platform.python_version(),
            "device_count": 1,
            "devices": ["NVIDIA RTX A4000"],
            "cuda": {"available": True, "nccl": True, "version": "13.0"},
        }):
            receipt = runner.run_qualification_suite(output_dir=tmp_path)
            assert receipt["status"] == "passed"
            assert receipt["environment"]["torch"] == mock_torch_version
            validate_qualification_receipt(receipt)

    def test_receipt_torch_fallback_when_inspect_returns_none(self, tmp_path):
        """When inspect_environment returns 'none', runner falls back to torch.__version__."""
        runner = RunPodRunner(dry_run=True)
        fake_torch = MagicMock(__version__="2.9.9+fallback")
        with patch("hypergan.runpod_runner.inspect_environment", return_value={"torch": "none"}), \
             patch.dict("sys.modules", {"torch": fake_torch}):
            receipt = runner.run_qualification_suite(output_dir=tmp_path)
            assert receipt["environment"]["torch"] == "2.9.9+fallback"
            validate_qualification_receipt(receipt)

    def test_live_runner_delegation_with_custom_client(self, tmp_path):
        """Live runner path with mock client delegating execute_qualification verifies envelope."""
        class CustomClient:
            def __init__(self):
                self.created_pods = []
                self.terminated_pods = []

            def get_gpu_catalog(self):
                from hypergan.runpod_runner import DEFAULT_GPU_CATALOG
                return DEFAULT_GPU_CATALOG

            def create_pod(self, gpu_type, image, timeout_seconds):
                pod_id = "custom-pod-123"
                self.created_pods.append(pod_id)
                return pod_id

            def execute_qualification(self, pod_id):
                # Return partial custom receipt
                return {
                    "qualification_id": f"qual-{pod_id}",
                    "status": "passed",
                    "suites": [
                        {"name": "single_device", "status": "passed", "duration_seconds": 0.05},
                        {"name": "custom_remote_suite", "status": "passed", "duration_seconds": 0.12},
                    ],
                }

            def terminate_pod(self, pod_id):
                self.terminated_pods.append(pod_id)
                return True

        client = CustomClient()
        runner = RunPodRunner(dry_run=True, client=client)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)

        assert receipt["qualification_id"] == "qual-custom-pod-123"
        assert receipt["status"] == "passed"
        assert receipt["schema_version"] == 1
        assert "environment" in receipt
        assert "receipt_sha256" in receipt
        assert len(receipt["suites"]) == 2
        assert client.created_pods == client.terminated_pods

        # Must pass schema validation
        validate_qualification_receipt(receipt)


# ==============================================================================
# 3. Cryptographic Integrity & Comprehensive Receipt Tamper Detection
# ==============================================================================

class TestReceiptCryptographicIntegrityAdversarial:
    """Empirically test that ANY mutation of receipt fields breaks receipt_sha256."""

    @pytest.fixture
    def base_receipt(self, tmp_path) -> Dict[str, Any]:
        runner = RunPodRunner(dry_run=True)
        receipt = runner.run_qualification_suite(output_dir=tmp_path)
        validate_qualification_receipt(receipt)
        return receipt

    def test_tamper_environment_platform(self, base_receipt):
        """Modifying environment['platform'] invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["platform"] = "freebsd_injected"
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_environment_python(self, base_receipt):
        """Modifying environment['python'] invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["python"] = "3.99.9"
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_environment_torch(self, base_receipt):
        """Modifying environment['torch'] invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["torch"] = "1.0.0.tampered"
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_environment_device_count(self, base_receipt):
        """Modifying environment['device_count'] invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["device_count"] = 8
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_environment_devices_list(self, base_receipt):
        """Modifying environment['devices'] invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["devices"] = ["NVIDIA H100 80GB"]
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_environment_cuda_metadata(self, base_receipt):
        """Modifying environment['cuda'] subfields invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["cuda"]["version"] = "99.0"
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_environment_runpod_metadata(self, base_receipt):
        """Modifying environment['runpod']['dry_run'] invalidates SHA-256 and fails validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["environment"]["runpod"]["dry_run"] = False
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_status(self, base_receipt):
        """Flipping status 'passed' -> 'failed' invalidates SHA-256."""
        mutated = copy.deepcopy(base_receipt)
        mutated["status"] = "failed"
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_invalid_status_value(self, base_receipt):
        """An invalid status string ('unknown', 'running') is rejected by validation."""
        mutated = copy.deepcopy(base_receipt)
        mutated["status"] = "in_progress"
        mutated["receipt_sha256"] = compute_receipt_sha256(mutated)
        with pytest.raises(AssertionError, match="Invalid status: in_progress"):
            validate_qualification_receipt(mutated)

    def test_tamper_duration_seconds(self, base_receipt):
        """Mutating duration_seconds by epsilon invalidates SHA-256."""
        mutated = copy.deepcopy(base_receipt)
        mutated["duration_seconds"] = round(base_receipt["duration_seconds"] + 0.01, 3)
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_duration_exceeding_hard_limit(self, base_receipt):
        """A duration > 180.0 is rejected by validation even with recomputed hash."""
        mutated = copy.deepcopy(base_receipt)
        mutated["duration_seconds"] = 180.1
        mutated["receipt_sha256"] = compute_receipt_sha256(mutated)
        with pytest.raises(AssertionError, match="Qualification exceeded 3-minute hard limit"):
            validate_qualification_receipt(mutated)

    def test_tamper_suite_durations(self, base_receipt):
        """Modifying any individual suite duration invalidates SHA-256."""
        for idx in range(len(base_receipt["suites"])):
            mutated = copy.deepcopy(base_receipt)
            mutated["suites"][idx]["duration_seconds"] += 0.05
            assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
            with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
                validate_qualification_receipt(mutated)

    def test_tamper_suite_name_or_status(self, base_receipt):
        """Changing a suite's name or status invalidates SHA-256."""
        mutated = copy.deepcopy(base_receipt)
        mutated["suites"][0]["status"] = "failed"
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_suite_ordering(self, base_receipt):
        """Reordering suites list invalidates SHA-256 (suites order is semantic)."""
        mutated = copy.deepcopy(base_receipt)
        mutated["suites"] = list(reversed(mutated["suites"]))
        assert compute_receipt_sha256(mutated) != base_receipt["receipt_sha256"]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    def test_tamper_sha_digest_corruption(self, base_receipt):
        """Corrupting the receipt_sha256 string itself raises mismatch."""
        mutated = copy.deepcopy(base_receipt)
        original_hash = base_receipt["receipt_sha256"]
        # Flip first character (e.g. 0 -> 1 or a -> b)
        corrupted_char = "0" if original_hash[0] != "0" else "1"
        mutated["receipt_sha256"] = corrupted_char + original_hash[1:]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(mutated)

    @pytest.mark.parametrize("missing_key", [
        "schema_version",
        "qualification_id",
        "timestamp",
        "status",
        "duration_seconds",
        "environment",
        "suites",
        "receipt_sha256",
    ])
    def test_missing_required_keys(self, base_receipt, missing_key):
        """Omitting any required top-level key causes validation failure."""
        mutated = copy.deepcopy(base_receipt)
        del mutated[missing_key]
        with pytest.raises(AssertionError, match="Qualification receipt missing required keys"):
            validate_qualification_receipt(mutated)

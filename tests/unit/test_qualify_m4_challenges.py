"""Adversarial stress-test suite for Milestone 4 qualification receipt and CLI.

Empirically challenges commit ec043036:
1. Receipt canonical SHA-256 calculation & comprehensive tamper detection across
   all top-level fields, nested environment properties, suite statuses, metrics,
   durations, timestamps, suite reordering, and unexpected injected fields.
2. Failure modes and exit codes: suite execution failures, timeouts, invalid CLI
   options, unwriteable output paths, and ensures status="failed" with valid receipt.
3. Suite filtering: isolation of single suites, skipping unrequested suites,
   case-insensitive handling, and rejection of invalid suite names.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from hypergan.qualify import (
    QualificationRunner,
    compute_receipt_sha256,
    run_qualification,
    validate_qualification_receipt,
)


class TestReceiptAdversarialTamperChallenges:
    """Stress-test receipt SHA-256 tamper sensitivity and schema validation."""

    @pytest.fixture
    def base_receipt(self, tmp_path):
        """Generate a pristine dry-run receipt."""
        receipt_path = tmp_path / "base_receipt.json"
        receipt = run_qualification(output=receipt_path, dry_run=True)
        validate_qualification_receipt(receipt)
        return receipt

    def test_top_level_field_mutations_detected(self, base_receipt):
        """Mutating any top-level key must invalidate receipt_sha256 or fail schema validation."""
        for key in base_receipt:
            if key == "receipt_sha256":
                continue
            tampered = copy.deepcopy(base_receipt)
            val = tampered[key]
            if isinstance(val, str):
                tampered[key] = val + "_tampered"
            elif isinstance(val, (int, float)):
                tampered[key] = val + 0.1
            elif isinstance(val, list):
                tampered[key] = val + [{"name": "injected", "status": "passed", "duration_seconds": 0.01}]
            elif isinstance(val, dict):
                tampered[key]["tampered_sub"] = "bad"

            with pytest.raises(AssertionError):
                validate_qualification_receipt(tampered)

    def test_environment_field_mutations_detected(self, base_receipt):
        """Mutating nested environment metadata must invalidate receipt_sha256."""
        mutations = [
            ("platform", "freebsd_unsupported"),
            ("python", "3.9.99"),
            ("torch", "0.4.0"),
            ("device_count", 999),
            ("devices", ["Tampered GPU 0"]),
        ]
        for key, val in mutations:
            tampered = copy.deepcopy(base_receipt)
            tampered["environment"][key] = val
            with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
                validate_qualification_receipt(tampered)

    def test_suite_field_mutations_detected(self, base_receipt):
        """Mutating nested suite attributes must invalidate receipt_sha256."""
        mutations = [
            ("status", "failed"),
            ("duration_seconds", 99.99),
            ("name", "injected_suite_name"),
            ("details", {"tampered_metric": 42}),
        ]
        for key, val in mutations:
            tampered = copy.deepcopy(base_receipt)
            tampered["suites"][0][key] = val
            with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
                validate_qualification_receipt(tampered)

    def test_suite_order_mutation_detected(self, base_receipt):
        """Reordering suites list must invalidate receipt_sha256 even though contents match."""
        tampered = copy.deepcopy(base_receipt)
        assert len(tampered["suites"]) >= 2
        tampered["suites"][0], tampered["suites"][1] = tampered["suites"][1], tampered["suites"][0]
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(tampered)

    def test_unexpected_injected_top_level_key_detected(self, base_receipt):
        """Injecting extra keys into receipt must invalidate receipt_sha256."""
        tampered = copy.deepcopy(base_receipt)
        tampered["unauthorized_field"] = "injected_data"
        with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(tampered)

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
    def test_missing_required_key_detected(self, base_receipt, missing_key):
        """Deleting any mandatory key must raise AssertionError."""
        tampered = copy.deepcopy(base_receipt)
        del tampered[missing_key]
        with pytest.raises(AssertionError, match="missing required keys"):
            validate_qualification_receipt(tampered)

    def test_corrupted_sha256_variants_detected(self, base_receipt):
        """Corrupted hash string variants must be rejected."""
        good_sha = base_receipt["receipt_sha256"]
        corruptions = [
            "",
            "0" * 64,
            good_sha[:-1] + ("a" if good_sha[-1] != "a" else "b"),
            good_sha.upper(),  # case mismatch
        ]
        for bad_sha in corruptions:
            tampered = copy.deepcopy(base_receipt)
            tampered["receipt_sha256"] = bad_sha
            with pytest.raises(AssertionError, match="Receipt SHA-256 mismatch"):
                validate_qualification_receipt(tampered)

    def test_empty_suites_list_rejected(self, base_receipt):
        """Empty suites list must be rejected even if hash is recomputed."""
        tampered = copy.deepcopy(base_receipt)
        tampered["suites"] = []
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(AssertionError, match="Receipt suites list cannot be empty"):
            validate_qualification_receipt(tampered)

    def test_duration_exceeding_180s_boundary_rejected(self, base_receipt):
        """Duration > 180s must be rejected even if hash is recomputed."""
        tampered = copy.deepcopy(base_receipt)
        tampered["duration_seconds"] = 180.001
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(AssertionError, match="exceeded 3-minute hard limit"):
            validate_qualification_receipt(tampered)

    def test_invalid_status_value_rejected(self, base_receipt):
        """Status other than passed or failed must be rejected even if hash is recomputed."""
        tampered = copy.deepcopy(base_receipt)
        tampered["status"] = "partially_passed"
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(AssertionError, match="Invalid status"):
            validate_qualification_receipt(tampered)


class TestQualifyCLIFailureModeChallenges:
    """Stress-test CLI failure modes, exit codes, and failed receipt emission."""

    def test_cli_marks_failed_status_and_exits_code_1_on_suite_error(self, tmp_path):
        """When a suite raises an exception, receipt status is 'failed', receipt is valid, and CLI exits 1."""
        rec_path = tmp_path / "failed_suite_receipt.json"
        code = f"""
import sys
from unittest.mock import patch
from hypergan.cli import main

def mock_fail(self, env_info=None):
    raise RuntimeError("Injected worker communicator failure")

with patch("hypergan.qualify.QualificationRunner._run_suite_cpu_replicated_gloo", mock_fail):
    sys.argv = ["hypergan", "qualify", "--dry-run", "--output", "{rec_path}"]
    sys.exit(main())
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 1, f"Expected CLI exit code 1, got {proc.returncode}"
        assert rec_path.exists()
        receipt = json.loads(rec_path.read_text())
        assert receipt["status"] == "failed"
        failed_suite = next(s for s in receipt["suites"] if "cpu_replicated_gloo" in s["name"])
        assert failed_suite["status"] == "failed"
        assert "Injected worker communicator failure" in failed_suite["error"]
        # Cryptographic verification: even failed receipts must be tamper-evident
        validate_qualification_receipt(receipt)

    def test_cli_marks_failed_status_on_subprocess_timeout(self, tmp_path):
        """Subprocess TimeoutExpired is captured as a suite failure."""
        rec_path = tmp_path / "timeout_receipt.json"
        code = f"""
import subprocess, sys
from unittest.mock import patch
from hypergan.cli import main

def mock_timeout(self, env_info=None):
    raise subprocess.TimeoutExpired(cmd="worker.py", timeout=35.0)

with patch("hypergan.qualify.QualificationRunner._run_suite_multinode_tcp", mock_timeout):
    sys.argv = ["hypergan", "qualify", "--dry-run", "--output", "{rec_path}"]
    sys.exit(main())
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 1
        assert rec_path.exists()
        receipt = json.loads(rec_path.read_text())
        assert receipt["status"] == "failed"
        suite = next(s for s in receipt["suites"] if "multinode_tcp" in s["name"])
        assert suite["status"] == "failed"
        validate_qualification_receipt(receipt)

    @pytest.mark.parametrize("invalid_timeout", ["-5", "0", "180.1", "300", "invalid"])
    def test_cli_rejects_invalid_timeout(self, invalid_timeout):
        """Out-of-range or malformed --timeout flags must exit with non-zero exit code."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--timeout", invalid_timeout],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0

    def test_cli_rejects_unknown_suite_with_exit_code_2(self):
        """Unknown suite specification must exit with code 2."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--suite", "unknown_suite_xyz"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "Unknown qualification suite" in proc.stderr


class TestSuiteFilteringAdversarialChallenges:
    """Stress-test suite selection and exclusion behaviors."""

    @pytest.mark.parametrize("suite_key,expected_suite", [
        ("single_device", "single_device"),
        ("cpu_replicated_gloo", "cpu_replicated_gloo"),
        ("cuda_replicated_nccl", "cuda_replicated_nccl"),
        ("aggregation", "aggregation_strategies"),
        ("multinode_tcp", "multinode_tcp"),
        ("checkpoint_compatibility", "checkpoint_compatibility"),
    ])
    def test_individual_suite_isolation(self, tmp_path, suite_key, expected_suite):
        """Selecting an individual suite runs only that suite and excludes all others."""
        rec_path = tmp_path / f"{suite_key}.json"
        receipt = run_qualification(output=rec_path, dry_run=True, suite=suite_key)
        validate_qualification_receipt(receipt)
        suite_names = [s["name"] for s in receipt["suites"]]
        assert suite_names == [expected_suite]

    def test_all_and_wildcard_filters_run_all_suites(self, tmp_path):
        """Filter values 'all', 'ALL', and '*' execute all 6 suites."""
        for val in ["all", "ALL", "*"]:
            rec_path = tmp_path / f"all_{val}.json"
            receipt = run_qualification(output=rec_path, dry_run=True, suite=val)
            validate_qualification_receipt(receipt)
            assert len(receipt["suites"]) == 6

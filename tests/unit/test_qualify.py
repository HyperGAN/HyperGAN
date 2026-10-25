"""Unit tests for hypergan qualify CLI and hardware qualification harness.

Covers:
1. CLI argument parsing and registration in src/hypergan/cli.py.
2. Local environment inspection and device capability detection.
3. Smoke suite execution and selective filtering (--suite).
4. Standardized qualification-receipt.json schema validation and canonical SHA-256 calculation.
5. Exit codes: 0 on passing suites, non-zero on test failures and invalid arguments.
6. Execution time boundary: the timeout is enforced and the duration is never clamped.
7. Honest statuses (dry_run, skipped), rank process cleanup, and checks that survive python -O.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Dict
from unittest.mock import patch

import pytest

from hypergan import qualify
from hypergan.cli import _parser
from hypergan.qualify import QualificationCheckFailed, ReceiptValidationError, validate_qualification_receipt


# ---------------------------------------------------------------------------
# Helpers & Oracles
# ---------------------------------------------------------------------------

def compute_receipt_sha256(receipt_dict: Dict[str, Any]) -> str:
    """Canonical receipt SHA-256 hash excluding the receipt_sha256 key."""
    canonical = {k: v for k, v in receipt_dict.items() if k != "receipt_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ==============================================================================
# 1. CLI Parsing & Command Registration
# ==============================================================================

class TestQualifyCLIParsing:
    """Verify qualify subcommand registration and CLI option parsing in cli.py."""

    def test_qualify_subcommand_registered_in_parser(self):
        """The qualify subcommand must be registered in the main CLI parser."""
        parser = _parser()
        subparsers = [a for a in parser._actions if a.dest == "command"]
        assert len(subparsers) == 1
        assert "qualify" in subparsers[0].choices

    def test_qualify_default_options(self):
        """Default arguments for qualify command match GLOSSARY contracts."""
        parser = _parser()
        args = parser.parse_args(["qualify"])
        assert args.command == "qualify"
        assert args.output is None
        assert args.dry_run is False
        assert args.suite is None
        assert args.timeout == pytest.approx(180.0)

    @pytest.mark.parametrize("flag,val,expected", [
        ("--output", "custom.json", Path("custom.json")),
        ("-o", "reports/receipt.json", Path("reports/receipt.json")),
    ])
    def test_qualify_output_flags(self, flag, val, expected):
        """--output and -o specify the output receipt file path."""
        parser = _parser()
        args = parser.parse_args(["qualify", flag, val])
        assert args.output == expected

    def test_qualify_dry_run_flag(self):
        """--dry-run enables dry-run execution."""
        parser = _parser()
        args = parser.parse_args(["qualify", "--dry-run"])
        assert args.dry_run is True

    @pytest.mark.parametrize("suite_name", [
        "cpu_replicated_gloo",
        "cuda_replicated_nccl",
        "aggregation",
        "multinode_tcp",
        "checkpoint_compatibility",
    ])
    def test_qualify_suite_flag(self, suite_name):
        """--suite selects an isolated test suite."""
        parser = _parser()
        args = parser.parse_args(["qualify", "--suite", suite_name])
        assert args.suite == suite_name

    def test_qualify_timeout_flag(self):
        """--timeout sets execution time limit."""
        parser = _parser()
        args = parser.parse_args(["qualify", "--timeout", "60"])
        assert args.timeout == pytest.approx(60.0)

    def test_qualify_help_text(self):
        """hypergan qualify --help displays all available flags."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0
        out = proc.stdout.lower()
        assert "--output" in out or "-o" in out
        assert "--dry-run" in out
        assert "--suite" in out
        assert "--timeout" in out


# ==============================================================================
# 2. Environment & Device Inspection
# ==============================================================================

class TestEnvironmentInspection:
    """Verify hardware and software environment inspection in qualify.py."""

    def test_inspect_environment_live(self):
        """inspect_environment returns valid schema on current machine."""
        from hypergan.qualify import inspect_environment
        env = inspect_environment()
        assert isinstance(env, dict)
        assert "platform" in env and isinstance(env["platform"], str)
        assert "python" in env and isinstance(env["python"], str)
        assert "torch" in env and isinstance(env["torch"], str)
        assert "device_count" in env and isinstance(env["device_count"], int)
        assert "devices" in env and isinstance(env["devices"], list)
        assert "cuda" in env and isinstance(env["cuda"], dict)
        assert len(env["devices"]) == env["device_count"]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_name", side_effect=["NVIDIA RTX A6000", "NVIDIA RTX A6000"])
    @patch("torch.distributed.is_nccl_available", return_value=True)
    def test_inspect_environment_mocked_cuda(self, mock_nccl, mock_name, mock_count, mock_avail):
        """inspect_environment captures 2x NVIDIA RTX A6000 GPU metadata."""
        from hypergan.qualify import inspect_environment
        env = inspect_environment()
        assert env["device_count"] == 2
        assert env["devices"] == ["NVIDIA RTX A6000", "NVIDIA RTX A6000"]
        assert env["cuda"]["available"] is True
        assert env["cuda"]["nccl"] is True

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.device_count", return_value=0)
    def test_inspect_environment_cpu_only(self, mock_count, mock_avail):
        """inspect_environment behaves correctly when CUDA is unavailable."""
        from hypergan.qualify import inspect_environment
        env = inspect_environment()
        assert env["device_count"] == 0
        assert env["devices"] == []
        assert env["cuda"]["available"] is False


# ==============================================================================
# 3. Smoke Suite Execution & Filtering
# ==============================================================================

class TestSmokeSuiteExecution:
    """Verify execution, suite filtering, and error handling in qualify.py."""

    def test_run_qualification_dry_run_all_suites(self, tmp_path):
        """run_qualification in dry_run mode reports dry_run, never passed, for every suite."""
        from hypergan.qualify import run_qualification
        out_file = tmp_path / "receipt.json"
        receipt = run_qualification(output=out_file, dry_run=True)
        assert receipt["status"] == "dry_run"
        assert {s["status"] for s in receipt["suites"]} == {"dry_run"}
        assert out_file.exists()
        validate_qualification_receipt(receipt)
        suite_names = [s["name"] for s in receipt["suites"]]
        for expected in ("cpu_replicated_gloo", "aggregation"):
            assert any(expected in name for name in suite_names)

    def test_run_qualification_suite_filtering(self, tmp_path):
        """Passing suite='aggregation' executes only the aggregation suite."""
        from hypergan.qualify import run_qualification
        out_file = tmp_path / "receipt_agg.json"
        receipt = run_qualification(output=out_file, dry_run=True, suite="aggregation")
        assert receipt["status"] == "dry_run"
        validate_qualification_receipt(receipt)
        assert all("aggregation" in s["name"] for s in receipt["suites"])

    def test_run_qualification_invalid_suite_raises(self, tmp_path):
        """Requesting an unknown suite name raises ValueError with descriptive message."""
        from hypergan.qualify import run_qualification
        with pytest.raises(ValueError, match="Unknown qualification suite"):
            run_qualification(output=tmp_path / "r.json", dry_run=True, suite="invalid_suite_xyz")

    def test_run_qualification_failure_captured_in_receipt(self, tmp_path):
        """If a test suite fails, status='failed' is recorded and receipt is still emitted."""
        from hypergan.qualify import QualificationRunner
        runner = QualificationRunner(dry_run=True)
        with patch.object(runner, "_run_suite_aggregation", side_effect=AssertionError("Parity drift")):
            receipt = runner.run(output=tmp_path / "failed_receipt.json")
            assert receipt["status"] == "failed"
            failed_suite = next(s for s in receipt["suites"] if "aggregation" in s["name"])
            assert failed_suite["status"] == "failed"
            assert "Parity drift" in failed_suite.get("error", "")
            validate_qualification_receipt(receipt)


# ==============================================================================
# 4. Receipt Schema Contract & SHA-256 Integrity
# ==============================================================================

class TestReceiptSchemaAndIntegrity:
    """Verify qualification receipt schema adherence and cryptographic integrity."""

    def test_receipt_sha256_canonical_hash(self, tmp_path):
        """Generated receipt hash matches canonical JSON SHA-256."""
        from hypergan.qualify import run_qualification
        out_file = tmp_path / "rec.json"
        receipt = run_qualification(output=out_file, dry_run=True)
        computed = compute_receipt_sha256(receipt)
        assert receipt["receipt_sha256"] == computed

    def test_receipt_sha256_tamper_detection(self, tmp_path):
        """Any modification to receipt content invalidates receipt_sha256."""
        from hypergan.qualify import run_qualification
        receipt = run_qualification(output=tmp_path / "rec.json", dry_run=True)
        tampered = copy.deepcopy(receipt)
        tampered["status"] = "failed" if receipt["status"] != "failed" else "dry_run"
        with pytest.raises(ReceiptValidationError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(tampered)

    def test_receipt_duration_boundary(self, tmp_path):
        """Qualification duration must be bounded to <= 180s."""
        from hypergan.qualify import run_qualification
        receipt = run_qualification(output=tmp_path / "rec.json", dry_run=True)
        assert receipt["duration_seconds"] <= 180.0

    def test_receipt_output_creates_parent_directories(self, tmp_path):
        """Emitting receipt to nested nonexistent directory creates parent directories."""
        from hypergan.qualify import run_qualification
        deep_file = tmp_path / "a" / "b" / "c" / "qualification-receipt.json"
        receipt = run_qualification(output=deep_file, dry_run=True)
        assert deep_file.is_file()
        assert json.loads(deep_file.read_text())["receipt_sha256"] == receipt["receipt_sha256"]


# ==============================================================================
# 5. CLI Execution & Process Exit Codes
# ==============================================================================

class TestQualifyCLIExecution:
    """Verify process exit codes and end-to-end command execution."""

    def test_cli_dry_run_exit_code_zero(self, tmp_path):
        """hypergan qualify --dry-run exits with code 0."""
        out_file = tmp_path / "cli_receipt.json"
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--dry-run", "--output", str(out_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"CLI qualify failed:\n{proc.stderr}"
        assert out_file.exists()
        receipt = json.loads(out_file.read_text())
        validate_qualification_receipt(receipt)

    def test_cli_suite_filtering_exit_code_zero(self, tmp_path):
        """hypergan qualify --suite aggregation --dry-run exits with code 0."""
        out_file = tmp_path / "cli_agg.json"
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--suite", "aggregation", "--dry-run", "--output", str(out_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0
        assert out_file.exists()

    def test_cli_invalid_argument_exit_code_nonzero(self):
        """Passing an invalid flag or value exits with non-zero code."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--timeout", "-10"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode != 0


# ==============================================================================
# 6. Review 2026-10-25 regressions: timeout, statuses, rank cleanup, python -O
# ==============================================================================

class TestTimeoutAndStatuses:
    """The 3-minute check must be able to fire, and nothing that did not run reports passed."""

    def test_timeout_fires_and_duration_is_not_clamped(self, tmp_path):
        """Suites that overrun the timeout fail the receipt; later suites are not started."""
        now = [0.0]
        runner = qualify.QualificationRunner(output_path=tmp_path / "r.json", timeout=60, clock=lambda: now[0])

        def slow(env_info=None):
            now[0] += 200.0
            return {"name": "single_device", "status": "passed", "duration_seconds": 200.0}

        with patch.object(runner, "_run_suite_single_device", side_effect=slow):
            receipt = runner.run()
        assert receipt["duration_seconds"] == pytest.approx(200.0)
        assert receipt["status"] == "failed"
        assert "60" in receipt["error"]
        later = [s for s in receipt["suites"] if s["name"] != "single_device"]
        assert later and all(s["status"] == "skipped" and "not started" in s["details"]["reason"] for s in later)
        validate_qualification_receipt(receipt)

    def test_passed_receipt_over_limit_is_invalid(self):
        """Only a failed receipt may record more than 180 seconds."""
        receipt = {"schema_version": 1, "qualification_id": "q", "timestamp": "t", "status": "passed",
                   "duration_seconds": 181.0,
                   "environment": {"platform": "linux", "python": "3", "torch": "2", "device_count": 0},
                   "suites": [{"name": "single_device", "status": "passed"}]}
        receipt["receipt_sha256"] = qualify.compute_receipt_sha256(receipt)
        with pytest.raises(ReceiptValidationError, match="exceeded 3-minute hard limit"):
            validate_qualification_receipt(receipt)
        receipt["status"] = "failed"
        receipt["receipt_sha256"] = qualify.compute_receipt_sha256(receipt)
        validate_qualification_receipt(receipt)

    def test_only_skipped_suites_is_not_a_pass(self, tmp_path, capsys):
        """When every selected suite is skipped the receipt says skipped and the CLI exits 1."""
        env = {"platform": "linux", "python": "3", "torch": "2", "device_count": 0, "cuda": {"available": False}}
        args = _parser().parse_args(["qualify", "--suite", "cuda_replicated_nccl", "--output", str(tmp_path / "r.json")])
        with patch("hypergan.qualify.inspect_environment", return_value=env):
            assert qualify.run_qualification_cli(args) == 1
        receipt = json.loads((tmp_path / "r.json").read_text())
        assert receipt["status"] == "skipped"
        assert [s["status"] for s in receipt["suites"]] == ["skipped"]
        validate_qualification_receipt(receipt)

    def test_dry_run_cli_exits_zero_and_says_dry_run(self, tmp_path, capsys):
        args = _parser().parse_args(["qualify", "--dry-run", "--output", str(tmp_path / "r.json")])
        assert qualify.run_qualification_cli(args) == 0
        out = capsys.readouterr().out
        assert "DRY_RUN" in out
        assert "PASSED" not in out

    def test_json_verbose_keeps_stdout_pure_json(self, tmp_path, capsys):
        args = _parser().parse_args(["qualify", "--dry-run", "--json", "-v", "--output", str(tmp_path / "r.json")])
        assert qualify.run_qualification_cli(args) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["status"] == "dry_run"
        assert "Running qualification suite" in captured.err

    def test_default_output_is_state_dir_not_cwd(self, tmp_path, monkeypatch):
        """Without an output path the receipt goes under the state directory, never the cwd."""
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setenv("HYPERGAN_STATE_DIR", str(tmp_path / "state"))
        runner = qualify.QualificationRunner(dry_run=True)
        receipt = runner.run()
        assert list(work.iterdir()) == []
        assert runner.last_output_path == (tmp_path / "state" / "qualification"
                                           / f"qualification-receipt-{receipt['qualification_id']}.json")
        assert runner.last_output_path.is_file()


class TestRankProcessCleanup:
    """_run_subprocess_ranks must never leave rank processes behind."""

    @pytest.fixture
    def spawned(self, monkeypatch):
        procs = []
        real_popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            procs.append(proc)
            return proc

        monkeypatch.setattr(qualify.subprocess, "Popen", recording_popen)
        yield procs
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_ranks_killed_on_timeout(self, monkeypatch, spawned):
        monkeypatch.setattr(qualify, "WORKER_DRIVER_CODE", "import time\ntime.sleep(60)\n")
        monkeypatch.setattr(qualify, "RANK_TIMEOUT_SECONDS", 0.3)
        runner = qualify.QualificationRunner()
        with pytest.raises(TimeoutError, match="did not finish"):
            runner._run_subprocess_ranks(backend="gloo", world_size=2)
        assert len(spawned) == 2
        assert all(p.poll() is not None for p in spawned)

    def test_peer_ranks_killed_when_one_rank_fails(self, monkeypatch, spawned):
        monkeypatch.setattr(qualify, "WORKER_DRIVER_CODE",
                            "import sys, time\nif sys.argv[2] == '0':\n    sys.exit(3)\ntime.sleep(60)\n")
        runner = qualify.QualificationRunner()
        with pytest.raises(QualificationCheckFailed, match="Rank 0 failed with code 3"):
            runner._run_subprocess_ranks(backend="gloo", world_size=2)
        assert len(spawned) == 2
        assert all(p.poll() is not None for p in spawned)


def test_checks_survive_python_optimize_mode(tmp_path):
    """Validation and suite checks raise explicitly, so python -O cannot turn them into passes."""
    script = tmp_path / "optimized.py"
    script.write_text(textwrap.dedent("""
        from unittest.mock import patch
        from hypergan import qualify

        try:
            qualify.validate_qualification_receipt({})
        except qualify.ReceiptValidationError:
            print("validator-raised")

        diverged = [{"passed": True, "final_weights": {"w": [0.0]}}, {"passed": True, "final_weights": {"w": [1.0]}}]
        runner = qualify.QualificationRunner()
        with patch.object(runner, "_run_subprocess_ranks", return_value=diverged):
            try:
                runner._run_suite_cpu_replicated_gloo({})
            except qualify.QualificationCheckFailed:
                print("suite-check-raised")
    """))
    proc = subprocess.run([sys.executable, "-O", str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["validator-raised", "suite-check-raised"]


@pytest.mark.heavy
def test_multinode_suite_is_labelled_single_host_and_uses_distinct_node_ranks(tmp_path):
    """The multi-node suite says it is a one-host simulation and gives each rank its own node_rank."""
    receipt = qualify.run_qualification(output=tmp_path / "r.json", suite="multinode")
    validate_qualification_receipt(receipt)
    (suite,) = receipt["suites"]
    assert suite["name"] == "multinode_tcp_single_host"
    assert suite["status"] == "passed", suite
    assert suite["details"]["simulated"] is True
    assert suite["details"]["hosts"] == 1
    assert suite["details"]["node_ranks"] == [0, 1]


@pytest.mark.heavy
def test_aggregation_suite_exercises_outer_momentum(tmp_path):
    """The pseudo_gradient check runs two outer steps, so outer_momentum must change the result."""
    receipt = qualify.run_qualification(output=tmp_path / "r.json", suite="aggregation")
    (suite,) = receipt["suites"]
    assert suite["status"] == "passed", suite
    assert suite["details"]["pseudo_gradient"] == "passed"

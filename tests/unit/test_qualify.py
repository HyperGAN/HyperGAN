"""Tests for `hypergan qualify`, its receipt and its rank workers.

Covers:
1. CLI argument parsing, help and exit codes (0 pass or dry run, 1 failure or nothing qualified,
   2 invalid arguments).
2. Environment inspection.
3. Suite selection: filters, isolation, case-insensitive substrings, unknown names.
4. The receipt contract enforced by validate_qualification_receipt: required keys, statuses,
   the unkeyed SHA-256 checksum (an edit that does not recompute it is caught; one that does
   is not, so it is no signature), the 3-minute limit, status agreement and receipt paths.
5. Timeout enforcement without clamping, honest dry_run and skipped statuses.
6. Rank workers (python -m hypergan.qualify_worker): killed on timeout or a peer's failure.
7. Checks that survive python -O.
8. Real suites on CPU (heavy), including the product-path CPU Gloo suite.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Callable, Dict
from unittest.mock import patch

import pytest

from hypergan import qualify
from hypergan.cli import _parser, main as cli_main
from hypergan.qualify import (
    QualificationCheckFailed,
    QualificationRunner,
    ReceiptValidationError,
    run_qualification,
    run_qualification_cli,
    validate_qualification_receipt,
)

ALL_SUITES = ["single_device", "cpu_replicated_gloo", "cuda_replicated_nccl", "aggregation_strategies",
              "multinode_tcp_single_host", "checkpoint_compatibility"]


def independent_sha256(receipt_dict: Dict[str, Any]) -> str:
    """Canonical receipt SHA-256 excluding receipt_sha256, written independently of the product."""
    canonical = {k: v for k, v in receipt_dict.items() if k != "receipt_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resealed(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """The receipt with its checksum recomputed, as anyone editing it could do."""
    receipt["receipt_sha256"] = qualify.compute_receipt_sha256(receipt)
    return receipt


def passed_receipt(**fields: Any) -> Dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "qualification_id": "qual-20261024-test-001",
        "timestamp": "2026-10-24T15:00:00Z",
        "status": "passed",
        "duration_seconds": 45.2,
        "environment": {"platform": "linux", "python": "3.14.8", "torch": "2.5.1", "device_count": 2,
                        "devices": ["NVIDIA RTX A6000", "NVIDIA RTX A6000"], "cuda": {"available": True, "nccl": True}},
        "suites": [
            {"name": "cpu_replicated_gloo", "status": "passed", "duration_seconds": 12.1},
            {"name": "cuda_replicated_nccl", "status": "passed", "duration_seconds": 15.3},
            {"name": "aggregation_strategies", "status": "passed", "duration_seconds": 10.4},
            {"name": "multinode_tcp_single_host", "status": "passed", "duration_seconds": 7.4},
        ],
    }
    receipt.update(fields)
    return resealed(receipt)


@pytest.fixture
def dry_receipt(tmp_path) -> Dict[str, Any]:
    """A pristine dry-run receipt for every suite."""
    receipt = run_qualification(output=tmp_path / "dry.json", dry_run=True)
    validate_qualification_receipt(receipt)
    return receipt


# ==============================================================================
# 1. CLI parsing, help and exit codes
# ==============================================================================

class TestQualifyCLI:

    def test_qualify_subcommand_registered_with_defaults(self):
        parser = _parser()
        (subparsers,) = [a for a in parser._actions if a.dest == "command"]
        assert "qualify" in subparsers.choices
        args = parser.parse_args(["qualify"])
        assert args.command == "qualify"
        assert args.output is None
        assert args.dry_run is False
        assert args.suite is None
        assert args.timeout == pytest.approx(180.0)

    @pytest.mark.parametrize("argv,attr,expected", [
        (["--output", "custom.json"], "output", Path("custom.json")),
        (["-o", "reports/receipt.json"], "output", Path("reports/receipt.json")),
        (["--dry-run"], "dry_run", True),
        (["--suite", "cpu_replicated_gloo"], "suite", "cpu_replicated_gloo"),
        (["--suite", "checkpoint_compatibility"], "suite", "checkpoint_compatibility"),
        (["--timeout", "60"], "timeout", 60.0),
        (["--timeout", "180"], "timeout", 180.0),
    ])
    def test_qualify_options_parse(self, argv, attr, expected):
        assert getattr(_parser().parse_args(["qualify", *argv]), attr) == expected

    @pytest.mark.parametrize("invalid_timeout", ["-10", "-5", "0", "180.1", "300", "nan", "invalid"])
    def test_out_of_range_or_malformed_timeout_exits_2(self, invalid_timeout):
        with pytest.raises(SystemExit) as exc:
            _parser().parse_args(["qualify", "--timeout", invalid_timeout])
        assert exc.value.code == 2

    def test_qualify_help_lists_every_flag(self):
        proc = subprocess.run([sys.executable, "-m", "hypergan", "qualify", "--help"],
                              capture_output=True, text=True, timeout=15)
        assert proc.returncode == 0, proc.stderr
        for flag in ("--output", "--dry-run", "--suite", "--timeout", "--json", "--verbose"):
            assert flag in proc.stdout

    def test_unknown_suite_exits_2_with_catalog(self, capsys):
        assert run_qualification_cli(_parser().parse_args(["qualify", "--suite", "bogus_suite_xyz"])) == 2
        err = capsys.readouterr().err
        assert "Unknown qualification suite" in err
        assert "single_device" in err

    def test_dry_run_exits_0_and_says_dry_run(self, tmp_path, capsys):
        assert cli_main(["qualify", "--dry-run", "--output", str(tmp_path / "r.json")]) == 0
        out = capsys.readouterr().out
        assert "DRY_RUN" in out
        assert "PASSED" not in out
        validate_qualification_receipt(json.loads((tmp_path / "r.json").read_text()))

    def test_suite_filtered_dry_run_exits_0(self, tmp_path):
        out_file = tmp_path / "agg.json"
        assert cli_main(["qualify", "--suite", "aggregation", "--dry-run", "--output", str(out_file)]) == 0
        assert [s["name"] for s in json.loads(out_file.read_text())["suites"]] == ["aggregation_strategies"]

    @pytest.mark.parametrize("suite_method,error,suite_name", [
        ("_run_suite_cpu_replicated_gloo", RuntimeError("Injected worker communicator failure"), "cpu_replicated_gloo"),
        ("_run_suite_multinode_tcp", subprocess.TimeoutExpired(cmd="worker", timeout=35.0), "multinode_tcp_single_host"),
        ("_run_suite_aggregation", AssertionError("Parity drift"), "aggregation_strategies"),
    ])
    def test_suite_error_fails_receipt_and_exits_1(self, tmp_path, suite_method, error, suite_name):
        """A suite that raises is recorded as failed in a still-valid receipt, and the CLI exits 1."""
        rec_path = tmp_path / "failed.json"
        with patch.object(QualificationRunner, suite_method, side_effect=error):
            assert cli_main(["qualify", "--dry-run", "--output", str(rec_path)]) == 1
        receipt = json.loads(rec_path.read_text())
        assert receipt["status"] == "failed"
        (failed,) = [s for s in receipt["suites"] if s["name"] == suite_name]
        assert failed["status"] == "failed"
        assert str(error) in failed["error"]
        validate_qualification_receipt(receipt)

    def test_only_skipped_suites_is_not_a_pass(self, tmp_path):
        """When every selected suite is skipped the receipt says skipped and the CLI exits 1."""
        env = {"platform": "linux", "python": "3", "torch": "2", "device_count": 0, "cuda": {"available": False}}
        args = _parser().parse_args(["qualify", "--suite", "cuda_replicated_nccl", "--output", str(tmp_path / "r.json")])
        with patch("hypergan.qualify.inspect_environment", return_value=env):
            assert run_qualification_cli(args) == 1
        receipt = json.loads((tmp_path / "r.json").read_text())
        assert receipt["status"] == "skipped"
        assert [s["status"] for s in receipt["suites"]] == ["skipped"]
        validate_qualification_receipt(receipt)

    def test_json_verbose_keeps_stdout_pure_json(self, tmp_path, capsys):
        args = _parser().parse_args(["qualify", "--dry-run", "--json", "-v", "--output", str(tmp_path / "r.json")])
        assert run_qualification_cli(args) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["status"] == "dry_run"
        assert "Running qualification suite" in captured.err

    @pytest.mark.heavy
    def test_cli_subprocess_dry_run_writes_valid_receipt(self, tmp_path):
        out_file = tmp_path / "cli_receipt.json"
        proc = subprocess.run([sys.executable, "-m", "hypergan", "qualify", "--dry-run", "--output", str(out_file)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        validate_qualification_receipt(json.loads(out_file.read_text()))


# ==============================================================================
# 2. Environment inspection
# ==============================================================================

class TestEnvironmentInspection:

    def test_inspect_environment_live(self):
        env = qualify.inspect_environment()
        for key, kind in (("platform", str), ("python", str), ("torch", str), ("device_count", int),
                          ("devices", list), ("cuda", dict)):
            assert isinstance(env[key], kind), key
        assert len(env["devices"]) == env["device_count"]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_name", side_effect=["NVIDIA RTX A6000", "NVIDIA RTX A6000"])
    @patch("torch.distributed.is_nccl_available", return_value=True)
    def test_inspect_environment_mocked_cuda(self, mock_nccl, mock_name, mock_count, mock_avail):
        env = qualify.inspect_environment()
        assert env["device_count"] == 2
        assert env["devices"] == ["NVIDIA RTX A6000", "NVIDIA RTX A6000"]
        assert env["cuda"]["available"] is True
        assert env["cuda"]["nccl"] is True

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.device_count", return_value=0)
    def test_inspect_environment_cpu_only(self, mock_count, mock_avail):
        env = qualify.inspect_environment()
        assert env["device_count"] == 0
        assert env["devices"] == []
        assert env["cuda"]["available"] is False


# ==============================================================================
# 3. Suite selection
# ==============================================================================

class TestSuiteSelection:

    def test_dry_run_reports_dry_run_for_every_suite(self, tmp_path):
        out_file = tmp_path / "receipt.json"
        receipt = run_qualification(output=out_file, dry_run=True)
        assert receipt["status"] == "dry_run"
        assert [s["name"] for s in receipt["suites"]] == ALL_SUITES
        assert {s["status"] for s in receipt["suites"]} == {"dry_run"}
        assert json.loads(out_file.read_text()) == receipt

    @pytest.mark.parametrize("suite_key,expected_suite", [
        ("single_device", "single_device"),
        ("cpu_replicated_gloo", "cpu_replicated_gloo"),
        ("cuda_replicated_nccl", "cuda_replicated_nccl"),
        ("aggregation", "aggregation_strategies"),
        ("multinode_tcp", "multinode_tcp_single_host"),
        ("checkpoint_compatibility", "checkpoint_compatibility"),
    ])
    def test_individual_suite_isolation(self, tmp_path, suite_key, expected_suite):
        receipt = run_qualification(output=tmp_path / "r.json", dry_run=True, suite=suite_key)
        validate_qualification_receipt(receipt)
        assert [s["name"] for s in receipt["suites"]] == [expected_suite]

    @pytest.mark.parametrize("value", ["all", "ALL", "*"])
    def test_all_and_wildcard_filters_run_every_suite(self, tmp_path, value):
        receipt = run_qualification(output=tmp_path / "r.json", dry_run=True, suite=value)
        assert [s["name"] for s in receipt["suites"]] == ALL_SUITES

    @pytest.mark.parametrize("suite_name,expected_contained", [
        ("SINGLE_DEVICE", "single_device"),
        ("Cpu_Replicated_Gloo", "cpu_replicated_gloo"),
        ("AGGREGATION", "aggregation_strategies"),
        ("checkpoint", "checkpoint_compatibility"),
        ("multinode", "multinode_tcp_single_host"),
    ])
    def test_case_insensitive_and_substring_suite_filtering(self, suite_name, expected_contained):
        runner = QualificationRunner(dry_run=True, suite_filter=suite_name)
        assert runner.suite_filter is not None
        assert runner.should_run_suite(expected_contained)

    @pytest.mark.parametrize("invalid_suite", ["invalid_suite_xyz", "quantum_gan", "tpu_cluster", "xyz"])
    def test_unknown_suite_raises_value_error(self, invalid_suite):
        with pytest.raises(ValueError, match="Unknown qualification suite"):
            QualificationRunner(dry_run=True, suite_filter=invalid_suite)


# ==============================================================================
# 4. Receipt contract (validate_qualification_receipt)
# ==============================================================================

def _set(path: str, value: Any) -> Callable[[Dict[str, Any]], None]:
    def mutate(receipt: Dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        target: Any = receipt
        for part in parents:
            target = target[int(part)] if isinstance(target, list) else target[part]
        target[int(leaf) if isinstance(target, list) else leaf] = value
    return mutate


def _swap_first_suites(receipt):
    receipt["suites"][0], receipt["suites"][1] = receipt["suites"][1], receipt["suites"][0]


def _bump_every_suite_duration(receipt):
    for suite in receipt["suites"]:
        suite["duration_seconds"] += 0.05


# One-field edits that leave the recorded checksum stale.
UNSEALED_EDITS = {
    "status": _set("status", "failed"),
    "duration_seconds": lambda r: r.update(duration_seconds=r["duration_seconds"] + 0.01),
    "qualification_id": _set("qualification_id", "qual-forged"),
    "timestamp": _set("timestamp", "2000-01-01T00:00:00+00:00"),
    "environment.platform": _set("environment.platform", "freebsd_unsupported"),
    "environment.python": _set("environment.python", "3.99.9"),
    "environment.torch": _set("environment.torch", "1.0.0.tampered"),
    "environment.device_count": _set("environment.device_count", 999),
    "environment.devices": _set("environment.devices", ["NVIDIA H100 80GB"]),
    "environment.cuda.version": _set("environment.cuda.version", "99.0"),
    "environment.injected": _set("environment.tampered_sub", "bad"),
    "suite.status": _set("suites.0.status", "failed"),
    "suite.duration_seconds": _set("suites.0.duration_seconds", 99.99),
    "suite.name": _set("suites.0.name", "injected_suite_name"),
    "suite.details": _set("suites.0.details", {"tampered_metric": 42}),
    "every_suite_duration": _bump_every_suite_duration,
    "suite_order_swapped": _swap_first_suites,
    "suite_order_reversed": lambda r: r.update(suites=list(reversed(r["suites"]))),
    "suite_appended": lambda r: r["suites"].append({"name": "injected", "status": "dry_run", "duration_seconds": 0.0}),
    "injected_top_level_key": _set("unauthorized_field", "injected_data"),
    "sha_empty": _set("receipt_sha256", ""),
    "sha_zeros": _set("receipt_sha256", "0" * 64),
    "sha_last_char": lambda r: r.update(receipt_sha256=r["receipt_sha256"][:-1] + ("a" if r["receipt_sha256"][-1] != "a" else "b")),
    "sha_first_char": lambda r: r.update(receipt_sha256=("0" if r["receipt_sha256"][0] != "0" else "1") + r["receipt_sha256"][1:]),
    "sha_uppercase": lambda r: r.update(receipt_sha256=r["receipt_sha256"].upper()),
}


class TestReceiptContract:

    def test_checksum_matches_independent_canonical_sha256(self, dry_receipt):
        assert dry_receipt["receipt_sha256"] == independent_sha256(dry_receipt)

    def test_complete_passed_receipt_validates(self):
        validate_qualification_receipt(passed_receipt())

    @pytest.mark.parametrize("edit", list(UNSEALED_EDITS.values()), ids=list(UNSEALED_EDITS))
    def test_edit_without_recomputed_checksum_is_rejected(self, dry_receipt, edit):
        tampered = copy.deepcopy(dry_receipt)
        edit(tampered)
        assert tampered != dry_receipt
        with pytest.raises(ReceiptValidationError, match="Receipt SHA-256 mismatch"):
            validate_qualification_receipt(tampered)

    def test_mutating_any_top_level_key_is_rejected(self, dry_receipt):
        """Whatever keys a receipt grows, editing one fails the checksum or the schema."""
        for key in dry_receipt:
            if key == "receipt_sha256":
                continue
            tampered = copy.deepcopy(dry_receipt)
            val = tampered[key]
            if isinstance(val, str):
                tampered[key] = val + "_tampered"
            elif isinstance(val, (int, float)):
                tampered[key] = val + 0.1
            elif isinstance(val, list):
                tampered[key] = val + [{"name": "injected", "status": "passed", "duration_seconds": 0.01}]
            elif isinstance(val, dict):
                tampered[key]["tampered_sub"] = "bad"
            with pytest.raises(ReceiptValidationError):
                validate_qualification_receipt(tampered)

    def test_recomputed_checksum_hides_an_edit(self, dry_receipt):
        """The checksum is unkeyed: an edit that recomputes it validates. It is no signature."""
        forged = copy.deepcopy(dry_receipt)
        forged["environment"]["devices"] = ["NVIDIA H100 80GB"]
        validate_qualification_receipt(resealed(forged))

    @pytest.mark.parametrize("missing_key", ["schema_version", "qualification_id", "timestamp", "status",
                                             "duration_seconds", "environment", "suites", "receipt_sha256"])
    def test_missing_required_key_rejected(self, dry_receipt, missing_key):
        tampered = copy.deepcopy(dry_receipt)
        del tampered[missing_key]
        with pytest.raises(ReceiptValidationError, match="Qualification receipt missing required keys"):
            validate_qualification_receipt(tampered)

    @pytest.mark.parametrize("invalid_status", ["partially_passed", "in_progress", "pending", "error", 1, None])
    def test_invalid_status_rejected_even_when_resealed(self, dry_receipt, invalid_status):
        tampered = resealed(dict(copy.deepcopy(dry_receipt), status=invalid_status))
        with pytest.raises(ReceiptValidationError, match="Invalid status"):
            validate_qualification_receipt(tampered)

    @pytest.mark.parametrize("field,value,match", [
        ("suites", [], "Receipt suites list cannot be empty"),
        ("duration_seconds", -1.5, "non-negative number"),
        ("schema_version", 2, "Expected schema_version 1"),
    ])
    def test_schema_violation_rejected_even_when_resealed(self, dry_receipt, field, value, match):
        tampered = resealed(dict(copy.deepcopy(dry_receipt), **{field: value}))
        with pytest.raises(ReceiptValidationError, match=match):
            validate_qualification_receipt(tampered)

    @pytest.mark.parametrize("duration", [180.001, 180.1, 181.0])
    def test_over_limit_is_valid_only_as_a_failure(self, duration):
        """Only a failed receipt may record more than 180 seconds."""
        with pytest.raises(ReceiptValidationError, match="Qualification exceeded 3-minute hard limit"):
            validate_qualification_receipt(passed_receipt(duration_seconds=duration))
        validate_qualification_receipt(passed_receipt(duration_seconds=duration, status="failed"))

    def test_just_under_limit_is_valid(self):
        validate_qualification_receipt(passed_receipt(duration_seconds=179.9))

    @pytest.mark.parametrize("claimed", ["passed", "skipped"])
    def test_status_contradicting_suites_rejected(self, dry_receipt, claimed):
        """A dry-run receipt cannot be relabelled passed or skipped, even with a recomputed checksum."""
        tampered = resealed(dict(copy.deepcopy(dry_receipt), status=claimed))
        with pytest.raises(ReceiptValidationError, match="contradicts its suites"):
            validate_qualification_receipt(tampered)

    def test_receipt_output_creates_parent_directories(self, tmp_path):
        deep_file = tmp_path / "a" / "b" / "c" / "qualification-receipt.json"
        receipt = run_qualification(output=deep_file, dry_run=True)
        assert json.loads(deep_file.read_text()) == receipt

    def test_output_existing_directory_raises_is_a_directory_error(self, tmp_path):
        output_dir = tmp_path / "existing_dir"
        output_dir.mkdir()
        with pytest.raises(IsADirectoryError):
            QualificationRunner(output_path=output_dir, dry_run=True).run()

    def test_default_output_is_state_dir_not_cwd(self, tmp_path, monkeypatch):
        """Without an output path the receipt goes under the state directory, never the cwd."""
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setenv("HYPERGAN_STATE_DIR", str(tmp_path / "state"))
        runner = QualificationRunner(dry_run=True)
        receipt = runner.run()
        assert list(work.iterdir()) == []
        assert runner.last_output_path == (tmp_path / "state" / "qualification"
                                           / f"qualification-receipt-{receipt['qualification_id']}.json")
        assert runner.last_output_path.is_file()


# ==============================================================================
# 5. Timeout and statuses
# ==============================================================================

class TestTimeoutAndStatuses:
    """The 3-minute check must be able to fire, and nothing that did not run reports passed."""

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_timeout_fires_and_duration_is_not_clamped(self, tmp_path, dry_run):
        """Suites that overrun the timeout fail the receipt; later suites are not started."""
        now = [1000.0]
        runner = QualificationRunner(output_path=tmp_path / "r.json", timeout=60, dry_run=dry_run,
                                     clock=lambda: now[0])

        def slow(env_info=None):
            now[0] += 250.0
            return {"name": "single_device", "status": "passed", "duration_seconds": 250.0}

        with patch.object(runner, "_run_suite_single_device", side_effect=slow):
            receipt = runner.run()
        assert receipt["duration_seconds"] == pytest.approx(250.0)
        assert receipt["status"] == "failed"
        assert "60" in receipt["error"] and "timeout" in receipt["error"]
        later = [s for s in receipt["suites"] if s["name"] != "single_device"]
        assert later and all(s["status"] == "skipped" and "not started" in s["details"]["reason"] for s in later)
        validate_qualification_receipt(receipt)

    @pytest.mark.parametrize("timeout", [0, -1, 180.1])
    def test_runner_rejects_out_of_range_timeout(self, timeout):
        with pytest.raises(ValueError, match="Timeout must be between 1 and 180 seconds"):
            QualificationRunner(timeout=timeout)


# ==============================================================================
# 6. Rank workers
# ==============================================================================

class TestRankWorkers:
    """Ranks run the real module hypergan.qualify_worker and are never left behind."""

    def test_ranks_run_the_worker_module(self):
        assert qualify.worker_command() == [sys.executable, "-m", "hypergan.qualify_worker"]
        spec = importlib.util.find_spec(qualify.WORKER_MODULE)
        assert spec is not None and Path(spec.origin).parent == Path(qualify.__file__).parent
        # No program hides in a string literal any more.
        assert not hasattr(qualify, "WORKER_DRIVER_CODE")

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

    @staticmethod
    def _fake_worker(monkeypatch, code):
        # argv after `-c` matches the module's: MODE RANK WORLD_SIZE ...
        monkeypatch.setattr(qualify, "worker_command", lambda: [sys.executable, "-c", code])

    def test_ranks_killed_on_timeout(self, monkeypatch, spawned):
        self._fake_worker(monkeypatch, "import time\ntime.sleep(60)\n")
        monkeypatch.setattr(qualify, "RANK_TIMEOUT_SECONDS", 0.3)
        with pytest.raises(TimeoutError, match="did not finish"):
            QualificationRunner()._run_subprocess_ranks(backend="gloo", world_size=2)
        assert len(spawned) == 2
        assert all(p.poll() is not None for p in spawned)

    def test_peer_ranks_killed_when_one_rank_fails(self, monkeypatch, spawned):
        self._fake_worker(monkeypatch, "import sys, time\nif sys.argv[2] == '0':\n    sys.exit(3)\ntime.sleep(60)\n")
        with pytest.raises(QualificationCheckFailed, match="Rank 0 failed with code 3"):
            QualificationRunner()._run_subprocess_ranks(backend="gloo", world_size=2)
        assert len(spawned) == 2
        assert all(p.poll() is not None for p in spawned)

    def test_rank_arguments_reach_the_worker(self, monkeypatch, spawned):
        """Each rank gets MODE RANK WORLD_SIZE INIT_METHOD BACKEND OUT_FILE EXTRA_JSON and its output is read back."""
        self._fake_worker(monkeypatch, textwrap.dedent("""
            import json, sys
            mode, rank, world, init, backend, out, extra = sys.argv[1:8]
            json.dump({"argv": [mode, rank, world, init[:6], backend], "extra": json.loads(extra)}, open(out, "w"))
        """))
        results = QualificationRunner()._run_subprocess_ranks(
            backend="gloo", world_size=2, steps=3, mode="checkpoint", run_dir="/run")
        assert [r["argv"] for r in results] == [["checkpoint", str(r), "2", "tcp://", "gloo"] for r in range(2)]
        assert results[0]["extra"] == {"steps": 3, "nodes": 1, "ranks_per_node": 2, "run_dir": "/run",
                                       "aggregation": {"strategy": "gradient_allreduce", "sync_every_steps": 1}}


class TestProductPathSuite:
    """The CPU Gloo suite trains through replicated_execution.run_train, not ReplicatedTrainer."""

    @staticmethod
    def _fake_run_train(tmp_path, diverge):
        import torch
        calls = []

        def run_train(config, run_dir, *, profile, checkpoint_every, service_policy):
            calls.append({"config": config, "profile": profile, "checkpoint_every": checkpoint_every,
                          "service_policy": service_policy})
            checkpoint = tmp_path / "ckpt"
            checkpoint.mkdir()
            for rank in range(2):
                tensor = {"w": torch.tensor([1.0 + (rank if diverge else 0)])}
                torch.save({"graph": tensor, "prior": {"z": torch.zeros(2)}, "ema_graph": tensor, "ema_prior": {}},
                           checkpoint / f"rank-{rank:05d}.pt")
            return {"status": "complete", "steps": 2, "checkpoint_path": str(checkpoint)}

        return run_train, calls

    def test_suite_drives_run_train_and_compares_checkpointed_replicas(self, tmp_path):
        fake, calls = self._fake_run_train(tmp_path, diverge=False)
        with patch("hypergan.replicated_execution.run_train", fake):
            result = QualificationRunner()._run_suite_cpu_replicated_gloo({})
        assert result["status"] == "passed"
        assert "replicated_execution.run_train" in result["details"]["path"]
        (call,) = calls
        assert call["profile"]["execution"] == {"name": "cpu-replicated-gloo", "world_size": 2, "accumulation_steps": 1}
        assert call["config"]["training"]["steps"] == call["checkpoint_every"] == 2
        assert call["service_policy"]["total_timeout"] <= qualify.PRODUCT_PATH_TIMEOUT_SECONDS

    def test_suite_fails_when_replicas_diverge(self, tmp_path):
        fake, _ = self._fake_run_train(tmp_path, diverge=True)
        with patch("hypergan.replicated_execution.run_train", fake):
            with pytest.raises(QualificationCheckFailed, match="graph diverged between rank 0 and rank 1"):
                QualificationRunner()._run_suite_cpu_replicated_gloo({})


# ==============================================================================
# 7. python -O
# ==============================================================================

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
                runner._run_suite_aggregation({})
            except qualify.QualificationCheckFailed:
                print("suite-check-raised")
    """))
    proc = subprocess.run([sys.executable, "-O", str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["validator-raised", "suite-check-raised"]


# ==============================================================================
# 8. Real suites on CPU
# ==============================================================================

@pytest.mark.heavy
def test_cpu_gloo_suite_passes_through_the_product_path(tmp_path):
    receipt = qualify.run_qualification(output=tmp_path / "r.json", suite="cpu_replicated_gloo")
    validate_qualification_receipt(receipt)
    (suite,) = receipt["suites"]
    assert suite["status"] == "passed", suite
    assert "replicated_execution.run_train" in suite["details"]["path"]


@pytest.mark.heavy
def test_worker_ranks_agree_and_record_every_step():
    """Two python -m hypergan.qualify_worker ranks train in lockstep under gradient_allreduce."""
    results = QualificationRunner()._run_subprocess_ranks(backend="gloo", world_size=2, steps=2)
    assert [r["rank"] for r in results] == [0, 1]
    assert all(r["passed"] and len(r["weight_history"]) == 2 for r in results)
    assert results[0]["weight_history"] == results[1]["weight_history"]


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


@pytest.mark.heavy
def test_checkpoint_suite_saves_and_restores_on_every_rank(tmp_path):
    receipt = qualify.run_qualification(output=tmp_path / "r.json", suite="checkpoint_compatibility")
    (suite,) = receipt["suites"]
    assert suite["status"] == "passed", suite
    assert suite["details"]["manifest_verified"] is True

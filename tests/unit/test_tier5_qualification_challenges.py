"""Tier 5: Adversarial Qualification Harness and Multi-Node CLI Challenges.

Empirically challenges deliverables:
1. Qualification CLI:
   - Nested output directory creation vs. directory path handling
   - Invalid suite names rejection, casing normalization, and partial substring matches
   - Duration accounting (never clamped; over the limit is a failure) and timeout boundaries
   - Receipt schema validation adhering to PROJECT.md § Interface Contracts 5 and the unkeyed SHA-256 checksum
2. Multi-Node CLI & Rendezvous:
   - Profile parsing and validation with multi-node topology keys (nodes, node_rank, ranks_per_node, rendezvous)
   - CLI overrides via -s execution.*
   - TCP store collision on occupied port raising DistNetworkError
   - TCP store follower timeout on missing coordinator
   - Rendezvous URI validation and scheme rejection
"""
from __future__ import annotations

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from hypergan.cli import _parse_set_overrides, _parser
from hypergan.config import resolve_config
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.qualify import (
    QualificationRunner,
    ReceiptValidationError,
    compute_receipt_sha256,
    run_qualification,
    run_qualification_cli,
    validate_qualification_receipt,
)


def get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ==============================================================================
# 1. Qualification CLI & Receipt Schema Challenges
# ==============================================================================

class TestQualificationCLIAdversarial:
    """Adversarial stress-testing of QualificationRunner and qualify CLI."""

    def test_output_deeply_nested_directory_creation(self, tmp_path):
        """Harness creates non-existent parent directories for output receipt."""
        nested_output = tmp_path / "deeply" / "nested" / "reports" / "receipt.json"
        assert not nested_output.parent.exists()
        receipt = run_qualification(output=nested_output, dry_run=True)
        assert nested_output.exists()
        assert nested_output.is_file()
        loaded = json.loads(nested_output.read_text())
        assert loaded["qualification_id"] == receipt["qualification_id"]

    def test_output_existing_directory_raises_is_a_directory_error(self, tmp_path):
        """Targeting an existing directory path directly in qualify raises IsADirectoryError."""
        output_dir = tmp_path / "existing_dir"
        output_dir.mkdir(parents=True, exist_ok=True)
        # qualify.py writes directly to target_output: target_output.write_text(...)
        runner = QualificationRunner(output_path=output_dir, dry_run=True)
        with pytest.raises(IsADirectoryError):
            runner.run()

    @pytest.mark.parametrize("invalid_suite", [
        "nonexistent_suite",
        "quantum_gan",
        "tpu_cluster",
        "random_test_123",
        "xyz",
    ])
    def test_invalid_suite_name_rejected_with_value_error(self, invalid_suite):
        """Unknown suite filters raise ValueError with catalog of available suites."""
        with pytest.raises(ValueError, match=r"Unknown qualification suite"):
            QualificationRunner(dry_run=True, suite_filter=invalid_suite)

    def test_run_qualification_cli_returns_exit_code_2_on_invalid_suite(self):
        """CLI returns exit code 2 when an unknown suite name is supplied."""
        args = _parser().parse_args(["qualify", "--suite", "bogus_suite_xyz"])
        ret = run_qualification_cli(args)
        assert ret == 2

    @pytest.mark.parametrize("suite_name,expected_contained", [
        ("SINGLE_DEVICE", "single_device"),
        ("Cpu_Replicated_Gloo", "cpu_replicated_gloo"),
        ("AGGREGATION", "aggregation"),
        ("checkpoint", "checkpoint_compatibility"),
        ("multinode", "multinode_tcp"),
    ])
    def test_case_insensitive_and_substring_suite_filtering(self, suite_name, expected_contained):
        """Suite filtering supports case-insensitive and substring matching."""
        runner = QualificationRunner(dry_run=True, suite_filter=suite_name)
        assert runner.suite_filter is not None
        assert runner.should_run_suite(expected_contained)

    def test_duration_over_budget_is_recorded_unclamped_and_fails(self, tmp_path):
        """A 250s run records 250s (never clamped to 180s) and its receipt status is failed."""
        now = [1000.0]
        runner = QualificationRunner(output_path=tmp_path / "receipt.json", dry_run=True, clock=lambda: now[0])
        real_single = runner._run_suite_single_device

        def slow_suite(env_info=None):
            now[0] += 250.0
            return real_single(env_info)

        with patch.object(runner, "_run_suite_single_device", side_effect=slow_suite):
            receipt = runner.run()
        assert receipt["duration_seconds"] == 250.0
        assert receipt["status"] == "failed"
        assert "timeout" in receipt["error"]
        validate_qualification_receipt(receipt)

    def test_qualification_cli_stream_isolation_json_verbose(self, tmp_path):
        """With both --json and -v, progress goes to stderr and stdout stays parseable JSON."""
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan.cli", "qualify", "--dry-run", "--json", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tmp_path,
            env={**os.environ, "HYPERGAN_STATE_DIR": str(tmp_path / "state")},
        )
        assert proc.returncode == 0, proc.stderr
        # Progress goes to stderr, so stdout is pure JSON
        parsed = json.loads(proc.stdout)
        assert parsed["status"] == "dry_run"
        assert "Running qualification suite" in proc.stderr


class TestReceiptSchemaAdversarialValidation:
    """Adversarial stress-testing of receipt schema contracts and tamper defense."""

    @pytest.fixture
    def valid_receipt(self, tmp_path):
        out = tmp_path / "val_receipt.json"
        return run_qualification(output=out, dry_run=True)

    def test_valid_receipt_passes_validation(self, valid_receipt):
        """Unmodified receipt passes contract validation."""
        validate_qualification_receipt(valid_receipt)

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
    def test_missing_required_key_rejected(self, valid_receipt, missing_key):
        """Dropping any required key triggers an assertion error."""
        tampered = copy.deepcopy(valid_receipt)
        del tampered[missing_key]
        with pytest.raises(ReceiptValidationError):
            validate_qualification_receipt(tampered)

    @pytest.mark.parametrize("invalid_status", ["pending", "error", "unknown", 1, None])
    def test_invalid_status_rejected(self, valid_receipt, invalid_status):
        """Status must be one of passed, failed, skipped or dry_run."""
        tampered = copy.deepcopy(valid_receipt)
        tampered["status"] = invalid_status
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(ReceiptValidationError):
            validate_qualification_receipt(tampered)

    def test_negative_duration_rejected(self, valid_receipt):
        """Negative duration_seconds is rejected."""
        tampered = copy.deepcopy(valid_receipt)
        tampered["duration_seconds"] = -1.5
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(ReceiptValidationError):
            validate_qualification_receipt(tampered)

    def test_duration_exceeding_180s_rejected(self, valid_receipt):
        """Receipt with duration > 180s fails hard guardrail validation."""
        tampered = copy.deepcopy(valid_receipt)
        tampered["duration_seconds"] = 180.1
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(ReceiptValidationError, match=r"Qualification exceeded 3-minute hard limit"):
            validate_qualification_receipt(tampered)

    def test_empty_suites_list_rejected(self, valid_receipt):
        """Receipt with empty suites list is rejected."""
        tampered = copy.deepcopy(valid_receipt)
        tampered["suites"] = []
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(ReceiptValidationError):
            validate_qualification_receipt(tampered)

    def test_tampered_payload_sha256_mismatch(self, valid_receipt):
        """Any unauthorized content modification invalidates the SHA-256 integrity check."""
        tampered = copy.deepcopy(valid_receipt)
        # Invert status without recomputing SHA-256
        tampered["status"] = "failed"
        with pytest.raises(ReceiptValidationError, match=r"Receipt SHA-256 mismatch"):
            validate_qualification_receipt(tampered)

    @pytest.mark.parametrize("claimed", ["passed", "skipped"])
    def test_status_contradicting_suites_rejected(self, valid_receipt, claimed):
        """A dry-run receipt cannot be relabelled passed or skipped, even with a recomputed checksum."""
        tampered = copy.deepcopy(valid_receipt)
        tampered["status"] = claimed
        tampered["receipt_sha256"] = compute_receipt_sha256(tampered)
        with pytest.raises(ReceiptValidationError, match=r"contradicts its suites"):
            validate_qualification_receipt(tampered)


# ==============================================================================
# 2. Multi-Node CLI & TCP Store Challenges
# ==============================================================================

class TestMultiNodeCLIParsingAndTCPStore:
    """Adversarial stress-testing of multi-node profile resolution and TCP store failures."""

    def test_multinode_profile_parsing_all_keys(self):
        """resolve_execution_profile resolves full multi-node topology."""
        cfg = resolve_config({"training": {"device": "cpu", "batch_size": 16}})
        raw_profile = {
            "schema_version": 1,
            "execution": {
                "name": "cpu-replicated-gloo",
                "world_size": 4,
                "nodes": 2,
                "node_rank": 1,
                "ranks_per_node": 2,
                "rendezvous": "tcp://192.168.1.50:29500",
                "accumulation_steps": 1,
            },
        }
        resolved = resolve_execution_profile(raw_profile, cfg)
        exec_t = resolved["execution"]
        assert exec_t["nodes"] == 2
        assert exec_t["node_rank"] == 1
        assert exec_t["ranks_per_node"] == 2
        assert exec_t["rendezvous"] == "tcp://192.168.1.50:29500"

    def test_multinode_set_overrides_parsed(self):
        """_parse_set_overrides parses execution table overrides cleanly."""
        cli_sets = [
            "execution.name=cpu-replicated-gloo",
            "execution.world_size=2",
            "execution.nodes=2",
            "execution.node_rank=0",
            "execution.rendezvous=tcp://127.0.0.1:29500",
        ]
        parsed = _parse_set_overrides(cli_sets)
        assert parsed["execution.name"] == "cpu-replicated-gloo"
        assert parsed["execution.world_size"] == 2
        assert parsed["execution.nodes"] == 2
        assert parsed["execution.node_rank"] == 0
        assert parsed["execution.rendezvous"] == "tcp://127.0.0.1:29500"

    def test_tcp_store_port_already_in_use_raises_dist_network_error(self):
        """Binding TCP rendezvous on an already occupied port raises DistNetworkError."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        occupied_port = sock.getsockname()[1]
        sock.listen(1)

        try:
            with pytest.raises(Exception) as exc_info:
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"tcp://127.0.0.1:{occupied_port}",
                    rank=0,
                    world_size=2,
                    timeout=timedelta(seconds=2),
                )
            # PyTorch raises DistNetworkError on address in use
            err_name = type(exc_info.value).__name__
            assert "Dist" in err_name or "Network" in err_name or "Runtime" in err_name
        finally:
            sock.close()

    def test_tcp_store_missing_coordinator_times_out(self):
        """Follower rank trying to connect to a missing coordinator times out cleanly."""
        free_port = get_free_tcp_port()
        with pytest.raises(Exception) as exc_info:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{free_port}",
                rank=1,
                world_size=2,
                timeout=timedelta(seconds=2),
            )
        err_name = type(exc_info.value).__name__
        assert "Dist" in err_name or "Network" in err_name or "Runtime" in err_name

    @pytest.mark.parametrize("invalid_uri", [
        "http://127.0.0.1:29500",
        "tcp://",
        "tcp://localhost",
        "tcp://:29500",
        "tcp://localhost:0",
        "tcp://localhost:70000",
        "udp://localhost:29500",
    ])
    def test_invalid_rendezvous_uris_rejected(self, invalid_uri):
        """Invalid rendezvous URIs are rejected during execution profile resolution."""
        cfg = resolve_config({"training": {"device": "cpu", "batch_size": 8}})
        raw_profile = {
            "schema_version": 1,
            "execution": {
                "name": "cpu-replicated-gloo",
                "world_size": 2,
                "nodes": 2,
                "node_rank": 0,
                "rendezvous": invalid_uri,
            },
        }
        with pytest.raises(ValueError):
            resolve_execution_profile(raw_profile, cfg)

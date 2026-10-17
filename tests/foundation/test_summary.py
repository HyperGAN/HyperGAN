"""Tests for atomic summary.json lifecycle, schema validation, step-named eval dirs, and metric cadence."""
from datetime import datetime
import json
import math
from pathlib import Path
import time
import pytest

from hypergan.config import resolve_config
from hypergan.metrics import DEFAULT_METRICS, metric_catalog, select_metrics
from hypergan.metric_evaluation import _EVAL_DIR_RE
from hypergan.summary import (
    atomic_write_json,
    collect_checkpoints,
    collect_evaluations,
    compute_best_evaluation,
    read_summary_json,
    write_summary_json,
)


def test_atomic_write_json_atomicity_and_fsync(tmp_path):
    target = tmp_path / "summary.json"
    data = {"run_id": "test_run", "step": 100, "status": "running"}
    atomic_write_json(target, data)
    assert target.is_file()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == data
    # Temporary files should be cleaned up
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_summary_json_initialization(tmp_path):
    manifest = {
        "run_id": "init_run",
        "status": "initializing",
        "steps": 0,
        "config": {"name": "cifar-init", "training": {"steps": 5000}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_summary_json(tmp_path, manifest)

    summary = read_summary_json(tmp_path)
    assert summary is not None
    assert summary["run_id"] == "init_run"
    assert summary["name"] == "cifar-init"
    assert summary["status"] == "initializing"
    assert summary["step"] == 0
    assert summary["total_steps"] == 5000
    assert summary["best"] is None
    assert summary["checkpoints"] == []
    assert summary["evaluations"] == []
    assert summary["updated_at"].endswith("+00:00") or summary["updated_at"].endswith("Z")


def test_write_summary_json_cadence_and_eta(tmp_path):
    manifest = {
        "run_id": "run_cadence",
        "status": "running",
        "steps": 1000,
        "steps_per_second": 20.0,
        "config": {"name": "cadence-test", "training": {"steps": 5000}},
    }
    metrics = {"loss/d": 0.35, "loss/g": 1.25}
    write_summary_json(tmp_path, manifest, metrics=metrics)

    summary = read_summary_json(tmp_path)
    assert summary["step"] == 1000
    assert summary["steps_per_sec"] == 20.0
    # Remaining: 4000 steps @ 20 step/s = 200.0s ETA
    assert summary["eta_seconds"] == 200.0
    assert summary["latest"]["loss/d"] == 0.35
    assert summary["latest"]["loss/g"] == 1.25


def test_write_summary_json_with_checkpoints_and_evaluations(tmp_path):
    # Setup checkpoints
    ckpt_dir = tmp_path / "checkpoints" / "0001-step-0001000-abcdef"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "manifest.json").write_text(json.dumps({"step": 1000, "state_sha256": "hash1"}), encoding="utf-8")

    # Setup step-named evaluation directory
    eval_dir = tmp_path / "metrics" / "evaluations" / "step-0001000-fid50k_train-12345678"
    eval_dir.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "metric_id": "fid50k_train",
        "source_step": 1000,
        "result": {"value": 15.42, "step": 1000},
    }
    (eval_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    manifest = {
        "run_id": "eval_run",
        "status": "running",
        "steps": 1000,
        "config": {"training": {"steps": 10000}, "checkpoints": {"keep_best": {"metric": "fid50k_train", "mode": "min"}}},
    }
    write_summary_json(tmp_path, manifest)

    summary = read_summary_json(tmp_path)
    assert len(summary["checkpoints"]) == 1
    assert summary["checkpoints"][0]["step"] == 1000
    assert summary["checkpoints"][0]["metrics"] == {"fid50k_train": 15.42}

    assert len(summary["evaluations"]) == 1
    assert summary["evaluations"][0]["metric"] == "fid50k_train"
    assert summary["evaluations"][0]["value"] == 15.42

    assert summary["best"] is not None
    assert summary["best"]["metric"] == "fid50k_train"
    assert summary["best"]["value"] == 15.42
    assert summary["best"]["step"] == 1000
    assert summary["best"]["checkpoint"] == "0001-step-0001000-abcdef"
    assert summary["best_fid"] == 15.42


def test_step_named_eval_dir_regex_matches_new_and_legacy():
    # New convention: step-0001000-fid50k_train-12345678
    assert _EVAL_DIR_RE.match("step-0001000-fid50k_train-12345678")
    assert _EVAL_DIR_RE.match("step-0050000-distance-abcdef12")
    # 8-digit intermediate convention
    assert _EVAL_DIR_RE.match("step-00001000-a1b2c3d4e5f60718293a4b5c6d7e8f90")
    # Legacy 32-hex UUID convention
    assert _EVAL_DIR_RE.match("a1b2c3d4e5f60718293a4b5c6d7e8f90")

    # Invalid names rejected
    assert not _EVAL_DIR_RE.match("invalid_name")
    assert not _EVAL_DIR_RE.match("step-abc-fid")
    assert not _EVAL_DIR_RE.match(".hidden")


def test_default_metric_cadence_every_steps_10():
    assert DEFAULT_METRICS["every_steps"] == 10
    config = resolve_config({})
    assert config["metrics"]["every_steps"] == 10

    catalog = metric_catalog(config)
    # At step 1: mode should be 'cadence' with empty metrics
    values, statuses, mode = select_metrics(config, catalog, {"d_loss": 1.0, "g_loss": 1.0}, 1, 0.1)
    assert mode == "cadence"
    assert values == {}

    # At step 10: mode should be 'sampled' with full metrics
    values, statuses, mode = select_metrics(config, catalog, {"d_loss": 1.0, "g_loss": 1.0}, 10, 0.1)
    assert mode == "sampled"
    assert "loss/total" in values


def test_compute_best_evaluation_modes():
    evals = [
        {"step": 1000, "metric": "fid50k_train", "value": 25.0},
        {"step": 2000, "metric": "fid50k_train", "value": 18.5},
        {"step": 3000, "metric": "fid50k_train", "value": 20.0},
    ]
    best = compute_best_evaluation(evals)
    assert best["value"] == 18.5
    assert best["step"] == 2000

    # Max mode (e.g. accuracy)
    acc_evals = [
        {"step": 1000, "metric": "accuracy", "value": 0.82},
        {"step": 2000, "metric": "accuracy", "value": 0.94},
    ]
    best_acc = compute_best_evaluation(acc_evals)
    assert best_acc["value"] == 0.94
    assert best_acc["step"] == 2000


def test_read_summary_json_nonexistent_or_invalid(tmp_path):
    assert read_summary_json(tmp_path) is None
    (tmp_path / "summary.json").write_text("not json", encoding="utf-8")
    assert read_summary_json(tmp_path) is None


def test_atomic_write_json_retries_windows_sharing_without_changing_visible_state(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    import hypergan.summary as summary
    target = tmp_path / 'summary.json'
    old, new = {'step': 1}, {'step': 2}
    atomic_write_json(target, old)
    replacement, attempts, delays = os.replace, [], []
    def replace(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text()) == old
        assert json.loads(Path(source).read_text()) == new
        if len(attempts) < 3:
            error = PermissionError(13, 'temporary Windows sharing conflict')
            error.winerror = 5 if len(attempts) == 1 else 32
            raise error
        replacement(source, destination)
    monkeypatch.setattr(summary, 'os', SimpleNamespace(**{**vars(os), 'name': 'nt', 'replace': replace}))
    monkeypatch.setattr(summary, 'time', SimpleNamespace(sleep=delays.append))
    atomic_write_json(target, new)
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert delays == [.01, .02]
    assert json.loads(target.read_text()) == new
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('platform, winerror, expected_attempts', [('nt', 5, 11), ('nt', 17, 1), ('posix', 5, 1)])
def test_atomic_write_json_propagates_permanent_replace_failure(tmp_path, monkeypatch, platform, winerror, expected_attempts):
    import os
    from types import SimpleNamespace
    import hypergan.summary as summary
    target = tmp_path / 'summary.json'
    atomic_write_json(target, {'step': 1})
    attempts, delays = [], []
    error = PermissionError(13, 'permanent replace failure')
    error.winerror = winerror
    def replace(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text()) == {'step': 1}
        raise error
    monkeypatch.setattr(summary, 'os', SimpleNamespace(**{**vars(os), 'name': platform, 'replace': replace}))
    monkeypatch.setattr(summary, 'time', SimpleNamespace(sleep=delays.append))
    with pytest.raises(PermissionError) as raised:
        atomic_write_json(target, {'step': 2})
    assert raised.value is error
    assert len(attempts) == expected_attempts
    assert len(delays) == expected_attempts - 1
    assert sum(delays) <= .75 + 1e-9
    assert json.loads(target.read_text()) == {'step': 1}
    assert not list(tmp_path.glob('*.tmp'))

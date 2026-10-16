"""Tests for hypergan status, runs ls, runs compare, and gc (Milestone 5).

Verifies inspection commands operate correctly, cleanly, and strictly torch-free.
"""
import csv
from hypergan.run_state import run_lock
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_cli(tmp_path, *args, isolated=True):
    cmd = [sys.executable]
    if isolated:
        cmd.append("-I")
    cmd.extend(["-m", "hypergan", *map(str, args)])
    return subprocess.run(
        cmd,
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )


def _make_mock_run(run_dir: Path, *, name="exp-test", status="completed", steps=100000,
                   total_steps=200000, best_fid=18.42, with_summary=True, with_views=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": status,
        "steps": steps,
        "seconds": 3600.0,
        "training_seconds": 3600.0,
        "steps_per_second": 42.0,
        "checkpoint_path": f"checkpoints/att-1-step-{steps:08d}/state.pt",
        "config": {
            "name": name,
            "train": {"device": "cuda:0", "steps": total_steps},
            "training": {"device": "cuda:0", "steps": total_steps},
        },
        "warnings": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Launch file for PID
    launch_dir = run_dir / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    (launch_dir / "0001.json").write_text(json.dumps({"pid": 12345}), encoding="utf-8")

    # Checkpoints
    ckpt_dir = run_dir / "checkpoints" / f"att-1-step-{steps:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "manifest.json").write_text(json.dumps({"step": steps}), encoding="utf-8")
    (ckpt_dir / "state.pt").write_bytes(b"mock_state")

    # Evaluations
    eval_dir = run_dir / "metrics" / "evaluations" / f"step-{steps:07d}-fid50k_train-abcd1234"
    eval_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "metric_id": "fid50k_train",
        "metric": "fid50k_train",
        "step": steps,
        "source_step": steps,
        "result": {"value": best_fid, "step": steps},
    }
    (eval_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    # Events
    events = [
        {"event": "start", "step": 0, "attempt_id": "0001"},
        {"event": "train", "step": 0, "metrics": {"loss/g_total": 2.5, "loss/d_total": 1.2}},
        {"event": "train", "step": steps // 2, "metrics": {"loss/g_total": 1.5, "loss/d_total": 0.8}},
        {"event": "checkpoint", "step": steps, "checkpoint_path": f"checkpoints/att-1-step-{steps:08d}/state.pt"},
        {"event": "evaluation", "step": steps, "metric_id": "fid50k_train", "value": best_fid},
        {"event": "train", "step": steps, "metrics": {"loss/g_total": 1.1, "loss/d_total": 0.5}},
        {"event": status, "step": steps},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    if with_views:
        views_dir = run_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        (views_dir / "contributions.jsonl").write_bytes(b"view data" * 1000)
        (views_dir / "projection.json").write_text("{}", encoding="utf-8")

    if with_summary:
        from hypergan.summary import write_summary_json
        write_summary_json(run_dir, status=status)


def test_torch_free_inspection_invariant(tmp_path):
    """Assert all inspection and GC functions execute strictly torch-free."""
    run_dir = tmp_path / "run_torch_free"
    _make_mock_run(run_dir)

    code = f"""
import sys
import hypergan.inspection as hi

# Run inspection calls
s = hi.run_status(r"{run_dir}")
assert s["name"] == "exp-test"

runs = hi.list_runs(r"{tmp_path}")
assert len(runs) >= 1

comp = hi.compare_runs([r"{run_dir}"], metric="fid50k_train", at_steps="100k,best,latest")
assert comp["metric"] == "fid50k_train"

gc_res = hi.gc_run(r"{run_dir}", dry_run=True)
assert gc_res["reclaimed_bytes"] > 0

# Strict invariant: neither torch nor particlegan loaded
assert "torch" not in sys.modules, f"torch leaked: {{[m for m in sys.modules if 'torch' in m]}}"
assert "particlegan" not in sys.modules, f"particlegan leaked: {{[m for m in sys.modules if 'particlegan' in m]}}"
"""
    probe = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr


def test_status_with_summary_fast_path(tmp_path):
    run_dir = tmp_path / "run_status_fast"
    _make_mock_run(run_dir, name="cifar-fast", steps=100000, total_steps=200000, best_fid=15.25)

    # Text summary output
    res = run_cli(tmp_path, "status", run_dir, isolated=False)
    assert res.returncode == 0, res.stderr
    assert "HyperGAN Run Status:" in res.stdout
    assert "cifar-fast" in res.stdout
    assert "100,000 / 200,000" in res.stdout
    assert "15.25" in res.stdout
    assert "checkpoints/" in res.stdout

    # JSON export
    res_json = run_cli(tmp_path, "status", run_dir, "--json", isolated=False)
    assert res_json.returncode == 0, res_json.stderr
    data = json.loads(res_json.stdout)
    assert data["name"] == "cifar-fast"
    assert data["step"] == 100000
    assert data["total_steps"] == 200000
    assert data["best"]["value"] == 15.25


def test_status_with_manifest_fallback(tmp_path):
    run_dir = tmp_path / "run_status_legacy"
    _make_mock_run(run_dir, name="legacy-run", steps=50000, with_summary=False)
    assert not (run_dir / "summary.json").exists()

    res = run_cli(tmp_path, "status", run_dir, isolated=False)
    assert res.returncode == 0, res.stderr
    assert "legacy-run" in res.stdout
    assert "50,000" in res.stdout

    res_json = run_cli(tmp_path, "status", run_dir, "--json", isolated=False)
    assert res_json.returncode == 0, res_json.stderr
    data = json.loads(res_json.stdout)
    assert data["name"] == "legacy-run"
    assert data["step"] == 50000


def test_status_active_run_lock(tmp_path):
    run_dir = tmp_path / "run_active"
    _make_mock_run(run_dir, name="active-run", status="running")

    lock_file = run_dir / ".run.lock"
    with run_lock(lock_file.parent):
        res = run_cli(tmp_path, "status", run_dir, isolated=False)
        assert res.returncode == 0, res.stderr
        assert "(active)" in res.stdout


def test_runs_ls_discovery_and_sorting(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    r1 = runs_dir / "run-alpha"
    _make_mock_run(r1, name="alpha", steps=50000, best_fid=25.0, status="running")

    r2 = runs_dir / "run-beta"
    _make_mock_run(r2, name="beta", steps=100000, best_fid=15.0, status="completed")

    r3 = runs_dir / "run-gamma"
    _make_mock_run(r3, name="gamma", steps=80000, best_fid=20.0, status="stopped")

    # Table listing
    res = run_cli(tmp_path, "runs", "ls", runs_dir, isolated=False)
    assert res.returncode == 0, res.stderr
    assert "alpha" in res.stdout
    assert "beta" in res.stdout
    assert "gamma" in res.stdout

    # Sort by FID (ascending: beta 15.0, gamma 20.0, alpha 25.0)
    res_fid = run_cli(tmp_path, "runs", "ls", runs_dir, "--sort", "fid", isolated=False)
    lines_fid = [l for l in res_fid.stdout.splitlines() if any(k in l for k in ("alpha", "beta", "gamma"))]
    assert "beta" in lines_fid[0]
    assert "gamma" in lines_fid[1]
    assert "alpha" in lines_fid[2]

    # Sort by step (descending: beta 100k, gamma 80k, alpha 50k)
    res_step = run_cli(tmp_path, "runs", "ls", runs_dir, "--sort", "step", isolated=False)
    lines_step = [l for l in res_step.stdout.splitlines() if any(k in l for k in ("alpha", "beta", "gamma"))]
    assert "beta" in lines_step[0]
    assert "gamma" in lines_step[1]
    assert "alpha" in lines_step[2]

    # Status filter
    res_filter = run_cli(tmp_path, "runs", "ls", runs_dir, "--status", "completed", isolated=False)
    assert "beta" in res_filter.stdout
    assert "alpha" not in res_filter.stdout
    assert "gamma" not in res_filter.stdout

    # JSON export
    res_json = run_cli(tmp_path, "runs", "ls", runs_dir, "--json", isolated=False)
    assert res_json.returncode == 0, res_json.stderr
    items = json.loads(res_json.stdout)
    assert len(items) == 3
    names = {it["name"] for it in items}
    assert names == {"alpha", "beta", "gamma"}


def test_runs_compare_table_and_csv(tmp_path):
    r1 = tmp_path / "run_comp_1"
    _make_mock_run(r1, name="exp1", steps=100000, best_fid=22.5)

    r2 = tmp_path / "run_comp_2"
    _make_mock_run(r2, name="exp2", steps=100000, best_fid=18.0)

    # Table comparison
    res = run_cli(tmp_path, "runs", "compare", r1, r2, "--metric", "fid50k_train", "--at", "100k,best,latest", isolated=False)
    assert res.returncode == 0, res.stderr
    assert "Metric: fid50k_train" in res.stdout
    assert "exp1" in res.stdout
    assert "exp2" in res.stdout
    assert "22.50" in res.stdout
    assert "18.00" in res.stdout

    # CSV comparison
    res_csv = run_cli(tmp_path, "runs", "compare", r1, r2, "--metric", "fid50k_train", "--at", "100k,best,latest", "--csv", isolated=False)
    assert res_csv.returncode == 0, res_csv.stderr
    reader = list(csv.reader(res_csv.stdout.strip().splitlines()))
    assert len(reader) >= 4  # Header + 100k + best + latest
    assert reader[0][0] == "step"
    assert "exp1" in reader[0][1]
    assert "exp2" in reader[0][2]


def test_gc_views_and_compact_events(tmp_path):
    run_dir = tmp_path / "run_to_gc"
    _make_mock_run(run_dir, name="gc-target", steps=20, total_steps=20, with_views=True)

    views_dir = run_dir / "views"
    assert views_dir.is_dir()
    events_file = run_dir / "events.jsonl"
    orig_events_len = len(events_file.read_text(encoding="utf-8").splitlines())

    # Add 25 routine train events with stride 1
    more_events = [{"event": "train", "step": i, "metrics": {"loss/g_total": 1.0}} for i in range(1, 20) if i not in (0, 10, 20)]
    with events_file.open("a", encoding="utf-8") as f:
        for ev in more_events:
            f.write(json.dumps(ev) + "\n")

    pre_gc_size = events_file.stat().st_size

    # Run GC
    res = run_cli(tmp_path, "gc", run_dir, "--stride", "10", isolated=False)
    assert res.returncode == 0, res.stderr
    assert "Cleaned run:" in res.stdout
    assert "Views removed:" in res.stdout
    assert "Events compacted:" in res.stdout

    # views/ directory should be gone
    assert not views_dir.exists()

    # events.jsonl should be compacted
    assert events_file.stat().st_size < pre_gc_size

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("events_compacted") is True
    assert "durable_event_boundary" in manifest


def test_gc_refuses_active_run(tmp_path):
    run_dir = tmp_path / "run_gc_active"
    _make_mock_run(run_dir, status="running")

    lock_file = run_dir / ".run.lock"
    with run_lock(lock_file.parent):
        res = run_cli(tmp_path, "gc", run_dir, isolated=False)
        assert res.returncode != 0
        assert "active training run" in res.stderr


def test_gc_dry_run(tmp_path):
    run_dir = tmp_path / "run_gc_dry"
    _make_mock_run(run_dir, with_views=True)

    views_dir = run_dir / "views"
    assert views_dir.is_dir()

    res = run_cli(tmp_path, "gc", run_dir, "--dry-run", isolated=False)
    assert res.returncode == 0, res.stderr
    assert "Would clean run:" in res.stdout
    # views/ directory still exists
    assert views_dir.is_dir()


def test_gc_all(tmp_path):
    runs_dir = tmp_path / "all_runs"
    runs_dir.mkdir()

    r1 = runs_dir / "run1"
    _make_mock_run(r1, name="r1", status="completed")

    r2 = runs_dir / "run2"
    _make_mock_run(r2, name="r2", status="running")

    lock_file = r2 / ".run.lock"
    with run_lock(lock_file.parent):
        res = run_cli(tmp_path, "gc", "--all", runs_dir, isolated=False)
        assert res.returncode == 0, res.stderr
        assert "GC complete across 2 runs:" in res.stdout
        assert "Cleaned:  1 runs" in res.stdout
        assert "Skipped:  1 active runs" in res.stdout

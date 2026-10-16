"""Exercise installed entrypoints from outside the source checkout."""

import json

from pathlib import Path

import shutil

import subprocess

import sys

import sysconfig

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

def test_cli_resume_without_config_applies_set_overrides(tmp_path):
    config = tmp_path / "model.toml"
    run_dir = tmp_path / "run"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    train_res = run_cli(tmp_path, "train", config, "--run-dir", run_dir, "--steps", "1", "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert train_res.returncode == 0, train_res.stderr

    resume_res = run_cli(tmp_path, "resume", run_dir, "--set", "train.steps=2", "--no-server", isolated=False)
    assert resume_res.returncode == 0, resume_res.stderr
    data = json.loads(resume_res.stdout)
    assert data["status"] == "complete"
    assert data["steps"] == 2

def test_cli_checkpoints_prune(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "att-1-step-00000100-uuid1"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "manifest.json").write_text(json.dumps({"step": 100}), encoding="utf-8")
    (ckpt_dir / "state.pt").write_bytes(b"dummy")

    # hypergan checkpoints prune RUN_DIR --dry-run
    res1 = run_cli(tmp_path, "checkpoints", "prune", run_dir, "--dry-run", isolated=False)
    assert res1.returncode == 0, res1.stderr
    data1 = json.loads(res1.stdout)
    assert data1["dry_run"] is True
    assert data1["retained_count"] == 1

    # alias: hypergan checkpoint prune RUN_DIR --dry-run
    res2 = run_cli(tmp_path, "checkpoint", "prune", run_dir, "--dry-run", isolated=False)
    assert res2.returncode == 0, res2.stderr
    data2 = json.loads(res2.stdout)
    assert data2["dry_run"] is True
    assert data2["retained_count"] == 1

def test_cli_train_optional_run_dir_and_provenance(tmp_path):
    ws_file = tmp_path / "hypergan.toml"
    runs_dir = tmp_path / "custom_runs"
    ws_file.write_text(f"""
[workspace]
root = "{tmp_path}"
runs_root = "{runs_dir}"
""")
    config = tmp_path / "my_experiment.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")

    # Train without --run-dir; should default to {runs_root}/train-{stem}
    train_res = run_cli(tmp_path, "train", config, "--steps", "1", "--no-server", isolated=False)
    assert train_res.returncode == 0, train_res.stderr

    expected_run = runs_dir / "train-my_experiment"
    assert expected_run.is_dir()
    assert (expected_run / "manifest.json").is_file()

    # Verify launch provenance and config snapshot were written
    launch_json = expected_run / "launch" / "0001.json"
    assert launch_json.is_file()
    launch_data = json.loads(launch_json.read_text(encoding="utf-8"))
    assert launch_data["attempt"] == 1
    assert "hypergan" in launch_data["git"]

    config_source = expected_run / "config-source"
    assert config_source.is_dir()
    assert (config_source / "my_experiment.toml").is_file()
    assert (config_source / "hypergan.toml").is_file()


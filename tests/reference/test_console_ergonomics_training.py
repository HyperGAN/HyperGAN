"""Tests for console ergonomics, clean headers, ETA lines, milestone lines, and CLI unification (Milestone 5)."""

import io

import json

from pathlib import Path

import subprocess

import sys

import pytest

import hypergan.api as hg

from hypergan.bounded_cli_output import CLIProgress, TrainingOutput

from hypergan.cli import _parser, _warnings, _print_training_header

def run_cli(tmp_path, *args, isolated=True):
    cmd = [sys.executable]
    if isolated:
        cmd.append("-I")
    cmd.extend(["-m", "hypergan", *map(str, args)])
    return subprocess.run(
        cmd,
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )

def test_resume_alias_of_train_with_run_dir(tmp_path):
    config = tmp_path / "model.toml"
    run_dir = tmp_path / "run_resume_alias"

    run_cli(tmp_path, "new", config, "--device", "cpu")
    t1 = run_cli(tmp_path, "train", config, "--run-dir", run_dir, "--steps", "1",
                 "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert t1.returncode == 0, t1.stderr

    # Resume using `hypergan train RUN_DIR` with --progress-json
    t2 = run_cli(tmp_path, "train", run_dir, "--steps", "2", "--progress-json", "--no-server", isolated=False)
    assert t2.returncode == 0, t2.stderr
    last_line = t2.stdout.strip().splitlines()[-1]
    res_event = json.loads(last_line)
    d2 = res_event.get("manifest", res_event)
    assert d2["steps"] == 2
    assert d2["status"] == "complete"

    # Resume using `hypergan resume RUN_DIR`
    t3 = run_cli(tmp_path, "resume", run_dir, "--steps", "3", "--no-server", isolated=False)
    assert t3.returncode == 0, t3.stderr
    d3 = json.loads(t3.stdout)
    assert d3["steps"] == 3
    assert d3["status"] == "complete"

def test_train_cli_human_summary_footer_and_progress_json(tmp_path):
    config = tmp_path / "model.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")

    # Interactive mode without --progress-json emits human summary footer on stdout
    run_dir_human = tmp_path / "run_human"
    res_human = run_cli(tmp_path, "train", config, "--run-dir", run_dir_human, "--steps", "1",
                        "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert res_human.returncode == 0, res_human.stderr
    assert "Training complete" in res_human.stdout
    assert "Steps:" in res_human.stdout
    assert "Checkpoint:" in res_human.stdout
    assert "Dashboard:" in res_human.stdout

    # Machine mode with --progress-json emits valid JSON result
    run_dir_json = tmp_path / "run_json"
    res_json = run_cli(tmp_path, "train", config, "--run-dir", run_dir_json, "--steps", "1",
                       "--progress-json", "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert res_json.returncode == 0, res_json.stderr
    last_line = res_json.stdout.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed.get("event") == "result"
    assert parsed["manifest"]["status"] == "complete"
    assert parsed["manifest"]["steps"] == 1

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


def test_clean_5_point_header():
    buf = io.StringIO()
    old_stderr = sys.stderr
    try:
        sys.stderr = buf
        _print_training_header(
            run_dir=Path("/tmp/train-demo"),
            experiment_name="demo.toml",
            device="cpu",
            dashboard_url="http://localhost:8765/runs/train-demo",
        )
    finally:
        sys.stderr = old_stderr

    text = buf.getvalue()
    assert "=== HyperGAN ===" in text
    assert "Run directory:  /tmp/train-demo" in text
    assert "Experiment:     demo.toml" in text
    assert "Devices:        cpu (CPU)" in text
    assert "Environment:    Python" in text
    assert "Dashboard:      http://localhost:8765/runs/train-demo" in text


def test_eta_progress_line():
    from hypergan.console_settings import ConsolePolicy

    class MockOutput:
        def __init__(self):
            self.stderr = io.StringIO()
            self.total_steps = 200000
            self.evaluation_reminder = None
            self.policy = ConsolePolicy(progress_every=1)
            self.progress_json = False

    mock_out = MockOutput()
    progress = CLIProgress(mock_out)

    event = {
        "event": "train",
        "step": 12450,
        "total_steps": 200000,
        "steps_per_second": 42.1,
    }
    progress(event)
    line = mock_out.stderr.getvalue()
    assert "step 12,450 / 200,000 (6.2%) · 42.1 it/s · ETA 01:14:14" in line


def test_milestone_lines_and_suppression():
    from hypergan.console_settings import ConsolePolicy

    class MockOutput:
        def __init__(self):
            self.stderr = io.StringIO()
            self.total_steps = 1000
            self.evaluation_reminder = None
            self.policy = ConsolePolicy(progress_every=1)
            self.progress_json = False

    mock_out = MockOutput()
    progress = CLIProgress(mock_out)

    # Intermediate metric and preview should be suppressed
    progress({"event": "metric", "step": 10})
    progress({"event": "preview", "step": 10})
    assert mock_out.stderr.getvalue() == ""

    # Checkpoint milestone
    progress({"event": "checkpoint", "step": 1000, "checkpoint_path": "checkpoints/att-1-step-00001000/state.pt"})
    assert "[CHECKPOINT] step 1,000: saved to checkpoints/att-1-step-00001000/state.pt" in mock_out.stderr.getvalue()

    # Evaluation milestone with new best
    progress({"event": "evaluation", "step": 1000, "metric_id": "fid50k_train", "value": 18.42, "best": True})
    assert "[EVAL] step 1,000: fid50k_train = 18.42 (new best!)" in mock_out.stderr.getvalue()


def test_warning_collapsing():
    buf = io.StringIO()
    old_stderr = sys.stderr
    try:
        sys.stderr = buf
        config = {
            "warnings": [
                "Custom metrics execute trusted Python in bounded workers; unqualified until separately evaluated.",
                "Custom constructors execute trusted Python code when training or sampling; configuration is not a sandbox.",
                "GPU device memory is low (1.2 GB available)",
            ]
        }
        _warnings(config)
    finally:
        sys.stderr = old_stderr

    text = buf.getvalue()
    # Trust warnings collapsed into single warning:
    assert "warning: custom recipe executes trusted Python code; combinations are unqualified until evaluated" in text
    # Operational warnings kept intact:
    assert "warning: GPU device memory is low (1.2 GB available)" in text
    # Not duplicated
    assert text.count("warning: custom recipe executes trusted Python code") == 1


def test_human_summary_footer():
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    output = TrainingOutput(stdout=stdout_buf, stderr=stderr_buf, progress_json=False)

    manifest = {
        "status": "complete",
        "steps": 200000,
        "seconds": 15150.0,
        "training_seconds": 15150.0,
        "steps_per_second": 42.1,
        "run_id": "train-demo",
        "checkpoint_path": "checkpoints/00200000/state.pt",
        "config": {"training": {"steps": 200000}},
    }

    output.result(manifest, run_dir="/tmp/train-demo", human=True)
    text = stdout_buf.getvalue()
    assert "============================================================" in text
    assert "Training complete" in text
    assert "Steps:       200,000 / 200,000 (100.0%)" in text
    assert "Time:        4h 12m (42.1 it/s)" in text
    assert "Checkpoint:  checkpoints/00200000/state.pt" in text
    assert "Export:      hypergan export /tmp/train-demo --checkpoint best" in text
    assert "Dashboard:   http://localhost:8765/runs/train-demo" in text
    assert "============================================================" in text


def test_human_summary_footer_interrupted():
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    output = TrainingOutput(stdout=stdout_buf, stderr=stderr_buf, progress_json=False)

    manifest = {
        "status": "interrupted",
        "steps": 12450,
        "seconds": 295.0,
        "training_seconds": 295.0,
        "steps_per_second": 42.1,
        "checkpoint_path": "checkpoints/00010000/state.pt",
    }

    output.result(manifest, run_dir="/tmp/train-demo", human=True)
    text = stdout_buf.getvalue()
    assert "Training interrupted at step 12,450" in text
    assert "Time elapsed: 4m 55s (42.1 it/s)" in text
    assert "Latest checkpoint: checkpoints/00010000/state.pt" in text
    assert "Resume run:   hypergan resume /tmp/train-demo" in text


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



def test_viewer_flag_aliases():
    parser = _parser()

    # train parser aliases
    train_args = parser.parse_args(["train", "model.toml", "--host", "127.0.0.1", "--port", "9000", "--dev"])
    assert train_args.server_host == "127.0.0.1"
    assert train_args.server_port == 9000
    assert train_args.viewer_dev is True

    train_alt = parser.parse_args(["train", "model.toml", "--server-host", "127.0.0.1", "--server-port", "9000", "--viewer-dev"])
    assert train_alt.server_host == "127.0.0.1"
    assert train_alt.server_port == 9000
    assert train_alt.viewer_dev is True

    # resume parser aliases
    resume_args = parser.parse_args(["resume", "runs/r1", "--host", "127.0.0.1", "--port", "9000", "--dev"])
    assert resume_args.server_host == "127.0.0.1"
    assert resume_args.server_port == 9000
    assert resume_args.viewer_dev is True

    # serve parser aliases
    serve_args = parser.parse_args(["serve", "runs/r1", "--host", "127.0.0.1", "--port", "9000", "--dev"])
    assert serve_args.host == "127.0.0.1"
    assert serve_args.port == 9000
    assert serve_args.dev is True

    serve_alt = parser.parse_args(["serve", "runs/r1", "--server-host", "127.0.0.1", "--server-port", "9000", "--viewer-dev"])
    assert serve_alt.host == "127.0.0.1"
    assert serve_alt.port == 9000
    assert serve_alt.dev is True

    # view parser aliases
    view_args = parser.parse_args(["view", "--host", "127.0.0.1", "--port", "9000", "--dev"])
    assert view_args.host == "127.0.0.1"
    assert view_args.port == 9000
    assert view_args.dev is True

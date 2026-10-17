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
    run_dir = Path("/tmp/train-demo")
    try:
        sys.stderr = buf
        _print_training_header(
            run_dir=run_dir,
            experiment_name="demo.toml",
            device="cpu",
            dashboard_url="http://localhost:8765/runs/train-demo",
        )
    finally:
        sys.stderr = old_stderr

    text = buf.getvalue()
    assert "=== HyperGAN ===" in text
    assert f"Run directory:  {run_dir}" in text
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
        "best_checkpoint": "checkpoints/00200000/state.pt",
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

    # Verify latest fallback when best_checkpoint is absent
    stdout_buf2 = io.StringIO()
    output2 = TrainingOutput(stdout=stdout_buf2, stderr=stderr_buf, progress_json=False)
    manifest_no_best = dict(manifest)
    manifest_no_best.pop("best_checkpoint", None)
    output2.result(manifest_no_best, run_dir="/tmp/train-demo", human=True)
    text2 = stdout_buf2.getvalue()
    assert "Export:      hypergan export /tmp/train-demo --checkpoint latest" in text2


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

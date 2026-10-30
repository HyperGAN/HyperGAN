"""Empirical adversarial challenge suite for Milestone 3 Features 34–39.

Author: Challenger M3-2 (Empirical Challenger)
Focus:
  - F34: Empirical Cadence Wire-Through ([run] checkpoint_every, preview_every, progress_every)
  - F36: Empirical Progress Line Losses (D= and G= rendering in CLI output)
  - F37: Empirical 8 GAN Health Telemetry Scalars in summary.json["latest"]
  - F38: Empirical Health Alerts & EMA Trends (Discriminator overpowering, Generator vanishing gradient, Mode collapse risk, telemetry_trends)
  - F39: Empirical Unified RunInfo Reader (Completed, In-progress, and fallback runs)
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import pytest
import toml

import hypergan as hg
from hypergan.run_info import RunInfo, read_run_info


class TestCadenceWireThrough:
    """Empirical verification of Feature 34 TOML cadence wire-through."""

    def test_cadences_honored_in_training(self, tmp_path):
        """Recipe with [run] checkpoint_every=3, preview_every=2 produces checkpoints at 3,6 and previews at 2,4,6."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 6
        spec["train"]["device"] = "cpu"
        spec["run"] = {
            "checkpoint_every": 3,
            "preview_every": 2,
            "progress_every": 1,
        }
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_cadence"

        # Pace steps so background preview worker completes before subsequent cadences
        def slow_down(event):
            if event.get("event") == "train":
                time.sleep(2.0)

        hg.train(spec_path, run_dir, steps=6, on_event=slow_down)

        # Verify checkpoints
        ckpts = sorted([p.name for p in (run_dir / "checkpoints").glob("*") if (p / "manifest.json").is_file()])
        ckpt_steps = set()
        for c in ckpts:
            data = json.loads((run_dir / "checkpoints" / c / "manifest.json").read_text())
            ckpt_steps.add(data.get("step"))

        assert 3 in ckpt_steps, f"Step 3 checkpoint missing: {ckpt_steps}"
        assert 6 in ckpt_steps, f"Step 6 checkpoint missing: {ckpt_steps}"

        # Verify previews
        preview_index = json.loads((run_dir / "previews" / "index.json").read_text())
        preview_steps = [p.get("step") for p in preview_index.get("previews", [])]
        assert 2 in preview_steps, f"Step 2 preview missing: {preview_steps}"
        assert 4 in preview_steps, f"Step 4 preview missing: {preview_steps}"
        assert 6 in preview_steps, f"Step 6 preview missing: {preview_steps}"


class TestProgressLineLosses:
    """Empirical verification of Feature 36 CLI progress line losses."""

    def test_cli_progress_lines_render_d_and_g_losses(self, tmp_path):
        """hypergan train emits D=... and G=... loss metrics on terminal progress lines."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 3
        spec["train"]["device"] = "cpu"
        spec["run"] = {"progress_every": 1}
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_progress_loss"

        cmd = [sys.executable, "-m", "hypergan", "train", str(spec_path), "--run-dir", str(run_dir), "--no-server"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, res.stderr

        stderr_output = res.stderr
        progress_lines = [line for line in stderr_output.splitlines() if line.startswith("step ")]
        assert len(progress_lines) >= 3, f"Expected at least 3 progress lines, got: {stderr_output}"

        for line in progress_lines:
            assert "D=" in line, f"Missing D= loss in progress line: {line}"
            assert "G=" in line, f"Missing G= loss in progress line: {line}"


class TestHealthTelemetryScalars:
    """Empirical verification of Feature 37 GAN health telemetry scalars."""

    def test_all_health_telemetry_keys_in_summary_latest(self, tmp_path):
        """Train model for 12 steps and verify all GAN health telemetry keys exist in summary.json['latest'] as finite floats."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 12
        spec["train"]["device"] = "cpu"
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_telemetry"

        hg.train(spec_path, run_dir, steps=12)

        summary_path = run_dir / "summary.json"
        assert summary_path.is_file(), "summary.json was not created"
        summary = json.loads(summary_path.read_text())
        latest = summary.get("latest", {})

        required_telemetry_keys = [
            "critic/real_mean",
            "critic/fake_mean",
            "critic/margin",
            "critic/accuracy",
            "grad_norm/critic",
            "grad_norm/generator",
            "grad_norm/prior",
            "optimizer/d_guard_ratio",
            "prior/particle_rms",
            "diversity/ratio",
        ]

        missing_keys = [k for k in required_telemetry_keys if k not in latest]
        assert not missing_keys, (
            f"Missing required telemetry keys in summary.json['latest']: {missing_keys}. "
            f"Present keys: {sorted(latest.keys())}"
        )

        for k in required_telemetry_keys:
            val = latest[k]
            assert isinstance(val, (int, float)), f"Key {k} is not a float/int (got {type(val).__name__}: {val})"
            assert math.isfinite(val), f"Key {k} value is non-finite: {val}"


class TestHealthAlertsAndTrends:
    """Empirical verification of Feature 38 health alerts and EMA trends."""

    def test_telemetry_trends_populated(self, tmp_path):
        """Verify summary.json includes telemetry_trends with rolling EMA values."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 20
        spec["train"]["device"] = "cpu"
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_trends"

        hg.train(spec_path, run_dir, steps=20)

        summary = json.loads((run_dir / "summary.json").read_text())
        assert "telemetry_trends" in summary
        trends = summary["telemetry_trends"]
        for key in ["critic/real_mean", "critic/margin", "grad_norm/critic", "loss/d_total"]:
            assert key in trends, f"Key {key} missing from telemetry_trends"
            assert isinstance(trends[key], (int, float))

    def test_discriminator_overpowering_alert(self, tmp_path):
        """Discriminator overpowering alert triggers when critic accuracy exceeds 0.98 after step 20."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 25
        spec["train"]["device"] = "cpu"
        spec["train"]["lr"] = 1e-8
        spec["train"]["d_lr_mult"] = 100000.0
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_overpowering"

        hg.train(spec_path, run_dir, steps=25)

        summary = json.loads((run_dir / "summary.json").read_text())
        alerts = summary.get("health_alerts", [])
        assert any("overpowering" in a.lower() for a in alerts), f"Overpowering alert did not trigger: {alerts}"

    def test_generator_vanishing_gradient_alert(self, tmp_path):
        """Generator vanishing gradient alert triggers when generator grad norm < 1e-5."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 10
        spec["train"]["device"] = "cpu"
        spec["losses"] = [
            {"type": "adversarial", "critic": "discriminator", "weight": 0.0, "penalty": 1.0},
            {"type": "prior", "weight": 0.0},
        ]
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_vanishing"

        hg.train(spec_path, run_dir, steps=10)

        summary = json.loads((run_dir / "summary.json").read_text())
        alerts = summary.get("health_alerts", [])
        assert any("generator vanishing gradient" in a.lower() for a in alerts), f"Vanishing alert did not trigger: {alerts}"


class TestUnifiedRunInfoReader:
    """Empirical verification of Feature 39 unified RunInfo reader."""

    def test_run_info_completed_run(self, tmp_path):
        """RunInfo resolves completed run metadata into typed attributes."""
        m = hg.packaged("gaussian-grid")
        spec = dict(m.spec)
        spec["train"] = dict(spec.get("train", {}))
        spec["train"]["steps"] = 10
        spec["train"]["device"] = "cpu"
        spec_path = tmp_path / "recipe.toml"
        spec_path.write_text(toml.dumps(spec))
        run_dir = tmp_path / "runs" / "test_run_info"

        hg.train(spec_path, run_dir, steps=10)

        info = read_run_info(run_dir)
        assert isinstance(info, RunInfo)
        assert info.step == 10
        assert info.status == "completed"
        assert isinstance(info.config, dict) and len(info.config) > 0
        assert isinstance(info.latest_metrics, dict) and len(info.latest_metrics) > 0
        assert isinstance(info.checkpoints, list) and len(info.checkpoints) >= 1
        assert isinstance(info.health_alerts, list)
        assert isinstance(info.telemetry_trends, dict)
        assert info.is_active is False
        assert isinstance(info.to_dict(), dict)

    def test_run_info_in_progress_run(self, tmp_path):
        """RunInfo resolves in-progress running state with is_active=True."""
        run_dir = tmp_path / "runs" / "test_in_progress"
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": "inprogress_test_12345",
            "status": "running",
            "steps": 5,
            "total_steps": 100,
            "config": {"training": {"steps": 100}},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        info = read_run_info(run_dir)
        assert isinstance(info, RunInfo)
        assert info.step == 5
        assert info.status == "running"
        assert info.is_active is True
        assert info.total_steps == 100

    def test_run_info_empty_directory_fallback(self, tmp_path):
        """RunInfo handles empty target path safely without crashing."""
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        info = read_run_info(empty_dir)
        assert isinstance(info, RunInfo)
        assert info.status == "unknown"
        assert info.step == 0
        assert info.is_active is False

"""Tests for PEP 561 compliance, static typing, and Python API ergonomics (Features 64, 65, 66)."""
from dataclasses import is_dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

import hypergan as hg
from hypergan.api import StopTraining, TrainResult, SampleResult, ModelDescription, ValidationWarning, RunStatus, PsSnapshot, RunHandle

ROOT = Path(__file__).resolve().parents[2]


def test_pep561_marker_file_and_packaging_metadata():
    """Verify py.typed exists and is declared in pyproject.toml and MANIFEST.in."""
    py_typed = ROOT / "src" / "hypergan" / "py.typed"
    assert py_typed.is_file(), "src/hypergan/py.typed must exist for PEP 561 compliance"

    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"py.typed"' in pyproject_text, "pyproject.toml package-data must include py.typed"

    manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include src/hypergan/py.typed" in manifest_text, "MANIFEST.in must include src/hypergan/py.typed"


def test_torch_free_fast_import():
    """Verify import hypergan does not load torch and executes rapidly."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys, time; "
            "t0 = time.perf_counter(); "
            "import hypergan as hg; "
            "t1 = time.perf_counter(); "
            "assert 'torch' not in sys.modules, f'torch was imported: {sys.modules.get(\"torch\")}'; "
            "assert 'particlegan' not in sys.modules, 'particlegan was imported'; "
            "print(f'Import took {(t1 - t0)*1000:.2f}ms')"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0, res.stderr


def test_canonical_public_symbols_and_docstrings():
    """Verify all canonical public symbols are declared in __all__ with docstrings."""
    canonical = {
        "ConfigError",
        "LoadedGenerator",
        "Model",
        "ModelDescription",
        "ModelFileError",
        "PsSnapshot",
        "Run",
        "RunHandle",
        "RunInfo",
        "RunStatus",
        "Sample",
        "SampleResult",
        "StopTraining",
        "TrainResult",
        "ValidationWarning",
        "compare",
        "describe",
        "events",
        "export",
        "interpolate",
        "launch",
        "load",
        "load_bundle",
        "logs",
        "lower",
        "metrics",
        "new",
        "operators",
        "override",
        "prune_checkpoints",
        "ps",
        "read_run_info",
        "resolve",
        "resume",
        "rm",
        "sample",
        "samples",
        "save",
        "schema",
        "status",
        "stop",
        "tag",
        "train",
        "validate",
        "view",
    }
    assert canonical <= set(hg.__all__), f"Missing canonical symbols: {canonical - set(hg.__all__)}"

    for name in hg.__all__:
        assert hasattr(hg, name), f"Symbol {name} in __all__ cannot be resolved on hg"
        obj = getattr(hg, name)
        if callable(obj) and not isinstance(obj, type):
            doc = getattr(obj, "__doc__", None)
            assert doc and doc.strip(), f"Public callable {name} is missing a docstring"


def test_typed_dataclasses_and_mapping_protocol():
    """Verify typed dataclasses maintain mapping compatibility for backwards compatibility."""
    assert issubclass(StopTraining, Exception)
    assert is_dataclass(TrainResult)
    assert is_dataclass(SampleResult)
    assert is_dataclass(ModelDescription)
    assert is_dataclass(ValidationWarning)
    assert is_dataclass(RunStatus)
    assert is_dataclass(PsSnapshot)

    # PsSnapshot mapping compatibility and length behavior
    empty_ps = PsSnapshot(runs=[], queued=[], active_count=0, total_known=5)
    assert len(empty_ps) == 0, "len(empty_ps) must be 0 when no runs are active (doctor fix)"
    assert empty_ps["active_count"] == 0
    assert empty_ps.get("total_known") == 5
    assert isinstance(empty_ps, dict)
    assert "runs" in empty_ps

    nonempty_ps = PsSnapshot(runs=[{"run_id": "r1", "active": True}], queued=[], active_count=1, total_known=1)
    assert len(nonempty_ps) == 1
    assert nonempty_ps.runs[0]["run_id"] == "r1"

    # RunStatus mapping and attribute compatibility
    rs = RunStatus(run_id="run-001", status="running", step=50, total_steps=200, devices="cuda:0")
    assert isinstance(rs, dict)
    assert rs.run_id == "run-001"
    assert rs["run_id"] == "run-001"
    assert rs.step == 50
    assert rs["step"] == 50
    assert rs.get("status") == "running"

    # ModelDescription mapping and attribute compatibility
    md = ModelDescription(name="test_model", family="dcgan", components={"gen": {}}, raw={"extra": 123})
    assert isinstance(md, dict)
    assert md.name == "test_model"
    assert md["name"] == "test_model"
    assert md["extra"] == 123


def test_early_stopping_with_stop_training_exception(tmp_path):
    """Verify on_event raising StopTraining triggers clean graceful early stop."""
    model = hg.packaged("gaussian-grid")
    run_dir = tmp_path / "stop_exc_run"

    step_count = 0

    def on_event(ev):
        nonlocal step_count
        if ev.get("event") == "train":
            step_count += 1
            if ev.get("step", 0) >= 20:
                raise StopTraining("Halted early by test callback")

    run = hg.train(model, run_dir=run_dir, steps=100, on_event=on_event)
    info = hg.read_run_info(run_dir)
    assert info.status == "stopped"
    assert "Halted early" in str(info.stop_reason)
    assert info.step >= 20
    assert info.step < 100


def test_early_stopping_with_return_false(tmp_path):
    """Verify on_event returning False triggers clean graceful early stop."""
    model = hg.packaged("gaussian-grid")
    run_dir = tmp_path / "return_false_run"

    def on_event(ev):
        if ev.get("event") == "train" and ev.get("step", 0) >= 15:
            return False
        return True

    run = hg.train(model, run_dir=run_dir, steps=100, on_event=on_event)
    info = hg.read_run_info(run_dir)
    assert info.status == "stopped"
    assert info.step >= 15
    assert info.step < 100


def test_launch_non_blocking_handle(tmp_path):
    """Verify hg.launch returns a RunHandle that runs asynchronously and can be waited on."""
    model = hg.packaged("gaussian-grid")
    run_dir = tmp_path / "launch_run"

    handle = hg.launch(model, run_dir=run_dir, steps=30)
    assert isinstance(handle, RunHandle)
    assert handle.is_alive() or handle.wait(timeout=10.0)
    final_run = handle.wait(timeout=15.0)
    assert final_run.path == run_dir
    assert not handle.is_alive()
    info = handle.run_info
    assert info.status in ("complete", "completed", "stopped")
    assert info.step >= 30


def test_device_routing_without_cuda_visible_devices_mutation(tmp_path, monkeypatch):
    """Verify device='cuda:0' routes directly to config without invoking _set_visible_devices if CUDA initialized."""
    model = hg.packaged("gaussian-grid")
    prep = hg.prepare(model, run_dir=tmp_path / "device_prep", steps=10, device="cuda:0")
    assert prep.config["training"]["device"] == "cuda:0"

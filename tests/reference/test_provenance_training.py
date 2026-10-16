"""Release identities remain useful outside a Git checkout."""

import json

from pathlib import Path

import subprocess

import pytest

from hypergan.provenance import hypergan_source

def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True,
                                   stderr=subprocess.DEVNULL).strip()

def test_run_train_writes_provenance_inside_run_lock(tmp_path):
    from hypergan.run_controller import run_train
    from hypergan.run_state import run_lock
    from hypergan.config import resolve_config
    import hypergan.api as hg

    # Create a simple valid configuration
    config_file = tmp_path / "model.toml"
    hg.save(hg.override(hg.load(Path(__file__).resolve().parents[2] / "examples" / "paired-linear.toml"),
                        {"training": {"device": "cpu", "steps": 1}}), config_file)
    cfg = resolve_config(hg.lower(hg.load(config_file)))
    cfg["_source_path"] = str(config_file)

    run_dir = tmp_path / "concurrent_run"
    run_dir.mkdir(parents=True)

    # Acquire lock externally simulating a concurrent training process
    with run_lock(run_dir):
        with pytest.raises(RuntimeError, match="Run is already locked"):
            from hypergan.single_execution import SingleProcessExecution
            run_train(cfg, run_dir, steps=1, execution_factory=SingleProcessExecution)

        # Confirm that no launch provenance or config snapshots were created while locked
        assert not (run_dir / "launch").exists()
        assert not (run_dir / "config-source").exists()

    # Now with lock released, run_train succeeds and creates provenance inside lock
    from hypergan.single_execution import SingleProcessExecution
    result = run_train(cfg, run_dir, steps=1, execution_factory=SingleProcessExecution)
    assert result is not None
    assert (run_dir / "launch" / "0001.json").is_file()
    assert (run_dir / "config-source" / "model.toml").is_file()


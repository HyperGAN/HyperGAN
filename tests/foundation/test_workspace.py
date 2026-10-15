"""Tests for standalone workspace discovery, loading, interpolation, and GPU aliasing."""

import os
from pathlib import Path
import pytest

from hypergan.workspace import (
    find_workspace,
    load_workspace,
    interpolate_variables,
    resolve_device_aliases,
)


def test_find_workspace_in_current_dir(tmp_path):
    ws_file = tmp_path / "hypergan.toml"
    ws_file.write_text("[workspace]\nruns_root = 'runs'\n")
    found = find_workspace(tmp_path)
    assert found == ws_file


def test_find_workspace_in_parent_dir(tmp_path):
    ws_file = tmp_path / "hypergan.toml"
    ws_file.write_text("[workspace]\nruns_root = 'runs'\n")
    sub = tmp_path / "experiments" / "cifar"
    sub.mkdir(parents=True)
    fake_config = sub / "exp1.toml"
    fake_config.write_text("format = 'hypergan-model/1'\n")

    found = find_workspace(fake_config)
    assert found == ws_file


def test_find_workspace_not_found(tmp_path):
    sub = tmp_path / "isolated"
    sub.mkdir()
    # Ensure no hypergan.toml anywhere up to root inside this branch
    found = find_workspace(sub)
    # Could be None or an ancestor workspace outside tmp_path; for isolated sub:
    # If there's a workspace in the dev repo, sub might find it if tmp_path is under dev repo,
    # but tmp_path is in /tmp, so it will reach root and return None.
    assert found is None


def test_load_workspace_resolves_roots_and_tables(tmp_path):
    ws_file = tmp_path / "hypergan.toml"
    ws_file.write_text("""
[workspace]
runs_root = "runs"
run_dir = "{runs_root}/train-{stem}"

[paths]
data = "${workspace.root}/data"
models = "${workspace.root}/models"

[gpus]
gpu0 = "GPU-ed080e41-3193-3755-6756-f3d46c433331"
gpu1 = "GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce"

[envs]
develop = { python = "python", src = "src" }
""")
    ws = load_workspace(ws_file)
    assert ws["workspace"]["root"] == str(tmp_path)
    assert ws["workspace"]["runs_root"] == str((tmp_path / "runs").resolve())
    assert ws["paths"]["data"] == str(tmp_path / "data")
    assert ws["gpus"]["gpu0"] == "GPU-ed080e41-3193-3755-6756-f3d46c433331"
    assert ws["envs"]["develop"]["python"] == "python"


def test_load_workspace_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workspace(tmp_path / "nonexistent.toml")


def test_interpolate_variables_basic_and_shorthand(tmp_path):
    ws = {
        "workspace": {
            "root": str(tmp_path),
            "runs_root": str(tmp_path / "runs"),
        },
        "paths": {
            "data": str(tmp_path / "data"),
            "networks": str(tmp_path / "networks"),
        },
    }

    template = "${paths.data}/cifar10"
    res = interpolate_variables(template, ws)
    assert res == str(tmp_path / "data") + "/cifar10"

    # Shorthand alias
    template_short = "${data}/cifar10"
    assert interpolate_variables(template_short, ws) == str(tmp_path / "data") + "/cifar10"

    # Workspace root
    template_root = "${workspace.root}/output"
    assert interpolate_variables(template_root, ws) == str(tmp_path) + "/output"


def test_interpolate_variables_recursive_dict_and_list(tmp_path):
    ws = {
        "workspace": {"root": "/test"},
        "paths": {"data": "/test/data", "checkpoints": "/test/chk"},
    }
    data = {
        "network": {
            "hndl": "${paths.data}/net.hndl",
            "weights": ["${paths.checkpoints}/w1.pt", "${paths.checkpoints}/w2.pt"],
        },
        "step": 100,
    }
    res = interpolate_variables(data, ws)
    assert res["network"]["hndl"] == "/test/data/net.hndl"
    assert res["network"]["weights"] == ["/test/chk/w1.pt", "/test/chk/w2.pt"]
    assert res["step"] == 100


def test_interpolate_variables_env_fallback(monkeypatch):
    monkeypatch.setenv("CUSTOM_RUN_DIR", "/custom/run/path")
    ws = {"workspace": {}, "paths": {}}
    res = interpolate_variables("${CUSTOM_RUN_DIR}/train", ws)
    assert res == "/custom/run/path/train"


def test_interpolate_variables_undefined_raises_key_error():
    ws = {"workspace": {}, "paths": {"data": "/data"}}
    with pytest.raises(KeyError, match=r"Undefined variable in template: '\$missing'"):
        interpolate_variables("${missing}/foo", ws)



def test_resolve_device_aliases():
    ws = {
        "gpus": {
            "gpu0": "GPU-ed080e41-3193-3755-6756-f3d46c433331",
            "gpu1": "GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce",
        }
    }

    # Single string
    assert resolve_device_aliases("gpu0", ws) == "GPU-ed080e41-3193-3755-6756-f3d46c433331"

    # Comma-separated string
    assert resolve_device_aliases("gpu0,gpu1", ws) == "GPU-ed080e41-3193-3755-6756-f3d46c433331,GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce"

    # List of strings
    assert resolve_device_aliases(["gpu0", "gpu1"], ws) == [
        "GPU-ed080e41-3193-3755-6756-f3d46c433331",
        "GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce",
    ]

    # Passthrough explicit indices or existing UUID
    assert resolve_device_aliases("cuda:0", ws) == "cuda:0"
    assert resolve_device_aliases(["0", "1"], ws) == ["0", "1"]

    # Unknown alias when aliases are defined
    with pytest.raises(ValueError, match="Unknown GPU alias 'gpu9'"):
        resolve_device_aliases("gpu9", ws)

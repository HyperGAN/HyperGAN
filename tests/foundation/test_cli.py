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


def test_help_and_version_are_lightweight(tmp_path):
    help_result = run_cli(tmp_path, "--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "validate" in help_result.stdout
    result = run_cli(tmp_path, "--version")
    assert result.returncode == 0
    assert result.stdout.strip() == "hypergan 2.0.0a1"
    probe = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys, hypergan, hypergan.cli; assert 'torch' not in sys.modules; "
         "assert 'particlegan' not in sys.modules"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr


def test_console_entrypoint(tmp_path):
    executable = (
        shutil.which("hypergan", path=sysconfig.get_path("scripts"))
        or shutil.which("hypergan")
        or shutil.which("hypergan", path=sysconfig.get_path("scripts", "posix_user"))
    )
    assert executable, "installed console entrypoint was not generated"
    result = subprocess.run([executable, "version"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hypergan 2.0.0a1"


def test_create_validate_and_no_overwrite(tmp_path):
    project = tmp_path / "my project"
    created = run_cli(tmp_path, "new", project)
    assert created.returncode == 0, created.stderr
    config_path = project / "model.toml" if (project / "model.toml").is_file() else project / "config.toml"
    original = config_path.read_bytes()
    validated = run_cli(tmp_path, "validate", project)
    assert validated.returncode == 0, validated.stderr
    assert isinstance(json.loads(validated.stdout), dict)
    duplicate = run_cli(tmp_path, "new", project)
    assert duplicate.returncode != 0
    assert config_path.read_bytes() == original


def test_bad_config_and_arguments_fail_cleanly(tmp_path):
    config = tmp_path / "invalid.toml"
    config.write_text("not valid = [", encoding="utf-8")
    result = run_cli(tmp_path, "validate", config)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    result = run_cli(tmp_path, "train", config, "--run-dir", tmp_path / "run", "--steps", "0")
    assert result.returncode != 0
    assert "positive integer" in result.stderr


def test_recipe_listing(tmp_path):
    result = run_cli(tmp_path, "recipes")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)


def test_manifest_inspection_without_model_loading(tmp_path):
    manifest = {"status": "completed", "qualification": "numerical reference only"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "model.pt").write_bytes(b"not a model")
    result = run_cli(tmp_path, "inspect", tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == manifest
    missing = run_cli(tmp_path, "inspect", tmp_path / "missing.json")
    assert missing.returncode != 0
    assert "Traceback" not in missing.stderr


@pytest.mark.heavy
@pytest.mark.parametrize("missing", ["numpy", "torch", "particlegan"])
def test_missing_runtime_has_install_guidance(tmp_path, missing):
    project = tmp_path / "project"
    created = run_cli(tmp_path, "new", project)
    assert created.returncode == 0, created.stderr
    # Block optional imports explicitly so this also runs in development envs
    # which already have the training extra installed.
    code = """
import importlib.abc
import sys
blocked = sys.argv.pop(1)
class MissingRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == blocked:
            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)
sys.meta_path.insert(0, MissingRuntime())
from hypergan.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, missing, "train", str(project), "--run-dir", str(tmp_path / "run"), "--steps", "6", "--no-server"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "hypergan[train]" in result.stderr
    assert "unqualified" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("device", [None, "cpu", "cuda:1"])
def test_new_cli_device_selection(tmp_path, device):
    project = tmp_path / "project"
    options = [] if device is None else ["--device", device]
    result = run_cli(tmp_path, "new", project, *options)
    assert result.returncode == 0, result.stderr
    result = run_cli(tmp_path, "validate", project)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["training"]["device"] == (device or "cuda")


def test_invalid_new_device_leaves_no_project(tmp_path):
    project = tmp_path / "project"
    result = run_cli(tmp_path, "new", project, "--device", "cuda:-1")
    assert result.returncode != 0
    assert "training.device" in result.stderr
    assert not project.exists()


def test_cli_convert_subcommand(tmp_path):
    legacy = Path(__file__).resolve().parents[2] / "examples" / "paired-linear.toml"
    out_model = tmp_path / "converted_model.toml"
    result = run_cli(tmp_path, "convert", legacy, "--output", out_model)
    assert result.returncode == 0, result.stderr
    assert out_model.is_file()
    validated = run_cli(tmp_path, "validate", out_model)
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["format"] == "hypergan-model/1"


def test_cli_set_malformed_fails_cleanly(tmp_path):
    config = tmp_path / "model.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    result = run_cli(tmp_path, "train", config, "--run-dir", tmp_path / "run", "--set", "missing_equal", "--no-server")
    assert result.returncode != 0
    assert "Invalid --set override 'missing_equal'; expected KEY=VALUE" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_set_invalid_keypath_fails_cleanly(tmp_path):
    config = tmp_path / "model.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    result = run_cli(tmp_path, "train", config, "--run-dir", tmp_path / "run", "--set", "losses.nonexistent.penalty=0.5", "--no-server")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr




def test_cli_resume_with_malformed_set_fails_cleanly(tmp_path):
    config = tmp_path / "model.toml"
    run_dir = tmp_path / "run"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    run_cli(tmp_path, "train", config, "--run-dir", run_dir, "--steps", "1", "--set", "train.lr_floor=1.0", "--no-server", isolated=False)

    result = run_cli(tmp_path, "resume", run_dir, "--set", "bad_set", "--no-server", isolated=False)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Invalid --set override 'bad_set'; expected KEY=VALUE" in result.stderr
    assert "Traceback" not in result.stderr




def test_cli_eval_help_and_export(tmp_path):
    # Verify eval alias and flags in help
    res_eval = run_cli(tmp_path, "eval", "--help", isolated=False)
    assert res_eval.returncode == 0, res_eval.stderr
    assert "--checkpoint" in res_eval.stdout
    assert "--devices" in res_eval.stdout

    res_evaluate = run_cli(tmp_path, "evaluate", "--help", isolated=False)
    assert res_evaluate.returncode == 0, res_evaluate.stderr
    assert "--checkpoint" in res_evaluate.stdout
    assert "--devices" in res_evaluate.stdout

    # Verify export help
    res_exp_help = run_cli(tmp_path, "export", "--help", isolated=False)
    assert res_exp_help.returncode == 0, res_exp_help.stderr
    assert "--checkpoint" in res_exp_help.stdout
    assert "--output" in res_exp_help.stdout


def test_cli_view_subcommand(tmp_path):
    # Verify help output for hypergan view
    res_help = run_cli(tmp_path, "view", "--help", isolated=False)
    assert res_help.returncode == 0, res_help.stderr
    assert "--status" in res_help.stdout
    assert "--stop" in res_help.stdout
    assert "--port" in res_help.stdout

    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # hypergan view [ROOT] --status when not running
    res_status = run_cli(tmp_path, "view", runs_root, "--status", isolated=False)
    assert res_status.returncode == 0, res_status.stderr
    status_data = json.loads(res_status.stdout)
    assert status_data["status"] == "not_running"

    # hypergan view [ROOT] --stop when not running
    res_stop = run_cli(tmp_path, "view", runs_root, "--stop", isolated=False)
    assert res_stop.returncode == 0, res_stop.stderr
    stop_data = json.loads(res_stop.stdout)
    assert stop_data["status"] == "not_running"


def test_cli_stop_server_all(tmp_path, monkeypatch, capsys):
    # Verify help for hypergan stop-server
    res_help = run_cli(tmp_path, "stop-server", "--help", isolated=False)
    assert res_help.returncode == 0, res_help.stderr
    assert "--all" in res_help.stdout

    # hypergan stop-server --all
    # Never inspect or terminate unrelated host viewers in a CLI unit test.
    import hypergan.api as hg
    from hypergan.cli import main
    monkeypatch.setattr(hg, "stop_all_servers", lambda: [])
    assert main(["stop-server", "--all"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "stopped"
    assert isinstance(data["killed_pids"], list)
    assert "count" in data

    # hypergan stop-server without run_dir or --all fails cleanly
    res_err = run_cli(tmp_path, "stop-server", isolated=False)
    assert res_err.returncode != 0
    assert "error: run_dir is required unless --all is specified" in res_err.stderr
    assert "Traceback" not in res_err.stderr




def test_cli_set_bracket_normalization(tmp_path):
    from hypergan.cli import _normalize_set_key, _parse_set_value, _parse_set_overrides

    assert _normalize_set_key("layers[0].width") == "layers.0.width"
    assert _normalize_set_key("networks[generator][hidden]") == "networks.generator.hidden"
    assert _parse_set_value("true") is True
    assert _parse_set_value("false") is False
    assert _parse_set_value("null") is None
    assert _parse_set_value("123") == 123
    assert _parse_set_value("1.5e-3") == 0.0015
    assert _parse_set_value("[1, 2, 3]") == [1, 2, 3]

    overrides = _parse_set_overrides(["layers[0].width=128", "enabled=true"])
    assert overrides == {"layers.0.width": 128, "enabled": True}


def test_cli_reexec_loop_detection_raises_runtime_error(tmp_path):
    import os
    ws_file = tmp_path / "hypergan.toml"
    ws_file.write_text("""
[workspace]
root = "."

[envs.fake]
python = "/nonexistent/bin/python"
""")
    config = tmp_path / "reexec_model.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    content = config.read_text(encoding="utf-8")
    config.write_text(content + '\n[run]\nenv = "fake"\n')

    cmd = [sys.executable, "-m", "hypergan", "train", str(config), "--run-dir", str(tmp_path / "run")]
    env = dict(os.environ, _HYPERGAN_REEXEC="1")
    p = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, env=env)
    assert p.returncode != 0
    assert "Re-exec loop detected" in p.stderr


def test_check_reexec_schema_interpreter_and_worktree(tmp_path, monkeypatch):
    from hypergan.cli import _check_reexec
    import hypergan.api as hg

    wt_dir = tmp_path / "my_worktree"
    (wt_dir / "src").mkdir(parents=True)
    venv_dir = tmp_path / "my_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.14.7\n")
    fake_py = venv_dir / "bin" / "python"
    fake_py.write_text("#!/bin/sh\n")

    ws_data = {
        "workspace": {"root": str(tmp_path)},
        "envs": {
            "default": {"interpreter": "python"},
            "env_schema_new": {
                "interpreter": str(fake_py),
                "worktree": str(wt_dir),
            },
            "env_schema_legacy": {
                "python": str(fake_py),
                "src": str(wt_dir / "src"),
            },
        },
    }

    # Test new schema (interpreter + worktree)
    needs_reexec, target_py, target_srcs, target_envs = _check_reexec({"env": "env_schema_new"}, ws_data)
    assert needs_reexec is True
    assert target_py == str(fake_py.absolute())
    assert target_srcs == [str((wt_dir / "src").resolve())]

    # Test legacy schema (python + src)
    needs_reexec_leg, target_py_leg, target_srcs_leg, _ = _check_reexec({"env": "env_schema_legacy"}, ws_data)
    assert needs_reexec_leg is True
    assert target_py_leg == str(fake_py.absolute())
    assert target_srcs_leg == [str((wt_dir / "src").resolve())]

    # PATH is controlled: "python" identifies this environment in this case.
    monkeypatch.setattr(shutil, "which", lambda name: sys.executable)
    # Test default python (interpreter = "python")
    needs_reexec_def, target_py_def, target_srcs_def, _ = _check_reexec({"env": "default"}, ws_data)
    assert needs_reexec_def is False

    # Test against real training-runs/hypergan.toml if available
    real_ws_file = Path("/home/martyn/dev/hypergan/training-runs/hypergan.toml")
    if real_ws_file.is_file():
        real_ws = hg.load_workspace(real_ws_file)
        # pg08 specifies interpreter
        needs_pg08, py_pg08, srcs_pg08, _ = _check_reexec({"env": "pg08"}, real_ws)
        assert needs_pg08 is True
        assert "particlegan08-env/bin/python" in py_pg08

        # pg07 specifies interpreter + worktree
        needs_pg07, py_pg07, srcs_pg07, _ = _check_reexec({"env": "pg07"}, real_ws)
        assert needs_pg07 is True
        assert "particlegan07-env/bin/python" in py_pg07
        assert any("develop-pg07-lazy16" in s for s in srcs_pg07)


def test_reconstruct_argv_for_reexec_strips_set_flags():
    from hypergan.cli import _reconstruct_argv_for_reexec

    # Test standard --set and -s with separate KEY=VALUE argument
    argv = [
        "hypergan", "train", "model.toml",
        "--set", "losses[0].weight=0.5",
        "-s", "train.lr_floor=1.0",
        "--steps", "50",
    ]
    new_argv = _reconstruct_argv_for_reexec(argv, "model.toml", "run_dir/recipe.toml", "run_dir")
    assert "--set" not in new_argv
    assert "-s" not in new_argv
    assert "losses[0].weight=0.5" not in new_argv
    assert "train.lr_floor=1.0" not in new_argv
    assert new_argv == ["train", "run_dir/recipe.toml", "--steps", "50", "--run-dir", "run_dir"]

    # Test inline --set=... and -s=...
    argv_inline = [
        "python", "-m", "hypergan", "train", "model.toml",
        "--set=losses[0].weight=0.5",
        "-s=networks.generator.hidden=128",
        "--run-dir", "/explicit/run",
    ]
    new_argv_inline = _reconstruct_argv_for_reexec(argv_inline, "model.toml", "run_dir/recipe.toml", "/explicit/run")
    assert not any(a.startswith("--set") for a in new_argv_inline)
    assert not any(a.startswith("-s") for a in new_argv_inline)
    assert new_argv_inline == ["train", "run_dir/recipe.toml", "--run-dir", "/explicit/run"]





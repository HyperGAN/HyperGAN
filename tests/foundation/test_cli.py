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



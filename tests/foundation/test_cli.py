"""Exercise installed entrypoints from outside the source checkout."""

import json
import shutil
import subprocess
import sys
import sysconfig


def run_cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-I", "-m", "hypergan", *map(str, args)],
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
    executable = shutil.which("hypergan", path=sysconfig.get_path("scripts"))
    assert executable, "installed console entrypoint was not generated"
    result = subprocess.run([executable, "version"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hypergan 2.0.0a1"


def test_create_validate_and_no_overwrite(tmp_path):
    project = tmp_path / "my project"
    created = run_cli(tmp_path, "new", project)
    assert created.returncode == 0, created.stderr
    config_path = project / "config.toml"
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


def test_missing_runtime_has_install_guidance(tmp_path):
    project = tmp_path / "project"
    created = run_cli(tmp_path, "new", project)
    assert created.returncode == 0, created.stderr
    # Block optional imports explicitly so this also runs in development envs
    # which already have the training extra installed.
    code = """
import importlib.abc
import sys
class MissingRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'particlegan'}:
            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)
sys.meta_path.insert(0, MissingRuntime())
from hypergan.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, "train", str(project), "--run-dir", str(tmp_path / "run"), "--steps", "6"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "hypergan[train]" in result.stderr
    assert "unqualified" in result.stderr
    assert "Traceback" not in result.stderr

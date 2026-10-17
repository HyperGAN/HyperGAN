"""UX Playbook 1: Greenfield ML Engineer (0-to-1 Project & Experiment Lifecycle).

Covers:
- `python -m hypergan --help` and `python -m hypergan.cli --help` equivalence
- `hypergan new --list`
- `hypergan new --project <dir> --template gaussian-grid --device cpu` + `validate <config>` + `validate <dir>`
- `hypergan new --project <dir2> --template dcgan --dataset <synth_32x32_dir> --device cpu` (8x 32x32 PNGs)
  + auto-detected `height = 32, width = 32` + `validate --data` and `validate --runtime`
- Training a short CPU run and exercising `status`, `logs`, `inspect`, `model` (zero-arg and from another cwd),
  `tag`, `prune --dry-run`, and `rm`.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

import hypergan.api as hg
import hypergan.cli as cli


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch):
    state_dir = tmp_path / "hypergan_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)
    return state_dir


def _invoke_cli(capsys, *args: str) -> tuple[int, str, str]:
    capsys.readouterr()
    try:
        rc = cli.main([str(a) for a in args])
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return rc or 0, captured.out, captured.err


def test_help_entrypoint_equivalence(isolated_env, tmp_path):
    """`python3 -m hypergan --help` and `python3 -m hypergan.cli --help` must exit 0 with identical output."""
    env = os.environ.copy()
    r_pkg = subprocess.run(
        [sys.executable, "-m", "hypergan", "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    r_cli = subprocess.run(
        [sys.executable, "-m", "hypergan.cli", "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert r_pkg.returncode == 0, r_pkg.stderr
    assert r_cli.returncode == 0, r_cli.stderr
    assert len(r_cli.stdout.strip()) > 0
    assert r_pkg.stdout == r_cli.stdout
    for expected_cmd in ("new", "validate", "train", "resume", "ps", "status", "logs", "checkpoint", "inspect", "view"):
        assert expected_cmd in r_pkg.stdout


def test_new_list_templates(isolated_env, capsys):
    """`hypergan new --list` displays all starter templates."""
    rc, out, err = _invoke_cli(capsys, "new", "--list")
    assert rc == 0, err
    for tmpl in ("dcgan", "tiny-transformer", "resnet", "gaussian-grid"):
        assert tmpl in out


def test_scaffold_and_validate_gaussian_grid_project(isolated_env, tmp_path, capsys):
    """Scaffolding a gaussian-grid project and validating both the config file and project directory."""
    proj_dir = tmp_path / "proj_gauss"
    # When invoked without --dataset or --template, defaults to gaussian-grid (R2-5)
    rc, out, err = _invoke_cli(
        capsys, "new", "--project", str(proj_dir), "--device", "cpu"
    )
    assert rc == 0, err
    cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
    assert cfg_file.is_file()
    assert (proj_dir / "hypergan.toml").is_file()
    assert (proj_dir / "runs").is_dir()

    # Validate explicit config path without --json prints concise human summary (R2-9)
    rc_cfg, out_cfg, err_cfg = _invoke_cli(capsys, "validate", str(cfg_file))
    assert rc_cfg == 0, err_cfg
    assert "Valid:" in out_cfg
    assert "Recipe:" in out_cfg
    assert "Device:" in out_cfg

    # Validate explicit config path with --json produces parseable JSON dictionary
    rc_json, out_json, err_json = _invoke_cli(capsys, "validate", str(cfg_file), "--json")
    assert rc_json == 0, err_json
    payload = json.loads(out_json)
    assert isinstance(payload, dict)

    # Validate project directory directly
    rc_dir, out_dir, err_dir = _invoke_cli(capsys, "validate", str(proj_dir))
    assert rc_dir == 0, err_dir


def test_scaffold_and_validate_dcgan_32x32_dataset(isolated_env, tmp_path, capsys):
    """Scaffolding a dcgan project with 8 synthetic 32x32 PNGs auto-detects 32x32 and passes --data and --runtime."""
    synth_dir = tmp_path / "synth_32x32"
    synth_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        img = Image.new("RGB", (32, 32), color=(i * 20, 100, 200 - i * 15))
        img.save(synth_dir / f"sample_{i:02d}.png")

    proj_dir = tmp_path / "proj_dcgan"
    rc, out, err = _invoke_cli(
        capsys,
        "new",
        "--project",
        str(proj_dir),
        "--template",
        "dcgan",
        "--dataset",
        str(synth_dir),
        "--device",
        "cpu",
    )
    assert rc == 0, err

    cfg_file = proj_dir / "configs" / "dcgan.toml"
    assert cfg_file.is_file()

    model = hg.load(cfg_file)
    assert model.spec["data"]["builtin"] == "image_folder"
    assert model.spec["data"]["args"]["height"] == 32
    assert model.spec["data"]["args"]["width"] == 32
    assert model.spec["networks"]["generator"]["output_shape"] == ["B", 3, 32, 32]
    assert model.spec["networks"]["discriminator"]["input_shape"] == ["B", 3, 32, 32]

    # Validate --data
    rc_data, out_data, err_data = _invoke_cli(capsys, "validate", str(cfg_file), "--data")
    assert rc_data == 0, err_data
    data_payload = json.loads(out_data)
    accepted = data_payload["inventory"].get("accepted", data_payload["inventory"].get("accepted_count"))
    assert accepted == 8

    # Validate --runtime
    rc_rt, out_rt, err_rt = _invoke_cli(capsys, "validate", str(cfg_file), "--runtime")
    assert rc_rt == 0, err_rt
    rt_payload = json.loads(out_rt)
    assert isinstance(rt_payload, dict)
    assert "devices" in rt_payload or rt_payload.get("runtime_checked") is True or rt_payload.get("stage") == "runtime"


def test_greenfield_lifecycle_zero_arg_and_cross_directory(isolated_env, tmp_path, monkeypatch, capsys):
    """End-to-end greenfield lifecycle: train on CPU, zero-arg & cross-cwd status/logs/inspect/model, tag, prune, rm."""
    proj_dir = tmp_path / "proj_lifecycle"
    rc, _, err = _invoke_cli(
        capsys, "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
    )
    assert rc == 0, err
    cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

    # Train for 2 steps on CPU
    rc_train, out_train, err_train = _invoke_cli(
        capsys,
        "train",
        str(cfg_file),
        "--steps",
        "2",
        "-d",
        "cpu",
        "--checkpoint-every",
        "1",
        "--no-server",
    )
    assert rc_train == 0, err_train

    # Switch to a completely separate working directory
    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_cwd)

    # Zero-arg status from other_cwd
    rc_st, out_st, err_st = _invoke_cli(capsys, "status", "--json")
    assert rc_st == 0, err_st
    st_data = json.loads(out_st)
    assert st_data["step"] == 2
    assert st_data["status"] == "completed"
    slug = st_data["slug"]
    short_id = st_data["short_id"]
    run_dir = Path(st_data["run_dir"])

    # Create a rebuildable view file so prune --dry-run has bytes to report
    views_dir = run_dir / "views" / "test_rev"
    views_dir.mkdir(parents=True, exist_ok=True)
    (views_dir / "contributions.jsonl").write_text('{"step": 1}\n' * 20, encoding="utf-8")

    # Zero-arg logs, inspect, model from other_cwd
    rc_logs, out_logs, err_logs = _invoke_cli(capsys, "logs", "-n", "10")
    assert rc_logs == 0, err_logs
    assert "Run started" in out_logs or "step" in out_logs.lower()

    rc_insp, out_insp, err_insp = _invoke_cli(capsys, "inspect")
    assert rc_insp == 0, err_insp
    manifest_data = json.loads(out_insp)
    assert manifest_data["run_id"] == st_data["run_id"]

    rc_mod, out_mod, err_mod = _invoke_cli(capsys, "model")
    assert rc_mod == 0, err_mod
    model_data = json.loads(out_mod)
    assert "formulation" in model_data or "networks" in model_data

    # Tag the run from other_cwd using its slug
    rc_tag, out_tag, err_tag = _invoke_cli(capsys, "tag", slug, "greenfield-v1", "--json")
    assert rc_tag == 0, err_tag
    tag_data = json.loads(out_tag)
    assert tag_data["name"] == "greenfield-v1"

    # Address by new custom tag from other_cwd
    rc_st2, out_st2, err_st2 = _invoke_cli(capsys, "status", "greenfield-v1", "--json")
    assert rc_st2 == 0, err_st2
    assert json.loads(out_st2)["name"] == "greenfield-v1"

    rc_mod2, out_mod2, err_mod2 = _invoke_cli(capsys, "model", short_id)
    assert rc_mod2 == 0, err_mod2
    assert json.loads(out_mod2)

    # Prune --dry-run by tag
    rc_prune, out_prune, err_prune = _invoke_cli(capsys, "prune", "greenfield-v1", "--dry-run")
    assert rc_prune == 0, err_prune
    assert "Would clean" in out_prune
    assert (views_dir / "contributions.jsonl").is_file()

    # Remove run by tag
    rc_rm, out_rm, err_rm = _invoke_cli(capsys, "rm", "greenfield-v1", "--json")
    assert rc_rm == 0, err_rm
    rm_data = json.loads(out_rm)
    assert rm_data.get("removed") is True
    assert not run_dir.exists()


@pytest.mark.parametrize("template_name", ["gaussian-grid", "dcgan", "resnet", "tiny-transformer"])
def test_all_starter_templates_pass_runtime_and_model_validation(isolated_env, tmp_path, capsys, template_name):
    """All 4 starter templates must pass validate --runtime and model --networks out of the box (R2-5)."""
    synth_dir = tmp_path / f"synth_{template_name}"
    synth_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(synth_dir / "sample.png")

    proj_dir = tmp_path / f"proj_{template_name}"
    cmd = ["new", "--project", str(proj_dir), "--template", template_name, "--device", "cpu"]
    if template_name != "gaussian-grid":
        cmd.extend(["--dataset", str(synth_dir)])
    rc, out, err = _invoke_cli(capsys, *cmd)
    assert rc == 0, err
    cfg_file = proj_dir / "configs" / f"{template_name}.toml"
    assert cfg_file.is_file()

    # 1. validate --runtime passes with exit code 0
    rc_rt, out_rt, err_rt = _invoke_cli(capsys, "validate", str(cfg_file), "--runtime", "--json")
    assert rc_rt == 0, f"validate --runtime failed for {template_name}: {err_rt}"
    rt_data = json.loads(out_rt)
    assert isinstance(rt_data, dict)

    # 2. model --networks passes without undefined variable 'z'
    rc_mod, out_mod, err_mod = _invoke_cli(capsys, "model", str(cfg_file), "--networks")
    assert rc_mod == 0, f"model --networks failed for {template_name}: {err_mod}"
    net_data = json.loads(out_mod)
    nets = net_data.get("networks", net_data)
    net_names = [n["name"] for n in nets] if isinstance(nets, list) else list(nets)
    assert "generator" in net_names


def test_standalone_pt_bundle_sampling_cli(isolated_env, tmp_path, capsys):
    """Users can sample directly from an exported .pt bundle with --count, --seed, --output (R2-6, R2-11)."""
    proj_dir = tmp_path / "proj_bundle"
    rc, _, err = _invoke_cli(capsys, "new", "--project", str(proj_dir), "--device", "cpu")
    assert rc == 0, err
    cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

    # Train a 1-step run to produce checkpoint
    rc_train, _, err_train = _invoke_cli(capsys, "train", str(cfg_file), "--steps", "1", "-d", "cpu", "--no-server")
    assert rc_train == 0, err_train

    # Export inference bundle
    bundle_file = tmp_path / "exported_bundle.pt"
    rc_exp, out_exp, err_exp = _invoke_cli(capsys, "export", "--output", str(bundle_file), "--json")
    assert rc_exp == 0, err_exp
    assert bundle_file.is_file()

    # Sample from bundle directly
    sample_out = tmp_path / "bundle_sample.json"
    rc_samp, out_samp, err_samp = _invoke_cli(
        capsys, "sample", str(bundle_file), "--count", "4", "--seed", "42", "--output", str(sample_out), "--json"
    )
    assert rc_samp == 0, err_samp
    assert sample_out.is_file()
    # Check permissions 0644
    assert oct(sample_out.stat().st_mode & 0o777) == "0o644"
    sample_data = json.loads(sample_out.read_text(encoding="utf-8"))
    assert sample_data.get("count") == 4
    assert sample_data.get("seed") == 42
    assert len(sample_data.get("samples", [])) == 4


def test_new_dataset_inspection_and_rgba_warning(isolated_env, tmp_path, capsys):
    """hypergan new --dataset reports count/res/mode to stderr and emits RGBA warning (R2-11)."""
    rgba_dir = tmp_path / "synth_rgba"
    rgba_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        img = Image.new("RGBA", (16, 16), color=(255, 0, 0, 128))
        img.save(rgba_dir / f"rgba_{i}.png")

    proj_dir = tmp_path / "proj_rgba"
    rc, out, err = _invoke_cli(capsys, "new", "--project", str(proj_dir), "--dataset", str(rgba_dir), "--device", "cpu")
    assert rc == 0, err
    assert "RGBA" in err or "alpha" in err.lower()
    assert "4 images" in err or "4" in err
    assert "16x16" in err


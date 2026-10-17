"""Tests for hypergan new NAME --dataset DIR recipe generator (Milestone 5).

Verifies image discovery, header-only sizing, architecture templates,
workspace path interpolation, validation, and refusal to overwrite.
"""
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

import hypergan.api as hg


def run_cli(tmp_path, *args, isolated=True):
    cmd = [sys.executable]
    if isolated:
        cmd.append("-I")
    cmd.extend(["-m", "hypergan", *map(str, args)])
    return subprocess.run(
        cmd,
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )


def _create_dummy_image(path: Path, size=(64, 64), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color=128 if mode == "L" else (128, 128, 128))
    img.save(path)


def test_new_dataset_cifar_32(tmp_path):
    dataset_dir = tmp_path / "cifar_dataset"
    for i in range(5):
        _create_dummy_image(dataset_dir / f"img_{i}.png", size=(32, 32), mode="RGB")

    recipe_path = tmp_path / "cifar_model.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode == 0, res.stderr
    assert recipe_path.is_file()

    # Load and validate
    m = hg.load(recipe_path)
    assert m.spec["format"] == "hypergan-model/1"
    assert m.spec["name"] == "cifar_model"
    assert m.spec["data"]["builtin"] == "image_folder"
    assert m.spec["data"]["args"]["height"] == 32
    assert m.spec["data"]["args"]["width"] == 32
    assert m.spec["data"]["args"]["mode"] == "RGB"
    assert m.spec["data"]["args"]["root"] == str(dataset_dir)

    # Networks should be sized for 32x32
    assert m.spec["networks"]["generator"]["output_shape"] == ["B", 3, 32, 32]
    assert m.spec["networks"]["discriminator"]["input_shape"] == ["B", 3, 32, 32]

    # Model must be valid
    warnings = hg.validate(m)
    assert isinstance(warnings, list)
    resolved = hg.resolve(m)
    assert resolved["name"] == "cifar_model"


def test_new_dataset_dcgan_64_grayscale(tmp_path):
    dataset_dir = tmp_path / "faces_gray"
    for i in range(10):
        _create_dummy_image(dataset_dir / f"face_{i}.jpg", size=(64, 64), mode="L")

    recipe_path = tmp_path / "faces_gray.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode == 0, res.stderr

    m = hg.load(recipe_path)
    assert m.spec["data"]["args"]["height"] == 64
    assert m.spec["data"]["args"]["width"] == 64
    assert m.spec["data"]["args"]["mode"] == "L"
    assert m.spec["networks"]["generator"]["output_shape"] == ["B", 1, 64, 64]
    assert m.spec["networks"]["discriminator"]["input_shape"] == ["B", 1, 64, 64]


def test_new_dataset_dcgan_128(tmp_path):
    dataset_dir = tmp_path / "anime_128"
    for i in range(5):
        _create_dummy_image(dataset_dir / f"anime_{i}.png", size=(128, 128), mode="RGB")

    recipe_path = tmp_path / "anime_128.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode == 0, res.stderr

    m = hg.load(recipe_path)
    assert m.spec["data"]["args"]["height"] == 128
    assert m.spec["data"]["args"]["width"] == 128
    assert m.spec["networks"]["generator"]["output_shape"] == ["B", 3, 128, 128]


def test_new_dataset_resnet_256(tmp_path):
    dataset_dir = tmp_path / "art_256"
    for i in range(5):
        _create_dummy_image(dataset_dir / f"art_{i}.webp", size=(256, 256), mode="RGB")

    recipe_path = tmp_path / "art_256.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode == 0, res.stderr

    m = hg.load(recipe_path)
    assert m.spec["data"]["args"]["height"] == 256
    assert m.spec["data"]["args"]["width"] == 256
    assert m.spec["networks"]["generator"]["output_shape"] == ["B", 3, 256, 256]


def test_new_dataset_workspace_path_interpolation(tmp_path):
    ws_file = tmp_path / "hypergan.toml"
    data_dir = tmp_path / "datasets"
    images_dir = data_dir / "cats"
    images_dir.mkdir(parents=True)
    _create_dummy_image(images_dir / "cat1.png", size=(64, 64))

    ws_file.write_text(f"""
[workspace]
root = "{tmp_path}"

[paths]
data = "{data_dir}"
""", encoding="utf-8")

    recipe_path = tmp_path / "cats_model.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", images_dir, isolated=False)
    assert res.returncode == 0, res.stderr

    # File on disk preserves ${paths.data}/cats template variable
    assert 'root = "${paths.data}/cats"' in recipe_path.read_text(encoding="utf-8")
    # Loaded model resolves ${paths.data} against workspace
    m = hg.load(recipe_path)
    assert m.spec["data"]["args"]["root"] == str(images_dir.resolve())


def test_new_dataset_refuses_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    recipe_path = tmp_path / "empty_model.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", empty_dir, isolated=False)
    assert res.returncode != 0
    assert "No supported image files found" in res.stderr


def test_new_dataset_refuses_overwrite(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _create_dummy_image(dataset_dir / "img.png", size=(64, 64))

    recipe_path = tmp_path / "existing.toml"
    recipe_path.write_text("existing content", encoding="utf-8")

    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode != 0
    assert "already exists" in res.stderr
    assert recipe_path.read_text(encoding="utf-8") == "existing content"


def test_new_without_dataset_generates_reference_gaussian_grid(tmp_path):
    ref_path = tmp_path / "ref_model.toml"
    res = run_cli(tmp_path, "new", ref_path, isolated=False)
    assert res.returncode == 0, res.stderr
    assert ref_path.is_file()

    m = hg.load(ref_path)
    assert "gaussian" in m.spec["name"]
    assert "data" in m.spec


def test_new_dataset_nonsquare_resolution_snapping(tmp_path):
    dataset_dir = tmp_path / "nonsquare_images"
    for i in range(5):
        _create_dummy_image(dataset_dir / f"img_{i}.png", size=(64, 48), mode="RGB")

    recipe_path = tmp_path / "nonsquare_model.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode == 0, res.stderr
    assert recipe_path.is_file()

    m = hg.load(recipe_path)
    # max(64, 48) = 64 -> snapped to preset 64
    assert m.spec["data"]["args"]["height"] == 64
    assert m.spec["data"]["args"]["width"] == 64
    assert m.spec["data"]["args"]["resize"] == "center_crop"
    assert m.spec["networks"]["generator"]["output_shape"] == ["B", 3, 64, 64]
    assert m.spec["networks"]["discriminator"]["input_shape"] == ["B", 3, 64, 64]

    warnings = hg.validate(m)
    assert isinstance(warnings, list)

    # Train for 1 step to ensure HNDL lowers cleanly and no E_CONSTRAINT error occurs
    run_dir = tmp_path / "run_nonsquare"
    train_res = run_cli(tmp_path, "train", recipe_path, "--run-dir", run_dir, "--steps", "1",
                        "--set", "train.device=cpu", "--no-server", isolated=False)
    assert train_res.returncode == 0, train_res.stderr


def test_new_dataset_nonpreset_resolution_snapping(tmp_path):
    dataset_dir = tmp_path / "nonpreset_images"
    for i in range(5):
        _create_dummy_image(dataset_dir / f"img_{i}.png", size=(48, 48), mode="RGB")

    recipe_path = tmp_path / "nonpreset_model.toml"
    res = run_cli(tmp_path, "new", recipe_path, "--dataset", dataset_dir, isolated=False)
    assert res.returncode == 0, res.stderr
    assert recipe_path.is_file()

    m = hg.load(recipe_path)
    # max(48, 48) = 48 -> snapped to preset 64
    assert m.spec["data"]["args"]["height"] == 64
    assert m.spec["data"]["args"]["width"] == 64
    assert m.spec["data"]["args"]["resize"] == "center_crop"
    assert m.spec["networks"]["generator"]["output_shape"] == ["B", 3, 64, 64]
    assert m.spec["networks"]["discriminator"]["input_shape"] == ["B", 3, 64, 64]

    warnings = hg.validate(m)
    assert isinstance(warnings, list)

    # Train for 1 step to ensure HNDL lowers cleanly and no E_CONSTRAINT error occurs
    run_dir = tmp_path / "run_nonpreset"
    train_res = run_cli(tmp_path, "train", recipe_path, "--run-dir", run_dir, "--steps", "1",
                        "--set", "train.device=cpu", "--no-server", isolated=False)
    assert train_res.returncode == 0, train_res.stderr


def test_new_list_templates(tmp_path):
    res = run_cli(tmp_path, "new", "--list", isolated=False)
    assert res.returncode == 0, res.stderr
    assert "dcgan" in res.stdout
    assert "tiny-transformer" in res.stdout
    assert "resnet" in res.stdout
    assert "gaussian-grid" in res.stdout


def test_new_template_without_dataset_and_self_documenting_comments(tmp_path):
    import tomllib

    out_path = tmp_path / "my_transformer.toml"
    res = run_cli(tmp_path, "new", "-t", "tiny-transformer", "-r", "64", "-o", out_path, "--device", "cpu", isolated=False)
    assert res.returncode == 0, res.stderr
    assert out_path.is_file()

    raw_text = out_path.read_text(encoding="utf-8")
    assert "# HyperGAN Experiment Configuration" in raw_text
    assert "# Start training: hypergan train my_transformer.toml" in raw_text
    assert "# Validate config: hypergan validate my_transformer.toml" in raw_text
    assert "[dataset]" in raw_text
    assert "[architecture]" in raw_text
    assert "[losses]" in raw_text
    assert "[trainer]" in raw_text
    assert "[SamplePreview]" in raw_text

    parsed = tomllib.loads(raw_text)
    assert parsed["format"] == "hypergan-model/1"
    assert parsed["data"]["args"]["height"] == 64
    assert parsed["data"]["args"]["width"] == 64
    assert "attention" in parsed["networks"]["generator"]["source"]

    m = hg.load(out_path)
    warnings = hg.validate_config(m)
    assert isinstance(warnings, list)


def test_new_project_scaffolding(tmp_path):
    proj_dir = tmp_path / "my_gan_project"
    res = run_cli(tmp_path, "new", "--project", proj_dir, "-t", "resnet", "-r", "64", "--device", "cpu", isolated=False)
    assert res.returncode == 0, res.stderr

    assert (proj_dir / "runs").is_dir()
    assert (proj_dir / "configs").is_dir()
    ws_file = proj_dir / "hypergan.toml"
    cfg_file = proj_dir / "configs" / "resnet.toml"
    assert ws_file.is_file()
    assert cfg_file.is_file()

    ws_text = ws_file.read_text(encoding="utf-8")
    assert 'runs_root = "runs"' in ws_text
    assert 'run_dir = "${runs_root}/${stem}"' in ws_text

    # Ensure no training run was started automatically
    assert list((proj_dir / "runs").iterdir()) == []

    # Validate generated config
    val_res = run_cli(tmp_path, "validate", cfg_file, isolated=False)
    assert val_res.returncode == 0, val_res.stderr


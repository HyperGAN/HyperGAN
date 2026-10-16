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

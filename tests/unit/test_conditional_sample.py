"""Tests for single-image and single-row conditional sampling in artifacts, API, and CLI."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch

import hypergan.api as hg
from hypergan.artifacts import _coerce_conditioning_input, decode_png_sample_tensor, save_bundle
from hypergan.config import DEFAULT, resolve_config
from hypergan.training import ReferenceTrainer


def _make_conditional_image_run(run_dir: Path, size: int = 8) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = copy.deepcopy(DEFAULT)
    pixels = 3 * size * size
    raw["prior"]["args"].update(num_particles=16, z_dim=4)
    raw["components"]["generator"] = {
        "factory": "hndl",
        "args": {
            "input_shape": {"x": ["B", 4], "gray": ["B", 1, size, size]},
            "output_shape": ["B", 3, size, size],
            "source": f"g = flatten(gray)\nconcat(x, g)\nlinear({pixels})\ntanh()\nreshape(3, {size}, {size})\n",
        },
        "inputs": {"x": "latent", "gray": "batch.gray"},
    }
    raw["components"]["discriminator"] = {
        "factory": "hndl",
        "args": {
            "input_shape": ["B", 3, size, size],
            "output_shape": ["B", 1],
            "source": "flatten()\nlinear(1)\n",
        },
        "inputs": {"x": "candidate"},
    }
    raw["sampling"].update(count=4, seed=42)
    raw["training"].update(steps=1, batch_size=2, device="cpu")
    raw["metrics"] = {"preset": "none"}
    config = resolve_config(raw)
    trainer = ReferenceTrainer(config)
    batch = {
        "real": torch.zeros(2, 3, size, size),
        "gray": torch.zeros(2, 1, size, size),
    }
    trainer.update(batch)
    save_bundle(run_dir, trainer, batch)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "step": trainer.step,
        "steps": 1,
        "bundle_path": "model.pt",
        "config": config,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_coerce_conditioning_input_image_and_unbatched_tensor(tmp_path):
    # Create an RGBA test image with transparency to verify white alpha compositing and BT.601 luminance
    img = Image.new("RGBA", (16, 8), (255, 0, 0, 128))
    png_path = tmp_path / "cond.png"
    img.save(png_path)

    expected_gray = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
    coerced_from_path = _coerce_conditioning_input(png_path, expected_gray)
    coerced_from_str = _coerce_conditioning_input(str(png_path), expected_gray)
    coerced_from_pil = _coerce_conditioning_input(img, expected_gray)

    assert coerced_from_path.shape == (1, 1, 8, 8)
    assert coerced_from_path.dtype == torch.float32
    assert torch.allclose(coerced_from_path, coerced_from_str)
    assert torch.allclose(coerced_from_path, coerced_from_pil)
    assert coerced_from_path.min() >= -1.0 and coerced_from_path.max() <= 1.0

    # Top and bottom padding rows (since 16x8 padded to 8x8 pads top/bottom with white=1.0)
    assert torch.allclose(coerced_from_path[0, 0, 0, :], torch.ones(8), atol=1e-4)
    assert torch.allclose(coerced_from_path[0, 0, -1, :], torch.ones(8), atol=1e-4)

    # 3-channel expected input preserves RGB
    expected_rgb = torch.zeros(2, 3, 8, 8, dtype=torch.float32)
    coerced_rgb = _coerce_conditioning_input(png_path, expected_rgb)
    assert coerced_rgb.shape == (1, 3, 8, 8)

    # Unbatched tensor (C, H, W) unsqueezes to (1, C, H, W)
    unbatched = torch.full((1, 8, 8), 0.25)
    coerced_unbatched = _coerce_conditioning_input(unbatched, expected_gray)
    assert coerced_unbatched.shape == (1, 1, 8, 8)
    assert torch.equal(coerced_unbatched[0], unbatched)


def test_sample_single_row_tensor_broadcasts_to_count(tmp_path):
    run_dir = _make_conditional_image_run(tmp_path / "cond_run", size=8)

    single_row = torch.full((1, 1, 8, 8), 0.5)
    s_direct = hg.sample(run_dir, count=4, seed=7, inputs=single_row, output=tmp_path / "direct.json")
    s_dict = hg.sample(run_dir, count=4, seed=7, inputs={"gray": single_row}, output=tmp_path / "dict.json")
    s_unbatched = hg.sample(run_dir, count=4, seed=7, inputs=single_row[0], output=tmp_path / "unbatched.json")

    assert s_direct.data.shape == (4, 3, 8, 8)
    assert torch.allclose(s_direct.data, s_dict.data)
    assert torch.allclose(s_direct.data, s_unbatched.data)
    assert s_direct.inputs["gray"].shape == (4, 1, 8, 8)
    for idx in range(4):
        assert torch.allclose(s_direct.inputs["gray"][idx], single_row[0])

    # Different conditioning tensor produces different output
    other_row = torch.full((1, 1, 8, 8), -0.5)
    s_other = hg.sample(run_dir, count=4, seed=7, inputs=other_row, output=tmp_path / "other.json")
    assert not torch.allclose(s_direct.data, s_other.data)

    # Mismatched batch size != 1 and != count still raises ValueError
    with pytest.raises(ValueError, match="Input gray must have 4 rows"):
        hg.sample(run_dir, count=4, seed=7, inputs={"gray": torch.zeros(3, 1, 8, 8)})


def test_sample_image_file_path_broadcasts_to_png_grid(tmp_path):
    run_dir = _make_conditional_image_run(tmp_path / "cond_run", size=8)

    img = Image.new("RGB", (16, 16), (40, 120, 200))
    png_path = tmp_path / "sketch.png"
    img.save(png_path)

    grid_a = tmp_path / "grid_a.png"
    grid_b = tmp_path / "grid_b.png"
    s_a = hg.sample(run_dir, count=4, seed=99, inputs=png_path, output=grid_a)
    s_b = hg.sample(run_dir, count=4, seed=99, inputs={"gray": png_path}, output=grid_b)

    assert grid_a.is_file() and grid_b.is_file()
    assert s_a.path == grid_a and s_b.path == grid_b
    decoded_a = decode_png_sample_tensor(grid_a)
    decoded_b = decode_png_sample_tensor(grid_b)
    assert decoded_a.shape == (4, 3, 8, 8)
    assert torch.allclose(decoded_a, decoded_b)


def test_cli_sample_with_input_flag(tmp_path):
    import os

    run_dir = _make_conditional_image_run(tmp_path / "cond_run", size=8)
    img = Image.new("RGB", (12, 12), (200, 50, 50))
    png_path = tmp_path / "input.png"
    img.save(png_path)

    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": src_dir}

    out_grid = tmp_path / "cli_grid.png"
    cmd = [
        sys.executable,
        "-m",
        "hypergan",
        "sample",
        str(run_dir),
        "--input",
        str(png_path),
        "--count",
        "4",
        "--seed",
        "123",
        "--output",
        str(out_grid),
    ]
    res = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env)
    assert res.returncode == 0, res.stderr
    assert out_grid.is_file()
    decoded = decode_png_sample_tensor(out_grid)
    assert decoded.shape == (4, 3, 8, 8)

    # Also test KEY=PATH form and standalone .pt bundle
    out_grid_kv = tmp_path / "cli_grid_kv.png"
    cmd_kv = [
        sys.executable,
        "-m",
        "hypergan",
        "sample",
        str(run_dir / "model.pt"),
        "-i",
        f"gray={png_path}",
        "--count",
        "4",
        "--seed",
        "123",
        "--output",
        str(out_grid_kv),
    ]
    res_kv = subprocess.run(cmd_kv, cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env)
    assert res_kv.returncode == 0, res_kv.stderr
    assert out_grid_kv.is_file()
    assert torch.allclose(decoded, decode_png_sample_tensor(out_grid_kv))

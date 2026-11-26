"""Structural, fairness, and dataset-preparer tests for whitepaper benchmark configs."""
from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from hypergan.config import load_config
from hypergan.model_file import lower
from hypergan.particlegan_integration import particlegan_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks" / "whitepaper"
PREP_SCRIPT = REPO_ROOT / "scripts" / "prepare_benchmark_dataset.py"

EXPECTED_PAIRS = [
    ("cifar10-dcgan-bn-v12.toml", "cifar10-dcgan-bn-adam.toml"),
    ("cifar10-resnet-scratch-v12.toml", "cifar10-resnet-scratch-adam.toml"),
    ("cifar10-transgan-scratch-v12.toml", "cifar10-transgan-scratch-adam.toml"),
    ("ffhq64-resnet-v12.toml", "ffhq64-resnet-adam.toml"),
]
ALL_EXPECTED_FILES = sorted(name for pair in EXPECTED_PAIRS for name in pair)

FORBIDDEN_V12_TRAIN_OVERRIDES = {
    "lr",
    "d_lr_mult",
    "prior_lr_mult",
    "betas",
    "prior_betas",
    "implementation",
    "ema",
    "lr_anneal_start",
    "lr_floor",
    "network_lr_floor",
    "network_lr_horizon_cap",
    "d_guard_ratio",
    "d_guard_min_steps",
    "latent_damping_max_rate",
    "input_noise_std",
    "input_noise_anneal_end",
    "output_noise_std",
    "output_noise_warmup",
    "optimizer",
}


def _load_raw(filename: str) -> dict:
    with (BENCH_DIR / filename).open("rb") as stream:
        return tomllib.load(stream)


def _load_prep_module():
    spec = importlib.util.spec_from_file_location("prepare_benchmark_dataset", PREP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["prepare_benchmark_dataset"] = module
    spec.loader.exec_module(module)
    return module


def test_expected_whitepaper_config_inventory():
    actual = sorted(p.name for p in BENCH_DIR.glob("*.toml"))
    assert actual == ALL_EXPECTED_FILES


@pytest.mark.parametrize("filename", ALL_EXPECTED_FILES)
def test_whitepaper_config_validates_and_uses_pure_hndl(filename: str):
    path = BENCH_DIR / filename
    resolved = load_config(path)

    # Pure declarative HNDL for both generator and discriminator (no custom PyTorch shortcuts).
    assert set(resolved["components"].keys()) == {"generator", "discriminator"}
    for role_name in ("generator", "discriminator"):
        comp = resolved["components"][role_name]
        assert comp["factory"] == "hndl", f"{filename} {role_name} must use pure HNDL"
        assert comp["args"]["source"].strip()

    # Standard benchmark protocol fields.
    assert resolved["training"]["batch_size"] == 256
    assert resolved["training"]["device"] == "cuda"
    assert resolved["training"]["steps"] == 100000
    assert "fid50k_train" in resolved["metrics"]["custom"]
    fid_spec = resolved["metrics"]["custom"]["fid50k_train"]
    assert fid_spec["factory"] == "hypergan.image_metrics:InceptionFID"
    assert fid_spec["evaluation"]["sample_count"] == 50000
    assert fid_spec["trigger"] == "interval"

    assert "inception_suite" in resolved["metrics"]["custom"]
    inc_suite = resolved["metrics"]["custom"]["inception_suite"]
    assert inc_suite["factory"] == "hypergan.image_metrics:InceptionSuite"
    assert inc_suite["evaluation"]["sample_count"] == 50000
    assert inc_suite["trigger"] == "manual"

    assert "dinov2_suite" in resolved["metrics"]["custom"]
    dino_suite = resolved["metrics"]["custom"]["dinov2_suite"]
    assert dino_suite["factory"] == "hypergan.image_metrics:DINOv2Suite"
    assert dino_suite["evaluation"]["sample_count"] == 50000
    assert dino_suite["trigger"] == "manual"

    # Verify all custom snapshot metric factory paths import cleanly without downloading weights.
    import importlib
    for eval_id in ("fid50k_train", "inception_suite", "dinov2_suite"):
        mod_name, attr_name = resolved["metrics"]["custom"][eval_id]["factory"].split(":", 1)
        factory_cls = getattr(importlib.import_module(mod_name), attr_name)
        assert callable(factory_cls)
        assert callable(getattr(factory_cls, "evaluate", None))

    # ImageFolder requires CPU data RNG generator.
    if resolved["data"]["factory"] == "image_folder":
        assert resolved["training"]["data_rng_device"] == "cpu"

    # HNDL graph compiles cleanly with static shape resolution (no GPU needed).
    from hypergan.hndl_networks import HNDLNetwork
    for role_name in ("generator", "discriminator"):
        net = HNDLNetwork(**resolved["components"][role_name]["args"])
        param_count = sum(p.numel() for p in net.parameters())
        assert param_count > 100_000

    # ParticleGAN recipe object constructs and passes all dependency invariants.
    import particlegan

    recipe = particlegan_recipe(resolved)
    if filename.endswith("-v12.toml"):
        expected = particlegan.get_recipe()
        assert recipe.name == expected.name
        assert recipe.optimizer_family == expected.optimizer_family
        assert recipe.effective_critic_formulation == expected.effective_critic_formulation
    else:
        assert recipe.optimizer_family == "adam"
        assert recipe.effective_critic_formulation == "k3p"
        assert recipe.loss == "non_saturating"
        assert recipe.reg_arm == "a_r1r2"
        assert recipe.reg_real_weight == 1.0
        assert recipe.reg_fake_weight == 0.0


@pytest.mark.parametrize("v12_name,adam_name", EXPECTED_PAIRS)
def test_v12_and_adam_pairs_have_identical_architectures_and_fair_contracts(v12_name: str, adam_name: str):
    raw_v12 = _load_raw(v12_name)
    raw_adam = _load_raw(adam_name)

    # Identical pure HNDL networks, shapes, dataset, seed, steps, batch_size, and evaluation protocol.
    assert raw_v12["networks"] == raw_adam["networks"]
    assert raw_v12["data"] == raw_adam["data"]
    assert raw_v12["evaluations"] == raw_adam["evaluations"]
    assert raw_v12["train"]["batch_size"] == raw_adam["train"]["batch_size"] == 256
    assert raw_v12["train"]["steps"] == raw_adam["train"]["steps"] == 100000
    assert raw_v12["train"]["seed"] == raw_adam["train"]["seed"]
    assert raw_v12["train"].get("initialization") == raw_adam["train"].get("initialization")
    assert raw_v12["prior"]["z_dim"] == raw_adam["prior"]["z_dim"]

    # ParticleGAN v12: ZERO recipe or optimizer overrides.
    assert raw_v12["defaults"] == "particlegan"
    assert "recipe" not in raw_v12
    assert "penalty" not in raw_v12
    leaked = set(raw_v12["train"]) & FORBIDDEN_V12_TRAIN_OVERRIDES
    assert not leaked, f"{v12_name} must have zero optimizer/recipe overrides, found: {sorted(leaked)}"
    assert raw_v12["prior"]["kind"] == "particles"
    assert any(loss.get("type") == "prior" for loss in raw_v12["losses"])

    # Honest Adam + R1 baseline: pure PyTorch Adam, R1-only penalty, Gaussian prior, EMA=0.999, no ParticleGAN guards.
    assert raw_adam["defaults"] == "hypergan"
    assert raw_adam["prior"]["kind"] == "gaussian"
    assert not any(loss.get("type") == "prior" for loss in raw_adam["losses"])
    assert raw_adam["penalty"]["anchor_weight"] == 0.0
    assert raw_adam["train"]["implementation"] == "torch_fused_adam"
    assert raw_adam["train"]["lr"] == 0.0002
    assert raw_adam["train"]["d_lr_mult"] == 1.0
    assert raw_adam["train"]["ema"] == 0.999
    assert raw_adam["train"]["d_guard_ratio"] == 0.0
    assert raw_adam["train"]["latent_damping_max_rate"] == 0.0
    assert raw_adam["train"]["input_noise_std"] == 0.0
    assert raw_adam["train"]["output_noise_std"] == 0.0

    # Also ensure comments include the finalized multi-metric suite hooks.
    for name in (v12_name, adam_name):
        text = (BENCH_DIR / name).read_text(encoding="utf-8")
        assert "hypergan.image_metrics:InceptionSuite" in text
        assert "hypergan.image_metrics:DINOv2Suite" in text

    # Ensure lowering succeeds cleanly for both.
    assert lower(raw_v12, base=BENCH_DIR)["components"]["generator"]["factory"] == "hndl"
    assert lower(raw_adam, base=BENCH_DIR)["components"]["generator"]["factory"] == "hndl"


def test_prepare_benchmark_dataset_zip_and_tgz_offline(tmp_path: Path):
    prep = _load_prep_module()

    # 1. Synthetic ZIP archive (FFHQ-64 style: 64x64 RGB PNGs + non-image dataset.json + unsafe traversal entry).
    zip_path = tmp_path / "synthetic_ffhq64.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dataset.json", '{"labels": null}')
        zf.writestr("../escape.png", b"not-a-valid-path")
        for idx in range(6):
            buf = BytesIO()
            img = Image.new("RGB", (64, 64), color=(idx * 30, 100, 200))
            img.save(buf, format="PNG")
            zf.writestr(f"00000/img{idx:08d}.png", buf.getvalue())

    out_ffhq = tmp_path / "out_ffhq64"
    manifest = prep.prepare_dataset(
        dataset="ffhq64",
        output_dir=out_ffhq,
        archive_path=zip_path,
        resolution=64,
        workers=2,
        max_images=6,
    )
    assert manifest["image_count"] == 6
    assert manifest["resized_count"] == 0
    assert manifest["resolution"] == 64
    verified = prep.verify_prepared_dataset(out_ffhq, dataset="ffhq64", resolution=64, expected_count=6)
    assert verified["image_count"] == 6

    # Idempotent re-run reuses existing verified directory without failing.
    manifest_again = prep.prepare_dataset(
        dataset="ffhq64",
        output_dir=out_ffhq,
        archive_path=zip_path,
        resolution=64,
        workers=2,
        max_images=6,
    )
    assert manifest_again["image_count"] == 6

    # 2. Synthetic TGZ archive (Flowers-102 style: non-square RGB/RGBA JPEGs/PNGs -> center-crop + Lanczos 64x64 PNGs).
    tgz_path = tmp_path / "synthetic_flowers.tgz"
    with tarfile.open(tgz_path, "w:gz") as tf:
        for idx, (size, mode, fmt) in enumerate([
            ((120, 80), "RGB", "JPEG"),
            ((80, 140), "RGB", "JPEG"),
            ((96, 96), "RGBA", "PNG"),
            ((64, 64), "L", "PNG"),
        ]):
            buf = BytesIO()
            color = (50, 120, 180, 128) if mode == "RGBA" else (128 if mode == "L" else (80, 160, 40))
            Image.new(mode, size, color=color).save(buf, format=fmt)
            raw = buf.getvalue()
            ext = "jpg" if fmt == "JPEG" else "png"
            info = tarfile.TarInfo(name=f"jpg/image_{idx:05d}.{ext}")
            info.size = len(raw)
            tf.addfile(info, BytesIO(raw))

    out_flowers = tmp_path / "out_flowers102"
    manifest_f = prep.prepare_dataset(
        dataset="flowers102",
        output_dir=out_flowers,
        archive_path=tgz_path,
        resolution=64,
        workers=2,
        max_images=4,
    )
    assert manifest_f["image_count"] == 4
    assert manifest_f["resized_count"] == 4
    assert json.loads((out_flowers / "dataset-manifest.json").read_text())["resize_filter"] == "lanczos"

    # Output loads cleanly in HyperGAN's ImageFolder.
    from hypergan.data import ImageFolder
    folder = ImageFolder(out_flowers, height=64, width=64, mode="RGB", resize="none", shuffle=False, workers=0)
    try:
        assert len(folder) == 4
    finally:
        folder.close()

"""Recipe generator creating runnable hypergan-model/1 image recipes sized from datasets."""
from pathlib import Path
import statistics
from typing import Any, Dict, Optional, Tuple, Union

from .workspace import find_workspace

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def inspect_image_dataset(dataset_dir: Union[Path, str]) -> Dict[str, Any]:
    """Scan dataset directory using header-only inspection without decoding pixel buffers into RAM."""
    from PIL import Image

    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {dataset_dir}")

    image_paths = []
    for p in dataset_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."):
            image_paths.append(p)

    if not image_paths:
        raise ValueError(f"No supported image files found in directory {dataset_dir}")

    image_paths.sort()

    widths = []
    heights = []
    has_rgb = False

    sample_paths = image_paths[:100]
    for img_path in sample_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                if img.mode in ("RGB", "RGBA", "CMYK"):
                    has_rgb = True
        except Exception:
            continue

    if not widths or not heights:
        raise ValueError(f"Failed to inspect image headers in directory {dataset_dir}")

    target_w = int(statistics.median(widths))
    target_h = int(statistics.median(heights))

    mode = "RGB" if has_rgb else "L"
    channels = 3 if has_rgb else 1

    has_varying = any(w != target_w or h != target_h for w, h in zip(widths, heights))

    max_dim = max(target_w, target_h)
    if max_dim <= 32:
        preset_res = 32
    elif max_dim <= 64:
        preset_res = 64
    elif max_dim <= 128:
        preset_res = 128
    else:
        preset_res = 256

    return {
        "count": len(image_paths),
        "width": target_w,
        "height": target_h,
        "preset_res": preset_res,
        "channels": channels,
        "mode": mode,
        "resize": "center_crop",
    }


STARTER_TEMPLATES = {
    "dcgan": {
        "name": "dcgan",
        "description": "Convolutional DCGAN starter (supports 32, 64, 128, 256px images)",
        "resolutions": [32, 64, 128, 256],
    },
    "tiny-transformer": {
        "name": "tiny-transformer",
        "description": "Tiny Transformer generator + conv critic (64 spatial tokens + upsampling)",
        "resolutions": [32, 64, 128, 256],
    },
    "resnet": {
        "name": "resnet",
        "description": "Residual upsampling generator + conv critic with skip connections",
        "resolutions": [32, 64, 128, 256],
    },
    "gaussian-grid": {
        "name": "gaussian-grid",
        "description": "2D mixture-of-Gaussians toy reference model (no external dataset needed)",
        "resolutions": [],
    },
}


def list_starter_templates() -> list[dict[str, Any]]:
    """Return available starter templates for `hypergan new`."""
    return list(STARTER_TEMPLATES.values())


def format_starter_templates() -> str:
    """Format starter templates and usage examples for `hypergan new --list`."""
    lines = [
        "Available HyperGAN Starter Templates:",
        "  Template          Resolutions       Description",
        "  ----------------  ----------------  -----------------------------------------------------------------",
    ]
    for t in STARTER_TEMPLATES.values():
        res_str = ", ".join(str(r) for r in t["resolutions"]) if t["resolutions"] else "N/A (2D toy)"
        lines.append(f"  {t['name']:<16}  {res_str:<16}  {t['description']}")
    lines.extend([
        "",
        "Examples:",
        "  hypergan new my_cifar.toml --dataset ./data/cifar10",
        "  hypergan new my_transformer.toml --template tiny-transformer --resolution 64",
        "  hypergan new my_resnet.toml --template resnet --dataset ./data/faces",
        "  hypergan new toy_2d.toml",
    ])
    return "\n".join(lines)


def _upsample_conv_tail(start_res: int, target_res: int, channels: int) -> str:
    """Build nearest-neighbor upsample + conv layers from start_res to target_res."""
    cur = start_res
    ch = 256
    blocks = []
    while cur < target_res:
        cur *= 2
        next_ch = max(32, ch // 2)
        blocks.append(f'h = upsample(h, 2, mode="nearest")')
        blocks.append(f'h = conv(h, {next_ch}, kernel_size=3, padding=1, name="conv{cur}")')
        blocks.append("h = leaky_relu(h, 0.2)")
        ch = next_ch
    blocks.append(f'h = conv(h, {channels}, kernel_size=3, padding=1, name="rgb")')
    blocks.append("out = tanh(h)")
    return "\n".join(blocks)


def get_hndl_templates(width: int, height: int, channels: int, template: str = "dcgan") -> Tuple[str, str]:
    """Select optimal generator and critic HNDL architecture templates for resolution and template style."""
    max_dim = max(width, height)
    if max_dim <= 32:
        preset_res = 32
    elif max_dim <= 64:
        preset_res = 64
    elif max_dim <= 128:
        preset_res = 128
    else:
        preset_res = 256

    if max_dim <= 32:
        # CIFAR-32 template
        gen = f"""
reshape(128, 1, 1)
deconv(256, kernel_size=4, stride=1, padding=0, bias=False)
batch_norm()
relu()
deconv(128, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(64, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv({channels}, kernel_size=4, stride=2, padding=1)
tanh()
"""
        disc = f"""
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(256, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
"""
    elif max_dim <= 64:
        # DCGAN-64 template
        gen = f"""
reshape(128, 1, 1)
deconv(512, kernel_size=4, stride=1, padding=0, bias=False)
batch_norm()
relu()
deconv(256, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(128, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(64, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv({channels}, kernel_size=4, stride=2, padding=1)
tanh()
"""
        disc = f"""
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(256, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
"""
    elif max_dim <= 128:
        # DCGAN-128 template
        gen = f"""
reshape(128, 1, 1)
deconv(512, kernel_size=4, stride=1, padding=0, bias=False)
batch_norm()
relu()
deconv(512, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(256, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(128, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(64, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv({channels}, kernel_size=4, stride=2, padding=1)
tanh()
"""
        disc = f"""
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(256, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
"""
    else:
        # DCGAN-256 template
        gen = f"""
reshape(128, 1, 1)
deconv(512, kernel_size=4, stride=1, padding=0, bias=False)
batch_norm()
relu()
deconv(512, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(256, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(128, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(64, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv(32, kernel_size=4, stride=2, padding=1, bias=False)
batch_norm()
relu()
deconv({channels}, kernel_size=4, stride=2, padding=1)
tanh()
"""
        disc = f"""
conv(32, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(256, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(512, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
"""

    if template == "tiny-transformer":
        tail = _upsample_conv_tail(8, preset_res, channels)
        gen = f"""
h = linear(z, 16384, name="input_projection")
h = reshape(h, 64, 256)
branch = layer_norm(h, name="block0_attention_norm")
branch = attention(branch, 4, dropout=0.0, qkv_bias=True, out_bias=True, name="block0_attention")
h = add(h, branch)
branch = layer_norm(h, name="block0_ffn_norm")
branch = feed_forward(branch, 512, activation="gelu", dropout=0.0, name="block0_ffn")
h = add(h, branch)
branch = layer_norm(h, name="block1_attention_norm")
branch = attention(branch, 4, dropout=0.0, qkv_bias=True, out_bias=True, name="block1_attention")
h = add(h, branch)
branch = layer_norm(h, name="block1_ffn_norm")
branch = feed_forward(branch, 512, activation="gelu", dropout=0.0, name="block1_ffn")
h = add(h, branch)
h = permute(h, 2, 1)
h = reshape(h, 256, 8, 8)
{tail}
"""
    elif template == "resnet":
        tail = _upsample_conv_tail(8, preset_res, channels)
        gen = f"""
h = linear(z, 16384, name="input_projection")
h = reshape(h, 256, 8, 8)
skip = h
h = conv(h, 256, kernel_size=3, padding=1, name="res0_conv1")
h = leaky_relu(h, 0.2)
h = conv(h, 256, kernel_size=3, padding=1, name="res0_conv2")
h = add(h, skip)
h = leaky_relu(h, 0.2)
{tail}
"""

    return gen.strip(), disc.strip()


def _add_recipe_comments(body: str, config_name: str) -> str:
    """Inject concise, self-documenting comments above each TOML section."""
    import hashlib
    import re

    commented = body
    section_comments = [
        (r"(?m)^(\[data\])", "# [dataset] Dataset source, resolution, and preprocessing\n\\1"),
        (r"(?m)^(\[prior\])", "# [prior] Latent particle distribution\n\\1"),
        (r"(?m)^(\[networks\.)", "# [architecture] Generator and critic network definitions\n\\1"),
        (r"(?m)^(\[\[losses\]\])", "# [losses] Adversarial and prior regularization objectives\n\\1"),
        (r"(?m)^(\[train\])", "# [trainer] Training steps, batch size, device, and initialization\n\\1"),
        (r"(?m)^(\[checkpoints\])", "# [checkpoints] Checkpoint retention and EMA export policy\n\\1"),
        (r"(?m)^(\[sampling\])", "# [SamplePreview] Periodic sample grid preview configuration\n\\1"),
    ]
    for pattern, replacement in section_comments:
        commented = re.sub(pattern, replacement, commented, count=1)

    digest = hashlib.sha256(commented.encode("utf-8")).hexdigest()
    header = (
        "# HyperGAN Experiment Configuration\n"
        f"# Start training: hypergan train {config_name}\n"
        f"# Validate config: hypergan validate {config_name}\n"
        f"# body-sha256: {digest}\n\n"
    )
    return header + commented


def generate_recipe(name: Optional[Union[str, Path]] = None,
                    dataset_dir: Optional[Union[Path, str]] = None,
                    output_path: Optional[Union[Path, str]] = None,
                    overwrite: bool = False,
                    device: str = "cuda",
                    template: str = "dcgan",
                    resolution: Optional[int] = None) -> Path:
    """Generate and save a runnable hypergan-model/1 recipe for dataset_dir or template/resolution."""
    from .config import validate_device
    validate_device(device)

    template = (template or "dcgan").strip().lower()
    if template not in STARTER_TEMPLATES:
        choices = ", ".join(STARTER_TEMPLATES.keys())
        raise ValueError(f"Unknown starter template {template!r}; choose one of: {choices}")

    # Support calling `generate_recipe(data_dir, output_path=..., template=..., resolution=...)`
    if dataset_dir is None and name is not None:
        if isinstance(name, Path) or Path(str(name)).is_dir():
            dataset_dir = name
            name = None

    if name is None:
        if output_path is not None:
            out_cand = Path(output_path)
            name = out_cand.stem if out_cand.suffix == ".toml" else out_cand.name
        elif dataset_dir is not None:
            name = Path(dataset_dir).resolve().name
        else:
            name = template
    name = str(name)

    if dataset_dir is not None:
        dataset_path = Path(dataset_dir).resolve()
        info = inspect_image_dataset(dataset_path)
        w = info["width"]
        h = info["height"]
        c = info["channels"]
        mode = info["mode"]
        preset_res = resolution or info.get("preset_res")
        if preset_res is None:
            max_dim = max(w, h)
            if max_dim <= 32:
                preset_res = 32
            elif max_dim <= 64:
                preset_res = 64
            elif max_dim <= 128:
                preset_res = 128
            else:
                preset_res = 256
    else:
        dataset_path = None
        c = 3
        mode = "RGB"
        preset_res = int(resolution or 64)
        if preset_res not in (32, 64, 128, 256):
            raise ValueError(f"Unsupported resolution {preset_res}; choose one of 32, 64, 128, 256")

    # Target path resolution
    if output_path is None:
        target_path = (Path.cwd() / f"{name}.toml").resolve()
    else:
        target_path = Path(output_path).resolve()
        if target_path.is_dir():
            target_path = target_path / f"{name}.toml"
        elif target_path.suffix != ".toml":
            target_path = target_path.with_suffix(".toml")

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Recipe file already exists: {target_path}")

    if template == "gaussian-grid" and dataset_path is None:
        from copy import deepcopy
        from .api import packaged
        pkg_model = packaged("gaussian-grid")
        spec = deepcopy(pkg_model.spec)
        if device:
            spec.setdefault("train", {})["device"] = device
    else:
        # Check workspace data path interpolation
        ws_file = find_workspace(target_path) or find_workspace(Path.cwd())
        if dataset_path is not None:
            root_str = str(dataset_path)
            if ws_file:
                from .workspace import load_workspace
                ws_data = load_workspace(ws_file)
                paths_sec = ws_data.get("paths", {})
                if "data" in paths_sec:
                    data_root = Path(paths_sec["data"]).resolve()
                    try:
                        rel = dataset_path.relative_to(data_root)
                        root_str = f"${{paths.data}}/{rel.as_posix()}" if str(rel) != "." else "${paths.data}"
                    except ValueError:
                        root_str = str(dataset_path)
        else:
            root_str = "./data"
            if ws_file:
                from .workspace import load_workspace
                ws_data = load_workspace(ws_file)
                if "data" in ws_data.get("paths", {}):
                    root_str = "${paths.data}"

        gen_source, disc_source = get_hndl_templates(preset_res, preset_res, c, template=template)

        spec = {
            "format": "hypergan-model/1",
            "name": name,
            "defaults": "particlegan",
            "data": {
                "builtin": "image_folder",
                "args": {
                    "root": root_str,
                    "height": preset_res,
                    "width": preset_res,
                    "mode": mode,
                    "resize": "center_crop",
                },
            },
            "prior": {
                "kind": "particles",
                "initialization_seed": 2,
                "z_dim": 128,
                "args": {"num_particles": 10000},
            },
            "networks": {
                "generator": {
                    "role": "generator",
                    "inputs": {"z": "latent"},
                    "input_shape": ["B", 128],
                    "output_shape": ["B", c, preset_res, preset_res],
                    "source": gen_source,
                },
                "discriminator": {
                    "role": "critic",
                    "inputs": {"x": "candidate"},
                    "input_shape": ["B", c, preset_res, preset_res],
                    "output_shape": ["B", 1],
                    "source": disc_source,
                },
            },
            "losses": [
                {
                    "type": "adversarial",
                    "critic": "discriminator",
                    "weight": 1.0,
                    "penalty": 1.0,
                },
                {
                    "type": "prior",
                },
            ],
            "train": {
                "steps": 200000,
                "batch_size": 64,
                "seed": 0,
                "device": device,
                "initialization": "deterministic_orthogonal",
            },
            "metrics": {
                "every_steps": 10,
            },
            "checkpoints": {
                "keep_last": 3,
                "keep_best": {"metric": "fid50k_train", "mode": "min", "count": 2},
                "keep_every": 50000,
                "export_bundle": "on_eval",
            },
            "sampling": {
                "count": 16,
                "seed": 123,
            },
        }

    from .toml_writer import dumps as _dumps
    body = _dumps(spec, table_depth=3)
    text = _add_recipe_comments(body, target_path.name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(target_path)
    return target_path.resolve()


def init_project(
    project_dir: Union[str, Path],
    *,
    template: str = "dcgan",
    resolution: Optional[int] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_config: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Initialize a HyperGAN project directory with hypergan.toml, runs/, and configs/<template>.toml."""
    template = (template or "dcgan").strip().lower()
    if template not in STARTER_TEMPLATES:
        choices = ", ".join(STARTER_TEMPLATES.keys())
        raise ValueError(f"Unknown starter template {template!r}; choose one of: {choices}")

    proj_path = Path(project_dir).resolve()
    runs_dir = proj_path / "runs"
    configs_dir = proj_path / "configs"
    proj_path.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    ws_path = proj_path / "hypergan.toml"
    if not ws_path.exists() or overwrite:
        ws_text = (
            "# HyperGAN Workspace Configuration\n"
            "[workspace]\n"
            "# Directory where training runs are stored (relative to this workspace)\n"
            'runs_root = "runs"\n'
            "# Naming pattern for new run directories\n"
            'run_dir = "${runs_root}/${stem}"\n'
        )
        ws_path.write_text(ws_text, encoding="utf-8")

    if output_config is not None:
        out_p = Path(output_config)
        cfg_target = out_p if out_p.is_absolute() else (proj_path / out_p)
    else:
        cfg_target = configs_dir / f"{template}.toml"

    cfg_path = generate_recipe(
        data_dir,
        output_path=cfg_target,
        template=template,
        resolution=resolution,
        device=device,
        overwrite=overwrite,
    )

    return {
        "project_dir": str(proj_path),
        "workspace_config": str(ws_path),
        "recipe_path": str(cfg_path),
        "template": template,
        "next_steps": [
            f"hypergan validate {cfg_path}",
            f"hypergan train {cfg_path}",
        ],
    }


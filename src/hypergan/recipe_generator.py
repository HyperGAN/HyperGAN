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

    return {
        "count": len(image_paths),
        "width": target_w,
        "height": target_h,
        "channels": channels,
        "mode": mode,
        "resize": "center_crop" if has_varying else "center_crop",
    }


def get_hndl_templates(width: int, height: int, channels: int) -> Tuple[str, str]:
    """Select optimal generator and critic HNDL architecture templates for resolution."""
    max_dim = max(width, height)

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

    return gen.strip(), disc.strip()


def generate_recipe(name: str,
                    dataset_dir: Union[Path, str],
                    output_path: Optional[Union[Path, str]] = None,
                    overwrite: bool = False,
                    device: str = "cuda") -> Path:
    """Generate and save a runnable hypergan-model/1 recipe for dataset_dir."""
    dataset_path = Path(dataset_dir).resolve()
    info = inspect_image_dataset(dataset_path)

    w = info["width"]
    h = info["height"]
    c = info["channels"]
    mode = info["mode"]

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

    # Check workspace data path interpolation
    ws_file = find_workspace(target_path) or find_workspace(Path.cwd())
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

    gen_source, disc_source = get_hndl_templates(w, h, c)

    spec: Dict[str, Any] = {
        "format": "hypergan-model/1",
        "name": name,
        "data": {
            "builtin": "image_folder",
            "args": {
                "root": root_str,
                "height": h,
                "width": w,
                "mode": mode,
                "resize": "center_crop",
            },
        },
        "prior": {
            "kind": "particles",
            "z_dim": 128,
            "args": {"num_particles": 10000},
        },
        "networks": {
            "generator": {
                "role": "generator",
                "inputs": {"z": "latent"},
                "input_shape": ["B", 128],
                "output_shape": ["B", c, h, w],
                "source": gen_source,
            },
            "discriminator": {
                "role": "critic",
                "inputs": {"x": "candidate"},
                "input_shape": ["B", c, h, w],
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
                "weight": 1.0,
            },
        ],
        "train": {
            "steps": 200000,
            "batch_size": 64,
            "seed": 42,
            "device": device,
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

    from .api import from_dict, save
    model_obj = from_dict(spec)
    saved = save(model_obj, target_path, overwrite=overwrite)
    return Path(saved).resolve()

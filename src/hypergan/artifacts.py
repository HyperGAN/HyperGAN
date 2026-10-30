"""Native EMA inference artifacts. These deliberately are not resume checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import uuid

import numpy as np
import torch

from .config import config_values, resolve_config, sampling_bindings, evaluation_bindings
from .recipes import ComponentGraph, generation_output, generation_particle_ids, make_prior
from .run_state import sync_directory


def bundle_state(trainer, batch, include_example_inputs: bool = True):
    """Assemble inference state; callers own copying and RNG isolation."""
    # Retain generation/sampling and explicitly selected evaluation dependency
    # graphs, omitting unrelated training components from inference artifacts.
    needed_components = set()
    called_components = set()
    def include(name):
        if name in called_components:
            return
        called_components.add(name)
        needed_components.add(name)
        if 'reuse' in trainer.config['components'][name]:
            needed_components.add(trainer.config['components'][name]['reuse'])
        for path in trainer.config["components"][name]["inputs"].values():
            if path.startswith("components."):
                include(path.split(".")[1])
    include("generator")
    for binding in sampling_bindings(trainer.config['sampling']):
        if binding.startswith('components.'):
            include(binding.split('.')[1])
    sampling_components = set(called_components)
    for binding in evaluation_bindings(trainer.config):
        if binding.startswith('components.'):
            include(binding.split('.')[1])
    specs = {name: spec for name, spec in trainer.config["components"].items() if name in needed_components}
    needed = {path.split(".")[1] for name in sampling_components for path in specs[name]["inputs"].values() if path.startswith("batch.")}
    models = {name: trainer.ema_graph.models[name] for name, spec in specs.items() if 'reuse' not in spec}
    raw_inputs = {k: v for k, v in batch.items() if k in needed}
    if not include_example_inputs:
        example_inputs = {}
        for k, v in raw_inputs.items():
            if isinstance(v, torch.Tensor):
                example_inputs[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                example_inputs[k] = v
    else:
        example_inputs = raw_inputs
    state = {"schema_version": 1, "kind": "ema-inference", "resume_supported": False, "step": trainer.step, "config": config_values(trainer.config), "components": specs, "model_states": {name: model.state_dict() for name, model in models.items()}, "prior": trainer.ema_prior.state_dict(), "example_inputs": example_inputs}
    state["identity"] = getattr(trainer, "artifact_identity", {})
    # state_dict deliberately omits nonpersistent buffers; custom inference
    # modules can still use these values in their forward pass.
    state["model_buffers"] = {name: dict(model.named_buffers()) for name, model in models.items()}
    state["prior_buffers"] = dict(trainer.ema_prior.named_buffers())
    return state


def save_bundle(run_dir, trainer, batch, include_example_inputs: bool = True):
    run_dir = Path(run_dir)
    state = bundle_state(trainer, batch, include_example_inputs=include_example_inputs)
    temporary = run_dir / "model.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(run_dir / "model.pt")
    digest = hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()
    checksum = run_dir / "model.sha256.tmp"
    with checksum.open("w", encoding="utf-8") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    checksum.replace(run_dir / "model.sha256")
    sync_directory(run_dir)


def sample(run_dir, count=16, seed=42, output=None, *, inputs=None, checkpoint=None, default_image_png=False, device=None, output_dir=None):
    """Sample with supplied batched inputs, or explicitly recorded example conditions.

    Custom component constructors are executable trusted Python, even though state
    tensors load with weights_only=True. An explicit .png output selects a bounded
    RGB/grayscale grid; the default and .json outputs retain generic tensor JSON.
    PNG stores provenance in its hypergan text chunk, JSON in its usual fields.
    """
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("count must be positive and seed nonnegative integers")
    if output is not None and Path(output).suffix.lower() == '.png':
        from .image_grids import MAX_COUNT
        if count > MAX_COUNT:
            raise ValueError(f'PNG sampling supports at most {MAX_COUNT} images')
    # Constructors and custom modules may use global RNGs even in eval mode.
    # An observer must neither consume training randomness nor depend on it.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_device = torch.cuda.current_device() if torch.cuda.is_initialized() else None
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            return _sample(run_dir, count, seed, output, inputs=inputs, checkpoint=checkpoint, default_image_png=default_image_png, device=device, output_dir=output_dir)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)


def _coerce_conditioning_input(value, expected_input):
    exp_shape = None
    exp_dtype = torch.float32
    if isinstance(expected_input, dict) and "shape" in expected_input:
        exp_shape = tuple(expected_input["shape"])
        dtype_str = expected_input.get("dtype", "float32").replace("torch.", "")
        exp_dtype = getattr(torch, dtype_str, torch.float32)
    elif isinstance(expected_input, torch.Tensor):
        exp_shape = tuple(expected_input.shape)
        exp_dtype = expected_input.dtype

    if isinstance(value, dict) and "shape" in value:
        spec_shape = tuple(value["shape"])
        spec_dtype_str = value.get("dtype", "float32").replace("torch.", "")
        spec_dtype = getattr(torch, spec_dtype_str, torch.float32)
        return torch.zeros(spec_shape, dtype=spec_dtype)

    is_path = isinstance(value, (str, Path))
    is_pil = not is_path and type(value).__module__.startswith("PIL.") and hasattr(value, "convert")

    if is_path:
        str_val = str(value).strip()
        path = Path(value)

        parsed_json = None
        if path.is_file() and path.suffix.lower() == ".json":
            try:
                parsed_json = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                parsed_json = None
        elif str_val.startswith(("{", "[")):
            try:
                parsed_json = json.loads(str_val)
            except Exception:
                parsed_json = None

        if parsed_json is not None:
            val_tensor = torch.as_tensor(parsed_json, dtype=exp_dtype)
            if exp_shape and len(exp_shape) >= 2 and val_tensor.shape == exp_shape[1:]:
                val_tensor = val_tensor.unsqueeze(0)
            return val_tensor

        if not path.is_file() and re.match(r"^[\d\s,.\-+eE\[\]]+$", str_val):
            try:
                cleaned = str_val.replace("[", "").replace("]", "")
                numbers = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
                val_tensor = torch.as_tensor(numbers, dtype=exp_dtype)
                if exp_shape and len(exp_shape) >= 2 and val_tensor.shape == exp_shape[1:]:
                    val_tensor = val_tensor.unsqueeze(0)
                return val_tensor
            except Exception:
                pass

        if not path.is_file():
            raise FileNotFoundError(f"Conditioning input image not found: {path}")

    if is_path or is_pil:
        from PIL import Image, ImageOps
        if is_path:
            with Image.open(path) as image:
                image.load()
                rgba = ImageOps.exif_transpose(image).convert("RGBA")
        else:
            rgba = ImageOps.exif_transpose(value).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb_image = Image.alpha_composite(background, rgba).convert("RGB")
        if exp_shape and len(exp_shape) == 4:
            c, h, w = (int(exp_shape[1]), int(exp_shape[2]), int(exp_shape[3]))
        else:
            c, h, w = (1, 128, 128)
        padded = ImageOps.pad(
            rgb_image,
            (w, h),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )
        tensor = torch.frombuffer(bytearray(padded.tobytes()), dtype=torch.uint8)
        rgb = tensor.reshape(1, h, w, 3).permute(0, 3, 1, 2).to(torch.float32).div_(127.5).sub_(1.0).contiguous()
        if c == 1:
            val_out = (rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114).contiguous()
        else:
            val_out = rgb
        return val_out.to(dtype=exp_dtype)

    val_tensor = torch.as_tensor(value, dtype=exp_dtype)
    if exp_shape and len(exp_shape) >= 2 and val_tensor.shape == exp_shape[1:]:
        val_tensor = val_tensor.unsqueeze(0)
    return val_tensor


def _sample(run_dir, count, seed, output, *, inputs, checkpoint=None, default_image_png=False, device=None, output_dir=None):
    from .recipes import execution_device
    exec_device = execution_device(device, fallback=True) if device is not None else torch.device("cpu")
    run_dir = Path(run_dir)
    is_direct_bundle = run_dir.is_file() and run_dir.suffix.lower() == ".pt"
    if is_direct_bundle:
        path = run_dir
        bundle_dir = run_dir.parent
        stem_sha = run_dir.with_suffix(".sha256")
        model_sha = bundle_dir / "model.sha256"
        if stem_sha.is_file():
            expected = stem_sha.read_text(encoding="utf-8").strip()
        elif model_sha.is_file():
            expected = model_sha.read_text(encoding="utf-8").strip()
        else:
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Inference artifact hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        bundle_dir = run_dir
        published_bundle = False
        manifest_path = run_dir / "manifest.json"
        if checkpoint is None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("bundle_path"):
                path = Path(manifest["bundle_path"])
                bundle_dir = (path if path.is_absolute() else run_dir / path).parent
                published_bundle = True
        target_bundle = bundle_dir / "model.pt"
        if checkpoint is not None or (not published_bundle and not target_bundle.is_file()):
            from .checkpoints import resolve_checkpoint, extract_ema_bundle
            target = checkpoint if checkpoint is not None else "latest"
            ckpt_dir = resolve_checkpoint(run_dir, target)
            state = extract_ema_bundle(ckpt_dir)
            source_file = next(
                (ckpt_dir / cand for cand in ("model.pt", "state.pt", "rank-00000.pt") if (ckpt_dir / cand).is_file()),
                ckpt_dir / "state.pt",
            )
            expected = hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file.is_file() else ""
        else:
            path = target_bundle
            if not path.is_file():
                raise ValueError("Published inference artifact is missing")
            expected = (bundle_dir / "model.sha256").read_text().strip()
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Inference artifact hash mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("kind") != "ema-inference":
        raise ValueError("Unsupported inference artifact schema")
    for key in ("config", "components", "model_states", "prior", "example_inputs"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"Invalid inference artifact: missing or invalid {key}")
    if "generator" not in state["components"] or set(state["model_states"]) != {name for name, spec in state["components"].items() if 'reuse' not in spec}:
        raise ValueError("Invalid inference artifact: component states must exactly match the generator graph")
    config = resolve_config(state["config"])
    graph = ComponentGraph(state["components"]).to(exec_device).float().eval().requires_grad_(False)
    for name, weights in state["model_states"].items():
        graph.models[name].load_state_dict(weights)
        if "model_buffers" in state:
            _restore_buffers(graph.models[name], state["model_buffers"].get(name))
    prior = make_prior(config["prior"], device=str(exec_device), recipe=config.get('recipe')).to(exec_device).float().eval().requires_grad_(False)
    prior.load_state_dict(state["prior"])
    if "prior_buffers" in state:
        _restore_buffers(prior, state["prior_buffers"])
    supplied = inputs is not None
    if inputs is None:
        batch = state["example_inputs"]
    elif isinstance(inputs, dict):
        batch = inputs
    elif (
        isinstance(inputs, (str, Path, torch.Tensor))
        or (type(inputs).__module__.startswith("PIL.") and hasattr(inputs, "convert"))
    ):
        if len(state["example_inputs"]) != 1:
            raise ValueError(
                f"Single-value conditioning input requires a model with 1 conditioning input, "
                f"found {len(state['example_inputs'])}: {list(state['example_inputs'])}"
            )
        only_key = next(iter(state["example_inputs"]))
        batch = {only_key: inputs}
    else:
        raise ValueError("Inference inputs must be a mapping of batched tensors")
    normalized = {}
    for key, value in batch.items():
        expected_input = state["example_inputs"].get(key)
        value = _coerce_conditioning_input(value, expected_input)
        if value.ndim < 1 or len(value) == 0:
            raise ValueError(f"Input {key} must have a nonempty batch dimension")
        if supplied and len(value) == 1 and count > 1:
            value = value.expand(count, *value.shape[1:]).contiguous()
        if supplied and len(value) != count:
            raise ValueError(f"Input {key} must have {count} rows")
        normalized[key] = (value if supplied else value[torch.arange(count) % len(value)]).to(exec_device)
    rng = torch.Generator(device=exec_device if exec_device.type == "cuda" else "cpu").manual_seed(seed)
    with torch.inference_mode():
        z, ids = prior.sample(count, generator=rng)
        context = graph.generate(z, normalized, prior=prior)
        values = generation_output(graph, context, config['sampling'])
        ids = generation_particle_ids(graph, context, ids, config['sampling'])
    values = values.cpu()
    if ids is not None and isinstance(ids, torch.Tensor):
        ids = ids.cpu()
    normalized = {key: val.cpu() for key, val in normalized.items()}
    if not isinstance(values, torch.Tensor) or values.ndim < 1 or len(values) != count:
        raise ValueError(f"Generator must return a tensor with {count} samples")
    if not torch.isfinite(values).all():
        raise ValueError("Inference produced nonfinite values")

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if values.ndim == 4 and values.shape[1] in (1, 3):
            from PIL import Image
            img_tensors = values.detach().clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255.0).to(torch.uint8)
            for i in range(count):
                single = img_tensors[i]
                if single.shape[0] == 1:
                    arr = single.squeeze(0).numpy()
                    im = Image.fromarray(arr, mode="L")
                else:
                    arr = single.permute(1, 2, 0).numpy()
                    im = Image.fromarray(arr, mode="RGB")
                im.save(out_dir / f"sample_{i:04d}.png")
        else:
            payload_data = {
                "schema_version": 1,
                "seed": seed,
                "count": count,
                "step": state.get("step"),
                "samples": values.tolist(),
                "inputs": {k: v.tolist() for k, v in normalized.items()}
            }
            (out_dir / "samples.json").write_text(json.dumps(payload_data, indent=2) + "\n", encoding="utf-8")
        if output is None:
            return out_dir

    if output is not None:
        output = Path(output)
    else:
        from .image_grids import MAX_COUNT
        is_image_tensor = (
            default_image_png
            and values.ndim == 4
            and values.shape[1] in (1, 3)
            and values.shape[2] >= 4
            and values.shape[3] >= 4
            and count <= MAX_COUNT
        )
        ext = "png" if is_image_tensor else "json"
        sub = (run_dir.parent / "samples") if is_direct_bundle else ((run_dir / "samples") if is_image_tensor else run_dir)
        output = sub / f"samples-step{state.get('step', 0):08d}-seed{seed}-n{count}-{uuid.uuid4().hex}.{ext}"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "seed": seed, "count": count, "step": state.get("step"), "bundle_sha256": expected, "shape": list(values.shape), "particle_ids": ids.tolist() if ids is not None else None, "conditioning": "supplied" if supplied else ("saved-example-inputs-cycled" if batch else "unconditional"), "resume_supported": False}
    payload["identity"] = state.get("identity", {})
    if output.suffix.lower() == '.png':
        from .image_grids import tensor_grid
        payload['input_shapes'] = {key: list(value.shape) for key, value in normalized.items()}
        encoded, _ = tensor_grid(values, payload)
    else:
        payload.update(samples=values.tolist(), inputs={key: value.tolist() for key, value in normalized.items()})
        encoded = (json.dumps(payload, allow_nan=False) + '\n').encode('utf-8')
    descriptor, temporary = tempfile.mkstemp(prefix=".sample-", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
        sync_directory(output.parent)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return output


def _restore_buffers(module, saved):
    buffers = dict(module.named_buffers())
    if not isinstance(saved, dict) or set(saved) != set(buffers):
        raise ValueError("Inference buffer inventory differs from the saved component")
    with torch.no_grad():
        for name, target in buffers.items():
            value = saved[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Incompatible inference buffer: {name}")
            target.copy_(value)


sample_run = sample


def decode_png_sample_tensor(path_or_bytes):
    """Reconstruct (N, C, H, W) float32 [-1, 1] tensor from a saved PNG sample grid."""
    import io
    from PIL import Image
    if isinstance(path_or_bytes, (str, Path)):
        img = Image.open(path_or_bytes, formats=["PNG"])
    else:
        img = Image.open(io.BytesIO(path_or_bytes), formats=["PNG"])
    raw_meta = img.info.get('hypergan', '{}')
    payload = json.loads(raw_meta) if raw_meta else {}
    grid_meta = payload.get('grid', {})
    count = grid_meta.get('count', payload.get('count', 1))
    rows = grid_meta.get('rows', 1)
    columns = grid_meta.get('columns', 1)
    tile_w = img.width // columns
    tile_h = img.height // rows

    tiles = []
    for idx in range(count):
        r, c = divmod(idx, columns)
        box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
        tile = img.crop(box)
        arr = np.array(tile, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        arr = np.transpose(arr, (2, 0, 1))
        arr = (arr / 127.5) - 1.0
        tiles.append(arr)
    return torch.from_numpy(np.stack(tiles, axis=0))


def interpolate(v0: torch.Tensor, v1: torch.Tensor, steps: int = 10, *, mode: str = "slerp", t: float | None = None) -> torch.Tensor:
    """Perform linear or spherical linear interpolation between two latent vectors."""
    if steps < 1:
        raise ValueError("steps must be at least 1")

    def _lerp(a, b, alpha):
        return (1.0 - alpha) * a + alpha * b

    def _slerp(a, b, alpha, eps=1e-7):
        a_norm = a / torch.norm(a, dim=-1, keepdim=True).clamp_min(eps)
        b_norm = b / torch.norm(b, dim=-1, keepdim=True).clamp_min(eps)
        dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        collinear = dot.abs() > 0.9995
        if collinear.all():
            return _lerp(a, b, alpha)
        safe_dot = torch.where(collinear, torch.zeros_like(dot), dot)
        omega = torch.acos(safe_dot)
        sin_omega = torch.sin(omega).clamp_min(eps)
        scale0 = torch.sin((1.0 - alpha) * omega) / sin_omega
        scale1 = torch.sin(alpha * omega) / sin_omega
        slerp_val = scale0 * a + scale1 * b
        return torch.where(collinear, _lerp(a, b, alpha), slerp_val)

    interp_fn = _slerp if mode == "slerp" else _lerp
    if t is not None:
        return interp_fn(v0, v1, float(t))
    if steps == 1:
        return interp_fn(v0, v1, 0.5)
    alphas = torch.linspace(0.0, 1.0, steps, device=v0.device)
    points = [interp_fn(v0, v1, float(a)) for a in alphas]
    return torch.cat(points, dim=0) if v0.ndim >= 2 else torch.stack(points, dim=0)


class LoadedGenerator(torch.nn.Module, dict):
    """Callable standalone generator module with baked particle table and inference helpers."""
    __hash__ = object.__hash__

    def __init__(self, bundle: dict | Path | str, device: str = "cpu"):
        torch.nn.Module.__init__(self)
        if isinstance(bundle, (str, Path)):
            bundle = torch.load(bundle, map_location=device, weights_only=True)
        dict.__init__(self, bundle)
        from .recipes import execution_device
        self.device = execution_device(device, fallback=True)
        self.config = bundle.get("config", {})
        self.components = bundle.get("components", {})
        self.example_inputs = bundle.get("example_inputs", {})

        # Standalone particle baking (zero particlegan dependency)
        prior_state = bundle.get("prior", {})
        if isinstance(prior_state, dict) and "z" in prior_state and isinstance(prior_state["z"], torch.Tensor):
            self.register_buffer("particle_z", prior_state["z"].clone().detach().to(self.device))
        else:
            self.particle_z = None

        if isinstance(prior_state, dict) and "sigma" in prior_state and isinstance(prior_state["sigma"], torch.Tensor):
            self.register_buffer("particle_sigma", prior_state["sigma"].clone().detach().to(self.device))
        else:
            self.particle_sigma = None

        # Build generator component graph
        self.graph = ComponentGraph(self.components).to(self.device).eval().requires_grad_(False)
        for name, weights in bundle.get("model_states", {}).items():
            if name in self.graph.models:
                self.graph.models[name].load_state_dict(weights)
        if "model_buffers" in bundle:
            for name, bufs in bundle["model_buffers"].items():
                if name in self.graph.models and isinstance(bufs, dict):
                    _restore_buffers(self.graph.models[name], bufs)

    def means(self):
        return self.particle_z

    @property
    def sigma(self):
        return self.particle_sigma

    def sample_latents(self, count: int, generator: torch.Generator | None = None, seed: int | None = None) -> torch.Tensor:
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif generator is not None and seed is not None:
            raise ValueError("Cannot provide both generator and seed")
        if self.particle_z is not None:
            idx = torch.randint(0, len(self.particle_z), (count,), generator=generator, device=self.device)
            z = self.particle_z[idx]
            if self.particle_sigma is not None and float(self.particle_sigma) > 0:
                noise = torch.randn(count, self.particle_z.shape[1], generator=generator, device=self.device, dtype=z.dtype)
                z = z + self.particle_sigma * noise
            return z
        else:
            z_dim = self.config.get("prior", {}).get("z_dim", 4)
            return torch.randn(count, z_dim, generator=generator, device=self.device)

    def sample_prior(self, count: int, generator: torch.Generator | None = None, seed: int | None = None) -> torch.Tensor:
        return self.sample_latents(count, generator=generator, seed=seed)

    def forward(self, z: torch.Tensor | None = None, inputs: dict | None = None, **kwargs) -> torch.Tensor:
        if inputs is None:
            inputs = kwargs.get("conditioning") or kwargs.get("batch") or {}
        if z is None:
            count = len(next(iter(inputs.values()))) if inputs else 1
            z = self.sample_latents(count)
        else:
            count = len(z)
        if inputs:
            inputs = {
                k: (v.expand(count, *v.shape[1:]) if isinstance(v, torch.Tensor) and len(v) == 1 and count > 1 else v)
                for k, v in inputs.items()
            }
        context = self.graph.generate(z, inputs, prior=self)
        return generation_output(self.graph, context, self.config.get("sampling", {}))

    def sample(self, count: int = 16, seed: int | None = None, inputs: dict | None = None, conditioning: dict | None = None) -> torch.Tensor:
        rng = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        z = self.sample_latents(count, generator=rng)
        cond = conditioning if conditioning is not None else inputs
        return self.forward(z=z, inputs=cond)

    def interpolate(self, seed_a: int | None = None, seed_b: int | None = None, steps: int = 10,
                    *, z_a: torch.Tensor | None = None, z_b: torch.Tensor | None = None,
                    mode: str = "slerp", inputs: dict | None = None, conditioning: dict | None = None) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if z_a is None or z_b is None:
            if seed_a is None or seed_b is None:
                raise ValueError("Must provide either (z_a, z_b) or (seed_a, seed_b)")
            rng_a = torch.Generator(device=self.device).manual_seed(seed_a)
            rng_b = torch.Generator(device=self.device).manual_seed(seed_b)
            z_a = self.sample_latents(1, generator=rng_a)
            z_b = self.sample_latents(1, generator=rng_b)
        else:
            z_a = torch.as_tensor(z_a, device=self.device)
            z_b = torch.as_tensor(z_b, device=self.device)
            if z_a.ndim == 1:
                z_a = z_a.unsqueeze(0)
            if z_b.ndim == 1:
                z_b = z_b.unsqueeze(0)
        z_walk = interpolate(z_a, z_b, steps=steps, mode=mode)
        cond = conditioning if conditioning is not None else inputs
        return self.forward(z=z_walk, inputs=cond)

    def to(self, *args, **kwargs):
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = torch.device(args[0])
        elif "device" in kwargs:
            device = torch.device(kwargs["device"])
        if device is not None:
            self.device = device
        return super().to(*args, **kwargs)


def export_generator(run_dir: Path | str, output_path: Path | str | None = None,
                     format: str = "bundle", include_example_inputs: bool = False,
                     checkpoint: str = "latest", device: str = "cpu") -> Path:
    """Export standalone generator across bundle, safetensors, pt2, or onnx formats."""
    from .checkpoints import resolve_checkpoint, extract_ema_bundle
    from .api import _run_path

    valid_formats = ("bundle", "safetensors", "pt2", "onnx")
    if format not in valid_formats:
        raise ValueError(f"Unknown export format {format!r}; expected one of {valid_formats}")

    resolved_run = _run_path(run_dir)
    ckpt_dir = resolve_checkpoint(resolved_run, target=checkpoint)
    bundle = extract_ema_bundle(ckpt_dir, include_example_inputs=include_example_inputs)

    ext_map = {"bundle": ".pt", "safetensors": ".safetensors", "pt2": ".pt2", "onnx": ".onnx"}
    target_ext = ext_map[format]

    if output_path is None:
        dest = resolved_run / f"model{target_ext}"
    else:
        dest = Path(output_path)
        if dest.is_dir() or (not dest.exists() and dest.suffix == ""):
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / f"model{target_ext}"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

    if format == "bundle":
        temp_pt = dest.with_suffix(".pt.tmp")
        with temp_pt.open("wb") as stream:
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp_pt.replace(dest)

        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        chk = dest.with_name("model.sha256") if dest.name == "model.pt" else dest.with_suffix(".sha256")
        chk.write_text(digest + "\n", encoding="utf-8")
        return dest

    elif format == "safetensors":
        import safetensors.torch
        tensors = {}
        for comp_name, wdict in bundle.get("model_states", {}).items():
            for k, v in wdict.items():
                tensors[f"{comp_name}.{k}"] = v.contiguous().cpu()
        if "prior" in bundle and isinstance(bundle["prior"], dict):
            for k, v in bundle["prior"].items():
                if isinstance(v, torch.Tensor):
                    tensors[f"prior.{k}"] = v.contiguous().cpu()
        meta = {
            "format": "safetensors",
            "schema_version": "1",
            "step": str(bundle.get("step", 0)),
            "config": json.dumps(bundle.get("config", {})),
            "example_inputs": json.dumps(bundle.get("example_inputs", {}))
        }
        safetensors.torch.save_file(tensors, dest, metadata=meta)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        chk = dest.with_suffix(".sha256")
        chk.write_text(digest + "\n", encoding="utf-8")
        return dest

    elif format == "pt2":
        lg = LoadedGenerator(bundle, device=device)
        z_dim = bundle.get("config", {}).get("prior", {}).get("z_dim", 4)
        dummy_z = torch.randn(1, z_dim, device=lg.device)
        try:
            exported = torch.export.export(lg, (dummy_z,))
            torch.export.save(exported, dest)
        except Exception:
            traced = torch.jit.trace(lg, (dummy_z,))
            torch.jit.save(traced, dest)
        return dest

    elif format == "onnx":
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise ImportError("Exporting to ONNX format requires the onnx package: pip install hypergan[onnx]")
        lg = LoadedGenerator(bundle, device=device)
        z_dim = bundle.get("config", {}).get("prior", {}).get("z_dim", 4)
        dummy_z = torch.randn(1, z_dim, device=lg.device)
        torch.onnx.export(
            lg, (dummy_z,), dest,
            input_names=["latent"], output_names=["output"],
            dynamic_axes={"latent": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False
        )
        return dest


"""Native EMA inference artifacts. These deliberately are not resume checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np
import torch

from .config import config_values, resolve_config, sampling_bindings, evaluation_bindings
from .recipes import ComponentGraph, generation_output, generation_particle_ids, make_prior
from .run_state import sync_directory


def bundle_state(trainer, batch):
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
    state = {"schema_version": 1, "kind": "ema-inference", "resume_supported": False, "step": trainer.step, "config": config_values(trainer.config), "components": specs, "model_states": {name: model.state_dict() for name, model in models.items()}, "prior": trainer.ema_prior.state_dict(), "example_inputs": {k: v for k, v in batch.items() if k in needed}}
    state["identity"] = getattr(trainer, "artifact_identity", {})
    # state_dict deliberately omits nonpersistent buffers; custom inference
    # modules can still use these values in their forward pass.
    state["model_buffers"] = {name: dict(model.named_buffers()) for name, model in models.items()}
    state["prior_buffers"] = dict(trainer.ema_prior.named_buffers())
    return state


def save_bundle(run_dir, trainer, batch):
    run_dir = Path(run_dir)
    state = bundle_state(trainer, batch)
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


def sample(run_dir, count=16, seed=42, output=None, *, inputs=None, checkpoint=None, default_image_png=False):
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
            return _sample(run_dir, count, seed, output, inputs=inputs, checkpoint=checkpoint, default_image_png=default_image_png)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)


def _sample(run_dir, count, seed, output, *, inputs, checkpoint=None, default_image_png=False):
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
            source_file = ckpt_dir / "model.pt" if (ckpt_dir / "model.pt").is_file() else ckpt_dir / "state.pt"
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
    graph = ComponentGraph(state["components"]).float().eval().requires_grad_(False)
    for name, weights in state["model_states"].items():
        graph.models[name].load_state_dict(weights)
        if "model_buffers" in state:
            _restore_buffers(graph.models[name], state["model_buffers"].get(name))
    prior = make_prior(config["prior"], device='cpu', recipe=config.get('recipe')).float().eval().requires_grad_(False)
    prior.load_state_dict(state["prior"])
    if "prior_buffers" in state:
        _restore_buffers(prior, state["prior_buffers"])
    supplied = inputs is not None
    batch = state["example_inputs"] if inputs is None else inputs
    if not isinstance(batch, dict):
        raise ValueError("Inference inputs must be a mapping of batched tensors")
    normalized = {}
    for key, value in batch.items():
        expected_input = state["example_inputs"].get(key)
        value = torch.as_tensor(value, dtype=expected_input.dtype if isinstance(expected_input, torch.Tensor) else None)
        if value.ndim < 1 or len(value) == 0:
            raise ValueError(f"Input {key} must have a nonempty batch dimension")
        if supplied and len(value) != count:
            raise ValueError(f"Input {key} must have {count} rows")
        normalized[key] = value if supplied else value[torch.arange(count) % len(value)]
    rng = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        z, ids = prior.sample(count, generator=rng)
        context = graph.generate(z, normalized, prior=prior, lazy=config['sampling'].get('generated', 'generated') != 'generated')
        values = generation_output(graph, context, config['sampling'])
        ids = generation_particle_ids(graph, context, ids, config['sampling'])
    if not isinstance(values, torch.Tensor) or values.ndim < 1 or len(values) != count:
        raise ValueError(f"Generator must return a tensor with {count} samples")
    if not torch.isfinite(values).all():
        raise ValueError("Inference produced nonfinite values")
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
        img = Image.open(path_or_bytes)
    else:
        img = Image.open(io.BytesIO(path_or_bytes))
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

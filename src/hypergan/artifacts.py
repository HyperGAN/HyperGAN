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

from .config import config_values, resolve_config
from .recipes import ComponentGraph, make_prior
from .run_state import sync_directory


def bundle_state(trainer, batch):
    """Assemble inference state; callers own copying and RNG isolation."""
    # Keep exactly the generator dependency graph, omitting objective-only encoders
    # and discriminator conditioners from the inference environment.
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
    specs = {name: spec for name, spec in trainer.config["components"].items() if name in needed_components}
    needed = {path.split(".")[1] for name in called_components for path in specs[name]["inputs"].values() if path.startswith("batch.")}
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


def sample(run_dir, count=16, seed=42, output=None, *, inputs=None):
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
            return _sample(run_dir, count, seed, output, inputs=inputs)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)


def _sample(run_dir, count, seed, output, *, inputs):
    run_dir = Path(run_dir)
    bundle_dir = run_dir
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle_path"):
            path = Path(manifest["bundle_path"])
            bundle_dir = (path if path.is_absolute() else run_dir / path).parent
    path = bundle_dir / "model.pt"
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
    prior = make_prior(config["prior"], device='cpu').float().eval().requires_grad_(False)
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
        values = graph.generate(z, normalized, prior=prior)["generated"]
    if not isinstance(values, torch.Tensor) or values.ndim < 1 or len(values) != count:
        raise ValueError(f"Generator must return a tensor with {count} samples")
    if not torch.isfinite(values).all():
        raise ValueError("Inference produced nonfinite values")
    output = Path(output) if output is not None else run_dir / f"samples-step{state.get('step', 0):08d}-seed{seed}-n{count}-{uuid.uuid4().hex}.json"
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
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(temporary, output)
        sync_directory(output.parent)
    finally:
        os.unlink(temporary)
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

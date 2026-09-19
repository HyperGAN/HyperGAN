"""Native EMA inference artifacts. These deliberately are not resume checkpoints."""
import hashlib
import json
from pathlib import Path

import torch

from .config import config_values, resolve_config
from .recipes import ComponentGraph, make_prior


def save_bundle(run_dir, trainer, batch):
    run_dir = Path(run_dir)
    # Keep exactly the generator dependency graph, omitting objective-only encoders
    # and discriminator conditioners from the inference environment.
    needed_components = set()
    def include(name):
        if name in needed_components:
            return
        needed_components.add(name)
        for path in trainer.config["components"][name]["inputs"].values():
            if path.startswith("components."):
                include(path.split(".")[1])
    include("generator")
    specs = {name: spec for name, spec in trainer.config["components"].items() if name in needed_components}
    needed = {path.split(".")[1] for spec in specs.values() for path in spec["inputs"].values() if path.startswith("batch.")}
    state = {"schema_version": 1, "kind": "ema-inference", "resume_supported": False, "config": config_values(trainer.config), "components": specs, "model_states": {name: trainer.ema_graph.models[name].state_dict() for name in specs}, "prior": trainer.ema_prior.state_dict(), "example_inputs": {k: v for k, v in batch.items() if k in needed}}
    temporary = run_dir / "model.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(run_dir / "model.pt")
    digest = hashlib.sha256((run_dir / "model.pt").read_bytes()).hexdigest()
    (run_dir / "model.sha256").write_text(digest + "\n")


def sample(run_dir, count=16, seed=42, output=None, *, inputs=None):
    """Sample with supplied batched inputs, or explicitly recorded example conditions.

    Custom component constructors are executable trusted Python, even though state
    tensors load with weights_only=True. JSON outputs include conditioning provenance.
    """
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("count must be positive and seed nonnegative integers")
    run_dir = Path(run_dir)
    path = run_dir / "model.pt"
    expected = (run_dir / "model.sha256").read_text().strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Inference artifact hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("kind") != "ema-inference":
        raise ValueError("Unsupported inference artifact schema")
    for key in ("config", "components", "model_states", "prior", "example_inputs"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"Invalid inference artifact: missing or invalid {key}")
    if "generator" not in state["components"] or set(state["model_states"]) != set(state["components"]):
        raise ValueError("Invalid inference artifact: component states must exactly match the generator graph")
    config = resolve_config(state["config"])
    graph = ComponentGraph(state["components"]).float().eval().requires_grad_(False)
    for name, weights in state["model_states"].items():
        graph.models[name].load_state_dict(weights)
    prior = make_prior(config["prior"]).float().eval().requires_grad_(False)
    prior.load_state_dict(state["prior"])
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
        values = graph.generate(z, normalized)["generated"]
    if not isinstance(values, torch.Tensor) or values.ndim < 1 or len(values) != count:
        raise ValueError(f"Generator must return a tensor with {count} samples")
    if not torch.isfinite(values).all():
        raise ValueError("Inference produced nonfinite values")
    output = Path(output) if output is not None else run_dir / f"samples-seed{seed}-n{count}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "seed": seed, "count": count, "shape": list(values.shape), "samples": values.tolist(), "particle_ids": ids.tolist() if ids is not None else None, "conditioning": "supplied" if supplied else ("saved-example-inputs-cycled" if batch else "unconditional"), "inputs": {k: v.tolist() for k, v in normalized.items()}, "resume_supported": False}
    with output.open("x") as stream:
        json.dump(payload, stream, allow_nan=False)
        stream.write("\n")
    return output

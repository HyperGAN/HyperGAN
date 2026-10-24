"""Lightweight 2-rank replicated recipe parity smoke tests against single-process reference trainer."""
import copy
from pathlib import Path
import pytest
import torch

from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.cpu_workers import launch_cpu_workers


def _assert_close(left, right, path=""):
    """Assert tensor, float or dict numerical equality within rtol=3e-5, atol=3e-6."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor):
            left = torch.tensor(left)
        if not isinstance(right, torch.Tensor):
            right = torch.tensor(right)
        torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6, msg=lambda m: f"{path}: {m}")
    elif isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, rel=3e-5, abs=3e-6), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"{path} keys mismatch: {left.keys()} vs {right.keys()}"
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{path} length mismatch: {len(left)} vs {len(right)}"
        for i, (a, b) in enumerate(zip(left, right)):
            _assert_close(a, b, f"{path}[{i}]")
    else:
        assert left == right, path


def _extract_state(trainer):
    """Capture parameter, EMA, and optimizer states."""
    return {
        "graph": copy.deepcopy(trainer.graph.state_dict()),
        "prior": copy.deepcopy(trainer.prior.state_dict()),
        "ema_graph": copy.deepcopy(trainer.ema_graph.state_dict()),
        "ema_prior": copy.deepcopy(trainer.ema_prior.state_dict()),
        "opt_g": copy.deepcopy(trainer.opt_g.state_dict()),
        "opt_d": copy.deepcopy(trainer.opt_d.state_dict()),
        "step": trainer.step,
    }


def _recipe_parity_worker(rank, world_size, config_dict, full_batch, shared_dir, accumulation_steps=1):
    """Worker executed on rank 0 and rank 1 inside a Gloo process group."""
    config = resolve_config(config_dict)
    replicated = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=accumulation_steps)

    # Slice batch per rank
    local_batch = {k: v.chunk(world_size)[rank] for k, v in full_batch.items() if not k.startswith("_")}

    # Handle prior latents or particle IDs if present
    ids = full_batch.get("_ids")
    prior_arg = None
    if ids is not None and hasattr(replicated.prior, "z"):
        local_ids = ids.chunk(world_size)[rank]
        prior_arg = (replicated.prior.z[local_ids], local_ids)
    elif "_z" in full_batch:
        local_z = full_batch["_z"].chunk(world_size)[rank]
        prior_arg = (local_z, None)

    # Perform replicated training step
    if prior_arg is not None:
        metrics, _ = replicated.update(local_batch, prior_arg)
    else:
        metrics, _ = replicated.update(local_batch)

    # Save rank output for verification
    state = _extract_state(replicated)
    out_file = Path(shared_dir) / f"rank-{rank}.pt"
    torch.save({"metrics": metrics, "state": state}, out_file)


def _run_parity_case(config_dict, full_batch, tmp_path, accumulation_steps=1):
    """Helper to run 2-rank worker and compare against ReferenceTrainer."""
    # 1. Run reference trainer on full batch
    ref_config = resolve_config(copy.deepcopy(config_dict))
    reference = ReferenceTrainer(ref_config)

    ids = full_batch.get("_ids")
    prior_arg = None
    if ids is not None and hasattr(reference.prior, "z"):
        prior_arg = (reference.prior.z[ids], ids)
    elif "_z" in full_batch:
        prior_arg = (full_batch["_z"], None)

    ref_batch = {k: v for k, v in full_batch.items() if not k.startswith("_")}
    if prior_arg is not None:
        ref_metrics, _ = reference.update(ref_batch, prior_arg)
    else:
        ref_metrics, _ = reference.update(ref_batch)
    ref_state = _extract_state(reference)

    # 2. Launch 2-rank Gloo workers via launch_cpu_workers
    launch_cpu_workers(
        _recipe_parity_worker,
        args=(config_dict, full_batch, str(tmp_path), accumulation_steps),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )

    # 3. Load rank results and assert parity
    rank0_data = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    rank1_data = torch.load(tmp_path / "rank-1.pt", weights_only=False)

    # Assert replica equality under gradient_allreduce
    _assert_close(rank0_data["state"], rank1_data["state"], "rank0_vs_rank1_state")

    # Assert parity with single-process ReferenceTrainer
    _assert_close(rank0_data["state"], ref_state, "replicated_vs_reference_state")

    # Assert metrics within rtol=3e-5, atol=3e-6
    for key in ("d_loss", "g_loss"):
        if key in ref_metrics:
            assert rank0_data["metrics"][key] == pytest.approx(ref_metrics[key], rel=3e-5, abs=3e-6)
    if "objectives" in ref_metrics and ref_metrics["objectives"]:
        assert len(rank0_data["metrics"]["objectives"]) == len(ref_metrics["objectives"])
        for o_rank, o_ref in zip(rank0_data["metrics"]["objectives"], ref_metrics["objectives"]):
            assert o_rank == pytest.approx(o_ref, rel=3e-5, abs=3e-6)


def test_replicated_parity_prior_bound_inputs(tmp_path):
    """Smoke test: recipe binding prior.means to component inputs matches reference step."""
    config = {
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 8, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "encoder": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 4, "bias": False},
                "inputs": {"input": "prior.means"},
                "trainable": True,
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "objectives": [
            {
                "factory": "mse",
                "inputs": {"input": "components.encoder", "target": "prior.means"},
                "weight": 0.5,
            }
        ],
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    batch = {
        "real": torch.linspace(-1, 1, 16).reshape(8, 2),
        "_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    }
    _run_parity_case(config, batch, tmp_path)


def test_replicated_parity_component_reuse(tmp_path):
    """Smoke test: recipe reusing component parameters across graph positions matches reference."""
    config = {
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 4, "bias": True},
                "inputs": {"input": "latent"},
            },
            "reconstruction": {
                "reuse": "generator",
                "inputs": {"input": "components.generator"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "objectives": [
            {
                "factory": "mse",
                "inputs": {"input": "components.reconstruction", "target": "batch.real"},
                "weight": 1.0,
            }
        ],
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    batch = {
        "real": torch.linspace(-1, 1, 32).reshape(8, 4),
        "_z": torch.linspace(0.5, -0.5, 32).reshape(8, 4),
    }
    _run_parity_case(config, batch, tmp_path)


def test_replicated_parity_multiple_adversarial_terms(tmp_path):
    """Smoke test: recipe with extra adversarial_terms critics evaluates and matches reference."""
    config = {
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
            "aux_critic": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "adversarial_terms": [
            {
                "id": "aux",
                "component": "aux_critic",
                "weight": 0.5,
                "real": "batch.real",
                "fake": "generated",
                "inputs": {"input": "candidate"},
                "penalty": False,
            }
        ],
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    batch = {
        "real": torch.linspace(-1, 1, 16).reshape(8, 2),
        "_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    }
    _run_parity_case(config, batch, tmp_path)


def test_replicated_parity_particlegan_preset(tmp_path):
    """Smoke test: resolved ParticleGAN preset recipe with recipe optimizer matches reference."""
    config = {
        "defaults": "particlegan",
        "gradient_penalty": {"lazy_k": 2},
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
    }
    batch = {
        "real": torch.linspace(-1, 1, 16).reshape(8, 2),
        "_ids": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    }
    _run_parity_case(config, batch, tmp_path)


def test_replicated_parity_accumulation_steps(tmp_path):
    """Smoke test: 2-rank replicated training with accumulation_steps=2 matches reference."""
    config = {
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    batch = {
        "real": torch.linspace(-1, 1, 16).reshape(8, 2),
        "_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    }
    _run_parity_case(config, batch, tmp_path, accumulation_steps=2)

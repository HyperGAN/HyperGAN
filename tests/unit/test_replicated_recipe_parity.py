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
        # Axiom enforces reg_every=1 (clock-free penalty) and low-D pullback; disabling
        # both here tests exact 2-rank parity of Axiom's coupled loss, transport,
        # simplicial prior, and optimizer.
        "gradient_penalty": {"coeff": 0.0},
        "recipe": {"t1_pullback_weight": 0.0},
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


def test_replicated_parity_particlegan_preset_mog(tmp_path):
    """Smoke test: resolved ParticleGAN preset with MoG prior matches reference."""
    config = {
        "defaults": "particlegan",
        "gradient_penalty": {"coeff": 0.0},
        "recipe": {"t1_pullback_weight": 0.0},
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
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
        "_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
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


def _unified_v11_highres_worker(rank, world_size, config_dict, full_batches, shared_dir):
    """Run 3 high-res 4D image steps on 2 ranks and assert bitwise replica agreement."""
    config = resolve_config(config_dict)
    replicated = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=1)
    for full_batch in full_batches:
        local_batch = {"real": full_batch["real"].chunk(world_size)[rank]}
        local_ids = full_batch["_ids"].chunk(world_size)[rank]
        prior_arg = (replicated.prior.z[local_ids], local_ids)
        replicated.update(local_batch, prior_arg)
    replicated._assert_replicas("unified_v11 highres step 3")
    rec = replicated.opt_d.record
    torch.save(
        {
            "digest": replicated._replica_digest(),
            "ema_margin": rec.ema_margin,
            "ema_data_scale": rec.ema_data_scale,
            "ema_push_res": rec.ema_push_res,
            "margin_initialized": rec.margin_initialized,
            "opt_g_ema_func_gain": replicated.opt_g._ema_func_gain,
            "opt_d_ema_func_gain": replicated.opt_d._ema_func_gain,
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def test_replicated_unified_v11_highres_image_replica_agreement(tmp_path):
    """2-rank unified_v11 on 4D high-res images (dim > 1024) maintains exact replica agreement."""
    config = {
        "defaults": "particlegan:unified_v11",
        "training": {
            "steps": 3,
            "batch_size": 8,
            "device": "cpu",
            "seed": 42,
            "initialization": "deterministic_orthogonal",
        },
        "prior": {"kind": "mog", "fixed_sigma": 0.2, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {
                    "source": "linear(1200)\nreshape(3,20,20)\ntanh()",
                    "input_shape": ["B", 4],
                    "output_shape": ["B", 3, 20, 20],
                },
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": "hndl",
                "args": {
                    "source": "conv(8, kernel_size=3, padding=1)\nleaky_relu(0.2)\nadaptive_avg_pool(2)\nflatten()\nlinear(1)",
                    "input_shape": ["B", 3, 20, 20],
                    "output_shape": ["B", 1],
                },
                "inputs": {"x": "candidate"},
            },
        },
    }
    gen = torch.Generator().manual_seed(7)
    full_batches = [
        {
            "real": torch.randn(8, 3, 20, 20, generator=gen),
            "_ids": torch.arange(8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    launch_cpu_workers(
        _unified_v11_highres_worker,
        args=(config, full_batches, str(tmp_path)),
        world_size=2,
        timeout=60,
        collective_timeout=20,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0 == r1
    assert r0["margin_initialized"] is True
    assert r0["ema_data_scale"] > 0.0


def _unified_v12_highres_worker(rank, world_size, config_dict, full_batches, shared_dir):
    """Run 3 high-res 4D image steps on 2 ranks with unified_v12 and test controller sync."""
    from hypergan.aggregation import _synchronize_optimizer_regularizers

    config = resolve_config(config_dict)
    replicated = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=1)
    assert replicated.recipe.name == "unified_v12"
    assert replicated.recipe.optimizer_family == "unified_v12_equil"
    for full_batch in full_batches:
        local_batch = {"real": full_batch["real"].chunk(world_size)[rank]}
        local_ids = full_batch["_ids"].chunk(world_size)[rank]
        prior_arg = (replicated.prior.z[local_ids], local_ids)
        replicated.update(local_batch, prior_arg)
    replicated._assert_replicas("unified_v12 highres step 3")

    # Exercise local-SGD regularizer synchronization for v12's _g_func_gain, _ema_flip_cos, and _prev_flip_sketch.
    rec = replicated.opt_d.record
    rec._g_func_gain = 0.4 if rank == 0 else 0.8
    replicated.opt_g._ema_flip_cos = -0.2 if rank == 0 else -0.6
    sketch = torch.zeros(67, dtype=torch.float32)
    sketch[0] = 1.0 if rank == 0 else 0.6
    sketch[1] = 0.0 if rank == 0 else 0.8
    replicated.opt_g._prev_flip_sketch = sketch
    _synchronize_optimizer_regularizers(replicated)
    replicated._assert_replicas("unified_v12 synced v12 controller state")

    torch.save(
        {
            "digest": replicated._replica_digest(),
            "ema_margin": rec.ema_margin,
            "ema_data_scale": rec.ema_data_scale,
            "ema_push_res": rec.ema_push_res,
            "margin_initialized": rec.margin_initialized,
            "g_func_gain": rec._g_func_gain,
            "opt_g_ema_func_gain": replicated.opt_g._ema_func_gain,
            "opt_g_ema_flip_cos": replicated.opt_g._ema_flip_cos,
            "opt_g_prev_flip_sketch": replicated.opt_g._prev_flip_sketch.tolist(),
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def test_replicated_unified_v12_highres_image_replica_agreement(tmp_path):
    """2-rank unified_v12 on 4D high-res images maintains exact replica agreement."""
    config = {
        "defaults": "particlegan:unified_v12",
        "training": {
            "steps": 3,
            "batch_size": 8,
            "device": "cpu",
            "seed": 42,
            "initialization": "deterministic_orthogonal",
        },
        "prior": {"kind": "mog", "fixed_sigma": 0.2, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {
                    "source": "linear(1200)\nreshape(3,20,20)\ntanh()",
                    "input_shape": ["B", 4],
                    "output_shape": ["B", 3, 20, 20],
                },
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": "hndl",
                "args": {
                    "source": "conv(8, kernel_size=3, padding=1)\nleaky_relu(0.2)\nadaptive_avg_pool(2)\nflatten()\nlinear(1)",
                    "input_shape": ["B", 3, 20, 20],
                    "output_shape": ["B", 1],
                },
                "inputs": {"x": "candidate"},
            },
        },
    }
    gen = torch.Generator().manual_seed(7)
    full_batches = [
        {
            "real": torch.randn(8, 3, 20, 20, generator=gen),
            "_ids": torch.arange(8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    launch_cpu_workers(
        _unified_v12_highres_worker,
        args=(config, full_batches, str(tmp_path)),
        world_size=2,
        timeout=60,
        collective_timeout=20,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0 == r1
    assert r0["margin_initialized"] is True
    assert r0["ema_data_scale"] > 0.0
    assert abs(r0["g_func_gain"] - 0.6) < 1e-6
    assert abs(r0["opt_g_ema_flip_cos"] - (-0.4)) < 1e-6
    assert len(r0["opt_g_prev_flip_sketch"]) == 67


def _axiom_image_worker(rank, world_size, config_dict, full_batches, shared_dir):
    """Run 3 4D image steps on 2 ranks with default Axiom preset and verify controller sync."""
    from hypergan.aggregation import _synchronize_optimizers

    config = resolve_config(config_dict)
    replicated = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=1)
    assert replicated.recipe.name == "axiom"
    assert replicated.recipe.optimizer_family == "axiom"
    assert replicated.recipe.effective_critic_formulation == "axiom"
    for full_batch in full_batches:
        local_batch = {"real": full_batch["real"].chunk(world_size)[rank]}
        local_ids = full_batch["_ids"].chunk(world_size)[rank]
        prior_arg = (replicated.prior.z[local_ids], local_ids)
        replicated.update(local_batch, prior_arg)
    replicated._assert_replicas("axiom image step 3")

    # Exercise local-SGD synchronization for AxiomOptimizer + AxiomStepRecord state.
    rec = replicated.opt_d.record
    rec.g_func_gain = 0.4 if rank == 0 else 0.8
    rec.thermal_scale = 0.5 if rank == 0 else 0.9
    replicated.opt_g.ema_step_cos = -0.2 if rank == 0 else -0.6
    replicated.opt_g.log2_damp = -0.2 if rank == 0 else -0.8
    replicated.opt_g.crit_skew_ema = 0.4 if rank == 0 else 1.2
    sketch = torch.zeros(67, dtype=torch.float32)
    sketch[0] = 1.0 if rank == 0 else 0.6
    sketch[1] = 0.0 if rank == 0 else 0.8
    replicated.opt_g._prev_sketch = sketch
    _synchronize_optimizers(replicated, None, world_size)
    replicated._assert_replicas("axiom synced controller state")

    torch.save(
        {
            "digest": replicated._replica_digest(),
            "ema_margin": rec.ema_margin,
            "ema_data_scale": rec.ema_data_scale,
            "ema_push_res": rec.ema_push_res,
            "margin_initialized": rec.margin_initialized,
            "g_func_gain": rec.g_func_gain,
            "thermal_scale": rec.thermal_scale,
            "opt_g_ema_step_cos": replicated.opt_g.ema_step_cos,
            "opt_g_log2_damp": replicated.opt_g.log2_damp,
            "opt_g_crit_skew_ema": replicated.opt_g.crit_skew_ema,
            "opt_g_prev_step_sketch": replicated.opt_g._prev_sketch.tolist(),
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def test_replicated_axiom_image_replica_agreement(tmp_path):
    """2-rank Axiom (default preset) on 4D images maintains exact replica agreement."""
    config = {
        "defaults": "particlegan",
        "training": {
            "steps": 3,
            "batch_size": 8,
            "device": "cpu",
            "seed": 42,
            "initialization": "deterministic_orthogonal",
        },
        "prior": {"kind": "mog", "fixed_sigma": 0.2, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {
                    "source": "linear(1200)\nreshape(3,20,20)\ntanh()",
                    "input_shape": ["B", 4],
                    "output_shape": ["B", 3, 20, 20],
                },
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": "hndl",
                "args": {
                    "source": "conv(8, kernel_size=3, padding=1)\nleaky_relu(0.2)\nadaptive_avg_pool(2)\nflatten()\nlinear(1)",
                    "input_shape": ["B", 3, 20, 20],
                    "output_shape": ["B", 1],
                },
                "inputs": {"x": "candidate"},
            },
        },
    }
    gen = torch.Generator().manual_seed(7)
    full_batches = [
        {
            "real": torch.randn(8, 3, 20, 20, generator=gen),
            "_ids": torch.arange(8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    launch_cpu_workers(
        _axiom_image_worker,
        args=(config, full_batches, str(tmp_path)),
        world_size=2,
        timeout=60,
        collective_timeout=20,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0 == r1
    assert r0["margin_initialized"] is True
    assert r0["ema_data_scale"] > 0.0
    assert abs(r0["g_func_gain"] - 0.6) < 1e-6
    assert abs(r0["thermal_scale"] - 0.7) < 1e-6
    assert abs(r0["opt_g_ema_step_cos"] - (-0.4)) < 1e-6
    assert abs(r0["opt_g_log2_damp"] - (-0.5)) < 1e-6
    assert abs(r0["opt_g_crit_skew_ema"] - 0.8) < 1e-6
    assert len(r0["opt_g_prev_step_sketch"]) == 67

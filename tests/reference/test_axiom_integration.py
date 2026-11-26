"""Integration tests for ParticleGAN Axiom default preset in HyperGAN."""
from __future__ import annotations

from pathlib import Path
import torch

from hypergan.config import load_config
from hypergan.training import ReferenceTrainer, resume, train


def _axiom_2d_model(*, multi_critic: bool = False):
    networks = {
        "generator": {
            "role": "generator",
            "inputs": {"x": "latent"},
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
            "source": "linear(16)\nleaky_relu(0.2)\nlinear(2)",
        },
        "discriminator": {
            "role": "critic",
            "inputs": {"x": "candidate"},
            "input_shape": ["B", 2],
            "output_shape": ["B", 1],
            "source": "linear(16)\nleaky_relu(0.2)\nlinear(1)",
        },
    }
    losses = [
        {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
        {"type": "prior"},
    ]
    if multi_critic:
        networks["discriminator_aux"] = {
            "role": "critic",
            "inputs": {"x": "candidate"},
            "input_shape": ["B", 2],
            "output_shape": ["B", 1],
            "source": "linear(16)\nleaky_relu(0.2)\nlinear(1)",
        }
        losses.insert(
            1,
            {
                "id": "aux",
                "type": "adversarial",
                "critic": "discriminator_aux",
                "weight": 0.5,
                "penalty": 1.0,
            },
        )
    return {
        "format": "hypergan-model/1",
        "name": "test/axiom-2d",
        "defaults": "particlegan",
        "data": {"builtin": "gaussian_grid"},
        "prior": {"kind": "particles", "z_dim": 4, "num_particles": 64, "initialization_seed": 2},
        "networks": networks,
        "losses": losses,
        "train": {
            "steps": 4,
            "batch_size": 16,
            "seed": 7,
            "device": "cpu",
            "initialization": "deterministic_orthogonal",
            "precision": "float32",
        },
    }


def test_axiom_record_binding_and_low_dim_generator_probe():
    cfg = load_config(_axiom_2d_model())
    trainer = ReferenceTrainer(cfg)
    assert trainer.recipe.name == "axiom"
    assert trainer.recipe.optimizer_family == "axiom"
    assert trainer.recipe.effective_critic_formulation == "axiom"

    rec = trainer.opt_d.record
    assert rec is not None
    assert trainer.gan.record is rec
    assert trainer.spread.record is rec
    assert trainer.spread.axiom_record is rec
    assert trainer.opt_g._axiom_record is rec
    assert trainer.opt_g.record is None
    assert trainer.prior._axiom_record is rec

    row, _ = trainer.update()
    assert torch.isfinite(torch.tensor(row["d_loss"]))
    assert torch.isfinite(torch.tensor(row["g_loss"]))
    # Low-D (2D <= 32) activates Axiom's generator verified-trust probe
    assert trainer.opt_g.probe_consumptions >= 1


def test_axiom_multi_critic_thermal_coordinator():
    cfg = load_config(_axiom_2d_model(multi_critic=True))
    trainer = ReferenceTrainer(cfg)
    assert trainer.thermal_coordinator.active is True
    assert trainer.thermal_coordinator.policy == "coupled_max"
    assert set(trainer.thermal_coordinator.critic_records) == {"discriminator", "discriminator_aux"}

    rec_primary = trainer.critic_optimizers["discriminator"].record
    rec_aux = trainer.critic_optimizers["discriminator_aux"].record
    rec_primary.thermal_scale = 0.25
    rec_aux.thermal_scale = 0.80
    synced = trainer.thermal_coordinator.synchronize()
    assert synced == 0.80
    assert rec_primary.thermal_scale == 0.80
    assert rec_aux.thermal_scale == 0.80

    row, _ = trainer.update()
    assert 0.0 < row["critic/thermal_scale"] <= 1.0
    assert "critic/aux/thermal_scale" in row
    assert "critic/aux/raw_thermal_scale" in row
    assert rec_primary.thermal_scale == rec_aux.thermal_scale


def test_axiom_checkpoint_resume_exact_continuation(tmp_path: Path):
    torch.set_num_threads(1)
    cfg = load_config(_axiom_2d_model())

    full_dir = tmp_path / "full"
    res_full = train(cfg, full_dir, steps=4, checkpoint_every=4)
    assert res_full["status"] == "complete"

    split_dir = tmp_path / "split"
    res_part = train(cfg, split_dir, steps=4, checkpoint_every=2, stop_after_steps=2)
    assert res_part["steps"] == 2
    res_resumed = resume(split_dir)
    assert res_resumed["status"] == "complete"
    assert res_resumed["steps"] == 4

    from hypergan.checkpoints import read_checkpoint
    _, _, full_state = read_checkpoint(full_dir)
    _, _, split_state = read_checkpoint(split_dir)
    for k, v in full_state["graph"].items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, split_state["graph"][k], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full_state["prior"]["z"], split_state["prior"]["z"], rtol=1e-6, atol=1e-6)

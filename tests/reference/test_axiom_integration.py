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


def test_unified_v15_record_binding_and_step():
    raw = _axiom_2d_model()
    raw["defaults"] = "particlegan:unified_v15"
    cfg = load_config(raw)
    trainer = ReferenceTrainer(cfg)
    assert trainer.recipe.name == "unified_v15"
    assert trainer.recipe.effective_critic_formulation == "unified_v15"
    rec = trainer.opt_d.record
    assert rec is not None
    assert trainer.gan.record is rec
    assert trainer.spread.unified_v15_record is rec
    assert trainer.opt_g._unified_v15_record is rec
    assert trainer.prior._unified_v15_record is rec
    row, _ = trainer.update()
    assert torch.isfinite(torch.tensor(row["d_loss"]))
    assert torch.isfinite(torch.tensor(row["g_loss"]))


class _SyntheticConvData:
    resume_stateless = True

    def __init__(self, channels: int = 3, height: int = 16, width: int = 16):
        self.channels = channels
        self.height = height
        self.width = width

    def __call__(self, batch_size: int, *, generator):
        return {
            "real": torch.randn(
                batch_size, self.channels, self.height, self.width, generator=generator
            )
        }


def _axiom_conv_model(*, gen_bn: bool = True, disc_bn: bool = True):
    g_bn = "batch_norm()\n" if gen_bn else ""
    d_bn = "batch_norm()\n" if disc_bn else ""
    return {
        "format": "hypergan-model/1",
        "name": "test/axiom-conv-bn",
        "defaults": "particlegan",
        "data": {
            "factory": f"{__name__}:_SyntheticConvData",
            "args": {"channels": 3, "height": 16, "width": 16},
        },
        "prior": {"kind": "particles", "z_dim": 16, "num_particles": 64, "initialization_seed": 2},
        "networks": {
            "generator": {
                "role": "generator",
                "inputs": {"z": "latent"},
                "input_shape": ["B", 16],
                "output_shape": ["B", 3, 16, 16],
                "source": (
                    "linear(256, bias=False)\nreshape(16, 4, 4)\n"
                    f"{g_bn}relu()\n"
                    "deconv(8, kernel_size=4, stride=2, padding=1, bias=False)\n"
                    f"{g_bn}relu()\n"
                    "deconv(3, kernel_size=4, stride=2, padding=1)\ntanh()"
                ),
            },
            "discriminator": {
                "role": "critic",
                "inputs": {"x": "candidate"},
                "input_shape": ["B", 3, 16, 16],
                "output_shape": ["B", 1],
                "source": (
                    "conv(8, kernel_size=4, stride=2, padding=1)\nleaky_relu(0.2)\n"
                    "conv(16, kernel_size=4, stride=2, padding=1, bias=False)\n"
                    f"{d_bn}leaky_relu(0.2)\nflatten()\nlinear(1)"
                ),
            },
        },
        "losses": [
            {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
            {"type": "prior"},
        ],
        "train": {
            "steps": 4,
            "batch_size": 8,
            "seed": 11,
            "device": "cpu",
            "initialization": "deterministic_orthogonal",
            "precision": "float32",
        },
    }


def test_axiom_bn_layout_scoping_and_g_step_bn_tracking(tmp_path: Path):
    torch.set_num_threads(1)

    # Active BN only in critic must not leak BN layout ids into a BN-free generator optimizer.
    mixed_cfg = load_config(_axiom_conv_model(gen_bn=False, disc_bn=True))
    mixed_trainer = ReferenceTrainer(mixed_cfg)
    assert not mixed_trainer.opt_g._bn_net_ids
    assert mixed_trainer.opt_d._bn_net_ids

    # Active BN in both G and D: both bind BN ids, and G-step forwards do not mutate critic BN tracking.
    bn_cfg = load_config(_axiom_conv_model(gen_bn=True, disc_bn=True))
    bn_trainer = ReferenceTrainer(bn_cfg)
    assert bn_trainer.opt_g._bn_net_ids
    assert bn_trainer.opt_d._bn_net_ids

    disc_bns = [
        m for m in bn_trainer.graph.models["discriminator"].modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert disc_bns
    row, _ = bn_trainer.update()
    assert torch.isfinite(torch.tensor(row["d_loss"]))
    assert torch.isfinite(torch.tensor(row["g_loss"]))
    # D step runs real + fake forwards (=2 tracked batches); G step keeps track_running_stats=False.
    for m in disc_bns:
        assert int(m.num_batches_tracked.item()) == 2
        assert m.track_running_stats is True

    # Exact checkpoint/resume continuation with Axiom BN momentum buffers (m_u).
    full_dir = tmp_path / "bn_full"
    assert train(bn_cfg, full_dir, steps=4, checkpoint_every=4)["status"] == "complete"
    split_dir = tmp_path / "bn_split"
    assert train(bn_cfg, split_dir, steps=4, checkpoint_every=2, stop_after_steps=2)["steps"] == 2
    assert resume(split_dir)["status"] == "complete"
    from hypergan.checkpoints import read_checkpoint
    _, _, full_state = read_checkpoint(full_dir)
    _, _, split_state = read_checkpoint(split_dir)
    for k, v in full_state["graph"].items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, split_state["graph"][k], rtol=1e-6, atol=1e-6)

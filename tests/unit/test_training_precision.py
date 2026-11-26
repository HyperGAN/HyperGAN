"""Unit tests for training_precision config, schema, manifest, resume, CPU parity, and HNDL autocast."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from hndl.errors import HNDLError

import hypergan as hg
from hypergan.config import (
    DEFAULT,
    VALID_TRAINING_PRECISIONS,
    config_values,
    fingerprint,
    load_config,
    numerical_values,
    resolve_config,
    resume_compatible,
)
from hypergan.model_file import TRAINING_FIELDS, lower
from hypergan.particlegan_integration import PARTICLEGAN_DEFAULT_FIELDS, particlegan_recipe
from hypergan.recipes import construct
from hypergan.schema_info import get_schema
from hypergan.training import ReferenceTrainer, resume, train


def _minimal_model_file(*, defaults="particlegan", precision=None, device="cpu"):
    train_tbl = {
        "steps": 4,
        "batch_size": 16,
        "seed": 7,
        "device": device,
        "initialization": "deterministic_orthogonal",
    }
    if precision is not None:
        train_tbl["precision"] = precision
    return {
        "format": "hypergan-model/1",
        "name": "test/precision-model",
        "defaults": defaults,
        "data": {"builtin": "gaussian_grid"},
        "prior": {"kind": "particles", "z_dim": 4, "num_particles": 64, "initialization_seed": 2},
        "networks": {
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
        },
        "losses": [
            {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
            {"type": "prior"},
        ],
        "train": train_tbl,
    }


def test_precision_schema_and_model_file_plumbing():
    assert "precision" in TRAINING_FIELDS
    assert ("training", "precision") in PARTICLEGAN_DEFAULT_FIELDS
    assert PARTICLEGAN_DEFAULT_FIELDS[("training", "precision")] == "training_precision"
    assert DEFAULT["training"]["precision"] == "float32"

    train_schema = get_schema("train")["fields"]
    assert "precision" in train_schema
    assert tuple(train_schema["precision"]["choices"]) == VALID_TRAINING_PRECISIONS
    assert train_schema["precision"]["default"] == "float32"


def test_precision_defaults_and_explicit_overrides():
    # defaults="hypergan" -> float32
    cfg_hg = load_config(_minimal_model_file(defaults="hypergan"))
    assert cfg_hg["training"]["precision"] == "float32"
    assert particlegan_recipe(cfg_hg).training_precision == "float32"

    # defaults="particlegan" (axiom) -> bf16_channels_last from preset
    cfg_pg = load_config(_minimal_model_file(defaults="particlegan"))
    assert cfg_pg["training"]["precision"] == "bf16_channels_last"
    assert cfg_pg["recipe"]["training_precision"] == "bf16_channels_last"
    assert particlegan_recipe(cfg_pg).training_precision == "bf16_channels_last"

    # Explicit train.precision="float32" overrides particlegan preset
    cfg_pg_fp32 = load_config(_minimal_model_file(defaults="particlegan", precision="float32"))
    assert cfg_pg_fp32["training"]["precision"] == "float32"
    assert cfg_pg_fp32["recipe"]["training_precision"] == "float32"
    assert particlegan_recipe(cfg_pg_fp32).training_precision == "float32"

    # Explicit train.precision="bf16_channels_last" overrides defaults="hypergan"
    cfg_hg_bf16 = load_config(_minimal_model_file(defaults="hypergan", precision="bf16_channels_last"))
    assert cfg_hg_bf16["training"]["precision"] == "bf16_channels_last"
    assert particlegan_recipe(cfg_hg_bf16).training_precision == "bf16_channels_last"

    # Preset whose upstream recipe uses float32 (ka2) resolves to float32
    cfg_ka2 = load_config(_minimal_model_file(defaults="particlegan:ka2"))
    assert cfg_ka2["training"]["precision"] == "float32"
    assert cfg_ka2["recipe"]["training_precision"] == "float32"

    # hg.override supports both train.precision and training.precision
    model = hg.from_dict(_minimal_model_file(defaults="particlegan"))
    overridden_a = hg.resolve(hg.override(model, {"train.precision": "float32"}))
    overridden_b = hg.resolve(hg.override(model, {"training.precision": "float32"}))
    assert overridden_a["training"]["precision"] == "float32"
    assert overridden_b["training"]["precision"] == "float32"

    # Lowering and lifting preserve explicit precision
    lowered = lower(_minimal_model_file(defaults="hypergan", precision="bf16_channels_last"))
    assert lowered["training"]["precision"] == "bf16_channels_last"
    lifted = hg.lift(cfg_pg_fp32)
    assert lifted.spec["train"]["precision"] == "float32"


@pytest.mark.parametrize("bad_precision", ["fp16", "bfloat16", "float16", "", None, 16, True])
def test_invalid_training_precision_rejected(bad_precision):
    with pytest.raises(ValueError, match="training.precision"):
        resolve_config({"training": {"precision": bad_precision}})


def test_precision_fingerprint_and_strict_resume_compatibility():
    fp32_pg = load_config(_minimal_model_file(defaults="particlegan", precision="float32"))
    bf16_pg = load_config(_minimal_model_file(defaults="particlegan", precision="bf16_channels_last"))
    assert fingerprint(fp32_pg) != fingerprint(bf16_pg)
    assert not resume_compatible(fp32_pg, bf16_pg)
    assert not resume_compatible(bf16_pg, fp32_pg)
    assert not resume_compatible(
        fp32_pg, bf16_pg, allow_device_change=True, allow_batch_size_change=True
    )
    assert "training.precision" in resume_compatible(bf16_pg, fp32_pg).differences

    fp32_hg = load_config(_minimal_model_file(defaults="hypergan", precision="float32"))
    bf16_hg = load_config(_minimal_model_file(defaults="hypergan", precision="bf16_channels_last"))
    assert "precision" not in numerical_values(fp32_hg)["training"]
    assert numerical_values(bf16_hg)["training"]["precision"] == "bf16_channels_last"
    assert config_values(fp32_hg)["training"]["precision"] == "float32"
    assert config_values(bf16_hg)["training"]["precision"] == "bf16_channels_last"
    assert fingerprint(fp32_hg) != fingerprint(bf16_hg)
    assert not resume_compatible(bf16_hg, fp32_hg)
    assert not resume_compatible(fp32_hg, bf16_hg)
    assert resume_compatible(bf16_hg, config_values(bf16_hg))
    assert resolve_config(config_values(bf16_pg)) == bf16_pg


def test_manifest_records_resolved_precision_and_rejects_precision_change_on_resume(tmp_path: Path):
    run_dir = tmp_path / "run_bf16"
    cfg_bf16 = load_config(_minimal_model_file(defaults="particlegan", precision="bf16_channels_last", device="cpu"))
    cfg_fp32 = load_config(_minimal_model_file(defaults="particlegan", precision="float32", device="cpu"))

    res = train(cfg_bf16, run_dir, steps=2, checkpoint_every=1, stop_after_steps=1)
    assert res["steps"] == 1

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_precision"] == "bf16_channels_last"
    assert manifest["config"]["training"]["precision"] == "bf16_channels_last"
    assert manifest["config"]["recipe"]["training_precision"] == "bf16_channels_last"

    # Resuming with changed precision must fail strictly
    with pytest.raises(ValueError, match="Resume configuration differs"):
        resume(run_dir, config_path=cfg_fp32, steps=2)

    # Resuming with matching precision succeeds and preserves recorded precision
    resumed = resume(run_dir, config_path=cfg_bf16, steps=2)
    assert resumed["status"] == "complete"
    assert resumed["steps"] == 2
    manifest_after = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_after["training_precision"] == "bf16_channels_last"
    assert manifest_after["config"]["training"]["precision"] == "bf16_channels_last"


def test_cpu_bf16_channels_last_matches_cpu_float32_bit_identically():
    torch.set_num_threads(1)
    cfg_fp32 = load_config(_minimal_model_file(defaults="particlegan", precision="float32", device="cpu"))
    cfg_bf16 = load_config(_minimal_model_file(defaults="particlegan", precision="bf16_channels_last", device="cpu"))

    t_fp32 = ReferenceTrainer(cfg_fp32)
    t_bf16 = ReferenceTrainer(cfg_bf16)

    for _ in range(3):
        row_a, _ = t_fp32.update()
        row_b, _ = t_bf16.update()
        assert row_a["d_loss"] == row_b["d_loss"]
        assert row_a["g_loss"] == row_b["g_loss"]

    for (k1, v1), (k2, v2) in zip(t_fp32.graph.state_dict().items(), t_bf16.graph.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)
        assert v2.dtype == torch.float32


def test_hndl_network_autocast_validation_preserves_shape_and_dtype_contracts():
    spec = {
        "factory": "hndl",
        "args": {
            "source": "linear(64)\nreshape(4, 4, 4)\nconv(32, kernel_size=3, padding=1)\nsilu()\nflatten()\nlinear(1)",
            "input_shape": ["B", 8],
            "output_shape": ["B", 1],
        },
        "inputs": {"x": "latent"},
    }
    net = construct(spec)
    x = torch.randn(2, 8)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out1 = net(x)
        out2 = net(x)
    assert out1.shape == (2, 1)
    assert out2.shape == (2, 1)
    assert torch.equal(out1, out2)

    # deepcopy must keep _AutocastAwareCheck bound to the new copy
    cloned = copy.deepcopy(net)
    assert cloned.network._check._module is cloned.network
    assert cloned.network._check._module is not net.network
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert cloned(x).shape == (2, 1)

    # Without autocast, wrong dtype must still raise HNDLError
    with pytest.raises(HNDLError):
        net(x.to(torch.bfloat16))

    # Under bf16 autocast, wrong shape and unrelated float64 dtype must still raise HNDLError
    with pytest.raises(HNDLError):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            net(torch.randn(2, 9))
    with pytest.raises(HNDLError):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            net(x.double())


class _SyntheticImagesCPU:
    resume_stateless = True

    def __init__(self, channels=3, size=20):
        self.channels = channels
        self.size = size

    def __call__(self, batch_size, *, generator):
        return {
            "real": torch.randn(batch_size, self.channels, self.size, self.size, generator=generator)
        }


def test_uses_simplicial_manifold_loss_delegates_to_particlegan_and_handles_fallback(monkeypatch):
    from types import SimpleNamespace
    import particlegan.training as pg_training
    from hypergan.objective_program import _uses_simplicial_manifold_loss

    prior = SimpleNamespace(z=torch.nn.Parameter(torch.zeros(8, 4)))
    v11 = SimpleNamespace(effective_critic_formulation="unified_v11", rhbi_simplex_weight=1.0)
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 1024), prior=prior, is_discrete_prior=True, conditioned=False) is True
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 1025), prior=prior, is_discrete_prior=True, conditioned=False) is False
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 3, 16, 16), prior=prior, is_discrete_prior=True, conditioned=False) is True
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 3, 32, 32), prior=prior, is_discrete_prior=True, conditioned=False) is False
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 16), prior=prior, is_discrete_prior=True, conditioned=True) is False

    # Future/custom recipe cap via recipe.simplicial_max_dim delegates directly to ParticleGAN
    future = SimpleNamespace(effective_critic_formulation="future_v99", rhbi_simplex_weight=1.0, simplicial_max_dim=64)
    assert _uses_simplicial_manifold_loss(future, torch.zeros(4, 64), prior=prior, is_discrete_prior=True, conditioned=False) is True
    assert _uses_simplicial_manifold_loss(future, torch.zeros(4, 65), prior=prior, is_discrete_prior=True, conditioned=False) is False

    # Graceful fallback when running on older ParticleGAN lacking uses_simplicial_manifold_loss
    monkeypatch.setattr(pg_training, "uses_simplicial_manifold_loss", None, raising=False)
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 3, 16, 16), prior=prior, is_discrete_prior=True, conditioned=False) is True
    assert _uses_simplicial_manifold_loss(v11, torch.zeros(4, 3, 32, 32), prior=prior, is_discrete_prior=True, conditioned=False) is False


def test_highres_unified_skip_skips_simplicial_loss_and_preserves_g_probe_cache():
    # High-dim (> 1024 per sample: 3x20x20 = 1200) must skip simplicial_manifold_loss
    # and never overwrite record._g_probe_cache via an extra generator forward.
    def _highres_cfg(defaults):
        return load_config(
            {
                "format": "hypergan-model/1",
                "name": "test/highres-skip-cpu",
                "defaults": defaults,
                "data": {"factory": f"{__name__}:_SyntheticImagesCPU", "args": {"channels": 3, "size": 20}},
                "prior": {"kind": "particles", "z_dim": 8, "num_particles": 16, "initialization_seed": 2},
                "networks": {
                    "generator": {
                        "role": "generator",
                        "inputs": {"x": "latent"},
                        "input_shape": ["B", 8],
                        "output_shape": ["B", 3, 20, 20],
                        "source": "linear(1200)\nreshape(3, 20, 20)\ntanh()",
                    },
                    "discriminator": {
                        "role": "critic",
                        "inputs": {"x": "candidate"},
                        "input_shape": ["B", 3, 20, 20],
                        "output_shape": ["B", 1],
                        "source": "flatten()\nlinear(16)\nsilu()\nlinear(1)",
                    },
                },
                "losses": [
                    {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
                    {"type": "prior"},
                ],
                "train": {"steps": 2, "batch_size": 4, "seed": 5, "device": "cpu"},
            }
        )

    # 1. unified_v12: skips simplicial_manifold_loss and preserves the single _g_probe_cache write
    trainer_v12 = ReferenceTrainer(_highres_cfg("particlegan:unified_v12"))
    t1c_v12 = getattr(trainer_v12.gan, "t1c", None) or getattr(trainer_v12.gan, "rhbi", None)
    assert t1c_v12 is not None
    v12_simplex_calls = []
    v12_bind_calls = []
    orig_simplex = t1c_v12.simplicial_manifold_loss
    orig_bind = t1c_v12.bind_discrete_prior_step
    t1c_v12.simplicial_manifold_loss = lambda *a, **kw: (v12_simplex_calls.append(1), orig_simplex(*a, **kw))[1]
    t1c_v12.bind_discrete_prior_step = lambda is_disc, simplicial_active=False: (
        v12_bind_calls.append((is_disc, simplicial_active)),
        orig_bind(is_disc, simplicial_active=simplicial_active),
    )[1]

    orig_probe_hook = trainer_v12.graph._probe_hook
    probe_writes = []

    def _spy_probe_hook(*args, **kwargs):
        orig_probe_hook(*args, **kwargs)
        cache = getattr(trainer_v12.opt_d.record, "_g_probe_cache", None)
        if cache is not None:
            probe_writes.append(id(cache))

    trainer_v12.graph._probe_hook = _spy_probe_hook
    trainer_v12.update()

    assert v12_simplex_calls == [], "simplicial_manifold_loss must be skipped when per-sample numel > 1024"
    assert (True, False) in v12_bind_calls
    assert all(not active for _, active in v12_bind_calls)
    # In shared phase_draws, generator runs once per step (never re-run by simplex loss)
    assert len(probe_writes) == 1

    # 2. Default particlegan (axiom): skips simplicial_manifold_loss on dim > 1024 and keeps high-D probe off (dim > 32)
    trainer_hi = ReferenceTrainer(_highres_cfg("particlegan"))
    t1c = getattr(trainer_hi.gan, "t1c", None) or getattr(trainer_hi.gan, "rhbi", None)
    assert t1c is not None
    simplex_calls = []
    bind_calls = []
    orig_ax_simplex = t1c.simplicial_manifold_loss
    orig_ax_bind = t1c.bind_discrete_prior_step
    t1c.simplicial_manifold_loss = lambda *a, **kw: (simplex_calls.append(1), orig_ax_simplex(*a, **kw))[1]
    t1c.bind_discrete_prior_step = lambda is_disc, simplicial_active=False: (
        bind_calls.append((is_disc, simplicial_active)),
        orig_ax_bind(is_disc, simplicial_active=simplicial_active),
    )[1]
    trainer_hi.update()
    assert simplex_calls == [], "axiom must skip simplicial_manifold_loss when per-sample numel > 1024"
    assert (True, False) in bind_calls
    assert all(not active for _, active in bind_calls)
    assert getattr(trainer_hi.opt_d.record, "_g_probe_cache", None) is None

    # 3. Low-dim (<= 1024: 2D gaussian_grid) must still run simplicial_manifold_loss cleanly on default preset
    cfg_lo = load_config(_minimal_model_file(defaults="particlegan", precision="float32", device="cpu"))
    trainer_lo = ReferenceTrainer(cfg_lo)
    t1c_lo = getattr(trainer_lo.gan, "t1c", None) or getattr(trainer_lo.gan, "rhbi", None)
    assert t1c_lo is not None
    lo_calls = []
    orig_lo = t1c_lo.simplicial_manifold_loss
    t1c_lo.simplicial_manifold_loss = lambda *a, **kw: (lo_calls.append(1), orig_lo(*a, **kw))[1]
    trainer_lo.update()
    assert len(lo_calls) == 1


def test_hndl_autocast_check_degrades_gracefully_for_future_hndl_versions():
    from hypergan.hndl_networks import _AutocastAwareCheck, _install_autocast_check

    class _NoCheck(torch.nn.Module):
        pass

    class _ChangedSig(torch.nn.Module):
        def _check(self, tensor):
            return tensor

    class _CompatibleExtendedSig(torch.nn.Module):
        def _check(self, value, expected, dtype, port, batch, strict=True):
            return (value.dtype, dtype, strict)

    m_none = _install_autocast_check(_NoCheck())
    assert not hasattr(m_none, "_check")

    m_changed = _install_autocast_check(_ChangedSig())
    assert not isinstance(m_changed._check, _AutocastAwareCheck)

    m_ext = _install_autocast_check(_CompatibleExtendedSig())
    assert isinstance(m_ext._check, _AutocastAwareCheck)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        val_dt, checked_dt, strict = m_ext._check(
            torch.zeros(2, dtype=torch.bfloat16), None, torch.float32, "x", 2, strict=False
        )
    assert (val_dt, checked_dt, strict) == (torch.bfloat16, torch.bfloat16, False)


def test_precision_plumbing_is_general_across_non_particlegan_and_named_presets(monkeypatch):
    # Non-ParticleGAN config (defaults='hypergan', no recipe snapshot) supports explicit bf16_channels_last
    cfg_hg = load_config(_minimal_model_file(defaults="hypergan", precision="bf16_channels_last", device="cpu"))
    assert cfg_hg["training"]["precision"] == "bf16_channels_last"
    assert cfg_hg["recipe"] == {}
    trainer_hg = ReferenceTrainer(cfg_hg)
    assert trainer_hg.recipe.training_precision == "bf16_channels_last"
    row, _ = trainer_hg.update()
    assert all(torch.isfinite(torch.tensor(row[k])) for k in ("d_loss", "g_loss"))

    # Named ParticleGAN presets dynamically inherit training_precision and allow float32 override
    for preset in ("unified_v11", "bispf_optimized"):
        cfg_default = load_config(_minimal_model_file(defaults=f"particlegan:{preset}", device="cpu"))
        assert cfg_default["training"]["precision"] == "bf16_channels_last"
        cfg_override = load_config(
            _minimal_model_file(defaults=f"particlegan:{preset}", precision="float32", device="cpu")
        )
        assert cfg_override["training"]["precision"] == "float32"

    # Older ParticleGAN Recipe lacking training_precision field stays compatible for default float32
    import hypergan.particlegan_integration as pgi

    captured = {}
    orig_restore = pgi.restore_particlegan_recipe

    def _fake_restore(fields):
        captured["fields"] = dict(fields)
        return orig_restore({k: v for k, v in fields.items() if k != "training_precision"})

    monkeypatch.setattr(pgi, "_recipe_supports_field", lambda name: False)
    monkeypatch.setattr(pgi, "restore_particlegan_recipe", _fake_restore)
    cfg_fp32 = load_config(_minimal_model_file(defaults="hypergan", precision="float32", device="cpu"))
    pgi.particlegan_recipe(cfg_fp32)
    assert "training_precision" not in captured["fields"]

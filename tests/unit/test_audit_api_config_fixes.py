"""Regression tests for forensic audit fixes across artifacts, API, CLI, config, queue, controller, and signals."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import hypergan as hg
from hypergan import api, cli
from hypergan.config import config_values, load_config, resolve_config
from hypergan.model_file import ModelFileError, lower, resolve_inheritance
from hypergan.provenance import snapshot_config_sources
from hypergan.queue import (
    _GRACEFUL_STOP_WAIT_SECONDS,
    check_device_availability,
    install_service,
    queue_add,
    queue_run,
    queue_stop,
)
from hypergan.run_signals import GracefulStop
from hypergan.schema_info import get_schema
from hypergan.workspace import load_workspace


def _make_trainer(cfg, gen, prior, *, step=1):
    import torch
    return SimpleNamespace(
        config=cfg,
        step=step,
        ema_generator=gen,
        prior=prior,
        ema_prior=prior,
        ema_graph=SimpleNamespace(models={"generator": gen}),
        device=torch.device("cpu"),
    )


# ==============================================================================
# 1. Artifacts & LoadedGenerator
# ==============================================================================

def test_loaded_generator_mog_standardize_true_and_false(tmp_path):
    torch = pytest.importorskip("torch")
    from particlegan import MoGParticlePrior
    from hypergan.artifacts import LoadedGenerator, save_bundle

    class _IdentityGen(torch.nn.Module):
        def forward(self, input):
            return input

    for standardize in (True, False):
        prior = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.05, standardize=standardize)
        with torch.no_grad():
            # Shift raw z far from zero mean / unit std so unstandardized vs standardized is unmistakable
            prior.z.copy_(torch.randn(32, 4) * 9.0 + 50.0)
        cfg = resolve_config({
            "prior": {"kind": "mog", "args": {"num_particles": 32, "z_dim": 4, "sigma": 0.05, "standardize": standardize}},
            "components": {
                "generator": {"factory": "identity", "args": {}, "inputs": {"input": "latent"}},
                "discriminator": {"factory": "linear", "args": {"in_features": 4, "out_features": 1}, "inputs": {"input": "candidate"}},
            },
        })
        trainer = _make_trainer(cfg, _IdentityGen(), prior, step=5)
        out_dir = tmp_path / f"mog_std_{standardize}"
        out_dir.mkdir()
        save_bundle(out_dir, trainer, {"real": torch.zeros(2, 4)})
        bundle_path = out_dir / "model.pt"
        lg = LoadedGenerator(bundle_path, device="cpu")

        assert torch.allclose(lg.means(), prior.means(), atol=1e-5)
        if standardize:
            assert lg.means().mean(dim=0).abs().max().item() < 1e-4
            assert (lg.means().std(dim=0) - 1.0).abs().max().item() < 1e-2
        else:
            assert lg.means().mean().item() > 30.0

        g1 = torch.Generator(device="cpu").manual_seed(99)
        g2 = torch.Generator(device="cpu").manual_seed(99)
        z_lg = lg.sample_latents(16, generator=g1)
        z_prior, _ = prior.sample(16, generator=g2)
        assert torch.allclose(z_lg, z_prior, atol=1e-5)


def test_loaded_generator_gaussian_init_std_and_fresh_gaussian_code_law(tmp_path):
    torch = pytest.importorskip("torch")
    from particlegan import GaussianPrior, ParticlePrior
    from hypergan.artifacts import LoadedGenerator, save_bundle

    class _IdentityGen(torch.nn.Module):
        def forward(self, input):
            return input

    # GaussianPrior with non-unit init_std
    g_prior = GaussianPrior(z_dim=8, init_std=3.5)
    g_cfg = resolve_config({
        "prior": {"kind": "gaussian", "args": {"z_dim": 8, "init_std": 3.5}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {"factory": "identity", "args": {}, "inputs": {"input": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 8, "out_features": 1}, "inputs": {"input": "candidate"}},
        },
    })
    trainer_g = _make_trainer(g_cfg, _IdentityGen(), g_prior)
    (tmp_path / "gauss").mkdir()
    save_bundle(tmp_path / "gauss", trainer_g, {})
    b_g = tmp_path / "gauss" / "model.pt"
    lg_g = LoadedGenerator(b_g, device="cpu")
    gen1 = torch.Generator(device="cpu").manual_seed(7)
    gen2 = torch.Generator(device="cpu").manual_seed(7)
    z_lg = lg_g.sample_latents(64, generator=gen1)
    z_pr, _ = g_prior.sample(64, generator=gen2)
    assert torch.allclose(z_lg, z_pr, atol=1e-6)
    assert z_lg.std().item() > 2.5

    # ParticlePrior with code_law="fresh_gaussian"
    p_prior = ParticlePrior(num_particles=16, z_dim=4, init_std=1.0)
    with torch.no_grad():
        p_prior.z.fill_(100.0)
    p_cfg = resolve_config({
        "recipe": {"prior_code_law": "fresh_gaussian"},
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {"factory": "identity", "args": {}, "inputs": {"input": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 4, "out_features": 1}, "inputs": {"input": "candidate"}},
        },
    })
    trainer_p = _make_trainer(p_cfg, _IdentityGen(), p_prior)
    (tmp_path / "fresh").mkdir()
    save_bundle(tmp_path / "fresh", trainer_p, {})
    b_p = tmp_path / "fresh" / "model.pt"
    lg_p = LoadedGenerator(b_p, device="cpu")
    z_fresh = lg_p.sample_latents(64, seed=11)
    assert z_fresh.abs().mean().item() < 5.0  # Not reading frozen z == 100.0


def test_loaded_generator_device_resolution_before_load_and_apply(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from particlegan import GaussianPrior
    import hypergan.recipes as recipes_mod
    from hypergan.artifacts import LoadedGenerator, save_bundle
    from hypergan.recipes import linear as hndl_linear

    cfg = resolve_config({
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    gen = hndl_linear(4, 2)
    trainer = _make_trainer(cfg, gen, GaussianPrior(z_dim=4))
    (tmp_path / "dev").mkdir()
    save_bundle(tmp_path / "dev", trainer, {})
    bundle_path = tmp_path / "dev" / "model.pt"

    call_order = []
    orig_exec_device = recipes_mod.execution_device
    orig_load = torch.load

    def tracked_exec_device(dev, **kwargs):
        call_order.append(("execution_device", dev, kwargs))
        return orig_exec_device(dev, **kwargs)

    def tracked_load(*args, **kwargs):
        call_order.append(("torch.load", kwargs.get("map_location")))
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(recipes_mod, "execution_device", tracked_exec_device)
    monkeypatch.setattr(torch, "load", tracked_load)

    lg = LoadedGenerator(bundle_path, device="cpu")
    assert call_order[0][0] == "execution_device"
    assert call_order[1][0] == "torch.load"
    assert call_order[1][1] == torch.device("cpu")

    # _apply updates self.device when module moves (even via .cpu() / .to(tensor))
    lg.device = torch.device("meta")
    lg.cpu()
    assert lg.device == torch.device("cpu")


def test_bundle_privacy_default_and_non_tensor_redaction(tmp_path):
    torch = pytest.importorskip("torch")
    from particlegan import GaussianPrior
    from hypergan.artifacts import bundle_state, save_bundle

    cfg = resolve_config({
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2},
                "inputs": {"x": "latent", "cond": "batch.real", "secret": "batch.user_secret", "meta": "batch.meta"},
            },
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    gen = torch.nn.Linear(4, 2)
    trainer = _make_trainer(cfg, gen, GaussianPrior(z_dim=4))
    batch = {
        "real": torch.ones(2, 2),
        "user_secret": "sensitive-patient-id-42",
        "meta": {"nested_id": 98765},
    }

    default_state = bundle_state(trainer, batch)
    assert "last_batch" not in default_state
    assert default_state["example_inputs"]["real"] == {"shape": [2, 2], "dtype": "torch.float32"}
    assert default_state["example_inputs"]["user_secret"] == {"type": "str", "redacted": True}
    assert default_state["example_inputs"]["meta"]["nested_id"] == {"type": "int", "redacted": True}

    (tmp_path / "priv").mkdir()
    save_bundle(tmp_path / "priv", trainer, batch)
    saved_path = tmp_path / "priv" / "model.pt"
    loaded = torch.load(saved_path, map_location="cpu", weights_only=True)
    assert loaded["example_inputs"]["user_secret"] == {"type": "str", "redacted": True}
    assert "sensitive-patient-id-42" not in repr(loaded)

    explicit_state = bundle_state(trainer, batch, include_example_inputs=True)
    assert torch.equal(explicit_state["example_inputs"]["real"], batch["real"])
    assert explicit_state["example_inputs"]["user_secret"] == "sensitive-patient-id-42"


def test_pt2_export_raises_on_failure_and_safetensors_preserves_buffers_and_extra_state(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from particlegan import MoGParticlePrior
    from hypergan.artifacts import export_generator, save_bundle

    class _BufGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 2)
            self.register_buffer("non_persistent_scale", torch.tensor([3.25, 4.5]), persistent=False)

        def forward(self, x, **kwargs):
            return self.lin(x) * self.non_persistent_scale

    cfg = resolve_config({
        "prior": {"kind": "mog", "args": {"num_particles": 16, "z_dim": 4, "sigma": 0.1, "standardize": True}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2},
                "inputs": {"x": "latent", "tag": "batch.tag"},
            },
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    gen = _BufGen()
    prior = MoGParticlePrior(num_particles=16, z_dim=4, sigma=0.1, standardize=True)
    trainer = _make_trainer(cfg, gen, prior, step=3)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    batch = {"real": torch.zeros(2, 2), "tag": "secret"}
    save_bundle(run_dir, trainer, batch)
    from hypergan.artifacts import bundle_state as _bundle_state
    state_dict_bundle = _bundle_state(trainer, batch)

    monkeypatch.setattr("hypergan.checkpoints.resolve_checkpoint", lambda *a, **kw: run_dir)
    monkeypatch.setattr("hypergan.checkpoints.extract_ema_bundle", lambda *a, **kw: state_dict_bundle)
    monkeypatch.setattr("hypergan.artifacts.LoadedGenerator", lambda path, device="cpu": SimpleNamespace(
        model=gen,
        z_dim=4,
        device=torch.device("cpu"),
        config=cfg,
        sample_latents=lambda n, seed=None: torch.zeros(n, 4),
    ))
    # PT2 must raise cleanly instead of silently writing a TorchScript archive
    uncond_cfg = resolve_config({
        "prior": {"kind": "mog", "args": {"num_particles": 16, "z_dim": 4, "sigma": 0.1, "standardize": True}},
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    uncond_bundle = _bundle_state(_make_trainer(uncond_cfg, gen, prior, step=3), {})
    monkeypatch.setattr("hypergan.checkpoints.extract_ema_bundle", lambda *a, **kw: uncond_bundle)
    monkeypatch.setattr(torch.export, "export", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("dynamic control flow")))
    pt2_out = tmp_path / "bad.pt2"
    with pytest.raises(RuntimeError, match="PT2 export"):
        export_generator(run_dir, pt2_out, format="pt2")
    assert not pt2_out.exists()

    # SafeTensors preserves non-persistent model_buffers and prior_extra_state
    monkeypatch.setattr("hypergan.checkpoints.extract_ema_bundle", lambda *a, **kw: state_dict_bundle)
    safetensors = pytest.importorskip("safetensors")
    st_out = tmp_path / "gen.safetensors"
    export_generator(run_dir, st_out, format="safetensors")
    with safetensors.safe_open(str(st_out), framework="pt", device="cpu") as f:
        meta = f.metadata()
        keys = set(f.keys())
        assert "generator.non_persistent_scale" in keys
        assert torch.allclose(f.get_tensor("generator.non_persistent_scale"), torch.tensor([3.25, 4.5]))
        assert "non_persistent_scale" in json.loads(meta["model_buffers"])["generator"]
        assert json.loads(meta["prior_extra_state"])["standardize"] is True
        assert json.loads(meta["example_inputs"])["tag"] == {"type": "str", "redacted": True}


def test_sha256_sidecar_no_model_fallback_and_rejects_empty_truncated(tmp_path):
    torch = pytest.importorskip("torch")
    from particlegan import GaussianPrior
    from hypergan.artifacts import _verify_sha256_file, sample, save_bundle
    from hypergan.recipes import linear as hndl_linear

    cfg = resolve_config({
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    gen = hndl_linear(4, 2)
    trainer = _make_trainer(cfg, gen, GaussianPrior(z_dim=4))
    bundle_dir = tmp_path / "b"
    bundle_dir.mkdir()
    save_bundle(bundle_dir, trainer, {})
    model_pt = bundle_dir / "model.pt"
    custom_pt = bundle_dir / "custom.pt"
    custom_pt.write_bytes(model_pt.read_bytes())

    # Corrupt unrelated model.sha256: sampling custom.pt must NOT fall back to model.sha256
    (bundle_dir / "model.sha256").write_text("0" * 64 + "  model.pt\n", encoding="utf-8")
    res_path = sample(custom_pt, count=4, seed=1, device="cpu")
    res = json.loads(Path(res_path).read_text(encoding="utf-8"))
    assert res["shape"] == [4, 2]

    # Empty and truncated custom.sha256 MUST fail verification
    custom_sha = bundle_dir / "custom.sha256"
    custom_sha.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or empty checksum"):
        _verify_sha256_file(custom_pt, custom_sha)
    with pytest.raises(ValueError, match="invalid or empty checksum"):
        sample(custom_pt, count=4, seed=1, device="cpu")

    custom_sha.write_text("abcd1234", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or empty checksum"):
        _verify_sha256_file(custom_pt, custom_sha)

    valid_digest = hashlib.sha256(custom_pt.read_bytes()).hexdigest()
    custom_sha.write_text(f"{valid_digest}  custom.pt\n", encoding="utf-8")
    res2_path = sample(custom_pt, count=4, seed=1, device="cpu")
    assert json.loads(Path(res2_path).read_text(encoding="utf-8"))["shape"] == [4, 2]


# ==============================================================================
# 2. API & Provenance
# ==============================================================================

def test_snapshot_config_sources_manifest_hndl_and_load_stored_run_model(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "g.hndl").write_text("linear(2)\n", encoding="utf-8")
    (src_dir / "d.hndl").write_text("linear(1)\n", encoding="utf-8")
    base_toml = src_dir / "a_base.toml"
    base_toml.write_text(
        'format = "hypergan-model/1"\n'
        '[data]\nbuiltin = "gaussian_grid"\n'
        '[networks.generator]\nrole = "generator"\nhndl = "g.hndl"\ninput_shape = ["B", 4]\noutput_shape = ["B", 2]\ninputs = { x = "latent" }\n'
        '[networks.discriminator]\nrole = "critic"\nhndl = "d.hndl"\ninput_shape = ["B", 2]\noutput_shape = ["B", 1]\ninputs = { x = "candidate" }\n'
        '[[losses]]\ntype = "adversarial"\ncritic = "discriminator"\n'
        '[train]\nsteps = 10\nbatch_size = 16\ndevice = "cpu"\n',
        encoding="utf-8",
    )
    child_toml = src_dir / "z_child.toml"
    child_toml.write_text('format = "hypergan-model/1"\nextends = "a_base.toml"\n[train]\nsteps = 20\n', encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot_config_sources(run_dir, [base_toml, child_toml], entry_file=child_toml)
    cs = run_dir / "config-source"
    cs_manifest = json.loads((cs / "manifest.json").read_text(encoding="utf-8"))
    assert cs_manifest["entry"] == "z_child.toml"
    assert (cs / "g.hndl").is_file()
    assert (cs / "d.hndl").is_file()

    # Case 1: manifest matches child_toml -> loads entry z_child.toml (not a_base.toml!)
    resolved_child = config_values(load_config(child_toml))
    (run_dir / "manifest.json").write_text(json.dumps({"config": resolved_child}), encoding="utf-8")
    loaded = api._load_stored_run_model(run_dir)
    assert loaded.spec["train"]["steps"] == 20

    # Case 2: CLI override changed steps to 99 in manifest.json -> prefers resolved manifest config!
    overridden = config_values(resolve_config({**resolved_child, "training": {**resolved_child["training"], "steps": 99}}))
    (run_dir / "manifest.json").write_text(json.dumps({"config": overridden}), encoding="utf-8")
    loaded_ovr = api._load_stored_run_model(run_dir)
    assert loaded_ovr.spec["train"]["steps"] == 99


def test_lift_preserves_recipe_aggregation_and_adversarial_inputs():
    raw = resolve_config({
        "recipe": {"name": "custom_test_recipe", "reg_coeff": 0.25},
        "aggregation": {"strategy": "pseudo_gradient", "sync_every_steps": 4, "outer_lr": 0.8, "outer_momentum": 0.5},
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"input": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"input": "candidate", "cond": "batch.cond"}},
        },
        "adversarial": {"weight": 1.0, "inputs": {"cond": "batch.cond"}},
    })
    lifted = api.lift(raw)
    assert lifted.spec["recipe"]["name"] == "custom_test_recipe"
    assert lifted.spec["aggregation"]["strategy"] == "pseudo_gradient"
    assert lifted.spec["aggregation"]["sync_every_steps"] == 4
    adv_losses = [l for l in lifted.spec["losses"] if l["type"] == "adversarial"]
    assert adv_losses[0]["inputs"] == {"cond": "batch.cond"}


def test_load_inference_restores_nested_nonpersistent_buffers(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from particlegan import GaussianPrior
    from hypergan.artifacts import save_bundle

    class _NestedGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = torch.nn.Module()
            self.sub.lin = torch.nn.Linear(4, 2)
            self.sub.register_buffer("non_persist_tag", torch.tensor([7.0, 9.0]), persistent=False)

        def forward(self, x):
            return self.sub.lin(x) + self.sub.non_persist_tag

    cfg = resolve_config({
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    trained_gen = _NestedGen()
    trainer = _make_trainer(cfg, trained_gen, GaussianPrior(z_dim=4))
    (tmp_path / "nested").mkdir()
    save_bundle(tmp_path / "nested", trainer, {})
    bundle_path = tmp_path / "nested" / "model.pt"

    class _FakeGraph(torch.nn.Module):
        def __init__(self, _specs):
            super().__init__()
            fresh = _NestedGen()
            with torch.no_grad():
                fresh.sub.non_persist_tag.zero_()
            self.models = torch.nn.ModuleDict({"generator": fresh})

    monkeypatch.setattr("hypergan.recipes.ComponentGraph", _FakeGraph)
    _state, _cfg, graph, _prior, _sha = api.load_inference(bundle_path)
    assert torch.allclose(graph.models["generator"].sub.non_persist_tag, torch.tensor([7.0, 9.0]))


def test_load_bundle_catches_value_error_and_honors_manifest_bundle_path(tmp_path):
    torch = pytest.importorskip("torch")
    from particlegan import GaussianPrior
    from hypergan.artifacts import save_bundle
    from hypergan.recipes import linear as hndl_linear

    cfg = resolve_config({
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        },
    })
    gen = hndl_linear(4, 2)
    trainer = _make_trainer(cfg, gen, GaussianPrior(z_dim=4))
    run_dir = tmp_path / "run_bundle"
    custom_bundle_dir = run_dir / "attempts" / "0001" / "inference"
    custom_bundle_dir.mkdir(parents=True)
    save_bundle(custom_bundle_dir, trainer, {})
    bundle_path = custom_bundle_dir / "model.pt"
    # Write manifest with bundle_path and NO checkpoints/ directory (so resolve_checkpoint raises ValueError)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "r1", "config": config_values(cfg), "bundle_path": str(bundle_path)}),
        encoding="utf-8",
    )
    lg = api.load_bundle(run_dir, device="cpu")
    assert lg.sample(2, seed=1).shape == (2, 2)


def test_diff_distinguishes_particlegan_presets():
    pytest.importorskip("particlegan")
    m_v11 = api.override(hg.packaged("gaussian-grid"), {"defaults": "particlegan:unified_v11"})
    m_v15 = api.override(hg.packaged("gaussian-grid"), {"defaults": "particlegan:unified_v15"})
    m_axiom = api.override(hg.packaged("gaussian-grid"), {"defaults": "particlegan:axiom"})
    d1 = api.diff(m_v11, m_v15)
    d2 = api.diff(m_v15, m_axiom)
    assert d1["identical"] is False
    assert d2["identical"] is False
    assert any(k.startswith("recipe.") for k in d1["differences"])
    assert any(k.startswith("recipe.") for k in d2["differences"])


def test_override_null_deletes_key_and_supports_minimal_legacy_recipe():
    m = hg.packaged("gaussian-grid")
    m2 = api.override(m, {"train.ema": 0.99})
    assert "ema" in m2.spec["train"]
    m3 = api.override(m2, {"train.ema": None})
    assert "ema" not in m3.spec["train"]

    # Minimal legacy engine recipe omitting optimizer/training still works with override()
    legacy = api.Model(
        spec={
            "components": {
                "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"input": "latent"}},
                "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"input": "candidate"}},
            }
        }
    )
    updated = api.override(legacy, {"training.steps": 25, "optimizer.lr": 0.001})
    assert updated.spec["training"]["steps"] == 25
    assert updated.spec["optimizer"]["lr"] == 0.001


# ==============================================================================
# 3. CLI Fixes
# ==============================================================================

def test_print_training_header_never_initializes_cuda(monkeypatch, capsys):
    import torch
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("CUDA initialized before backend policy!")),
    )
    monkeypatch.setattr(
        "hypergan.gpu_inventory.inspect_gpu_hardware",
        lambda: [{"index": 0, "name": "Mock RTX 5090", "memory_total": 32768.0, "uuid": "GPU-mock"}],
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cli._print_training_header(Path("/tmp/run"), "test-model", "cuda:0")
    err = capsys.readouterr().err
    assert "Mock RTX 5090" in err


def test_cli_port_zero_non_loopback_warning_reexec_and_subparsers(tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext

    # Nested subparsers use _CustomSubcommandParser
    parser = cli._parser()
    for action in parser._actions:
        if isinstance(action, cli.argparse._SubParsersAction):
            for cmd_name in ("runs", "checkpoints", "queue"):
                sub = action.choices[cmd_name]
                for sub_act in sub._actions:
                    if isinstance(sub_act, cli.argparse._SubParsersAction):
                        assert sub_act._parser_class is cli._CustomSubcommandParser

    # --port 0 must not fall back to DEFAULT_VIEWER_PORT, and 0.0.0.0 + auth=none must warn
    captured = {}
    monkeypatch.setattr(
        "hypergan.web_autostart.ensure_dashboard_running",
        lambda *a, **kw: captured.update(kw) or "http://0.0.0.0:12345",
    )
    monkeypatch.setattr("hypergan.web_autostart.training_viewer", lambda *a, **kw: nullcontext())
    args = SimpleNamespace(
        run_dir=str(tmp_path),
        no_server=False,
        server=True,
        open=False,
        server_port=0,
        server_host="0.0.0.0",
        auth="none",
        public_origin=None,
        viewer_dev=False,
        node_rank=None,
    )
    cli._training_viewer(args)
    assert captured["port"] == 0
    assert "warning:" in capsys.readouterr().err.lower()

    # _reconstruct_argv_for_reexec handles packaged model name + project dir + preserves --set when not replaced
    argv_pkg = ["hypergan", "train", "gaussian-grid", "--set", "train.steps=5", "--steps", "5"]
    rebuilt_pkg = cli._reconstruct_argv_for_reexec(
        argv_pkg, "/pkg/gaussian-grid.toml", "/tmp/run/recipe.toml", "/tmp/run", orig_arg_str="gaussian-grid"
    )
    assert "/tmp/run/recipe.toml" in rebuilt_pkg
    assert "--set" not in rebuilt_pkg

    argv_unmatched = ["hypergan", "train", "other.toml", "--set", "train.steps=5"]
    rebuilt_unmatched = cli._reconstruct_argv_for_reexec(
        argv_unmatched, "/some/model.toml", "/tmp/run/recipe.toml", "/tmp/run"
    )
    assert "--set" in rebuilt_unmatched

    # Unknown [run] env name raises ValueError explicitly
    ws = {"envs": {"known_env": {"interpreter": sys.executable}}}
    with pytest.raises(ValueError, match="Unknown \\[run\\] env 'missing_env'"):
        cli._check_reexec({"env": "missing_env"}, ws)


def test_resume_completed_json_and_sample_pt_sampler(tmp_path, monkeypatch, capsys):
    # resume <completed> --json must output valid JSON, not plain text
    run_dir = tmp_path / "completed_run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "completed", "steps": 10, "total_steps": 10, "run_id": "abc"}),
        encoding="utf-8",
    )
    rc = cli.main(["resume", str(run_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    inner = payload.get("manifest", payload)
    assert inner["already_complete"] is True
    assert inner["steps"] == 10

    # sample <bundle.pt> --sampler <name> invokes hg.view and saves rendered view
    bundle_file = tmp_path / "gen.pt"
    bundle_file.write_bytes(b"placeholder")
    called = {}

    class _FakeView:
        def save(self, path):
            Path(path).write_bytes(b"PNGDATA")

    out_dir = tmp_path / "sample_out"
    out_dir.mkdir()
    sample_json = out_dir / "samples.json"
    sample_json.write_text(json.dumps({"shape": [4, 2], "samples": [[0.0, 1.0]] * 4, "step": 5}), encoding="utf-8")
    monkeypatch.setattr(api, "sample", lambda *a, **kw: SimpleNamespace(path=sample_json, shape=[4, 2], step=5))
    monkeypatch.setattr(api, "view", lambda sample_dict, sampler_name, run=None: called.update(name=sampler_name, run=run) or _FakeView())
    monkeypatch.setattr("torch.load", lambda *a, **kw: {"config": {"samplers": {"hist": {"fn": "x:y"}}}})
    rc_s = cli.main(["sample", str(bundle_file), "--sampler", "hist", "--output-dir", str(out_dir), "--json"])
    assert rc_s == 0
    assert called["name"] == "hist"
    out_payload = json.loads(capsys.readouterr().out)
    assert "view" in out_payload
    assert Path(out_payload["view"]).read_bytes() == b"PNGDATA"


# ==============================================================================
# 4. Config, ModelFile, Schema & Workspace
# ==============================================================================

def test_model_file_eval_data_train_owner_every_steps_and_particlegan_defaults():
    base_nets = {
        "g": {"role": "generator", "module": "mod:Gen", "inputs": {"z": "latent"}},
        "e": {"role": "encoder", "module": "mod:Enc", "inputs": {"x": "batch.real"}},
        "d": {"role": "critic", "module": "mod:Critic", "inputs": {"x": "candidate"}},
    }
    # 1) data = "train" across CIFAR10Data, ImageFolder, ItemData, ColorizationData
    for data_spec, check in [
        ({"factory": "hypergan.datasets:CIFAR10Data", "root": "/tmp/c10", "sampling": "random", "horizontal_flip": True},
         lambda args: args["sampling"] == "sequential" and args["horizontal_flip"] is False),
        ({"factory": "hypergan.datasets:ImageFolder", "root": "/tmp/im", "height": 16, "width": 16, "shuffle": True},
         lambda args: args["shuffle"] is False),
        ({"factory": "hypergan.datasets:ItemData", "args": {"dataset": "mod:DS", "shuffle": True}},
         lambda args: args["shuffle"] is False),
        ({"factory": "hypergan.colorization_data:ColorizationData", "root": "/tmp/col", "shuffle": True, "bad_image_policy": "skip"},
         lambda args: args["shuffle"] is False and args["bad_image_policy"] == "error"),
    ]:
        spec = {
            "format": "hypergan-model/1",
            "data": data_spec,
            "networks": base_nets,
            "losses": [
                {"type": "adversarial", "critic": "d"},
                {"type": "objective", "id": "enc_rec", "fn": "mse", "inputs": {"input": "e", "target": "batch.real"}, "owner": "e"},
            ],
            "train": {"steps": 10, "every_steps": 25},
            "metrics": {"preset": "standard"},
            "evaluations": {
                "div": {"factory": "hypergan.diversity_metrics:SampleDiversity", "samples": 16, "batch_size": 4, "seed": 1, "data": "train"}
            },
        }
        lowered = lower(spec)
        # losses[].owner stripped from objective term so resolve_config succeeds!
        assert "owner" not in lowered["objectives"][0]
        # train.every_steps NOT clobbered to 10 when [metrics]/[evaluations] exist
        assert lowered["metrics"]["every_steps"] == 25
        eval_args = lowered["metrics"]["custom"]["div"]["evaluation"]["data"]["args"]
        assert check(eval_args), f"Failed eval data check for {data_spec}: {eval_args}"
        resolved = resolve_config(lowered)
        assert resolved["metrics"]["every_steps"] == 25

    # Unknown owner rejected
    bad_owner = {
        "format": "hypergan-model/1",
        "data": {"builtin": "gaussian_grid"},
        "networks": base_nets,
        "losses": [
            {"type": "adversarial", "critic": "d"},
            {"type": "objective", "id": "enc_rec", "fn": "mse", "inputs": {"input": "e", "target": "batch.real"}, "owner": "nonexistent"},
        ],
    }
    with pytest.raises(ModelFileError, match="must name a declared network"):
        lower(bad_owner)

    # defaults = "particlegan" does not clobber preset reg_coeff or [recipe] overrides
    pytest.importorskip("particlegan")
    pg_spec = {
        "format": "hypergan-model/1",
        "defaults": "particlegan",
        "data": {"builtin": "gaussian_grid"},
        "networks": {
            "g": {"role": "generator", "source": "linear(2)\n", "input_shape": ["B", 4], "output_shape": ["B", 2], "inputs": {"x": "latent"}},
            "d": {"role": "critic", "source": "linear(1)\n", "input_shape": ["B", 2], "output_shape": ["B", 1], "inputs": {"x": "candidate"}},
        },
        "losses": [{"type": "adversarial", "critic": "d"}],
        "recipe": {"lr": 0.00123, "reg_coeff": 0.42},
    }
    pg_resolved = resolve_config(lower(pg_spec))
    assert pg_resolved["optimizer"]["lr"] == pytest.approx(0.00123)
    assert pg_resolved["recipe"]["lr"] == pytest.approx(0.00123)
    assert pg_resolved["gradient_penalty"]["coeff"] == pytest.approx(0.42)
    assert pg_resolved["recipe"]["reg_coeff"] == pytest.approx(0.42)


def test_config_z_dim_auto_inference_generated_cycle_mog_modes_and_aggregation():
    # prior.args.z_dim auto-inferred from generator input_shape even when [prior] omits z_dim
    res = resolve_config({
        "prior": {"kind": "gaussian", "args": {"init_std": 0.5}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {"factory": "hndl", "args": {"source": "linear(2)\n", "input_shape": ["B", 16], "output_shape": ["B", 2]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "hndl", "args": {"source": "linear(1)\n", "input_shape": ["B", 2], "output_shape": ["B", 1]}, "inputs": {"x": "candidate"}},
        },
    })
    assert res["prior"]["args"]["z_dim"] == 16

    # Cycle via "generated" binding must raise
    with pytest.raises(ValueError, match="Cyclic component binding"):
        resolve_config({
            "components": {
                "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"input": "components.aux"}},
                "aux": {"factory": "linear", "args": {"in_features": 2, "out_features": 4}, "inputs": {"input": "generated"}},
                "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"input": "candidate"}},
            }
        })

    # Invalid "modes" arg on MoG prior must be rejected
    with pytest.raises(ValueError, match="Unknown prior field.*modes"):
        resolve_config({"prior": {"kind": "mog", "args": {"z_dim": 4, "modes": 8}}})

    # Aggregation validation
    with pytest.raises(ValueError, match="Unknown aggregation.strategy"):
        resolve_config({"aggregation": {"strategy": "not_a_strategy"}})
    with pytest.raises(ValueError, match="gradient_allreduce requires sync_every_steps = 1"):
        resolve_config({"aggregation": {"strategy": "gradient_allreduce", "sync_every_steps": 4}})
    with pytest.raises(ValueError, match="outer_momentum"):
        resolve_config({"aggregation": {"strategy": "pseudo_gradient", "outer_momentum": 1.0}})


def test_schema_info_defaults_and_workspace_multipass_and_multilevel_extends(tmp_path):
    # Schema defaults reconciled with runtime
    schema = get_schema()
    assert schema["sections"]["defaults"]["default"] == "hypergan"
    assert schema["sections"]["prior"]["fields"]["num_particles"]["default"] == 20000
    assert schema["sections"]["train"]["fields"]["device"]["default"] == "cpu"
    assert schema["sections"]["data"]["fields"]["resize"]["default"] == "none"
    assert schema["sections"]["data"]["fields"]["alpha"]["default"] == "error"
    assert "default" not in schema["sections"]["data"]["fields"]["width"]
    assert "default" not in schema["sections"]["data"]["fields"]["height"]

    # Workspace multi-pass interpolation before anchoring runs_root
    ws_file = tmp_path / "ws" / "hypergan.toml"
    ws_file.parent.mkdir()
    ws_file.write_text(
        '[workspace]\nruns_root = "${paths.out}/runs"\n\n'
        '[paths]\nbase = "${workspace.root}/storage"\nout = "${paths.base}/artifacts"\n'
        'exports = "${workspace.runs_root}/exports"\n',
        encoding="utf-8",
    )
    ws = load_workspace(ws_file)
    expected_runs_root = str((ws_file.parent / "storage" / "artifacts" / "runs").resolve())
    assert ws["workspace"]["runs_root"] == expected_runs_root
    assert ws["paths"]["out"] == str((ws_file.parent / "storage" / "artifacts").resolve())
    assert ws["paths"]["exports"] == f"{expected_runs_root}/exports"

    # Multi-level extends anchoring: each ancestor anchored once to its own directory
    gp_dir = tmp_path / "gp"
    p_dir = tmp_path / "p"
    c_dir = tmp_path / "c"
    for d in (gp_dir, p_dir, c_dir):
        d.mkdir()
    (gp_dir / "gp.hndl").write_text("linear(2)\n", encoding="utf-8")
    (p_dir / "p.hndl").write_text("linear(1)\n", encoding="utf-8")
    (gp_dir / "gp.toml").write_text(
        'format = "hypergan-model/1"\n'
        '[data]\nbuiltin = "image_folder"\nroot = "gp_data"\n'
        '[networks.generator]\nrole = "generator"\nhndl = "gp.hndl"\ninput_shape = ["B", 4]\noutput_shape = ["B", 2]\ninputs = { x = "latent" }\n',
        encoding="utf-8",
    )
    (p_dir / "parent.toml").write_text(
        'format = "hypergan-model/1"\nextends = "../gp/gp.toml"\n'
        '[networks.discriminator]\nrole = "critic"\nhndl = "p.hndl"\ninput_shape = ["B", 2]\noutput_shape = ["B", 1]\ninputs = { x = "candidate" }\n',
        encoding="utf-8",
    )
    child_path = c_dir / "child.toml"
    child_path.write_text('format = "hypergan-model/1"\nextends = "../p/parent.toml"\n', encoding="utf-8")
    merged = resolve_inheritance({"format": "hypergan-model/1", "extends": "../p/parent.toml"}, config_path=child_path)
    assert merged["data"]["root"] == str((gp_dir / "gp_data").resolve())
    assert merged["networks"]["generator"]["hndl"] == str((gp_dir / "gp.hndl").resolve())
    assert merged["networks"]["discriminator"]["hndl"] == str((p_dir / "p.hndl").resolve())


# ==============================================================================
# 5. Queue, Run Controller & Signals
# ==============================================================================

def test_queue_install_service_security_and_invalid_gpu_and_scheduler_fixes(tmp_path, monkeypatch, capsys):
    # Path traversal & directive injection rejected in install_service
    with pytest.raises(ValueError, match="Invalid systemd service name"):
        install_service(name="../escape", dry_run=True)
    with pytest.raises(ValueError, match="Invalid systemd service name"):
        install_service(name="sub/name", dry_run=True)
    with pytest.raises(ValueError, match="newlines"):
        install_service(devices="0\nExecStartPre=/bin/evil", dry_run=True)

    # Graceful stop wait timeout must exceed GracefulStop's 30s deadline
    assert _GRACEFUL_STOP_WAIT_SECONDS >= 35.0

    # Invalid GPU index rejected instead of returning (True, None)
    monkeypatch.setattr(
        "hypergan.queue.inspect_gpu_hardware",
        lambda: [{"index": 0, "uuid": "GPU-0", "memory_free": 40000.0}],
    )
    monkeypatch.setattr("hypergan.queue.inspect_gpu_processes", lambda: [])
    ok, reason = check_device_availability("99")
    assert ok is False
    assert "Invalid or unknown GPU device" in reason

    # No head-of-line starvation + no --name stem collision + multi-runner GPU exclusion
    q_file = tmp_path / ".hypergan" / "queue.json"
    m = tmp_path / "same_stem.toml"
    m.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    j_gpu0 = queue_add(m, devices="0", queue_file=q_file)
    j_gpu1 = queue_add(m, devices="1", queue_file=q_file)

    # Mark GPU 0 busy, GPU 1 free; runner should skip j_gpu0 and run j_gpu1 immediately!
    monkeypatch.setattr(
        "hypergan.queue.check_device_availability",
        lambda devs, ws=None: (False, "GPU 0 busy") if str(devs) == "0" else (True, None),
    )
    launched_cmds = []

    class _MockPopen:
        def __init__(self, cmd, **kwargs):
            launched_cmds.append(cmd)
            self.pid = 77777
            # Cancel j_gpu0 once j_gpu1 starts so once=True exits promptly
            queue_stop(job_id=j_gpu0["id"], queue_file=q_file)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _MockPopen)
    rc = queue_run(once=True, poll_interval=0.01, queue_file=q_file, json_output=True)
    assert rc == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["total_executed"] == 1
    assert out_json["executed"][0]["id"] == j_gpu1["id"]
    # --name must NOT be passed when user did not explicitly supply custom_name
    assert "--name" not in launched_cmds[0]

    # Multi-runner GPU exclusion: a running job on GPU 0 blocks another queued job on GPU 0
    q_file2 = tmp_path / ".hypergan" / "queue2.json"
    from hypergan.queue import _process_identity, load_queue, queue_lock, save_queue
    queue_add(m, devices="0", queue_file=q_file2)
    j_conflict = queue_add(m, devices="0", queue_file=q_file2)
    queue_add(m, devices="1", queue_file=q_file2)
    with queue_lock(q_file2):
        q = load_queue(q_file2)
        q["jobs"][0]["status"] = "running"
        q["jobs"][0]["runner"] = _process_identity(os.getpid())
        q["jobs"][0]["pid"] = os.getpid()
        q["jobs"][0]["process"] = _process_identity(os.getpid())
        save_queue(q_file2, q)

    monkeypatch.setattr("hypergan.queue.check_device_availability", lambda devs, ws=None: (True, None))
    launched2 = []

    class _MockPopen2:
        def __init__(self, cmd, **kwargs):
            launched2.append(cmd)
            self.pid = 77778
            queue_stop(job_id=j_conflict["id"], queue_file=q_file2)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _MockPopen2)
    assert queue_run(once=True, poll_interval=0.01, queue_file=q_file2) == 0
    assert len(launched2) == 1
    assert "--devices" in launched2[0] and launched2[0][launched2[0].index("--devices") + 1] == "1"


def test_run_controller_worktree_anchoring_poll_requests_and_graceful_stop(tmp_path):
    from hypergan.run_controller import CompletedUpdate, ExecutionInfo, _record_launch, run_train
    from hypergan.run_requests import request_status, submit_request

    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    (ws_root / "hypergan.toml").write_text('[envs.e1]\nworktree = "rel_wt"\n', encoding="utf-8")
    cfg_file = ws_root / "m.toml"
    cfg_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    run_dir = tmp_path / "run_wt"
    run_dir.mkdir()
    _record_launch(run_dir, 1, {"run": {"env": "e1"}, "_source_path": str(cfg_file)}, cfg_file)
    launch = json.loads((run_dir / "launch" / "0001.json").read_text(encoding="utf-8"))
    assert launch["worktree"] == str((ws_root / "rel_wt").resolve())

    # poll_requests survives corrupt checkpoint manifest and still executes stop request
    class _FakeExec:
        inference_available = False

        def __init__(self, cfg):
            self.step = 0
            self.cfg = cfg

        def environment(self):
            return {"runtime": {"device": "cpu"}, "source": {"git": "test"}}

        def start(self):
            return ExecutionInfo(
                step=0,
                data_identity={"id": "fake"},
                recovery_reasons=[],
                checkpoint_metadata={},
                environment=None,
                model={"name": "fake"},
            )

        def checkpoint(self, rdir, metadata):
            ckpt = rdir / "checkpoints" / f"step-{self.step:06d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            # Write corrupt manifest on step 0 so reconciliation on step 1 fails!
            if self.step == 0:
                (ckpt / "manifest.json").write_text("{corrupt json", encoding="utf-8")
            else:
                (ckpt / "manifest.json").write_text(
                    json.dumps({"step": self.step, "attempt_id": metadata["attempt_id"], "request_ids": metadata.get("request_ids", [])}),
                    encoding="utf-8",
                )
            return ckpt

        def update(self):
            self.step += 1
            if self.step == 1:
                m_data = json.loads((poll_run / "manifest.json").read_text(encoding="utf-8"))
                submit_request(poll_run, action="checkpoint", run_id=m_data["run_id"], attempt_id=m_data["attempt_id"], request_id="req-ckpt")
                submit_request(poll_run, action="stop", run_id=m_data["run_id"], attempt_id=m_data["attempt_id"], request_id="req-stop")
            return CompletedUpdate(
                step=self.step,
                metrics={"d_loss": 0.5, "g_loss": 0.5, "d_adversarial": 0.5, "g_adversarial": 0.5,
                         "d_adversarial_weighted": 0.5, "g_adversarial_weighted": 0.5,
                         "gradient_penalty": 0.0, "prior_loss": 0.0, "lr_scale": 1.0, "objectives": []},
            )

        def observe(self, cb, row):
            return cb(row)

        def shutdown(self):
            pass

    poll_run = tmp_path / "poll_run"
    result = run_train(
        {"training": {"steps": 10, "batch_size": 4, "device": "cpu"}},
        poll_run,
        checkpoint_every=1,
        execution_factory=_FakeExec,
    )
    assert result["status"] == "stopped"
    assert result["steps"] == 1
    assert request_status(poll_run, "req-stop")["status"] == "succeeded"

    # execution.observe returning False also stops training cleanly (for replicated early stop)
    class _EarlyStopExec(_FakeExec):
        def update(self):
            self.step += 1
            return CompletedUpdate(
                step=self.step,
                metrics={"d_loss": 0.5, "g_loss": 0.5, "d_adversarial": 0.5, "g_adversarial": 0.5,
                         "d_adversarial_weighted": 0.5, "g_adversarial_weighted": 0.5,
                         "gradient_penalty": 0.0, "prior_loss": 0.0, "lr_scale": 1.0, "objectives": []},
            )

        def observe(self, cb, row):
            return False if row.get("event") == "train" and row.get("step") == 2 else None

    es_run = tmp_path / "es_run"
    es_res = run_train(
        {"training": {"steps": 10, "batch_size": 4, "device": "cpu"}},
        es_run,
        checkpoint_every=10,
        on_event=lambda e: None,
        execution_factory=_EarlyStopExec,
    )
    assert es_res["status"] == "stopped"
    assert es_res["steps"] == 2
    assert es_res["stop_reason"] == "early_stop"

    # GracefulStop._receive must never spawn a thread/Timer inside the signal handler
    created_threads = []
    orig_thread_start = threading.Thread.start

    with GracefulStop(timeout=10.0) as stop:
        def _spy_start(self, *a, **kw):
            created_threads.append(self)
            return orig_thread_start(self, *a, **kw)

        threading.Thread.start = _spy_start
        try:
            stop._receive(signal.SIGTERM, None)
            assert stop.reason == "SIGTERM"
            assert created_threads == []
        finally:
            threading.Thread.start = orig_thread_start

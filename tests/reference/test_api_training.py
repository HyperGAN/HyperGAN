"""Tests for hypergan.api: public functional interface, builders, explain, and guards."""

import inspect

from pathlib import Path

import pytest

import hypergan.api as hg

from hypergan.config import DEFAULT, resolve_config

ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = ROOT / 'examples' / 'api' / 'config_first'

def dummy_fn(x, hidden=32):
    return x * hidden

def test_api_checkpoints_and_export(tmp_path):
    import json
    import torch
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "att-1-step-00000100-uuid1"
    ckpt_dir.mkdir(parents=True)
    state = {
        "step": 100,
        "ema_graph": {"models.generator.weight": torch.randn(2, 2)},
        "ema_prior": {"prior": torch.randn(2)},
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    (ckpt_dir / "manifest.json").write_text(json.dumps({
        "step": 100,
        "config": {"components": {"generator": {"factory": "linear", "inputs": {}}}},
    }), encoding="utf-8")

    # 1. resolve_checkpoint
    resolved = hg.resolve_checkpoint(run_dir, "latest")
    assert resolved == ckpt_dir

    # 2. extract_ema_bundle
    bundle = hg.extract_ema_bundle(ckpt_dir)
    assert bundle["kind"] == "ema-inference"
    assert bundle["step"] == 100

    # 3. export
    out_pt = hg.export(run_dir, checkpoint="latest")
    assert out_pt.is_file()
    assert out_pt.name == "model.pt"
    assert (out_pt.parent / "model.sha256").is_file()

    # 4. prune_checkpoints
    report = hg.prune_checkpoints(run_dir, dry_run=True)
    assert report["dry_run"] is True
    assert report["retained_count"] == 1

def test_api_load_inference(tmp_path):
    import json
    import torch
    from hypergan.config import resolve_config
    from hypergan.recipes import ComponentGraph, make_prior

    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "att-1-step-00000100-uuid1"
    ckpt_dir.mkdir(parents=True)
    cfg = resolve_config({
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 2, "out_features": 2}, "inputs": {"z": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        }
    })
    cg = ComponentGraph(cfg["components"])
    gen_sd = {f"models.generator.{k}": v for k, v in cg.models["generator"].state_dict().items()}
    prior = make_prior(cfg["prior"], device="cpu")
    state = {
        "step": 100,
        "ema_graph": gen_sd,
        "ema_prior": prior.state_dict(),
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    (ckpt_dir / "manifest.json").write_text(json.dumps({
        "step": 100,
        "config": cfg,
    }), encoding="utf-8")

    out_pt = hg.export(run_dir, checkpoint="latest")
    loaded_state, loaded_cfg, graph, loaded_prior, sha = hg.load_inference(out_pt.parent)
    assert loaded_state["kind"] == "ema-inference"
    assert loaded_state["step"] == 100
    assert graph is not None
    assert loaded_prior is not None
    assert sha is not None

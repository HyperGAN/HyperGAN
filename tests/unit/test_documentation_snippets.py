"""Unit tests verifying semantic correctness of documented workflows and snippets.

Covers:
- docs/custom-pytorch.md: Custom HNDL operator registration, summary/shape/args, and network execution
- docs/deployment.md: LoadedGenerator.sample_latents & sample_prior with seed/generator parameters
- docs/python-api.md: ValidationWarning string subclass & .message property
- docs/python-api.md: Run.stop_reason property on early stopping
- docs/python-api.md: RunHandle.is_running and RunHandle.join aliases
- docs/python-api.md: PsSnapshot.runs normalization and RunStatus attribute/mapping access
- docs/python-api.md: hg.export with positional target argument
"""
from dataclasses import is_dataclass
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

import hypergan as hg
from hypergan.artifacts import LoadedGenerator
from hypergan.api import ValidationWarning, RunStatus, PsSnapshot, RunHandle, Run


def test_custom_hndl_operator_registration_and_execution():
    """Verify custom HNDL operator registration from docs/custom-pytorch.md."""
    from hndl import Arg, Registry, operator
    from hypergan.hndl_networks import build_network

    @operator(
        "custom_squash_doc_test",
        summary="Squash activation function.",
        shape="x[B, ...] -> out[B, ...]",
        args={
            "dim": Arg(int, -1, help="Dimension along which to compute norm"),
            "eps": Arg(float, 1e-8, help="Epsilon for numerical stability"),
        },
        category="activation",
    )
    class CustomSquash(nn.Module):
        def __init__(self, dim: int = -1, eps: float = 1e-8):
            super().__init__()
            self.dim = int(dim)
            self.eps = float(eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
            scale = (norm ** 2) / (1.0 + norm ** 2)
            return scale * (x / (norm + self.eps))

    registry = Registry.builtins()
    registry.add(CustomSquash)

    hndl_source = "linear(32)\ncustom_squash_doc_test(dim=-1)\nlinear(10)"
    net = build_network(hndl_source, input_shape=("B", 16), output_shape=("B", 10), registry=registry)
    x = torch.randn(4, 16)
    out = net(x)
    assert out.shape == torch.Size([4, 10])


def test_loaded_generator_latent_sampling_with_seed():
    """Verify LoadedGenerator.sample_latents and sample_prior support seed parameter."""
    with tempfile.TemporaryDirectory() as d:
        bundle_path = Path(d) / "bundle.pt"
        torch.save({
            "schema_version": 1,
            "kind": "ema-inference",
            "step": 100,
            "config": {"prior": {"z_dim": 8}, "sampling": {}},
            "components": {
                "generator": {
                    "factory": "torch.nn:Linear",
                    "args": {"in_features": 8, "out_features": 8},
                    "inputs": {"input": "latent"},
                    "trainable": True,
                }
            },
            "model_states": {
                "generator": {
                    "weight": torch.eye(8),
                    "bias": torch.zeros(8),
                }
            },
            "prior": {
                "z": torch.randn(20, 8),
                "sigma": torch.tensor(0.01),
            },
            "example_inputs": {},
        }, bundle_path)

        generator = hg.load_bundle(bundle_path, device="cpu")
        generator.eval()

        # 1. Sample with explicit seed
        latents_a1 = generator.sample_latents(count=4, seed=123)
        latents_a2 = generator.sample_latents(count=4, seed=123)
        assert torch.equal(latents_a1, latents_a2), "latents with same seed must be equal"

        # 2. Sample with generator
        rng = torch.Generator(device="cpu").manual_seed(123)
        latents_b = generator.sample_latents(count=4, generator=rng)
        assert torch.equal(latents_a1, latents_b), "seed and generator with same seed must match"

        # 3. Sample prior with seed
        prior_latents = generator.sample_prior(count=4, seed=123)
        assert torch.equal(latents_a1, prior_latents)

        # 4. Error on providing both
        with pytest.raises(ValueError, match="Cannot provide both"):
            generator.sample_latents(count=4, generator=rng, seed=123)

        # 5. Forward pass with latents
        images = generator(z=latents_a1)
        assert images.shape == torch.Size([4, 8])


def test_validation_warning_attributes_and_string_compatibility():
    """Verify ValidationWarning behaves as a string while providing .message property."""
    model = hg.packaged("cifar10-transgan")
    warnings = hg.validate(model)
    assert isinstance(warnings, list)
    for w in warnings:
        assert isinstance(w, str)
        assert hasattr(w, "message")
        assert w.message == str(w)
        assert f"Warning: {w.message}" == f"Warning: {w}"
        assert w == str(w)

    custom_w = ValidationWarning("invalid parameter", field="generator.lr", level="warning")
    assert is_dataclass(custom_w)
    assert isinstance(custom_w, str)
    assert custom_w.message == "invalid parameter"
    assert custom_w.field == "generator.lr"
    assert custom_w == "invalid parameter"
    assert [custom_w] == ["invalid parameter"]


def test_run_stop_reason_and_early_stopping(tmp_path):
    """Verify Run.stop_reason property and early stopping callback from docs/python-api.md."""
    def check_divergence(event):
        if event.get("event") == "train":
            if event.get("step", 0) >= 3:
                return False
        return True

    run_dir = tmp_path / "early_stop_run"
    result = hg.train(
        hg.packaged("gaussian-grid"),
        run_dir=run_dir,
        steps=10,
        devices="cpu",
        previews=0,
        on_event=check_divergence,
    )
    assert isinstance(result, Run)
    assert result.step == 3
    assert result.stop_reason == "early_stop"
    assert hasattr(result, "stop_reason")


def test_run_handle_is_running_and_join_aliases(tmp_path):
    """Verify RunHandle.is_running and join aliases from docs/python-api.md."""
    run_dir = tmp_path / "async_run"
    handle = hg.launch(
        hg.packaged("gaussian-grid"),
        run_dir=run_dir,
        steps=4,
        devices="cpu",
        previews=0,
    )
    assert isinstance(handle, RunHandle)
    # Check is_running alias
    assert handle.is_running() == handle.is_alive()

    # Wait with join alias
    result = handle.join(timeout=15.0)
    assert isinstance(result, Run)
    assert result.step == 4
    assert handle.is_running() is False
    assert handle.is_alive() is False


def test_ps_snapshot_and_run_status_attribute_and_mapping_access(tmp_path):
    """Verify PsSnapshot.runs normalization and RunStatus attribute/mapping access."""
    run_dir = tmp_path / "ps_run"
    hg.train(
        hg.packaged("gaussian-grid"),
        run_dir=run_dir,
        steps=3,
        devices="cpu",
        previews=0,
    )

    ps = hg.ps(all_runs=True, root=tmp_path)
    assert isinstance(ps, PsSnapshot)
    assert len(ps) >= 1
    assert ps.total_count >= 1

    for run in ps.runs:
        assert isinstance(run, RunStatus)
        assert isinstance(run, dict)
        # Attribute access
        assert hasattr(run, "slug")
        assert hasattr(run, "step")
        assert hasattr(run, "status")
        assert hasattr(run, "device")
        assert hasattr(run, "pid")
        # Key access parity
        assert run["slug"] == run.slug
        assert run["step"] == run.step
        assert run["status"] == run.status


def test_hg_export_positional_and_keyword_target(tmp_path):
    """Verify hg.export works with both positional output and explicit output keyword."""
    run_dir = tmp_path / "export_run"
    hg.train(
        hg.packaged("gaussian-grid"),
        run_dir=run_dir,
        steps=2,
        devices="cpu",
        previews=0,
    )

    # Positional target
    out_pos = tmp_path / "generator_pos.pt"
    p1 = hg.export(run_dir, out_pos, format="bundle")
    assert Path(p1).is_file()

    # Keyword target
    out_kw = tmp_path / "generator_kw.pt"
    p2 = hg.export(run_dir, output=out_kw, format="bundle")
    assert Path(p2).is_file()

    # Standalone loading of both
    gen1 = hg.load_bundle(p1, device="cpu")
    gen2 = hg.load_bundle(p2, device="cpu")
    assert gen1.sample(count=2, seed=42).shape[0] == 2
    assert gen2.sample(count=2, seed=42).shape[0] == 2

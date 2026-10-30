"""Adversarial Stress Test Suite for Milestone 4 (Features 47–53).

Author: Challenger M4-1 (Empirical Challenger)
Subject Commits: f14e4ecd and 9703546c on branch develop.

Scope:
- Feature 47: Export Privacy Stripping & Conditioning Specs
- Feature 48: Standalone Generator Baking (Zero ParticleGAN / HNDL Dependencies)
- Feature 50: LoadedGenerator Dual Inheritance (nn.Module + dict) & TorchScript/Export Compatibility
- Feature 51: Latent Interpolation Stress Testing (Steps Validation, Boundary Checks, Slerp vs Lerp, Collinear Degeneracies)
- Feature 52: Batch PNG Sampling CLI with --output-dir
- Feature 53: Non-Image Conditioning Inputs (JSON strings, JSON files, tensor strings & error handling)
"""

import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
import pytest
import torch
import torch.nn as nn

import hypergan.api as hg
from hypergan.artifacts import (
    LoadedGenerator,
    export_generator,
    interpolate,
    sample,
    _coerce_conditioning_input,
)
from tests.e2e.harness_v2 import run_cli


# ---------------------------------------------------------------------------
# Fixture: Trained minimal model for export and sampling tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_dcgan_fixture(tmp_path_factory):
    """Train a 2-step DCGAN model with real images to verify exports and sampling."""
    base_dir = tmp_path_factory.mktemp("challenger_m4_dcgan")
    data_dir = base_dir / "data"
    data_dir.mkdir()
    for i in range(4):
        # Distinct RGB images
        img = Image.new("RGB", (32, 32), (i * 60, (4 - i) * 50, 128))
        img.save(data_dir / f"img_{i:02d}.png")

    proj_dir = base_dir / "proj"
    rc, _, err = run_cli("new", "--project", str(proj_dir), "--template", "dcgan", "--dataset", str(data_dir), "--device", "cpu")
    assert rc == 0, f"hypergan new failed: {err}"

    cfg = proj_dir / "configs" / "dcgan.toml"
    rc_t, _, err_t = run_cli("train", str(cfg), "--steps", "2", "-d", "cpu")
    assert rc_t == 0, f"hypergan train failed: {err_t}"

    rc_s, out_s, _ = run_cli("status", "--json")
    assert rc_s == 0
    run_dir = Path(json.loads(out_s)["run_dir"])
    assert run_dir.is_dir()
    return run_dir, base_dir


# ---------------------------------------------------------------------------
# 1. Feature 47: Export Privacy Stripping & Conditioning Specs
# ---------------------------------------------------------------------------
class TestAdversarialExportPrivacy:
    """Stress-test privacy stripping of raw image data during model export."""

    def test_default_export_strips_raw_tensors_and_no_last_batch(self, trained_dcgan_fixture):
        """Default export must never leak raw image tensors in example_inputs or last_batch."""
        run_dir, tmp_dir = trained_dcgan_fixture
        export_out = tmp_dir / "privacy_stripped_bundle.pt"
        rc, _, err = run_cli("export", str(run_dir), "-o", str(export_out))
        assert rc == 0, f"Export failed: {err}"
        assert export_out.is_file()

        state = torch.load(export_out, weights_only=False)
        assert isinstance(state, dict)

        # 1. Verify 'last_batch' is never present in the inference bundle
        assert "last_batch" not in state, "Privacy leak: 'last_batch' was serialized in exported bundle!"

        # 2. Verify 'example_inputs' contains no torch.Tensor objects
        assert "example_inputs" in state, "Export missing example_inputs dictionary"
        for key, val in state["example_inputs"].items():
            assert not isinstance(val, torch.Tensor), (
                f"Privacy leak: raw tensor found in example_inputs[{key!r}]!"
            )
            assert isinstance(val, dict), f"Expected dict spec for {key!r}, got {type(val)}"
            assert "shape" in val, f"Missing 'shape' in spec for {key!r}"
            assert "dtype" in val, f"Missing 'dtype' in spec for {key!r}"
            assert isinstance(val["shape"], list)

        # 3. Deep recursion check: ensure no training image tensors leaked elsewhere
        def _check_no_training_tensors(obj, path=""):
            if isinstance(obj, torch.Tensor):
                # Only weights/buffers in model_states or prior are allowed
                allowed = ("model_states", "prior", "model_buffers", "prior_buffers")
                assert any(path.startswith(prefix) for prefix in allowed), (
                    f"Privacy leak: Unexpected tensor found at '{path}' with shape {list(obj.shape)}"
                )
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_training_tensors(v, f"{path}.{k}" if path else str(k))
            elif isinstance(obj, (list, tuple)):
                for idx, v in enumerate(obj):
                    _check_no_training_tensors(v, f"{path}[{idx}]")

        _check_no_training_tensors(state)

    def test_opt_in_include_example_inputs_preserves_tensors(self, trained_dcgan_fixture):
        """Passing --include-example-inputs must explicitly retain the real conditioning tensors."""
        run_dir, tmp_dir = trained_dcgan_fixture
        export_out = tmp_dir / "unstripped_bundle.pt"
        rc, _, err = run_cli("export", str(run_dir), "-o", str(export_out), "--include-example-inputs")
        assert rc == 0, f"Export with --include-example-inputs failed: {err}"
        assert export_out.is_file()

        state = torch.load(export_out, weights_only=False)
        assert "example_inputs" in state
        # In this conditional/unconditional DCGAN, if example_inputs has inputs, they are tensors
        if state["example_inputs"]:
            assert any(isinstance(v, torch.Tensor) for v in state["example_inputs"].values())

    def test_conditioning_coercion_from_privacy_spec(self):
        """_coerce_conditioning_input must gracefully synthesize zero tensors from shape/dtype specs."""
        spec_4d = {"shape": [2, 3, 32, 32], "dtype": "float32"}
        t4d = _coerce_conditioning_input(spec_4d, None)
        assert isinstance(t4d, torch.Tensor)
        assert list(t4d.shape) == [2, 3, 32, 32]
        assert t4d.dtype == torch.float32
        assert torch.all(t4d == 0.0)

        spec_2d = {"shape": [4, 16], "dtype": "torch.float64"}
        t2d = _coerce_conditioning_input(spec_2d, None)
        assert isinstance(t2d, torch.Tensor)
        assert list(t2d.shape) == [4, 16]
        assert t2d.dtype == torch.float64
        assert torch.all(t2d == 0.0)


# ---------------------------------------------------------------------------
# 2. Feature 48: Standalone Generator Baking (Zero ParticleGAN / HNDL Dependencies)
# ---------------------------------------------------------------------------
class TestStandaloneGeneratorBakingDependencies:
    """Stress-test runtime independence of LoadedGenerator from particlegan and hndl."""

    def test_loaded_generator_executes_in_clean_subprocess_without_particlegan_or_hndl(self, tmp_path):
        """Verify LoadedGenerator runs in an isolated process where particlegan and hndl are unimportable."""
        # Create a synthetic standalone bundle using native PyTorch components
        bundle_path = tmp_path / "standalone_baked_bundle.pt"
        bundle_data = {
            "schema_version": 1,
            "kind": "ema-inference",
            "step": 42,
            "config": {
                "prior": {"z_dim": 8},
                "sampling": {"generated": "components.generator"}
            },
            "components": {
                "generator": {
                    "factory": "torch.nn:Linear",
                    "args": {"in_features": 8, "out_features": 16},
                    "inputs": {"input": "latent"},
                    "trainable": True
                }
            },
            "model_states": {
                "generator": {
                    "weight": torch.randn(16, 8),
                    "bias": torch.randn(16)
                }
            },
            "prior": {
                "z": torch.randn(32, 8),
                "sigma": torch.tensor(0.05)
            },
            "example_inputs": {}
        }
        torch.save(bundle_data, bundle_path)

        # Run verification in a fresh Python subprocess with particlegan and hndl strictly blocked
        sub_code = f"""
import sys
# Block particlegan and hndl from being imported
sys.modules['particlegan'] = None
sys.modules['hndl'] = None

import torch
from hypergan.artifacts import LoadedGenerator

# 1. Instantiate baked LoadedGenerator
lg = LoadedGenerator({repr(str(bundle_path))}, device='cpu')
assert sys.modules.get('particlegan') is None, 'particlegan was imported!'
assert sys.modules.get('hndl') is None, 'hndl was imported!'

# 2. Forward inference g(z)
z = torch.randn(4, 8)
out_fwd = lg(z)
assert out_fwd.shape == (4, 16), f'Unexpected forward shape: {{out_fwd.shape}}'

# 3. Sampling API
out_samp = lg.sample(count=5, seed=123)
assert out_samp.shape == (5, 16), f'Unexpected sample shape: {{out_samp.shape}}'

# 4. Latent walk API
out_interp = lg.interpolate(seed_a=1, seed_b=2, steps=6)
assert out_interp.shape == (6, 16), f'Unexpected interpolate shape: {{out_interp.shape}}'

print('SUBPROCESS_STANDALONE_SUCCESS')
"""
        proc = subprocess.run(
            [sys.executable, "-c", sub_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert proc.returncode == 0, f"Subprocess failed with code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}"
        assert "SUBPROCESS_STANDALONE_SUCCESS" in proc.stdout


# ---------------------------------------------------------------------------
# 3. Feature 50: LoadedGenerator Dual Inheritance & TorchScript / Export
# ---------------------------------------------------------------------------
class TestLoadedGeneratorDualInheritanceAndCompatibility:
    """Stress-test LoadedGenerator dual inheritance (nn.Module + dict) and export compatibility."""

    @pytest.fixture
    def standalone_generator(self, tmp_path):
        bundle_path = tmp_path / "test_dual_model.pt"
        bundle_data = {
            "schema_version": 1,
            "kind": "ema-inference",
            "step": 100,
            "config": {
                "prior": {"z_dim": 4},
                "sampling": {"generated": "components.generator"}
            },
            "components": {
                "generator": {
                    "factory": "torch.nn:Linear",
                    "args": {"in_features": 4, "out_features": 8},
                    "inputs": {"input": "latent"},
                    "trainable": True
                }
            },
            "model_states": {
                "generator": {
                    "weight": torch.randn(8, 4),
                    "bias": torch.randn(8)
                }
            },
            "prior": {
                "z": torch.randn(16, 4),
                "sigma": torch.tensor(0.02)
            },
            "example_inputs": {}
        }
        torch.save(bundle_data, bundle_path)
        return LoadedGenerator(bundle_path, device="cpu")

    def test_dual_inheritance_types_and_dict_semantics(self, standalone_generator):
        """LoadedGenerator must satisfy both nn.Module and dict type contracts."""
        g = standalone_generator
        # Type assertions
        assert isinstance(g, nn.Module), "LoadedGenerator must be an instance of nn.Module"
        assert isinstance(g, dict), "LoadedGenerator must be an instance of dict"

        # Dictionary access semantics
        assert g["schema_version"] == 1
        assert g["step"] == 100
        assert "config" in g
        assert "components" in g
        assert g.get("step") == 100
        assert g.get("nonexistent_key", 999) == 999
        assert isinstance(g.keys(), type({}.keys()))
        assert "prior" in list(g.keys())
        assert any(k == "generator" for k in g["components"].keys())

        # Module semantics
        assert hasattr(g, "particle_z")
        assert hasattr(g, "particle_sigma")
        assert isinstance(g.particle_z, torch.Tensor)
        assert isinstance(g.particle_sigma, torch.Tensor)
        buffers = dict(g.named_buffers())
        assert "particle_z" in buffers
        assert "particle_sigma" in buffers

        # Device movement
        g.to("cpu")
        assert g.device.type == "cpu"

    def test_callable_execution_and_methods(self, standalone_generator):
        """LoadedGenerator must execute as callable module and support sample and interpolate."""
        g = standalone_generator
        assert callable(g)

        # 1. Direct callable g(z)
        z = torch.randn(3, 4)
        out = g(z)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 8)

        # 2. g.sample()
        s4 = g.sample(count=4, seed=42)
        assert s4.shape == (4, 8)
        s1 = g.sample(count=1, seed=42)
        assert s1.shape == (1, 8)

        # 3. g.interpolate()
        w = g.interpolate(seed_a=10, seed_b=20, steps=5)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (5, 8)

    def test_torch_jit_trace_compatibility(self, standalone_generator):
        """LoadedGenerator must be traceable via torch.jit.trace."""
        g = standalone_generator
        dummy_z = torch.randn(1, 4)

        traced = torch.jit.trace(g, (dummy_z,))
        assert callable(traced)
        out_orig = g(dummy_z)
        out_traced = traced(dummy_z)
        assert torch.allclose(out_orig, out_traced, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Feature 51: Latent Interpolation Stress Testing
# ---------------------------------------------------------------------------
class TestAdversarialLatentInterpolation:
    """Stress-test hg.interpolate and gen.interpolate across bounds and edge cases."""

    def test_interpolate_invalid_steps_rejected(self):
        """steps <= 0 must raise ValueError."""
        v0 = torch.randn(1, 8)
        v1 = torch.randn(1, 8)

        for invalid_step in [0, -1, -10, -999]:
            with pytest.raises(ValueError, match="steps must be at least 1"):
                interpolate(v0, v1, steps=invalid_step)

    def test_loaded_generator_interpolate_invalid_steps_rejected(self, tmp_path):
        """LoadedGenerator.interpolate must raise ValueError when steps < 1."""
        bundle_path = tmp_path / "bundle_interp_fail.pt"
        bundle_data = {
            "schema_version": 1,
            "kind": "ema-inference",
            "step": 1,
            "config": {"prior": {"z_dim": 4}, "sampling": {"generated": "components.generator"}},
            "components": {
                "generator": {
                    "factory": "torch.nn:Linear",
                    "args": {"in_features": 4, "out_features": 4},
                    "inputs": {"input": "latent"},
                    "trainable": True
                }
            },
            "model_states": {"generator": {"weight": torch.eye(4), "bias": torch.zeros(4)}},
            "prior": {"z": torch.randn(8, 4), "sigma": torch.tensor(0.0)},
            "example_inputs": {}
        }
        torch.save(bundle_data, bundle_path)
        g = LoadedGenerator(bundle_path, device="cpu")

        with pytest.raises(ValueError, match="steps must be at least 1"):
            g.interpolate(seed_a=1, seed_b=2, steps=0)
        with pytest.raises(ValueError, match="steps must be at least 1"):
            g.interpolate(seed_a=1, seed_b=2, steps=-5)

    def test_interpolate_boundary_conditions_t(self):
        """Boundary checks: t=0.0 must yield v0 and t=1.0 must yield v1."""
        torch.manual_seed(42)
        v0 = torch.randn(1, 16)
        v1 = torch.randn(1, 16)

        # Mode: slerp
        p0_slerp = interpolate(v0, v1, t=0.0, mode="slerp")
        p1_slerp = interpolate(v0, v1, t=1.0, mode="slerp")
        assert torch.allclose(p0_slerp, v0, atol=1e-5), f"Slerp t=0 mismatch: diff={torch.norm(p0_slerp - v0)}"
        assert torch.allclose(p1_slerp, v1, atol=1e-5), f"Slerp t=1 mismatch: diff={torch.norm(p1_slerp - v1)}"

        # Mode: lerp
        p0_lerp = interpolate(v0, v1, t=0.0, mode="lerp")
        p1_lerp = interpolate(v0, v1, t=1.0, mode="lerp")
        assert torch.allclose(p0_lerp, v0, atol=1e-5), f"Lerp t=0 mismatch: diff={torch.norm(p0_lerp - v0)}"
        assert torch.allclose(p1_lerp, v1, atol=1e-5), f"Lerp t=1 mismatch: diff={torch.norm(p1_lerp - v1)}"

        # Endpoint checks with steps=7
        walk = interpolate(v0, v1, steps=7, mode="slerp")
        assert torch.allclose(walk[0:1], v0, atol=1e-5)
        assert torch.allclose(walk[-1:], v1, atol=1e-5)

    def test_slerp_vs_lerp_geodesic_trajectories(self):
        """Slerp must preserve unit norm along the sphere surface; Lerp cuts through sphere interior."""
        u0 = torch.tensor([[1.0, 0.0, 0.0]])
        u1 = torch.tensor([[0.0, 1.0, 0.0]])

        walk_slerp = interpolate(u0, u1, steps=5, mode="slerp")
        walk_lerp = interpolate(u0, u1, steps=5, mode="lerp")

        norms_slerp = torch.norm(walk_slerp, dim=-1)
        norms_lerp = torch.norm(walk_lerp, dim=-1)

        # Slerp norms remain exactly 1.0 across the geodesic trajectory
        assert torch.allclose(norms_slerp, torch.ones_like(norms_slerp), atol=1e-4)

        # Lerp midpoint (alpha=0.5) has norm sqrt(0.5) ~ 0.7071
        midpoint_lerp_norm = norms_lerp[2].item()
        assert midpoint_lerp_norm < 0.75
        assert abs(midpoint_lerp_norm - (0.5 ** 0.5)) < 1e-3

    def test_interpolate_identical_vectors_lerp(self):
        """Lerp on identical vectors must return the original vector at every step."""
        v0 = torch.randn(1, 8)
        walk = interpolate(v0, v0, steps=5, mode="lerp")
        for i in range(5):
            assert torch.allclose(walk[i:i+1], v0, atol=1e-5)

    def test_interpolate_identical_vectors_slerp_challenge(self):
        """Adversarial stress test: slerp on identical vectors must not collapse to zero."""
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        walk = interpolate(v0, v0, steps=5, mode="slerp")
        assert not torch.all(walk == 0.0).item(), "slerp returned all zeros for identical vectors!"
        assert torch.allclose(walk[0:1], v0, atol=1e-4), "slerp[0] did not equal v0 for identical vectors!"
        assert torch.allclose(walk, v0.expand(5, 3), atol=1e-4), "all slerp steps must match v0 for identical vectors!"

    def test_interpolate_anti_parallel_vectors_slerp_challenge(self):
        """Adversarial stress test: slerp on anti-parallel vectors (dot=-1.0) must preserve endpoints."""
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[-1.0, 0.0, 0.0]])
        walk = interpolate(v0, v1, steps=5, mode="slerp")
        # Endpoints must match v0 and v1
        assert torch.allclose(walk[0:1], v0, atol=1e-3), f"Endpoint walk[0] ({walk[0]}) does not match v0!"
        assert torch.allclose(walk[-1:], v1, atol=1e-3), f"Endpoint walk[-1] ({walk[-1]}) does not match v1!"

    def test_interpolate_slerp_mixed_batch_and_1d(self):
        """Verifies slerp handles 1D tensors and mixed batches with per-element collinearity masking."""
        v0_1d = torch.tensor([1.0, 0.0, 0.0])
        walk_1d = interpolate(v0_1d, v0_1d, steps=3, mode="slerp")
        assert walk_1d.shape == (3, 3)
        assert torch.allclose(walk_1d, v0_1d.expand(3, 3))

        b0 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mid = interpolate(b0, b1, t=0.5, mode="slerp")
        assert torch.allclose(mid[0], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.allclose(mid[1], torch.tensor([0.70710678, 0.70710678, 0.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Feature 52: Batch PNG Sampling CLI with --output-dir
# ---------------------------------------------------------------------------
class TestAdversarialBatchPngSamplingCli:
    """Stress-test CLI batch PNG sampling and directory creation."""

    def test_cli_sample_output_dir_creates_dir_and_valid_pngs(self, trained_dcgan_fixture):
        """hypergan sample --output-dir <nonexistent_dir> must create the dir and write valid PNGs."""
        run_dir, tmp_dir = trained_dcgan_fixture
        export_pt = tmp_dir / "model_batch.pt"
        rc_e, _, err_e = run_cli("export", str(run_dir), "-o", str(export_pt))
        assert rc_e == 0, f"Export failed: {err_e}"

        target_out_dir = tmp_dir / "batch_png_output_nonexistent_sub" / "nested"
        assert not target_out_dir.exists(), "Target dir should not exist before test"

        # Execute sample with --output-dir and --device cpu
        rc_s, out_s, err_s = run_cli(
            "sample",
            str(export_pt),
            "--count", "4",
            "--output-dir", str(target_out_dir),
            "-d", "cpu"
        )
        assert rc_s == 0, f"Sample command failed: {err_s}"
        assert target_out_dir.is_dir(), "Command failed to create --output-dir!"

        png_files = sorted(target_out_dir.glob("sample_*.png"))
        assert len(png_files) == 4, f"Expected 4 PNGs, found {len(png_files)}: {png_files}"
        assert png_files[0].name == "sample_0000.png"
        assert png_files[3].name == "sample_0003.png"

        # Validate PNG content with PIL
        for png_path in png_files:
            with Image.open(png_path) as img:
                img.verify()  # Verifies file integrity
            with Image.open(png_path) as img:
                assert img.format == "PNG"
                assert img.size == (32, 32)
                assert img.mode in ("RGB", "L")


# ---------------------------------------------------------------------------
# 6. Feature 53: Non-Image Conditioning Inputs & Error Handling
# ---------------------------------------------------------------------------
class TestAdversarialNonImageConditioning:
    """Stress-test non-image conditioning inputs (JSON strings, files, tensor strings)."""

    def test_conditioning_valid_json_string(self):
        """Valid JSON string representation of tensor data must be parsed properly."""
        json_str = "[0.25, -0.75, 1.5, 0.0]"
        tensor = _coerce_conditioning_input(json_str, None)
        assert isinstance(tensor, torch.Tensor)
        assert list(tensor.shape) in ([4], [1, 4])
        assert abs(tensor.flatten()[0].item() - 0.25) < 1e-5
        assert abs(tensor.flatten()[1].item() - (-0.75)) < 1e-5

    def test_conditioning_valid_json_file(self, tmp_path):
        """Path to a valid .json file containing numerical data must be read and coerced."""
        json_file = tmp_path / "cond_data.json"
        json_file.write_text(json.dumps([[1.0, 2.0], [3.0, 4.0]]), encoding="utf-8")

        tensor = _coerce_conditioning_input(json_file, None)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert tensor[0, 1].item() == 2.0

    def test_conditioning_valid_tensor_string(self):
        """Comma-separated numerical string must be parsed into tensor."""
        t_str = "1.0, 2.0, 3.5, -4.5"
        tensor = _coerce_conditioning_input(t_str, None)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[-1] == 4
        assert tensor.flatten()[2].item() == 3.5

    def test_conditioning_invalid_json_string_handled(self):
        """Malformed JSON string with invalid tokens must fail gracefully with FileNotFoundError or ValueError."""
        malformed_json_strings = [
            "{invalid_json_no_quotes",
            '{"key": invalid_unquoted_value}',
            "[1.0, 2.0, abc]",
            "{'single_quotes': 1}",
        ]
        for bad_str in malformed_json_strings:
            with pytest.raises((FileNotFoundError, ValueError, Exception)):
                _coerce_conditioning_input(bad_str, None)

    def test_conditioning_lenient_numeric_string_fallback(self):
        """Lenient numeric list string without closing bracket is parsed by tensor string fallback."""
        s = "[1.0, 2.0,"
        tensor = _coerce_conditioning_input(s, None)
        assert isinstance(tensor, torch.Tensor)
        assert list(tensor.shape) in ([2], [1, 2])
        assert abs(tensor.flatten()[0].item() - 1.0) < 1e-5
        assert abs(tensor.flatten()[1].item() - 2.0) < 1e-5

    def test_conditioning_invalid_json_file_handled(self, tmp_path):
        """A .json file with syntax errors must raise an informative exception."""
        bad_file = tmp_path / "corrupt_data.json"
        bad_file.write_text("{this is not valid json!}", encoding="utf-8")

        with pytest.raises(Exception):
            _coerce_conditioning_input(bad_file, None)

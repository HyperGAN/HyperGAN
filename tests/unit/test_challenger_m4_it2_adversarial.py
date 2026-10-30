"""Milestone 4 Iteration 2 Adversarial Challenge Test Suite.

Empirical Challenger 1: Adversarial Stress-Testing on Export, Standalone Generators,
and Latent Interpolation (Slerp Collinearity, Anti-Parallel Stability, Mixed Batches,
Device Transfers, Seed Determinism, and Batch Generation).
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import hypergan.api as hg
from hypergan.artifacts import LoadedGenerator, interpolate


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_dummy_bundle(tmp_path: Path, z_dim: int = 8, out_features: int = 16, num_particles: int = 32) -> Path:
    """Create a synthetic standalone inference bundle for testing."""
    bundle_data = {
        "schema_version": 1,
        "kind": "ema-inference",
        "step": 42,
        "config": {
            "prior": {"z_dim": z_dim},
            "sampling": {"generated": "components.generator"}
        },
        "components": {
            "generator": {
                "factory": "torch.nn:Linear",
                "args": {"in_features": z_dim, "out_features": out_features},
                "inputs": {"input": "latent"},
                "trainable": True
            }
        },
        "model_states": {
            "generator": {
                "weight": torch.randn(out_features, z_dim),
                "bias": torch.randn(out_features)
            }
        },
        "prior": {
            "z": torch.randn(num_particles, z_dim),
            "sigma": torch.tensor(0.05)
        },
        "example_inputs": {}
    }
    pt_path = tmp_path / "synthetic_bundle.pt"
    torch.save(bundle_data, pt_path)
    return pt_path


def make_gaussian_prior_bundle(tmp_path: Path, z_dim: int = 8, out_features: int = 16) -> Path:
    """Create a synthetic standalone inference bundle without particle prior (Gaussian prior)."""
    bundle_data = {
        "schema_version": 1,
        "kind": "ema-inference",
        "step": 10,
        "config": {
            "prior": {"z_dim": z_dim},
            "sampling": {"generated": "components.generator"}
        },
        "components": {
            "generator": {
                "factory": "torch.nn:Linear",
                "args": {"in_features": z_dim, "out_features": out_features},
                "inputs": {"input": "latent"},
                "trainable": True
            }
        },
        "model_states": {
            "generator": {
                "weight": torch.randn(out_features, z_dim),
                "bias": torch.randn(out_features)
            }
        },
        "prior": {},  # No particle table
        "example_inputs": {}
    }
    pt_path = tmp_path / "gaussian_bundle.pt"
    torch.save(bundle_data, pt_path)
    return pt_path


# ---------------------------------------------------------------------------
# 1. Adversarial Stress-Testing on Slerp & Latent Interpolation
# ---------------------------------------------------------------------------

class TestAdversarialSlerpDegeneracies:
    """Adversarial stress-testing of _slerp and interpolate under mathematical edge cases."""

    def test_slerp_identical_vectors_single_and_batch_no_nan_or_zero_collapse(self):
        """Stress-test exactly identical vectors (v0 == v1): must never collapse to 0 or produce NaN/Inf."""
        # 1D single vector
        v_1d = torch.randn(16)
        walk_1d = hg.interpolate(v_1d, v_1d, steps=10, mode="slerp")
        assert not torch.isnan(walk_1d).any(), "NaN found in 1D identical slerp!"
        assert not torch.isinf(walk_1d).any(), "Inf found in 1D identical slerp!"
        assert walk_1d.shape == (10, 16)
        for s in range(10):
            assert torch.allclose(walk_1d[s], v_1d, atol=1e-6), f"Step {s} collapsed or drifted: {walk_1d[s]}"

        # 2D batch of identical vectors
        v_2d = torch.randn(5, 16)
        walk_2d = hg.interpolate(v_2d, v_2d, steps=8, mode="slerp")
        assert not torch.isnan(walk_2d).any(), "NaN found in 2D identical slerp!"
        assert walk_2d.shape == (8 * 5, 16)
        walk_reshaped = walk_2d.view(8, 5, 16)
        for s in range(8):
            assert torch.allclose(walk_reshaped[s], v_2d, atol=1e-6), f"Step {s} drifted from v_2d"

    def test_slerp_zero_vectors_and_near_zero_magnitudes(self):
        """Stress-test with zero vectors and near-zero vectors smaller than eps (1e-7)."""
        # Zero vectors
        v_zero = torch.zeros(8)
        walk_zero = hg.interpolate(v_zero, v_zero, steps=5, mode="slerp")
        assert not torch.isnan(walk_zero).any()
        assert torch.all(walk_zero == 0.0)

        # Sub-epsilon vectors
        v_subeps = torch.full((8,), 1e-9)
        walk_subeps = hg.interpolate(v_subeps, v_subeps, steps=5, mode="slerp")
        assert not torch.isnan(walk_subeps).any()
        assert not torch.isinf(walk_subeps).any()

        # Large magnitude vectors (1e6)
        v_huge = torch.randn(8) * 1e6
        walk_huge = hg.interpolate(v_huge, v_huge, steps=5, mode="slerp")
        assert not torch.isnan(walk_huge).any()
        assert torch.allclose(walk_huge[0], v_huge, atol=1e-2)

    def test_slerp_anti_parallel_vectors_monotonic_continuity_and_midpoint_stability(self):
        """Stress-test anti-parallel vectors (v0 == -v1): verify continuity and midpoint stability without sign flips."""
        v0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v1 = -v0
        steps = 11  # Odd steps guarantees exact alpha = 0.5 at index 5
        walk = hg.interpolate(v0, v1, steps=steps, mode="slerp")

        assert not torch.isnan(walk).any(), "NaN in anti-parallel slerp!"
        assert not torch.isinf(walk).any(), "Inf in anti-parallel slerp!"
        assert walk.shape == (steps, 4)

        # Midpoint stability: at step index 5 (alpha=0.5), value must strictly equal 0.0
        midpoint = walk[5]
        assert torch.allclose(midpoint, torch.zeros(4), atol=1e-6), f"Midpoint was not zero: {midpoint}"

        # Monotonic continuity: projection along v0 must strictly decrease monotonically
        projections = [(walk[s] * v0).sum().item() for s in range(steps)]
        for s in range(steps - 1):
            assert projections[s] > projections[s + 1], (
                f"Projection not strictly decreasing at step {s} -> {s + 1}: {projections[s]} <= {projections[s + 1]}"
            )
        assert abs(projections[0] - 1.0) < 1e-5
        assert abs(projections[-1] - (-1.0)) < 1e-5

    def test_slerp_mixed_batch_all_degeneracies_simultaneously(self):
        """Stress-test a single batch containing collinear, anti-parallel, orthogonal, acute, obtuse, and zeros."""
        e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        e2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        acute = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
        obtuse = torch.tensor([math.cos(3 * math.pi / 4), math.sin(3 * math.pi / 4), 0.0, 0.0])
        near_collinear = torch.tensor([math.cos(0.01), math.sin(0.01), 0.0, 0.0])
        near_antiparallel = torch.tensor([math.cos(math.pi - 0.01), math.sin(math.pi - 0.01), 0.0, 0.0])

        v0_batch = torch.stack([
            e1,                # 0: identical
            e1,                # 1: orthogonal
            e1,                # 2: anti-parallel
            e1,                # 3: acute (45 deg)
            e1,                # 4: obtuse (135 deg)
            torch.zeros(4),    # 5: zero vector
            e1,                # 6: near-collinear (|dot| > 0.9995)
            e1                 # 7: near-antiparallel (|dot| > 0.9995)
        ])
        v1_batch = torch.stack([
            e1,
            e2,
            -e1,
            acute,
            obtuse,
            torch.zeros(4),
            near_collinear,
            near_antiparallel
        ])

        steps = 9  # Midpoint at index 4
        walk = hg.interpolate(v0_batch, v1_batch, steps=steps, mode="slerp")
        assert not torch.isnan(walk).any(), "NaN in mixed batch slerp!"
        assert not torch.isinf(walk).any(), "Inf in mixed batch slerp!"
        assert walk.shape == (steps * 8, 4)

        walk_by_step = walk.view(steps, 8, 4)

        # 0: Identical pair stays equal to e1 at all steps
        for s in range(steps):
            assert torch.allclose(walk_by_step[s, 0], e1, atol=1e-5), f"Row 0 step {s} failed"

        # 1: Orthogonal pair preserves unit norm at all steps (geodesic arc)
        for s in range(steps):
            norm = torch.norm(walk_by_step[s, 1]).item()
            assert abs(norm - 1.0) < 1e-4, f"Row 1 step {s} norm deviation: {norm}"

        # 2: Anti-parallel pair midpoint is zero
        assert torch.allclose(walk_by_step[4, 2], torch.zeros(4), atol=1e-5), "Row 2 midpoint not zero"

        # 3: Acute pair preserves unit norm
        for s in range(steps):
            norm = torch.norm(walk_by_step[s, 3]).item()
            assert abs(norm - 1.0) < 1e-4, f"Row 3 step {s} norm deviation: {norm}"

        # 4: Obtuse pair preserves unit norm
        for s in range(steps):
            norm = torch.norm(walk_by_step[s, 4]).item()
            assert abs(norm - 1.0) < 1e-4, f"Row 4 step {s} norm deviation: {norm}"

        # 5: Zero vectors stay zero
        assert torch.all(walk_by_step[:, 5] == 0.0), "Row 5 zero vector failed"

        # 6: Near-collinear lerps safely
        assert not torch.isnan(walk_by_step[:, 6]).any()

        # 7: Near-antiparallel lerps safely
        assert not torch.isnan(walk_by_step[:, 7]).any()

    def test_1d_vs_2d_tensor_semantics_and_single_step(self):
        """Stress-test tensor shapes: 1D vs 2D, t parameter, and steps=1."""
        v0_1d = torch.randn(8)
        v1_1d = torch.randn(8)

        # 1D steps=5
        out_1d = hg.interpolate(v0_1d, v1_1d, steps=5)
        assert out_1d.shape == (5, 8)

        # 1D steps=1 (midpoint)
        out_1d_s1 = hg.interpolate(v0_1d, v1_1d, steps=1)
        assert out_1d_s1.shape == (8,)

        # 1D explicit t=0.5
        out_1d_t = hg.interpolate(v0_1d, v1_1d, t=0.5)
        assert out_1d_t.shape == (8,)
        assert torch.allclose(out_1d_s1, out_1d_t, atol=1e-5)

        # 2D (1, 8)
        v0_2d_1 = v0_1d.unsqueeze(0)
        v1_2d_1 = v1_1d.unsqueeze(0)
        out_2d_1 = hg.interpolate(v0_2d_1, v1_2d_1, steps=5)
        assert out_2d_1.shape == (5, 8)

        # 2D (3, 8)
        v0_2d_3 = torch.randn(3, 8)
        v1_2d_3 = torch.randn(3, 8)
        out_2d_3 = hg.interpolate(v0_2d_3, v1_2d_3, steps=5)
        assert out_2d_3.shape == (15, 8)

        # 2D explicit t=0.3
        out_2d_t = hg.interpolate(v0_2d_3, v1_2d_3, t=0.3)
        assert out_2d_t.shape == (3, 8)

    def test_interpolate_invalid_steps_rejected(self):
        """Verify invalid steps (< 1) raise ValueError."""
        v0 = torch.randn(4)
        v1 = torch.randn(4)
        with pytest.raises(ValueError, match="steps must be at least 1"):
            hg.interpolate(v0, v1, steps=0)
        with pytest.raises(ValueError, match="steps must be at least 1"):
            hg.interpolate(v0, v1, steps=-3)

    def test_slerp_cuda_execution_preserves_device(self):
        """Verify slerp executes on CUDA without CPU sync or device changes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        v0 = torch.randn(8, device="cuda:0")
        v1 = torch.randn(8, device="cuda:0")
        walk = hg.interpolate(v0, v1, steps=7, mode="slerp")
        assert walk.device.type == "cuda"
        assert walk.device.index == 0
        assert walk.shape == (7, 8)


# ---------------------------------------------------------------------------
# 2. Stress-Testing StandaloneGenerator (LoadedGenerator) and load_bundle
# ---------------------------------------------------------------------------

class TestLoadedGeneratorDeviceTransfersAndDeterminism:
    """Stress-test LoadedGenerator across devices, seed determinism, and batch generation."""

    def test_device_transfers_cpu_cuda0_cuda1_and_back(self, tmp_path):
        """Stress-test device movement across CPU, CUDA:0, and CUDA:1."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        bundle_file = make_dummy_bundle(tmp_path, z_dim=8, out_features=16)

        # 1. Load on CPU
        lg = hg.load_bundle(bundle_file, device="cpu")
        assert lg.device == torch.device("cpu")
        z_fixed = torch.randn(4, 8)
        out_fwd_cpu = lg(z_fixed)
        out_sample_cpu1 = lg.sample(count=4, seed=123)
        out_sample_cpu2 = lg.sample(count=4, seed=123)
        assert torch.equal(out_sample_cpu1, out_sample_cpu2), "CPU sample determinism failed"

        # 2. Move to CUDA:0
        lg.to("cuda:0")
        assert lg.device == torch.device("cuda:0")
        assert lg.particle_z.device.type == "cuda" and lg.particle_z.device.index == 0
        assert lg.particle_sigma.device.type == "cuda" and lg.particle_sigma.device.index == 0
        out_fwd_cuda0 = lg(z_fixed.to("cuda:0"))
        assert out_fwd_cuda0.device.type == "cuda" and out_fwd_cuda0.device.index == 0
        # Forward pass numerical equivalence given identical latent input
        assert torch.allclose(out_fwd_cpu, out_fwd_cuda0.cpu(), atol=1e-5)

        # CUDA sample determinism
        out_sample_cuda1 = lg.sample(count=4, seed=123)
        out_sample_cuda2 = lg.sample(count=4, seed=123)
        assert torch.equal(out_sample_cuda1, out_sample_cuda2), "CUDA sample determinism failed"

        # 3. Multi-GPU move to CUDA:1 if available
        if torch.cuda.device_count() >= 2:
            lg.to("cuda:1")
            assert lg.device == torch.device("cuda:1")
            assert lg.particle_z.device.type == "cuda" and lg.particle_z.device.index == 1
            out_fwd_cuda1 = lg(z_fixed.to("cuda:1"))
            assert out_fwd_cuda1.device.type == "cuda" and out_fwd_cuda1.device.index == 1
            assert torch.allclose(out_fwd_cuda0.cpu(), out_fwd_cuda1.cpu(), atol=1e-5)
            out_sample_cuda1_1 = lg.sample(count=4, seed=123)
            out_sample_cuda1_2 = lg.sample(count=4, seed=123)
            assert torch.equal(out_sample_cuda1_1, out_sample_cuda1_2), "CUDA:1 sample determinism failed"

        # 4. Move back to CPU
        lg.to("cpu")
        assert lg.device == torch.device("cpu")
        assert lg.particle_z.device.type == "cpu"
        out_fwd_back = lg(z_fixed)
        assert out_fwd_back.device.type == "cpu"
        assert torch.allclose(out_fwd_cpu, out_fwd_back, atol=1e-5)

    def test_seed_determinism_and_rng_isolation(self, tmp_path):
        """Stress-test that sample(seed=...) and interpolate(seed_a=..., seed_b=...) isolate global RNG."""
        bundle_file = make_dummy_bundle(tmp_path, z_dim=8, out_features=16)
        lg = hg.load_bundle(bundle_file, device="cpu")

        # Global RNG isolation during sample()
        torch.manual_seed(99999)
        rng_before = torch.get_rng_state()
        out1 = lg.sample(count=5, seed=42)
        rng_after = torch.get_rng_state()
        assert torch.equal(rng_before, rng_after), "lg.sample() mutated PyTorch global RNG state!"

        # Determinism across calls
        out2 = lg.sample(count=5, seed=42)
        assert torch.equal(out1, out2), "lg.sample() is not deterministic with the same seed!"

        # Different seed produces different output
        out3 = lg.sample(count=5, seed=43)
        assert not torch.equal(out1, out3), "lg.sample() produced identical output with different seeds!"

        # Global RNG isolation during interpolate()
        torch.manual_seed(88888)
        rng_before_interp = torch.get_rng_state()
        walk1 = lg.interpolate(seed_a=10, seed_b=20, steps=6)
        rng_after_interp = torch.get_rng_state()
        assert torch.equal(rng_before_interp, rng_after_interp), "lg.interpolate() mutated global RNG state!"

        walk2 = lg.interpolate(seed_a=10, seed_b=20, steps=6)
        assert torch.equal(walk1, walk2), "lg.interpolate() is not deterministic with identical seeds!"

    def test_gaussian_prior_standalone_generator(self, tmp_path):
        """Stress-test StandaloneGenerator with standard Gaussian prior (no particle table)."""
        bundle_file = make_gaussian_prior_bundle(tmp_path, z_dim=6, out_features=12)
        lg = hg.load_bundle(bundle_file, device="cpu")

        assert lg.particle_z is None
        assert lg.particle_sigma is None
        assert lg.means() is None
        assert lg.sigma is None

        # Sample with seed determinism
        s1 = lg.sample(count=4, seed=101)
        s2 = lg.sample(count=4, seed=101)
        assert torch.equal(s1, s2)
        assert s1.shape == (4, 12)

        # Interpolate between seeds
        walk = lg.interpolate(seed_a=1, seed_b=2, steps=5)
        assert walk.shape == (5, 12)

    def test_batch_generation_various_sizes_and_defaults(self, tmp_path):
        """Stress-test forward generation across various batch sizes and default z=None."""
        bundle_file = make_dummy_bundle(tmp_path, z_dim=8, out_features=16)
        lg = hg.load_bundle(bundle_file, device="cpu")

        # Dynamic batch sizes
        for b in [1, 2, 7, 16, 64]:
            z = torch.randn(b, 8)
            out = lg(z)
            assert out.shape == (b, 16)

        # Default z=None generates 1 sample
        out_default = lg()
        assert out_default.shape == (1, 16)

    def test_interpolate_latent_coercion_and_conditioning_expansion(self, tmp_path):
        """Stress-test that z_a and z_b coerce lists/numpy arrays and conditioning expands."""
        bundle_file = make_dummy_bundle(tmp_path, z_dim=4, out_features=8)
        lg = hg.load_bundle(bundle_file, device="cpu")

        # Python lists
        walk_list = lg.interpolate(z_a=[1.0, 0.0, 0.0, 0.0], z_b=[0.0, 1.0, 0.0, 0.0], steps=4)
        assert walk_list.shape == (4, 8)

        # NumPy arrays
        walk_np = lg.interpolate(
            z_a=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            z_b=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
            steps=5
        )
        assert walk_np.shape == (5, 8)

        # 1D vs 2D tensors passed to interpolate
        walk_t1d = lg.interpolate(z_a=torch.randn(4), z_b=torch.randn(4), steps=3)
        assert walk_t1d.shape == (3, 8)

        walk_t2d = lg.interpolate(z_a=torch.randn(1, 4), z_b=torch.randn(1, 4), steps=3)
        assert walk_t2d.shape == (3, 8)

    def test_dual_inheritance_dict_and_module_mutation_safety(self, tmp_path):
        """Stress-test dual inheritance: nn.Module and dict behavior."""
        bundle_file = make_dummy_bundle(tmp_path, z_dim=8, out_features=16)
        lg = hg.load_bundle(bundle_file, device="cpu")

        assert isinstance(lg, torch.nn.Module)
        assert isinstance(lg, dict)
        assert lg["schema_version"] == 1
        assert "generator" in lg["components"]
        assert lg.get("step") == 42

        # Dict operations do not break Module forward
        lg["custom_metadata"] = "verified"
        assert lg["custom_metadata"] == "verified"
        out = lg(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_load_bundle_path_types_and_error_handling(self, tmp_path):
        """Stress-test hg.load_bundle and hg.load with Path, str, and invalid targets."""
        bundle_file = make_dummy_bundle(tmp_path, z_dim=8, out_features=16)

        # Path object
        lg1 = hg.load_bundle(bundle_file)
        assert isinstance(lg1, LoadedGenerator)

        # String path
        lg2 = hg.load_bundle(str(bundle_file))
        assert isinstance(lg2, LoadedGenerator)

        # hg.load dispatch for .pt
        lg3 = hg.load(bundle_file)
        assert isinstance(lg3, LoadedGenerator)

        # Non-existent file
        with pytest.raises(Exception):
            hg.load_bundle(tmp_path / "non_existent.pt")


# ---------------------------------------------------------------------------
# 3. Clean Subprocess Zero-Dependency Verification
# ---------------------------------------------------------------------------

class TestZeroDependencyCleanSubprocessExecution:
    """Stress-test that LoadedGenerator runs on CPU and GPU without particlegan or hndl."""

    def test_clean_subprocess_cpu_and_cuda_without_particlegan_or_hndl(self, tmp_path):
        """Run in a completely isolated Python process with particlegan and hndl blocked."""
        bundle_file = make_dummy_bundle(tmp_path, z_dim=8, out_features=16)

        sub_code = f"""
import sys
# Strictly block particlegan and hndl
sys.modules['particlegan'] = None
sys.modules['hndl'] = None

import torch
import hypergan.api as hg

# 1. Load standalone generator
lg = hg.load_bundle({repr(str(bundle_file))}, device='cpu')
assert sys.modules.get('particlegan') is None
assert sys.modules.get('hndl') is None

# 2. CPU execution
out_cpu = lg.sample(count=4, seed=42)
assert out_cpu.shape == (4, 16)
walk_cpu = lg.interpolate(seed_a=1, seed_b=2, steps=5)
assert walk_cpu.shape == (5, 16)

# 3. GPU execution if available
if torch.cuda.is_available():
    lg.to('cuda')
    assert lg.device.type == 'cuda'
    out_cuda = lg.sample(count=4, seed=42)
    assert out_cuda.device.type == 'cuda'
    assert out_cuda.shape == (4, 16)
    walk_cuda = lg.interpolate(seed_a=1, seed_b=2, steps=5)
    assert walk_cuda.device.type == 'cuda'
    assert walk_cuda.shape == (5, 16)

print('SUBPROCESS_SUCCESS')
"""
        result = subprocess.run([sys.executable, "-c", sub_code], capture_output=True, text=True)
        assert result.returncode == 0, f"Clean subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "SUBPROCESS_SUCCESS" in result.stdout

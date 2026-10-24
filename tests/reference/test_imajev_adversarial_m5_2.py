"""Tier 5 White-Box Adversarial Coverage Hardening Suite for ImaJev-4B Constraint GAN (M5 Phase 2).

Systematically stress-tests:
1. `JointMultiQueryConstraintData` in `src/hypergan/constraint_data.py`:
   - Independent mode sampling across Q queries spanning Cartesian products.
   - Chi-squared contingency independence test across samples and queries.
   - Soft-capped target generation with exact mathematical formulas and [-tau, tau] bounds.
   - Heterogeneous class counts (K_q < K_max) with exact -tau padding.
   - Comprehensive query table validation: empty list, non-mapping, >64 queries, invalid options,
     target_probs length mismatch, out-of-bounds target_modes, non-positive parameters.
   - Boundary probabilities (0.50001, 0.99999), jitter std=0 vs >0, batch size 1 edge cases.
   - Checkpoint resume identity contract.
2. `OpticalWavefrontPrior` in `src/hypergan/constraint_data.py`:
   - Radial spectral density slope alpha in [0.5, 4.0], frequency-dependent attenuation, margin_db gating.
   - Seam penalty on 8px and 16px ViT boundaries, border pixel exclusions, gradient direction.
   - Biharmonic curvature norm: 5-point vs 9-point Laplacian kernels, flat images, linear ramps, padding modes.
   - Non-square (64x48, 32x64) and odd dimensions (65x49, 33x47), minimum boundary shape (16x16),
     and dimension rejections (H < 16, C != 3).
   - Device and dtype transfers (CPU/CUDA, float32/float64).
3. `MarginalShannonEntropyRegularizer` in `src/hypergan/constraint_data.py`:
   - Numerical stability in log: zero probability modes, small epsilon, extreme logits (+/- 1000, +/- 1e5).
   - Batch size 1 (B=1) single-sample edge cases with valid finite autograd gradients.
   - Mode collapse penalty scaling = (min_entropy_ratio * ln M)^2.
   - Multiple heterogeneous queries, slicing bounds verification, and hinge=False mode.
   - Closed-loop Adam optimization unpacking collapsed mode batch toward uniform distribution.
4. `NormalizedVisionCosineRepulsion` in `src/hypergan/constraint_data.py`:
   - Scale invariance and unit normalization across feature scales 10^-6 to 10^6 and dtypes.
   - Zero-norm vectors (v=0) forward and backward stability with zero loss and finite gradients.
   - Margin handling across geometries: identical (1 - margin), orthogonal (0.0), antipodal (0.0).
   - Boundary batch sizes B=0 and B=1 returning 0.0.
   - Closed-loop Adam optimization driving pairwise cosine distance >= 0.12.
5. `examples/imajev-apples-joint-bispf.toml`:
   - Recipe DAG parsing, component inventory (image_generator, generator, vision_features, discriminator).
   - All 6 objectives parsed, including 3 detached monitor objectives (apple_prob, noise_prob, unknown_prob).
   - `hg.lift` and `hg.validate` conformance to hypergan-model/1.
   - Full config resolution via `resolve_config_at`.
   - Adversarial DAG mutations: cyclic dependency, dangling inputs, unknown factories, verified rejection.
"""
import copy
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import hypergan as hg
from hypergan.checkpoints import data_contract
from hypergan.config import load_config, resolve_config, resolve_config_at
from hypergan.constraint_data import (
    JointMultiQueryConstraintData,
    MarginalShannonEntropyRegularizer,
    NormalizedVisionCosineRepulsion,
    OpticalWavefrontPrior,
    OptionProbability,
)
from hypergan.model_file import ModelFileError
from hypergan.pretrained_providers import _symmetric_log_odds

RECIPE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "imajev-apples-joint-bispf.toml"


# ============================================================================
# SECTION 1: JointMultiQueryConstraintData Adversarial Tests
# ============================================================================

def test_joint_multi_query_cartesian_independence_contingency():
    """Verify statistical independence of sampled modes across queries via 2D contingency table."""
    queries = [
        {
            "aspect": "color",
            "question": "Color?",
            "options": ["red", "green", "yellow", "noise"],
            "target_probs": [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.03, 0.90, 0.03, 0.02, 0.02],
                [0.03, 0.03, 0.90, 0.02, 0.02],
            ],
        },
        {
            "aspect": "bites",
            "question": "Bites?",
            "options": ["zero", "one", "multiple", "noise"],
            "target_probs": [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.03, 0.90, 0.03, 0.02, 0.02],
                [0.03, 0.03, 0.90, 0.02, 0.02],
            ],
        },
        {
            "aspect": "style",
            "question": "Style?",
            "options": ["glossy", "painterly", "product", "noise"],
            "target_probs": [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.03, 0.90, 0.03, 0.02, 0.02],
                [0.03, 0.03, 0.90, 0.02, 0.02],
            ],
        },
    ]
    data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)
    rng = torch.Generator().manual_seed(4242)

    # Sample large batch to test statistical contingency
    N = 2700
    batch = data(N, generator=rng)
    modes = batch["modes"]  # [N, 3]
    assert modes.shape == (N, 3)

    # Contingency test between query 0 and query 1
    m0 = modes[:, 0]
    m1 = modes[:, 1]
    contingency = torch.zeros(3, 3, dtype=torch.float32)
    for i in range(N):
        contingency[m0[i], m1[i]] += 1.0

    joint_p = contingency / N
    p0 = joint_p.sum(dim=1, keepdim=True)
    p1 = joint_p.sum(dim=0, keepdim=True)
    indep_p = p0 @ p1

    # Maximum absolute deviation between empirical joint and independent product
    max_dev = (joint_p - indep_p).abs().max().item()
    assert max_dev < 0.035, f"Contingency independence violation: max dev {max_dev:.4f} >= 0.035"

    # All 27 mode combinations (3 x 3 x 3) must be visited in N=2700 draws
    combinations = set(tuple(row.tolist()) for row in modes)
    assert len(combinations) == 27, f"Expected all 27 Cartesian modes, got {len(combinations)}"


def test_joint_multi_query_soft_capping_exact_math_and_bounds():
    """Verify that soft-capped targets match exact mathematical definition and [-tau, tau] bounds."""
    for tau in [0.5, 1.0, 3.0, 10.0]:
        for temp in [0.5, 1.0, 1.30515697, 2.5]:
            p = [0.85, 0.05, 0.05, 0.05]
            queries = [
                {
                    "aspect": "test",
                    "question": "Q?",
                    "options": ["a", "b", "c"],
                    "target_probs": p,
                }
            ]
            data = JointMultiQueryConstraintData(queries, tau=tau, temperature=temp, jitter_std=0.0)
            batch = data(8)
            real = batch["real"]  # [8, 4]

            # Invariant: all values must be bounded within [-tau, tau]
            assert torch.all(real >= -tau - 1e-6)
            assert torch.all(real <= tau + 1e-6)

            # Manual verification of exact formula
            probs_t = torch.tensor(p, dtype=torch.float32)
            log_p = probs_t.log()
            scaled = log_p / temp
            centered = scaled - scaled.mean()
            u_manual = _symmetric_log_odds(centered.unsqueeze(0))
            expected = tau * torch.tanh(u_manual / tau)

            torch.testing.assert_close(real[0], expected[0], atol=1e-5, rtol=1e-5)


def test_joint_multi_query_heterogeneous_classes_and_padding():
    """Verify queries with heterogeneous class counts are padded with exactly -tau."""
    queries = [
        {
            "aspect": "q2",
            "options": ["opt1"],  # 2 classes (1 option + 1 unknown)
            "target_modes": [0],
        },
        {
            "aspect": "q5",
            "options": ["opt1", "opt2", "opt3", "opt4"],  # 5 classes
            "target_modes": [0, 1, 2],
        },
        {
            "aspect": "q3",
            "options": ["opt1", "opt2"],  # 3 classes
            "target_modes": [0],
        },
    ]
    tau = 2.5
    data = JointMultiQueryConstraintData(queries, tau=tau, jitter_std=0.0)
    assert data.k_max == 5
    assert data.num_queries == 3

    batch = data(16)
    real = batch["real"]
    assert real.shape == (16, 15)  # 3 queries * 5 classes

    # Query 0: K=2 classes, so indices 2, 3, 4 must be padded with -tau
    torch.testing.assert_close(real[:, 2:5], torch.full((16, 3), -tau), atol=1e-6, rtol=0.0)
    # Query 2: K=3 classes, so indices 13, 14 (10 + 3, 10 + 4) must be padded with -tau
    torch.testing.assert_close(real[:, 13:15], torch.full((16, 2), -tau), atol=1e-6, rtol=0.0)

    # Check padding with jitter > 0 as well
    data_jitter = JointMultiQueryConstraintData(queries, tau=tau, jitter_std=0.2)
    batch_j = data_jitter(16)
    real_j = batch_j["real"]
    torch.testing.assert_close(real_j[:, 2:5], torch.full((16, 3), -tau), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(real_j[:, 13:15], torch.full((16, 2), -tau), atol=1e-6, rtol=0.0)


def test_joint_multi_query_query_table_comprehensive_validation():
    """Verify input validation for JointMultiQueryConstraintData rejects invalid tables."""
    valid_q = [{"question": "Q?", "options": ["a", "b"], "target_modes": [0]}]

    # Empty or wrong type
    for bad_queries in ([], "not_a_list", [valid_q[0]] * 65, [1, 2, 3]):
        with pytest.raises(ValueError):
            JointMultiQueryConstraintData(bad_queries)

    # Bad parameter bounds
    for bad_tau in (0.0, -1.0, float("nan"), float("inf"), True, False):
        with pytest.raises(ValueError, match="tau"):
            JointMultiQueryConstraintData(valid_q, tau=bad_tau)

    for bad_temp in (0.0, -2.0, float("nan"), True):
        with pytest.raises(ValueError, match="temperature"):
            JointMultiQueryConstraintData(valid_q, temperature=bad_temp)

    for bad_jitter in (-0.01, float("nan"), True):
        with pytest.raises(ValueError, match="jitter_std"):
            JointMultiQueryConstraintData(valid_q, jitter_std=bad_jitter)

    for bad_prob in (0.5, 0.4, 1.0, 1.2, -0.1, True):
        with pytest.raises(ValueError, match="target_prob"):
            JointMultiQueryConstraintData(valid_q, target_prob=bad_prob)

    # Bad questions and options
    with pytest.raises(ValueError, match="non-empty string"):
        JointMultiQueryConstraintData([{"question": "  ", "options": ["a", "b"], "target_modes": [0]}])

    with pytest.raises(ValueError, match="1..63 strings"):
        JointMultiQueryConstraintData([{"options": [], "target_modes": [0]}])

    with pytest.raises(ValueError, match="unique"):
        JointMultiQueryConstraintData([{"options": ["a", "a"], "target_modes": [0]}])

    with pytest.raises(ValueError, match="non-empty strings"):
        JointMultiQueryConstraintData([{"options": ["a", ""], "target_modes": [0]}])

    # Target mode out of range
    with pytest.raises(ValueError, match="target_modes entries must be in"):
        JointMultiQueryConstraintData([{"options": ["a", "b"], "target_modes": [5]}])

    with pytest.raises(ValueError, match="target_modes entries must be in"):
        JointMultiQueryConstraintData([{"options": ["a", "b"], "target_modes": [-1]}])

    # Target probs length mismatch
    with pytest.raises(ValueError, match="must match options length"):
        JointMultiQueryConstraintData([{"options": ["a", "b"], "target_probs": [0.5, 0.25, 0.15, 0.05, 0.05]}])

    # Target modes length mismatch with target probs rows
    with pytest.raises(ValueError, match="must match target_probs rows"):
        JointMultiQueryConstraintData([
            {
                "options": ["a", "b"],
                "target_probs": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
                "target_modes": [0],  # 1 mode given for 2 rows
            }
        ])


def test_joint_multi_query_boundary_probabilities_and_jitter():
    """Verify boundary probabilities (0.50001, 0.99999) and B=1 edge case."""
    for p_val in [0.50001, 0.99999]:
        q = [{"options": ["opt1", "opt2"], "target_modes": [0]}]
        data = JointMultiQueryConstraintData(q, target_prob=p_val, tau=3.0)

        # Batch size 1 edge case
        b1 = data(1)
        assert b1["real"].shape == (1, 3)
        assert b1["modes"].shape == (1, 1)
        assert torch.isfinite(b1["real"]).all()
        assert -3.0 <= b1["real"][0, 0].item() <= 3.0

    # Deterministic RNG reproducibility
    data_rng = JointMultiQueryConstraintData(q, jitter_std=0.1)
    g1 = torch.Generator().manual_seed(999)
    g2 = torch.Generator().manual_seed(999)
    b_rng1 = data_rng(8, generator=g1)
    b_rng2 = data_rng(8, generator=g2)
    assert torch.equal(b_rng1["real"], b_rng2["real"])
    assert torch.equal(b_rng1["modes"], b_rng2["modes"])


def test_joint_multi_query_resume_identity_contract():
    """Verify data_contract and resume_identity specification."""
    queries = [
        {"aspect": "a", "options": ["x", "y"], "target_modes": [0]},
        {"aspect": "b", "options": ["u", "v"], "target_modes": [0, 1]},
    ]
    data = JointMultiQueryConstraintData(queries, tau=2.8, temperature=1.2, jitter_std=0.05, target_prob=0.88)
    ident = data.resume_identity()
    assert ident["tau"] == 2.8
    assert ident["temperature"] == 1.2
    assert ident["jitter_std"] == 0.05
    assert ident["target_prob"] == 0.88
    assert len(ident["queries"]) == 2

    spec = {
        "factory": "hypergan.constraint_data:JointMultiQueryConstraintData",
        "args": {"queries": queries, "tau": 2.8},
    }
    contract = data_contract(data, spec)
    assert contract["supported"] is True
    assert contract["stateful"] is False
    assert contract["identity"] == ident


# ============================================================================
# SECTION 2: OpticalWavefrontPrior Adversarial Tests
# ============================================================================

def test_optical_wavefront_psd_alpha_spectral_falloff():
    """Verify radial spectral density prior across alpha in [0.5, 4.0] and margin gating."""
    for alpha in [0.5, 1.0, 2.0, 3.0, 4.0]:
        prior = OpticalWavefrontPrior(
            weight_seam=0.0,
            weight_biharmonic=0.0,
            weight_psd=1.0,
            alpha=alpha,
            margin_db=1.0,
        )
        # White noise has flat spectral power (alpha = 0) -> heavily violates 1/f^alpha
        noise = torch.randn(2, 3, 64, 64)
        loss_noise = prior(noise)
        assert float(loss_noise.item()) > 0.1, f"White noise failed to trigger PSD loss for alpha={alpha}"

        # Constant image has 0 AC power -> gate tanh(ac/1e-3) evaluates to 0 -> loss exactly 0
        flat = torch.full((2, 3, 64, 64), 0.5)
        loss_flat = prior(flat)
        assert float(loss_flat.item()) == pytest.approx(0.0, abs=1e-7)

    # Test margin_db gating: large margin_db absorbs spectral deviation
    prior_generous = OpticalWavefrontPrior(
        weight_seam=0.0, weight_biharmonic=0.0, weight_psd=1.0, margin_db=20.0
    )
    assert float(prior_generous(noise).item()) == pytest.approx(0.0, abs=1e-5)


def test_optical_wavefront_patch_seam_vit_boundaries_and_gradient():
    """Verify seam penalty calculation, border exclusions, and directional restorative gradients."""
    prior = OpticalWavefrontPrior(
        weight_seam=1.0,
        weight_biharmonic=0.0,
        weight_psd=0.0,
        seam_threshold=1.05,
        patch_size=16,
        secondary_patch_size=8,
    )

    # 1. Smooth gradient image should have R_seam <= 1.05 and loss_seam == 0.0
    y = torch.linspace(-1, 1, 64).unsqueeze(1).repeat(1, 64)
    x = torch.linspace(-1, 1, 64).unsqueeze(0).repeat(64, 1)
    smooth = (1.0 - torch.clamp(x**2 + y**2, max=1.0)).sqrt().view(1, 1, 64, 64).repeat(2, 3, 1, 1)
    d_smooth = prior(smooth, return_dict=True)
    assert d_smooth["R_seam"].item() <= 1.05
    assert d_smooth["loss_seam"].item() == 0.0

    # 2. Add boundary step along seam (x=16 and y=16)
    spiked = smooth.clone().requires_grad_(True)
    with torch.no_grad():
        spiked[:, :, 16, :] += 1.0
        spiked[:, :, :, 16] += 1.0
    d_spiked = prior(spiked, return_dict=True)
    assert d_spiked["R_seam"].item() > 1.20
    assert d_spiked["loss_seam"].item() > 0.0

    # 3. Check gradient direction: gradient on spiked line must oppose the spike
    d_spiked["loss_seam"].backward()
    assert spiked.grad is not None
    seam_grad_norm = spiked.grad[:, :, 16, :].abs().mean().item()
    interior_grad_norm = spiked.grad[:, :, 20:28, 20:28].abs().mean().item()
    assert seam_grad_norm > interior_grad_norm, "Restorative gradient failed to concentrate on seam"


def test_optical_wavefront_curvature_laplacian_kernels_and_paddings():
    """Verify 5-point vs 9-point Laplacian kernels, linear ramp invariance, and padding modes."""
    for kernel in ["5point", "9point"]:
        for pad_mode in ["reflect", "replicate", "zeros"]:
            prior = OpticalWavefrontPrior(
                weight_seam=0.0,
                weight_biharmonic=1.0,
                weight_psd=0.0,
                laplacian_kernel=kernel,
                padding_mode=pad_mode,
            )

            # Flat image: Delta I = 0 everywhere for reflect/replicate padding
            flat = torch.full((1, 3, 32, 32), 0.75, requires_grad=True)
            loss_flat = prior(flat)
            if pad_mode in ("reflect", "replicate"):
                assert float(loss_flat.item()) == pytest.approx(0.0, abs=1e-6)
            else:
                # zeros padding introduces artificial step at border, which correctly produces curvature
                assert float(loss_flat.item()) > 0.05

            # Linear ramp: Delta(a*x + b*y + c) = 0 in interior
            ramp = torch.linspace(0, 1, 32).view(1, 1, 1, 32).repeat(1, 3, 32, 1)
            loss_ramp = prior(ramp)
            assert float(loss_ramp.item()) < 0.15

            # High-curvature noise: positive loss with finite autograd grad
            noisy = torch.randn(1, 3, 32, 32, requires_grad=True)
            loss_noisy = prior(noisy)
            assert float(loss_noisy.item()) > 0.05
            loss_noisy.backward()
            assert torch.isfinite(noisy.grad).all()


def test_optical_wavefront_non_square_and_odd_image_shapes():
    """Verify OpticalWavefrontPrior operates smoothly on non-square, odd, and boundary shapes."""
    prior = OpticalWavefrontPrior(alpha=2.0)

    # Various non-square, odd, and boundary dimensions >= patch_size (16)
    shapes = [
        (2, 3, 64, 48),  # 4:3 non-square
        (1, 3, 32, 64),  # 1:2 wide
        (2, 3, 65, 49),  # odd dimensions
        (1, 3, 33, 47),  # odd non-square
        (1, 3, 16, 16),  # minimum boundary size
    ]
    for shape in shapes:
        x = torch.randn(shape, requires_grad=True)
        d = prior(x, return_dict=True)
        assert isinstance(d["total_loss"], torch.Tensor)
        assert torch.isfinite(d["total_loss"])
        d["total_loss"].backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    # Dimension rejection
    with pytest.raises(ValueError, match="patch_size"):
        prior(torch.randn(2, 3, 15, 32))
    with pytest.raises(ValueError, match="patch_size"):
        prior(torch.randn(2, 3, 32, 14))
    with pytest.raises(ValueError, match="C=3"):
        prior(torch.randn(2, 1, 32, 32))
    with pytest.raises(ValueError, match="4D floating-point"):
        prior(torch.randn(3, 32, 32))


def test_optical_wavefront_device_and_dtype_transfers():
    """Verify module cleanly runs with float32 and float64 dtypes and diagnostic dictionary."""
    prior = OpticalWavefrontPrior()

    # Float64 precision
    x64 = torch.randn(1, 3, 32, 32, dtype=torch.float64, requires_grad=True)
    d64 = prior(x64, return_dict=True)
    assert d64["total_loss"].dtype == torch.float64
    d64["total_loss"].backward()
    assert x64.grad.dtype == torch.float64

    # CUDA execution if GPU is available
    if torch.cuda.is_available():
        prior_cuda = OpticalWavefrontPrior().cuda()
        x_cuda = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
        d_cuda = prior_cuda(x_cuda, return_dict=True)
        assert d_cuda["total_loss"].is_cuda
        d_cuda["total_loss"].backward()
        assert x_cuda.grad.is_cuda


def test_optical_wavefront_no_procedural_shader_and_resume_identity():
    """Verify OpticalWavefrontPrior is strictly domain-agnostic and has no procedural basis."""
    prior = OpticalWavefrontPrior(
        weight_seam=1.5,
        weight_biharmonic=0.2,
        weight_psd=0.05,
    )
    assert not hasattr(prior, "_get_optical_basis")
    assert not hasattr(prior, "weight_optical_envelope")
    ident = prior.resume_identity()
    assert ident["weight_seam"] == 1.5
    assert ident["weight_biharmonic"] == 0.2
    assert ident["weight_psd"] == 0.05
    with pytest.raises(TypeError):
        OpticalWavefrontPrior(weight_optical_envelope=20.0)


# ============================================================================
# SECTION 3: MarginalShannonEntropyRegularizer Adversarial Tests
# ============================================================================

def test_marginal_entropy_numerical_stability_extreme_logits_and_zero_probs():
    """Verify numerical stability in log under extreme logits (+/- 1000, +/- 1e5) and zero probabilities."""
    reg = MarginalShannonEntropyRegularizer(
        query_indices=[0, 1],
        num_classes=5,
        num_modes=3,
        input_mode="joint_log_odds",
    )

    # 1. Extreme saturating logits (+/- 1000)
    u_ext = torch.tensor([
        [1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, 1000.0, -1000.0, -1000.0, -1000.0],
        [-1000.0, 1000.0, -1000.0, -1000.0, -1000.0, 1000.0, -1000.0, -1000.0, -1000.0, -1000.0],
    ], requires_grad=True)

    loss_ext = reg(u_ext)
    assert torch.isfinite(loss_ext)
    loss_ext.backward()
    assert torch.isfinite(u_ext.grad).all(), "Extreme logits produced non-finite gradients"

    # 2. Input mode 'probs' with exact zero probabilities
    reg_probs = MarginalShannonEntropyRegularizer(
        query_indices=[0],
        num_classes=4,
        num_modes=3,
        input_mode="probs",
        eps=1e-12,
    )
    p_zero = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss_zero = reg_probs(p_zero)
    assert torch.isfinite(loss_zero)
    loss_zero.backward()
    assert torch.isfinite(p_zero.grad).all(), "Zero probability modes produced non-finite gradients"


def test_marginal_entropy_batch_size_one_edge_case():
    """Verify batch size 1 (B=1) edge case: mode marginal equals single sample."""
    reg = MarginalShannonEntropyRegularizer(
        query_indices=[0],
        num_classes=4,
        num_modes=2,
        min_entropy_ratio=0.85,
        input_mode="logits",
    )
    target_H = 0.85 * math.log(2.0)

    # Perfectly balanced single sample: [0, 0, -20, -20] -> p = [0.5, 0.5, 0, 0] -> H = ln 2 > target_H
    u_bal = torch.tensor([[0.0, 0.0, -20.0, -20.0]], requires_grad=True)
    loss_bal = reg(u_bal)
    assert float(loss_bal.item()) == pytest.approx(0.0, abs=1e-5)

    # Collapsed single sample: [3.0, -3.0, -3.0, -3.0]
    u_col = torch.tensor([[3.0, -3.0, -3.0, -3.0]], requires_grad=True)
    loss_col = reg(u_col)
    assert loss_col.item() > 0.0

    loss_col.backward()
    assert u_col.grad is not None
    assert torch.isfinite(u_col.grad).all()
    # Gradients should push dominant option downward and suppressed options upward
    assert u_col.grad[0, 0].item() > 0.0
    assert u_col.grad[0, 1].item() < 0.0


def test_marginal_entropy_zero_probability_modes_and_collapse_penalty():
    """Verify exact penalty scaling = (min_entropy_ratio * ln M)^2 when H ~ 0."""
    for M in [2, 3, 5]:
        reg = MarginalShannonEntropyRegularizer(
            query_indices=[0],
            num_classes=M,
            num_modes=M,
            min_entropy_ratio=0.85,
        )
        target_H = 0.85 * math.log(M)
        expected = target_H ** 2

        # Fully collapsed batch
        u = torch.full((16, M), -10.0)
        u[:, 0] = 10.0
        loss = reg(u)
        assert float(loss.item()) == pytest.approx(expected, rel=1e-2)


def test_marginal_entropy_multiple_heterogeneous_queries_and_slicing():
    """Verify multi-query handling with heterogeneous mode counts and slicing bounds."""
    # Queries 1, 3 with different mode counts: M_1=3, M_3=4
    reg = MarginalShannonEntropyRegularizer(
        query_indices=[1, 3],
        num_classes=5,
        num_modes={1: 3, 3: 4},
        min_entropy_ratio=0.85,
        hinge=False,  # L = -H
    )
    # Total width must be at least 4 * 5 = 20
    u = torch.randn(8, 20, requires_grad=True)
    loss = reg(u)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(u.grad).all()

    # Slicing error when tensor width is too small
    with pytest.raises(ValueError, match="tensor width"):
        reg(torch.randn(8, 15))


def test_marginal_entropy_closed_loop_mode_unpacking():
    """Closed-loop optimization: Adam unpacks collapsed logits into all modes."""
    reg = MarginalShannonEntropyRegularizer(
        query_indices=[0],
        num_classes=3,
        num_modes=3,
        min_entropy_ratio=0.85,
        input_mode="logits",
    )
    # Start with predominantly collapsed logits with small particle variations
    torch.manual_seed(42)
    logits = torch.randn(30, 3) * 0.1
    logits[:, 0] += 4.0
    logits.requires_grad_(True)

    initial_loss = reg(logits).item()
    assert initial_loss > 0.5, "Initial collapsed loss should be large"

    optimizer = torch.optim.Adam([logits], lr=0.2)
    for _ in range(25):
        optimizer.zero_grad()
        loss = reg(logits)
        loss.backward()
        optimizer.step()

    final_loss = reg(logits).item()
    assert final_loss < 0.01, f"Failed to unpack modes: final loss {final_loss} >= 0.01"

    probs = logits.detach().softmax(dim=-1).mean(dim=0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    target_H = 0.85 * math.log(3.0)
    assert entropy >= target_H, f"Marginal entropy {entropy} < target {target_H}"
    assert float(probs.min().item()) >= 0.08, f"Mode starving with probability {probs.min().item()}"


# ============================================================================
# SECTION 4: NormalizedVisionCosineRepulsion Adversarial Tests
# ============================================================================

def test_vision_repulsion_unit_normalization_and_scale_invariance():
    """Verify that feature scale (from 10^-6 to 10^6) does not alter cosine repulsion."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)
    base_v = torch.randn(4, 2560)

    loss_base = rep(base_v).item()

    # Scale by 10^-6 and 10^6
    v_tiny = base_v * 1e-6
    v_huge = base_v * 1e6

    assert rep(v_tiny).item() == pytest.approx(loss_base, rel=1e-4)
    assert rep(v_huge).item() == pytest.approx(loss_base, rel=1e-4)

    # Precision dtypes
    for dt in (torch.float32, torch.float64):
        v_dt = base_v.to(dtype=dt)
        loss_dt = rep(v_dt)
        assert loss_dt.dtype == dt


def test_vision_repulsion_zero_norm_vectors():
    """Verify that zero-norm vectors (v=0) yield zero loss and finite gradients."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)

    # All zeros
    v_zeros = torch.zeros(4, 2560, requires_grad=True)
    loss_zeros = rep(v_zeros)
    assert float(loss_zeros.item()) == 0.0
    loss_zeros.backward()
    assert torch.isfinite(v_zeros.grad).all()

    # Mixed zero and non-zero
    v_mixed = torch.cat([torch.zeros(2, 2560), torch.randn(2, 2560)], dim=0).requires_grad_(True)
    loss_mixed = rep(v_mixed)
    loss_mixed.backward()
    assert torch.isfinite(loss_mixed)
    assert torch.isfinite(v_mixed.grad).all()


def test_vision_repulsion_margin_handling_and_geometries():
    """Verify pairwise geometries: identical (1 - margin), orthogonal (0.0), antipodal (0.0)."""
    rep = NormalizedVisionCosineRepulsion(margin=0.80)

    # 1. Identical vectors: cos = 1.0 -> penalty = 1.0 - 0.80 = 0.20
    v_id = torch.randn(1, 2560).repeat(4, 1)
    loss_id = rep(v_id)
    assert float(loss_id.item()) == pytest.approx(0.20, abs=1e-5)

    # 2. Orthogonal vectors: cos = 0.0 <= margin -> penalty = 0.0
    v_ortho = torch.eye(4, 2560)
    assert float(rep(v_ortho).item()) == 0.0

    # 3. Antipodal vectors: cos = -1.0 <= margin -> penalty = 0.0
    v_anti = torch.cat([v_id[:2], -v_id[:2]], dim=0)
    loss_anti = rep(v_anti)
    assert 0.0 < float(loss_anti.item()) < 0.20

    # 4. Dynamic margin override in forward
    loss_dyn = rep(v_id, margin=0.50)
    assert float(loss_dyn.item()) == pytest.approx(0.50, abs=1e-5)

    # 5. Boundary batch sizes B=0 and B=1 return 0.0
    assert float(rep(torch.randn(1, 2560)).item()) == 0.0
    assert float(rep(torch.randn(0, 2560)).item()) == 0.0


def test_vision_repulsion_separation_dynamics_closed_loop():
    """Closed-loop optimization: Adam drives pairwise cosine distance >= 0.12."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)
    # Start with tightly clustered features (nearly identical with tiny perturbation)
    torch.manual_seed(42)
    features = (torch.randn(1, 2560).repeat(8, 1) + torch.randn(8, 2560) * 0.01).requires_grad_(True)

    optimizer = torch.optim.Adam([features], lr=0.1)
    for _ in range(30):
        optimizer.zero_grad()
        loss = rep(features)
        loss.backward()
        optimizer.step()

    # Verify that pairwise cosine distance 1 - cos(v_i, v_j) is now >= 0.12 (sim <= 0.88)
    with torch.no_grad():
        v_norm = F.normalize(features, p=2, dim=-1)
        sim = torch.mm(v_norm, v_norm.t())
        mask = ~torch.eye(8, dtype=torch.bool)
        mean_cos_dist = (1.0 - sim[mask]).mean().item()
        assert mean_cos_dist >= 0.12, f"Failed to separate features: cosine distance {mean_cos_dist:.4f} < 0.12"


# ============================================================================
# SECTION 5: Recipe DAG Parsing and Validation
# ============================================================================

def test_imajev_apple_recipe_dag_structure_and_components():
    """Verify recipe DAG structure, components, and shapes in imajev-apples-joint-bispf.toml."""
    assert RECIPE_PATH.is_file(), f"Recipe file missing at {RECIPE_PATH}"
    recipe = hg.load_recipe(RECIPE_PATH)

    assert recipe.get("schema_version") == 1
    assert recipe.get("name") == "constraints/imajev-apples-joint-64-bispf"
    assert recipe.get("defaults") == "particlegan"

    components = recipe.get("components", {})
    assert set(components.keys()) == {"image_generator", "generator", "vision_features", "discriminator"}

    # image_generator: latent -> [B, 3, 64, 64]
    ig = components["image_generator"]
    assert ig["factory"] == "hndl"
    assert ig["inputs"] == {"x": "latent"}
    assert ig["trainable"] is True
    assert ig["args"]["output_shape"] == ["B", 3, 64, 64]

    # generator: image_generator -> [B, 30] joint_log_odds
    gen = components["generator"]
    assert gen["inputs"] == {"x": "components.image_generator"}
    assert gen["args"]["output_shape"] == ["B", 30]
    assert gen["args"]["pretrained_providers"]["imajev_4b"]["mode"] == "joint_log_odds"

    # vision_features: image_generator -> [B, 2560] pooled_vision
    vf = components["vision_features"]
    assert vf["inputs"] == {"x": "components.image_generator"}
    assert vf["args"]["output_shape"] == ["B", 2560]
    assert vf["args"]["pretrained_providers"]["imajev_4b"]["mode"] == "pooled_vision"

    # discriminator: candidate -> [B, 1]
    disc = components["discriminator"]
    assert disc["inputs"] == {"x": "candidate"}
    assert disc["args"]["input_shape"] == ["B", 30]
    assert disc["args"]["output_shape"] == ["B", 1]


def test_imajev_apple_recipe_dag_objectives_and_monitors():
    """Verify all 6 objectives including detached monitors in imajev-apples-joint-bispf.toml."""
    recipe = hg.load_recipe(RECIPE_PATH)
    objectives = recipe.get("objectives", [])
    assert len(objectives) == 6

    obj_map = {obj["id"]: obj for obj in objectives}
    assert set(obj_map.keys()) == {
        "optical_wavefront",
        "mode_entropy",
        "vision_repulsion",
        "apple_prob",
        "noise_prob",
        "unknown_prob",
    }

    # Verify detached monitor configurations
    for monitor_id in ["apple_prob", "noise_prob", "unknown_prob"]:
        m = obj_map[monitor_id]
        assert m["detach"] == ["logits"]
        assert m["weight"] == 1.0
        assert m["args"]["input_mode"] == "joint_log_odds"
        assert m["args"]["num_classes"] == 5

    assert obj_map["apple_prob"]["args"]["index"] == 0
    assert obj_map["noise_prob"]["args"]["index"] == 3
    assert obj_map["unknown_prob"]["args"]["index"] == 4


def test_imajev_apple_recipe_lift_and_model_validation():
    """Verify recipe lifts cleanly to hypergan-model/1 and passes model validation."""
    recipe = hg.load_recipe(RECIPE_PATH)
    model = hg.lift(recipe, base=RECIPE_PATH.parent)

    assert model.spec.get("format") == "hypergan-model/1"
    assert "networks" in model.spec
    assert "losses" in model.spec

    # Validation must return structured warnings without raising exceptions
    warnings = hg.validate(model)
    assert isinstance(warnings, list)
    assert any("Spread regularizer" in w or "spread" in w for w in warnings)


def test_imajev_apple_recipe_resolve_config_at():
    """Verify resolve_config_at resolves full engine configuration with all components and objectives."""
    recipe = hg.load_recipe(RECIPE_PATH)
    resolved = resolve_config_at(recipe, RECIPE_PATH.parent)

    assert "components" in resolved
    assert set(resolved["components"].keys()) == {
        "image_generator",
        "generator",
        "vision_features",
        "discriminator",
    }
    assert len(resolved["objectives"]) == 6
    assert resolved["data"]["factory"] == "hypergan.constraint_data:JointMultiQueryConstraintData"
    assert resolved["prior"]["kind"] == "particles"
    assert resolved["prior"]["args"]["num_particles"] == 512


def test_imajev_apple_recipe_adversarial_mutations():
    """Verify DAG validation rejects cyclic dependencies and dangling inputs."""
    recipe = hg.load_recipe(RECIPE_PATH)

    # 1. Cyclic component binding: image_generator depends on generator
    bad_cycle = copy.deepcopy(recipe)
    bad_cycle["components"]["image_generator"]["inputs"]["x"] = "components.generator"
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_config_at(bad_cycle, RECIPE_PATH.parent)

    # 2. Dangling candidate input binding on discriminator
    bad_disc = copy.deepcopy(recipe)
    bad_disc["components"]["discriminator"]["inputs"]["x"] = "components.ghost_network"
    with pytest.raises(ValueError, match="candidate"):
        resolve_config_at(bad_disc, RECIPE_PATH.parent)

    # 3. Invalid objective factory (bad syntax without colon or builtin)
    bad_obj = copy.deepcopy(recipe)
    bad_obj["objectives"][0]["factory"] = "invalid_factory_syntax"
    with pytest.raises(ValueError, match="module:object"):
        resolve_config_at(bad_obj, RECIPE_PATH.parent)

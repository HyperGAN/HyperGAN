"""Adversarial empirical challenge suite for Milestone 3 (Features 11 & 12).

Stress-tests:
1. Feature 11: On-Manifold Particle Diversity
   - MarginalShannonEntropyRegularizer:
     * Mode collapse penalty: H ~ 0 triggers strong penalty (0.85 ln M)^2.
     * Uniform multi-mode batch: H >= 0.85 ln M triggers zero penalty.
     * Boundary sensitivity: sharp transition across target entropy threshold.
     * Gradient direction: gradients push option probabilities away from collapse.
     * Closed-loop unpacking: Adam optimizer unpacks collapsed batch into all modes.
     * Multi-query heterogeneous mode configurations and slicing bounds.
     * Input modes (joint_log_odds, logits, probs) and extreme inputs (+/- 1000, +/- 1e5).
     * Numerical autograd gradcheck.
   - NormalizedVisionCosineRepulsion:
     * Margin calibration: identical vectors incur exact (1.0 - margin) penalty (0.12 for margin 0.88).
     * Orthogonal and antipodal vectors incur 0.0 penalty.
     * Boundary batch sizes: B <= 1 returns 0.0.
     * Separation dynamics: gradients push vectors apart; Adam drives pairwise cosine distance >= 0.12.
     * Numerical scale invariance and autograd gradcheck.
2. Feature 12: Multi-Scale Pyramid Generator
   - Exact parameter count verification (~918k, exactly 918,076 parameters).
   - Frequency-domain Nyquist corner checkerboard attenuation (>20x reduction vs transposed-conv).
   - Optical patch-seam ratio coherence (R_seam <= 1.15).
   - Smooth latent spatial interpolation (bounded Lipschitz constant, no jump discontinuities).
   - End-to-end backward gradient flow to z across all pyramid stages and batch sizes.
"""
import copy
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypergan.constraint_data import (
    MarginalShannonEntropyRegularizer,
    NormalizedVisionCosineRepulsion,
    OpticalWavefrontPrior,
)
from hypergan.hndl_networks import build_network

ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# FEATURE 11: MarginalShannonEntropyRegularizer Stress Tests
# ============================================================================

def test_marginal_shannon_entropy_collapse_penalty_scaling():
    """Verify that mode collapse triggers strong squared penalty = (min_entropy_ratio * ln M)^2."""
    for num_modes in [2, 3, 5, 8]:
        num_classes = max(5, num_modes)
        reg = MarginalShannonEntropyRegularizer(
            query_indices=[0],
            num_classes=num_classes,
            num_modes=num_modes,
            min_entropy_ratio=0.85,
            hinge=True,
        )
        target_H = 0.85 * math.log(num_modes)
        expected_penalty = target_H ** 2

        for B in [1, 2, 8, 32, 128]:
            # Completely collapsed batch: mode 0 has large logit (+15), others have (-15)
            u = torch.full((B, num_classes), -15.0)
            u[:, 0] = 15.0

            loss = reg(u)
            # H ~ 0, so loss should closely match (0.85 * ln M)^2
            assert float(loss.item()) == pytest.approx(expected_penalty, rel=1e-2), (
                f"Collapsed batch B={B}, M={num_modes} penalty {loss.item()} != expected {expected_penalty}"
            )


def test_marginal_shannon_entropy_uniform_and_threshold_boundary():
    """Verify that uniform multi-mode batches incur 0 penalty, and test threshold boundary."""
    reg = MarginalShannonEntropyRegularizer(
        query_indices=[0],
        num_classes=5,
        num_modes=3,
        min_entropy_ratio=0.85,
        hinge=True,
    )
    target_H = 0.85 * math.log(3)

    # 1. Perfectly balanced batch across 3 modes: H = ln(3) > 0.85 ln(3) -> loss must be exactly 0.0
    B = 30
    u_uniform = torch.full((B, 5), -10.0)
    for i in range(B):
        u_uniform[i, i % 3] = 10.0

    loss_uniform = reg(u_uniform)
    assert loss_uniform.item() == 0.0, f"Uniform batch incurred non-zero penalty: {loss_uniform.item()}"

    # 2. Test near-threshold behavior using input_mode='probs'
    # Distribution with H just above target
    p_above = torch.tensor([[0.60, 0.25, 0.15, 0.0, 0.0]])
    H_above = -(p_above * torch.log(p_above.clamp_min(1e-12))).sum().item()
    assert H_above > target_H
    reg_probs = MarginalShannonEntropyRegularizer(
        query_indices=[0], num_classes=5, num_modes=3, min_entropy_ratio=0.85, input_mode="probs"
    )
    assert reg_probs(p_above).item() == 0.0

    # Distribution with H just below target
    p_below = torch.tensor([[0.75, 0.15, 0.10, 0.0, 0.0]])
    H_below = -(p_below * torch.log(p_below.clamp_min(1e-12))).sum().item()
    assert H_below < target_H
    expected_deficit = (target_H - H_below) ** 2
    loss_below = reg_probs(p_below).item()
    assert loss_below > 0.0
    assert loss_below == pytest.approx(expected_deficit, rel=1e-4)


def test_marginal_shannon_entropy_gradient_push_and_unpacking_optimization():
    """Verify gradients push collapsed probabilities down and starved options up,
    and verify Adam optimization unpacks a collapsed batch into balanced modes."""
    reg = MarginalShannonEntropyRegularizer(
        query_indices=[0],
        num_classes=5,
        num_modes=3,
        min_entropy_ratio=0.85,
        hinge=True,
    )

    # 1. Verify gradient signs on collapsed batch
    u = torch.full((16, 5), -4.0, requires_grad=True)
    with torch.no_grad():
        u[:, 0] = 4.0  # Collapsed onto option 0

    loss = reg(u)
    loss.backward()

    # Decreasing loss requires increasing H, which requires lowering p_0 and raising p_1, p_2
    assert u.grad[:, 0].mean().item() > 0.0, "Gradient on collapsed option must be positive"
    assert u.grad[:, 1].mean().item() < 0.0, "Gradient on starved option 1 must be negative"
    assert u.grad[:, 2].mean().item() < 0.0, "Gradient on starved option 2 must be negative"

    # Single gradient descent step
    u_next = u - 0.2 * u.grad
    probs_before = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=-1, keepdim=True)
    probs_after = torch.sigmoid(u_next) / torch.sigmoid(u_next).sum(dim=-1, keepdim=True)
    assert probs_after[:, 0].mean().item() < probs_before[:, 0].mean().item()
    assert probs_after[:, 1:].mean().item() > probs_before[:, 1:].mean().item()

    # 2. Closed-loop optimization unpacking test
    torch.manual_seed(101)
    u_opt = (u.detach().clone() + 0.05 * torch.randn(16, 5)).requires_grad_(True)
    opt = torch.optim.Adam([u_opt], lr=0.1)

    steps = 0
    for steps in range(1, 100):
        opt.zero_grad()
        l = reg(u_opt)
        if l.item() == 0.0:
            break
        l.backward()
        opt.step()

    assert reg(u_opt).item() == 0.0, "Optimizer failed to eliminate entropy deficit"
    assert steps < 25, f"Unpacking took too many steps: {steps}"


def test_marginal_shannon_entropy_multi_query_and_heterogeneous_modes():
    """Verify multi-query handling with heterogeneous mode counts across queries."""
    # 6 queries, 5 classes each -> width 30
    # Query 1: 3 modes, Query 2: 2 modes, Query 4: 4 modes
    query_indices = [1, 2, 4]
    modes_dict = {1: 3, 2: 2, 4: 4}

    reg = MarginalShannonEntropyRegularizer(
        query_indices=query_indices,
        num_classes=5,
        num_modes=modes_dict,
        min_entropy_ratio=0.85,
    )

    B = 16
    u = torch.randn(B, 30, requires_grad=True)
    loss = reg(u)
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert loss.item() >= 0.0
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()

    # Slicing out of bounds: requires at least (4 + 1) * 5 = 25 columns
    u_narrow = torch.randn(B, 24)
    with pytest.raises(ValueError, match="required 25"):
        reg(u_narrow)


def test_marginal_shannon_entropy_input_modes_and_extremes():
    """Verify logits mode, probs mode, non-hinge mode, and extreme inputs (+/- 1000)."""
    # 1. input_mode="logits" (softmax)
    reg_logits = MarginalShannonEntropyRegularizer(
        query_indices=[0], num_classes=4, num_modes=2, input_mode="logits"
    )
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]], requires_grad=True)
    loss_l = reg_logits(logits)
    assert loss_l.item() > 0.0
    loss_l.backward()
    assert torch.isfinite(logits.grad).all()

    # 2. hinge=False (returns -H)
    reg_linear = MarginalShannonEntropyRegularizer(
        query_indices=[0], num_classes=4, num_modes=2, hinge=False
    )
    loss_lin = reg_linear(logits)
    assert loss_lin.item() < 0.0  # -H is negative

    # 3. Extreme logit inputs (+/- 1000, +/- 1e5) must not produce NaNs or Infs
    reg_default = MarginalShannonEntropyRegularizer(query_indices=[0], num_classes=5, num_modes=3)
    u_extreme = torch.tensor([[1000.0, -1000.0, -1000.0, 1e5, -1e5]], requires_grad=True)
    loss_ext = reg_default(u_extreme)
    assert torch.isfinite(loss_ext)
    loss_ext.backward()
    assert torch.isfinite(u_extreme.grad).all()

    # 4. Gradcheck
    u_double = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    reg_double = MarginalShannonEntropyRegularizer(query_indices=[0], num_classes=5, num_modes=3)
    assert torch.autograd.gradcheck(lambda x: reg_double(x), (u_double,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_marginal_shannon_entropy_validation_errors():
    """Verify input validation for constructor parameters."""
    with pytest.raises(ValueError, match="query_indices"):
        MarginalShannonEntropyRegularizer(query_indices=[])
    with pytest.raises(ValueError, match="nonnegative"):
        MarginalShannonEntropyRegularizer(query_indices=[-1])
    with pytest.raises(ValueError, match="duplicates"):
        MarginalShannonEntropyRegularizer(query_indices=[0, 0])
    with pytest.raises(ValueError, match="num_classes"):
        MarginalShannonEntropyRegularizer(num_classes=1)
    with pytest.raises(ValueError, match="num_modes"):
        MarginalShannonEntropyRegularizer(num_modes=1)
    with pytest.raises(ValueError, match="min_entropy_ratio"):
        MarginalShannonEntropyRegularizer(min_entropy_ratio=0.0)
    with pytest.raises(ValueError, match="min_entropy_ratio"):
        MarginalShannonEntropyRegularizer(min_entropy_ratio=1.5)
    with pytest.raises(ValueError, match="input_mode"):
        MarginalShannonEntropyRegularizer(input_mode="invalid")
    with pytest.raises(ValueError, match="eps"):
        MarginalShannonEntropyRegularizer(eps=-1e-4)

    reg = MarginalShannonEntropyRegularizer()
    with pytest.raises(ValueError, match="2D floating-point"):
        reg(torch.zeros(4, 5, dtype=torch.int32))
    with pytest.raises(ValueError, match="2D floating-point"):
        reg(torch.zeros(4, 5, 2))


# ============================================================================
# FEATURE 11: NormalizedVisionCosineRepulsion Stress Tests
# ============================================================================

def test_normalized_vision_cosine_repulsion_identical_margin_penalty():
    """Verify identical vectors incur exactly (1.0 - margin) penalty across margins and dimensions."""
    for margin in [0.0, 0.5, 0.88, 0.95]:
        rep = NormalizedVisionCosineRepulsion(margin=margin)
        expected_penalty = 1.0 - margin

        for D in [16, 128, 2560]:
            for B in [2, 4, 16]:
                # Identical vectors: repeat single vector across batch
                v = torch.randn(1, D).repeat(B, 1)
                loss = rep(v)
                assert float(loss.item()) == pytest.approx(expected_penalty, abs=1e-5), (
                    f"Margin {margin}, D={D}, B={B}: loss {loss.item()} != {expected_penalty}"
                )


def test_normalized_vision_cosine_repulsion_orthogonal_and_antipodal_zero():
    """Verify orthogonal and antipodal vectors incur strictly 0.0 penalty."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)

    # 1. Orthogonal vectors: cos(v_i, v_j) = 0.0 <= 0.88 -> 0.0
    v_ortho = torch.eye(8, 2560)
    assert rep(v_ortho).item() == 0.0

    # 2. Antipodal vectors: v_1 = -v_0 -> cos(v_0, v_1) = -1.0 <= 0.88 -> 0.0
    v_base = torch.randn(1, 2560)
    v_anti = torch.cat([v_base, -v_base], dim=0)
    assert rep(v_anti).item() == 0.0

    # 3. Disjoint support vectors
    v_disjoint = torch.zeros(4, 2560)
    v_disjoint[0, :640] = 1.0
    v_disjoint[1, 640:1280] = 1.0
    v_disjoint[2, 1280:1920] = 1.0
    v_disjoint[3, 1920:] = 1.0
    assert rep(v_disjoint).item() == 0.0


def test_normalized_vision_cosine_repulsion_boundary_batch_sizes():
    """Verify B <= 1 returns 0.0 with proper dtype and device without division errors."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)

    # B = 1
    res1 = rep(torch.randn(1, 2560))
    assert res1.item() == 0.0
    assert res1.dtype == torch.float32

    # B = 0 (empty batch)
    res0 = rep(torch.zeros(0, 2560))
    assert res0.item() == 0.0

    # Gradient requires_grad with B = 1
    v1 = torch.randn(1, 2560, requires_grad=True)
    l1 = rep(v1)
    assert l1.item() == 0.0


def test_normalized_vision_cosine_repulsion_gradient_separation_and_optimization():
    """Verify gradient separation dynamics: Adam optimizer drives clustered batch to distance >= 0.12."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)

    # 1. Gradient direction test on two close unit vectors
    torch.manual_seed(42)
    v0 = F.normalize(torch.randn(1, 2560), dim=-1)
    noise = F.normalize(torch.randn(1, 2560), dim=-1)
    v1 = F.normalize(v0 + 0.25 * noise, dim=-1)

    v = torch.cat([v0, v1], dim=0).requires_grad_(True)
    cos_init = F.cosine_similarity(v[0:1], v[1:2]).item()
    assert cos_init > 0.88

    loss = rep(v)
    loss.backward()

    # Step in negative gradient direction must reduce cosine similarity
    v_new = v.detach() - 0.5 * v.grad
    cos_step = F.cosine_similarity(v_new[0:1], v_new[1:2]).item()
    assert cos_step < cos_init, f"Gradient step failed to separate vectors: {cos_step} >= {cos_init}"

    # 2. Multi-vector optimization loop
    torch.manual_seed(202)
    base = torch.randn(1, 256)
    # 8 clustered vectors
    v_cluster = (base.repeat(8, 1) + 0.001 * torch.randn(8, 256)).requires_grad_(True)
    opt = torch.optim.Adam([v_cluster], lr=0.05)

    for step in range(1, 60):
        opt.zero_grad()
        l = rep(v_cluster)
        if l.item() == 0.0:
            break
        l.backward()
        opt.step()

    assert rep(v_cluster).item() == 0.0, "Failed to separate cluster"
    # Verify pairwise cosine distance >= 0.12 (similarity <= 0.88)
    v_norm = F.normalize(v_cluster.detach(), p=2, dim=-1)
    sims = torch.mm(v_norm, v_norm.t())
    mask = ~torch.eye(8, dtype=torch.bool)
    max_sim = sims[mask].max().item()
    assert max_sim <= 0.88 + 1e-4, f"Maximum cosine similarity {max_sim} exceeds margin 0.88"


def test_normalized_vision_cosine_repulsion_numerical_stability_and_gradcheck():
    """Verify eps prevents NaN on zero vectors, scale-invariance, and finite-difference gradient check."""
    rep = NormalizedVisionCosineRepulsion(margin=0.88)

    # 1. Zero vector inputs
    v_zeros = torch.zeros(4, 128, requires_grad=True)
    l_zero = rep(v_zeros)
    assert torch.isfinite(l_zero)
    l_zero.backward()
    assert torch.isfinite(v_zeros.grad).all()

    # 2. Scale invariance: scaling vectors by 1e-6 or 1e6 does not change penalty
    v_base = torch.randn(4, 128)
    l_base = rep(v_base).item()
    l_tiny = rep(v_base * 1e-6).item()
    l_huge = rep(v_base * 1e6).item()
    assert abs(l_tiny - l_base) < 1e-5
    assert abs(l_huge - l_base) < 1e-5

    # 3. Finite differences gradient check in float32 (the native precision of vision tower)
    torch.manual_seed(123)
    x = torch.randn(3, 16, dtype=torch.float32, requires_grad=True)
    loss = rep(x, margin=0.0)
    loss.backward()

    eps = 1e-3
    num_grad = torch.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.clone().detach()
            x_plus[i, j] += eps
            x_minus = x.clone().detach()
            x_minus[i, j] -= eps
            l_p = rep(x_plus, margin=0.0)
            l_m = rep(x_minus, margin=0.0)
            num_grad[i, j] = (l_p - l_m) / (2 * eps)

    torch.testing.assert_close(x.grad, num_grad, atol=1e-3, rtol=1e-3)


def test_normalized_vision_cosine_repulsion_validation():
    """Verify argument validations for NormalizedVisionCosineRepulsion."""
    for bad_margin in [-0.1, 1.0, 1.5, float('nan'), float('inf'), True]:
        with pytest.raises(ValueError, match="margin"):
            NormalizedVisionCosineRepulsion(margin=bad_margin)
    for bad_eps in [0.0, -1e-5, float('nan'), float('inf'), True]:
        with pytest.raises(ValueError, match="eps"):
            NormalizedVisionCosineRepulsion(eps=bad_eps)

    rep = NormalizedVisionCosineRepulsion()
    with pytest.raises(ValueError, match="requires a 2D"):
        rep(torch.randn(10))
    with pytest.raises(ValueError, match="requires a 2D"):
        rep(torch.randn(2, 3, 4))


# ============================================================================
# FEATURE 12: Multi-Scale Pyramid Generator Empirical Stress Tests
# ============================================================================

def test_multiscale_pyramid_generator_exact_parameter_count():
    """Verify both multiscale-pyramid-generator-64 and simple-apple-generator-64 have exactly 918,076 parameters."""
    pyramid_path = ROOT / 'examples' / 'networks' / 'multiscale-pyramid-generator-64.hndl'
    simple_path = ROOT / 'examples' / 'networks' / 'simple-apple-generator-64.hndl'

    assert pyramid_path.is_file(), f"Missing {pyramid_path}"
    assert simple_path.is_file(), f"Missing {simple_path}"

    pyramid_net = build_network(source=pyramid_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])
    simple_net = build_network(source=simple_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])

    p_pyramid = sum(p.numel() for p in pyramid_net.parameters())
    p_simple = sum(p.numel() for p in simple_net.parameters())

    assert p_pyramid == 918_076, f"Expected 918,076 params for multiscale pyramid, got {p_pyramid}"
    assert p_simple == 918_076, f"Expected 918,076 params for simple apple generator, got {p_simple}"


def test_multiscale_pyramid_generator_checkerboard_spectral_attenuation():
    """Verify that multi-scale pyramid generator strongly attenuates 2D Nyquist checkerboard frequencies (>20x reduction)."""
    # 1. Classical deconv stride=2 baseline known for checkerboard artifacts
    deconv_hndl = """
z = reshape(128, 1, 1)
h4 = deconv(z, 256, kernel_size=4, stride=1, padding=0, bias=False)
h8 = deconv(h4, 128, kernel_size=4, stride=2, padding=1, bias=False)
h16 = deconv(h8, 64, kernel_size=4, stride=2, padding=1, bias=False)
h32 = deconv(h16, 32, kernel_size=4, stride=2, padding=1, bias=False)
out = deconv(h32, 3, kernel_size=4, stride=2, padding=1, bias=True)
out = tanh(out)
"""
    deconv_net = build_network(source=deconv_hndl, input_shape=['B', 128], output_shape=['B', 3, 64, 64])

    pyramid_path = ROOT / 'examples' / 'networks' / 'multiscale-pyramid-generator-64.hndl'
    pyramid_net = build_network(source=pyramid_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])

    torch.manual_seed(42)
    z = torch.randn(16, 128)

    out_deconv = deconv_net(z)
    out_pyramid = pyramid_net(z)

    # Frequency domain 2D FFT: measure power at Nyquist frequency corners (highest diagonal frequencies)
    fft_deconv = torch.fft.fftshift(torch.fft.fft2(out_deconv), dim=(-2, -1)).abs().pow(2)
    fft_pyramid = torch.fft.fftshift(torch.fft.fft2(out_pyramid), dim=(-2, -1)).abs().pow(2)

    # 4 corner regions around Nyquist frequencies
    nyq_deconv = (fft_deconv[:, :, 0:4, 0:4].mean() + fft_deconv[:, :, -4:, -4:].mean()) / fft_deconv.mean()
    nyq_pyramid = (fft_pyramid[:, :, 0:4, 0:4].mean() + fft_pyramid[:, :, -4:, -4:].mean()) / fft_pyramid.mean()

    ratio = (nyq_deconv / nyq_pyramid).item()
    assert ratio > 20.0, f"Pyramid generator Nyquist corner attenuation ratio {ratio:.1f}x is not > 20x"

    # Also check that OpticalWavefrontPrior patch seam ratio R_seam <= 1.15
    prior = OpticalWavefrontPrior()
    metrics = prior(out_pyramid, return_dict=True)
    assert metrics["R_seam"].item() <= 1.15, f"Patch seam ratio {metrics['R_seam'].item()} exceeds 1.15"


def test_multiscale_pyramid_generator_smooth_latent_spatial_interpolation():
    """Verify smooth spatial interpolation in latent space without abrupt jumps or flash artifacts."""
    pyramid_path = ROOT / 'examples' / 'networks' / 'multiscale-pyramid-generator-64.hndl'
    pyramid_net = build_network(source=pyramid_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])

    torch.manual_seed(99)
    z0 = torch.randn(1, 128)
    z1 = torch.randn(1, 128)

    # 50 interpolation steps
    steps = 50
    alphas = torch.linspace(0.0, 1.0, steps)
    images = []
    with torch.no_grad():
        for a in alphas:
            za = (1.0 - a) * z0 + a * z1
            images.append(pyramid_net(za))

    step_diffs = [torch.norm(images[k + 1] - images[k]).item() for k in range(steps - 1)]
    z_step = torch.norm((alphas[1] - alphas[0]) * (z1 - z0)).item()

    # Lipschitz estimate along the interpolation chord
    lipschitz_estimates = [d / z_step for d in step_diffs]

    # Check 1: No jump discontinuities (max diff / min diff is bounded)
    diff_ratio = max(step_diffs) / min(step_diffs)
    assert diff_ratio < 3.0, f"Interpolation exhibited jump discontinuities: max/min ratio {diff_ratio:.2f} >= 3.0"

    # Check 2: Bounded maximum Lipschitz constant
    assert max(lipschitz_estimates) < 15.0, f"Excessive Lipschitz constant: {max(lipschitz_estimates):.2f}"

    # Check 3: Monotonic distance progression from start image
    dist_from_start = [torch.norm(img - images[0]).item() for img in images]
    # Verify overall upward trend
    assert dist_from_start[-1] > dist_from_start[steps // 2] > dist_from_start[1]


def test_multiscale_pyramid_generator_backward_gradient_flow_to_z():
    """Verify backward gradient flow to z across batch sizes and pyramid scales."""
    pyramid_path = ROOT / 'examples' / 'networks' / 'multiscale-pyramid-generator-64.hndl'
    pyramid_net = build_network(source=pyramid_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])

    for B in [1, 4, 16]:
        z = torch.randn(B, 128, requires_grad=True)
        out = pyramid_net(z)
        assert out.shape == (B, 3, 64, 64)

        loss = out.pow(2).mean()
        loss.backward()

        assert z.grad is not None, f"z.grad is None for B={B}"
        assert torch.isfinite(z.grad).all(), f"Non-finite gradients found for B={B}"

        # No gradient vanishing: norm is bounded away from 0
        grad_norm = z.grad.norm().item()
        assert grad_norm > 0.01, f"Vanishing gradient norm {grad_norm} for B={B}"

        # No gradient explosion
        assert grad_norm < 1e5, f"Exploding gradient norm {grad_norm} for B={B}"

    # Verify simple-apple-generator-64 as well
    simple_path = ROOT / 'examples' / 'networks' / 'simple-apple-generator-64.hndl'
    simple_net = build_network(source=simple_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])
    z_s = torch.randn(4, 128, requires_grad=True)
    out_s = simple_net(z_s)
    out_s.sum().backward()
    assert z_s.grad is not None and torch.isfinite(z_s.grad).all() and z_s.grad.norm().item() > 0.01


def test_multiscale_pyramid_generator_output_bounds_and_spatial_variance():
    """Verify output tensor strict [-1, 1] bounding and spatial non-degeneracy."""
    pyramid_path = ROOT / 'examples' / 'networks' / 'multiscale-pyramid-generator-64.hndl'
    pyramid_net = build_network(source=pyramid_path.read_text(encoding='utf-8'), input_shape=['B', 128], output_shape=['B', 3, 64, 64])

    torch.manual_seed(777)
    z = torch.randn(8, 128)
    out = pyramid_net(z)

    # 1. Output bounding in [-1.0, 1.0]
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0

    # 2. Non-flat images: standard deviation across spatial dimensions > 0.05
    spatial_std = out.std(dim=(-2, -1)).mean().item()
    assert spatial_std > 0.05, f"Generated images are degenerately flat: spatial std {spatial_std}"

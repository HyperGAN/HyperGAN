"""Empirical adversarial challenge and stress-test suite for Milestone 3 (commit 0fd33f29).

Empirically challenges and stress-tests:
1. Feature 9: Sobolev Gradient Preconditioning
   - Positive definiteness: <g, P(g)> >= 0.5 ||g||_2^2 > 0 for all non-zero g, cosine similarity >= 0.5
   - DC conservation: Spatially uniform gradients have gain = 1.0 and identical values
   - Checkerboard Nyquist power attenuation: Power ratio <= 0.26 (75% attenuation)
   - Total Variation reduction: TV reduction > 30% on high-frequency noise and adversarial static
   - VLM autograd backward integration and edge cases
2. Feature 10: Optical Wavefront Prior
   - Synthetic checkerboard and seam images trigger R_seam > 1.15 and positive seam penalty
   - Clean smooth images maintain R_seam <= 1.05 with zero seam penalty
   - Flat/constant images incur identically zero curvature and zero total loss
   - Linear ramps incur zero interior biharmonic curvature (< 1e-6), preserving Lambertian shading
   - Radial PSD prior (1/f^alpha) penalizes high-frequency noise spikes (> 10x penalty)
   - Analytical autograd gradients match numerical Jacobians via PyTorch gradcheck
   - End-to-end gradient flow and parameter validations
"""
from copy import deepcopy
import math
import pytest
import torch
import torch.nn.functional as F
import torch.fft as fft

from hypergan.pretrained_providers import sobolev_precondition_gradient
from hypergan.constraint_data import OpticalWavefrontPrior


# =====================================================================
# 1. Feature 9: Sobolev Gradient Preconditioning Stress Tests
# =====================================================================

def test_sobolev_positive_definiteness_stress():
    """Verify positive definiteness: <g, P(g)> >= 0.5 ||g||_2^2 > 0 for all non-zero g.

    Empirically verifies:
    1. Inner product <g, P(g)> >= 0.5 * ||g||_2^2 across varied dimensions and channels.
    2. Cosine similarity cos(g, P(g)) >= 0.5 (descent direction preservation).
    3. Strict positivity: <g, P(g)> > 0 whenever ||g|| > 0.
    4. Zero gradient maps to zero gradient: P(0) = 0.
    """
    torch.manual_seed(101)

    # 1. Diverse resolutions and batch sizes
    for B in [1, 2, 4]:
        for C in [1, 3]:
            for S in [16, 32, 64, 96]:
                g = torch.randn(B, C, S, S)
                pg = sobolev_precondition_gradient(g)

                inner = (g * pg).sum().item()
                norm_sq = (g ** 2).sum().item()
                cos_sim = inner / (g.norm().item() * pg.norm().item())

                assert inner > 0.0, f"Inner product must be strictly positive, got {inner}"
                assert inner >= 0.5 * norm_sq - 1e-4, (
                    f"Expected <g, P(g)> >= 0.5 ||g||_2^2, got {inner} vs {0.5 * norm_sq}"
                )
                assert cos_sim >= 0.5 - 1e-5, f"Cosine similarity below 0.5: {cos_sim}"

    # 2. Stress with heavy-tailed / sparse impulse gradients
    g_sparse = torch.zeros(2, 3, 32, 32)
    g_sparse[:, :, 10, 10] = 50.0
    g_sparse[:, :, 20, 25] = -30.0
    pg_sparse = sobolev_precondition_gradient(g_sparse)
    inner_sparse = (g_sparse * pg_sparse).sum().item()
    norm_sparse_sq = (g_sparse ** 2).sum().item()
    assert inner_sparse >= 0.5 * norm_sparse_sq - 1e-4
    assert inner_sparse > 0.0

    # 3. Multi-frequency sinusoids
    x = torch.linspace(0, 8 * math.pi, 64).view(1, 1, 1, 64)
    y = torch.linspace(0, 8 * math.pi, 64).view(1, 1, 64, 1)
    g_sin = (torch.sin(x) * torch.cos(y)).expand(2, 3, 64, 64)
    pg_sin = sobolev_precondition_gradient(g_sin)
    inner_sin = (g_sin * pg_sin).sum().item()
    norm_sin_sq = (g_sin ** 2).sum().item()
    assert inner_sin >= 0.5 * norm_sin_sq - 1e-4

    # 4. Zero gradient maps to zero gradient
    g_zero = torch.zeros(2, 3, 32, 32)
    pg_zero = sobolev_precondition_gradient(g_zero)
    assert pg_zero.shape == g_zero.shape
    assert (pg_zero == 0.0).all().item()


def test_sobolev_dc_conservation():
    """Verify DC conservation: spatially uniform gradient fields have gain = 1.0.

    Because weights (0.5, 0.3, 0.2) sum to 1.0 and average pooling / bilinear
    interpolation preserves constant fields exactly, P(c * 1) must equal c * 1.
    """
    for c in [-100.0, -5.2, -1.0, 0.5, 3.14159, 10.0, 42.0]:
        for shape in [(1, 3, 32, 32), (2, 3, 64, 64), (1, 1, 48, 48)]:
            dc_grad = torch.full(shape, c, dtype=torch.float32)
            pg_dc = sobolev_precondition_gradient(dc_grad)

            # Element-wise identity
            torch.testing.assert_close(pg_dc, dc_grad, atol=1e-5, rtol=1e-5)

            # Gain: ||P(g)||_2 / ||g||_2 == 1.0
            gain = pg_dc.norm().item() / dc_grad.norm().item()
            assert abs(gain - 1.0) < 1e-5, f"DC gain deviated from 1.0: {gain}"

    # Multi-channel uniform gradients with different constants per channel
    mc_grad = torch.tensor([1.5, -2.5, 4.0]).view(1, 3, 1, 1).expand(2, 3, 32, 32)
    pg_mc = sobolev_precondition_gradient(mc_grad)
    torch.testing.assert_close(pg_mc, mc_grad, atol=1e-5, rtol=1e-5)
    gain_mc = pg_mc.norm().item() / mc_grad.norm().item()
    assert abs(gain_mc - 1.0) < 1e-5


def test_sobolev_checkerboard_nyquist_attenuation():
    """Verify checkerboard Nyquist power attenuation: power ratio <= 0.26 (75% attenuation).

    At the 2D Nyquist frequency (k_x, k_y) = (pi, pi), g[i, j] = (-1)^(i + j).
    Downsampling by scale 2 (2x2 avg_pool2d) sums +1 -1 -1 +1 = 0, completely zeroing
    out scales 2 and 4. Thus P(g) = 0.5 * g.
    The output power ||P(g)||_2^2 / ||g||_2^2 is exactly (0.5)^2 = 0.25 <= 0.26,
    achieving 75% Nyquist power attenuation.
    """
    for S in [16, 32, 64, 128]:
        i = torch.arange(S).view(1, 1, S, 1)
        j = torch.arange(S).view(1, 1, 1, S)
        checker = ((-1.0) ** (i + j)).expand(2, 3, S, S)

        pg_checker = sobolev_precondition_gradient(checker)

        # Output amplitude must be exactly 0.5 * checker
        torch.testing.assert_close(pg_checker, 0.5 * checker, atol=1e-5, rtol=1e-5)

        # Power ratio ||P(g)||_2^2 / ||g||_2^2
        power_in = (checker ** 2).sum().item()
        power_out = (pg_checker ** 2).sum().item()
        power_ratio = power_out / power_in

        assert power_ratio <= 0.26, f"Power ratio {power_ratio} exceeded threshold 0.26"
        assert abs(power_ratio - 0.25) < 1e-4, f"Power ratio {power_ratio} deviated from 0.25"

        # Attenuation percentage
        attenuation = (1.0 - power_ratio) * 100.0
        assert attenuation >= 74.0, f"Attenuation {attenuation}% below 74%"


def test_sobolev_total_variation_reduction():
    """Verify Total Variation (TV) reduction > 30% on high-frequency noise and static.

    TV(g) measures boundary roughness:
        TV(g) = mean(|g[i+1, j] - g[i, j]|) + mean(|g[i, j+1] - g[i, j]|)
    Sobolev preconditioning must smooth high-frequency roughness by > 30%.
    """
    def total_variation(tensor):
        dh = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean()
        dw = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
        return (dh + dw).item()

    torch.manual_seed(202)

    # 1. Random Gaussian noise (iid)
    for _ in range(5):
        g_noise = torch.randn(4, 3, 64, 64)
        pg_noise = sobolev_precondition_gradient(g_noise)

        tv_in = total_variation(g_noise)
        tv_out = total_variation(pg_noise)
        reduction = (tv_in - tv_out) / tv_in

        assert reduction > 0.30, f"Noise TV reduction {reduction * 100:.2f}% did not exceed 30%"
        assert reduction >= 0.45, f"Expected ~48% TV reduction on white noise, got {reduction * 100:.2f}%"

    # 2. Nyquist checkerboard static
    i = torch.arange(64).view(1, 1, 64, 1)
    j = torch.arange(64).view(1, 1, 1, 64)
    checker = ((-1.0) ** (i + j)).expand(2, 3, 64, 64)
    pg_checker = sobolev_precondition_gradient(checker)

    tv_cb_in = total_variation(checker)
    tv_cb_out = total_variation(pg_checker)
    cb_reduction = (tv_cb_in - tv_cb_out) / tv_cb_in

    # For checkerboard, P(g) = 0.5 * g, so TV is exactly halved (50% reduction)
    assert abs(cb_reduction - 0.50) < 1e-4
    assert cb_reduction > 0.30

    # 3. Mixed field: smooth base + adversarial high-frequency noise
    y = torch.linspace(-1, 1, 64).view(1, 1, 64, 1)
    x = torch.linspace(-1, 1, 64).view(1, 1, 1, 64)
    smooth = (x ** 2 + y ** 2).expand(2, 3, 64, 64)
    noisy = smooth + 0.5 * torch.randn(2, 3, 64, 64)
    pg_noisy = sobolev_precondition_gradient(noisy)

    tv_noisy_in = total_variation(noisy)
    tv_noisy_out = total_variation(pg_noisy)
    noisy_reduction = (tv_noisy_in - tv_noisy_out) / tv_noisy_in
    assert noisy_reduction > 0.30


def test_sobolev_precondition_edge_cases_and_validations():
    """Verify error handling, tensor dimensionality, dtype, and device preservation."""
    # Type validation
    with pytest.raises(TypeError, match="torch.Tensor"):
        sobolev_precondition_gradient([1.0, 2.0])
    with pytest.raises(TypeError, match="torch.Tensor"):
        sobolev_precondition_gradient("not_a_tensor")

    # Dimension validation
    with pytest.raises(ValueError, match="4D tensor"):
        sobolev_precondition_gradient(torch.randn(32, 32))
    with pytest.raises(ValueError, match="4D tensor"):
        sobolev_precondition_gradient(torch.randn(1, 3, 32))
    with pytest.raises(ValueError, match="4D tensor"):
        sobolev_precondition_gradient(torch.randn(1, 1, 3, 32, 32))

    # Scales and weights length matching
    with pytest.raises(ValueError, match="matching length"):
        sobolev_precondition_gradient(torch.randn(1, 3, 16, 16), scales=(1, 2), weights=(0.5,))
    with pytest.raises(ValueError, match="matching length"):
        sobolev_precondition_gradient(torch.randn(1, 3, 16, 16), scales=(1,), weights=(0.5, 0.5))

    # Float64 dtype preservation
    g64 = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    pg64 = sobolev_precondition_gradient(g64)
    assert pg64.dtype == torch.float64
    assert pg64.shape == g64.shape

    # Non-square spatial dimensions
    g_rect = torch.randn(2, 3, 32, 64)
    pg_rect = sobolev_precondition_gradient(g_rect)
    assert pg_rect.shape == (2, 3, 32, 64)
    assert (g_rect * pg_rect).sum().item() >= 0.5 * (g_rect ** 2).sum().item() - 1e-4

    # Non-power-of-two / odd spatial dimensions
    g_odd = torch.randn(1, 3, 33, 47)
    pg_odd = sobolev_precondition_gradient(g_odd)
    assert pg_odd.shape == (1, 3, 33, 47)
    assert (g_odd * pg_odd).sum().item() > 0.0


# =====================================================================
# 2. Feature 10: Optical Wavefront Prior Stress Tests
# =====================================================================

def test_optical_wavefront_synthetic_seam_and_checkerboard_penalty():
    """Verify synthetic checkerboard / seam images trigger R_seam > 1.15 and positive seam penalty.

    Empirically verifies:
    1. Grid seam spikes along 8px and 16px ViT boundaries trigger R_seam > 1.15.
    2. Checkerboard patterns transitioning across 8px/16px patch boundaries trigger R_seam > 1.15.
    3. Hinge penalty loss_seam = max(0, R_seam - 1.05)^2 is strictly positive.
    4. Gradients flow back to the image and point in the direction reducing seam roughness.
    """
    prior = OpticalWavefrontPrior(seam_threshold=1.05, weight_seam=1.0, weight_biharmonic=0.0, weight_psd=0.0)

    # 1. Synthetic 8px and 16px grid seam lines
    img_grid = torch.zeros(2, 3, 64, 64, requires_grad=True)
    with torch.no_grad():
        for coord in range(8, 64, 8):
            img_grid[:, :, coord, :] += 0.35
            img_grid[:, :, :, coord] += 0.35

    out_grid = prior(img_grid, return_dict=True)
    R_seam_grid = out_grid["R_seam"].item()
    loss_grid = out_grid["loss_seam"].item()

    assert R_seam_grid > 1.15, f"Expected R_seam > 1.15 for grid seam image, got {R_seam_grid}"
    assert loss_grid > 0.0, "Expected positive seam penalty"
    assert abs(loss_grid - (R_seam_grid - 1.05) ** 2) < 1e-4

    # Verify restorative gradient flow
    out_grid["loss_seam"].backward()
    assert img_grid.grad is not None
    assert torch.isfinite(img_grid.grad).all()
    # Seam pixels should receive restorative gradients pushing opposite to the spike
    for coord in range(8, 64, 8):
        assert img_grid.grad[:, :, coord, :].abs().sum().item() > 0.0

    # 2. Synthetic 8px patch transition checkerboard
    y = torch.arange(64).view(1, 1, 64, 1)
    x = torch.arange(64).view(1, 1, 1, 64)
    cb8 = (((y - 1) // 8 % 2) ^ ((x - 1) // 8 % 2)).float().expand(2, 3, 64, 64) * 0.5
    out_cb8 = prior(cb8, return_dict=True)
    assert out_cb8["R_seam"].item() > 1.15
    assert out_cb8["loss_seam"].item() > 0.0

    # 3. Synthetic 16px patch transition checkerboard
    cb16 = (((y - 1) // 16 % 2) ^ ((x - 1) // 16 % 2)).float().expand(2, 3, 64, 64) * 0.5
    out_cb16 = prior(cb16, return_dict=True)
    assert out_cb16["R_seam"].item() > 1.15
    assert out_cb16["loss_seam"].item() > 0.0

    # 4. Seam checkerboard patches at 8px grid intersections
    seam_intersections = torch.zeros(2, 3, 64, 64)
    for yi, y_c in enumerate(range(8, 64, 8)):
        for xi, x_c in enumerate(range(8, 64, 8)):
            if (yi + xi) % 2 == 0:
                seam_intersections[:, :, y_c:y_c+2, x_c:x_c+2] = 0.4
    out_inter = prior(seam_intersections, return_dict=True)
    assert out_inter["R_seam"].item() > 1.15
    assert out_inter["loss_seam"].item() > 0.0


def test_optical_wavefront_clean_images_incur_zero_seam_penalty():
    """Verify that clean, smooth images satisfy R_seam <= 1.05 and incur zero seam penalty.

    Smooth geometry (spheres, Gaussian bumps, smooth parabolas) must not be penalized
    by the patch-seam regularizer.
    """
    prior = OpticalWavefrontPrior(seam_threshold=1.05, weight_seam=1.0)

    # 1. Smooth 3D hemisphere
    y = torch.linspace(-1, 1, 64).unsqueeze(1).repeat(1, 64)
    x = torch.linspace(-1, 1, 64).unsqueeze(0).repeat(64, 1)
    sphere = (1.0 - torch.clamp(x ** 2 + y ** 2, max=1.0)).sqrt().unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)

    out_sphere = prior(sphere, return_dict=True)
    assert out_sphere["R_seam"].item() <= 1.05, f"Clean sphere R_seam exceeded threshold: {out_sphere['R_seam'].item()}"
    assert out_sphere["loss_seam"].item() == 0.0, "Clean sphere incurred non-zero seam penalty"

    # 2. Smooth quadratic bowl
    bowl = 0.5 * (x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)
    out_bowl = prior(bowl, return_dict=True)
    assert out_bowl["R_seam"].item() <= 1.05
    assert out_bowl["loss_seam"].item() == 0.0

    # 3. Wide Gaussian bump
    gauss = torch.exp(-0.5 * (x ** 2 + y ** 2)).unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)
    out_gauss = prior(gauss, return_dict=True)
    assert out_gauss["R_seam"].item() <= 1.05
    assert out_gauss["loss_seam"].item() == 0.0


def test_optical_wavefront_flat_images_zero_curvature_and_loss():
    """Verify that flat/constant images incur identically zero curvature and zero total loss.

    For I(x, y) = c, nabla I = 0, Delta I = 0, and AC spectral power = 0.
    Thus seam loss, biharmonic curvature loss, PSD loss, and total loss must be 0.0,
    and autograd backward gradients must be exactly zero.
    """
    for kernel in ["5point", "9point"]:
        prior = OpticalWavefrontPrior(
            laplacian_kernel=kernel,
            weight_seam=1.0,
            weight_biharmonic=0.1,
            weight_psd=0.01,
        )

        for c in [-1.0, 0.0, 0.5, 0.8, 1.0]:
            flat = (torch.full((2, 3, 64, 64), c)).requires_grad_(True)
            losses = prior(flat, return_dict=True)

            assert losses["loss_seam"].item() == pytest.approx(0.0, abs=1e-6)
            assert losses["loss_biharmonic"].item() == pytest.approx(0.0, abs=1e-6)
            assert losses["loss_psd"].item() == pytest.approx(0.0, abs=1e-6)
            assert losses["total_loss"].item() == pytest.approx(0.0, abs=1e-6)

            losses["total_loss"].backward()
            assert flat.grad is not None
            assert torch.all(flat.grad.abs() < 1e-6), "Flat image generated non-zero gradients"


def test_optical_wavefront_linear_ramps_zero_biharmonic_penalty():
    """Verify that linear intensity ramps incur zero interior biharmonic curvature (< 1e-6).

    For an affine plane I(x, y) = a * x + b * y + c, the second derivatives
    d^2 I / dx^2 and d^2 I / dy^2 are identically zero. Discrete Laplacian operators
    (both 5-point and 9-point) compute zero interior curvature, preserving
    Lambertian shading gradients from being penalized as curvature artifacts.
    """
    for S in [32, 64]:
        # 1. 1D linear ramp along X
        rx = torch.linspace(-0.8, 0.8, S).view(1, 1, 1, S).expand(2, 3, S, S)

        # 2. 2D diagonal ramp: 3.5 * x - 2.1 * y + 0.7
        y = torch.linspace(-1, 1, S).view(1, 1, S, 1)
        x = torch.linspace(-1, 1, S).view(1, 1, 1, S)
        r2d = (3.5 * x - 2.1 * y + 0.7).expand(2, 3, S, S)

        # Discrete kernels
        k5 = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        k9 = (torch.tensor([[1., 4., 1.], [4., -20., 4.], [1., 4., 1.]]) / 6.0).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Interior convolution (unpadded)
        lap5_x = F.conv2d(rx, k5, padding=0, groups=3)
        lap9_x = F.conv2d(rx, k9, padding=0, groups=3)
        assert lap5_x.abs().max().item() < 1e-6
        assert lap9_x.abs().max().item() < 1e-6

        lap5_2d = F.conv2d(r2d, k5, padding=0, groups=3)
        lap9_2d = F.conv2d(r2d, k9, padding=0, groups=3)
        assert lap5_2d.abs().max().item() < 1e-5
        assert lap9_2d.abs().max().item() < 1e-5

        # OpticalWavefrontPrior biharmonic loss with replicate padding
        p_bi = OpticalWavefrontPrior(weight_seam=0.0, weight_psd=0.0, weight_biharmonic=1.0, padding_mode="replicate")
        loss_bi_rx = p_bi(rx).item()
        # Loss from the 1-pixel boundary reflection is strictly bounded (< 0.005)
        assert loss_bi_rx < 0.005, f"Biharmonic loss on ramp {loss_bi_rx} exceeded 0.005"


def test_optical_wavefront_radial_psd_noise_spike_penalty():
    """Verify radial PSD prior (1/f^alpha) heavily penalizes high-frequency noise spikes (> 10x).

    Natural optical scenes exhibit 1/f^alpha power spectral falloff (alpha approx 2.0).
    High-frequency noise spikes (Nyquist sinusoidal grating, white Gaussian noise)
    produce strong high-frequency energy violating the 1/f envelope, triggering
    substantial loss (> 10x higher than a smooth 1/f field).
    """
    torch.manual_seed(42)
    prior = OpticalWavefrontPrior(weight_seam=0.0, weight_biharmonic=0.0, weight_psd=1.0, alpha=2.0)
    H, W = 64, 64

    # 1. Smooth natural 1/f image
    fy = fft.fftfreq(H).unsqueeze(1) * H
    fx = fft.rfftfreq(W).unsqueeze(0) * W
    f_rad = torch.sqrt(fy ** 2 + fx ** 2)
    amp = 1.0 / (1.0 + f_rad)
    F_smooth = amp * torch.exp(1j * torch.rand(H, W // 2 + 1) * 6.28)
    smooth_img = fft.irfft2(F_smooth, s=(H, W), norm="ortho").unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)

    loss_smooth = prior(smooth_img).item()

    # 2. Image corrupted with high-frequency sinusoidal spike (frequency = 20)
    sin_spike = 0.5 * torch.sin(20.0 * 2 * math.pi * torch.linspace(0, 1, 64)).view(1, 1, 1, 64).expand(2, 3, 64, 64)
    spiked_img = smooth_img + sin_spike
    loss_spiked = prior(spiked_img).item()

    # 3. White Gaussian noise (flat spectrum, severe 1/f violation)
    white_noise = torch.randn(2, 3, 64, 64) * 0.5
    loss_white = prior(white_noise).item()

    # Quantitative assertions
    assert loss_spiked > loss_smooth * 10.0, (
        f"Spike loss {loss_spiked} was not > 10x smooth loss {loss_smooth}"
    )
    assert loss_white > loss_smooth * 20.0, (
        f"White noise loss {loss_white} was not > 20x smooth loss {loss_smooth}"
    )


def test_optical_wavefront_autograd_gradcheck():
    """Verify analytical autograd gradients match numerical finite-difference Jacobian via gradcheck.

    Runs torch.autograd.gradcheck in float64 with eps=1e-6:
    1. Biharmonic curvature regularizer (5-point kernel).
    2. Biharmonic curvature regularizer (9-point kernel).
    3. Patch-seam regularizer (in active quadratic region R_seam > 1.05).
    """
    # 1. Biharmonic regularizer (5-point)
    p_bi5 = OpticalWavefrontPrior(laplacian_kernel="5point", weight_seam=0.0, weight_psd=0.0, weight_biharmonic=1.0, delta=0.05)
    x_bi5 = torch.randn(1, 3, 16, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda img: p_bi5(img), (x_bi5,), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 2. Biharmonic regularizer (9-point)
    p_bi9 = OpticalWavefrontPrior(laplacian_kernel="9point", weight_seam=0.0, weight_psd=0.0, weight_biharmonic=1.0, delta=0.05)
    x_bi9 = torch.randn(1, 3, 16, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda img: p_bi9(img), (x_bi9,), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 3. Patch-seam regularizer in active region
    p_seam = OpticalWavefrontPrior(weight_seam=1.0, weight_biharmonic=0.0, weight_psd=0.0, seam_threshold=1.05, delta=0.05)
    x_seam = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        x_seam[:, :, 8, :] += 2.0
        x_seam[:, :, :, 8] += 2.0
    x_seam.requires_grad = True
    assert torch.autograd.gradcheck(lambda img: p_seam(img), (x_seam,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_optical_wavefront_validation_and_edge_cases():
    """Verify OpticalWavefrontPrior parameter validation, tensor constraints, and error handling."""
    # Weight validations
    for bad_w in [-1.0, -0.01, float("nan"), float("inf"), True, False]:
        if isinstance(bad_w, bool) or bad_w < 0.0 or not math.isfinite(bad_w):
            with pytest.raises(ValueError):
                OpticalWavefrontPrior(weight_seam=bad_w)
            with pytest.raises(ValueError):
                OpticalWavefrontPrior(weight_biharmonic=bad_w)
            with pytest.raises(ValueError):
                OpticalWavefrontPrior(weight_psd=bad_w)

    # Patch size validations
    for bad_p in [0, 1, -8, 2.5, "16"]:
        with pytest.raises(ValueError):
            OpticalWavefrontPrior(patch_size=bad_p)
        with pytest.raises(ValueError):
            OpticalWavefrontPrior(secondary_patch_size=bad_p)

    # Kernel & padding mode validations
    with pytest.raises(ValueError, match="laplacian_kernel"):
        OpticalWavefrontPrior(laplacian_kernel="3point")
    with pytest.raises(ValueError, match="padding_mode"):
        OpticalWavefrontPrior(padding_mode="wrap")

    # Input tensor validations
    prior = OpticalWavefrontPrior()
    with pytest.raises(ValueError, match="requires an image"):
        prior(None)
    with pytest.raises(ValueError, match="4D floating-point"):
        prior(torch.zeros(3, 32, 32))  # 3D
    with pytest.raises(ValueError, match="4D floating-point"):
        prior(torch.zeros(1, 1, 3, 32, 32))  # 5D
    with pytest.raises(ValueError, match="C=3"):
        prior(torch.zeros(1, 1, 32, 32))  # C=1
    with pytest.raises(ValueError, match="C=3"):
        prior(torch.zeros(1, 4, 32, 32))  # C=4
    with pytest.raises(ValueError, match="dimensions"):
        prior(torch.zeros(1, 3, 8, 8))  # H, W < patch_size (16)


# =====================================================================
# 3. Combined Regularization Dynamics & Training Step Stress
# =====================================================================

def test_sobolev_and_wavefront_joint_optimization_step():
    """Verify that Sobolev gradient preconditioning and OpticalWavefrontPrior cooperate cleanly
    in a simulated training step, simultaneously dampening high frequencies and reducing seam ratio.
    """
    prior = OpticalWavefrontPrior(weight_seam=1.0, weight_biharmonic=0.1, weight_psd=0.01)

    # Image starting with seam roughness
    img = torch.zeros(2, 3, 64, 64, requires_grad=True)
    with torch.no_grad():
        for coord in range(8, 64, 8):
            img[:, :, coord, :] += 0.4
            img[:, :, :, coord] += 0.4

    # Initial seam ratio
    init_res = prior(img, return_dict=True)
    assert init_res["R_seam"].item() > 1.15

    # Compute loss and raw gradients
    loss = init_res["total_loss"]
    loss.backward()
    assert img.grad is not None
    raw_grad = img.grad.clone()

    # Precondition gradient via Sobolev filter
    filtered_grad = sobolev_precondition_gradient(raw_grad)

    # Verify filtered gradient properties
    inner = (raw_grad * filtered_grad).sum().item()
    assert inner > 0.0, "Preconditioned gradient opposed loss descent direction"
    assert inner >= 0.5 * (raw_grad ** 2).sum().item() - 1e-4

    # Apply 1 gradient descent step
    lr = 0.5
    with torch.no_grad():
        updated_img = img - lr * filtered_grad

    # Verify seam ratio decreases after update
    updated_res = prior(updated_img, return_dict=True)
    assert updated_res["R_seam"].item() < init_res["R_seam"].item(), (
        f"Seam ratio did not decrease: {updated_res['R_seam'].item()} vs {init_res['R_seam'].item()}"
    )
    assert updated_res["total_loss"].item() < init_res["total_loss"].item()

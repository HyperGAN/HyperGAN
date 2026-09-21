"""Native HNDL image operators under deterministic CUDA, with double backward."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import pytest
import torch
from particlegan import GradientPenalty

from hypergan.hndl_networks import build_network
from hypergan.image_components import _PixelDiscriminator


@pytest.fixture(autouse=True)
def deterministic_cuda():
    assert torch.cuda.is_available(), 'CUDA acceptance requires a local GPU'
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize('size', [7, 16])
def test_native_feature_pool_deterministic_second_derivative(size):
    pool = build_network('adaptive_avg_pool(4)', input_shape=('B', 8, size, size),
                         output_shape=('B', 8, 4, 4)).cuda()
    source = torch.randn(2, 8, size, size, device='cuda')
    repeated = []
    for _ in range(2):
        x = source.detach().clone().requires_grad_()
        output = pool(x)
        first, = torch.autograd.grad(output.sin().square().sum(), x, create_graph=True)
        second, = torch.autograd.grad(first.square().sum(), x)
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        repeated.append((output.detach(), first.detach(), second.detach()))
    for actual, expected in zip(*repeated):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference = torch.nn.functional.adaptive_avg_pool2d(source, 4)
    torch.testing.assert_close(repeated[0][0], reference, atol=1e-6, rtol=1e-6)


def test_native_sagan_broadcast_and_constants_support_pixel_bcap():
    pixel = _PixelDiscriminator(width=8).cuda()
    real = torch.randn(2, 3, 32, 32, device='cuda')
    fake = torch.randn_like(real)

    def critic(x):
        return pixel(x, torch.zeros_like(x)).flatten()

    penalty = GradientPenalty(arm='b_cap', kappa=0)(critic, real, fake)
    (penalty + critic(real).square().mean()).backward()
    assert torch.isfinite(penalty) and penalty >= 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in pixel.parameters())

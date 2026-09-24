"""The style transformer matches the coordinate renderer and opens latent paths."""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from hypergan.hndl_networks import build_network

SOURCE = Path(__file__).parents[2] / 'examples/networks/simple-styletransformer-generator-128.hndl'


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        return build_network(SOURCE.read_text(), input_shape=('B', 512),
                             output_shape=('B', 3, 128, 128))


def reference(m, z):
    """Plain PyTorch equations, including packed MultiheadAttention convention."""
    def linear(x, layer):
        return F.linear(x, layer.weight, layer.bias)

    w = z
    for i in range(3):
        w = F.leaky_relu(linear(w, m[f'mapping{i}']), .2)
    theta = linear(w, m['affine_prediction']).reshape(-1, 2, 2)
    theta = theta + torch.eye(2, device=z.device, dtype=z.dtype)

    def coords(side):
        y, x = torch.meshgrid(torch.linspace(-1, 1, side, device=z.device, dtype=z.dtype),
                              torch.linspace(-1, 1, side, device=z.device, dtype=z.dtype), indexing='ij')
        return torch.stack((x, y), -1).reshape(1, side*side, 2).expand(z.shape[0], -1, -1) @ theta.transpose(1, 2)

    def fourier(c):
        p = c @ m['fourier'].frequencies.t()
        return torch.cat((p.sin(), p.cos()), -1)

    h = linear(fourier(coords(16)), m['input_projection'])
    for i in range(6):
        shifts = linear(F.silu(w), m[f'block{i}_modulation']).chunk(6, dim=1)
        sa, ca, ga, sm, cm, gm = (a[:, None, :] for a in shifts)
        a = h * (1 + ca) + sa
        attn = m[f'block{i}_attention']
        q, k, v = [linear(a, p).reshape(z.shape[0], 256, 8, 48).transpose(1, 2)
                   for p in (attn.q_proj, attn.k_proj, attn.v_proj)]
        scores = (q @ k.transpose(-1, -2) / 48**.5).softmax(-1)
        a = (scores @ v).transpose(1, 2).reshape(z.shape[0], 256, 384)
        h = h + ga * linear(a, attn.o_proj)
        ff = m[f'block{i}_ffn']
        h = h + gm * linear(F.gelu(linear(h * (1 + cm) + sm, ff.up)), ff.down)
    grid = coords(128).reshape(-1, 128, 128, 2)
    h = h.transpose(1, 2).reshape(-1, 384, 16, 16)
    h = F.grid_sample(h, grid, align_corners=False).flatten(2).transpose(1, 2)
    h = torch.cat((h, fourier(grid.flatten(1, 2))), -1)
    for name in ('renderer0', 'renderer1'):
        h = F.leaky_relu(linear(h, m[name]), .2)
    return linear(h, m['renderer_rgb']).tanh().transpose(1, 2).reshape(-1, 3, 128, 128)


def test_structure_and_initial_zero_gates(model):
    assert sum(p.numel() for p in model.parameters()) == 19_036_935
    assert len([n for n in model.plan.nodes if n.id.endswith('_attention')]) == 6
    assert model['fourier'].frequencies.shape == (192, 2)
    for name in ['affine_prediction', *(f'block{i}_modulation' for i in range(6))]:
        assert model[name].weight.count_nonzero() == 0
        assert model[name].bias.count_nonzero() == 0
    z = torch.arange(1024).float().cos().reshape(2, 512).requires_grad_()
    rng = torch.get_rng_state().clone()
    out = model(z)
    assert out.shape == (2, 3, 128, 128) and torch.isfinite(out).all()
    assert out.abs().max() <= 1
    torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)
    dz, dtheta, dstyle = torch.autograd.grad(out.square().mean(),
        (z, model['affine_prediction'].weight, model['block0_modulation'].weight))
    assert dz.count_nonzero() == 0
    assert torch.isfinite(dtheta).all() and dtheta.abs().sum() > 0
    assert torch.isfinite(dstyle).all() and dstyle.abs().sum() > 0
    assert torch.equal(rng, torch.get_rng_state())


def test_nonidentity_geometry_and_active_blocks_match_reference_and_gradients(model):
    # Exercise the full graph beyond the initial zero-gated state, without RNG.
    with torch.no_grad():
        model['affine_prediction'].weight.copy_(
            torch.arange(2048).reshape(4, 512).sin() * .003)
        model['affine_prediction'].bias.copy_(torch.tensor([.02, -.03, .01, -.02]))
        for i in range(6):
            model[f'block{i}_modulation'].weight.fill_(.0001)
            model[f'block{i}_modulation'].bias.copy_(torch.arange(2304).sin() * .04)
    z = torch.arange(1024).float().cos().reshape(2, 512).requires_grad_()
    actual = model(z)
    expected = reference(model, z)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    selected = (z, model['affine_prediction'].weight, model['mapping0'].weight,
                model['block0_attention'].q_proj.weight, model['block5_ffn'].up.weight)
    actual_grad = torch.autograd.grad(actual.square().mean(), selected)
    expected_grad = torch.autograd.grad(expected.square().mean(), selected)
    for a, b in zip(actual_grad, expected_grad):
        assert torch.isfinite(a).all() and a.abs().sum() > 0
        torch.testing.assert_close(a, b, atol=2e-7, rtol=5e-4)
    with torch.no_grad():
        isolated = model(z[:1])
    torch.testing.assert_close(actual[:1], isolated, atol=2e-6, rtol=2e-5)
    assert (actual[0] - actual[1]).abs().max() > 1e-6


def test_one_update_opens_latent_gradient(model):
    opt = torch.optim.Adam(model.parameters(), lr=.00002, betas=(.5, .999))
    z = torch.arange(512).float().cos().reshape(1, 512).requires_grad_()
    model(z).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    out = model(z)
    dz, dmapping = torch.autograd.grad(out.square().mean(), (z, model['mapping0'].weight))
    assert torch.isfinite(dz).all() and dz.abs().sum() > 0
    assert torch.isfinite(dmapping).all() and dmapping.abs().sum() > 0

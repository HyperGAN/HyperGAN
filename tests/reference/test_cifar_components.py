"""Small CPU contracts; real pinned source/weight parity is external evidence."""
import hashlib

import pytest
import torch

from hypergan.image_components import CIFARDiscriminator, CIFARGenerator, CIFARRoutingEncoder, SAGANAttention


def test_pretrained_artifact_missing_or_wrong_fails_without_download(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Artifact validation must never download')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    with pytest.raises(FileNotFoundError, match='no automatic downloads'):
        CIFARDiscriminator(str(tmp_path / 'missing'))
    bad = tmp_path / 'weights'
    bad.write_bytes(b'not the declared weight artifact')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        CIFARDiscriminator(str(bad))
    with pytest.raises(ValueError, match='lowercase SHA256'):
        CIFARDiscriminator(str(bad), weights_sha256='bad')


def test_generator_encoder_reconstruction_gradient_ownership():
    torch.manual_seed(42)
    g, e = CIFARGenerator(z_dim=8, width=8), CIFARRoutingEncoder(z_dim=8, width=8)
    means = torch.randn(64, 8, requires_grad=True)
    images = torch.randn(2, 3, 32, 32).tanh()
    routed = e(images, means, torch.tensor(.2))
    assert set(routed) == {'latent', 'ids', 'offset', 'soft'}
    assert torch.equal(routed['latent'], means[routed['ids']])  # Offset starts zero.
    torch.testing.assert_close(routed['soft'].sum(1), torch.ones(2))
    assert routed['offset'].abs().max() <= 3
    original = [p.requires_grad for p in g.parameters()]
    g.requires_grad_(False)
    generated = g(routed['latent'])
    # Matches the graph alias: restore ownership after building the forward.
    for parameter, trainable in zip(g.parameters(), original):
        parameter.requires_grad_(trainable)
    torch.nn.functional.mse_loss(generated, images).backward()
    assert generated.shape == images.shape and generated.abs().max() <= 1
    assert means.grad is None and all(p.grad is None for p in g.parameters())
    assert e.query.weight.grad.abs().sum() > 0
    assert e.offset.weight.grad.abs().sum() > 0


def test_attention_has_active_residual_and_double_backward():
    torch.manual_seed(123)
    layer = SAGANAttention(8)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    out = layer(x)
    assert not torch.equal(out, x)
    grad, = torch.autograd.grad(out.square().sum(), x, create_graph=True)
    grad.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='positive and finite'):
        CIFARRoutingEncoder(temperature=temperature)


@pytest.mark.parametrize('size', [4, 8, 16])
def test_deterministic_feature_pool_source_forward_and_two_derivatives(size):
    from hypergan.hndl_networks import build_network
    torch.manual_seed(218)
    x = torch.randn(2, 64, size, size, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    pool = build_network('adaptive_avg_pool(4)', input_shape=('B', 64, size, size),
                         output_shape=('B', 64, 4, 4)).double()
    actual = pool(x)
    expected = torch.nn.functional.adaptive_avg_pool2d(other, 4)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    first, = torch.autograd.grad(actual.sin().square().sum(), x, create_graph=True)
    reference, = torch.autograd.grad(expected.sin().square().sum(), other, create_graph=True)
    torch.testing.assert_close(first, reference, atol=1e-14, rtol=1e-14)
    second, = torch.autograd.grad(first.square().sum(), x)
    second_reference, = torch.autograd.grad(reference.square().sum(), other)
    torch.testing.assert_close(second, second_reference, atol=1e-14, rtol=1e-14)


def test_native_deterministic_pool_supports_ragged_shapes():
    from hypergan.hndl_networks import build_network
    pool = build_network('adaptive_avg_pool(4)', input_shape=('B', 1, 7, 7),
                         output_shape=('B', 1, 4, 4)).double()
    x = torch.randn(1, 1, 7, 7, dtype=torch.float64, requires_grad=True)
    actual = pool(x)
    expected = torch.nn.functional.adaptive_avg_pool2d(x, 4)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    gradient, = torch.autograd.grad(actual.square().sum(), x, create_graph=True)
    second, = torch.autograd.grad(gradient.square().sum(), x)
    assert torch.isfinite(second).all()


def test_generator_architecture_can_be_replaced_by_configuration():
    from hndl.torch import Network
    generator = CIFARGenerator(z_dim=8, networks={
        'image_generator': 'linear(3072)\nreshape(3, 32, 32)\nsigmoid()'})
    assert isinstance(generator.network, Network)
    image = generator(torch.randn(2, 8))
    assert image.shape == (2, 3, 32, 32)
    assert image.min() >= 0 and image.max() <= 1
    assert [node.op.split('@')[0] for node in generator.network.plan.nodes] == [
        'linear', 'reshape', 'sigmoid']

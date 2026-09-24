"""The small generator has an immediate, sample-independent latent path."""
from pathlib import Path
import tomllib

import pytest
import torch

from hypergan.hndl_networks import build_network

NETWORKS = Path(__file__).parents[2] / 'examples/networks'


@pytest.mark.parametrize('side,z_dim', [(32, 64), (128, 512), (128, 64)])
def test_initial_conditioning_gradients_and_checkpoint(side, z_dim):
    if (side, z_dim) == (128, 64):
        config = tomllib.loads((NETWORKS.parent / 'logos-tiny-transformer-resnet-features-128.toml').read_text())
        source = config['components']['generator']['args']['source']
    else:
        source = (NETWORKS / f'tiny-transformer-generator-{side}.hndl').read_text()
    with torch.random.fork_rng(devices=[]):
        model = build_network(source, input_shape=('B', z_dim),
                              output_shape=('B', 3, side, side))
        restored = build_network(source, input_shape=('B', z_dim),
                                 output_shape=('B', 3, side, side))
    z = torch.arange(2 * z_dim).float().cos().reshape(2, z_dim).requires_grad_()
    rng = torch.get_rng_state().clone()
    out = model(z)
    assert out.shape == (2, 3, side, side)
    assert torch.isfinite(out).all() and out.abs().max() <= 1
    assert (out[0] - out[1]).square().mean().sqrt() > 1e-4
    out.square().mean().backward()
    assert torch.isfinite(z.grad).all()
    assert (z.grad.norm(dim=1) > 0).all()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # Both transformer branches and the decoder must participate immediately.
    for name in ('input_projection', 'block0_attention', 'block1_attention',
                 'block0_ffn', 'block1_ffn', 'conv16', f'conv{side}', 'rgb'):
        assert sum(p.grad.abs().sum() for p in model[name].parameters()) > 0, name
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(z), out, rtol=0, atol=0)
        torch.testing.assert_close(model(z[:1]), out[:1], rtol=1e-4, atol=1e-6)
    assert torch.equal(rng, torch.get_rng_state())

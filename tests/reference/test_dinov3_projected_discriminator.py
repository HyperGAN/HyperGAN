"""128px DINOv3 projected critic under ParticleGAN defaults."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from hypergan.config import load_config
from hypergan.hndl_networks import HNDLNetwork, build_network
from hypergan.training import initialize_fresh_parameters
from tests.dinov3_fixtures import dinov3_assets


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _UpstreamDINOFixture(nn.Module):
    """Four native 8x8 patch maps."""
    def __init__(self, side=128):
        super().__init__()
        self.network = build_network('avg_pool(16)\nconv(384, kernel_size=1)\nlayer_norm()\ntanh()',
                                     input_shape=('B', 3, side, side),
                                     output_shape=('B', 384, side // 16, side // 16))
        self.calls = []

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        self.calls.append((tuple(x.shape), n, reshape, norm))
        features = self.network(x)
        return tuple(features * (index + 1) / 4 for index in range(4))


def assets(monkeypatch, tmp_path, side=128):
    source, commit, weights, digest = dinov3_assets(monkeypatch, tmp_path, lambda: _UpstreamDINOFixture(side))
    return {'dinov3_vits16': {'source_path': source, 'source_commit': commit}}, {
        'weights_path': weights, 'weights_sha256': digest}


def test_dinov3_projected_128_orthogonal_init_and_second_derivatives(monkeypatch, tmp_path):
    providers, parameters = assets(monkeypatch, tmp_path)
    root = Path(__file__).parents[2]
    source = (root / 'examples/networks/dinov3-projected-discriminator-128.hndl').read_text()
    model = HNDLNetwork(source, input_shape=('B', 3, 128, 128), output_shape=('B', 1),
                        parameters=parameters, pretrained_providers=providers)
    backbone_module = model.network['backbone']
    backbone = backbone_module.model
    frozen_before = copy.deepcopy(backbone.state_dict())

    graph = nn.Module()
    graph.models = nn.ModuleDict({'discriminator': model})
    initialize_fresh_parameters(graph, seed=0)

    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen_before[key], rtol=0, atol=0)
    assert all(not p.requires_grad for p in backbone.parameters())

    outside = [m for m in model.network.modules() if id(m) not in {id(b) for b in backbone_module.modules()}]
    convs = [m for m in outside if isinstance(m, nn.Conv2d)]
    assert len(convs) == 12
    assert not any(hasattr(m, 'parametrizations') for m in convs)
    assert not any(isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)) for m in outside)

    x = torch.linspace(-1, 1, 2 * 3 * 128 * 128).reshape(2, 3, 128, 128).requires_grad_()
    score = model(x=x)
    assert score.shape == (2, 1) and torch.isfinite(score).all()
    assert backbone.calls == [((2, 3, 128, 128), (2, 5, 8, 11), True, True)]

    first, = torch.autograd.grad(score.mean(), x, create_graph=True, retain_graph=True)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    (score.mean() + first.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for conv in convs:
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
        assert conv.weight.grad.abs().sum() > 0


def test_logos_projected_example_uses_particlegan_defaults():
    root = Path(__file__).parents[2]
    resolved = load_config(root / 'examples/logos-tiny-transformer-dinov3-projected-128.toml')
    assert resolved['training']['initialization'] == 'deterministic_orthogonal'
    assert resolved['components']['generator']['args']['source'] == (
        root / 'examples/networks/tiny-transformer-generator-128.hndl').read_text()
    assert resolved['components']['discriminator']['args']['source'] == (
        root / 'examples/networks/dinov3-projected-discriminator-128.hndl').read_text()
    assert resolved['components']['discriminator']['args']['output_shape'] == ['B', 1]

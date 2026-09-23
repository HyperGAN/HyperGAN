"""Projected DINOv3 critic: one frozen backbone call and four spatial heads."""
from pathlib import Path

import pytest
import torch
from torch import nn

from hypergan.hndl_networks import HNDLNetwork, build_network
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
    """Four native patch maps. Only the external model API is mocked."""
    def __init__(self, side):
        super().__init__()
        self.network = build_network('avg_pool(16)\nconv(384, kernel_size=1)\nbatch_norm()\ntanh()',
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


def _outside(model, backbone):
    hidden = {id(module) for module in backbone.modules()}
    return [module for module in model.network.modules() if id(module) not in hidden]


@pytest.mark.parametrize('batch_size', [1, 3])
def test_projected_dino128_spatial_heads_and_frozen_projection(monkeypatch, tmp_path, batch_size):
    providers, parameters = assets(monkeypatch, tmp_path)
    source = (Path(__file__).parents[2] /
              'examples/networks/dinov3-projected-discriminator-128-stable.hndl').read_text()
    assert source.count('kaiming_uniform(a=2.23606797749979)') == 4
    model = HNDLNetwork(source, input_shape=('B', 3, 128, 128), output_shape=('B', 4, 4, 4),
                        parameters=parameters, pretrained_providers=providers)
    backbone_module = model.network['backbone']
    backbone = backbone_module.model
    for mode in (False, True):
        model.train(mode)
        assert all(not module.training for module in backbone.modules())
    model.train(True)
    outside = _outside(model, backbone_module)
    ccms = [module for module in outside
            if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)
            and (module.in_channels, module.out_channels) == (384, 384) and module.bias is None
            and not hasattr(module, 'parametrizations')]
    discs = [module for module in outside
             if isinstance(module, nn.Conv2d) and hasattr(module, 'parametrizations')]
    assert len(ccms) == 4 and len(discs) == 8
    assert not any(isinstance(module, (nn.BatchNorm2d, nn.Linear)) for module in outside)
    downs = [module for module in discs if module.out_channels == 128 and module.stride == (2, 2)
             and module.kernel_size == (3, 3) and module.padding == (1, 1) and module.bias is None]
    logits = [module for module in discs if module.out_channels == 1 and module.stride == (1, 1)
              and module.kernel_size == (3, 3) and module.padding == (1, 1) and module.bias is None]
    assert len(downs) == 4 and len(logits) == 4
    assert all(not parameter.requires_grad for module in ccms for parameter in module.parameters())
    assert all(parameter.requires_grad for module in discs for parameter in module.parameters())
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    disc_weights = [module.parametrizations.weight.original for module in discs]
    ccm_weights = [module.weight for module in ccms]
    assert len({id(weight) for weight in disc_weights + ccm_weights}) == 12
    captured = {}

    def capture(_module, inputs, _output):
        captured['normalized'] = inputs[0].detach()

    def capture_augmented(_module, _inputs, output):
        captured['augmented'] = output.detach().clone()

    handle = backbone_module.register_forward_hook(capture)
    aug_handle = model.network['diffaug'].register_forward_hook(capture_augmented)
    x = torch.linspace(-1, 1, batch_size * 3 * 128 * 128).reshape(batch_size, 3, 128, 128)
    x.requires_grad_()
    try:
        score = model(x=x)
    finally:
        handle.remove()
        aug_handle.remove()
    assert score.shape == (batch_size, 4, 4, 4) and torch.isfinite(score).all()
    assert score.std(dim=1, unbiased=False).sum() > 0
    assert backbone.calls == [((batch_size, 3, 128, 128), (2, 5, 8, 11), True, True)]
    mean = x.new_tensor((.485, .456, .406)).reshape(1, 3, 1, 1)
    std = x.new_tensor((.229, .224, .225)).reshape(1, 3, 1, 1)
    torch.testing.assert_close(captured['normalized'], (captured['augmented'] * .5 + .5 - mean) / std, rtol=2e-6, atol=2e-7)
    assert not torch.equal(captured['augmented'], x.detach())
    # The endpoint penalty differentiates through augmentation and frozen DINO.
    first, = torch.autograd.grad(score.mean(), x, create_graph=True, retain_graph=True)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    (score.mean() + first.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for weight in disc_weights:
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    for weight in ccm_weights:
        assert weight.grad is None
    for parameter in backbone.parameters():
        assert parameter.grad is None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in trainable)
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    generated = x.detach().clone().requires_grad_()
    model(x=generated).mean().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert model.network['diffaug'].training
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        first, second = model(x=x.detach()), model(x=x.detach())
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(first, second, rtol=0, atol=0)

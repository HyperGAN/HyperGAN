"""A shared critic head must retain gradients through both feature sources."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hypergan import colorization_components as color


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _Backbone(nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 16, stride=16)
        self.project = nn.Linear(8, 384)
        self.constant = constant
        self.calls = 0

    def forward_features(self, x):
        self.calls += 1
        if self.constant:
            return {'x_norm_patchtokens': x.new_zeros(len(x), 256, 384)}
        h = self.patch(x).flatten(2).transpose(1, 2)[:, None]
        h = F.scaled_dot_product_attention(h, h, h)[:, 0]
        return {'x_norm_patchtokens': self.project(h)}


def _model(monkeypatch, pixel_width=2, constant=False):
    monkeypatch.setattr(color, '_load_dinov3', lambda *args: _Backbone(constant))
    return color.DINOv3ProjectedDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                               feature_width=4, head='conv', pixel_width=pixel_width)


def test_pixel_features_supply_image_gradient_when_dino_is_constant(monkeypatch):
    torch.manual_seed(971)
    model = _model(monkeypatch, constant=True).train()
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    calls = []
    hook = model.attention.register_forward_pre_hook(lambda module, inputs: calls.append(tuple(inputs[0].shape)))
    logits = model(x)
    hook.remove()
    assert logits.shape == (2, 1)
    assert model.backbone.calls == 1
    assert calls == [(2, 8, 16, 16)]
    gradient, = torch.autograd.grad(logits.sum(), x, create_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # Real/fake forwards share spectral-norm buffers, followed by b-cap-like
    # second derivatives through the learned RGB stem and shared critic.
    other_logits = model(torch.randn_like(x))
    (F.softplus(other_logits - logits).mean() + gradient.square().sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pixel_features.parameters())
    assert model.pixel_features[0].weight.grad.abs().sum() > 0
    assert model.feature_output[0].weight_orig.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.feature_project.parameters())


def test_dino_image_gradient_remains_when_pixel_features_are_disabled(monkeypatch):
    model = _model(monkeypatch, pixel_width=0)
    assert not hasattr(model, 'pixel_features')
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    gradient, = torch.autograd.grad(model(x).sum(), x, create_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    gradient.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.attention.query.weight.grad.abs().sum() > 0


def test_pixel_features_freeze_masks_and_checkpoint_reload(monkeypatch):
    model = _model(monkeypatch)
    model.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())
    model.requires_grad_(True).train()
    assert not model.backbone.training and not model.feature_project.training
    assert model.pixel_features.training and model.attention.training and model.feature_output.training
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()
              if name.startswith(('backbone.', 'feature_project.'))}
    assert all(p.requires_grad == (name not in frozen) for name, p in model.named_parameters())
    x = torch.randn(2, 3, 256, 256)
    before_pixel = model.pixel_features[0].weight.detach().clone()
    model(x).square().mean().backward()
    torch.optim.SGD(model.parameters(), lr=.1).step()
    assert not torch.equal(before_pixel, model.pixel_features[0].weight)
    for name, p in model.named_parameters():
        if name in frozen:
            assert p.grad is None
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    model.eval()
    restored = _model(monkeypatch).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize('pixel_width', [-1, True, 1.5, None])
def test_pixel_width_validation_precedes_backbone_loading(pixel_width):
    with pytest.raises(ValueError, match='pixel_width must be a nonnegative integer'):
        color.DINOv3ProjectedDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                           head='conv', pixel_width=pixel_width)


def test_pixel_features_require_convolutional_head():
    with pytest.raises(ValueError, match="requires head='conv'"):
        color.DINOv3ProjectedDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                           pixel_width=2)

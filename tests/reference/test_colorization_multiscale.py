"""Small CPU fixtures for the multidepth projected discriminator contract."""
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


class _IntermediateBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 16, stride=16)
        self.project = nn.Linear(8, 384)
        self.calls = []

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        self.calls.append((n, reshape, norm))
        h = self.patch(x).flatten(2).transpose(1, 2)[:, None]
        h = F.scaled_dot_product_attention(h, h, h)[:, 0]
        h = self.project(h).transpose(1, 2).reshape(len(x), 384, 16, 16)
        return tuple(h * (index + 1) / 4 for index in range(4))


def _model(monkeypatch):
    monkeypatch.setattr(color, '_load_dinov3', lambda *args: _IntermediateBackbone())
    return color.DINOv3MultiScaleDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                               feature_width=2)


def test_multiscale_single_backbone_pass_and_each_head_contributes(monkeypatch):
    model = _model(monkeypatch).eval()
    values, sizes, attention_sizes = [], [], []

    def capture_head(module, inputs, output):
        sizes.append(inputs[0].shape[-1])
        values.append(output)

    hooks = [head.register_forward_hook(capture_head) for head in model.heads]
    hooks += [module.register_forward_pre_hook(lambda module, inputs: attention_sizes.append(inputs[0].shape[-1]))
              for module in model.modules() if isinstance(module, color.SAGANAttention)]
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    for hook in hooks:
        hook.remove()
    assert model.backbone.calls == [((2, 5, 8, 11), True, True)]
    assert sizes == [32, 16, 8, 4]
    assert attention_sizes == [16, 16, 8, 4]
    assert logits.shape == (2, 1)
    torch.testing.assert_close(logits, torch.stack(values).mean(0))
    # Every branch, rather than just the deepest one, receives learning signal.
    logits.sum().backward()
    for head in model.heads:
        assert head.layers[-1].weight_orig.grad.abs().sum() > 0


def test_multiscale_frozen_projection_and_image_double_backward_reload(monkeypatch):
    torch.manual_seed(853)
    model = _model(monkeypatch)
    model.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    model.requires_grad_(True).train()
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if name.startswith(('backbone.', 'feature_project.'))}
    assert not model.backbone.training and not model.feature_project.training
    assert model.heads.training
    assert all(parameter.requires_grad == (name not in frozen)
               for name, parameter in model.named_parameters())
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    logits = model(x)
    gradient, = torch.autograd.grad(logits.sum(), x, create_graph=True)
    other = model(torch.randn_like(x))
    (F.softplus(other - logits).mean() + gradient.square().sum()).backward()
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.heads.parameters())
    before = model.heads[0].layers[-1].weight_orig.detach().clone()
    torch.optim.SGD(model.parameters(), lr=.1).step()
    assert not torch.equal(before, model.heads[0].layers[-1].weight_orig)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    model.eval()
    restored = _model(monkeypatch).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x.detach()), model(x.detach()), rtol=0, atol=0)


def test_multiscale_rejects_bad_input_and_backbone_maps(monkeypatch):
    model = _model(monkeypatch)
    x = torch.randn(1, 3, 256, 256)
    with pytest.raises(ValueError, match='requires x'):
        model(x[:, :, :128])
    with pytest.raises(TypeError):
        model(x, gray=x[:, :1])
    monkeypatch.setattr(model.backbone, 'get_intermediate_layers', lambda *args, **kwargs: ())
    with pytest.raises(ValueError, match='four .* intermediate maps'):
        model(x)
    monkeypatch.setattr(model.backbone, 'get_intermediate_layers',
                        lambda *args, **kwargs: [torch.zeros(1, 384, 8, 8)] * 4)
    with pytest.raises(ValueError, match='four .* intermediate maps'):
        model(x)


@pytest.mark.parametrize('feature_width', [0, -1, True, 2.5])
def test_multiscale_rejects_invalid_width_before_loading_weights(feature_width):
    with pytest.raises(ValueError, match='feature_width must be a positive integer'):
        color.DINOv3MultiScaleDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                            feature_width=feature_width)

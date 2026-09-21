"""Small CPU fixtures for the multidepth projected discriminator contract."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hypergan import colorization_components as color
from tests.hndl_fixtures import fixture_network
from tests.dinov3_fixtures import dinov3_assets, backbone_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _IntermediateBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('colorization_backbone', (3, 256, 256), (256, 384))
        self.calls = []

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        self.calls.append((n, reshape, norm))
        h = self.network(x).transpose(1, 2).reshape(len(x), 384, 16, 16)
        return tuple(h * (index + 1) / 4 for index in range(4))


def _model(monkeypatch, tmp_path):
    args = dinov3_assets(monkeypatch, tmp_path, _IntermediateBackbone)
    return color.DINOv3MultiScaleDiscriminator(*args, feature_width=2)


def test_multiscale_single_backbone_pass_and_each_head_contributes(monkeypatch, tmp_path):
    model = _model(monkeypatch, tmp_path).eval()
    values, sizes, attention_sizes = [], [], []

    def capture_head(module, inputs, output):
        sizes.append(inputs[0].shape[-1])
        values.append(output)

    hooks = [head.register_forward_hook(capture_head) for head in model.heads]
    hooks += [module.register_forward_pre_hook(lambda module, inputs: attention_sizes.append(inputs[0].shape[-1]))
              for head in model.heads for node in head.layers.plan.nodes
              if node.id == 'attention_query' for module in [head.layers[node.id]]]
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    for hook in hooks:
        hook.remove()
    assert backbone_model(model).calls == [((2, 5, 8, 11), True, True)]
    assert sizes == [32, 16, 8, 4]
    assert attention_sizes == [16, 16, 8, 4]
    assert logits.shape == (2, 1)
    torch.testing.assert_close(logits, torch.stack(values).mean(0))
    # Every branch, rather than just the deepest one, receives learning signal.
    logits.sum().backward()
    for head in model.heads:
        assert head.layers['output'].parametrizations.weight.original.grad.abs().sum() > 0


def test_multiscale_frozen_projection_and_image_double_backward_reload(monkeypatch, tmp_path):
    torch.manual_seed(853)
    model = _model(monkeypatch, tmp_path)
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    model.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    for parameter, flag in flags:
        parameter.requires_grad_(flag)
    model.train()
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if name.startswith(('backbone.', 'feature_project.'))}
    assert not backbone_model(model).training
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
    before = model.heads[0].layers['output'].parametrizations.weight.original.detach().clone()
    torch.optim.SGD(model.parameters(), lr=.1).step()
    assert not torch.equal(before, model.heads[0].layers['output'].parametrizations.weight.original)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    model.eval()
    restored = _model(monkeypatch, tmp_path).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x.detach()), model(x.detach()), rtol=0, atol=0)


def test_multiscale_rejects_bad_input_and_backbone_maps(monkeypatch, tmp_path):
    model = _model(monkeypatch, tmp_path)
    x = torch.randn(1, 3, 256, 256)
    with pytest.raises(ValueError, match='requires x'):
        model(x[:, :, :128])
    with pytest.raises(TypeError):
        model(x, gray=x[:, :1])
    monkeypatch.setattr(backbone_model(model), 'get_intermediate_layers', lambda *args, **kwargs: ())
    with pytest.raises(ValueError, match='four .* intermediate maps'):
        model(x)
    monkeypatch.setattr(backbone_model(model), 'get_intermediate_layers',
                        lambda *args, **kwargs: [torch.zeros(1, 384, 8, 8)] * 4)
    with pytest.raises(ValueError, match='four .* intermediate maps'):
        model(x)


@pytest.mark.parametrize('feature_width', [0, -1, True, 2.5])
def test_multiscale_rejects_invalid_width_before_loading_weights(feature_width):
    with pytest.raises(ValueError, match='feature_width must be a positive integer'):
        color.DINOv3MultiScaleDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                            feature_width=feature_width)

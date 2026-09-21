"""CPU resolution contracts with synthetic, locally pinned ResNet-shaped weights."""
import copy
import hashlib
import sys
from types import ModuleType

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from particlegan import GradientPenalty

from hypergan import image_components as image


class _SyntheticResNet(nn.Module):
    """Keep native stage resolutions/channels and BN without pretrained assets."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self.stage(64, 64, 1)
        self.layer2 = self.stage(64, 128, 2)
        self.layer3 = self.stage(128, 256, 2)

    @staticmethod
    def stage(cin, cout, stride):
        return nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def pinned_backbone(tmp_path, monkeypatch):
    # No torchvision dependency or download is needed in the fast CPU suite.
    models = ModuleType('torchvision.models')
    models.resnet18 = lambda *, weights: _SyntheticResNet()
    torchvision = ModuleType('torchvision')
    torchvision.models = models
    monkeypatch.setitem(sys.modules, 'torchvision', torchvision)
    monkeypatch.setitem(sys.modules, 'torchvision.models', models)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(583)
        network = _SyntheticResNet()
    path = tmp_path / 'synthetic-resnet.pth'
    torch.save(network.state_dict(), path)
    features = nn.ModuleList([
        nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool, network.layer1),
        network.layer2, network.layer3])
    digest = hashlib.sha256()
    for name, value in features.state_dict().items():
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return {'weights_path': str(path), 'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'feature_state_sha256': digest.hexdigest(), 'width': 1}


def test_256_native_maps_attention_and_frozen_backbone_double_backward(pinned_backbone):
    torch.manual_seed(584)
    model = image.CIFARDiscriminator(**pinned_backbone, image_size=256, feature_size=256)
    model.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    model.requires_grad_(True).train()
    assert all(not module.training for module in model.critic.features.modules())
    assert all(not parameter.requires_grad for parameter in model.critic.features.parameters())
    assert all(parameter.requires_grad for parameter in model.critic.pixel.parameters())
    assert all(parameter.requires_grad for parameter in model.critic.project.parameters())
    frozen = copy.deepcopy(model.critic.features.state_dict())
    assert len(model.critic.pixel.blocks) == 6
    observed = []
    hook = model.critic.pixel.attention.register_forward_pre_hook(
        lambda module, inputs: observed.append(tuple(inputs[0].shape)))
    real, fake = torch.randn(1, 3, 256, 256).tanh(), torch.randn(1, 3, 256, 256).tanh()
    real_logits, fake_logits = model(real), model(fake)
    hook.remove()
    assert real_logits.shape == fake_logits.shape == (1,)
    assert observed == [(1, 4, 16, 16), (1, 4, 16, 16)]
    assert [tuple(value.shape) for value in model._context_features] == [
        (1, 64, 64, 64), (1, 128, 32, 32), (1, 256, 16, 16)]
    # Zero cap makes the penalty active even for a weak synthetic critic.
    penalty = GradientPenalty(arm='b_cap', kappa=0)(model, real, fake)
    (F.softplus(fake_logits - real_logits).mean() + penalty).backward()
    for module in (model.critic.pixel, model.critic.project):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in module.parameters())
    assert model.critic.project[0][0].weight.grad.abs().sum() > 0
    assert model.critic.pixel.input.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.critic.features.parameters())
    for name, value in model.critic.features.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


class _ZeroPixel(nn.Module):
    def forward(self, x, context):
        return x.new_zeros(len(x), 1)


def test_candidate_gradient_survives_through_frozen_feature_path(pinned_backbone):
    model = image.CIFARDiscriminator(**pinned_backbone, image_size=256, feature_size=256)
    model.critic.pixel = _ZeroPixel()
    model.requires_grad_(False).eval()
    candidate = torch.randn(1, 3, 256, 256, requires_grad=True)
    gradient, = torch.autograd.grad(model(candidate).sum(), candidate, create_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    second, = torch.autograd.grad(gradient.square().sum(), candidate)
    assert torch.isfinite(second).all()
    assert all(not value.requires_grad for value in model._context_features)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_strict_reload_and_dtype_move_invalidate_derived_context(pinned_backbone):
    model = image.CIFARDiscriminator(**pinned_backbone, image_size=256, feature_size=256).eval()
    candidate = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        expected = model(candidate)
        cache = model._context_features
        model(candidate)
        assert model._context_features is cache
        state = copy.deepcopy(model.state_dict())
        assert not any('_context_features' in name for name in state)
        model.critic.features[0][1].running_mean.add_(1)
        model.load_state_dict(state, strict=True)
        assert model._context_features is None
        torch.testing.assert_close(model(candidate), expected, rtol=0, atol=0)
        model.double()
        assert model._context_features is None
        model(candidate.double())
        assert all(value.dtype == torch.float64 for value in model._context_features)
    missing_context = dict(state)
    del missing_context['context']
    with pytest.raises(RuntimeError, match='Missing key'):
        model.load_state_dict(missing_context, strict=True)


class _LegacyPixel(nn.Module):
    """Previous 32px layer order and attention placement, as a parity oracle."""
    def __init__(self, width, image_size=32):
        super().__init__()
        assert image_size == 32
        self.emb_dim = 4 * width
        self.input = nn.Conv2d(6, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            image._ResBlock(a, b, self.emb_dim, affine_condition=False)
            for a, b in [(width, 2 * width), (2 * width, 4 * width), (4 * width, 4 * width)]])
        self.output = nn.Linear(width * 4 * 4 * 4, 1)

    def forward(self, x, context):
        embedding = x.new_zeros(len(x), self.emb_dim)
        h = self.input(torch.cat((x, context), dim=1))
        for index, block in enumerate(self.blocks):
            h = F.avg_pool2d(block(h, embedding), 2)
            if index == 0:
                h = self.attention(h)
        return self.output(F.leaky_relu(h, .2).flatten(1))


def test_default_cifar_matches_legacy_pixel_layout_values_and_rng(pinned_backbone, monkeypatch):
    before = torch.get_rng_state()
    actual = image.CIFARDiscriminator(**pinned_backbone).eval()
    after = torch.get_rng_state()
    monkeypatch.setattr(image, '_PixelDiscriminator', _LegacyPixel)
    torch.set_rng_state(before)
    expected = image.CIFARDiscriminator(**pinned_backbone).eval()
    assert torch.equal(torch.get_rng_state(), after)
    assert actual.image_size == 32 and actual.critic.feature_size == 64
    assert list(actual.state_dict()) == list(expected.state_dict())
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    candidate = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(actual(candidate), expected(candidate), rtol=0, atol=0)


@pytest.mark.parametrize('option,value', [('image_size', 16), ('image_size', 48),
                                         ('image_size', True), ('feature_size', 32),
                                         ('feature_size', 96), ('feature_size', 64.)])
def test_bad_resolution_fails_before_weight_loading(option, value):
    with pytest.raises(ValueError, match=option + ' must be a power of two'):
        image.CIFARDiscriminator('missing', **{option: value})


def test_wrong_image_shape_is_rejected(pinned_backbone):
    model = image.CIFARDiscriminator(**pinned_backbone, image_size=256, feature_size=256)
    for shape in ((1, 3, 32, 32), (1, 1, 256, 256), (3, 256, 256)):
        with pytest.raises(ValueError, match=r'\[batch,3,256,256\]'):
            model(torch.zeros(shape))

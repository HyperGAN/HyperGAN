"""Small CPU derivative and posterior fixtures; no pretrained downloads or GPU."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hypergan import colorization_components as color
from hypergan.config import tomllib
from hypergan.recipes import ComponentGraph


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _encoding():
    encoder = color.GrayscaleRoutingEncoder(z_dim=8, width=2)
    gray = torch.randn(2, 1, 256, 256)
    means = torch.randn(16, 8, requires_grad=True)
    sigma = torch.tensor(.2)
    return encoder, gray, means, sigma


def test_hard_posterior_exact_matching_sigma_and_rng_restore():
    encoder, gray, means, sigma = _encoding()
    state = torch.get_rng_state()
    expected_noise = torch.randn(2, 8)
    torch.set_rng_state(state)
    result = encoder(gray, means, sigma)
    torch.testing.assert_close(result['latent'], means[result['ids']] + sigma * expected_noise,
                               rtol=0, atol=0)
    torch.testing.assert_close(result['reconstruction_latent'], result['latent'], rtol=0, atol=0)
    assert result['ids'].shape == (2,)
    assert result['soft'].shape == (2, 16)
    torch.set_rng_state(state)
    repeated = encoder(gray, means, sigma)
    torch.testing.assert_close(result['latent'], repeated['latent'], rtol=0, atol=0)
    changed = encoder(gray, means, sigma)
    assert torch.equal(changed['ids'], result['ids'])
    assert not torch.equal(changed['latent'], result['latent'])


def test_reconstruction_gradients_update_only_encoder_through_frozen_decoder():
    encoder, gray, means, sigma = _encoding()
    generator = color.ColorizationGenerator(z_dim=8, width=2).requires_grad_(False)
    result = encoder(gray, means, sigma)
    output = generator(result['reconstruction_latent'])
    assert output.shape == (2, 3, 256, 256)
    assert output.min() >= -1 and output.max() <= 1
    output.square().mean().backward()
    assert means.grad is None
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert encoder.query.weight.grad is not None
    assert encoder.query.weight.grad.abs().sum() > 0


def test_grayscale_reconstruction_matches_data_luminance_and_preserves_gradient():
    rgb = torch.tensor([[[[1.]], [[0.]], [[-1.]]]], requires_grad=True)
    gray = color.GrayscaleImage()(rgb)
    torch.testing.assert_close(gray, torch.tensor([[[[.185]]]]))
    gray.sum().backward()
    torch.testing.assert_close(rgb.grad.flatten(), torch.tensor([.299, .587, .114]))
    with pytest.raises(ValueError, match='requires RGB'):
        color.GrayscaleImage()(rgb[:, :1])


def test_random_gan_and_conditional_reconstruction_have_separate_gradient_owners():
    """Exercise the actual recipe bindings without pretrained assets or a GPU."""
    torch.manual_seed(97)
    path = Path(__file__).resolve().parents[2] / 'examples/logos-colorization-256.toml'
    recipe = tomllib.loads(path.read_text())
    specs = recipe['components']
    specs['discriminator'] = {'factory': 'identity', 'inputs': {'input': 'candidate'}}
    for name in ('generator', 'encoder'):
        specs[name]['args'].update(z_dim=8, width=2)
    for spec in specs.values():
        spec.setdefault('trainable', True)
    graph = ComponentGraph(specs)
    means = torch.randn(16, 8, requires_grad=True)
    prior = SimpleNamespace(means=lambda: means, sigma=torch.tensor(.2))
    rgb = torch.randn(2, 3, 256, 256).tanh()
    batch = {'real': rgb, 'gray': color.GrayscaleImage()(rgb)}
    latent = means[torch.tensor([1, 12])] + .2 * torch.randn(2, 8)
    context = graph.generate(latent, batch, prior=prior)
    assert set(context['components']) == {'generator'}
    context['generated'].square().mean().backward()
    assert means.grad.abs().sum() > 0
    assert graph.models['generator'].output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in graph.models['encoder'].parameters())
    graph.zero_grad(set_to_none=True)
    means.grad = None
    objective = recipe['objectives'][0]
    reconstruction = graph.resolve(objective['inputs']['input'], context)
    target = graph.resolve(objective['inputs']['target'], context).detach()
    F.mse_loss(reconstruction, target).backward()
    assert means.grad is None
    assert all(p.grad is None for p in graph.models['generator'].parameters())
    assert graph.models['encoder'].query.weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in graph.models['generator'].parameters())


def test_adversarial_latent_updates_selected_means_and_encoder():
    encoder, gray, means, sigma = _encoding()
    result = encoder(gray, means, sigma)
    result['latent'].square().mean().backward()
    assert means.grad.abs().sum() > 0
    unselected = torch.ones(len(means), dtype=torch.bool)
    unselected[result['ids']] = False
    assert torch.equal(means.grad[unselected], torch.zeros_like(means.grad[unselected]))
    assert encoder.query.weight.grad.abs().sum() > 0


def test_explicit_detached_prior_option():
    encoder, gray, means, sigma = _encoding()
    encoder.detach_means = True
    encoder(gray, means, sigma)['latent'].sum().backward()
    assert means.grad is None
    assert encoder.query.weight.grad.abs().sum() > 0


class _TinyBackbone(nn.Module):
    """Actual differentiable attention, with the upstream patch-token contract."""
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 16, stride=16)
        self.project = nn.Linear(8, 384)

    def forward_features(self, x):
        h = self.patch(x).flatten(2).transpose(1, 2)[:, None]
        h = F.scaled_dot_product_attention(h, h, h)[:, 0]
        return {'x_norm_patchtokens': self.project(h)}


def test_discriminator_keeps_features_frozen_but_allows_image_double_backward(monkeypatch):
    monkeypatch.setattr(color, '_load_dinov3', lambda *args: _TinyBackbone())
    model = color.DINOv3Discriminator('unused', '0' * 40, 'unused', '0' * 64,
                                      width=2, feature_width=4)
    model.requires_grad_(False).requires_grad_(True).train()
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    gray = torch.randn(1, 1, 256, 256)
    logits = model(x, gray)
    assert logits.shape == (1, 1)
    gradient, = torch.autograd.grad(logits.sum(), x, create_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    gradient.square().sum().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.attention.query.weight.grad.abs().sum() > 0
    assert model.pixel.output.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    with torch.no_grad():
        assert not torch.equal(model(x.detach(), gray), model(x.detach(), gray + .2))


def _projected_discriminator(monkeypatch):
    monkeypatch.setattr(color, '_load_dinov3', lambda *args: _TinyBackbone())
    return color.DINOv3ProjectedDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                               feature_width=4)


def test_projected_discriminator_single_path_frozen_masks_and_double_backward(monkeypatch):
    model = _projected_discriminator(monkeypatch)
    model.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    model.requires_grad_(True).train()
    assert not model.backbone.training and not model.feature_project.training
    assert model.attention.training and model.feature_output.training
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if name.startswith(('backbone.', 'feature_project.'))}
    assert all(parameter.requires_grad == (name not in frozen)
               for name, parameter in model.named_parameters())
    calls = []
    hook = model.backbone.patch.register_forward_hook(lambda *_: calls.append(True))
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    logits = model(x)
    hook.remove()
    assert calls == [True]
    assert logits.shape == (1, 1)
    assert not hasattr(model, 'pixel')
    gradient, = torch.autograd.grad(logits.sum(), x, create_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    gradient.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.attention.query.weight.grad.abs().sum() > 0
    assert model.feature_output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name in frozen)
    # Even an optimizer over all parameters cannot update the fixed projection.
    before_head = model.feature_output.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=.1).step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    assert not torch.equal(before_head, model.feature_output.weight)


def test_projected_discriminator_state_reload_preserves_random_projection(monkeypatch, tmp_path):
    model = _projected_discriminator(monkeypatch).eval()
    x = torch.randn(1, 3, 256, 256)
    expected = model(x).detach()
    path = tmp_path / 'projected.pt'
    torch.save(model.state_dict(), path)
    restored = _projected_discriminator(monkeypatch).eval()
    assert not torch.equal(model.feature_project[0].weight, restored.feature_project[0].weight)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in restored.feature_project.parameters())
    with pytest.raises(RuntimeError, match='Missing key|Unexpected key|size mismatch'):
        color.DINOv3Discriminator('unused', '0' * 40, 'unused', '0' * 64,
                                  width=2, feature_width=4).load_state_dict(model.state_dict())


def test_projected_discriminator_rejects_conditioning_and_bad_feature_contract(monkeypatch):
    model = _projected_discriminator(monkeypatch)
    x = torch.randn(1, 3, 256, 256)
    with pytest.raises(TypeError):
        model(x, gray=x[:, :1])
    with pytest.raises(ValueError, match='requires x'):
        model(x[:, :, :128])
    monkeypatch.setattr(model.backbone, 'forward_features',
                        lambda x: {'x_norm_patchtokens': torch.zeros(len(x), 64, 384)})
    with pytest.raises(ValueError, match='256 patch tokens'):
        model(x)


@pytest.mark.parametrize('feature_width', [0, -1, True, 4.5])
def test_projected_discriminator_rejects_invalid_feature_width(feature_width):
    with pytest.raises(ValueError, match='feature_width must be a positive integer'):
        color.DINOv3ProjectedDiscriminator('unused', '0' * 40, 'unused', '0' * 64,
                                           feature_width=feature_width)


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf'), True])
def test_bad_routing_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        color.GrayscaleRoutingEncoder(temperature=temperature)


def test_incompatible_input_and_posterior_rejected():
    encoder, gray, means, sigma = _encoding()
    with pytest.raises(ValueError, match='gray'):
        encoder(gray.expand(-1, 3, -1, -1), means, sigma)
    with pytest.raises(ValueError, match='means'):
        encoder(gray, means[:, :4], sigma)
    with pytest.raises(ValueError, match='scalar sigma'):
        encoder(gray, means, torch.ones(8))
    with pytest.raises(RuntimeError, match='positive finite'):
        encoder(gray, means, torch.tensor(0.))
    with pytest.raises(ValueError, match='z '):
        color.ColorizationGenerator(z_dim=8, width=2)(torch.randn(1, 4))


def test_pretrained_loader_requires_full_source_pin(tmp_path):
    with pytest.raises(ValueError, match='full lowercase Git SHA'):
        color._load_dinov3(tmp_path, 'latest', tmp_path / 'weights.pth', '0' * 64)


def test_generator_default_stays_under_ten_million_parameters():
    assert sum(parameter.numel() for parameter in color.ColorizationGenerator().parameters()) < 10_000_000


def test_pretrained_loader_rejects_changed_external_source(tmp_path):
    (tmp_path / 'dinov3' / 'hub').mkdir(parents=True)
    implementation = tmp_path / 'dinov3' / 'hub' / 'backbones.py'
    implementation.write_text('# pinned source\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    git('init')
    git('add', 'dinov3')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'Pinned source')
    commit = git('rev-parse', 'HEAD')
    with pytest.raises(ValueError, match='source must be clean'):
        color._load_dinov3(tmp_path, '0' * 40, 'missing.pth', '0' * 64)
    implementation.write_text('# modified source\n')
    with pytest.raises(ValueError, match='dirty=True'):
        color._load_dinov3(tmp_path, commit, 'missing.pth', '0' * 64)

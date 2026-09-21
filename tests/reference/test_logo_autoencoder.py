"""Fast AE recipe parity and gradient ownership; no pretrained assets or GPU."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from particlegan import get_recipe, particle_ae

from hypergan.autoencoder_components import ParticleAEEncoder256
from hypergan.config import tomllib
from hypergan.recipes import ComponentGraph


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_ae_matches_particle_gan_values_gradients_and_rng():
    torch.manual_seed(91)
    prior = get_recipe('ae_gan', num_particles=16, z_dim=8).make_prior()
    encoder = ParticleAEEncoder256(z_dim=8, width=2)
    # Nonzero offsets exercise the bounded local code, not just nearest centers.
    torch.nn.init.normal_(encoder.offset.weight, std=.2)
    x = torch.randn(2, 3, 256, 256)
    means = prior.means()
    rng = torch.get_rng_state()
    actual = encoder(x, means, prior.sigma)
    assert torch.equal(torch.get_rng_state(), rng)
    h = encoder.features(x)
    query = F.layer_norm(encoder.query(h), (8,))
    expected = particle_ae(query, encoder.offset(h), prior,
                           temperature=.125, distance_reduction='mean')
    torch.testing.assert_close(actual['latent'], expected.codes[:, 0], rtol=0, atol=0)
    assert torch.equal(actual['ids'], expected.indices[:, 0])
    assert actual['offset'].abs().max() <= 3
    parameters = [*encoder.parameters(), prior.z]
    probe = torch.randn_like(actual['latent'])
    first = torch.autograd.grad((actual['latent'] * probe).sum(), parameters)
    second = torch.autograd.grad((expected.codes[:, 0] * probe).sum(), parameters)
    for got, want in zip(first, second):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert first[-1].abs().sum() > 0
    assert encoder(x, prior.means(), prior.sigma)['latent'].equal(actual['latent'])


def test_rgb_reconstruction_updates_encoder_generator_and_prior():
    torch.manual_seed(92)
    path = Path(__file__).resolve().parents[2] / 'examples/logos-ae-gan-256.toml'
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
    # No grayscale key: all AE bindings must use RGB.
    batch = {'real': torch.randn(2, 3, 256, 256).tanh()}
    latent = means[[1, 12]] + .2 * torch.randn(2, 8)
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
    assert means.grad.abs().sum() > 0
    assert graph.models['generator'].output.weight.grad.abs().sum() > 0
    encoder = graph.models['encoder']
    assert encoder.query.weight.grad.abs().sum() > 0
    assert encoder.offset.weight.grad.abs().sum() > 0
    selected = context['components']['encoder']['ids']
    unselected = torch.ones(len(means), dtype=torch.bool)
    unselected[selected] = False
    assert torch.count_nonzero(means.grad[unselected]) == 0


def test_ae_initial_offset_zero_and_input_contract():
    encoder = ParticleAEEncoder256(z_dim=8, width=2)
    means, sigma = torch.randn(16, 8), torch.tensor(.2)
    result = encoder(torch.randn(1, 3, 256, 256), means, sigma)
    assert torch.equal(result['latent'], means[result['ids']])
    assert not torch.count_nonzero(result['offset'])
    with pytest.raises(ValueError, match='requires RGB'):
        encoder(torch.randn(1, 1, 256, 256), means, sigma)
    with pytest.raises(ValueError, match='means'):
        encoder(torch.randn(1, 3, 256, 256), means[:, :4], sigma)
    with pytest.raises(ValueError, match='scalar'):
        encoder(torch.randn(1, 3, 256, 256), means, sigma.expand(2))

"""Color-word mechanism gate. The 200-step run is heavy and is not a second seed."""
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest
import torch
from torch import nn

from hypergan.checkpoints import data_contract
from hypergan.color_words import (CANONICAL, LABELS, ColorGenerator, ColorWords, ImageCritic,
                                  ImageEncoder, JointCritic, TextCritic, TextEncoder, ZEmbedding,
                                  color_word_metrics, route_particles)
from hypergan.config import load_config, resolve_config
from hypergan.execution_profiles import validate_replicated_recipe
from hypergan.recipes import make_prior
from hypergan.training import ReferenceTrainer


RECIPE = Path(__file__).resolve().parents[2] / 'examples' / 'color-words.toml'
SEED = 25021


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _recipe_raw():
    with RECIPE.open('rb') as source:
        return tomllib.load(source)


def _short_config():
    """Mechanism gate: toml seed, one or two steps, batch 8. Not a second seed."""
    config = load_config(RECIPE)
    config['training']['steps'] = 2
    config['training']['batch_size'] = 8
    assert config['training']['seed'] == SEED
    return config


def _clone(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, after):
    return any(not torch.equal(saved, current) for saved, current in zip(before, after))


def _nonzero(grads):
    return any(grad is not None and grad.abs().sum() > 0 for grad in grads)


def _grads(module):
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in module.parameters()]


def test_color_words_shapes_labels_and_resume_contract():
    data = ColorWords()
    assert data.resume_stateless is True
    generator = torch.Generator().manual_seed(SEED)
    batch = data(8, generator=generator)
    assert batch['real'].shape == (8, 3, 16, 16) and batch['real'].dtype == torch.float32
    assert batch['label'].shape == (8, 1) and batch['label'].dtype == torch.int64
    assert torch.isfinite(batch['real']).all()
    assert batch['real'].min() >= -1 and batch['real'].max() <= 1
    exact = ColorWords(noise=0)
    clean = exact(8, generator=torch.Generator().manual_seed(1))
    expected = CANONICAL[clean['label'].view(-1)].view(8, 3, 1, 1).expand_as(clean['real'])
    assert torch.equal(clean['real'], expected)
    covered = ColorWords()(256, generator=torch.Generator().manual_seed(2))['label'].view(-1)
    assert set(covered.tolist()) == set(range(len(LABELS)))
    contract = data_contract(data, {'factory': 'hypergan.color_words:ColorWords', 'args': {'noise': 0.05}})
    assert contract['supported'] is True and contract['stateful'] is False


def test_router_straight_through_updates_means_and_query():
    means = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    query = torch.tensor([[0.45, 0.05], [0.05, 0.45]], requires_grad=True)
    latent, ids = route_particles(query, torch.zeros_like(query), means, torch.tensor(0.2), 0.125)
    assert ids.dtype == torch.int64 and latent.shape == query.shape
    latent.sum().backward()
    assert means.grad.abs().sum() > 0 and query.grad.abs().sum() > 0
    selected = torch.zeros(len(means), dtype=torch.bool)
    selected[ids] = True
    assert means.grad[selected].abs().sum() > 0
    assert torch.count_nonzero(means.grad[~selected]) == 0

    torch.manual_seed(3)
    encoder = ImageEncoder(z_dim=4, width=4)
    table = torch.randn(6, 4, requires_grad=True)
    image = torch.randn(2, 3, 16, 16)
    encoded = encoder(image, table, torch.tensor(0.05))
    assert torch.allclose(encoded['latent'], table[encoded['ids']])
    encoded['latent'].sum().backward()
    assert table.grad.abs().sum() > 0
    assert encoder.query.weight.grad.abs().sum() > 0


def test_module_shapes_and_prior_fixed_sigma():
    torch.manual_seed(4)
    means, sigma = torch.randn(8, 16), torch.tensor(0.05)
    image, label, latent = torch.randn(2, 3, 16, 16), torch.zeros(2, 1, dtype=torch.int64), torch.randn(2, 16)
    encoded = ImageEncoder()(image, means, sigma)
    text = TextEncoder()(label, means, sigma)
    assert encoded['latent'].shape == (2, 16) and encoded['ids'].shape == (2,) and encoded['ids'].dtype == torch.int64
    assert text['embedding'].shape == (2, 16) and text['latent'].shape == (2, 16) and text['ids'].dtype == torch.int64
    generated = ColorGenerator()(latent)
    assert generated.shape == (2, 3, 16, 16) and generated.min() >= -1 and generated.max() <= 1
    assert ZEmbedding()(latent).shape == (2, 16)
    for critic in (ImageCritic(), TextCritic(), JointCritic()):
        assert any(parameter.requires_grad for parameter in critic.parameters())
    assert ImageCritic()(generated).shape == (2, 1)
    assert TextCritic()(text['embedding']).shape == (2, 1)
    assert JointCritic()(generated, text['embedding']).shape == (2, 1)
    prior = make_prior({'kind': 'mog', 'initialization_device': 'cpu', 'initialization_seed': SEED,
                        'fixed_sigma': 0.05, 'args': {'num_particles': 64, 'z_dim': 16}},
                       device=torch.device('cpu'))
    assert torch.allclose(prior.sigma, torch.tensor(0.05))


def test_color_word_metrics_define_recovery_agreement_and_swap_gap():
    real = CANONICAL[0].view(1, 3, 1, 1).expand(2, 3, 16, 16).contiguous()
    label = torch.zeros(2, 1, dtype=torch.int64)
    text_image = real.clone()
    reconstruction = real + 0.25

    class _Image(nn.Module):
        def forward(self, image, means, sigma):
            return {'latent': torch.ones(len(image), 1), 'ids': torch.tensor([0, 0])}

    class _Text(nn.Module):
        def forward(self, labels, means, sigma):
            labels = labels.view(-1)
            return {'latent': torch.zeros(len(labels), 1),
                    'embedding': torch.nn.functional.one_hot(labels, 8).float(),
                    'ids': torch.tensor([0, 1])}

    class _Generator(nn.Module):
        def forward(self, latent):
            return text_image if torch.all(latent[:, 0] == 0) else reconstruction

    class _Joint(nn.Module):
        def forward(self, candidate, embedding):
            return embedding[:, :1]

    metrics = color_word_metrics(real, label, _Generator(), _Image(), _Text(), _Joint(),
                                 torch.zeros(4, 1), torch.tensor(0.05))
    assert metrics['label_recovery'] == pytest.approx(1.0)
    assert metrics['text_image_mae'] == pytest.approx(0.0)
    assert metrics['reconstruction_mae'] == pytest.approx(0.25)
    assert metrics['particle_agreement'] == pytest.approx(0.5)
    assert metrics['distinct_particles'] == 2
    # Label 0 scores 1; forced label 1 scores 0. Gap is true minus swapped.
    assert metrics['joint_swap_gap'] == pytest.approx(1.0)


def test_recipe_declares_shared_generator_and_three_terms():
    raw = _recipe_raw()
    assert raw['data']['factory'] == 'hypergan.color_words:ColorWords'
    assert raw['components']['generator']['factory'] == 'hypergan.color_words:ColorGenerator'
    assert raw['components']['reconstruction']['reuse'] == 'generator'
    assert raw['components']['text_image']['reuse'] == 'generator'
    assert raw['adversarial']['mode'] == 'vanilla' and raw['adversarial']['weight'] == 1.0
    assert raw['prior']['kind'] == 'mog' and raw['prior']['fixed_sigma'] == 0.05
    assert raw['training']['seed'] == SEED and raw['training']['phase_draws'] == 'shared'
    assert raw['optimizer']['implementation'] == 'device_adam'
    assert [term['id'] for term in raw['objectives']] == ['rgb-image', 'rgb-text', 'latent-text']
    assert raw['adversarial_terms'] == [
        {'id': 'image-from-text', 'component': 'discriminator', 'real': 'batch.real',
         'fake': 'components.text_image', 'penalty': False},
        {'id': 'text-marginal', 'component': 'text_critic', 'real': 'components.text_encoder.embedding',
         'fake': 'components.z_embedding', 'penalty': True, 'penalty_coeff': 0.1},
        {'id': 'joint', 'component': 'joint_critic', 'real': 'batch.real',
         'fake': 'components.text_image', 'penalty': True, 'penalty_coeff': 0.1},
    ]
    factories = ' '.join(spec.get('factory', '') for spec in raw['components'].values())
    assert 'clip' not in factories.casefold()


def test_legacy_resolve_config_allows_replicated_execution():
    validate_replicated_recipe(resolve_config({}))


def test_color_recipe_rejects_replicated_execution():
    config = load_config(RECIPE)
    with pytest.raises(ValueError):
        validate_replicated_recipe(config)


def test_phase_gradients_and_optimizer_ownership():
    trainer = ReferenceTrainer(_short_config())
    models = trainer.graph.models
    discriminator, text_critic, joint = models['discriminator'], models['text_critic'], models['joint_critic']
    generator, image_encoder = models['generator'], models['image_encoder']
    text_encoder, z_embedding = models['text_encoder'], models['z_embedding']
    critics = (discriminator, text_critic, joint)
    generators = (generator, image_encoder, text_encoder, z_embedding)
    disc_seen, gen_seen, joint_flags = [], [], []

    def wrap(module, sink, argument):
        original = module.forward

        def forward(*args, **kwargs):
            value = kwargs[argument] if argument in kwargs else args[0]
            sink.append(value.detach().clone())
            return original(*args, **kwargs)

        module.forward = forward

    wrap(discriminator, disc_seen, 'candidate')
    original_generator = generator.forward

    def generator_forward(*args, **kwargs):
        value = original_generator(*args, **kwargs)
        gen_seen.append(value.detach().clone())
        return value

    generator.forward = generator_forward
    original_joint = joint.forward

    def joint_forward(*args, **kwargs):
        embedding = kwargs['embedding'] if 'embedding' in kwargs else args[1]
        joint_flags.append(bool(embedding.requires_grad))
        return original_joint(*args, **kwargs)

    joint.forward = joint_forward
    before = {module: _clone(module) for module in critics + generators}
    captured = {}
    opt_d, opt_g = trainer.opt_d.step, trainer.opt_g.step

    def d_step(*args, **kwargs):
        captured['d_disc'] = list(disc_seen)
        captured['d_gen'] = list(gen_seen)
        captured['d_joint'] = list(joint_flags)
        captured['d_grads'] = {module: _grads(module) for module in critics + (z_embedding,)}
        opt_d(*args, **kwargs)
        captured['after_d'] = {module: _clone(module) for module in critics + generators}

    def g_step(*args, **kwargs):
        captured['g_grads'] = _grads(z_embedding)
        opt_g(*args, **kwargs)
        captured['after_g'] = {module: _clone(module) for module in critics + generators}

    trainer.opt_d.step = d_step
    trainer.opt_g.step = g_step
    trainer.update()

    # image-from-text scores G(E(text)) with the image critic during the critic phase.
    # The first generator call is the legacy G(z) sample.
    legacy = captured['d_gen'][0]
    later = [output for output in captured['d_gen'][1:] if not torch.allclose(output, legacy)]
    assert any(torch.allclose(candidate, legacy) for candidate in captured['d_disc'])
    assert later and any(torch.allclose(candidate, output) for candidate in captured['d_disc'] for output in later)
    assert _nonzero(captured['d_grads'][discriminator])
    assert _nonzero(captured['d_grads'][text_critic])
    assert _nonzero(captured['d_grads'][joint])
    assert not _nonzero(captured['d_grads'][z_embedding])
    # Legacy detach: text_encoder.embedding does not require grad on joint critic forwards.
    assert captured['d_joint'] and not any(captured['d_joint'])
    for module in critics:
        assert _changed(before[module], captured['after_d'][module])
    for module in generators:
        assert not _changed(before[module], captured['after_d'][module])
    assert _nonzero(captured['g_grads'])
    for module in generators:
        assert _changed(captured['after_d'][module], captured['after_g'][module])
    for module in critics:
        assert not _changed(captured['after_d'][module], captured['after_g'][module])


def test_latent_objective_weight_zero_still_runs():
    config = _short_config()
    term = next(item for item in config['objectives'] if item['id'] == 'latent-text')
    term['weight'] = 0.0
    trainer = ReferenceTrainer(config)
    index = next(i for i, item in enumerate(trainer.config['objectives']) if item['id'] == 'latent-text')
    calls = {'n': 0}
    objective = trainer.objectives[index]
    original = objective.forward

    def wrapped(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    objective.forward = wrapped
    trainer.update()
    assert calls['n'] >= 1


@pytest.mark.heavy
def test_color_words_two_hundred_steps_metrics_are_finite():
    config = load_config(RECIPE)
    assert config['training']['seed'] == SEED and config['training']['steps'] == 200
    trainer = ReferenceTrainer(config)
    batch = None
    for _ in range(200):
        _, batch = trainer.update()
    metrics = color_word_metrics(
        batch['real'], batch['label'], trainer.graph.models['generator'], trainer.graph.models['image_encoder'],
        trainer.graph.models['text_encoder'], trainer.graph.models['joint_critic'], trainer.prior.means(), trainer.prior.sigma)
    print('color-words-200', metrics)
    floats = ('label_recovery', 'text_image_mae', 'reconstruction_mae', 'particle_agreement', 'joint_swap_gap')
    assert all(type(metrics[name]) is float and math.isfinite(metrics[name]) for name in floats)
    assert type(metrics['distinct_particles']) is int and metrics['distinct_particles'] >= 1

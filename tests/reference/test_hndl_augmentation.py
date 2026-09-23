"""DiffAugment graph routing, derivative ownership and recipe provenance."""
import copy
import json

import pytest
import torch
from hndl import HNDLError, Registry, ResolvedPlan
from hndl.torch import build
from particlegan import GradientPenalty

from hypergan.config import config_values, fingerprint, load_config, resolve_config
from hypergan.diff_augment import DiffAugment
from hypergan.hndl_augmentation import register_augmentation
from hypergan.hndl_networks import HNDLNetwork, build_network
from hypergan.recipes import ComponentGraph


CRITIC = '''augmented = diff_augment(x, transforms="color,translation,cutout", name="diffaug")
features = conv(augmented, 4, kernel_size=3, padding=1)
features = tanh(features)
features = flatten(features)
out = linear(features, 1, name="score")'''
IMAGE_SHAPE = ['B', 3, 8, 8]


def _specs():
    return {
        'generator': {'factory': 'hndl', 'trainable': True, 'inputs': {'x': 'latent'},
                      'args': {'source': 'linear(192)\ntanh()\nreshape(3, 8, 8)',
                               'input_shape': ['B', 2], 'output_shape': IMAGE_SHAPE}},
        'discriminator': {'factory': 'hndl', 'trainable': True, 'inputs': {'x': 'candidate'},
                          'args': {'source': CRITIC, 'input_shape': IMAGE_SHAPE,
                                   'output_shape': ['B', 1]}},
    }


def test_registration_is_local_idempotent_and_construction_preserves_rng():
    registry, other = Registry.builtins(), Registry.builtins()
    assert register_augmentation(registry) is registry
    register_augmentation(registry)
    assert 'diff_augment' not in other.aliases
    rng = torch.get_rng_state().clone()
    model = build_network('diff_augment()', input_shape=IMAGE_SHAPE,
                          output_shape=IMAGE_SHAPE, registry=registry)
    assert torch.equal(rng, torch.get_rng_state())
    assert isinstance(model[0], DiffAugment)
    assert not list(model.parameters()) and not list(model.buffers())
    assert model(torch.ones(2, 3, 8, 8)).device.type == 'cpu'
    with pytest.raises(HNDLError, match='E_'):
        build_network('diff_augment()', input_shape=['B', 192], output_shape=['B', 192])


def test_named_adapter_eval_identity_dtype_and_copy_without_rng():
    model = HNDLNetwork('out = diff_augment(candidate, name="diffaug")',
                        {'candidate': IMAGE_SHAPE}, {'out': IMAGE_SHAPE}).double()
    pixels = torch.linspace(-1, 1, 2 * 3 * 8 * 8, dtype=torch.float64).reshape(2, 3, 8, 8)
    model.eval()
    rng = torch.get_rng_state().clone()
    clone = copy.deepcopy(model)
    assert torch.equal(model(candidate=pixels), pixels)
    assert torch.equal(clone(candidate=pixels), pixels)
    assert torch.equal(rng, torch.get_rng_state())
    model.train()
    assert model.network['diffaug'].training
    assert not clone.network['diffaug'].training
    output = model(candidate=pixels)
    assert output.shape == pixels.shape and output.dtype == pixels.dtype
    assert not torch.equal(output, pixels)


def test_critic_and_generator_routes_augment_with_input_and_penalty_gradients():
    graph = ComponentGraph(_specs())
    critic = graph.models['discriminator']
    augment = critic.network['diffaug']
    real = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8).requires_grad_()
    latent = torch.tensor([[.1, -.2], [.7, .5]], requires_grad=True)
    context = graph.generate(latent, {'real': real})
    fake = context['generated']
    calls = []
    handle = augment.register_forward_hook(
        lambda module, inputs, output: calls.append(
            (module.training, inputs[0].detach().clone(), output.detach().clone())))
    try:
        real_score = graph.critic(real, context)
        fake_score = graph.critic(fake.detach(), context)
        penalty = GradientPenalty(arm='b_cap', kappa=0)(
            lambda candidate: graph.critic(candidate, context), real, fake.detach())
        loss = real_score.square().mean() + fake_score.square().mean() + penalty
        loss.backward()
        assert torch.isfinite(loss) and penalty > 0
        assert real.grad is not None and torch.isfinite(real.grad).all() and real.grad.norm() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
        assert all(p.grad is None for p in graph.models['generator'].parameters())
        assert len(calls) >= 3 and all(training for training, _, _ in calls)
        assert all(not torch.equal(before, after) for _, before, after in calls)

        graph.zero_grad(set_to_none=True)
        critic.requires_grad_(False)
        generated = graph.generate(latent, {'real': real})
        generated['generated'].retain_grad()
        graph.critic(generated['generated'], generated).square().mean().backward()
        assert augment.training and calls[-1][0]
        assert not torch.equal(calls[-1][1], calls[-1][2])
        assert generated['generated'].grad.norm() > 0
        assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.norm() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in graph.models['generator'].parameters())
        assert all(p.grad is None for p in critic.parameters())
    finally:
        handle.remove()


def test_hndl_plan_and_config_source_survive_save_load(tmp_path):
    source_path = tmp_path / 'critic.hndl'
    source_path.write_text(CRITIC)
    config_path = tmp_path / 'config.toml'
    config_path.write_text('''[components.generator]
factory = "hndl"
[components.generator.inputs]
x = "latent"
[components.generator.args]
source = "linear(192)\\ntanh()\\nreshape(3, 8, 8)"
input_shape = ["B", 2]
output_shape = ["B", 3, 8, 8]
[components.discriminator]
factory = "hndl"
[components.discriminator.inputs]
x = "candidate"
[components.discriminator.args]
file = "critic.hndl"
input_shape = ["B", 3, 8, 8]
output_shape = ["B", 1]
''')
    config = load_config(config_path)
    assert config['components']['discriminator']['args']['source'] == CRITIC
    config_copy = resolve_config(json.loads(json.dumps(config_values(config))))
    assert fingerprint(config_copy) == fingerprint(config)
    changed = config_values(config)
    changed['components']['discriminator']['args']['source'] = CRITIC.replace(
        'color,translation,cutout', 'color')
    assert fingerprint(resolve_config(changed)) != fingerprint(config)
    model = HNDLNetwork(**config_copy['components']['discriminator']['args']).eval()
    plan_text = model.network.plan.to_json()
    assert 'hypergan.diff_augment@1' in plan_text
    assert 'color,translation,cutout' in plan_text
    registry = register_augmentation(Registry.builtins())
    plan = ResolvedPlan.from_json(plan_text, registry=registry)
    restored = build(plan, device='cpu', registry=registry).eval()
    state_path = tmp_path / 'critic.pt'
    torch.save(model.network.state_dict(), state_path)
    restored.load_state_dict(torch.load(state_path, weights_only=True), strict=True)
    pixels = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)
    restored_output, = restored(x=pixels).values()
    torch.testing.assert_close(restored_output, model(pixels), rtol=0, atol=0)

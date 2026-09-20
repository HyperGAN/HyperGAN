"""Native image phase contract, with a small caller-owned numerical oracle.

The oracle follows the D/G draw and update order in ParticleGAN
9e9ce96c:experiments/train_cifar_ae_sagan.py:858-900. It deliberately uses tiny
ordinary components and the installed public primitives, not private imports.
Image architecture/source parity is a separate CUDA qualification.
"""
import copy
import json

import pytest
import torch

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import config_values, resolve_config
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.training import DeviceAdam, ReferenceTrainer, update_ema


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)

    def forward(self, x, means, sigma):
        return self.linear(x) + sigma * means.detach()[0]


class PriorGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, z, means, sigma):
        return self.linear(z) + sigma * means[0, :2]


def config():
    return resolve_config({'components': {
        'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
        'discriminator': {'factory': 'mlp', 'args': {'input_dim': 2, 'output_dim': 1, 'hidden': [8]}, 'inputs': {'x': 'candidate'}},
        'encoder': {'factory': f'{__name__}:Encoder', 'inputs': {'x': 'batch.real', 'means': 'prior.means', 'sigma': 'prior.sigma'}},
        'reconstruction': {'reuse': 'generator', 'freeze_parameters': True, 'inputs': {'x': 'components.encoder'}}},
        'objectives': [{'factory': 'mse', 'inputs': {'input': 'components.reconstruction', 'target': 'batch.real'}}],
        'prior': {'kind': 'mog', 'args': {'num_particles': 16, 'z_dim': 4}, 'initialization_device': 'cpu', 'initialization_seed': 43, 'fixed_sigma': .2},
        'prior_regularizer': {'rows': 'full'},
        'optimizer': {'implementation': 'torch_fused_adam', 'prior_betas': [.5, .999]},
        'gradient_penalty': {'lazy_k': 2, 'kappa': .001},
        'training': {'steps': 4, 'batch_size': 8, 'phase_draws': 'independent', 'data_rng_device': 'execution', 'data_seed_offset': 2, 'prior_seed_offset': 3, 'lr_floor': 1.}})


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def close(left, right, *, exact=False):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0 if exact else 2e-6, atol=0 if exact else 2e-7)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            close(left[key], right[key], exact=exact)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            close(a, b, exact=exact)
    else:
        assert left == right


def test_independent_phases_match_direct_updates_and_encoder_only_gradient_routing():
    trainer = ReferenceTrainer(config())
    g, d, e = [copy.deepcopy(trainer.graph.models[name]) for name in ('generator', 'discriminator', 'encoder')]
    prior = copy.deepcopy(trainer.prior)
    ema_g, ema_e, ema_prior = [copy.deepcopy(module) for module in (g, e, prior)]
    opt = trainer.config['optimizer']
    og = DeviceAdam([{'params': [*g.parameters(), *e.parameters()], 'lr': opt['lr']},
                    {'params': prior.parameters(), 'lr': opt['lr'] * opt['prior_lr_mult'], 'betas': tuple(opt['prior_betas'])}],
                   betas=tuple(opt['betas']), fused=True)
    od = DeviceAdam(d.parameters(), lr=opt['lr'] * opt['d_lr_mult'], betas=tuple(opt['betas']), fused=True)
    data_rng = torch.Generator().manual_seed(441)
    d_ids, g_ids = torch.arange(8), torch.arange(8, 16)
    for step in range(1, 5):
        d_real, g_real = [torch.randn(8, 2, generator=data_rng) for _ in range(2)]
        d_eps, g_eps = [torch.randn(8, 4, generator=data_rng) for _ in range(2)]
        d.requires_grad_(True)
        with torch.no_grad():
            fake = g(prior(d_ids, eps=d_eps))
        od.zero_grad(set_to_none=True)
        dp = trainer.penalty(d, d_real, fake, step)
        dl = trainer.gan.d_loss(d(d_real), d(fake)) + dp
        dl.backward()
        od.step()
        d.requires_grad_(False)
        og.zero_grad(set_to_none=True)
        with torch.no_grad():
            dr = d(g_real)
        gl = trainer.gan.g_loss(d(g(prior(g_ids, eps=g_eps))), dr)
        encoded = e(g_real, prior.means(), prior.sigma)
        g.requires_grad_(False)
        rec = (g(encoded) - g_real).square().mean()
        g.requires_grad_(True)
        total = gl + rec + trainer.spread(prior.z)
        total.backward()
        og.step()
        for target, source in ((ema_g, g), (ema_e, e), (ema_prior, prior)):
            update_ema(target, source, trainer.config['training']['ema'])
        row, last = trainer.update({'real': d_real}, (trainer.prior(d_ids, eps=d_eps), d_ids),
                                   generator_batch={'real': g_real}, generator_latent_draw=(trainer.prior(g_ids, eps=g_eps), g_ids))
        assert row['d_loss'] == pytest.approx(float(dl.detach()), rel=2e-6, abs=2e-7)
        assert row['g_loss'] == pytest.approx(float(total.detach()), rel=2e-6, abs=2e-7)
        assert row['gradient_penalty'] > 0 if step % 2 == 0 else row['gradient_penalty'] == 0
        assert torch.equal(last['real'], g_real)
        for name, expected in (('generator', g), ('encoder', e), ('discriminator', d)):
            close(trainer.graph.models[name].state_dict(), expected.state_dict())
            for actual_p, expected_p in zip(trainer.graph.models[name].parameters(), expected.parameters()):
                close(actual_p.grad, expected_p.grad)
        close(trainer.prior.state_dict(), prior.state_dict())
        close(trainer.prior.z.grad, prior.z.grad)
        close(trainer.opt_g.state_dict(), og.state_dict())
        close(trainer.opt_d.state_dict(), od.state_dict())
        for actual, expected in ((trainer.ema_graph.models['generator'], ema_g), (trainer.ema_graph.models['encoder'], ema_e), (trainer.ema_prior, ema_prior)):
            close(actual.state_dict(), expected.state_dict())


def test_reconstruction_forward_has_only_encoder_parameter_gradients():
    trainer = ReferenceTrainer(config())
    context = trainer.graph.generate(torch.zeros(8, 4), {'real': torch.ones(8, 2)}, prior=trainer.prior)
    rec = trainer.graph.resolve('components.reconstruction', context).square().mean()
    e = list(trainer.graph.models['encoder'].parameters())
    excluded = [*trainer.graph.models['generator'].parameters(), trainer.prior.z]
    grads = torch.autograd.grad(rec, e + excluded, allow_unused=True)
    assert all(gradient is not None and gradient.abs().sum() > 0 for gradient in grads[:len(e)])
    assert all(gradient is None for gradient in grads[len(e):])
    assert 'reconstruction' not in trainer.graph.models
    assert all(parameter.requires_grad for parameter in excluded)


@pytest.mark.parametrize('boundary', [1, 2])
def test_complete_recovery_before_and_after_lazy_penalty(boundary):
    trainer = ReferenceTrainer(config())
    initial_data = trainer.streams['data'].get_state().clone()
    for _ in range(boundary):
        _, batch = trainer.update()
    checkpoint = copy.deepcopy(trainer_state(trainer, batch))
    for _ in range(4 - boundary):
        _, batch = trainer.update()
    expected = copy.deepcopy(trainer_state(trainer, batch))
    restored = ReferenceTrainer(config())
    restore_trainer(restored, checkpoint)
    for _ in range(4 - boundary):
        _, batch = restored.update()
    close(trainer_state(restored, batch), expected, exact=True)
    # Four updates consumed eight independent real draws, not four reused batches.
    stream = torch.Generator().set_state(initial_data)
    for _ in range(8):
        restored.data(8, generator=stream)
    assert torch.equal(restored.streams['data'].get_state(), stream.get_state())


def test_native_only_recipe_rejected_by_replicated_profile_before_runtime():
    profile = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}}
    with pytest.raises(ValueError, match='require native execution'):
        resolve_execution_profile(profile, config())
    assert resolve_execution_profile({'schema_version': 1, 'execution': {'name': 'cpu-single'}}, config())


def test_alias_validation_and_resolved_roundtrip():
    assert resolve_config(config_values(config())) == config()
    raw = config_values(config())
    raw['components']['reconstruction']['reuse'] = 'discriminator'
    with pytest.raises(ValueError, match='reuse'):
        resolve_config(raw)
    raw = config_values(config())
    raw['prior']['kind'] = 'particles'
    with pytest.raises(ValueError, match='MoG'):
        resolve_config(raw)


def test_component_construction_order_is_independent_of_json_key_order():
    original = config()
    reordered = resolve_config(json.loads(json.dumps(config_values(original), sort_keys=True)))
    first = ReferenceTrainer(original)
    state = copy.deepcopy(trainer_state(first, None))
    second = ReferenceTrainer(reordered)
    close(trainer_state(second, None), state, exact=True)


def test_alias_and_prior_bindings_survive_inference_bundle_and_preview_snapshot(tmp_path):
    from hypergan.artifacts import sample, save_bundle
    from hypergan.preview_snapshot import capture_snapshot, renderer_command
    from hypergan.previews import render_preview
    raw = config_values(config())
    raw['components'] = {
        'generator': {'factory': f'{__name__}:PriorGenerator', 'inputs': {'z': 'components.reused', 'means': 'prior.means', 'sigma': 'prior.sigma'}},
        'projection': {'factory': 'linear', 'args': {'in_features': 4, 'out_features': 4}, 'inputs': {'input': 'batch.unused'}},
        'reused': {'reuse': 'projection', 'inputs': {'input': 'latent'}},
        'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1}, 'inputs': {'input': 'candidate'}}}
    raw['objectives'] = []
    trainer = ReferenceTrainer(resolve_config(raw))
    _, batch = trainer.update()
    save_bundle(tmp_path, trainer, batch)
    saved = torch.load(tmp_path / 'model.pt', weights_only=True)
    assert saved['model_states'].keys() == {'generator', 'projection'}
    assert saved['components'].keys() == {'generator', 'projection', 'reused'}
    assert saved['example_inputs'] == {}
    generated = json.loads(sample(tmp_path, count=8, seed=123).read_text())
    assert generated['shape'] == [8, 2]
    expected = render_preview(trainer, batch, {'sample_sequence': 1})
    path, output = tmp_path / 'snapshot.pt', tmp_path / 'preview.json'
    descriptor = capture_snapshot(trainer, batch, {'sample_sequence': 1}, path)
    renderer_command((str(path), descriptor, {'sample_sequence': 1}, trainer.step, str(output)), 'render', None)
    assert json.loads(output.read_text()) == expected

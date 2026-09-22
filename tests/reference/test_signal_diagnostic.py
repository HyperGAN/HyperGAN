"""Analytic gradient transmission and observation-only state checks."""
import copy

import pytest
import torch
from torch import nn

from hypergan.config import resolve_config
from hypergan.signal_diagnostic import _probe, _digest_state
from hypergan.training import ReferenceTrainer


class Chain(nn.Module):
    def __init__(self, gain=0.5, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(depth)])
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(gain * torch.eye(2))
                layer.bias.zero_()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Critic(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 1))
        self.fail = fail

    def forward(self, x):
        if self.fail:
            raise ValueError('deliberate critic failure')
        return x @ self.weight


class SaturatingCritic(Critic):
    def __init__(self):
        super().__init__()
        self.saturation = nn.Tanh()

    def forward(self, x):
        return self.saturation(64 * (x @ self.weight))


class PretrainedCritic(Critic):
    def __init__(self):
        from hndl.operators.pretrained import Pretrained
        class LocalPretrained(Pretrained):
            def __init__(self):
                nn.Module.__init__(self)
                self.model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                with torch.no_grad():
                    self.model[0].weight.copy_(torch.eye(2))
                    self.model[0].bias.zero_()
                self.eval().requires_grad_(False)

            def forward(self, x):
                return self.model(x)
        super().__init__()
        self.features = LocalPretrained()

    def forward(self, x):
        return self.features(x) @ self.weight


class FrozenFeatures(Chain):
    def __init__(self):
        super().__init__()
        self.features = nn.BatchNorm1d(2).eval().requires_grad_(False)

    def forward(self, x):
        return self.features(super().forward(x))


class MutatingChain(Chain):
    def forward(self, x):
        with torch.no_grad():
            self.layers[0].weight.add_(0.01)
        return super().forward(x)


class MutatingFrozenFeatures(FrozenFeatures):
    def forward(self, x):
        with torch.no_grad():
            self.features.running_mean.add_(0.01)
        return super().forward(x)


def _trainer(factory='Chain', critic='Critic', **args):
    config = resolve_config({
        'components': {
            'generator': {'factory': f'{__name__}:{factory}', 'args': args, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': f'{__name__}:{critic}', 'inputs': {'x': 'candidate'}}},
        'prior': {'kind': 'gaussian', 'args': {'z_dim': 2}},
        'prior_regularizer': {'weight': 0.0},
        'training': {'batch_size': 4, 'device': 'cpu'},
    })
    return ReferenceTrainer(config)


def _draw():
    return {'real': torch.ones(4, 2)}, (torch.tensor([[.2, .3], [.4, .5], [.6, .7], [.8, .9]]), None)


def test_chain_detects_early_attenuation_without_any_update():
    trainer = _trainer()
    batch, latent = _draw()
    before = _digest_state((('g', trainer.graph), ('p', trainer.prior)))
    result = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    rows = {r['path']: r for r in result['activations']}
    assert rows['generator.layers.0']['gradient_to_output_rms_ratio'] == pytest.approx(.5 ** 3)
    assert rows['generator.layers.3']['gradient_to_output_rms_ratio'] == pytest.approx(1)
    assert before == _digest_state((('g', trainer.graph), ('p', trainer.prior)))
    assert all(p.grad is None for p in trainer.graph.parameters())
    assert trainer.step == 0
    assert not trainer.opt_g.state and not trainer.opt_d.state
    assert result['state_verification']['optimizer_steps'] == 0


def test_frozen_features_pass_input_gradients_and_keep_buffers():
    trainer = _trainer('FrozenFeatures')
    features = trainer.graph.models['generator'].features
    before = copy.deepcopy(features.state_dict())
    batch, latent = _draw()
    result = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    assert result['summary']['first_observed_to_output_rms_ratio'] > 0
    for name, value in features.state_dict().items():
        assert torch.equal(value, before[name])
    assert not features.training
    assert all(not p.requires_grad and p.grad is None for p in features.parameters())


def test_hook_and_requires_grad_cleanup_after_failed_forward():
    trainer = _trainer()
    trainer.graph.models['discriminator'].fail = True
    before = [(p, p.requires_grad) for p in trainer.graph.parameters()]
    batch, latent = _draw()
    with pytest.raises(ValueError, match='deliberate critic failure'):
        _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    assert all(p.requires_grad == flag for p, flag in before)
    assert all(not module._forward_hooks for module in trainer.graph.modules())
    assert all(not module._forward_pre_hooks for module in trainer.graph.modules())


@pytest.mark.parametrize(('factory', 'path'), [
    ('MutatingChain', 'graph.parameter.models.generator.layers.0.weight'),
    ('MutatingFrozenFeatures', 'graph.buffer.models.generator.features.running_mean'),
])
def test_state_audit_refuses_mutation_and_identifies_registered_tensor(factory, path):
    trainer = _trainer(factory)
    batch, latent = _draw()
    flags = [(parameter, parameter.requires_grad) for parameter in trainer.graph.parameters()]
    buffers = [(value, value.clone()) for value in trainer.graph.buffers()]
    with pytest.raises(ValueError, match='report refused') as failure:
        _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    assert path in str(failure.value)
    assert all(parameter.requires_grad == flag for parameter, flag in flags)
    assert all(torch.equal(value, saved) for value, saved in buffers)
    assert all(not module._forward_hooks for module in trainer.graph.modules())
    assert all(not module._forward_pre_hooks for module in trainer.graph.modules())
    assert trainer.step == 0
    assert not trainer.opt_g.state and not trainer.opt_d.state


def test_entry_hashes_do_not_override_aggregate_failure(monkeypatch):
    import hypergan.signal_diagnostic as diagnostic
    trainer = _trainer()
    batch, latent = _draw()
    calls = 0

    def inconsistent_aggregate(modules, *, entries=None):
        nonlocal calls
        digest = _digest_state(modules, entries=entries)
        calls += 1
        return digest if calls == 1 else 'different-aggregate'

    monkeypatch.setattr(diagnostic, '_digest_state', inconsistent_aggregate)
    with pytest.raises(ValueError, match='none identified; aggregate mismatch remains fatal'):
        _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)


def test_optional_entry_hashes_preserve_existing_aggregate():
    module = nn.Linear(2, 2)
    module.register_buffer('scalar', torch.tensor(1.0), persistent=False)
    entries = {}
    before = _digest_state((('test', module),))
    assert _digest_state((('test', module),), entries=entries) == before
    assert set(entries) == {'test.parameter.weight', 'test.parameter.bias', 'test.buffer.scalar'}
    with torch.no_grad():
        module.scalar.add_(1.)
    changed = {}
    assert _digest_state((('test', module),), entries=changed) != before
    assert {name for name in entries if entries[name] != changed[name]} == {'test.buffer.scalar'}


def test_total_objective_matches_adversarial_without_auxiliary_terms():
    trainer = _trainer()
    batch, latent = _draw()
    adv = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    total = _probe(trainer, 'total', batch=batch, latent_draw=latent)
    assert adv['loss'] == total['loss']
    assert adv['parameters'] == total['parameters']


def test_saturating_critic_blocks_signal_despite_healthy_generator_transmission():
    from hypergan.initialization_tuning import _structural
    trainer = _trainer(gain=1., critic='SaturatingCritic')
    batch, latent = _draw()
    before = _digest_state((('graph', trainer.graph), ('prior', trainer.prior)))
    layers = [(name, module) for name, module in trainer.graph.models['generator'].named_modules()
              if isinstance(module, nn.Linear)]
    structural = _structural(trainer, batch, latent, layers)
    assert structural['score'] == pytest.approx(0.)
    result = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    profile = result['summary']['discriminator_profiles'][0]
    # The default relativistic loss has sigmoid(0) / batch at equal scores.
    assert profile['fake_score_gradient_rms'] == pytest.approx(.5 / len(batch['real']))
    assert profile['fake_input_gradients'][0]['gradient_rms'] == 0.
    assert profile['fake_input_gradients'][0]['gradient_to_score_rms_ratio'] == 0.
    assert result['summary']['generated_output_gradient_rms'] == 0.
    rows = result['discriminator_activations']
    assert {row['sample'] for row in rows} == {'fake', 'real'}
    assert all(row['gradient'] is None for row in rows if row['sample'] == 'real')
    assert all(row['critic_invocation'] == (0 if row['sample'] == 'fake' else 1) for row in rows)
    assert _digest_state((('graph', trainer.graph), ('prior', trainer.prior))) == before
    assert all(parameter.grad is None for parameter in trainer.graph.parameters())
    assert result['state_verification']['parameters_and_restored_buffers_unchanged']


def test_pretrained_critic_interfaces_pass_gradients_without_instrumenting_internals():
    trainer = _trainer(gain=1., critic='PretrainedCritic')
    batch, latent = _draw()
    features = trainer.graph.models['discriminator'].features
    before = copy.deepcopy(features.state_dict())
    result = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    rows = result['discriminator_activations']
    assert not any('.features.model' in row['path'] for row in rows)
    boundaries = [row for row in rows if '.features' in row['path'] and row['sample'] == 'fake']
    assert {row['boundary'] for row in boundaries} == {'input', 'output'}
    assert all(row['gradient']['rms'] > 0 for row in boundaries)
    assert result['summary']['discriminator_profiles'][0]['fake_input_gradients'][0]['gradient_rms'] > 0
    for name, value in features.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in features.parameters())
    assert not features.training

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


class FrozenFeatures(Chain):
    def __init__(self):
        super().__init__()
        self.features = nn.BatchNorm1d(2).eval().requires_grad_(False)

    def forward(self, x):
        return self.features(super().forward(x))


def _trainer(factory='Chain', **args):
    config = resolve_config({
        'components': {
            'generator': {'factory': f'{__name__}:{factory}', 'args': args, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': f'{__name__}:Critic', 'inputs': {'x': 'candidate'}}},
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


def test_total_objective_matches_adversarial_without_auxiliary_terms():
    trainer = _trainer()
    batch, latent = _draw()
    adv = _probe(trainer, 'adversarial', batch=batch, latent_draw=latent)
    total = _probe(trainer, 'total', batch=batch, latent_draw=latent)
    assert adv['loss'] == total['loss']
    assert adv['parameters'] == total['parameters']

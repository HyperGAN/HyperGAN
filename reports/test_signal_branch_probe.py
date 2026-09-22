"""Analytic CPU guard/reconstruction checks for the research-only branch helper."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from hypergan.checkpoints import capture_rng
from hypergan.config import resolve_config
from hypergan.distributed_checkpoints import _digest
from hypergan.training import ReferenceTrainer

_spec = importlib.util.spec_from_file_location('signal_branch_probe', Path(__file__).with_name('signal_branch_probe.py'))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
probe_branches = _module.probe_branches


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 48)

    def forward(self, x):
        return self.layer(x).reshape(-1, 3, 4, 4)


class FiveBranches(nn.Module):
    def __init__(self, bad_mix=False):
        super().__init__()
        self.bad_mix = bad_mix
        self.network = nn.Module()
        self.network.nodes = nn.ModuleDict({'n_' + name: nn.Linear(48, 1, bias=False)
                                           for name in _module.COEFFICIENTS})
        with torch.no_grad():
            for value, layer in zip((1., 1., -1., .5, -.5), self.network.nodes.values()):
                layer.weight.fill_(value / 48)

    def forward(self, x):
        scores = {name: self.network.nodes['n_' + name](x.flatten(1)) for name in _module.COEFFICIENTS}
        result = sum(_module.COEFFICIENTS[name] * score for name, score in scores.items())
        return result * (2 if self.bad_mix else 1)


def _trainer(bad_mix=False):
    return ReferenceTrainer(resolve_config({
        'components': {
            'generator': {'factory': f'{__name__}:Generator', 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': f'{__name__}:FiveBranches', 'args': {'bad_mix': bad_mix},
                              'inputs': {'x': 'candidate'}}},
        'prior': {'kind': 'gaussian', 'args': {'z_dim': 2}},
        'training': {'device': 'cpu', 'batch_size': 3},
        'prior_regularizer': {'weight': 0.0},
    }))


def _inputs():
    return {'real': torch.linspace(-1, 1, 144).reshape(3, 3, 4, 4)}, (torch.tensor([[.2, .3], [.4, .1], [.5, -.2]]), None)


def _state(trainer):
    return _digest({'rng': capture_rng(), 'streams': {name: stream.get_state() for name, stream in trainer.streams.items()},
                    'graph': trainer.graph.state_dict(), 'prior': trainer.prior.state_dict(),
                    'flags': [p.requires_grad for p in trainer.graph.parameters()],
                    'modes': [m.training for m in trainer.graph.modules()],
                    'optimizers': [trainer.opt_g.state_dict(), trainer.opt_d.state_dict()]})


def test_linear_heads_reconstruct_actual_loss_gradient_and_reveal_cancellation():
    trainer = _trainer()
    batch, latent = _inputs()
    before = _state(trainer)
    result = probe_branches(trainer, batch, latent)
    assert _state(trainer) == before
    assert result['state_verification']['unchanged'] and trainer.step == 0
    assert result['gradient_reconstruction']['relative_l2_error'] < 1e-6
    # Four feature gradients cancel exactly, leaving the pixel branch.
    assert result['branches']['pixel_score']['norm_ratio_to_total_image_gradient'] == pytest.approx(1, rel=1e-6)
    assert result['branches']['feature1_score']['norm_ratio_to_total_image_gradient'] == pytest.approx(.5, rel=1e-6)
    assert result['cancellation_fraction'] == pytest.approx(.6, abs=1e-6)
    pairs = {(row['left'], row['right']): row['cosine'] for row in result['pairwise_branch_gradient_cosines']}
    assert pairs['feature1_score', 'feature2_score'] == pytest.approx(-1, abs=1e-6)
    assert result['total_image_gradient']['sample_coordinate_pairwise_cosine_mean'] == pytest.approx(1, abs=1e-6)
    pooled = result['total_image_gradient']['pooled_spatial_gradient']['4']
    assert pooled['residual_high_frequency_energy_fraction'] == pytest.approx(0, abs=1e-8)
    assert all(p.grad is None for p in trainer.graph.parameters())


def test_architecture_mixture_guard_preserves_state_on_failure():
    trainer = _trainer(bad_mix=True)
    batch, latent = _inputs()
    before = _state(trainer)
    with pytest.raises(ValueError, match='guarded pixel-plus-four-feature'):
        probe_branches(trainer, batch, latent)
    assert _state(trainer) == before
    assert all(not module._forward_hooks for module in trainer.graph.modules())


def test_missing_named_branch_rejects_before_probe():
    trainer = _trainer()
    del trainer.graph.models['discriminator'].network.nodes['n_feature4_score']
    before = _state(trainer)
    with pytest.raises(ValueError, match='n_feature4_score'):
        probe_branches(trainer, *_inputs())
    assert _state(trainer) == before


@pytest.mark.parametrize('scale', [0., 1e-12])
def test_zero_and_tiny_branch_gradients_report_finite_statistics(scale):
    import json
    trainer = _trainer()
    with torch.no_grad():
        for value in trainer.graph.models['discriminator'].parameters():
            value.mul_(scale)
    result = probe_branches(trainer, *_inputs())
    json.dumps(result, allow_nan=False)
    assert result['gradient_reconstruction']['relative_l2_error'] < 1e-6
    if scale == 0:
        assert result['total_image_gradient']['rms'] == 0
        assert result['cancellation_fraction'] is None
        assert all(row['norm_ratio_to_total_image_gradient'] is None for row in result['branches'].values())
    else:
        assert 0 < result['total_image_gradient']['rms'] < 1e-12
        assert result['cancellation_fraction'] == pytest.approx(.6, abs=1e-6)
        assert result['branches']['pixel_score']['norm_ratio_to_total_image_gradient'] == pytest.approx(1, rel=1e-6)

"""Research-only feature estimator and read-only probe failure contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hypergan.checkpoints import trainer_state
from hypergan.config import resolve_config
from hypergan.startup_dynamics import _same_state, _snapshot
from hypergan.training import ReferenceTrainer


_spec = importlib.util.spec_from_file_location(
    'frozen_feature_probe', Path(__file__).resolve().parents[2] / 'reports/frozen_feature_probe.py')
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_polynomial_mmd_is_unbiased_and_does_not_clamp_negative_estimates():
    real = torch.tensor([[0.], [1.]])
    fake = torch.tensor([[2.], [3.]])
    # Within-real=1, within-fake=343; cross mean=(1+1+27+64)/4.
    assert probe.polynomial_mmd2_unbiased(real, fake) == pytest.approx(297.5)
    assert probe.polynomial_mmd2_unbiased(real, real) == pytest.approx(-3.5)
    assert probe.polynomial_mmd2_unbiased(real.flip(0), fake.flip(0)) == pytest.approx(297.5)


@pytest.mark.parametrize('real,fake', [
    (torch.ones(1, 2), torch.ones(2, 2)),
    (torch.ones(2, 2), torch.ones(2, 3)),
    (torch.full((2, 2), float('nan')), torch.ones(2, 2)),
])
def test_polynomial_mmd_rejects_invalid_inputs(real, fake):
    with pytest.raises(ValueError):
        probe.polynomial_mmd2_unbiased(real, fake)


class FakeBackbone(torch.nn.Module):
    def __init__(self, mutate=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.register_buffer('counter', torch.zeros(()))
        self.mutate = mutate

    def forward(self, value):
        if self.mutate:
            self.counter.add_(1)
        pooled = value[:, :1].mean((-2, -1), keepdim=True)
        depths = torch.arange(1536, device=value.device)[None, :, None, None] / 1000
        return (pooled + depths).expand(-1, -1, 8, 8)


@pytest.mark.parametrize('failure', [None, 'normalization', 'protected_mutation'])
def test_measure_restores_full_native_state_and_removes_hook_even_on_failure(monkeypatch, failure):
    trainer = ReferenceTrainer(resolve_config({'training': {'steps': 32, 'batch_size': 2}}))
    backbone = FakeBackbone(mutate=failure == 'protected_mutation')
    trainer.graph.add_module('feature_test', backbone)
    fake = torch.stack((torch.zeros(3, 128, 128), torch.ones(3, 128, 128) * .25))
    real = torch.stack((torch.ones(3, 128, 128) * -.25, torch.ones(3, 128, 128) * .5))
    bank = ({'real': real}, (torch.zeros(2, 2), None))
    term = SimpleNamespace(weight=1., generator_phase=None, gan=SimpleNamespace(
        g_loss=lambda f, r: (f - r).square().mean(),
        d_loss=lambda r, f: (r - f).square().mean()))
    monkeypatch.setattr(probe, '_architecture', lambda _: (term, 'n_backbone', backbone))

    def draw(batch, latent):
        # Exercise restoration of RNG and ordinary trainer metadata.
        torch.rand(3)
        trainer.step += 1
        return batch, None, {'generated': fake, 'batch': batch}

    monkeypatch.setattr(trainer, '_draw', draw)

    def scores(term, context, graph, phase_name, phase, *, first):
        outputs = []
        for candidate in (context['generated'], context['batch']['real']):
            joined = torch.cat((candidate, torch.zeros_like(candidate)))
            mean = joined.new_tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
            std = joined.new_tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
            normalized = (joined * .5 + .5 - mean) / std
            if failure == 'normalization':
                normalized[len(candidate):].add_(1)
            feature = backbone(normalized)
            outputs.append(feature[:len(candidate)].mean((1, 2, 3))[:, None])
        return real, fake, outputs[1], outputs[0]

    monkeypatch.setattr(probe, '_bound_scores', scores)
    before = _snapshot(trainer)
    if failure:
        with pytest.raises((ValueError, RuntimeError), match='context changed|mutated protected'):
            probe.measure_frozen_features(trainer, bank)
    else:
        result = probe.measure_frozen_features(trainer, bank)
        assert result['samples'] == 2
        assert result['feature_width'] == 384
        assert result['pretrained_images_including_gray_context'] == 8
        # Constant context rows were excluded: fake spread comes from the two
        # actual candidates, not four rows diluted by duplicated gray images.
        assert result['dino_fake_feature_spread'] == pytest.approx(.25 * .5 / .229 / 2, rel=1e-6)
        assert all(value is None or isinstance(value, (str, int, float, bool)) for value in result.values())
    assert not backbone._forward_hooks
    assert _same_state(before['state'], trainer_state(trainer, None))

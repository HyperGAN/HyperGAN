"""Research-only feature estimator and read-only probe failure contracts."""
import importlib.util
from dataclasses import replace
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


def test_architecture_accepts_native_real_score_stop_gradient():
    from hndl.operators.pretrained import Pretrained
    from hypergan.pretrained_providers import _dinov3_multidepth

    trainer = ReferenceTrainer(resolve_config({'training': {'steps': 32, 'batch_size': 2}}))
    native_term = trainer.program.adversarial_terms[0]
    assert native_term.generator_phase.real.detach_score is True
    # Construct only the native wrapper identity/metadata needed by the guard;
    # no pretrained artifact is loaded, downloaded or executed in this test.
    backbone = Pretrained.__new__(Pretrained)
    torch.nn.Module.__init__(backbone)
    backbone.readout = 'multidepth'
    backbone.provider = SimpleNamespace(read=_dinov3_multidepth)
    backbone.source = SimpleNamespace(config={'provider': 'dinov3_vits16'})
    backbone.model = torch.nn.Linear(2, 2).requires_grad_(False).eval()
    critic = torch.nn.Module()
    critic.add_module('n_backbone', backbone)
    term = replace(native_term, module=critic)
    guarded_trainer = SimpleNamespace(program=SimpleNamespace(adversarial_terms=(term,)))
    assert probe._architecture(guarded_trainer) == (term, 'n_backbone', backbone)


class FakeBackbone(torch.nn.Module):
    def __init__(self, mutate=False, grid=None, channels=1536):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.register_buffer('counter', torch.zeros(()))
        self.mutate = mutate
        self.grid = grid
        self.channels = channels

    def forward(self, value):
        if self.mutate:
            self.counter.add_(1)
        pooled = value[:, :1].mean((-2, -1), keepdim=True)
        depths = torch.arange(self.channels, device=value.device)[None, :, None, None] / 1000
        # Default grid follows the image under test. An explicit grid can disagree.
        grid = self.grid if self.grid is not None else {128: 8, 64: 4}[int(value.shape[-1])]
        return (pooled + depths).expand(-1, -1, grid, grid)


def _prepare(monkeypatch, *, edge, grid=None, channels=1536, failure=None):
    trainer = ReferenceTrainer(resolve_config({'training': {'steps': 32, 'batch_size': 2}}))
    backbone = FakeBackbone(mutate=failure == 'protected_mutation', grid=grid, channels=channels)
    trainer.graph.add_module('feature_test', backbone)
    fake = torch.stack((torch.zeros(3, edge, edge), torch.ones(3, edge, edge) * .25))
    real = torch.stack((torch.ones(3, edge, edge) * -.25, torch.ones(3, edge, edge) * .5))
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
    return trainer, backbone, bank, _snapshot(trainer)


def _assert_restored(backbone, before, trainer):
    assert not backbone._forward_hooks
    assert _same_state(before['state'], trainer_state(trainer, None))


@pytest.mark.parametrize('failure', [None, 'normalization', 'protected_mutation'])
def test_measure_restores_full_native_state_and_removes_hook_even_on_failure(monkeypatch, failure):
    trainer, backbone, bank, before = _prepare(monkeypatch, edge=128, failure=failure)
    if failure:
        with pytest.raises((ValueError, RuntimeError), match='context changed|mutated protected'):
            probe.measure_frozen_features(trainer, bank)
    else:
        result = probe.measure_frozen_features(trainer, bank)
        assert result['samples'] == 2
        assert result['feature_width'] == 384
        assert result['protocol'] == 'online_dinov3_block11_spatial_mean_candidate_only_poly3'
        assert result['image_edge'] == 128
        assert result['patch_grid_edge'] == 8
        assert result['patch_size'] == 16
        assert result['pretrained_images_including_gray_context'] == 8
        # Constant context rows were excluded: fake spread comes from the two
        # actual candidates, not four rows diluted by duplicated gray images.
        assert result['dino_fake_feature_spread'] == pytest.approx(.25 * .5 / .229 / 2, rel=1e-6)
        assert all(value is None or isinstance(value, (str, int, float, bool)) for value in result.values())
    _assert_restored(backbone, before, trainer)


def test_measure_accepts_declared_64px_grid_with_distinct_protocol(monkeypatch):
    trainer, backbone, bank, before = _prepare(monkeypatch, edge=64)
    result = probe.measure_frozen_features(trainer, bank)
    assert result['protocol'] == 'online_dinov3_block11_spatial_mean_candidate_only_poly3_64px_4x4'
    assert result['image_edge'] == 64
    assert result['patch_grid_edge'] == 4
    assert result['patch_size'] == 16
    assert result['feature_width'] == 384
    assert all(value is None or isinstance(value, (str, int, float, bool)) for value in result.values())
    _assert_restored(backbone, before, trainer)


@pytest.mark.parametrize('edge,grid,size', [(64, 8, '8x8'), (128, 4, '4x4')])
def test_measure_rejects_declared_image_with_the_other_grid(monkeypatch, edge, grid, size):
    trainer, backbone, bank, before = _prepare(monkeypatch, edge=edge, grid=grid)
    with pytest.raises(ValueError, match=f'rejected spatial size {size}'):
        probe.measure_frozen_features(trainer, bank)
    _assert_restored(backbone, before, trainer)


@pytest.mark.parametrize('edge', [32, 96, 256])
def test_measure_rejects_undeclared_square_edges(monkeypatch, edge):
    trainer, backbone, bank, before = _prepare(monkeypatch, edge=edge)
    with pytest.raises(ValueError, match=rf'rejected spatial size {edge}x{edge}'):
        probe.measure_frozen_features(trainer, bank)
    _assert_restored(backbone, before, trainer)


def test_measure_rejects_matching_image_with_wrong_channel_count(monkeypatch):
    trainer, backbone, bank, before = _prepare(monkeypatch, edge=64, channels=768)
    with pytest.raises(ValueError, match=r'rejected shape \(4, 768, 4, 4\)'):
        probe.measure_frozen_features(trainer, bank)
    _assert_restored(backbone, before, trainer)

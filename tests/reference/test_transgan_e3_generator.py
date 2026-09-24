"""Projected latent shortcuts preserve the backbone and provide direct gradients."""
import gc
from pathlib import Path

import pytest
import torch

from hypergan.hndl_networks import build_network


NETWORKS = Path(__file__).parents[2] / 'examples/networks'
SHORTCUTS = ((32, 256), (64, 64))


@pytest.fixture(scope='module')
def generators():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    models = []
    try:
        for variant in ('stable', 'e3'):
            with torch.random.fork_rng(devices=[]):
                model = build_network(
                    (NETWORKS / f'transgan-generator-128-{variant}.hndl').read_text(),
                    input_shape=('B', 512), output_shape=('B', 3, 128, 128))
            models.append(model)
        yield models
    finally:
        models.clear()
        gc.collect()
        torch.set_num_threads(previous)


def test_backbone_weights_load_and_zero_shortcuts_recover_baseline(generators):
    baseline, shortcuts = generators
    before, after = baseline.state_dict(), shortcuts.state_dict()
    extra = set(after) - set(before)
    assert len(extra) == 2 and all('latent_projection' in key for key in extra)
    assert sum(p.numel() for p in shortcuts.parameters()) == 151_512_455 + 163_840
    for name, tensor in before.items():
        assert name in after and after[name].shape == tensor.shape
    result = shortcuts.load_state_dict(before, strict=False)
    assert set(result.missing_keys) == extra and not result.unexpected_keys
    projections = [shortcuts[f'stage{side}_latent_projection'] for side, _ in SHORTCUTS]
    saved = [module.weight.detach().clone() for module in projections]
    try:
        with torch.no_grad():
            for module in projections:
                module.weight.zero_()
            z = torch.arange(1024, dtype=torch.float32).cos().reshape(2, 512)
            torch.testing.assert_close(shortcuts(z), baseline(z), rtol=0, atol=0)
    finally:
        with torch.no_grad():
            for module, weight in zip(projections, saved, strict=True):
                module.weight.copy_(weight)


@pytest.mark.parametrize('active_side', [32, 64])
def test_each_shortcut_supplies_original_z_and_gradients_without_stem(generators, active_side):
    _, model = generators
    flags = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    active = model[f'stage{active_side}_latent_projection']
    active.weight.requires_grad_(True)
    # Two different samples detect accidental broadcasting across batch entries.
    z = torch.arange(1024, dtype=torch.float32).cos().reshape(2, 512).requires_grad_()
    observed = []
    handles = [model['input_projection'].register_forward_hook(
        lambda module, inputs, output: output.detach())]

    def check_latent(module, inputs):
        assert inputs[0] is z

    def check_injection(module, inputs, output, *, side, channels):
        features, latent = inputs
        assert features.shape == (2, side * side, channels)
        assert latent.shape == (2, 1, channels)
        torch.testing.assert_close(output, features + latent.expand_as(features), rtol=0, atol=0)
        expected = model[f'stage{side}_latent_projection'].weight.detach()
        torch.testing.assert_close(latent[:, 0], z.detach() @ expected.T)
        assert not torch.equal(latent[0], latent[1])
        observed.append(side)

    for side, channels in SHORTCUTS:
        projection = model[f'stage{side}_latent_projection']
        assert projection.weight.shape == (channels, 512) and projection.bias is None
        handles.append(projection.register_forward_pre_hook(check_latent))
        if side != active_side:
            handles.append(projection.register_forward_hook(
                lambda module, inputs, output: output.detach()))
        handles.append(model[f'stage{side}_latent_injection'].register_forward_hook(
            lambda module, inputs, output, side=side, channels=channels:
            check_injection(module, inputs, output, side=side, channels=channels)))
    rng = torch.get_rng_state().clone()
    try:
        output = model(z)
        assert output.shape == (2, 3, 128, 128) and torch.isfinite(output).all()
        dz, dw = torch.autograd.grad(output.square().mean(), (z, active.weight))
        assert torch.isfinite(dz).all() and (dz.abs().sum(dim=1) > 0).all()
        assert torch.isfinite(dw).all() and dw.abs().sum() > 0
        assert observed == [32, 64]
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(model.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)

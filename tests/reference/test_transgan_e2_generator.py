"""E2 residual scaling preserves weights while changing the depth dynamics."""
import gc
import math
from pathlib import Path

import pytest
import torch

from hypergan.hndl_networks import build_network


NETWORKS = Path(__file__).parents[2] / 'examples/networks'
STAGES = ((8, 5), (16, 4), (32, 4), (64, 4), (128, 4))
RESIDUALS = tuple(name for side, depth in STAGES for block in range(depth)
                  for name in (f'stage{side}_block{block}_attention_residual',
                               f'stage{side}_block{block}'))
SCALE = 1 / math.sqrt(42)


@pytest.fixture(scope='module')
def generators():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    models = []
    for variant in ('stable', 'e2'):
        # Replay the same construction RNG, without advancing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            model = build_network(
                (NETWORKS / f'transgan-generator-128-{variant}.hndl').read_text(),
                input_shape=('B', 512), output_shape=('B', 3, 128, 128))
        models.append(model)
    try:
        yield models
    finally:
        models.clear()
        del model
        gc.collect()
        torch.set_num_threads(previous)


def test_e2_preserves_all_weight_tensors_and_initialization(generators):
    baseline, scaled = generators
    assert sum(p.numel() for p in scaled.parameters()) == 151_512_455
    before, after = baseline.state_dict(), scaled.state_dict()
    assert list(before) == list(after)
    for name in before:
        assert torch.equal(before[name], after[name]), name
    # Generator state dictionaries are compatible; full training checkpoints
    # still have their own recipe-identity checks.
    scaled.load_state_dict(before, strict=True)
    scale_nodes = {node.id for node in scaled.plan.nodes if node.id.endswith('_scale')}
    assert scale_nodes == {f'{name}_scale' for name in RESIDUALS}
    for name in scale_nodes:
        module = scaled[name]
        assert module.factor == SCALE
        assert not list(module.parameters()) and not list(module.buffers())


def test_e2_scales_each_branch_before_addition_and_preserves_latent_gradient(generators):
    baseline, scaled = generators
    flags = [p.requires_grad for p in scaled.parameters()]
    scaled.requires_grad_(False)
    scaled.train()
    raw_branches = {}
    observed = []
    handles = []

    def capture_branch(module, inputs, *, name):
        raw_branches[name] = inputs[0].detach()

    def check_residual(module, inputs, output, *, name):
        bypass, branch = inputs
        expected_branch = raw_branches.pop(name) * SCALE
        torch.testing.assert_close(branch, expected_branch, rtol=0, atol=0)
        torch.testing.assert_close(output, bypass + expected_branch, rtol=0, atol=0)
        observed.append(name)

    for name in RESIDUALS:
        handles.append(scaled[f'{name}_scale'].register_forward_pre_hook(
            lambda module, inputs, name=name: capture_branch(module, inputs, name=name)))
        handles.append(scaled[name].register_forward_hook(
            lambda module, inputs, output, name=name:
            check_residual(module, inputs, output, name=name)))
    latent = torch.arange(512, dtype=torch.float32).cos().reshape(1, 512).requires_grad_()
    rng = torch.get_rng_state().clone()
    try:
        output = scaled(latent)
        assert output.shape == (1, 3, 128, 128) and torch.isfinite(output).all()
        derivative, = torch.autograd.grad(output.square().mean(), latent)
        assert torch.isfinite(derivative).all() and derivative.abs().sum() > 0
        assert observed == list(RESIDUALS) and not raw_branches
        with torch.no_grad():
            original_output = baseline(latent.detach())
        assert not torch.allclose(output, original_output)
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(scaled.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)

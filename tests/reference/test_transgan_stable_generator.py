"""Paper-depth 128px TransGAN generator: geometry, initialization and raw RGB."""
import gc
import math
from pathlib import Path

import pytest
import torch

from hypergan.hndl_networks import build_network


SOURCE = Path(__file__).parents[2] / 'examples/networks/transgan-generator-128-stable.hndl'
STAGES = ((8, 1024, 5), (16, 1024, 4), (32, 256, 4), (64, 64, 4), (128, 16, 4))


@pytest.fixture(scope='module')
def generator():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        model = build_network(SOURCE.read_text(), input_shape=('B', 512),
                              output_shape=('B', 3, 128, 128))
    try:
        yield model
    finally:
        del model
        gc.collect()
        torch.set_num_threads(previous)


def test_stage_depths_positions_and_pixelnorm_are_enabled(generator):
    expected_attention = {f'stage{side}_block{block}_attention'
                          for side, _, depth in STAGES for block in range(depth)}
    actual_attention = {node.id for node in generator.plan.nodes if node.id.endswith('_attention')}
    assert actual_attention == expected_attention
    assert len({id(generator[name]) for name in actual_attention}) == 21
    for side, channels, depth in STAGES:
        position = generator[f'stage{side}_position'].weight
        assert position.shape == (side * side, channels)
        assert position.requires_grad and torch.isfinite(position).all() and position.count_nonzero() > 0
        grid = side if side <= 32 else 16
        for block in range(depth):
            attention = generator[f'stage{side}_block{block}_attention']
            assert attention.heads == 4 and attention.spatial_shape == (grid, grid)
            assert attention.relative_position_bias and attention.dropout == 0
            bias = attention.relative_position_bias_table
            assert bias.shape == ((2 * grid - 1) ** 2, 4)
            assert bias.requires_grad and torch.isfinite(bias).all() and bias.count_nonzero() > 0
            for branch in (1, 2):
                norm = generator[f'stage{side}_block{block}_norm{branch}']
                assert not list(norm.parameters()) and norm.eps == 1e-8
                token = torch.linspace(.5, 1.5, channels).reshape(1, 1, channels) * 1e-4
                expected = token / torch.sqrt(token.square().mean(-1, keepdim=True) + 1e-8)
                torch.testing.assert_close(norm(token), expected, rtol=1e-6, atol=1e-7)


def test_512_latent_projection_preserves_fan_in_initialization(generator):
    stem = generator['input_projection']
    assert (stem.in_features, stem.out_features) == (512, 64 * 1024)
    bound = 1 / math.sqrt(512)
    assert torch.isfinite(stem.weight).all() and stem.weight.detach().abs().max() <= bound + 1e-7
    assert torch.isfinite(stem.bias).all() and stem.bias.detach().abs().max() <= bound + 1e-7
    assert stem.bias.count_nonzero() > 0
    assert stem.weight.detach().var(unbiased=False).item() == pytest.approx(1 / (3 * 512), rel=.02)


def test_full_depth_forward_and_latent_gradient_are_finite(generator):
    generator.train()
    flags = [parameter.requires_grad for parameter in generator.parameters()]
    # Only the input derivative is needed; avoid allocating parameter gradients
    # or retaining activations solely for the full-size weight derivatives.
    generator.requires_grad_(False)
    observed = []
    handles = []
    for side, _, depth in STAGES:
        for block in range(depth):
            name = f'stage{side}_block{block}_attention'
            handles.append(generator[name].register_forward_pre_hook(
                lambda module, inputs, name=name: observed.append((name, tuple(inputs[0].shape)))))
    latent = torch.arange(512, dtype=torch.float32).cos().reshape(1, 512).requires_grad_()
    rng = torch.get_rng_state().clone()
    try:
        output = generator(latent)
        assert output.shape == (1, 3, 128, 128) and torch.isfinite(output).all()
        derivative, = torch.autograd.grad(output.square().mean(), latent)
        assert torch.isfinite(derivative).all() and derivative.abs().sum() > 0
        assert torch.equal(rng, torch.get_rng_state())
        expected = [(f'stage{side}_block{block}_attention',
                     (1 if side <= 32 else (side // 16) ** 2,
                      side * side if side <= 32 else 256, channels))
                    for side, channels, depth in STAGES for block in range(depth)]
        assert observed == expected
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(generator.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)


def test_rgb_readout_preserves_values_outside_unit_range(generator):
    readout = generator['output_projection']
    weight, bias = readout.weight.detach().clone(), readout.bias.detach().clone()
    try:
        with torch.no_grad():
            readout.weight.zero_()
            readout.bias.copy_(torch.tensor([-3., .25, 4.]))
            output = generator(torch.zeros(1, 512))
        expected = readout.bias.detach().reshape(1, 3, 1, 1).expand(1, 3, 128, 128)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        with torch.no_grad():
            readout.weight.copy_(weight)
            readout.bias.copy_(bias)

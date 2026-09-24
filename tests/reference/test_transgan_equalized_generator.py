"""All TransGAN affine projections use runtime equalized learning rate."""
import gc
from pathlib import Path

import pytest
import torch
from torch import nn

from hypergan.hndl_networks import build_network


SOURCE = Path(__file__).parents[2] / 'examples/networks/transgan-generator-128-equalized.hndl'
STAGES = ((8, 1024, 5), (16, 1024, 4), (32, 256, 4), (64, 64, 4), (128, 16, 4))


@pytest.fixture(scope='module')
def generator():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            model = build_network(SOURCE.read_text(), input_shape=('B', 512),
                                  output_shape=('B', 3, 128, 128))
        yield model
    finally:
        gc.collect()
        torch.set_num_threads(previous)


def test_all_128_affine_projections_are_equalized(generator):
    projections = [m for m in generator.modules() if isinstance(m, nn.Linear)]
    assert len(projections) == 128  # Stem + RGB + 21 * (Q/K/V/O + FFN up/down).
    assert sum(p.numel() for p in generator.parameters()) == 151_512_455
    for module in projections:
        assert module.equalized
        assert torch.isfinite(module.weight).all()
        if module.weight.numel() >= 4096:
            assert module.weight.detach().var().item() == pytest.approx(1, rel=.08)
        if module.bias is not None:
            assert module.bias.count_nonzero() == 0
    for side, channels, depth in STAGES:
        assert generator[f'stage{side}_position'].weight.shape == (side * side, channels)
        for block in range(depth):
            attention = generator[f'stage{side}_block{block}_attention']
            assert attention.heads == 4 and attention.relative_position_bias
            assert attention.q_proj.bias is None and attention.o_proj.bias is not None
            bias = attention.relative_position_bias_table
            assert bias.detach().std().item() == pytest.approx(.02, rel=.1)
            ffn = generator[f'stage{side}_block{block}_ffn']
            assert ffn.activation == 'gelu' and ffn.dropout.p == 0
    assert not any('latent_projection' in node.id or node.id.endswith('_scale')
                   for node in generator.plan.nodes)


def test_equalized_forward_latent_and_selected_weight_gradients(generator):
    flags = [p.requires_grad for p in generator.parameters()]
    generator.requires_grad_(False)
    # A late attention and both FFN matrices check nested trainable projections
    # without allocating gradients for the entire 151M-parameter model.
    selected = [generator['stage128_block3_attention'].q_proj.weight,
                generator['stage128_block3_ffn'].up.weight,
                generator['stage128_block3_ffn'].down.weight,
                generator['output_projection'].weight]
    for parameter in selected:
        parameter.requires_grad_(True)
    z = torch.arange(512, dtype=torch.float32).cos().reshape(1, 512).requires_grad_()
    rng = torch.get_rng_state().clone()
    try:
        output = generator(z)
        assert output.shape == (1, 3, 128, 128) and torch.isfinite(output).all()
        derivatives = torch.autograd.grad(output.square().mean(), (z, *selected))
        for derivative in derivatives:
            assert torch.isfinite(derivative).all() and derivative.abs().sum() > 0
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        for parameter, flag in zip(generator.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)

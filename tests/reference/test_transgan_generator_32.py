"""Small CIFAR TransGAN build, training gradients, and reconstruction reuse."""
import copy
from pathlib import Path

import pytest
import torch

from hypergan.hndl_networks import build_network
from hypergan.image_components import CIFARRoutingEncoder
from hypergan.training import update_ema


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def generator():
    source = (Path(__file__).parents[2] / 'examples/networks/transgan-generator-32.hndl').read_text()
    with torch.random.fork_rng(devices=[]):
        return build_network(source, input_shape=('B', 64), output_shape=('B', 3, 32, 32))


@pytest.mark.parametrize('batch_size', [1, 3])
def test_cifar_generator_forward_latent_and_parameter_gradients(generator, batch_size):
    generator.train()
    z = torch.arange(batch_size * 64, dtype=torch.float32).cos().reshape(batch_size, 64).requires_grad_()
    stage_shapes = {}
    def capture(side):
        def record(module, inputs, output):
            stage_shapes[side] = tuple(output.shape)
        return record
    handles = [generator[f'stage{side}_position'].register_forward_hook(capture(side)) for side in (8, 16, 32)]
    rng = torch.get_rng_state().clone()
    try:
        output = generator(z)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(rng, torch.get_rng_state())
    assert stage_shapes == {8: (batch_size, 64, 256), 16: (batch_size, 256, 64),
                            32: (batch_size, 1024, 16)}
    assert output.shape == (batch_size, 3, 32, 32)
    assert torch.isfinite(output).all() and output.min() >= -1 and output.max() <= 1
    output.square().mean().backward()
    assert torch.isfinite(z.grad).all() and (z.grad.abs().sum(1) > 0).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in generator.parameters())
    for side in (8, 16, 32):
        for block in (0, 1):
            attention = generator[f'stage{side}_block{block}_attention']
            assert attention.heads == 4 and attention.spatial_shape == (side, side)
            table = attention.relative_position_bias_table
            assert table.grad.abs().sum() > 0
    output_layer = generator['output_projection']
    before = output_layer.weight.detach().clone()
    torch.optim.SGD(generator.parameters(), lr=.001).step()
    assert not torch.equal(output_layer.weight, before)


def test_cifar_generator_deepcopy_ema_and_state_restore(generator):
    generator.eval()
    z = torch.arange(64, dtype=torch.float32).sin().reshape(1, 64)
    with torch.no_grad():
        expected = generator(z)
    rng = torch.get_rng_state().clone()
    average = copy.deepcopy(generator).eval().requires_grad_(False)
    assert torch.equal(rng, torch.get_rng_state())
    for live, copied in zip(generator.parameters(), average.parameters()):
        assert live.data_ptr() != copied.data_ptr()
        torch.testing.assert_close(live, copied, rtol=0, atol=0)
    for live, copied in zip(generator.buffers(), average.buffers()):
        assert live.dtype == copied.dtype == torch.int64
        assert live.data_ptr() != copied.data_ptr()
        torch.testing.assert_close(live, copied, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(average(z), expected, rtol=0, atol=0)
        original_bias = average['output_projection'].bias.clone()
        generator['output_projection'].bias.add_(.2)
    update_ema(average, generator, .75)
    torch.testing.assert_close(average['output_projection'].bias, original_bias + .05)
    with torch.no_grad():
        assert torch.isfinite(average(z)).all()
    average.load_state_dict(generator.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(average(z), generator(z), rtol=0, atol=0)


def test_existing_hndl_encoder_trains_through_frozen_transgan_reconstruction(generator):
    # Preserve the CIFAR recipe's reconstruction contract: the reused G is
    # frozen, while its latent derivative trains the routing encoder.
    encoder = CIFARRoutingEncoder(z_dim=64, width=32, temperature=.125)
    real = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).cos().reshape(2, 3, 32, 32)
    means = torch.arange(4 * 64, dtype=torch.float32).sin().reshape(4, 64).requires_grad_()
    generator.requires_grad_(False)
    latent = encoder(real, means, torch.tensor(.2))['latent']
    assert latent.shape == (2, 64)
    reconstructed = generator(latent)
    (reconstructed - real).square().mean().backward()
    assert encoder.query.weight.grad is not None and torch.isfinite(encoder.query.weight.grad).all()
    assert encoder.query.weight.grad.abs().sum() > 0
    assert encoder.offset.weight.grad is not None and encoder.offset.weight.grad.abs().sum() > 0
    assert means.grad is None
    assert all(parameter.grad is None for parameter in generator.parameters())

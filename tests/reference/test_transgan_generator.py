"""CPU behavior of the editable TransGAN generator and native grid attention."""
import copy
import gc
from pathlib import Path
import random

import pytest
import torch

from hypergan.hndl_networks import build_network
from hypergan.training import update_ema


SOURCE_PATH = Path(__file__).parents[2] / 'examples/networks/transgan-generator-128.hndl'


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def generator():
    # Reuse the large model; tests change only mode and restore any edited state.
    with torch.random.fork_rng(devices=[]):
        model = build_network(SOURCE_PATH.read_text(), input_shape=('B', 128), output_shape=('B', 3, 128, 128))
    yield model
    del model
    gc.collect()


def latents(count):
    return torch.arange(count * 128, dtype=torch.float32).reshape(count, 128).cos()


@pytest.mark.parametrize('batch_size', [1, 3])
def test_rgb_output_range_and_latent_gradients(generator, batch_size):
    generator.train()
    z = latents(batch_size).requires_grad_()
    output = generator(z)
    assert output.shape == (batch_size, 3, 128, 128)
    assert torch.isfinite(output).all() and output.min() >= -1 and output.max() <= 1
    gradient, = torch.autograd.grad(output.square().mean(), z)
    assert torch.isfinite(gradient).all() and (gradient.abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize('training', [True, False])
def test_forward_uses_only_supplied_latent_without_random_draws(generator, training):
    generator.train(training)
    z = latents(1)
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    with torch.no_grad():
        first, second = generator(z), generator(z)
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == python_rng
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_eval_batch_companions_do_not_change_each_sample(generator):
    generator.eval()
    z = latents(3)
    with torch.no_grad():
        together = generator(z)
        separate = torch.cat([generator(row[None]) for row in z])
        reordered = generator(z.flip(0)).flip(0)
    # Native RMS statistics use float32, including when a model is cast double.
    # Allow accumulation roundoff from kernels selected for different N.
    torch.testing.assert_close(together, separate, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(together, reordered, rtol=2e-4, atol=2e-5)


def source_section(begin, end):
    """Exercise production statements directly, without a parallel network."""
    source = SOURCE_PATH.read_text()
    assert source.count(begin) == source.count(end) == 1
    return source.split(begin, 1)[1].split('\n', 1)[1].split(end, 1)[0]


@pytest.mark.parametrize('side', [64, 128])
@pytest.mark.parametrize('batch_size', [1, 3])
def test_grid_partition_orders_windows_and_exactly_inverts_for_each_batch(side, batch_size):
    partition = source_section(f'# BEGIN stage{side} window partition:', f'# END stage{side} window partition.')
    inverse = source_section(f'# BEGIN stage{side} window inverse:', f'# END stage{side} window inverse.')
    model = build_network('h = x\n' + partition + inverse,
                          input_shape=('B', 2, side, side), output_shape=('B', 2, side, side))
    captured = []
    hook = model[f'stage{side}_windows'].register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().clone()))
    pixels = torch.arange(batch_size * 2 * side * side, dtype=torch.float32).reshape(batch_size, 2, side, side)
    pixels.requires_grad_()
    try:
        output = model(pixels)
    finally:
        hook.remove()
    # Columns are stacked after rows: column, then row, then original batch.
    expected_windows = torch.cat([pixels[:, :, row:row + 16, column:column + 16]
                                  for column in range(0, side, 16) for row in range(0, side, 16)])
    torch.testing.assert_close(captured[0], expected_windows, rtol=0, atol=0)
    torch.testing.assert_close(output, pixels, rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(pixels.grad, torch.ones_like(pixels), rtol=0, atol=0)


@pytest.mark.parametrize('side,channels', [(64, 64), (128, 16)])
def test_grid_blocks_share_weights_preserve_locality_and_use_pixelnorm(side, channels):
    section = source_section(f'# BEGIN grid{side}\n', f'# END grid{side}\n')
    model = build_network('h = x\n' + section, input_shape=('B', channels, side, side),
                          output_shape=('B', channels, side, side)).eval()
    captured, calls = {}, []
    def attention_call(module, inputs):
        calls.append((id(module), tuple(inputs[0].shape)))
    def norm_call(module, inputs, output):
        captured['norm'] = (inputs[0].detach().clone(), output.detach().clone())
    handles = [model[f'stage{side}_block{block}_attention'].register_forward_pre_hook(attention_call)
               for block in (0, 1)]
    handles.append(model[f'stage{side}_block0_norm1'].register_forward_hook(norm_call))
    tile = torch.arange(channels * 16 * 16, dtype=torch.float32).cos().reshape(1, channels, 16, 16)
    pixels = tile.repeat(1, 1, side // 16, side // 16)
    try:
        with torch.no_grad():
            output = model(pixels)
    finally:
        for handle in handles:
            handle.remove()
    windows = (side // 16) ** 2
    assert len(calls) == 2 and len({identity for identity, shape in calls}) == 2
    assert all(shape == (windows, 256, channels) for identity, shape in calls)
    # The same two blocks process every window in one batched invocation.
    for row in range(0, side, 16):
        for column in range(0, side, 16):
            torch.testing.assert_close(output[:, :, row:row + 16, column:column + 16], output[:, :, :16, :16], rtol=0, atol=0)
    norm_input, norm_output = captured['norm']
    expected = norm_input * torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + 1e-8)
    torch.testing.assert_close(norm_output, expected, rtol=1e-6, atol=1e-7)
    assert list(model[f'stage{side}_block0_norm1'].parameters()) == []
    perturbed = pixels.clone()
    perturbed[:, :, 0, 0] += .5
    with torch.no_grad():
        changed = model(perturbed)
    assert not torch.equal(changed[:, :, :16, :16], output[:, :, :16, :16])
    torch.testing.assert_close(changed[:, :, 16:, :], output[:, :, 16:, :], rtol=0, atol=0)
    torch.testing.assert_close(changed[:, :, :16, 16:], output[:, :, :16, 16:], rtol=0, atol=0)


def test_deepcopy_state_restoration_and_ema_preserve_independent_weights(generator):
    generator.eval()
    z = latents(1)
    with torch.no_grad():
        before = generator(z)
    rng = torch.get_rng_state().clone()
    average = copy.deepcopy(generator).eval().requires_grad_(False)
    assert torch.equal(rng, torch.get_rng_state())
    for live, copied in zip(generator.parameters(), average.parameters()):
        assert live.data_ptr() != copied.data_ptr()
        torch.testing.assert_close(live, copied, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(average(z), before, rtol=0, atol=0)
    live, copied = next(generator.parameters()), next(average.parameters())
    original = live.flatten()[0].item()
    try:
        with torch.no_grad():
            live.flatten()[0].add_(.25)
        assert copied.flatten()[0].item() == original
        update_ema(average, generator, .75)
        assert copied.flatten()[0].item() == pytest.approx(original + .0625, abs=1e-7)
        with torch.no_grad():
            assert torch.isfinite(average(z)).all()
    finally:
        with torch.no_grad():
            live.flatten()[0].fill_(original)
    average.load_state_dict(generator.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(average(z), before, rtol=0, atol=0)
    del average
    gc.collect()

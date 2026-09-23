"""CPU behavior of the editable TransGAN generator and native grid attention."""
import copy
import gc
import math
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


def test_all_stage_norms_use_parameter_free_channel_scaling_with_small_epsilon(generator):
    # Tiny and zero tokens expose a wrong epsilon that ordinary activations
    # conceal. Nonzero channel means also distinguish this from LayerNorm.
    for side, channels in ((8, 1024), (16, 1024), (32, 256), (64, 64), (128, 16)):
        values = torch.linspace(.5, 1.5, channels).reshape(1, 1, channels)
        scales = torch.tensor([0., 1e-6, 1e-4, 1.]).reshape(1, 4, 1)
        x = (values * scales).repeat(2, 1, 1).requires_grad_()
        expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-8)
        expected_gradient, = torch.autograd.grad(expected.sum(), x)
        for block in (0, 1):
            for branch in (1, 2):
                norm = generator[f'stage{side}_block{block}_norm{branch}']
                assert list(norm.parameters()) == []
                actual = norm(x)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                gradient, = torch.autograd.grad(actual.sum(), x)
                torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=1e-6)
                assert torch.isfinite(gradient).all()


@pytest.mark.parametrize('stage,block,grid', [(8, 0, 8), (64, 0, 16), (128, 1, 16)])
def test_relative_position_attention_matches_2d_oracle_and_learns(stage, block, grid):
    # Reuse an actual configured attention operation at a small channel width.
    # Native global/grid geometry, projection bias flags and init stay intact.
    full_source = SOURCE_PATH.read_text()
    name = f'stage{stage}_block{block}_attention'
    start = full_source.rindex('branch = attention(', 0, full_source.index(f'name="{name}"'))
    end = full_source.index('\nh = add(', start)
    source = 'branch = x\n' + full_source[start:end]
    tokens = grid * grid
    model = build_network(source, input_shape=('B', tokens, 8), output_shape=('B', tokens, 8)).double().eval()
    attention = model[name]
    table = attention.relative_position_bias_table
    expected_index = torch.tensor([
        [(query_row - key_row + grid - 1) * (2 * grid - 1) + query_column - key_column + grid - 1
         for key_row in range(grid) for key_column in range(grid)]
        for query_row in range(grid) for query_column in range(grid)], dtype=torch.int64)
    torch.testing.assert_close(attention.relative_position_index, expected_index, rtol=0, atol=0)
    assert attention.relative_position_index.dtype == torch.int64
    assert attention.q_proj.bias is attention.k_proj.bias is attention.v_proj.bias is None
    assert attention.o_proj.bias is not None
    # Deliberately asymmetric offsets expose swapped axes or reversed q/k.
    with torch.no_grad():
        table.copy_(torch.arange(table.numel(), dtype=torch.float64).sin().reshape_as(table) * .1)
    x = torch.arange(2 * tokens * 8, dtype=torch.float64).cos().reshape(2, tokens, 8)
    q, k, v = [projection(x).reshape(2, tokens, 4, 2).transpose(1, 2)
               for projection in (attention.q_proj, attention.k_proj, attention.v_proj)]
    bias = table[expected_index].permute(2, 0, 1)
    probabilities = (q @ k.transpose(-2, -1) / math.sqrt(2) + bias).softmax(-1)
    expected = attention.o_proj((probabilities @ v).transpose(1, 2).reshape(2, tokens, 8))
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    before = table.detach().clone()
    actual.square().mean().backward()
    assert table.grad is not None and torch.isfinite(table.grad).all() and table.grad.abs().sum() > 0
    torch.optim.Adam([table], lr=.01).step()
    assert not torch.equal(table, before)
    with torch.no_grad():
        updated = model(x)
        assert torch.isfinite(updated).all() and not torch.equal(updated, actual)
    restored = build_network(source, input_shape=('B', tokens, 8), output_shape=('B', tokens, 8)).double().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored[name].relative_position_index, expected_index, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), updated, rtol=0, atol=0)


def test_configured_projection_biases_and_position_initialization(generator):
    for side in (8, 16, 32, 64, 128):
        grid = min(side, 32) if side < 64 else 16
        absolute = generator[f'stage{side}_position'].weight
        assert torch.isfinite(absolute).all() and absolute.abs().max() <= 2
        assert absolute.count_nonzero() > 0
        for block in (0, 1):
            attention = generator[f'stage{side}_block{block}_attention']
            assert attention.spatial_shape == (grid, grid)
            assert attention.relative_position_bias
            table = attention.relative_position_bias_table
            assert table.shape == ((2 * grid - 1) ** 2, 4)
            assert torch.isfinite(table).all() and table.abs().max() <= 2
            assert table.count_nonzero() > 0
            assert attention.relative_position_index.dtype == torch.int64
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                assert projection.bias is None
            assert attention.o_proj.bias is not None
    # Official Linear layers retain PyTorch's fan-in initialization; only the
    # convolution-equivalent RGB readout uses Xavier weights. Biases retain
    # their fan-in uniform initialization, including the RGB bias.
    for module in generator.modules():
        if isinstance(module, torch.nn.Linear):
            bias_bound = 1 / math.sqrt(module.in_features)
            bound = (math.sqrt(6 / (module.in_features + module.out_features))
                     if module is generator['output_projection'] else bias_bound)
            assert torch.isfinite(module.weight).all() and module.weight.abs().max() <= bound + 1e-7
            assert module.weight.count_nonzero() > 0
            if module.bias is not None:
                assert torch.isfinite(module.bias).all() and module.bias.abs().max() <= bias_bound + 1e-7
                assert module.bias.count_nonzero() > 0


def test_stem_and_first_residual_branches_keep_fan_in_activation_scale(generator):
    """Catch a tiny Xavier stem feeding much larger residual branches.

    This checks the initialization mechanism behind saturation, without
    asserting that untrained image diversity predicts eventual training quality.
    """
    generator.eval()
    stem = generator['input_projection']
    expected_variance = 1 / (3 * stem.in_features)
    # Over eight million entries make this analytic distribution check stable;
    # the previous fan-in+fan-out initializer had about 1/85 of this variance.
    assert stem.weight.detach().var(unbiased=False).item() == pytest.approx(expected_variance, rel=.02)
    captured = {}
    def capture(name):
        def record(module, inputs, output):
            captured[name] = output.detach().square().mean().sqrt().item()
        return record
    names = ('input_projection', 'stage8_block0_attention', 'stage8_block0_ffn')
    handles = [generator[name].register_forward_hook(capture(name)) for name in names]
    z = latents(3)
    try:
        with torch.no_grad():
            generator(z)
    finally:
        for handle in handles:
            handle.remove()
    # Independent uniform weights and bias imply E[(Wz+b)^2] =
    # (||z||^2 + 1)/(3*fan_in), averaged over the fixed supplied examples.
    expected_rms = math.sqrt((z.square().sum(1).mean().item() + 1) * expected_variance)
    assert captured['input_projection'] == pytest.approx(expected_rms, rel=.1)
    for name in names[1:]:
        assert 0 < captured[name] < 2 * captured['input_projection']


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
    for live, copied in zip(generator.buffers(), average.buffers()):
        assert live.dtype == copied.dtype and live.data_ptr() != copied.data_ptr()
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
        for name, buffer in average.named_buffers():
            torch.testing.assert_close(buffer, dict(generator.named_buffers())[name], rtol=0, atol=0)
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

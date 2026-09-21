"""Behavioral coverage for the editable 128px SAGAN/AdaIN generator."""
import copy
from pathlib import Path
import random

import pytest
import torch

from hypergan.hndl_networks import build_network
from hypergan.training import update_ema


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def generator():
    source = (Path(__file__).parents[2] / 'examples/networks/sagan-adain-generator-128.hndl').read_text()
    with torch.random.fork_rng(devices=[]):
        return build_network(source, input_shape=('B', 128), output_shape=('B', 3, 128, 128))


def latents(count):
    return torch.arange(count * 128, dtype=torch.float32).reshape(count, 128).cos()


@pytest.mark.parametrize('batch_size', [1, 3])
def test_rgb_output_range_and_both_latent_halves_receive_gradients(generator, batch_size):
    generator.eval()
    z = latents(batch_size).requires_grad_()
    output = generator(z)
    assert output.shape == (batch_size, 3, 128, 128)
    assert torch.isfinite(output).all() and output.min() >= -1 and output.max() <= 1
    gradient, = torch.autograd.grad(output.square().mean(), z)
    assert torch.isfinite(gradient).all()
    assert (gradient[:, :64].abs().sum(dim=1) > 0).all()
    assert (gradient[:, 64:].abs().sum(dim=1) > 0).all()


def test_style_only_changes_affines_and_output_with_population_adain_oracle(generator):
    generator.eval()
    z = latents(1).expand(2, -1).clone()
    z[1, 64:] = -z[0, 64:]
    captured = {}
    def record(name):
        def hook(module, inputs, output):
            captured[name] = (tuple(value.detach().clone() for value in inputs), output.detach().clone())
        return hook
    style_names = [f'block{block}_style{index}' for block in range(5) for index in (1, 2)] + ['output_style']
    handles = [generator[name].register_forward_hook(record(name))
               for name in ['input_projection', 'block0_norm1', *style_names]]
    try:
        with torch.no_grad():
            output = generator(z)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(captured['input_projection'][1][0], captured['input_projection'][1][1], rtol=0, atol=0)
    style_inputs, styles = captured['block0_style1']
    torch.testing.assert_close(style_inputs[0], z[:, 64:], rtol=0, atol=0)
    for name in style_names:
        torch.testing.assert_close(captured[name][0][0], z[:, 64:], rtol=0, atol=0)
    assert not torch.equal(styles[0], styles[1])
    (features, parameters), actual = captured['block0_norm1']
    torch.testing.assert_close(parameters, styles, rtol=0, atol=0)
    gamma, beta = parameters.chunk(2, dim=1)
    normalized = (features - features.mean(dim=(-2, -1), keepdim=True)) / torch.sqrt(
        features.var(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-5)
    expected = (1 + gamma[:, :, None, None]) * normalized + beta[:, :, None, None]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert (output[0] - output[1]).abs().mean() > 1e-5


def test_eval_samples_do_not_depend_on_batch_companions(generator):
    # Double precision isolates batch coupling from accumulated float32
    # roundoff in the different convolution kernels used for N=1 and N=3.
    generator.double().eval()
    z = latents(3).double()
    with torch.no_grad():
        together = generator(z)
        alone = torch.cat([generator(row[None]) for row in z])
        reversed_batch = generator(z.flip(0)).flip(0)
    torch.testing.assert_close(together, alone, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(together, reversed_batch, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('training', [False, True])
def test_forward_consumes_no_randomness_in_either_mode(generator, training):
    generator.train(training)
    z = latents(1)
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    with torch.no_grad():
        output = generator(z)
    assert torch.isfinite(output).all()
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()


def test_attention_gate_learns_before_attention_weights_and_opens_branch(generator):
    generator.eval()  # Keep spectral power-iteration buffers fixed for this comparison.
    gamma = generator['attention_gain'].gamma
    assert gamma.shape == (1,) and gamma.item() == 0
    attention_parameters = [generator[name].parametrizations.weight.original
                            for name in ('attention_query', 'attention_key', 'attention_value', 'attention_project')]
    captured = {}
    def record(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().clone()
        return hook
    handles = [generator[name].register_forward_hook(record(name))
               for name in ('block2_out', 'attention_out', 'attention_probabilities')]
    z = latents(1)
    try:
        before = generator(z)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(captured['attention_out'], captured['block2_out'], rtol=0, atol=0)
    assert captured['attention_probabilities'].shape == (1, 1024, 256)
    torch.testing.assert_close(captured['attention_probabilities'].sum(-1), torch.ones(1, 1024))
    before.square().mean().backward()
    assert gamma.grad is not None and torch.isfinite(gamma.grad).all() and gamma.grad.abs().sum() > 0
    assert all(parameter.grad is not None and parameter.grad.count_nonzero() == 0 for parameter in attention_parameters)
    torch.optim.Adam([gamma], lr=.01).step()
    assert gamma.item() != 0
    generator.zero_grad(set_to_none=True)
    after = generator(z)
    assert not torch.equal(before, after)
    after.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in attention_parameters)


def test_spectral_generator_deepcopy_and_ema_are_independent_after_forward(generator):
    generator.train()
    z = latents(1)
    generator(z).square().mean().backward()
    generator.eval()
    rng = torch.get_rng_state().clone()
    average = copy.deepcopy(generator).eval().requires_grad_(False)
    assert torch.equal(rng, torch.get_rng_state())
    original_state = generator.state_dict()
    for name, value in average.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        assert value.data_ptr() != original_state[name].data_ptr()
    with torch.no_grad():
        torch.testing.assert_close(average(z), generator(z), rtol=0, atol=0)
    torch.optim.SGD(generator.parameters(), lr=.001).step()
    with torch.no_grad():
        generator['attention_gain'].gamma.add_(.2)
    assert average['attention_gain'].gamma.item() == 0
    expected_gamma = generator['attention_gain'].gamma.item() * .25
    update_ema(average, generator, .75)
    assert average['attention_gain'].gamma.item() == pytest.approx(expected_gamma)
    for name, buffer in average.named_buffers():
        torch.testing.assert_close(buffer, dict(generator.named_buffers())[name], rtol=0, atol=0)
    with torch.no_grad():
        assert torch.isfinite(average(z)).all()

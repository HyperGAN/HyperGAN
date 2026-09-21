"""Fused nonfinite refusal and fused EMA: same contract, one host read per phase.

The trainer previously read every parameter's gradient back to the host, one
blocking transfer per tensor. These checks pin the behaviour that replaces it:
the same distinct messages in the same order, an optimizer step still refused
with parameters and optimizer state untouched, gradients left bitwise unchanged
by the screening kernel, and an EMA update bitwise identical to the per-tensor
loop it replaces. The device-level synchronization count is qualified separately
in tests/cuda/test_trainer_host_syncs.py.
"""
import copy

import pytest
import torch
from tests.hndl_fixtures import fixture_linear, fixture_norm, fixture_network

from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer, update_ema


class Encoder(torch.nn.Module):
    """A small auxiliary component so the generator phase owns more than one model."""
    def __init__(self):
        super().__init__()
        self.network = fixture_network('prior_projection',
            {'x': (2,), 'prior_mean': (4,), 'sigma': (4,)}, (4,))

    @property
    def linear(self):
        return self.network['projection']

    def forward(self, x, means, sigma):
        return self.network(x=x, prior_mean=means.detach()[0].expand(len(x), 4),
                            sigma=sigma.expand(len(x), 4))


def config():
    """Generator, discriminator, auxiliary encoder, objective and trainable prior."""
    return resolve_config({'components': {
        'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
        'discriminator': {'factory': 'mlp', 'args': {'input_dim': 2, 'output_dim': 1, 'hidden': [8]}, 'inputs': {'x': 'candidate'}},
        'encoder': {'factory': f'{__name__}:Encoder', 'inputs': {'x': 'batch.real', 'means': 'prior.means', 'sigma': 'prior.sigma'}},
        'reconstruction': {'reuse': 'generator', 'freeze_parameters': True, 'inputs': {'x': 'components.encoder'}}},
        'objectives': [{'factory': 'mse', 'inputs': {'input': 'components.reconstruction', 'target': 'batch.real'}}],
        'prior': {'kind': 'mog', 'args': {'num_particles': 16, 'z_dim': 4}, 'initialization_device': 'cpu', 'initialization_seed': 43, 'fixed_sigma': .2},
        'prior_regularizer': {'rows': 'full'},
        'optimizer': {'implementation': 'torch_fused_adam', 'prior_betas': [.5, .999]},
        'gradient_penalty': {'lazy_k': 2, 'kappa': .001},
        'training': {'steps': 4, 'batch_size': 8, 'phase_draws': 'independent', 'data_rng_device': 'execution',
                     'data_seed_offset': 2, 'prior_seed_offset': 3, 'lr_floor': 1.}})


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def previous_update_ema(average, current, decay):
    """The per-tensor loop this change replaces, kept as the numerical oracle."""
    with torch.no_grad():
        for target, source in zip(average.parameters(), current.parameters()):
            target.lerp_(source, 1.0 - decay)
        for target, source in zip(average.buffers(), current.buffers()):
            target.copy_(source)


def bits(tensor):
    """Raw bytes, so NaN, signed zero and subnormals compare exactly."""
    return tensor.detach().cpu().contiguous().flatten().view(torch.uint8)


def assert_bitwise(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(bits(actual), bits(expected))


def comparable(value):
    """Plain comparable values, including exact tensor bytes."""
    if isinstance(value, torch.Tensor):
        return str(value.dtype), tuple(value.shape), tuple(bits(value).tolist())
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    return value


def assert_state_dicts_bitwise(actual, expected):
    left, right = actual.state_dict(), expected.state_dict()
    assert left.keys() == right.keys()
    for name in left:
        assert comparable(left[name]) == comparable(right[name]), name


class Mixed(torch.nn.Module):
    """Parameter and buffer inventory that forces several foreach groups."""
    def __init__(self):
        super().__init__()
        self.linear = fixture_linear(7, 5)
        self.norm = fixture_norm(3, image=True)  # float buffers plus int64 num_batches_tracked
        self.wide = torch.nn.Parameter(torch.randn(129, 17, dtype=torch.float64))
        self.narrow = torch.nn.Parameter(torch.randn(4, 4).half())
        self.register_buffer('counter', torch.tensor([3], dtype=torch.int64))
        self.register_buffer('scale', torch.randn(2, 2, dtype=torch.float64))


def diverge(module, generator):
    """Give the online module different values so the EMA actually moves."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator).to(parameter.dtype))
        for buffer in module.buffers():
            buffer.add_(7 if buffer.dtype == torch.int64 else 0.25)
    return module


@pytest.mark.parametrize('decay', [0.995, 0.0, 1.0, 0.5])
def test_update_ema_is_bitwise_identical_to_the_previous_per_tensor_loop(decay):
    torch.manual_seed(11)
    online = diverge(Mixed(), torch.Generator().manual_seed(12))
    average = Mixed()
    expected, source = copy.deepcopy(average), comparable(online.state_dict())
    previous_update_ema(expected, online, decay)
    update_ema(average, online, decay)
    assert_state_dicts_bitwise(average, expected)
    assert comparable(online.state_dict()) == source  # the online module is only read


def test_update_ema_matches_the_previous_loop_over_a_trained_graph_and_prior():
    trainer = ReferenceTrainer(config())
    for _ in range(2):
        trainer.update()
    decay = trainer.config['training']['ema']
    for average, current in ((trainer.ema_graph, trainer.graph), (trainer.ema_prior, trainer.prior)):
        expected = copy.deepcopy(average)
        previous_update_ema(expected, current, decay)
        update_ema(average, current, decay)
        assert_state_dicts_bitwise(average, expected)


def test_update_ema_accepts_empty_parameter_and_buffer_inventories():
    empty, other = torch.nn.Module(), torch.nn.Module()
    update_ema(empty, other, 0.9)
    assert list(empty.parameters()) == [] and list(empty.buffers()) == []


def test_gradient_screening_leaves_gradients_bitwise_unchanged_and_flags_each_group():
    trainer = ReferenceTrainer(config())
    gradients = [torch.tensor([-0.0, 0.0, 1.4e-45, 5.9e-39, 3.4e38, -1.7e38]),
                 torch.randn(5, 3), torch.randn(4, dtype=torch.float64),
                 torch.randn(3, dtype=torch.complex64)]
    parameters = [torch.nn.Parameter(torch.zeros(gradient.shape, dtype=gradient.dtype)) for gradient in gradients]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    parameters.append(torch.nn.Parameter(torch.zeros(2)))  # grad is None: contributes no flag
    flags = trainer._gradient_flags(parameters)
    # The complex gradient the screening kernel does not accept keeps the
    # elementwise reduction; the float32 and float64 gradients are one group each.
    assert len(flags) == 3 and [bool(flag) for flag in flags] == [False, False, False]
    for parameter, gradient in zip(parameters, gradients):
        assert_bitwise(parameter.grad, gradient)
    for index, expected in ((3, [True, False, False]), (1, [False, True, False]), (2, [False, False, True])):
        original = parameters[index].grad.clone()
        parameters[index].grad.view(-1)[1] = float('nan') if index != 2 else float('-inf')
        assert [bool(flag) for flag in trainer._gradient_flags(parameters)] == expected
        parameters[index].grad = original
    assert [bool(flag) for flag in trainer._gradient_flags(parameters)] == [False, False, False]
    assert trainer._gradient_flags([]) == []


def untouched(trainer, phase):
    """State that a refused optimizer step must leave exactly as it was."""
    state = {'prior': comparable(trainer.prior.state_dict()), 'step': trainer.step,
             'ema_graph': comparable(trainer.ema_graph.state_dict()),
             'ema_prior': comparable(trainer.ema_prior.state_dict()),
             'opt_g': comparable(trainer.opt_g.state_dict())}
    if phase == 'discriminator':
        state['graph'] = comparable(trainer.graph.state_dict())
        state['opt_d'] = comparable(trainer.opt_d.state_dict())
    else:
        # The discriminator phase legitimately completed before the generator failure.
        state['graph'] = comparable({name: module.state_dict() for name, module
                                     in trainer.graph.models.items() if name != 'discriminator'})
    return state


def nan_gradient(parameter):
    parameter.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))


def wrap(owner, name, transform):
    original = getattr(owner, name)
    setattr(owner, name, lambda *args, **kwargs: transform(original(*args, **kwargs)))


@pytest.mark.parametrize('broken,phase,message', [
    ('d_loss', 'discriminator', 'Nonfinite discriminator loss; run stopped'),
    ('d_loss_and_gradient', 'discriminator', 'Nonfinite discriminator loss; run stopped'),
    ('d_gradient', 'discriminator', 'Nonfinite discriminator gradient; run stopped'),
    ('g_loss', 'generator', 'Nonfinite generator loss; run stopped'),
    ('g_loss_and_gradient', 'generator', 'Nonfinite generator loss; run stopped'),
    ('g_gradient', 'generator', 'Nonfinite generator/auxiliary gradient; run stopped'),
    ('encoder_gradient', 'generator', 'Nonfinite generator/auxiliary gradient; run stopped'),
    ('prior_gradient', 'generator', 'Nonfinite prior gradient; run stopped'),
])
def test_nonfinite_loss_and_gradients_refuse_the_step_with_their_own_message(broken, phase, message):
    trainer = ReferenceTrainer(config())
    if broken == 'd_loss':  # a nonfinite loss whose gradients stay finite
        wrap(trainer.gan, 'd_loss', lambda value: value + float('inf'))
    elif broken == 'd_loss_and_gradient':  # both nonfinite: the loss is reported
        wrap(trainer.gan, 'd_loss', lambda value: value * float('nan'))
    elif broken == 'g_loss':
        wrap(trainer.gan, 'g_loss', lambda value: value + float('inf'))
    elif broken == 'g_loss_and_gradient':
        wrap(trainer.gan, 'g_loss', lambda value: value * float('nan'))
    elif broken == 'd_gradient':
        nan_gradient(trainer.graph.models['discriminator'].network[0].weight)
    elif broken == 'g_gradient':
        nan_gradient(trainer.graph.models['generator'].network[2].bias)
    elif broken == 'encoder_gradient':
        nan_gradient(trainer.graph.models['encoder'].linear.bias)
    elif broken == 'prior_gradient':
        nan_gradient(trainer.prior.z)
    before = untouched(trainer, phase)
    discriminator = comparable(trainer.graph.models['discriminator'].state_dict())
    with pytest.raises(ValueError, match=message):
        trainer.update()
    assert untouched(trainer, phase) == before
    assert all(parameter.requires_grad for parameter in trainer.graph.models['discriminator'].parameters())
    if phase == 'generator':  # the failure was reached, not skipped before the D step
        assert comparable(trainer.graph.models['discriminator'].state_dict()) != discriminator


def test_a_finite_step_still_updates_every_module():
    trainer = ReferenceTrainer(config())
    before = untouched(trainer, 'discriminator')
    row, _ = trainer.update()
    assert trainer.step == 1 and untouched(trainer, 'discriminator') != before
    assert row['d_loss'] == row['d_loss'] and row['g_loss'] == row['g_loss']


def test_each_phase_reads_the_host_once_instead_of_once_per_parameter():
    trainer = ReferenceTrainer(config())
    counts = {'isfinite': 0, 'screen': 0, 'read': 0}
    isfinite, screen, tolist = torch.isfinite, torch._amp_foreach_non_finite_check_and_unscale_, torch.Tensor.tolist

    def counted_isfinite(*args, **kwargs):
        counts['isfinite'] += 1
        return isfinite(*args, **kwargs)

    def counted_screen(*args, **kwargs):
        counts['screen'] += 1
        return screen(*args, **kwargs)

    def counted_tolist(self, *args, **kwargs):
        counts['read'] += 1
        return tolist(self, *args, **kwargs)

    assert len(list(trainer.graph.parameters())) + len(list(trainer.prior.parameters())) > 8
    torch.isfinite, torch._amp_foreach_non_finite_check_and_unscale_ = counted_isfinite, counted_screen
    torch.Tensor.tolist = counted_tolist
    try:
        trainer.update()
    finally:
        torch.isfinite, torch._amp_foreach_non_finite_check_and_unscale_ = isfinite, screen
        torch.Tensor.tolist = tolist
    # One loss reduction and one stacked host read per phase, and one screening
    # call per gradient group rather than one host read per parameter tensor.
    assert counts['isfinite'] == 2
    assert counts['read'] == 2
    assert counts['screen'] == 3


def test_default_recipe_still_trains():
    trainer = ReferenceTrainer(resolve_config({}))
    rows = [trainer.update()[0] for _ in range(3)]
    assert [row['step'] for row in rows] == [1, 2, 3]
    assert all(row['d_loss'] == row['d_loss'] for row in rows)

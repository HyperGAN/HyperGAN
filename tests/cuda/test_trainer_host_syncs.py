"""Explicit single-GPU qualification of the fused nonfinite check and EMA.

CPU CI does not run this. It pins the device-visible half of the contract: one
host read per phase instead of one per parameter gradient, gradients left
bitwise unchanged by the screening kernel, a CUDA EMA update bitwise identical
to the per-tensor loop it replaces, and a refused optimizer step on a nonfinite
CUDA gradient. Message coverage for every phase lives in
tests/reference/test_nonfinite_refusal_and_ema_fusion.py.
"""
import copy

import pytest
import torch

from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer, update_ema


_METRIC_SCALARS = ('d_loss', 'd_adversarial', 'd_adversarial_weighted', 'g_adversarial_weighted',
                   'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty')


def config():
    return resolve_config({'training': {'device': 'cuda', 'data_rng_device': 'cpu', 'steps': 8}})


def bits(tensor):
    return tensor.detach().cpu().contiguous().flatten().view(torch.uint8)


def comparable(value):
    if isinstance(value, torch.Tensor):
        return str(value.dtype), tuple(value.shape), tuple(bits(value).tolist())
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    return value


def previous_update_ema(average, current, decay):
    """The per-tensor loop this change replaces, kept as the numerical oracle."""
    with torch.no_grad():
        for target, source in zip(average.parameters(), current.parameters()):
            target.lerp_(source, 1.0 - decay)
        for target, source in zip(average.buffers(), current.buffers()):
            target.copy_(source)


class Mixed(torch.nn.Module):
    """Parameter and buffer inventory that forces several foreach groups."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 5)
        self.norm = torch.nn.BatchNorm2d(3)  # float buffers plus int64 num_batches_tracked
        self.wide = torch.nn.Parameter(torch.randn(129, 17, dtype=torch.float64))
        self.narrow = torch.nn.Parameter(torch.randn(4, 4).half())
        self.register_buffer('counter', torch.tensor([3], dtype=torch.int64))
        self.register_buffer('scale', torch.randn(2, 2, dtype=torch.float64))


def test_one_update_reads_the_host_once_per_phase_not_once_per_parameter():
    assert torch.cuda.is_available()
    trainer = ReferenceTrainer(config())
    assert trainer.device.type == 'cuda'
    parameters = [p for p in trainer.graph.parameters()] + [p for p in trainer.prior.parameters()]
    assert len(parameters) > 4
    for _ in range(2):
        trainer.update()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        row, _ = trainer.update()
    events = {event.key: event.count for event in profile.key_averages()}
    # One stacked flag read per phase; the metric fence keeps its pinned copies.
    assert events.get('Memcpy DtoH (Device -> Pageable)') == 2
    assert events.get('aten::stack') == 2
    assert events.get('aten::isfinite') == 2  # one loss reduction per phase
    assert events.get('aten::_amp_foreach_non_finite_check_and_unscale_') == 3
    # The per-parameter checks this replaces read each gradient back through an
    # unpinned device transfer, so that count scaled with the parameter
    # inventory; two remain. The metric fence stages its own scalars through
    # pinned buffers and is qualified in test_metric_scalar_transfer.py.
    assert len(row['objectives']) == 0
    assert events.get('Memcpy DtoH (Device -> Pinned)') == len(_METRIC_SCALARS) + 1
    # Fused EMA: one lerp launch for this single-device float32 graph rather
    # than one per tensor. (The foreach Adam also issues its own lerps, so the
    # EMA is profiled on its own.)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as ema:
        update_ema(trainer.ema_graph, trainer.graph, trainer.config['training']['ema'])
    ema_events = {event.key: event.count for event in ema.key_averages()}
    assert ema_events.get('aten::_foreach_lerp_') == 1
    assert ema_events.get('aten::lerp_', 0) == 0
    assert ema_events.get('aten::copy_', 0) == 0
    assert len(list(trainer.graph.parameters())) > 1


def test_gradient_screening_leaves_cuda_gradients_bitwise_unchanged():
    assert torch.cuda.is_available()
    trainer = ReferenceTrainer(config())
    device = trainer.device
    gradients = [torch.tensor([-0.0, 0.0, 1.4e-45, 5.9e-39, 3.4e38, -1.7e38], device=device),
                 torch.randn(5, 3, device=device), torch.randn(4, dtype=torch.float64, device=device)]
    parameters = [torch.nn.Parameter(torch.zeros(g.shape, dtype=g.dtype, device=device)) for g in gradients]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    expected = [comparable(gradient) for gradient in gradients]
    assert [bool(flag) for flag in trainer._gradient_flags(parameters)] == [False, False]
    assert [comparable(parameter.grad) for parameter in parameters] == expected
    parameters[0].grad[0] = float('inf')
    assert [bool(flag) for flag in trainer._gradient_flags(parameters)] == [True, False]
    assert all(flag.device == device for flag in trainer._gradient_flags(parameters))


@pytest.mark.parametrize('decay', [0.995, 0.0, 1.0])
def test_update_ema_on_cuda_is_bitwise_identical_to_the_previous_loop(decay):
    assert torch.cuda.is_available()
    torch.manual_seed(5)
    online, average = Mixed().cuda(), Mixed().cuda()
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(torch.randn(parameter.shape, device='cuda').to(parameter.dtype))
        for buffer in online.buffers():
            buffer.add_(7 if buffer.dtype == torch.int64 else 0.25)
    expected, source = copy.deepcopy(average), comparable(online.state_dict())
    previous_update_ema(expected, online, decay)
    update_ema(average, online, decay)
    assert comparable(average.state_dict()) == comparable(expected.state_dict())
    assert comparable(online.state_dict()) == source


def test_update_ema_over_a_trained_cuda_graph_matches_the_previous_loop():
    assert torch.cuda.is_available()
    trainer = ReferenceTrainer(config())
    trainer.update()
    decay = trainer.config['training']['ema']
    for average, current in ((trainer.ema_graph, trainer.graph), (trainer.ema_prior, trainer.prior)):
        expected = copy.deepcopy(average)
        previous_update_ema(expected, current, decay)
        update_ema(average, current, decay)
        assert comparable(average.state_dict()) == comparable(expected.state_dict())


@pytest.mark.parametrize('broken,message', [
    ('d_gradient', 'Nonfinite discriminator gradient; run stopped'),
    ('g_gradient', 'Nonfinite generator/auxiliary gradient; run stopped'),
    ('prior_gradient', 'Nonfinite prior gradient; run stopped'),
    ('d_loss', 'Nonfinite discriminator loss; run stopped'),
])
def test_nonfinite_cuda_loss_and_gradients_refuse_the_optimizer_step(broken, message):
    assert torch.cuda.is_available()
    trainer = ReferenceTrainer(config())
    if broken == 'd_loss':
        original = trainer.gan.d_loss
        trainer.gan.d_loss = lambda *args, **kwargs: original(*args, **kwargs) + float('inf')
    else:
        owner = {'d_gradient': trainer.graph.models['discriminator'].network[0].weight,
                 'g_gradient': trainer.graph.models['generator'].network[0].weight,
                 'prior_gradient': trainer.prior.z}[broken]
        owner.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
    before = (comparable(trainer.prior.state_dict()), comparable(trainer.opt_g.state_dict()), trainer.step)
    discriminator = comparable(trainer.graph.models['discriminator'].state_dict())
    generator = comparable(trainer.graph.models['generator'].state_dict())
    with pytest.raises(ValueError, match=message):
        trainer.update()
    assert (comparable(trainer.prior.state_dict()), comparable(trainer.opt_g.state_dict()), trainer.step) == before
    assert comparable(trainer.graph.models['generator'].state_dict()) == generator
    if broken.startswith('d_'):
        assert comparable(trainer.graph.models['discriminator'].state_dict()) == discriminator
        assert not trainer.opt_d.state_dict()['state']
    else:  # the discriminator phase legitimately completed first
        assert comparable(trainer.graph.models['discriminator'].state_dict()) != discriminator
    assert all(parameter.requires_grad for parameter in trainer.graph.models['discriminator'].parameters())

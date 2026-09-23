"""Distribution, RNG replay, and gradient contracts for TransGAN DiffAugment."""

import pytest
import torch

from hypergan.diff_augment import DiffAugment, diff_augment


def test_color_matches_reference_distribution_and_keeps_input(monkeypatch):
    draws = iter((0.75, 0.25, 0.75))
    monkeypatch.setattr(torch, 'rand', lambda shape, **kwargs: torch.full(shape, next(draws), **kwargs))
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2) / 10
    before = x.clone()
    actual = diff_augment(x, 'color')
    bright = before + 0.25
    saturated = bright.mean(1, keepdim=True) + 0.5 * (bright - bright.mean(1, keepdim=True))
    mean = saturated.mean((1, 2, 3), keepdim=True)
    torch.testing.assert_close(actual, mean + 1.25 * (saturated - mean))
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual.max() > 1  # Do not clamp the generator's range.


def test_translation_uses_independent_zero_padded_integer_shifts(monkeypatch):
    draws = iter((torch.tensor([1, -1]), torch.tensor([-1, 1])))

    def randint(low, high, shape, **kwargs):
        assert (low, high) == (-1, 2)
        return next(draws).reshape(shape).to(**kwargs)

    monkeypatch.setattr(torch, 'randint', randint)
    x = torch.arange(1, 129, dtype=torch.float32).reshape(2, 1, 8, 8)
    expected = torch.zeros_like(x)
    expected[0, :, :-1, 1:] = x[0, :, 1:, :-1]
    expected[1, :, 1:, :-1] = x[1, :, :-1, 1:]
    torch.testing.assert_close(diff_augment(x, 'translation'), expected, rtol=0, atol=0)


@pytest.mark.parametrize('side, centers', [(8, (0, 4, 8)), (5, (0, 2, 4)), (1, (0,))])
def test_cutout_reference_boundary_clipping(monkeypatch, side, centers):
    cut = int(side * 0.5 + 0.5)

    def randint(low, high, shape, **kwargs):
        assert (low, high) == (0, side + 1 - cut % 2)
        return torch.tensor(centers, **kwargs).reshape(shape)

    monkeypatch.setattr(torch, 'randint', randint)
    x = torch.ones(len(centers), 3, side, side, dtype=torch.float64)
    expected = x.clone()
    # Scalar reference follows the official implementation's clipped index mask.
    for batch, center in enumerate(centers):
        for dy in range(cut):
            for dx in range(cut):
                y = max(0, min(side - 1, center + dy - cut // 2))
                column = max(0, min(side - 1, center + dx - cut // 2))
                expected[batch, :, y, column] = 0
    torch.testing.assert_close(diff_augment(x, 'cutout'), expected, rtol=0, atol=0)
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('official', [False, True])
def test_training_rng_replay_and_evaluation_identity(device, official):
    module = (DiffAugment('translation,cutout,color', translation_ratio=0.2, cutout_probability=0.3)
              if official else DiffAugment())
    x = torch.linspace(-1, 1, 2 * 3 * 16 * 12, device=device).reshape(2, 3, 16, 12)
    before = x.clone()
    get_rng = torch.get_rng_state if device == 'cpu' else torch.cuda.get_rng_state
    set_rng = torch.set_rng_state if device == 'cpu' else torch.cuda.set_rng_state
    state = get_rng()
    first = module(x)
    after = get_rng()
    assert not torch.equal(state, after)
    set_rng(state)
    torch.testing.assert_close(module(x), first, rtol=0, atol=0)
    assert first.shape == x.shape and first.device == x.device and first.dtype == x.dtype
    assert not torch.equal(first, x)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    module.eval()
    assert module(x) is x
    assert torch.equal(after, get_rng())
    module.train()
    assert module(x) is not x


@pytest.mark.parametrize('probability', [0.0, 0.3, 1.0])
def test_augmentation_supports_generator_and_gradient_penalty_backward(probability):
    x = torch.linspace(-1, 1, 3 * 8 * 8, dtype=torch.float64).reshape(1, 3, 8, 8).requires_grad_()
    state = torch.get_rng_state()

    def replay(value):
        torch.set_rng_state(state)
        return diff_augment(value, 'translation,cutout,color', translation_ratio=0.2,
                            cutout_probability=probability)

    assert torch.autograd.gradcheck(replay, (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(replay, (x,), fast_mode=True)
    head = torch.nn.Conv2d(3, 1, 1).double()
    score = head(replay(x)).square().mean()
    gradient, = torch.autograd.grad(score, x, create_graph=True)
    (score + gradient.square().mean()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0


def test_empty_policy_is_identity_and_invalid_configuration_fails_early():
    x = torch.ones(1, 3, 8, 8)
    state = torch.get_rng_state()
    assert DiffAugment('')(x) is x
    assert diff_augment(x, '') is x
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match='Unknown DiffAugment policy'):
        DiffAugment('color,typo')
    with pytest.raises(TypeError, match='comma-separated string'):
        DiffAugment(None)
    with pytest.raises(ValueError, match='floating-point NCHW'):
        DiffAugment()(torch.ones(1, 8, 8))


def test_official_translation_ratio_at_128(monkeypatch):
    draws = iter((26, -26))

    def randint(low, high, shape, **kwargs):
        assert (low, high) == (-26, 27)
        return torch.full(shape, next(draws), dtype=torch.long, **kwargs)

    monkeypatch.setattr(torch, 'randint', randint)
    x = torch.ones(2, 3, 128, 128)
    expected = torch.zeros_like(x)
    expected[:, :, :-26, 26:] = 1
    torch.testing.assert_close(diff_augment(x, 'translation', translation_ratio=0.2), expected)


@pytest.mark.parametrize('draw, applied', [(0.0, True), (0.299, True), (0.3, False), (0.99, False)])
def test_cutout_probability_is_a_single_whole_batch_gate(monkeypatch, draw, applied):
    gates = []

    def rand(shape, **kwargs):
        assert shape == ()
        gates.append(kwargs['device'])
        return torch.full(shape, draw, **kwargs)

    def randint(low, high, shape, **kwargs):
        return torch.tensor([0, 4, 8], **kwargs).reshape(shape)

    monkeypatch.setattr(torch, 'rand', rand)
    monkeypatch.setattr(torch, 'randint', randint)
    x = torch.ones(3, 3, 8, 8)
    actual = DiffAugment('cutout', cutout_probability=0.3)(x)
    removed = (actual == 0).sum((1, 2, 3))
    torch.testing.assert_close(removed, torch.tensor([12, 48, 12]) if applied else torch.zeros(3, dtype=torch.long))
    assert gates == [x.device]


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_disabled_cutout_is_identity_without_rng(device):
    x = torch.ones(2, 3, 8, 8, device=device).transpose(2, 3)
    get_rng = torch.get_rng_state if device == 'cpu' else torch.cuda.get_rng_state
    state = get_rng()
    assert diff_augment(x, 'cutout', cutout_probability=0) is x
    assert DiffAugment('cutout', cutout_probability=0)(x) is x
    assert torch.equal(state, get_rng())


@pytest.mark.parametrize('name', ['translation_ratio', 'cutout_probability'])
@pytest.mark.parametrize('value', [-0.01, 1.01, float('nan'), float('inf'), '0.3', None])
def test_invalid_sampling_parameters_fail_before_rng(name, value):
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        DiffAugment(**{name: value})
    with pytest.raises(ValueError, match=name):
        diff_augment(torch.ones(1, 3, 8, 8), **{name: value})
    assert torch.equal(state, torch.get_rng_state())

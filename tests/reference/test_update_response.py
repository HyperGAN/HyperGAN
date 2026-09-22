"""Held-out validation and tensor-response measurements; no GAN runs."""
import json
import math

import pytest
import torch

from hypergan.update_response import (
    tensor_change,
    verify_heldout_decrease,
    verify_pair_validation,
    verify_player_validation,
)


def test_validation_requires_resolved_decrease_on_both_independent_banks():
    assert verify_player_validation([(1., .9), (2., 1.8)], changed=True)['accepted']
    for bad in ((1., 1.), (1., 1.01), (1., 1. - 1e-8), (math.nan, 0.)):
        result = verify_player_validation([(1., .9), bad], changed=True)
        assert not result['accepted']
        json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='exactly two'):
        verify_player_validation([(1., .9)], changed=True)


def test_pair_failure_cannot_silently_become_a_single_player_proposal():
    passed = [(1., .9), (2., 1.8)]
    failed = [(1., .9), (2., 2.1)]
    assert not verify_pair_validation(passed, failed, g_factor=.5, d_factor=.5)['accepted']
    assert verify_pair_validation(passed, [], g_factor=.5, d_factor=1.)['accepted']
    result = verify_pair_validation([], [], g_factor=1., d_factor=1.)
    assert result['reason'] == 'no_reduction_proposed'


def test_pair_validation_accepts_measured_factors_below_the_historical_floor():
    passed = [(1., .9), (2., 1.8)]
    assert verify_pair_validation(passed, passed, g_factor=.001565, d_factor=.0002)['accepted']
    assert verify_pair_validation(passed, [], g_factor=.001565, d_factor=1.)['accepted']


@pytest.mark.parametrize('factor', [0., -1., True, None, '0.1', math.nan, math.inf, 1.01])
@pytest.mark.parametrize('player', ['g', 'd'])
def test_pair_validation_rejects_invalid_factors(factor, player):
    factors = {'g_factor': .5, 'd_factor': .5}
    factors[player + '_factor'] = factor
    with pytest.raises(ValueError, match='player factors'):
        verify_pair_validation([(1., .9)] * 2, [(1., .9)] * 2, **factors)


def test_validation_loss_scaling_does_not_create_an_absolute_tolerance_floor():
    for scale in (1e-20, 1., 1e20):
        assert verify_heldout_decrease(2. * scale, 1. * scale)['accepted']


def test_tensor_response_reports_rotation_and_motion_without_mutating_autograd():
    first = torch.tensor([3., 4.], requires_grad=True)
    second = torch.tensor([-4., 3.], requires_grad=True)
    originals = first.clone(), second.clone()
    report = tensor_change(first, second)
    assert report['status'] == 'finite'
    assert report['before_rms'] == pytest.approx(math.sqrt(12.5))
    assert report['change_rms'] == pytest.approx(5.)
    assert report['relative_change'] == pytest.approx(math.sqrt(2.))
    assert report['norm_ratio'] == pytest.approx(1.)
    assert report['cosine'] == pytest.approx(0., abs=1e-15)
    assert first.grad is second.grad is None
    assert first.requires_grad and second.requires_grad
    assert torch.equal(first, originals[0]) and torch.equal(second, originals[1])


def test_near_zero_signal_has_no_fabricated_relative_health_score():
    report = tensor_change(torch.zeros(2), torch.ones(2))
    assert report['change_rms'] == 1.
    assert report['relative_status'] == 'near_zero_baseline'
    assert report['relative_change'] is report['norm_ratio'] is report['cosine'] is None
    zero_after = tensor_change(torch.ones(2), torch.zeros(2))
    assert zero_after['norm_ratio'] == 0.
    assert zero_after['cosine'] is None
    nonfinite = tensor_change(torch.zeros(2), torch.tensor([math.inf, 0.]))
    assert nonfinite['status'] == 'nonfinite'
    json.dumps(nonfinite, allow_nan=False)


@pytest.mark.parametrize('epsilon', [0., -1., 1., math.inf])
def test_invalid_precision_is_a_contract_error(epsilon):
    with pytest.raises(ValueError, match='epsilon'):
        verify_heldout_decrease(1., .5, epsilon=epsilon)

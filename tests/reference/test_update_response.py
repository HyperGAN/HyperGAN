"""Deterministic algebra and real Adam displacement checks; no GAN runs."""
import json
import math

import pytest
import torch

from hypergan.update_response import (
    aggregate_proposals,
    fit_directional_quadratic,
    tensor_change,
    verify_heldout_decrease,
    verify_pair_validation,
    verify_player_validation,
)


DOUBLE_EPSILON = torch.finfo(torch.float64).eps


def quadratic_fit(factor, *, scale=1., offset=0.):
    values = [scale * ((s - factor) ** 2 + offset) for s in (0., .5, 1.)]
    return fit_directional_quadratic(*values, epsilon=DOUBLE_EPSILON)


def test_actual_adam_displacement_defines_the_direction_and_known_optimum():
    theta = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=2., betas=(0., .9), eps=0.)
    before = theta.detach().clone()

    def loss(parameter):
        return .5 * (parameter.square() * parameter.new_tensor([1., 3.])).sum()

    loss(theta).backward()
    gradient = theta.grad.clone()
    optimizer.step()
    delta = theta.detach() - before
    assert torch.allclose(delta, torch.tensor([-2., -2.], dtype=torch.float64))
    assert not torch.allclose(delta, -2. * gradient)
    values = [float(loss(before + s * delta)) for s in (0., .5, 1.)]
    fitted = fit_directional_quadratic(*values, slope=float(gradient @ delta),
                                     epsilon=DOUBLE_EPSILON)
    assert fitted['a'] == pytest.approx(-14.)
    assert fitted['k'] == pytest.approx(16.)
    assert fitted['selected_factor'] == pytest.approx(.875)
    assert loss(before + .875 * delta) < loss(before + delta)


def test_positive_loss_rescaling_preserves_fixed_direction_proposal():
    for scale in (1e-20, 1., 1e20):
        fitted = quadratic_fit(.3, scale=scale, offset=2.)
        assert fitted['status'] == 'valid'
        assert fitted['selected_factor'] == pytest.approx(.3)


@pytest.mark.parametrize(('values', 'reason'), [
    ((1., .5, 0.), 'unresolved_curvature'),
    ((1., .25, -1.), 'negative_curvature'),
    ((0., .25, 1.), 'unresolved_slope'),
    ((0., .75, 2.), 'non_descent'),
    ((1., math.nan, 0.), 'nonfinite_loss_or_slope'),
    ((1., .5, math.inf), 'nonfinite_loss_or_slope'),
])
def test_flat_negative_non_descent_and_nonfinite_abstain(values, reason):
    result = fit_directional_quadratic(*values)
    assert result['status'] == 'unresolved'
    assert result['reason'] == reason
    assert result['selected_factor'] is None
    json.dumps(result, allow_nan=False)


def test_loss_cancellation_below_evaluation_precision_abstains():
    result = fit_directional_quadratic(1e8 + .25, 1e8, 1e8 + .25)
    assert result['status'] == 'unresolved'
    assert result['reason'] == 'unresolved_slope'
    # Python float arithmetic cannot recover missing precision in an fp32 loss.
    precise = fit_directional_quadratic(1e8 + .25, 1e8, 1e8 + .25,
                                       epsilon=DOUBLE_EPSILON)
    assert precise['selected_factor'] == pytest.approx(.5)


def test_actual_matching_bank_non_descent_overrides_interpolated_descent():
    fitted = fit_directional_quadratic(.25, 0., .25, slope=.1)
    assert fitted['reason'] == 'non_descent_supplied_slope'
    assert fitted['selected_factor'] is None


def test_reduction_bounds_abstain_below_floor_and_keep_above_one():
    assert quadratic_fit(.01)['reason'] == 'below_minimum_factor'
    assert fit_directional_quadratic(0., 15., 80.)['selected_factor'] == pytest.approx(.1)
    assert quadratic_fit(3.)['selected_factor'] == 1.


def test_two_banks_choose_smaller_valid_factor_and_do_not_salvage_bad_bank():
    result = aggregate_proposals([quadratic_fit(.3), quadratic_fit(.7)])
    assert result['status'] == 'selected'
    assert result['factor'] == pytest.approx(.3)
    result = aggregate_proposals([quadratic_fit(.3), fit_directional_quadratic(0., 1., 3.)])
    assert result['status'] == 'unresolved'
    assert result['factor'] == 1.
    assert aggregate_proposals([quadratic_fit(2.), quadratic_fit(3.)])['status'] == 'unchanged'
    with pytest.raises(ValueError, match='exactly two'):
        aggregate_proposals([quadratic_fit(.3)])


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
        fit_directional_quadratic(1., 0., 1., epsilon=epsilon)

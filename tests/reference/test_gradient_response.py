"""Deterministic observed-secant algebra; no training runs or seed sweeps."""
import copy
import json
import math

import pytest
import torch

from hypergan.gradient_response import (
    aggregate_gradient_response,
    differentiation_interval,
    fit_gradient_response,
)


EPS = torch.finfo(torch.float64).eps


def measurement(g0, gh, delta, metric=None, h=.01):
    g0, gh, delta = [torch.as_tensor(value, dtype=torch.float64) for value in (g0, gh, delta)]
    metric = torch.ones_like(delta) if metric is None else torch.as_tensor(metric, dtype=torch.float64)
    dual = lambda value: float(torch.sqrt((value.square() / metric).sum()))
    d = float(torch.sqrt((delta.square() * metric).sum()))
    z = dual(gh - g0)
    return dict(status='finite', slope0=float(g0 @ delta),
                slope0_absolute_product_sum=float((g0 * delta).abs().sum()),
                epsilon=EPS, h=h, realized_parameter_perturbation_norm=float((h * delta).norm()),
                gradient0_connected_tensors=1, gradient_h_connected_tensors=1,
                adam_metric=dict(status='measured', delta_metric_norm=d,
                                 gradient_change_dual_norm=z, gradient0_dual_norm=dual(g0),
                                 gradient_h_dual_norm=dual(gh), cauchy_curvature=d * z / h))


def test_gradient_descent_recovers_published_observed_secant_rate():
    g0 = torch.tensor([2., -1.], dtype=torch.float64)
    eta, h = 3., .01
    delta = -eta * g0
    gh = g0 + torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64) @ (h * delta)
    result = fit_gradient_response(measurement(g0, gh, delta, h=h))
    published_rate = float((h * delta).norm() / (2 * (gh - g0).norm()))
    assert result['status'] == 'valid'
    assert eta * result['selected_factor'] == pytest.approx(published_rate)


def test_negative_curvature_with_gradient_rotation_is_still_measurable():
    # A symmetric indefinite Hessian gives actual negative directional curvature.
    g0 = torch.tensor([1., 0.], dtype=torch.float64)
    delta = torch.tensor([-1., 0.], dtype=torch.float64)
    hessian = torch.tensor([[-2., 30.], [30., 1.]], dtype=torch.float64)
    gh = g0 + hessian @ (.01 * delta)
    assert float(delta @ (gh - g0) / .01) < 0
    result = fit_gradient_response(measurement(g0, gh, delta))
    assert result['status'] == 'valid'
    assert result['selected_factor'] == pytest.approx(1 / (2 * math.sqrt(904)))
    assert result['selected_factor'] < .1


def test_fixed_metric_transforms_covariantly_under_coordinate_change():
    g0 = torch.tensor([3., -2.], dtype=torch.float64)
    gh = torch.tensor([2.9, -2.4], dtype=torch.float64)
    delta = torch.tensor([-.7, .2], dtype=torch.float64)
    metric = torch.tensor([.1, 30.], dtype=torch.float64)
    original = fit_gradient_response(measurement(g0, gh, delta, metric))
    # x=A*theta implies g_x=A^-1*g and M_x=A^-2*M. This is NOT a
    # claim that recomputing Adam in different coordinates yields that metric.
    transform = torch.tensor([100., .03], dtype=torch.float64)
    transformed = fit_gradient_response(measurement(
        g0 / transform, gh / transform, delta * transform, metric / transform.square()))
    assert transformed['selected_factor'] == pytest.approx(original['selected_factor'])
    for scale in (1e-12, 1e12):
        scaled = fit_gradient_response(measurement(g0, gh, delta, metric * scale))
        assert scaled['selected_factor'] == pytest.approx(original['selected_factor'])


def test_actual_adam_direction_and_poststep_metric_are_used():
    theta = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=2., betas=(.9, .99), eps=1e-8)
    hessian = torch.tensor([1., 10.], dtype=torch.float64)
    before = theta.detach().clone()
    (.5 * (hessian * theta.square()).sum()).backward()
    g0 = theta.grad.clone()
    optimizer.step()
    delta = theta.detach() - before
    denom = (optimizer.state[theta]['exp_avg_sq'] / (1 - .99)).sqrt() + 1e-8
    gh = hessian * (before + .01 * delta)
    assert not torch.allclose(delta, -2 * g0)
    result = fit_gradient_response(measurement(g0, gh, delta, denom))
    c = float(torch.sqrt((delta.square() * denom).sum())
              * torch.sqrt(((gh - g0).square() / denom).sum()) / .01)
    assert result['selected_factor'] == pytest.approx(min(1., float(-g0 @ delta) / (2 * c)))


@pytest.mark.parametrize(('path', 'value', 'reason'), [
    (('status',), 'nonfinite', 'unresolved_measurement'),
    (('slope0',), math.nan, 'nonfinite_or_missing_measurement'),
    (('slope0',), .5, 'non_descent'),
    (('slope0',), 0., 'unresolved_slope'),
    (('realized_parameter_perturbation_norm',), 0., 'unrealized_perturbation'),
    (('gradient0_connected_tensors',), 0, 'disconnected_objective'),
    (('adam_metric', 'status'), 'not_supplied', 'unresolved_adam_metric'),
    (('adam_metric', 'cauchy_curvature'), math.inf, 'nonfinite_or_missing_measurement'),
    (('adam_metric', 'gradient_change_dual_norm'), 0., 'unresolved_gradient_change'),
    (('epsilon',), True, 'nonfinite_or_missing_measurement'),
    (('h',), -1., 'invalid_measurement'),
])
def test_invalid_or_unresolved_measurements_abstain_with_finite_json(path, value, reason):
    row = measurement([1., 2.], [.8, 1.6], [-1., -1.])
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = fit_gradient_response(row)
    assert result['status'] == 'unresolved'
    assert result['reason'] == reason
    assert result['selected_factor'] is None
    json.dumps(result, allow_nan=False)


def test_cancellation_and_unresolved_gradient_difference_abstain():
    row = measurement([1., 1.], [.9, .8], [-1., -1.])
    row['slope0_absolute_product_sum'] = 1e16
    assert fit_gradient_response(row)['reason'] == 'unresolved_slope'
    row = measurement([1., 1.], [.9, .8], [-1., -1.])
    row['adam_metric']['gradient0_dual_norm'] = 1e16
    assert fit_gradient_response(row)['reason'] == 'unresolved_gradient_change'


def test_two_banks_require_both_valid_and_use_smaller_without_floor():
    small = fit_gradient_response(measurement([1.], [2.], [-1.]))
    large = fit_gradient_response(measurement([1.], [1.1], [-1.]))
    assert small['selected_factor'] == pytest.approx(.005)
    assert aggregate_gradient_response([small, large])['factor'] == pytest.approx(.005)
    invalid = copy.deepcopy(small)
    invalid['status'] = 'unresolved'
    result = aggregate_gradient_response([small, invalid])
    assert result['status'] == 'unresolved' and result['factor'] == 1.
    invalid.update(status='valid', selected_factor=math.nan)
    json.dumps(aggregate_gradient_response([small, invalid]), allow_nan=False)
    unchanged = fit_gradient_response(measurement([1.], [1.0001], [-1.]))
    assert aggregate_gradient_response([unchanged, unchanged])['status'] == 'unchanged'
    with pytest.raises(ValueError, match='exactly two'):
        aggregate_gradient_response([small])


def test_interval_is_bounded_and_uses_direction_scale_at_zero_parameters():
    assert differentiation_interval(0., 2., epsilon=EPS)['h'] == pytest.approx(math.sqrt(EPS))
    assert differentiation_interval(1e20, 1., epsilon=EPS)['h'] == .1
    assert differentiation_interval(3., 1., epsilon=EPS)['h'] == pytest.approx(3 * math.sqrt(EPS))
    for theta, delta in ((1., 0.), (-1., 1.), (math.inf, 1.), (1., math.nan)):
        result = differentiation_interval(theta, delta)
        assert result['status'] == 'unresolved'
        json.dumps(result, allow_nan=False)

"""Pure proposals from an observed gradient secant in a fixed Adam metric.

These are measured local response scales, not certified smoothness bounds or
GAN quality scores. The caller must hold the opponent, prior, draws, buffers,
and Adam metric fixed, and independently validate any proposed rate change.
"""
from __future__ import annotations

import math


FLOAT32_EPSILON = 2.0 ** -23


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def differentiation_interval(parameter_norm, direction_norm, *, epsilon=FLOAT32_EPSILON):
    """Choose h=min(.1, sqrt(eps)*max(||theta||,||delta||)/||delta||).

    This is a coordinate-dependent numerical heuristic, not a response target.
    Zero theta uses the direction's scale; zero delta cannot be probed. Actual
    representability of theta+h*delta must still be checked by the caller.
    """
    theta, delta, eps = map(_finite, (parameter_norm, direction_norm, epsilon))
    report = dict(status='unresolved', reason='invalid_norm_or_precision', h=None,
                  parameter_norm=theta, direction_norm=delta, epsilon=eps)
    if (theta is None or delta is None or eps is None
            or theta < 0 or delta <= 0 or not 0 < eps < 1):
        return report
    # Divide after multiplying by sqrt(eps), avoiding an overflowing norm ratio.
    h = min(.1, math.sqrt(eps) * max(theta, delta) / delta)
    if not math.isfinite(h) or h <= 0:
        report['reason'] = 'unresolved_interval'
        return report
    report.update(status='valid', reason=None, h=h)
    return report


def fit_gradient_response(measurement):
    """Return s=min(1, -a/(2*C)) for resolved a=g0.dot(delta)<0.

    C=||delta||_M * ||gh-g0||_(M^-1)/h with the actual update's fixed,
    positive Adam denominator M. Cauchy bounds the *observed directional
    secant*, including when signed directional curvature is negative. It does
    not bound curvature elsewhere. The 1/2 margin reproduces the adaptive-GD
    secant rate ||dx||/(2||dg||) for ordinary gradient descent. Applying it to
    arbitrary Adam directions does not inherit that algorithm's convergence
    result (Malitsky & Mishchenko, ICML 2020).

    The 8*eps guards detect cancellation at the reported precision; they are
    heuristics, not bounds on full backpropagation, TF32, or stochastic error.
    """
    report = dict(status='unresolved', reason=None, selected_factor=None,
                  unconstrained_factor=None, slope0=None, cauchy_curvature=None,
                  slope_resolution=None, gradient_change_resolution=None,
                  h=None, epsilon=None,
                  formula='min(1, -slope0 / (2 * adam_metric.cauchy_curvature))')
    if not isinstance(measurement, dict) or measurement.get('status') != 'finite':
        report['reason'] = 'unresolved_measurement'
        return report
    metric = measurement.get('adam_metric')
    if not isinstance(metric, dict) or metric.get('status') != 'measured':
        report['reason'] = 'unresolved_adam_metric'
        return report
    a, absolute, eps, h, realized = [_finite(measurement.get(key)) for key in (
        'slope0', 'slope0_absolute_product_sum', 'epsilon', 'h',
        'realized_parameter_perturbation_norm')]
    c, delta, difference, g0, gh = [_finite(metric.get(key)) for key in (
        'cauchy_curvature', 'delta_metric_norm', 'gradient_change_dual_norm',
        'gradient0_dual_norm', 'gradient_h_dual_norm')]
    report.update(slope0=a, cauchy_curvature=c, h=h, epsilon=eps)
    if any(value is None for value in (a, absolute, eps, h, realized, c, delta, difference, g0, gh)):
        report['reason'] = 'nonfinite_or_missing_measurement'
    elif not 0 < eps < 1 or h <= 0 or min(absolute, realized, c, delta, difference, g0, gh) < 0:
        report['reason'] = 'invalid_measurement'
    elif realized == 0 or delta == 0:
        report['reason'] = 'unrealized_perturbation'
    elif any(measurement.get(key) == 0 for key in (
            'gradient0_connected_tensors', 'gradient_h_connected_tensors')):
        report['reason'] = 'disconnected_objective'
    else:
        # Scale before adding to avoid overflow for otherwise finite norms.
        ar = _finite((8 * eps) * absolute)
        gr = _finite((8 * eps) * g0 + (8 * eps) * gh)
        report.update(slope_resolution=ar, gradient_change_resolution=gr)
        if ar is None or gr is None:
            report['reason'] = 'nonfinite_resolution'
        elif absolute < abs(a):
            report['reason'] = 'inconsistent_absolute_product_sum'
        elif a >= -ar:
            report['reason'] = 'non_descent' if a > ar else 'unresolved_slope'
        elif difference <= gr or c == 0:
            report['reason'] = 'unresolved_gradient_change'
        else:
            # Divide numerator first to avoid overflowing 2*C.
            factor = _finite((-a * .5) / c)
            report['unconstrained_factor'] = factor
            if factor is None or factor <= 0:
                report['reason'] = 'unrepresentable_factor'
            else:
                report.update(status='valid', selected_factor=min(1., factor))
    return report


def aggregate_gradient_response(fits):
    """Require exactly two valid banks; use the smaller reduction-only factor."""
    fits = list(fits)
    if len(fits) != 2:
        raise ValueError('exactly two fitting banks are required')
    # Do not copy untrusted measurements (potential NaN/Inf) into finite JSON.
    report = dict(status='unresolved', reason='unresolved_fitting_bank', factor=1.,
                  bank_factors=[None, None])
    if any(not isinstance(fit, dict) or fit.get('status') != 'valid' for fit in fits):
        return report
    factors = [_finite(fit.get('selected_factor')) for fit in fits]
    report['bank_factors'] = factors
    if any(factor is None or not 0 < factor <= 1 for factor in factors):
        report['reason'] = 'invalid_fitting_factor'
        return report
    factor = min(factors)
    report.update(status='selected' if factor < 1 else 'unchanged', reason=None, factor=factor)
    return report

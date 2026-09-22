"""Pure measurements for bounded startup proposals, not a GAN quality metric.

Losses must replay the same state and stochastic draws within each bank. The
two validation banks must be distinct from the two banks used for fitting.
These helpers cannot establish or enforce that separation themselves.
"""
from __future__ import annotations

import math

import torch


FLOAT32_EPSILON = 2.0 ** -23
MIN_FACTOR = 0.1


def _number(value):
    """Never emit NaN/Inf into a persisted JSON report."""
    value = float(value)
    return value if math.isfinite(value) else None


def _epsilon(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("epsilon must be a finite floating-point precision in (0, 1)")
    return value


def _resolution(values, coefficients, epsilon):
    # Conservative cancellation bound, not an estimate of minibatch variance.
    # No floor of 1: such a floor would make the result depend on loss units.
    return _number(8 * epsilon * sum(abs(c * v) for c, v in zip(coefficients, values)))


def fit_directional_quadratic(loss_zero, loss_half, loss_one, *, slope=None,
                              epsilon=FLOAT32_EPSILON):
    """Fit phi(s)=phi(0)+a*s+k*s²/2 at s=(0, .5, 1).

    ``slope``, when supplied, must be gradient·delta on this same bank/state,
    not the gradient which originally produced delta on a training minibatch.
    ``epsilon`` describes the precision of the evaluated losses; converting a
    float32 loss to a Python float does not make its evaluation float64.
    """
    epsilon = _epsilon(epsilon)
    losses = [_number(v) for v in (loss_zero, loss_half, loss_one)]
    report = dict(status="unresolved", reason=None, losses=losses, a=None, k=None,
                  slope_resolution=None, curvature_resolution=None,
                  supplied_slope=None if slope is None else _number(slope),
                  unconstrained_factor=None, selected_factor=None)
    if any(v is None for v in losses) or (slope is not None and report['supplied_slope'] is None):
        report['reason'] = 'nonfinite_loss_or_slope'
        return report
    l0, lh, l1 = losses
    a = _number(4 * (lh - l0) - (l1 - l0))
    k = _number(4 * ((l1 - l0) - 2 * (lh - l0)))
    ar = _resolution(losses, (3, 4, 1), epsilon)
    kr = _resolution(losses, (4, 8, 4), epsilon)
    report.update(a=a, k=k, slope_resolution=ar, curvature_resolution=kr)
    if any(v is None for v in (a, k, ar, kr)):
        report['reason'] = 'nonfinite_fit'
    elif a >= -ar:
        report['reason'] = 'non_descent' if a > ar else 'unresolved_slope'
    elif slope is not None and report['supplied_slope'] >= -ar:
        report['reason'] = 'non_descent_supplied_slope' if report['supplied_slope'] > ar else 'unresolved_supplied_slope'
    elif k <= kr:
        report['reason'] = 'negative_curvature' if k < -kr else 'unresolved_curvature'
    else:
        factor = _number(-a / k)
        report['unconstrained_factor'] = factor
        if factor is None:
            report['reason'] = 'nonfinite_factor'
        elif factor < MIN_FACTOR:
            report['reason'] = 'below_minimum_factor'
        else:
            report.update(status='valid', reason=None, selected_factor=min(1., factor))
    return report


def aggregate_proposals(fits):
    """Require both fitting banks to resolve descent and positive curvature."""
    fits = list(fits)
    if len(fits) != 2:
        raise ValueError('exactly two fitting banks are required')
    report = dict(status='unresolved', reason=None, factor=1., banks=fits)
    if any(fit['status'] != 'valid' for fit in fits):
        report['reason'] = 'unresolved_fitting_bank'
        return report
    factors = [fit['selected_factor'] for fit in fits]
    if any(v is None or not math.isfinite(v) or not MIN_FACTOR <= v <= 1 for v in factors):
        report['reason'] = 'invalid_fitting_factor'
        return report
    factor = min(factors)
    report.update(status='selected' if factor < 1 else 'unchanged', factor=factor)
    return report


def verify_heldout_decrease(before, after, *, epsilon=FLOAT32_EPSILON):
    """Strict resolved decrease on one independent bank; this is not Armijo."""
    epsilon = _epsilon(epsilon)
    before, after = _number(before), _number(after)
    report = dict(accepted=False, reason=None, before=before, after=after,
                  decrease=None, resolution=None)
    if before is None or after is None:
        report['reason'] = 'nonfinite_validation_loss'
        return report
    decrease = _number(before - after)
    resolution = _resolution((before, after), (1, 1), epsilon)
    report.update(decrease=decrease, resolution=resolution)
    if decrease is None or resolution is None:
        report['reason'] = 'nonfinite_validation_difference'
    elif decrease <= resolution:
        report['reason'] = 'validation_loss_increased' if decrease < -resolution else 'unresolved_validation_decrease'
    else:
        report['accepted'] = True
    return report


def verify_player_validation(banks, *, changed, epsilon=FLOAT32_EPSILON):
    """Unchanged players defer to the coupled rollout, without a loss test."""
    if not changed:
        return dict(accepted=True, status='unchanged', reason=None, banks=[])
    banks = list(banks)
    if len(banks) != 2:
        raise ValueError('exactly two validation banks are required for a changed player')
    rows = [verify_heldout_decrease(*bank, epsilon=epsilon) for bank in banks]
    accepted = all(row['accepted'] for row in rows)
    return dict(accepted=accepted, status='accepted' if accepted else 'rejected',
                reason=None if accepted else 'changed_player_validation_failed', banks=rows)


def verify_pair_validation(g_banks, d_banks, *, g_factor, d_factor,
                           epsilon=FLOAT32_EPSILON):
    """Reject the entire pair, never salvage a new combination after failure."""
    for factor in (g_factor, d_factor):
        if isinstance(factor, bool) or not math.isfinite(factor) or not MIN_FACTOR <= factor <= 1:
            raise ValueError('player factors must be finite and in [0.1, 1]')
    g = verify_player_validation(g_banks, changed=g_factor < 1, epsilon=epsilon)
    d = verify_player_validation(d_banks, changed=d_factor < 1, epsilon=epsilon)
    changed = g_factor < 1 or d_factor < 1
    accepted = changed and g['accepted'] and d['accepted']
    return dict(accepted=accepted, reason=None if accepted else (
        'no_reduction_proposed' if not changed else 'pair_validation_failed'), generator=g, discriminator=d)


def tensor_change(before, after, *, zero_tolerance=1e-12):
    """Read-only finite displacement / fixed-image cotangent comparison.

    Absolute RMS is always reported when finite. Relative change and norm ratio
    require a resolved baseline; cosine requires both norms resolved. The zero
    tolerance is explicitly in the tensor's units, not a universal health level.
    Callers must separately ensure identical inputs and stochastic state.
    """
    if before.shape != after.shape or before.numel() == 0:
        raise ValueError('comparison requires equal, nonempty tensor shapes')
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError('zero_tolerance must be finite and nonnegative')
    report = dict(status='nonfinite', before_rms=None, after_rms=None,
                  change_rms=None, relative_change=None, norm_ratio=None,
                  cosine=None, relative_status='nonfinite')
    with torch.no_grad():
        first = before.detach().to(dtype=torch.float64).reshape(-1)
        second = after.detach().to(device=first.device, dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
            return report
        root_n = math.sqrt(first.numel())
        baseline = _number(torch.linalg.vector_norm(first) / root_n)
        current = _number(torch.linalg.vector_norm(second) / root_n)
        change = _number(torch.linalg.vector_norm(second - first) / root_n)
        report.update(before_rms=baseline, after_rms=current, change_rms=change)
        if any(v is None for v in (baseline, current, change)):
            return report
        report.update(status='finite', relative_status='near_zero_baseline')
        if baseline > zero_tolerance:
            relative, ratio = _number(change / baseline), _number(current / baseline)
            report.update(relative_change=relative, norm_ratio=ratio,
                          relative_status='resolved' if relative is not None and ratio is not None else 'nonfinite')
        if baseline > zero_tolerance and current > zero_tolerance:
            cosine = _number(torch.dot(first / (baseline * root_n), second / (current * root_n)))
            report['cosine'] = None if cosine is None else min(1., max(-1., cosine))
    return report

"""Research-only FLeRM measurements and GeN model checks; never select rates."""
import math

import torch

from hypergan.objective_program import _bound_scores
from hypergan.startup_dynamics import _snapshot, _restore, _protected
from hypergan.startup_response_probe import (
    _parameters, _copy, _registered_parameters, phase_loss,
)
from hypergan.signal_structure import _hash


def symmetric_model(points, slope):
    """Fit -1/0/+1, assess unused +.5; report extrapolation without applying it."""
    minus, zero, plus, middle = (points[x] for x in (-1., 0., 1., .5))
    a = (plus - minus) / 2
    curvature = plus + minus - 2 * zero
    predicted = zero + .5 * a + .125 * curvature
    return {
        'symmetric_slope': a, 'curvature': curvature,
        'exact_slope_at_zero': slope,
        'slope_error': a - slope,
        'unused_half_step_predicted_loss': predicted,
        'unused_half_step_actual_loss': middle,
        'unused_half_step_prediction_error': predicted - middle,
        'unused_half_step_actual_change': middle - zero,
        'stationary_factor': -a / curvature if a < 0 and curvature > 0 else None,
        'status': 'positive_curvature_descent' if a < 0 and curvature > 0 else 'unresolved_signs',
        'interpretation': 'Descriptive model check only; numerical significance and independent data validation still required. No rate is applied.',
    }


def measure_gen_stencil(trainer, anchor, role, bank):
    entry = _snapshot(trainer)
    protected = _protected(trainer)
    original_hash = _hash(protected)
    points, slope, absolute_sum = {}, None, None
    try:
        owned = _parameters(trainer, role)
        for factor in (-1., 0., 1., .5):
            _restore(trainer, anchor['snapshot'])
            _copy(owned, [before + factor * delta
                          for before, delta in zip(anchor['before'], anchor['delta'])])
            expected = _hash(_registered_parameters(trainer))
            loss = phase_loss(trainer, role, *bank, step=anchor['step'])
            points[factor] = float(loss.detach())
            if factor == 0:
                gradients = torch.autograd.grad(loss, owned, allow_unused=True)
                products = [g.detach().cpu().double() * delta.double()
                            for g, delta in zip(gradients, anchor['delta']) if g is not None]
                slope = sum(float(value.sum()) for value in products)
                absolute_sum = sum(float(value.abs().sum()) for value in products)
                del gradients, products
            if _hash(_registered_parameters(trainer)) != expected or _hash(protected) != original_hash:
                raise ValueError('GeN diagnostic modified registered or protected parameters')
            del loss
        if not all(math.isfinite(x) for x in (*points.values(), slope, absolute_sum)):
            raise FloatingPointError('Nonfinite GeN measurement')
        return {'player': role, 'anchor_step': anchor['step'],
                'points': [{'factor': x, 'loss': y} for x, y in points.items()],
                'gradient_dot_delta_absolute_sum': absolute_sum,
                'model': symmetric_model(points, slope),
                'phase_loss_evaluations': 4, 'player_gradient_evaluations': 1}
    finally:
        _restore(trainer, entry)


def _output(trainer, role, bank):
    _, _, context = trainer._draw(*bank)
    if role == 'generator':
        return context['generated'].reshape(-1)
    if role != 'discriminator':
        raise ValueError('Only owned G/D directions are supported')
    values = []
    for term in trainer.program.adversarial_terms:
        _, _, real, fake = _bound_scores(term, context, trainer.graph, 'critic',
                                         term.critic_phase, first='real')
        values.extend((real.reshape(-1), fake.reshape(-1)))
    return torch.cat(values)


def crossed_progress(trainer, initial, final, bank):
    """Separate G and D parameter effects with initial prior/buffers held fixed."""
    entry = _snapshot(trainer)
    protected = _protected(trainer)
    expected_protected = _hash(protected)
    values, losses = {}, {}
    try:
        for label, state in (('old', initial), ('new', final)):
            _restore(trainer, state)
            values[label] = {role: [p.detach().cpu().clone() for p in _parameters(trainer, role)]
                             for role in ('generator', 'discriminator')}
        for g_state in ('old', 'new'):
            for d_state in ('old', 'new'):
                _restore(trainer, initial)
                _copy(_parameters(trainer, 'generator'), values[g_state]['generator'])
                _copy(_parameters(trainer, 'discriminator'), values[d_state]['discriminator'])
                expected = _hash(_registered_parameters(trainer))
                with torch.no_grad():
                    loss = float(phase_loss(trainer, 'generator', *bank, step=1))
                if not math.isfinite(loss):
                    raise FloatingPointError('Nonfinite crossed objective')
                if _hash(_registered_parameters(trainer)) != expected or _hash(protected) != expected_protected:
                    raise ValueError('Crossed objective changed registered or protected state')
                losses[g_state + '_g_' + d_state + '_d'] = loss
        return {'losses': losses,
                'g_progress_against_initial_d': losses['new_g_old_d'] - losses['old_g_old_d'],
                'g_progress_against_final_d': losses['new_g_new_d'] - losses['old_g_new_d'],
                'd_drift_at_initial_g': losses['old_g_new_d'] - losses['old_g_old_d'],
                'd_drift_at_final_g': losses['new_g_new_d'] - losses['new_g_old_d'],
                'phase_loss_evaluations': 4,
                'control': 'Same real bank and fixed initial latent values; initial prior, buffers and random draws. Only owned G/D parameters are crossed.',
                'interpretation': 'Negative G differences mean descent against that fixed critic; these are not distribution quality scores.'}
    finally:
        _restore(trainer, entry)


def measure_function_space(trainer, anchor, role, bank, projections=4):
    """Isotropic VJP estimates, including inter-block cross terms and finite check.

    Four projections are descriptive, noisy estimates, not a matching target or
    the paper's lower-variance Kronecker approximation. Measurement RNG is local.
    """
    if not 1 <= projections <= 8:
        raise ValueError('Expected 1..8 fixed measurement projections')
    entry = _snapshot(trainer)
    protected = _protected(trainer)
    original_hash = _hash(protected)
    try:
        _restore(trainer, anchor['snapshot'])
        owned = _parameters(trainer, role)
        names = {id(value): name for name, value in _registered_parameters(trainer)}
        expected = _hash(_registered_parameters(trainer))
        output = _output(trainer, role, bank)
        before = output.detach().cpu().double()
        rng = torch.Generator(device='cpu').manual_seed(0)
        rows, omegas = [], []
        for index in range(projections):
            omega = torch.randint(0, 2, (output.numel(),), generator=rng,
                                  dtype=torch.int8).double().mul_(2).sub_(1)
            omega /= math.sqrt(output.numel())
            gradients = torch.autograd.grad(output, owned,
                grad_outputs=omega.to(output.device, output.dtype),
                retain_graph=index + 1 < projections, allow_unused=True)
            rows.append([0. if g is None else float((g.detach().cpu().double() * delta.double()).sum())
                         for g, delta in zip(gradients, anchor['delta'])])
            omegas.append(omega)
            del gradients
        del output
        if _hash(_registered_parameters(trainer)) != expected or _hash(protected) != original_hash:
            raise ValueError('Function-space VJP modified registered or protected parameters')
        _restore(trainer, anchor['snapshot'])
        _copy(owned, [before + delta for before, delta in zip(anchor['before'], anchor['delta'])])
        expected = _hash(_registered_parameters(trainer))
        with torch.no_grad():
            after = _output(trainer, role, bank).detach().cpu().double()
        finite_delta = after - before
        projected = torch.tensor(rows, dtype=torch.float64)
        linear_total = projected.sum(dim=1)
        finite_total = torch.tensor([float((omega * finite_delta).sum()) for omega in omegas],
                                    dtype=torch.float64)
        if not bool(torch.isfinite(projected).all() and torch.isfinite(finite_delta).all()):
            raise FloatingPointError('Nonfinite function-space response')
        if _hash(_registered_parameters(trainer)) != expected or _hash(protected) != original_hash:
            raise ValueError('Function-space finite probe modified registered or protected parameters')
        return {
            'player': role, 'anchor_step': anchor['step'], 'projections': projections,
            'output_definition': 'generated_pixels' if role == 'generator' else 'concatenated_real_fake_critic_scores',
            'input_control': 'same real batch, latent values, opponent, prior, buffers and random draws',
            'output_elements': before.numel(),
            'output_before_rms': float(before.square().mean().sqrt()),
            'finite_update_rms': float(finite_delta.square().mean().sqrt()),
            'estimated_linear_update_rms': float(linear_total.square().mean().sqrt()),
            'estimated_sum_block_squared_responses': float(projected.square().mean(dim=0).sum()),
            'estimated_total_squared_response': float(linear_total.square().mean()),
            'estimated_cross_terms': float(linear_total.square().mean() - projected.square().mean(dim=0).sum()),
            'projected_linear_totals': linear_total.tolist(),
            'projected_finite_totals': finite_total.tolist(),
            'estimated_linearization_error_rms': float((finite_total - linear_total).square().mean().sqrt()),
            'parameters': [{'path': names[id(parameter)],
                            'estimated_output_response_rms': float(projected[:, index].square().mean().sqrt()),
                            'projections': projected[:, index].tolist()}
                           for index, parameter in enumerate(owned)],
            'output_forwards': 2, 'projection_backwards': projections,
            'interpretation': 'Low-count unbiased squared-norm estimator with potentially high variance; square roots are biased. No target or automatic layer adjustment.',
        }
    finally:
        _restore(trainer, entry)

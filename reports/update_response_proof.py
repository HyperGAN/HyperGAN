"""Deterministic CPU algebra checks for the update-response research memo.

No GAN training, random inputs, seed experiments, or production imports.
Prints JSON; the caller chooses whether and where to save it.
"""
import json
import itertools
import math

import torch


def main():
    dtype = torch.float64
    matrix = torch.tensor([[2., -.5], [.3, 1.5], [-1., .7]], dtype=dtype)
    target = torch.tensor([.1, -.2, .4], dtype=dtype)
    theta = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=dtype))
    optimizer = torch.optim.Adam([theta], lr=.02, betas=(.5, .999))

    def generator(parameter):
        return torch.tanh(matrix @ parameter)

    before = theta.detach().clone()
    output = generator(theta)
    loss = (output - target).square().mean()
    q = torch.autograd.grad(loss, output, retain_graph=True)[0]
    loss.backward()
    gradient = theta.grad.detach().clone()
    optimizer.step()
    delta = theta.detach() - before
    _, tangent = torch.func.jvp(generator, (before,), (delta,))
    actual = generator(theta.detach()) - generator(before)
    pullback_prediction = float(gradient @ delta)
    output_prediction = float(q @ tangent)
    assert math.isclose(pullback_prediction, output_prediction, rel_tol=1e-12)
    errors = {}
    for scale in (1., .5):
        measured = generator(before + scale * delta) - generator(before)
        errors[str(scale)] = float(torch.linalg.vector_norm(measured - scale * tangent)
                                   / torch.linalg.vector_norm(scale * tangent))
    assert errors['0.5'] < errors['1.0']

    # Same physical displacement in rescaled coordinates has the same JVP.
    _, rescaled_tangent = torch.func.jvp(
        lambda phi: generator(phi / 10.), (10. * before,), (10. * delta,))
    assert torch.allclose(tangent, rescaled_tangent, rtol=1e-12, atol=1e-14)

    # Enumerate every sign projection in this tiny example instead of sampling.
    # One VJP exposes all parameter blocks; cross terms matter for the total.
    _, vjp = torch.func.vjp(generator, before)
    projected_blocks = []
    for signs in itertools.product((-1., 1.), repeat=3):
        cotangent = torch.tensor(signs, dtype=dtype) / math.sqrt(3)
        projected_blocks.append(delta * vjp(cotangent)[0])
    projections = torch.stack(projected_blocks)
    block_squared_rms = projections.square().mean(dim=0)
    total_squared_rms = projections.sum(dim=1).square().mean()
    assert torch.allclose(total_squared_rms, tangent.square().mean(), rtol=1e-12)
    for block in range(2):
        block_delta = torch.zeros_like(delta)
        block_delta[block] = delta[block]
        _, block_tangent = torch.func.jvp(generator, (before,), (block_delta,))
        assert torch.allclose(block_squared_rms[block], block_tangent.square().mean(), rtol=1e-12)

    # A fixed three-point stencil recovers a quadratic minimum; flat/negative
    # curvature does not identify a positive interior minimizing step.
    fits = {}
    for name, function in (
        ('positive', lambda s: (1. - 4. * s) ** 2 + .5),
        ('flat', lambda s: 1. - s),
        ('negative', lambda s: 1. - s - s * s),
    ):
        l0, lh, l1 = (function(s) for s in (0., .5, 1.))
        a = 4. * lh - l1 - 3. * l0
        k = 4. * (l1 - 2. * lh + l0)
        fits[name] = {'slope': a, 'curvature': k,
                      'interior_minimum': -a / k if a < 0 < k else None}
    assert fits['positive']['interior_minimum'] == .25
    assert fits['flat']['interior_minimum'] is None
    assert fits['negative']['interior_minimum'] is None

    # Worst singular direction need not be the optimizer's direction.
    jacobian = torch.diag(torch.tensor([1000., 1.], dtype=dtype))
    direction = torch.tensor([0., .1], dtype=dtype)
    bound = float(torch.linalg.matrix_norm(jacobian, ord=2)
                  * torch.linalg.vector_norm(direction))
    directional_motion = float(torch.linalg.vector_norm(jacobian @ direction))
    assert math.isclose(bound / directional_motion, 1000.)

    # Both players have own curvature 1. Simultaneous unit steps minimize each
    # frozen-opponent quadratic but destroy an otherwise stable small-step map.
    field_jacobian = torch.tensor([[1., 10.], [-10., 1.]], dtype=dtype)
    radii = {}
    for rate in (.01, 1.):
        update_map = torch.eye(2, dtype=dtype) - rate * field_jacobian
        radii[str(rate)] = float(torch.linalg.eigvals(update_map).abs().max())
    assert radii['0.01'] < 1 < radii['1.0']

    hidden_delta = 1.
    saturated_output_delta = math.tanh(7.) - math.tanh(6.)
    assert saturated_output_delta < 2e-5
    print(json.dumps({
        'kind': 'deterministic_algebra_only',
        'torch_version': torch.__version__,
        'device': 'cpu', 'dtype': 'float64',
        'adam_direction': {'parameter_delta': delta.tolist(),
                           'actual_output_delta': actual.tolist(),
                           'jvp': tangent.tolist(),
                           'gradient_dot_delta': pullback_prediction,
                           'q_dot_jvp': output_prediction,
                           'relative_linearization_error': errors,
                           'rescaled_coordinates_same_jvp': True},
        'quadratic_fits': fits,
        'all_sign_projections': {
            'count': len(projected_blocks),
            'layer_squared_rms': block_squared_rms.tolist(),
            'total_squared_rms': float(total_squared_rms),
            'sum_of_layer_squared_rms': float(block_squared_rms.sum()),
            'note': 'Exact enumeration validates identity; few projections in a real model remain noisy.',
        },
        'worst_direction': {'spectral_bound': bound,
                            'actual_directional_motion': directional_motion},
        'coupled_quadratic': {'simultaneous_update_spectral_radius': radii},
        'saturated_tanh': {'preactivation_delta': hidden_delta,
                           'output_delta': saturated_output_delta},
        'interpretation': 'Checks identities and counterexamples, not GAN rate selection or quality.',
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

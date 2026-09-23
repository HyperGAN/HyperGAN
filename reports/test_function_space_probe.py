"""CPU checks of research-only FLeRM/GeN probes, including trainer rollback."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from hypergan.checkpoints import trainer_state
from hypergan.startup_dynamics import _same_state, _snapshot


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load('research_function_space_probe', Path(__file__).with_name('function_space_probe.py'))
fixtures = _load('research_native_probe_fixtures', Path(__file__).resolve().parents[1]
                 / 'tests/reference/test_startup_response_probe.py')


def assert_restored(trainer, snapshot):
    assert _same_state(snapshot['state'], trainer_state(trainer, None))
    for parameter, original, saved in snapshot['gradients']:
        assert parameter.grad is original
        assert saved is None or _same_state(saved, parameter.grad)


def test_crossed_progress_uses_both_parameter_states_and_restores(monkeypatch):
    trainer = fixtures.make_trainer()
    initial = _snapshot(trainer)
    _, banks, _, _ = fixtures.capture_update(trainer, step=1)
    final = _snapshot(trainer)
    # Bilinear game makes the opponent's effect on G's direction explicit.
    g = trainer.program.generator_parameters[0]
    d = trainer.program.critic_parameters[0]
    monkeypatch.setattr(probe, 'phase_loss', lambda *_args, **_kwargs:
                        g.flatten()[0] * d.flatten()[0])
    result = probe.crossed_progress(trainer, initial, final, banks['generator'])
    assert len(result['losses']) == 4
    losses = result['losses']
    assert result['g_progress_against_initial_d'] == losses['new_g_old_d'] - losses['old_g_old_d']
    assert result['g_progress_against_final_d'] == losses['new_g_new_d'] - losses['old_g_new_d']
    assert result['d_drift_at_initial_g'] == losses['old_g_new_d'] - losses['old_g_old_d']
    assert_restored(trainer, final)


@pytest.mark.parametrize('role,step', [('generator', 1), ('discriminator', 8)])
def test_native_phase_and_function_probes_restore_full_state_and_grad_identity(role, step):
    trainer = fixtures.make_trainer()
    observer, banks, _, _ = fixtures.capture_update(trainer, step=step)
    entry = _snapshot(trainer)
    anchor = observer.anchors[role]
    model = probe.measure_gen_stencil(trainer, anchor, role, banks[role])
    assert [row['factor'] for row in model['points']] == [-1., 0., 1., .5]
    assert model['phase_loss_evaluations'] == 4
    assert model['player_gradient_evaluations'] == 1
    assert_restored(trainer, entry)
    response = probe.measure_function_space(trainer, anchor, role, banks[role])
    assert response['output_forwards'] == 2
    assert response['projection_backwards'] == 4
    assert response['output_elements'] > 0
    assert response['finite_update_rms'] > 0
    assert response['estimated_total_squared_response'] == pytest.approx(
        response['estimated_sum_block_squared_responses'] + response['estimated_cross_terms'])
    assert_restored(trainer, entry)
    # Local measurement RNG and restored runtime RNG give the same observations.
    assert response == probe.measure_function_space(trainer, anchor, role, banks[role])
    assert_restored(trainer, entry)
    json.dumps([model, response], allow_nan=False)


@pytest.mark.parametrize('second,total,cross', [(1., 9., 4.), (-1., 1., -4.)])
def test_scalar_output_exact_squared_norm_and_block_cross_terms(monkeypatch, second, total, cross):
    trainer = fixtures.make_trainer()
    observer, banks, _, _ = fixtures.capture_update(trainer, step=1)
    anchor = dict(observer.anchors['generator'])
    parameters = trainer.program.generator_parameters
    anchor['delta'] = [torch.zeros_like(value) for value in anchor['before']]
    anchor['delta'][0].view(-1)[0] = 2.
    anchor['delta'][1].view(-1)[0] = second

    # A single output coordinate makes Rademacher squared norms exact, so this
    # catches lost cross terms without treating a low-count estimate as exact.
    monkeypatch.setattr(probe, '_output', lambda *_:
                        (parameters[0].view(-1)[0] + parameters[1].view(-1)[0]).reshape(1))
    entry = _snapshot(trainer)
    result = probe.measure_function_space(trainer, anchor, 'generator', banks['generator'])
    assert result['estimated_sum_block_squared_responses'] == pytest.approx(5.)
    assert result['estimated_total_squared_response'] == pytest.approx(total)
    assert result['estimated_cross_terms'] == pytest.approx(cross)
    arithmetic_resolution = 4 * torch.finfo(torch.float32).eps * total ** .5
    assert result['finite_update_rms'] == pytest.approx(total ** .5, abs=arithmetic_resolution)
    assert result['estimated_linearization_error_rms'] < arithmetic_resolution
    assert_restored(trainer, entry)


def test_symmetric_fit_sign_and_unused_point_detect_nonquadratic_model():
    quadratic = lambda s: 7 - 4 * s + 2 * s * s
    points = {s: quadratic(s) for s in (-1., 0., 1., .5)}
    result = probe.symmetric_model(points, -4.)
    assert result['symmetric_slope'] == -4.
    assert result['curvature'] == 4.
    assert result['stationary_factor'] == 1.
    assert result['unused_half_step_prediction_error'] == 0.
    assert result['slope_error'] == 0.
    # A cubic vanishes at all three fitting points; independent point and exact
    # slope both expose it despite a mathematically perfect interpolating fit.
    nonlinear = lambda s: quadratic(s) + 3 * s * (s*s - 1)
    result = probe.symmetric_model({s: nonlinear(s) for s in points}, -7.)
    assert result['curvature'] == 4.
    assert result['slope_error'] == 3.
    assert result['unused_half_step_prediction_error'] == pytest.approx(1.125)
    negative = probe.symmetric_model({s: 7 - s - s*s for s in points}, -1.)
    assert negative['stationary_factor'] is None
    assert negative['status'] == 'unresolved_signs'


@pytest.mark.parametrize('kind', ['function', 'gen'])
def test_protected_buffer_mutation_detected_before_restore(monkeypatch, kind):
    trainer = fixtures.make_trainer()
    frozen = torch.nn.Linear(1, 1).requires_grad_(False)
    frozen.register_buffer('marker', torch.tensor(0.))
    trainer.graph.add_module('frozen_probe_marker', frozen)
    observer, banks, _, _ = fixtures.capture_update(trainer, step=1)
    anchor = observer.anchors['generator']
    attribute = '_output' if kind == 'function' else 'phase_loss'
    original = getattr(probe, attribute)

    def mutate(*args, **kwargs):
        frozen.marker.add_(1.)
        return original(*args, **kwargs)

    monkeypatch.setattr(probe, attribute, mutate)
    entry = _snapshot(trainer)
    helper = probe.measure_function_space if kind == 'function' else probe.measure_gen_stencil
    with pytest.raises(ValueError, match='protected'):
        helper(trainer, anchor, 'generator', banks['generator'])
    assert frozen.marker.item() == 0.
    assert_restored(trainer, entry)


@pytest.mark.parametrize('kind', ['function', 'gen'])
def test_failure_restores_parameters_streams_modes_and_existing_gradients(monkeypatch, kind):
    trainer = fixtures.make_trainer()
    observer, banks, _, _ = fixtures.capture_update(trainer, step=1)
    entry = _snapshot(trainer)

    def fail(*args, **kwargs):
        torch.rand(3)
        torch.rand(3, generator=trainer.streams['prior'])
        trainer.graph.eval()
        with torch.no_grad():
            trainer.program.generator_parameters[0].add_(1.)
        raise RuntimeError('injected probe failure')

    monkeypatch.setattr(probe, '_output' if kind == 'function' else 'phase_loss', fail)
    helper = probe.measure_function_space if kind == 'function' else probe.measure_gen_stencil
    with pytest.raises(RuntimeError, match='injected probe'):
        helper(trainer, observer.anchors['generator'], 'generator', banks['generator'])
    assert_restored(trainer, entry)

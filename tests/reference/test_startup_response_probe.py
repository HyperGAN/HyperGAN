"""Phase probes replay actual native objectives and isolate their interventions."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from hypergan.checkpoints import trainer_state
from hypergan.config import resolve_config
from hypergan.signal_structure import _hash
from hypergan.startup_dynamics import _restore, _same_state, _snapshot
from hypergan.startup_response_probe import PhaseProbes, UpdateObserver, phase_loss
from hypergan.training import ReferenceTrainer


def make_trainer():
    return ReferenceTrainer(resolve_config({
        'training': {'steps': 32, 'batch_size': 8},
        'sampling': {'count': 4},
        'gradient_penalty': {'lazy_k': 8, 'kappa': .001},
    }))


def frozen_bank(batch, ids, context):
    return (deepcopy(batch), (context['latent'].detach().clone(),
                             None if ids is None else ids.detach().clone()))


def capture_update(trainer, *, step=8, protected=()):
    """One real native update at the requested schedule boundary, no seed search."""
    trainer.step = step - 1
    budget = defaultdict(int)
    observer = UpdateObserver(trainer, _snapshot, protected, _hash(protected), budget)
    banks = {}

    def observe(event, **kwargs):
        if event.startswith('before'):
            role = 'discriminator' if event.endswith('_d') else 'generator'
            banks[role] = frozen_bank(kwargs['batch'], kwargs['ids'], kwargs['context'])
        observer(event, **kwargs)

    trainer._update_response_observer = observe
    row, _ = trainer.update()
    del trainer._update_response_observer
    return observer, banks, row, budget


@pytest.mark.parametrize('step', [7, 8])
def test_phase_loss_matches_native_discriminator_and_lazy_penalty(step):
    trainer = make_trainer()
    trainer.step = step - 1
    captured = {}

    def observe(event, **kwargs):
        if event == 'before_d':
            captured['snapshot'] = _snapshot(trainer)
            captured['bank'] = frozen_bank(kwargs['batch'], kwargs['ids'], kwargs['context'])

    trainer._update_response_observer = observe
    row, _ = trainer.update()
    del trainer._update_response_observer
    _restore(trainer, captured['snapshot'])
    measured = phase_loss(trainer, 'discriminator', *captured['bank'], step=step)
    assert float(measured.detach()) == pytest.approx(row['d_loss'], rel=1e-6, abs=1e-7)
    if step == 7:
        assert row['gradient_penalty'] == 0.
    else:
        assert row['gradient_penalty'] > 0.
        # The native lazy event multiplies the ordinary coefficient by eight.
        _restore(trainer, captured['snapshot'])
        term = trainer.program.adversarial_terms[0]
        eager_penalty = deepcopy(term.penalty_fn)
        eager_penalty.lazy_k = 1
        trainer.program = replace(trainer.program, adversarial_terms=(replace(term, penalty_fn=eager_penalty),))
        eager = float(phase_loss(trainer, 'discriminator', *captured['bank'], step=step).detach())
        eager_penalty_value = eager - row['d_adversarial_weighted']
        assert eager_penalty_value == pytest.approx(row['gradient_penalty'] / 8., rel=2e-3, abs=1e-7)


def test_actual_adam_delta_and_post_discriminator_anchor_replay(monkeypatch):
    import hypergan.startup_response_probe as module
    trainer = make_trainer()
    original_critic = [value.detach().clone() for value in trainer.program.critic_parameters]
    observer, banks, row, budget = capture_update(trainer, step=1)
    anchor = observer.anchors['generator']
    assert anchor['step'] == 1
    # The captured displacement reconstructs the actual optimizer result.
    for value, before, delta in zip(trainer.program.generator_parameters, anchor['before'], anchor['delta']):
        torch.testing.assert_close(before + delta, value.detach().cpu(), rtol=0, atol=0)
    assert any(not torch.equal(old, new) for old, new in zip(original_critic, trainer.program.critic_parameters))
    after_critic = [value.detach().clone() for value in trainer.program.critic_parameters]
    _restore(trainer, anchor['snapshot'])
    assert all(torch.equal(old, new) for old, new in zip(after_critic, trainer.program.critic_parameters))
    assert all(not value.requires_grad for value in trainer.program.critic_parameters)
    value = phase_loss(trainer, 'generator', *banks['generator'], step=1)
    assert float(value.detach()) == pytest.approx(row['g_loss'], rel=1e-6, abs=1e-7)
    _restore(trainer, anchor['snapshot'])
    prior = deepcopy(trainer.prior.state_dict())
    actual_phase_loss = module.phase_loss
    observed = []

    def audited(*args, **kwargs):
        assert all(torch.equal(old, new) for old, new in zip(after_critic, trainer.program.critic_parameters))
        assert _same_state(prior, trainer.prior.state_dict())
        observed.append([value.detach().clone() for value in trainer.program.generator_parameters])
        return actual_phase_loss(*args, **kwargs)

    monkeypatch.setattr(module, 'phase_loss', audited)
    probes = PhaseProbes(trainer, _restore, [], _hash([]), budget)
    for factor in (0., .5, 1.):
        probes.evaluate(anchor, 'generator', banks['generator'], factor, category='fit_phase_loss_evaluations')
        assert _same_state(anchor['snapshot']['state'], trainer_state(trainer, None))
    assert len(observed) == 3
    for factor, parameters in zip((0., .5, 1.), observed):
        for value, before, delta in zip(parameters, anchor['before'], anchor['delta']):
            torch.testing.assert_close(value, before + factor * delta, rtol=0, atol=0)


def test_fixed_image_signal_generates_once_and_reuses_identical_scored_input(monkeypatch):
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer)
    anchor = observer.anchors['discriminator']
    generate = trainer.graph.generate
    generated, scored = [], []

    def deliberately_nonrepeatable(*args, **kwargs):
        context = generate(*args, **kwargs)
        image = context['generated'] + .1 * (len(generated) + 1)
        context['generated'] = image
        context['components']['generator'] = image
        generated.append(image.detach().clone())
        return context

    monkeypatch.setattr(trainer.graph, 'generate', deliberately_nonrepeatable)
    hook = trainer.program.adversarial_terms[0].module.register_forward_pre_hook(
        lambda module, args, kwargs: scored.append(kwargs['x'].detach().clone()), with_kwargs=True)
    try:
        report = PhaseProbes(trainer, _restore, [], _hash([]), budget).d_signal_response(
            anchor, [banks['discriminator'], banks['discriminator']])
    finally:
        hook.remove()
    assert report['status'] == 'measured'
    assert len(generated) == budget['q_image_forwards'] == 2
    assert budget['d_signal_input_backwards'] == 4
    # Each factor scores fake then real. Its fake is the same actual tensor value.
    assert len(scored) == 8
    for bank_index in range(2):
        assert torch.equal(scored[bank_index * 4], generated[bank_index])
        assert torch.equal(scored[bank_index * 4 + 2], generated[bank_index])
    assert _same_state(anchor['snapshot']['state'], trainer_state(trainer, None))


@pytest.mark.parametrize('change', ['detach_sample', 'detach_score', 'path'])
def test_fixed_image_signal_skips_unsupported_fake_binding(change):
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer)
    term = trainer.program.adversarial_terms[0]
    fake = replace(term.generator_phase.fake, **{change: 'batch.real' if change == 'path' else True})
    term = replace(term, generator_phase=replace(term.generator_phase, fake=fake))
    trainer.program = replace(trainer.program, adversarial_terms=(term,))
    report = PhaseProbes(trainer, _restore, [], _hash([]), budget).d_signal_response(
        observer.anchors['discriminator'], [banks['discriminator']])
    assert report['status'] == 'skipped'
    assert budget['q_image_forwards'] == budget['d_signal_input_backwards'] == 0


def test_fixed_image_signal_reports_disconnection_without_aborting_tuning(monkeypatch):
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer)
    anchor = observer.anchors['discriminator']
    critic = trainer.program.adversarial_terms[0].module
    weight = trainer.program.critic_parameters[0]
    monkeypatch.setattr(critic, 'forward', lambda x: weight.sum().expand(len(x), 1))
    report = PhaseProbes(trainer, _restore, [], _hash([]), budget).d_signal_response(
        anchor, [banks['discriminator']])
    assert report['status'] == 'skipped' and 'disconnected' in report['reason']
    assert _same_state(anchor['snapshot']['state'], trainer_state(trainer, None))


def test_protected_buffer_write_is_detected_before_restore(monkeypatch):
    import hypergan.startup_response_probe as module
    trainer = make_trainer()
    trainer.graph.register_buffer('_protected_probe_marker', torch.tensor(0.))
    protected = [('marker', trainer.graph._protected_probe_marker)]
    observer, banks, _, budget = capture_update(trainer, step=1, protected=protected)
    anchor = observer.anchors['generator']
    original = module.phase_loss

    def mutate(*args, **kwargs):
        trainer.graph._protected_probe_marker.add_(1.)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'phase_loss', mutate)
    probes = PhaseProbes(trainer, _restore, protected, _hash(protected), budget)
    with pytest.raises(ValueError, match='protected frozen/pretrained'):
        probes.evaluate(anchor, 'generator', banks['generator'], .5, category='fit_phase_loss_evaluations')
    assert trainer.graph._protected_probe_marker.item() == 0.
    assert _same_state(anchor['snapshot']['state'], trainer_state(trainer, None))


def test_exception_during_phase_probe_restores_parameters_rng_and_existing_gradients(monkeypatch):
    import hypergan.startup_response_probe as module
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer)
    anchor = observer.anchors['discriminator']

    def fail(*args, **kwargs):
        torch.rand(3)
        torch.rand(3, generator=trainer.streams['prior'])
        with torch.no_grad():
            trainer.program.generator_parameters[0].add_(1.)
        raise RuntimeError('injected objective failure')

    monkeypatch.setattr(module, 'phase_loss', fail)
    with pytest.raises(RuntimeError, match='injected objective'):
        PhaseProbes(trainer, _restore, [], _hash([]), budget).evaluate(
            anchor, 'discriminator', banks['discriminator'], .5, category='fit_phase_loss_evaluations')
    assert _same_state(anchor['snapshot']['state'], trainer_state(trainer, None))
    for parameter, original, saved in anchor['snapshot']['gradients']:
        assert parameter.grad is original
        assert saved is None or _same_state(saved, parameter.grad)


def test_generator_anchor_keeps_first_update_while_discriminator_uses_eighth():
    trainer = make_trainer()
    observer = UpdateObserver(trainer, _snapshot, (), _hash(()), defaultdict(int))
    trainer._update_response_observer = observer
    try:
        trainer.update()
        first = observer.anchors['generator']
        saved_first = deepcopy(first['before'])
        assert 'discriminator' not in observer.anchors
        for _ in range(7):
            trainer.update()
    finally:
        del trainer._update_response_observer
    assert observer.anchors['generator'] is first
    assert first['step'] == 1 and first['snapshot']['state']['step'] == 0
    assert _same_state(first['before'], saved_first)
    assert observer.anchors['discriminator']['step'] == 8
    assert observer.anchors['discriminator']['snapshot']['state']['step'] == 7
    assert [row['step'] for row in observer.observations if row['player'] == 'generator'] == [1, 8]


@pytest.mark.parametrize('step', [1, 8])
def test_final_affine_response_exposes_motion_hidden_by_tanh_without_extra_forwards(step):
    config = resolve_config({})
    config['components']['generator']['args']['source'] = 'linear(2)\ntanh()'
    config['prior']['args']['num_particles'] = 32
    trainer = ReferenceTrainer(config)
    affine = next(module for module in trainer.graph.models['generator'].modules() if isinstance(module, torch.nn.Linear))
    with torch.no_grad():
        affine.weight.zero_()
        affine.bias.fill_(3.)
    batch, ids, context = trainer._draw(None, None)
    budget = defaultdict(int)
    observer = UpdateObserver(trainer, _snapshot, (), _hash(()), budget, capture_anchors=False)
    observer('before_g', step=step, batch=batch, ids=ids, context=context)
    with torch.no_grad():
        affine.bias.add_(1.)
    # A known parameter displacement lets this test compare the observation to
    # analytic activations; no extra training candidate or randomized trial.
    expected = deepcopy(trainer_state(trainer, None))
    observer('after_g', step=step, batch=batch, ids=ids, context=context)
    row = observer.observations[0]
    activation = row['generator_final_affine_response']
    assert activation['status'] == 'measured'
    assert activation['relationship_to_output'] == 'pre_tanh'
    assert activation['downstream_operations'] == ['tanh']
    assert activation['response']['before_rms'] == pytest.approx(3.)
    assert activation['response']['after_rms'] == pytest.approx(4.)
    assert activation['response']['change_rms'] == pytest.approx(1.)
    expected_output_change = float(torch.tanh(torch.tensor(4.)) - torch.tanh(torch.tensor(3.)))
    assert row['generator_output_response']['full']['change_rms'] == pytest.approx(expected_output_change)
    assert budget['g_response_forwards'] == 2
    assert not affine._forward_hooks
    assert _same_state(expected, trainer_state(trainer, None))


def test_final_affine_response_follows_layout_path_and_skips_unsupported_tail():
    from hypergan.startup_response_probe import _final_owned_affine
    config = resolve_config({})
    config['prior']['args']['num_particles'] = 32
    config['components']['generator']['args']['source'] = 'linear(2)\nreshape(1, 2)\npermute(2, 1)\nreshape(2)\ntanh()'
    trainer = ReferenceTrainer(config)
    module, description = _final_owned_affine(trainer)
    assert isinstance(module, torch.nn.Linear)
    assert description['relationship_to_output'] == 'pre_tanh'
    assert description['downstream_operations'] == ['reshape', 'permute', 'reshape', 'tanh']
    config['components']['generator']['args']['source'] = 'linear(2)\nleaky_relu(0.2)'
    trainer = ReferenceTrainer(config)
    module, description = _final_owned_affine(trainer)
    assert module is None and description['status'] == 'skipped'


def test_final_affine_hook_is_removed_and_forward_fence_restored_on_exception(monkeypatch):
    trainer = make_trainer()
    batch, ids, context = trainer._draw(None, None)
    observer = UpdateObserver(trainer, _snapshot, (), _hash(()), defaultdict(int), capture_anchors=False)
    assert observer.affine_module is not None
    observer('before_g', step=1, batch=batch, ids=ids, context=context)
    with torch.no_grad():
        trainer.program.generator_parameters[0].add_(.01)
    expected = deepcopy(trainer_state(trainer, None))
    def fail(*args, **kwargs):
        torch.rand(3)
        torch.rand(3, generator=trainer.streams['prior'])
        raise RuntimeError('matched forward failed')
    monkeypatch.setattr(trainer.graph, 'generate', fail)
    with pytest.raises(RuntimeError, match='matched forward failed'):
        observer('after_g', step=1, batch=batch, ids=ids, context=context)
    assert not observer.affine_module._forward_hooks
    assert _same_state(expected, trainer_state(trainer, None))


def test_direction_audit_matches_local_loss_derivative_and_preserves_callers_state_and_gradients():
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer, step=1)
    anchor = observer.anchors['generator']
    # The caller is already after its real update; the audit must restore that
    # state, not leave it at the earlier supplied phase anchor.
    for parameter in trainer.program.generator_parameters:
        parameter.grad = torch.ones_like(parameter)
    expected = deepcopy(trainer_state(trainer, None))
    gradients = [(parameter, parameter.grad) for parameter in trainer.graph.parameters()]
    probes = PhaseProbes(trainer, _restore, [], _hash([]), budget)
    rows = [probes.measure_direction(anchor, 'generator', banks['generator'], factor, gradient=factor == .1)
            for factor in (0., .1, .2)]
    central_difference = (rows[2]['loss'] - rows[0]['loss']) / .2
    assert rows[1]['gradient_status'] == 'finite'
    assert rows[1]['gradient_dot_delta_absolute_sum'] >= abs(rows[1]['gradient_dot_delta'])
    assert rows[1]['gradient_dot_delta'] == pytest.approx(central_difference, rel=.02, abs=1e-5)
    assert rows[0]['gradient_status'] == 'not_requested'
    assert budget['direction_audit_phase_loss_evaluations'] == 3
    assert budget['direction_audit_player_gradient_evaluations'] == 1
    assert all(row['output']['status'] == 'finite' and row['final_affine']['status'] == 'measured' for row in rows)
    assert _same_state(expected, trainer_state(trainer, None))
    assert all(parameter.grad is original for parameter, original in gradients)
    assert all(not module._forward_hooks for module in trainer.graph.modules())


def test_direction_audit_reports_tanh_saturation_at_known_stencil_point():
    config = resolve_config({})
    config['components']['generator']['args']['source'] = 'linear(2)\ntanh()'
    config['prior']['args']['num_particles'] = 32
    trainer = ReferenceTrainer(config)
    affine = next(module for module in trainer.graph.models['generator'].modules() if isinstance(module, torch.nn.Linear))
    with torch.no_grad():
        affine.weight.zero_()
        affine.bias.fill_(3.)
    snapshot = _snapshot(trainer)
    anchor = {'snapshot': snapshot, 'step': 1,
              'before': [parameter.detach().clone() for parameter in trainer.program.generator_parameters],
              'delta': [torch.zeros_like(parameter) if parameter is affine.weight else torch.ones_like(parameter)
                        for parameter in trainer.program.generator_parameters]}
    with torch.no_grad():
        batch, ids, context = trainer._draw(None, None)
    bank = frozen_bank(batch, ids, context)
    budget = defaultdict(int)
    report = PhaseProbes(trainer, _restore, [], _hash([]), budget).measure_direction(
        anchor, 'generator', bank, 1., gradient=True)
    pre_tanh = report['final_affine']['activation']
    assert pre_tanh['rms'] == pytest.approx(4.)
    assert pre_tanh['tanh_response']['mean_derivative'] == pytest.approx(1. - torch.tensor(4., dtype=torch.float64).tanh().item() ** 2)
    assert pre_tanh['tanh_response']['derivative_below_0_01_fraction'] == 1.
    assert report['output']['absolute_above_0_99_fraction'] == 1.
    assert report['gradient_status'] == 'finite'


def test_direction_audit_exception_restores_caller_and_removes_all_hooks(monkeypatch):
    import hypergan.startup_response_probe as module
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer, step=1)
    expected = deepcopy(trainer_state(trainer, None))
    def fail(*args, **kwargs):
        torch.rand(2)
        torch.rand(2, generator=trainer.streams['data'])
        with torch.no_grad():
            trainer.program.critic_parameters[0].add_(1.)
        raise RuntimeError('audit objective failed')
    monkeypatch.setattr(module, 'phase_loss', fail)
    with pytest.raises(RuntimeError, match='audit objective failed'):
        PhaseProbes(trainer, _restore, [], _hash([]), budget).measure_direction(
            observer.anchors['generator'], 'generator', banks['generator'], .01, gradient=True)
    assert _same_state(expected, trainer_state(trainer, None))
    assert all(not item._forward_hooks for item in trainer.graph.modules())


def test_captured_adam_metric_uses_post_step_bias_corrected_moments_without_mutation():
    from hypergan.startup_response_probe import capture_adam_denominators
    trainer = make_trainer()
    assert capture_adam_denominators(trainer, 'generator')['status'] == 'unsupported'
    trainer.update()
    expected = deepcopy(trainer_state(trainer, None))
    metric = capture_adam_denominators(trainer, 'generator')
    assert metric['status'] == 'measured'
    assert metric['steps'] == [1] * len(trainer.program.generator_parameters)
    group = trainer.opt_g.param_groups[0]
    for parameter, denominator in zip(trainer.program.generator_parameters, metric['denominators']):
        moment = trainer.opt_g.state[parameter]['exp_avg_sq'].double()
        expected_denominator = (moment / (1. - group['betas'][1])).sqrt() + group['eps']
        torch.testing.assert_close(denominator, expected_denominator, rtol=0, atol=0)
        assert denominator.device.type == 'cpu'
    assert _same_state(expected, trainer_state(trainer, None))
    group['amsgrad'] = True
    assert capture_adam_denominators(trainer, 'generator')['status'] == 'unsupported'


@pytest.mark.parametrize('sign', [1., -1.])
def test_gradient_field_matches_analytic_quadratic_in_euclidean_and_adam_metrics(monkeypatch, sign):
    import hypergan.startup_response_probe as module
    config = resolve_config({})
    config['components']['generator']['args']['source'] = 'linear(2)'
    config['prior']['args']['num_particles'] = 32
    trainer = ReferenceTrainer(config)
    owned = trainer.program.generator_parameters
    coefficients, delta, denominators = [], [], []
    with torch.no_grad():
        for index, parameter in enumerate(owned):
            parameter.fill_(.5)
            coefficients.append(torch.full_like(parameter, sign * (index + 1)))
            delta.append(torch.full_like(parameter, -.25 * (index + 1)))
            denominators.append(torch.full_like(parameter, 2. + index, dtype=torch.float64))
    anchor = {'snapshot': _snapshot(trainer), 'step': 1,
              'before': [parameter.detach().clone() for parameter in owned], 'delta': delta}
    def quadratic(*args, **kwargs):
        return sum(.5 * (coefficient * parameter.square()).sum()
                   for coefficient, parameter in zip(coefficients, owned))
    monkeypatch.setattr(module, 'phase_loss', quadratic)
    paths = {id(parameter): path for path, parameter in module._registered_parameters(trainer)}
    metric = {'status': 'measured', 'player': 'generator', 'parameter_paths': [paths[id(parameter)] for parameter in owned],
              'denominators': denominators}
    for parameter in owned:
        parameter.grad = torch.ones_like(parameter)
    expected_state = deepcopy(trainer_state(trainer, None))
    original_gradients = [parameter.grad for parameter in owned]
    budget = defaultdict(int)
    report = PhaseProbes(trainer, _restore, [], _hash([]), budget).gradient_change(
        anchor, 'generator', (None, None), .1, adam_denominators=metric)
    vector = torch.cat([value.double().reshape(-1) for value in delta])
    diagonal = torch.cat([value.double().reshape(-1) for value in coefficients])
    denominator = torch.cat([value.reshape(-1) for value in denominators])
    derivative = diagonal * vector
    expected_slope = float((diagonal * .5 * vector).sum())
    expected_signed_curvature = float((vector * derivative).sum())
    expected_cauchy = float(vector.norm() * derivative.norm())
    expected_metric = float((vector.square() * denominator).sum().sqrt()
                            * (derivative.square() / denominator).sum().sqrt())
    assert report['status'] == 'finite'
    assert report['slope0'] == pytest.approx(expected_slope)
    assert report['directional_secant_curvature'] == pytest.approx(expected_signed_curvature, rel=1e-5)
    assert report['unweighted_cauchy_curvature'] == pytest.approx(expected_cauchy, rel=1e-5)
    assert report['adam_metric']['cauchy_curvature'] == pytest.approx(expected_metric, rel=1e-5)
    assert report['adam_metric']['gradient0_dual_norm'] > 0
    assert report['realized_parameter_perturbation_norm'] == pytest.approx(float(.1 * vector.norm()), rel=1e-5)
    assert budget['gradient_field_phase_loss_evaluations'] == budget['gradient_field_player_gradient_evaluations'] == 2
    assert _same_state(expected_state, trainer_state(trainer, None))
    assert all(parameter.grad is original for parameter, original in zip(owned, original_gradients))


def test_gradient_field_second_probe_exception_restores_full_callers_state(monkeypatch):
    import hypergan.startup_response_probe as module
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer, step=1)
    expected = deepcopy(trainer_state(trainer, None))
    calls = []
    original = module.phase_loss
    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            torch.rand(2, generator=trainer.streams['prior'])
            with torch.no_grad():
                trainer.program.critic_parameters[0].add_(1.)
            raise RuntimeError('second gradient probe failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'phase_loss', fail_second)
    with pytest.raises(RuntimeError, match='second gradient probe failed'):
        PhaseProbes(trainer, _restore, [], _hash([]), budget).gradient_change(
            observer.anchors['generator'], 'generator', banks['generator'], .1)
    assert len(calls) == 2
    assert _same_state(expected, trainer_state(trainer, None))


def test_gradient_field_zero_realized_perturbation_is_unresolved():
    trainer = make_trainer()
    observer, banks, _, budget = capture_update(trainer, step=1)
    anchor = observer.anchors['generator']
    anchor['delta'] = [torch.zeros_like(value) for value in anchor['delta']]
    report = PhaseProbes(trainer, _restore, [], _hash([]), budget).gradient_change(
        anchor, 'generator', banks['generator'], .1)
    assert report['status'] == 'unresolved'
    assert report['realized_parameter_perturbation_norm'] == 0.
    assert report['adam_metric']['status'] == 'not_supplied'

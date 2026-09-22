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
    observer, banks, row, budget = capture_update(trainer)
    anchor = observer.anchors['generator']
    # The captured displacement reconstructs the actual optimizer result.
    for value, before, delta in zip(trainer.program.generator_parameters, anchor['before'], anchor['delta']):
        torch.testing.assert_close(before + delta, value.detach().cpu(), rtol=0, atol=0)
    assert any(not torch.equal(old, new) for old, new in zip(original_critic, trainer.program.critic_parameters))
    after_critic = [value.detach().clone() for value in trainer.program.critic_parameters]
    _restore(trainer, anchor['snapshot'])
    assert all(torch.equal(old, new) for old, new in zip(after_critic, trainer.program.critic_parameters))
    assert all(not value.requires_grad for value in trainer.program.critic_parameters)
    value = phase_loss(trainer, 'generator', *banks['generator'], step=8)
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
    observer, banks, _, budget = capture_update(trainer, protected=protected)
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

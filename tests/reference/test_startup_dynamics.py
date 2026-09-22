"""Bounded measured-update orchestration and exact rollback of actual training."""
import copy

import pytest
import torch

from hypergan.checkpoints import trainer_state
from hypergan.config import resolve_config
from hypergan.startup_dynamics import _same_state, tune_startup_dynamics
from hypergan.training import ReferenceTrainer


def make_trainer():
    return ReferenceTrainer(resolve_config({
        'training': {'steps': 100, 'batch_size': 4, 'lr_floor': 1.},
        'prior': {'args': {'num_particles': 32, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 8}}))


def assert_restored(trainer, state):
    assert _same_state(state, trainer_state(trainer, None))
    assert trainer.step == 0
    assert not trainer.opt_g.state and not trainer.opt_d.state
    assert not hasattr(trainer, '_update_response_observer')


def controlled_proposal(monkeypatch):
    """Exercise orchestration branches independently of the numeric fit tests."""
    import hypergan.update_response as numerical
    monkeypatch.setattr(numerical, 'aggregate_proposals', lambda fits: {
        'status': 'selected', 'reason': None, 'factor': .5, 'banks': fits})
    monkeypatch.setattr(numerical, 'verify_player_validation', lambda banks, **kw: {
        'accepted': True, 'status': 'accepted', 'reason': None, 'banks': list(banks)})


def test_native_measured_baseline_records_real_updates_and_restores_every_state():
    trainer = make_trainer()
    gradients = []
    for root in (trainer.graph, trainer.prior, trainer.ema_graph, trainer.ema_prior):
        for parameter in root.parameters():
            parameter.grad = torch.ones_like(parameter)
            gradients.append((parameter, parameter.grad))
    gradients[0][1].requires_grad_(True)
    caches = [(module, module._compiled) for module in trainer.graph.modules() if hasattr(module, '_compiled')]
    state = copy.deepcopy(trainer_state(trainer, None))
    events = []
    report = tune_startup_dynamics(trainer, progress=events.append)
    assert report['method'] == 'measured-update-response'
    assert report['outcome'] in ('kept_baseline', 'unresolved', 'selected')
    assert report['disposable_completed_updates'] in (8, 16)
    assert report['maximum_disposable_updates'] == 16
    assert report['probe_budget']['fit_phase_loss_evaluations'] == 12
    assert report['probe_budget']['validation_phase_loss_evaluations'] <= 8
    assert report['probe_budget']['fit_bank_gradient_evaluations'] == 0
    assert report['probe_budget']['d_signal_input_backwards'] == 4
    assert report['probe_budget']['q_image_forwards'] == 2
    assert report['all_trial_state_restored']
    assert all(event['phase'] == 'dynamics' for event in events)
    assert 'fit' in {event['stage'] for event in events}
    observations = report['candidates'][0]['update_response']
    assert [(row['step'], row['player']) for row in observations] == [
        (1, 'discriminator'), (1, 'generator'), (8, 'discriminator'), (8, 'generator')]
    assert all(row['optimizer_motion']['finite'] and row['optimizer_motion']['changed'] for row in observations)
    assert all(row['prior_optimizer_motion']['finite'] for row in observations if row['player'] == 'generator')
    assert_restored(trainer, state)
    assert all(module._compiled is original for module, original in caches)
    for parameter, original in gradients:
        assert parameter.grad is original
        assert torch.equal(parameter.grad, torch.ones_like(parameter))


def test_single_coupled_replay_changes_both_player_rates_but_not_prior_or_draws(monkeypatch):
    import hypergan.startup_dynamics as dynamics
    controlled_proposal(monkeypatch)
    monkeypatch.setattr(dynamics, '_candidate_guards', lambda *args: ([], []))
    trainer = make_trainer()
    state = copy.deepcopy(trainer_state(trainer, None))
    calls, original_update = [], trainer.update
    def observed_update():
        calls.append((trainer.step, copy.deepcopy(trainer.base_lrs), trainer.streams['data'].get_state().clone()))
        return original_update()
    trainer.update = observed_update
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'selected'
    assert report['selected_g_lr_factor'] == report['selected_d_lr_factor'] == .5
    assert report['selected_candidate'] == 'measured_update_pair'
    assert len(calls) == report['disposable_completed_updates'] == 16
    assert len(report['candidates']) == 2
    assert report['probe_budget']['fit_phase_loss_evaluations'] == 12
    assert report['probe_budget']['validation_phase_loss_evaluations'] == 8
    assert report['probe_budget']['g_response_forwards'] == 8
    for first, second in zip(calls[:8], calls[8:]):
        assert first[0] == second[0]
        assert torch.equal(first[2], second[2])
        assert second[1][0][0] == .5 * first[1][0][0]
        assert second[1][1][0] == .5 * first[1][1][0]
        assert second[1][0][1:] == first[1][0][1:]
    assert_restored(trainer, state)


def test_one_changed_players_heldout_failure_rejects_whole_pair_without_replay(monkeypatch):
    import hypergan.update_response as numerical
    controlled_proposal(monkeypatch)
    checks = []
    def validate(banks, **kwargs):
        checks.append(list(banks))
        return {'accepted': len(checks) == 1, 'status': 'rejected', 'reason': 'controlled_failure', 'banks': []}
    monkeypatch.setattr(numerical, 'verify_player_validation', validate)
    trainer = make_trainer()
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'unresolved'
    assert report['selected_g_lr_factor'] == report['selected_d_lr_factor'] == 1.
    assert report['disposable_completed_updates'] == 8
    assert report['probe_budget']['validation_phase_loss_evaluations'] == 8
    assert len(checks) == 2 and all(len(banks) == 2 for banks in checks)
    assert 'whole pair' in report['reason']
    assert_restored(trainer, state)


def test_failed_coupled_replay_does_not_try_another_combination(monkeypatch):
    import hypergan.startup_dynamics as dynamics
    controlled_proposal(monkeypatch)
    guard_calls = []
    def guards(*args):
        guard_calls.append(True)
        return ([] if len(guard_calls) == 1 else ['controlled_coupled_failure']), []
    monkeypatch.setattr(dynamics, '_candidate_guards', guards)
    trainer = make_trainer()
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'unresolved'
    assert report['selected_g_lr_factor'] == report['selected_d_lr_factor'] == 1.
    assert report['disposable_completed_updates'] == 16
    assert len(report['candidates']) == 2 and not report['candidates'][1]['accepted']
    assert_restored(trainer, state)


@pytest.mark.parametrize('stage', ['measure', 'fit', 'validate', 'replay'])
def test_observer_exception_at_each_stage_restores_partial_trial(monkeypatch, stage):
    controlled_proposal(monkeypatch)
    trainer = make_trainer()
    state = copy.deepcopy(trainer_state(trainer, None))
    def progress(event):
        if event['stage'] == stage and (stage not in ('measure', 'replay') or event['trial_step'] == 3):
            raise RuntimeError('observer failed')
    with pytest.raises(RuntimeError, match='observer failed'):
        tune_startup_dynamics(trainer, progress=progress)
    assert_restored(trainer, state)


def test_reserved_banks_clone_reused_data_storage_and_restore_cursor(monkeypatch):
    import hypergan.startup_response_probe as probes
    trainer = make_trainer()
    class Stateful:
        def __init__(self):
            self.state = {'cursor': [0]}
            self.batch = {'real': torch.zeros(4, 2)}
        def __call__(self, *args, **kwargs):
            self.state['cursor'][0] += 1
            self.batch['real'].fill_(self.state['cursor'][0] / 100.)
            return self.batch
        def state_dict(self):
            return {'cursor': self.state['cursor'], 'batch': self.batch}
        def load_state_dict(self, state):
            self.state = {'cursor': state['cursor']}
            self.batch = state['batch']
    trainer.data = Stateful()
    state = copy.deepcopy(trainer_state(trainer, None))
    observed = []
    original = probes.PhaseProbes.evaluate
    def evaluate(self, anchor, role, bank, factor, *, category):
        observed.append((category, float(bank[0]['real'][0, 0])))
        return original(self, anchor, role, bank, factor, category=category)
    monkeypatch.setattr(probes.PhaseProbes, 'evaluate', evaluate)
    controlled_proposal(monkeypatch)
    tune_startup_dynamics(trainer)
    fitting = {value for category, value in observed if category == 'fit_phase_loss_evaluations'}
    validation = {value for category, value in observed if category == 'validation_phase_loss_evaluations'}
    assert len(fitting) == len(validation) == 2
    assert not fitting & validation
    assert trainer.data.state == {'cursor': [0]}
    assert_restored(trainer, state)


def test_trainable_pretrained_is_skipped_before_any_optimizer_update():
    from hndl.operators.pretrained import Pretrained
    trainer = make_trainer()
    pretrained = Pretrained.__new__(Pretrained)
    torch.nn.Module.__init__(pretrained)
    pretrained.inner = torch.nn.Linear(2, 2)
    trainer.graph.models['generator'].add_module('foreign', pretrained)
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'skipped'
    assert 'Trainable pretrained' in report['reason']
    assert_restored(trainer, state)


def test_frozen_mutation_fails_before_rollback_can_hide_it():
    from hndl.operators.pretrained import Pretrained
    trainer = make_trainer()
    pretrained = Pretrained.__new__(Pretrained)
    torch.nn.Module.__init__(pretrained)
    pretrained.register_parameter('nonfinite_named_fixed', torch.nn.Parameter(torch.ones(2), requires_grad=False))
    trainer.graph.models['discriminator'].add_module('foreign', pretrained)
    state = copy.deepcopy(trainer_state(trainer, None))
    original_update = trainer.update
    def mutate():
        result = original_update()
        with torch.no_grad():
            pretrained.nonfinite_named_fixed.add_(1.)
        return result
    trainer.update = mutate
    with pytest.raises(ValueError, match='changed protected'):
        tune_startup_dynamics(trainer)
    assert_restored(trainer, state)


def test_nonfinite_partial_optimizer_update_abstains_and_rolls_back():
    trainer = make_trainer()
    state = copy.deepcopy(trainer_state(trainer, None))
    def fail_after_critic_step():
        trainer.opt_d.zero_grad(set_to_none=True)
        loss = sum(parameter.square().mean() for parameter in trainer.program.critic_parameters)
        loss.backward()
        trainer.opt_d.step()
        raise ValueError('Nonfinite generator gradient; run stopped')
    trainer.update = fail_after_critic_step
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'unresolved'
    assert report['selected_g_lr_factor'] == report['selected_d_lr_factor'] == 1.
    assert 'nonfinite' in report['reason'].lower()
    assert_restored(trainer, state)


def test_trainable_storage_alias_of_pretrained_tensor_skips_before_update():
    from hndl.operators.pretrained import Pretrained
    trainer = make_trainer()
    parameter = next(trainer.graph.models['generator'].parameters())
    pretrained = Pretrained.__new__(Pretrained)
    torch.nn.Module.__init__(pretrained)
    pretrained.register_parameter('foreign_weight', torch.nn.Parameter(parameter.detach(), requires_grad=False))
    trainer.graph.models['discriminator'].add_module('foreign', pretrained)
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'skipped'
    assert 'alias frozen/pretrained storage' in report['reason']
    assert report['disposable_completed_updates'] == 0
    assert_restored(trainer, state)


def test_additional_critic_optimizer_group_skips_before_trials():
    trainer = make_trainer()
    extra = torch.nn.Parameter(torch.ones(1))
    trainer.opt_d.add_param_group({'params': [extra]})
    trainer.base_lrs[1].append(trainer.opt_d.param_groups[1]['lr'])
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'skipped'
    assert 'one native critic optimizer group' in report['reason']
    assert_restored(trainer, state)


def test_distinct_player_parameters_sharing_storage_skip_before_trials():
    trainer = make_trainer()
    generator = next(parameter for parameter in trainer.program.generator_parameters if parameter.shape == (64,))
    discriminator = next(parameter for parameter in trainer.program.critic_parameters if parameter.shape == (64,))
    assert generator is not discriminator
    discriminator.data = generator.detach()
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'skipped'
    assert 'share tensor storage' in report['reason']
    assert report['disposable_completed_updates'] == 0
    assert_restored(trainer, state)

"""Short disposable optimizer trials select cautiously and restore exact state."""
import copy

import pytest
import torch

from hypergan.checkpoints import trainer_state
from hypergan.config import resolve_config
from hypergan.startup_dynamics import _same_state, tune_startup_dynamics
from hypergan.training import ReferenceTrainer, update_ema


def make_trainer():
    config = resolve_config({})
    config['training']['steps'] = 100
    return ReferenceTrainer(config)


def assert_restored(trainer, state):
    assert _same_state(state, trainer_state(trainer, None))
    assert trainer.step == 0
    assert not trainer.opt_g.state and not trainer.opt_d.state


def test_actual_gan_baseline_passes_and_every_training_state_is_restored():
    trainer = make_trainer()
    gradients = []
    for root in (trainer.graph, trainer.prior, trainer.ema_graph, trainer.ema_prior):
        for parameter in root.parameters():
            parameter.grad = torch.ones_like(parameter)
            gradients.append((parameter, parameter.grad))
    gradients[0][1].requires_grad_(True)
    state = copy.deepcopy(trainer_state(trainer, None))
    events = []
    report = tune_startup_dynamics(trainer, progress=events.append)
    assert report['outcome'] == 'kept_baseline'
    assert report['selected_g_lr_factor'] == 1
    assert report['disposable_completed_updates'] == 8
    assert len(report['candidates']) == 1
    assert report['all_trial_state_restored']
    assert all(event['phase'] == 'dynamics' for event in events)
    assert_restored(trainer, state)
    for parameter, original in gradients:
        assert parameter.grad is original
        assert torch.equal(parameter.grad, torch.ones_like(parameter))


def saturating_optimizer_fixture():
    """Known toy optimization: bias += LR makes final tanh saturate at LR=1.

    This uses the trainer's real configured Adam instances, prior, stream draws
    and EMA operations. Its analytic bias objective makes rollback/selection
    behavior deterministic without depending on a GAN loss finding a mode.
    """
    config = resolve_config({})
    config['training']['steps'] = 100
    config['components']['generator']['args']['source'] = 'linear(2)\ntanh()'
    config['optimizer'].update(lr=1., d_lr_mult=.0002, prior_lr_mult=.0002)
    trainer = ReferenceTrainer(config)
    last = [m for m in trainer.graph.models['generator'].modules() if isinstance(m, torch.nn.Linear)][-1]
    calls = []
    def update():
        calls.append({'step': trainer.step, 'data_rng': trainer.streams['data'].get_state().clone(),
                      'g_lr': trainer.opt_g.param_groups[0]['lr'], 'd_lr': trainer.opt_d.param_groups[0]['lr'],
                      'prior_lr': trainer.opt_g.param_groups[1]['lr']})
        trainer._draw(None, None)
        trainer.opt_d.zero_grad(set_to_none=True)
        d_loss = sum(p.square().mean() for p in trainer.program.critic_parameters)
        d_loss.backward()
        trainer.opt_d.step()
        trainer.opt_g.zero_grad(set_to_none=True)
        g_loss = -last.bias.sum() + trainer.prior.z.square().mean() * .001
        g_loss.backward()
        trainer.opt_g.step()
        update_ema(trainer.ema_graph, trainer.graph, trainer.config['training']['ema'])
        update_ema(trainer.ema_prior, trainer.prior, trainer.config['training']['ema'])
        trainer.step += 1
        return {'g_loss': float(g_loss.detach()), 'd_loss': float(d_loss.detach())}, None
    trainer.update = update
    return trainer, calls


def test_formulaic_rate_confirmation_prevents_optimizer_induced_tanh_collapse():
    trainer, calls = saturating_optimizer_fixture()
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'selected'
    assert report['selected_g_lr_factor'] == .1
    assert report['disposable_completed_updates'] == 16
    assert len(report['candidates']) == 2
    assert not report['candidates'][0]['accepted'] and report['candidates'][1]['accepted']
    assert len(calls) == 16
    for first, second in zip(calls[:8], calls[8:]):
        assert first['step'] == second['step']
        assert torch.equal(first['data_rng'], second['data_rng'])
        assert first['d_lr'] == second['d_lr']
        assert first['prior_lr'] == second['prior_lr']
        assert second['g_lr'] == .1 * first['g_lr']
    assert_restored(trainer, state)


def test_no_repeated_search_when_confirmation_fails(monkeypatch):
    import hypergan.startup_dynamics as dynamics
    trainer, calls = saturating_optimizer_fixture()
    state = copy.deepcopy(trainer_state(trainer, None))
    original_guards = dynamics._guards
    def reject_confirmation(before, after):
        reasons, comparisons = original_guards(before, after)
        return reasons or ['heldout_confirmation_failed'], comparisons
    monkeypatch.setattr(dynamics, '_guards', reject_confirmation)
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'unresolved'
    assert report['selected_g_lr_factor'] == 1
    assert len(calls) == 16 and len(report['candidates']) == 2
    assert_restored(trainer, state)


def test_callback_exception_after_real_updates_restores_partial_trial():
    trainer = make_trainer()
    state = copy.deepcopy(trainer_state(trainer, None))
    def progress(event):
        if event['trial_step'] == 3:
            raise RuntimeError('observer failed')
    with pytest.raises(RuntimeError, match='observer failed'):
        tune_startup_dynamics(trainer, progress=progress)
    assert_restored(trainer, state)


def test_stateful_data_cursor_is_restored_even_after_reserved_bank_draws():
    trainer, _ = saturating_optimizer_fixture()
    source = trainer.data
    class Stateful:
        def __init__(self):
            self.state = {'cursor': [0]}
        def __call__(self, *args, **kwargs):
            self.state['cursor'][0] += 1
            return source(*args, **kwargs)
        def state_dict(self):
            return self.state
        def load_state_dict(self, state):
            # A legal loader that retains the mutable dictionary it is given.
            self.state = state
    trainer.data = Stateful()
    state = copy.deepcopy(trainer_state(trainer, None))
    report = tune_startup_dynamics(trainer)
    assert report['outcome'] == 'selected'
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
    trainer = make_trainer()
    parameter = next(trainer.graph.models['discriminator'].parameters())
    parameter.requires_grad_(False)
    state = copy.deepcopy(trainer_state(trainer, None))
    original_update = trainer.update
    def mutate():
        result = original_update()
        with torch.no_grad():
            parameter.add_(1.)
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
    assert report['selected_g_lr_factor'] == 1
    assert len(report['candidates']) == 1
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

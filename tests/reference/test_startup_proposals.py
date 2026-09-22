"""Actual optimizer effects and exact rollback for explicit research proposals."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from hypergan.checkpoints import trainer_state
from hypergan.config import resolve_config
from hypergan.startup_dynamics import _restore, _same_state, _snapshot
from hypergan.training import ReferenceTrainer

spec = importlib.util.spec_from_file_location('research_startup_proposals', Path(__file__).resolve().parents[2] / 'research/startup_tuning/proposals.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
apply_proposal = module.apply_proposal


def make_trainer():
    return ReferenceTrainer(resolve_config({'training': {'steps': 32, 'batch_size': 8},
                                           'sampling': {'count': 4}}))


def names(trainer):
    return {id(p): 'graph.' + name for name, p in trainer.graph.named_parameters()}


def assert_restored(trainer, entry):
    assert _same_state(entry['state'], trainer_state(trainer, None))
    for parameter, original, saved in entry['gradients']:
        assert parameter.grad is original
        assert saved is None or _same_state(saved, parameter.grad)


def test_actual_adam_layer_update_scales_with_rate_after_moments_accumulate():
    trainer = make_trainer()
    parameter = trainer.program.generator_parameters[0]
    target = names(trainer)[id(parameter)]
    initial = _snapshot(trainer)
    before = parameter.detach().clone()
    original_prior_group = trainer.opt_g.param_groups[1]
    prior_rate = original_prior_group['lr']

    def updates():
        for step in range(3):
            trainer.opt_g.zero_grad(set_to_none=True)
            for group in trainer.opt_g.param_groups:
                for value in group['params']:
                    value.grad = torch.full_like(value, 1. + step)
            trainer.opt_g.step()

    updates()
    control_delta = parameter.detach() - before
    _restore(trainer, initial)
    original_prior_group = trainer.opt_g.param_groups[1]
    entry_groups = trainer.opt_g.param_groups
    with apply_proposal(trainer, {'schema_version': 1,
            'layer_lr_multipliers': [{'pattern': target, 'multiplier': .25}]}) as resolved:
        assert trainer.opt_g.param_groups[1] is original_prior_group
        assert trainer.base_lrs[0][1] == prior_rate
        updates()
        assert float(trainer.opt_g.state[parameter]['step']) == 3.
        torch.testing.assert_close(parameter.detach() - before, .25 * control_delta, rtol=5e-4, atol=3e-8)
        row = next(row for row in resolved['optimizers'][0]['parameters'] if row['path'] == target)
        assert row['effective_lr'] == trainer.config['optimizer']['lr'] * .25
    assert trainer.opt_g.param_groups is entry_groups
    assert trainer.opt_g.param_groups[1] is original_prior_group
    assert_restored(trainer, initial)


def test_native_updates_and_snapshot_restore_work_with_both_split_optimizers():
    trainer = make_trainer()
    paths = names(trainer)
    g = trainer.program.generator_parameters[0]
    d = trainer.program.critic_parameters[0]
    plan = {'schema_version': 1, 'g_lr': .0001, 'd_lr': .0003,
            'layer_lr_multipliers': [{'pattern': paths[id(g)], 'multiplier': .2},
                                     {'pattern': paths[id(d)], 'multiplier': .5}]}
    entry = _snapshot(trainer)
    with apply_proposal(trainer, plan):
        assert len(trainer.opt_g.param_groups) == 3
        assert len(trainer.opt_d.param_groups) == 2
        assert trainer.base_lrs[0][1] == entry['state']['base_lrs'][0][1]
        split_start = _snapshot(trainer)
        trainer.update()
        after = _snapshot(trainer)
        _restore(trainer, split_start)
        trainer.update()
        assert _same_state(after['state'], trainer_state(trainer, None))
        for parameter, _, saved in after['gradients']:
            assert saved is None or _same_state(saved, parameter.grad)
        _restore(trainer, after)
        assert_restored(trainer, after)
    assert_restored(trainer, entry)


def test_initialization_exact_parameters_and_ema_then_exception_restores_everything():
    trainer = make_trainer()
    target = trainer.program.generator_parameters[0]
    target.grad = torch.ones_like(target)
    path = names(trainer)[id(target)]
    ema = dict(trainer.ema_graph.named_parameters())[path.removeprefix('graph.')]
    before = target.detach().clone()
    entry = _snapshot(trainer)
    with pytest.raises(RuntimeError, match='injected'):
        with apply_proposal(trainer, {'schema_version': 1, 'g_lr': .0001,
                'init_scales': [{'pattern': path, 'multiplier': .3}],
                'evidence': {'purpose': 'explicit test'}}) as resolved:
            torch.testing.assert_close(target, before * .3, rtol=0, atol=0)
            torch.testing.assert_close(ema, target, rtol=0, atol=0)
            assert resolved['initialization_parameters'][0]['path'] == path
            trainer.update()
            torch.rand(3, generator=trainer.streams['prior'])
            raise RuntimeError('injected')
    assert_restored(trainer, entry)


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf'), True])
def test_invalid_factors_reject_before_any_mutation(invalid):
    trainer = make_trainer()
    path = names(trainer)[id(trainer.program.generator_parameters[0])]
    entry = _snapshot(trainer)
    with pytest.raises(ValueError):
        with apply_proposal(trainer, {'schema_version': 1, 'g_lr': .0001,
                'init_scales': [{'pattern': path, 'multiplier': invalid}]}):
            pytest.fail('invalid plan yielded')
    assert_restored(trainer, entry)


@pytest.mark.parametrize('case', ['zero_match', 'prior', 'frozen', 'alias', 'overlap', 'overflow'])
def test_protected_unknown_and_ambiguous_patterns_are_atomic(case):
    trainer = make_trainer()
    target = trainer.program.generator_parameters[0]
    path = names(trainer)[id(target)]
    multiplier = .5
    if case == 'zero_match':
        pattern = 'graph.models.no_such_parameter.*'
    elif case == 'prior':
        pattern = 'prior.*'
    elif case == 'frozen':
        trainer.graph.register_parameter('foreign_frozen', torch.nn.Parameter(torch.ones(1), requires_grad=False))
        pattern = 'graph.foreign_frozen'
    elif case == 'alias':
        trainer.graph.register_buffer('foreign_frozen_alias', target.detach())
        pattern = path
    else:
        pattern = path
    if case == 'overflow':
        multiplier = 1e300
    rules = [{'pattern': pattern, 'multiplier': multiplier}]
    if case == 'overlap':
        rules.append(copy.deepcopy(rules[0]))
    entry = _snapshot(trainer)
    with pytest.raises((ValueError, RuntimeError)):
        with apply_proposal(trainer, {'schema_version': 1, 'g_lr': .0001, 'init_scales': rules}):
            pytest.fail('unsafe proposal yielded')
    assert_restored(trainer, entry)


def test_nonfresh_trainer_and_warmup_are_rejected():
    trainer = make_trainer()
    trainer.g_lr_warmup = {'steps': 4, 'start_g_lr': .0001, 'target_g_lr': .0002}
    with pytest.raises(ValueError, match='fresh'):
        with apply_proposal(trainer, {'schema_version': 1}):
            pass
    del trainer.g_lr_warmup
    trainer.update()
    with pytest.raises(ValueError, match='fresh'):
        with apply_proposal(trainer, {'schema_version': 1}):
            pass

"""Optional G-only startup ramp has exact endpoints and resumable state."""
from copy import deepcopy

import pytest
from particlegan import learning_rate_scale

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.tuning_overrides import generator_lr_override, optimizer_lr_override, scheduled_generator_lr
from .test_recovery import equal


def _fixture(d_factor=None):
    config = resolve_config({'training': {'device': 'cpu', 'steps': 10, 'lr_anneal_start': .2,
                                         'lr_floor': .5},
                             'prior': {'args': {'num_particles': 32, 'z_dim': 4}}})
    trainer = ReferenceTrainer(config)
    baseline = deepcopy(trainer.base_lrs)
    override = (generator_lr_override(config, .3) if d_factor is None
                else optimizer_lr_override(config, .3, d_factor))
    warmup = {'steps': 5, 'start_g_lr': override['effective_g_lr'],
              'target_g_lr': override['baseline_g_lr']}
    trainer.g_lr_warmup = deepcopy(warmup)
    trainer.base_lrs[0][0] = trainer.opt_g.param_groups[0]['lr'] = warmup['start_g_lr']
    if d_factor is not None:
        trainer.base_lrs[1][0] = trainer.opt_d.param_groups[0]['lr'] = override['effective_d_lr']
    metadata = {'initialization_tuning': {'status': 'complete', 'dynamics_outcome': 'selected',
                                         'selected_g_lr_factor': .3, 'optimizer_override': override,
                                         'g_lr_warmup': warmup}}
    if d_factor is not None:
        metadata['initialization_tuning']['selected_d_lr_factor'] = d_factor
    return trainer, baseline, metadata


def test_ramp_starts_on_first_update_reaches_target_and_changes_only_g_main_rate():
    trainer, baseline, _ = _fixture()
    assert len(trainer.opt_g.param_groups) == 2  # Learned prior has its own absolute rate.
    start, target = trainer.g_lr_warmup['start_g_lr'], baseline[0][0]
    expected = [start, start + (target - start) / 4, start + (target - start) / 2,
                start + (target - start) * .75, target, target]
    assert scheduled_generator_lr(start, trainer.g_lr_warmup, 0) == start
    assert scheduled_generator_lr(start, None, 100) == start
    for step, expected_g in enumerate(expected, start=1):
        trainer.update()
        scale = learning_rate_scale(step - 1, 10, start=.2, floor=.5)
        assert trainer.opt_g.param_groups[0]['lr'] == pytest.approx(expected_g * scale)
        assert trainer.opt_g.param_groups[1]['lr'] == baseline[0][1] * scale
        assert trainer.opt_d.param_groups[0]['lr'] == baseline[1][0] * scale
    assert trainer.base_lrs[0][0] == start


@pytest.mark.parametrize('split', [0, 1, 3, 5, 6])
@pytest.mark.parametrize('d_factor', [None, .5])
def test_resume_continues_identical_ramp_and_training_without_reapplying_factor(split, d_factor):
    uninterrupted, baseline, metadata = _fixture(d_factor)
    batch = None
    for _ in range(split):
        _, batch = uninterrupted.update()
    saved = deepcopy(trainer_state(uninterrupted, batch))
    for _ in range(7 - split):
        _, batch = uninterrupted.update()
    expected = deepcopy(trainer_state(uninterrupted, batch))
    resumed = ReferenceTrainer(uninterrupted.config)
    batch = restore_trainer(resumed, saved, metadata=metadata)
    assert resumed.g_lr_warmup == metadata['initialization_tuning']['g_lr_warmup']
    assert resumed.g_lr_warmup is not metadata['initialization_tuning']['g_lr_warmup']
    for _ in range(7 - split):
        _, batch = resumed.update()
    equal(trainer_state(resumed, batch), expected)
    assert resumed.base_lrs[1] == [rate * (d_factor or 1.) for rate in baseline[1]]
    assert resumed.base_lrs[0][1:] == baseline[0][1:]


@pytest.mark.parametrize('change', ['missing-override', 'steps-small', 'steps-bool', 'steps-float',
                                   'start', 'target', 'nonfinite', 'extra', 'null', 'live-g', 'live-prior'])
def test_invalid_ramp_or_saved_rate_is_rejected_before_model_restore(change):
    trainer, _, metadata = _fixture()
    _, batch = trainer.update()
    _, batch = trainer.update()
    saved = deepcopy(trainer_state(trainer, batch))
    tuning = metadata['initialization_tuning']
    warmup = tuning['g_lr_warmup']
    if change == 'missing-override':
        del tuning['optimizer_override']
    elif change.startswith('steps-'):
        warmup['steps'] = {'steps-small': 1, 'steps-bool': True, 'steps-float': 5.0}[change]
    elif change == 'start':
        warmup['start_g_lr'] *= .5
    elif change == 'target':
        warmup['target_g_lr'] *= 2
    elif change == 'nonfinite':
        warmup['target_g_lr'] = float('nan')
    elif change == 'extra':
        warmup['scope'] = 'all-optimizers'
    elif change == 'null':
        tuning['g_lr_warmup'] = None
    else:
        index = 0 if change == 'live-g' else 1
        saved['optimizers'][0]['param_groups'][index]['lr'] *= .5
    target = ReferenceTrainer(trainer.config)
    before = deepcopy(target.graph.state_dict())
    with pytest.raises(ValueError, match='learning.rate'):
        restore_trainer(target, saved, metadata=metadata)
    equal(target.graph.state_dict(), before)


def test_legacy_untuned_restore_clears_any_in_memory_schedule():
    trainer, _, _ = _fixture()
    baseline = ReferenceTrainer(trainer.config)
    saved = deepcopy(trainer_state(baseline, None))
    target = ReferenceTrainer(trainer.config)
    target.g_lr_warmup = trainer.g_lr_warmup
    restore_trainer(target, saved)
    assert target.g_lr_warmup is None

"""Only completed, validated startup metadata may change restored G/D rates."""
from copy import deepcopy

import pytest
import torch

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.tuning_overrides import generator_lr_override, optimizer_lr_override
from .test_recovery import equal


def _fixture(step=0, factor=.3, d_factor=None, *, version=None):
    config = resolve_config({'training': {'device': 'cpu', 'steps': 10, 'lr_anneal_start': .2, 'lr_floor': .5},
                             'prior': {'args': {'num_particles': 32, 'z_dim': 4}}})
    trainer = ReferenceTrainer(config)
    baseline = deepcopy(trainer.base_lrs)
    override = (generator_lr_override(config, factor) if d_factor is None
                else optimizer_lr_override(config, factor, d_factor))
    if version is not None:
        override['schema_version'] = version
    trainer.base_lrs[0][0] = trainer.opt_g.param_groups[0]['lr'] = override['effective_g_lr']
    if d_factor is not None:
        trainer.base_lrs[1][0] = trainer.opt_d.param_groups[0]['lr'] = override['effective_d_lr']
    batch = None
    for _ in range(step):
        _, batch = trainer.update()
    state = deepcopy(trainer_state(trainer, batch))
    metadata = {'initialization_tuning': {'status': 'complete', 'dynamics_outcome': 'selected',
                                         'selected_g_lr_factor': factor, 'optimizer_override': override}}
    if d_factor is not None:
        metadata['initialization_tuning']['selected_d_lr_factor'] = d_factor
    if factor == 1 and d_factor in (None, 1):
        metadata['initialization_tuning']['dynamics_outcome'] = 'kept_baseline'
    return config, baseline, state, metadata


@pytest.mark.parametrize('step', [0, 6])
def test_tuned_rates_restore_exactly_at_zero_and_inside_annealing(step):
    config, baseline, state, metadata = _fixture(step)
    target = ReferenceTrainer(config)
    batch = restore_trainer(target, state, metadata=metadata)
    equal(trainer_state(target, batch), state)
    assert target.base_lrs[0][0] == baseline[0][0] * .3
    assert target.base_lrs[0][1:] == baseline[0][1:]
    assert target.base_lrs[1] == baseline[1]


@pytest.mark.parametrize('change', ['missing', 'removed', 'status', 'scope', 'factor-bool', 'factor-nan',
    'factor-small', 'factor-large', 'baseline', 'effective', 'hash', 'outcome', 'selection', 'extra'])
def test_invalid_override_metadata_cannot_relax_original_rate_contract(change):
    config, _, state, metadata = _fixture()
    tuning = metadata['initialization_tuning']
    override = tuning['optimizer_override']
    if change == 'missing':
        metadata = None
    elif change == 'removed':
        del tuning['optimizer_override']
    elif change == 'status':
        tuning['status'] = 'running'
    elif change == 'scope':
        override['scope'] = 'prior'
    elif change.startswith('factor-'):
        override['factor'] = {'factor-bool': True, 'factor-nan': float('nan'),
                              'factor-small': .01, 'factor-large': 1.1}[change]
    elif change == 'baseline':
        override['baseline_g_lr'] *= 2
    elif change == 'effective':
        override['effective_g_lr'] *= 2
    elif change == 'hash':
        override['optimizer_config_sha256'] = '0' * 64
    elif change == 'outcome':
        tuning['dynamics_outcome'] = 'unresolved'
    elif change == 'selection':
        tuning['selected_g_lr_factor'] = .1
    else:
        override['extra'] = 'unsupported'
    target = ReferenceTrainer(config)
    before = deepcopy(target.graph.state_dict())
    with pytest.raises(ValueError):
        restore_trainer(target, state, metadata=metadata)
    equal(target.graph.state_dict(), before)


@pytest.mark.parametrize('change', ['prior-base', 'discriminator-base', 'g-live-rate', 'd-live-rate'])
def test_override_never_permits_other_base_rates_or_unscheduled_live_rates(change):
    config, _, state, metadata = _fixture()
    if change == 'prior-base':
        assert len(state['base_lrs'][0]) == 2
        state['base_lrs'][0][1] *= .3
    elif change == 'discriminator-base':
        state['base_lrs'][1][0] *= .3
    elif change == 'g-live-rate':
        state['optimizers'][0]['param_groups'][0]['lr'] *= .3
    else:
        state['optimizers'][1]['param_groups'][0]['lr'] *= .3
    with pytest.raises(ValueError, match='learning rate'):
        restore_trainer(ReferenceTrainer(config), state, metadata=metadata)


def test_formulaic_factor_is_supported_without_compounding():
    config, baseline, state, metadata = _fixture(factor=.271828)
    first = ReferenceTrainer(config)
    restore_trainer(first, state, metadata=metadata)
    assert first.base_lrs[0][0] == baseline[0][0] * .271828
    with pytest.raises(ValueError, match='original configuration'):
        restore_trainer(first, state, metadata=metadata)


@pytest.mark.parametrize('step', [0, 6])
@pytest.mark.parametrize(('g_factor', 'd_factor'), [(1., .5), (.3, 1.), (.3, .5), (1., 1.)])
def test_schema_two_restores_recorded_main_rates_without_changing_prior(step, g_factor, d_factor):
    config, baseline, state, metadata = _fixture(step, g_factor, d_factor, version=2)
    assert metadata['initialization_tuning']['optimizer_override']['schema_version'] == 2
    target = ReferenceTrainer(config)
    batch = restore_trainer(target, state, metadata=metadata)
    equal(trainer_state(target, batch), state)
    assert target.base_lrs[0][0] == baseline[0][0] * g_factor
    assert target.base_lrs[1][0] == baseline[1][0] * d_factor
    assert target.base_lrs[0][1:] == baseline[0][1:]
    assert target.opt_g.param_groups[1]['lr'] == state['optimizers'][0]['param_groups'][1]['lr']


@pytest.mark.parametrize('change', ['missing-selection', 'selection', 'd-bool', 'd-nan', 'd-small', 'd-large',
                                   'baseline', 'effective', 'outcome', 'prior-base', 'd-live', 'g-live',
                                   'downgrade-schema', 'scope'])
def test_schema_two_rejects_inconsistent_d_override_before_restoring_models(change):
    config, _, state, metadata = _fixture(step=3, factor=1., d_factor=.5, version=2)
    tuning = metadata['initialization_tuning']
    override = tuning['optimizer_override']
    if change == 'missing-selection':
        del tuning['selected_d_lr_factor']
    elif change == 'selection':
        tuning['selected_d_lr_factor'] = 1.
    elif change in ('d-bool', 'd-nan', 'd-small', 'd-large'):
        override['d_factor'] = {'d-bool': True, 'd-nan': float('nan'), 'd-small': .01, 'd-large': 2.}[change]
    elif change in ('baseline', 'effective'):
        override[change + '_d_lr'] *= 2
    elif change == 'outcome':
        tuning['dynamics_outcome'] = 'kept_baseline'
    elif change == 'prior-base':
        state['base_lrs'][0][1] *= .5
    elif change == 'd-live':
        state['optimizers'][1]['param_groups'][0]['lr'] *= .5
    elif change == 'g-live':
        state['optimizers'][0]['param_groups'][0]['lr'] *= .5
    elif change == 'downgrade-schema':
        override['schema_version'] = 1
    elif change == 'scope':
        override['scope'] = 'generator-main-group'
    target = ReferenceTrainer(config)
    before = deepcopy(target.graph.state_dict())
    with pytest.raises(ValueError):
        restore_trainer(target, state, metadata=metadata)
    equal(target.graph.state_dict(), before)


@pytest.mark.parametrize('step', [0, 6])
def test_schema_three_small_factors_resume_without_compounding_or_changing_prior(step):
    config, baseline, state, metadata = _fixture(step, .003, .025)
    assert metadata['initialization_tuning']['optimizer_override']['schema_version'] == 3
    target = ReferenceTrainer(config)
    batch = restore_trainer(target, state, metadata=metadata)
    equal(trainer_state(target, batch), state)
    assert target.base_lrs[0][0] == baseline[0][0] * .003
    assert target.base_lrs[1][0] == baseline[1][0] * .025
    assert target.base_lrs[0][1:] == baseline[0][1:]
    with pytest.raises(ValueError, match='original configuration'):
        restore_trainer(target, state, metadata=metadata)


def test_schema_three_small_rate_split_resume_matches_continuous_training():
    config, _, state, metadata = _fixture(3, .003, .025)
    continued = ReferenceTrainer(config)
    restore_trainer(continued, deepcopy(state), metadata=metadata)
    for _ in range(2):
        _, batch = continued.update()
    expected = deepcopy(trainer_state(continued, batch))
    resumed = ReferenceTrainer(config)
    restore_trainer(resumed, deepcopy(state), metadata=metadata)
    _, batch = resumed.update()
    intermediate = deepcopy(trainer_state(resumed, batch))
    resumed = ReferenceTrainer(config)
    restore_trainer(resumed, intermediate, metadata=metadata)
    _, batch = resumed.update()
    equal(trainer_state(resumed, batch), expected)


@pytest.mark.parametrize('player', ['generator', 'discriminator'])
@pytest.mark.parametrize('value', [0., -1., True, float('nan'), float('inf'), 1.01, 5e-324])
def test_schema_three_writer_refuses_invalid_or_underflowing_effective_rates(player, value):
    config = resolve_config({})
    factors = (value, 1.) if player == 'generator' else (1., value)
    with pytest.raises(ValueError, match=player):
        optimizer_lr_override(config, *factors)


@pytest.mark.parametrize('player', ['generator', 'discriminator'])
@pytest.mark.parametrize('value', [0., float('nan'), 5e-324])
def test_schema_three_reader_refuses_invalid_or_underflowing_effective_rates(player, value):
    config, _, state, metadata = _fixture(factor=.003, d_factor=.025)
    tuning = metadata['initialization_tuning']
    override = tuning['optimizer_override']
    field, rate_name, index = ('factor', 'g', 0) if player == 'generator' else ('d_factor', 'd', 1)
    effective = override[f'baseline_{rate_name}_lr'] * value
    override[field] = tuning[f'selected_{rate_name}_lr_factor'] = value
    override[f'effective_{rate_name}_lr'] = effective
    state['base_lrs'][index][0] = state['optimizers'][index]['param_groups'][0]['lr'] = effective
    target = ReferenceTrainer(config)
    before = deepcopy(target.graph.state_dict())
    with pytest.raises(ValueError, match=player):
        restore_trainer(target, state, metadata=metadata)
    equal(target.graph.state_dict(), before)


@pytest.mark.parametrize('version,player', [(1, 'generator'), (2, 'generator'), (2, 'discriminator')])
def test_historical_schemas_keep_the_original_factor_floor(version, player):
    config, _, state, metadata = _fixture(d_factor=None if version == 1 else .5, version=version)
    tuning = metadata['initialization_tuning']
    override = tuning['optimizer_override']
    field, rate_name, index = ('factor', 'g', 0) if player == 'generator' else ('d_factor', 'd', 1)
    override[field] = tuning[f'selected_{rate_name}_lr_factor'] = .01
    effective = override[f'baseline_{rate_name}_lr'] * .01
    override[f'effective_{rate_name}_lr'] = effective
    state['base_lrs'][index][0] = state['optimizers'][index]['param_groups'][0]['lr'] = effective
    with pytest.raises(ValueError, match=player):
        restore_trainer(ReferenceTrainer(config), state, metadata=metadata)

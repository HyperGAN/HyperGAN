"""Only completed, validated startup metadata may change restored G base rates."""
from copy import deepcopy

import pytest
import torch

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.tuning_overrides import generator_lr_override
from .test_recovery import equal


def _fixture(step=0, factor=.3):
    config = resolve_config({'training': {'device': 'cpu', 'steps': 10, 'lr_anneal_start': .2, 'lr_floor': .5},
                             'prior': {'args': {'num_particles': 32, 'z_dim': 4}}})
    trainer = ReferenceTrainer(config)
    baseline = deepcopy(trainer.base_lrs)
    override = generator_lr_override(config, factor)
    trainer.base_lrs[0][0] = trainer.opt_g.param_groups[0]['lr'] = override['effective_g_lr']
    batch = None
    for _ in range(step):
        _, batch = trainer.update()
    state = deepcopy(trainer_state(trainer, batch))
    metadata = {'initialization_tuning': {'status': 'complete', 'dynamics_outcome': 'selected',
                                         'selected_g_lr_factor': factor, 'optimizer_override': override}}
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

"""Only audited dependency source bytes qualify the explicit-sigma migration."""
from copy import deepcopy

import pytest

from hypergan.checkpoint_compatibility import (
    PARTICLEGAN_060_MIGRATION,
    validate_implementation,
    validate_runtime,
)


def implementations():
    saved = {key: old for key, (old, _) in PARTICLEGAN_060_MIGRATION.items()}
    current = {key: new for key, (_, new) in PARTICLEGAN_060_MIGRATION.items()}
    saved['particlegan.grad_regularizers'] = current['particlegan.grad_regularizers'] = 'same'
    return saved, current


def test_qualified_upgrade_warns_and_does_not_mutate_metadata():
    saved, current = implementations()
    before = deepcopy((saved, current))
    messages = []
    assert validate_runtime({'particlegan': '0.5.0'}, {'particlegan': '0.6.0'},
                            saved_implementation=saved, current_implementation=current,
                            warn=messages.append) == messages
    assert len(messages) == 1 and 'audited 0.6.0' in messages[0]
    validate_implementation(saved, current)
    assert (saved, current) == before


@pytest.mark.parametrize('damage', ['old_prior', 'new_prior', 'recipe', 'penalty', 'missing', 'added'])
def test_unknown_dependency_changes_remain_rejected(damage):
    saved, current = implementations()
    if damage == 'old_prior':
        saved['particlegan.particle_prior'] = 'unknown'
    elif damage == 'new_prior':
        current['particlegan.particle_prior'] = 'unknown'
    elif damage == 'recipe':
        current['particlegan.recipes'] = 'unknown'
    elif damage == 'penalty':
        current['particlegan.grad_regularizers'] = 'changed'
    elif damage == 'missing':
        saved.pop('particlegan.recipes')
    else:
        current['custom.loss'] = 'new'
    with pytest.raises(ValueError):
        validate_runtime({'particlegan': '0.5.0'}, {'particlegan': '0.6.0'},
                         saved_implementation=saved, current_implementation=current)
    with pytest.raises(ValueError, match='implementation'):
        validate_implementation(saved, current)


def test_reverse_upgrade_and_other_runtime_changes_are_not_qualified():
    saved, current = implementations()
    with pytest.raises(ValueError, match='runtime'):
        validate_runtime({'particlegan': '0.6.0'}, {'particlegan': '0.5.0'},
                         saved_implementation=current, current_implementation=saved)
    with pytest.raises(ValueError, match='implementation'):
        validate_implementation(current, saved)
    with pytest.raises(ValueError, match='torch'):
        validate_runtime({'particlegan': '0.5.0', 'torch': 'old'},
                         {'particlegan': '0.6.0', 'torch': 'changed'},
                         saved_implementation=saved, current_implementation=current)

"""Pinned training exclusions survive restore without changing healthy sampling."""
import json
from concurrent.futures import wait
from copy import deepcopy

import pytest
import torch
from PIL import Image

from hypergan.checkpoints import read_checkpoint
from hypergan.colorization_data import ColorizationData, prepare_manifest
from hypergan.config import (
    config_values,
    load_config,
    resolve_config,
    resume_compatible,
)
from hypergan.training import resume, train
from tests.reference.test_image_recovery import _equal, _project


@pytest.fixture
def pinned(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    for i in range(30):
        Image.new('RGB', (2, 2), (i * 7, i, 255 - i)).save(root / f'{i:02}.png')
    manifest = tmp_path / 'inventory.json'
    info = prepare_manifest(root, manifest, height=2, width=2, workers=2)
    return {'root': root, 'manifest': manifest, 'manifest_sha256': info['sha256']}


@pytest.mark.parametrize('workers', [0, 1, 4])
def test_missing_changed_and_prefetched_images_resume_exactly(pinned, workers, caplog):
    strict = ColorizationData(**pinned, workers=0)
    data = ColorizationData(**pinned, workers=workers, bad_image_policy='skip')
    rng = torch.Generator().manual_seed(71)
    restored = None
    try:
        # Healthy sampling and a legacy state retain their exact meaning.
        legacy = strict.state_dict()
        data.load_state_dict(legacy)
        expected = strict(3, generator=torch.Generator().manual_seed(71))
        actual = data(3, generator=rng)
        assert torch.equal(actual['real'], expected['real'])
        wait(data._pending.values())
        order = data._permutation[data._cursor:]
        missing, changed = order[:2]
        deleted = pinned['root'] / data.entries[missing]['path']
        repaired = pinned['root'] / data.entries[changed]['path']
        original = repaired.read_bytes()
        deleted.unlink()
        repaired.write_bytes(b'corrupt after prefetch')
        batch = data(5, generator=rng)
        assert batch['real'].shape == (5, 3, 2, 2)
        assert set(data._excluded) == {missing, changed}
        assert 'content changed' in data._excluded[changed]
        assert 'No such file' in data._excluded[missing]
        assert caplog.text.count('Skipping image') == 2
        state, rng_state = data.state_dict(), rng.get_state()
        assert state['schema_version'] == 2
        # Repairing an excluded source must not put it back into this run.
        repaired.write_bytes(original)
        restored = ColorizationData(**pinned, workers=0, bad_image_policy='skip')
        restored.load_state_dict(json.loads(json.dumps(state)))
        other_rng = torch.Generator().set_state(rng_state)
        for size in (1, 55, 7):
            expected = data(size, generator=rng)
            actual = restored(size, generator=other_rng)
            assert all(torch.equal(actual[k], expected[k]) for k in actual)
            assert data.state_dict() == restored.state_dict()
            assert torch.equal(rng.get_state(), other_rng.get_state())
        assert caplog.text.count('Skipping image') == 2
    finally:
        strict.close()
        data.close()
        if restored:
            restored.close()


@pytest.mark.parametrize('limit', ['total', 'consecutive', 'all'])
@pytest.mark.parametrize('workers', [0, 4])
def test_failure_limits_rollback_sampler_rng_and_exclusions(pinned, limit, workers):
    data = ColorizationData(**pinned, workers=workers, bad_image_policy='skip',
                            max_bad_images=1 if limit == 'total' else 100,
                            max_consecutive_bad_images=2 if limit == 'consecutive' else 100)
    rng = torch.Generator().manual_seed(71)
    try:
        data(1, generator=rng)
        state, before_rng = data.state_dict(), rng.get_state()
        if limit == 'all':
            targets = list(range(len(data.entries)))
        else:
            # Separate failures for total-limit coverage; consecutive for streak.
            offsets = (0, 2) if limit == 'total' else (0, 1)
            targets = [data._permutation[data._cursor + offset] for offset in offsets]
        for index in targets:
            (pinned['root'] / data.entries[index]['path']).unlink()
        with pytest.raises(ValueError, match='Image failure limit reached'):
            data(len(data.entries) + 2, generator=rng)
        assert data.state_dict() == state
        assert torch.equal(rng.get_state(), before_rng)
        assert not data._pending
    finally:
        data.close()


def test_programming_errors_are_not_skipped(pinned, monkeypatch):
    data = ColorizationData(**pinned, workers=4, bad_image_policy='skip')
    rng = torch.Generator()
    state, rng_state = data.state_dict(), rng.get_state()
    def broken(entry):
        raise ValueError('implementation bug')
    monkeypatch.setattr(data, '_pixels', broken)
    try:
        with pytest.raises(ValueError, match='implementation bug'):
            data(3, generator=rng)
        assert data.state_dict() == state
        assert torch.equal(rng.get_state(), rng_state)
    finally:
        data.close()


@pytest.mark.parametrize('kwargs', [
    {'bad_image_policy': 'ignore'}, {'max_bad_images': 0}, {'max_bad_images': True},
    {'max_consecutive_bad_images': 1.5}, {'split': 'heldout'}, {'shuffle': False},
])
def test_invalid_settings_and_evaluation_reject_skipping(pinned, kwargs):
    with pytest.raises(ValueError):
        ColorizationData(**pinned, **({'bad_image_policy': 'skip'} | kwargs))


@pytest.mark.parametrize('damage', ['index', 'path', 'reason', 'duplicate', 'streak', 'strict'])
def test_invalid_exclusions_do_not_mutate_loader(pinned, damage):
    data = ColorizationData(**pinned, bad_image_policy='error' if damage == 'strict' else 'skip')
    rng = torch.Generator()
    try:
        data(1, generator=rng)
        before = data.state_dict()
        state = deepcopy(before)
        record = {'index': 0, 'path': data.entries[0]['path'], 'reason': 'missing'}
        state.update(schema_version=2, excluded=[record], bad_image_streak=0)
        if damage == 'index':
            record['index'] = True
        elif damage == 'path':
            record['path'] = '../outside'
        elif damage == 'reason':
            record['reason'] = ''
        elif damage == 'duplicate':
            state['excluded'].append(record.copy())
        elif damage == 'streak':
            state['bad_image_streak'] = 8
        with pytest.raises(ValueError):
            data.load_state_dict(state)
        assert data.state_dict() == before
    finally:
        data.close()


def test_resume_opt_in_is_narrow_and_preserves_original_config(pinned):
    args = {k: str(v) for k, v in pinned.items()}
    original = resolve_config({'data': {'factory': 'hypergan.colorization_data:ColorizationData', 'args': args}})
    saved = deepcopy(original)
    current = deepcopy(original)
    current['data']['args'].update(bad_image_policy='skip', max_bad_images=100,
                                   max_consecutive_bad_images=8)
    assert resume_compatible(current, original, include_observation=True)
    assert original == saved
    assert not resume_compatible(original, current)
    for section, key, value in [('data', 'manifest_sha256', 'changed'),
                                ('data', 'shuffle', False), ('data', 'max_bad_images', True),
                                ('training', 'lr', .5)]:
        changed = deepcopy(current)
        target = changed['data']['args'] if section == 'data' else changed[section]
        target[key] = value
        assert not resume_compatible(changed, original)
    changed = deepcopy(current)
    changed['data']['args']['max_bad_images'] = 101
    assert not resume_compatible(changed, current)


def test_training_can_enable_skipping_then_resume_full_checkpoint(tmp_path):
    config, root = _project(tmp_path)
    manifest = tmp_path / 'inventory.json'
    info = prepare_manifest(root, manifest, height=2, width=2, workers=2)
    source = config.read_text()
    start, end = source.index('[data]'), source.index('[components.generator]')
    source = source[:start] + f'''[data]
factory = "hypergan.colorization_data:ColorizationData"
[data.args]
root = {json.dumps(str(root))}
manifest = {json.dumps(str(manifest))}
manifest_sha256 = "{info['sha256']}"
''' + source[end:]
    config.write_text(source)
    left, right = tmp_path / 'continuous', tmp_path / 'resumed'
    for run in (left, right):
        train(config, run, stop_after_steps=2, checkpoint_every=1)
    old_config = config_values(load_config(config))
    data = ColorizationData(root, manifest, info['sha256'])
    (root / data.entries[0]['path']).unlink()
    config.write_text(source.replace('[data.args]', '[data.args]\nbad_image_policy = "skip"'))
    finished = resume(left, config_path=config, require_same_config=True)
    stopped = resume(right, config_path=config, require_same_config=True, stop_after_steps=2)
    assert stopped['last_durable_step'] == 4
    assert stopped['config']['data']['args']['bad_image_policy'] == 'skip'
    completed = resume(right)
    assert finished['status'] == completed['status'] == 'complete'
    _, _, expected = read_checkpoint(left)
    _, _, actual = read_checkpoint(right)
    assert len(actual['data']['excluded']) == 1
    assert actual['data']['schema_version'] == 2
    _equal(actual, expected)
    assert 'bad_image_policy' not in old_config['data']['args']

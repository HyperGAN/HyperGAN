"""Fault injection across the event/payload/pointer publication transaction."""
import json
from pathlib import Path

import pytest
import torch

from hypergan.checkpoints import read_checkpoint
from hypergan.config import write_default
from hypergan.run_state import validate_event_boundary
from hypergan.training import train, resume


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('stage', ['events', 'payload', 'rename', 'pointer', 'after-pointer', 'manifest'])
def test_failure_at_publication_stages_has_complete_recoverable_prefix(tmp_path, monkeypatch, stage):
    import hypergan.checkpoints as checkpoints
    import hypergan.run_controller as controller
    import hypergan.observation_io as observation_io
    config = write_default(tmp_path / 'config', device='cpu')
    train(config, tmp_path / 'full')
    original_save = torch.save
    original_rename = Path.rename
    original_atomic = checkpoints.atomic_json
    original_boundary = observation_io.EventJournal.commit_boundary
    failed = False
    def fail_once():
        nonlocal failed
        if not failed:
            failed = True
            raise OSError('injected publication failure')
    def boundary(journal):
        value = original_boundary(journal)
        if stage == 'events' and value['step'] == 1:
            fail_once()
        return value
    def save(value, destination, *args, **kwargs):
        if stage == 'payload' and value.get('step') == 1:
            destination.write(b'partial')
            fail_once()
        return original_save(value, destination, *args, **kwargs)
    def rename(path, target):
        if stage == 'rename' and '-step-00000001-' in path.name:
            fail_once()
        return original_rename(path, target)
    def atomic(path, value):
        path = Path(path)
        selected = path.name == 'latest.json' and value.get('step') == 1
        if selected and stage == 'pointer':
            fail_once()
        if stage == 'manifest' and path.name == 'manifest.json' and value.get('last_durable_step') == 1:
            fail_once()
        original_atomic(path, value)
        if selected and stage == 'after-pointer':
            fail_once()
    with monkeypatch.context() as patch:
        patch.setattr(observation_io.EventJournal, 'commit_boundary', boundary)
        patch.setattr(torch, 'save', save)
        patch.setattr(Path, 'rename', rename)
        patch.setattr(checkpoints, 'atomic_json', atomic)
        patch.setattr(controller, 'atomic_json', atomic)
        patch.setattr(observation_io, 'atomic_json', atomic)
        with pytest.raises(OSError, match='injected publication'):
            train(config, tmp_path / 'run', checkpoint_every=1)
    _, info, state = read_checkpoint(tmp_path / 'run')
    expected_step = 1 if stage in ('after-pointer', 'manifest') else 0
    assert info['step'] == state['step'] == expected_step
    validate_event_boundary(tmp_path / 'run', info)
    boundary = info['event_boundary']
    prefix = (tmp_path / 'run/events.jsonl').read_bytes()[:boundary['offset']]
    assert json.loads(prefix.splitlines()[-1])['step'] == expected_step
    resume(tmp_path / 'run')
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'run')[2])


def test_missing_boundary_fails_before_new_attempt(tmp_path):
    config = write_default(tmp_path / 'config', device='cpu')
    result = train(config, tmp_path / 'run', stop_after_steps=1)
    checkpoint = Path(result['checkpoint_path']) / 'manifest.json'
    info = json.loads(checkpoint.read_text())
    info.pop('event_boundary')
    checkpoint.write_text(json.dumps(info))
    before = (tmp_path / 'run/manifest.json').read_bytes()
    with pytest.raises(ValueError, match='durable event boundary'):
        resume(tmp_path / 'run')
    assert (tmp_path / 'run/manifest.json').read_bytes() == before


def test_lost_committed_metrics_reject_latest_and_allow_explicit_intact_snapshot(tmp_path):
    config = write_default(tmp_path / 'config', device='cpu')
    train(config, tmp_path / 'full')
    run = tmp_path / 'run'
    train(config, run, stop_after_steps=1)
    earlier = next((run / 'checkpoints').glob('*-step-00000000-*'))
    info = json.loads((earlier / 'manifest.json').read_text())
    events = run / 'events.jsonl'
    events.write_bytes(events.read_bytes()[:info['event_boundary']['offset']])
    before = (run / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='missing committed events'):
        resume(run)
    assert (run / 'manifest.json').read_bytes() == before
    restored = resume(run, checkpoint=earlier)
    assert restored['recovery_parent']['step'] == 0
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(run)[2])

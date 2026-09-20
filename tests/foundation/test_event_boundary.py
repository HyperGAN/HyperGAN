"""The published numerical snapshot and its immutable event prefix agree."""
import hashlib
import json

import pytest

from hypergan.run_state import EventJournal, validate_event_boundary


def row(sequence=1, step=0, **values):
    return dict(run_id='run', attempt_id='attempt', step=step, sequence=sequence, **values)


def test_durable_prefix_matches_bytes_and_allows_later_attempts(tmp_path):
    journal = EventJournal(tmp_path)
    journal.append(row(event='start'))
    journal.append(row(2, 1, event='train', metrics={'loss': 1.25}))
    boundary = journal.commit_boundary()
    content = (tmp_path / 'events.jsonl').read_bytes()
    assert boundary['offset'] == len(content)
    assert boundary['sha256'] == hashlib.sha256(content).hexdigest()
    checkpoint = dict(row(step=1), event_boundary=boundary)
    journal.append(dict(row(1, 1, event='resume'), attempt_id='next'))
    with journal.path.open('ab') as output:
        output.write(b'{"partial')
    validate_event_boundary(tmp_path, checkpoint)
    reopened = EventJournal(tmp_path)
    reopened.append(dict(row(2, 2, event='train'), attempt_id='next'))
    assert reopened.commit_boundary()['sha256'] == hashlib.sha256(journal.path.read_bytes()).hexdigest()
    validate_event_boundary(tmp_path, checkpoint)


@pytest.mark.parametrize('damage', ['truncate', 'edit', 'identity', 'sequence', 'partial'])
def test_missing_or_mismatched_committed_events_are_rejected(tmp_path, damage):
    journal = EventJournal(tmp_path)
    journal.append(row(event='start'))
    checkpoint = dict(row(), event_boundary=journal.commit_boundary())
    boundary = checkpoint['event_boundary']
    if damage == 'truncate':
        journal.path.write_bytes(journal.path.read_bytes()[:-1])
    elif damage == 'edit':
        journal.path.write_bytes(journal.path.read_bytes().replace(b'start', b'stops'))
    elif damage == 'identity':
        boundary['attempt_id'] = 'different'
    elif damage == 'sequence':
        boundary['sequence'] += 1
    else:
        content = journal.path.read_bytes()[:-1]
        journal.path.write_bytes(content)
        boundary.update(offset=len(content), sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match='event boundary'):
        validate_event_boundary(tmp_path, checkpoint)


def test_large_last_event_has_no_implicit_megabyte_limit(tmp_path):
    journal = EventJournal(tmp_path)
    journal.append(row(event='start'))
    journal.append(row(2, 1, event='train', data='x' * 2100000))
    validate_event_boundary(tmp_path, dict(row(step=1), event_boundary=journal.commit_boundary()))


def test_event_sync_failure_never_returns_a_boundary(tmp_path, monkeypatch):
    journal = EventJournal(tmp_path)
    journal.append(row(event='start'))
    import hypergan.run_state as state
    def fail(fd):
        raise OSError('event fsync failed')
    monkeypatch.setattr(state.os, 'fsync', fail)
    with pytest.raises(OSError, match='event fsync'):
        journal.commit_boundary()


def test_partial_append_cannot_be_followed_by_an_error_row_or_commit(tmp_path, monkeypatch):
    from pathlib import Path
    journal = EventJournal(tmp_path)
    journal.append(row(event='start'))
    committed = journal.commit_boundary()
    original_open = Path.open
    class BrokenWriter:
        def __enter__(self):
            self.output = original_open(journal.path, 'ab')
            return self
        def write(self, data):
            self.output.write(data[:len(data) // 2])
            self.output.flush()
            raise OSError('disk full during row')
        def __exit__(self, *unused):
            self.output.close()
    def opened(path, mode='r', *args, **kwargs):
        return BrokenWriter() if path == journal.path and mode == 'ab' else original_open(path, mode, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', opened)
        with pytest.raises(OSError, match='disk full'):
            journal.append(row(2, 1, event='train'))
    tail = journal.path.read_bytes()
    with pytest.raises(OSError, match='incomplete write'):
        journal.append(row(3, 1, event='failed'))
    with pytest.raises(OSError, match='incomplete event'):
        journal.commit_boundary()
    assert journal.path.read_bytes() == tail
    repaired = EventJournal(tmp_path)
    assert repaired.offset == committed['offset']
    validate_event_boundary(tmp_path, dict(row(), event_boundary=committed))

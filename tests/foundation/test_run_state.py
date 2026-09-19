"""Event reconnect and manual checkpoint protocols require only the base package."""
import base64
import json
import subprocess
import sys

import pytest

from hypergan.run_events import read_event_page
from hypergan.run_requests import (
    acknowledge_request, checkpoint_request_status, pending_requests,
    submit_checkpoint_request,
)
from hypergan.run_state import repair_event_tail


def event(sequence, attempt='0001-first', run='run-one', step=None):
    return {'schema_version': 1, 'run_id': run, 'attempt_id': attempt,
            'sequence': sequence, 'step': sequence - 1 if step is None else step,
            'event': 'train', 'seconds': float(sequence)}


def append(root, *rows):
    with (root / 'events.jsonl').open('ab') as output:
        for row in rows:
            output.write((json.dumps(row) + '\n').encode())


def submit(root, name='request-one', attempt='0001-first', run='run-one'):
    return submit_checkpoint_request(root, run_id=run, attempt_id=attempt, request_id=name)


def succeed(root, name='request-one', attempt='0001-first', step=2):
    return acknowledge_request(root, name, status='succeeded', attempt_id=attempt,
                               step=step, checkpoint_path=str(root / 'checkpoints' / 'generation'))


def test_forward_pages_reconnect_empty_log_and_attempt_transition(tmp_path):
    initial = read_event_page(tmp_path)
    assert initial['events'] == [] and not initial['has_more'] and not initial['partial_tail']
    append(tmp_path, event(1), event(2), event(3), event(1, '0002-second'), event(2, '0002-second'))
    page = read_event_page(tmp_path, initial['cursor'], limit=2)
    assert page['events'] == [event(1), event(2)] and page['has_more']
    # Retrying a cursor returns the same existing rows, not silent implicit advancement.
    assert read_event_page(tmp_path, initial['cursor'], limit=2) == page
    second = read_event_page(tmp_path, page['cursor'], limit=2)
    assert second['events'] == [event(3), event(1, '0002-second')]
    third = read_event_page(tmp_path, second['cursor'], limit=2)
    assert third['events'] == [event(2, '0002-second')] and not third['has_more']
    assert read_event_page(tmp_path, third['cursor'])['events'] == []
    append(tmp_path, event(3, '0002-second'))
    assert read_event_page(tmp_path, third['cursor'])['events'] == [event(3, '0002-second')]


def test_partial_unicode_tail_waits_without_advancing_and_repair_starts_new_attempt(tmp_path):
    append(tmp_path, event(1))
    second = dict(event(2), detail='snowman ☃')
    encoded = (json.dumps(second, ensure_ascii=False) + '\n').encode()
    split = encoded.index('☃'.encode()) + 1
    with (tmp_path / 'events.jsonl').open('ab') as output:
        output.write(encoded[:split])
    page = read_event_page(tmp_path)
    assert page['events'] == [event(1)] and page['partial_tail'] and not page['has_more']
    waiting = read_event_page(tmp_path, page['cursor'])
    assert waiting['events'] == [] and waiting['cursor'] == page['cursor']
    with (tmp_path / 'events.jsonl').open('ab') as output:
        output.write(encoded[split:])
    completed = read_event_page(tmp_path, page['cursor'])
    assert completed['events'] == [second] and not completed['partial_tail']
    with (tmp_path / 'events.jsonl').open('ab') as output:
        output.write(b'{"half":')
    repair_event_tail(tmp_path / 'events.jsonl')
    append(tmp_path, event(1, '0002-second'))
    assert read_event_page(tmp_path, completed['cursor'])['events'] == [event(1, '0002-second')]


def test_byte_budget_never_skips_an_oversized_event(tmp_path):
    first, second = event(1), dict(event(2), detail='x' * 1000)
    append(tmp_path, first, second)
    budget = len((json.dumps(first) + '\n').encode())
    page = read_event_page(tmp_path, max_bytes=budget)
    assert page['events'] == [first] and page['has_more']
    with pytest.raises(ValueError, match='exceeds max_bytes'):
        read_event_page(tmp_path, page['cursor'], max_bytes=budget)
    assert read_event_page(tmp_path, page['cursor'], max_bytes=4096)['events'] == [second]


@pytest.mark.parametrize('change', ['truncate', 'replace', 'edit', 'remove'])
def test_stale_cursor_is_explicit(tmp_path, change):
    append(tmp_path, event(1), event(2))
    cursor = read_event_page(tmp_path)['cursor']
    path = tmp_path / 'events.jsonl'
    if change == 'truncate':
        path.write_bytes(b'')
    elif change == 'replace':
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif change == 'edit':
        path.write_bytes(path.read_bytes().replace(b'train', b'drain'))
    else:
        path.unlink()
    with pytest.raises(ValueError, match='Stale event cursor'):
        read_event_page(tmp_path, cursor)


def test_cursor_invalid_cross_run_mid_record_and_bounds(tmp_path):
    append(tmp_path, event(1))
    cursor = read_event_page(tmp_path)['cursor']
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises(ValueError, match='different run directory'):
        read_event_page(other, cursor)
    for malformed in ('not-base64?', '', 1, 'a' * 5000, base64.b64encode(b'[]').decode()):
        with pytest.raises(ValueError, match='Invalid event cursor'):
            read_event_page(tmp_path, malformed)
    state = json.loads(base64.urlsafe_b64decode(cursor))
    state['offset'] -= 1
    forged = base64.urlsafe_b64encode(json.dumps(state).encode()).decode()
    with pytest.raises(ValueError, match='consumed boundary changed'):
        read_event_page(tmp_path, forged)
    for value in (0, -1, True, 1.5, 10001):
        with pytest.raises(ValueError, match='limit must'):
            read_event_page(tmp_path, limit=value)
    for value in (0, -1, True, 1.5, 16777217):
        with pytest.raises(ValueError, match='max_bytes must'):
            read_event_page(tmp_path, max_bytes=value)


@pytest.mark.parametrize('bad', [b'not-json\n', b'[]\n', b'{"schema_version":2}\n'])
def test_complete_corrupt_rows_fail_instead_of_being_skipped(tmp_path, bad):
    append(tmp_path, event(1))
    cursor = read_event_page(tmp_path)['cursor']
    with (tmp_path / 'events.jsonl').open('ab') as output:
        output.write(bad)
    with pytest.raises(ValueError, match='Corrupt complete event row'):
        read_event_page(tmp_path, cursor)


@pytest.mark.parametrize('second', [event(3), event(1, run='other-run'), event(2, attempt='other-attempt')])
def test_event_identity_and_sequence_discontinuity_rejected(tmp_path, second):
    append(tmp_path, event(1), second)
    with pytest.raises(ValueError, match='Corrupt complete event row'):
        read_event_page(tmp_path)


def test_request_receipt_idempotency_and_conflicting_targets(tmp_path):
    first = submit(tmp_path)
    assert first['status'] == 'pending'
    assert submit(tmp_path) == first
    assert pending_requests(tmp_path) == [first['request']]
    for kwargs in ({'attempt': '0002-second'}, {'run': 'another-run'}):
        with pytest.raises(ValueError, match='Conflicting checkpoint request ID'):
            submit(tmp_path, **kwargs)
    receipt = succeed(tmp_path)
    assert receipt['status'] == 'succeeded' and receipt['step'] == 2
    assert checkpoint_request_status(tmp_path, 'request-one') == receipt
    assert submit(tmp_path) == receipt
    assert succeed(tmp_path) == receipt
    assert pending_requests(tmp_path) == []
    with pytest.raises(ValueError, match='Conflicting checkpoint acknowledgement'):
        succeed(tmp_path, step=3)


def test_stale_target_requires_rejection_not_success_under_new_attempt(tmp_path):
    submit(tmp_path)
    with pytest.raises(ValueError, match='different target attempt'):
        succeed(tmp_path, attempt='0002-second')
    receipt = acknowledge_request(tmp_path, 'request-one', status='rejected', attempt_id='0002-second', error='Target attempt has ended')
    assert receipt['request']['attempt_id'] == '0001-first'
    assert receipt['attempt_id'] == '0002-second'
    assert submit(tmp_path) == receipt
    assert pending_requests(tmp_path) == []


@pytest.mark.parametrize('bad', ['', '../escape', '/absolute', 'a.b', 'a/b', 'a\\b', '☃', '_first', 'x' * 129])
def test_request_ids_cannot_escape_paths(tmp_path, bad):
    for field in ('request_id', 'attempt_id', 'run_id'):
        kwargs = {'run_id': 'run-one', 'attempt_id': '0001-first', 'request_id': 'safe'}
        kwargs[field] = bad
        with pytest.raises(ValueError, match='ASCII'):
            submit_checkpoint_request(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        checkpoint_request_status(tmp_path, bad)
    assert not (tmp_path / 'checkpoint_requests').exists()


def test_generated_id_ordering_limits_and_windows_reserved_basename(tmp_path):
    result = submit_checkpoint_request(tmp_path, run_id='run-one', attempt_id='0001-first')
    assert len(result['request']['request_id']) == 32
    for name in ('z-last', 'a-first', 'CON'):
        submit(tmp_path, name)
    requests = pending_requests(tmp_path, limit=2)
    assert len(requests) == 2
    assert requests == sorted(requests, key=lambda r: r['request_id'])
    assert checkpoint_request_status(tmp_path, 'CON')['status'] == 'pending'
    for limit in (0, True, 257, 2.5):
        with pytest.raises(ValueError, match='limit must'):
            pending_requests(tmp_path, limit=limit)


def test_pending_and_payload_work_are_bounded(tmp_path):
    from hypergan import run_requests
    for i in range(256):
        submit(tmp_path, f'r{i:03d}')
    with pytest.raises(ValueError, match='queue is full'):
        submit(tmp_path, 'overfull')
    assert len(pending_requests(tmp_path, limit=3)) == 3
    # Existing receipts still work at capacity; no extra queue entry is allocated.
    assert submit(tmp_path, 'r000')['status'] == 'pending'
    path = tmp_path / 'checkpoint_requests' / 'pending' / 'request-r000.json'
    path.write_bytes(b' ' * (run_requests._MAX_RECORD_BYTES + 1))
    with pytest.raises(ValueError, match='exceeds 8192 bytes'):
        pending_requests(tmp_path)


def test_failed_atomic_request_publication_leaves_no_visible_partial(tmp_path, monkeypatch):
    from hypergan import run_requests
    real_link = run_requests.os.link
    def disk_full(source, target):
        raise OSError('simulated publication failure')
    monkeypatch.setattr(run_requests.os, 'link', disk_full)
    with pytest.raises(OSError, match='publication failure'):
        submit(tmp_path)
    assert pending_requests(tmp_path) == []
    assert list((tmp_path / 'checkpoint_requests' / 'pending').iterdir()) == []
    monkeypatch.setattr(run_requests.os, 'link', real_link)
    assert submit(tmp_path)['status'] == 'pending'


def test_ack_publication_before_pending_removal_and_crash_recovery(tmp_path, monkeypatch):
    from hypergan import run_requests
    original = submit(tmp_path)
    def crash(path):
        raise OSError('simulated death after durable receipt')
    monkeypatch.setattr(run_requests, '_remove_pending', crash)
    with pytest.raises(OSError, match='durable receipt'):
        succeed(tmp_path)
    assert checkpoint_request_status(tmp_path, 'request-one')['status'] == 'succeeded'
    assert submit(tmp_path)['status'] == 'succeeded'
    monkeypatch.undo()
    assert pending_requests(tmp_path) == []
    assert succeed(tmp_path)['request'] == original['request']


def test_failed_ack_keeps_request_pending(tmp_path, monkeypatch):
    from hypergan import run_requests
    submit(tmp_path)
    def fail(path, value):
        raise OSError('simulated receipt write failure')
    monkeypatch.setattr(run_requests, '_publish', fail)
    with pytest.raises(OSError, match='receipt write failure'):
        succeed(tmp_path)
    assert checkpoint_request_status(tmp_path, 'request-one')['status'] == 'pending'
    assert len(pending_requests(tmp_path)) == 1


def test_queue_lock_is_distinct_from_trainer_lock_and_nonblocking(tmp_path):
    from hypergan.run_requests import _queue_lock
    from hypergan.run_state import run_lock
    with run_lock(tmp_path):
        submit(tmp_path)
    control = tmp_path / 'checkpoint_requests'
    with _queue_lock(control):
        with pytest.raises(RuntimeError, match='queue is busy'):
            submit(tmp_path, 'another')
    assert submit(tmp_path, 'another')['status'] == 'pending'


def test_bad_files_wrong_ids_unknown_status_and_symlink_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match='Unknown checkpoint request'):
        checkpoint_request_status(tmp_path, 'unknown')
    submit(tmp_path)
    path = tmp_path / 'checkpoint_requests' / 'pending' / 'request-request-one.json'
    value = json.loads(path.read_text())
    value['request_id'] = 'mismatch'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='ID differs from filename'):
        pending_requests(tmp_path)
    with pytest.raises(ValueError, match='ID differs from filename'):
        checkpoint_request_status(tmp_path, 'request-one')
    path.write_text('[]')
    with pytest.raises(ValueError, match='Invalid checkpoint request fields'):
        pending_requests(tmp_path)


def test_protocol_imports_do_not_load_optional_dependencies(tmp_path):
    code = """
import importlib.abc, sys
class BaseOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','PIL','particlegan'}:
            raise AssertionError('run-state API imported optional dependency')
sys.meta_path.insert(0, BaseOnly())
from hypergan.run_events import read_event_page
from hypergan.run_requests import submit_checkpoint_request, pending_requests
assert read_event_page(sys.argv[1])['events'] == []
submit_checkpoint_request(sys.argv[1],run_id='run-one',attempt_id='attempt-one',request_id='request-one')
assert len(pending_requests(sys.argv[1])) == 1
"""
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=tmp_path, check=True, timeout=30)


def test_nonfinite_and_deeply_nested_complete_json_fail_cleanly(tmp_path):
    row = event(1)
    row['loss'] = float('nan')
    append(tmp_path, row)
    with pytest.raises(ValueError, match='nonfinite'):
        read_event_page(tmp_path)
    (tmp_path / 'events.jsonl').write_bytes(b'[' * 2000 + b']' * 2000 + b'\n')
    with pytest.raises(ValueError, match='Corrupt complete event row'):
        read_event_page(tmp_path)
    submit(tmp_path)
    path = tmp_path / 'checkpoint_requests' / 'pending' / 'request-request-one.json'
    path.write_bytes(b'[' * 2000 + b']' * 2000)
    with pytest.raises(ValueError, match='Invalid checkpoint request'):
        pending_requests(tmp_path)


def test_concurrent_same_id_submission_preserves_one_request(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def contender(_):
        try:
            return submit(tmp_path)
        except RuntimeError as exc:
            assert 'queue is busy' in str(exc)
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(contender, range(24)))
    accepted = [value for value in results if value is not None]
    assert accepted and all(value == accepted[0] for value in accepted)
    assert len(pending_requests(tmp_path)) == 1
    assert list((tmp_path / 'checkpoint_requests' / 'pending').glob('request-*.json')) == [tmp_path / 'checkpoint_requests' / 'pending' / 'request-request-one.json']


def test_status_retries_receipt_lookup_when_pending_disappears(tmp_path, monkeypatch):
    from hypergan import run_requests
    submit(tmp_path)
    real_read = run_requests._read
    fired = False
    def racing_read(path):
        nonlocal fired
        if path.parent.name == 'pending' and not fired:
            fired = True
            succeed(tmp_path)
        return real_read(path)
    monkeypatch.setattr(run_requests, '_read', racing_read)
    assert checkpoint_request_status(tmp_path, 'request-one')['status'] == 'succeeded'


def test_control_directory_rejects_an_existing_non_directory(tmp_path):
    (tmp_path / 'checkpoint_requests').write_text('unrelated local file')
    with pytest.raises(ValueError, match='ordinary directory'):
        submit(tmp_path)
    assert (tmp_path / 'checkpoint_requests').read_text() == 'unrelated local file'

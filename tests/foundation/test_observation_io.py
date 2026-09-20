"""Training admission never waits for a slow observation filesystem."""
from threading import Event, Thread
import json

import pytest

from hypergan.observation_io import ObservationIO
from hypergan.run_state import validate_event_boundary


def row(step):
    return dict(schema_version=1, run_id='run', attempt_id='attempt', sequence=step + 1,
                event='train', step=step, metrics={'loss': 0.5})


def test_slow_append_is_bounded_and_commit_waits_for_exact_prefix(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    writer = ObservationIO(tmp_path, attempt, capacity=2)
    entered, release, committed = Event(), Event(), Event()
    original = writer.journal.append
    def slow(value):
        entered.set()
        assert release.wait(5)
        original(value)
    monkeypatch.setattr(writer.journal, 'append', slow)
    try:
        assert writer.append(row(0))
        assert entered.wait(5)
        assert writer.append(row(1)) and writer.append(row(2))
        assert writer.append(row(3)) is False
        outcome = []
        def commit():
            outcome.append(writer.commit_boundary())
            committed.set()
        thread = Thread(target=commit)
        thread.start()
        assert not committed.wait(0.05)
        release.set()
        assert committed.wait(5)
        thread.join()
        assert outcome[0]['step'] == 2
        validate_event_boundary(tmp_path, dict(event_boundary=outcome[0], run_id='run', attempt_id='attempt', step=2))
    finally:
        release.set()
        writer.close()


def test_slow_status_write_coalesces_and_owns_snapshot(tmp_path, monkeypatch):
    import hypergan.observation_io as module
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    writer = ObservationIO(tmp_path, attempt)
    entered, release = Event(), Event()
    original = module.atomic_json
    def slow(path, value):
        entered.set()
        assert release.wait(5)
        original(path, value)
    monkeypatch.setattr(module, 'atomic_json', slow)
    try:
        writer.publish({'steps': 0})
        assert entered.wait(5)
        for step in range(1, 100):
            value = {'steps': step, 'nested': {'value': step}}
            writer.publish(value)
            value['nested']['value'] = -1
        assert writer.status == {'steps': 99, 'nested': {'value': 99}}
        release.set()
        writer.flush()
        assert json.loads((tmp_path / 'manifest.json').read_text()) == {'steps': 99, 'nested': {'value': 99}}
        assert (tmp_path / 'manifest.json').read_bytes() == (attempt / 'manifest.json').read_bytes()
    finally:
        release.set()
        writer.close()


def test_writer_failure_prevents_durable_boundary_and_closes(tmp_path, monkeypatch):
    writer = ObservationIO(tmp_path, tmp_path)
    def failed(_):
        raise OSError('volume unavailable')
    monkeypatch.setattr(writer.journal, 'append', failed)
    writer.append(row(0))
    with pytest.raises(OSError, match='volume unavailable'):
        writer.commit_boundary()
    with pytest.raises(OSError, match='volume unavailable'):
        writer.close()
    assert not writer.worker.is_alive()


def test_event_queue_byte_and_structure_bounds(tmp_path, monkeypatch):
    writer = ObservationIO(tmp_path, tmp_path, max_bytes=1024 * 1024)
    entered, release = Event(), Event()
    original = writer.journal.append
    def slow(value):
        entered.set()
        assert release.wait(5)
        original(value)
    monkeypatch.setattr(writer.journal, 'append', slow)
    try:
        assert writer.append(row(0))
        assert entered.wait(5)
        large = dict(row(1), detail='x' * 100000)
        assert writer.append(large)
        assert not writer.append(large)
        with pytest.raises(ValueError, match='byte bound'):
            writer.append(dict(row(2), detail='x' * 1000000))
        with pytest.raises(ValueError, match='structure bound'):
            writer.append(dict(row(2), detail=[0] * 20000))
    finally:
        release.set()
        writer.close()

"""Projection recovery, bounded map workers and raw agent pages need no torch."""
import base64
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import pytest

from hypergan.event_views import (ArtifactDescriptor, MapSpec, Projector, ViewSpec,
                                 read_projection_page)
from hypergan.run_events import read_event_page


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def catalog(root, definition='a' * 64):
    descriptor = {'kind': 'scalar', 'source': 'g_loss', 'version': definition}
    descriptor['definition_hash'] = hashlib.sha256(encode(descriptor)).hexdigest()
    value = {'schema_version': 1, 'metrics': {'loss/g_total': descriptor}}
    revision = hashlib.sha256(encode(value)).hexdigest()
    (root / 'metrics').mkdir(exist_ok=True)
    (root / 'metrics' / f'catalog-{revision}.json').write_bytes(encode(value))
    return revision


def append(root, count, *, start=1, definition='a' * 64, empty=False):
    revision = catalog(root, definition)
    with (root / 'events.jsonl').open('ab') as handle:
        for seq in range(start, start + count):
            event = dict(schema_version=2, run_id='run', stream_id='training', stream_generation='gen',
                         attempt_id='a1', sequence=seq, step=seq, event='train', catalog=revision,
                         metrics={} if empty else {'loss/g_total': seq / 2})
            handle.write(encode(event) + b'\n')


def test_default_map_recovery_all_emissions_and_empty_documents(tmp_path):
    append(tmp_path, 2)
    append(tmp_path, 1, start=3, empty=True)
    spec = MapSpec()
    with Projector(tmp_path, spec) as projector:
        assert projector.project(limit=2)['documents'] == 2
        with pytest.raises(RuntimeError, match='locked'):
            with Projector(tmp_path, spec):
                pass
    page = read_projection_page(tmp_path, spec.revision)
    assert len(page['frames']) == 2
    assert [f['emissions'][0]['value'] for f in page['frames']] == [.5, 1]
    source_cursor = page['frames'][-1]['source_cursor']
    assert read_event_page(tmp_path, source_cursor)['events'][0]['sequence'] == 3
    path = tmp_path / 'views' / spec.revision / 'contributions.jsonl'
    with path.open('ab') as handle:
        handle.write(b'{"torn":')
    with Projector(tmp_path, spec) as projector:
        assert projector.project()['documents'] == 1
        assert projector.project()['documents'] == 0
    suffix = read_projection_page(tmp_path, spec.revision, page['cursor'])
    assert len(suffix['frames']) == 1
    assert suffix['frames'][0]['emissions'] == []
    assert suffix['frames'][0]['projection_sequence'] == 3
    with Projector(tmp_path, spec) as projector:
        assert projector.sequence == 3
    whole = read_projection_page(tmp_path, spec.revision)
    assert len(whole['frames']) == 3
    assert len({e['id'] for f in whole['frames'] for e in f['emissions']}) == 2


def test_map_revision_shared_views_and_definition_partitions(tmp_path):
    append(tmp_path, 1)
    append(tmp_path, 1, start=2, definition='b' * 64)
    spec = MapSpec()
    mean = ViewSpec(spec.revision, reducer='mean/v1')
    envelope = ViewSpec(spec.revision)
    assert mean.revision != envelope.revision
    assert mean.map_revision == envelope.map_revision
    with Projector(tmp_path, spec) as projector:
        projector.project()
    rows = read_projection_page(tmp_path, spec.revision)['frames']
    assert rows[0]['emissions'][0]['definition_hash'] != rows[1]['emissions'][0]['definition_hash']
    with pytest.raises(ValueError, match='partitions'):
        ViewSpec(spec.revision, group_by=('metric_id',))


def test_each_frame_cursor_reconnect_and_faithful_copy(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    append(root, 4)
    with Projector(root) as projector:
        projector.project()
    page = read_projection_page(root, MapSpec().revision)
    for index, cursor in enumerate(page['frame_cursors']):
        suffix = read_projection_page(root, MapSpec().revision, cursor)
        assert [f['projection_sequence'] for f in suffix['frames']] == list(range(index + 2, 5))
    replica = tmp_path / 'replica'
    shutil.copytree(root, replica)
    assert read_projection_page(replica, MapSpec().revision, page['frame_cursors'][1])['frames'] == page['frames'][2:]
    path = root / 'views' / MapSpec().revision / 'contributions.jsonl'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='Stale'):
        read_projection_page(root, MapSpec().revision, page['cursor'])


def test_corrupt_complete_frame_not_repaired(tmp_path):
    append(tmp_path, 1)
    with Projector(tmp_path) as projector:
        projector.project()
    path = projector.path
    with path.open('ab') as handle:
        handle.write(b'{"broken":true}\n')
    size = path.stat().st_size
    with pytest.raises(ValueError):
        with Projector(tmp_path):
            pass
    assert path.stat().st_size == size
    with pytest.raises(ValueError):
        read_projection_page(tmp_path, MapSpec().revision)


def test_source_replacement_rejected_without_projection_mutation(tmp_path):
    append(tmp_path, 1)
    with Projector(tmp_path) as projector:
        projector.project()
    original = projector.path.read_bytes()
    source = tmp_path / 'events.jsonl'
    source.rename(tmp_path / 'old.jsonl')
    source.write_bytes((tmp_path / 'old.jsonl').read_bytes())
    with pytest.raises(ValueError, match='replaced'):
        with Projector(tmp_path):
            pass
    assert projector.path.read_bytes() == original


def changed_device_cursor(projector):
    frame = json.loads(projector.path.read_bytes())
    cursor = json.loads(base64.urlsafe_b64decode(frame['source_cursor']))
    cursor['file'][0] += 1
    frame['source_cursor'] = base64.urlsafe_b64encode(encode(cursor)).decode()
    projector.path.write_bytes(encode(frame) + b'\n')


def test_automatic_projector_can_resume_after_filesystem_device_change(tmp_path):
    append(tmp_path, 1)
    with Projector(tmp_path) as projector:
        projector.project()
    generation = projector.generation
    changed_device_cursor(projector)
    original = projector.path.read_bytes()
    with pytest.raises(ValueError, match='replaced'):
        with Projector(tmp_path):
            pass
    append(tmp_path, 1, start=2)
    with Projector(tmp_path, allow_device_change=True) as resumed:
        assert resumed.generation == generation
        assert resumed.sequence == 1
        assert resumed.project()['documents'] == 1
    assert projector.path.read_bytes().startswith(original)
    with Projector(tmp_path) as resumed:
        assert resumed.sequence == 2


@pytest.mark.parametrize('change', ['inode', 'boundary', 'truncate'])
def test_automatic_device_rebind_still_rejects_changed_source(tmp_path, change):
    append(tmp_path, 1)
    with Projector(tmp_path) as projector:
        projector.project()
    changed_device_cursor(projector)
    original = projector.path.read_bytes()
    source = tmp_path / 'events.jsonl'
    if change == 'inode':
        source.rename(tmp_path / 'old.jsonl')
        source.write_bytes((tmp_path / 'old.jsonl').read_bytes())
    elif change == 'boundary':
        source.write_bytes(source.read_bytes().replace(b'0.5', b'0.6'))
    else:
        source.write_bytes(b'')
    with pytest.raises(ValueError, match='Stale event cursor'):
        with Projector(tmp_path, allow_device_change=True):
            pass
    assert projector.path.read_bytes() == original


def test_v2_validation_and_per_event_cursors(tmp_path):
    append(tmp_path, 5)
    page = read_event_page(tmp_path, include_cursors=True)
    for index, cursor in enumerate(page['event_cursors']):
        assert len(read_event_page(tmp_path, cursor)['events']) == 4 - index
    path = tmp_path / 'events.jsonl'
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'0.5', b'1e999', 1))
    with pytest.raises(ValueError, match='Corrupt complete'):
        read_event_page(tmp_path)


@pytest.fixture
def maps(tmp_path, monkeypatch):
    name = '_hypergan_projection_maps'
    path = tmp_path / (name + '.py')
    path.write_text('''import os, sys, time
from pathlib import Path

def scaled(event, scale=1, receipt=None):
    if receipt:
        Path(receipt).write_text(str(os.getpid()) + ':' + str('torch' in sys.modules))
    for key, value in event.get('metrics', {}).items():
        yield [key, event['attempt_id'], event['step']], value * scale
        yield [key, event['attempt_id'], event['step']], value

def hang(event):
    time.sleep(120)

def unbounded(event):
    while True:
        yield ['loss/g_total', event['attempt_id'], event['step']], 1
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, hashlib.sha256(path.read_bytes()).hexdigest(), path
    sys.modules.pop(name, None)


def test_custom_map_worker_isolated_multiple_emissions_and_revision(tmp_path, maps):
    name, digest, _ = maps
    receipt = tmp_path / 'receipt'
    spec = MapSpec(name + ':scaled', config={'scale': 3, 'receipt': str(receipt)}, source_digest=digest)
    old_revision = spec.revision
    spec.config['scale'] = 100
    assert spec.revision == old_revision
    append(tmp_path, 2)
    with Projector(tmp_path, spec, timeout=10) as projector:
        assert projector.project()['documents'] == 2
    assert receipt.read_text().endswith(':False')
    rows = read_projection_page(tmp_path, spec.revision)['frames']
    assert [e['value'] for e in rows[0]['emissions']] == [1.5, .5]
    assert len({e['id'] for f in rows for e in f['emissions']}) == 4


@pytest.mark.heavy
def test_custom_map_timeout_and_output_bound_leave_source_cursor(tmp_path, maps):
    name, digest, _ = maps
    append(tmp_path, 1)
    for function in ('hang', 'unbounded'):
        spec = MapSpec(name + ':' + function, source_digest=digest)
        started = time.monotonic()
        with pytest.raises((RuntimeError, TimeoutError), match='deadline|budget'):
            with Projector(tmp_path, spec, timeout=2, total_timeout=4) as projector:
                projector.project()
        assert time.monotonic() - started < 12
        assert read_projection_page(tmp_path, spec.revision)['frames'] == []


def test_custom_source_digest_change_rejected(tmp_path, maps):
    name, digest, path = maps
    append(tmp_path, 1)
    spec = MapSpec(name + ':scaled', source_digest=digest)
    path.write_text(path.read_text() + '\n# changed\n')
    with pytest.raises(RuntimeError, match='digest changed'):
        with Projector(tmp_path, spec, timeout=10):
            pass


def test_modality_neutral_artifact_descriptor():
    artifact = ArtifactDescriptor('sample-1', 'sample', 'audio', 'audio/wav', 'a' * 64,
                                  1234, {'run_id': 'run', 'attempt_id': 'a', 'step': 7},
                                  {'sample_rate': 48000})
    assert artifact.descriptor()['metadata']['sample_rate'] == 48000
    with pytest.raises(ValueError, match='role'):
        ArtifactDescriptor('x', 'image', 'image', 'image/png', 'a' * 64, 1,
                           {'run_id': 'run', 'attempt_id': 'a', 'step': 1})


@pytest.mark.parametrize('damage', ['key', 'nonfinite', 'sequence', 'definition', 'metadata'])
def test_malformed_nested_frames_fail_actionably(tmp_path, damage):
    append(tmp_path, 1)
    with Projector(tmp_path) as projector:
        projector.project()
    row = json.loads(projector.path.read_bytes())
    if damage == 'key':
        row['emissions'][0]['key'] = []
    elif damage == 'nonfinite':
        row['emissions'][0]['value'] = 'not a scalar'
    elif damage == 'sequence':
        row['projection_sequence'] = True
    elif damage == 'definition':
        row['emissions'][0]['definition_hash'] = 'wrong'
    elif damage == 'metadata':
        (projector.directory / 'projection.json').write_text('{}')
    projector.path.write_bytes(encode(row) + b'\n')
    with pytest.raises(ValueError):
        read_projection_page(tmp_path, MapSpec().revision)


def test_partial_tail_read_and_atomic_per_source_rollback(tmp_path, maps):
    append(tmp_path, 1)
    spec = MapSpec()
    with Projector(tmp_path, spec) as projector:
        projector.project()
    page = read_projection_page(tmp_path, spec.revision)
    with projector.path.open('ab') as handle:
        handle.write(b'{"emissions":[')
    tail = read_projection_page(tmp_path, spec.revision, page['cursor'])
    assert tail['frames'] == [] and tail['partial_tail']
    assert tail['cursor'] == page['cursor']
    # A mapping exception must never publish progress independently of values.
    name, digest, _ = maps
    bad = MapSpec(name + ':unbounded', source_digest=digest)
    with Projector(tmp_path, bad, timeout=10) as projector:
        with pytest.raises(RuntimeError, match='budget'):
            projector.project()
        with pytest.raises(RuntimeError, match='close and reopen'):
            projector.project()
    assert read_projection_page(tmp_path, bad.revision)['frames'] == []


def test_large_backfill_batches_and_bounded_reopen(tmp_path, monkeypatch):
    append(tmp_path, 600)
    import hypergan.event_views as module
    real_read = module.read_event_page
    calls = []
    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return real_read(*args, **kwargs)
    monkeypatch.setattr(module, 'read_event_page', tracked)
    with Projector(tmp_path) as projector:
        assert projector.project(limit=1000)['documents'] == 600
    assert len(calls) == 2  # Startup boundary check + one source batch.
    assert projector.path.stat().st_size > 2 * 65536
    with Projector(tmp_path) as resumed:
        assert resumed.sequence == 600
        assert resumed.project()['documents'] == 0
    # Raw readers independently validate every frame, including older corruption.
    assert len(read_projection_page(tmp_path, MapSpec().revision, limit=1000)['frames']) == 600


def test_missing_projection_reader_does_not_create_state(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_projection_page(tmp_path, MapSpec().revision)
    assert list(tmp_path.iterdir()) == []


def test_source_generation_transition_fails_before_append(tmp_path):
    append(tmp_path, 1)
    with Projector(tmp_path) as projector:
        projector.project()
    original = projector.path.read_bytes()
    append(tmp_path, 1, start=2)
    source = tmp_path / 'events.jsonl'
    rows = source.read_bytes().splitlines()
    second = json.loads(rows[1])
    second['stream_generation'] = 'replacement'
    source.write_bytes(rows[0] + b'\n' + encode(second) + b'\n')
    with Projector(tmp_path) as projector:
        with pytest.raises(ValueError, match='stream identity changed'):
            projector.project()
    assert projector.path.read_bytes() == original


def test_recovery_rejects_source_cursor_ahead_of_committed_document(tmp_path):
    append(tmp_path, 3)
    with Projector(tmp_path) as projector:
        projector.project(limit=1)
    frame = json.loads(projector.path.read_bytes())
    frame['source_cursor'] = read_event_page(tmp_path)['cursor']
    projector.path.write_bytes(encode(frame) + b'\n')
    with pytest.raises(ValueError, match='cursor does not match'):
        with Projector(tmp_path):
            pass

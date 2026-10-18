"""Atomic run metadata and bounded, torch-free observation."""
import json
import hashlib
from pathlib import Path

from foldlitedb.fileio import (AppendJournal, atomic_json, file_lock,
                             repair_jsonl_tail as repair_event_tail, sync_directory)


def run_lock(run_dir):
    """Keep the established HyperGAN writer lock filename."""
    return file_lock(run_dir, name='.run.lock')


class EventJournal(AppendJournal):
    """Training document journal with HyperGAN checkpoint identity."""
    def __init__(self, run_dir):
        super().__init__(Path(run_dir) / 'events.jsonl')

    def commit_boundary(self):
        if self.failed:
            raise OSError('Cannot commit an incomplete event write')
        if self.last is None:
            raise ValueError('A checkpoint requires an event boundary')
        boundary = self.sync()
        return dict(schema_version=1, **boundary,
                    **{key: self.last[key] for key in ('run_id', 'attempt_id', 'step', 'sequence')})


def validate_event_boundary(run_dir, info):
    """Validate a controller snapshot before restoring any numerical state.

    Low-level numerical checkpoint callers may omit an event journal entirely.
    Controller snapshots carry this field and must match every committed byte;
    newer events and an incomplete uncommitted tail are deliberately permitted.
    """
    boundary = info.get('event_boundary')
    if boundary is None:
        return
    manifest_path = Path(run_dir) / 'manifest.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('events_compacted'):
                return
        except Exception:
            pass
    required = {'schema_version', 'offset', 'sha256', 'run_id', 'attempt_id', 'step', 'sequence'}
    if (not isinstance(boundary, dict) or set(boundary) != required
            or type(boundary['schema_version']) is not int or boundary['schema_version'] != 1
            or any(type(boundary[key]) is not int or boundary[key] < 1 for key in ('offset', 'sequence'))
            or type(boundary['step']) is not int or boundary['step'] < 0
            or any(boundary[key] != info[key] for key in ('run_id', 'attempt_id', 'step'))):
        raise ValueError('Invalid checkpoint event boundary identity')
    digest, remaining, tail = hashlib.sha256(), boundary['offset'], b''
    try:
        with (Path(run_dir) / 'events.jsonl').open('rb') as source:
            while remaining:
                block = source.read(min(1048576, remaining))
                if not block:
                    raise ValueError('Checkpoint event boundary is missing committed events')
                digest.update(block)
                remaining -= len(block)
                tail += block
                lines = tail.rsplit(b'\n', 2)
                if len(lines) == 3:
                    tail = b'\n'.join(lines[-2:])
        if digest.hexdigest() != boundary['sha256'] or not tail.endswith(b'\n'):
            raise ValueError('Checkpoint event boundary digest or complete-row mismatch')
        last = json.loads(tail.rstrip(b'\n').rsplit(b'\n', 1)[-1])
        if any(last[key] != boundary[key] for key in ('run_id', 'attempt_id', 'step', 'sequence')):
            raise ValueError('Checkpoint event boundary cursor differs from its last event')
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('Checkpoint event boundary is missing or invalid') from exc


def read_events(run_dir, limit=100, max_bytes=1048576):
    """Read at most max_bytes of the log tail; ignore partial trailing writes."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError('limit must be between 1 and 10000')
    if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 16777216:
        raise ValueError('max_bytes must be between 1 and 16777216')
    path = Path(run_dir) / 'events.jsonl'
    if not path.exists():
        return []
    with path.open('rb') as handle:
        handle.seek(0, 2)
        start = max(0, handle.tell() - max_bytes)
        handle.seek(start)
        data = handle.read(max_bytes)
    if start:
        data = data.partition(b'\n')[2]
    lines = data.split(b'\n')[:-1]
    rows = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Corrupt complete event row in run log") from exc
    return rows

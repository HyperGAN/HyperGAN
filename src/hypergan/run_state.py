"""Atomic run metadata and bounded, torch-free observation."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path

from foldlitedb.fileio import (AppendJournal, atomic_json, file_lock,
                             repair_jsonl_tail as repair_event_tail, sync_directory,
                             validate_jsonl_prefix, PrefixTruncatedError, PrefixMismatchError)
from foldlitedb.streams import read_jsonl_tail


@contextmanager
def run_lock(run_dir):
    """Keep the established HyperGAN writer lock filename."""
    stack = ExitStack()
    try:
        stack.enter_context(file_lock(run_dir, name='.run.lock'))
    except RuntimeError as exc:
        raise RuntimeError(f"Run is already locked by another process: {run_dir}") from exc
    with stack:
        yield


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
    try:
        last = validate_jsonl_prefix(Path(run_dir) / 'events.jsonl',
                                    offset=boundary['offset'], sha256=boundary['sha256'])
        if any(last[key] != boundary[key] for key in ('run_id', 'attempt_id', 'step', 'sequence')):
            raise ValueError('Checkpoint event boundary cursor differs from its last event')
    except PrefixTruncatedError as exc:
        raise ValueError('Checkpoint event boundary is missing committed events') from exc
    except PrefixMismatchError as exc:
        raise ValueError('Checkpoint event boundary digest or complete-row mismatch') from exc
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('Checkpoint event boundary is missing or invalid') from exc


def read_events(run_dir, limit=100, max_bytes=1048576):
    """Read at most max_bytes of the log tail; ignore partial trailing writes."""
    return read_jsonl_tail(Path(run_dir) / 'events.jsonl', limit=limit, max_bytes=max_bytes,
                          error_message='Corrupt complete event row in run log')

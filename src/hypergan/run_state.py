"""Atomic run metadata and bounded, torch-free observation."""
from contextlib import contextmanager
import json
import hashlib
import os
from pathlib import Path
import tempfile


def sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def run_lock(run_dir):
    """An OS-owned lock releases on process death; the file itself may remain."""
    path = Path(run_dir) / '.run.lock'
    with path.open('a+b') as handle:
        # msvcrt permits locking beyond EOF. Do not write an initialization byte
        # that another Windows handle may have locked since the file was opened.
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Run is already locked by another process: {run_dir}') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def repair_event_tail(path):
    """Drop an incomplete final write before appending a new attempt."""
    path = Path(path)
    if not path.exists():
        return
    with path.open('r+b') as handle:
        handle.seek(0, 2)
        end = handle.tell()
        while end:
            start = max(0, end - 65536)
            handle.seek(start)
            block = handle.read(end - start)
            last = block.rfind(b'\n')
            if last >= 0:
                handle.truncate(start + last + 1)
                return
            end = start
        handle.truncate(0)


class EventJournal:
    """One controller's append-only log with a durable, content-addressed prefix.

    Opening hashes retained history once; appends update that hash incrementally.
    The run lock excludes another writer. Projection readers need no lock.
    """
    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'events.jsonl'
        repair_event_tail(self.path)
        self.digest = hashlib.sha256()
        self.offset = 0
        if self.path.exists():
            with self.path.open('rb') as source:
                for block in iter(lambda: source.read(1048576), b''):
                    self.digest.update(block)
                    self.offset += len(block)
        self.last = None
        self.failed = False

    def append(self, row):
        if self.failed:
            raise OSError('Event journal has an incomplete write; resume must repair its tail')
        data = (json.dumps(row, allow_nan=False) + '\n').encode('utf-8')
        try:
            with self.path.open('ab') as output:
                output.write(data)
                output.flush()
        except BaseException:
            self.failed = True
            raise
        self.digest.update(data)
        self.offset += len(data)
        self.last = row

    def commit_boundary(self):
        if self.failed:
            raise OSError('Cannot commit an incomplete event write')
        if self.last is None:
            raise ValueError('A checkpoint requires an event boundary')
        with self.path.open('r+b') as source:
            if os.fstat(source.fileno()).st_size != self.offset:
                raise ValueError('Training event log changed outside its controller')
            os.fsync(source.fileno())
        sync_directory(self.path.parent)
        return dict(schema_version=1, offset=self.offset, sha256=self.digest.hexdigest(),
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

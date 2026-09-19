"""Atomic run metadata and bounded, torch-free observation."""
from contextlib import contextmanager
import json
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
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
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

"""Local, torch-free control requests and immutable durable acknowledgements.

A control request asks the running attempt to `checkpoint`, `sample` or `stop`.
The trainer is the only action executor, at complete update boundaries. Submission
means durable receipt, not execution: a request racing the final boundary may stay
pending until a later attempt rejects its stale target. Publication is at-least-once
until acknowledgement; retries after a durable acknowledgement do not repeat work.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .run_state import sync_directory

_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
_MAX_PENDING = 256
_MAX_RECORD_BYTES = 8192
_REQUEST_FIELDS = {'schema_version', 'request_id', 'run_id', 'attempt_id', 'action'}
_VALID_ACTIONS = {'checkpoint', 'stop', 'sample'}
# The queue directory predates the stop and sample actions; the on-disk name
# stays so existing runs and their receipts keep working.
CONTROL_DIRECTORY = 'checkpoint_requests'


def _id(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f'{name} must be 1-128 ASCII letters/digits, underscores or hyphens, beginning with a letter/digit')
    return value


def _paths(run_dir, create=False):
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Run directory does not exist: {root}')
    control = root / CONTROL_DIRECTORY
    pending, receipts = control / 'pending', control / 'receipts'
    for path in (control, pending, receipts):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f'Control request directory must be an ordinary directory: {path}')
        if create and not path.exists():
            path.mkdir(exist_ok=True)
            sync_directory(path.parent)
    return control, pending, receipts


@contextmanager
def _queue_lock(control):
    """Nonblocking; callers retry with the same request ID if another writer holds it."""
    path = control / 'queue.lock'
    if path.is_symlink():
        raise ValueError('Control request lock must not be a symlink')
    with path.open('a+b') as handle:
        # msvcrt permits locking beyond EOF. An initialization write before
        # acquisition can race a Windows contender's mandatory byte-range lock.
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Control request queue is busy; retry with the same request ID') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _path(directory, request_id):
    return directory / f'request-{_id(request_id, "request_id")}.json'


def _read(path):
    if path.is_symlink():
        raise ValueError(f'Control request files must not be symlinks: {path.name}')
    try:
        with path.open('rb') as stream:
            content = stream.read(_MAX_RECORD_BYTES + 1)
        if len(content) > _MAX_RECORD_BYTES:
            raise ValueError(f'record exceeds {_MAX_RECORD_BYTES} bytes')
        return json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'Invalid control request record {path.name}: {exc}') from exc


def _request(value):
    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        raise ValueError('Invalid control request fields')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['action'] not in _VALID_ACTIONS:
        raise ValueError('Unsupported control request schema/action')
    for key in ('request_id', 'run_id', 'attempt_id'):
        _id(value[key], key)
    return value


def _receipt(value):
    base_fields = {'schema_version', 'request', 'status', 'attempt_id', 'checkpoint_path', 'step', 'error'}
    optional_fields = {'artifact_path', 'preview_path'}
    if (
        not isinstance(value, dict)
        or not base_fields.issubset(value)
        or not set(value).issubset(base_fields | optional_fields)
        or type(value['schema_version']) is not int
        or value['schema_version'] != 1
    ):
        raise ValueError('Invalid control acknowledgement fields/schema')
    request = _request(value['request'])
    _id(value['attempt_id'], 'acknowledgement attempt_id')
    action = request['action']
    if value['status'] == 'succeeded':
        if value['attempt_id'] != request['attempt_id']:
            raise ValueError('A control request cannot succeed in a different target attempt')
        if type(value['step']) is not int or not 0 <= value['step'] < 2 ** 63:
            raise ValueError('Successful control acknowledgement requires a nonnegative step')
        if value['error'] is not None:
            raise ValueError('Successful control acknowledgement requires no error')
        if action == 'checkpoint':
            if not isinstance(value['checkpoint_path'], str) or not 1 <= len(value['checkpoint_path']) <= 4096:
                raise ValueError('Successful checkpoint acknowledgement requires a path and no error')
        elif action == 'sample':
            sample_path = value.get('artifact_path') or value.get('preview_path') or value.get('checkpoint_path')
            if not isinstance(sample_path, str) or not 1 <= len(sample_path) <= 4096:
                raise ValueError('Successful sample acknowledgement requires an artifact/preview path and no error')
        elif action == 'stop':
            if value['checkpoint_path'] is not None and (
                not isinstance(value['checkpoint_path'], str) or not 1 <= len(value['checkpoint_path']) <= 4096
            ):
                raise ValueError('Invalid checkpoint_path on stop acknowledgement')
    elif value['status'] == 'rejected':
        if (
            value['checkpoint_path'] is not None
            or value.get('artifact_path') is not None
            or value.get('preview_path') is not None
            or value['step'] is not None
            or not isinstance(value['error'], str)
            or not 1 <= len(value['error']) <= 2048
        ):
            raise ValueError('Rejected control acknowledgement requires an error (1-2048 characters), no path or step')
    else:
        raise ValueError('Control acknowledgement status must be succeeded or rejected')
    return value


def _publish(path, value):
    """Create one immutable JSON record atomically; never replace an existing ID."""
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError(f'Control request record exceeds {_MAX_RECORD_BYTES} bytes')
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _remove_pending(path):
    if path.exists():
        path.unlink()
        sync_directory(path.parent)


def _inventory(pending):
    """Bound directory scan as well as payload work; refuse overfull/tampered queues."""
    files = []
    with os.scandir(pending) as entries:
        for index, entry in enumerate(entries):
            # Atomic writers clean their own temporary files; a crash can leave one.
            # Count even those entries so externally populated directories stay bounded.
            if index >= _MAX_PENDING * 2:
                raise ValueError('Control request directory exceeds its scan bound; inspect pending/orphan files')
            if entry.name.startswith('.pending-'):
                continue
            if not entry.name.startswith('request-') or not entry.name.endswith('.json'):
                raise ValueError(f'Unexpected control request file: {entry.name}')
            request_id = entry.name[len('request-'):-len('.json')]
            _id(request_id, 'request filename ID')
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f'Control request must be an ordinary file: {entry.name}')
            files.append((request_id, Path(entry.path)))
            if len(files) > _MAX_PENDING:
                raise ValueError(f'Control request queue exceeds {_MAX_PENDING} pending requests')
    return sorted(files)


def _pending_receipt(request):
    return {'schema_version': 1, 'request': request, 'status': 'pending'}


def request_status(run_dir, request_id):
    """Return durable pending/succeeded/rejected receipt, or FileNotFoundError."""
    _, pending, receipts = _paths(run_dir)
    request_path, receipt_path = _path(pending, request_id), _path(receipts, request_id)
    try:
        receipt = _receipt(_read(receipt_path))
        if receipt['request']['request_id'] != request_id:
            raise ValueError('Control request receipt ID differs from filename')
        return receipt
    except FileNotFoundError:
        pass
    try:
        request = _request(_read(request_path))
        if request['request_id'] != request_id:
            raise ValueError('Control request ID differs from filename')
        return _pending_receipt(request)
    except FileNotFoundError:
        # The writer may have published acknowledgement and removed pending between reads.
        try:
            receipt = _receipt(_read(receipt_path))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Unknown control request: {request_id}') from exc
        if receipt['request']['request_id'] != request_id:
            raise ValueError('Control request receipt ID differs from filename')
        return receipt


checkpoint_request_status = request_status


def submit_request(run_dir, *, action, run_id, attempt_id, request_id=None):
    """Durably enqueue an explicitly targeted control request, or return the same receipt.

    Same ID plus another run/attempt/action conflicts. Native callers supply current run
    status; the CLI rejects new terminal-run requests. Pending does not imply that
    the target is alive or will execute. ID receipts are retained for future retries.
    """
    request = _request({'schema_version': 1, 'request_id': uuid.uuid4().hex if request_id is None else request_id,
                        'run_id': run_id, 'attempt_id': attempt_id, 'action': action})
    control, pending, _ = _paths(run_dir, create=True)
    with _queue_lock(control):
        try:
            existing = request_status(run_dir, request['request_id'])
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing['request'] != request:
                raise ValueError('Conflicting control request ID: reuse requires the same run, attempt and action')
            return existing
        if len(_inventory(pending)) >= _MAX_PENDING:
            raise ValueError(f'Control request queue is full ({_MAX_PENDING}); wait for acknowledgements')
        _publish(_path(pending, request['request_id']), request)
        return _pending_receipt(request)


def submit_checkpoint_request(run_dir, *, run_id, attempt_id, request_id=None):
    """Durably enqueue an explicitly targeted checkpoint, or return the same receipt."""
    return submit_request(run_dir, action='checkpoint', run_id=run_id, attempt_id=attempt_id, request_id=request_id)


def submit_stop_request(run_dir, *, run_id, attempt_id, request_id=None):
    """Durably enqueue an explicitly targeted graceful stop request, or return the same receipt."""
    return submit_request(run_dir, action='stop', run_id=run_id, attempt_id=attempt_id, request_id=request_id)


def submit_sample_request(run_dir, *, run_id, attempt_id, request_id=None):
    """Durably enqueue an explicitly targeted sample/preview request, or return the same receipt."""
    return submit_request(run_dir, action='sample', run_id=run_id, attempt_id=attempt_id, request_id=request_id)


def pending_requests(run_dir, *, limit=32):
    """Read at most limit requests, ordered by ID; only trainer should consume/ack.

    Trainer compares run_id/attempt_id and rejects stale targets rather than applying
    them to a newer attempt. Queue lock contention raises RuntimeError for retry.
    """
    if type(limit) is not int or not 1 <= limit <= _MAX_PENDING:
        raise ValueError(f'limit must be an integer between 1 and {_MAX_PENDING}')
    control, pending, receipts = _paths(run_dir)
    if not pending.exists():
        return []
    with _queue_lock(control):
        result = []
        for request_id, path in _inventory(pending):
            request = _request(_read(path))
            if request['request_id'] != request_id:
                raise ValueError('Control request ID differs from filename')
            receipt_path = _path(receipts, request_id)
            if receipt_path.exists():
                receipt = _receipt(_read(receipt_path))
                if receipt['request'] != request:
                    raise ValueError('Control request conflicts with its durable acknowledgement')
                _remove_pending(path)
                continue
            result.append(request)
            if len(result) == limit:
                break
        return result


def acknowledge_request(
    run_dir,
    request_id,
    *,
    status,
    attempt_id,
    checkpoint_path=None,
    artifact_path=None,
    preview_path=None,
    step=None,
    error=None,
):
    """Publish an immutable result after trainer safely executes/rejects a request.

    Only acknowledge success after publishing the complete checkpoint or preview.
    The helper validates protocol identity and receipt fields; it does not inspect
    torch state. Repeating the exact acknowledgement is safe; conflicting outcomes are errors.
    """
    control, pending, _ = _paths(run_dir)
    if not control.exists():
        raise FileNotFoundError(f'Unknown control request: {request_id}')
    with _queue_lock(control):
        existing = request_status(run_dir, request_id)
        action = existing['request']['action']
        payload = {
            'schema_version': 1,
            'request': existing['request'],
            'status': status,
            'attempt_id': attempt_id,
            'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else None,
            'step': step,
            'error': error,
        }
        if action == 'sample':
            # A sample receipt names its preview in `preview_path` and
            # `artifact_path`. `checkpoint_path` stays None unless a caller
            # passes one; older receipts repeated the preview path there and
            # remain valid.
            resolved_sample = preview_path or artifact_path or checkpoint_path
            sample_str = str(resolved_sample) if resolved_sample is not None else None
            payload['artifact_path'] = str(artifact_path) if artifact_path is not None else sample_str
            payload['preview_path'] = str(preview_path) if preview_path is not None else sample_str
        elif artifact_path is not None or preview_path is not None:
            if artifact_path is not None:
                payload['artifact_path'] = str(artifact_path)
            if preview_path is not None:
                payload['preview_path'] = str(preview_path)
        receipt = _receipt(payload)
        if existing['status'] != 'pending':
            if existing != receipt:
                raise ValueError('Conflicting control acknowledgement: an existing result is immutable')
            _remove_pending(_path(pending, request_id))
            return existing
        _publish(_path(control / 'receipts', request_id), receipt)
        _remove_pending(_path(pending, request_id))
        return receipt

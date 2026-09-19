"""Bounded, torch-free forward event pages with opaque reconnect cursors.

Cursors are continuation markers, not credentials. Logs are append-only: replacing,
truncating or changing the consumed boundary requires restarting without a cursor.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re

_MAX_BYTES = 16 * 1024 * 1024
_ANCHOR_BYTES = 256
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _identity(stat):
    return [stat.st_dev, stat.st_ino]


def _decode_cursor(cursor, location):
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096:
            raise ValueError('cursor must be a nonempty string of at most 4096 characters')
        raw = base64.b64decode(cursor.encode('ascii'), altchars=b'-_', validate=True)
        state = json.loads(raw)
        if not isinstance(state, dict) or set(state) != {'version', 'location', 'file', 'offset', 'anchor', 'last'}:
            raise ValueError('invalid cursor fields')
        if type(state['version']) is not int or state['version'] != 1:
            raise ValueError('unsupported cursor version')
        if state['location'] != location:
            raise ValueError('cursor belongs to a different run directory')
        _integer(state['offset'], 'cursor offset', 0, 2 ** 63 - 1)
        if not isinstance(state['anchor'], str) or not _HEX.fullmatch(state['anchor']):
            raise ValueError('invalid cursor anchor')
        identity = state['file']
        if identity is not None and (not isinstance(identity, list) or len(identity) != 2 or any(type(x) is not int or x < 0 for x in identity)):
            raise ValueError('invalid cursor file identity')
        last = state['last']
        if state['offset'] == 0:
            if last is not None or state['anchor'] != _hash(b''):
                raise ValueError('invalid initial cursor')
        else:
            if identity is None or not isinstance(last, dict) or set(last) != {'run_id', 'attempt_id', 'sequence'}:
                raise ValueError('invalid cursor event identity')
            _event_identity(last)
        return state
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'Invalid event cursor: {exc}') from exc


def _portable_cursor(cursor, identity, location):
    """Translate a public stream marker to the private reader's validated state."""
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096:
            raise ValueError('invalid cursor size')
        value = json.loads(base64.b64decode(cursor.encode('ascii'), altchars=b'-_', validate=True))
        if not isinstance(value, dict) or set(value) != {'version', 'run_id', 'stream_id', 'stream_generation', 'offset', 'anchor', 'last'}:
            raise ValueError('invalid public cursor fields')
        if any(value[key] != identity[key] for key in identity):
            raise ValueError('cursor belongs to a different run, stream or generation')
        private = {key: value[key] for key in ('version', 'offset', 'anchor', 'last')}
        private.update(location=location, file=[0, 0] if value['offset'] else None)
        state = _decode_cursor(_encode_cursor(private), location)
        if state['last'] is not None and state['last']['run_id'] != identity['run_id']:
            raise ValueError('cursor last event belongs to a different run')
        state['file'] = None  # file identities are local and never constrain public replay
        return state
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'Invalid public event cursor: {exc}') from exc


def _encode_cursor(state):
    return base64.urlsafe_b64encode(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).decode()


def _event_identity(row):
    for key in ('run_id', 'attempt_id'):
        if not isinstance(row.get(key), str) or not 1 <= len(row[key]) <= 128:
            raise ValueError(f'event {key} must be a nonempty string of at most 128 characters')
    _integer(row.get('sequence'), 'event sequence', 1, 2 ** 63 - 1)


def _event(line, previous):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    try:
        row = json.loads(line, parse_constant=reject_constant)
        # allow_nan also rejects a finite-looking exponent parsed as infinity.
        json.dumps(row, allow_nan=False)
        if not isinstance(row, dict) or type(row.get('schema_version')) is not int or row['schema_version'] not in (1, 2):
            raise ValueError('unsupported event schema')
        if row['schema_version'] == 2:
            for key in ('stream_id', 'stream_generation'):
                if not isinstance(row.get(key), str) or not 1 <= len(row[key]) <= 128:
                    raise ValueError(f'event {key} must be a nonempty string of at most 128 characters')
            if not isinstance(row.get('catalog'), str) or not _HEX.fullmatch(row['catalog']):
                raise ValueError('event catalog must be a lowercase SHA256 digest')
            if 'metrics' in row:
                if not isinstance(row['metrics'], dict) or any(
                    not isinstance(key, str) or not 1 <= len(key) <= 256 or type(value) not in (int, float)
                    for key, value in row['metrics'].items()
                ):
                    raise ValueError('event metrics must map IDs to finite scalar numbers')
            if 'measurement_status' in row and not isinstance(row['measurement_status'], dict):
                raise ValueError('measurement_status must be an object')
        _event_identity(row)
        _integer(row.get('step'), 'event step', 0, 2 ** 63 - 1)
        if not isinstance(row.get('event'), str) or not row['event']:
            raise ValueError('event name must be a nonempty string')
        if previous is None:
            if row['sequence'] != 1:
                raise ValueError('the first event must start an attempt at sequence 1')
        elif row['run_id'] != previous['run_id']:
            raise ValueError('run identity changed within the event log')
        elif row['attempt_id'] == previous['attempt_id']:
            if row['sequence'] != previous['sequence'] + 1:
                raise ValueError('event sequence is not contiguous within the attempt')
        elif row['sequence'] != 1:
            raise ValueError('a new attempt must start at sequence 1')
        return row
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'Corrupt complete event row: {exc}') from exc


def read_event_page(run_dir, cursor=None, *, limit=100, max_bytes=1048576, include_cursors=False, _open_file=None, _stream_identity=None):
    """Read forward from a cursor (or the beginning), preserving incomplete tails.

    Returns events, cursor, has_more and partial_tail. A page reads at most
    max_bytes plus a fixed 256-byte boundary anchor. An event exceeding the byte
    budget raises when it is next; increase max_bytes, up to 16 MiB. Complete
    corrupt rows fail visibly. A trailing unfinished row never advances a cursor.
    Cursors are local to this directory and file generation; attempts may change.
    Internal HTTP callers may supply _stream_identity (run_id, stream_id,
    stream_generation) for portable public cursors. This mode additionally reads
    one bounded first row to verify the source generation on every page.
    """
    _integer(limit, 'limit', 1, 10000)
    _integer(max_bytes, 'max_bytes', 1, _MAX_BYTES)
    if type(include_cursors) is not bool:
        raise ValueError('include_cursors must be a boolean')
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Run directory does not exist: {root}')
    location = _hash(os.fsencode(root))
    if _stream_identity is not None:
        if (not isinstance(_stream_identity, dict) or set(_stream_identity) != {'run_id', 'stream_id', 'stream_generation'}
                or any(not isinstance(value, str) or not 1 <= len(value) <= 128 for value in _stream_identity.values())):
            raise ValueError('Invalid public stream identity')
    def encode(value):
        if _stream_identity is None:
            return _encode_cursor(value)
        return _encode_cursor({**_stream_identity, **{key: value[key] for key in ('version', 'offset', 'anchor', 'last')}})
    state = ((_portable_cursor(cursor, _stream_identity, location) if _stream_identity else _decode_cursor(cursor, location))
             if cursor is not None else {
        'version': 1, 'location': location, 'file': None, 'offset': 0,
        'anchor': _hash(b''), 'last': None,
    })
    path = root / 'events.jsonl'
    try:
        handle = _open_file(path) if _open_file else path.open('rb')
    except FileNotFoundError:
        if state['file'] is not None or state['offset']:
            raise ValueError('Stale event cursor: the log was removed; restart without a cursor')
        result = {'events': [], 'cursor': encode(state), 'has_more': False, 'partial_tail': False}
        if include_cursors:
            result['event_cursors'] = []
        return result
    with handle as handle:
        stat = os.fstat(handle.fileno())
        identity = _identity(stat)
        if state['file'] is not None and state['file'] != identity:
            raise ValueError('Stale event cursor: the log was replaced; restart without a cursor')
        if _stream_identity is not None and stat.st_size:
            first = handle.readline(max_bytes + 1)
            if len(first) > max_bytes:
                raise ValueError('Public stream first document exceeds max_bytes')
            if first.endswith(b'\n'):
                first_event = _event(first, None)
                if any(first_event.get(key) != value for key, value in _stream_identity.items()):
                    raise ValueError('Source document run/stream/generation differs from its registration')
        offset = state['offset']
        if offset > stat.st_size:
            raise ValueError('Stale event cursor: the log was truncated; restart without a cursor')
        handle.seek(max(0, offset - _ANCHOR_BYTES))
        anchor = handle.read(min(offset, _ANCHOR_BYTES))
        if _hash(anchor) != state['anchor'] or (offset and not anchor.endswith(b'\n')):
            raise ValueError('Stale event cursor: its consumed boundary changed; restart without a cursor')
        handle.seek(offset)
        data = handle.read(max_bytes)
        read_end = offset + len(data)
        rows, consumed, last, event_cursors = [], 0, state['last'], []
        for _ in range(limit):
            end = data.find(b'\n', consumed)
            if end == -1:
                break
            row = _event(data[consumed:end], last)
            if _stream_identity is not None and any(row.get(key) != value for key, value in _stream_identity.items()):
                raise ValueError('Source document run/stream/generation differs from its registration')
            rows.append(row)
            last = {key: row[key] for key in ('run_id', 'attempt_id', 'sequence')}
            consumed = end + 1
            if include_cursors:
                boundary = (anchor + data[:consumed])[-256:] if consumed < 256 else data[consumed - 256:consumed]
                event_cursors.append(encode(dict(state, file=identity,
                    offset=offset + consumed, anchor=_hash(boundary), last=last)))
        if not rows and len(data) == max_bytes and read_end < stat.st_size:
            raise ValueError('Next event exceeds max_bytes; increase the byte limit (maximum 16777216)')
        next_offset = offset + consumed
        # Reuse at most 256 bytes from the already-read old anchor and this page.
        next_anchor = (anchor + data[:consumed])[-min(next_offset, _ANCHOR_BYTES):] if next_offset else b''
        updated = dict(state, file=identity, offset=next_offset, anchor=_hash(next_anchor), last=last)
        remaining = data[consumed:]
        has_more = read_end < stat.st_size or b'\n' in remaining
        partial_tail = read_end == stat.st_size and bool(data) and not data.endswith(b'\n')
        try:
            current_identity = _identity(path.stat())
        except FileNotFoundError as exc:
            raise ValueError('Stale event cursor: the log was removed while reading; retry without a cursor') from exc
        if current_identity != identity:
            raise ValueError('Stale event cursor: the log was replaced while reading; retry without a cursor')
        result = {'events': rows, 'cursor': encode(updated), 'has_more': has_more, 'partial_tail': partial_tail}
        if include_cursors:
            result['event_cursors'] = event_cursors
        return result

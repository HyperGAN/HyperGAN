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
        if not isinstance(row, dict) or type(row.get('schema_version')) is not int or row['schema_version'] != 1:
            raise ValueError('unsupported event schema')
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


def read_event_page(run_dir, cursor=None, *, limit=100, max_bytes=1048576):
    """Read forward from a cursor (or the beginning), preserving incomplete tails.

    Returns events, cursor, has_more and partial_tail. A page reads at most
    max_bytes plus a fixed 256-byte boundary anchor. An event exceeding the byte
    budget raises when it is next; increase max_bytes, up to 16 MiB. Complete
    corrupt rows fail visibly. A trailing unfinished row never advances a cursor.
    Cursors are local to this directory and file generation; attempts may change.
    """
    _integer(limit, 'limit', 1, 10000)
    _integer(max_bytes, 'max_bytes', 1, _MAX_BYTES)
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Run directory does not exist: {root}')
    location = _hash(os.fsencode(root))
    state = _decode_cursor(cursor, location) if cursor is not None else {
        'version': 1, 'location': location, 'file': None, 'offset': 0,
        'anchor': _hash(b''), 'last': None,
    }
    path = root / 'events.jsonl'
    try:
        handle = path.open('rb')
    except FileNotFoundError:
        if state['file'] is not None:
            raise ValueError('Stale event cursor: the log was removed; restart without a cursor')
        return {'events': [], 'cursor': _encode_cursor(state), 'has_more': False, 'partial_tail': False}
    with handle:
        stat = os.fstat(handle.fileno())
        identity = _identity(stat)
        if state['file'] is not None and state['file'] != identity:
            raise ValueError('Stale event cursor: the log was replaced; restart without a cursor')
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
        rows, consumed, last = [], 0, state['last']
        for _ in range(limit):
            end = data.find(b'\n', consumed)
            if end == -1:
                break
            row = _event(data[consumed:end], last)
            rows.append(row)
            last = {key: row[key] for key in ('run_id', 'attempt_id', 'sequence')}
            consumed = end + 1
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
        return {'events': rows, 'cursor': _encode_cursor(updated), 'has_more': has_more, 'partial_tail': partial_tail}

"""File-backed event maps and reusable contributions; no training or HTTP imports.

One map revision owns one projection. Views select its contributions and reducers;
changing a renderer/reducer does not execute Python again. Custom maps are trusted
Python isolated by the existing worker broker, not a security sandbox.
"""
from contextlib import ExitStack
from collections import OrderedDict
from dataclasses import dataclass, field
import base64
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import uuid

from .run_events import read_event_page
from .run_state import atomic_json, run_lock

MAX_FRAME_BYTES = 65536
MAX_EMISSIONS = 128
MAX_KEY_BYTES = 1024
MAX_KEYS = 4096
_HEX = re.compile(r'[0-9a-f]{64}\Z')


def _json(value):
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f'Expected finite JSON: {exc}') from exc


def _load(raw):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    value = json.loads(raw, parse_constant=reject)
    _json(value)  # Also rejects overflowing exponents (1e999).
    return value


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _hash(value, name):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')
    return value


def _text(value, name, limit=128):
    if not isinstance(value, str) or not 1 <= len(value) <= limit:
        raise ValueError(f'{name} must have 1..{limit} characters')
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


@dataclass(frozen=True)
class MapSpec:
    """Pure event -> iterable of ``([metric_id, attempt_id, step], value)``.

    Custom functions accept ``event, **config``. Pin their module file digest;
    workers verify it before calling the mapper. Source dependencies belong in version/config.
    Only metric IDs registered in the source catalog may be emitted in v1.
    """
    reference: str = 'builtin:metrics'
    version: str = '1'
    config: dict = field(default_factory=dict)
    source_digest: str | None = None

    def __post_init__(self):
        _text(self.reference, 'map reference', 512)
        _text(self.version, 'map version')
        if self.reference != 'builtin:metrics':
            if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', self.reference):
                raise ValueError('Map reference must be module:function')
            _hash(self.source_digest, 'custom map source_digest')
        elif self.source_digest is not None or self.config:
            raise ValueError('builtin:metrics accepts no source_digest or config')
        if type(self.config) is not dict or len(_json(self.config)) > 8192:
            raise ValueError('Map config must be a JSON object of at most 8192 bytes')
        # Detach user-owned containers; descriptors below always return snapshots.
        object.__setattr__(self, 'config', _load(_json(self.config)))
        object.__setattr__(self, '_config_json', _json(self.config))

    def descriptor(self):
        return _load(_json({'schema_version': 1, 'reference': self.reference,
            'version': self.version, 'config': _load(self._config_json), 'source_digest': self.source_digest,
            'key_schema': ['metric_id', 'attempt_id', 'step']}))

    @property
    def revision(self):
        return _digest(self.descriptor())


@dataclass(frozen=True)
class ViewSpec:
    map_revision: str
    reducer: str = 'envelope/v1'
    group_by: tuple = ('metric_id', 'definition_hash', 'attempt_id', 'step_bucket')
    bucket_steps: int = 1
    renderer: str = 'line'

    def __post_init__(self):
        _hash(self.map_revision, 'map_revision')
        if self.reducer not in ('mean/v1', 'envelope/v1'):
            raise ValueError('Supported reducers: mean/v1, envelope/v1')
        if (not isinstance(self.group_by, (tuple, list)) or len(set(self.group_by)) != len(self.group_by)
                or not {'metric_id', 'definition_hash', 'attempt_id'} <= set(self.group_by)
                or set(self.group_by) - {'metric_id', 'definition_hash', 'attempt_id', 'step_bucket'}):
            raise ValueError('group_by must preserve metric_id, definition_hash and attempt_id partitions')
        _integer(self.bucket_steps, 'bucket_steps', 1, 2 ** 53 - 1)
        _text(self.renderer, 'renderer')
        object.__setattr__(self, 'group_by', tuple(self.group_by))

    def descriptor(self):
        return {'schema_version': 1, 'map_revision': self.map_revision, 'reducer': self.reducer,
            'group_by': list(self.group_by), 'bucket_steps': self.bucket_steps, 'renderer': self.renderer}

    @property
    def revision(self):
        return _digest(self.descriptor())


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Indexed out-of-line output; samples and measurements have distinct roles."""
    artifact_id: str
    role: str
    modality: str
    media_type: str
    sha256: str
    size_bytes: int
    provenance: dict
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        _text(self.artifact_id, 'artifact_id')
        if self.role not in ('sample', 'measurement', 'diagnostic'):
            raise ValueError('Artifact role must be sample, measurement or diagnostic')
        _text(self.modality, 'modality')
        _text(self.media_type, 'media_type')
        _hash(self.sha256, 'artifact sha256')
        _integer(self.size_bytes, 'artifact size_bytes', 0, 2 ** 53 - 1)
        for name, value in (('provenance', self.provenance), ('metadata', self.metadata)):
            if type(value) is not dict or len(_json(value)) > 8192:
                raise ValueError(f'Artifact {name} must be an object of at most 8192 bytes')
            object.__setattr__(self, '_' + name + '_json', _json(value))
        for name in ('run_id', 'attempt_id'):
            _text(self.provenance.get(name), 'artifact provenance ' + name)
        _integer(self.provenance.get('step'), 'artifact provenance step', 0, 2 ** 53 - 1)

    def descriptor(self):
        return dict(schema_version=1, artifact_id=self.artifact_id, role=self.role,
                    modality=self.modality, media_type=self.media_type, sha256=self.sha256,
                    size_bytes=self.size_bytes, provenance=_load(self._provenance_json),
                    metadata=_load(self._metadata_json))


def _map_worker(rank, world_size, descriptor):
    reference = descriptor['reference']
    module_name, name = reference.split(':')
    # Trusted import occurs only inside the supervised worker.
    module = importlib.import_module(module_name)
    path = inspect.getsourcefile(module)
    if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != descriptor['source_digest']:
        raise ValueError('Custom map source digest changed; create a new map revision')
    function = getattr(module, name)
    if not inspect.isfunction(function):
        raise ValueError('Custom map must resolve to a module-level function')
    return function, descriptor['config']


def _map_command(state, operation, payload):
    if operation != 'map':
        raise ValueError('Unknown map operation')
    function, config = state
    output = []
    result = function(payload, **config)
    if result is not None:
        for item in result:
            output.append(item)
            if len(output) > MAX_EMISSIONS or len(_json(output)) > 32768:
                raise ValueError('Map emissions exceed bounded output budget')
    return _load(_json(output))


def _builtin(event):
    return [([metric, event['attempt_id'], event['step']], value)
            for metric, value in event.get('metrics', {}).items()]


def _definitions(root, event):
    if not event.get('metrics') and not event.get('catalog'):
        return {}
    from .metrics import read_catalog
    revision = _hash(event.get('catalog'), 'event catalog')
    catalog = read_catalog(root, revision)
    entries = catalog.get('metrics')
    if not isinstance(entries, dict) or len(entries) > MAX_KEYS:
        raise ValueError('Invalid or oversized metric catalog')
    return {metric: _hash(item['definition_hash'], 'metric definition_hash')
            for metric, item in entries.items()}


def _emissions(event, output, revision, definitions):
    if not isinstance(output, (list, tuple)) or len(output) > MAX_EMISSIONS:
        raise ValueError('Map emitted too many values')
    result = []
    source_id = [event['run_id'], event.get('stream_id', 'training'),
                 event['attempt_id'], event['sequence']]
    for index, pair in enumerate(output):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('Map emissions must be (key, value) pairs')
        key, value = pair
        if not isinstance(key, (tuple, list)) or len(key) != 3:
            raise ValueError('Map key must be [metric_id, attempt_id, step]')
        metric, attempt, step = key
        _text(metric, 'metric_id', 256)
        _text(attempt, 'attempt_id')
        _integer(step, 'mapped step', 0, 2 ** 53 - 1)
        if attempt != event['attempt_id']:
            raise ValueError('Map cannot change the source attempt partition')
        if metric not in definitions:
            raise ValueError(f'Mapped metric {metric!r} is absent from the source catalog')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('V1 mapped values must be finite scalars')
        if len(_json(key)) > MAX_KEY_BYTES:
            raise ValueError('Map key exceeds byte budget')
        result.append({'id': _digest([revision, source_id, index]), 'key': list(key),
                       'definition_hash': definitions[metric], 'value': value})
    return result


def _validate_source(source):
    if not isinstance(source, dict) or set(source) != {'run_id', 'stream_id', 'stream_generation', 'attempt_id', 'sequence'}:
        raise ValueError('Invalid projection source identity')
    for name in ('run_id', 'stream_id', 'stream_generation', 'attempt_id'):
        _text(source[name], name)
    _integer(source['sequence'], 'source sequence', 1, 2 ** 63 - 1)


def _source_contiguous(source, previous):
    if previous is None:
        if source['sequence'] != 1:
            raise ValueError('Projection must begin at source sequence 1')
        return
    if any(source[name] != previous[name] for name in ('run_id', 'stream_id', 'stream_generation')):
        raise ValueError('Projection source stream identity changed')
    expected = previous['sequence'] + 1 if source['attempt_id'] == previous['attempt_id'] else 1
    if source['sequence'] != expected:
        raise ValueError('Projection source sequence is not contiguous')


def _frame_unchecked(row, revision, previous):
    if (type(row) is not dict or type(row.get('schema_version')) is not int or row.get('schema_version') != 1
            or row.get('map_revision') != revision
            or type(row.get('projection_sequence')) is not int or row.get('projection_sequence') != previous + 1):
        raise ValueError('Corrupt projection frame identity or sequence')
    _text(row.get('source_cursor'), 'source_cursor', 4096)
    source = row.get('source')
    if not isinstance(source, dict) or set(source) != {'run_id', 'stream_id', 'stream_generation', 'attempt_id', 'sequence'}:
        raise ValueError('Corrupt projection source')
    for name in ('run_id', 'stream_id', 'stream_generation', 'attempt_id'):
        _text(source.get(name), name)
    _integer(source.get('sequence'), 'source sequence', 1, 2 ** 63 - 1)
    saved_source = _load(base64.b64decode(row['source_cursor'].encode('ascii'), altchars=b'-_', validate=True))
    if not isinstance(saved_source, dict) or saved_source.get('last') != {
            key: source[key] for key in ('run_id', 'attempt_id', 'sequence')} :
        raise ValueError('Source cursor does not match projection document identity')
    if previous == 0 and source['sequence'] != 1:
        raise ValueError('Projection must begin at the first source event')
    emissions = row.get('emissions')
    if not isinstance(emissions, list) or len(emissions) > MAX_EMISSIONS:
        raise ValueError('Corrupt projection emissions')
    for index, item in enumerate(emissions):
        if not isinstance(item, dict) or set(item) != {'id', 'key', 'definition_hash', 'value'}:
            raise ValueError('Corrupt projection emission')
        definition = _hash(item['definition_hash'], 'definition_hash')
        checked = _emissions(source, [(item['key'], item['value'])], revision,
                             {item['key'][0]: definition})[0]
        expected_id = _digest([revision, [source[k] for k in ('run_id', 'stream_id', 'attempt_id', 'sequence')], index])
        if checked['key'] != item['key'] or item['id'] != expected_id:
            raise ValueError('Corrupt projection emission identity')
    return row


def _frame(row, revision, previous):
    try:
        return _frame_unchecked(row, revision, previous)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError(f'Corrupt projection frame: {exc}') from exc


def _metadata(directory, revision, _open_file=None):
    path = directory / 'projection.json'
    with (_open_file(path) if _open_file else path.open('rb')) as handle:
        raw = handle.read(MAX_FRAME_BYTES + 1)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError('Projection metadata exceeds byte budget')
    value = _load(raw)
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version'] != 1 or value.get('map_revision') != revision
            or not isinstance(value.get('map'), dict) or _digest(value['map']) != revision
            or not isinstance(value.get('generation'), str)
            or not re.fullmatch('[0-9a-f]{32}', value['generation'])):
        raise ValueError('Invalid projection metadata identity')
    return value


class Projector:
    """Owned, sequential projection service; open explicitly outside HTTP handlers.

    The lock and optional supervised worker live for this context. ``project``
    advances at most limit source documents; each newline commits emissions and
    its source cursor together. Restart validates only the bounded final frame and repairs only
    an incomplete tail. Complete corrupt rows fail. No source log is modified.
    Custom worker total lifetime is bounded by total_timeout (restart explicitly).
    """
    def __init__(self, run_dir, spec=None, *, timeout=5.0, total_timeout=300.0):
        self.root = Path(run_dir).resolve()
        self.spec = spec or MapSpec()
        self.descriptor = self.spec.descriptor()
        self.revision = _digest(self.descriptor)
        self.directory = self.root / 'views' / self.revision
        self.path = self.directory / 'contributions.jsonl'
        self.timeout, self.total_timeout = timeout, total_timeout
        self._stack = self._service = None
        self.cursor = None
        self.sequence = 0
        self.source = None
        self._failed = False
        self._catalogs = OrderedDict()

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError('Projector is already open')
        if not self.root.is_dir():
            raise ValueError('Run directory does not exist')
        self.directory.mkdir(parents=True, exist_ok=True)
        stack = ExitStack()
        try:
            stack.enter_context(run_lock(self.directory))
            metadata_path = self.directory / 'projection.json'
            if metadata_path.exists():
                metadata = _metadata(self.directory, self.revision)
                if metadata.get('map') != self.descriptor:
                    raise ValueError('Projection map descriptor changed')
            else:
                if self.path.exists():
                    raise ValueError('Projection log has no identity metadata')
                metadata = {'schema_version': 1, 'map_revision': self.revision,
                            'generation': uuid.uuid4().hex, 'map': self.descriptor}
                atomic_json(metadata_path, metadata)
            self.generation = metadata['generation']
            self.cursor, self.sequence, self._failed, self.source = None, 0, False, None
            self._output = stack.enter_context(self.path.open('a+b'))
            # Only the final complete frame is needed to recover progress. Read
            # at most two frame budgets even for a multi-gigabyte projection.
            self._output.seek(0, os.SEEK_END)
            size = self._output.tell()
            begin = max(0, size - 2 * MAX_FRAME_BYTES)
            self._output.seek(begin)
            tail = self._output.read()
            end = tail.rfind(b'\n')
            if end < 0:
                if size > MAX_FRAME_BYTES:
                    raise ValueError('Projection incomplete tail exceeds byte budget')
                self._output.truncate(0)
            else:
                start = tail.rfind(b'\n', 0, end) + 1
                if begin and not start:
                    raise ValueError('Projection final frame exceeds byte budget')
                line = tail[start:end]
                if len(line) + 1 > MAX_FRAME_BYTES or len(tail) - end - 1 > MAX_FRAME_BYTES:
                    raise ValueError('Projection tail exceeds byte budget')
                row = _load(line)
                if not isinstance(row, dict):
                    raise ValueError('Corrupt projection final frame')
                _integer(row.get('projection_sequence'), 'projection sequence', 1, 2 ** 63 - 1)
                _frame(row, self.revision, row['projection_sequence'] - 1)
                self.cursor, self.sequence, self.source = row['source_cursor'], row['projection_sequence'], row['source']
                self._output.truncate(begin + end + 1)
            self._output.seek(0, os.SEEK_END)
            # Validate saved source generation/boundary before appending anything.
            read_event_page(self.root, self.cursor, limit=1, max_bytes=MAX_FRAME_BYTES)
            if self.descriptor['reference'] != 'builtin:metrics':
                from .cpu_worker_service import CPUWorkerService
                self._service = stack.enter_context(CPUWorkerService(_map_worker, _map_command,
                    args=(self.descriptor,), run_id='projection', attempt_id=self.revision,
                    world_size=1, initialize_process_group=False, startup_timeout=self.timeout,
                    command_timeout=self.timeout, collective_timeout=self.timeout,
                    total_timeout=self.total_timeout))
            self._stack = stack
            return self
        except BaseException:
            stack.close()
            self._service = None
            raise

    def project(self, *, limit=100):
        if self._failed:
            raise RuntimeError('Projection failed; close and reopen to validate committed progress')
        try:
            return self._project(limit=limit)
        except BaseException:
            self._failed = True
            raise

    def _project(self, *, limit):
        if self._stack is None:
            raise RuntimeError('Projector must be used as a context manager')
        _integer(limit, 'limit', 1, 10000)
        page = read_event_page(self.root, self.cursor, limit=limit,
                               max_bytes=1048576, include_cursors=True)
        emitted = 0
        for event, source_cursor in zip(page['events'], page['event_cursors']):
            source = {key: event.get(key, 'training' if key == 'stream_id' else event['run_id'])
                      for key in ('run_id', 'stream_id', 'stream_generation', 'attempt_id', 'sequence')}
            _source_contiguous(source, self.source)
            revision = event.get('catalog')
            if revision not in self._catalogs:
                self._catalogs[revision] = _definitions(self.root, event)
                if len(self._catalogs) > 16:
                    self._catalogs.popitem(last=False)
            definitions = self._catalogs[revision]
            if self._service is None:
                output = _builtin(event)
            else:
                output = self._service.command('map', event)['results'][0]
            row = {'schema_version': 1, 'map_revision': self.revision,
                   'projection_sequence': self.sequence + 1, 'source_cursor': source_cursor,
                   'source': source,
                   'emissions': _emissions(event, output, self.revision, definitions)}
            encoded = _json(row) + b'\n'
            if len(encoded) > MAX_FRAME_BYTES:
                raise ValueError('Projection frame exceeds 65536 bytes')
            self._output.write(encoded)
            self._output.flush()
            self.cursor, self.sequence, self.source = source_cursor, row['projection_sequence'], source
            emitted += 1
        return {'documents': emitted, 'source_cursor': self.cursor, 'has_more': page['has_more'],
                'partial_tail': page['partial_tail'], 'projection_sequence': self.sequence}

    def __exit__(self, *exc):
        stack, self._stack = self._stack, None
        self._service = None
        return stack.__exit__(*exc)


def _cursor(state):
    return base64.urlsafe_b64encode(_json(state)).decode('ascii')


def read_projection_page(run_dir, map_revision, cursor=None, *, limit=100, max_bytes=1048576, _open_file=None):
    """Read unchanged complete frames with an end cursor for each frame.

    Cursors bind revision, generation, offset, sequence and boundary digest; they
    contain no local path/inode, so faithfully copied projections retain cursors.
    Missing projections are explicit errors; this function never runs a mapper.
    """
    _hash(map_revision, 'map_revision')
    _integer(limit, 'limit', 1, 10000)
    _integer(max_bytes, 'max_bytes', 1, 16 * 1048576)
    directory = Path(run_dir) / 'views' / map_revision
    metadata = _metadata(directory, map_revision, _open_file)
    state = {'version': 1, 'map_revision': map_revision, 'generation': metadata['generation'],
             'offset': 0, 'sequence': 0, 'anchor': hashlib.sha256(b'').hexdigest(), 'source': None}
    if cursor is not None:
        try:
            if not isinstance(cursor, str) or len(cursor) > 4096:
                raise ValueError('Invalid projection cursor size')
            saved = _load(base64.b64decode(cursor.encode('ascii'), altchars=b'-_', validate=True))
            if set(saved) != set(state) or any(saved[k] != state[k] for k in ('version', 'map_revision', 'generation')):
                raise ValueError('Projection cursor revision/generation mismatch')
            _integer(saved['offset'], 'cursor offset', 0, 2 ** 63 - 1)
            _integer(saved['sequence'], 'cursor sequence', 0, 2 ** 63 - 1)
            _hash(saved['anchor'], 'cursor anchor')
            if (saved['sequence'] == 0) != (saved['source'] is None):
                raise ValueError('Invalid cursor source identity')
            if saved['source'] is not None:
                _validate_source(saved['source'])
            state = saved
        except (ValueError, TypeError, UnicodeError, KeyError) as exc:
            raise ValueError(f'Invalid projection cursor: {exc}') from exc
    path = directory / 'contributions.jsonl'
    with (_open_file(path) if _open_file else path.open('rb')) as handle:
        file_stat = os.fstat(handle.fileno())
        size = file_stat.st_size
        offset = state['offset']
        handle.seek(max(0, offset - 256))
        anchor = handle.read(min(offset, 256))
        if offset > size or hashlib.sha256(anchor).hexdigest() != state['anchor'] or (offset and not anchor.endswith(b'\n')):
            raise ValueError('Stale projection cursor: consumed boundary changed')
        handle.seek(offset)
        raw = handle.read(max_bytes)
        frames, cursors, consumed = [], [], 0
        for _ in range(limit):
            end = raw.find(b'\n', consumed)
            if end < 0:
                break
            if end + 1 - consumed > MAX_FRAME_BYTES:
                raise ValueError('Projection frame exceeds byte budget')
            row = _frame(_load(raw[consumed:end]), map_revision, state['sequence'])
            _source_contiguous(row['source'], state['source'])
            frames.append(row)
            anchor = (anchor + raw[consumed:end + 1])[-256:]
            consumed = end + 1
            state = dict(state, offset=offset + consumed, sequence=row['projection_sequence'],
                         anchor=hashlib.sha256(anchor).hexdigest(), source=row['source'])
            cursors.append(_cursor(state))
        if not frames and len(raw) == max_bytes and offset + len(raw) < size:
            raise ValueError('Next projection frame exceeds max_bytes')
        current = (directory / 'contributions.jsonl').stat()
        if (current.st_dev, current.st_ino) != (file_stat.st_dev, file_stat.st_ino):
            raise ValueError('Projection replaced while reading; reconnect from a validated cursor')
        return {'frames': frames, 'frame_cursors': cursors, 'cursor': _cursor(state),
                'has_more': offset + len(raw) < size or b'\n' in raw[consumed:],
                'partial_tail': offset + len(raw) == size and bool(raw) and not raw.endswith(b'\n')}

"""Bounded observation services shared by HTTP, browser and agent consumers.

One tail per file fans unchanged frames out. Only explicit historical bootstrap
jobs reduce; connected viewers never create live reduced state on the server.
"""
import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from itertools import islice
import re
import time

from .event_views import MapSpec, ViewSpec, read_projection_page
from .run_events import read_event_page
from .web_files import read_json, read_bytes, _open

MAX_QUEUE_BYTES = 1048576
MAX_GROUPS = 2048
MAX_STREAMS = 64
MAX_ATTEMPTS = 4096
MAX_BOOTSTRAPS = 8
# Preview retention defaults to the whole run, so the sample history a viewer
# scrubs through is long: a 100k-step run at --preview-every 100 publishes a
# thousand generations. The index stays bounded and validated, with room for
# several times that before a run must opt into --preview-keep.
MAX_PREVIEWS = 4096
PREVIEW_INDEX_BYTES = 16 * 1048576
_HEX = re.compile('[0-9a-f]{64}\\Z')


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


_SAMPLE_NAME = re.compile('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}')


def _sample_name(*candidates):
    """Short bounded group name for the viewer; never a path or artifact identity."""
    for value in candidates:
        if isinstance(value, str) and _SAMPLE_NAME.fullmatch(value):
            return value
    return 'sample'


def cursor_offset(cursor):
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or len(cursor) > 4096:
        raise ValueError('Invalid stream cursor')
    try:
        data = json.loads(base64.b64decode(cursor.encode('ascii'), altchars=b'-_', validate=True))
        offset = data['offset']
        if type(offset) is not int or offset < 0:
            raise ValueError('Invalid stream offset')
        return offset
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise ValueError('Invalid stream cursor') from exc


def sse(event, value):
    # No SSE id: native EventSource acknowledgement is not an application commit.
    return ('event: ' + event + '\ndata: ' + _json(value) + '\n\n').encode()


class Subscriber:
    def __init__(self, stream_id, *, max_bytes=MAX_QUEUE_BYTES):
        self.stream_id = stream_id
        self.queue = asyncio.Queue(maxsize=256)
        self.size = 0
        self.max_bytes = max_bytes
        self.closed = False

    def accepts(self, stream_id):
        return self.stream_id == '*' or self.stream_id == stream_id

    def offer(self, event, data):
        if self.closed:
            return
        encoded = sse(event, data)
        if self.size + len(encoded) > self.max_bytes or self.queue.full():
            self.closed = True
            while not self.queue.empty():
                self.queue.get_nowait()
            encoded = sse('gap', {'reason': 'consumer_queue_exceeded', 'recover': 'reconnect_from_last_applied_cursor'})
            self.size = len(encoded)
            self.queue.put_nowait(encoded)
            return
        self.size += len(encoded)
        self.queue.put_nowait(encoded)

    async def get(self):
        result = await self.queue.get()
        self.size -= len(result)
        return result


@dataclass
class Stream:
    stream_id: str
    cursor: str | None = None
    sequence: int = 0
    caught_up: bool = False
    error: str | None = None
    generation: str | None = None
    projected_offset: int | None = None
    task: object = None


@dataclass
class BootstrapJob:
    key: str
    query: dict
    task: object = None
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.monotonic)
    completed: float | None = None
    delivered: bool = False


class ObservationService:
    def __init__(self, run_dir, *, poll_seconds=.25, history_timeout=180):
        self.root = Path(run_dir).resolve()
        if type(history_timeout) not in (int, float) or not math.isfinite(history_timeout) or not 1 <= history_timeout <= 3600:
            raise ValueError('history timeout must be finite seconds in 1..3600')
        if type(poll_seconds) not in (int, float) or not math.isfinite(poll_seconds) or not .005 <= poll_seconds <= 60:
            raise ValueError('poll interval must be finite seconds in .005..60')
        self.poll_seconds = poll_seconds
        self.history_timeout = history_timeout
        self.streams = {}
        self.source_paths = {'training': 'events.jsonl'}
        self.subscribers = set()
        self.jobs = OrderedDict()
        self.attempts = {}
        self.artifacts = {}
        self._artifact_signature = None
        self._artifact_lock = asyncio.Lock()
        self.run_id = None
        self.manifest = {'status': 'waiting'}
        self.metadata_revision = None
        self._discovery = None
        self._closed = False
        self._history_slots = asyncio.Semaphore(1)
        self._source_revision = 0
        self.discovery_error = None

    async def start(self):
        await self._activate()
        self._discovery = asyncio.create_task(self._watch())
        return self

    async def _activate(self):
        if not (self.root / 'manifest.json').exists():
            return
        self.manifest = await asyncio.to_thread(read_json, self.root, 'manifest.json')
        self.run_id = self.manifest['run_id']
        self.metadata_revision = self._metadata_key(self.manifest)
        self._register('training')
        await self.discover()
        await self.refresh_artifacts()
        self.notify('metadata', {'run': self.public_manifest(), 'revision': self.metadata_revision})

    @staticmethod
    def _metadata_key(manifest):
        return _digest({k: manifest.get(k) for k in ('run_id', 'attempt_id', 'metrics_catalog', 'status', 'recovery_parent')})

    def public_manifest(self):
        fields = ('schema_version', 'run_id', 'attempt_id', 'attempt_index', 'status', 'steps',
                  'total_steps', 'last_durable_step', 'seconds', 'metrics_catalog',
                  'observation_sha256', 'checkpoint_every', 'stop_reason', 'possible_lost_steps',
                  'evaluation_schedule')
        result = {key: self.manifest[key] for key in fields if key in self.manifest}
        result['name'] = self.manifest.get('config', {}).get('name', self.root.name)
        boundary = self.manifest.get('durable_event_boundary')
        if isinstance(boundary, dict):
            result['durable_event_boundary'] = boundary
        result['metric_consistency'] = self.metric_consistency()
        return result

    def metric_consistency(self):
        """Report projection progress separately from the durable training prefix.

        This is server-observed projection progress, not an acknowledgement that
        a particular browser has rendered those frames or a new durability check.
        A caught-up training tail alone says nothing about the metric projector.
        """
        boundary = self.manifest.get('durable_event_boundary')
        result = {'status': 'unavailable', 'committed_step': None,
                  'committed_offset': None, 'projected_offset': None}
        if (not isinstance(boundary, dict) or boundary.get('run_id') != self.run_id
                or type(boundary.get('offset')) is not int or boundary['offset'] < 1
                or type(boundary.get('step')) is not int or boundary['step'] < 0):
            return result
        result.update(committed_step=boundary['step'], committed_offset=boundary['offset'])
        stream = self.streams.get('projection:' + MapSpec().revision)
        if stream is not None and stream.error:
            return result
        projected = stream.projected_offset if stream is not None else None
        result.update(projected_offset=projected,
                      status='caught_up' if projected is not None and projected >= boundary['offset'] else 'pending')
        return result

    def artifact_index(self):
        return {'schema_version': 1, 'artifacts': {
            key: {name: value for name, value in record.items() if name != 'path'}
            for key, record in self.artifacts.items()}}

    async def refresh_artifacts(self):
        async with self._artifact_lock:
            def inventory():
                signature = [self.manifest.get('sample_path')]
                for relative in ('previews/index.json', 'artifacts/index.json'):
                    path = self.root / relative
                    try:
                        stat = path.lstat()
                        signature.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
                    except FileNotFoundError:
                        signature.append(None)
                if signature == self._artifact_signature:
                    return None
                records = {}
                def relative_path(value):
                    path = Path(value)
                    if path.is_absolute():
                        path = path.relative_to(self.root)
                    return path.as_posix()
                if signature[1] is not None:
                    index = read_json(self.root, 'previews/index.json',
                                      max_bytes=PREVIEW_INDEX_BYTES)
                    previews = index.get('previews')
                    if (index.get('schema_version') != 1 or index.get('run_id') != self.run_id
                            or not isinstance(previews, list) or len(previews) > MAX_PREVIEWS):
                        raise ValueError('Invalid bounded preview index')
                    for preview in previews:
                        identity = preview['identity']
                        # Digest keys stay the artifact identity; `name` indexes the
                        # source across steps so a viewer can group its history.
                        key = 'preview-' + _digest(identity)[:32]
                        name = _sample_name(preview.get('name'), identity.get('name'))
                        records[key] = dict(path=relative_path(preview['path']), bytes=preview['bytes'],
                            sha256=preview.get('sha256'), role='sample', modality='tensor', name=name,
                            media_type='application/json', provenance=dict(identity, step=preview['step'], name=name),
                            shape=preview['shape'])
                        if not isinstance(records[key]['sha256'], str):
                            records[key].update(status='unavailable', reason='Preview predates indexed content digests')
                        for field, suffix in (('image_grid', '-grid'), ('real_image_grid', '-real-grid')):
                            grid = preview.get(field)
                            if grid is None:
                                continue
                            grid_name = _sample_name(grid.get('name'), name if field == 'image_grid' else 'x')
                            record = dict(path=relative_path(grid['path']), bytes=grid['bytes'],
                                sha256=grid['sha256'], role='sample', modality='image', media_type='image/png',
                                name=grid_name,
                                provenance=dict(identity, step=preview['step'], name=grid_name),
                                width=grid['width'], height=grid['height'])
                            if field == 'image_grid':
                                record['shape'] = preview['shape']
                            records[key + suffix] = record
                if signature[2] is not None:
                    index = read_json(self.root, 'artifacts/index.json')
                    indexed = index.get('artifacts')
                    if index.get('schema_version') != 1 or not isinstance(indexed, dict) or len(indexed) > 256:
                        raise ValueError('Invalid bounded artifact index')
                    for key, record in indexed.items():
                        if key in records or not isinstance(key, str) or not 1 <= len(key) <= 128:
                            raise ValueError('Invalid/duplicate artifact ID')
                        # Explicit indexes may name their source; otherwise the stable
                        # artifact ID is its own group name.
                        records[key] = dict(record, path=relative_path(record['path']),
                                            name=_sample_name(record.get('name'), key))
                if signature[0]:
                    path = relative_path(signature[0])
                    key = 'final-sample-' + _digest(path)[:32]
                    try:
                        data = read_bytes(self.root, path, max_bytes=16 * 1048576)
                        payload = json.loads(data)
                        saved = payload.get('identity') if isinstance(payload.get('identity'), dict) else {}
                        name = _sample_name(payload.get('name'), saved.get('name'))
                        records[key] = dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            role='sample', modality='tensor', media_type='application/json', name=name,
                            provenance=dict(saved, step=payload.get('step'), name=name),
                            shape=payload.get('shape'))
                    except (ValueError, OSError) as exc:
                        records[key] = dict(status='unavailable', role='sample', reason=str(exc), name='final')
                return signature, records
            result = await asyncio.to_thread(inventory)
            if result is not None:
                self._artifact_signature, self.artifacts = result
                self.notify('artifacts', {'revision': _digest(self.artifact_index())})

    async def close(self):
        self._closed = True
        tasks = [s.task for s in self.streams.values()] + [job.task for job in self.jobs.values()]
        if self._discovery:
            tasks.append(self._discovery)
        for task in tasks:
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in tasks if t), return_exceptions=True)

    def notify(self, event, value):
        for subscriber in list(self.subscribers):
            subscriber.offer(event, value)

    def _register(self, stream_id):
        if stream_id in self.streams:
            return
        if len(self.streams) >= MAX_STREAMS:
            raise ValueError('Run exceeds 64 registered observation streams')
        stream = Stream(stream_id)
        self.streams[stream_id] = stream
        self.notify('stream_added', {'stream_id': stream_id})
        stream.task = asyncio.create_task(self._tail(stream))

    async def discover(self):
        # Bad/over-budget optional inventories must not freeze run metadata or
        # trigger endless bootstrap resets for already registered consumers.
        previous = self.discovery_error
        try:
            await self._discover()
            self.discovery_error = None
        except (OSError, ValueError, KeyError) as exc:
            self.discovery_error = str(exc)
        if self.discovery_error != previous:
            self.notify('discovery_error', {'reason': self.discovery_error or '', 'status': 'error' if self.discovery_error else 'available'})

    async def _discover(self):
        for relative, kind in (('views', 'projection'), ('metrics/evaluations', 'evaluation')):
            directory = self.root / relative
            if not directory.exists():
                continue
            found = await asyncio.to_thread(lambda: list(islice(directory.iterdir(), MAX_STREAMS + 1)))
            if len(found) > MAX_STREAMS:
                raise ValueError('Run exceeds bounded observation directory count')
            for path in sorted(found):
                if path.is_symlink() or not path.is_dir():
                    continue
                if kind == 'projection' and _HEX.fullmatch(path.name) and (path / 'projection.json').is_file():
                    self._register('projection:' + path.name)
                elif kind == 'evaluation' and re.fullmatch('[0-9a-f]{32}', path.name) and (path / 'stream.json').is_file():
                    stream_id = 'evaluation:' + path.name
                    if stream_id in self.streams:
                        continue
                    record = await asyncio.to_thread(read_json, self.root, relative + '/' + path.name + '/stream.json')
                    expected = relative + '/' + path.name + '/events.jsonl'
                    if (record.get('schema_version') != 1 or record.get('run_id') != self.run_id
                            or record.get('stream_id') != stream_id or record.get('stream_generation') != path.name
                            or record.get('path') != expected):
                        raise ValueError('Invalid evaluation stream registration')
                    self._register(stream_id)
                    self.source_paths[stream_id] = expected

    async def _watch(self):
        while True:
            try:
                if self.run_id is None:
                    await self._activate()
                    await asyncio.sleep(self.poll_seconds)
                    continue
                await self.discover()
                manifest = await asyncio.to_thread(read_json, self.root, 'manifest.json')
                if manifest.get('run_id') != self.run_id:
                    raise ValueError('Observed run identity changed')
                revision = self._metadata_key(manifest)
                changed = manifest != self.manifest
                self.manifest = manifest
                await self.refresh_artifacts()
                if changed:
                    self.notify('heartbeat', {'run': self.public_manifest()})
                if revision != self.metadata_revision:
                    self.manifest, self.metadata_revision = manifest, revision
                    self.notify('metadata', {'run': self.public_manifest(), 'revision': revision})
            except (OSError, ValueError, KeyError) as exc:
                self.notify('reset_required', {'reason': str(exc)})
            await asyncio.sleep(self.poll_seconds)

    def page(self, stream_id, cursor, *, limit=256):
        def safe_open(path):
            return _open(self.root, Path(path).relative_to(self.root).as_posix())
        if stream_id in self.source_paths:
            source = self.root / self.source_paths[stream_id]
            if not source.exists():
                raise FileNotFoundError('Training event stream has not been published yet')
            page = read_event_page(source.parent, cursor, limit=limit, max_bytes=1048576, include_cursors=True, _open_file=safe_open,
                _stream_identity={'run_id': self.run_id, 'stream_id': stream_id,
                                  'stream_generation': self.run_id if stream_id == 'training' else stream_id[11:]})
            for event in page['events']:
                if event.get('run_id') != self.run_id or event.get('stream_id', 'training') != stream_id:
                    raise ValueError('Source document run/stream identity differs from its registration')
            return page['events'], page['event_cursors'], page
        if not stream_id.startswith('projection:') or not _HEX.fullmatch(stream_id[11:]):
            raise ValueError('Unknown stream selector')
        if not (self.root / 'views' / stream_id[11:] / 'contributions.jsonl').exists():
            raise FileNotFoundError('Projection log has not been published yet')
        page = read_projection_page(self.root, stream_id[11:], cursor, limit=limit, max_bytes=1048576, _open_file=safe_open)
        if any(frame['source']['run_id'] != self.run_id for frame in page['frames']):
            raise ValueError('Projection document run identity differs from the selected run')
        return page['frames'], page['frame_cursors'], page

    def _index_source(self, event):
        if event.get('event') in ('start', 'resume'):
            attempt = event['attempt_id']
            if attempt in self.attempts or event['sequence'] != 1:
                raise ValueError('Duplicate or misplaced attempt lifecycle event')
            if event.get('parent_attempt_id') is not None and not isinstance(event['parent_attempt_id'], str):
                raise ValueError('Invalid recovery parent attempt identity')
            if attempt not in self.attempts and len(self.attempts) >= MAX_ATTEMPTS:
                raise ValueError('Run exceeds bounded lineage index of 4096 attempts')
            self.attempts[attempt] = {'parent': event.get('parent_attempt_id'),
                                     'restored_step': event.get('restored_step', 0)}
            self.notify('bootstrap_ready', {'status': 'lineage_ready'})

    async def _tail(self, stream):
        while True:
            try:
                frames, cursors, page = await asyncio.to_thread(self.page, stream.stream_id, stream.cursor)
                for frame, cursor in zip(frames, cursors):
                    if stream.stream_id in self.source_paths:
                        generation = frame.get('stream_generation', frame['run_id'])
                        if stream.generation is not None and stream.generation != generation:
                            raise ValueError('Source stream generation changed')
                        stream.generation = generation
                    if stream.stream_id == 'training':
                        self._index_source(frame)
                    elif stream.stream_id.startswith('projection:'):
                        if frame['source']['stream_id'] != 'training' or frame['source']['stream_generation'] != self.run_id:
                            raise ValueError('Metric projection is not from the training stream')
                        offset = cursor_offset(frame['source_cursor'])
                        if offset <= (stream.projected_offset or 0):
                            raise ValueError('Projection source cursor did not advance')
                        stream.projected_offset = offset
                    stream.cursor = cursor
                    stream.sequence = frame.get('projection_sequence', stream.sequence + 1)
                    envelope = {'stream_id': stream.stream_id, 'cursor': cursor, 'frame': frame}
                    for subscriber in list(self.subscribers):
                        if subscriber.accepts(stream.stream_id):
                            subscriber.offer('frame', envelope)
                stream.cursor = page['cursor']
                was_caught_up = stream.caught_up
                stream.caught_up = not page['has_more']
                if stream.caught_up and not was_caught_up:
                    self.notify('bootstrap_ready', {'stream_id': stream.stream_id, 'status': 'index_ready'})
                stream.error = None
                if frames and stream.stream_id == 'projection:' + MapSpec().revision:
                    self.notify('heartbeat', {'run': self.public_manifest()})
                if not page['has_more']:
                    await asyncio.sleep(self.poll_seconds)
                else:
                    await asyncio.sleep(0)
            except FileNotFoundError as exc:
                if stream.projected_offset is not None:
                    changed = stream.error != str(exc)
                    stream.error = str(exc)
                    if changed and stream.stream_id == 'projection:' + MapSpec().revision:
                        self.notify('heartbeat', {'run': self.public_manifest()})
                await asyncio.sleep(self.poll_seconds)
            except (OSError, ValueError, KeyError) as exc:
                stream.error = str(exc)
                if stream.stream_id == 'projection:' + MapSpec().revision:
                    self.notify('heartbeat', {'run': self.public_manifest()})
                self.notify('reset_required', {'stream_id': stream.stream_id, 'reason': str(exc)})
                # Corrupt/replaced complete history must not be silently skipped.
                return

    def subscribe(self, stream_id):
        if len(self.subscribers) >= 32:
            raise ValueError('Viewer has reached its 32-consumer limit')
        if stream_id != '*' and stream_id != 'training' and not (
                (stream_id.startswith('projection:') and _HEX.fullmatch(stream_id[11:])) or
                (stream_id.startswith('evaluation:') and re.fullmatch('[0-9a-f]{32}', stream_id[11:]))):
            raise ValueError('Invalid stream selector')
        subscriber = Subscriber(stream_id)
        self.subscribers.add(subscriber)
        # No await between registration and snapshot: queued live suffix starts
        # after these watermarks. Historical replay through them cannot lose events.
        watermarks = {key: value.cursor for key, value in self.streams.items() if subscriber.accepts(key)}
        return subscriber, watermarks

    async def events(self, stream_id, cursor=None):
        subscriber, watermarks = self.subscribe(stream_id)
        try:
            if cursor is not None and stream_id == '*':
                raise ValueError('A replay cursor requires one explicit stream')
            if cursor is not None:
                if stream_id not in self.streams:
                    raise ValueError('Cannot replay a missing projection')
                target = cursor_offset(watermarks[stream_id])
                if cursor_offset(cursor) > target:
                    # Indexer is behind the acknowledged client. Wait for the
                    # shared tail; do not invent a reduced state or lose its suffix.
                    target = cursor_offset(cursor)
                replay = cursor
                replay_count = 0
                while cursor_offset(replay) < target:
                    frames, cursors, page = await asyncio.to_thread(self.page, stream_id, replay)
                    if not frames:
                        raise ValueError('Replay watermark is no longer available')
                    for frame, end in zip(frames, cursors):
                        if cursor_offset(end) > target:
                            break
                        replay_count += 1
                        if replay_count > 4096:
                            yield sse('gap', {'reason': 'replay_budget_exceeded', 'recover': 'request_fresh_bootstrap', 'retry_after_seconds': 5})
                            return
                        replay = end
                        yield sse('frame', {'stream_id': stream_id, 'cursor': end, 'frame': frame})
                    if subscriber.closed:
                        break
                # Validate even an empty replay against the underlying file.
                await asyncio.to_thread(self.page, stream_id, replay, limit=1)
                watermarks[stream_id] = replay
            yield sse('ready', {'streams': watermarks, 'run': self.public_manifest()})
            while True:
                try:
                    encoded = await asyncio.wait_for(subscriber.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield sse('heartbeat', {'run': self.public_manifest()})
                    continue
                if encoded.startswith(b'event: frame\n'):
                    value = json.loads(encoded.split(b'data: ', 1)[1])
                    marker = watermarks.get(value['stream_id'])
                    if marker is not None and cursor_offset(value['cursor']) <= cursor_offset(marker):
                        continue
                    watermarks[value['stream_id']] = value['cursor']
                yield encoded
                if subscriber.closed:
                    return
        except (ValueError, OSError, KeyError) as exc:
            yield sse('reset_required', {'reason': str(exc)})
        finally:
            self.subscribers.discard(subscriber)

    def lineage(self):
        source = self.streams.get('training')
        if source is None:
            raise LookupError('Run is waiting for its first events')
        if not source.caught_up:
            raise LookupError('Source lineage index is catching up')
        if source.error:
            raise ValueError(source.error)
        current = self.manifest.get('attempt_id')
        result, seen, through = [], set(), None
        while current is not None:
            if current in seen or len(seen) >= MAX_ATTEMPTS:
                raise ValueError('Invalid recovery lineage cycle/length')
            seen.add(current)
            attempt = self.attempts.get(current)
            if attempt is None:
                raise LookupError('Selected attempt is not yet indexed')
            result.append({'attempt_id': current, 'through_step': through})
            restored = attempt['restored_step']
            if type(restored) is not int or restored < 0:
                raise ValueError('Invalid restored step in source lineage')
            through = restored if through is None else min(through, restored)
            current = attempt['parent']
        return list(reversed(result))

    def bootstrap(self, map_revision, *, series, bucket_steps, step_from=0, step_to=None):
        if not _HEX.fullmatch(map_revision):
            raise ValueError('Invalid map revision')
        if type(bucket_steps) is not int or not 1 <= bucket_steps <= 2 ** 53 - 1:
            raise ValueError('bucket_steps must be a positive safe integer')
        if type(step_from) is not int or not 0 <= step_from <= 2 ** 53 - 1:
            raise ValueError('step_from must be a nonnegative safe integer')
        if step_to is not None and (type(step_to) is not int or not step_from <= step_to <= 2 ** 53 - 1):
            raise ValueError('step_to must be a safe integer at least step_from')
        if not isinstance(series, list) or not 1 <= len(series) <= 32 or len(set(series)) != len(series):
            raise ValueError('Select 1..32 distinct metric IDs')
        if any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in series):
            raise ValueError('Invalid metric IDs')
        query = dict(map_revision=map_revision, series=sorted(series), bucket_steps=bucket_steps,
                     step_from=step_from, step_to=step_to)
        lineage = self.lineage()
        query['lineage_revision'] = _digest(lineage)
        key = _digest(query)
        stream = self.streams.get('projection:' + map_revision)
        if stream is None:
            raise FileNotFoundError('Projection missing; run the explicit projection service')
        if stream.error:
            raise ValueError(stream.error)
        if key in self.jobs:
            cached = self.jobs[key]
            age = time.monotonic() - cached.completed if cached.completed is not None else 0
            lag = stream.sequence - cached.result['projection_sequence'] if cached.result is not None else 0
            # One fixed-H job while pending, then a completion lease for its
            # notification/refetch. Only a later explicit query may refresh it.
            if (cached.completed is None or not cached.delivered or age < 5
                    or lag == 0 or (lag <= 4096 and age <= 30)):
                if cached.completed is not None:
                    cached.delivered = True
                self.jobs.move_to_end(key)
                return cached
            self.jobs.pop(key)
        if not stream.caught_up:
            raise LookupError('Projection index is catching up')
        if len(self.jobs) >= MAX_BOOTSTRAPS:
            completed = next((k for k, job in self.jobs.items() if job.task.done()), None)
            if completed is None:
                raise ValueError('Historical bootstrap queue is full')
            self.jobs.pop(completed)
        job = BootstrapJob(key, query)
        self.jobs[key] = job
        job.task = asyncio.create_task(self._reduce_history(job, stream.cursor, stream.sequence, lineage))
        return job

    async def _reduce_history(self, job, cursor, sequence, lineage):
        try:
            async with self._history_slots:
                result = await self._history(job.query, cursor, sequence, lineage)
            job.result = result
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            job.error = str(exc)
        job.completed = time.monotonic()
        self.notify('bootstrap_ready', {'query': job.query, 'job_id': job.key,
                                       'status': 'error' if job.error else 'ready'})

    async def _history(self, query, cursor, sequence, lineage):
        from .metrics_reducer import Reducer
        reducer = await asyncio.to_thread(Reducer)
        started = time.monotonic()
        states, replay = {}, None
        allowed = {x['attempt_id']: x['through_step'] for x in lineage}
        selected = set(query['series'])
        target = cursor_offset(cursor)
        while cursor_offset(replay) < target:
            if time.monotonic() - started > self.history_timeout:
                raise ValueError('Historical reduction exceeded its time budget; retry with a larger explicit history budget or prepare a smaller projection')
            frames, cursors, page = await asyncio.to_thread(self.page, 'projection:' + query['map_revision'], replay)
            if not frames:
                raise ValueError('Historical projection watermark disappeared')
            pending = {}
            for frame, end in zip(frames, cursors):
                if cursor_offset(end) > target:
                    break
                replay = end
                for emission in frame['emissions']:
                    metric, attempt, step = emission['key']
                    if (metric not in selected or attempt not in allowed or
                            (allowed[attempt] is not None and step > allowed[attempt]) or
                            step < query['step_from'] or
                            (query['step_to'] is not None and step > query['step_to'])):
                        continue
                    key = (metric, emission['definition_hash'], attempt,
                           step // query['bucket_steps'] * query['bucket_steps'])
                    pending.setdefault(key, []).append({'value': emission['value'], 'position': [step, emission['id']]})
            if len(set(states) | set(pending)) > MAX_GROUPS:
                raise ValueError('View exceeds 2048 groups; increase bucket_steps or narrow step range')
            def reduce_page():
                for key, points in pending.items():
                    state = states.get(key)
                    if state is None:
                        state = reducer.identity('envelope/v1')
                    for start in range(0, len(points), 1024):
                        state = reducer.add(state, points[start:start + 1024])
                    states[key] = state
            await asyncio.to_thread(reduce_page)
            await asyncio.sleep(0)
        view = ViewSpec(query['map_revision'], bucket_steps=query['bucket_steps'])
        result = dict(schema_version=1, run_id=self.run_id, map_revision=query['map_revision'],
                      view_revision=view.revision, module_sha256=reducer.module_sha256,
                      bucket_steps=query['bucket_steps'], lineage_revision=query['lineage_revision'],
                      lineage=lineage, groups=[{'key': list(k), 'state': v} for k, v in sorted(states.items())],
                      cursor=cursor, projection_sequence=sequence, coverage={'complete': True},
                      step_from=query['step_from'], step_to=query['step_to'])
        if len(_json(result).encode()) > 1048576:
            raise ValueError('Bootstrap response exceeds 1 MiB; narrow the view')
        return result

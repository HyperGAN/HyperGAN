import asyncio
import hashlib
import json
from pathlib import Path
import socket

import httpx
import pytest
from starlette.testclient import TestClient

from hypergan.event_views import MapSpec, Projector
from hypergan.metrics import digest
from hypergan.metrics_reducer import Reducer
from hypergan.run_state import atomic_json
from hypergan.web_service import ObservationService, Subscriber
from hypergan.web_server import create_app
from hypergan.web_session import LocalSession


def fixture_run(root, count=5):
    root.mkdir(exist_ok=True)
    definition = {'kind': 'scalar', 'source': 'g_loss', 'label': 'Generator loss'}
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {'loss/g_total': definition}}
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    manifest = dict(schema_version=1, run_id='run', attempt_id='a', steps=count,
                    status='running', metrics_catalog=revision)
    atomic_json(root / 'manifest.json', manifest)
    append_event(root, 1, 0, event='start')
    for step in range(1, count + 1):
        append_event(root, step + 1, step)
    with Projector(root) as projector:
        projector.project(limit=10000)
    return manifest


def append_event(root, sequence, step, *, attempt='a', event='train', parent=None, restored=0):
    manifest = json.loads((root / 'manifest.json').read_bytes())
    row = dict(schema_version=2, run_id='run', attempt_id=attempt, stream_id='training',
               stream_generation='run', sequence=sequence, step=step, event=event,
               catalog=manifest['metrics_catalog'])
    if event in ('start', 'resume'):
        row.update(parent_attempt_id=parent, restored_step=restored)
    else:
        row['metrics'] = {'loss/g_total': float(step)}
    with (root / 'events.jsonl').open('ab') as handle:
        handle.write(json.dumps(row).encode() + b'\n')


async def caught_up(service):
    for _ in range(300):
        if all(s.caught_up or s.error for s in service.streams.values()) and service.streams:
            return
        await asyncio.sleep(.01)
    raise AssertionError('index did not finish')


async def get_bootstrap(service, **kwargs):
    await caught_up(service)
    job = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2, **kwargs)
    await job.task
    assert job.error is None, job.error
    return job.result


def test_auth_api_schema_artifact_and_no_mapper(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    import hypergan.event_views as maps
    monkeypatch.setattr(maps.Projector, '__enter__', lambda self: pytest.fail('HTTP executed mapper'))
    session = LocalSession(8123)
    session.write_credentials(tmp_path / 'session.json')
    token = json.loads((tmp_path / 'session.json').read_text())['token']
    app = create_app(tmp_path, session, poll_seconds=.01)
    with TestClient(app, base_url=session.origin) as client:
        assert client.get('/api/v1/capabilities').status_code == 401
        assert client.post('/api/v1/session', json={'token': token}, headers={'origin': 'https://foreign.test'}).status_code == 403
        assert client.post('/api/v1/session', json={'token': token}).status_code == 200
        assert 'HttpOnly' in client.cookies.__repr__() or client.cookies.get(session.cookie_name)
        capability = client.get('/api/v1/capabilities').json()
        assert capability['run_id'] == 'run'
        assert client.get('/api/v1/runs/run').json()['attempt_id'] == 'a'
        assert client.get('/api/v1/runs/run/metrics/catalog').json()['metrics']['loss/g_total']['kind'] == 'scalar'
        assert client.get('/api/v1/runs/run/events?limit=2').json()['has_more']
        assert client.get('/api/v1/runs/run/events?limit=0').status_code == 400
        assert client.get('/api/v1/runs/run/views').json()['status'] == 'available'
        assert client.get('/api/v1/openapi.json').json()['openapi'] == '3.1.0'
        assert client.get('/api/v1/runs/foreign').status_code == 404
        assert client.get('/api/v1/capabilities', headers={'host': 'malicious.example'}).status_code == 403
        assert client.get('/reducers/reducer.wasm').content.startswith(b'\x00asm')
        assert 'wasm-unsafe-eval' in client.get('/reducers/host.js').headers['content-security-policy']
        assert client.get('/api/v1/runs/run/artifacts').json()['artifacts'] == {}
        data = b'bounded sample'
        (tmp_path / 'sample.bin').write_bytes(data)
        atomic_json(tmp_path / 'artifacts' / 'index.json', {'schema_version': 1, 'artifacts': {
            'sample': {'path': 'sample.bin', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}})
        artifact = client.get('/api/v1/runs/run/artifacts/sample')
        assert artifact.content == data and artifact.headers['content-disposition'].startswith('attachment')
        (tmp_path / 'sample.bin').write_bytes(b'corrupt')
        assert client.get('/api/v1/runs/run/artifacts/sample').status_code == 400


def test_bootstrap_fixed_watermark_then_live_continuation(tmp_path, monkeypatch):
    fixture_run(tmp_path, 4)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            initial = await get_bootstrap(service)
            assert initial['projection_sequence'] == 5
            assert initial['lineage'] == [{'attempt_id': 'a', 'through_step': None}]
            reducer = Reducer()
            totals = sum(reducer.finalize(g['state'])['count'] for g in initial['groups'])
            assert totals == 4
            # Live fanout must not execute reduction even with multiple viewers.
            monkeypatch.setattr(Reducer, 'add', lambda *args: pytest.fail('live server reduction'))
            stream = 'projection:' + MapSpec().revision
            subscribers = [service.events(stream, initial['cursor']) for _ in range(2)]
            for subscriber in subscribers:
                assert (await anext(subscriber)).startswith(b'event: ready')
            append_event(tmp_path, 6, 5)
            with Projector(tmp_path) as projector:
                projector.project()
            frames = [await asyncio.wait_for(anext(subscriber), 3) for subscriber in subscribers]
            assert frames[0] == frames[1]
            frame = json.loads(frames[0].split(b'data: ', 1)[1])
            assert frame['frame']['projection_sequence'] == 6
            # Already cached bootstrap keeps H; consumer replay supplies its suffix.
            assert service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2).result == initial
            for subscriber in subscribers:
                await subscriber.aclose()
        finally:
            await service.close()
    asyncio.run(scenario())


def test_replay_gapless_and_slow_subscriber_bounded(tmp_path):
    fixture_run(tmp_path, 2)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            initial = await get_bootstrap(service)
            append_event(tmp_path, 4, 3)
            append_event(tmp_path, 5, 4)
            with Projector(tmp_path) as projector:
                projector.project()
            await asyncio.sleep(.1)
            iterator = service.events('projection:' + MapSpec().revision, initial['cursor'])
            one, two, ready = await anext(iterator), await anext(iterator), await anext(iterator)
            assert json.loads(one.split(b'data: ')[1])['frame']['projection_sequence'] == 4
            assert json.loads(two.split(b'data: ')[1])['frame']['projection_sequence'] == 5
            assert ready.startswith(b'event: ready')
            await iterator.aclose()
            slow = Subscriber('training', max_bytes=150)
            slow.offer('frame', {'huge': 'x' * 300})
            assert slow.closed and slow.queue.qsize() == 1
            assert (await slow.get()).startswith(b'event: gap')
        finally:
            await service.close()
    asyncio.run(scenario())


def test_earlier_checkpoint_lineage_zero_updates_and_new_stream(tmp_path):
    fixture_run(tmp_path, 5)
    manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
    manifest.update(attempt_id='b', steps=2, recovery_parent={'attempt_id': 'a', 'step': 2})
    atomic_json(tmp_path / 'manifest.json', manifest)
    append_event(tmp_path, 1, 2, attempt='b', event='resume', parent='a', restored=2)
    with Projector(tmp_path) as projector:
        projector.project()
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            result = await get_bootstrap(service)
            assert result['lineage'] == [{'attempt_id': 'a', 'through_step': 2}, {'attempt_id': 'b', 'through_step': None}]
            assert sum(g['state']['count'] for g in result['groups']) == 2
            subscriber, _ = service.subscribe('*')
            other = MapSpec(version='2')
            with Projector(tmp_path, other) as projector:
                projector.project()
            await service.discover()
            frames = []
            for _ in range(10):
                event = await asyncio.wait_for(subscriber.get(), 3)
                frames.append(event)
                if event.startswith(b'event: frame'):
                    break
            assert frames[0].startswith(b'event: stream_added')
            assert any(other.revision.encode() in e for e in frames)
            service.subscribers.remove(subscriber)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_corruption_visible_and_missing_projection_never_materialized(tmp_path):
    fixture_run(tmp_path, 1)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await caught_up(service)
            with pytest.raises(FileNotFoundError):
                service.bootstrap('a' * 64, series=['loss/g_total'], bucket_steps=1)
            assert not (tmp_path / 'views' / ('a' * 64)).exists()
            subscriber, _ = service.subscribe('*')
            path = tmp_path / 'views' / MapSpec().revision / 'contributions.jsonl'
            with path.open('ab') as handle:
                handle.write(b'{"bad":true}\n')
            while True:
                event = await asyncio.wait_for(subscriber.get(), 3)
                if event.startswith(b'event: reset_required'):
                    break
            assert service.streams['projection:' + MapSpec().revision].error
            service.subscribers.remove(subscriber)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_pending_run_activation(tmp_path):
    root = tmp_path / 'future'
    async def scenario():
        service = await ObservationService(root, poll_seconds=.01).start()
        try:
            assert service.run_id is None
            subscriber, _ = service.subscribe('*')
            fixture_run(root, 1)
            for _ in range(20):
                data = await asyncio.wait_for(subscriber.get(), 3)
                if data.startswith(b'event: metadata'):
                    break
            assert service.run_id == 'run'
            service.subscribers.remove(subscriber)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_real_asgi_stream_reconnect(tmp_path):
    fixture_run(tmp_path, 2)
    async def scenario():
        import uvicorn
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        session = LocalSession(sock.getsockname()[1])
        session.write_credentials(tmp_path / 'session.json')
        token = json.loads((tmp_path / 'session.json').read_bytes())['token']
        app = create_app(tmp_path, session, poll_seconds=.01)
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=sock.getsockname()[1], log_level='error'))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            for _ in range(300):
                if server.started:
                    break
                await asyncio.sleep(.01)
            async with httpx.AsyncClient(base_url=session.origin, headers={'Authorization': 'Bearer ' + token}, timeout=5) as client:
                assert (await client.get('/api/v1/capabilities')).status_code == 200
                await caught_up(app.state.observations)
                initial = await get_bootstrap(app.state.observations)
                url = '/api/v1/runs/run/stream'
                async with client.stream('GET', url, params={'stream_id': 'projection:' + MapSpec().revision, 'cursor': initial['cursor']}) as response:
                    assert response.status_code == 200
                    lines = response.aiter_lines()
                    assert await anext(lines) == 'event: ready'
                    append_event(tmp_path, 4, 3)
                    with Projector(tmp_path) as projector:
                        projector.project()
                    found = False
                    for _ in range(20):
                        line = await anext(lines)
                        if line == 'event: frame':
                            value = json.loads((await anext(lines))[6:])
                            assert value['frame']['projection_sequence'] == 4
                            found = True
                            break
                    assert found
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)
            sock.close()
    asyncio.run(scenario())


def test_preview_index_final_sample_and_safe_reader_paths(tmp_path):
    fixture_run(tmp_path, 1)
    from hypergan.previews import publish_preview_payload
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'attempt_index': 1, 'sample_sequence': 1}
    payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'step': 1,
               'count': 1, 'shape': [1, 2], 'samples': [[.2, .3]]}
    result = publish_preview_payload(tmp_path, payload, identity, 1)
    assert result[0]['sha256'] == hashlib.sha256(Path(result[0]['path']).read_bytes()).hexdigest()
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            index = service.artifact_index()
            assert len(index['artifacts']) == 1
            item = next(iter(index['artifacts'].values()))
            assert item['role'] == 'sample' and 'path' not in item
            final = tmp_path / 'final.json'
            final.write_text(json.dumps({'shape': [1, 2], 'step': 1, 'identity': {'attempt_id': 'a'}}))
            service.manifest['sample_path'] = str(final)
            await service.refresh_artifacts()
            assert len(service.artifacts) == 2
            # Event source paths reject symlinks rather than serving outside data.
            linked = ObservationService(tmp_path / 'linked')
            linked.root.mkdir()
            linked.run_id = 'run'
            try:
                (linked.root / 'events.jsonl').symlink_to(tmp_path / 'events.jsonl')
            except OSError:
                return  # Windows account may lack symlink privilege; other cases still ran.
            with pytest.raises(ValueError, match='unavailable|traverse links'):
                linked.page('training', None)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_future_evaluation_stream_registration(tmp_path):
    fixture_run(tmp_path, 1)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await caught_up(service)
            subscriber, _ = service.subscribe('*')
            evaluation_id = 'e' * 32
            directory = tmp_path / 'metrics' / 'evaluations' / evaluation_id
            directory.mkdir(parents=True)
            source = json.loads((tmp_path / 'events.jsonl').read_bytes().splitlines()[0])
            source.update(stream_id='evaluation:' + evaluation_id, stream_generation=evaluation_id,
                          event='evaluation', metrics={'loss/g_total': 42.0})
            (directory / 'events.jsonl').write_bytes(json.dumps(source).encode() + b'\n')
            atomic_json(directory / 'stream.json', {'schema_version': 1, 'run_id': 'run',
                'stream_id': source['stream_id'], 'stream_generation': evaluation_id,
                'path': f'metrics/evaluations/{evaluation_id}/events.jsonl', 'role': 'measurement'})
            await service.discover()
            assert (await subscriber.get()).startswith(b'event: stream_added')
            while True:
                event = await asyncio.wait_for(subscriber.get(), 3)
                if event.startswith(b'event: frame'):
                    assert json.loads(event.split(b'data: ')[1])['frame']['metrics']['loss/g_total'] == 42
                    break
            service.subscribers.remove(subscriber)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_real_cli_server_private_credentials_and_shutdown(tmp_path):
    import os
    import subprocess
    import sys
    import time
    fixture_run(tmp_path / 'run', 1)
    credential = tmp_path / 'credential.json'
    process = subprocess.Popen([sys.executable, '-I', '-m', 'hypergan', 'serve', str(tmp_path / 'run'),
                                '--port', '0', '--session-file', str(credential)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(500):
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            if credential.exists():
                break
            time.sleep(.01)
        record = json.loads(credential.read_bytes())
        if os.name != 'nt':
            assert credential.stat().st_mode & 0o077 == 0
        for _ in range(500):
            try:
                result = httpx.get(record['origin'] + '/api/v1/capabilities',
                                   headers={'Authorization': 'Bearer ' + record['token']}, timeout=1)
                break
            except httpx.ConnectError:
                time.sleep(.01)
        assert result.status_code == 200 and result.json()['run_id'] == 'run'
        process.terminate()
        out, err = process.communicate(timeout=15)
        assert record['token'] not in out + err
        # SIGTERM cleanup path follows the normal uvicorn return on POSIX.
        if os.name != 'nt':
            assert not credential.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_cached_bootstrap_rejected_after_stream_corruption(tmp_path):
    fixture_run(tmp_path, 1)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await get_bootstrap(service)
            stream = service.streams['projection:' + MapSpec().revision]
            stream.error = 'Projection generation changed'
            with pytest.raises(ValueError, match='generation changed'):
                service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_bootstrap_request_driven_freshness_and_pending_coalescing(tmp_path):
    fixture_run(tmp_path, 1)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await get_bootstrap(service)
            first = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2)
            stream = service.streams['projection:' + MapSpec().revision]
            stream.sequence += 5000
            assert service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2) is first
            first.completed -= 6  # Large lag after completed-result lease expires.
            assert service.jobs[first.key] is first  # No automatic refresh on live tail.
            second = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2)
            assert second is not first
            assert service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2) is second
            await second.task
        finally:
            await service.close()
    asyncio.run(scenario())


def test_blocked_asgi_send_closes_subscription(tmp_path):
    fixture_run(tmp_path, 1)
    async def scenario():
        from starlette.requests import ClientDisconnect
        session = LocalSession(8123)
        session.write_credentials(tmp_path / 'session.json')
        token = json.loads((tmp_path / 'session.json').read_bytes())['token']
        app = create_app(tmp_path, session, poll_seconds=.01)
        await app.state.observations.start()
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
                 'http_version': '1.1', 'method': 'GET', 'scheme': 'http',
                 'path': '/api/v1/stream', 'raw_path': b'/api/v1/stream', 'query_string': b'',
                 'headers': [(b'host', session.host.encode()), (b'authorization', ('Bearer ' + token).encode())],
                 'server': ('127.0.0.1', 8123), 'client': ('127.0.0.1', 9999), 'root_path': ''}
        async def receive():
            await asyncio.sleep(60)
            return {'type': 'http.disconnect'}
        async def send(message):
            if message['type'] == 'http.response.body':
                await asyncio.sleep(60)
        try:
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(app(scope, receive, send), 10)
            assert not app.state.observations.subscribers
        finally:
            await app.state.observations.close()
    asyncio.run(scenario())


def test_public_source_cursor_copy_generation_and_boundary(tmp_path):
    import base64
    import shutil
    original = tmp_path / 'original'
    fixture_run(original, 3)
    session = LocalSession(8123)
    session.write_credentials(tmp_path / 'credentials.json')
    token = json.loads((tmp_path / 'credentials.json').read_bytes())['token']
    headers = {'authorization': 'Bearer ' + token}
    with TestClient(create_app(original, session), base_url=session.origin) as client:
        page = client.get('/api/v1/runs/run/events?limit=2', headers=headers).json()
        cursor = page['cursor']
        encoded = json.loads(base64.urlsafe_b64decode(cursor))
        assert set(encoded) == {'version', 'run_id', 'stream_id', 'stream_generation', 'offset', 'anchor', 'last'}
        assert encoded['run_id'] == encoded['stream_generation'] == 'run'
        assert encoded['stream_id'] == 'training'
    copied = tmp_path / 'copied'
    shutil.copytree(original, copied)
    assert original.stat().st_ino != copied.stat().st_ino
    with TestClient(create_app(copied, session), base_url=session.origin) as client:
        result = client.get('/api/v1/runs/run/events', params={'cursor': cursor}, headers=headers)
        assert result.status_code == 200
        assert [event['step'] for event in result.json()['events']] == [2, 3]
        # An exhausted cursor must still check the first document's generation.
        end = result.json()['cursor']
        path = copied / 'events.jsonl'
        rows = path.read_bytes().splitlines()
        first = json.loads(rows[0]); first['stream_generation'] = 'new'
        rows[0] = json.dumps(first).encode()
        path.write_bytes(b'\n'.join(rows) + b'\n')
        result = client.get('/api/v1/runs/run/events', params={'cursor': end}, headers=headers)
        assert result.status_code == 400 and 'generation' in result.text
        first['stream_generation'] = 'run'; rows[0] = json.dumps(first).encode()
        row = json.loads(rows[1]); row['metrics']['loss/g_total'] = 9.0
        rows[1] = json.dumps(row).encode(); path.write_bytes(b'\n'.join(rows) + b'\n')
        result = client.get('/api/v1/runs/run/events', params={'cursor': cursor}, headers=headers)
        assert result.status_code == 400 and 'boundary' in result.text


def test_discovery_overflow_keeps_metadata_live(tmp_path):
    fixture_run(tmp_path, 1)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            directory = tmp_path / 'metrics/evaluations'
            directory.mkdir(parents=True)
            for index in range(65):
                (directory / f'{index:032x}').mkdir()
            await service.discover()
            assert 'bounded observation directory' in service.discovery_error
            manifest = dict(service.manifest, status='completed', steps=42)
            atomic_json(tmp_path / 'manifest.json', manifest)
            for _ in range(100):
                if service.manifest['steps'] == 42:
                    break
                await asyncio.sleep(.02)
            assert service.manifest['steps'] == 42
            assert service.manifest['status'] == 'completed'
            assert service.page('training', None)[0]
        finally:
            await service.close()
    asyncio.run(scenario())

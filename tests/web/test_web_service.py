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


def fixture_run(root, count=5, extra_metrics=None, evaluation_schedule=None):
    root.mkdir(exist_ok=True)
    definition = {'kind': 'scalar', 'source': 'g_loss', 'label': 'Generator loss'}
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {'loss/g_total': definition, **(extra_metrics or {})}}
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    manifest = dict(schema_version=1, run_id='run', attempt_id='a', steps=count,
                    status='running', metrics_catalog=revision)
    if evaluation_schedule is not None:
        manifest['evaluation_schedule'] = evaluation_schedule
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


def test_initialization_tuning_progress_is_public_and_updates_live(tmp_path):
    manifest = fixture_run(tmp_path)
    progress = {'status': 'running', 'candidate': 1, 'total_candidates': 3}
    manifest.update(status='tuning', initialization_tuning=progress)
    atomic_json(tmp_path / 'manifest.json', manifest)

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            assert service.public_manifest()['initialization_tuning'] == progress
            subscriber, _ = service.subscribe('*')
            result = {'status': 'complete', 'outcome': 'kept_baseline'}
            manifest.update(status='running', initialization_tuning=result)
            atomic_json(tmp_path / 'manifest.json', manifest)
            for _ in range(300):
                message = await asyncio.wait_for(subscriber.get(), timeout=3)
                if b'event: heartbeat' in message and b'kept_baseline' in message:
                    break
            else:
                raise AssertionError('Tuning completion was not published')
            assert service.public_manifest()['status'] == 'running'
            assert service.public_manifest()['initialization_tuning'] == result
        finally:
            await service.close()
    asyncio.run(scenario())


def test_auth_api_schema_artifact_and_no_mapper(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    import hypergan.event_views as maps
    monkeypatch.setattr(maps.Projector, '__enter__', lambda self: pytest.fail('HTTP executed mapper'))
    session = LocalSession(8123, auth="token")
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


def test_manual_snapshot_metric_trigger_and_empty_schedule_reach_the_browser(tmp_path):
    """The viewer's unscheduled notice derives from data the public API already sends."""
    snapshot = {'kind': 'scalar', 'source': 'custom:fid', 'label': 'FID50k', 'scope': 'snapshot',
                'specification': {'mode': 'snapshot', 'trigger': 'manual',
                                  'evaluation': {'device': 'cuda:0'}}}
    snapshot['definition_hash'] = digest(snapshot)
    fixture_run(tmp_path, extra_metrics={'fid50k_train': snapshot}, evaluation_schedule={})
    session = LocalSession(8124, auth='token')
    session.write_credentials(tmp_path / 'session.json')
    token = json.loads((tmp_path / 'session.json').read_text())['token']
    app = create_app(tmp_path, session, poll_seconds=.01)
    with TestClient(app, base_url=session.origin) as client:
        assert client.post('/api/v1/session', json={'token': token}).status_code == 200
        run = client.get('/api/v1/runs/run').json()
        assert run['evaluation_schedule'] == {}
        catalog = client.get('/api/v1/runs/run/metrics/catalog').json()
        definition = catalog['metrics']['fid50k_train']
        assert definition['scope'] == 'snapshot'
        assert definition['specification']['trigger'] == 'manual'
        # Nothing in the catalog is scheduled, which is exactly the notice's condition.
        assert not [name for name, value in catalog['metrics'].items()
                    if value.get('scope') == 'snapshot'
                    and value.get('specification', {}).get('trigger') == 'interval']
        schema = client.get('/api/v1/openapi.json').json()['components']['schemas']['Run']
        assert 'evaluation_schedule' in schema['properties']


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
        session = LocalSession(sock.getsockname()[1], auth="token")
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
            # A periodic preview is not the finished sample.
            assert 'final' not in item['provenance']
            final = tmp_path / 'final.json'
            final.write_text(json.dumps({'shape': [1, 2], 'step': 1, 'identity': {'attempt_id': 'a'}}))
            service.manifest['sample_path'] = str(final)
            await service.refresh_artifacts()
            assert len(service.artifacts) == 2
            # The run's finished sample says so, so a viewer can name it.
            saved = next(value for key, value in service.artifacts.items()
                         if key.startswith('final-sample-'))
            assert saved['provenance']['final'] is True and saved['modality'] == 'tensor'
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


def test_png_preview_is_authenticated_bounded_digest_checked_and_inline(tmp_path):
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    fixture_run(tmp_path, 1)
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    png = encode_png(bytes([255, 0, 0]), 1, 1, 3, {'step': 1})
    payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'step': 1,
               'count': 1, 'shape': [1, 3, 1, 1], 'samples': [[[[1]], [[-1]], [[-1]]]],
               'image_grid': {'width': 1, 'height': 1, 'channels': 3,
                              'png_base64': base64.b64encode(png).decode('ascii')}}
    published = publish_preview_payload(tmp_path, payload, identity, 1)[0]
    session = LocalSession(8123, auth="token")
    session.write_credentials(tmp_path / 'session.json')
    token = json.loads((tmp_path / 'session.json').read_text())['token']
    app = create_app(tmp_path, session, poll_seconds=.01)
    with TestClient(app, base_url=session.origin) as client:
        assert client.get('/api/v1/runs/run/artifacts/unknown').status_code == 401
        client.post('/api/v1/session', json={'token': token})
        records = client.get('/api/v1/runs/run/artifacts').json()['artifacts']
        image_id, record = next((key, item) for key, item in records.items() if item['modality'] == 'image')
        assert record['width'] == record['height'] == 1 and 'path' not in record
        route = '/api/v1/runs/run/artifacts/' + image_id
        response = client.get(route)
        assert response.content == png and response.headers['content-type'] == 'image/png'
        assert response.headers['content-disposition'].startswith('inline')
        assert response.headers['x-content-type-options'] == 'nosniff'
        assert "default-src 'self'" in response.headers['content-security-policy']
        Path(published['image_grid']['path']).write_bytes(b'changed')
        assert client.get(route).status_code == 400
        # Even an indexed and correctly hashed HTML payload cannot claim image/png.
        bad = b'<svg onload="alert(1)"></svg>'
        (tmp_path / 'bad.png').write_bytes(bad)
        atomic_json(tmp_path / 'artifacts/index.json', {'schema_version': 1, 'artifacts': {
            'bad-png': dict(path='bad.png', bytes=len(bad), sha256=hashlib.sha256(bad).hexdigest(),
                            modality='image', media_type='image/png', width=1, height=1)}})
        assert client.get('/api/v1/runs/run/artifacts/bad-png').status_code == 400


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
                                '--port', '0', '--auth', 'token', '--session-file', str(credential)],
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
        session = LocalSession(8123, auth="token")
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
    session = LocalSession(8123, auth="token")
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


@pytest.mark.parametrize('auth', ['none', 'token'])
def test_remote_binding_auth_mode_and_same_origin(tmp_path, auth):
    fixture_run(tmp_path)
    session = LocalSession(8123, host='0.0.0.0', auth=auth)
    with TestClient(create_app(tmp_path, session), base_url='http://training.example:8123') as client:
        response = client.get('/api/v1/capabilities')
        if auth == 'token':
            assert response.status_code == 401
            assert client.post('/api/v1/session', json={'token': session._token},
                               headers={'origin': 'http://training.example:8123'}).status_code == 200
        assert client.get('/api/v1/capabilities').json()['auth_mode'] == auth
        assert client.get('/api/v1/runs/run').status_code == 200
        assert client.get('/api/v1/capabilities', headers={'origin': 'http://other.example:8123'}).status_code == 403
        schema = client.get('/api/v1/openapi.json').json()
        if auth == 'none':
            assert not any(operation.get('security') for path in schema['paths'].values() for operation in path.values())


def test_console_control_is_no_longer_served(tmp_path):
    """The CLI cadence is a flag on the run, not something the viewer changes."""
    fixture_run(tmp_path)
    session = LocalSession(8123, auth="token")
    session.write_credentials(tmp_path / 'session.json')
    token = json.loads((tmp_path / 'session.json').read_text())['token']
    with TestClient(create_app(tmp_path, session), base_url=session.origin) as client:
        route = '/api/v1/runs/run/console'
        assert client.get(route).status_code == 401
        client.post('/api/v1/session', json={'token': token})
        assert client.get(route).status_code == 404
        assert client.put(route, json={'progress_every': 7}).status_code == 404
        assert not (tmp_path / 'console.json').exists()
        assert client.get('/api/v1/capabilities').json()['controls'] == []
        assert not [name for name in client.get('/api/v1/openapi.json').json()['paths'] if 'console' in name]


@pytest.mark.parametrize('status,next_step', [('running', 30000), ('disabled', None)])
def test_snapshot_evaluation_schedule_is_public_and_documented(tmp_path, status, next_step):
    manifest = fixture_run(tmp_path)
    manifest['evaluation_schedule'] = {'fid': {
        'status': status, 'source_step': 10000, 'next_step': next_step,
        'evaluation_id': 'a' * 32, 'skipped_busy': 1,
        'last_skipped_step': 20000, 'reason': 'worker_busy',
    }}
    atomic_json(tmp_path / 'manifest.json', manifest)
    session = LocalSession(8123, auth='none')
    app = create_app(tmp_path, session, poll_seconds=.01)
    with TestClient(app, base_url=session.origin) as client:
        response = client.get('/api/v1/runs/run')
        assert response.status_code == 200
        assert response.json()['evaluation_schedule'] == manifest['evaluation_schedule']
        schema = client.get('/api/v1/openapi.json').json()['components']['schemas']
        schedule = schema['Run']['properties']['evaluation_schedule']['additionalProperties']['properties']
        assert {'skipped', 'cancelled', 'disabled'} <= set(schedule['status']['enum'])
        assert 'cancelled' in schema['Event']['properties']['status']['enum']
        assert {'type': 'null'} in schedule['next_step']['oneOf']
        assert {'source_step', 'next_step', 'skipped_busy', 'last_skipped_step'} <= schedule.keys()


def test_artifact_records_carry_stable_sample_names_within_the_bounded_index(tmp_path):
    """Named samples group the shelf; digest artifact IDs stay unchanged."""
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    fixture_run(tmp_path, 1)
    png = encode_png(bytes([255, 0, 0]), 1, 1, 3, {'step': 1})
    real = encode_png(bytes([0, 0, 255]), 1, 1, 3, {'step': 1, 'name': 'x'})
    def publish(sequence, step):
        identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32,
                    'sample_sequence': sequence, 'name': 'g'}
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'name': 'g',
                   'step': step, 'count': 1, 'shape': [1, 3, 1, 1],
                   'samples': [[[[1]], [[-1]], [[-1]]]],
                   'image_grid': {'width': 1, 'height': 1, 'channels': 3, 'name': 'g',
                                  'png_base64': base64.b64encode(png).decode('ascii')},
                   'real_image_grid': {'width': 1, 'height': 1, 'channels': 3, 'name': 'x',
                                       'png_base64': base64.b64encode(real).decode('ascii')}}
        return publish_preview_payload(tmp_path, payload, identity, step, keep=4)
    for sequence, step in ((1, 1), (2, 2)):
        publish(sequence, step)
    unnamed = json.dumps({'shape': [1], 'samples': [1.]}).encode()
    (tmp_path / 'other.json').write_bytes(unnamed)
    atomic_json(tmp_path / 'artifacts/index.json', {'schema_version': 1, 'artifacts': {
        'diagnostic': dict(path='other.json', bytes=len(unnamed),
                           sha256=hashlib.sha256(unnamed).hexdigest(), role='diagnostic',
                           modality='tensor', media_type='application/json', shape=[1])}})
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            records = service.artifact_index()['artifacts']
            assert len(records) == 7
            previews = {key: value for key, value in records.items() if key.startswith('preview-')}
            assert len(previews) == 6 and all(len(key) <= 64 for key in previews)
            names = sorted((value['name'], value['modality']) for value in previews.values())
            assert names == [('g', 'image'), ('g', 'image'), ('g', 'tensor'),
                             ('g', 'tensor'), ('x', 'image'), ('x', 'image')]
            for key, value in previews.items():
                # Names are additive: provenance, role and digest keys are unchanged.
                assert value['provenance']['name'] == value['name'] and value['role'] == 'sample'
                assert value['provenance']['step'] in (1, 2)
                assert key.endswith('-real-grid') == (value['name'] == 'x')
            # An explicit index without a name groups under its own artifact ID.
            assert records['diagnostic']['name'] == 'diagnostic'
            # Retention keeps the whole run, so a long history must load: the
            # slider reaches back to the first sample, not the last hundred.
            from hypergan.web_service import MAX_PREVIEWS
            assert MAX_PREVIEWS >= 4096
            indexed = json.loads((tmp_path / 'previews/index.json').read_text())
            template = indexed['previews'][0]
            def resized(count):
                return dict(indexed, previews=[
                    dict(template, identity=dict(template['identity'], sample_sequence=n))
                    for n in range(count)])
            atomic_json(tmp_path / 'previews/index.json', resized(MAX_PREVIEWS))
            await service.refresh_artifacts()
            served = service.artifact_index()['artifacts']
            assert sum(key.startswith('preview-') for key in served) == 3 * MAX_PREVIEWS
            # The served index stays explicitly bounded above that.
            atomic_json(tmp_path / 'previews/index.json', resized(MAX_PREVIEWS + 1))
            with pytest.raises(ValueError, match='Invalid bounded preview index'):
                await service.refresh_artifacts()
        finally:
            await service.close()
    asyncio.run(scenario())


def test_png_only_colorization_artifacts_are_served_as_named_images(tmp_path):
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    fixture_run(tmp_path, 1)
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    payload = dict(schema_version=1, kind='ema-preview', identity=identity, name='g',
                   step=1, count=8, shape=[8, 3, 256, 256], samples=None, representation='png')
    for field, name, channels in [('image_grid', 'g', 3), ('real_image_grid', 'x', 3),
                                 ('input_image_grid_0', 'gray', 1)]:
        png = encode_png(bytes([128]) * 768 * 768 * channels, 768, 768, channels)
        payload[field] = dict(width=768, height=768, channels=channels, name=name,
                              png_base64=base64.b64encode(png).decode('ascii'))
    publish_preview_payload(tmp_path, payload, identity, 1)
    session = LocalSession(8123, auth='none')
    with TestClient(create_app(tmp_path, session, poll_seconds=.01), base_url=session.origin) as client:
        records = client.get('/api/v1/runs/run/artifacts').json()['artifacts']
        assert sorted((record['name'], record['modality']) for record in records.values()) == [
            ('g', 'image'), ('gray', 'image'), ('x', 'image')]
        for key, record in records.items():
            response = client.get('/api/v1/runs/run/artifacts/' + key)
            assert response.status_code == 200 and response.headers['content-type'] == 'image/png'
            assert len(response.content) == record['bytes']


def test_default_retention_publishes_a_whole_run_history_to_the_viewer(tmp_path):
    """Sixty real publications with the default keep stay listed; none are pruned."""
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import DEFAULT_KEEP, publish_preview_payload
    fixture_run(tmp_path, 1)
    png = encode_png(bytes([255, 0, 0]), 1, 1, 3, {'step': 1})
    real = encode_png(bytes([0, 0, 255]), 1, 1, 3, {'step': 1, 'name': 'x'})
    for sequence in range(1, 61):
        step = sequence * 500
        identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32,
                    'sample_sequence': sequence, 'name': 'g'}
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'name': 'g',
                   'step': step, 'count': 1, 'shape': [1, 3, 1, 1],
                   'samples': [[[[1]], [[-1]], [[-1]]]],
                   'image_grid': {'width': 1, 'height': 1, 'channels': 3, 'name': 'g',
                                  'png_base64': base64.b64encode(png).decode('ascii')},
                   'real_image_grid': {'width': 1, 'height': 1, 'channels': 3, 'name': 'x',
                                       'png_base64': base64.b64encode(real).decode('ascii')}}
        # No `keep`: exactly what a default `hypergan train` publishes.
        publish_preview_payload(tmp_path, payload, identity, step)
    index = json.loads((tmp_path / 'previews/index.json').read_text())
    # Sixty publications sit inside the default bound, so nothing is thinned yet.
    assert index['keep'] == DEFAULT_KEEP == 128 and index['retention'] == 'thinned'
    assert len(index['previews']) == 60
    assert (index['previews'][0]['step'], index['previews'][-1]['step']) == (500, 30000)
    generations = [entry for entry in (tmp_path / 'previews').iterdir() if entry.is_dir()]
    assert len(generations) == 60, 'the oldest generations must survive the newest publication'
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            records = service.artifact_index()['artifacts']
            previews = {key: value for key, value in records.items() if key.startswith('preview-')}
            # One tensor and two image grids for each of the sixty publications.
            assert len(previews) == 180
            steps = sorted({value['provenance']['step'] for value in previews.values()})
            assert steps == [n * 500 for n in range(1, 61)]
            generated = [value for value in previews.values()
                         if value['modality'] == 'image' and value['name'] == 'g']
            assert len(generated) == 60
        finally:
            await service.close()
    asyncio.run(scenario())


def test_real_tuned_run_establishes_viewer_lineage_before_tuning(tmp_path):
    from hypergan.config import write_default
    from hypergan.run_controller import CompletedUpdate, ExecutionInfo, run_train

    class TuningExecution:
        def __init__(self, config):
            self.step = 0

        def environment(self):
            return {'runtime': {}, 'source': {}}

        def start(self):
            return ExecutionInfo(0, {}, [], {})

        def tune(self, run_dir, on_event=None):
            on_event({'candidate': 1, 'total_candidates': 1})
            return {'outcome': 'kept_baseline', 'selected_candidate': 'baseline'}

        def checkpoint(self, run_dir, metadata):
            target = run_dir / f'checkpoint-{self.step}'
            target.mkdir()
            atomic_json(target / 'manifest.json', dict(metadata, step=self.step))
            return target

        def update(self):
            self.step += 1
            return CompletedUpdate(self.step, {'g_loss': .5})

        inference_available = False

        def observe(self, callback, event):
            callback(event)

        def shutdown(self):
            pass

    root = tmp_path / 'run'
    result = run_train(write_default(tmp_path / 'project', device='cpu'), root,
                       steps=1, tune=True, execution_factory=TuningExecution)
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'start' and events[0]['sequence'] == 1
    assert any(row['event'] == 'tuning' for row in events[1:])
    assert sum(row['event'] == 'start' for row in events) == 1
    with Projector(root) as projector:
        projector.project(limit=10000)

    async def scenario():
        service = await ObservationService(root, poll_seconds=.01).start()
        try:
            bootstrap = await get_bootstrap(service)
            assert bootstrap['lineage'] == [{'attempt_id': result['attempt_id'], 'through_step': None}]
            assert service.public_manifest()['initialization_tuning']['status'] == 'complete'
        finally:
            await service.close()
    asyncio.run(scenario())

import asyncio
import hashlib
import json
from pathlib import Path
import socket
import time

import httpx
import pytest
from starlette.testclient import TestClient

from hypergan.event_views import MapSpec, Projector
from hypergan.config import config_values, fingerprint, resolve_config
from hypergan.metrics import digest
from hypergan.metrics_reducer import Reducer
from hypergan.model_description import MODEL_FILE
from hypergan.run_state import atomic_json
from hypergan.web_service import ObservationService, Subscriber
from hypergan.web_server import RUN_PREFIXES, RUN_ROUTES, ServiceRegistry, create_app
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
            eval_stream = service.streams['evaluation:' + evaluation_id]
            await asyncio.wait_for(eval_stream.task, 3)
            assert eval_stream.caught_up and eval_stream.task.done()
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


def test_discovery_overflow_keeps_metadata_live(tmp_path, monkeypatch):
    import hypergan.web_service as ws
    assert ws.MAX_STREAMS == 4096
    fixture_run(tmp_path, 1)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            directory = tmp_path / 'metrics/evaluations'
            directory.mkdir(parents=True)
            for index in range(100):
                (directory / f'step-{index * 500}-random_spread').mkdir()
            await service.discover()
            assert service.discovery_error is None
            monkeypatch.setattr(ws, 'MAX_STREAMS', 64)
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
def test_remote_binding_auth_mode_and_same_origin(tmp_path, auth, monkeypatch):
    fixture_run(tmp_path)
    monkeypatch.setenv('HYPERGAN_VIEWER_HOSTS', 'training.example')
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
        assert client.get('/api/v1/capabilities').json()['controls'] == ['checkpoint', 'stop', 'sample']
        assert not [name for name in client.get('/api/v1/openapi.json').json()['paths'] if 'console' in name]


def test_authenticated_control_endpoints_checkpoint_sample_stop_and_request_receipts(tmp_path):
    manifest = fixture_run(tmp_path)
    manifest['status'] = 'running'
    manifest['hostname'] = 'remote-gpu-worker-01.internal'
    atomic_json(tmp_path / 'manifest.json', manifest)
    session = LocalSession(8123, auth='token')
    session.write_credentials(tmp_path / 'session.json')
    token = json.loads((tmp_path / 'session.json').read_text())['token']

    with TestClient(create_app(tmp_path, session, poll_seconds=.01), base_url=session.origin) as client:
        # Both prefixes require the credential for reads and for every control.
        for prefix in ('/api/v1/runs/run', '/api/runs/run'):
            assert client.get(prefix).status_code == 401
            for action in ('checkpoint', 'sample', 'stop'):
                assert client.post(f'{prefix}/{action}').status_code == 401

        assert client.post('/api/v1/session', json={'token': token}).status_code == 200

        # Remote running fixture is recognized as active and exposes hostname + remote in manifest
        pub = client.get('/api/v1/runs/run').json()
        assert pub['status'] == 'running'
        assert pub['remote'] is True
        assert pub['hostname'] == 'remote-gpu-worker-01.internal'

        # Idempotent checkpoint request + receipt lookup
        ckpt1 = client.post('/api/v1/runs/run/checkpoint', json={'request_id': 'ckpt-req-1'}).json()
        ckpt2 = client.post('/api/runs/run/checkpoint', json={'request_id': 'ckpt-req-1'}).json()
        assert ckpt1['ok'] is True and ckpt1['request_id'] == 'ckpt-req-1'
        assert ckpt1['receipt'] == ckpt2['receipt']
        assert client.get('/api/v1/runs/run/requests/ckpt-req-1').json() == ckpt1['receipt']

        # Idempotent sample request + receipt lookup
        samp1 = client.post('/api/v1/runs/run/sample', json={'request_id': 'sample-req-1'}).json()
        samp2 = client.post('/api/runs/run/sample', json={'request_id': 'sample-req-1'}).json()
        assert samp1['ok'] is True and samp1['request_id'] == 'sample-req-1'
        assert samp1['receipt']['request']['action'] == 'sample'
        assert samp1['receipt'] == samp2['receipt']
        assert client.get('/api/runs/run/requests/sample-req-1').json() == samp1['receipt']

        # Idempotent stop request + receipt lookup on remote cluster run (no local PID required)
        stop1 = client.post('/api/v1/runs/run/stop', json={'request_id': 'stop-req-1'}).json()
        stop2 = client.post('/api/runs/run/stop', json={'request_id': 'stop-req-1'}).json()
        assert stop1['ok'] is True and stop1['request_id'] == 'stop-req-1'
        assert stop1['remote'] is True and stop1['signaled'] is False
        assert stop1['receipt']['request']['action'] == 'stop'
        assert stop1['receipt'] == stop2['receipt']
        assert client.get('/api/v1/runs/run/requests/stop-req-1').json() == stop1['receipt']
        assert client.get('/api/v1/runs/run/requests/nonexistent').status_code == 404


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
            # An invalid optional index is reported, not fatal: the run stays
            # served with what was last indexed.
            await service.refresh_artifacts()
            assert 'Invalid bounded preview index' in service.artifact_error
            assert 'Invalid bounded preview index' in service.artifact_index()['error']
            assert sum(key.startswith('preview-') for key in service.artifact_index()['artifacts']) == 3 * MAX_PREVIEWS
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


# GET /api/v1/runs/{run_id}/model: authenticated, run-checked, bounded and redacted.
def configured_run(root):
    manifest = fixture_run(root)
    config = resolve_config({})
    manifest.update(config=config_values(config), config_sha256=fingerprint(config),
                    source={'particlegan_distribution_version': '0.8.0'}, runtime={'hndl': '0.6.0'})
    manifest['config']['data']['args']['root'] = '/home/someone/private-data'
    atomic_json(root / 'manifest.json', manifest)
    return manifest


def authenticated(client, session, root):
    session.write_credentials(root / 'session.json')
    token = json.loads((root / 'session.json').read_text())['token']
    assert client.post('/api/v1/session', json={'token': token}).status_code == 200


def test_model_route_serves_the_recorded_configuration(tmp_path):
    manifest = configured_run(tmp_path)
    session = LocalSession(8123, auth='token')
    with TestClient(create_app(tmp_path, session), base_url=session.origin) as client:
        assert client.get('/api/v1/runs/run/model').status_code == 401
        authenticated(client, session, tmp_path)
        assert client.get('/api/v1/runs/foreign/model').status_code == 404
        response = client.get('/api/v1/runs/run/model')
        assert response.status_code == 200
        body = response.json()
        assert body['run_id'] == 'run' and body['config_sha256'] == manifest['config_sha256']
        assert body['formulation']['family'] == 'particlegan' and body['formulation']['particlegan'] == '0.8.0'
        assert [n['name'] for n in body['networks']] == ['generator', 'discriminator']
        assert body['networks'][0]['source']['text'].strip()
        assert body['networks'][0]['graph']['status'] == 'unavailable'
        assert body['losses']['totals']['generator'] == 'loss/g_total'
        # The fixture catalog publishes only loss/g_total; the others say so.
        adversarial = body['losses']['generator'][0]
        assert adversarial['metric'] == 'loss/g_adversarial' and 'catalog' in adversarial['metric_note']
        assert 'someone' not in response.text and body['data']['args']['root'] == '…/private-data'
        assert client.get('/api/v1/runs/run/model', headers={'origin': 'http://evil.example'}).status_code == 403


def test_model_route_merges_recorded_network_detail(tmp_path):
    manifest = configured_run(tmp_path)
    node = {'id': 'n0', 'op': 'linear', 'category': 'core', 'out': {'out': ['B', 2]}, 'params': 10}
    atomic_json(tmp_path / MODEL_FILE, {'schema_version': 1, 'config_sha256': manifest['config_sha256'],
        'origin': 'recorded', 'components': {'generator': {'status': 'built', 'parameters': {'total': 10},
            'subgraphs': [{'module_path': 'network', 'node_count': 1, 'nodes': [node]}]}}})
    session = LocalSession(8123, auth='none')
    with TestClient(create_app(tmp_path, session), base_url=session.origin) as client:
        body = client.get('/api/v1/runs/run/model').json()
        generator, discriminator = body['networks']
        assert generator['graph']['status'] == 'built' and generator['graph']['subgraphs'][0]['nodes'] == [node]
        assert discriminator['graph']['status'] == 'unavailable'
        # A file for another configuration is never attached.
        atomic_json(tmp_path / MODEL_FILE, {'schema_version': 1, 'config_sha256': 'f' * 64, 'components': {}})
        body = client.get('/api/v1/runs/run/model').json()
        assert 'different configuration' in body['networks'][0]['graph']['reason']
        (tmp_path / MODEL_FILE).write_text('{not json')
        body = client.get('/api/v1/runs/run/model').json()
        assert 'unreadable' in body['networks'][0]['graph']['reason']


def test_model_summary_cache_follows_the_attempt(tmp_path):
    manifest = configured_run(tmp_path)
    session = LocalSession(8123, auth='none')
    with TestClient(create_app(tmp_path, session, poll_seconds=.01), base_url=session.origin) as client:
        first = client.get('/api/v1/runs/run/model').json()
        assert first['warnings'] == []
        # A new attempt with the same configuration, catalog and model.json. It is
        # written to disk because the watcher replaces the in-memory manifest on every poll.
        for _ in range(20):
            try:
                atomic_json(tmp_path / 'manifest.json', {**manifest, 'attempt_index': 2, 'warnings': ['new attempt warning'],
                                                         'source': {**manifest['source'], 'hypergan_commit': 'abcdef1234567'}})
                break
            except PermissionError:
                time.sleep(.05)
        deadline = time.monotonic() + 10
        body = client.get('/api/v1/runs/run/model').json()
        while body['run']['attempt_index'] != 2 and time.monotonic() < deadline:
            time.sleep(.01)
            body = client.get('/api/v1/runs/run/model').json()
        assert body['warnings'] == ['new attempt warning'] and body['run']['attempt_index'] == 2
        assert body['provenance']['hypergan_commit'] == 'abcdef1234567'


def test_model_route_without_configuration_is_not_found(tmp_path):
    fixture_run(tmp_path)
    session = LocalSession(8123, auth='none')
    with TestClient(create_app(tmp_path, session), base_url=session.origin) as client:
        response = client.get('/api/v1/runs/run/model')
        assert response.status_code == 404 and 'configuration' in response.json()['error']


@pytest.mark.parametrize('auth', ['token', 'none'])
def test_model_route_is_in_the_openapi_contract(tmp_path, auth):
    fixture_run(tmp_path)
    session = LocalSession(8123, auth=auth)
    with TestClient(create_app(tmp_path, session), base_url=session.origin) as client:
        if auth == 'token':
            authenticated(client, session, tmp_path)
        schema = client.get('/api/v1/openapi.json').json()
    operation = schema['paths']['/api/v1/runs/{run_id}/model']['get']
    assert operation['responses']['200']['content']['application/json']['schema'] == {'$ref': '#/components/schemas/Model'}
    assert {'Model', 'Network', 'NetworkGraph', 'LossTerm'} <= set(schema['components']['schemas'])
    assert bool(operation.get('security')) == (auth == 'token')

def test_foldlite_history_reopens_from_persisted_cursor_and_regenerates(tmp_path, monkeypatch):
    """Reopening reduces only a new suffix; deleting derived state is equivalent."""
    import shutil
    fixture_run(tmp_path, 4)

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            first = await get_bootstrap(service)
            query = next(iter(service.jobs.values())).query
        finally:
            await service.close()
        cache = service.view_cache_dir(MapSpec().revision)
        assert list(cache.glob('*.fold'))
        assert not (tmp_path / 'views' / MapSpec().revision / 'foldlite').exists()
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await caught_up(service)
            page = service.page
            cursors = []
            def tracked(stream, cursor, *, limit=256):
                if stream.startswith('projection:'):
                    assert cursor is not None, 'Cached history rescanned its source'
                    cursors.append(cursor)
                return page(stream, cursor, limit=limit)
            monkeypatch.setattr(service, 'page', tracked)
            reopened = await service._history(query, first['cursor'], first['projection_sequence'], first['lineage'])
            assert reopened == first and cursors
            append_event(tmp_path, 6, 5)
            with Projector(tmp_path) as projector:
                projector.project()
            for _ in range(300):
                stream = service.streams['projection:' + MapSpec().revision]
                if stream.sequence == 6 and stream.caught_up:
                    break
                await asyncio.sleep(.01)
            else:
                raise AssertionError('Projection suffix did not become visible')
            suffix = await service._history(query, stream.cursor, stream.sequence, first['lineage'])
            assert suffix['projection_sequence'] == 6
            assert sum(Reducer().finalize(group['state'])['count'] for group in suffix['groups']) == 5
        finally:
            await service.close()
        shutil.rmtree(cache)
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            regenerated = await get_bootstrap(service)
            assert regenerated == suffix
        finally:
            await service.close()
    asyncio.run(scenario())


def test_foldlite_cached_history_revalidates_source_boundary(tmp_path):
    fixture_run(tmp_path, 3)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            initial = await get_bootstrap(service)
            query = next(iter(service.jobs.values())).query
            path = tmp_path / 'views' / MapSpec().revision / 'contributions.jsonl'
            raw = path.read_bytes()
            path.write_bytes(raw[:-2] + b' \n')
            with pytest.raises(ValueError, match='boundary changed'):
                await service._history(query, initial['cursor'], initial['projection_sequence'], initial['lineage'])
        finally:
            await service.close()
    asyncio.run(scenario())


def test_failed_bootstrap_job_can_retry_after_its_delivery_lease(tmp_path):
    fixture_run(tmp_path, 3)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await get_bootstrap(service)
            failed = next(iter(service.jobs.values()))
            failed.result, failed.error = None, 'temporary historical read failure'
            failed.delivered, failed.completed = True, time.monotonic() - 6
            retried = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2)
            assert retried is not failed
            await retried.task
            assert retried.error is None and retried.result is not None
        finally:
            await service.close()
    asyncio.run(scenario())


def test_corrupt_foldlite_cache_reports_job_error_and_allows_regeneration(tmp_path):
    fixture_run(tmp_path, 3)
    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            expected = await get_bootstrap(service)
            cache = next(service.view_cache_dir(MapSpec().revision).glob('*.fold'))
            service.jobs.clear()
            cache.write_bytes(b'corrupt derived cache')
            failed = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2)
            await failed.task
            assert failed.result is None
            assert 'derived view cache' in failed.error and 'remove' in failed.error and '.fold' in failed.error
            # The completed job has closed its cache; no historical reader or
            # writer is active. Its error lease can expire and regeneration retry.
            cache.unlink()
            failed.delivered, failed.completed = True, time.monotonic() - 6
            retried = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=2)
            await retried.task
            assert retried is not failed and retried.error is None
            assert retried.result == expected
        finally:
            await service.close()
    asyncio.run(scenario())


def test_stream_disconnect_during_replay_burst_and_sigint_with_open_streams(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    runs_root = tmp_path / 'runs'
    runs_root.mkdir()
    fixture_run(runs_root / 'run', 300)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
    process = subprocess.Popen(
        [sys.executable, '-m', 'hypergan.cli', 'view', str(runs_root),
         '--host', '127.0.0.1', '--port', '0', '--auth', 'none', '--no-browser'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    sockets = []
    try:
        line = process.stdout.readline()
        if not line:
            raise AssertionError(process.communicate())
        record = json.loads(line)
        credential = Path(record['session_file'])
        origin = record['origin']
        host, port_str = origin.split('://')[1].split(':')
        port = int(port_str)

        proj_stream = None
        for _ in range(500):
            try:
                views = httpx.get(origin + '/api/v1/runs/run/views', timeout=2).json()
                proj_stream = next(s for s in views['streams'] if s['stream_id'].startswith('projection:'))
                if proj_stream['caught_up'] and proj_stream['cursor']:
                    break
            except (httpx.ConnectError, StopIteration):
                pass
            time.sleep(.01)
        assert proj_stream is not None and proj_stream['caught_up']
        stream_id = proj_stream['stream_id']
        first_page = httpx.get(
            origin + '/api/v1/runs/run/events',
            params={'stream_id': stream_id, 'limit': 1},
            timeout=2,
        ).json()
        first_cursor = first_page['cursor']

        # Abruptly close several SSE streams mid-replay of ~299 frames.
        for _ in range(6):
            sock = socket.create_connection((host, port), timeout=5)
            req = (
                f'GET /api/v1/runs/run/stream?stream_id={stream_id}&cursor={first_cursor} HTTP/1.1\r\n'
                f'Host: {host}:{port}\r\n\r\n'
            )
            sock.sendall(req.encode('ascii'))
            buf = b''
            while b'event: frame' not in buf:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                buf += chunk
            sock.close()

        # Keep both root discovery and run SSE streams open when SIGINT arrives.
        for path in ('/api/v1/stream', f'/api/v1/runs/run/stream?stream_id={stream_id}'):
            sock = socket.create_connection((host, port), timeout=5)
            sockets.append(sock)
            sock.sendall(f'GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n'.encode('ascii'))
            buf = b''
            while b'event: ready' not in buf:
                chunk = sock.recv(1024)
                assert chunk
                buf += chunk

        started = time.monotonic()
        process.send_signal(signal.SIGINT if os.name != 'nt' else signal.SIGTERM)
        out, err = process.communicate(timeout=5)
        elapsed = time.monotonic() - started
        assert 'socket.send() raised exception.' not in err
        assert 'Exception in ASGI application' not in err
        assert elapsed < 3.0
        if os.name != 'nt':
            assert process.returncode == 128 + signal.SIGINT
            assert not credential.exists()
    finally:
        for sock in sockets:
            sock.close()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


# Access control and service lifecycle (2026-10-25 review, slice 4).

def _signed_out(tmp_path, port=8123, host='127.0.0.1'):
    session = LocalSession(port, host=host, auth='token')
    return session, create_app(tmp_path, session, poll_seconds=.01)


def _concrete(path):
    return (path.replace('{run_id}', 'run').replace('{request_id}', 'r1')
            .replace('{map_revision}', MapSpec().revision).replace('{artifact_id}', 'a1'))


def test_token_mode_requires_the_credential_for_every_api_read_and_stream(tmp_path):
    """Both run prefixes come from one table and the auth layer reads that table."""
    fixture_run(tmp_path)
    session, app = _signed_out(tmp_path)
    with TestClient(app, base_url=session.origin) as client:
        checked = 0
        for prefix in RUN_PREFIXES:
            assert client.get(prefix).status_code == 401
            for suffix, _, methods in RUN_ROUTES:
                method = (methods or ['GET'])[0]
                response = client.request(method, _concrete(prefix + '/{run_id}' + suffix))
                assert response.status_code == 401, (method, prefix + suffix)
                checked += 1
        assert checked == 2 * len(RUN_ROUTES)
        for path in ('/api/v1/stream', '/api/v1/capabilities', '/api/v1/openapi.json', '/api/unknown'):
            assert client.get(path).status_code == 401
        # The shell stays public so the browser can show its sign-in form.
        assert client.get('/assets/app.js').status_code == 200
        assert client.get('/reducers/reducer.json').status_code == 200
        assert client.get('/runs/run').status_code == 200
        assert client.get('/runs/no-such-run').status_code == 200  # existence is not disclosed
        client.headers['Authorization'] = 'Bearer ' + session._token
        assert client.get('/api/runs/run').status_code == 200
        assert client.get('/runs/no-such-run').status_code == 404


def test_token_dashboard_html_carries_no_run_data_until_signed_in(tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()
    fixture_run(root / 'secret-run')
    session, app = _signed_out(root)
    with TestClient(app, base_url=session.origin) as client:
        page = client.get('/').text
        assert 'id="login"' in page and 'secret-run' not in page and str(root) not in page
        client.headers['Authorization'] = 'Bearer ' + session._token
        page = client.get('/').text
        assert 'secret-run' in page and str(root) in page


def test_wildcard_bind_rejects_rebound_dns_names_but_accepts_ip_literals_and_own_names(tmp_path):
    """DNS rebinding: an attacker's page reaches the viewer under its own hostname."""
    fixture_run(tmp_path)
    session = LocalSession(8123, host='0.0.0.0', auth='none')
    app = create_app(tmp_path, session, poll_seconds=.01)
    with TestClient(app, base_url=session.origin) as client:
        evil = {'host': 'evil.example:8123', 'origin': 'http://evil.example:8123'}
        assert client.get('/api/runs', headers=evil).status_code == 403
        assert client.post('/api/runs/run/stop', headers=evil).status_code == 403
        for host in ('192.168.1.20:8123', '[fe80::1]:8123', 'localhost:8123', '127.0.0.1:8123',
                     socket.gethostname().lower() + ':8123'):
            assert client.get('/api/runs', headers={'host': host}).status_code == 200, host
        assert client.get('/api/runs', headers={'host': '192.168.1.20:9999'}).status_code == 403


@pytest.mark.parametrize('host', ['localhost:8123', '127.0.0.1:8123', '[::1]:8123'])
def test_loopback_bind_accepts_every_loopback_spelling(tmp_path, host):
    """The printed http://localhost:PORT link works for a 127.0.0.1 bind."""
    fixture_run(tmp_path)
    session = LocalSession(8123, host='127.0.0.1')
    with TestClient(create_app(tmp_path, session), base_url='http://' + host) as client:
        assert client.get('/api/v1/capabilities').status_code == 200
        assert client.get('/api/v1/capabilities', headers={'host': '192.168.1.20:8123'}).status_code == 403


def test_explicit_allowed_hosts_extend_the_accepted_names(monkeypatch):
    monkeypatch.setenv('HYPERGAN_VIEWER_HOSTS', 'Box.Tailnet.TS.net, other.lan')
    session = LocalSession(8123, host='0.0.0.0')
    assert session.permits_request('box.tailnet.ts.net:8123')
    assert session.permits_request('other.lan:8123', 'http://other.lan:8123')
    assert not session.permits_request('evil.example:8123')


def test_single_run_viewer_ignores_the_global_registry(tmp_path):
    """Whether `/` is the run page must not depend on unrelated registered runs."""
    from hypergan import registry
    elsewhere = tmp_path / 'elsewhere'
    fixture_run(elsewhere)
    registry.register_run(elsewhere, status='stopped')
    run = tmp_path / 'one'
    fixture_run(run)
    session = LocalSession(8123)
    with TestClient(create_app(run, session), base_url=session.origin) as client:
        assert '<title>HyperGAN Runs' not in client.get('/').text
    assert ServiceRegistry(run)._iter_registered_run_dirs() == []
    # Even a registered run nested inside the served run is not listed: the
    # root filter alone would admit it, so this is the single-run rule.
    nested = run / 'nested'
    fixture_run(nested)
    registry.register_run(nested, status='stopped')
    assert ServiceRegistry(run)._iter_registered_run_dirs() == []


def test_moved_run_with_absolute_preview_index_still_opens(tmp_path):
    from hypergan.previews import publish_preview_payload
    original = tmp_path / 'workspace' / 'runs' / 'x'
    original.parent.mkdir(parents=True)
    fixture_run(original, 1)
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'attempt_index': 1, 'sample_sequence': 1}
    payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'step': 1,
               'count': 1, 'shape': [1, 2], 'samples': [[.2, .3]]}
    publish_preview_payload(original, payload, identity, 1)
    # Producers now index relative paths; older runs hold absolute ones, which
    # name the old location once the run moves.
    index_file = original / 'previews' / 'index.json'
    index = json.loads(index_file.read_text())
    for record in index['previews']:
        assert not Path(record['path']).is_absolute()
        record['path'] = str(original / record['path'])
    atomic_json(index_file, index)
    moved = tmp_path / 'home' / 'runs' / 'renamed'
    moved.parent.mkdir(parents=True)
    original.rename(moved)
    session = LocalSession(8123)
    with TestClient(create_app(moved, session, poll_seconds=.01), base_url=session.origin) as client:
        assert client.get('/api/v1/runs/run').status_code == 200
        assert client.get('/api/v1/runs/run/metrics/catalog').status_code == 200
        artifacts = client.get('/api/v1/runs/run/artifacts').json()
        assert 'error' not in artifacts
        (artifact_id, record), = artifacts['artifacts'].items()
        data = client.get(f'/api/v1/runs/run/artifacts/{artifact_id}')
        assert data.status_code == 200 and hashlib.sha256(data.content).hexdigest() == record['sha256']


def test_unusable_preview_index_is_reported_once_and_never_resets_the_page(tmp_path):
    fixture_run(tmp_path, 1)
    atomic_json(tmp_path / 'previews' / 'index.json', {
        'schema_version': 1, 'run_id': 'run', 'previews': [{
            'identity': {'name': 'g'}, 'step': 1, 'path': '/nowhere/else/preview.json',
            'representation': 'tensor', 'bytes': 1, 'sha256': '0' * 64, 'shape': [1]}]})

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        subscriber, _ = service.subscribe('*')
        try:
            assert service.run_id == 'run' and 'training' in service.streams
            # Only the record naming a missing file is unavailable, not the index.
            assert service.artifact_error is None
            (record,) = service.artifacts.values()
            assert record['status'] == 'unavailable' and 'not inside this run' in record['reason']
            await asyncio.sleep(.2)
            events = []
            while not subscriber.queue.empty():
                events.append((await subscriber.get()).split(b'\n', 1)[0])
            assert b'event: reset_required' not in events
        finally:
            service.subscribers.discard(subscriber)
            await service.close()
    asyncio.run(scenario())


def test_capabilities_during_run_start_does_not_block_stream_registration(tmp_path):
    root = tmp_path / 'pending'
    session = LocalSession(8123)
    app = create_app(root, session, poll_seconds=.5)
    with TestClient(app, base_url=session.origin) as client:
        service = app.state.observations
        assert service.run_id is None
        fixture_run(root, 2)
        # Inside the watcher's poll window, before it activated the run.
        assert client.get('/api/v1/capabilities').status_code == 200
        assert service.run_id is None
        for _ in range(200):
            if 'training' in service.streams:
                break
            client.get('/api/v1/capabilities')
            import time
            time.sleep(.02)
        assert 'training' in service.streams and service.run_id == 'run'


def test_retired_service_tells_its_subscribers_to_reconnect(tmp_path):
    fixture_run(tmp_path, 1)

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        subscriber, _ = service.subscribe('*')
        # Frames still queued for a slow page must not hide the reason: a closed
        # subscriber delivers only its first queued frame.
        subscriber.offer('frame', {'n': 1})
        subscriber.offer('frame', {'n': 2})
        service.retire('released')
        events = []
        while not subscriber.queue.empty():
            events.append(await subscriber.get())
        assert len(events) == 1, events
        assert events[0].startswith(b'event: reset_required') and b'released' in events[0]
        assert subscriber.closed
        await service.close()
    asyncio.run(scenario())


def test_lru_keeps_watched_services_and_retires_idle_ones(tmp_path):
    for name in ('a', 'b', 'c'):
        fixture_run(tmp_path / name, 1)

    async def scenario():
        registry = ServiceRegistry(tmp_path, poll_seconds=.01, max_services=1)
        watched = await registry.get('a')
        subscriber, _ = watched.subscribe('*')
        idle = await registry.get('b')
        assert set(registry.services) == {'a', 'b'}  # a is watched; b is the newest
        await registry.get('c')
        assert set(registry.services) == {'a', 'c'} and idle._closed and not watched._closed
        watched.subscribers.discard(subscriber)
        await registry.close_all()
    asyncio.run(scenario())


def test_rebuilt_projection_gets_a_new_tail_and_corrupt_one_stays_failed(tmp_path):
    fixture_run(tmp_path, 3)

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        stream_id = 'projection:' + MapSpec().revision
        try:
            await get_bootstrap(service)
            path = tmp_path / 'views' / MapSpec().revision / 'contributions.jsonl'
            with path.open('ab') as handle:
                handle.write(b'{"bad":true}\n')
            for _ in range(300):
                if service.streams[stream_id].error:
                    break
                await asyncio.sleep(.01)
            failed = service.streams[stream_id]
            assert failed.error
            await service.discover()
            assert service.streams[stream_id] is failed  # unchanged file: still failed
            subscriber, _ = service.subscribe('*')
            # Rebuilding the projection (a new file) gets a fresh tail and a reload notice.
            for name in ('contributions.jsonl', 'projection.json'):
                (path.parent / name).unlink()
            with Projector(tmp_path) as projector:
                projector.project(limit=10000)
            await service.discover()
            assert service.streams[stream_id] is not failed
            events = [await subscriber.get() for _ in range(subscriber.queue.qsize())]
            assert any(event.startswith(b'event: reset_required') for event in events)
            service.subscribers.discard(subscriber)
            assert (await get_bootstrap(service))['projection_sequence'] == 4
        finally:
            await service.close()
    asyncio.run(scenario())


def test_bootstrap_leaves_nothing_in_the_run_directory(tmp_path, private_viewer_cache):
    fixture_run(tmp_path, 3)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await get_bootstrap(service)
            await get_bootstrap(service, step_from=1)
            assert list(service.view_cache_dir(MapSpec().revision).glob('*.fold'))
        finally:
            await service.close()
    asyncio.run(scenario())
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before
    assert list(private_viewer_cache.rglob('*.fold'))


def test_view_caches_are_bounded(tmp_path, monkeypatch, private_viewer_cache):
    import hypergan.web_service as module
    monkeypatch.setattr(module, 'MAX_VIEW_CACHES', 2)
    monkeypatch.setattr(module, 'MAX_VIEW_CACHE_RUNS', 2)
    for name in ('a', 'b', 'c'):
        fixture_run(tmp_path / name, 4)
    records = private_viewer_cache / '.projection-failures'
    records.mkdir()

    async def scenario():
        for name in ('a', 'b', 'c'):
            service = await ObservationService(tmp_path / name, poll_seconds=.01).start()
            try:
                for step_from in range(4):
                    await get_bootstrap(service, step_from=step_from)
                assert len(list(service.view_cache_dir(MapSpec().revision).glob('*.fold'))) == 2
            finally:
                await service.close()
        # Only the two most recently used runs keep a cache; other viewer state stays.
        assert records.is_dir()
        assert len([path for path in private_viewer_cache.iterdir() if path != records]) == 2
        assert service.view_cache_dir(MapSpec().revision).is_dir()
    asyncio.run(scenario())


def test_stream_ends_cleanly_on_shutdown_through_the_asgi_contract(tmp_path):
    """No server internals: shutdown alone ends an idle SSE response and completes it."""
    fixture_run(tmp_path, 1)

    async def scenario():
        session = LocalSession(8123)
        app = create_app(tmp_path, session, poll_seconds=.01)
        await app.state.observations.start()
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                 'http_version': '1.1', 'method': 'GET', 'scheme': 'http',
                 'path': '/api/v1/stream', 'raw_path': b'/api/v1/stream', 'query_string': b'',
                 'headers': [(b'host', session.host.encode())],
                 'server': ('127.0.0.1', 8123), 'client': ('127.0.0.1', 9999), 'root_path': ''}
        never = asyncio.Event()
        sent = []

        async def receive():
            await never.wait()

        async def send(message):
            sent.append(message)
            if message['type'] == 'http.response.body' and message.get('more_body'):
                app.state.shutdown_event.set()
        try:
            await asyncio.wait_for(app(scope, receive, send), 5)
            assert sent[0]['status'] == 200
            assert sent[-1] == {'type': 'http.response.body', 'body': b'', 'more_body': False}
            assert not app.state.observations.subscribers
        finally:
            await app.state.observations.close()
    asyncio.run(scenario())


def test_inactive_run_refuses_new_requests_and_replays_only_acknowledged_ones(tmp_path):
    manifest = fixture_run(tmp_path, 1)
    manifest['status'] = 'stopped'
    atomic_json(tmp_path / 'manifest.json', manifest)
    session = LocalSession(8123)
    with TestClient(create_app(tmp_path, session), base_url=session.origin) as client:
        for action in ('checkpoint', 'sample', 'stop'):
            response = client.post(f'/api/v1/runs/run/{action}', json={'request_id': 'fresh-' + action})
            assert response.status_code == 400 and response.json()['action'] == action
        assert not (tmp_path / 'requests').exists() or not any((tmp_path / 'requests').rglob('*fresh*'))
        assert client.post('/api/v1/runs/run/stop', content=b'{"x":1}').status_code == 400
        # A request the run queued before it stopped keeps its receipt on retry,
        # but only for the action it was submitted as.
        from hypergan.run_requests import submit_checkpoint_request
        submit_checkpoint_request(tmp_path, run_id='run', attempt_id='a', request_id='queued-1')
        replay = client.post('/api/v1/runs/run/checkpoint', json={'request_id': 'queued-1'})
        assert replay.status_code == 200, replay.text
        assert replay.json()['request_id'] == 'queued-1' and replay.json()['status'] == 'pending'
        assert client.post('/api/v1/runs/run/sample', json={'request_id': 'queued-1'}).status_code == 400


def test_running_dashboard_is_reused_by_view_for_its_root(tmp_path, monkeypatch):
    import hypergan.web_autostart as startup
    from hypergan.web_launch import running_dashboard
    root = tmp_path / 'runs'
    root.mkdir()
    fixture_run(root / 'r1')
    monkeypatch.setattr(startup, '_registered_dashboard', lambda r: ({'host': '0.0.0.0'}, 'tok'))
    seen = []

    def probe(host, port, auth_token=None, **kwargs):
        seen.append(auth_token)
        return {'runs_root': str(root.resolve()), 'auth_mode': 'none'}
    monkeypatch.setattr(startup, 'probe_dashboard', probe)
    assert running_dashboard(root, 8765, 'none') == 'http://localhost:8765'
    assert running_dashboard(root / 'r1', None, 'none') == 'http://localhost:8765/runs/r1'
    assert seen == ['tok', 'tok']
    assert running_dashboard(tmp_path / 'other', 8765, 'none') is None
    with pytest.raises(OSError, match='--auth none'):
        running_dashboard(root, 8765, 'token')
    assert running_dashboard(root, 0, 'none') is None


def test_moved_run_marks_only_the_missing_preview_unavailable(tmp_path):
    from hypergan.previews import index_path, publish_preview_payload
    original = tmp_path / 'workspace' / 'runs' / 'x'
    original.parent.mkdir(parents=True)
    fixture_run(original, 2)
    for sequence in (1, 2):
        identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'attempt_index': 1,
                    'sample_sequence': sequence}
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'step': sequence,
                   'count': 1, 'shape': [1, 2], 'samples': [[.2, .3]]}
        publish_preview_payload(original, payload, identity, sequence)
    # An older run's absolute index: the moved run's files are found by suffix.
    index_file = original / 'previews' / 'index.json'
    index = json.loads(index_file.read_text())
    for record in index['previews']:
        record['path'] = str(index_path(original, record['path']))
    atomic_json(index_file, index)
    lost, kept = index['previews'][0], index['previews'][1]
    Path(lost['path']).unlink()
    moved = tmp_path / 'home' / 'renamed'
    moved.parent.mkdir(parents=True)
    original.rename(moved)
    session = LocalSession(8123)
    with TestClient(create_app(moved, session, poll_seconds=.01), base_url=session.origin) as client:
        artifacts = client.get('/api/v1/runs/run/artifacts').json()
        assert 'error' not in artifacts
        by_step = {record['provenance']['step']: (key, record) for key, record in artifacts['artifacts'].items()}
        assert by_step[lost['step']][1]['status'] == 'unavailable'
        key, record = by_step[kept['step']]
        assert 'status' not in record
        data = client.get(f'/api/v1/runs/run/artifacts/{key}')
        assert data.status_code == 200 and hashlib.sha256(data.content).hexdigest() == record['sha256']
        assert client.get(f'/api/v1/runs/run/artifacts/{by_step[lost["step"]][0]}').status_code == 400


def test_wildcard_bind_accepts_the_mdns_name_and_refusals_name_the_setting(tmp_path):
    fixture_run(tmp_path)
    session = LocalSession(8123, host='0.0.0.0', auth='none')
    short = socket.gethostname().lower().split('.')[0]
    assert session.permits_request(f'{short}.local:8123', f'http://{short}.local:8123')
    assert not session.permits_request('evil.local:8123')
    with TestClient(create_app(tmp_path, session, poll_seconds=.01), base_url=session.origin) as client:
        refused = client.get('/api/runs', headers={'host': 'evil.example:8123'})
        assert refused.status_code == 403
        assert 'HYPERGAN_VIEWER_HOSTS' in refused.json()['error']


def test_failed_background_projection_is_reported_instead_of_indexing_forever(tmp_path, monkeypatch):
    import shutil
    import hypergan.web_autostart as startup
    fixture_run(tmp_path, 2)
    shutil.rmtree(tmp_path / 'views')
    monkeypatch.setattr(startup, '_PROJECTION_FAILURES', {})
    calls = []

    def unwritable(run_dir):
        calls.append(run_dir)
        raise PermissionError(13, 'Permission denied', str(run_dir / 'views'))
    monkeypatch.setattr(startup, 'project_run', unwritable)

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            series = dict(series=['loss/g_total'], bucket_steps=1)
            # Nothing has failed yet: the projection is still expected.
            reason = None
            for _ in range(500):
                try:
                    service.bootstrap(MapSpec().revision, **series)
                except LookupError as exc:
                    reason = str(exc)
                    if 'being built' in reason:
                        break
                await asyncio.sleep(.01)
            assert reason == 'Projection is being built'
            projector = startup.Projector(tmp_path)
            assert projector.poll_once() == []
            assert projector.poll_once() == []
            assert len(calls) == 1, 'An unchanged failing log is not retried on every poll'
            with pytest.raises(FileNotFoundError, match='could not be built: PermissionError'):
                service.bootstrap(MapSpec().revision, **series)
            monkeypatch.setattr(startup, 'project_run', lambda run_dir: {'documents': 0})
            append_event(tmp_path, 4, 3)
            projector.poll_once()
            assert startup.projection_failure(tmp_path) is None
        finally:
            await service.close()
    asyncio.run(scenario())


def test_stream_client_that_stops_reading_is_dropped_without_an_error_log(tmp_path, monkeypatch):
    import logging
    import uvicorn
    import hypergan.web_server as web_server
    fixture_run(tmp_path)
    monkeypatch.setattr(web_server, 'STREAM_SEND_DEADLINE', .2)
    session = LocalSession(8123)
    app = create_app(tmp_path, session, poll_seconds=.01)
    with TestClient(app, base_url=session.origin) as client:
        async def stalled():
            sent = []

            async def receive():
                await asyncio.Event().wait()

            async def send(message):
                sent.append(message['type'])
                if message['type'] == 'http.response.body':
                    await asyncio.sleep(3600)  # a client that never drains
            scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1',
                     'method': 'GET', 'scheme': 'http', 'path': '/api/v1/runs/run/stream',
                     'raw_path': b'/api/v1/runs/run/stream', 'query_string': b'', 'root_path': '',
                     'headers': [(b'host', session.host.encode())], 'client': ('127.0.0.1', 1),
                     'server': ('127.0.0.1', 8123)}
            try:
                await asyncio.wait_for(app(scope, receive, send), 10)
            except Exception as exc:
                return sent, exc
            return sent, None
        sent, error = client.portal.call(stalled)
    assert sent[0] == 'http.response.start' and error is not None
    assert web_server._is_send_deadline(error), repr(error)

    # The server logs the dropped connection through uvicorn.error; that record is
    # filtered out, while any other application error still reaches the log.
    def record(exc):
        return logging.LogRecord('uvicorn.error', logging.ERROR, __file__, 1, 'Exception in ASGI application\n',
                                 None, (type(exc), exc, exc.__traceback__))
    assert web_server._quiet_send_deadline(record(error)) is False
    assert web_server._quiet_send_deadline(record(RuntimeError('boom'))) is True
    monkeypatch.setattr(uvicorn.Server, 'run', lambda self, sockets=None: None)
    listener = socket.socket()
    try:
        listener.bind(('127.0.0.1', 0))
        web_server.run_socket(tmp_path, listener, session)
    finally:
        listener.close()
    assert web_server._quiet_send_deadline in logging.getLogger('uvicorn.error').filters


def test_root_empty_at_startup_streams_as_a_dashboard_once_runs_appear(tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()

    async def scenario():
        session = LocalSession(8123)
        app = create_app(root, session, poll_seconds=.01)
        # Empty, the root looked like one pending run; now it holds runs.
        assert app.state.observations is not None
        fixture_run(root / 'r1', 1)
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                 'http_version': '1.1', 'method': 'GET', 'scheme': 'http',
                 'path': '/api/v1/stream', 'raw_path': b'/api/v1/stream', 'query_string': b'stream_id=bogus',
                 'headers': [(b'host', session.host.encode())],
                 'server': ('127.0.0.1', 8123), 'client': ('127.0.0.1', 9999), 'root_path': ''}
        sent = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)
            if message['type'] == 'http.response.body' and message.get('more_body'):
                app.state.shutdown_event.set()
        await asyncio.wait_for(app(scope, receive, send), 5)
        # The dashboard stream ignores run selectors; the single-run stream it
        # replaced would have refused the unknown selector with 400.
        assert sent[0]['status'] == 200
        assert sent[1]['body'].startswith(b'event: ready')
    asyncio.run(scenario())


def test_tailscale_self_names_are_accepted_without_dns_rebinding_exposure(monkeypatch):
    import hypergan.web_session as ws
    extracted = ws._extract_tailscale_self_names({
        'Self': {
            'HostName': 'MLServer',
            'DNSName': 'MLServer.tail12345.ts.net.',
        }
    })
    assert extracted == {'mlserver', 'mlserver.local', 'mlserver.tail12345.ts.net'}
    monkeypatch.setattr(ws, '_tailscale_machine_names', lambda: frozenset(extracted))
    session = LocalSession(8123, host='0.0.0.0')
    assert session.permits_request('mlserver:8123', 'http://mlserver:8123')
    assert session.permits_request('mlserver.tail12345.ts.net:8123', 'http://mlserver.tail12345.ts.net:8123')
    assert not session.permits_request('other-node.tail12345.ts.net:8123')
    assert not session.permits_request('evil.example:8123')


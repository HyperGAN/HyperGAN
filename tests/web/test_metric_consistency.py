"""The server distinguishes an indexed event log from projected metrics."""
import asyncio
import hashlib
import importlib.util
from pathlib import Path

from hypergan.event_views import MapSpec, Projector
from hypergan.run_state import atomic_json
from hypergan.web_service import ObservationService

_spec = importlib.util.spec_from_file_location('_web_fixtures', Path(__file__).with_name('test_web_service.py'))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def test_checkpoint_projection_lag_catches_up_without_manifest_change(tmp_path):
    manifest = _fixtures.fixture_run(tmp_path, 2)
    _fixtures.append_event(tmp_path, 4, 3)
    data = (tmp_path / 'events.jsonl').read_bytes()
    boundary = dict(schema_version=1, run_id='run', attempt_id='a', step=3,
                    sequence=4, offset=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest.update(steps=3, last_durable_step=3, durable_event_boundary=boundary)
    atomic_json(tmp_path / 'manifest.json', manifest)

    async def scenario():
        service = await ObservationService(tmp_path, poll_seconds=.01).start()
        try:
            await _fixtures.caught_up(service)
            pending = service.public_manifest()
            assert service.streams['training'].caught_up
            assert pending['durable_event_boundary'] == boundary
            assert pending['metric_consistency']['status'] == 'pending'
            assert pending['metric_consistency']['projected_offset'] < len(data)
            subscriber, _ = service.subscribe('*')
            with Projector(tmp_path) as projector:
                projector.project()
            for _ in range(300):
                if service.metric_consistency()['status'] == 'caught_up':
                    break
                await asyncio.sleep(.01)
            assert service.metric_consistency() == {
                'status': 'caught_up', 'committed_step': 3,
                'committed_offset': len(data), 'projected_offset': len(data)}
            messages = []
            while not subscriber.queue.empty():
                messages.append(await subscriber.get())
            assert any(b'event: heartbeat' in message and b'"status":"caught_up"' in message for message in messages)
            # A new attempt can restore an earlier step while extending history:
            # byte boundaries, not step comparisons, determine projection lag.
            _fixtures.append_event(tmp_path, 1, 1, attempt='b', event='resume', parent='a', restored=1)
            newer = (tmp_path / 'events.jsonl').read_bytes()
            service.manifest['durable_event_boundary'] = dict(boundary, attempt_id='b', step=1,
                sequence=1, offset=len(newer), sha256=hashlib.sha256(newer).hexdigest())
            assert service.metric_consistency()['status'] == 'pending'
            # Invalid projection history must not retain a successful status.
            stream = service.streams['projection:' + MapSpec().revision]
            stream.error = 'corrupt complete frame'
            assert service.metric_consistency()['status'] == 'unavailable'
        finally:
            await service.close()
    asyncio.run(scenario())


def test_no_durable_event_contract_is_explicitly_unavailable(tmp_path):
    service = ObservationService(tmp_path)
    service.run_id = 'run'
    service.manifest = {'steps': 100, 'last_durable_step': 100}
    assert service.public_manifest()['metric_consistency']['status'] == 'unavailable'
    assert service.public_manifest()['metric_consistency']['committed_step'] is None

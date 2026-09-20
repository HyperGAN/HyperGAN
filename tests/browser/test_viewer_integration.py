"""Real ASGI, file projection, Python bootstrap, WASM worker and browser acceptance."""
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import sync_playwright
import uvicorn

from hypergan.event_views import MapSpec, Projector, read_projection_page
from hypergan.metrics import digest
from hypergan.metrics_reducer import Reducer
from hypergan.run_state import atomic_json
from hypergan.web_launch import bind_loopback
from hypergan.web_server import create_app
from hypergan.web_session import LocalSession


class Experiment:
    def __init__(self, root):
        self.root = root
        self.manifest = None
        self.sequence = 0
        self.catalog_revision = None

    def catalog(self, version):
        descriptor = dict(kind='scalar', source='g_loss', label='Generator total', version=version)
        descriptor['definition_hash'] = digest(descriptor)
        value = dict(schema_version=1, metrics={'loss/g_total': descriptor})
        self.catalog_revision = digest(value)
        atomic_json(self.root / 'metrics' / f'catalog-{self.catalog_revision}.json', value)

    def create(self, count=3):
        self.root.mkdir()
        self.catalog('original')
        self.manifest = dict(schema_version=1, run_id='browser-run', attempt_id='attempt-a',
                             config={'name':'Browser acceptance'}, steps=0, total_steps=20,
                             last_durable_step=1, status='running', metrics_catalog=self.catalog_revision)
        self.event('start', 0, parent_attempt_id=None, restored_step=0)
        for step in range(1, count+1):
            self.event('train', step, metrics={'loss/g_total':float(step)})
        self.publish()
        self.project()

    def event(self, kind, step, **values):
        self.sequence += 1
        event = dict(schema_version=2, run_id=self.manifest['run_id'], stream_id='training',
                     stream_generation='browser-run', attempt_id=self.manifest['attempt_id'],
                     sequence=self.sequence, step=step, event=kind,
                     catalog=self.catalog_revision, **values)
        with (self.root / 'events.jsonl').open('ab') as handle:
            handle.write(json.dumps(event, allow_nan=False).encode()+b'\n')
        self.manifest['steps'] = step

    def publish(self):
        self.manifest['metrics_catalog'] = self.catalog_revision
        atomic_json(self.root / 'manifest.json', self.manifest)

    def project(self):
        with Projector(self.root) as projector:
            projector.project(limit=10000)


@pytest.fixture
def real_viewer(tmp_path):
    experiment = Experiment(tmp_path / 'run')
    listener = bind_loopback()
    session = LocalSession(listener.getsockname()[1])
    session.write_credentials(tmp_path / 'session.json')
    token = json.loads((tmp_path / 'session.json').read_text())['token']
    app = create_app(experiment.root, session, poll_seconds=.02)
    experiment.service = app.state.observations
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    deadline = time.monotonic()+10
    while not server.started:
        if not thread.is_alive() or time.monotonic()>deadline:
            raise AssertionError('ASGI server did not start')
        time.sleep(.01)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={'width':1280,'height':900})
            page = context.new_page()
            errors, requests = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(request.url))
            yield experiment, session, token, page, context, errors, requests
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), 'ASGI server did not shut down'


def sign_in(page, session, token):
    page.goto(session.origin)
    page.get_by_label('Session token', exact=True).fill(token)
    page.get_by_role('button', name='Open workspace').click()


def test_real_png_grid_loads_and_updates_without_selected_metrics(real_viewer):
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    def publish(sequence, step, pixels):
        identity = {'run_id': 'browser-run', 'attempt_id': '0001-' + 'a' * 32,
                    'sample_sequence': sequence}
        png = encode_png(pixels, 2, 1, 3, {'step': step})
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity,
                   'step': step, 'count': 2, 'shape': [2, 3, 1, 1],
                   'image_grid': {'width': 2, 'height': 1, 'channels': 3,
                                  'png_base64': base64.b64encode(png).decode('ascii')}}
        return publish_preview_payload(experiment.root, payload, identity, step, keep=2)
    publish(1, 1, bytes([255, 0, 0, 0, 255, 0]))
    sign_in(page, session, token)
    image = page.get_by_role('img', name='Generated image grid at step 1', exact=True)
    image.wait_for()
    image.scroll_into_view_if_needed()
    page.wait_for_function("() => document.querySelector('img.image-grid')?.naturalWidth === 2")
    # Decode the actual authenticated image in the browser and inspect its pixels.
    pixels = image.evaluate('''image => { const canvas = document.createElement('canvas');
      canvas.width = 2; canvas.height = 1; const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0); return [...ctx.getImageData(0, 0, 2, 1).data]; }''')
    assert pixels == [255, 0, 0, 255, 0, 255, 0, 255]
    page.get_by_role('button', name='Clear', exact=True).click()
    page.locator('#coverage').filter(has_text='No metrics selected').wait_for()
    latest_record = publish(2, 2, bytes([0, 0, 255, 255, 255, 255]))[0]
    latest = page.get_by_role('img', name='Generated image grid at step 2', exact=True)
    latest.wait_for(timeout=10000)
    latest.scroll_into_view_if_needed()
    page.wait_for_function("() => [...document.querySelectorAll('img.image-grid')].every(img => img.naturalWidth === 2)")
    assert page.locator('img.image-grid').count() == 2
    item = page.locator('#artifact-items li').filter(has=latest)
    with page.expect_download() as download:
        item.get_by_role('link', name='Download', exact=True).click()
    assert Path(download.value.path()).read_bytes() == Path(latest_record['image_grid']['path']).read_bytes()
    assert not errors


def test_actual_bootstrap_live_reconnect_and_recovery_lineage(real_viewer, monkeypatch):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create()
    page.goto(session.origin)
    assert page.locator('#workspace').is_hidden()
    page.get_by_label('Session token', exact=True).fill('wrong-token')
    page.get_by_role('button', name='Open workspace').click()
    page.locator('#login-error').filter(has_text='session').wait_for()
    page.get_by_label('Session token', exact=True).fill(token)
    page.get_by_role('button', name='Open workspace').click()
    page.locator('#g-loss').filter(has_text='3').wait_for()
    assert page.locator('#run-name').inner_text()=='Browser acceptance'
    assert page.locator('.chart-canvas canvas').count()==1
    page.locator('.data-table summary').click()
    reductions = []
    original_add = Reducer.add
    def tracked_add(self, *args, **kwargs):
        reductions.append(1)
        return original_add(self, *args, **kwargs)
    monkeypatch.setattr(Reducer, 'add', tracked_add)

    experiment.event('train',4,metrics={'loss/g_total':4.})
    experiment.publish();experiment.project()
    page.locator('#g-loss').filter(has_text='4').wait_for()
    applied_cursor=read_projection_page(experiment.root,MapSpec().revision)['cursor']
    context.set_offline(True)
    def disconnect_consumers():
        for subscriber in list(experiment.service.subscribers):
            subscriber.offer('heartbeat', {})
            subscriber.closed = True
    loop = next(iter(experiment.service.streams.values())).task.get_loop()
    loop.call_soon_threadsafe(disconnect_consumers)
    page.locator('#connection').filter(has_text='Disconnected').wait_for(timeout=10000)
    experiment.event('train',5,metrics={'loss/g_total':5.})
    experiment.publish();experiment.project()
    context.set_offline(False)
    page.locator('#g-loss').filter(has_text='5').wait_for(timeout=15000)
    reconnect_cursors=[parse_qs(urlparse(url).query).get('cursor',[None])[0]
                       for url in requests if '/stream?' in url]
    assert applied_cursor in reconnect_cursors
    assert reductions == [], 'Live fanout or reconnect reran server reduction'

    # Recover the parent at step2 with no new updates. Its abandoned3..5 values
    # must disappear immediately after the lineage notification/bootstrap.
    experiment.manifest['attempt_id']='attempt-b'
    experiment.manifest['recovery_parent']={'attempt_id':'attempt-a','step':2}
    experiment.sequence=0
    experiment.catalog('changed-definition')
    experiment.event('resume',2,parent_attempt_id='attempt-a',restored_step=2)
    experiment.publish();experiment.project()
    page.locator('#g-loss').filter(has_text='2').wait_for(timeout=15000)
    assert 'attempt-b' not in page.locator('#values-table').inner_text()
    experiment.event('train',3,metrics={'loss/g_total':33.})
    experiment.publish();experiment.project()
    page.locator('#g-loss').filter(has_text='33').wait_for(timeout=15000)
    table=page.locator('#values-table').inner_text()
    assert 'attempt-a' in table and 'attempt-b' in table
    assert len(page.locator('#values-table tr').all())==2
    assert not errors
    assert not any(token in url for url in requests)


def test_real_waiting_server_discovers_new_run_without_polling(real_viewer):
    experiment,session,token,page,context,errors,requests=real_viewer
    sign_in(page,session,token)
    page.locator('#run-name').filter(has_text='Waiting for training').wait_for()
    before=len([url for url in requests if url.endswith('/capabilities')])
    page.wait_for_timeout(200)
    assert len([url for url in requests if url.endswith('/capabilities')])==before
    experiment.create(2)
    page.locator('#g-loss').filter(has_text='2').wait_for(timeout=15000)
    assert not errors


def test_artifact_shelf_streams_while_metrics_are_unselected(real_viewer):
    import hashlib
    experiment,session,token,page,context,errors,requests=real_viewer
    experiment.create(2)
    sign_in(page,session,token)
    page.locator('#g-loss').filter(has_text='2').wait_for()
    payload=json.dumps({'shape':[2,2],'samples':[[1.,2.],[3.,4.]]}).encode()
    (experiment.root/'samples.json').write_bytes(payload)
    audio=b'RIFF fixture audio bytes'
    (experiment.root/'audio.bin').write_bytes(audio)
    records={
        'numbers':dict(path='samples.json',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                       role='sample',modality='tensor',media_type='application/json',shape=[2,2],provenance={'step':2}),
        'audio':dict(path='audio.bin',bytes=len(audio),sha256=hashlib.sha256(audio).hexdigest(),
                     role='sample',modality='audio',media_type='audio/wav',provenance={'step':2}),
    }
    # Publish in the interval after the old SSE closes and before its
    # replacement subscribes. Ready must refresh inventory to cover that gap.
    def during_reconnect(route):
        atomic_json(experiment.root/'artifacts/index.json',{'schema_version':1,'artifacts':records})
        time.sleep(.15)
        route.continue_()
    page.route('**/stream?**', during_reconnect, times=1)
    page.get_by_role('button',name='Clear',exact=True).click()
    page.locator('#coverage').filter(has_text='No metrics selected').wait_for()
    page.get_by_role('button',name='Preview numbers').wait_for(timeout=10000)
    assert 'sample · audio' in page.locator('#artifact-items').inner_text().lower()
    assert page.locator('#artifact-items img, #artifact-items audio, #artifact-items video').count()==0
    page.get_by_role('button',name='Preview numbers').click()
    page.locator('.numeric-preview').filter(has_text='1, 2, 3, 4').wait_for()
    assert 'Shape 2 × 2' in page.locator('#artifact-items').inner_text()
    # A redundant inventory notification must preserve the open preview node.
    page.locator('.numeric-preview').filter(has_text='1, 2, 3, 4').evaluate("el => el.dataset.retained = 'yes'")
    before = len([url for url in requests if url.endswith('/artifacts')])
    next(iter(experiment.service.streams.values())).task.get_loop().call_soon_threadsafe(
        experiment.service.notify, 'artifacts', {})
    page.wait_for_function("count => performance.getEntriesByType('resource').filter(x => x.name.endsWith('/artifacts')).length > count", arg=before)
    page.locator('.numeric-preview[data-retained="yes"]').filter(has_text='1, 2, 3, 4').wait_for()

    with page.expect_download() as download:
        page.locator('a[href$="/artifacts/audio"]').click()
    assert Path(download.value.path()).read_bytes()==audio
    bad = json.dumps({'shape':[2,2], 'samples':[[1,2,3],[4]]}).encode()
    (experiment.root/'bad.json').write_bytes(bad)
    records['bad'] = dict(records['numbers'], path='bad.json', bytes=len(bad), sha256=hashlib.sha256(bad).hexdigest())
    atomic_json(experiment.root/'artifacts/index.json',{'schema_version':1,'artifacts':records})
    bad_item = page.locator('#artifact-items li').filter(has=page.locator('a[href$="/artifacts/bad"]'))
    bad_item.get_by_role('button',name='Preview numbers').click()
    bad_item.locator('.numeric-preview').filter(has_text='differ from its shape').wait_for()
    assert not errors


def publish_evaluation(experiment, evaluation_id, kind, value=None, *, failed=False, step=1, protocol="c" * 64, attempt="saved-attempt"):
    # Exact terminal M4 source protocol, including a catalog absent from the
    # active training manifest. Publish registration last, as the evaluator does.
    definition = dict(kind=kind, source='custom:quality', scope='snapshot',
                      label='Snapshot quality', unit='distance', evaluation_protocol={'seed': 42})
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {'quality': definition}}
    revision = digest(catalog)
    atomic_json(experiment.root / 'metrics' / f'catalog-{revision}.json', catalog)
    event = dict(schema_version=2, event='evaluation', run_id=experiment.manifest['run_id'],
                 stream_id='evaluation:' + evaluation_id, stream_generation=evaluation_id,
                 attempt_id=evaluation_id if failed else attempt, step=0 if failed else step, seconds=12.5,
                 sequence=1, catalog=revision, evaluation_id=evaluation_id,
                 source_position_known=not failed, status='failed' if failed else 'complete',
                 metrics={}, measurement_status={}, snapshot_sha256='b'*64,
                 snapshot_identity={} if failed else {'attempt_id': attempt},
                 protocol_sha256=protocol, evaluation_protocol={'sample_count': 7, 'seed': 42})
    if failed:
        event['measurement_status']['quality'] = {'status': 'failed', 'reason': 'Evaluation fixture timeout'}
    else:
        event.setdefault('metrics' if kind == 'scalar' else 'distributions', {})['quality'] = value
    directory = experiment.root / 'metrics/evaluations' / evaluation_id
    directory.mkdir(parents=True)
    (directory / 'events.jsonl').write_text(json.dumps(event) + '\n')
    atomic_json(directory / 'stream.json', dict(schema_version=1, stream_id=event['stream_id'],
                stream_generation=evaluation_id, run_id=event['run_id'],
                path=f'metrics/evaluations/{evaluation_id}/events.jsonl', role='measurement', modality=kind))
    return event


def test_snapshot_scalar_histogram_failure_discovery_and_export(real_viewer, monkeypatch):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    experiment.manifest['status'] = 'complete'; experiment.publish()
    scalar = publish_evaluation(experiment, '1'*32, 'scalar', .25)
    sign_in(page, session, token)
    page.locator('#g-loss').filter(has_text='2').wait_for()
    page.locator('#evaluation-items li').filter(has_text='0.25 distance').wait_for()
    assert 'Snapshot quality · Evaluation' in page.locator('#metric-list').inner_text()
    page.get_by_role('button', name='Clear', exact=True).click()
    page.locator('#coverage').filter(has_text='No metrics selected').wait_for()
    monkeypatch.setattr(Reducer, 'add', lambda *a, **k: pytest.fail('Evaluation ran a server reduction'))
    histogram = publish_evaluation(experiment, '2'*32, 'histogram', {'edges':[0.,.5,1.], 'counts':[.25,.75]})
    publish_evaluation(experiment, '3'*32, 'scalar', failed=True)
    hist = page.locator('#evaluation-items li').filter(has_text='2 histogram bins')
    hist.wait_for(timeout=10000)
    failure = page.locator('#evaluation-items li').filter(has_text='Evaluation fixture timeout')
    failure.wait_for(timeout=10000)
    assert 'Source position unknown' in failure.inner_text()
    assert 'Source step 0' not in failure.inner_text()
    assert failure.locator('svg').count() == 0
    hist.get_by_text('Plot and accessible values', exact=True).click()
    hist.locator('svg rect').first.wait_for()
    assert hist.locator('svg rect').count() == 2
    assert hist.locator('table tbody tr').count() == 2
    assert '0.75' in hist.locator('table').inner_text()
    hist.get_by_text('Definition and evaluation protocol', exact=True).click()
    assert 'sample_count' in hist.locator('pre').inner_text()
    assert histogram['catalog'] in hist.locator('pre').inner_text()
    scalar_card = page.locator('#evaluation-items li').filter(has_text='0.25 distance')
    scalar_card.get_by_text('Plot and accessible values', exact=True).click()
    scalar_card.locator('svg circle').wait_for()
    assert 'Source step 1' in scalar_card.inner_text() and 'saved-attempt' in scalar_card.inner_text()
    export = scalar_card.get_by_role('link', name='Export raw evaluation').get_attribute('href')
    exported = context.request.get(session.origin + export).json()
    assert exported['events'] == [scalar]
    schema = context.request.get(session.origin + '/api/v1/openapi.json').json()
    assert 'distributions' in schema['components']['schemas']['Event']['properties']
    # Reconnect/reload discovers existing immutable sources without polling.
    page.reload()
    page.locator('#evaluation-items li').filter(has_text='2 histogram bins').wait_for(timeout=10000)
    page.locator('#evaluation-items li').filter(has_text='Evaluation fixture timeout').wait_for()
    assert page.locator('#evaluation-items li').count() == 3
    assert not errors
    assert not any(token in url for url in requests)


def test_single_measurement_has_a_visible_chart_mark(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(1)
    sign_in(page, session, token)
    page.locator('#g-loss').filter(has_text='1').wait_for()
    # With no second observation there is no line segment; the value must
    # still appear on the actual canvas, not only in the adjacent value card.
    page.wait_for_function("""() => {
      const canvas = document.querySelector('.chart-canvas canvas');
      if (!canvas) return false;
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let marks = 0;
      for (let i = 0; i < pixels.length; i += 4)
        if (Math.abs(pixels[i] - 215) < 5 && Math.abs(pixels[i+1] - 187) < 5 && Math.abs(pixels[i+2] - 129) < 5 && pixels[i+3] > 200) marks++;
      return marks > 3;
    }""")
    assert not errors


def test_cifar_final_tensor_preview_and_oversize_explanation(real_viewer):
    import hashlib
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    # The final CIFAR sample has 64 RGB 32x32 tensors: larger than both old
    # browser limits (4,096 values/64KiB) and the periodic-preview producer cap.
    shape = [64, 3, 32, 32]
    samples = [[[[.125 for _ in range(32)] for _ in range(32)] for _ in range(3)] for _ in range(64)]
    payload = json.dumps({'shape': shape, 'samples': samples}).encode()
    (experiment.root / 'samples.json').write_bytes(payload)
    record = dict(path='samples.json', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                  role='sample', modality='tensor', media_type='application/json', shape=shape, provenance={'step': 2})
    records = {'cifar': record, 'oversize': dict(record, shape=[65, 3, 64, 64])}
    atomic_json(experiment.root / 'artifacts/index.json', {'schema_version': 1, 'artifacts': records})
    sign_in(page, session, token)
    item = page.locator('#artifact-items li').filter(has=page.locator('a[href$="/artifacts/cifar"]'))
    item.get_by_role('button', name='Preview numbers').click()
    item.locator('.numeric-preview').filter(has_text='Showing 128 of 196608 values.').wait_for()
    assert item.locator('.numeric-preview').inner_text().count('0.125') == 128
    item.get_by_role('button', name='Hide numbers').click()
    assert item.locator('.numeric-preview').is_hidden()
    oversized = page.locator('#artifact-items li').filter(has=page.locator('a[href$="/artifacts/oversize"]'))
    assert oversized.get_by_role('button', name='Preview numbers').count() == 0
    assert 'Download this tensor to inspect it.' in oversized.inner_text()
    with page.expect_download() as download:
        item.get_by_role('link', name='Download').click()
    assert Path(download.value.path()).read_bytes() == payload
    assert download.value.suggested_filename == "samples.json"
    assert not errors


def test_evaluation_metrics_sort_snapshots_preserve_repeats_and_protocols(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    for index, step in enumerate([10000, 30000, 40000, 20000, 20000], 1):
        publish_evaluation(experiment, f'{index:032x}', 'scalar', 50 - step / 1000 + index / 10, step=step, attempt=f'saved-{index}')
    publish_evaluation(experiment, '6'*32, 'scalar', 9.0, step=20000, protocol='d'*64)
    publish_evaluation(experiment, '7'*32, 'scalar', 8.0, step=20000, attempt='recovered-attempt')
    sign_in(page, session, token)
    page.locator('#evaluation-items li[data-step]').nth(6).wait_for()
    assert page.locator('#evaluation-items li').evaluate_all('items => items.map(x => Number(x.dataset.step))') == [10000, 20000, 20000, 20000, 20000, 30000, 40000]
    assert page.locator('#metric-count').inner_text() == '3'
    assert page.get_by_role('checkbox', name='Snapshot quality · Evaluation', exact=True).count() == 2
    cards = page.locator('.chart-card').filter(has_text='Snapshot quality · Evaluation')
    assert cards.count() == 2
    assert cards.locator('canvas').count() == 2
    page.locator('.data-table summary').click()
    rows = page.locator('#values-table tr').filter(has_text='quality')
    assert rows.count() == 7
    assert rows.evaluate_all('rows => rows.filter(r => r.dataset.metric.endsWith("c".repeat(64))).map(r => Number(r.dataset.step))') == [10000, 20000, 20000, 20000, 30000, 40000]
    # The same-step repeated measurements remain individual exact observations.
    assert rows.filter(has_text='00000000000000000000000000000004').count() == 1
    assert rows.filter(has_text='00000000000000000000000000000005').count() == 1
    assert rows.filter(has_text='recovered-attempt').count() == 1
    page.locator('#smoothing').select_option('0.5')
    assert cards.filter(has_text='EMA uses').count() == 0
    assert cards.filter(has_text='discrete snapshot measurements').count() == 2
    # Evaluation-only selection works without a training bootstrap request.
    page.get_by_role('checkbox', name='Generator total', exact=True).uncheck()
    page.locator('#coverage').filter(has_text='Evaluation snapshots').wait_for()
    page.locator('#step-from').fill('20000'); page.locator('#step-to').fill('20000')
    page.get_by_role('button', name='Apply range').click()
    assert page.locator('#values-table tr').count() == 4
    # A new result arrives after the charts were first displayed.
    publish_evaluation(experiment, '8'*32, 'scalar', 7.0, step=20000)
    page.locator('#values-table tr').filter(has_text='88888888888888888888888888888888').wait_for(timeout=10000)
    assert page.locator('#values-table tr').count() == 5
    assert 'Evaluation duration: 12.5 seconds' in page.locator('#evaluation-items').inner_text()
    assert not errors


def test_evaluation_rejects_missing_protocol_identity(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    publish_evaluation(experiment, '9'*32, 'scalar', 1.0, protocol=None)
    sign_in(page, session, token)
    page.locator('#evaluation-items').filter(has_text='Invalid evaluation definition or protocol identity').wait_for()
    assert page.get_by_role('checkbox', name='Snapshot quality · Evaluation', exact=True).count() == 0
    assert not errors

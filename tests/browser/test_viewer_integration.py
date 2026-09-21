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
    session = LocalSession(listener.getsockname()[1], auth="token")
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


def test_color256_png_only_preview_displays_three_named_grids(real_viewer):
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(1)
    identity = {'run_id': 'browser-run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    payload = dict(schema_version=1, kind='ema-preview', identity=identity, name='g',
                   step=1, count=8, shape=[8, 3, 256, 256], samples=None, representation='png')
    for field, name, channels in [('image_grid', 'g', 3), ('real_image_grid', 'x', 3),
                                 ('input_image_grid_0', 'gray', 1)]:
        png = encode_png(bytes([128]) * 768 * 768 * channels, 768, 768, channels)
        payload[field] = dict(width=768, height=768, channels=channels, name=name,
                              png_base64=base64.b64encode(png).decode('ascii'))
    publish_preview_payload(experiment.root, payload, identity, 1)
    sign_in(page, session, token)
    page.wait_for_function("""() => {
      const images = [...document.querySelectorAll('#artifact-items img.image-grid')];
      return images.length === 3 && images.every(image => image.complete && image.naturalWidth === 768);
    }""")
    cards = page.locator('#artifact-items > li')
    assert sorted(cards.evaluate_all('cards => cards.map(card => card.dataset.sample)')) == ['g', 'gray', 'x']
    assert page.locator('#artifact-items .raw-tensor').count() == 0
    assert page.locator('#artifact-items li[data-modality="tensor"]').count() == 0
    assert not errors


def test_real_png_grid_loads_and_updates_without_selected_metrics(real_viewer):
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    def publish(sequence, step, pixels):
        identity = {'run_id': 'browser-run', 'attempt_id': '0001-' + 'a' * 32,
                    'sample_sequence': sequence, 'name': 'g'}
        png = encode_png(pixels, 2, 1, 3, {'step': step})
        real = encode_png(bytes([255, 255, 255, 0, 0, 0]), 2, 1, 3, {'step': step, 'name': 'x'})
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity,
                   'name': 'g', 'step': step, 'count': 2, 'shape': [2, 3, 1, 1],
                   'image_grid': {'width': 2, 'height': 1, 'channels': 3, 'name': 'g',
                                  'png_base64': base64.b64encode(png).decode('ascii')},
                   'real_image_grid': {'width': 2, 'height': 1, 'channels': 3, 'name': 'x',
                                       'png_base64': base64.b64encode(real).decode('ascii')}}
        return publish_preview_payload(experiment.root, payload, identity, step, keep=4)
    publish(1, 1, bytes([255, 0, 0, 0, 255, 0]))
    sign_in(page, session, token)
    image = page.get_by_role('img', name='Sample g image grid at step 1', exact=True)
    image.wait_for()
    image.scroll_into_view_if_needed()
    page.wait_for_function("() => document.querySelector('img.image-grid')?.naturalWidth === 2")
    # Decode the actual authenticated image in the browser and inspect its pixels.
    read = '''image => { const canvas = document.createElement('canvas');
      canvas.width = 2; canvas.height = 1; const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0); return [...ctx.getImageData(0, 0, 2, 1).data]; }'''
    assert image.evaluate(read) == [255, 0, 0, 255, 0, 255, 0, 255]
    # The comparable real batch is indexed beside it under its own short name.
    page.get_by_role('img', name='Sample x image grid at step 1', exact=True).wait_for()
    page.get_by_role('button', name='Clear', exact=True).click()
    page.locator('#coverage').filter(has_text='No metrics selected').wait_for()
    latest_record = publish(2, 2, bytes([0, 0, 255, 255, 255, 255]))[0]
    latest = page.get_by_role('img', name='Sample g image grid at step 2', exact=True)
    latest.wait_for(timeout=10000)
    latest.scroll_into_view_if_needed()
    page.wait_for_function("() => [...document.querySelectorAll('img.image-grid')].every(img => img.naturalWidth === 2)")
    # Each name shows one image: the newest. Older versions live behind the slider.
    generated = page.locator('#artifact-items li[data-sample="g"][data-modality="image"]')
    assert generated.count() == 1 and page.locator('img.image-grid').count() == 2
    assert page.get_by_role('img', name='Sample g image grid at step 1').count() == 0
    with page.expect_download() as download:
        generated.get_by_role('link', name='Download', exact=True).click()
    assert Path(download.value.path()).read_bytes() == Path(latest_record['image_grid']['path']).read_bytes()
    # The numbers the grid was drawn from ride along as a download on the same
    # card; they are not a second, unexplained group beside the picture.
    assert page.locator('#artifact-items li[data-modality="tensor"]').count() == 0
    with page.expect_download() as tensor:
        generated.get_by_role('link', name='Download raw tensor (JSON, shape 2 × 3 × 1 × 1)',
                              exact=True).click()
    assert json.loads(Path(tensor.value.path()).read_text())['step'] == 2
    generated.locator('input[type="range"]').focus()
    page.keyboard.press('Home')
    older = page.get_by_role('img', name='Sample g image grid at step 1', exact=True)
    older.wait_for()
    page.wait_for_function("() => document.querySelector('li[data-sample=\\'g\\'] img')?.naturalWidth === 2")
    assert older.evaluate(read) == [255, 0, 0, 255, 0, 255, 0, 255]
    assert 'Version 1 of 2 · step 1' in generated.locator('.sample-position').inner_text()
    page.keyboard.press('End')
    generated.locator('.sample-position').filter(has_text='Version 2 of 2').wait_for()
    assert not errors


def test_slider_reaches_the_first_sample_of_a_whole_run_history(real_viewer):
    """The retained history spans the run, so the slider reaches its first sample."""
    import base64
    from hypergan.image_grids import encode_png
    from hypergan.previews import publish_preview_payload
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    png = encode_png(bytes([255, 0, 0, 0, 255, 0]), 2, 1, 3, {'step': 1})
    real = encode_png(bytes([255, 255, 255, 0, 0, 0]), 2, 1, 3, {'step': 1, 'name': 'x'})
    for sequence in range(1, 61):
        step = sequence * 500
        identity = {'run_id': 'browser-run', 'attempt_id': '0001-' + 'a' * 32,
                    'sample_sequence': sequence, 'name': 'g'}
        payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity,
                   'name': 'g', 'step': step, 'count': 2, 'shape': [2, 3, 1, 1],
                   'image_grid': {'width': 2, 'height': 1, 'channels': 3, 'name': 'g',
                                  'png_base64': base64.b64encode(png).decode('ascii')},
                   'real_image_grid': {'width': 2, 'height': 1, 'channels': 3, 'name': 'x',
                                       'png_base64': base64.b64encode(real).decode('ascii')}}
        # No `keep`: the default bound a plain `hypergan train` publishes with,
        # which sixty publications stay inside.
        publish_preview_payload(experiment.root, payload, identity, step)
    sign_in(page, session, token)
    generated = page.locator('#artifact-items li[data-sample="g"][data-modality="image"]')
    generated.locator('.sample-position').filter(has_text='Version 60 of 60').wait_for()
    slider = generated.locator('input[type="range"]')
    assert (slider.get_attribute('min'), slider.get_attribute('max')) == ('0', '59')
    # One picture per name at a time, so scrubbing fetches the shown step only
    # and a long history never downloads itself up front.
    assert page.locator('img.image-grid').count() == 2
    fetched = [url for url in requests if '/artifacts/' in url]
    assert len(fetched) <= 4, fetched
    slider.focus()
    page.keyboard.press('Home')
    page.get_by_role('img', name='Sample g image grid at step 500', exact=True).wait_for()
    assert 'Version 1 of 60 · step 500' in generated.locator('.sample-position').inner_text()
    page.keyboard.press('End')
    generated.locator('.sample-position').filter(has_text='Version 60 of 60').wait_for()
    assert len([url for url in requests if '/artifacts/' in url]) <= 8
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


def publish_evaluation(experiment, evaluation_id, kind, value=None, *, failed=False, cancelled=False, step=1, protocol="c" * 64, attempt="saved-attempt", metric='quality'):
    # Exact terminal M4 source protocol, including a catalog absent from the
    # active training manifest. Publish registration last, as the evaluator does.
    definition = dict(kind=kind, source=f'custom:{metric}', scope='snapshot',
                      label=f'Snapshot {metric}', unit='distance', evaluation_protocol={'seed': 42})
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {metric: definition}}
    revision = digest(catalog)
    atomic_json(experiment.root / 'metrics' / f'catalog-{revision}.json', catalog)
    event = dict(schema_version=2, event='evaluation', run_id=experiment.manifest['run_id'],
                 stream_id='evaluation:' + evaluation_id, stream_generation=evaluation_id,
                 attempt_id=evaluation_id if failed else attempt, step=0 if failed else step, seconds=12.5,
                 sequence=1, catalog=revision, evaluation_id=evaluation_id,
                 source_position_known=not failed, status='cancelled' if cancelled else 'failed' if failed else 'complete',
                 metrics={}, measurement_status={}, snapshot_sha256='b'*64,
                 snapshot_identity={} if failed else {'attempt_id': attempt},
                 protocol_sha256=protocol, evaluation_protocol={'sample_count': 7, 'seed': 42})
    if cancelled:
        event['measurement_status'][metric] = {'status': 'cancelled', 'reason': 'Training stopped before evaluation finished'}
    elif failed:
        event['measurement_status'][metric] = {'status': 'failed', 'reason': 'Evaluation fixture timeout'}
    else:
        event.setdefault('metrics' if kind == 'scalar' else 'distributions', {})[metric] = value
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
    quality = page.locator('#evaluation-items li[data-metric="quality"]')
    quality.filter(has_text='0.25 distance').wait_for()
    # One scalar result is a chart with a visible point, not a text card.
    assert quality.locator('.chart-canvas canvas').count() == 1
    # Snapshot metrics live here alone: Learning curves never lists or charts one.
    assert 'Snapshot quality' not in page.locator('#metric-list').inner_text()
    assert page.locator('#charts').get_by_text('Snapshot quality', exact=True).count() == 0
    # The metric id is not repeated under a label that already differs from it.
    assert quality.locator('h3').inner_text() == 'Snapshot quality'
    page.get_by_role('button', name='Clear', exact=True).click()
    page.locator('#coverage').filter(has_text='No metrics selected').wait_for()
    monkeypatch.setattr(Reducer, 'add', lambda *a, **k: pytest.fail('Evaluation ran a server reduction'))
    histogram = publish_evaluation(experiment, '2'*32, 'histogram', {'edges':[0.,.5,1.], 'counts':[.25,.75]}, metric='spread')
    publish_evaluation(experiment, '3'*32, 'scalar', failed=True)
    hist = page.locator('#evaluation-items li[data-metric="spread"]')
    hist.filter(has_text='2 histogram bins').wait_for(timeout=10000)
    # The failure is a compact status line under its own metric, with no step zero.
    failure = quality.locator('.evaluation-status li')
    failure.wait_for(timeout=10000)
    assert failure.count() == 1
    assert 'Source position unknown · Failed · Evaluation fixture timeout' == failure.inner_text()
    assert 'Source step 0' not in quality.inner_text()
    assert hist.locator('.chart-canvas').count() == 0
    hist.locator('details.evaluation-results > summary').click()
    hist.locator('details.evaluation-result > summary').click()
    hist.locator('svg').first.wait_for()
    hist.locator('svg rect').first.wait_for()
    assert hist.locator('svg rect').count() == 2
    assert hist.locator('table tbody tr').filter(has_text='0.75').count() == 1
    hist.get_by_text('Definition and evaluation protocol', exact=True).click()
    assert 'sample_count' in hist.locator('pre').inner_text()
    assert histogram['catalog'] in hist.locator('pre').inner_text()
    # Per-result details stay collapsed until asked for, and keep the raw export.
    assert quality.locator('details.evaluation-results').evaluate('node => node.open') is False
    quality.locator('details.evaluation-results > summary').click()
    quality.locator('details.evaluation-results tbody tr').first.wait_for()
    assert 'Source step 1' in quality.inner_text() and 'saved-attempt' in quality.inner_text()
    export = quality.locator('details.evaluation-result a.text-link').first.get_attribute('href')
    exported = context.request.get(session.origin + export).json()
    assert exported['events'] == [scalar]
    schema = context.request.get(session.origin + '/api/v1/openapi.json').json()
    assert 'distributions' in schema['components']['schemas']['Event']['properties']
    # Reconnect/reload discovers existing immutable sources without polling.
    page.reload()
    page.locator('#evaluation-items li[data-metric="spread"]').filter(has_text='2 histogram bins').wait_for(timeout=10000)
    page.locator('#evaluation-items li[data-metric="quality"] .evaluation-status li').wait_for()
    assert page.locator('#evaluation-items > li').count() == 2
    assert not errors
    assert not any(token in url for url in requests)


def test_snapshot_chart_draws_one_point_and_extends_across_steps(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    publish_evaluation(experiment, '1'*32, 'scalar', 42.5, step=10000)
    sign_in(page, session, token)
    card = page.locator('#evaluation-items li[data-metric="quality"]')
    card.filter(has_text='42.5 distance').wait_for()
    assert card.locator('.chart-canvas canvas').count() == 1
    assert '1 completed evaluation' in card.locator('.chart-note').inner_text()
    # A single result still paints its marker on the canvas, not an empty line.
    page.wait_for_function("""() => {
      const canvas = document.querySelector('#evaluation-items .chart-canvas canvas');
      if (!canvas) return false;
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let marks = 0;
      for (let i = 0; i < pixels.length; i += 4)
        if (Math.abs(pixels[i] - 215) < 5 && Math.abs(pixels[i+1] - 187) < 5 && Math.abs(pixels[i+2] - 129) < 5 && pixels[i+3] > 200) marks++;
      return marks > 3;
    }""")
    assert card.get_attribute('data-steps') == '10000'
    card.locator('details.evaluation-results > summary').click()
    rows = card.locator('details.evaluation-results tbody tr')
    rows.first.wait_for()
    assert rows.evaluate_all('rows => rows.map(r => r.dataset.step)') == ['10000']
    assert '12.5' in rows.first.inner_text()
    # New streams extend the same chart in place, in step order, losing nothing.
    publish_evaluation(experiment, '2'*32, 'scalar', 21.5, step=30000, attempt='saved-2')
    publish_evaluation(experiment, '3'*32, 'scalar', 30.5, step=20000, attempt='saved-3')
    publish_evaluation(experiment, '4'*32, 'scalar', cancelled=True, step=40000)
    page.locator('#evaluation-items li[data-metric="quality"][data-steps="10000,20000,30000"]').wait_for(timeout=10000)
    card.locator('.evaluation-status li').filter(has_text='Cancelled').wait_for(timeout=10000)
    assert page.locator('#evaluation-items > li').count() == 1
    assert card.locator('.chart-canvas canvas').count() == 1
    assert '3 completed evaluations' in card.locator('.chart-note').inner_text()
    assert 'Source step 40000 · Cancelled · Training stopped' in card.locator('.evaluation-status li').inner_text()
    # The details list stayed open across the update and lists every step.
    assert rows.evaluate_all('rows => rows.map(r => r.dataset.step)') == ['10000', '20000', '30000', '40000']
    assert card.locator('details.evaluation-result').count() == 4
    assert page.locator('#evaluation-items canvas').count() == 1
    assert not errors


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
    card = page.locator('#evaluation-items li[data-metric="quality"]')
    # Every stream of one metric shares a single chart; history keeps its order.
    page.locator('#evaluation-items li[data-steps="10000,20000,20000,20000,20000,30000,40000"]').wait_for(timeout=10000)
    assert page.locator('#evaluation-items > li').count() == 1
    assert card.locator('.chart-canvas canvas').count() == 1
    card.locator('details.evaluation-results > summary').click()
    card.locator('details.evaluation-results tbody tr').first.wait_for()
    assert card.locator('details.evaluation-results tbody tr').evaluate_all(
        'rows => rows.map(r => Number(r.dataset.step))') == [10000, 20000, 20000, 20000, 20000, 30000, 40000]
    # Learning curves counts, lists and charts the training stream only; every
    # protocol of a snapshot metric stays inside this one tile.
    assert page.locator('#metric-count').inner_text() == '1'
    assert page.get_by_role('checkbox', name='Snapshot quality', exact=False).count() == 0
    assert page.locator('.chart-card').filter(has_text='Snapshot quality').count() == 0
    assert card.locator('.chart-canvas canvas').count() == 1
    page.locator('.data-table summary').click()
    assert page.locator('#values-table tr').filter(has_text='quality').count() == 0
    # The same-step repeated measurements remain individual exact observations.
    rows = card.locator('details.evaluation-results tbody tr')
    assert rows.filter(has_text='00000000000000000000000000000004').count() == 1
    assert rows.filter(has_text='00000000000000000000000000000005').count() == 1
    assert rows.filter(has_text='recovered-attempt').count() == 1
    page.locator('#smoothing').select_option('0.5')
    assert page.locator('.chart-card').filter(has_text='discrete snapshot measurements').count() == 0
    # Clearing the training selection leaves the snapshot panel untouched.
    page.get_by_role('checkbox', name='Generator total', exact=True).uncheck()
    page.locator('#coverage').filter(has_text='No metrics selected').wait_for()
    assert card.locator('.chart-canvas canvas').count() == 1
    # A new result arrives after the charts were first displayed.
    publish_evaluation(experiment, '8'*32, 'scalar', 7.0, step=20000)
    # The late result joins the same card, and duration stays in its details.
    page.locator('#evaluation-items li[data-metric="quality"]'
                 '[data-steps="10000,20000,20000,20000,20000,20000,30000,40000"]').wait_for(timeout=10000)
    assert card.locator('details.evaluation-results tbody tr').count() == 8
    assert card.locator('details.evaluation-results tbody tr').first.inner_text().count('12.5') == 1
    assert not errors


def test_evaluation_rejects_missing_protocol_identity(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create(2)
    publish_evaluation(experiment, '9'*32, 'scalar', 1.0, protocol=None)
    sign_in(page, session, token)
    page.locator('#evaluation-items').filter(has_text='Invalid evaluation definition or protocol identity').wait_for()
    assert page.get_by_role('checkbox', name='Snapshot quality').count() == 0
    assert not errors


def test_configured_snapshot_metrics_visible_before_results_and_live_schedule(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create()
    catalog_path = experiment.root / 'metrics' / f'catalog-{experiment.catalog_revision}.json'
    catalog = json.loads(catalog_path.read_text())
    for metric, spec in {
        'fid50k_train': {'trigger': 'interval', 'every_steps': 10000, 'on_busy': 'skip', 'evaluation': {'device': 'cuda:1'}},
        'fid_manual': {'trigger': 'manual', 'evaluation': {'device': 'cuda:0'}},
    }.items():
        definition = dict(kind='scalar', source='custom:fid', scope='snapshot', label=metric, specification=spec)
        definition['definition_hash'] = digest(definition)
        catalog['metrics'][metric] = definition
    experiment.catalog_revision = digest(catalog)
    atomic_json(experiment.root / 'metrics' / f'catalog-{experiment.catalog_revision}.json', catalog)
    experiment.publish()
    sign_in(page, session, token)
    # One tile per metric, whether or not it has published anything yet.
    items = page.locator('#evaluation-items')
    interval = items.locator('li[data-metric="fid50k_train"]')
    manual = items.locator('li[data-metric="fid_manual"]')
    interval.locator('.evaluation-schedule-status').filter(has_text='Every 10000 steps').wait_for()
    assert items.locator('> li').count() == 2
    # One scheduled metric is enough; the unscheduled notice stays out of the way.
    assert page.locator('#evaluation-unscheduled').is_hidden()
    # The empty state is one quiet line where the chart will be, not a wall of text.
    assert interval.locator('.evaluation-empty').inner_text() == 'No evaluations yet · first at step 10000'
    assert manual.locator('.evaluation-schedule-status').inner_text() == 'Manual'
    assert manual.locator('.evaluation-empty').inner_text() == 'No evaluations yet · run hypergan evaluate'
    # The label already is the metric id, so the id is not repeated under it.
    assert interval.locator('p.quiet').count() == 0
    assert interval.locator('.chart-canvas').count() == 0
    # Device and busy policy are one expansion away, not in the tile body.
    assert 'cuda:1' not in interval.inner_text()
    interval.locator('details.evaluation-results > summary').click()
    interval.locator('details p.quiet').filter(has_text='Device cuda:1').wait_for()
    assert 'On busy · skip' in interval.inner_text()
    assert page.locator('#evaluations').is_visible()
    experiment.manifest['evaluation_schedule'] = {'fid50k_train': {
        'status': 'running', 'source_step': 10000, 'next_step': 30000,
        'skipped_busy': 1, 'last_skipped_step': 20000, 'reason': 'worker_busy',
    }}
    experiment.publish()
    interval.locator('.evaluation-schedule-status').filter(
        has_text='Every 10000 steps · running since step 10000').wait_for()
    assert interval.locator('.evaluation-empty').inner_text() == 'No evaluations yet · next at step 30000'
    assert '1 skipped while busy · last at step 20000' in interval.inner_text()
    assert 'An evaluator was busy' in interval.inner_text()
    experiment.manifest['evaluation_schedule']['fid50k_train'].update(status='failed', reason='Evaluator exceeded its deadline')
    experiment.publish()
    interval.locator('.evaluation-schedule-status').filter(
        has_text='Every 10000 steps · failed at step 10000').wait_for()
    assert 'Evaluator exceeded its deadline' in interval.inner_text()
    experiment.manifest['status'] = 'stopped'
    experiment.publish()
    interval.locator('.evaluation-empty').filter(has_text='when training resumes').wait_for()
    page.reload()
    interval.locator('.evaluation-schedule-status').filter(
        has_text='Every 10000 steps · failed at step 10000').wait_for()
    experiment.manifest['evaluation_schedule']['fid50k_train'].update(status='disabled', reason='Interval evaluations are disabled')
    experiment.publish()
    interval.locator('.evaluation-schedule-status').filter(
        has_text='Every 10000 steps · disabled').wait_for()
    assert interval.locator('.evaluation-empty').inner_text() == 'No evaluations yet'
    assert manual.locator('.evaluation-schedule-status').inner_text() == 'Manual'
    assert not errors


def test_all_manual_snapshot_metrics_show_a_persistent_unscheduled_notice(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create()
    catalog_path = experiment.root / 'metrics' / f'catalog-{experiment.catalog_revision}.json'
    catalog = json.loads(catalog_path.read_text())
    for metric in ('fid50k_train', 'fid_smoke'):
        definition = dict(kind='scalar', source='custom:fid', scope='snapshot', label=metric,
                          specification={'trigger': 'manual', 'evaluation': {'device': 'cuda:0'}})
        definition['definition_hash'] = digest(definition)
        catalog['metrics'][metric] = definition
    experiment.catalog_revision = digest(catalog)
    atomic_json(experiment.root / 'metrics' / f'catalog-{experiment.catalog_revision}.json', catalog)
    experiment.publish()
    sign_in(page, session, token)
    notice = page.locator('#evaluation-unscheduled')
    notice.filter(has_text='No automatic evaluation is scheduled.').wait_for()
    text = notice.inner_text()
    assert 'Snapshot metrics fid50k_train, fid_smoke set trigger = "manual"' in text
    assert 'set trigger = "interval" with every_steps' in text
    assert 'hypergan resume RUN --config CONFIG' in text
    assert page.locator('#evaluations').is_visible()
    # It is a panel notice, not a launch-time toast: training on does not retire it.
    for step in range(4, 7):
        experiment.event('train', step, metrics={'loss/g_total': float(step)})
    experiment.publish()
    experiment.project()
    page.locator('#step').filter(has_text='6').wait_for()
    assert notice.is_visible()
    # Each manual metric is one tile, never a schedule tile beside a result tile.
    assert page.locator('#evaluation-items li[data-metric="fid_smoke"]').count() == 1
    assert page.locator('#evaluation-items > li').count() == 2
    # A run that does have a schedule never sees it.
    experiment.manifest['evaluation_schedule'] = {'fid50k_train': {'status': 'complete', 'source_step': 6}}
    experiment.publish()
    page.locator('#evaluation-items li[data-metric="fid50k_train"] .evaluation-schedule-status').filter(
        has_text='Manual · last at step 6').wait_for()
    assert notice.is_hidden()
    assert not errors


def test_cancelled_evaluation_retains_source_and_export_without_failure_or_value(real_viewer):
    experiment, session, token, page, context, errors, requests = real_viewer
    experiment.create()
    publish_evaluation(experiment, 'c' * 32, 'scalar', cancelled=True, step=10000)
    sign_in(page, session, token)
    card = page.locator('#evaluation-items li[data-metric="quality"]')
    status = card.locator('.evaluation-status li')
    status.wait_for()
    assert status.inner_text() == 'Source step 10000 · Cancelled · Training stopped before evaluation finished'
    # Nothing was measured: no chart, no value and no failure styling.
    assert card.locator('.chart-canvas, .evaluation-failure, .evaluation-value').count() == 0
    assert card.get_attribute('data-steps') == ''
    assert page.locator('#charts').get_by_text('Snapshot quality', exact=True).count() == 0
    card.locator('details.evaluation-results > summary').click()
    card.locator('details.evaluation-results tbody tr').first.wait_for()
    assert 'Cancelled' in card.locator('details.evaluation-results tbody tr').inner_text()
    export = card.locator('details.evaluation-result a.text-link').get_attribute('href')
    response = context.request.get(session.origin + export)
    assert response.ok
    assert response.json()['events'][0]['status'] == 'cancelled'
    assert not errors

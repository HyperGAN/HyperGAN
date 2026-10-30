"""Actual browser UI with fixture public HTTP/SSE API; no training/runtime imports."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import sync_playwright

from hypergan.metrics_reducer import Reducer, assets

MAP = 'a' * 64
VIEW = 'b' * 64
DEFINITION = 'c' * 64
LINEAGE = 'd' * 64
RUN = 'fixture-run'
METRICS = {'loss/g_total': 'Generator total', 'loss/d_total': 'Discriminator total',
           'loss/gradient_penalty':'D gradient penalty contribution'}


@pytest.fixture
def viewer():
    reducer = Reducer()
    control = {'paths': [], 'streams': [], 'step': 2, 'pending': False, 'release': False,
               'status': 'running',
               'artifacts': {}, 'assets': {}, 'artifact_revision': 0}
    condition = threading.Condition()
    def frame(step):
        values = {'loss/g_total': 1 / step, 'loss/d_total': -.5 if step == 2 else 2 / step,
                  'loss/gradient_penalty':.01 * step}
        return {'stream_id': f'projection:{MAP}', 'cursor': f'cursor:{step}', 'frame': {
            'map_revision': MAP, 'projection_sequence': step,
            'source': {'run_id': RUN, 'attempt_id': 'attempt-1'},
            'emissions': [{'id': f'{step*10+i:064x}', 'key':[metric,'attempt-1',step],
                           'definition_hash':DEFINITION,'value':value}
                          for i,(metric,value) in enumerate(values.items())]}}
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def send(self, status, value, mime='application/json'):
            body=value if isinstance(value,bytes) else json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',len(body));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            length=int(self.headers.get('Content-Length') or 0)
            request=json.loads(self.rfile.read(length)) if length else {}
            if self.path=='/api/v1/session':
                if request!={'token':'fixture-secret'}:
                    return self.send(401,{'error':'Invalid session token'})
                body=b'{"ok":true}';self.send_response(200);self.send_header('Set-Cookie','session=valid; HttpOnly; SameSite=Strict; Path=/')
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',len(body));self.end_headers();self.wfile.write(body)
                return
            if 'session=valid' not in self.headers.get('Cookie',''):return self.send(401,{'error':'Session required'})
            if self.path in (f'/api/v1/runs/{RUN}/stop',f'/api/v1/runs/{RUN}/checkpoint',f'/api/v1/runs/{RUN}/sample'):
                action=self.path.rsplit('/',1)[-1]
                control.setdefault('posts',[]).append(self.path)
                return self.send(200,control.get(f'{action}_response',{'ok':True,'action':action,'run_id':RUN,'request_id':'req-1','status':'pending'}))
            return self.send(404,{'error':'Not found'})
        def do_GET(self):
            parsed=urlparse(self.path);path=parsed.path;query=parse_qs(parsed.query,keep_blank_values=True)
            control['paths'].append(self.path)
            if path=='/' or path.startswith('/runs/'): return self.send(200,files('hypergan.web_assets').joinpath('index.html').read_bytes(),'text/html')
            if path.startswith('/assets/') or path.startswith('/reducers/'):
                resource=(files('hypergan.web_assets') if path.startswith('/assets/') else assets()).joinpath(path.rsplit('/',1)[-1])
                mime='text/javascript' if path.endswith('.js') else 'text/css' if path.endswith('.css') else 'image/png' if path.endswith('.png') else 'application/wasm' if path.endswith('.wasm') else 'application/octet-stream'
                return self.send(200,resource.read_bytes(),mime)
            if 'session=valid' not in self.headers.get('Cookie',''):return self.send(401,{'error':'Session required'})
            if path=='/api/v1/capabilities':return self.send(200,{'run_id':None if control.get('waiting') else RUN,'reducer':reducer.spec})
            if path==f'/api/v1/runs/{RUN}':
                run={'run_id':RUN,'status':control['status'],'steps':control['step'],'last_durable_step':1,'total_steps':100,'config':{'name':'Color / reference study'}}
                if control.get('progress',True):
                    run.update(steps_per_second=12.5,training_seconds=5025.4,samples_seen=control['step']*32,global_batch_size=32)
                if control.get('run_extra'):
                    run.update(control['run_extra'])
                return self.send(200,run)
            if f'/api/v1/runs/{RUN}/requests/' in path:
                req_id=path.rsplit('/',1)[-1]
                control.setdefault('request_polls',[]).append(req_id)
                receipt=control.get('requests',{}).get(req_id)
                return self.send(404,{'error':'Not found'}) if receipt is None else self.send(200,receipt)
            if path.endswith('/metrics/catalog'):return self.send(200,{'schema_version':1,'metrics':{metric:(label if isinstance(label,dict) else {'label':label,'kind':'scalar','definition_hash':DEFINITION})for metric,label in control.get('catalog',METRICS).items()}})
            if '/artifacts/' in path:
                time.sleep(control.get('asset_delay',{}).get(path.rsplit('/',1)[-1],0))
                asset=control['assets'].get(path.rsplit('/',1)[-1])
                mime='application/json' if asset and asset[:1]==b'{' else 'image/png'
                return self.send(404,{'error':'Not found'}) if asset is None else self.send(200,asset,mime)
            if path.endswith('/artifacts'):return self.send(200,{'schema_version':1,'artifacts':control['artifacts']})
            if path.endswith('/model'):
                return self.send(200,control['model']) if control.get('model') else self.send(404,{'error':'No model configuration recorded for this run'})
            if path.endswith('/views'):return self.send(200,{'map_revision':MAP})
            if path.endswith('/bootstrap'):
                if control['pending'] and not control['release']:return self.send(202,{'status':'pending'})
                selected=query.get('series',[''])[0].split(',');bucket=int(query['bucket_steps'][0]);groups={}
                for step in range(1,control['step']+1):
                    if step<int(query.get('step_from',['0'])[0]) or step>int(query.get('step_to',['9007199254740991'])[0]):continue
                    for emission in frame(step)['frame']['emissions']:
                        if emission['key'][0] not in selected:continue
                        key=[emission['key'][0],DEFINITION,'attempt-1',step//bucket*bucket];encoded=json.dumps(key)
                        old=groups.get(encoded,{'key':key,'state':reducer.identity('envelope/v1')})
                        old['state']=reducer.add(old['state'],[{'value':emission['value'],'position':[step,emission['id']]}]);groups[encoded]=old
                return self.send(200,{'schema_version':1,'run_id':RUN,'map_revision':MAP,'view_revision':VIEW,'module_sha256':reducer.module_sha256,
                    'lineage_revision':LINEAGE,'lineage':[{'attempt_id':'attempt-1','through_step':None}], 'bucket_steps':bucket,
                    'groups':list(groups.values()),'cursor':f'cursor:{control["step"]}','projection_sequence':control['step'],'coverage':{'complete':True}})
            if path.endswith('/stream'):
                cursor=query.get('cursor',['cursor:0'])[0];control['streams'].append(cursor)
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers()
                def emit(name,value):
                    self.wfile.write(f'event: {name}\ndata: {json.dumps(value)}\n\n'.encode());self.wfile.flush()
                try:
                    emit('ready',{})
                    next_step=int(cursor.split(':')[-1])+1
                    deadline=time.monotonic()+15
                    sent_ready=False
                    published=control['artifact_revision']
                    while time.monotonic()<deadline:
                        with condition:condition.wait(.03)
                        if path=='/api/v1/stream':
                            if not control.get('waiting'):emit('metadata',{'run_id':RUN});return
                            continue
                        if control['artifact_revision']!=published:
                            published=control['artifact_revision'];emit('artifacts',{'revision':f'{published:064x}'})
                        if control['release'] and not sent_ready:emit('bootstrap_ready',{});sent_ready=True
                        if control['step']>=next_step:
                            emit('frame',frame(next_step));next_step+=1
                        if control.get('gap_at') and next_step > control['gap_at']:
                            control.pop('gap_at');emit('gap',{'reason':'replay_budget_exceeded'});return
                        if control.get('disconnect') and cursor!='cursor:0':control['disconnect']=False;return
                except (BrokenPipeError,ConnectionResetError):pass
                return
            return self.send(404,{'error':'Not found'})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.on('console',lambda msg:errors.append(msg.text) if 'Content Security Policy' in msg.text else None)
            page.goto(f'http://127.0.0.1:{server.server_port}')
            yield page,control,condition,errors
            browser.close()
    finally:server.shutdown();thread.join(timeout=5);server.server_close()


def login(page):
    page.get_by_label('Session token',exact=True).fill('fixture-secret')
    page.get_by_role('button',name='Open workspace').click()
    page.locator('#stream-position[data-projection="2"]').wait_for()


def test_login_plots_live_ack_reconnect_and_exact_table(viewer,tmp_path):
    page,control,condition,errors=viewer
    assert page.locator('#workspace').is_hidden()
    login(page)
    assert page.locator('#g-loss').inner_text()=='0.5'
    assert page.locator('#d-loss').inner_text()=='-0.5'
    assert page.locator('.chart-canvas canvas').count()==3
    page.locator('.data-table summary').click()
    assert '0.5' in page.locator('#values-table').inner_text()
    with condition:control['step']=3;condition.notify_all()
    page.locator('#stream-position[data-projection="3"]').wait_for()
    page.locator('#values-table').filter(has_text='0.3333333333333333').wait_for()
    with condition:control['disconnect']=True;condition.notify_all()
    page.locator('#connection').filter(has_text='Disconnected').wait_for()
    page.locator('#connection').filter(has_text='Live stream').wait_for(timeout=5000)
    assert 'cursor:3' in control['streams']
    assert not any('fixture-secret' in path for path in control['paths'])
    page.screenshot(path=str(tmp_path/'viewer-desktop.png'),full_page=True)
    assert not errors


@pytest.mark.parametrize('status,label',[('running','Training'),('failed','Failed'),('quiesced','Quiesced')])
def test_status_badge_reads_in_plain_words_beside_a_labelled_run_id(viewer,status,label):
    page,control,condition,errors=viewer
    control['status']=status;login(page)
    badge=page.locator('#run-status')
    # text_content, not inner_text: the badge is uppercased by the stylesheet.
    assert badge.text_content()==label
    assert badge.get_attribute('data-status')==status
    assert page.locator('#run-id').text_content()==RUN
    assert page.locator('.run-identity').text_content().strip().startswith('Run id')
    page.get_by_role('button',name='Copy',exact=True).click()
    # Either the clipboard took it, or the id was selected for a keystroke.
    page.locator('#run-id-status:not(:empty)').wait_for()
    assert 'opy' in page.locator('#run-id-status').text_content()
    assert not errors


def test_select_search_symlog_smoothing_range_and_responsive_accessibility(viewer):
    page,control,condition,errors=viewer;login(page)
    page.get_by_label('Search metrics').fill('generator')
    assert page.locator('.metric-option').count()==1
    page.get_by_label('Search metrics').fill('')
    page.locator('#scale').select_option('symlog')
    assert page.locator('#d-loss').inner_text()=='-0.5'
    assert 'excluded' not in page.locator('#charts').inner_text()
    page.locator('#smoothing').select_option('0.2')
    assert page.get_by_text('EMA uses visible envelope points;',exact=False).count()==3
    page.locator('#scale').select_option('linear')
    assert page.locator('.chart-canvas canvas').count()==3
    page.locator('#scale').select_option('symlog')
    page.locator('#step-from').fill('2');page.locator('#step-to').fill('2');page.get_by_role('button',name='Apply range').click()
    page.locator('#stream-position[data-projection="2"]').wait_for()
    assert any('step_from=2' in path and 'step_to=2' in path for path in control['paths'])
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.get_by_role('button',name='Clear',exact=True).click()
    assert page.locator('#empty').is_visible()
    assert page.locator('.chart-canvas canvas').count()==0
    assert not errors


def test_pending_history_waits_for_stream_notification_without_polling(viewer):
    page,control,condition,errors=viewer
    control['pending']=True
    page.get_by_label('Session token',exact=True).fill('fixture-secret');page.get_by_role('button',name='Open workspace').click()
    page.locator('#coverage').filter(has_text='waiting for stream notification').wait_for()
    count=sum('/bootstrap?' in path for path in control['paths'])
    page.wait_for_timeout(300)
    assert sum('/bootstrap?' in path for path in control['paths'])==count
    with condition:control['release']=True;condition.notify_all()
    page.locator('#stream-position[data-projection="2"]').wait_for()
    assert page.locator('#g-loss').inner_text()=='0.5'
    assert not errors


def test_server_waiting_for_run_activates_from_global_stream(viewer):
    page,control,condition,errors=viewer
    control['waiting']=True
    page.get_by_label('Session token',exact=True).fill('fixture-secret');page.get_by_role('button',name='Open workspace').click()
    page.locator('#run-name').filter(has_text='Waiting for training').wait_for()
    count=control['paths'].count('/api/v1/capabilities')
    page.wait_for_timeout(300)
    assert control['paths'].count('/api/v1/capabilities')==count
    with condition:control['waiting']=False;condition.notify_all()
    page.locator('#stream-position[data-projection="2"]').wait_for()
    assert not errors


def test_worker_stages_whole_frame_and_ignores_replay(viewer):
    page,control,condition,errors=viewer;login(page)
    result=page.evaluate('''async ({map,definition,run})=>{
      const response=await fetch(`/api/v1/runs/${run}/views/${map}/bootstrap?series=loss%2Fg_total&bucket_steps=10`);
      const bootstrap=await response.json();const worker=new Worker('/assets/view-worker.js',{type:'module'});let id=0;
      const call=message=>new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Worker fixture deadline')),5000);worker.onmessage=({data})=>{clearTimeout(timer);resolve(data);};worker.postMessage({...message,id:++id});});
      try{
        const initial=await call({op:'bootstrap',bootstrap,selected:['loss/g_total']});
        const emission={id:'f'.repeat(64),key:['loss/g_total','attempt-1',3],definition_hash:definition,value:3};
        const envelope={stream_id:`projection:${map}`,cursor:'cursor:3',frame:{map_revision:map,projection_sequence:3,source:{run_id:run,attempt_id:'attempt-1'},emissions:[emission,{...emission,value:'bad'}]}};
        const failed=await call({op:'frame',envelope});
        envelope.frame.emissions=[emission];const complete=await call({op:'frame',envelope});
        const duplicate=await call({op:'frame',envelope});
        envelope.frame.projection_sequence=5;const gap=await call({op:'frame',envelope});
        return {initial,failed,complete,duplicate,gap};
      }finally{worker.terminate();}
    }''',{'map':MAP,'definition':DEFINITION,'run':RUN})
    assert 'error' in result['failed']
    assert result['complete']['ok']['groups'][0]['value']['count']==3
    assert result['complete']['ok']['cursor']=='cursor:3'
    assert result['duplicate']['ok']['replay'] is True
    assert result['duplicate']['ok']['groups']==[]
    assert 'Coverage gap' in result['gap']['error']
    assert not errors


def test_delivery_gap_keeps_reduced_state_and_resumes_acknowledged_cursor(viewer):
    page, control, condition, errors = viewer
    login(page)
    page.locator('#stream-position[data-projection="2"]').wait_for()
    initial_requests = sum('/bootstrap?' in path for path in control['paths'])
    control['gap_at'] = 4
    control['step'] = 7
    with condition:
        condition.notify_all()
    page.locator('#stream-position[data-projection="7"]').wait_for()
    assert 'cursor:4' in control['streams']
    assert sum('/bootstrap?' in path for path in control['paths']) == initial_requests
    assert not errors


def test_named_samples_group_with_latest_image_and_history_slider(viewer):
    """Samples are indexed by name; images show the newest with a scrubber."""
    from hypergan.image_grids import encode_png
    page, control, condition, errors = viewer
    def image(identifier, name, step, pixel):
        control['assets'][identifier] = encode_png(pixel, 1, 1, 3, {'step': step})
        control['artifacts'][identifier] = {
            'role': 'sample', 'modality': 'image', 'media_type': 'image/png', 'name': name,
            'bytes': len(control['assets'][identifier]), 'width': 1, 'height': 1,
            'provenance': {'step': step, 'sample_sequence': step // 10, 'name': name}}
    image('g-10', 'g', 10, bytes([255, 0, 0]))
    image('g-30', 'g', 30, bytes([0, 0, 255]))
    image('g-20', 'g', 20, bytes([0, 255, 0]))
    image('x-30', 'x', 30, bytes([255, 255, 255]))
    control['artifacts']['numbers-30'] = {
        'role': 'sample', 'modality': 'tensor', 'media_type': 'application/json', 'name': 'g',
        'bytes': 32, 'shape': [1, 2], 'provenance': {'step': 30, 'sample_sequence': 3, 'name': 'g'}}
    login(page)
    items = page.locator('#artifact-items li')
    items.first.wait_for()
    assert items.count() == 2
    # One card per named picture; the tensor behind a grid is not a card of its own.
    assert [items.nth(i).get_attribute('data-sample') for i in range(2)] == ['g', 'x']
    assert [items.nth(i).get_attribute('data-modality') for i in range(2)] == ['image', 'image']
    assert page.locator('#artifact-items li[data-modality="tensor"]').count() == 0
    assert page.locator('#artifact-items img.image-grid').count() == 2
    generated = page.locator('#artifact-items li[data-sample="g"][data-modality="image"]')
    assert generated.locator('strong.sample-name').inner_text() == 'g'
    # Only the most recent generated image is shown by default.
    assert generated.locator('img').get_attribute('src').endswith('/artifacts/g-30')
    # The picture is named as a picture; its numbers are a secondary download.
    assert 'Image grid · Step 30' in generated.locator('span').first.inner_text()
    raw = generated.get_by_role('link', name='Download raw tensor (JSON, shape 1 × 2)', exact=True)
    assert raw.get_attribute('href').endswith('/artifacts/numbers-30')
    assert 'same sample as numbers' in generated.locator('.sample-note').inner_text()
    assert 'Version 3 of 3' in generated.locator('.sample-position').inner_text()
    assert 'step 30' in generated.locator('.sample-position').inner_text()
    assert 'latest' in generated.locator('.sample-position').inner_text()
    assert generated.get_by_role('button', name='Latest').is_hidden()
    # A single version needs no history control.
    assert page.locator('li[data-sample="x"] input[type="range"]').count() == 0
    slider = generated.locator('input[type="range"]')
    assert slider.get_attribute('aria-label') == 'g sample history'
    assert (slider.get_attribute('min'), slider.get_attribute('max')) == ('0', '2')
    # The slider scrubs back through earlier versions with the keyboard.
    slider.focus()
    page.keyboard.press('Home')
    generated.locator('.sample-position').filter(has_text='Version 1 of 3 · step 10').wait_for()
    generated.locator('img[src$="/artifacts/g-10"]').wait_for()
    assert generated.get_by_role('button', name='Latest').is_visible()
    # An older picture published no numbers of its own, so it offers no tensor.
    assert generated.get_by_role('link', name='Download raw tensor (JSON, shape 1 × 2)').count() == 0
    assert generated.locator('.sample-note').count() == 0
    page.keyboard.press('ArrowRight')
    generated.locator('.sample-position').filter(has_text='Version 2 of 3 · step 20').wait_for()
    generated.locator('img[src$="/artifacts/g-20"]').wait_for()
    assert 'Image grid · Step 20' in generated.locator('span').first.inner_text()
    generated.get_by_role('button', name='Latest').click()
    generated.locator('.sample-position').filter(has_text='Version 3 of 3').wait_for()
    generated.locator('img[src$="/artifacts/g-30"]').wait_for()
    assert raw.is_visible()
    page.wait_for_function(
        "() => [...document.querySelectorAll('#artifact-items img')].every(i => i.naturalWidth === 1)")
    assert not errors


def test_numerical_run_names_its_raw_tensors_and_final_sample(viewer):
    """A run without pictures keeps its tensors, said in words rather than 'sample · tensor'."""
    page, control, condition, errors = viewer
    def tensor(identifier, step, values, **extra):
        payload = json.dumps({'shape': [1, 2], 'samples': [values]}).encode()
        control['assets'][identifier] = payload
        control['artifacts'][identifier] = {
            'role': 'sample', 'modality': 'tensor', 'media_type': 'application/json', 'name': 'g',
            'bytes': len(payload), 'shape': [1, 2],
            'provenance': {'step': step, 'name': 'g', **extra}}
    tensor('numbers-10', 10, [1., 2.], sample_sequence=1)
    tensor('numbers-20', 20, [3., 4.], sample_sequence=2)
    # The final sample shares the last step; it is the newest version of it.
    tensor('final-sample', 20, [5., 6.], final=True)
    login(page)
    items = page.locator('#artifact-items li')
    items.first.wait_for()
    assert items.count() == 1 and items.get_attribute('data-modality') == 'tensor'
    assert page.locator('#artifact-items img').count() == 0
    card = page.locator('#artifact-items li[data-sample="g"]')
    details = card.locator('span').first
    assert 'Final sample · raw generator output (JSON numbers) · Step 20' in details.inner_text()
    assert 'Shape 1 × 2' in details.inner_text()
    assert 'saved when the run finished' in card.locator('.sample-note').inner_text()
    card.get_by_role('button', name='Preview numbers').click()
    card.locator('.numeric-preview').filter(has_text='5, 6').wait_for()
    # Earlier versions are the periodic previews, named for what they are.
    card.locator('input[type="range"]').focus()
    page.keyboard.press('Home')
    card.locator('.sample-position').filter(has_text='Version 1 of 3 · step 10').wait_for()
    assert 'Raw generator output (JSON numbers) · Step 10' in details.inner_text()
    note = card.locator('.sample-note').inner_text()
    assert 'no picture to draw from it' in note and '"Preview numbers" shows the first values' in note
    card.get_by_role('button', name='Preview numbers').click()
    card.locator('.numeric-preview').filter(has_text='1, 2').wait_for()


def test_sample_slider_follows_latest_and_holds_an_earlier_pick(viewer):
    """A new sample advances the slider only when it already showed the latest."""
    from hypergan.image_grids import encode_png
    page, control, condition, errors = viewer
    def image(step):
        identifier = f'g-{step}'
        control['assets'][identifier] = encode_png(bytes([step, 0, 0]), 1, 1, 3, {'step': step})
        control['artifacts'][identifier] = {
            'role': 'sample', 'modality': 'image', 'media_type': 'image/png', 'name': 'g',
            'bytes': len(control['assets'][identifier]), 'width': 1, 'height': 1,
            'provenance': {'step': step, 'sample_sequence': step // 10, 'name': 'g'}}
    def published():
        with condition:
            control['artifact_revision'] += 1
            condition.notify_all()
    for step in (10, 20, 30):
        image(step)
    login(page)
    generated = page.locator('#artifact-items li[data-sample="g"][data-modality="image"]')
    position = generated.locator('.sample-position')
    slider = generated.locator('input[type="range"]')
    position.filter(has_text='Version 3 of 3 \u00b7 step 30 \u00b7 latest').wait_for()
    # Showing the latest: a new sample advances to it and the range grows.
    image(40)
    published()
    position.filter(has_text='Version 4 of 4 \u00b7 step 40 \u00b7 latest').wait_for()
    generated.locator('img[src$="/artifacts/g-40"]').wait_for()
    assert slider.get_attribute('max') == '3'
    # Scrub back to an earlier sample.
    slider.focus()
    page.keyboard.press('Home')
    page.keyboard.press('ArrowRight')
    position.filter(has_text='Version 2 of 4 \u00b7 step 20').wait_for()
    # Slider positions shift when the oldest sample ages out and newer ones
    # arrive. The viewer holds the sample that was picked, not that position.
    control['artifacts'].pop('g-10')
    image(50)
    published()
    position.filter(has_text='Version 1 of 4 \u00b7 step 20').wait_for()
    generated.locator('img[src$="/artifacts/g-20"]').wait_for()
    assert (slider.get_attribute('min'), slider.get_attribute('max')) == ('0', '3')
    # Latest resumes following, so the next sample advances the slider again.
    generated.get_by_role('button', name='Latest').click()
    position.filter(has_text='Version 4 of 4 \u00b7 step 50 \u00b7 latest').wait_for()
    image(60)
    published()
    position.filter(has_text='Version 5 of 5 \u00b7 step 60 \u00b7 latest').wait_for()
    generated.locator('img[src$="/artifacts/g-60"]').wait_for()
    assert not errors


def test_sample_slider_keeps_a_picture_on_screen_while_the_next_loads(viewer):
    """Scrubbing never blanks the card: the shown picture stays until the next is decoded."""
    from hypergan.image_grids import encode_png
    page, control, condition, errors = viewer
    for step in range(10, 130, 10):
        identifier = f'g-{step}'
        control['assets'][identifier] = encode_png(bytes([step, 0, 0]), 1, 1, 3, {'step': step})
        control['artifacts'][identifier] = {
            'role': 'sample', 'modality': 'image', 'media_type': 'image/png', 'name': 'g',
            'bytes': len(control['assets'][identifier]), 'width': 1, 'height': 1,
            'provenance': {'step': step, 'sample_sequence': step // 10, 'name': 'g'}}
    # The oldest picture is slow to arrive and is not a neighbour of the latest.
    control['asset_delay'] = {'g-10': 1.0}
    login(page)
    generated = page.locator('#artifact-items li[data-sample="g"][data-modality="image"]')
    generated.locator('img[src$="/artifacts/g-120"]').wait_for()
    # Neighbours of the slider position are fetched ahead.
    page.wait_for_function("() => performance.getEntriesByType('resource')"
                           ".some(e => e.name.endsWith('/artifacts/g-80'))")
    assert not any(path.endswith('/artifacts/g-10') for path in control['paths'])
    page.evaluate("""() => {
        const body = document.querySelector('li[data-sample="g"] .sample-body');
        window.blanked = false;
        new MutationObserver(() => {
            if (!body.querySelector('img.image-grid')) window.blanked = true;
        }).observe(body, {childList: true, subtree: true});
    }""")
    slider = generated.locator('input[type="range"]')
    slider.focus()
    page.keyboard.press('Home')
    generated.locator('.sample-position').filter(has_text='Version 1 of 12 · step 10').wait_for()
    # Still loading: the previous picture stays and the card says it is busy.
    assert generated.locator('.sample-body').get_attribute('aria-busy') == 'true'
    assert generated.locator('img.image-grid').get_attribute('src').endswith('/artifacts/g-120')
    generated.locator('img[src$="/artifacts/g-10"]').wait_for()
    assert generated.locator('.sample-body').get_attribute('aria-busy') is None
    # Prefetched neighbours swap in at once; the card never went without a picture.
    page.keyboard.press('ArrowRight')
    generated.locator('img[src$="/artifacts/g-20"]').wait_for()
    assert page.evaluate('window.blanked') is False
    assert generated.locator('img.image-grid').count() == 1
    assert not errors


def test_headline_stats_report_throughput_training_time_and_samples(viewer):
    page,control,condition,errors=viewer;login(page)
    assert page.locator('#steps-per-second').inner_text()=='12.5'
    assert page.locator('#training-time').inner_text()=='1h 23m'
    assert page.locator('#samples-seen').inner_text()=='64'
    assert page.locator('#samples-batch').inner_text()=='32 per completed update'
    # A run that publishes no progress yet reads as unknown, never as zero.
    control['progress']=False
    page.reload()
    page.locator('#stream-position[data-projection="2"]').wait_for()
    assert [page.locator(f'#{tile}').inner_text() for tile in
            ('steps-per-second','training-time','samples-seen')]==['—','—','—']
    assert page.locator('#samples-batch').inner_text()=='Updates × global batch'
    assert not errors


def model_document(metrics=METRICS):
    from hypergan.config import resolve_config
    from hypergan.model_description import describe_run
    config = resolve_config({})
    rows = [{'id': f'n{i}', 'op': op, 'category': category, 'line': i + 2, 'out': {'out': ['B', 64]},
             'params': params, 'trainable_params': params, 'trainable': True}
            for i, (op, category, params) in enumerate([('linear', 'core', 320), ('leaky_relu', 'activation', 0)])]
    recorded = {'config_sha256': 'e' * 64, 'origin': 'recorded', 'components': {'generator': {
        'status': 'built', 'parameters': {'total': 320, 'trainable': 320, 'frozen': 0},
        'subgraphs': [{'module_path': 'network', 'node_count': 2, 'nodes': rows}]}}}
    model = describe_run({'config': config, 'config_sha256': 'e' * 64, 'run_id': RUN},
                         recorded=recorded, catalog={'metrics': dict.fromkeys(metrics, {})})
    return model


def test_inline_model_section_shows_formulation_losses_networks_and_plots_a_loss(viewer):
    page, control, condition, errors = viewer
    control['catalog'] = {
        **METRICS,
        'fid50k': {
            'label': 'FID 50k',
            'kind': 'scalar',
            'scope': 'snapshot',
            'definition_hash': DEFINITION,
            'specification': {'trigger': 'interval', 'every_steps': 5000},
        },
    }
    control['model'] = model_document(METRICS)
    login(page)

    # Tabs and duplicate "← All Runs" nav are gone; inline section jump bar is present.
    assert page.locator('nav.tabs, #tab-metrics, #tab-model, .nav-all-runs').count() == 0
    assert page.get_by_text('← All Runs').count() == 0
    for href in ('#learning-curves', '#evaluations', '#artifacts', '#model-section'):
        assert page.locator(f'#section-nav a[href="{href}"]').is_visible()

    # Inline model section loads eagerly on initial render without clicking any tab.
    page.locator('.model-network').first.wait_for()
    assert page.locator('#evaluations').is_visible()
    assert page.locator('#artifacts').is_visible()
    assert page.locator('#model-section').is_visible()

    # Evaluations, Samples & artifacts, and Model & configuration span full workspace width.
    catalog_box = page.locator('.catalog').bounding_box()
    charts_box = page.locator('#charts').bounding_box()
    min_full_width = catalog_box['width'] + charts_box['width'] * 0.9
    for section_sel in ('#evaluations', '#artifacts', '#model-section'):
        section_box = page.locator(section_sel).bounding_box()
        assert section_box['width'] > min_full_width

    text = page.locator('#model-content').inner_text()
    for expected in ('ParticleGAN', 'Discriminator (critic)', 'critic_regularizer', 'adversarial_g',
                     'hypergan/networks/reference.hndl', 'gaussian_grid'):
        assert expected in text
    generator = page.locator('.model-network').first
    generator.locator('.model-layers summary').click()
    assert generator.locator('.model-layers tbody tr').count() >= 2
    assert 'B × 64' in generator.locator('.model-layers').inner_text()
    generator.locator('.model-source summary').click()
    assert 'linear(64)' in generator.locator('.model-source code').inner_text()
    # The discriminator has no recorded detail and says why instead.
    assert 'Layer detail unavailable' in page.locator('.model-network').nth(1).inner_text()
    # A loss row plots its series in Learning curves; a series outside the catalog is plain text.
    assert page.locator('.model-table code', has_text='loss/d_adversarial').count() == 1
    page.locator('input[value="loss/gradient_penalty"]').uncheck()
    assert not page.locator('input[value="loss/gradient_penalty"]').is_checked()
    page.locator('.model-table button', has_text='loss/gradient_penalty').click()
    assert page.locator('input[value="loss/gradient_penalty"]').is_checked()
    page.wait_for_function("document.querySelector('.chart-canvas canvas').getBoundingClientRect().width > 200")
    assert not errors


def test_model_section_plots_a_loss_when_the_view_already_holds_eight_series(viewer):
    page, control, condition, errors = viewer
    extra = [
        'loss/prior_regularizer',
        'critic/accuracy',
        'diversity/generated_rms',
        'diversity/ratio',
        'diversity/pooled4_ratio',
        'throughput/steps_per_second',
        'loss/d_adversarial',
    ]
    control['catalog'] = {**METRICS, **{metric: metric for metric in extra}}
    control['model'] = model_document(control['catalog'])
    login(page)
    # Default selection holds 8 learning metrics, includes critic/accuracy, and excludes throughput.
    assert page.locator('#metric-list input:checked').count() == 8
    assert page.locator('input[value="critic/accuracy"]').is_checked()
    assert not page.locator('input[value="throughput/steps_per_second"]').is_checked()

    btn = page.locator('.model-table button', has_text='loss/d_adversarial')
    btn.wait_for()
    btn.click()
    assert page.locator('#learning-curves').is_visible()
    assert page.locator('input[value="loss/d_adversarial"]').is_checked()
    assert not page.locator('input[value="loss/g_total"]').is_checked()
    assert page.locator('#metric-list input:checked').count() == 8
    # The notice says what was replaced and outlives the view reload.
    notice = page.locator('#notice', has_text='in place of loss/g_total')
    notice.wait_for()
    page.wait_for_function("document.querySelector('#coverage').textContent === 'History loaded'")
    assert any('d_adversarial' in p for p in control['paths'] if '/bootstrap' in p)
    assert notice.is_visible()
    assert not errors


def test_model_section_missing_configuration_and_hash_deeplink(viewer):
    page, control, condition, errors = viewer
    login(page)
    page.locator('#model-status', has_text='No model configuration is recorded').wait_for()
    assert page.locator('#model-content').inner_text() == ''
    assert page.locator('.chart-canvas canvas').count() == 3

    # Navigating directly to /runs/{id}#model scrolls/targets #model-section without JS errors.
    control['model'] = model_document()
    origin = page.url.split('#')[0].split('?')[0].rstrip('/')
    page.goto(f'{origin}/runs/{RUN}#model')
    page.locator('#stream-position[data-projection="2"]').wait_for()
    page.locator('.model-network').first.wait_for()
    assert page.evaluate('location.hash') == '#model'
    page.wait_for_function("""() => {
      const top = document.getElementById('model-section').getBoundingClientRect().top;
      return window.scrollY > 0 && top < window.innerHeight * 0.6;
    }""")
    assert not errors


def test_summary_grid_5x2_active_gated_throughput_header_pills_and_health_alerts(viewer):
    page, control, condition, errors = viewer
    page.set_viewport_size({'width': 1280, 'height': 900})
    control['run_extra'] = {
        'active': True,
        'running': True,
        'global_batch_size': 256,
        'local_batch_size': 128,
        'batch_label': 'Batch 256',
        'batch_detail': '128 / GPU',
        'steps_per_second': 1.5,
        'samples_per_second': 384.0,
        'eta_seconds': 65.0,
        'samples_seen': 512,
        'best_fid': 14.25,
        'best_step': 2,
        'latest_fid': 15.80,
        'topology': {
            'device_kind': 'cuda',
            'world_size': 2,
            'nodes': 1,
            'ranks_per_node': 2,
            'gpu_count': 2,
            'gpu_name': 'RTX A6000',
            'label': '2 GPUs',
            'detail': '2× RTX A6000',
        },
        'health_alerts': ['Evaluator shares cuda:0 with active training rank 0 (VRAM contention risk)'],
    }
    login(page)

    # 10 summary cards arranged in 2 balanced rows of 5 on 1280x900 desktop.
    cards = page.locator('.stats article')
    assert cards.count() == 10
    tops = cards.evaluate_all('els => els.map(el => Math.round(el.getBoundingClientRect().top))')
    assert len(set(tops[:5])) == 1
    assert len(set(tops[5:])) == 1
    assert tops[5] > tops[0]

    assert page.locator('#run-topology-pill').is_visible()
    assert page.locator('#run-topology-pill').inner_text() == '2 GPUs · 2× RTX A6000'
    assert page.locator('#run-batch-pill').is_visible()
    assert page.locator('#run-batch-pill').inner_text() == 'Batch 256 · 128 / GPU'
    assert page.locator('#steps-per-second').inner_text() == '1.5'
    assert page.locator('#samples-per-second').inner_text() == '384.0'
    assert page.locator('#batch-size').inner_text() == '256'
    assert page.locator('#batch-topology').inner_text() == '2 GPUs · 2× RTX A6000 · 128 / GPU'
    assert page.locator('#best-fid').inner_text() == '14.25'
    assert 'Step 2' in page.locator('#fid-detail').inner_text()
    assert 'Latest 15.80' in page.locator('#fid-detail').inner_text()
    for action_id in ('#action-sample', '#action-checkpoint', '#action-stop'):
        assert page.locator(action_id).is_visible()

    assert page.locator('#health-alerts').is_visible()
    assert 'Evaluator shares cuda:0' in page.locator('#health-alerts-list').inner_text()
    page.locator('#dismiss-health-alerts').click()
    assert page.locator('#health-alerts').is_hidden()

    # Transition to stopped/inactive with historical throughput values retained.
    control['status'] = 'stopped'
    control['run_extra'].update(active=False, running=False)
    page.locator('#reconnect').click()
    page.locator('#run-status[data-status="stopped"]').wait_for()
    assert page.locator('#steps-per-second').inner_text() == '—'
    assert page.locator('#samples-per-second').inner_text() == '—'
    assert page.locator('#steps-rate-note').inner_text() == 'Last: 1.5 steps/s'
    assert page.locator('#samples-rate-note').inner_text() == 'Last: 384.0 samples/s'
    for action_id in ('#action-sample', '#action-checkpoint', '#action-stop'):
        assert page.locator(action_id).is_hidden()
    assert not errors


def test_two_click_confirm_stop_and_live_request_receipt_tracking(viewer):
    page, control, condition, errors = viewer
    control['requests'] = {'req-1': {'status': 'pending'}}
    control['stop_response'] = {
        'ok': True,
        'action': 'stop',
        'run_id': RUN,
        'request_id': 'req-1',
        'status': 'pending',
    }
    login(page)

    stop_btn = page.locator('#action-stop')
    cancel_btn = page.locator('#action-stop-cancel')
    assert stop_btn.is_visible() and stop_btn.inner_text() == 'Stop'
    assert cancel_btn.is_hidden()

    # First click arms without posting.
    stop_btn.click()
    assert control.get('posts', []) == []
    assert stop_btn.inner_text() == 'Confirm stop?'
    assert cancel_btn.is_visible()

    # Cancel disarms and resets to "Stop".
    cancel_btn.click()
    assert stop_btn.inner_text() == 'Stop'
    assert cancel_btn.is_hidden()
    assert control.get('posts', []) == []

    # Two clicks confirm, post /stop, poll /requests/req-1, and announce completion.
    stop_btn.click()
    assert stop_btn.inner_text() == 'Confirm stop?'
    stop_btn.click()
    page.locator('#notice', has_text='Stop requested at next update boundary').wait_for()
    assert control['posts'] == [f'/api/v1/runs/{RUN}/stop']

    deadline = time.monotonic() + 5
    while 'req-1' not in control.get('request_polls', []):
        assert time.monotonic() < deadline, 'Did not poll /requests/req-1'
        time.sleep(0.05)

    control['status'] = 'stopped'
    control['run_extra'] = {'active': False, 'running': False}
    control['requests']['req-1'] = {'status': 'completed', 'completed': True, 'step': 2}
    page.locator('#notice', has_text='Stop completed at step 2.').wait_for(timeout=5000)
    page.locator('#run-status[data-status="stopped"]').wait_for(timeout=5000)
    assert stop_btn.is_hidden()
    assert not errors


def test_samples_workbench_side_by_side_synced_sliders_pixel_toggle_and_lightbox(viewer):
    from hypergan.image_grids import encode_png
    page, control, condition, errors = viewer
    page.set_viewport_size({'width': 1280, 'height': 900})

    def image(identifier, name, step, pixel):
        control['assets'][identifier] = encode_png(pixel, 1, 1, 3, {'step': step})
        control['artifacts'][identifier] = {
            'role': 'sample',
            'modality': 'image',
            'media_type': 'image/png',
            'name': name,
            'bytes': len(control['assets'][identifier]),
            'width': 1,
            'height': 1,
            'provenance': {'step': step, 'sample_sequence': step // 10, 'name': name},
        }

    image('g-10', 'g', 10, bytes([255, 0, 0]))
    image('g-20', 'g', 20, bytes([0, 255, 0]))
    image('x-10', 'x', 10, bytes([0, 0, 255]))
    image('x-20', 'x', 20, bytes([255, 255, 0]))
    login(page)

    g_card = page.locator('#artifact-items li[data-sample="g"]')
    x_card = page.locator('#artifact-items li[data-sample="x"]')
    g_card.locator('img[src$="/artifacts/g-20"]').wait_for()
    x_card.locator('img[src$="/artifacts/x-20"]').wait_for()

    # Side-by-side layout on 1280x900 desktop (same top Y coordinate).
    g_box = g_card.bounding_box()
    x_box = x_card.bounding_box()
    assert abs(g_box['y'] - x_box['y']) <= 2
    assert x_box['x'] > g_box['x']

    # Dragging/stepping g's slider to step 10 syncs x's slider to step 10 when #sync-samples is checked.
    assert page.locator('#sync-samples').is_checked()
    g_slider = g_card.locator('input[type="range"]')
    x_slider = x_card.locator('input[type="range"]')
    g_slider.fill('0')
    g_card.locator('.sample-position').filter(has_text='Version 1 of 2 · step 10').wait_for()
    x_card.locator('.sample-position').filter(has_text='Version 1 of 2 · step 10').wait_for()
    assert x_slider.input_value() == '0'
    g_card.locator('img[src$="/artifacts/g-10"]').wait_for()
    x_card.locator('img[src$="/artifacts/x-10"]').wait_for()

    # Pixel rendering toggle switches between Pixels: Crisp and Pixels: Smooth.
    pixel_btn = page.locator('#toggle-pixelated')
    assert pixel_btn.inner_text() == 'Pixels: Crisp'
    assert pixel_btn.get_attribute('aria-pressed') == 'true'
    assert g_card.locator('img.image-grid').evaluate('el => getComputedStyle(el).imageRendering') == 'pixelated'
    pixel_btn.click()
    assert pixel_btn.inner_text() == 'Pixels: Smooth'
    assert pixel_btn.get_attribute('aria-pressed') == 'false'
    assert g_card.locator('img.image-grid').evaluate('el => getComputedStyle(el).imageRendering') == 'auto'
    pixel_btn.click()
    assert pixel_btn.inner_text() == 'Pixels: Crisp'

    # Clicking sample image opens <dialog id="sample-lightbox">; ArrowRight/ArrowLeft scrubs, Escape closes.
    g_card.locator('img.image-grid').click()
    lightbox = page.locator('#sample-lightbox')
    assert lightbox.is_visible()
    assert lightbox.evaluate('d => d.open') is True
    assert page.locator('#lightbox-name').inner_text() == 'g'
    assert 'Step 10' in page.locator('#lightbox-caption').inner_text()
    assert page.locator('#lightbox-image').get_attribute('src').endswith('/artifacts/g-10')

    page.keyboard.press('ArrowRight')
    page.locator('#lightbox-caption').filter(has_text='Step 20').wait_for()
    assert page.locator('#lightbox-image').get_attribute('src').endswith('/artifacts/g-20')

    page.keyboard.press('ArrowLeft')
    page.locator('#lightbox-caption').filter(has_text='Step 10').wait_for()
    assert page.locator('#lightbox-image').get_attribute('src').endswith('/artifacts/g-10')

    page.keyboard.press('Escape')
    assert lightbox.evaluate('d => d.open') is False
    assert not errors

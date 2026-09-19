"""Dedicated installed-package browser gate; missing Chromium is a failure, not a skip."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest
from playwright.sync_api import sync_playwright

from hypergan.metrics_reducer import Reducer, append_frame, assets, bootstrap


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    root = tmp_path_factory.mktemp("reducer-browser")
    for resource in assets().iterdir():
        if resource.is_file():
            (root / resource.name).write_bytes(resource.read_bytes())
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1",0), partial(QuietHandler,directory=str(root)))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/proof.html")
            page.wait_for_function("typeof window.runProof === 'function'")
            yield page
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_python_bootstrap_browser_worker_live_suffix(page):
    reducer = Reducer()
    values = [{"value":v,"position":[i,f"source-{i}"]} for i,v in enumerate([None,1,5,-4,12,3,12,0])]
    cases = {}
    for kind in ["mean/v1","envelope/v1"]:
        view = bootstrap(reducer,kind,"run/generation/map/view/metric/bucket")
        view = append_frame(reducer,view,identity=view["identity"],start=0,end=5,values=values[:5])
        frames = [{"identity":view["identity"],"start":5,"end":8,"values":values[5:]}]
        # Duplicate replay after application must not double count.
        frames.append(frames[0])
        cases[kind] = {"bootstrap":view,"frames":frames,
                       "expected":reducer.finalize(reducer.add(reducer.identity(kind),values))}
    results = page.evaluate("payload => window.runProof(payload)", {"cases":cases})
    for kind,result in results.items():
        assert result["value"] == cases[kind]["expected"]
        assert result["module_sha256"] == reducer.module_sha256
        assert result["bootstrap"]["end"] == 8
        assert result["memory_bytes"] <= 16*1024*1024


def test_browser_worker_rejects_bad_data_and_recovers(page):
    result = page.evaluate("""async () => {
      const {ReducerWorker} = await import('./client.js');
      const worker = new ReducerWorker();
      const errors = [];
      try {
        const state = await worker.call({op:'request',request:{op:'identity',reducer:'mean/v1'}});
        for (const value of [NaN,Infinity,undefined]) {
          try { await worker.call({op:'request',request:{op:'add',state,values:[{value,position:[0,'x']}]}}); }
          catch(error) { errors.push(error.message); }
        }
        try { await worker.call({op:'request',request:{op:'add',state,values:[{value:1,position:[9007199254740992,'x']}]}}); }
        catch(error) { errors.push(error.message); }
        const final = await worker.call({op:'request',request:{op:'finalize',state}});
        return {errors,final};
      } finally {worker.close();}
    }""")
    assert len(result["errors"]) == 4
    assert result["final"] == {"count":0,"value":None}


def test_browser_coverage_and_shared_module_mismatch(page):
    reducer = Reducer()
    view = bootstrap(reducer,"mean/v1","scope")
    result = page.evaluate("""async view => {
      const {ReducerWorker} = await import('./client.js');
      const worker = new ReducerWorker(); const errors=[];
      try {
        for (const changed of [{...view,module_sha256:'wrong'},{...view,end:1,state:{...view.state,count:1,sum:2}}]) {
          try { await worker.call({op:'continue',bootstrap:changed,frames:[{identity:'scope',start:2,end:3,values:[]}]}); }
          catch(error) { errors.push(error.message); }
        }
        try { await worker.call({op:'merge',left:{...view,end:2},right:{...view,start:1,end:3}}); }
        catch(error) { errors.push(error.message); }
        return errors;
      } finally {worker.close();}
    }""",view)
    assert len(result) == 3
    assert "digest" in result[0]
    assert "gap" in result[1]
    assert "disjoint" in result[2]


def test_worker_deadline_terminates_without_blocking_page(page):
    result = page.evaluate("""async () => {
      const {ReducerWorker} = await import('./client.js');
      const url = URL.createObjectURL(new Blob(['while(true) {}'],{type:'text/javascript'}));
      const worker = new ReducerWorker({timeoutMs:50,workerURL:url});
      let ticks=0; const timer=setInterval(()=>ticks++,5);
      try { await worker.call({op:'request'}); return {error:'unexpected success'}; }
      catch(error) {return {error:error.message,ticks,closed:worker.closed};}
      finally {clearInterval(timer);worker.close();URL.revokeObjectURL(url);}
    }""")
    assert "deadline" in result["error"]
    assert result["closed"] is True and result["ticks"] > 0

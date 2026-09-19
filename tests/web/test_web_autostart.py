"""Real spawned lifecycle, isolated producer, and preflight behavior."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from hypergan.event_views import MapSpec, read_projection_page
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_autostart import Viewer, training_viewer
from hypergan.web_launch import bind_loopback


def until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError('condition timed out')


def make_run(root):
    root.mkdir()
    definition = dict(kind='scalar', source='g_loss', label='Generator')
    definition['definition_hash'] = digest(definition)
    catalog = dict(schema_version=1, metrics={'loss/g_total': definition})
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    rows = [dict(schema_version=2, run_id='r', attempt_id='a', stream_id='training',
                 stream_generation='r', sequence=i+1, step=i, catalog=revision,
                 event='start' if i == 0 else 'train',
                 **({'parent_attempt_id': None, 'restored_step': 0} if i == 0 else {'metrics': {'loss/g_total': float(i)}}))
            for i in range(251)]
    (root / 'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    atomic_json(root / 'manifest.json', dict(schema_version=1, run_id='r', attempt_id='a',
                                          steps=250, status='running', metrics_catalog=revision))


def test_pending_start_does_not_create_run_then_separate_producer_catches_up(tmp_path):
    root = tmp_path / 'run'
    with training_viewer(root, required=True) as viewer:
        assert not root.exists()
        assert viewer.info['processes']['server'] != viewer.info['processes']['projector']
        assert len({os.getpid(), viewer.process.pid, *viewer.info['processes'].values()}) == 4
        credentials = json.loads(viewer.credential_path.read_text())
        assert credentials['origin'] == viewer.session.origin
        make_run(root)
        def projected():
            try:
                page = read_projection_page(root, MapSpec().revision, limit=1000)
                return page if len(page['frames']) == 251 else None
            except (OSError, ValueError):
                return None
        page = until(projected)
        assert page['frames'][-1]['projection_sequence'] == 251
        until(lambda: list((root / 'observations').glob('viewer-*.json')))
    assert not viewer.process.is_alive()
    assert not viewer.credential_path.exists()
    receipt = json.loads(next((root / 'observations').glob('viewer-*.json')).read_text())
    assert receipt['status'] == 'stopped'
    assert credentials['token'] not in json.dumps(receipt)
    with socket.socket() as client:
        assert client.connect_ex(('127.0.0.1', int(viewer.session.host.rsplit(':', 1)[1]))) != 0


def test_default_does_not_wait_and_optional_bind_failure_is_independent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Viewer, 'wait_ready', lambda *args: pytest.fail('automatic startup waited'))
    with training_viewer(tmp_path / 'pending'):
        pass
    with bind_loopback() as occupied:
        port = occupied.getsockname()[1]
        with training_viewer(tmp_path / 'pending', port=port) as viewer:
            assert viewer is None
        with pytest.raises(OSError):
            with training_viewer(tmp_path / 'pending', required=True, port=port):
                pytest.fail('numerical work started after failed preflight')
    assert 'training continues headless' in capsys.readouterr().err


def test_viewer_crash_does_not_fail_training_and_exception_cleans_up(tmp_path):
    with pytest.raises(ValueError, match='numerical failure'):
        with training_viewer(tmp_path / 'pending', required=True) as viewer:
            os.kill(viewer.info['processes']['server'], signal.SIGTERM)
            until(lambda: not __import__('hypergan.web_autostart', fromlist=['_probe'])._probe(viewer.session))
            assert viewer.process.is_alive()
            raise ValueError('numerical failure')
    assert not viewer.process.is_alive()
    assert not viewer.credential_path.exists()


def test_killed_parent_cleans_up_server_and_credentials(tmp_path):
    script = tmp_path / 'parent.py'
    info = tmp_path / 'info.json'
    script.write_text('''from hypergan.web_autostart import training_viewer
import json, time
from pathlib import Path
if __name__ == '__main__':
    with training_viewer(Path(__file__).parent / 'pending', required=True) as viewer:
        Path(__file__).with_name('info.json').write_text(json.dumps(viewer.info))
        time.sleep(60)
''')
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        until(info.exists)
        record = json.loads(info.read_text())
        process.kill()
        process.wait(5)
        until(lambda: not Path(record['session_file']).exists())
        port = int(record['origin'].rsplit(':', 1)[1])
        def disconnected():
            with socket.socket() as client:
                return client.connect_ex(('127.0.0.1', port)) != 0
        until(disconnected)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
        process.stderr.close()


def test_cli_explicit_bind_preflight_before_numerical_import_or_run_creation(tmp_path, monkeypatch, capsys):
    import builtins
    from hypergan.cli import main
    from hypergan.config import write_default
    config = write_default(tmp_path / 'project', device='cpu')
    root = tmp_path / 'run'
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in ('training', 'hypergan.training', 'torch'), 'numerical import before preflight'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with bind_loopback() as occupied:
        assert main(['train', str(config), '--run-dir', str(root), '--server-port',
                     str(occupied.getsockname()[1])]) == 1
    assert not root.exists()
    assert capsys.readouterr().out == ''


_STOP_OWNER_DEATH = '''
from hypergan import web_autostart as web
from pathlib import Path
import faulthandler,json,os,signal,sys,time

real_broker,real_project=web._broker,web._project

def interrupted_server(root,listener,session,stop):
    # Kill inside the old Event's condition critical section. Lock-free stop
    # flags have no cross-process critical section to interrupt.
    condition=getattr(stop,'_cond',None)
    if condition is not None:
        condition.acquire()
    (root.parent/'server.pid').write_text(str(os.getpid()))
    time.sleep(60)

def recorded_project(root,stop):
    (root.parent/'projector.pid').write_text(str(os.getpid()))
    real_project(root,stop)

def fault_broker(*args):
    web._server,web._project=interrupted_server,recorded_project
    real_broker(*args)

if __name__ == '__main__':
    faulthandler.enable()
    faulthandler.dump_traceback_later(10,repeat=True)
    web._broker=fault_broker
    directory=Path(sys.argv[1])
    viewer=web.Viewer(directory/'pending')
    (directory/'credential-path.json').write_text(json.dumps(str(viewer.credential_path)))
    deadline=time.monotonic()+10
    while not all((directory/name).exists() for name in ('server.pid','projector.pid')):
        if time.monotonic()>deadline:
            raise RuntimeError('worker PID receipt timed out')
        time.sleep(.02)
    pids=[viewer.process.pid,*[int((directory/name).read_text()) for name in ('server.pid','projector.pid')]]
    (directory/'pids.json').write_text(json.dumps(pids))
    os.kill(pids[1],signal.SIGTERM)
    started=time.monotonic()
    viewer.close()
    (directory/'closed.json').write_text(json.dumps({'seconds':time.monotonic()-started}))
'''


def _process_running(pid):
    if sys.platform == 'win32':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    return not (stat.exists() and stat.read_text().split()[2] == 'Z')


def test_killed_stop_owner_cannot_strand_broker_or_projector(tmp_path):
    script = tmp_path / 'stop_owner.py'
    script.write_text(_STOP_OWNER_DEATH)
    with (tmp_path / 'parent.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, str(script), str(tmp_path)], stdout=log, stderr=log)
        passed = False
        try:
            assert process.wait(timeout=18) == 0
            pids = json.loads((tmp_path / 'pids.json').read_text())
            until(lambda: not any(_process_running(pid) for pid in pids), timeout=3)
            assert json.loads((tmp_path / 'closed.json').read_text())['seconds'] < 5
            credential = json.loads((tmp_path / 'credential-path.json').read_text())
            assert not Path(credential).exists()
            passed = True
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(5)
            # Preserve failed assertions while cleaning deliberate pre-fix leaks.
            receipt = tmp_path / 'pids.json'
            if receipt.exists():
                for pid in json.loads(receipt.read_text()):
                    if _process_running(pid):
                        os.kill(pid, signal.SIGTERM if sys.platform == 'win32' else signal.SIGKILL)
            credentials = tmp_path / 'credential-path.json'
            if credentials.exists():
                path = Path(json.loads(credentials.read_text()))
                path.unlink(missing_ok=True)
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
            if not passed:
                print((tmp_path / 'parent.log').read_text(), file=sys.stderr)

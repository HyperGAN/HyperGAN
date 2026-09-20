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


def test_pending_start_survives_context_exit_then_reuses_and_stops(tmp_path):
    from hypergan.web_autostart import _probe, stop_viewer
    root = tmp_path / 'run'
    try:
        with training_viewer(root, required=True, auth="token") as viewer:
            assert not root.exists()
            assert viewer.info['processes']['server'] != viewer.info['processes']['projector']
            assert len({os.getpid(), viewer.info['supervisor_pid'], *viewer.info['processes'].values()}) == 4
            credentials = json.loads(viewer.credential_path.read_text())
            make_run(root)
            def projected():
                try:
                    page = read_projection_page(root, MapSpec().revision, limit=1000)
                    return page if len(page['frames']) == 251 else None
                except (OSError, ValueError):
                    return None
            assert until(projected)['frames'][-1]['projection_sequence'] == 251
            until(lambda: list((root / 'observations').glob('viewer-*.json')))
        assert _probe(viewer.session)
        with training_viewer(root, required=True) as resumed:
            assert resumed.info['server_instance_id'] == viewer.info['server_instance_id']
            assert resumed.info['supervisor_pid'] == viewer.info['supervisor_pid']
            assert resumed.session.auth_mode == 'token'
        assert stop_viewer(root)['status'] == 'stopped'
        assert stop_viewer(root)['status'] == 'not_running'
        assert not viewer.credential_path.exists()
        receipt = json.loads(next((root / 'observations').glob('viewer-*.json')).read_text())
        assert receipt['status'] == 'stopped'
        assert credentials['token'] not in json.dumps(receipt)
        assert not _probe(viewer.session)
    finally:
        stop_viewer(root)


def test_default_does_not_wait_and_optional_bind_failure_is_independent(tmp_path, monkeypatch, capsys):
    from hypergan.web_autostart import stop_viewer
    monkeypatch.setattr(Viewer, 'wait_ready', lambda *args: pytest.fail('automatic startup waited'))
    root = tmp_path / 'pending'
    try:
        with training_viewer(root):
            pass
    finally:
        stop_viewer(root)
    with bind_loopback() as occupied:
        port = occupied.getsockname()[1]
        with training_viewer(root, port=port) as viewer:
            assert viewer is None
        with pytest.raises(OSError):
            with training_viewer(root, required=True, port=port):
                pytest.fail('numerical work started after failed preflight')
    assert 'training continues headless' in capsys.readouterr().err


def test_numerical_failure_keeps_viewer_and_viewer_failure_does_not_fail_training(tmp_path):
    from hypergan.web_autostart import _probe, stop_viewer
    root = tmp_path / 'pending'
    try:
        with pytest.raises(ValueError, match='numerical failure'):
            with training_viewer(root, required=True) as viewer:
                raise ValueError('numerical failure')
        assert _probe(viewer.session)
        with training_viewer(root, required=True) as reused:
            os.kill(reused.info['processes']['server'], signal.SIGTERM)
            until(lambda: reused.failed.is_set())
            # This remains an observation failure, independent of numerical work.
            assert not _probe(reused.session)
        until(lambda: not viewer.credential_path.exists())
    finally:
        stop_viewer(root)


@pytest.mark.parametrize('ending', ['normal', 'term', 'kill'])
def test_parent_exit_or_signal_leaves_server_available(tmp_path, ending):
    from hypergan.web_autostart import _probe, _session, stop_viewer
    script = tmp_path / 'parent.py'
    info = tmp_path / 'info.json'
    root = tmp_path / 'pending'
    script.write_text("""from hypergan.web_autostart import training_viewer
import json, time, sys
from pathlib import Path
if __name__ == '__main__':
    with training_viewer(Path(__file__).parent / 'pending', required=True) as viewer:
        Path(__file__).with_name('info.json').write_text(json.dumps(viewer.info))
        if sys.argv[1] != 'normal':
            time.sleep(60)
""")
    process = subprocess.Popen([sys.executable, str(script), ending], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        until(info.exists)
        record = json.loads(info.read_text())
        if ending == 'term':
            process.terminate()
        elif ending == 'kill':
            process.kill()
        process.wait(5)
        assert _probe(_session(record))
        # A fresh CLI owns shutdown; no launcher process needs to remain alive.
        result = subprocess.run([sys.executable, '-m', 'hypergan', 'stop-server', str(root)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['status'] == 'stopped'
        assert not Path(record['session_file']).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
        process.stderr.close()
        stop_viewer(root)


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


def test_broker_death_cleans_children_and_stale_registry_can_restart(tmp_path):
    from hypergan.web_autostart import _probe, stop_viewer
    root = tmp_path / 'pending'
    viewer = Viewer(root)
    try:
        viewer.wait_ready()
        children = list(viewer.info['processes'].values())
        viewer.process.kill()
        viewer.process.wait(3)
        until(lambda: not any(_process_running(pid) for pid in children))
        assert not _probe(viewer.session)
        with training_viewer(root, required=True) as restarted:
            assert restarted.info['server_instance_id'] != viewer.info['server_instance_id']
            assert _probe(restarted.session)
    finally:
        viewer.detach()
        stop_viewer(root)


def test_reuse_checks_explicit_settings_and_concurrent_attach(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hypergan.web_autostart import stop_viewer
    root = tmp_path / 'pending'
    viewers = []
    try:
        def attach(_):
            viewer = Viewer(root, host='127.0.0.1', auth='token')
            viewer.wait_ready()
            return viewer
        with ThreadPoolExecutor(max_workers=3) as workers:
            viewers = list(workers.map(attach, range(3)))
        assert len({viewer.info['server_instance_id'] for viewer in viewers}) == 1
        assert len({viewer.info['supervisor_pid'] for viewer in viewers}) == 1
        for options in [dict(auth='none'), dict(host='0.0.0.0'), dict(port=1)]:
            with pytest.raises(ValueError, match='different bind/auth'):
                Viewer(root, **options)
    finally:
        for viewer in viewers:
            viewer.detach()
        stop_viewer(root)


def test_stop_is_bounded_when_child_cannot_cooperate(tmp_path):
    from hypergan.web_autostart import stop_viewer
    viewer = Viewer(tmp_path / 'pending')
    try:
        viewer.wait_ready()
        children = list(viewer.info['processes'].values())
        # Windows has no SIGSTOP; a killed child exercises the failed-service
        # cleanup path there, while POSIX forces escalation of a stuck child.
        os.kill(viewer.info['processes']['projector'],
                signal.SIGTERM if os.name == 'nt' else signal.SIGSTOP)
        started = time.monotonic()
        viewer.close()
        assert time.monotonic() - started < 7
        until(lambda: not any(_process_running(pid) for pid in children))
    finally:
        viewer.detach()
        stop_viewer(tmp_path / 'pending')


def test_short_optional_attempt_prints_discovery_and_startup_can_be_cancelled(tmp_path, capsys):
    from hypergan.web_autostart import _registry, stop_viewer, viewer_status
    root = tmp_path / 'pending'
    try:
        with training_viewer(root):
            pass
        assert 'hypergan server-status' in capsys.readouterr().err
        until(lambda: viewer_status(root)['status'] == 'ready')
        result = subprocess.run([sys.executable, '-m', 'hypergan', 'server-status', str(root)],
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['origin'].startswith('http://127.0.0.1:')
        assert 'token' not in json.loads(result.stdout)
    finally:
        stop_viewer(root)
    # A reservation whose launcher died before broker lock acquisition has no
    # service to stop. Cancel it immediately, with no 15-second startup wait.
    private = _registry(tmp_path / 'unstarted')
    atomic_json(private / 'state.json', dict(launch_id='unstarted', status='starting',
                                           started_at=time.time()))
    started = time.monotonic()
    assert stop_viewer(tmp_path / 'unstarted')['status'] == 'stopped'
    assert time.monotonic() - started < 1

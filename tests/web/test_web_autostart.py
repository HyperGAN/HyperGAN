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


def test_automatic_viewer_keeps_fixed_port_and_shares_dashboard(tmp_path):
    """The root dashboard uses fixed port 8765, and concurrent runs under root share it."""
    from hypergan.ports import DEFAULT_VIEWER_PORT
    from hypergan.web_autostart import stop_viewer
    first, second = tmp_path / 'first', tmp_path / 'second'

    def port_of(viewer):
        return int(viewer.session.origin.rsplit(':', 1)[1])

    try:
        with training_viewer(first, required=True) as viewer:
            chosen = port_of(viewer)
        assert chosen == DEFAULT_VIEWER_PORT
        assert stop_viewer(first)['status'] == 'stopped'
        # A restarted run returns to the fixed port.
        with training_viewer(first, required=True) as restarted:
            assert port_of(restarted) == DEFAULT_VIEWER_PORT
            assert restarted.info['origin'].endswith(f':{DEFAULT_VIEWER_PORT}')
            # A concurrent run under the same runs root shares that dashboard on port 8765.
            with training_viewer(second, required=True) as concurrent:
                other = port_of(concurrent)
                assert other == DEFAULT_VIEWER_PORT
    finally:
        stop_viewer(first)
        stop_viewer(second)


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
    except BaseException:
        from hypergan.web_autostart import _registry
        log = _registry(root) / 'viewer.log'
        if log.exists():
            print(log.read_text(errors='replace'), file=sys.stderr)
        raise
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
        assert main(['train', str(config), '--run-dir', str(root), '--port',
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


def test_failed_terminal_state_publication_still_reaps_and_removes_credentials(tmp_path):
    from hypergan.web_autostart import _read_state, _registry, _locked
    root = tmp_path / 'pending'
    private = _registry(root)
    launch_id = 'terminal-write-fault'
    atomic_json(private / 'state.json', dict(schema_version=1, launch_id=launch_id, root=str(root),
        host='127.0.0.1', auth='token', port=0, mode='explicit', status='starting', started_at=time.time()))
    script = tmp_path / 'broker.py'
    script.write_text('''from pathlib import Path
import sys
from hypergan import run_state, web_autostart
if __name__ == '__main__':
    original = run_state.atomic_json
    failures = [0]
    def fail_terminal(path, value):
        if Path(path).name == 'state.json' and value.get('status') == 'failed':
            failures[0] += 1
            if failures[0] == 2:
                raise PermissionError('injected terminal registry replace failure')
        return original(path, value)
    run_state.atomic_json = fail_terminal
    web_autostart._broker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
''')
    with (tmp_path / 'broker.log').open('wb') as output:
        process = subprocess.Popen([sys.executable, str(script), str(root), str(private), launch_id],
                                   stdout=output, stderr=output)
        record = None
        try:
            record = until(lambda: (state if (state := _read_state(private))['status'] == 'ready' else None))
            os.kill(record['processes']['server'], signal.SIGTERM)
            assert process.wait(timeout=8) != 0
            assert 'injected terminal registry replace failure' in (tmp_path / 'broker.log').read_text()
            assert not Path(record['session_file']).exists()
            assert not _locked(private)
            until(lambda: not any(_process_running(pid) for pid in record['processes'].values()))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if record is not None:
                until(lambda: not any(_process_running(pid) for pid in record['processes'].values()))
            credential = private / ('session-' + launch_id + '.json')
            credential.unlink(missing_ok=True)


def test_credential_cleanup_waits_for_own_active_reader(tmp_path, monkeypatch):
    import threading
    from hypergan.web_autostart import _cleanup_session, _registry, _session
    from hypergan.web_session import LocalSession
    private = _registry(tmp_path / 'pending')
    launch_id = 'credential-read-race'
    session = LocalSession(8123, auth='token')
    credential = session.write_credentials(private / ('session-' + launch_id + '.json'))
    state = dict(session_file=str(credential), server_instance_id=session.instance_id)
    opened, release, cleanup_started, cleaned = [threading.Event() for _ in range(4)]
    failures = []
    read_text = Path.read_text
    def held_read(path, *args, **kwargs):
        if path != credential:
            return read_text(path, *args, **kwargs)
        with path.open(encoding='utf-8') as stream:
            opened.set()
            assert release.wait(3), 'test did not release credential reader'
            return stream.read()
    monkeypatch.setattr(Path, 'read_text', held_read)
    def read():
        try:
            assert _session(state).instance_id == session.instance_id
        except BaseException as exc:
            failures.append(exc)
    def cleanup():
        cleanup_started.set()
        try:
            _cleanup_session(private, launch_id)
            cleaned.set()
        except BaseException as exc:
            failures.append(exc)
    reader, remover = threading.Thread(target=read), threading.Thread(target=cleanup)
    reader.start()
    try:
        assert opened.wait(3)
        remover.start()
        assert cleanup_started.wait(3)
        assert not cleaned.wait(.1), 'cleanup passed an active credential reader'
        assert credential.exists()
    finally:
        release.set()
        reader.join(3)
        if remover.ident is not None:
            remover.join(3)
        credential.unlink(missing_ok=True)
    assert not reader.is_alive() and not remover.is_alive()
    assert not failures, failures
    assert cleaned.is_set()


def test_get_port_owner_detects_listening_process():
    from hypergan.web_autostart import get_port_owner
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        owner = get_port_owner(port)
        assert owner is not None
        pid, comm = owner
        assert pid == os.getpid()
    finally:
        listener.close()


def test_bind_available_raises_actionable_error_on_conflict():
    from hypergan.web_launch import bind_available
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("0.0.0.0", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with pytest.raises(OSError, match=f"Port {port} is already in use by PID {os.getpid()}"):
            bind_available(port)
    finally:
        listener.close()


def test_ensure_dashboard_running_auto_restart_on_version_mismatch(tmp_path, monkeypatch):
    from hypergan.web_autostart import ensure_dashboard_running
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    probed_state = {
        "version": "0.1.0-outdated",
        "asset_hash": "old_hash",
        "runs_root": str(runs_root.resolve()),
        "status": "ready",
    }
    stop_called = []
    launch_called = []

    monkeypatch.setattr("hypergan.web_autostart.probe_dashboard", lambda *args, **kwargs: probed_state)
    monkeypatch.setattr("hypergan.web_autostart.stop_dashboard", lambda root: stop_called.append(root))
    monkeypatch.setattr("hypergan.web_autostart.launch_dashboard_daemon", lambda *args, **kwargs: launch_called.append(args))

    # The probe after launch should return the new version so loop terminates
    probe_count = 0
    def mock_probe(*args, **kwargs):
        nonlocal probe_count
        probe_count += 1
        if probe_count > 1:
            return {"version": "2.0.0a1", "status": "ready"}
        return probed_state

    monkeypatch.setattr("hypergan.web_autostart.probe_dashboard", mock_probe)

    url = ensure_dashboard_running(runs_root, check_version=True, run_id="my_run")
    assert stop_called == [runs_root.resolve()]
    assert len(launch_called) == 1
    assert "8765/runs/my_run" in url


def test_projector_clean_shutdown_on_terminal_manifest(tmp_path):
    from hypergan.web_autostart import _project, _StopFlag
    run_dir = tmp_path / "run_complete"
    run_dir.mkdir()
    manifest = {
        'schema_version': 1,
        'run_id': 'run_complete',
        'status': 'complete',
        'metrics_catalog': 'rev1',
    }
    atomic_json(run_dir / 'manifest.json', manifest)

    import multiprocessing as mp
    context = mp.get_context('spawn')
    stop = _StopFlag(context)

    # _project should notice manifest status is 'complete' and exit cleanly
    start = time.monotonic()
    _project(run_dir, stop)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, "Projector did not terminate within timeout on terminal status"


def test_stop_projector_and_stop_all_servers(tmp_path):
    from hypergan.web_autostart import stop_projector, stop_all_servers
    run_dir = tmp_path / "run_proj"
    run_dir.mkdir()
    # Should not raise even if no projector is running
    stop_projector(run_dir)

    # stop_all_servers sweeps and returns list of PIDs
    killed = stop_all_servers()
    assert isinstance(killed, list)


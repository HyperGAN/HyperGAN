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
    config = write_default(tmp_path / 'project', device='cpu', defaults='hypergan')
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
    """Only this root's own registered dashboard is restarted, and its credential is used to probe."""
    import hypergan.web_autostart as startup
    from hypergan.web_autostart import ensure_dashboard_running
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    probed_state = {
        "version": "0.1.0-outdated",
        "asset_hash": "old_hash",
        "runs_root": str(runs_root.resolve()),
        "server_instance_id": "old-instance",
        "status": "ready",
    }
    registered = {"launch_id": "L1", "server_instance_id": "old-instance", "port": 8765}
    stopped = []
    launched = []
    tokens = []
    monkeypatch.setattr(startup, "_registered_dashboard",
                        lambda root: (registered, "secret") if not launched else (dict(registered, server_instance_id="new"), "fresh"))
    monkeypatch.setattr(startup, "_stop_registered_dashboard", lambda root, state: stopped.append((root, state["launch_id"])))
    monkeypatch.setattr(startup, "launch_dashboard_daemon", lambda *args, **kwargs: launched.append(args))

    def mock_probe(host, port, auth_token=None, **kwargs):
        tokens.append(auth_token)
        if launched:
            return {"version": "2.0.0a1", "status": "ready", "runs_root": str(runs_root.resolve())}
        return None if stopped else probed_state

    monkeypatch.setattr(startup, "probe_dashboard", mock_probe)

    url = ensure_dashboard_running(runs_root, check_version=True, run_id="my_run")
    assert stopped == [(runs_root.resolve(), "L1")]
    assert len(launched) == 1
    assert tokens[0] == "secret" and tokens[-1] == "fresh"
    assert url == "http://localhost:8765/runs/my_run"


def test_ensure_dashboard_running_never_stops_another_roots_or_unregistered_dashboard(tmp_path, monkeypatch):
    import hypergan.web_autostart as startup
    from hypergan.web_autostart import ensure_dashboard_running
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    monkeypatch.setattr(startup, "_stop_registered_dashboard", lambda *a: pytest.fail("stopped a dashboard"))
    monkeypatch.setattr(startup, "launch_dashboard_daemon", lambda *a, **k: pytest.fail("launched a daemon"))

    # An outdated dashboard for another root holds the port: report it, stop nothing.
    monkeypatch.setattr(startup, "_registered_dashboard", lambda root: ({"launch_id": "B", "server_instance_id": "b"}, None))
    monkeypatch.setattr(startup, "probe_dashboard", lambda *a, **k: {
        "version": "0.0.1", "runs_root": str(root_a.resolve()), "server_instance_id": "a"})
    with pytest.raises(OSError, match="dashboard for root"):
        ensure_dashboard_running(root_b)

    # An outdated dashboard for this root that this root's registry did not start.
    monkeypatch.setattr(startup, "probe_dashboard", lambda *a, **k: {
        "version": "0.0.1", "runs_root": str(root_b.resolve()), "server_instance_id": "someone-else"})
    with pytest.raises(OSError, match="cannot restart"):
        ensure_dashboard_running(root_b)

    # A token-mode dashboard this root holds no credential for.
    monkeypatch.setattr(startup, "_registered_dashboard", lambda root: (None, None))
    monkeypatch.setattr(startup, "probe_dashboard", lambda *a, **k: {"status": "auth_required", "auth_required": True})
    with pytest.raises(OSError, match="token-protected"):
        ensure_dashboard_running(root_b)


def test_ensure_dashboard_running_reuses_a_current_token_dashboard(tmp_path, monkeypatch):
    """A token-mode dashboard answers the registered bearer, so a train does not restart it."""
    from hypergan import __version__
    import hypergan.web_autostart as startup
    from hypergan.web_autostart import ensure_dashboard_running
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(startup, "_registered_dashboard", lambda r: ({"launch_id": "L", "server_instance_id": "i"}, "tok"))
    monkeypatch.setattr(startup, "_stop_registered_dashboard", lambda *a: pytest.fail("restarted"))
    monkeypatch.setattr(startup, "launch_dashboard_daemon", lambda *a, **k: pytest.fail("relaunched"))
    monkeypatch.setattr(startup, "_dashboard_mismatch", lambda probed: [])

    def probe(host, port, auth_token=None, **kwargs):
        if auth_token != "tok":
            return {"status": "auth_required", "auth_required": True}
        return {"version": __version__, "runs_root": str(root.resolve()), "server_instance_id": "i",
                "auth_mode": "token"}
    monkeypatch.setattr(startup, "probe_dashboard", probe)
    assert ensure_dashboard_running(root, host="127.0.0.1", auth="token") == "http://localhost:8765"


def test_ensure_dashboard_running_refuses_to_reuse_a_dashboard_without_requested_auth(tmp_path, monkeypatch):
    """train --auth token must not silently reuse this root's auth-none dashboard."""
    from hypergan import __version__
    import hypergan.web_autostart as startup
    from hypergan.web_autostart import ensure_dashboard_running
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(startup, "_registered_dashboard", lambda r: ({"launch_id": "L", "server_instance_id": "i"}, None))
    monkeypatch.setattr(startup, "_stop_registered_dashboard", lambda *a: pytest.fail("restarted"))
    monkeypatch.setattr(startup, "launch_dashboard_daemon", lambda *a, **k: pytest.fail("relaunched"))
    monkeypatch.setattr(startup, "_dashboard_mismatch", lambda probed: [])
    monkeypatch.setattr(startup, "probe_dashboard", lambda *a, **k: {
        "version": __version__, "runs_root": str(root.resolve()), "server_instance_id": "i", "auth_mode": "none"})
    with pytest.raises(OSError, match="--auth none"):
        ensure_dashboard_running(root, auth="token")
    # Asking for no auth still reuses it, as before.
    assert ensure_dashboard_running(root, host="127.0.0.1") == "http://localhost:8765"


def test_dashboard_url_names_an_address_the_bind_accepts():
    from hypergan.web_autostart import dashboard_url
    assert dashboard_url("0.0.0.0", 8765) == "http://localhost:8765"
    assert dashboard_url("127.0.0.1", 8765, "r") == "http://localhost:8765/runs/r"
    assert dashboard_url("100.64.0.7", 9000) == "http://100.64.0.7:9000"


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


def test_stop_projector_and_stop_all_servers(tmp_path, monkeypatch):
    import hypergan.web_autostart as startup
    from hypergan.web_autostart import stop_projector, stop_all_servers
    run_dir = tmp_path / "run_proj"
    run_dir.mkdir()
    # Should not raise even if no projector is running
    stop_projector(run_dir)

    # Isolate temp/proc roots so running tests never kills live viewers on the host.
    monkeypatch.setattr(startup.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("HYPERGAN_PROC_ROOT", str(tmp_path / "proc"))
    # stop_all_servers sweeps and returns list of PIDs
    killed = stop_all_servers()
    assert isinstance(killed, list)


def test_stop_viewer_handles_none_state_without_type_error(tmp_path, monkeypatch):
    """Verify that stop_viewer handles unlinked/None state.json without TypeError."""
    from hypergan.web_autostart import stop_viewer, _registry
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    private = _registry(run_dir)
    private.mkdir(parents=True, exist_ok=True)
    initial_state = {
        'schema_version': 1, 'launch_id': 'test_launch_id',
        'status': 'starting', 'mode': 'dashboard', 'root': str(run_dir),
        'started_at': time.time(),
    }
    atomic_json(private / 'state.json', initial_state)

    calls = [0]
    orig_read_state = __import__('hypergan.web_autostart', fromlist=['_read_state'])._read_state

    def mock_read_state(priv, locked=False):
        calls[0] += 1
        if calls[0] > 1:
            return None  # Simulate state.json unlinked during shutdown
        return orig_read_state(priv, locked=locked)

    monkeypatch.setattr('hypergan.web_autostart._read_state', mock_read_state)
    result = stop_viewer(run_dir, launch_id='test_launch_id', timeout=2)
    assert result['status'] == 'stopped'


def test_probe_dashboard_silent_socket_returns_alien():
    """Verify probe_dashboard classifies a resetting/silent TCP listener as alien rather than free port."""
    from hypergan.web_autostart import probe_dashboard
    import socket, threading

    # Create an alien TCP listener on loopback that immediately closes incoming sockets
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(('127.0.0.1', 0))
    srv.listen(5)
    port = srv.getsockname()[1]
    stop_event = threading.Event()

    def accept_loop():
        srv.settimeout(0.2)
        while not stop_event.is_set():
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                pass

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    try:
        probed = probe_dashboard("127.0.0.1", port=port, timeout=0.3)
        assert probed == {'alien': True}
    finally:
        stop_event.set()
        t.join(timeout=1)
        srv.close()


def test_ensure_dashboard_running_alien_raises_oserror_immediately(tmp_path):
    """Alien port occupancy must immediately raise OSError naming PID/comm without 10s timeout."""
    from hypergan.web_autostart import ensure_dashboard_running
    import socket, threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(('127.0.0.1', 0))
    srv.listen(5)
    port = srv.getsockname()[1]

    start = time.monotonic()
    try:
        with pytest.raises(OSError, match=rf"Port {port} is already in use"):
            ensure_dashboard_running(tmp_path, port=port)
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, f"ensure_dashboard_running hung for {elapsed:.2f}s instead of raising immediately"
    finally:
        srv.close()



@pytest.mark.parametrize('address', ['127.0.0.1', '[::1]'])
def test_windows_port_owner_uses_numeric_listener_and_process_tables(monkeypatch, address):
    from types import SimpleNamespace
    from hypergan.web_autostart import _windows_port_owner
    def command(args, **kwargs):
        if args == ['netstat', '-ano']:
            return SimpleNamespace(returncode=0, stdout=(
                'TCP 127.0.0.1:8765 127.0.0.1:9000 ESTABLISHED 11\n'
                'UDP 0.0.0.0:8765 *:* 22\n'
                f'TCP {address}:87650 0.0.0.0:0 LISTENING 33\n'
                f'TCP {address}:8765 0.0.0.0:0 ABHOREN 44\n'))
        assert args == ['tasklist', '/FI', 'PID eq 44', '/FO', 'CSV', '/NH']
        return SimpleNamespace(returncode=0, stdout='"python.exe","44","Console","1","10 K"\n')
    monkeypatch.setattr(subprocess, 'run', command)
    assert _windows_port_owner(8765) == (44, 'python.exe')


def test_windows_port_owner_preserves_pid_if_process_name_query_is_unavailable(monkeypatch):
    from types import SimpleNamespace
    from hypergan.web_autostart import _windows_port_owner
    def command(args, **kwargs):
        if args[0] == 'tasklist':
            raise OSError('tasklist unavailable')
        return SimpleNamespace(returncode=0, stdout='TCP 0.0.0.0:8765 0.0.0.0:0 LISTENING 44\n')
    monkeypatch.setattr(subprocess, 'run', command)
    assert _windows_port_owner(8765) == (44, 'unknown')


@pytest.mark.parametrize('stale', [False, True])
def test_stop_all_servers_handles_windows_process_termination(tmp_path, monkeypatch, stale):
    from types import SimpleNamespace
    import hypergan.web_autostart as startup
    from hypergan.run_state import run_lock
    registry = tmp_path / 'hypergan-viewers-test' / 'root'
    (registry / 'service').mkdir(parents=True)
    atomic_json(registry / 'state.json', {'launch_id': 'L', 'status': 'ready', 'supervisor_pid': 44})
    signals = []
    def kill(pid, signum):
        signals.append((pid, signum))
        if stale:
            raise OSError(22, 'invalid stale process')
    monkeypatch.setattr(startup.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(startup, 'os', SimpleNamespace(
        name='nt', getpid=lambda: 1, kill=kill, path=SimpleNamespace(exists=lambda path: False)))
    # The supervisor's lifetime lock is what makes its recorded PID trustworthy.
    with run_lock(registry / 'service'):
        assert startup.stop_all_servers() == ([] if stale else [44])
    assert signals == [(44, signal.SIGTERM)]
    assert not (registry / 'state.json').exists()


def test_stop_all_servers_never_signals_pids_from_stopped_or_stale_states(tmp_path, monkeypatch):
    """A PID in a stopped viewer's state file can belong to a training job after reuse."""
    import hypergan.web_autostart as startup
    from hypergan.run_state import run_lock
    victim = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    proc = tmp_path / 'proc'
    proc.mkdir()
    try:
        viewers = tmp_path / 'hypergan-viewers-test'
        for name, status in (('stopped', 'stopped'), ('stale', 'ready')):
            (viewers / name / 'service').mkdir(parents=True)
            atomic_json(viewers / name / 'state.json', {
                'launch_id': name, 'status': status, 'supervisor_pid': victim.pid,
                'processes': {'server': victim.pid, 'projector': victim.pid}})
        monkeypatch.setattr(startup.tempfile, 'gettempdir', lambda: str(tmp_path))
        monkeypatch.setenv('HYPERGAN_PROC_ROOT', str(proc))
        assert startup.stop_all_servers() == []
        assert victim.poll() is None

        # A live supervisor (its lock is held) is stopped, but its recorded
        # children only when the process table names it as their parent.
        (proc / str(victim.pid)).mkdir()
        (proc / str(victim.pid) / 'stat').write_text(f'{victim.pid} (python) S 1 0 0')
        signalled = []
        monkeypatch.setattr(startup.os, 'kill', lambda pid, sig: signalled.append((pid, sig)))
        (viewers / 'live' / 'service').mkdir(parents=True)
        atomic_json(viewers / 'live' / 'state.json', {
            'launch_id': 'live', 'status': 'ready', 'supervisor_pid': 999999,
            'processes': {'server': victim.pid}})
        with run_lock(viewers / 'live' / 'service'):
            assert startup.stop_all_servers() == [999999]
        assert all(pid == 999999 for pid, _ in signalled)
    finally:
        monkeypatch.undo()  # restores os.kill
        victim.kill()
        victim.wait()


def test_ready_dashboard_receipts_new_runs_without_rewriting_existing_receipts(tmp_path):
    from hypergan.web_autostart import stop_viewer
    runs = tmp_path / 'runs'
    runs.mkdir()
    first = runs / 'existing'
    make_run(first)
    viewer = Viewer(runs, port=0, host='127.0.0.1', auth='none')
    try:
        viewer.wait_ready()
        name = 'viewer-' + viewer.session.instance_id + '.json'
        existing = first / 'observations' / name
        def ready(path):
            try:
                value = json.loads(path.read_text())
                return value if value['status'] == 'ready' else None
            except OSError:
                return None
        previous = until(lambda: ready(existing))
        modified = existing.stat().st_mtime_ns
        fresh = runs / 'created-after-ready'
        make_run(fresh)
        receipt = until(lambda: ready(fresh / 'observations' / name))
        assert receipt['server_instance_id'] == previous['server_instance_id']
        assert receipt['processes'] == previous['processes']
        assert existing.stat().st_mtime_ns == modified
        assert viewer.info['status'] == 'ready'
        assert stop_viewer(runs)['status'] == 'stopped'
        assert json.loads((fresh / 'observations' / name).read_text())['status'] == 'stopped'
    finally:
        stop_viewer(runs)

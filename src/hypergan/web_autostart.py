"""CLI-only viewer supervision. Never imported by the Python training API.

The broker, HTTP server, and bounded built-in projection producer are separate
spawned processes. Their supervisor is detached from training and reused per run.
No HTTP request executes a Python event map.
"""
from contextlib import contextmanager, nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import subprocess
import hashlib
from types import SimpleNamespace
import tempfile
import threading
import time


class _StopFlag:
    """One-way cooperative stop with no cross-process lock to poison on death."""

    def __init__(self, context):
        # One broker writes this one byte once (0 -> 1); children only read it.
        # Unlike generic RawValue read/modify/write operations, these aligned
        # single-byte loads/stores need no shared lock on our supported hosts.
        self._value = context.RawValue('b', 0)

    def set(self):
        self._value.value = 1

    def is_set(self):
        return bool(self._value.value)

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return True


def _parent_alive():
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def _watch_parent():
    # Keep monitoring even after cooperative stop: the broker can die during
    # shutdown, before it has reaped a child stuck in other work.
    while True:
        time.sleep(0.2)
        if not _parent_alive():
            os._exit(1)


def _server(root, listener, session, stop):
    from .web_server import run_socket

    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    def watch():
        requested = False
        while True:
            time.sleep(0.2)
            if not _parent_alive():
                # Parent death must also interrupt a main thread stuck in native
                # code, where a Python SIGTERM handler cannot execute promptly.
                os._exit(1)
            if stop.is_set() and not requested:
                requested = True
                os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch, daemon=True).start()
    run_socket(root, listener, session)


def _project(root, stop):
    from .event_views import Projector

    threading.Thread(target=_watch_parent, daemon=True).start()
    while not (root / 'manifest.json').is_file():
        if stop.wait(0.1):
            return
    with Projector(root) as projector:
        drain_deadline = None
        while True:
            if stop.is_set():
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + 1
                if time.monotonic() >= drain_deadline:
                    return
            progress = projector.project(limit=100)
            if not progress['has_more']:
                if stop.wait(0.1):
                    # One final bounded page catches the last training event.
                    projector.project(limit=100)
                    return


def _registry(root):
    """Private state outside experiment artifacts, stable across CLI invocations."""
    owner = str(os.getuid()) if hasattr(os, 'getuid') else hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
    base = Path(tempfile.gettempdir()) / ('hypergan-viewers-' + owner)
    private = base / hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()
    for path in (base, private):
        path.mkdir(mode=0o700, exist_ok=True)
        if path.is_symlink() or (os.name != 'nt' and
                (path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077)):
            raise ValueError('Viewer registry must be private and owned by the current user')
    (private / 'service').mkdir(exist_ok=True)
    return private


def _read_state(private, *, locked=False):
    # Windows readers can temporarily deny replacement of an open file. Use
    # the launch lock for our readers and writers, not timing-dependent retries.
    with nullcontext() if locked else _launch_lock(private):
        try:
            return json.loads((private / 'state.json').read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None


def _session(state):
    credentials = json.loads(Path(state['session_file']).read_text(encoding='utf-8'))
    if credentials['server_instance_id'] != state['server_instance_id']:
        raise ValueError('Viewer credentials do not match the registered incarnation')
    return SimpleNamespace(host=credentials['origin'].removeprefix('http://'),
                           origin=credentials['origin'], instance_id=credentials['server_instance_id'],
                           _token=credentials.get('token', ''), auth_mode=credentials['auth_mode'],
                           bind_host=credentials['bind_host'])


def _probe(session):
    import http.client
    connection = http.client.HTTPConnection(session.host, timeout=0.25)
    try:
        headers = {'Authorization': 'Bearer ' + session._token} if session.auth_mode == 'token' else {}
        connection.request('GET', '/api/v1/capabilities', headers=headers)
        response = connection.getresponse()
        body = response.read(65537)
        return (response.status == 200 and len(body) <= 65536
                and json.loads(body).get('server_instance_id') == session.instance_id)
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _locked(private):
    from .run_state import run_lock
    service = private / 'service'
    service.mkdir(exist_ok=True)
    try:
        with run_lock(service):
            return False
    except RuntimeError:
        return True


@contextmanager
def _launch_lock(private):
    from contextlib import ExitStack
    from .run_state import run_lock
    deadline = time.monotonic() + 3
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(run_lock(private))
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Another viewer operation is still in progress')
                time.sleep(.02)
        yield


def _running(private, state):
    # The OS lifetime lock is authoritative, never an unverified PID. A short
    # startup reservation bridges Popen until the new broker acquires its lock.
    return bool(state and (_locked(private) or
        (state['status'] == 'starting' and time.time() - state['started_at'] < 15)))


def _receipt(root, session, mode, status, children):
    from .run_state import atomic_json
    if not (root / 'manifest.json').is_file():
        return
    observations = root / 'observations'
    if observations.is_symlink():
        raise ValueError('Viewer observations directory cannot be a symlink')
    observations.mkdir(exist_ok=True)
    atomic_json(observations / ('viewer-' + session.instance_id + '.json'),
        {'schema_version': 1, 'mode': mode, 'status': status,
         'server_instance_id': session.instance_id, 'origin': session.origin,
         'auth_mode': session.auth_mode, 'bind_host': session.bind_host,
         'processes': {'supervisor': os.getpid(), **{name: child.pid for name, child in children.items()}}})


def _broker(root, private, launch_id):
    from .run_state import atomic_json, run_lock
    from .web_launch import bind_server
    from .web_session import LocalSession

    from contextlib import ExitStack
    context = mp.get_context('spawn')
    stop = _StopFlag(context)
    children = {}
    listener = session = None
    credential_path = private / ('session-' + launch_id + '.json')
    def terminate(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with ExitStack() as ownership:
        # Serialize the startup handoff with cancellation. A stop command can
        # cancel an unstarted broker without waiting on a reservation timeout.
        with _launch_lock(private):
            ownership.enter_context(run_lock(private / 'service'))
            state = _read_state(private, locked=True)
            if state['launch_id'] != launch_id or state['status'] == 'stopped':
                return
        # Metadata publication can fail after children are already reaped.
        # Credential cleanup must run regardless, before the lifetime lock exits.
        ownership.callback(credential_path.unlink, missing_ok=True)
        ownership.callback((private / ('stop-' + launch_id)).unlink, missing_ok=True)
        def publish(status, error=None):
            state.update(status=status, error=error, supervisor_pid=os.getpid(),
                         processes={name: child.pid for name, child in children.items()})
            if session is not None:
                state.update(origin=session.origin, server_instance_id=session.instance_id,
                             session_file=str(credential_path))
            with _launch_lock(private):
                atomic_json(private / 'state.json', state)
            if session is not None:
                try:
                    _receipt(root, session, state['mode'], status, children)
                except (OSError, ValueError) as exc:
                    print(f'Viewer receipt unavailable: {exc}', file=sys.stderr, flush=True)
        try:
            listener = bind_server(state['port'], state['host'])
            session = LocalSession(listener.getsockname()[1], host=state['host'], auth=state['auth'])
            session.write_credentials(credential_path)
            for name, target, args in [('server', _server, (root, listener, session, stop)),
                                      ('projector', _project, (root, stop))]:
                child = context.Process(target=target, args=args, name='hypergan-' + name)
                child.start()
                children[name] = child
            listener.close()
            publish('starting')
            deadline = time.monotonic() + 10
            receipt_exists = False
            while not stop.wait(.1):
                if (private / ('stop-' + launch_id)).exists():
                    break
                failed = [name for name, child in children.items() if not child.is_alive()]
                if failed:
                    publish('failed', 'Viewer process exited: ' + ', '.join(failed))
                    break
                if state['status'] == 'starting':
                    if _probe(session):
                        publish('ready')
                    elif time.monotonic() >= deadline:
                        publish('failed', 'Viewer did not become ready within 10 seconds')
                        break
                if not receipt_exists and (root / 'manifest.json').is_file():
                    publish(state['status'])
                    receipt_exists = True
        except BaseException as exc:
            publish('failed', f'{type(exc).__name__}: {exc}')
        finally:
            stop.set()
            deadline = time.monotonic() + 2
            for child in children.values():
                child.join(max(0, deadline - time.monotonic()))
            for child in children.values():
                if child.is_alive():
                    child.kill()
            for child in children.values():
                child.join(2)
            if listener is not None:
                listener.close()
            publish('stopped' if state['status'] != 'failed' else 'failed', state.get('error'))


def stop_viewer(root, *, launch_id=None, timeout=8):
    """Stop only the registered incarnation; never signal a PID from a file."""
    from .run_state import atomic_json
    private = _registry(root)
    with _launch_lock(private):
        state = _read_state(private, locked=True)
        if not state or (launch_id is not None and state['launch_id'] != launch_id):
            return {'status': 'not_running'}
        if not _running(private, state):
            (private / ('session-' + state['launch_id'] + '.json')).unlink(missing_ok=True)
            return {'status': 'not_running'}
        launch_id = state['launch_id']
        if not _locked(private):
            state['status'] = 'stopped'
            atomic_json(private / 'state.json', state)
            return {'status': 'stopped', 'server_instance_id': state.get('server_instance_id')}
        (private / ('stop-' + launch_id)).touch(mode=0o600)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = _read_state(private)
        if current['launch_id'] != launch_id or (not _locked(private) and current['status'] != 'starting'):
            return {'status': 'stopped', 'server_instance_id': state.get('server_instance_id')}
        time.sleep(.05)
    raise RuntimeError('Viewer shutdown timed out; inspect ' + str(private / 'viewer.log'))


def viewer_status(root):
    """Read launch/connection information without starting a viewer or training."""
    private = _registry(root)
    state = _read_state(private)
    if state is None:
        return {'status': 'not_running'}
    result = dict(state, log_file=str(private / 'viewer.log'))
    if not _running(private, state) and state['status'] not in {'stopped', 'failed'}:
        result['status'] = 'not_running'
    return result


class Viewer:
    """Attach to one persistent per-run supervisor; optional startup stays async."""

    def __init__(self, root, *, port=0, host=None, auth=None, mode='auto', open_browser=False):
        from .run_state import atomic_json, run_lock
        from .web_launch import bind_server, require_web
        import secrets

        require_web()
        self.root = Path(root).resolve()
        self.private = _registry(root)
        self.process = None
        self.ready = threading.Event()
        self.failed = threading.Event()
        self.info = self.session = None
        self._closing = threading.Event()
        self._open_browser = open_browser
        with _launch_lock(self.private):
            state = _read_state(self.private, locked=True)
            if _running(self.private, state):
                if ((host is not None and host != state['host']) or
                    (auth is not None and auth != state['auth']) or
                    (port and port != int(state.get('origin', ':0').rsplit(':', 1)[1]) and port != state['port'])):
                    raise ValueError('Existing viewer has different bind/auth settings; use hypergan stop-server RUN first')
            else:
                if state:
                    # An abruptly killed broker cannot unlink credentials. They
                    # cannot authenticate a new incarnation; remove this stale
                    # private file when the next launcher takes over.
                    (self.private / ('session-' + state['launch_id'] + '.json')).unlink(missing_ok=True)
                host, auth = host or '0.0.0.0', auth or 'none'
                # Fail explicit invalid ports before numerical imports. The
                # broker repeats binding; a race remains an actionable error.
                with bind_server(port, host):
                    pass
                if auth not in {'none', 'token'}:
                    raise ValueError("auth must be 'none' or 'token'")
                state = {'schema_version': 1, 'launch_id': secrets.token_hex(16),
                         'root': str(self.root), 'host': host, 'auth': auth, 'port': port,
                         'mode': mode, 'status': 'starting', 'started_at': time.time()}
                atomic_json(self.private / 'state.json', state)
                options = {'start_new_session': True} if os.name != 'nt' else {
                    'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
                with (self.private / 'viewer.log').open('ab') as log:
                    try:
                        self.process = subprocess.Popen([sys.executable, '-m', 'hypergan.web_autostart',
                            str(self.root), str(self.private), state['launch_id']], stdin=subprocess.DEVNULL,
                            stdout=log, stderr=log, close_fds=True, **options)
                    except BaseException:
                        state['status'] = 'failed'
                        atomic_json(self.private / 'state.json', state)
                        raise
        self.launch_id = state['launch_id']
        self.credential_path = self.private / ('session-' + self.launch_id + '.json')
        print(f'Viewer status: hypergan server-status {str(self.root)!r}\n'
              f'Stop viewer: hypergan stop-server {str(self.root)!r}', file=sys.stderr, flush=True)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def _monitor(self):
        deadline = time.monotonic() + 12
        while not self._closing.wait(.1):
            try:
                state = _read_state(self.private)
                if not state or state['launch_id'] != self.launch_id:
                    raise RuntimeError('Viewer incarnation changed')
                if state['status'] in {'failed', 'stopped'} or not _running(self.private, state):
                    raise RuntimeError(state.get('error') or 'Viewer supervisor stopped')
                if not self.ready.is_set() and state['status'] == 'ready':
                    session = _session(state)
                    if _probe(session):
                        self.session, self.info = session, state
                        self.ready.set()
                        print(f"Viewer: {session.origin} (listening on {session.bind_host})", file=sys.stderr, flush=True)
                        if session.auth_mode == 'token':
                            print(f'Credentials: {self.credential_path} (copy its token into the sign-in form)',
                                  file=sys.stderr, flush=True)
                        if self._open_browser:
                            import webbrowser
                            threading.Thread(target=webbrowser.open, args=(session.origin,), daemon=True).start()
                if not self.ready.is_set() and time.monotonic() >= deadline:
                    raise RuntimeError('Viewer startup timed out; inspect ' + str(self.private / 'viewer.log'))
            except (OSError, ValueError, RuntimeError) as exc:
                print(f'warning: viewer: {exc}; training continues independently', file=sys.stderr, flush=True)
                self.failed.set()
                return

    def wait_ready(self, timeout=12):
        deadline = time.monotonic() + timeout
        while not self.ready.wait(.05):
            if self.failed.is_set() or time.monotonic() >= deadline:
                raise RuntimeError('Requested viewer failed startup; training has not started')

    def detach(self):
        self._closing.set()
        self.monitor.join(.5)
        if self.process is not None:
            # Reap an eventual exit without keeping the training command alive.
            threading.Thread(target=self.process.wait, daemon=True).start()

    def close(self):
        self.detach()
        result = stop_viewer(self.root, launch_id=self.launch_id)
        if self.process is not None:
            self.process.wait(timeout=2)
        return result


@contextmanager
def training_viewer(root, *, required=False, port=0, host=None, auth=None, open_browser=False):
    if not required:
        from importlib.util import find_spec
        if any(find_spec(name) is None for name in ('starlette', 'uvicorn', 'wasmtime')):
            yield None
            return
    viewer = None
    try:
        try:
            viewer = Viewer(root, port=port, host=host, auth=auth,
                            mode='explicit' if required else 'auto', open_browser=open_browser)
            if required:
                viewer.wait_ready()
        except (OSError, RuntimeError, ImportError) as exc:
            if viewer is not None:
                viewer.close()
                viewer = None
            if required:
                raise
            print(f'warning: viewer unavailable: {exc}; training continues headless', file=sys.stderr, flush=True)
        yield viewer
    finally:
        if viewer is not None:
            viewer.detach()


if __name__ == '__main__':
    _broker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])

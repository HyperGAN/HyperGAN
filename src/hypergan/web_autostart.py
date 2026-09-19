"""CLI-only viewer supervision. Never imported by the Python training API.

The broker, HTTP server, and bounded built-in projection producer are separate
spawned processes. No HTTP request executes a Python event map.
"""
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
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


def _probe(session):
    import http.client

    connection = http.client.HTTPConnection('127.0.0.1', int(session.host.rsplit(':', 1)[1]), timeout=0.25)
    try:
        connection.request('GET', '/api/v1/capabilities', headers={'Authorization': 'Bearer ' + session._token})
        response = connection.getresponse()
        body = response.read(65537)
        return (response.status == 200 and len(body) <= 65536
                and json.loads(body).get('server_instance_id') == session.instance_id)
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _receipt(root, session, mode, status, children):
    # Pending viewers must not race the trainer's exclusive new-run mkdir.
    if not (root / 'manifest.json').is_file():
        return
    observations = root / 'observations'
    if observations.is_symlink():
        raise ValueError('Viewer observations directory cannot be a symlink')
    observations.mkdir(exist_ok=True)
    path = observations / ('viewer-' + session.instance_id + '.json')
    temporary = observations / ('.viewer-' + session.instance_id + '.tmp')
    record = {'schema_version': 1, 'mode': mode, 'status': status,
              'server_instance_id': session.instance_id, 'origin': session.origin,
              'processes': {name: child.pid for name, child in children.items()}}
    with temporary.open('x', encoding='utf-8') as output:
        json.dump(record, output, allow_nan=False)
        output.write('\n')
    temporary.replace(path)


def _broker(root, listener, session, credential_path, connection, mode):
    context = mp.get_context('spawn')
    stop = _StopFlag(context)
    children = {}
    status = 'starting'
    receipt_status = None
    def send(kind, detail):
        try:
            connection.send_bytes(json.dumps({'kind': kind, 'detail': detail}).encode())
        except (OSError, EOFError):
            pass
    try:
        for name, target, args in [('server', _server, (root, listener, session, stop)),
                                    ('projector', _project, (root, stop))]:
            child = context.Process(target=target, args=args, name='hypergan-' + name)
            child.start()
            children[name] = child
        listener.close()
        deadline = time.monotonic() + 10
        reported = set()
        while _parent_alive():
            if connection.poll(0.1):
                try:
                    if connection.recv_bytes(1024) == b'stop':
                        break
                except (EOFError, OSError):
                    break
            for name, child in children.items():
                if not child.is_alive() and name not in reported:
                    reported.add(name)
                    status = 'degraded'
                    send('warning', f'{name} exited (code {child.exitcode}); training continues independently')
            if status == 'starting':
                if _probe(session):
                    status = 'ready'
                    send('ready', {'origin': session.origin, 'session_file': str(credential_path),
                                   'server_instance_id': session.instance_id,
                                   'processes': {name: child.pid for name, child in children.items()}})
                elif time.monotonic() >= deadline:
                    status = 'degraded'
                    send('warning', 'server did not become ready within 10 seconds; training continues independently')
            if receipt_status != status and (root / 'manifest.json').is_file():
                try:
                    _receipt(root, session, mode, status, children)
                except OSError as exc:
                    send('warning', f'could not record viewer health: {exc}')
                receipt_status = status
    except BaseException as exc:
        send('warning', f'viewer supervisor failed: {type(exc).__name__}: {exc}')
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
        try:
            _receipt(root, session, mode, 'stopped', children)
        except (OSError, ValueError):
            pass
        listener.close()
        credential_path.unlink(missing_ok=True)
        try:
            credential_path.parent.rmdir()
        except OSError:
            pass
        connection.close()


class Viewer:
    """A bounded supervision handle; construction never waits for HTTP startup."""

    def __init__(self, root, *, port=0, mode='auto', open_browser=False):
        from .web_launch import bind_loopback, require_web
        from .web_session import LocalSession

        require_web()
        self.process = None
        self.connection = None
        self.ready = threading.Event()
        self.failed = threading.Event()
        self.info = None
        self._closing = threading.Event()
        self._open_browser = open_browser
        with bind_loopback(port) as listener:
            self.session = LocalSession(listener.getsockname()[1])
            private = Path(tempfile.mkdtemp(prefix='hypergan-viewer-'))
            self.credential_path = private / 'session.json'
            try:
                self.session.write_credentials(self.credential_path)
                context = mp.get_context('spawn')
                self.connection, child_connection = context.Pipe()
                self.process = context.Process(target=_broker,
                    args=(Path(root).resolve(), listener, self.session, self.credential_path, child_connection, mode),
                    name='hypergan-viewer-supervisor')
                try:
                    self.process.start()
                finally:
                    child_connection.close()
            except BaseException:
                self.credential_path.unlink(missing_ok=True)
                private.rmdir()
                if self.connection is not None:
                    self.connection.close()
                raise
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def _monitor(self):
        while not self._closing.is_set():
            try:
                if not self.connection.poll(0.1):
                    if not self.process.is_alive():
                        raise EOFError
                    continue
                message = json.loads(self.connection.recv_bytes(65536))
            except (EOFError, OSError, ValueError):
                if not self._closing.is_set():
                    print('warning: viewer supervisor disconnected; training continues independently', file=sys.stderr, flush=True)
                    self.failed.set()
                return
            if message['kind'] == 'ready':
                self.info = message['detail']
                self.ready.set()
                print(f"Viewer: {self.info['origin']}\nCredentials: {self.credential_path} (copy its token into the sign-in form)",
                      file=sys.stderr, flush=True)
                if self._open_browser:
                    import webbrowser
                    threading.Thread(target=webbrowser.open, args=(self.info['origin'],), daemon=True).start()
            else:
                print('warning: viewer: ' + message['detail'], file=sys.stderr, flush=True)
                if not self.ready.is_set():
                    self.failed.set()

    def wait_ready(self, timeout=12):
        deadline = time.monotonic() + timeout
        while not self.ready.wait(0.05):
            if self.failed.is_set() or time.monotonic() >= deadline:
                raise RuntimeError('Requested viewer failed startup; training has not started')

    def close(self):
        self._closing.set()
        try:
            self.connection.send_bytes(b'stop')
        except (OSError, EOFError):
            pass
        self.process.join(6)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)
        self.monitor.join(0.5)
        self.connection.close()
        self.credential_path.unlink(missing_ok=True)
        try:
            self.credential_path.parent.rmdir()
        except OSError:
            pass


@contextmanager
def training_viewer(root, *, required=False, port=0, open_browser=False):
    if not required:
        from importlib.util import find_spec
        if any(find_spec(name) is None for name in ('starlette', 'uvicorn', 'wasmtime')):
            yield None
            return
    viewer = None
    previous = None
    try:
        try:
            viewer = Viewer(root, port=port, mode='explicit' if required else 'auto', open_browser=open_browser)
            if required:
                viewer.wait_ready()
        except (OSError, RuntimeError, ImportError) as exc:
            if viewer is not None:
                viewer.close()
                viewer = None
            if required:
                raise
            print(f'warning: viewer unavailable: {exc}; training continues headless', file=sys.stderr, flush=True)
        if viewer is not None and threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGTERM)
            def terminate(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminate)
        yield viewer
    finally:
        if viewer is not None:
            viewer.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)

"""Explicit local viewer launch, keeping credentials outside experiment artifacts."""
import importlib.util
import json
from pathlib import Path
import socket
import signal
import sys
import tempfile

from .ports import DEFAULT_VIEWER_PORT


def require_web():
    missing = [name for name in ('starlette', 'uvicorn')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Local serving requires 'hypergan[web]' (missing " + ', '.join(missing) + ')')


def bind_server(port=0, host="0.0.0.0"):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('port must be in 0..65535; zero selects an available port')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Windows otherwise permits overlapping wildcard/specific listeners,
        # defeating occupied-port preflight and making request routing ambiguous.
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        elif sys.platform.startswith('linux'):
            # A restart must not fail on the previous viewer's TIME_WAIT
            # connections; on Linux a live listener on any address still makes
            # bind raise EADDRINUSE. BSD/macOS would let a wildcard bind share a
            # port with a loopback listener, so the option stays off there.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(128)
        return listener
    except BaseException:
        listener.close()
        raise


def bind_available(port=None, host="0.0.0.0", *, attempts=1):
    """Bind the viewer socket to fixed default port 8765, or explicit requested port.

    Does not hunt across port range. If the target port is occupied, raises an
    actionable OSError identifying the conflicting process.
    """
    if isinstance(port, str) and isinstance(host, int):
        port, host = host, port
    target_port = DEFAULT_VIEWER_PORT if port is None else port
    last_error = None
    for attempt in range(max(1, attempts)):
        try:
            return bind_server(target_port, host)
        except OSError as error:
            last_error = error
            if attempt < attempts - 1:
                import time
                time.sleep(0.05)
    if target_port == 0:
        raise last_error
    from .web_autostart import get_port_owner
    owner = get_port_owner(target_port)
    if owner:
        pid, comm = owner
        raise OSError(f"Port {target_port} is already in use by PID {pid} ({comm}); "
                      f"specify a different root or stop conflicting process") from last_error
    raise OSError(f"Port {target_port} is already in use; "
                  f"specify a different root or stop conflicting process") from last_error


def bind_loopback(port=0):
    """Explicit loopback socket for internal clients and test fixtures."""
    return bind_server(port, "127.0.0.1")


def running_dashboard(root, port=None, auth="none"):
    """URL of the dashboard already serving `root` (or the root holding this run) on `port`.

    `hypergan train` starts a persistent dashboard; `hypergan view` for the same
    root then opens it instead of failing to bind. A dashboard for another root
    is not reused, and one with different auth is reported rather than reused.
    """
    from .web_autostart import _registered_dashboard, dashboard_url, find_runs_root, probe_dashboard
    target_port = DEFAULT_VIEWER_PORT if port is None else port
    if target_port == 0:
        return None
    root = Path(root).resolve()
    runs_root = find_runs_root(root)
    state, token = _registered_dashboard(runs_root)
    if state is None and runs_root != root:
        state, token = _registered_dashboard(root)
    probed = probe_dashboard("127.0.0.1", target_port, auth_token=token)
    if not probed or probed.get('alien') or probed.get('auth_required') or not probed.get('runs_root'):
        return None
    served = Path(probed['runs_root']).resolve()
    if served == root:
        run_id = None
    elif (root / 'manifest.json').is_file() and root.is_relative_to(served):
        run_id = root.name
    else:
        return None
    if probed.get('auth_mode') != auth:
        raise OSError(f"Port {target_port} already serves this root's dashboard with --auth {probed.get('auth_mode')}; "
                      f"stop it with 'hypergan view {served} --stop' to change its settings")
    host = state.get('host', '0.0.0.0') if state else '0.0.0.0'
    return dashboard_url(host, target_port, run_id)


def serve(run_dir=None, *, port=None, host="0.0.0.0", auth="none", session_file=None,
          open_browser=False, public_origin=None):
    require_web()
    from contextlib import ExitStack
    import os
    import secrets
    import threading
    import time
    from .run_state import atomic_json, run_lock
    from . import web_autostart as _wa
    from .web_files import read_json
    from .web_server import run_socket
    from .web_session import LocalSession

    if hasattr(run_dir, "path"):
        run_dir = getattr(run_dir, "path")
    if run_dir is None:
        raw_path = _wa.find_runs_root(Path.cwd())
    else:
        raw_path = Path(run_dir)
        if not raw_path.exists():
            from .registry import resolve_run_target
            try:
                raw_path = resolve_run_target(str(run_dir), allow_latest=(str(run_dir) == "latest"))
            except Exception:
                pass
    root = raw_path.resolve(strict=True)
    if (root / 'manifest.json').is_file():
        manifest = read_json(root, 'manifest.json')
        if not isinstance(manifest, dict) or not isinstance(manifest.get('run_id'), str):
            raise ValueError('Selected directory has no valid run manifest')
        _wa.ensure_run_projected(root)
    elif not root.is_dir():
        raise ValueError('Selected directory does not exist')
    if session_file is not None and Path(session_file).resolve().is_relative_to(root):
        raise ValueError('Viewer credentials must be stored outside the run directory')
    existing = running_dashboard(root, port, auth)
    if existing is not None:
        print(json.dumps({'origin': existing, 'reused': True}), flush=True)
        print(f'Viewer: {existing} (already running for this root)', file=sys.stderr, flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open(existing)
        return
    with tempfile.TemporaryDirectory(prefix='hypergan-viewer-') as private:
        credential_path = Path(session_file) if session_file is not None else Path(private) / 'session.json'
        with bind_available(port, host) as listener:
            bound_port = listener.getsockname()[1]
            session = LocalSession(bound_port, host=host, auth=auth,
                                   public_origin=public_origin)
            session.write_credentials(credential_path)

            launch_id = secrets.token_hex(16)
            started_at = time.time()
            runs_root = _wa.find_runs_root(root)
            candidate_roots = [runs_root, root]
            if root.parent != root:
                candidate_roots.append(root.parent.resolve())
                if root.parent.name == "runs" and root.parent.parent != root.parent:
                    candidate_roots.append(root.parent.parent.resolve())
            if root.name == "runs" and root.parent != root:
                candidate_roots.append(root.parent.resolve())
            if (root / "runs").is_dir():
                candidate_roots.append((root / "runs").resolve())
            unique_roots = []
            for cand in candidate_roots:
                rc = Path(cand).resolve()
                if rc not in unique_roots:
                    unique_roots.append(rc)

            state_payload = {
                'schema_version': 1,
                'launch_id': launch_id,
                'root': str(root),
                'runs_root': str(runs_root),
                'host': host,
                'auth': auth,
                'port': bound_port,
                'origin': session.origin,
                'public_origin': session.public_origin,
                'server_instance_id': session.instance_id,
                'session_file': str(credential_path),
                'mode': 'dashboard',
                'status': 'ready',
                'started_at': started_at,
                'supervisor_pid': os.getpid(),
                'pid': os.getpid(),
                'processes': {'server': os.getpid()},
            }

            # Uvicorn restores and re-raises SIGTERM after its graceful shutdown.
            # Translate that second delivery into Python unwinding so private
            # credential cleanup runs instead of the OS immediately terminating.
            previous_term = signal.getsignal(signal.SIGTERM)
            previous_int = signal.getsignal(signal.SIGINT)
            def terminate(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)

            stop_event = threading.Event()
            owned_reg_dirs = []
            with ExitStack() as ownership:
                for cand in unique_roots:
                    try:
                        reg_dir = _wa._registry(cand)
                        if reg_dir in owned_reg_dirs:
                            continue
                        ownership.enter_context(run_lock(reg_dir / 'service'))
                        with _wa._launch_lock(reg_dir):
                            reg_cred = reg_dir / f'session-{launch_id}.json'
                            session.write_credentials(reg_cred)
                            reg_state = dict(state_payload, session_file=str(reg_cred))
                            atomic_json(reg_dir / 'state.json', reg_state)
                        owned_reg_dirs.append(reg_dir)
                    except Exception:
                        pass

                if owned_reg_dirs:
                    def _watch_stop():
                        while not stop_event.wait(0.1):
                            if any((reg_dir / f'stop-{launch_id}').exists() for reg_dir in owned_reg_dirs):
                                try:
                                    os.kill(os.getpid(), signal.SIGTERM)
                                except OSError:
                                    pass
                                break
                    threading.Thread(target=_watch_stop, daemon=True).start()

                def _bg_project():
                    bg_projector = _wa.Projector(root, poll_interval=0.25)
                    while not stop_event.wait(0.25):
                        if (root / '.stop_projector').exists():
                            break
                        try:
                            bg_projector.poll_once()
                        except Exception:
                            pass
                threading.Thread(target=_bg_project, daemon=True).start()

                try:
                    print(json.dumps({'origin': session.origin, 'session_file': str(credential_path),
                                      'server_instance_id': session.instance_id,
                                      **({'public_origin': session.public_origin} if session.public_origin else {})}),
                          flush=True)
                    print(f'Viewer: {session.origin} (listening on {host})', file=sys.stderr, flush=True)
                    if session.public_origin:
                        print(f'Public viewer: {session.public_origin} (through your own TLS proxy)',
                              file=sys.stderr, flush=True)
                    if auth == "token":
                        print(f'Credentials: {credential_path} (copy its token into the sign-in form)',
                              file=sys.stderr, flush=True)
                    if open_browser:
                        import webbrowser
                        webbrowser.open(session.browser_origin)
                    run_socket(root, listener, session)
                finally:
                    stop_event.set()
                    for reg_dir in owned_reg_dirs:
                        try:
                            stopped_state = dict(state_payload, status='stopped',
                                                 session_file=str(reg_dir / f'session-{launch_id}.json'))
                            with _wa._launch_lock(reg_dir):
                                atomic_json(reg_dir / 'state.json', stopped_state)
                                (reg_dir / f'session-{launch_id}.json').unlink(missing_ok=True)
                                (reg_dir / f'stop-{launch_id}').unlink(missing_ok=True)
                        except Exception:
                            pass
                    credential_path.unlink(missing_ok=True)
                    signal.signal(signal.SIGTERM, previous_term)
                    signal.signal(signal.SIGINT, previous_int)


def dashboard_status(runs_root=None):
    """Report status of the root dashboard or viewer for a given runs root, run directory, tag, or slug."""
    from .web_autostart import dashboard_status as _ds
    return _ds(runs_root)


def stop_dashboard(runs_root=None, timeout: float = 8.0):
    """Cleanly stop the root dashboard or viewer for a given runs root, run directory, tag, or slug."""
    from .web_autostart import stop_dashboard as _sd
    return _sd(runs_root, timeout=timeout)

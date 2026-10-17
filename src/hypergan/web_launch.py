"""Explicit local viewer launch, keeping credentials outside experiment artifacts."""
import importlib.util
import json
from pathlib import Path
import socket
import signal
import sys
import tempfile

from .ports import DEFAULT_VIEWER_PORT, PORT_SEARCH_LIMIT


def require_web():
    missing = [name for name in ('starlette', 'uvicorn', 'wasmtime')
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
            dash_state_files = []
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
                    if hasattr(_wa, "_dashboard_state_path"):
                        try:
                            sf = Path(_wa._dashboard_state_path(cand))
                            atomic_json(sf, {
                                "pid": os.getpid(),
                                "port": bound_port,
                                "host": host,
                                "url": session.origin,
                                "origin": session.origin,
                                "runs_root": str(runs_root),
                                "root": str(root),
                                "status": "ready",
                                "started_at": started_at,
                                "server_instance_id": session.instance_id,
                            })
                            dash_state_files.append(sf)
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
                    for sf in dash_state_files:
                        try:
                            sf.unlink(missing_ok=True)
                        except OSError:
                            pass
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


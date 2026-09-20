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


def bind_available(port=None, host="0.0.0.0", *, attempts=PORT_SEARCH_LIMIT):
    """Bind the viewer socket, preferring one stable port across restarts.

    ``None`` requests the default port and permits a bounded upward search, so
    a bookmarked URL keeps working while a second concurrent run still starts.
    Every other value is taken literally, including ``0`` for an OS-assigned
    port: an occupied explicit port is an error, never a silent relocation.
    Each candidate is bound for real, because probing a port and binding it
    afterwards leaves a window for another process to take it.
    """
    if port is not None:
        return bind_server(port, host)
    if type(attempts) is not int or attempts < 1:
        raise ValueError('attempts must be a positive integer')
    last = min(DEFAULT_VIEWER_PORT + attempts, 65536) - 1
    failure = None
    for candidate in range(DEFAULT_VIEWER_PORT, last + 1):
        try:
            return bind_server(candidate, host)
        except OSError as error:
            failure = error
    raise OSError(f'No free viewer port in {DEFAULT_VIEWER_PORT}..{last} on {host}; '
                  f'free one or request a port explicitly ({failure})')


def bind_loopback(port=0):
    """Explicit loopback socket for internal clients and test fixtures."""
    return bind_server(port, "127.0.0.1")


def serve(run_dir, *, port=None, host="0.0.0.0", auth="none", session_file=None, open_browser=False):
    require_web()
    from .web_files import read_json
    from .web_server import run_socket
    from .web_session import LocalSession

    root = Path(run_dir).resolve(strict=True)
    manifest = read_json(root, 'manifest.json')
    if not isinstance(manifest, dict) or not isinstance(manifest.get('run_id'), str):
        raise ValueError('Selected directory has no valid run manifest')
    if session_file is not None and Path(session_file).resolve().is_relative_to(root):
        raise ValueError('Viewer credentials must be stored outside the run directory')
    with tempfile.TemporaryDirectory(prefix='hypergan-viewer-') as private:
        credential_path = Path(session_file) if session_file is not None else Path(private) / 'session.json'
        with bind_available(port, host) as listener:
            session = LocalSession(listener.getsockname()[1], host=host, auth=auth)
            session.write_credentials(credential_path)
            # Uvicorn restores and re-raises SIGTERM after its graceful shutdown.
            # Translate that second delivery into Python unwinding so private
            # credential cleanup runs instead of the OS immediately terminating.
            previous_term = signal.getsignal(signal.SIGTERM)
            def terminate(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminate)
            try:
                print(json.dumps({'origin': session.origin, 'session_file': str(credential_path),
                                  'server_instance_id': session.instance_id}), flush=True)
                print(f'Viewer: {session.origin} (listening on {host})', file=sys.stderr, flush=True)
                if auth == "token":
                    print(f'Credentials: {credential_path} (copy its token into the sign-in form)',
                          file=sys.stderr, flush=True)
                if open_browser:
                    import webbrowser
                    webbrowser.open(session.origin)
                run_socket(root, listener, session)
            finally:
                credential_path.unlink(missing_ok=True)
                signal.signal(signal.SIGTERM, previous_term)

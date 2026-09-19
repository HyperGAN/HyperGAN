"""Explicit local viewer launch, keeping credentials outside experiment artifacts."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile


def require_web():
    missing = [name for name in ('starlette', 'uvicorn', 'wasmtime')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Local serving requires 'hypergan[web]' (missing " + ', '.join(missing) + ')')


def bind_loopback(port=0):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('port must be in 0..65535; zero selects an available port')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        return listener
    except BaseException:
        listener.close()
        raise


def serve(run_dir, *, port=0, session_file=None, open_browser=False):
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
        with bind_loopback(port) as listener:
            session = LocalSession(listener.getsockname()[1])
            session.write_credentials(credential_path)
            try:
                print(json.dumps({'origin': session.origin, 'session_file': str(credential_path),
                                  'server_instance_id': session.instance_id}), flush=True)
                print(f'Viewer: {session.origin}\nCredentials: {credential_path} (copy its token into the sign-in form)',
                      file=sys.stderr, flush=True)
                if open_browser:
                    import webbrowser
                    webbrowser.open(session.origin)
                run_socket(root, listener, session)
            finally:
                credential_path.unlink(missing_ok=True)

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
import socket
import subprocess
import hashlib
from types import SimpleNamespace
import tempfile
import threading
import time

from .ports import DEFAULT_VIEWER_PORT

TERMINAL_STATUSES = {'complete', 'stopped', 'failed', 'interrupted'}


def _windows_port_owner(port):
    """Read Windows' numeric TCP table without requiring an optional package."""
    try:
        table = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=3.)
        if table.returncode != 0:
            return None
        for line in table.stdout.splitlines():
            fields = line.split()
            if (len(fields) != 5 or fields[0].upper() != 'TCP'
                    or fields[1].rsplit(':', 1)[-1] != str(port)
                    or fields[2].rsplit(':', 1)[-1] != '0' or not fields[-1].isdigit()):
                continue
            # A listener has a zero remote port; avoid localized state names.
            pid = int(fields[-1])
            if pid <= 0:
                continue
            comm = 'unknown'
            try:
                import csv
                process = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                                         capture_output=True, text=True, timeout=3.)
                if process.returncode == 0:
                    for row in csv.reader(process.stdout.splitlines()):
                        if len(row) >= 2 and row[1] == str(pid):
                            comm = row[0]
                            break
            except (OSError, subprocess.SubprocessError, UnicodeError):
                pass
            return pid, comm
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    return None


def get_port_owner(port: int) -> tuple[int, str] | None:
    """Identify a TCP listener via the operating system's process table."""
    if not isinstance(port, int) or port <= 0:
        return None
    if os.name == 'nt':
        return _windows_port_owner(port)
    port_hex = f"{port:04X}"
    target_inodes = set()
    for net_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(net_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 10:
                        local_addr = parts[1]
                        st = parts[3]
                        if local_addr.endswith(":" + port_hex) and st == "0A":
                            target_inodes.add(parts[9])
        except (OSError, UnicodeError):
            pass

    if target_inodes:
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                fd_dir = f"/proc/{entry}/fd"
                try:
                    for fd in os.listdir(fd_dir):
                        try:
                            link = os.readlink(f"{fd_dir}/{fd}")
                            if any(link == f"socket:[{ino}]" for ino in target_inodes):
                                comm = ""
                                try:
                                    comm = Path(f"/proc/{entry}/comm").read_text(encoding="utf-8").strip()
                                except OSError:
                                    pass
                                return (int(entry), comm or "unknown")
                        except (OSError, UnicodeError):
                            pass
                except (OSError, PermissionError):
                    pass
        except OSError:
            pass

    # Fallback to ss command
    try:
        proc = subprocess.run(["ss", "-lptn", f"sport = :{port}"],
                              capture_output=True, text=True, timeout=1.0)
        if proc.returncode == 0:
            import re
            m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', proc.stdout)
            if m:
                return (int(m.group(2)), m.group(1))
            m2 = re.search(r'pid=(\d+)', proc.stdout)
            if m2:
                return (int(m2.group(1)), "unknown")
    except Exception:
        pass

    # Fallback to lsof command
    try:
        proc = subprocess.run(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpcn"],
                              capture_output=True, text=True, timeout=1.0)
        if proc.returncode == 0:
            lines = proc.stdout.splitlines()
            pid, comm = None, "unknown"
            for line in lines:
                if line.startswith("p") and line[1:].isdigit():
                    pid = int(line[1:])
                elif line.startswith("c"):
                    comm = line[1:]
            if pid is not None:
                return (pid, comm)
    except Exception:
        pass

    return None


def find_runs_root(path: Path | str | None = None, is_run: bool | None = None) -> Path:
    """Resolve the runs root directory for a given path.

    Checks parent directories for a workspace hypergan.toml defining runs_root,
    or falls back to the parent directory of a run (a directory containing manifest.json
    or a run directory under a runs folder), or the path itself.
    """
    if path is None:
        p = Path.cwd().resolve()
    else:
        p = Path(path).resolve()

    # 1. Walk up to find hypergan.toml workspace
    curr = p if not (p / "manifest.json").is_file() else p.parent
    while True:
        workspace_file = curr / "hypergan.toml"
        if workspace_file.is_file():
            try:
                import tomllib
                data = tomllib.loads(workspace_file.read_text(encoding="utf-8"))
                workspace_section = data.get("workspace", {})
                if isinstance(workspace_section, dict) and "runs_root" in workspace_section:
                    runs_root_val = workspace_section["runs_root"]
                    return (curr / runs_root_val).resolve()
            except Exception:
                pass
        parent = curr.parent
        if parent == curr:
            break
        curr = parent

    # 2. Check if explicitly marked as run or has manifest.json or does not exist yet
    if is_run is False:
        return p
    if is_run is True or (p / "manifest.json").is_file() or not p.exists():
        return p.parent

    # If p has subdirs containing manifest.json (directly or under p/runs), p is already runs_root:
    try:
        if p.is_dir() and (
            any((d / "manifest.json").is_file() for d in p.iterdir() if d.is_dir())
            or ((p / "runs").is_dir() and any((d / "manifest.json").is_file() for d in (p / "runs").iterdir() if d.is_dir()))
        ):
            return p
    except OSError:
        pass

    # If p.parent has subdirs with manifest.json, p is likely a run under p.parent:
    try:
        if p.parent.is_dir() and any((d / "manifest.json").is_file() for d in p.parent.iterdir() if d.is_dir()):
            return p.parent
    except OSError:
        pass

    return p


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


def _projection_state_mtime_ns(run_dir: Path) -> int | None:
    """Return latest mtime_ns of existing projection state files on disk, or None if missing."""
    from .event_views import MapSpec
    run_path = Path(run_dir).resolve()
    rev = MapSpec().revision
    contrib_file = run_path / 'views' / rev / 'contributions.jsonl'
    candidates = (
        contrib_file,
        run_path / 'views' / rev / 'projection.json',
        run_path / 'views' / 'default' / 'state.json',
        run_path / 'projections' / 'state.json',
    )
    if not contrib_file.is_file():
        return None
    try:
        if contrib_file.stat().st_size == 0:
            return None
        manifest_file = run_path / 'manifest.json'
        if manifest_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
            expected_rid = manifest.get('run_id') if isinstance(manifest, dict) else None
            if expected_rid:
                with contrib_file.open('r', encoding='utf-8') as f:
                    first_line = f.readline()
                if first_line:
                    row = json.loads(first_line)
                    src_rid = row.get('source', {}).get('run_id') if isinstance(row, dict) else None
                    if src_rid and src_rid != expected_rid:
                        return None
    except (OSError, ValueError, TypeError):
        return None
    mtimes: list[int] = []
    for p in candidates:
        try:
            if p.is_file():
                mtimes.append(p.stat().st_mtime_ns)
        except OSError:
            pass
    return max(mtimes) if mtimes else None


def is_projection_stale_or_missing(run_dir: Path | str) -> bool:
    """Return True if events.jsonl exists with events and projection view is missing or stale."""
    run_path = Path(run_dir).resolve()
    events_file = run_path / 'events.jsonl'
    try:
        if not events_file.is_file():
            return False
        events_stat = events_file.stat()
        if events_stat.st_size == 0:
            return False
    except OSError:
        return False

    proj_mtime = _projection_state_mtime_ns(run_path)
    if proj_mtime is None:
        return True
    return events_stat.st_mtime_ns > proj_mtime


def project_run(run_dir: Path | str, *, limit: int = 100) -> dict:
    """Project training events into view contributions for a single run directory."""
    from .event_views import MapSpec, Projector as EventProjector
    run_path = Path(run_dir).resolve()
    events_file = run_path / 'events.jsonl'
    if not run_path.is_dir() or not events_file.is_file():
        return {'documents': 0, 'has_more': False}
    last_progress = {'documents': 0, 'has_more': False}
    rev = MapSpec().revision
    view_dir = run_path / 'views' / rev
    try:
        with EventProjector(run_path, allow_device_change=True) as projector:
            while True:
                last_progress = projector.project(limit=limit)
                if not last_progress.get('has_more'):
                    break
    except ValueError as exc:
        msg = str(exc)
        if 'Stale event cursor' in msg or 'Projection source stream identity changed' in msg:
            for fname in ('contributions.jsonl', 'projection.json'):
                try:
                    (view_dir / fname).unlink(missing_ok=True)
                except OSError:
                    pass
            with EventProjector(run_path, allow_device_change=True) as projector:
                while True:
                    last_progress = projector.project(limit=limit)
                    if not last_progress.get('has_more'):
                        break
        else:
            raise
    try:
        ev_mtime = events_file.stat().st_mtime_ns
        proj_json = view_dir / 'projection.json'
        if proj_json.is_file() and proj_json.stat().st_mtime_ns < ev_mtime:
            os.utime(proj_json, ns=(ev_mtime, ev_mtime))
    except OSError:
        pass
    return last_progress


def ensure_run_projected(run_dir: Path | str) -> bool:
    """Lazily project run_dir on demand if its projection view is missing or stale."""
    run_path = Path(run_dir).resolve()
    if not is_projection_stale_or_missing(run_path):
        return False
    try:
        project_run(run_path)
        return True
    except Exception:
        return False


# Last projection failure per resolved run directory, so the viewer can answer
# with the reason instead of waiting for a projection the background projector
# cannot build (a read-only run, a failing projection). The broker runs the
# projector and the server in separate processes, so the failure is also kept
# as a small record in the private viewer cache, never in the run.
_PROJECTION_FAILURES: dict[Path, str] = {}
PROJECTION_RETRY_SECONDS = 30.0
_FAILURE_RECORDS = '.projection-failures'


def _failure_record(run_dir: Path) -> Path:
    from .web_service import view_cache_root
    name = hashlib.sha256(str(run_dir).encode()).hexdigest()[:32]
    return view_cache_root() / _FAILURE_RECORDS / (name + '.json')


def _record_projection_failure(run_dir: Path, error: str | None) -> None:
    """Remember (or with None, forget) why projecting `run_dir` failed, for any process."""
    if error is None:
        _PROJECTION_FAILURES.pop(run_dir, None)
    else:
        _PROJECTION_FAILURES[run_dir] = error
    try:
        record = _failure_record(run_dir)
        if error is None:
            record.unlink(missing_ok=True)
            return
        if record.parent.is_symlink() or record.parent.parent.is_symlink():
            return
        record.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = record.with_name(f'{record.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps({'run_dir': str(run_dir), 'error': error}), encoding='utf-8')
        os.replace(tmp, record)
    except OSError:
        pass


def projection_failure(run_dir: Path | str) -> str | None:
    """Why the viewer's background projector last failed on `run_dir`, if it did."""
    run_dir = Path(run_dir).resolve()
    failure = _PROJECTION_FAILURES.get(run_dir)
    if failure is not None:
        return failure
    try:
        record = _failure_record(run_dir)
        if record.is_symlink():
            return None
        data = json.loads(record.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get('run_dir') != str(run_dir):
        return None
    error = data.get('error')
    return error[:500] if isinstance(error, str) and error else None


def registered_run_dirs(root: Path | str) -> list[Path]:
    """Registered runs nested anywhere under `root` (never `root` itself or another root's)."""
    from . import registry as _reg_mod
    root = Path(root).resolve()
    try:
        data = _reg_mod._read_registry_unlocked(_reg_mod.get_registry_dir(root))
        entries = list(data.get('runs', {}).values())
    except (OSError, ValueError, AttributeError):
        return []
    dirs: list[Path] = []
    seen: set[Path] = set()
    for item in entries:
        rdir_val = item.get('run_dir') if isinstance(item, dict) else None
        if not rdir_val:
            continue
        try:
            p = Path(rdir_val).resolve()
            if (p not in seen and p != root and p.is_relative_to(root) and p.is_dir()
                    and ((p / 'manifest.json').is_file() or (p / 'summary.json').is_file())):
                seen.add(p)
                dirs.append(p)
        except (OSError, ValueError):
            continue
    return dirs


class Projector:
    """Background projector for every run under a root whose events.jsonl changed.

    A run is projected only when its event log's watermark moved since the last
    projection, whether or not it is training, so an idle persistent dashboard
    does no projection work and opens no projector.
    """

    def __init__(self, root: Path | str, *, poll_interval: float = 0.25):
        self.root = Path(root).resolve()
        self.runs_root = self.root
        self.poll_interval = poll_interval
        self._last_projected_watermark: dict[Path, tuple[int, int]] = {}
        self._last_failure: dict[Path, tuple[tuple[int, int] | None, float]] = {}
        # Runs whose failure record (possibly left by an earlier viewer) is known gone.
        self._cleared: set[Path] = set()

    def _candidate_runs(self) -> list[Path]:
        if (self.root / 'manifest.json').is_file() or (self.root / 'events.jsonl').is_file():
            return [self.root]
        search_dirs = [self.root]
        runs_sub = self.root / 'runs'
        if runs_sub.is_dir():
            search_dirs.append(runs_sub)
        candidates: list[Path] = []
        seen: set[Path] = set()
        for search_dir in search_dirs:
            try:
                for d in search_dir.iterdir():
                    if d.is_dir() and not d.name.startswith('.'):
                        if (d / 'manifest.json').is_file() or (d / 'events.jsonl').is_file():
                            resolved = d.resolve()
                            if resolved not in seen:
                                seen.add(resolved)
                                candidates.append(resolved)
            except OSError:
                pass
        # The same registered runs the dashboard serves, at any depth under the
        # root: a served run must have a projector, since requests never project.
        if not (self.root / 'summary.json').is_file():
            for d in registered_run_dirs(self.root):
                if d not in seen and (d / 'events.jsonl').is_file():
                    seen.add(d)
                    candidates.append(d)
        return sorted(candidates)

    def _should_project(self, run_dir: Path) -> tuple[bool, tuple[int, int] | None]:
        run_dir = run_dir.resolve()
        events_file = run_dir / 'events.jsonl'
        watermark: tuple[int, int] | None = None
        try:
            if events_file.is_file():
                st = events_file.stat()
                watermark = (st.st_mtime_ns, st.st_size)
        except OSError:
            watermark = None

        if watermark is None or watermark[1] == 0:
            return False, watermark

        last_wm = self._last_projected_watermark.get(run_dir)
        if last_wm is not None and last_wm == watermark:
            return False, watermark

        # A failed log is retried when it grows or after a pause, not every poll.
        failed = self._last_failure.get(run_dir)
        if (failed is not None and failed[0] == watermark
                and time.monotonic() - failed[1] < PROJECTION_RETRY_SECONDS):
            return False, watermark

        proj_mtime = _projection_state_mtime_ns(run_dir)
        if last_wm is None and proj_mtime is not None and proj_mtime >= watermark[0]:
            self._last_projected_watermark[run_dir] = watermark
            if run_dir not in self._cleared:
                _record_projection_failure(run_dir, None)
                self._cleared.add(run_dir)
            return False, watermark

        return True, watermark

    def poll_once(self) -> list[Path]:
        projected: list[Path] = []
        for run_dir in self._candidate_runs():
            should_proj, watermark = self._should_project(run_dir)
            if not should_proj:
                continue
            try:
                project_run(run_dir)
                try:
                    st = (run_dir / 'events.jsonl').stat()
                    watermark = (st.st_mtime_ns, st.st_size)
                except OSError:
                    pass
                if watermark is not None:
                    self._last_projected_watermark[run_dir] = watermark
                if self._last_failure.pop(run_dir, None) is not None or run_dir not in self._cleared:
                    _record_projection_failure(run_dir, None)
                    self._cleared.add(run_dir)
                projected.append(run_dir)
            except Exception as exc:
                self._last_failure[run_dir] = (watermark, time.monotonic())
                self._cleared.discard(run_dir)
                _record_projection_failure(run_dir, f'{type(exc).__name__}: {exc}'[:500])
        return projected

    def _loop(self, stop=None, *, max_iterations: int | None = None) -> None:
        iterations = 0
        while True:
            if (self.root / '.stop_projector').exists():
                break
            self.poll_once()
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            if stop is not None:
                if stop.wait(self.poll_interval):
                    break
            else:
                time.sleep(self.poll_interval)


def _has_sub_runs(root: Path) -> bool:
    if not (root / 'manifest.json').is_file() and any(
            (d / 'events.jsonl').is_file() for d in registered_run_dirs(root)):
        return True
    for search_dir in (root, root / 'runs'):
        try:
            if search_dir.is_dir() and any(
                (d / 'manifest.json').is_file() or (d / 'events.jsonl').is_file()
                for d in search_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ):
                return True
        except OSError:
            pass
    return False


def _project(root, stop):
    from .event_views import Projector as EventProjector
    from .web_files import read_json

    threading.Thread(target=_watch_parent, daemon=True).start()

    has_sub_runs = _has_sub_runs(root)

    if not has_sub_runs:
        while not (root / 'manifest.json').is_file():
            if stop.wait(0.1):
                return
            if (root / '.stop_projector').exists():
                return
            if _has_sub_runs(root):
                has_sub_runs = True
                break

    if not has_sub_runs and (root / 'manifest.json').is_file():
        with EventProjector(root, allow_device_change=True) as projector:
            drain_deadline = None
            while True:
                if stop.is_set() or (root / '.stop_projector').exists():
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + 1
                    if time.monotonic() >= drain_deadline:
                        return
                progress = projector.project(limit=100)
                if not progress['has_more']:
                    if (root / 'manifest.json').is_file():
                        try:
                            manifest = read_json(root, 'manifest.json')
                            if manifest.get('status') in TERMINAL_STATUSES:
                                projector.project(limit=100)
                                return
                        except (OSError, ValueError):
                            pass
                    if stop.wait(0.1) or (root / '.stop_projector').exists():
                        # One final bounded page catches the last training event.
                        projector.project(limit=100)
                        return
    else:
        bg_projector = Projector(root)
        bg_projector._loop(stop=stop)



def stop_projector(run_dir: Path | str) -> None:
    """Request projector shutdown for run_dir and release resources."""
    run_dir = Path(run_dir).resolve()
    try:
        (run_dir / '.stop_projector').touch(exist_ok=True)
    except OSError:
        pass
    runs_root = find_runs_root(run_dir)
    private = _registry(runs_root)
    state = _read_state(private)
    if state and isinstance(state.get('processes'), dict):
        proj_pid = state['processes'].get('projector')
        if proj_pid:
            try:
                os.kill(proj_pid, signal.SIGTERM)
            except OSError:
                pass


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
    credential_path = Path(state['session_file'])
    with _launch_lock(credential_path.parent):
        credentials = json.loads(credential_path.read_text(encoding='utf-8'))
    if credentials['server_instance_id'] != state['server_instance_id']:
        raise ValueError('Viewer credentials do not match the registered incarnation')
    public_origin = credentials.get('public_origin')
    return SimpleNamespace(host=credentials['origin'].removeprefix('http://'),
                           origin=credentials['origin'], instance_id=credentials['server_instance_id'],
                           _token=credentials.get('token', ''), auth_mode=credentials['auth_mode'],
                           bind_host=credentials['bind_host'], public_origin=public_origin,
                           browser_origin=public_origin or credentials['origin'])


def _cleanup_session(private, launch_id):
    # Match the credential reader's lock, so Windows deletion never races our
    # own open handle. External programs holding the file remain outside it.
    with _launch_lock(private):
        (private / ('session-' + launch_id + '.json')).unlink(missing_ok=True)
        (private / ('stop-' + launch_id)).unlink(missing_ok=True)


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
        recorded = set()
        for subdir in _manifest_roots(root):
            try:
                recorded.update(_receipt(subdir, session, mode, status, children))
            except OSError:
                pass
        return recorded
    observations = root / 'observations'
    if observations.is_symlink():
        raise ValueError('Viewer observations directory cannot be a symlink')
    observations.mkdir(exist_ok=True)
    atomic_json(observations / ('viewer-' + session.instance_id + '.json'),
        {'schema_version': 1, 'mode': mode, 'status': status,
         'server_instance_id': session.instance_id, 'origin': session.origin,
         'public_origin': session.public_origin,
         'auth_mode': session.auth_mode, 'bind_host': session.bind_host,
         'processes': {'supervisor': os.getpid(), **{name: child.pid for name, child in children.items()}}})
    return {root}


def _manifest_roots(root):
    if (root / 'manifest.json').is_file():
        return {root}
    roots = set()
    for search_dir in (root, root / 'runs'):
        try:
            if search_dir.is_dir():
                for child in search_dir.iterdir():
                    if child.is_dir() and (child / 'manifest.json').is_file():
                        roots.add(child)
        except OSError:
            pass
    return roots


def _broker(root, private, launch_id):
    from .run_state import atomic_json, run_lock
    from .web_launch import bind_available
    from .web_session import LocalSession

    from contextlib import ExitStack
    context = mp.get_context('spawn')
    stop = _StopFlag(context)
    children = {}
    receipted_roots = set()
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
            if not state or state['launch_id'] != launch_id or state['status'] == 'stopped':
                return
        # Metadata publication can fail after children are already reaped.
        # Credential cleanup must run regardless, before the lifetime lock exits.
        ownership.callback(_cleanup_session, private, launch_id)
        def publish(status, error=None):
            state.update(status=status, error=error, supervisor_pid=os.getpid(),
                         processes={name: child.pid for name, child in children.items()})
            if session is not None:
                state.update(origin=session.origin, public_origin=session.public_origin,
                             server_instance_id=session.instance_id,
                             session_file=str(credential_path))
            with _launch_lock(private):
                atomic_json(private / 'state.json', state)
            if session is not None:
                try:
                    receipted_roots.update(_receipt(root, session, state['mode'], status, children))
                except (OSError, ValueError) as exc:
                    print(f'Viewer receipt unavailable: {exc}', file=sys.stderr, flush=True)
        try:
            listener = bind_available(state['port'], state['host'])
            session = LocalSession(listener.getsockname()[1], host=state['host'], auth=state['auth'],
                                   public_origin=state.get('public_origin'))
            with _launch_lock(private):
                session.write_credentials(credential_path)
            for name, target, args in [('server', _server, (root, listener, session, stop)),
                                      ('projector', _project, (root, stop))]:
                child = context.Process(target=target, args=args, name='hypergan-' + name)
                child.start()
                children[name] = child
            listener.close()
            publish('starting')
            deadline = time.monotonic() + 10
            while not stop.wait(.1):
                if (private / ('stop-' + launch_id)).exists():
                    break
                failed = []
                for name, child in children.items():
                    if not child.is_alive():
                        if name == 'projector' and child.exitcode == 0:
                            continue
                        failed.append(name)
                if failed:
                    publish('failed', 'Viewer process exited: ' + ', '.join(failed))
                    break
                if state['status'] == 'starting':
                    if _probe(session):
                        publish('ready')
                    elif time.monotonic() >= deadline:
                        publish('failed', 'Viewer did not become ready within 10 seconds')
                        break
                if session is not None:
                    for run_root in _manifest_roots(root) - receipted_roots:
                        try:
                            receipted_roots.update(_receipt(
                                run_root, session, state['mode'], state['status'], children))
                        except (OSError, ValueError):
                            pass
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


def _select_registry(root):
    p = Path(root).resolve()
    runs_root = find_runs_root(p)
    private = _registry(runs_root)
    if runs_root != p:
        candidate = _registry(p)
        if (candidate / 'state.json').is_file():
            c_state = _read_state(candidate)
            r_state = _read_state(private) if (private / 'state.json').is_file() else None
            if c_state and (_running(candidate, c_state) or not r_state or not _running(private, r_state)):
                return candidate
    if not (private / 'state.json').is_file():
        extras = []
        if (p / 'runs').is_dir():
            extras.append((p / 'runs').resolve())
        if p.name == 'runs' and p.parent != p:
            extras.append(p.parent.resolve())
        for extra in extras:
            cand = _registry(extra)
            if (cand / 'state.json').is_file():
                return cand
    return private


def stop_viewer(root, *, launch_id=None, timeout=8):
    """Stop only the registered incarnation; never signal a PID from a file."""
    from .run_state import atomic_json
    private = _select_registry(root)
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
        if not current or current['launch_id'] != launch_id or (not _locked(private) and current['status'] != 'starting'):
            return {'status': 'stopped', 'server_instance_id': state.get('server_instance_id')}
        time.sleep(.05)
    raise RuntimeError('Viewer shutdown timed out; inspect ' + str(private / 'viewer.log'))


def viewer_status(root):
    """Read launch/connection information without starting a viewer or training."""
    private = _select_registry(root)
    state = _read_state(private)
    if state is None:
        return {'status': 'not_running'}
    result = dict(state, log_file=str(private / 'viewer.log'))
    if not _running(private, state) and state['status'] not in {'stopped', 'failed'}:
        result['status'] = 'not_running'
    return result


def probe_dashboard(host: str = "127.0.0.1", port: int = DEFAULT_VIEWER_PORT,
                    auth_token: str | None = None, timeout: float = 0.5) -> dict | None:
    """Probe dashboard on host:port for capabilities.

    Returns capabilities dict if running HyperGAN dashboard,
    None if port is free / connection refused,
    or dict with {'alien': True} if occupied by non-HyperGAN process.
    """
    import http.client
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        conn = http.client.HTTPConnection(connect_host, port, timeout=timeout)
        headers = {}
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        conn.request('GET', '/api/v1/capabilities', headers=headers)
        res = conn.getresponse()
        body = res.read(65536)
        conn.close()
    except ConnectionRefusedError:
        return None
    except OSError as exc:
        import errno
        if getattr(exc, 'errno', None) == errno.ECONNREFUSED:
            return None
        return {'alien': True}
    except Exception:
        return {'alien': True}

    if res.status == 200:
        try:
            data = json.loads(body.decode('utf-8'))
            if isinstance(data, dict) and ('api_version' in data or 'server_instance_id' in data):
                return data
        except Exception:
            pass
        return {'alien': True}
    elif res.status == 401:
        return {'status': 'auth_required', 'auth_required': True}
    else:
        return {'alien': True}


def launch_dashboard_daemon(runs_root: Path, port: int = DEFAULT_VIEWER_PORT,
                            host: str = "127.0.0.1", auth: str = "none",
                            public_origin: str | None = None) -> None:
    from .run_state import atomic_json
    from .web_session import normalize_public_origin
    import secrets

    runs_root = Path(runs_root).resolve()
    private = _registry(runs_root)
    public_origin = normalize_public_origin(public_origin)
    with _launch_lock(private):
        state = _read_state(private, locked=True)
        if _running(private, state):
            return
        if state:
            (private / ('session-' + state['launch_id'] + '.json')).unlink(missing_ok=True)
        launch_id = secrets.token_hex(16)
        state = {'schema_version': 1, 'launch_id': launch_id,
                 'root': str(runs_root), 'host': host, 'auth': auth, 'port': port,
                 'public_origin': public_origin, 'mode': 'dashboard',
                 'status': 'starting', 'started_at': time.time()}
        atomic_json(private / 'state.json', state)
        options = {'start_new_session': True} if os.name != 'nt' else {
            'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        with (private / 'viewer.log').open('ab') as log:
            subprocess.Popen([
                sys.executable, '-m', 'hypergan.web_autostart',
                str(runs_root), str(private), launch_id
            ], stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True, **options)


def _registered_dashboard(runs_root: Path) -> tuple[dict | None, str | None]:
    """This root's running registered viewer and its bearer token, if any.

    The token comes from the private session file, so probing a token-mode
    dashboard reads its real identity instead of a bare 401.
    """
    try:
        private = _registry(runs_root)
        state = _read_state(private)
        if not state or not _running(private, state):
            return None, None
    except (OSError, ValueError, KeyError, RuntimeError):
        return None, None
    token = None
    if state.get('session_file') and state.get('server_instance_id'):
        try:
            token = _session(state)._token or None
        except (OSError, ValueError, KeyError, RuntimeError):
            token = None
    return state, token


def _stop_registered_dashboard(runs_root: Path, state: dict) -> None:
    stop_viewer(runs_root, launch_id=state.get('launch_id'))


def dashboard_url(host: str, port: int, run_id: str | None = None) -> str:
    """The local link to print: loopback for a wildcard or loopback bind, else the bind address."""
    from .web_session import WILDCARD_HOSTS, _is_loopback
    name = 'localhost' if host in WILDCARD_HOSTS or _is_loopback(host) else host
    url = f"http://{'[' + name + ']' if ':' in name else name}:{port}"
    return f"{url}/runs/{run_id}" if run_id else url


def ensure_dashboard_running(runs_root: Path | str, *, port: int = DEFAULT_VIEWER_PORT,
                             host: str = "127.0.0.1", auth: str = "none",
                             public_origin: str | None = None,
                             check_version: bool = True, run_id: str | None = None) -> str:
    """Ensure a persistent root dashboard for runs_root is serving `port`, and return its URL.

    - Probes the port on loopback, with this root's registered credential when it has one.
    - A dashboard for another root, or one this root did not register, is never
      stopped: that is reported as an error naming how to stop it.
    - This root's own dashboard is restarted when its version, assets or commit
      differ from the current checkout; otherwise it is reused.
    - With no server on the port, a background dashboard daemon is started.
    """
    runs_root = Path(runs_root).resolve()
    if (runs_root / 'manifest.json').is_file() and run_id is None:
        ensure_run_projected(runs_root)
        run_id = runs_root.name
        runs_root = runs_root.parent
    elif run_id is not None:
        if (runs_root / str(run_id)).is_dir():
            ensure_run_projected(runs_root / str(run_id))
        elif (runs_root / 'runs' / str(run_id)).is_dir():
            ensure_run_projected(runs_root / 'runs' / str(run_id))

    state, token = _registered_dashboard(runs_root)
    probed = probe_dashboard("127.0.0.1", port, auth_token=token)
    if probed is not None and probed.get('alien'):
        owner = get_port_owner(port)
        if owner:
            pid, comm = owner
            raise OSError(f"Port {port} is already in use by PID {pid} ({comm}); "
                          f"specify a different root or stop conflicting process")
        raise OSError(f"Port {port} is already in use; specify a different root or stop conflicting process")
    if probed is None:
        owner = get_port_owner(port)
        if owner:
            pid, comm = owner
            raise OSError(f"Port {port} is already in use by PID {pid} ({comm}); "
                          f"specify a different root or stop conflicting process")
    elif probed.get('auth_required'):
        raise OSError(f"Port {port} is serving a token-protected dashboard that this root did not start; "
                      f"stop it with 'hypergan stop-server --all' or choose another port")
    else:
        remote_root = probed.get('runs_root')
        if remote_root and Path(remote_root).resolve() != runs_root:
            raise OSError(f"Port {port} is already running a dashboard for root '{remote_root}'; "
                          f"stop it with 'hypergan view {remote_root} --stop' or specify a different root.")
        if auth != 'none' and probed.get('auth_mode') != auth:
            # Reusing it would silently serve this run without the requested auth.
            raise OSError(f"Port {port} already serves this root's dashboard with --auth "
                          f"{probed.get('auth_mode')}; stop it with 'hypergan view {runs_root} --stop' "
                          f"to start it with --auth {auth}")
        reasons = _dashboard_mismatch(probed) if check_version else []
        if reasons:
            own = state is not None and probed.get('server_instance_id') == state.get('server_instance_id')
            if not own:
                raise OSError(f"Port {port} is serving an outdated dashboard ({', '.join(reasons)}) that this "
                              f"command cannot restart; stop it with 'hypergan view {runs_root} --stop' and retry")
            print(f"Restarting dashboard on port {port} ({', '.join(reasons)})...", file=sys.stderr, flush=True)
            _stop_registered_dashboard(runs_root, state)
            deadline = time.monotonic() + 5.0
            while probe_dashboard("127.0.0.1", port) is not None and time.monotonic() < deadline:
                time.sleep(0.1)
            probed = None

    if probed is None:
        launch_dashboard_daemon(runs_root, port=port, host=host, auth=auth, public_origin=public_origin)
        deadline = time.monotonic() + 10.0
        while True:
            state, token = _registered_dashboard(runs_root)
            probed = probe_dashboard("127.0.0.1", port, auth_token=token)
            if probed and not probed.get('alien') and not probed.get('auth_required'):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Dashboard failed to start on port {port} within 10 seconds")
            time.sleep(0.1)
        remote_root = probed.get('runs_root')
        if remote_root and Path(remote_root).resolve() != runs_root:
            raise OSError(f"Port {port} was taken by a dashboard for root '{remote_root}' while this one started")

    return dashboard_url(host, port, run_id)


def _dashboard_mismatch(probed: dict) -> list[str]:
    from . import __version__
    from .web_assets import compute_asset_hash
    from .provenance import hypergan_source

    local_hash = compute_asset_hash()
    local_commit = hypergan_source().get('hypergan_commit')
    reasons = []
    if probed.get('version') != __version__:
        reasons.append(f"version {probed.get('version')} != {__version__}")
    if local_hash != 'unknown' and probed.get('asset_hash') and probed.get('asset_hash') != local_hash:
        reasons.append(f"asset_hash {probed.get('asset_hash', '')[:8]} != {local_hash[:8]}")
    if local_commit and probed.get('git_commit') and probed.get('git_commit') != local_commit:
        reasons.append(f"git_commit {probed.get('git_commit', '')[:8]} != {local_commit[:8]}")
    return reasons


def _resolve_dashboard_target(runs_root: Path | str | None = None) -> Path:
    """Resolve a runs root directory, run directory, slug, tag, or short ID to a concrete filesystem Path."""
    if hasattr(runs_root, "path"):
        runs_root = getattr(runs_root, "path")
    if runs_root is None or str(runs_root).strip() == "" or str(runs_root) == "None":
        return find_runs_root(Path.cwd())
    raw_path = Path(runs_root)
    if not raw_path.exists():
        from .registry import resolve_run_target
        try:
            raw_path = resolve_run_target(str(runs_root), allow_latest=(str(runs_root) == "latest"))
        except Exception:
            pass
    return raw_path


def stop_dashboard(runs_root: Path | str | None = None, timeout: float = 8.0) -> dict:
    target = _resolve_dashboard_target(runs_root)
    if not target.exists():
        return {"status": "not_running"}
    return stop_viewer(target.resolve(), timeout=timeout)


def dashboard_status(runs_root: Path | str | None = None) -> dict:
    target = _resolve_dashboard_target(runs_root)
    if not target.exists():
        return {"status": "not_running"}
    info = viewer_status(target.resolve())
    if info.get("status") == "ready":
        port = info.get("port") or DEFAULT_VIEWER_PORT
        try:
            token = _session(info)._token or None
        except (OSError, ValueError, KeyError, RuntimeError):
            token = None
        probed = probe_dashboard("127.0.0.1", port, auth_token=token)
        if probed and not probed.get("alien"):
            info.update({
                "version": probed.get("version"),
                "git_commit": probed.get("git_commit"),
                "asset_hash": probed.get("asset_hash"),
                "run_count": probed.get("run_count", 0),
            })
    return info


def _process_tokens(proc_root: str, pid: int) -> list[str] | None:
    try:
        raw = Path(f'{proc_root}/{pid}/cmdline').read_bytes()
    except OSError:
        return None
    return [token.decode('utf-8', errors='ignore') for token in raw.split(b'\0') if token]


def _parent_pid(proc_root: str, pid: int) -> int | None:
    try:
        # The command name may contain spaces or parentheses; fields follow the last ')'.
        return int(Path(f'{proc_root}/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _is_viewer_command(tokens: list[str]) -> bool:
    """True for `python -m hypergan.web_autostart|web_server` and `hypergan view|serve`."""
    if any(mod in tokens for mod in ('hypergan.web_autostart', 'hypergan.web_server')):
        return True
    if len(tokens) < 2:
        return False
    exe = Path(tokens[0]).name
    if not (exe == 'hypergan' or (exe.startswith('python') and len(tokens) >= 3 and tokens[1] == '-m'
                                  and tokens[2] in ('hypergan', 'hypergan.cli'))):
        return False
    subcmd = next((tok for tok in tokens[1:] if not tok.startswith('-') and tok not in ('hypergan', 'hypergan.cli')), None)
    return subcmd in ('view', 'serve')


def stop_all_servers(force: bool = False) -> list[int]:
    """Terminate every viewer, dashboard and supervisor of the current user on this host.

    A PID is signalled only when it is known to be a viewer process now: a
    supervisor whose registry lifetime lock is held, that supervisor's own
    children, or a process whose command line is a viewer command. A PID left
    in a stopped or stale state file can belong to anything after reuse
    (a training job, for one), so it is never signalled.
    """
    current_pid = os.getpid()
    current_uid = os.getuid() if hasattr(os, 'getuid') else None
    proc_root = getattr(os, 'environ', {}).get('HYPERGAN_PROC_ROOT', '/proc')
    has_proc = os.path.exists(proc_root)
    candidates: set[int] = set()

    # 1. Registered viewers whose broker or server still holds its lifetime lock.
    tmp_dir = Path(tempfile.gettempdir())
    registries = []
    for viewers_dir in tmp_dir.glob('hypergan-viewers-*'):
        try:
            registries.extend(reg for reg in viewers_dir.iterdir() if reg.is_dir())
        except OSError:
            pass
    for reg in registries:
        try:
            state = json.loads((reg / 'state.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not isinstance(state, dict):
            continue
        launch_id = state.get('launch_id')
        if launch_id:
            try:
                (reg / f'stop-{launch_id}').touch(mode=0o600)
            except OSError:
                pass
        try:
            locked = _locked(reg)
        except (OSError, RuntimeError):
            locked = False
        if not locked:
            continue
        supervisor = state.get('supervisor_pid')
        if type(supervisor) is not int or supervisor == current_pid:
            continue
        candidates.add(supervisor)
        processes = state.get('processes', {})
        for child in (processes.values() if isinstance(processes, dict) else ()):
            if type(child) is int and child != current_pid and (
                    not has_proc or child == supervisor or _parent_pid(proc_root, child) == supervisor):
                candidates.add(child)

    # 2. Viewer commands in the host process table owned by the current user.
    if has_proc:
        try:
            for entry in os.listdir(proc_root):
                if not entry.isdigit() or int(entry) == current_pid:
                    continue
                try:
                    if current_uid is not None and os.stat(f'{proc_root}/{entry}').st_uid != current_uid:
                        continue
                except OSError:
                    continue
                tokens = _process_tokens(proc_root, int(entry))
                if tokens and _is_viewer_command(tokens):
                    candidates.add(int(entry))
        except OSError:
            pass

    # 3. Terminate candidate processes gracefully first (SIGTERM), then forcibly (SIGKILL)
    active_killed = []
    for pid in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
            active_killed.append(pid)
        except OSError:
            pass

    # Windows SIGTERM already uses TerminateProcess and has no SIGKILL.
    if active_killed and os.name != 'nt':
        time.sleep(0.5)
        for pid in active_killed:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    # 4. Clean up registry directories
    for reg in registries:
        try:
            for f in reg.glob('session-*.json'):
                f.unlink(missing_ok=True)
            for f in reg.glob('stop-*'):
                f.unlink(missing_ok=True)
            (reg / 'state.json').unlink(missing_ok=True)
        except OSError:
            pass

    return sorted(active_killed)


class Viewer:
    """Attach to one persistent per-run supervisor; optional startup stays async."""

    def __init__(self, root, *, port=None, host=None, auth=None, mode='auto', open_browser=False,
                 public_origin=None):
        from .run_state import atomic_json, run_lock
        from .web_launch import bind_available, require_web
        from .web_session import normalize_public_origin
        import secrets

        require_web()
        # Refuse an unusable proxy origin here, in the command the owner typed,
        # rather than inside a detached supervisor whose failure is a log line.
        public_origin = normalize_public_origin(public_origin)
        self.root = Path(root).resolve()
        self.runs_root = find_runs_root(root)
        target_port = DEFAULT_VIEWER_PORT if port is None else port
        self.private = _registry(self.runs_root)
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
                    (public_origin is not None and public_origin != state.get('public_origin')) or
                    (target_port and target_port != int(state.get('origin', ':0').rsplit(':', 1)[1]) and target_port != state['port'])):
                    raise ValueError('Existing viewer has different bind/auth settings; use hypergan stop-server RUN first')
            else:
                if state:
                    # An abruptly killed broker cannot unlink credentials. They
                    # cannot authenticate a new incarnation; remove this stale
                    # private file when the next launcher takes over.
                    (self.private / ('session-' + state['launch_id'] + '.json')).unlink(missing_ok=True)
                host, auth = host or '127.0.0.1', auth or 'none'
                # Fail unusable ports before numerical imports. The broker
                # repeats the same selection, so a port lost in between is
                # re-searched for a default request and an actionable error
                # for an explicitly requested one.
                with bind_available(target_port, host):
                    pass
                if auth not in {'none', 'token'}:
                    raise ValueError("auth must be 'none' or 'token'")
                state = {'schema_version': 1, 'launch_id': secrets.token_hex(16),
                         'root': str(self.runs_root), 'host': host, 'auth': auth, 'port': target_port,
                         'public_origin': public_origin,
                         'mode': mode, 'status': 'starting', 'started_at': time.time()}
                atomic_json(self.private / 'state.json', state)
                options = {'start_new_session': True} if os.name != 'nt' else {
                    'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
                with (self.private / 'viewer.log').open('ab') as log:
                    try:
                        self.process = subprocess.Popen([sys.executable, '-m', 'hypergan.web_autostart',
                            str(self.runs_root), str(self.private), state['launch_id']], stdin=subprocess.DEVNULL,
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
                        if session.public_origin:
                            print(f'Public viewer: {session.public_origin} (through your own TLS proxy)',
                                  file=sys.stderr, flush=True)
                        if session.auth_mode == 'token':
                            print(f'Credentials: {self.credential_path} (copy its token into the sign-in form)',
                                  file=sys.stderr, flush=True)
                        if self._open_browser:
                            import webbrowser
                            threading.Thread(target=webbrowser.open, args=(session.browser_origin,), daemon=True).start()
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
def training_viewer(root, *, required=False, port=None, host=None, auth=None, open_browser=False,
                    public_origin=None, config=None):
    if config and isinstance(config, dict) and config.get('run', {}).get('viewer') is False:
        yield None
        return
    if not required:
        from importlib.util import find_spec
        if any(find_spec(name) is None for name in ('starlette', 'uvicorn')):
            yield None
            return
    viewer = None
    try:
        try:
            viewer = Viewer(root, port=port, host=host, auth=auth, public_origin=public_origin,
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

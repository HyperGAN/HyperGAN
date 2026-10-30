"""Multi-GPU job queue scheduler, runner daemon, and systemd service generator.

Advisory file locking ensures safety across concurrent CLI invocations and runner daemons.
Zero torch/particlegan dependencies: can be run and queried from any Python environment.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

from ._file_lock import lock_file as _lock_file, unlock_file as _unlock_file
from .summary import atomic_write_json
from .workspace import find_workspace, load_workspace, resolve_device_aliases


def find_queue_file(target: Path | str | None = None) -> Path:
    """Resolve queue file location via hierarchy: explicit, env var, workspace, runs_root, cwd."""
    if target is not None:
        p = Path(target).resolve()
        if p.is_file() or p.suffix == ".json":
            return p
        if p.is_dir():
            return p / ".hypergan" / "queue.json"
        return p

    if os.environ.get("HYPERGAN_QUEUE_FILE"):
        return Path(os.environ["HYPERGAN_QUEUE_FILE"]).resolve()

    ws_path = find_workspace(Path.cwd())
    if ws_path:
        return ws_path.parent / ".hypergan" / "queue.json"

    try:
        from .web_autostart import find_runs_root
        runs_root = find_runs_root(Path.cwd())
        if runs_root:
            return runs_root / ".hypergan" / "queue.json"
    except Exception:
        pass

    return Path.cwd() / ".hypergan" / "queue.json"


@contextmanager
def queue_lock(queue_path: Path):
    """Advisory file lock protecting queue reads and writes across processes."""
    lock_path = queue_path.with_name(queue_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    held = _held_queue_locks()
    key = os.path.abspath(queue_path)
    try:
        _lock_file(fd, blocking=True)
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)
            _unlock_file(fd)
    finally:
        os.close(fd)


_lock_state = threading.local()


def _held_queue_locks() -> set:
    """Queue files whose lock this thread holds."""
    if not hasattr(_lock_state, "held"):
        _lock_state.held = set()
    return _lock_state.held


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_queue(next_job_id: int = 1) -> dict:
    return {
        "schema_version": 1,
        "next_job_id": next_job_id,
        "stop_generation": 0,
        "jobs": [],
    }


def _next_free_job_number(logs_dir: Path) -> int:
    """First job number whose log file does not exist yet, so new jobs never append to old logs."""
    highest = 0
    try:
        for p in logs_dir.glob("job-*.log"):
            m = re.fullmatch(r"job-(\d+)\.log", p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    except OSError:
        pass
    return highest + 1


def load_queue(queue_path: Path) -> dict:
    """Load queue state from disk or initialize empty state.

    An unreadable queue file is moved aside to ``queue.json.corrupt-<timestamp>``
    (never overwritten) and a fresh queue is saved in its place, so job numbering
    resumes after the newest job log even when the caller only reads. Only a caller
    holding ``queue_lock`` moves it; an unlocked reader just sees an empty queue, so
    it can never move aside a fresh queue a locked writer has just saved. A missing
    queue file also resumes numbering after the newest job log.
    """
    if not queue_path.is_file():
        return _empty_queue(_next_free_job_number(queue_path.parent / "logs"))
    try:
        with queue_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as exc:
        reason = f"invalid JSON: {exc}"
    else:
        if isinstance(data, dict) and isinstance(data.get("jobs"), list):
            if data.get("schema_version") == 1:
                return data
            raise ValueError(
                f"Queue file {queue_path} has unsupported schema_version {data.get('schema_version')!r}; "
                "it was written by a different HyperGAN version"
            )
        reason = "not a queue document"
    if os.path.abspath(queue_path) not in _held_queue_locks():
        return _empty_queue(_next_free_job_number(queue_path.parent / "logs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    aside = queue_path.with_name(f"{queue_path.name}.corrupt-{stamp}")
    os.replace(queue_path, aside)
    fresh = _empty_queue(_next_free_job_number(queue_path.parent / "logs"))
    save_queue(queue_path, fresh)
    print(
        f"[queue] warning: {queue_path} is unreadable ({reason}); moved it to {aside.name} and started an empty queue.",
        file=sys.stderr,
    )
    return fresh


def save_queue(queue_path: Path, data: dict) -> None:
    """Persist queue state atomically to disk."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(queue_path, data)


def _find_job(q: dict, job_id: str) -> Optional[dict]:
    return next((j for j in q.get("jobs", []) if j.get("id") == job_id), None)


# Process identity: a bare PID is reused after the process exits or the host reboots,
# so jobs record (pid, host, boot id, start time) and are only trusted or signalled
# while all of them still match. The start time is platform specific: clock ticks
# since boot on Linux (hence the boot id), and an absolute creation time on macOS
# and Windows, so a PID recycled after a reboot never matches there either.

def _boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def _linux_start_time(pid: int) -> Optional[int]:
    """Field 22 of /proc/<pid>/stat: clock ticks since boot."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # The command name in field 2 may contain spaces and ')'.
    return int(stat.rsplit(")", 1)[1].split()[19])


def _macos_start_time(pid: int) -> Optional[int]:
    """kinfo_proc.kp_proc.p_starttime (a struct timeval at offset 0), in microseconds."""
    import ctypes
    import ctypes.util
    import struct

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.sysctl.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t,
    ]
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
    size = ctypes.c_size_t(1024)  # sizeof(struct kinfo_proc) is 648 on 64-bit macOS
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0) != 0 or size.value < 16:
        # A PID without a process succeeds with an empty result.
        return None
    sec, usec = struct.unpack_from("=qi", buf.raw, 0)
    return sec * 1_000_000 + usec


def _windows_start_time(pid: int) -> Optional[int]:
    """GetProcessTimes creation time (100 ns FILETIME units) of a still-running process."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        # An exited process can still be opened while anyone (e.g. its Popen) holds a handle.
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


def _process_start_time(pid: int) -> Optional[int]:
    """Start time of a running process, or None when it is gone or this OS cannot tell."""
    try:
        if sys.platform.startswith("linux"):
            return _linux_start_time(pid)
        if sys.platform == "darwin":
            return _macos_start_time(pid)
        if os.name == "nt":
            return _windows_start_time(pid)
    except (OSError, AttributeError, IndexError, ValueError, OverflowError, TypeError):
        pass
    return None


def _process_identity(pid: int) -> dict:
    return {
        "pid": pid,
        "host": socket.gethostname(),
        "boot_id": _boot_id(),
        "start_time": _process_start_time(pid),
    }


def _same_host(identity) -> bool:
    return isinstance(identity, dict) and identity.get("host") == socket.gethostname()


def _identity_alive(identity) -> Optional[bool]:
    """Whether the recorded process still runs: True/False, or None when this host cannot tell."""
    if not _same_host(identity) or not isinstance(identity.get("pid"), int):
        return None
    pid = identity["pid"]
    boot_id = _boot_id()
    if identity.get("boot_id") and boot_id and identity["boot_id"] != boot_id:
        return False
    if identity.get("start_time") is not None:
        return _process_start_time(pid) == identity["start_time"]
    if os.name == "nt":
        # No start time was recorded, and os.kill(pid, 0) terminates the process on Windows.
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user, so not a process this queue started.
        return False
    except (OSError, OverflowError):
        return None
    return True


def _signal_job_process(job: dict, sig: int) -> bool:
    """Signal a job's trainer only while it is still the process the runner started.

    A trainer on this host whose identity cannot be checked (or a legacy job without
    one) is signalled by PID as before; one on another host never is. On Windows,
    os.kill(pid, SIGINT) terminates the process.
    """
    pid = job.get("pid")
    if not isinstance(pid, int):
        return False
    identity = job.get("process")
    if identity is not None:
        alive = _identity_alive(identity)
        if alive is False or (alive is None and not _same_host(identity)):
            return False
    try:
        os.kill(pid, sig)
    except (OSError, OverflowError):
        return False
    return True


def _interrupt_child(proc) -> None:
    """Stop the runner's own trainer: one SIGINT, or TerminateProcess on Windows, which has no SIGINT for it."""
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGINT)
    except (OSError, AttributeError, ValueError):
        pass


def _reconcile_orphans(q: dict) -> bool:
    """Mark running jobs whose runner died (crash, SIGKILL, reboot) as interrupted.

    A trainer that outlived its runner keeps the job running until it exits too.
    Jobs claimed on another host, or by a runner without a recorded identity, are left alone.
    """
    changed = False
    for j in q.get("jobs", []):
        if j.get("status") != "running" or _identity_alive(j.get("runner")) is not False:
            continue
        if _identity_alive(j.get("process")) is True:
            continue
        j["status"] = "interrupted"
        j["pid"] = None
        j["ended_at"] = _now()
        j["error"] = f"Queue runner (PID {j['runner']['pid']}) exited while the job was running"
        changed = True
    return changed


def _load_reconciled(queue_path: Path) -> dict:
    """load_queue plus orphan reconciliation; call with the queue lock held."""
    q = load_queue(queue_path)
    if _reconcile_orphans(q):
        save_queue(queue_path, q)
    return q


def _peek_stop_generation(queue_path: Path) -> Optional[int]:
    """Read stop_generation without the lock (writes are atomic replaces)."""
    try:
        with queue_path.open("r", encoding="utf-8") as f:
            value = json.load(f).get("stop_generation", 0)
        return value if isinstance(value, int) else None
    except (OSError, ValueError, AttributeError):
        return None


def _normalize_devices(devices, workspace_dict: dict | None = None) -> Optional[frozenset]:
    """Canonical device set for runner/job matching: aliases resolved, 'cuda:N' == 'N'."""
    if devices is None:
        return None
    items = devices.split(",") if isinstance(devices, str) else [str(d) for d in devices]
    items = [d.strip() for d in items if str(d).strip()]
    if not items:
        return None
    if not any(d.lower() == "auto" for d in items):
        try:
            resolved = resolve_device_aliases(items, workspace_dict)
            items = [str(d).strip() for d in resolved if str(d).strip()]
        except ValueError:
            pass
    canonical = set()
    for d in items:
        low = d.lower()
        if low.startswith("cuda:") and low[5:].isdigit():
            d = str(int(low[5:]))
        elif low.isdigit():
            d = str(int(low))
        elif low in ("cpu", "auto"):
            d = low
        canonical.add(d)
    return frozenset(canonical)


def _devices_compatible(runner_devices: Optional[frozenset], job_devices: Optional[frozenset]) -> bool:
    return runner_devices is None or job_devices is None or runner_devices == job_devices


def inspect_gpu_hardware() -> list[dict]:
    """Query GPU device inventory via provider without importing PyTorch."""
    from .gpu_inventory import inspect_gpu_hardware as _inv_hardware
    return _inv_hardware()


def inspect_gpu_processes() -> list[dict]:
    """Query active compute applications per GPU via provider."""
    from .gpu_inventory import inspect_gpu_processes as _inv_processes
    return _inv_processes()


def check_device_availability(
    devices: str | list[str] | None,
    workspace_dict: dict | None = None,
    min_free_mb: int = 2048
) -> tuple[bool, Optional[str]]:
    """Verify GPU free memory and ensure no conflicting hypergan/train compute processes."""
    if devices is None:
        return True, None

    resolved = resolve_device_aliases(devices, workspace_dict)
    dev_list = [d.strip() for d in (resolved if isinstance(resolved, list) else resolved.split(",")) if d.strip()]

    if not dev_list or (len(dev_list) == 1 and dev_list[0].lower() == "cpu"):
        return True, None

    hw = inspect_gpu_hardware()
    if not hw:
        # If nvidia-smi is not available, assume compute is available (e.g. CPU or CI container)
        return True, None

    hw_by_uuid = {g["uuid"]: g for g in hw}
    hw_by_index = {str(g["index"]): g for g in hw}

    target_gpus = []
    for d in dev_list:
        if d in hw_by_uuid:
            target_gpus.append(hw_by_uuid[d])
        elif d in hw_by_index:
            target_gpus.append(hw_by_index[d])
        elif d.startswith("cuda:"):
            idx = d.split(":")[1]
            if idx in hw_by_index:
                target_gpus.append(hw_by_index[idx])

    if not target_gpus:
        # Passthrough if no matching GPU found by index/UUID
        return True, None

    procs = inspect_gpu_processes()

    for g in target_gpus:
        if g["memory_free"] < min_free_mb:
            return False, f"GPU {g['index']} ({g['uuid']}) free memory ({g['memory_free']:.0f} MiB) < threshold ({min_free_mb} MiB)"

        for p in procs:
            if p["gpu_uuid"] == g["uuid"]:
                cmd = (p.get("cmdline") or "").lower()
                pname = (p.get("process_name") or "").lower()
                if "hypergan" in cmd or "hypergan" in pname or "train" in cmd:
                    return False, f"GPU {g['index']} ({g['uuid']}) is busy running process {p['pid']} ({p.get('process_name')})"

    return True, None


def queue_add(
    model_file: Path | str,
    devices: str | None = None,
    run_dir: Path | str | None = None,
    name: str | None = None,
    steps: int | None = None,
    sets: list[str] | dict | None = None,
    no_web: bool = False,
    wait: bool = False,
    queue_file: Path | str | None = None,
    **kwargs,
) -> dict:
    """Enqueue a training experiment or wait directive."""
    qpath = find_queue_file(queue_file)
    with queue_lock(qpath):
        q = load_queue(qpath)
        job_id = f"job-{q.get('next_job_id', 1):04d}"
        q["next_job_id"] = q.get("next_job_id", 1) + 1

        # Normalize sets to list of strings
        set_items: list[str] = []
        if sets:
            if isinstance(sets, dict):
                set_items = [f"{k}={v}" for k, v in sets.items()]
            elif isinstance(sets, (list, tuple)):
                set_items = [str(x) for x in sets]
            else:
                set_items = [str(sets)]

        no_web_flag = bool(no_web or kwargs.get("no_server", False))

        if wait:
            kind = "wait"
            target_run_dir = str(Path(model_file).resolve())
            job_name = name or f"wait-{Path(model_file).name}"
            m_file = None
        else:
            kind = "experiment"
            m_path = Path(model_file).resolve()
            m_file = str(m_path)
            job_name = name or m_path.stem
            target_run_dir = str(Path(run_dir).resolve()) if run_dir else None

        log_path = str(qpath.parent / "logs" / f"{job_id}.log")
        job = {
            "id": job_id,
            "name": job_name,
            "custom_name": name,
            "kind": kind,
            "model_file": m_file,
            "run_dir": target_run_dir,
            "devices": devices,
            "steps": int(steps) if steps is not None else None,
            "sets": set_items,
            "no_web": no_web_flag,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "ended_at": None,
            "pid": None,
            "exit_code": None,
            "log_file": log_path,
            "error": None,
        }
        q.setdefault("jobs", []).append(job)
        save_queue(qpath, q)
        return job


def queue_ls(
    status_filter: str | None = None,
    queue_file: Path | str | None = None,
) -> list[dict]:
    """List jobs from the queue with optional status filtering."""
    qpath = find_queue_file(queue_file)
    with queue_lock(qpath):
        q = _load_reconciled(qpath)
        jobs = q.get("jobs", [])
        if status_filter:
            jobs = [j for j in jobs if j.get("status") == status_filter]
        return list(jobs)


def queue_status(
    job_id: str | None = None,
    queue_file: Path | str | None = None,
    lines: int = 20,
) -> dict:
    """Inspect active job progress, summary metrics, and log tail."""
    qpath = find_queue_file(queue_file)
    with queue_lock(qpath):
        q = _load_reconciled(qpath)
        jobs = q.get("jobs", [])
        target = None
        if job_id:
            target = next((j for j in jobs if j.get("id") == job_id), None)
            if not target:
                raise ValueError(f"Job '{job_id}' not found in queue {qpath}")
        else:
            running = [j for j in jobs if j.get("status") == "running"]
            if running:
                target = running[0]
            else:
                queued = [j for j in jobs if j.get("status") == "queued"]
                if queued:
                    target = queued[0]
                elif jobs:
                    target = jobs[-1]

    info = {
        "queue_file": str(qpath),
        "jobs_total": len(jobs),
        "active_job": target,
        "summary": None,
        "log_tail": [],
    }
    if target and target.get("run_dir"):
        sum_path = Path(target["run_dir"]) / "summary.json"
        if sum_path.is_file():
            try:
                from .summary import read_summary_json
                info["summary"] = read_summary_json(target["run_dir"])
            except Exception:
                pass
    if target and target.get("log_file"):
        l_path = Path(target["log_file"])
        if l_path.is_file():
            try:
                with l_path.open("r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                    info["log_tail"] = [line.rstrip() for line in all_lines[-lines:]]
            except Exception:
                pass
    return info


def queue_stop(
    job_id: str | None = None,
    stop_all: bool = False,
    queue_file: Path | str | None = None,
) -> dict:
    """Cancel queued jobs or signal running jobs to halt.

    Each running trainer receives exactly one SIGINT (a second one forces an
    exit without the final checkpoint). ``stop_all`` also bumps the queue's
    ``stop_generation``, which halts every runner that was started before it.
    """
    qpath = find_queue_file(queue_file)
    if stop_all:
        with queue_lock(qpath):
            q = _load_reconciled(qpath)
            count = 0
            signalled = 0
            for j in q.get("jobs", []):
                if j.get("status") == "queued":
                    j["status"] = "cancelled"
                    j["ended_at"] = _now()
                    j["error"] = "Cancelled by queue stop --all"
                    count += 1
                elif j.get("status") == "running" and j.get("pid") and not j.get("stop_signalled"):
                    # Flag only delivered signals, so the runner of a trainer this
                    # host could not reach (another host) still signals it itself.
                    if _signal_job_process(j, signal.SIGINT):
                        j["stop_signalled"] = True
                        signalled += 1
            q["stop_generation"] = int(q.get("stop_generation", 0)) + 1
            q["stop_requested_at"] = _now()
            save_queue(qpath, q)

        return {
            "stopped_all": True,
            "cancelled_count": count,
            "signalled_count": signalled,
            "stop_generation": q["stop_generation"],
            "message": (
                f"Cancelled {count} queued job(s), asked running queue runners to halt, "
                f"and signaled {signalled} running process(es)."
            ),
        }

    if not job_id:
        raise ValueError("Must provide job_id or specify stop_all=True")

    with queue_lock(qpath):
        q = _load_reconciled(qpath)
        job = _find_job(q, job_id)
        if not job:
            raise ValueError(f"Job '{job_id}' not found in queue")

        if job.get("status") == "queued":
            job["status"] = "cancelled"
            job["ended_at"] = _now()
            job["error"] = "Cancelled by user"
            save_queue(qpath, q)
            return {"job_id": job_id, "status": "cancelled", "message": f"Cancelled queued job {job_id}"}

        if job.get("status") == "running":
            pid = job.get("pid")
            already = bool(job.get("stop_signalled"))
            job["status"] = "cancelled"
            job["ended_at"] = _now()
            job["error"] = "Stopped by user"
            # Without a pid the runner has claimed but not launched the job; it
            # signals the trainer itself once it records the pid and sees this.
            signalled = False if already else _signal_job_process(job, signal.SIGINT)
            if signalled:
                job["stop_signalled"] = True
            save_queue(qpath, q)
            if signalled:
                message = f"Signaled running job {job_id} (PID {pid})"
            elif pid and not already and job.get("process") is not None and not _same_host(job["process"]):
                message = f"Marked job {job_id} cancelled; its process (PID {pid}) runs on another host and was not signaled"
            elif pid and not already:
                message = f"Marked job {job_id} cancelled; its process (PID {pid}) is no longer running"
            else:
                message = f"Marked job {job_id} cancelled"
            return {"job_id": job_id, "status": "cancelled", "signalled": signalled, "message": message}

        return {"job_id": job_id, "status": job.get("status"), "message": f"Job {job_id} already has terminal status '{job.get('status')}'"}


def queue_clear(queue_file: Path | str | None = None) -> dict:
    """Cancel all queued jobs and signal runner to halt (alias for queue_stop(stop_all=True))."""
    return queue_stop(stop_all=True, queue_file=queue_file)


def _extract_run_dir_from_log(log_path: Path | str) -> Optional[str]:
    """Scan training process log output for reported run directory."""
    log_p = Path(log_path)
    if not log_p.is_file():
        return None
    try:
        with log_p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Run directory:"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        candidate = parts[1].strip()
                        if candidate:
                            cand_path = Path(candidate)
                            if cand_path.is_dir() or cand_path.parent.is_dir():
                                return str(cand_path.resolve())
    except Exception:
        pass
    return None


def _find_run_dir_from_registry(pid: int | None, job_name: str | None) -> Optional[str]:
    """Inspect registry for active or recent run matching pid or name."""
    if pid is None and not job_name:
        return None
    try:
        from .registry import _read_registry_unlocked, get_registry_dir
        reg_dir = get_registry_dir()
        if reg_dir.is_dir():
            data = _read_registry_unlocked(reg_dir)
            for path_str, entry in data.get("runs", {}).items():
                if pid and entry.get("pid") == pid:
                    return path_str
                if job_name and entry.get("name") == job_name:
                    return path_str
    except Exception:
        pass
    return None


def _run_is_active(run_dir: Path | str) -> bool:
    """A run is active only while its trainer holds .run.lock (or a remote host reports it live).

    Manifest status alone is not enough: a SIGKILLed trainer leaves ``running`` behind.
    """
    from .inspection import check_run_active
    try:
        return bool(check_run_active(run_dir)[0])
    except Exception:
        return False


def _update_job(queue_path: Path, job_id: str, **fields) -> Optional[dict]:
    """Merge fields into one job under the queue lock; returns a copy of the job as saved."""
    with queue_lock(queue_path):
        q = load_queue(queue_path)
        j = _find_job(q, job_id)
        if j is None:
            return None
        j.update(fields)
        save_queue(queue_path, q)
        return dict(j)


def queue_run(
    devices: str | None = None,
    poll_interval: float = 5.0,
    once: bool = False,
    no_web: bool = False,
    queue_file: Path | str | None = None,
    json_output: bool = False,
    **kwargs,
) -> int:
    """Daemon scheduler loop executing queued experiments sequentially in subprocesses."""
    qpath = find_queue_file(queue_file)
    logs_dir = qpath.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stop_file = qpath.with_name(qpath.name + ".stop")

    ws_path = find_workspace(qpath.parent)
    ws_data = load_workspace(ws_path) if ws_path else {}
    working_dir = ws_path.parent if ws_path else qpath.parent.parent
    runner_devices = _normalize_devices(devices, ws_data)
    runner_identity = _process_identity(os.getpid())

    with queue_lock(qpath):
        q = _load_reconciled(qpath)
        # Only `queue stop --all` issued after this runner started halts it.
        baseline_generation = int(q.get("stop_generation", 0))

    stop_requested = False
    had_failure = False
    executed_jobs: list[dict] = []

    def _sig_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    def _halt_requested() -> bool:
        # A manual stop file is consumed so it halts one runner pass, not every later `queue run`.
        if stop_file.is_file():
            try:
                stop_file.unlink()
            except FileNotFoundError:
                pass
            if not json_output:
                print(f"[queue] Stop file '{stop_file.name}' detected; halting runner.", file=sys.stderr)
            return True
        generation = _peek_stop_generation(qpath)
        if generation is not None and generation > baseline_generation:
            if not json_output:
                print("[queue] Queue stop requested; halting runner.", file=sys.stderr)
            return True
        return False

    prev_sigint = signal.signal(signal.SIGINT, _sig_handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

    try:
        while not stop_requested:
            if _halt_requested():
                return 0

            target_job = None
            with queue_lock(qpath):
                q = _load_reconciled(qpath)
                for j in q.get("jobs", []):
                    if j.get("status") == "queued":
                        # If runner is bound to specific devices, pick only jobs targeting or compatible with them
                        if not _devices_compatible(runner_devices, _normalize_devices(j.get("devices"), ws_data)):
                            continue
                        target_job = dict(j)
                        break

            if target_job is None:
                if once:
                    if json_output:
                        print(json.dumps({
                            "queue_file": str(qpath),
                            "total_executed": len(executed_jobs),
                            "executed": executed_jobs,
                            "status": "failed" if had_failure else "completed",
                        }, indent=2))
                    return 1 if had_failure else 0
                time.sleep(poll_interval)
                continue

            job_id = target_job["id"]
            job_devices = target_job.get("devices") or devices

            if target_job.get("kind") == "wait":
                if _run_is_active(target_job["run_dir"]):
                    # Still waiting
                    time.sleep(poll_interval)
                    continue

                # Finished waiting (unless the wait was cancelled meanwhile)
                with queue_lock(qpath):
                    q = load_queue(qpath)
                    j = _find_job(q, job_id)
                    finished = j is not None and j.get("status") == "queued"
                    if finished:
                        j["status"] = "completed"
                        j["ended_at"] = _now()
                        save_queue(qpath, q)
                if finished:
                    executed_jobs.append({"id": job_id, "name": target_job.get("name"), "status": "completed", "exit_code": 0, "run_dir": target_job.get("run_dir")})
                continue

            # Hardware preflight check
            avail, reason = check_device_availability(job_devices, ws_data)
            if not avail:
                if not json_output:
                    print(f"[queue] Hardware busy for {job_id}: {reason}. Retrying in {poll_interval:.1f}s...", file=sys.stderr)
                time.sleep(poll_interval)
                continue

            # Run lock collision check
            if target_job.get("run_dir"):
                rdir = Path(target_job["run_dir"])
                if _run_is_active(rdir):
                    if not json_output:
                        print(f"[queue] Run directory {rdir} locked by active process. Retrying in {poll_interval:.1f}s...", file=sys.stderr)
                    time.sleep(poll_interval)
                    continue

            # Claim: the job must still be queued (not cancelled, not taken by another runner)
            with queue_lock(qpath):
                q = load_queue(qpath)
                j = _find_job(q, job_id)
                claimed = j is not None and j.get("status") == "queued"
                if claimed:
                    j["status"] = "running"
                    j["started_at"] = _now()
                    j["runner"] = runner_identity
                    j["stop_signalled"] = False
                    save_queue(qpath, q)
                    target_job = dict(j)
            if not claimed:
                if not json_output:
                    print(f"[queue] {job_id} was cancelled or claimed by another runner; skipping.", file=sys.stderr)
                continue

            log_path = Path(target_job["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Both aliases refer to the live log inode. Windows may require
            # elevated privileges for symlinks; a hard link does not.
            current_log = logs_dir / "current.log"
            try:
                if current_log.is_symlink() or current_log.is_file():
                    current_log.unlink()
                try:
                    current_log.symlink_to(log_path.name)
                except OSError:
                    if os.name != 'nt':
                        raise
                    log_path.touch(exist_ok=True)
                    os.link(log_path, current_log)
            except Exception:
                pass

            cmd = [sys.executable, "-m", "hypergan", "train", str(target_job["model_file"])]
            if job_devices:
                cmd.extend(["--devices", str(job_devices)])
            if target_job.get("run_dir"):
                cmd.extend(["--run-dir", str(target_job["run_dir"])])

            custom_name = target_job.get("custom_name") or target_job.get("name")
            if custom_name and target_job.get("kind") != "wait":
                cmd.extend(["--name", str(custom_name)])

            if target_job.get("steps") is not None:
                cmd.extend(["--steps", str(target_job["steps"])])

            sets_list = target_job.get("sets") or target_job.get("set_overrides") or []
            if isinstance(sets_list, dict):
                sets_list = [f"{k}={v}" for k, v in sets_list.items()]
            for s_item in sets_list:
                cmd.extend(["-s", str(s_item)])

            if target_job.get("no_web") or target_job.get("no_server") or no_web:
                cmd.append("--no-web")

            if not json_output:
                print(f"[queue] Starting {job_id} ({target_job['name']}): {' '.join(cmd)}")
            try:
                with log_path.open("ab") as log_fp:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(working_dir),
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                    )
            except OSError as exc:
                had_failure = True
                _update_job(qpath, job_id, status="failed", ended_at=_now(), error=f"Failed to launch trainer: {exc}")
                executed_jobs.append({"id": job_id, "name": target_job.get("name"), "status": "failed", "exit_code": None, "run_dir": target_job.get("run_dir")})
                continue

            def _signal_once() -> None:
                """SIGINT the trainer unless someone already did: a second SIGINT forces os._exit."""
                with queue_lock(qpath):
                    q = load_queue(qpath)
                    j = _find_job(q, job_id)
                    already = j is not None and bool(j.get("stop_signalled"))
                    if j is not None and not already:
                        j["stop_signalled"] = True
                        save_queue(qpath, q)
                if not already:
                    _interrupt_child(proc)

            saved = _update_job(qpath, job_id, pid=proc.pid, process=_process_identity(proc.pid))
            signalled = saved is not None and saved.get("status") != "running"
            if signalled:
                # Cancelled between claim and launch: `queue stop` had no pid to signal.
                _interrupt_child(proc)

            halting = False
            detected_run_dir = target_job.get("run_dir")
            exit_code = None
            while exit_code is None and not stop_requested:
                try:
                    exit_code = proc.wait(timeout=0.5)
                except TypeError:
                    # Mock objects in tests without timeout argument
                    exit_code = proc.wait()
                except subprocess.TimeoutExpired:
                    pass

                if exit_code is None and not halting and _halt_requested():
                    halting = True
                if halting and not signalled:
                    signalled = True
                    _signal_once()

                if not detected_run_dir:
                    discovered = _extract_run_dir_from_log(log_path)
                    if discovered:
                        detected_run_dir = discovered
                        _update_job(qpath, job_id, run_dir=detected_run_dir)

            if exit_code is None:
                if stop_requested and not signalled:
                    signalled = True
                    _signal_once()
                try:
                    exit_code = proc.wait(timeout=10.0)
                except (subprocess.TimeoutExpired, TypeError):
                    try:
                        proc.kill()
                    except (OSError, ProcessLookupError, AttributeError):
                        pass
                    try:
                        exit_code = proc.wait(timeout=5.0)
                    except (subprocess.TimeoutExpired, TypeError):
                        exit_code = proc.poll()

            if not detected_run_dir:
                detected_run_dir = (
                    _extract_run_dir_from_log(log_path)
                    or _find_run_dir_from_registry(proc.pid, target_job.get("name"))
                )

            if exit_code != 0:
                had_failure = True

            with queue_lock(qpath):
                q = load_queue(qpath)
                for j in q.get("jobs", []):
                    if j.get("id") == job_id:
                        j["pid"] = None
                        j["exit_code"] = exit_code
                        j["ended_at"] = _now()
                        if detected_run_dir:
                            j["run_dir"] = detected_run_dir
                        if j.get("status") == "running":
                            j["status"] = "completed" if exit_code == 0 else "failed"
                            if exit_code != 0:
                                j["error"] = f"Process exited with code {exit_code}"
                save_queue(qpath, q)

            executed_jobs.append({
                "id": job_id,
                "name": target_job.get("name"),
                "status": "completed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "run_dir": detected_run_dir,
            })

            if not json_output:
                print(f"[queue] Finished {job_id} with exit code {exit_code}")

            if halting:
                return 0

        return 0
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)


def install_service(
    user: bool = True,
    devices: str | None = None,
    name: str | None = None,
    dry_run: bool = False,
) -> tuple[Path, str]:
    """Generate and install systemd user service unit for hypergan queue."""
    service_name = name or ("hypergan-queue" if not devices else f"hypergan-queue-{devices.replace(',', '-')}")
    ws_path = find_workspace(Path.cwd())
    working_dir = ws_path.parent if ws_path else Path.cwd()

    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_path = unit_dir / f"{service_name}.service"

    dev_flag = f" --devices {devices}" if devices else ""
    desc_dev = f" ({devices})" if devices else ""

    content = f"""[Unit]
Description=HyperGAN Job Queue Runner{desc_dev}
After=network.target

[Service]
Type=simple
WorkingDirectory={working_dir}
ExecStart={sys.executable} -m hypergan queue run{dev_flag}
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    instructions = f"""Installed systemd service: {unit_path}

To reload systemd and enable service:
  systemctl --user daemon-reload
  systemctl --user enable --now {service_name}

To check status and live logs:
  systemctl --user status {service_name}
  journalctl --user -u {service_name} -f
"""
    if dry_run:
        instructions = f"Generated unit file ({unit_path}):\n{content}\n" + instructions
    else:
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(content, encoding="utf-8")

    return unit_path, instructions


def format_queue_table(jobs: list[dict]) -> str:
    """Format queue jobs into a clean CLI table."""
    if not jobs:
        return "Queue is empty."

    headers = ["ID", "NAME", "DEVICES", "STATUS", "PID/EXIT", "CREATED"]
    rows = []
    for j in jobs:
        pid_or_exit = "-"
        if j.get("status") == "running" and j.get("pid"):
            pid_or_exit = f"PID:{j['pid']}"
        elif j.get("exit_code") is not None:
            pid_or_exit = f"code:{j['exit_code']}"

        created = j.get("created_at", "")
        if "T" in created:
            created = created.split("T")[1][:8]

        dev_str = j.get("devices") or "-"
        rows.append([
            j.get("id", "-"),
            j.get("name", "-"),
            dev_str,
            j.get("status", "-"),
            pid_or_exit,
            created,
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)))

    return "\n".join(lines)


def format_queue_status(status_info: dict) -> str:
    """Format detailed status for an active or targeted job."""
    target = status_info.get("active_job")
    if not target:
        return f"Queue Store: {status_info.get('queue_file')}\nTotal Jobs: {status_info.get('jobs_total')}\nNo active or queued jobs."

    lines = [
        f"Queue Store:    {status_info.get('queue_file')}",
        f"Total Jobs:     {status_info.get('jobs_total')}",
        f"Job ID:         {target.get('id')}",
        f"Name:           {target.get('name')}",
        f"Status:         {target.get('status')}",
        f"Devices:        {target.get('devices') or '-'}",
        f"Model File:     {target.get('model_file') or '-'}",
        f"Run Directory:  {target.get('run_dir') or '-'}",
    ]
    if target.get("pid"):
        lines.append(f"PID:            {target['pid']}")
    if target.get("exit_code") is not None:
        lines.append(f"Exit Code:      {target['exit_code']}")
    if target.get("error"):
        lines.append(f"Error:          {target['error']}")

    summary = status_info.get("summary")
    if summary and isinstance(summary, dict):
        step = summary.get("step")
        total_steps = summary.get("total_steps")
        # Keys as written by summary.write_summary_json.
        speed = summary.get("steps_per_sec")
        eta = summary.get("eta_seconds")
        metrics = summary.get("latest") or {}

        progress_str = f"Step {step}/{total_steps}" if total_steps else f"Step {step}"
        if speed:
            progress_str += f" ({speed:.2f} it/s)"
        if eta:
            eta_min = eta / 60
            progress_str += f", ETA {eta_min:.1f}m"
        lines.append(f"Progress:       {progress_str}")

        if metrics:
            metric_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
            lines.append(f"Metrics:        {', '.join(metric_strs[:6])}")

    log_tail = status_info.get("log_tail", [])
    if log_tail:
        lines.append("\nLog Tail:")
        lines.append("---------")
        lines.extend(log_tail)

    return "\n".join(lines)

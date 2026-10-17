"""Multi-GPU job queue scheduler, runner daemon, and systemd service generator.

Advisory file locking ensures safety across concurrent CLI invocations and runner daemons.
Zero torch/particlegan dependencies: can be run and queried from any Python environment.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
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
    try:
        _lock_file(fd, blocking=True)
        try:
            yield
        finally:
            _unlock_file(fd)
    finally:
        os.close(fd)


def load_queue(queue_path: Path) -> dict:
    """Load queue state from disk or initialize empty state."""
    if not queue_path.is_file():
        return {
            "schema_version": 1,
            "next_job_id": 1,
            "jobs": [],
        }
    try:
        with queue_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("schema_version") == 1 and isinstance(data.get("jobs"), list):
            return data
    except Exception:
        pass
    return {
        "schema_version": 1,
        "next_job_id": 1,
        "jobs": [],
    }


def save_queue(queue_path: Path, data: dict) -> None:
    """Persist queue state atomically to disk."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(queue_path, data)


def inspect_gpu_hardware() -> list[dict]:
    """Query GPU device inventory via nvidia-smi without importing PyTorch."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total": float(parts[3]),
                    "memory_free": float(parts[4]),
                    "memory_used": float(parts[5]),
                    "utilization": float(parts[6]),
                })
        return gpus
    except Exception:
        return []


def inspect_gpu_processes() -> list[dict]:
    """Query active compute applications per GPU via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        procs = []
        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                pid = int(parts[1])
                cmdline = ""
                cmd_path = Path(f"/proc/{pid}/cmdline")
                if cmd_path.exists():
                    try:
                        cmdline = cmd_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
                    except Exception:
                        pass
                procs.append({
                    "gpu_uuid": parts[0],
                    "pid": pid,
                    "process_name": parts[2],
                    "used_memory": float(parts[3]),
                    "cmdline": cmdline,
                })
        return procs
    except Exception:
        return []


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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
        q = load_queue(qpath)
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
        q = load_queue(qpath)
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
    """Cancel queued jobs or signal running jobs to halt."""
    qpath = find_queue_file(queue_file)
    if stop_all:
        with queue_lock(qpath):
            q = load_queue(qpath)
            count = 0
            running_pids = []
            for j in q.get("jobs", []):
                if j.get("status") == "queued":
                    j["status"] = "cancelled"
                    j["ended_at"] = datetime.now(timezone.utc).isoformat()
                    j["error"] = "Cancelled by queue stop --all"
                    count += 1
                elif j.get("status") == "running" and j.get("pid"):
                    running_pids.append(j["pid"])
            stop_file = qpath.with_name(qpath.name + ".stop")
            stop_file.touch()
            save_queue(qpath, q)

        for pid in running_pids:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass

        return {
            "stopped_all": True,
            "cancelled_count": count,
            "stop_file": str(stop_file),
            "message": f"Cancelled {count} queued job(s), wrote stop file, and signaled running process(es).",
        }

    if not job_id:
        raise ValueError("Must provide job_id or specify stop_all=True")

    with queue_lock(qpath):
        q = load_queue(qpath)
        job = next((j for j in q.get("jobs", []) if j.get("id") == job_id), None)
        if not job:
            raise ValueError(f"Job '{job_id}' not found in queue")

        if job.get("status") == "queued":
            job["status"] = "cancelled"
            job["ended_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = "Cancelled by user"
            save_queue(qpath, q)
            return {"job_id": job_id, "status": "cancelled", "message": f"Cancelled queued job {job_id}"}

        if job.get("status") == "running":
            pid = job.get("pid")
            if pid:
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            job["status"] = "cancelled"
            job["ended_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = "Stopped by user"
            save_queue(qpath, q)
            return {"job_id": job_id, "status": "cancelled", "message": f"Signaled running job {job_id} (PID {pid})"}

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

    stop_requested = False
    had_failure = False
    executed_jobs: list[dict] = []

    def _sig_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _sig_handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

    try:
        while not stop_requested:
            if stop_file.is_file():
                if not json_output:
                    print(f"[queue] Stop file '{stop_file.name}' detected; halting runner.", file=sys.stderr)
                return 0

            target_job = None
            with queue_lock(qpath):
                q = load_queue(qpath)
                for j in q.get("jobs", []):
                    if j.get("status") == "queued":
                        # If runner is bound to specific devices, pick only jobs targeting or compatible with them
                        if devices is not None and j.get("devices") and j["devices"] != devices:
                            continue
                        target_job = j
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

            # Hardware preflight check
            avail, reason = check_device_availability(job_devices, ws_data)
            if not avail:
                if not json_output:
                    print(f"[queue] Hardware busy for {job_id}: {reason}. Retrying in {poll_interval:.1f}s...", file=sys.stderr)
                time.sleep(poll_interval)
                continue

            if target_job.get("kind") == "wait":
                target_rd = Path(target_job["run_dir"])
                manifest_path = target_rd / "manifest.json"
                is_running = True
                if manifest_path.is_file():
                    try:
                        with manifest_path.open("r", encoding="utf-8") as f:
                            man = json.load(f)
                        if man.get("status") not in ("running", "initializing"):
                            is_running = False
                    except Exception:
                        pass
                else:
                    is_running = False

                if is_running:
                    # Still waiting
                    time.sleep(poll_interval)
                    continue

                # Finished waiting
                with queue_lock(qpath):
                    q = load_queue(qpath)
                    for j in q.get("jobs", []):
                        if j.get("id") == job_id:
                            j["status"] = "completed"
                            j["ended_at"] = datetime.now(timezone.utc).isoformat()
                    save_queue(qpath, q)
                executed_jobs.append({"id": job_id, "name": target_job.get("name"), "status": "completed", "exit_code": 0, "run_dir": target_job.get("run_dir")})
                continue

            # Run lock collision check
            if target_job.get("run_dir"):
                rdir = Path(target_job["run_dir"])
                lock_file = rdir / ".run.lock"
                if lock_file.is_file():
                    is_locked = False
                    try:
                        fd = os.open(str(lock_file), os.O_RDWR)
                        try:
                            _lock_file(fd)
                            _unlock_file(fd)
                        except (BlockingIOError, OSError):
                            is_locked = True
                        finally:
                            os.close(fd)
                    except Exception:
                        pass
                    if is_locked:
                        if not json_output:
                            print(f"[queue] Run directory {rdir} locked by active process. Retrying in {poll_interval:.1f}s...", file=sys.stderr)
                        time.sleep(poll_interval)
                        continue

            # Transition to running state under lock
            log_path = Path(target_job["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with queue_lock(qpath):
                q = load_queue(qpath)
                for j in q.get("jobs", []):
                    if j.get("id") == job_id:
                        j["status"] = "running"
                        j["started_at"] = datetime.now(timezone.utc).isoformat()
                save_queue(qpath, q)

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
            with log_path.open("ab") as log_fp:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(working_dir),
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )

            with queue_lock(qpath):
                q = load_queue(qpath)
                for j in q.get("jobs", []):
                    if j.get("id") == job_id:
                        j["pid"] = proc.pid
                save_queue(qpath, q)

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

                if stop_file.is_file():
                    try:
                        proc.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass

                if not detected_run_dir:
                    discovered = _extract_run_dir_from_log(log_path)
                    if discovered:
                        detected_run_dir = discovered
                        with queue_lock(qpath):
                            q = load_queue(qpath)
                            for j in q.get("jobs", []):
                                if j.get("id") == job_id:
                                    j["run_dir"] = detected_run_dir
                            save_queue(qpath, q)

            if exit_code is None:
                exit_code = proc.wait()

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
                        j["ended_at"] = datetime.now(timezone.utc).isoformat()
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
        speed = summary.get("speed_it_per_sec")
        eta = summary.get("eta_seconds")
        metrics = summary.get("latest_metrics", {})

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

#!/usr/bin/env python3
"""Crash-proof Google Colab GPU benchmark & evaluation orchestrator for HyperGAN.

Built from battle-tested Google Colab CLI operational guarantees:
1. Isolated per-worker `--config <worker>.colab.json` files + staggered VM boots
   (`--stagger-seconds 20`) prevent `sessions.json.lock` contention and Colab 503 bursts.
2. Two-layer teardown: local `try/finally` + `SIGINT`/`SIGTERM`/`atexit` + durable
   session tracking under `runs/colab/sessions/` stops every VM immediately on finish,
   error, or Ctrl-C; VM-side detached `watchdog.py` calls
   `POST http://$TBE_RUNTIME_ADDR/unassign` if the local controller disconnects,
   training stalls (`--stall-minutes`), or `--max-minutes` expires.
3. OS-level process-group hard timeout (`SIGTERM -> SIGKILL`) wraps every `colab` CLI
   subprocess so wedged websocket threads can never leak idle GPU credits.
4. Hard `--max-workers` cap of 3 (refuses `> 3` unless `--allow-over-cap` is passed).
5. Disconnect resilience for long 50k-100k runs: optional chunked local sync
   (`--chunk-steps`), background periodic checkpoint pull-back (`--sync-interval-minutes`),
   optional Google Drive mirror (`--drive-dir`), and automatic `hypergan resume` from
   local synced checkpoints if a Colab runtime resets or preempts.
6. Seamless local Observatory integration: unpacks `summary.json`, `manifest.json`,
   `model.json`, `events.jsonl`, `metrics/`, `previews/`, EMA `model.pt`, `train.log`,
   and checkpoints into `<runs-dir>/<run_name>/` and registers the run locally.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import queue as stdlib_queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Empirically measured Colab CU/hr rates & VRAM for cost planning & summary logging.
COLAB_GPU_SPECS: dict[str, dict[str, Any]] = {
    "cpu": {"cuda_model": None, "vram_gb": 0.0, "cu_per_hr": 0.07, "colab_flag": None},
    "T4": {"cuda_model": "Tesla T4", "vram_gb": 15.6, "cu_per_hr": 1.15, "colab_flag": "T4"},
    "L4": {"cuda_model": "NVIDIA L4", "vram_gb": 23.7, "cu_per_hr": 1.94, "colab_flag": "L4"},
    "G4": {"cuda_model": "NVIDIA G4", "vram_gb": 24.0, "cu_per_hr": 3.00, "colab_flag": "G4"},
    "A100": {"cuda_model": "NVIDIA A100-SXM4-40GB", "vram_gb": 40.0, "cu_per_hr": 6.80, "colab_flag": "A100"},
    "H100": {"cuda_model": "NVIDIA H100 80GB HBM3", "vram_gb": 80.0, "cu_per_hr": 12.00, "colab_flag": "H100"},
}

HARD_MAX_WORKERS = 3
DEFAULT_STAGGER_SECONDS = 20.0
DEFAULT_REPO_URL = "https://github.com/HyperGAN/HyperGAN.git"
DEFAULT_PARTICLEGAN_SPEC = "particlegan==0.12.0"
DEFAULT_HNDL_SPEC = "hndl>=0.12.0,<1"
DEFAULT_TORCH_FIDELITY_SPEC = "torch-fidelity==0.4.0"
DEFAULT_FOLDLITEDB_SPEC = "foldlitedb>=0.5.0,<0.6"
DEFAULT_SAFETENSORS_SPEC = "safetensors>=0.8,<1"

REMOTE_REPO_DIR = "/content/HyperGAN"
REMOTE_RUNS_ROOT = "/content/runs"
REMOTE_CONFIGS_ROOT = "/content/bench_configs"
REMOTE_EVAL_ROOT = "/content/bench_eval"

KNOWN_EVAL_FACTORIES: dict[str, str] = {
    "InceptionFID": "hypergan.image_metrics:InceptionFID",
    "InceptionSuite": "hypergan.image_metrics:InceptionSuite",
    "DINOv2FD": "hypergan.image_metrics:DINOv2FD",
    "DINOv2Suite": "hypergan.image_metrics:DINOv2Suite",
}

IMAGE_METRICS_CLI_ALIASES: dict[str, str] = {
    "InceptionSuite": "inception",
    "DINOv2Suite": "dinov2",
    "InceptionFID": "inception_fid",
    "DINOv2FD": "dinov2_fd",
    "inception": "inception",
    "dinov2": "dinov2",
    "inception_fid": "inception_fid",
    "dinov2_fd": "dinov2_fd",
}

_NAN_LOG_RE = re.compile(
    r"(?:\bnan\b|\binf\b|-inf|non[- ]?finite|FloatingPointError|gradient.*exploded|diverged|collapsed)",
    re.IGNORECASE,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def sanitize_name(raw: str, *, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip("-._")
    if not cleaned:
        raise ValueError(f"Invalid empty/unsafe name: {raw!r}")
    return cleaned[:max_len]


def resolve_colab_bin(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    found = shutil.which("colab")
    if found:
        return found
    local_bin = Path.home() / ".local" / "bin" / "colab"
    if local_bin.is_file():
        return str(local_bin)
    return "colab"


def normalize_gpu(gpu: str) -> str:
    raw = str(gpu).strip()
    if raw.lower() == "cpu":
        return "cpu"
    upper = raw.upper()
    if upper not in COLAB_GPU_SPECS:
        choices = ", ".join(COLAB_GPU_SPECS.keys())
        raise ValueError(f"Unsupported --gpu {gpu!r}; choose from {choices}")
    return upper


def parse_gpu_list(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse an ordered comma-separated list (or sequence) of Colab GPU names."""
    if raw is None:
        return []
    tokens: list[str] = []
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        for item in raw:
            tokens.extend(part.strip() for part in str(item).split(",") if part.strip())
    result: list[str] = []
    for tok in tokens:
        norm = normalize_gpu(tok)
        if norm not in result:
            result.append(norm)
    return result


def validate_worker_count(
    workers: int,
    *,
    max_workers: int = HARD_MAX_WORKERS,
    allow_over_cap: bool = False,
) -> int:
    """Enforce the hard concurrent Colab GPU VM safety cap (default max 3)."""
    if type(workers) is not int or workers < 1:
        raise ValueError(f"--workers must be a positive integer, got {workers!r}")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError(f"--max-workers must be a positive integer, got {max_workers!r}")
    if max_workers > HARD_MAX_WORKERS and not allow_over_cap:
        raise ValueError(
            f"--max-workers={max_workers} exceeds hard safety cap of {HARD_MAX_WORKERS} concurrent Colab VMs; "
            "pass --allow-over-cap to override explicitly."
        )
    if workers > max_workers:
        raise ValueError(
            f"--workers={workers} exceeds --max-workers={max_workers} "
            f"(hard cap {HARD_MAX_WORKERS} concurrent Colab VMs)."
        )
    if workers > HARD_MAX_WORKERS and not allow_over_cap:
        raise ValueError(
            f"--workers={workers} exceeds hard safety cap of {HARD_MAX_WORKERS} concurrent Colab VMs; "
            "pass --allow-over-cap to override explicitly."
        )
    return workers


def estimate_cost(gpu: str, wall_seconds: float, *, actual_cuda_model: str | None = None) -> dict[str, Any]:
    norm = normalize_gpu(gpu)
    spec = COLAB_GPU_SPECS[norm]
    secs = max(0.0, float(wall_seconds))
    gpu_hours = secs / 3600.0
    cu = gpu_hours * float(spec["cu_per_hr"])
    return {
        "gpu": norm,
        "cuda_model": actual_cuda_model or spec["cuda_model"] or "CPU",
        "vram_gb": spec["vram_gb"],
        "wall_seconds": round(secs, 2),
        "gpu_hours": round(gpu_hours, 4),
        "cu_per_hr": spec["cu_per_hr"],
        "estimated_cu": round(cu, 4),
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# OS-level hard-timeout subprocess runner (process-group SIGTERM -> SIGKILL)
# ---------------------------------------------------------------------------


def run_with_hard_timeout(
    cmd: list[str],
    *,
    timeout_seconds: float,
    kill_grace: float = 10.0,
    stdin_text: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    capture: bool = False,
) -> tuple[int, str]:
    """Run a subprocess in its own process group with guaranteed SIGTERM -> SIGKILL timeout.

    Prevents hung `google-colab-cli` websocket background threads from blocking indefinitely
    and leaking idle Colab GPU compute units.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    chunks: list[str] = []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("a", encoding="utf-8") if log_file else None
    writer_lock = threading.Lock()

    def _kill_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if capture:
                chunks.append(line)
            if log_handle is not None:
                with writer_lock:
                    log_handle.write(line)
                    log_handle.flush()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        if stdin_text is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            msg = (
                f"[{utc_stamp()}] [timeout] Hard-killing hung command after {timeout_seconds}s: "
                f"{' '.join(cmd[:6])}\n"
            )
            if capture:
                chunks.append(msg)
            if log_handle is not None:
                with writer_lock:
                    log_handle.write(msg)
                    log_handle.flush()
            _kill_group(signal.SIGTERM)
            try:
                proc.wait(timeout=kill_grace)
            except subprocess.TimeoutExpired:
                _kill_group(signal.SIGKILL)
                proc.wait(timeout=5)
            rc = 124
    finally:
        reader_thread.join(timeout=2.0)
        if log_handle is not None:
            log_handle.close()
    return rc, "".join(chunks)


# ---------------------------------------------------------------------------
# VM-side detached crash/stall/max-wall-clock watchdog
# ---------------------------------------------------------------------------


def build_watchdog_arm_script(
    *,
    max_seconds: int = 7200,
    idle_grace_seconds: int = 600,
    stall_seconds: int = 1800,
    heartbeat_file: str = "/content/bench_heartbeat",
    done_file: str = "/content/bench_job.done",
    job_log: str = "/content/bench_job.log",
    runs_root: str = REMOTE_RUNS_ROOT,
) -> str:
    """Build the VM-side Python snippet that writes and launches detached `/content/watchdog.py`.

    The watchdog polls every 15 seconds and sends `POST http://$TBE_RUNTIME_ADDR/unassign` if:
    - total VM lifetime exceeds `max_seconds`,
    - a job finished (`done_file`) and neither `done_file` nor `heartbeat_file` refreshed
      within `idle_grace_seconds` (local controller crashed/disconnected), or
    - active training stalled (`job_log`, `heartbeat_file`, and all `/content/runs/*/{events.jsonl,summary.json}`
      remained unmodified for longer than `stall_seconds`).
    """
    if max_seconds <= 0 or idle_grace_seconds <= 0 or stall_seconds <= 0:
        raise ValueError("Watchdog timeouts (max_seconds, idle_grace_seconds, stall_seconds) must be > 0")
    return textwrap.dedent(
        f"""
        import os, subprocess, sys, textwrap
        addr = os.environ.get("TBE_RUNTIME_ADDR")
        if not addr:
            raise RuntimeError("TBE_RUNTIME_ADDR missing in Colab kernel environment")
        max_s = int(os.environ.get("CB_MAX_SECONDS", "{max_seconds}"))
        grace_s = int(os.environ.get("CB_IDLE_GRACE_SECONDS", "{idle_grace_seconds}"))
        stall_s = int(os.environ.get("CB_STALL_SECONDS", "{stall_seconds}"))
        watchdog_code = textwrap.dedent('''
            import glob, os, sys, time, urllib.request
            addr, max_s, grace_s, stall_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
            t0 = time.monotonic()
            logf = "/content/watchdog.log"
            hb_file = {heartbeat_file!r}
            done_file = {done_file!r}
            job_log = {job_log!r}
            runs_root = {runs_root!r}
            def log(msg):
                with open(logf, "a") as f:
                    f.write(f"[{{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}}] {{msg}}\\n")
            def unassign(reason):
                log(f"TRIGGERING UNASSIGN: {{reason}}")
                try:
                    req = urllib.request.Request(f"http://{{addr}}/unassign", method="POST")
                    with urllib.request.urlopen(req, timeout=15) as r:
                        log(f"unassign HTTP {{r.status}}")
                except Exception as e:
                    log(f"unassign error: {{e!r}}")
                sys.exit(0)
            def freshest_activity_mtime():
                candidates = []
                for p in (hb_file, job_log):
                    if os.path.exists(p):
                        try:
                            candidates.append(os.path.getmtime(p))
                        except OSError:
                            pass
                for pattern in (
                    os.path.join(runs_root, "*", "events.jsonl"),
                    os.path.join(runs_root, "*", "summary.json"),
                    os.path.join(runs_root, "*", "train.log"),
                ):
                    for p in glob.glob(pattern):
                        try:
                            candidates.append(os.path.getmtime(p))
                        except OSError:
                            pass
                return max(candidates) if candidates else None
            with open(hb_file, "w") as f:
                f.write(str(time.time()))
            log(f"Watchdog armed: addr={{addr}} max={{max_s}}s grace={{grace_s}}s stall={{stall_s}}s")
            while True:
                time.sleep(15)
                now_mono = time.monotonic()
                now_wall = time.time()
                if now_mono - t0 > max_s:
                    unassign(f"max wall-clock {{max_s}}s exceeded")
                if os.path.exists(done_file):
                    try:
                        age = now_wall - os.path.getmtime(done_file)
                    except OSError:
                        age = 0.0
                    try:
                        hb_age = now_wall - os.path.getmtime(hb_file) if os.path.exists(hb_file) else age
                    except OSError:
                        hb_age = age
                    if age > grace_s and hb_age > grace_s:
                        unassign(f"post-job idle grace {{grace_s}}s exceeded (local controller disconnected)")
                latest = freshest_activity_mtime()
                if latest is not None and (now_wall - latest) > stall_s:
                    unassign(f"activity stall timeout {{stall_s}}s exceeded")
        ''').strip()
        with open("/content/watchdog.py", "w") as f:
            f.write(watchdog_code + "\\n")
        subprocess.Popen(
            [sys.executable, "/content/watchdog.py", addr, str(max_s), str(grace_s), str(stall_s)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"WATCHDOG_OK addr={{addr}} max={{max_s}}s grace={{grace_s}}s stall={{stall_s}}s")
        """
    ).strip()


# ---------------------------------------------------------------------------
# Thread-safe + disk-durable active session registry
# ---------------------------------------------------------------------------


@dataclass
class ActiveSessionRegistry:
    """Tracks every provisioned Colab session in memory and under `runs/colab/sessions/`."""

    colab_bin: str = "colab"
    state_dir: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _session_record_path(self, session: str) -> Path | None:
        if self.state_dir is None:
            return None
        return self.state_dir / "sessions" / f"{sanitize_name(session)}.json"

    def register(
        self,
        session: str,
        cfg: Path,
        *,
        gpu: str,
        batch_dir: Path | None = None,
    ) -> None:
        record = {
            "schema_version": 1,
            "session": session,
            "config_path": str(Path(cfg).resolve()),
            "gpu": gpu,
            "pid": os.getpid(),
            "batch_dir": str(batch_dir.resolve()) if batch_dir else None,
            "started_at": utc_stamp(),
            "status": "active",
            "stopped_at": None,
        }
        with self.lock:
            self.sessions[session] = record
        rec_path = self._session_record_path(session)
        if rec_path is not None:
            _atomic_write_json(rec_path, record)

    def unregister(self, session: str) -> None:
        with self.lock:
            record = self.sessions.pop(session, None)
        rec_path = self._session_record_path(session)
        if rec_path is not None:
            if record is None and rec_path.is_file():
                try:
                    record = json.loads(rec_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    record = {"session": session}
            if record is not None:
                record["status"] = "stopped"
                record["stopped_at"] = utc_stamp()
                _atomic_write_json(rec_path, record)

    def stop_all(self) -> None:
        with self.lock:
            items = list(self.sessions.items())
            self.sessions.clear()
        for session, record in items:
            cfg = record["config_path"]
            try:
                run_with_hard_timeout(
                    [self.colab_bin, "--config", str(cfg), "stop", "-s", session],
                    timeout_seconds=30,
                )
            except OSError as err:
                print(f"[colab_bench] stop_all failed for {session}: {err}", file=sys.stderr)
            rec_path = self._session_record_path(session)
            if rec_path is not None:
                record["status"] = "stopped"
                record["stopped_at"] = utc_stamp()
                try:
                    _atomic_write_json(rec_path, record)
                except OSError:
                    pass


_GLOBAL_REGISTRY = ActiveSessionRegistry()
atexit.register(_GLOBAL_REGISTRY.stop_all)


# ---------------------------------------------------------------------------
# Job manifest schema, local run inspection & completion/resume checks
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkJob:
    run_name: str
    config: str
    gpu: str = "L4"
    steps: int | None = None
    stop_after_steps: int | None = None
    checkpoint_every: int | None = None
    preview_every: int | None = None
    no_previews: bool = False
    progress_every: int | None = None
    set_overrides: list[str] = field(default_factory=list)
    dataset: str | None = None
    dataset_prep_cmd: str | None = None
    dataset_tarball: str | None = None
    dataset_target_dir: str | None = None
    chunk_steps: int = 0
    sync_interval_minutes: float = 0.0
    max_retries: int = 2
    export_bundle: bool = True
    drive_dir: str | None = None
    estimated_minutes: float = 60.0
    timeout_minutes: float | None = None

    def target_step(self) -> int | None:
        if self.stop_after_steps is not None:
            return self.stop_after_steps
        return self.steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "config": self.config,
            "gpu": self.gpu,
            "steps": self.steps,
            "stop_after_steps": self.stop_after_steps,
            "checkpoint_every": self.checkpoint_every,
            "preview_every": self.preview_every,
            "no_previews": self.no_previews,
            "progress_every": self.progress_every,
            "set_overrides": list(self.set_overrides),
            "dataset": self.dataset,
            "dataset_prep_cmd": self.dataset_prep_cmd,
            "dataset_tarball": self.dataset_tarball,
            "dataset_target_dir": self.dataset_target_dir,
            "chunk_steps": self.chunk_steps,
            "sync_interval_minutes": self.sync_interval_minutes,
            "max_retries": self.max_retries,
            "export_bundle": self.export_bundle,
            "drive_dir": self.drive_dir,
            "estimated_minutes": self.estimated_minutes,
            "timeout_minutes": self.timeout_minutes,
        }


def _resolve_config_ref(config_str: str, *, manifest_dir: Path | None, repo_root: Path) -> str:
    """Resolve a TOML config path relative to manifest_dir or repo_root, preserving bare packaged names."""
    p = Path(config_str).expanduser()
    if p.is_absolute():
        return str(p)
    if manifest_dir is not None and (manifest_dir / p).is_file():
        return str((manifest_dir / p).resolve())
    if (repo_root / p).is_file():
        return str((repo_root / p).resolve())
    return config_str


def load_manifest(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    default_gpu: str = "L4",
    cli_overrides: dict[str, Any] | None = None,
) -> list[BenchmarkJob]:
    """Load a JSON or TOML benchmark job manifest into validated `BenchmarkJob` objects."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        doc = tomllib.loads(raw_text)
    elif suffix == ".json":
        doc = json.loads(raw_text)
    else:
        try:
            doc = json.loads(raw_text)
        except json.JSONDecodeError:
            doc = tomllib.loads(raw_text)

    if isinstance(doc, list):
        defaults: dict[str, Any] = {}
        raw_jobs = doc
    elif isinstance(doc, dict):
        defaults = dict(doc.get("defaults") or {})
        raw_jobs = doc.get("jobs")
        if not isinstance(raw_jobs, list) or not raw_jobs:
            raise ValueError(f"Manifest {path} must define a non-empty 'jobs' list")
    else:
        raise ValueError(f"Manifest {path} must be a JSON/TOML object or list")

    overrides = dict(cli_overrides or {})
    jobs: list[BenchmarkJob] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(raw_jobs):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest {path} job #{idx} must be a mapping")
        merged = {**defaults, **entry}
        for k, v in overrides.items():
            if v is not None:
                if k == "set_overrides":
                    merged["set_overrides"] = [*list(merged.get("set_overrides") or []), *list(v)]
                else:
                    merged[k] = v

        raw_config = merged.get("config")
        if not raw_config or not isinstance(raw_config, str):
            raise ValueError(f"Manifest {path} job #{idx} missing required 'config' string")
        resolved_cfg = _resolve_config_ref(raw_config, manifest_dir=path.parent, repo_root=repo_root)
        raw_name = merged.get("run_name") or merged.get("name") or Path(raw_config).stem
        run_name = sanitize_name(str(raw_name))
        if run_name in seen_names:
            raise ValueError(f"Duplicate run_name {run_name!r} in manifest {path}")
        seen_names.add(run_name)

        gpu = normalize_gpu(str(merged.get("gpu") or default_gpu))
        set_overrides = list(merged.get("set_overrides") or merged.get("set") or [])
        if any(not isinstance(item, str) or "=" not in item for item in set_overrides):
            raise ValueError(f"Job {run_name!r} has invalid set_overrides (expected KEY=VALUE): {set_overrides!r}")

        dataset_tarball = merged.get("dataset_tarball")
        if dataset_tarball:
            dtp = Path(str(dataset_tarball)).expanduser()
            if not dtp.is_absolute():
                if (path.parent / dtp).is_file():
                    dataset_tarball = str((path.parent / dtp).resolve())
                elif (repo_root / dtp).is_file():
                    dataset_tarball = str((repo_root / dtp).resolve())

        jobs.append(
            BenchmarkJob(
                run_name=run_name,
                config=resolved_cfg,
                gpu=gpu,
                steps=int(merged["steps"]) if merged.get("steps") is not None else None,
                stop_after_steps=(
                    int(merged["stop_after_steps"]) if merged.get("stop_after_steps") is not None else None
                ),
                checkpoint_every=(
                    int(merged["checkpoint_every"]) if merged.get("checkpoint_every") is not None else None
                ),
                preview_every=int(merged["preview_every"]) if merged.get("preview_every") is not None else None,
                no_previews=bool(merged.get("no_previews", False)),
                progress_every=int(merged["progress_every"]) if merged.get("progress_every") is not None else None,
                set_overrides=set_overrides,
                dataset=str(merged["dataset"]) if merged.get("dataset") is not None else None,
                dataset_prep_cmd=(
                    str(merged["dataset_prep_cmd"]) if merged.get("dataset_prep_cmd") is not None else None
                ),
                dataset_tarball=str(dataset_tarball) if dataset_tarball else None,
                dataset_target_dir=(
                    str(merged["dataset_target_dir"]) if merged.get("dataset_target_dir") is not None else None
                ),
                chunk_steps=max(0, int(merged.get("chunk_steps", 0))),
                sync_interval_minutes=max(0.0, float(merged.get("sync_interval_minutes", 0.0))),
                max_retries=max(0, int(merged.get("max_retries", 2))),
                export_bundle=bool(merged.get("export_bundle", True)),
                drive_dir=str(merged["drive_dir"]) if merged.get("drive_dir") else None,
                estimated_minutes=max(1.0, float(merged.get("estimated_minutes", 60.0))),
                timeout_minutes=(
                    float(merged["timeout_minutes"]) if merged.get("timeout_minutes") is not None else None
                ),
            )
        )
    return jobs


def _has_local_checkpoint(run_dir: Path) -> bool:
    for subdir in ("checkpoints", "distributed-checkpoints"):
        root = run_dir / subdir
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                if (child / "state.pt").is_file() or (child / "manifest.json").is_file():
                    return True
    return False


def inspect_local_run(run_dir: Path, *, target_step: int | None = None) -> dict[str, Any]:
    """Inspect `<runs_dir>/<run_name>` to decide whether to skip, resume, or start fresh."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    has_ckpt = _has_local_checkpoint(run_dir)

    if not summary_path.is_file():
        return {
            "exists": run_dir.is_dir(),
            "status": None,
            "step": 0,
            "total_steps": None,
            "completed": False,
            "resumable": bool(manifest_path.is_file() and has_ckpt),
            "has_checkpoints": has_ckpt,
            "best_fid": None,
            "error": None,
        }

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "exists": True,
            "status": "corrupt_summary",
            "step": 0,
            "total_steps": None,
            "completed": False,
            "resumable": bool(manifest_path.is_file() and has_ckpt),
            "has_checkpoints": has_ckpt,
            "best_fid": None,
            "error": "Unable to parse summary.json",
        }

    status = str(summary.get("status") or "").lower()
    step = int(summary.get("step") or 0)
    total_steps = summary.get("total_steps")
    effective_target = target_step if target_step is not None else (int(total_steps) if total_steps else None)

    completed = False
    if status in ("stopped", "complete"):
        if effective_target is not None:
            completed = step >= effective_target
        else:
            completed = status == "complete"

    resumable = (
        not completed
        and manifest_path.is_file()
        and has_ckpt
        and status in ("running", "stopped", "interrupted", "initializing", "")
    )
    return {
        "exists": True,
        "status": status,
        "step": step,
        "total_steps": int(total_steps) if isinstance(total_steps, int) else None,
        "completed": completed,
        "resumable": resumable,
        "has_checkpoints": has_ckpt,
        "best_fid": summary.get("best_fid"),
        "error": summary.get("error"),
        "health_alerts": list(summary.get("health_alerts") or []),
    }


DEFAULT_CIFAR10_FAST_MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz"
PREPARABLE_BENCHMARK_DATASETS = ("ffhq64", "flowers102")


def detect_job_dataset(job: BenchmarkJob) -> str:
    """Infer whether a job needs built-in CIFAR-10 or `prepare_benchmark_dataset.py` (`ffhq64`/`flowers102`)."""
    if job.dataset:
        return job.dataset.strip().lower()
    cfg_path = Path(job.config)
    if cfg_path.is_file():
        try:
            doc = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
            data_tbl = doc.get("data") or {}
            if isinstance(data_tbl, dict):
                builtin = str(data_tbl.get("builtin") or "").strip().lower()
                if builtin == "cifar10":
                    return "cifar10"
                args_tbl = data_tbl.get("args") if isinstance(data_tbl.get("args"), dict) else {}
                root_val = str(args_tbl.get("root") or data_tbl.get("root") or "").lower()
                for ds in PREPARABLE_BENCHMARK_DATASETS:
                    if ds in root_val:
                        return ds
        except (OSError, tomllib.TOMLDecodeError):
            pass
    lower_ref = f"{job.config} {job.run_name}".lower()
    for ds in PREPARABLE_BENCHMARK_DATASETS:
        if ds in lower_ref:
            return ds
    if "cifar10" in lower_ref:
        return "cifar10"
    return "none"


# ---------------------------------------------------------------------------
# Git / source bundle preparation (git ref, unpushed bundle, or local tarball)
# ---------------------------------------------------------------------------


def _git_rev_parse(root: Path, ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=root, text=True).strip()


def _git_is_pushed(root: Path, sha: str) -> bool:
    proc = subprocess.run(
        ["git", "branch", "-r", "--contains", sha],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def prepare_source_bundle(
    root: Path,
    out_dir: Path,
    *,
    ref: str = "HEAD",
    source_mode: str = "auto",
) -> dict[str, Any]:
    """Prepare HyperGAN source sync payload for Colab VMs.

    Modes:
    - `git`: clone directly from `--repo-url` at `--ref` (requires pushed commit).
    - `bundle`: create a compact `git bundle` + optional dirty worktree overlay tarball.
    - `tarball`: archive the local checkout directly (`src/`, `scripts/`, `benchmarks/`, `pyproject.toml`, etc.).
    - `auto`: use `git` when `--ref` is pushed and worktree is clean; otherwise `bundle` + overlay.
    """
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = source_mode.strip().lower()
    if mode not in ("auto", "git", "bundle", "tarball"):
        raise ValueError(f"Unsupported --source-mode {source_mode!r}; choose auto, git, bundle, or tarball")

    if mode == "tarball":
        tar_path = out_dir / "hypergan-source.tar.gz"
        include_items = (
            "pyproject.toml",
            "setup.py",
            "MANIFEST.in",
            "README.md",
            "LICENSE",
            "src",
            "scripts",
            "benchmarks",
            "examples",
        )
        exclude_parts = {"__pycache__", ".pytest_cache", ".git", "node_modules", ".mypy_cache"}
        with tarfile.open(tar_path, "w:gz") as tf:
            for item in include_items:
                src_p = root / item
                if not src_p.exists():
                    continue
                if src_p.is_file():
                    tf.add(src_p, arcname=item)
                else:
                    for child in sorted(src_p.rglob("*")):
                        if any(part in exclude_parts for part in child.parts):
                            continue
                        if child.is_file() and child.suffix not in (".pyc", ".pyo", ".pt"):
                            tf.add(child, arcname=str(child.relative_to(root)))
        return {
            "mode": "tarball",
            "ref": ref,
            "sha": None,
            "pushed": False,
            "tarball_path": str(tar_path),
            "bundle_path": None,
            "overlay_path": None,
            "dirty_file_count": 0,
        }

    sha = _git_rev_parse(root, ref)
    pushed = _git_is_pushed(root, sha)
    bundle_path: Path | None = None
    overlay_path: Path | None = None
    base_ref: str | None = None

    if mode == "bundle" or (mode == "auto" and not pushed):
        base_proc = subprocess.run(
            ["git", "merge-base", "origin/develop", sha],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        base_ref = base_proc.stdout.strip() if base_proc.returncode == 0 and base_proc.stdout.strip() else None
        bundle_path = out_dir / "branch.bundle"
        rev_range = f"{base_ref}..{sha}" if base_ref and base_ref != sha else sha
        subprocess.run(
            ["git", "bundle", "create", str(bundle_path), rev_range, sha],
            cwd=root,
            check=True,
            capture_output=True,
        )

    dirty_files: list[str] = []
    if mode != "git" and ref in ("HEAD", "@"):
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
        )
        allowed_prefixes = ("src/", "scripts/", "benchmarks/", "examples/", "pyproject.toml")
        for line in status_out.splitlines():
            if len(line) < 4:
                continue
            rel = line[3:].strip()
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            if any(rel == p or rel.startswith(p) for p in allowed_prefixes):
                full = root / rel
                if full.is_file() and full.suffix not in (".pyc", ".pyo", ".pt"):
                    dirty_files.append(rel)
        if dirty_files:
            overlay_path = out_dir / "worktree-overlay.tar.gz"
            with tarfile.open(overlay_path, "w:gz") as tf:
                for rel in sorted(set(dirty_files)):
                    tf.add(root / rel, arcname=rel)

    return {
        "mode": "git" if (bundle_path is None and overlay_path is None) else "bundle",
        "ref": ref,
        "sha": sha,
        "base_sha": base_ref,
        "pushed": pushed,
        "tarball_path": None,
        "bundle_path": str(bundle_path) if bundle_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "dirty_file_count": len(dirty_files),
    }


# ---------------------------------------------------------------------------
# Artifact unpacking, path relocation, resume packing & telemetry inspection
# ---------------------------------------------------------------------------


_MANIFEST_PATH_KEYS = (
    "run_dir",
    "checkpoint_path",
    "resumed_from",
    "attempt_dir",
    "exported_bundle_path",
    "bundle_path",
    "sample_path",
)


def _rewrite_prefix(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if not isinstance(value, str):
        return value
    old_clean = old_prefix.rstrip("/")
    new_clean = new_prefix.rstrip("/")
    if value == old_clean:
        return new_clean
    if value.startswith(old_clean + "/"):
        return new_clean + value[len(old_clean) :]
    return value


def relocate_manifest_paths(run_dir: Path, *, old_prefix: str, new_prefix: str) -> int:
    """Rewrite absolute run path prefixes inside `manifest.json` and `attempts/*/manifest.json`."""
    run_dir = Path(run_dir)
    updated = 0
    targets = [run_dir / "manifest.json"]
    attempts_dir = run_dir / "attempts"
    if attempts_dir.is_dir():
        targets.extend(sorted(attempts_dir.glob("*/manifest.json")))

    for path in targets:
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        changed = False
        for key in _MANIFEST_PATH_KEYS:
            if key in doc:
                rewritten = _rewrite_prefix(doc[key], old_prefix, new_prefix)
                if rewritten != doc[key]:
                    doc[key] = rewritten
                    changed = True
        if changed:
            _atomic_write_json(path, doc)
            updated += 1
    return updated


def unpack_run_tarball(
    tar_path: Path,
    dest_run_dir: Path,
    *,
    remote_run_dir: str | None = None,
) -> dict[str, Any]:
    """Safely unpack a Colab run artifact tarball into `dest_run_dir` with traversal guards."""
    tar_path = Path(tar_path)
    dest_run_dir = Path(dest_run_dir).resolve()
    dest_run_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: list[str] = []
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            orig_name = member.name
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Refusing unsafe link/device member in archive: {orig_name!r}")
            if orig_name.startswith("/") or Path(orig_name).is_absolute() or ".." in Path(orig_name).parts:
                raise ValueError(f"Unsafe archive member path: {orig_name!r}")
            raw_name = orig_name
            while raw_name.startswith("./"):
                raw_name = raw_name[2:]
            if not raw_name or raw_name == ".":
                continue
            target = (dest_run_dir / raw_name).resolve()
            if not target.is_relative_to(dest_run_dir):
                raise ValueError(f"Archive member escapes destination: {orig_name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                tmp_target = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                try:
                    with src, tmp_target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.replace(tmp_target, target)
                finally:
                    if tmp_target.exists():
                        tmp_target.unlink(missing_ok=True)
                extracted_files.append(raw_name)

    relocated = 0
    if remote_run_dir:
        relocated = relocate_manifest_paths(
            dest_run_dir,
            old_prefix=remote_run_dir,
            new_prefix=str(dest_run_dir),
        )

    return {
        "dest_run_dir": str(dest_run_dir),
        "file_count": len(extracted_files),
        "has_summary": (dest_run_dir / "summary.json").is_file(),
        "has_manifest": (dest_run_dir / "manifest.json").is_file(),
        "has_model_json": (dest_run_dir / "model.json").is_file(),
        "has_events": (dest_run_dir / "events.jsonl").is_file(),
        "has_bundle": (dest_run_dir / "model.pt").is_file(),
        "has_train_log": (dest_run_dir / "train.log").is_file(),
        "manifests_relocated": relocated,
    }


def _select_latest_checkpoint_dir(run_dir: Path) -> Path | None:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    latest_json = ckpt_root / "latest.json"
    if latest_json.is_file():
        try:
            ptr = json.loads(latest_json.read_text(encoding="utf-8"))
            name = ptr.get("checkpoint")
            if isinstance(name, str) and (ckpt_root / name).is_dir():
                return ckpt_root / name
        except (OSError, json.JSONDecodeError):
            pass
    dirs = [d for d in sorted(ckpt_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    return dirs[-1] if dirs else None


def pack_local_run_for_resume(
    local_run_dir: Path,
    out_tar: Path,
    *,
    remote_run_dir: str,
) -> Path:
    """Pack a local run directory + latest checkpoint into a tarball ready for `hypergan resume` on Colab."""
    local_run_dir = Path(local_run_dir).resolve()
    out_tar = Path(out_tar).resolve()
    out_tar.parent.mkdir(parents=True, exist_ok=True)

    latest_ckpt = _select_latest_checkpoint_dir(local_run_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoints found under {local_run_dir / 'checkpoints'} for resume")

    with tempfile.TemporaryDirectory(prefix="colab-resume-stage-") as stage_tmp:
        stage_root = Path(stage_tmp)
        for fname in ("manifest.json", "summary.json", "model.json", "console.json", "events.jsonl", "train.log"):
            src = local_run_dir / fname
            if src.is_file():
                shutil.copy2(src, stage_root / fname)
        for dname in ("metrics", "previews", "views", "config-source", "attempts", "launch"):
            src_d = local_run_dir / dname
            if src_d.is_dir():
                shutil.copytree(src_d, stage_root / dname, dirs_exist_ok=True)
        ckpt_dest = stage_root / "checkpoints"
        ckpt_dest.mkdir(parents=True, exist_ok=True)
        if (local_run_dir / "checkpoints" / "latest.json").is_file():
            shutil.copy2(local_run_dir / "checkpoints" / "latest.json", ckpt_dest / "latest.json")
        else:
            _atomic_write_json(ckpt_dest / "latest.json", {"checkpoint": latest_ckpt.name})
        shutil.copytree(latest_ckpt, ckpt_dest / latest_ckpt.name, dirs_exist_ok=True)

        relocate_manifest_paths(stage_root, old_prefix=str(local_run_dir), new_prefix=remote_run_dir)

        with tarfile.open(out_tar, "w:gz") as tf:
            for item in sorted(stage_root.rglob("*")):
                if item.is_file():
                    tf.add(item, arcname=str(item.relative_to(stage_root)))
    return out_tar


def analyze_run_telemetry(run_dir: Path) -> dict[str, Any]:
    """Inspect pulled-back `summary.json`, `events.jsonl`, and `train.log` for NaNs/collapse/scores."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}

    latest = summary.get("latest") if isinstance(summary.get("latest"), dict) else {}
    alerts = list(summary.get("health_alerts") or [])
    error_text = summary.get("error")
    status = str(summary.get("status") or "unknown")

    nan_detected = False
    for key, val in latest.items():
        if isinstance(val, (int, float)) and not math.isfinite(val):
            nan_detected = True
            alerts.append(f"Non-finite latest metric: {key}={val}")

    if isinstance(error_text, str) and _NAN_LOG_RE.search(error_text):
        nan_detected = True

    train_log = run_dir / "train.log"
    log_tail_warnings: list[str] = []
    cpu_fallback_detected = False
    if train_log.is_file():
        try:
            full_log = train_log.read_text(encoding="utf-8", errors="replace")
            if "CUDA is unavailable" in full_log or "falling back to cpu" in full_log:
                cpu_fallback_detected = True
                log_tail_warnings.append("CUDA requested but HyperGAN fell back to CPU")
            tail = full_log[-16384:]
            for line in tail.splitlines():
                if _NAN_LOG_RE.search(line):
                    nan_detected = True
                    log_tail_warnings.append(line.strip()[:240])
        except OSError:
            pass

    diversity = latest.get("diversity/ratio")
    collapse_detected = bool(
        nan_detected
        or status == "failed"
        or (isinstance(diversity, (int, float)) and math.isfinite(diversity) and diversity < 0.05)
        or any("collapse" in str(a).lower() for a in alerts)
    )

    return {
        "status": status,
        "step": int(summary.get("step") or 0),
        "total_steps": summary.get("total_steps"),
        "steps_per_second": summary.get("steps_per_second") or summary.get("steps_per_sec"),
        "best": summary.get("best"),
        "best_fid": summary.get("best_fid"),
        "best_step": summary.get("best_step"),
        "latest_fid": summary.get("latest_fid"),
        "evaluations": list(summary.get("evaluations") or []),
        "latest_metrics": {
            k: latest[k]
            for k in ("loss/d_total", "loss/g_total", "loss/gradient_penalty", "critic/margin", "critic/accuracy", "diversity/ratio")
            if k in latest
        },
        "health_alerts": alerts,
        "error": error_text,
        "nan_detected": nan_detected,
        "collapse_detected": collapse_detected,
        "cpu_fallback_detected": cpu_fallback_detected,
        "log_warnings": log_tail_warnings[-5:],
        "has_bundle": (run_dir / "model.pt").is_file(),
    }


def register_local_run(run_dir: Path) -> bool:
    """Register a pulled-back Colab run into HyperGAN's local Observatory run registry."""
    try:
        from hypergan.registry import register_run

        register_run(Path(run_dir).resolve())
        return True
    except Exception:
        return False


def parse_colab_sessions_output(output: str) -> list[str]:
    """Parse active session names from `colab sessions` stdout."""
    sessions: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        token = line.strip()
        if (
            not token
            or token.startswith("[colab]")
            or token.lower().startswith("no active sessions")
            or token.startswith("─")
            or token.startswith("╭")
            or token.startswith("╰")
        ):
            continue
        if "|" in token:
            left = token.split("|", 1)[0].strip()
            parts = left.split()
            if parts:
                name = parts[-1].strip()
                if name and name not in seen and re.fullmatch(r"[A-Za-z0-9._-]+", name):
                    seen.add(name)
                    sessions.append(name)
    return sessions


# ---------------------------------------------------------------------------
# Warm Colab GPU worker for HyperGAN training & bundle evaluation
# ---------------------------------------------------------------------------


class ColabBenchWorker:
    """One warm Colab VM worker with isolated `--config`, VM watchdog, and guaranteed teardown."""

    def __init__(
        self,
        *,
        worker_name: str,
        gpu: str,
        batch_dir: Path,
        state_dir: Path,
        repo_root: Path,
        repo_url: str,
        source_sync: dict[str, Any],
        colab_bin: str = "colab",
        boot_timeout: int = 260,
        boot_retries: int = 2,
        fallback_gpus: list[str] | str | None = None,
        max_minutes: int = 240,
        idle_grace_minutes: int = 10,
        stall_minutes: int = 30,
        particlegan_spec: str = DEFAULT_PARTICLEGAN_SPEC,
        hndl_spec: str = DEFAULT_HNDL_SPEC,
        torch_fidelity_spec: str = DEFAULT_TORCH_FIDELITY_SPEC,
        foldlitedb_spec: str = DEFAULT_FOLDLITEDB_SPEC,
        safetensors_spec: str = DEFAULT_SAFETENSORS_SPEC,
        drive_dir: str | None = None,
        registry: ActiveSessionRegistry | None = None,
    ) -> None:
        self.worker_name = sanitize_name(worker_name, max_len=48)
        self.requested_gpu = normalize_gpu(gpu)
        self.gpu = self.requested_gpu
        self.batch_dir = Path(batch_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.repo_url = repo_url
        self.source_sync = source_sync
        self.colab_bin = colab_bin
        self.boot_timeout = boot_timeout
        self.boot_retries = max(1, int(boot_retries))
        self.fallback_gpus = [g for g in parse_gpu_list(fallback_gpus) if g != self.requested_gpu]
        self.max_seconds = max_minutes * 60
        self.idle_grace_seconds = idle_grace_minutes * 60
        self.stall_seconds = stall_minutes * 60
        self.particlegan_spec = particlegan_spec
        self.hndl_spec = hndl_spec
        self.torch_fidelity_spec = torch_fidelity_spec
        self.foldlitedb_spec = foldlitedb_spec
        self.safetensors_spec = safetensors_spec
        self.drive_dir = drive_dir
        self.registry = registry or _GLOBAL_REGISTRY
        self.cfg = self.batch_dir / f"{self.worker_name}.colab.json"
        self.log = self.batch_dir / f"{self.worker_name}.log"
        self.alive = False
        self.started_mono: float | None = None
        self.stopped_mono: float | None = None
        self.actual_cuda_model: str | None = None
        self._prepared_datasets: set[str] = set()
        self._ssh_lock = threading.Lock()

    def _log(self, msg: str) -> None:
        line = f"[{utc_stamp()}] [{self.worker_name}] {msg}\n"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(line)
        print(line, end="", flush=True)

    def _colab(
        self,
        args: list[str],
        *,
        timeout: float,
        stdin_text: str | None = None,
        capture: bool = False,
    ) -> tuple[int, str]:
        return run_with_hard_timeout(
            [self.colab_bin, "--config", str(self.cfg), *args],
            timeout_seconds=timeout,
            stdin_text=stdin_text,
            log_file=self.log,
            capture=capture,
        )

    def wall_seconds(self) -> float:
        if self.started_mono is None:
            return 0.0
        end = self.stopped_mono if self.stopped_mono is not None else time.monotonic()
        return max(0.0, end - self.started_mono)

    def cost_summary(self) -> dict[str, Any]:
        return {
            "worker": self.worker_name,
            "requested_gpu": self.requested_gpu,
            **estimate_cost(self.gpu, self.wall_seconds(), actual_cuda_model=self.actual_cuda_model),
        }

    def start(self) -> None:
        gpu_candidates = [self.requested_gpu, *self.fallback_gpus]

        last_rc = 1
        last_out = ""
        for idx_g, candidate_gpu in enumerate(gpu_candidates):
            spec = COLAB_GPU_SPECS[candidate_gpu]
            cmd = ["new", "-s", self.worker_name]
            if spec["colab_flag"]:
                cmd += ["--gpu", spec["colab_flag"]]
            for boot_attempt in range(1, self.boot_retries + 1):
                self._log(
                    f"Provisioning Colab {candidate_gpu} VM (attempt {boot_attempt}/{self.boot_retries}, "
                    f"boot timeout {self.boot_timeout}s)..."
                )
                last_rc, last_out = self._colab(cmd, timeout=self.boot_timeout, capture=True)
                if last_rc == 0:
                    self.gpu = candidate_gpu
                    self.started_mono = time.monotonic()
                    break
                # Best-effort stop in case Colab allocated the backend right as `colab new` timed out.
                self._colab(["stop", "-s", self.worker_name], timeout=30)
                if boot_attempt < self.boot_retries:
                    self._log(
                        f"colab new ({candidate_gpu}) attempt {boot_attempt}/{self.boot_retries} failed (rc={last_rc}); "
                        "waiting 20s before retry..."
                    )
                    time.sleep(20)
            if last_rc == 0:
                break
            if idx_g + 1 < len(gpu_candidates):
                self._log(f"Colab {candidate_gpu} allocation failed; falling back to {gpu_candidates[idx_g + 1]}...")
        if last_rc != 0:
            raise RuntimeError(
                f"colab new failed for {self.worker_name} (tried={gpu_candidates}, rc={last_rc}): {last_out[-400:]}"
            )

        self.alive = True
        self.registry.register(self.worker_name, self.cfg, gpu=self.gpu, batch_dir=self.batch_dir)
        time.sleep(5)  # Websocket warmup guard

        # Layer 2 teardown: arm VM-side detached watchdog with 5 retries.
        watchdog_script = build_watchdog_arm_script(
            max_seconds=self.max_seconds,
            idle_grace_seconds=self.idle_grace_seconds,
            stall_seconds=self.stall_seconds,
        )
        armed = False
        for attempt in range(1, 6):
            rc, _ = self._colab(
                [
                    "exec",
                    "-s",
                    self.worker_name,
                    "--timeout",
                    "30",
                    "--env",
                    f"CB_MAX_SECONDS={self.max_seconds}",
                    "--env",
                    f"CB_IDLE_GRACE_SECONDS={self.idle_grace_seconds}",
                    "--env",
                    f"CB_STALL_SECONDS={self.stall_seconds}",
                ],
                timeout=45,
                stdin_text=watchdog_script,
            )
            if rc == 0:
                armed = True
                break
            self._log(f"Watchdog arm attempt {attempt}/5 failed (rc={rc}); retrying in 5s...")
            time.sleep(5)
        if not armed:
            self.stop()
            raise RuntimeError(f"Failed to arm VM watchdog on {self.worker_name}; VM terminated for safety.")

        if self.drive_dir:
            self._log("Mounting Google Drive on Colab VM...")
            rc, _ = self._colab(["drivemount", "-s", self.worker_name, "/content/drive"], timeout=120)
            if rc != 0:
                self._log(f"Warning: drivemount returned rc={rc}; continuing with local checkpoint sync.")

        # Upload optional source payloads (tarball, git bundle, dirty worktree overlay).
        if self.source_sync.get("tarball_path"):
            self._log("Uploading HyperGAN source tarball...")
            rc, _ = self._colab(
                ["upload", "-s", self.worker_name, self.source_sync["tarball_path"], "/content/hypergan-source.tar.gz"],
                timeout=180,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to upload hypergan-source.tar.gz (rc={rc})")
        if self.source_sync.get("bundle_path"):
            self._log("Uploading unpushed branch git bundle...")
            rc, _ = self._colab(
                ["upload", "-s", self.worker_name, self.source_sync["bundle_path"], "/content/branch.bundle"],
                timeout=120,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to upload branch.bundle (rc={rc})")
        if self.source_sync.get("overlay_path"):
            self._log(f"Uploading worktree overlay ({self.source_sync['dirty_file_count']} file(s))...")
            rc, _ = self._colab(
                ["upload", "-s", self.worker_name, self.source_sync["overlay_path"], "/content/worktree-overlay.tar.gz"],
                timeout=120,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to upload worktree-overlay.tar.gz (rc={rc})")

        self._bootstrap_vm()

    def _bootstrap_vm(self) -> None:
        sha = self.source_sync.get("sha") or ""
        base_sha = self.source_sync.get("base_sha") or ""
        mode = self.source_sync.get("mode", "git")
        pkgs = [
            self.particlegan_spec,
            self.hndl_spec,
            self.torch_fidelity_spec,
            self.foldlitedb_spec,
            self.safetensors_spec,
        ]
        quoted_pkgs = " ".join(shlex.quote(p) for p in pkgs)
        self._log(f"Bootstrapping HyperGAN ({mode}, ref={self.source_sync.get('ref')}, sha={sha[:12]})...")
        bootstrap_sh = textwrap.dedent(
            f"""
            set -euo pipefail
            mkdir -p {REMOTE_RUNS_ROOT} {REMOTE_CONFIGS_ROOT} {REMOTE_EVAL_ROOT} ~/.hypergan/cache
            exec > >(tee -a /content/bench_job.log) 2>&1
            if command -v nvidia-smi >/dev/null 2>&1; then
              GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1)
              echo "COLAB_GPU_INFO: $GPU_LINE"
            fi
            rm -rf {REMOTE_REPO_DIR}
            if [ -f /content/hypergan-source.tar.gz ]; then
              mkdir -p {REMOTE_REPO_DIR}
              tar -xzf /content/hypergan-source.tar.gz -C {REMOTE_REPO_DIR}
              cd {REMOTE_REPO_DIR}
            else
              git clone --depth 1 --branch develop {shlex.quote(self.repo_url)} {REMOTE_REPO_DIR}
              cd {REMOTE_REPO_DIR}
              ACTUAL=$(git rev-parse HEAD)
              if [ -n "{sha}" ] && [ "$ACTUAL" != "{sha}" ]; then
                if [ -f /content/branch.bundle ]; then
                  if [ -n "{base_sha}" ] && ! git cat-file -e "{base_sha}^{{commit}}" 2>/dev/null; then
                    git fetch --depth 1 origin {base_sha} || true
                  fi
                  git fetch /content/branch.bundle {sha}:refs/heads/colab-branch || git fetch /content/branch.bundle HEAD:refs/heads/colab-branch
                  git checkout {sha}
                else
                  git fetch --depth 1 origin {sha}
                  git checkout {sha}
                fi
              fi
              if [ -f /content/worktree-overlay.tar.gz ]; then
                tar -xzf /content/worktree-overlay.tar.gz -C {REMOTE_REPO_DIR}
              fi
            fi
            if command -v uv >/dev/null 2>&1; then
              INSTALLER="uv pip install --system"
            else
              INSTALLER="python3 -m pip install -q"
            fi
            $INSTALLER --no-deps {quoted_pkgs}
            $INSTALLER --no-deps -e {REMOTE_REPO_DIR}
            python3 - <<'PY'
            import importlib.metadata as md, sys
            import torch, hypergan, particlegan, hndl, torch_fidelity, foldlitedb
            def _ver(pkg):
                try:
                    return md.version(pkg)
                except md.PackageNotFoundError:
                    return "installed"
            want_cuda = {self.gpu != "cpu"!r}
            cuda_ok = torch.cuda.is_available()
            print(
                f"[vm] ready: torch={{torch.__version__}} cuda={{cuda_ok}} "
                f"hypergan={{_ver('hypergan')}} particlegan={{_ver('particlegan')}} "
                f"hndl={{_ver('hndl')}} torch_fidelity={{_ver('torch-fidelity')}} "
                f"foldlitedb={{_ver('foldlitedb')}}"
            )
            if want_cuda and not cuda_ok:
                print("[vm] ERROR: GPU VM requested but torch.cuda.is_available() is False", file=sys.stderr)
                sys.exit(2)
            PY
            touch /content/bench_heartbeat
            """
        ).strip()
        rc, out = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", bootstrap_sh], timeout=420, capture=True)
        for line in out.splitlines():
            if line.startswith("COLAB_GPU_INFO:"):
                self.actual_cuda_model = line.split(":", 1)[1].split(",", 1)[0].strip()
                break
        if rc != 0:
            raise RuntimeError(f"VM bootstrap failed on {self.worker_name} (rc={rc})")
        self._log(f"Warm VM ready (GPU={self.actual_cuda_model or self.gpu}).")

    def prepare_dataset(self, job: BenchmarkJob) -> None:
        dataset = detect_job_dataset(job)
        if dataset == "cifar10" and "cifar10" not in self._prepared_datasets:
            self._log("Preparing cached CIFAR-10 dataset on VM...")
            prep_sh = textwrap.dedent(
                f"""
                set -euo pipefail
                touch /content/bench_heartbeat
                if [ ! -f ~/.hypergan/cache/cifar10/dataset-manifest.json ]; then
                  hypergan download-data cifar10 --output ~/.hypergan/cache/cifar10 \
                    --url {shlex.quote(DEFAULT_CIFAR10_FAST_MIRROR)} --timeout 45 \
                  || hypergan download-data cifar10 --output ~/.hypergan/cache/cifar10 --timeout 60
                fi
                touch /content/bench_heartbeat
                """
            ).strip()
            rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", prep_sh], timeout=240)
            if rc != 0:
                raise RuntimeError(f"CIFAR-10 dataset prep failed on {self.worker_name} (rc={rc})")
            self._prepared_datasets.add("cifar10")

        if dataset in PREPARABLE_BENCHMARK_DATASETS and dataset not in self._prepared_datasets:
            self._log(f"Preparing cached {dataset} benchmark dataset on VM via scripts/prepare_benchmark_dataset.py...")
            prep_sh = textwrap.dedent(
                f"""
                set -euo pipefail
                cd {REMOTE_REPO_DIR}
                touch /content/bench_heartbeat
                if [ ! -f ~/.hypergan/cache/datasets/{dataset}/dataset-manifest.json ]; then
                  python3 scripts/prepare_benchmark_dataset.py --dataset {shlex.quote(dataset)}
                fi
                touch /content/bench_heartbeat
                """
            ).strip()
            rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", prep_sh], timeout=900)
            if rc != 0:
                raise RuntimeError(f"{dataset} dataset prep failed on {self.worker_name} (rc={rc})")
            self._prepared_datasets.add(dataset)

        if job.dataset_tarball:
            key = f"tar:{job.dataset_tarball}:{job.dataset_target_dir or ''}"
            if key not in self._prepared_datasets:
                remote_tar = f"/content/dataset-{hashlib.sha256(key.encode()).hexdigest()[:8]}.tar.gz"
                target_dir = job.dataset_target_dir or "/content/datasets"
                self._log(f"Uploading dataset tarball {job.dataset_tarball} -> {target_dir}...")
                rc, _ = self._colab(["upload", "-s", self.worker_name, job.dataset_tarball, remote_tar], timeout=600)
                if rc != 0:
                    raise RuntimeError(f"Dataset tarball upload failed (rc={rc})")
                extract_sh = (
                    f"set -euo pipefail; mkdir -p {shlex.quote(target_dir)}; "
                    f"tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(target_dir)}; "
                    f"rm -f {shlex.quote(remote_tar)}; touch /content/bench_heartbeat"
                )
                rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", extract_sh], timeout=300)
                if rc != 0:
                    raise RuntimeError(f"Dataset tarball extraction failed (rc={rc})")
                self._prepared_datasets.add(key)

        if job.dataset_prep_cmd:
            cmd_hash = hashlib.sha256(job.dataset_prep_cmd.encode()).hexdigest()[:12]
            key = f"cmd:{cmd_hash}"
            if key not in self._prepared_datasets:
                self._log(f"Running dataset preparation command ({cmd_hash})...")
                prep_sh = textwrap.dedent(
                    f"""
                    set -euo pipefail
                    cd {REMOTE_REPO_DIR}
                    touch /content/bench_heartbeat
                    if [ ! -f /content/.dataset_ready_{cmd_hash} ]; then
                      {job.dataset_prep_cmd}
                      touch /content/.dataset_ready_{cmd_hash}
                    fi
                    touch /content/bench_heartbeat
                    """
                ).strip()
                rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", prep_sh], timeout=600)
                if rc != 0:
                    raise RuntimeError(f"Custom dataset_prep_cmd failed on {self.worker_name} (rc={rc})")
                self._prepared_datasets.add(key)

    def _stage_job_config(self, job: BenchmarkJob) -> str:
        """Upload local TOML config if needed and return remote config path or bare packaged name."""
        cfg_path = Path(job.config)
        if not cfg_path.is_file():
            return job.config
        resolved = cfg_path.resolve()
        try:
            rel = resolved.relative_to(self.repo_root)
            remote_path = f"{REMOTE_REPO_DIR}/{rel.as_posix()}"
            mkdir_cmd = f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}"
            self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", mkdir_cmd], timeout=45)
        except ValueError:
            remote_path = f"{REMOTE_CONFIGS_ROOT}/{job.run_name}.toml"
        rc, _ = self._colab(["upload", "-s", self.worker_name, str(resolved), remote_path], timeout=60)
        if rc != 0:
            raise RuntimeError(f"Failed to upload config {resolved} -> {remote_path} (rc={rc})")
        return remote_path

    def _pack_and_pull_remote_run(
        self,
        *,
        run_name: str,
        remote_run_dir: str,
        local_run_dir: Path,
        pull_checkpoints: str,
        export_bundle: bool,
        tag: str = "final",
    ) -> tuple[int, dict[str, Any] | None]:
        """Pack remote run artifacts on the VM, download tarball, and unpack safely locally."""
        tar_remote = f"/content/out-{run_name}-{tag}.tar.gz"
        tar_local = self.batch_dir / f"{run_name}-{tag}.tar.gz"
        pack_py = textwrap.dedent(
            f"""
            import json, os, shutil, subprocess, sys, tarfile
            from pathlib import Path

            run_dir = Path({remote_run_dir!r})
            tar_out = Path({tar_remote!r})
            pull_ckpts = {pull_checkpoints!r}
            want_bundle = {bool(export_bundle)!r}
            ext_log = Path({f"/content/logs/{run_name}.train.log"!r})

            run_dir.mkdir(parents=True, exist_ok=True)
            if ext_log.is_file():
                shutil.copy2(ext_log, run_dir / "train.log")

            # Ensure top-level EMA model.pt is staged if available.
            if want_bundle and not (run_dir / "model.pt").is_file():
                copied = False
                attempts_root = run_dir / "attempts"
                if attempts_root.is_dir():
                    candidates = sorted(attempts_root.glob("*/inference/model.pt"))
                    if candidates:
                        shutil.copy2(candidates[-1], run_dir / "model.pt")
                        copied = True
                if not copied and (run_dir / "checkpoints").is_dir():
                    subprocess.run(
                        ["hypergan", "export", str(run_dir), "--output", str(run_dir / "model.pt"), "--device", "cpu"],
                        check=False,
                        capture_output=True,
                    )

            keep_ckpts = set()
            ckpt_root = run_dir / "checkpoints"
            if ckpt_root.is_dir() and pull_ckpts != "none":
                all_dirs = [d.name for d in sorted(ckpt_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
                if pull_ckpts == "all":
                    keep_ckpts.update(all_dirs)
                else:
                    if pull_ckpts in ("latest", "best") and (ckpt_root / "latest.json").is_file():
                        try:
                            latest_name = json.loads((ckpt_root / "latest.json").read_text()).get("checkpoint")
                            if latest_name in all_dirs:
                                keep_ckpts.add(latest_name)
                        except Exception:
                            pass
                    if pull_ckpts == "best" and (run_dir / "summary.json").is_file():
                        try:
                            best = json.loads((run_dir / "summary.json").read_text()).get("best") or {{}}
                            best_ckpt = best.get("checkpoint")
                            if best_ckpt in all_dirs:
                                keep_ckpts.add(best_ckpt)
                        except Exception:
                            pass
                    if not keep_ckpts and all_dirs:
                        keep_ckpts.add(all_dirs[-1])

            top_files = ("summary.json", "manifest.json", "model.json", "console.json", "events.jsonl", "model.pt", "model.sha256", "train.log")
            top_dirs = ("metrics", "previews", "views", "config-source", "attempts", "launch")

            with tarfile.open(tar_out, "w:gz") as tf:
                for fname in top_files:
                    p = run_dir / fname
                    if p.is_file():
                        tf.add(p, arcname=fname)
                for dname in top_dirs:
                    d = run_dir / dname
                    if d.is_dir():
                        for child in sorted(d.rglob("*")):
                            if child.is_file() and not child.name.startswith(".tmp"):
                                tf.add(child, arcname=str(child.relative_to(run_dir)))
                if ckpt_root.is_dir() and pull_ckpts != "none":
                    if (ckpt_root / "latest.json").is_file():
                        tf.add(ckpt_root / "latest.json", arcname="checkpoints/latest.json")
                    for cname in sorted(keep_ckpts):
                        cdir = ckpt_root / cname
                        if cdir.is_dir():
                            for child in sorted(cdir.rglob("*")):
                                if child.is_file():
                                    tf.add(child, arcname=str(child.relative_to(run_dir)))
            print(f"[pack] wrote {{tar_out}} ({{tar_out.stat().st_size}} bytes)")
            """
        ).strip()
        pack_cmd = f"python3 - <<'PY'\n{pack_py}\nPY\ntouch /content/bench_heartbeat"
        with self._ssh_lock:
            pack_rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", pack_cmd], timeout=240)
            if pack_rc != 0:
                return pack_rc, None
            dl_rc, _ = self._colab(["download", "-s", self.worker_name, tar_remote, str(tar_local)], timeout=300)
        if dl_rc != 0 or not tar_local.is_file():
            return (dl_rc or 1), None
        unpack_info = unpack_run_tarball(tar_local, local_run_dir, remote_run_dir=remote_run_dir)
        return 0, unpack_info

    def run_training_job(
        self,
        job: BenchmarkJob,
        *,
        local_run_dir: Path,
        pull_checkpoints: str = "latest",
        force: bool = False,
        no_resume: bool = False,
        register_run_local: bool = True,
    ) -> dict[str, Any]:
        """Run one HyperGAN training job (with optional chunked/periodic checkpoint sync & resume)."""
        local_run_dir = Path(local_run_dir).resolve()
        remote_run_dir = f"{REMOTE_RUNS_ROOT}/{job.run_name}"
        self._log(f"Starting job {job.run_name} (config={job.config}, gpu={self.gpu})")

        self.prepare_dataset(job)
        remote_config = self._stage_job_config(job)

        target_step = job.target_step()
        local_state = inspect_local_run(local_run_dir, target_step=target_step)
        is_resume = bool(not force and not no_resume and local_state["resumable"])
        current_step = int(local_state["step"]) if is_resume else 0

        if is_resume:
            self._log(f"Uploading local run state (step {current_step}) for `hypergan resume`...")
            resume_tar = self.batch_dir / f"{job.run_name}-resume-upload.tar.gz"
            pack_local_run_for_resume(local_run_dir, resume_tar, remote_run_dir=remote_run_dir)
            remote_resume_tar = f"/content/resume-{job.run_name}.tar.gz"
            rc, _ = self._colab(["upload", "-s", self.worker_name, str(resume_tar), remote_resume_tar], timeout=300)
            if rc != 0:
                raise RuntimeError(f"Resume tarball upload failed for {job.run_name} (rc={rc})")
            unpack_cmd = (
                f"set -euo pipefail; rm -rf {shlex.quote(remote_run_dir)}; "
                f"mkdir -p {shlex.quote(remote_run_dir)}; "
                f"tar -xzf {shlex.quote(remote_resume_tar)} -C {shlex.quote(remote_run_dir)}; "
                f"rm -f {shlex.quote(remote_resume_tar)}; touch /content/bench_heartbeat"
            )
            rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", unpack_cmd], timeout=180)
            if rc != 0:
                raise RuntimeError(f"Resume tarball extraction failed on VM for {job.run_name} (rc={rc})")
        else:
            prep_cmd = f"rm -rf {shlex.quote(remote_run_dir)} && mkdir -p {shlex.quote(REMOTE_RUNS_ROOT)} /content/logs"
            self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", prep_cmd], timeout=60)

        # `hypergan train -d` maps directly to CUDA_VISIBLE_DEVICES, so single-GPU MUST be "0" (not "cuda").
        device_arg = "cpu" if self.gpu == "cpu" else "0"
        t0 = time.monotonic()
        chunks_completed = 0
        last_rc = 0
        last_unpack: dict[str, Any] | None = None
        disconnected = False

        while True:
            remaining_steps = None
            if target_step is not None:
                remaining_steps = target_step - current_step
                if remaining_steps <= 0:
                    break
            chunk_delta = (
                min(job.chunk_steps, remaining_steps)
                if (job.chunk_steps > 0 and remaining_steps is not None)
                else (job.chunk_steps if job.chunk_steps > 0 else remaining_steps)
            )

            if is_resume or chunks_completed > 0:
                hg_cmd = ["hypergan", "resume", remote_run_dir, "-d", device_arg, "--no-server"]
            else:
                hg_cmd = [
                    "hypergan",
                    "train",
                    remote_config,
                    "--run-dir",
                    remote_run_dir,
                    "-n",
                    job.run_name,
                    "-d",
                    device_arg,
                    "--no-server",
                ]
                if job.steps is not None:
                    hg_cmd += ["--steps", str(job.steps)]
                for ov in job.set_overrides:
                    hg_cmd += ["-s", ov]

            if chunk_delta is not None:
                hg_cmd += ["--stop-after-steps", str(chunk_delta)]
            if job.checkpoint_every is not None:
                hg_cmd += ["--checkpoint-every", str(job.checkpoint_every)]
            if job.no_previews:
                hg_cmd += ["--no-previews"]
            elif job.preview_every is not None:
                hg_cmd += ["--preview-every", str(job.preview_every)]
            if job.progress_every is not None:
                hg_cmd += ["--progress-every", str(job.progress_every)]

            quoted_hg = " ".join(shlex.quote(part) for part in hg_cmd)
            drive_sync_sh = ""
            if self.drive_dir:
                drive_target = f"{self.drive_dir.rstrip('/')}/{job.run_name}"
                drive_sync_sh = (
                    f"if [ -d /content/drive/MyDrive ]; then "
                    f"mkdir -p {shlex.quote(drive_target)} && "
                    f"cp -r {shlex.quote(remote_run_dir)}/summary.json {shlex.quote(remote_run_dir)}/manifest.json "
                    f"{shlex.quote(remote_run_dir)}/checkpoints {shlex.quote(drive_target)}/ 2>/dev/null || true; fi"
                )

            remote_train_log = f"/content/logs/{job.run_name}.train.log"
            run_sh = textwrap.dedent(
                f"""
                set -euo pipefail
                rm -f /content/bench_job.done
                touch /content/bench_heartbeat
                cd {REMOTE_REPO_DIR}
                mkdir -p {shlex.quote(REMOTE_RUNS_ROOT)} /content/logs
                set +e
                {quoted_hg} 2>&1 | tee -a {shlex.quote(remote_train_log)} /content/bench_job.log
                RUN_RC=${{PIPESTATUS[0]}}
                set -e
                {drive_sync_sh}
                touch /content/bench_heartbeat /content/bench_job.done
                echo "[vm] job={job.run_name} chunk={chunks_completed} rc=$RUN_RC"
                exit $RUN_RC
                """
            ).strip()

            # Optional background periodic checkpoint sync thread while continuous training runs.
            stop_sync = threading.Event()
            sync_thread: threading.Thread | None = None
            if job.sync_interval_minutes > 0 and job.chunk_steps == 0:
                interval_s = max(60.0, job.sync_interval_minutes * 60.0)

                def _periodic_sync_loop() -> None:
                    sync_idx = 0
                    while not stop_sync.wait(interval_s):
                        sync_idx += 1
                        self._log(f"Periodic checkpoint sync #{sync_idx} for {job.run_name}...")
                        try:
                            self._pack_and_pull_remote_run(
                                run_name=job.run_name,
                                remote_run_dir=remote_run_dir,
                                local_run_dir=local_run_dir,
                                pull_checkpoints="latest",
                                export_bundle=False,
                                tag=f"sync{sync_idx}",
                            )
                        except Exception as exc:
                            self._log(f"Periodic sync #{sync_idx} warning: {exc}")

                sync_thread = threading.Thread(target=_periodic_sync_loop, daemon=True)
                sync_thread.start()

            ssh_timeout = (
                int(job.timeout_minutes * 60)
                if job.timeout_minutes is not None
                else max(600, min(self.max_seconds, int(job.estimated_minutes * 60 * 2.0) + 600))
            )
            try:
                rc, ssh_out = self._colab(
                    ["ssh", "-s", self.worker_name, "--", "bash", "-lc", run_sh],
                    timeout=ssh_timeout,
                    capture=True,
                )
            finally:
                stop_sync.set()
                if sync_thread is not None:
                    sync_thread.join(timeout=5.0)

            last_rc = rc
            chunks_completed += 1

            # Detect Colab connection loss / timeout so orchestrator can trigger clean retry/resume.
            if rc == 124 or "Connection was lost" in ssh_out or "Websocket" in ssh_out:
                disconnected = True

            # Always pull back run artifacts (even on NaN/failure!) so telemetry is preserved locally.
            pull_rc, unpack_info = self._pack_and_pull_remote_run(
                run_name=job.run_name,
                remote_run_dir=remote_run_dir,
                local_run_dir=local_run_dir,
                pull_checkpoints=pull_checkpoints,
                export_bundle=job.export_bundle,
                tag=f"chunk{chunks_completed}",
            )
            if pull_rc == 0 and unpack_info is not None:
                last_unpack = unpack_info
                if register_run_local:
                    register_local_run(local_run_dir)
            elif disconnected:
                break

            refreshed = inspect_local_run(local_run_dir, target_step=target_step)
            current_step = int(refreshed["step"])

            if rc != 0:
                break
            if refreshed["completed"] or target_step is None or current_step >= target_step or job.chunk_steps <= 0:
                break
            is_resume = True

        elapsed = time.monotonic() - t0
        telemetry = analyze_run_telemetry(local_run_dir)
        if self.gpu != "cpu" and telemetry.get("cpu_fallback_detected"):
            self._log(
                f"ERROR: GPU run {job.run_name} silently fell back to CPU on {self.gpu} VM; marking failed."
            )
            if last_rc == 0:
                last_rc = 2
        self._log(
            f"Finished job {job.run_name} in {elapsed:.1f}s "
            f"(rc={last_rc}, step={telemetry['step']}, status={telemetry['status']}, "
            f"best_fid={telemetry['best_fid']}, nan={telemetry['nan_detected']})"
        )
        return {
            "run_name": job.run_name,
            "config": job.config,
            "requested_gpu": self.requested_gpu,
            "gpu": self.gpu,
            "cuda_model": self.actual_cuda_model or COLAB_GPU_SPECS[self.gpu]["cuda_model"] or "CPU",
            "local_run_dir": str(local_run_dir),
            "run_rc": last_rc,
            "disconnected": disconnected,
            "chunks_completed": chunks_completed,
            "elapsed_seconds": round(elapsed, 2),
            "unpack": last_unpack,
            "telemetry": telemetry,
        }

    def run_bundle_evaluations(
        self,
        *,
        bundles: list[dict[str, Any]],
        metrics: list[str],
        dataset_config: dict[str, Any],
        samples: int = 50000,
        batch_size: int = 128,
        seed: int = 34002,
        dataset_prep_cmd: str | None = None,
        dataset_tarball: str | None = None,
        dataset_target_dir: str | None = None,
        output_dir: Path,
        import_to_run: bool = False,
    ) -> dict[str, Any]:
        """Upload one or more EMA `model.pt` bundles to this Colab GPU VM and run snapshot metrics."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dataset once on the VM.
        dataset_name = str(dataset_config.get("_prepare_dataset") or dataset_config.get("builtin") or "none").lower()
        clean_dataset_config = {k: v for k, v in dataset_config.items() if not k.startswith("_")}
        prep_job = BenchmarkJob(
            run_name="eval-prep",
            config="eval-prep",
            gpu=self.gpu,
            dataset=dataset_name,
            dataset_prep_cmd=dataset_prep_cmd,
            dataset_tarball=dataset_tarball,
            dataset_target_dir=dataset_target_dir,
        )
        self.prepare_dataset(prep_job)

        remote_manifest_items: list[dict[str, Any]] = []
        for idx, item in enumerate(bundles):
            local_bundle = Path(item["bundle_path"]).resolve()
            if not local_bundle.is_file():
                raise FileNotFoundError(f"EMA bundle not found: {local_bundle}")
            bundle_id = sanitize_name(item.get("name") or local_bundle.parent.name or f"bundle-{idx}")
            remote_bundle = f"{REMOTE_EVAL_ROOT}/{bundle_id}.pt"
            self._log(f"Uploading EMA bundle {local_bundle} -> {remote_bundle}...")
            rc, _ = self._colab(["upload", "-s", self.worker_name, str(local_bundle), remote_bundle], timeout=300)
            if rc != 0:
                raise RuntimeError(f"Failed to upload bundle {local_bundle} (rc={rc})")
            remote_manifest_items.append(
                {
                    "name": bundle_id,
                    "local_bundle_path": str(local_bundle),
                    "local_run_dir": item.get("run_dir"),
                    "remote_bundle_path": remote_bundle,
                }
            )

        resolved_metrics = []
        cli_metrics: list[str] = []
        custom_metrics: list[dict[str, str]] = []
        for raw_m in metrics:
            alias = IMAGE_METRICS_CLI_ALIASES.get(raw_m)
            if alias is not None:
                if alias not in cli_metrics:
                    cli_metrics.append(alias)
                resolved_metrics.append({"metric_id": alias, "name": raw_m, "cli_alias": alias})
            else:
                factory = KNOWN_EVAL_FACTORIES.get(raw_m, raw_m)
                metric_id = sanitize_name(raw_m.split(":")[-1].lower())
                item_m = {"metric_id": metric_id, "name": raw_m, "factory": factory}
                custom_metrics.append(item_m)
                resolved_metrics.append(item_m)

        eval_spec_payload = {
            "bundles": remote_manifest_items,
            "metrics": resolved_metrics,
            "cli_metrics": cli_metrics,
            "custom_metrics": custom_metrics,
            "dataset": clean_dataset_config,
            "samples": samples,
            "batch_size": batch_size,
            "seed": seed,
            "device": "cpu" if self.gpu == "cpu" else "cuda",
        }
        local_spec_json = self.batch_dir / "eval-spec.json"
        _atomic_write_json(local_spec_json, eval_spec_payload)
        rc, _ = self._colab(
            ["upload", "-s", self.worker_name, str(local_spec_json), f"{REMOTE_EVAL_ROOT}/eval-spec.json"],
            timeout=60,
        )
        if rc != 0:
            raise RuntimeError(f"Failed to upload eval-spec.json (rc={rc})")

        runner_py = textwrap.dedent(
            f"""
            import hashlib, json, subprocess, sys, time, traceback
            from pathlib import Path
            import torch

            from hypergan.metric_evaluation_worker import evaluate_snapshot
            from hypergan.metric_plugins import _factory

            spec_doc = json.loads(Path("{REMOTE_EVAL_ROOT}/eval-spec.json").read_text(encoding="utf-8"))
            out_dir = Path("{REMOTE_EVAL_ROOT}/receipts")
            out_dir.mkdir(parents=True, exist_ok=True)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

            def sha256_file(path: Path) -> str:
                h = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()

            def lower_dataset(ds: dict) -> dict:
                eval_data = dict(ds)
                if "builtin" in eval_data:
                    builtin = str(eval_data["builtin"]).strip()
                    d_args = dict(eval_data.get("args") or {{}})
                    if builtin == "cifar10":
                        return {{"factory": "hypergan.image_data:CIFAR10Data", "args": d_args}}
                    return {{"factory": builtin, "args": d_args}}
                return eval_data

            data_cfg = lower_dataset(spec_doc["dataset"])
            cli_metrics = list(spec_doc.get("cli_metrics") or [])
            custom_metrics = list(spec_doc.get("custom_metrics") or [])

            summary_receipts = []
            for bitem in spec_doc["bundles"]:
                bpath = Path(bitem["remote_bundle_path"])
                bsha = sha256_file(bpath)
                saved = torch.load(bpath, map_location="cpu", weights_only=True)
                owner = saved.get("identity") or {{}}
                run_id = owner.get("run_id") or bitem["name"]
                step = int(saved.get("step") or 0)
                attempt_id = owner.get("attempt_id") or "eval"

                # 1. Built-in publication metrics via `python -m hypergan.image_metrics`
                if cli_metrics:
                    t0 = time.monotonic()
                    bundle_cli_out = out_dir / f"{{bitem['name']}}__image_metrics.json"
                    cli_cmd = [
                        sys.executable,
                        "-m",
                        "hypergan.image_metrics",
                        str(bpath),
                        "--metrics",
                        *cli_metrics,
                        "--samples",
                        str(int(spec_doc["samples"])),
                        "--batch-size",
                        str(int(spec_doc["batch_size"])),
                        "--seed",
                        str(int(spec_doc["seed"])),
                        "--device",
                        str(spec_doc["device"]),
                        "--output",
                        str(bundle_cli_out),
                    ]
                    raw_ds = spec_doc["dataset"]
                    if raw_ds.get("builtin") == "cifar10":
                        d_args = raw_ds.get("args") or {{}}
                        cli_cmd += [
                            "--cifar10-root",
                            str(Path(d_args.get("root", "~/.hypergan/cache/cifar10")).expanduser()),
                            "--split",
                            str(d_args.get("split", "train")),
                        ]
                    elif data_cfg.get("factory") == "image_folder":
                        d_args = data_cfg.get("args") or {{}}
                        cli_cmd += ["--data-folder", str(Path(str(d_args["root"])).expanduser())]
                        if d_args.get("width") is not None:
                            cli_cmd += ["--width", str(d_args["width"])]
                        if d_args.get("height") is not None:
                            cli_cmd += ["--height", str(d_args["height"])]
                    else:
                        cli_cmd += ["--data-json", json.dumps(data_cfg)]

                    proc = subprocess.run(cli_cmd, capture_output=True, text=True)
                    if proc.stdout:
                        print(proc.stdout, end="", flush=True)
                    if proc.stderr:
                        print(proc.stderr, file=sys.stderr, end="", flush=True)
                    elapsed_cli = round(time.monotonic() - t0, 2)

                    if proc.returncode == 0 and bundle_cli_out.is_file():
                        doc = json.loads(bundle_cli_out.read_text(encoding="utf-8"))
                        for mkey, entry in (doc.get("metrics") or {{}}).items():
                            rec = {{
                                "schema_version": 1,
                                "bundle_name": bitem["name"],
                                "local_bundle_path": bitem["local_bundle_path"],
                                "local_run_dir": bitem.get("local_run_dir"),
                                "bundle_sha256": doc.get("bundle_sha256", bsha),
                                "metric_id": mkey,
                                "metric_name": mkey,
                                "factory": f"python -m hypergan.image_metrics --metrics {{mkey}}",
                                "kind": entry.get("kind", "scalars"),
                                "step": int(doc.get("step", step)),
                                "attempt_id": attempt_id,
                                "gpu_model": gpu_name,
                                "status": "complete",
                                "error": None,
                                "values": entry.get("values"),
                                "outputs": entry.get("outputs"),
                                "protocol": {{
                                    "schema_version": 1,
                                    "metric_alias": mkey,
                                    "sample_count": doc.get("sample_count"),
                                    "batch_size": doc.get("batch_size"),
                                    "seed": doc.get("seed"),
                                    "data_seed": doc.get("data_seed"),
                                    "prior_seed": doc.get("prior_seed"),
                                    "device": doc.get("device"),
                                    "generated_binding": doc.get("generated_binding"),
                                    "data_identity": doc.get("data_identity"),
                                    "ema": True,
                                }},
                                "snapshot_identity": dict(doc.get("snapshot_identity") or owner, source_step=int(doc.get("step", step))),
                                "elapsed_seconds": round(float(entry.get("seconds", elapsed_cli)), 2),
                            }}
                            if "value" in entry:
                                rec["value"] = entry["value"]
                            rfile = out_dir / f"{{bitem['name']}}__{{mkey}}.json"
                            rfile.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
                            summary_receipts.append(rec)
                            print(f"[eval] {{bitem['name']}} / {{mkey}}: values={{rec.get('values')}} ({{rec['elapsed_seconds']}}s)", flush=True)
                    else:
                        for mkey in cli_metrics:
                            rec = {{
                                "schema_version": 1,
                                "bundle_name": bitem["name"],
                                "local_bundle_path": bitem["local_bundle_path"],
                                "local_run_dir": bitem.get("local_run_dir"),
                                "bundle_sha256": bsha,
                                "metric_id": mkey,
                                "metric_name": mkey,
                                "step": step,
                                "attempt_id": attempt_id,
                                "gpu_model": gpu_name,
                                "status": "failed",
                                "error": f"python -m hypergan.image_metrics failed rc={{proc.returncode}}: {{proc.stderr[-400:]}}",
                                "elapsed_seconds": elapsed_cli,
                            }}
                            summary_receipts.append(rec)
                    Path("/content/bench_heartbeat").touch()

                # 2. Custom `module:Class` snapshot evaluators via evaluate_snapshot
                for mitem in custom_metrics:
                    t0 = time.monotonic()
                    rec = {{
                        "schema_version": 1,
                        "bundle_name": bitem["name"],
                        "local_bundle_path": bitem["local_bundle_path"],
                        "local_run_dir": bitem.get("local_run_dir"),
                        "bundle_sha256": bsha,
                        "metric_id": mitem["metric_id"],
                        "metric_name": mitem["name"],
                        "factory": mitem["factory"],
                        "step": step,
                        "attempt_id": attempt_id,
                        "gpu_model": gpu_name,
                        "status": "complete",
                        "error": None,
                    }}
                    try:
                        _, expected = _factory(mitem["factory"], {{}})
                        descriptor = expected["descriptor"]
                        eval_spec = {{
                            "factory": mitem["factory"],
                            "args": {{}},
                            "inputs": {{"generated": "evaluation.generated", "reference": "evaluation.reference"}},
                            "evaluation": {{
                                "device": spec_doc["device"],
                                "sample_count": int(spec_doc["samples"]),
                                "batch_size": int(spec_doc["batch_size"]),
                                "seed": int(spec_doc["seed"]),
                                "data": data_cfg,
                            }},
                        }}
                        result = evaluate_snapshot(
                            eval_spec,
                            expected,
                            str(bpath),
                            bsha,
                            {{"run_id": run_id}},
                        )
                        rec["kind"] = descriptor["kind"]
                        if descriptor["kind"] == "scalars":
                            rec["values"] = result["values"]
                            rec["outputs"] = descriptor["outputs"]
                        else:
                            rec["value"] = result["value"]
                            rec["values"] = {{mitem["metric_id"]: result["value"]}}
                            rec["outputs"] = {{
                                mitem["metric_id"]: {{
                                    "label": descriptor.get("label", mitem["name"]),
                                    "unit": descriptor.get("unit", ""),
                                    "direction": descriptor.get("direction", "minimize"),
                                }}
                            }}
                        rec["protocol"] = result["protocol"]
                        rec["snapshot_identity"] = result["snapshot_identity"]
                    except Exception as exc:
                        rec["status"] = "failed"
                        rec["error"] = f"{{type(exc).__name__}}: {{exc}}"
                        traceback.print_exc()
                    rec["elapsed_seconds"] = round(time.monotonic() - t0, 2)
                    rfile = out_dir / f"{{bitem['name']}}__{{mitem['metric_id']}}.json"
                    rfile.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
                    summary_receipts.append(rec)
                    Path("/content/bench_heartbeat").touch()
                    print(f"[eval] {{bitem['name']}} / {{mitem['name']}}: status={{rec['status']}} values={{rec.get('values')}} ({{rec['elapsed_seconds']}}s)", flush=True)

            (out_dir / "all-receipts.json").write_text(json.dumps(summary_receipts, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            """
        ).strip()

        runner_local = self.batch_dir / "eval_runner.py"
        runner_local.write_text(runner_py + "\n", encoding="utf-8")
        up_rc, _ = self._colab(
            ["upload", "-s", self.worker_name, str(runner_local), f"{REMOTE_EVAL_ROOT}/eval_runner.py"],
            timeout=120,
        )
        if up_rc != 0:
            raise RuntimeError(f"Failed to upload eval_runner.py (rc={up_rc})")

        eval_sh = textwrap.dedent(
            f"""
            set -euo pipefail
            rm -f /content/bench_job.done
            touch /content/bench_heartbeat
            cd {REMOTE_REPO_DIR}
            python3 {REMOTE_EVAL_ROOT}/eval_runner.py 2>&1 | tee -a /content/bench_job.log
            tar -czf /content/eval-receipts.tar.gz -C {REMOTE_EVAL_ROOT}/receipts .
            touch /content/bench_heartbeat /content/bench_job.done
            """
        ).strip()

        t0 = time.monotonic()
        timeout_s = max(900, min(self.max_seconds, len(bundles) * len(metrics) * 900 + 300))
        rc, _ = self._colab(["ssh", "-s", self.worker_name, "--", "bash", "-lc", eval_sh], timeout=timeout_s)
        elapsed = time.monotonic() - t0

        local_tar = self.batch_dir / "eval-receipts.tar.gz"
        dl_rc, _ = self._colab(["download", "-s", self.worker_name, "/content/eval-receipts.tar.gz", str(local_tar)], timeout=180)
        if dl_rc != 0 or not local_tar.is_file():
            raise RuntimeError(f"Failed to download evaluation receipts (run_rc={rc}, dl_rc={dl_rc})")

        unpack_run_tarball(local_tar, output_dir)
        all_receipts_path = output_dir / "all-receipts.json"
        receipts = json.loads(all_receipts_path.read_text(encoding="utf-8")) if all_receipts_path.is_file() else []

        imported: list[dict[str, Any]] = []
        if import_to_run:
            allowed_output_meta = {"label", "unit", "direction", "description"}
            for rec in receipts:
                if rec.get("status") != "complete" or not rec.get("local_run_dir"):
                    continue
                target_run = Path(rec["local_run_dir"])
                if not (target_run / "manifest.json").is_file():
                    continue
                try:
                    from hypergan.evaluation_results import import_scores

                    raw_outputs = rec.get("outputs") or {}
                    clean_outputs = {
                        out_key: {mk: mv for mk, mv in meta.items() if mk in allowed_output_meta}
                        for out_key, meta in raw_outputs.items()
                        if isinstance(meta, dict)
                    }
                    imp = import_scores(
                        target_run,
                        rec["metric_id"],
                        step=int(rec["step"]),
                        values=rec["values"],
                        outputs=clean_outputs,
                        attempt_id=rec.get("attempt_id"),
                        snapshot_identity=rec.get("snapshot_identity"),
                        snapshot_sha256=rec.get("bundle_sha256"),
                        protocol=rec.get("protocol"),
                    )
                    imported.append({"run_dir": str(target_run), "metric_id": rec["metric_id"], "evaluation_id": imp.get("evaluation_id")})
                except Exception as exc:
                    imported.append({"run_dir": str(target_run), "metric_id": rec["metric_id"], "error": str(exc)})

        return {
            "run_rc": rc,
            "elapsed_seconds": round(elapsed, 2),
            "output_dir": str(output_dir),
            "receipts": receipts,
            "imported": imported,
        }

    def stop(self) -> None:
        if not self.alive:
            if self.started_mono is not None and self.stopped_mono is None:
                self.stopped_mono = time.monotonic()
            return
        self.alive = False
        self.registry.unregister(self.worker_name)
        self._log(f"Stopping Colab VM {self.worker_name}...")
        self._colab(["stop", "-s", self.worker_name], timeout=30)
        self.stopped_mono = time.monotonic()
        cost = self.cost_summary()
        self._log(
            f"Session released (wall={cost['wall_seconds']}s, "
            f"gpu_hours={cost['gpu_hours']}, est_cu={cost['estimated_cu']})."
        )


# ---------------------------------------------------------------------------
# Parallel warm-pool benchmark executor
# ---------------------------------------------------------------------------


def execute_benchmark_jobs(
    *,
    repo_root: Path,
    runs_dir: Path,
    state_dir: Path,
    jobs: list[BenchmarkJob],
    workers: int = HARD_MAX_WORKERS,
    max_workers: int = HARD_MAX_WORKERS,
    allow_over_cap: bool = False,
    stagger_seconds: float = DEFAULT_STAGGER_SECONDS,
    boot_timeout: int = 260,
    boot_retries: int = 2,
    fallback_gpus: list[str] | str | None = None,
    max_minutes: int = 240,
    idle_grace_minutes: int = 10,
    stall_minutes: int = 30,
    repo_url: str = DEFAULT_REPO_URL,
    ref: str = "HEAD",
    source_mode: str = "auto",
    colab_bin: str = "colab",
    pull_checkpoints: str = "latest",
    force: bool = False,
    no_resume: bool = False,
    register_runs: bool = True,
    dry_run: bool = False,
    particlegan_spec: str = DEFAULT_PARTICLEGAN_SPEC,
    hndl_spec: str = DEFAULT_HNDL_SPEC,
    torch_fidelity_spec: str = DEFAULT_TORCH_FIDELITY_SPEC,
    foldlitedb_spec: str = DEFAULT_FOLDLITEDB_SPEC,
    safetensors_spec: str = DEFAULT_SAFETENSORS_SPEC,
    drive_dir: str | None = None,
) -> dict[str, Any]:
    """Schedule and execute benchmark jobs across up to `max_workers` warm Colab GPU VMs."""
    validate_worker_count(workers, max_workers=max_workers, allow_over_cap=allow_over_cap)
    repo_root = Path(repo_root).resolve()
    runs_dir = Path(runs_dir).resolve()
    state_dir = Path(state_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    _GLOBAL_REGISTRY.colab_bin = colab_bin
    _GLOBAL_REGISTRY.state_dir = state_dir

    skipped: list[dict[str, Any]] = []
    pending: list[BenchmarkJob] = []
    for job in jobs:
        local_dir = runs_dir / job.run_name
        local_state = inspect_local_run(local_dir, target_step=job.target_step())
        if not force and local_state["completed"]:
            skipped.append(
                {
                    "run_name": job.run_name,
                    "status": "skipped_completed",
                    "step": local_state["step"],
                    "target_step": job.target_step(),
                    "best_fid": local_state["best_fid"],
                    "local_run_dir": str(local_dir),
                }
            )
        else:
            pending.append(job)

    if not pending:
        return {
            "schema_version": 1,
            "completed_at": utc_stamp(),
            "status": "already_completed",
            "skipped": skipped,
            "results": [],
            "workers_cost": [],
            "total_gpu_hours": 0.0,
            "total_estimated_cu": 0.0,
        }

    # Plan GPU-matched worker slots (never exceeding min(workers, len(pending))).
    by_gpu_list: dict[str, list[BenchmarkJob]] = {}
    for job in pending:
        by_gpu_list.setdefault(job.gpu, []).append(job)

    total_workers = min(workers, len(pending))
    worker_plan: list[tuple[str, str]] = []
    stamp = run_id_stamp()
    batch_dir = state_dir / f"bench-{stamp}"
    gpu_keys = sorted(by_gpu_list.keys())
    w_idx = 0
    while len(worker_plan) < total_workers:
        added = False
        for gpu in gpu_keys:
            if len(worker_plan) >= total_workers:
                break
            assigned = sum(1 for _, g in worker_plan if g == gpu)
            if assigned < len(by_gpu_list[gpu]):
                w_idx += 1
                worker_plan.append((f"hg-{gpu.lower()}-w{w_idx}-{stamp[-6:]}", gpu))
                added = True
        if not added:
            break

    if dry_run:
        return {
            "schema_version": 1,
            "status": "dry_run",
            "batch_dir": str(batch_dir),
            "runs_dir": str(runs_dir),
            "worker_plan": [{"worker": w, "gpu": g} for w, g in worker_plan],
            "pending_jobs": [j.to_dict() for j in pending],
            "skipped": skipped,
        }

    batch_dir.mkdir(parents=True, exist_ok=True)
    source_sync = prepare_source_bundle(repo_root, batch_dir, ref=ref, source_mode=source_mode)
    _atomic_write_json(
        batch_dir / "plan.json",
        {
            "created_at": utc_stamp(),
            "source_sync": source_sync,
            "worker_plan": worker_plan,
            "pending_jobs": [j.to_dict() for j in pending],
            "skipped": skipped,
        },
    )

    queues: dict[str, stdlib_queue.Queue[tuple[BenchmarkJob, int]]] = {}
    for gpu, jlist in by_gpu_list.items():
        q: stdlib_queue.Queue[tuple[BenchmarkJob, int]] = stdlib_queue.Queue()
        for job in jlist:
            q.put((job, 0))
        queues[gpu] = q

    results_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    worker_costs: list[dict[str, Any]] = []
    worker_errors: list[dict[str, Any]] = []

    def _run_worker_slot(slot_idx: int, base_name: str, gpu: str) -> None:
        if slot_idx > 0 and stagger_seconds > 0:
            time.sleep(slot_idx * stagger_seconds)
        q = queues[gpu]
        reincarnation = 0
        while True:
            if q.empty():
                break
            worker_name = base_name if reincarnation == 0 else f"{base_name}-r{reincarnation}"
            worker = ColabBenchWorker(
                worker_name=worker_name,
                gpu=gpu,
                batch_dir=batch_dir,
                state_dir=state_dir,
                repo_root=repo_root,
                repo_url=repo_url,
                source_sync=source_sync,
                colab_bin=colab_bin,
                boot_timeout=boot_timeout,
                boot_retries=boot_retries,
                fallback_gpus=fallback_gpus,
                max_minutes=max_minutes,
                idle_grace_minutes=idle_grace_minutes,
                stall_minutes=stall_minutes,
                particlegan_spec=particlegan_spec,
                hndl_spec=hndl_spec,
                torch_fidelity_spec=torch_fidelity_spec,
                foldlitedb_spec=foldlitedb_spec,
                safetensors_spec=safetensors_spec,
                drive_dir=drive_dir,
            )
            need_reprovision = False
            try:
                try:
                    worker.start()
                except (RuntimeError, OSError, ValueError) as exc:
                    with results_lock:
                        worker_errors.append({"worker": worker_name, "gpu": gpu, "stage": "start", "error": str(exc)})
                    worker._log(f"Worker startup failed: {exc}")
                    break

                while True:
                    try:
                        job, attempt_num = q.get_nowait()
                    except stdlib_queue.Empty:
                        break
                    try:
                        res = worker.run_training_job(
                            job,
                            local_run_dir=runs_dir / job.run_name,
                            pull_checkpoints=pull_checkpoints,
                            force=force if attempt_num == 0 else False,
                            no_resume=no_resume,
                            register_run_local=register_runs,
                        )
                        res["attempt"] = attempt_num + 1
                        if res["disconnected"] and attempt_num < job.max_retries:
                            worker._log(
                                f"Colab disconnect detected for {job.run_name}; "
                                f"re-queueing for automatic resume (retry {attempt_num + 1}/{job.max_retries})."
                            )
                            q.put((job, attempt_num + 1))
                            need_reprovision = True
                            break
                        with results_lock:
                            results.append(res)
                    except (RuntimeError, OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as exc:
                        with results_lock:
                            worker_errors.append(
                                {"worker": worker_name, "gpu": gpu, "stage": f"job:{job.run_name}", "error": str(exc)}
                            )
                        worker._log(f"Job {job.run_name} errored: {exc}")
                        if attempt_num < job.max_retries:
                            q.put((job, attempt_num + 1))
                            need_reprovision = True
                            break
                        with results_lock:
                            results.append(
                                {
                                    "run_name": job.run_name,
                                    "config": job.config,
                                    "gpu": gpu,
                                    "local_run_dir": str(runs_dir / job.run_name),
                                    "run_rc": 1,
                                    "error": str(exc),
                                    "attempt": attempt_num + 1,
                                }
                            )
                    finally:
                        q.task_done()
            finally:
                worker.stop()
                with results_lock:
                    worker_costs.append(worker.cost_summary())

            if not need_reprovision:
                break
            reincarnation += 1
            if reincarnation > 3:
                break
            time.sleep(min(20.0, max(5.0, stagger_seconds)))

    with ThreadPoolExecutor(max_workers=len(worker_plan)) as pool:
        futs = [pool.submit(_run_worker_slot, i, wname, gpu) for i, (wname, gpu) in enumerate(worker_plan)]
        for fut in as_completed(futs):
            fut.result()

    if not results and worker_errors:
        raise RuntimeError(f"All Colab workers failed to start/run: {worker_errors[:3]}")

    total_wall = sum(c["wall_seconds"] for c in worker_costs)
    total_gpu_hours = round(sum(c["gpu_hours"] for c in worker_costs), 4)
    total_cu = round(sum(c["estimated_cu"] for c in worker_costs), 4)
    all_ok = all(r.get("run_rc") == 0 for r in results) and len(results) == len(pending)

    summary = {
        "schema_version": 1,
        "completed_at": utc_stamp(),
        "status": "complete" if all_ok else "partial_or_failed",
        "batch_dir": str(batch_dir),
        "runs_dir": str(runs_dir),
        "source_sync": source_sync,
        "skipped": skipped,
        "results": results,
        "worker_errors": worker_errors,
        "workers_cost": worker_costs,
        "total_wall_seconds": round(total_wall, 2),
        "total_gpu_hours": total_gpu_hours,
        "total_estimated_cu": total_cu,
    }
    _atomic_write_json(batch_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = args.root.resolve()
    runs_dir = (args.runs_dir if args.runs_dir is not None else repo_root / "runs").resolve()
    state_dir = (args.state_dir if args.state_dir is not None else repo_root / "runs" / "colab").resolve()
    colab_bin = resolve_colab_bin(args.colab_bin)

    validate_worker_count(args.workers, max_workers=args.max_workers, allow_over_cap=args.allow_over_cap)

    jobs: list[BenchmarkJob] = []
    if args.manifest:
        cli_overrides = {
            "gpu": args.gpu if args.gpu_explicit else None,
            "steps": args.steps,
            "stop_after_steps": args.stop_after_steps,
            "checkpoint_every": args.checkpoint_every,
            "preview_every": args.preview_every,
            "no_previews": True if args.no_previews else None,
            "progress_every": args.progress_every,
            "set_overrides": args.set_overrides or None,
            "chunk_steps": args.chunk_steps if args.chunk_steps > 0 else None,
            "sync_interval_minutes": args.sync_interval_minutes if args.sync_interval_minutes > 0 else None,
            "drive_dir": args.drive_dir,
            "max_retries": args.max_retries,
        }
        jobs = load_manifest(args.manifest, repo_root=repo_root, default_gpu=args.gpu, cli_overrides=cli_overrides)
        if args.only:
            wanted = set(args.only)
            jobs = [j for j in jobs if j.run_name in wanted]
            if not jobs:
                raise ValueError(f"--only filter {sorted(wanted)} matched no jobs in {args.manifest}")
    elif args.config:
        run_name = sanitize_name(args.run_name or Path(args.config).stem)
        resolved_cfg = _resolve_config_ref(args.config, manifest_dir=Path.cwd(), repo_root=repo_root)
        jobs = [
            BenchmarkJob(
                run_name=run_name,
                config=resolved_cfg,
                gpu=normalize_gpu(args.gpu),
                steps=args.steps,
                stop_after_steps=args.stop_after_steps,
                checkpoint_every=args.checkpoint_every,
                preview_every=args.preview_every,
                no_previews=args.no_previews,
                progress_every=args.progress_every,
                set_overrides=list(args.set_overrides or []),
                dataset=args.dataset,
                dataset_prep_cmd=args.dataset_prep_cmd,
                dataset_tarball=str(Path(args.dataset_tarball).resolve()) if args.dataset_tarball else None,
                dataset_target_dir=args.dataset_target_dir,
                chunk_steps=args.chunk_steps,
                sync_interval_minutes=args.sync_interval_minutes,
                max_retries=args.max_retries,
                export_bundle=not args.no_export_bundle,
                drive_dir=args.drive_dir,
                estimated_minutes=args.estimated_minutes,
                timeout_minutes=args.timeout_minutes,
            )
        ]
    else:
        raise ValueError("Specify either --config <model.toml> or --manifest <jobs.json|jobs.toml>")

    summary = execute_benchmark_jobs(
        repo_root=repo_root,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=jobs,
        workers=args.workers,
        max_workers=args.max_workers,
        allow_over_cap=args.allow_over_cap,
        stagger_seconds=args.stagger_seconds,
        boot_timeout=args.boot_timeout,
        boot_retries=args.boot_retries,
        fallback_gpus=args.fallback_gpus,
        max_minutes=args.max_minutes,
        idle_grace_minutes=args.idle_grace_minutes,
        stall_minutes=args.stall_minutes,
        repo_url=args.repo_url,
        ref=args.ref,
        source_mode=args.source_mode,
        colab_bin=colab_bin,
        pull_checkpoints=args.pull_checkpoints,
        force=args.force,
        no_resume=args.no_resume,
        register_runs=not args.no_register,
        dry_run=args.dry_run,
        particlegan_spec=args.particlegan_spec,
        hndl_spec=args.hndl_spec,
        torch_fidelity_spec=args.torch_fidelity_spec,
        foldlitedb_spec=args.foldlitedb_spec,
        safetensors_spec=args.safetensors_spec,
        drive_dir=args.drive_dir,
    )
    print(json.dumps(summary, indent=2))
    if args.dry_run or summary.get("status") in ("complete", "already_completed"):
        return 0
    return 1


def _resolve_eval_bundle_specs(raw_targets: list[str]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for raw in raw_targets:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            candidate = p / "model.pt"
            if not candidate.is_file():
                raise FileNotFoundError(f"Run directory {p} does not contain model.pt")
            resolved.append({"name": p.name, "bundle_path": str(candidate), "run_dir": str(p)})
        elif p.is_file():
            run_dir = str(p.parent) if (p.parent / "manifest.json").is_file() else None
            name = p.parent.name if run_dir else p.stem
            resolved.append({"name": name, "bundle_path": str(p), "run_dir": run_dir})
        else:
            raise FileNotFoundError(f"Bundle or run directory not found: {p}")
    return resolved


def execute_eval_bundles(
    *,
    repo_root: Path,
    state_dir: Path,
    bundles: list[dict[str, Any]],
    metrics: list[str],
    dataset_config: dict[str, Any],
    gpu: str = "L4",
    samples: int = 50000,
    batch_size: int = 128,
    seed: int = 34002,
    dataset_prep_cmd: str | None = None,
    dataset_tarball: str | None = None,
    dataset_target_dir: str | None = None,
    output_dir: Path,
    import_to_run: bool = False,
    boot_timeout: int = 260,
    boot_retries: int = 2,
    fallback_gpus: list[str] | str | None = None,
    max_minutes: int = 120,
    idle_grace_minutes: int = 10,
    stall_minutes: int = 25,
    repo_url: str = DEFAULT_REPO_URL,
    ref: str = "HEAD",
    source_mode: str = "auto",
    colab_bin: str = "colab",
    particlegan_spec: str = DEFAULT_PARTICLEGAN_SPEC,
    hndl_spec: str = DEFAULT_HNDL_SPEC,
    torch_fidelity_spec: str = DEFAULT_TORCH_FIDELITY_SPEC,
    foldlitedb_spec: str = DEFAULT_FOLDLITEDB_SPEC,
    safetensors_spec: str = DEFAULT_SAFETENSORS_SPEC,
) -> dict[str, Any]:
    """Run one or more snapshot metric suites over local EMA bundles on a single Colab GPU VM."""
    repo_root = Path(repo_root).resolve()
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    _GLOBAL_REGISTRY.colab_bin = colab_bin
    _GLOBAL_REGISTRY.state_dir = state_dir

    stamp = run_id_stamp()
    batch_dir = state_dir / f"eval-{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    source_sync = prepare_source_bundle(repo_root, batch_dir, ref=ref, source_mode=source_mode)

    worker = ColabBenchWorker(
        worker_name=f"hg-eval-{normalize_gpu(gpu).lower()}-{stamp[-6:]}",
        gpu=gpu,
        batch_dir=batch_dir,
        state_dir=state_dir,
        repo_root=repo_root,
        repo_url=repo_url,
        source_sync=source_sync,
        colab_bin=colab_bin,
        boot_timeout=boot_timeout,
        boot_retries=boot_retries,
        fallback_gpus=fallback_gpus,
        max_minutes=max_minutes,
        idle_grace_minutes=idle_grace_minutes,
        stall_minutes=stall_minutes,
        particlegan_spec=particlegan_spec,
        hndl_spec=hndl_spec,
        torch_fidelity_spec=torch_fidelity_spec,
        foldlitedb_spec=foldlitedb_spec,
        safetensors_spec=safetensors_spec,
    )
    try:
        worker.start()
        eval_res = worker.run_bundle_evaluations(
            bundles=bundles,
            metrics=metrics,
            dataset_config=dataset_config,
            samples=samples,
            batch_size=batch_size,
            seed=seed,
            dataset_prep_cmd=dataset_prep_cmd,
            dataset_tarball=dataset_tarball,
            dataset_target_dir=dataset_target_dir,
            output_dir=output_dir,
            import_to_run=import_to_run,
        )
    finally:
        worker.stop()

    cost = worker.cost_summary()
    summary = {
        "schema_version": 1,
        "completed_at": utc_stamp(),
        "batch_dir": str(batch_dir),
        "worker_cost": cost,
        **eval_res,
    }
    _atomic_write_json(batch_dir / "summary.json", summary)
    return summary


def cmd_eval_bundles(args: argparse.Namespace) -> int:
    repo_root = args.root.resolve()
    state_dir = (args.state_dir if args.state_dir is not None else repo_root / "runs" / "colab").resolve()
    colab_bin = resolve_colab_bin(args.colab_bin)
    bundles = _resolve_eval_bundle_specs(args.bundle)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("--metrics must list at least one metric (e.g. InceptionFID,InceptionSuite,DINOv2Suite)")

    if args.dataset_json:
        dataset_config = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    elif args.dataset_dir:
        dataset_config = {
            "factory": "image_folder",
            "args": {
                "root": args.dataset_dir,
                "width": args.image_size,
                "height": args.image_size,
                "sampling": "sequential",
                "horizontal_flip": False,
            },
        }
    elif args.dataset.strip().lower() in PREPARABLE_BENCHMARK_DATASETS:
        ds = args.dataset.strip().lower()
        dataset_config = {
            "factory": "image_folder",
            "args": {
                "root": f"~/.hypergan/cache/datasets/{ds}",
                "width": args.image_size,
                "height": args.image_size,
                "sampling": "sequential",
                "horizontal_flip": False,
            },
            "_prepare_dataset": ds,
        }
    else:
        dataset_config = {
            "builtin": args.dataset,
            "args": {
                "root": f"~/.hypergan/cache/{args.dataset}",
                "split": args.dataset_split,
                "sampling": "sequential",
                "horizontal_flip": False,
            },
        }

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (state_dir / f"eval-receipts-{run_id_stamp()}").resolve()
    )

    summary = execute_eval_bundles(
        repo_root=repo_root,
        state_dir=state_dir,
        bundles=bundles,
        metrics=metrics,
        dataset_config=dataset_config,
        gpu=args.gpu,
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        dataset_prep_cmd=args.dataset_prep_cmd,
        dataset_tarball=str(Path(args.dataset_tarball).resolve()) if args.dataset_tarball else None,
        dataset_target_dir=args.dataset_target_dir,
        output_dir=output_dir,
        import_to_run=args.import_to_run,
        boot_timeout=args.boot_timeout,
        boot_retries=args.boot_retries,
        fallback_gpus=args.fallback_gpus,
        max_minutes=args.max_minutes,
        idle_grace_minutes=args.idle_grace_minutes,
        stall_minutes=args.stall_minutes,
        repo_url=args.repo_url,
        ref=args.ref,
        source_mode=args.source_mode,
        colab_bin=colab_bin,
        particlegan_spec=args.particlegan_spec,
        hndl_spec=args.hndl_spec,
        torch_fidelity_spec=args.torch_fidelity_spec,
        foldlitedb_spec=args.foldlitedb_spec,
        safetensors_spec=args.safetensors_spec,
    )
    print(json.dumps(summary, indent=2))
    receipts = summary.get("receipts") or []
    ok = summary.get("run_rc") == 0 and all(r.get("status") == "complete" for r in receipts)
    return 0 if ok else 1


def _load_tracked_sessions(state_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sess_dir = state_dir / "sessions"
    if sess_dir.is_dir():
        for path in sorted(sess_dir.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(doc, dict):
                    doc["_record_path"] = str(path)
                    records.append(doc)
            except (OSError, json.JSONDecodeError):
                continue
    return records


def cmd_status(args: argparse.Namespace) -> int:
    repo_root = args.root.resolve()
    state_dir = (args.state_dir if args.state_dir is not None else repo_root / "runs" / "colab").resolve()
    colab_bin = resolve_colab_bin(args.colab_bin)

    tracked = _load_tracked_sessions(state_dir)
    rc_sess, out_sess = run_with_hard_timeout([colab_bin, "sessions"], timeout_seconds=30, capture=True)
    rc_usage, out_usage = run_with_hard_timeout([colab_bin, "usage"], timeout_seconds=30, capture=True)

    active_server = parse_colab_sessions_output(out_sess)
    active_tracked = [r for r in tracked if r.get("status") == "active"]

    payload = {
        "state_dir": str(state_dir),
        "server_sessions": active_server,
        "tracked_active": [
            {k: v for k, v in r.items() if k != "_record_path"}
            for r in active_tracked
        ],
        "tracked_total": len(tracked),
        "colab_sessions_raw": out_sess.strip(),
        "colab_usage_raw": out_usage.strip(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(out_sess, end="" if out_sess.endswith("\n") else "\n")
        print(out_usage, end="" if out_usage.endswith("\n") else "\n")
        print(f"[colab_bench] Tracked sessions under {state_dir}: {len(active_tracked)} active / {len(tracked)} total")
        for r in active_tracked:
            print(f"  - {r.get('session')} (gpu={r.get('gpu')}, started={r.get('started_at')}, cfg={r.get('config_path')})")
    return 0 if (rc_sess == 0 and rc_usage == 0) else 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    repo_root = args.root.resolve()
    state_dir = (args.state_dir if args.state_dir is not None else repo_root / "runs" / "colab").resolve()
    colab_bin = resolve_colab_bin(args.colab_bin)

    tracked = _load_tracked_sessions(state_dir)
    targets: dict[str, Path | None] = {}

    # 1. Active sessions tracked under runs/colab/sessions/*.json
    for rec in tracked:
        name = rec.get("session")
        if not name:
            continue
        if rec.get("status") == "active" or (args.session and name in args.session):
            cfg_p = Path(rec["config_path"]) if rec.get("config_path") else None
            targets[str(name)] = cfg_p if (cfg_p and cfg_p.is_file()) else None

    # 2. Live sessions reported directly by Colab server
    rc_sess, out_sess = run_with_hard_timeout([colab_bin, "sessions"], timeout_seconds=30, capture=True)
    if rc_sess == 0:
        for name in parse_colab_sessions_output(out_sess):
            targets.setdefault(name, None)

    if args.session:
        wanted = set(args.session)
        targets = {k: v for k, v in targets.items() if k in wanted}
        for name in wanted:
            targets.setdefault(name, None)

    stopped: list[dict[str, Any]] = []
    for name, cfg_p in sorted(targets.items()):
        cmd = [colab_bin]
        if cfg_p is not None:
            cmd += ["--config", str(cfg_p)]
        cmd += ["stop", "-s", name]
        if args.dry_run:
            stopped.append({"session": name, "config": str(cfg_p) if cfg_p else None, "action": "dry_run"})
            continue
        rc, out = run_with_hard_timeout(cmd, timeout_seconds=30, capture=True)
        # Fallback to default config if isolated config didn't know the session.
        if rc != 0 and cfg_p is not None:
            rc, out = run_with_hard_timeout([colab_bin, "stop", "-s", name], timeout_seconds=30, capture=True)
        stopped.append({"session": name, "config": str(cfg_p) if cfg_p else None, "rc": rc, "output": out.strip()})

    if not args.dry_run:
        for rec in tracked:
            name = rec.get("session")
            if name in targets and rec.get("_record_path"):
                rec_path = Path(rec["_record_path"])
                clean_rec = {k: v for k, v in rec.items() if k != "_record_path"}
                clean_rec["status"] = "stopped"
                clean_rec["stopped_at"] = utc_stamp()
                _atomic_write_json(rec_path, clean_rec)

    rc_after, out_after = run_with_hard_timeout([colab_bin, "sessions"], timeout_seconds=30, capture=True)
    remaining = parse_colab_sessions_output(out_after) if rc_after == 0 else []
    result = {
        "state_dir": str(state_dir),
        "dry_run": args.dry_run,
        "stopped": stopped,
        "remaining_sessions": remaining,
    }
    print(json.dumps(result, indent=2))
    return 0 if (args.dry_run or not remaining) else 1


# ---------------------------------------------------------------------------
# CLI parser & entrypoint
# ---------------------------------------------------------------------------


class _TrackExplicitAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crash-proof Google Colab GPU benchmark & evaluation orchestrator for HyperGAN."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="HyperGAN repository root")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Colab orchestrator state/logs directory (default: <root>/runs/colab)",
    )
    parser.add_argument("--colab-bin", default=None, help="Path to google-colab-cli binary (default: auto-detect)")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_vm_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--gpu",
            default="L4",
            action=_TrackExplicitAction,
            help="Colab accelerator: cpu, T4, L4 (default), G4, A100, H100",
        )
        sp.set_defaults(gpu_explicit=False)
        sp.add_argument("--boot-timeout", type=int, default=260, help="Seconds to wait for `colab new`")
        sp.add_argument("--boot-retries", type=int, default=2, help="Outer `colab new` retries on transient 503 bursts")
        sp.add_argument(
            "--fallback-gpus",
            "--fallback-gpu",
            dest="fallback_gpus",
            default=None,
            help="Optional comma-separated fallback accelerator list on repeated 503 allocation failures (e.g. A100,L4,T4 or L4,T4)",
        )
        sp.add_argument("--max-minutes", type=int, default=240, help="Hard VM-side watchdog wall-clock cap (minutes)")
        sp.add_argument(
            "--idle-grace-minutes",
            type=int,
            default=10,
            help="VM watchdog self-unassign grace after job completion if local controller disconnects",
        )
        sp.add_argument(
            "--stall-minutes",
            type=int,
            default=30,
            help="VM watchdog self-unassign timeout if logs/events/heartbeat stall",
        )
        sp.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git remote URL for HyperGAN")
        sp.add_argument("--ref", default="HEAD", help="Git ref/branch/SHA to bootstrap on Colab (default: HEAD)")
        sp.add_argument(
            "--source-mode",
            choices=("auto", "git", "bundle", "tarball"),
            default="auto",
            help="How to ship HyperGAN source to Colab (default: auto)",
        )
        sp.add_argument("--particlegan-spec", default=DEFAULT_PARTICLEGAN_SPEC)
        sp.add_argument("--hndl-spec", default=DEFAULT_HNDL_SPEC)
        sp.add_argument("--torch-fidelity-spec", default=DEFAULT_TORCH_FIDELITY_SPEC)
        sp.add_argument("--foldlitedb-spec", default=DEFAULT_FOLDLITEDB_SPEC)
        sp.add_argument("--safetensors-spec", default=DEFAULT_SAFETENSORS_SPEC)

    rn = sub.add_parser("run", help="Run one benchmark config or a JSON/TOML job manifest across warm Colab GPU VMs")
    rn.add_argument("--config", default=None, help="Single HyperGAN TOML model file or packaged model name")
    rn.add_argument("--manifest", type=Path, default=None, help="JSON or TOML job manifest file")
    rn.add_argument("--only", action="append", default=None, help="Restrict manifest to specific run_name(s); repeatable")
    rn.add_argument("--run-name", default=None, help="Run directory name under --runs-dir (for single --config)")
    rn.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Local destination runs directory for Observatory integration (default: <root>/runs)",
    )
    rn.add_argument(
        "--workers",
        type=int,
        default=HARD_MAX_WORKERS,
        help=f"Max concurrent warm Colab VMs (default {HARD_MAX_WORKERS}, clamped to pending job count)",
    )
    rn.add_argument(
        "--max-workers",
        type=int,
        default=HARD_MAX_WORKERS,
        help=f"Hard concurrent VM ceiling (default {HARD_MAX_WORKERS}; refuses > {HARD_MAX_WORKERS} unless --allow-over-cap)",
    )
    rn.add_argument(
        "--allow-over-cap",
        action="store_true",
        help=f"Explicitly override the hard {HARD_MAX_WORKERS}-GPU safety cap",
    )
    rn.add_argument(
        "--stagger-seconds",
        type=float,
        default=DEFAULT_STAGGER_SECONDS,
        help=f"Delay between parallel VM allocations (default {DEFAULT_STAGGER_SECONDS}s)",
    )
    rn.add_argument("--steps", type=int, default=None, help="Override total training steps (`--steps`)")
    rn.add_argument("--stop-after-steps", type=int, default=None, help="Target step ceiling for benchmark runs")
    rn.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint cadence in steps")
    rn.add_argument("--preview-every", type=int, default=None, help="Preview image cadence in steps")
    rn.add_argument("--no-previews", action="store_true", help="Disable periodic preview rendering")
    rn.add_argument("--progress-every", type=int, default=None, help="Progress log cadence in steps")
    rn.add_argument(
        "-s",
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forward HyperGAN `-s KEY=VALUE` config overrides (repeatable)",
    )
    rn.add_argument("--dataset", default=None, help="Dataset identifier (e.g. cifar10, none)")
    rn.add_argument("--dataset-prep-cmd", default=None, help="Shell command run once per VM to prepare dataset")
    rn.add_argument("--dataset-tarball", default=None, help="Local dataset .tar.gz uploaded once per VM")
    rn.add_argument("--dataset-target-dir", default=None, help="Remote extraction directory for --dataset-tarball")
    rn.add_argument(
        "--chunk-steps",
        type=int,
        default=0,
        help="Train in chunks of N steps with guaranteed local checkpoint pull-back between chunks (0=single attempt)",
    )
    rn.add_argument(
        "--sync-interval-minutes",
        type=float,
        default=0.0,
        help="Background periodic checkpoint pull-back interval in minutes during continuous runs (0=disabled)",
    )
    rn.add_argument("--max-retries", type=int, default=2, help="Max automatic resume retries on Colab disconnect")
    rn.add_argument("--drive-dir", default=None, help="Optional Google Drive path (e.g. /content/drive/MyDrive/hypergan-runs)")
    rn.add_argument(
        "--pull-checkpoints",
        choices=("latest", "best", "all", "none"),
        default="latest",
        help="Which checkpoints to pull back locally (default: latest)",
    )
    rn.add_argument("--no-export-bundle", action="store_true", help="Skip staging top-level EMA model.pt before pull-back")
    rn.add_argument("--force", action="store_true", help="Re-run jobs even if local summary.json is already complete")
    rn.add_argument("--no-resume", action="store_true", help="Start fresh instead of auto-resuming from local checkpoints")
    rn.add_argument("--no-register", action="store_true", help="Do not register pulled runs into local Observatory registry")
    rn.add_argument("--estimated-minutes", type=float, default=60.0, help="Estimated runtime per job for timeout sizing")
    rn.add_argument("--timeout-minutes", type=float, default=None, help="Explicit per-job SSH hard timeout in minutes")
    rn.add_argument("--dry-run", "--plan-only", dest="dry_run", action="store_true", help="Print plan without booting VMs")
    _add_vm_args(rn)

    ev = sub.add_parser("eval-bundles", help="Evaluate local EMA model.pt bundles on a single Colab GPU VM")
    ev.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Local EMA model.pt path or HyperGAN run directory (repeatable)",
    )
    ev.add_argument(
        "--metrics",
        default="InceptionSuite,DINOv2Suite",
        help="Comma-separated metrics: InceptionSuite, DINOv2Suite, InceptionFID, DINOv2FD (via python -m hypergan.image_metrics), or custom module:Class",
    )
    ev.add_argument("--dataset", default="cifar10", help="Built-in dataset name (default: cifar10)")
    ev.add_argument("--dataset-split", default="train", help="Built-in dataset split (default: train)")
    ev.add_argument("--dataset-dir", default=None, help="Remote image_folder path for non-builtin datasets")
    ev.add_argument("--dataset-json", default=None, help="Explicit JSON file defining evaluation data config")
    ev.add_argument("--image-size", type=int, default=64, help="Image resolution for --dataset-dir (default: 64)")
    ev.add_argument("--dataset-prep-cmd", default=None, help="Shell command run once on VM to prepare evaluation dataset")
    ev.add_argument("--dataset-tarball", default=None, help="Local dataset .tar.gz uploaded once to VM")
    ev.add_argument("--dataset-target-dir", default=None, help="Remote extraction directory for --dataset-tarball")
    ev.add_argument("--samples", type=int, default=50000, help="Evaluation sample count (default: 50000)")
    ev.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size (default: 128)")
    ev.add_argument("--seed", type=int, default=34002, help="Deterministic evaluation seed")
    ev.add_argument("--output-dir", type=Path, default=None, help="Local directory for downloaded JSON receipts")
    ev.add_argument(
        "--import-to-run",
        action="store_true",
        help="Import evaluation receipts cleanly into each local run directory via hypergan.evaluation_results",
    )
    _add_vm_args(ev)

    st = sub.add_parser("status", help="Inspect active Colab VMs, tracked runs/colab/ sessions, and CU balance")
    st.add_argument("--json", action="store_true", help="Print status as JSON")

    cl = sub.add_parser("cleanup", help="Tear down any active or orphaned Colab VMs tracked under runs/colab/ or server")
    cl.add_argument("--session", action="append", default=None, help="Specific session name(s) to stop (default: all active)")
    cl.add_argument("--dry-run", action="store_true", help="List sessions that would be stopped without stopping them")

    return parser


def main(argv: list[str] | None = None) -> int:
    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"\n[colab_bench] Received signal {signum}; releasing all active Colab VMs...", file=sys.stderr, flush=True)
        _GLOBAL_REGISTRY.stop_all()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "eval-bundles":
        return cmd_eval_bundles(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "cleanup":
        return cmd_cleanup(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Run inspection, discovery, comparison, and garbage collection.

Operates strictly on run metadata (manifest.json, summary.json, events.jsonl).
Never imports torch, particlegan, or training execution dependencies.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from .summary import atomic_write_json, read_summary_json
from .workspace import find_workspace

_FID_RE = re.compile(r'(?:^|[^a-z0-9])fid(?:$|[^a-z]|50k)', re.IGNORECASE)


def _is_fid_metric(metric_name: Any) -> bool:
    if not metric_name:
        return False
    return bool(_FID_RE.search(str(metric_name)))



def format_bytes(num_bytes: int) -> str:
    """Format byte count into human-readable unit (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.1f} GB"


def format_duration(seconds: Optional[Union[int, float]]) -> str:
    """Format seconds into HH:MM:SS string."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def inspect_run_host(run_dir: Union[Path, str]) -> Dict[str, Any]:
    """Extract recorded hostname, PID, and whether the run belongs to a remote/cluster host."""
    import socket

    run_dir = Path(run_dir).resolve()
    local_hostname = socket.gethostname()
    hostname: Optional[str] = None
    pid: Optional[int] = None

    lock_file = run_dir / ".run.lock"
    if lock_file.is_file():
        try:
            raw_lock = lock_file.read_bytes()[:4096].decode("utf-8", errors="ignore").strip()
            if raw_lock.startswith("{"):
                lock_data = json.loads(raw_lock)
                if isinstance(lock_data, dict):
                    h = lock_data.get("hostname") or lock_data.get("host") or lock_data.get("node")
                    if isinstance(h, str) and h.strip():
                        hostname = h.strip()
                    p = lock_data.get("pid")
                    if isinstance(p, int):
                        pid = p
        except Exception:
            pass

    launch_dir = run_dir / "launch"
    if launch_dir.is_dir():
        attempts = sorted(launch_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        for att_file in attempts:
            try:
                data = json.loads(att_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if hostname is None:
                        h = data.get("hostname") or data.get("host") or data.get("node")
                        if isinstance(h, str) and h.strip():
                            hostname = h.strip()
                    if pid is None:
                        p = data.get("pid") or data.get("supervisor_pid")
                        if isinstance(p, int):
                            pid = p
                    if hostname is not None or pid is not None:
                        break
            except Exception:
                pass

    manifest = _read_manifest_safe(run_dir)
    if hostname is None and manifest:
        runtime_dict = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
        cluster_dict = manifest.get("cluster") if isinstance(manifest.get("cluster"), dict) else {}
        h = (
            manifest.get("hostname")
            or manifest.get("host")
            or runtime_dict.get("hostname")
            or cluster_dict.get("hostname")
        )
        if isinstance(h, str) and h.strip():
            hostname = h.strip()
    if pid is None and isinstance(manifest.get("pid"), int):
        pid = manifest["pid"]

    remote = bool(hostname and hostname != local_hostname)
    return {
        "hostname": hostname,
        "pid": pid,
        "remote": remote,
        "local_hostname": local_hostname,
    }


def check_run_active(run_dir: Union[Path, str]) -> Tuple[bool, Optional[int]]:
    """Test non-blocking flock on .run.lock (or remote cluster run state) to check if training is active."""
    run_dir = Path(run_dir).resolve()
    lock_file = run_dir / ".run.lock"
    is_active = False
    if lock_file.exists():
        try:
            with lock_file.open("a+b") as f:
                f.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    is_active = False
                except (BlockingIOError, OSError):
                    is_active = True
        except Exception:
            pass

    if not is_active:
        host_info = inspect_run_host(run_dir)
        if host_info["remote"]:
            manifest = _read_manifest_safe(run_dir)
            m_status = str(manifest.get("status") or "").strip().lower() if manifest else ""
            summary = read_summary_json(run_dir) if not m_status else None
            s_status = str(summary.get("status") or "").strip().lower() if isinstance(summary, dict) else ""
            terminal = {"complete", "completed", "stopped", "failed", "interrupted", "cancelled", "canceled"}
            if m_status in ("running", "initializing") or (not m_status and s_status in ("running", "initializing")):
                if m_status not in terminal and s_status not in terminal:
                    return True, host_info["pid"]
        return False, None

    pid = None
    launch_dir = run_dir / "launch"
    if launch_dir.is_dir():
        attempts = sorted(launch_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        for att_file in attempts:
            try:
                data = json.loads(att_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    p = data.get("pid") or data.get("supervisor_pid")
                    if isinstance(p, int):
                        pid = p
                        break
            except Exception:
                pass

    if pid is None:
        summary = read_summary_json(run_dir)
        if summary and isinstance(summary.get("pid"), int):
            pid = summary["pid"]
        elif (run_dir / "manifest.json").is_file():
            try:
                m_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                if isinstance(m_data, dict) and isinstance(m_data.get("pid"), int):
                    pid = m_data["pid"]
            except Exception:
                pass

    if is_active and pid is None:
        from .registry import find_live_pid_for_run
        pid = find_live_pid_for_run(run_dir)

    return is_active, pid


def _resolve_run_dir(
    run_target: Optional[Union[Path, str]] = None,
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> Path:
    from .registry import resolve_run_target
    return resolve_run_target(run_target, cwd=cwd)


def _read_manifest_safe(run_dir: Path) -> Dict[str, Any]:
    """Read manifest.json safely without crashing on missing or corrupted JSON."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


def _read_events_tail_fallback(run_dir: Path) -> Dict[str, Any]:
    """Extract fallback run metadata from the tail of events.jsonl when summary/manifest are missing or corrupt."""
    events_path = run_dir / "events.jsonl"
    info: Dict[str, Any] = {
        "run_id": None,
        "step": None,
        "steps_per_sec": None,
        "latest_metrics": {},
        "best_fid": None,
        "best_step": None,
        "status": None,
    }
    if not events_path.is_file():
        return info

    try:
        with events_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        for raw_line in lines:
            try:
                ev = json.loads(raw_line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("run_id") and not info["run_id"]:
                info["run_id"] = str(ev["run_id"])
            ev_step = ev.get("step")
            if isinstance(ev_step, int) and (info["step"] is None or ev_step > info["step"]):
                info["step"] = ev_step
            kind = ev.get("event")
            if kind == "train":
                if isinstance(ev.get("metrics"), dict):
                    info["latest_metrics"] = dict(ev["metrics"])
                sps = ev.get("steps_per_second") or ev.get("steps_per_sec")
                if isinstance(sps, (int, float)):
                    info["steps_per_sec"] = float(sps)
            elif kind in ("evaluation", "evaluation_complete"):
                m_id = str(
                    ev.get("metric_id")
                    or ev.get("metric")
                    or (ev.get("result", {}).get("metric") if isinstance(ev.get("result"), dict) else "")
                    or ""
                )
                if _is_fid_metric(m_id):
                    val = ev.get("value")
                    if val is None and isinstance(ev.get("result"), dict):
                        val = ev["result"].get("value")
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        if info["best_fid"] is None or float(val) < info["best_fid"]:
                            info["best_fid"] = float(val)
                            info["best_step"] = int(ev_step or 0)
            elif kind in ("complete", "completed", "stopped", "failed"):
                info["status"] = "completed" if kind == "complete" else str(kind)
    except Exception:
        pass
    return info


def _resolve_valid_status(
    summary: Optional[Dict[str, Any]],
    manifest: Dict[str, Any],
    *,
    is_active: bool,
    run_dir: Path,
) -> str:
    """Resolve run status across summary.json and manifest.json, marking 'incomplete' only if neither has valid status."""
    if is_active:
        return "running"
    s_status = summary.get("status") if isinstance(summary, dict) else None
    m_status = manifest.get("status") if isinstance(manifest, dict) else None

    raw = None
    for cand in (s_status, m_status):
        if isinstance(cand, str) and cand.strip().lower() not in ("", "unknown"):
            raw = cand.strip().lower()
            break

    if raw is None:
        return "incomplete"
    if raw in ("running", "initializing") and (run_dir / ".run.lock").exists():
        return "stopped"
    if raw == "complete":
        return "completed"
    return raw


def run_status(
    run_dir: Optional[Union[Path, str]] = None,
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Compile human-friendly status summary for a run (fast-path summary.json or manifest/events fallback)."""
    resolved_dir = _resolve_run_dir(run_dir, cwd=cwd)
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    is_active, pid = check_run_active(resolved_dir)
    summary = read_summary_json(resolved_dir)
    manifest = _read_manifest_safe(resolved_dir)
    events_fallback = _read_events_tail_fallback(resolved_dir)

    from .registry import _read_registry_unlocked, extract_or_derive_slug, get_registry_dir

    reg_entry: Optional[Dict[str, Any]] = None
    try:
        reg_data = _read_registry_unlocked(get_registry_dir(resolved_dir))
        reg_entry = reg_data.get("runs", {}).get(str(resolved_dir))
    except Exception:
        reg_entry = None

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = (
        (summary.get("run_id") if summary else None)
        or manifest.get("run_id")
        or events_fallback.get("run_id")
        or (reg_entry.get("run_id") if reg_entry else None)
        or resolved_dir.name
    )
    raw_name = (
        (summary.get("name") if summary else None)
        or manifest.get("name")
        or config.get("name")
        or (reg_entry.get("name") if reg_entry else None)
        or run_id
    )
    if reg_entry and reg_entry.get("_old_dir_name") and str(raw_name) == str(reg_entry["_old_dir_name"]):
        raw_name = reg_entry.get("name") or resolved_dir.name
    name = str(raw_name)

    slug = (
        (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or (reg_entry.get("slug") if reg_entry else None)
        or extract_or_derive_slug(str(run_id), name=name, run_dir_name=resolved_dir.name)
    )

    status_str = _resolve_valid_status(summary, manifest, is_active=is_active, run_dir=resolved_dir)

    step = summary.get("step") if summary else None
    if step is None:
        step = manifest.get("steps")
    if step is None:
        step = events_fallback.get("step")
    if step is None:
        step = 0

    total_steps = (
        (summary.get("total_steps") if summary else None)
        or (summary.get("max_steps") if summary else None)
        or config.get("training", {}).get("steps")
        or config.get("train", {}).get("steps")
        or manifest.get("total_steps")
    )
    percent = round((step / total_steps * 100), 1) if (total_steps and total_steps > 0) else None

    sps = (
        (summary.get("steps_per_sec") or summary.get("steps_per_second") or summary.get("speed_its") or summary.get("speed"))
        if summary
        else None
    )
    if sps is None:
        sps = manifest.get("steps_per_second") or events_fallback.get("steps_per_sec")

    eta_seconds = summary.get("eta_seconds") if summary else None
    if eta_seconds is None and sps and sps > 0 and total_steps and total_steps > step:
        eta_seconds = round((total_steps - step) / sps, 1)

    elapsed_seconds = manifest.get("training_seconds") or manifest.get("seconds")

    devices_str = (
        (summary.get("device") if summary else None)
        or config.get("training", {}).get("device")
        or config.get("train", {}).get("device")
        or manifest.get("device")
        or "unknown"
    )
    launch_dir = resolved_dir / "launch"
    if launch_dir.is_dir():
        attempts = sorted(launch_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        if attempts:
            try:
                att_data = json.loads(attempts[0].read_text(encoding="utf-8"))
                if isinstance(att_data, dict):
                    cuda_devs = (att_data.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
                    if cuda_devs:
                        devices_str = f"cuda:{cuda_devs}"
                    if is_active and pid is None and isinstance(att_data.get("pid"), int):
                        pid = att_data["pid"]
            except Exception:
                pass

    if not is_active:
        pid = None

    latest_metrics = dict(summary.get("latest", {}) if (summary and isinstance(summary.get("latest"), dict)) else {})
    if not latest_metrics:
        latest_metrics = dict(events_fallback.get("latest_metrics", {}))

    best = summary.get("best") if (summary and isinstance(summary.get("best"), dict)) else None
    best_fid = summary.get("best_fid") if summary else None
    best_step = summary.get("best_step") if summary else None
    best_checkpoint = None

    if best:
        b_metric = str(best.get("metric") or "fid50k_train")
        if _is_fid_metric(b_metric):
            if best_fid is None:
                best_fid = best.get("value")
        else:
            best_fid = None
        if best_step is None:
            best_step = best.get("step")
        best_checkpoint = best.get("checkpoint")
    else:
        # Fallback scan evaluations
        eval_root = resolved_dir / "metrics" / "evaluations"
        best_val = None
        best_s = None
        if eval_root.is_dir():
            for rec in eval_root.glob("*/receipt.json"):
                try:
                    data = json.loads(rec.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("status") == "complete" and isinstance(data.get("result"), dict):
                        m_id = str(data.get("metric_id") or data.get("metric") or data["result"].get("metric") or "")
                        if not _is_fid_metric(m_id):
                            continue
                        val = data["result"].get("value")
                        s = data.get("source_step") or data["result"].get("step")
                        if val is not None and (best_val is None or val < best_val):
                            best_val = val
                            best_s = s
                except Exception:
                    continue
        if best_val is None and events_fallback.get("best_fid") is not None:
            best_val = events_fallback["best_fid"]
            best_s = events_fallback.get("best_step")
        if best_val is not None:
            best_fid = best_val
            best_step = best_s
            best = {"metric": "fid50k_train", "value": best_val, "step": best_s, "checkpoint": None}

    # Collect evaluation values by step to enrich checkpoint rows
    eval_by_step: Dict[int, Dict[str, float]] = {}
    if summary and isinstance(summary.get("evaluations"), list):
        for ev in summary["evaluations"]:
            if isinstance(ev, dict) and ev.get("step") is not None and isinstance(ev.get("value"), (int, float)):
                m_name = str(ev.get("metric") or "fid50k_train")
                eval_by_step.setdefault(int(ev["step"]), {})[m_name] = float(ev["value"])
    eval_root = resolved_dir / "metrics" / "evaluations"
    if eval_root.is_dir():
        for rec in eval_root.glob("*/receipt.json"):
            try:
                rdata = json.loads(rec.read_text(encoding="utf-8"))
                if isinstance(rdata, dict) and rdata.get("status") == "complete" and isinstance(rdata.get("result"), dict):
                    val = rdata["result"].get("value")
                    s = rdata.get("source_step") or rdata["result"].get("step")
                    m_name = str(rdata.get("metric_id") or rdata.get("metric") or "fid50k_train")
                    if s is not None and isinstance(val, (int, float)):
                        eval_by_step.setdefault(int(s), {})[m_name] = float(val)
            except Exception:
                continue
    if best_step is not None:
        if best_fid is not None and isinstance(best_fid, (int, float)):
            eval_by_step.setdefault(int(best_step), {}).setdefault("fid50k_train", float(best_fid))
        if isinstance(best, dict) and isinstance(best.get("value"), (int, float)):
            b_m = str(best.get("metric") or "fid50k_train")
            eval_by_step.setdefault(int(best_step), {}).setdefault(b_m, float(best["value"]))

    # Recent checkpoints
    raw_checkpoints = summary.get("checkpoints") if (summary and isinstance(summary.get("checkpoints"), list)) else None
    if raw_checkpoints is None:
        ckpt_root = resolved_dir / "checkpoints"
        ckpts = []
        if ckpt_root.is_dir():
            for cm in ckpt_root.glob("*/manifest.json"):
                try:
                    cdata = json.loads(cm.read_text(encoding="utf-8"))
                    if isinstance(cdata, dict):
                        ckpts.append({
                            "step": cdata.get("step", 0),
                            "name": cm.parent.name,
                            "path": f"checkpoints/{cm.parent.name}",
                            "metrics": {},
                        })
                except Exception:
                    continue
        ckpts.sort(key=lambda c: c["step"])
        raw_checkpoints = ckpts

    checkpoints = []
    for c in (raw_checkpoints or []):
        if not isinstance(c, dict):
            continue
        c_copy = dict(c)
        c_metrics = dict(c_copy.get("metrics") if isinstance(c_copy.get("metrics"), dict) else {})
        c_step_int = int(c_copy.get("step", 0) or 0)
        if c_step_int in eval_by_step:
            for m_k, m_v in eval_by_step[c_step_int].items():
                c_metrics.setdefault(m_k, m_v)
                if _is_fid_metric(m_k):
                    c_metrics.setdefault("fid50k_train", m_v)
        c_copy["metrics"] = c_metrics
        checkpoints.append(c_copy)

    recent_ckpts = list(reversed(checkpoints[-5:])) if checkpoints else []

    warnings = list(manifest.get("warnings", []) if isinstance(manifest.get("warnings"), list) else []) + list(
        manifest.get("resume_warnings", []) if isinstance(manifest.get("resume_warnings"), list) else []
    )
    if not summary and not manifest and (resolved_dir / "events.jsonl").is_file():
        corrupt_warn = (
            "Metadata files (manifest.json / summary.json) are missing or corrupted; reconstructed from events.jsonl"
        )
        if corrupt_warn not in warnings:
            warnings.append(corrupt_warn)

    host_info = inspect_run_host(resolved_dir)

    return {
        "run_id": run_id,
        "short_id": str(run_id)[:8],
        "slug": slug,
        "name": name,
        "run_dir": str(resolved_dir),
        "status": status_str,
        "active": is_active,
        "pid": pid if is_active else None,
        "hostname": host_info["hostname"],
        "remote": host_info["remote"],
        "devices": devices_str,
        "step": step,
        "total_steps": total_steps,
        "percent": percent,
        "steps_per_sec": sps,
        "elapsed": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "latest_metrics": latest_metrics,
        "best": best,
        "best_fid": best_fid,
        "best_step": best_step,
        "best_checkpoint": best_checkpoint,
        "recent_checkpoints": recent_ckpts,
        "warnings": warnings,
    }


status = run_status


def format_run_status(info: Dict[str, Any]) -> str:
    """Format status dict into single-screen terminal summary."""
    sps = info.get("steps_per_sec")
    speed_str = f"{sps:.1f} it/s" if (sps is not None and isinstance(sps, (int, float))) else "-- it/s"
    short_id = info.get("short_id") or str(info["run_id"])[:8]
    slug = info.get("slug") or short_id
    lines = [
        "=" * 80,
        f"HyperGAN Run Status: {info['run_id']}",
        "=" * 80,
        f"Experiment:  {info['name']}",
        f"Slug / ID:   {slug} ({short_id})",
        f"Run Dir:     {info['run_dir']}",
        f"Status:      {str(info['status']).upper()}" + (" (active)" if info["active"] else " (stopped/idle)"),
        f"Devices:     {info['devices']}",
        f"Process:     PID {info['pid'] if info['pid'] is not None else '--'} " + ("(active)" if info["active"] else "(inactive)"),
        "",
        "Progress:",
        f"  Step:      {info['step']:,}" + (f" / {info['total_steps']:,} ({info['percent']}%)" if info['total_steps'] else ""),
        f"  Speed:     {speed_str}",
        f"  Elapsed:   {format_duration(info.get('elapsed'))}",
        f"  ETA:       {format_duration(info.get('eta_seconds'))}",
        "",
        "Metrics (Latest & Best):",
    ]

    latest = info.get("latest_metrics", {})
    metric_pairs = []
    for k in ("loss/g_total", "loss_g", "loss/d_total", "loss_d", "d_guard_ratio", "lr"):
        if k in latest and isinstance(latest[k], (int, float)):
            metric_pairs.append(f"{k}: {latest[k]:.4f}")
    if metric_pairs:
        lines.append("  " + "    ".join(metric_pairs[:4]))
    else:
        lines.append("  (No intermediate scalar metrics logged)")

    best_obj = info.get("best")
    if isinstance(best_obj, dict) and isinstance(best_obj.get("value"), (int, float)):
        b_metric = str(best_obj.get("metric") or "fid50k_train")
        b_val = best_obj["value"]
        b_step = best_obj.get("step") if best_obj.get("step") is not None else info.get("best_step")
        if _is_fid_metric(b_metric):
            best_str = f"Best FID:  {b_val:.2f}"
        else:
            best_str = f"Best {b_metric}:  {b_val:.4g}"
        if b_step is not None:
            best_str += f" at step {b_step:,}"
        lines.append(f"  {best_str}")
    elif info.get("best_fid") is not None:
        best_str = f"Best FID:  {info['best_fid']:.2f}"
        if info.get("best_step") is not None:
            best_str += f" at step {info['best_step']:,}"
        lines.append(f"  {best_str}")

    lines.append("")
    lines.append("Recent Checkpoints:")
    lines.append("  Step     Path                                                  FID")
    lines.append("  -------  ----------------------------------------------------  -------")
    recent = info.get("recent_checkpoints", [])
    if recent:
        for c in recent:
            c_step = f"{c.get('step', 0):,}"
            c_path = c.get("path", "")
            if len(c_path) > 52:
                c_path = c_path[:24] + "..." + c_path[-25:]
            c_metrics = c.get("metrics") if isinstance(c.get("metrics"), dict) else {}
            fid_val = c_metrics.get("fid50k_train")
            if fid_val is None:
                fid_val = c_metrics.get("fid")
            if fid_val is None and c.get("fid") is not None:
                fid_val = c.get("fid")
            if fid_val is None and info.get("best_step") is not None and c.get("step") == info.get("best_step") and info.get("best_fid") is not None:
                fid_val = info.get("best_fid")
            fid_str = f"{fid_val:.2f}" if (fid_val is not None and isinstance(fid_val, (int, float))) else "--"
            lines.append(f"  {c_step:>7}  {c_path:<52}  {fid_str:>7}")
    else:
        lines.append("  (No checkpoints saved yet)")

    lines.append("")
    lines.append("Warnings:")
    warns = info.get("warnings", [])
    if warns:
        for w in warns:
            lines.append(f"  - {w}")
    else:
        lines.append("  None")
    lines.append("=" * 80)
    return "\n".join(lines)


format_status = format_run_status


def _collect_ps_row(
    run_dir: Union[Path, str],
    cached_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect a single run row for list_runs / ps_runs using Tier 1 mtime cache or safe metadata fallback."""
    from .registry import _read_registry_unlocked, extract_or_derive_slug, get_registry_dir

    run_path = Path(run_dir).resolve()
    is_active, pid = check_run_active(run_path)

    if cached_entry is None:
        try:
            reg_data = _read_registry_unlocked(get_registry_dir(run_path))
            cached_entry = reg_data.get("runs", {}).get(str(run_path))
        except Exception:
            cached_entry = None

    summary_path = run_path / "summary.json"
    summary_mtime_ns: Optional[int] = None
    try:
        if summary_path.is_file():
            summary_mtime_ns = summary_path.stat().st_mtime_ns
    except OSError:
        summary_mtime_ns = None

    # Tier 1 O(1) cache reuse if inactive and summary_mtime_ns matches
    if (
        cached_entry is not None
        and not is_active
        and cached_entry.get("status") not in ("running", "initializing")
        and summary_mtime_ns is not None
        and cached_entry.get("summary_mtime_ns") == summary_mtime_ns
        and cached_entry.get("run_id")
    ):
        step_val = int(cached_entry.get("step") or 0)
        max_steps_val = (
            cached_entry.get("max_steps")
            if cached_entry.get("max_steps") is not None
            else cached_entry.get("total_steps")
        )
        speed_val = (
            cached_entry.get("speed")
            if cached_entry.get("speed") is not None
            else cached_entry.get("steps_per_sec")
        )
        rid = str(cached_entry["run_id"])
        return {
            "run_id": rid,
            "short_id": cached_entry.get("short_id") or rid[:8],
            "slug": cached_entry.get("slug") or extract_or_derive_slug(rid, name=str(cached_entry.get("name", "")), run_dir_name=run_path.name),
            "run_dir": str(run_path),
            "name": str(cached_entry.get("name") or run_path.name),
            "status": str(cached_entry.get("status") or "completed"),
            "active": False,
            "pid": None,
            "step": step_val,
            "max_steps": max_steps_val,
            "total_steps": max_steps_val,
            "speed": speed_val,
            "steps_per_sec": speed_val,
            "latest_metric": cached_entry.get("latest_metric", "--"),
            "best_fid": cached_entry.get("best_fid"),
            "device": str(cached_entry.get("device") or "cuda"),
            "updated_at": str(cached_entry.get("updated_at") or "--"),
            "summary_mtime_ns": summary_mtime_ns,
        }

    summary = read_summary_json(run_path)
    manifest = _read_manifest_safe(run_path)
    events_fallback = _read_events_tail_fallback(run_path) if (not summary or not manifest) else {}

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = (
        (summary.get("run_id") if summary else None)
        or manifest.get("run_id")
        or events_fallback.get("run_id")
        or (cached_entry.get("run_id") if cached_entry else None)
        or run_path.name
    )
    raw_name = (
        (summary.get("name") if summary else None)
        or manifest.get("name")
        or config.get("name")
        or (cached_entry.get("name") if cached_entry else None)
        or run_id
    )
    if cached_entry and cached_entry.get("_old_dir_name") and str(raw_name) == str(cached_entry["_old_dir_name"]):
        raw_name = cached_entry.get("name") or run_path.name
    name = str(raw_name)

    slug = (
        (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or (cached_entry.get("slug") if cached_entry else None)
        or extract_or_derive_slug(str(run_id), name=name, run_dir_name=run_path.name)
    )

    status_val = _resolve_valid_status(summary, manifest, is_active=is_active, run_dir=run_path)

    step = summary.get("step") if summary else None
    if step is None:
        step = manifest.get("steps")
    if step is None:
        if not events_fallback:
            events_fallback = _read_events_tail_fallback(run_path)
        step = events_fallback.get("step")
    if step is None:
        step = 0

    total_steps = (
        (summary.get("total_steps") if summary else None)
        or (summary.get("max_steps") if summary else None)
        or config.get("training", {}).get("steps")
        or config.get("train", {}).get("steps")
        or manifest.get("total_steps")
    )
    sps = (
        (summary.get("steps_per_sec") or summary.get("steps_per_second") or summary.get("speed_its") or summary.get("speed"))
        if summary
        else None
    )
    if sps is None:
        sps = manifest.get("steps_per_second") or events_fallback.get("steps_per_sec")

    best = summary.get("best") if (summary and isinstance(summary.get("best"), dict)) else None
    best_fid = summary.get("best_fid") if summary else None
    if best and isinstance(best, dict):
        b_metric = str(best.get("metric") or "")
        if _is_fid_metric(b_metric) and isinstance(best.get("value"), (int, float)):
            best_fid = best.get("value")
        elif not _is_fid_metric(b_metric):
            best_fid = None
    elif best_fid is not None and isinstance(best, dict) and not _is_fid_metric(best.get("metric", "")):
        best_fid = None

    if best_fid is None and (run_path / "metrics" / "evaluations").is_dir():
        for rec in (run_path / "metrics" / "evaluations").glob("*/receipt.json"):
            try:
                rdata = json.loads(rec.read_text(encoding="utf-8"))
                if isinstance(rdata, dict) and rdata.get("status") == "complete" and isinstance(rdata.get("result"), dict):
                    m_id = str(
                        rdata.get("metric_id")
                        or rdata.get("metric")
                        or (rdata.get("result", {}).get("metric") if isinstance(rdata.get("result"), dict) else "")
                        or ""
                    )
                    if _is_fid_metric(m_id):
                        val = rdata["result"].get("value")
                        if val is not None and (best_fid is None or val < best_fid):
                            best_fid = val
            except Exception:
                continue
    if best_fid is None and events_fallback.get("best_fid") is not None:
        best_fid = events_fallback["best_fid"]

    latest = (summary.get("latest", {}) if (summary and isinstance(summary.get("latest"), dict)) else {}) or (
        events_fallback.get("latest_metrics", {}) if events_fallback else {}
    )
    latest_metric_str = "--"
    for k in ("loss/g_total", "loss_g", "loss/total"):
        if k in latest and isinstance(latest[k], (int, float)):
            latest_metric_str = f"loss_g={latest[k]:.2f}"
            break

    device = (
        (summary.get("device") if summary else None)
        or config.get("training", {}).get("device")
        or config.get("train", {}).get("device")
        or manifest.get("device")
        or (cached_entry.get("device") if cached_entry else None)
        or "cuda"
    )

    updated_at = (summary.get("updated_at") if summary else None) or manifest.get("updated_at")
    if not updated_at:
        try:
            manifest_path = run_path / "manifest.json"
            target_file = summary_path if summary_path.is_file() else (manifest_path if manifest_path.is_file() else (run_path / "events.jsonl"))
            mtime = target_file.stat().st_mtime
            updated_at = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            updated_at = "--"

    return {
        "run_id": run_id,
        "short_id": str(run_id)[:8],
        "slug": slug,
        "run_dir": str(run_path),
        "name": name,
        "status": status_val,
        "active": is_active,
        "pid": pid if is_active else None,
        "step": int(step or 0),
        "max_steps": total_steps,
        "total_steps": total_steps,
        "speed": sps,
        "steps_per_sec": sps,
        "latest_metric": latest_metric_str,
        "best_fid": best_fid,
        "device": str(device),
        "updated_at": updated_at,
        "summary_mtime_ns": summary_mtime_ns,
    }


def list_runs(
    root: Optional[Union[Path, str]] = None,
    sort: str = "fid",
    status_filter: Optional[str] = None,
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Discover and list all runs under root (or globally via registry when root is None)."""
    from .registry import (
        _ensure_unique_slug,
        _is_isolated_test_env,
        find_workspace_runs_root,
        is_run_dir,
        list_registered_runs,
    )

    candidates: List[Path] = []
    cached_by_dir: Dict[str, Dict[str, Any]] = {}
    base_cwd = Path(cwd).resolve() if cwd is not None else (_is_isolated_test_env() or Path.cwd().resolve())

    if root is None:
        reg_entries = list_registered_runs(cwd=base_cwd)
        seen_dirs = set()
        for entry in reg_entries:
            p = Path(entry["run_dir"]).resolve()
            cached_by_dir[str(p)] = entry
            if p.is_dir() and str(p) not in seen_dirs:
                seen_dirs.add(str(p))
                candidates.append(p)
        ws_root = find_workspace_runs_root(base_cwd)
        has_ws = find_workspace(base_cwd) is not None
        check_roots: List[Path] = []
        if ws_root:
            check_roots.append(ws_root)
        if has_ws:
            if str(base_cwd.resolve()) != "/tmp":
                check_roots.append(base_cwd)
        else:
            if str(base_cwd.resolve()) == "/tmp":
                cwd_runs = base_cwd / "runs"
                if cwd_runs.is_dir():
                    check_roots.append(cwd_runs)
            else:
                cwd_runs = base_cwd / "runs"
                if cwd_runs.is_dir():
                    check_roots.append(cwd_runs)
                check_roots.append(base_cwd)

        for check_root in check_roots:
            if check_root and check_root.is_dir():
                if is_run_dir(check_root):
                    if str(check_root) not in seen_dirs:
                        seen_dirs.add(str(check_root))
                        candidates.append(check_root)
                else:
                    try:
                        for child in check_root.iterdir():
                            if child.name.startswith(".") or not child.is_dir():
                                continue
                            if is_run_dir(child):
                                rp = child.resolve()
                                if str(rp) not in seen_dirs:
                                    seen_dirs.add(str(rp))
                                    candidates.append(rp)
                    except OSError:
                        pass
    else:
        target_root = Path(root).resolve()
        if not target_root.exists():
            return []
        if is_run_dir(target_root):
            candidates = [target_root]
        else:
            try:
                for child in target_root.iterdir():
                    if child.name.startswith(".") or not child.is_dir():
                        continue
                    if is_run_dir(child):
                        candidates.append(child.resolve())
            except OSError:
                return []

    runs: List[Dict[str, Any]] = []
    for run_dir in candidates:
        try:
            row = _collect_ps_row(run_dir, cached_entry=cached_by_dir.get(str(run_dir.resolve())))
            row["slug"] = _ensure_unique_slug(str(row.get("slug") or ""), str(row.get("run_id") or ""), runs)
            runs.append(row)
        except Exception:
            continue

    # Filtering
    if status_filter:
        s_filter = status_filter.strip().lower()
        if s_filter == "complete":
            s_filter = "completed"
        runs = [r for r in runs if r["status"] == s_filter]

    # Sorting
    sort_key = sort.strip().lower()
    if sort_key in ("fid", "best"):
        # Best FID ascending, None values last
        runs.sort(key=lambda r: (0 if r["best_fid"] is not None else 1, r["best_fid"] if r["best_fid"] is not None else float("inf"), -r["step"]))
    elif sort_key == "step":
        runs.sort(key=lambda r: (-r["step"], r["name"]))
    elif sort_key == "name":
        runs.sort(key=lambda r: r["name"].lower())
    else:  # updated (active runs first, then updated_at descending)
        runs.sort(key=lambda r: (1 if r.get("active") else 0, str(r["updated_at"])), reverse=True)

    return runs


def format_runs_table(runs: List[Dict[str, Any]]) -> str:
    """Format runs list into dynamic terminal table."""
    if not runs:
        return "No runs found."

    headers = ["Run ID", "Slug", "Experiment", "Status", "Step", "it/s", "Latest Metric", "Best FID", "Updated"]
    rows = []
    for r in runs:
        run_id_short = r.get("short_id") or r["run_id"][:8]
        slug = r.get("slug") or "--"
        name = r["name"]
        if len(name) > 30:
            name = name[:13] + "..." + name[-14:]
        status = r["status"]
        if r.get("total_steps"):
            step_str = f"{r['step']:,} / {r['total_steps']:,}"
        else:
            step_str = f"{r['step']:,}"
        sps_str = f"{r['steps_per_sec']:.1f}" if r.get("steps_per_sec") else "--"
        latest = r.get("latest_metric", "--")
        fid_str = f"{r['best_fid']:.2f}" if r.get("best_fid") is not None else "--"
        updated = str(r.get("updated_at", "--"))
        if "T" in updated:
            updated = updated.split(".")[0].replace("T", " ")
        rows.append([run_id_short, slug, name, status, step_str, sps_str, latest, fid_str, updated])

    # Dynamic column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def make_line(items: List[str]) -> str:
        return "  ".join(items[i].ljust(widths[i]) if i not in (4, 5, 7) else items[i].rjust(widths[i]) for i in range(len(items)))

    table_lines = [
        make_line(headers),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        table_lines.append(make_line(row))

    return "\n".join(table_lines)


def parse_step_token(token: str) -> Union[int, str]:
    """Parse step notation (e.g. '50k' -> 50000, '1M' -> 1000000, 'best' -> 'best')."""
    token_str = str(token).strip().lower()
    if token_str in ("best", "latest"):
        return token_str
    if token_str.endswith("k"):
        return int(float(token_str[:-1]) * 1000)
    if token_str.endswith("m"):
        return int(float(token_str[:-1]) * 1000000)
    return int(token_str)


_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "loss_g": ("loss_g", "loss/g_total", "g_loss", "loss/g"),
    "loss/g_total": ("loss/g_total", "loss_g", "g_loss", "loss/g"),
    "g_loss": ("g_loss", "loss/g_total", "loss_g", "loss/g"),
    "loss_d": ("loss_d", "loss/d_total", "d_loss", "loss/d"),
    "loss/d_total": ("loss/d_total", "loss_d", "d_loss", "loss/d"),
    "d_loss": ("d_loss", "loss/d_total", "loss_d", "loss/d"),
    "fid": ("fid", "fid50k_train", "fid50k_full"),
    "fid50k_train": ("fid50k_train", "fid"),
}


def _candidate_metric_keys(metric: str) -> List[str]:
    m_clean = str(metric).strip()
    m_lower = m_clean.lower()
    keys: List[str] = [m_clean]
    if m_lower in _METRIC_ALIASES:
        for a in _METRIC_ALIASES[m_lower]:
            if a not in keys:
                keys.append(a)
    for alt in (m_clean.replace("/", "_"), m_clean.replace("_", "/")):
        if alt not in keys:
            keys.append(alt)
    return keys


def compare_runs(run_dirs: List[Union[Path, str]],
                 metric: str = "fid50k_train",
                 at_steps: Optional[Union[List[Union[int, str]], str]] = None) -> Dict[str, Any]:
    """Compare multiple runs for a given metric across step milestones."""
    if isinstance(at_steps, str):
        steps = [parse_step_token(t.strip()) for t in at_steps.split(",") if t.strip()]
    elif at_steps:
        steps = [parse_step_token(t) for t in at_steps]
    else:
        steps = [50000, 100000, "best", "latest"]

    cand_keys = _candidate_metric_keys(metric)
    cand_keys_lower = {k.lower() for k in cand_keys}

    runs_data = []
    for r_dir in run_dirs:
        resolved = _resolve_run_dir(r_dir)
        summary = read_summary_json(resolved)
        manifest_path = resolved / "manifest.json"
        manifest = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or resolved.name
        name = (summary.get("name") if summary else None) or manifest.get("config", {}).get("name") or run_id

        # Collect evaluation metrics
        evals_by_step: Dict[int, float] = {}
        if summary and summary.get("evaluations"):
            for ev in summary["evaluations"]:
                if isinstance(ev, dict) and str(ev.get("metric") or "").lower() in cand_keys_lower:
                    if ev.get("step") is not None and isinstance(ev.get("value"), (int, float)):
                        evals_by_step[int(ev["step"])] = float(ev["value"])
        if not evals_by_step and (resolved / "metrics" / "evaluations").is_dir():
            for receipt_file in (resolved / "metrics" / "evaluations").glob("*/receipt.json"):
                try:
                    rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
                    if rdata.get("status") == "complete" and "result" in rdata:
                        m_id = str(rdata.get("metric_id") or rdata.get("metric") or "")
                        if m_id.lower() in cand_keys_lower:
                            s = rdata.get("source_step") or rdata["result"].get("step")
                            val = rdata["result"].get("value")
                            if s is not None and val is not None:
                                evals_by_step[int(s)] = float(val)
                except Exception:
                    continue

        # If not an evaluation metric, check events.jsonl for training scalars
        if not evals_by_step and (resolved / "events.jsonl").is_file():
            try:
                with (resolved / "events.jsonl").open("rb") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            ev = json.loads(line)
                            if ev.get("event") == "train" and isinstance(ev.get("metrics"), dict):
                                s = ev.get("step")
                                m_dict = ev["metrics"]
                                val = None
                                for ck in cand_keys:
                                    if ck in m_dict and isinstance(m_dict[ck], (int, float)):
                                        val = m_dict[ck]
                                        break
                                if val is None:
                                    for mk, mv in m_dict.items():
                                        if str(mk).lower() in cand_keys_lower and isinstance(mv, (int, float)):
                                            val = mv
                                            break
                                if s is not None and val is not None:
                                    evals_by_step[int(s)] = float(val)
                        except Exception:
                            continue
            except Exception:
                pass

        # Fallback to summary.json latest metrics if events.jsonl had no match
        if not evals_by_step and summary and isinstance(summary.get("latest"), dict):
            latest_dict = summary["latest"]
            s_step = int(summary.get("step") or 0)
            for ck in cand_keys:
                if ck in latest_dict and isinstance(latest_dict[ck], (int, float)):
                    evals_by_step[s_step] = float(latest_dict[ck])
                    break

        runs_data.append({
            "run_id": run_id,
            "name": name,
            "dir": str(resolved),
            "evals": evals_by_step,
        })

    # Build matrix
    matrix: Dict[str, Dict[str, Any]] = {}
    for st in steps:
        st_key = str(st)
        matrix[st_key] = {}
        for r in runs_data:
            evals = r["evals"]
            if not evals:
                matrix[st_key][r["run_id"]] = None
                continue

            if st == "best":
                # Find min value (for FID/loss)
                best_step = min(evals.keys(), key=lambda s: (evals[s], -s))
                matrix[st_key][r["run_id"]] = {"value": evals[best_step], "step": best_step}
            elif st == "latest":
                latest_step = max(evals.keys())
                matrix[st_key][r["run_id"]] = {"value": evals[latest_step], "step": latest_step}
            else:
                target_step = int(st)
                if target_step in evals:
                    matrix[st_key][r["run_id"]] = {"value": evals[target_step], "step": target_step}
                else:
                    matrix[st_key][r["run_id"]] = None

    return {
        "metric": metric,
        "steps": steps,
        "runs": runs_data,
        "matrix": matrix,
    }


def format_comparison_table(comparison: Dict[str, Any]) -> str:
    """Format comparison data into side-by-side terminal table."""
    metric = comparison["metric"]
    runs = comparison["runs"]
    matrix = comparison["matrix"]
    steps = comparison["steps"]

    run_headers = [f"{r['name']} ({r['run_id'][:8]})" for r in runs]
    headers = ["Step", *run_headers]

    rows = []
    for st in steps:
        st_key = str(st)
        row_label = f"{st:,}" if isinstance(st, int) else str(st).capitalize()
        cells = [row_label]
        for r in runs:
            data = matrix.get(st_key, {}).get(r["run_id"])
            if data is None:
                cells.append("--")
            elif isinstance(st, (str)):
                cells.append(f"{data['value']:.2f} (step {data['step']:,})")
            else:
                cells.append(f"{data['value']:.2f}")
        rows.append(cells)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def make_line(items: List[str]) -> str:
        return "  ".join(items[i].ljust(widths[i]) if i == 0 else items[i].rjust(widths[i]) for i in range(len(items)))

    lines = [
        f"Metric: {metric}",
        "-" * (sum(widths) + 2 * (len(headers) - 1)),
        make_line(headers),
        make_line(["-" * w for w in widths]),
    ]
    for r in rows:
        lines.append(make_line(r))
    return "\n".join(lines)


def format_comparison_csv(comparison: Dict[str, Any]) -> str:
    """Format comparison data into CSV."""
    runs = comparison["runs"]
    matrix = comparison["matrix"]
    steps = comparison["steps"]

    run_headers = [f"{r['name']} ({r['run_id'][:8]})" for r in runs]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["step", *run_headers])

    for st in steps:
        st_key = str(st)
        row = [st]
        for r in runs:
            data = matrix.get(st_key, {}).get(r["run_id"])
            if data is None:
                row.append("")
            else:
                row.append(f"{data['value']:.4f}")
        writer.writerow(row)

    return output.getvalue()


def compact_events(run_dir: Union[Path, str], stride: int = 10) -> int:
    """Compact redundant train events in events.jsonl, returning reclaimed bytes."""
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    run_dir = Path(run_dir).resolve()
    is_active, _ = check_run_active(run_dir)
    if is_active:
        raise RuntimeError(f"Cannot compact events on active training run: {run_dir}")

    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return 0

    from .run_state import repair_event_tail
    repair_event_tail(events_path)

    before_size = events_path.stat().st_size
    if before_size == 0:
        return 0

    # Discover checkpoint and evaluation steps to preserve
    pinned_steps = {0}
    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.is_dir():
        for cm in ckpt_root.glob("*/manifest.json"):
            try:
                cdata = json.loads(cm.read_text(encoding="utf-8"))
                if "step" in cdata:
                    pinned_steps.add(int(cdata["step"]))
            except Exception:
                pass

    eval_root = run_dir / "metrics" / "evaluations"
    if eval_root.is_dir():
        for rec in eval_root.glob("*/receipt.json"):
            try:
                rdata = json.loads(rec.read_text(encoding="utf-8"))
                s = rdata.get("source_step") or rdata.get("result", {}).get("step")
                if s is not None:
                    pinned_steps.add(int(s))
            except Exception:
                pass

    # Read events to find final step
    last_train_step = None
    try:
        with events_path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("event") == "train" and "step" in ev:
                        last_train_step = int(ev["step"])
                except Exception:
                    continue
    except Exception:
        pass
    if last_train_step is not None:
        pinned_steps.add(last_train_step)

    # Read manifest for default run_id / attempt_id if needed
    manifest_path = run_dir / "manifest.json"
    manifest_data = {}
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_data = {}

    temp_path = run_dir / f".events.jsonl.compact.{os.getpid()}.tmp"
    digest = hashlib.sha256()
    new_size = 0
    last_event = None

    current_attempt_id = None
    current_sequence = 0

    try:
        with events_path.open("rb") as src, temp_path.open("wb") as dst:
            for line in src:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    if not isinstance(ev, dict):
                        continue
                except Exception:
                    # Incomplete trailing write or unparseable line: discard during compaction
                    continue

                if ev.get("event") == "train":
                    step = ev.get("step")
                    if step not in pinned_steps and (step % stride != 0):
                        continue

                # Renumber sequence contiguously per attempt_id
                ev_att = ev.get("attempt_id")
                if ev_att is not None and ev_att != current_attempt_id:
                    current_attempt_id = ev_att
                    current_sequence = 1
                elif current_attempt_id is None:
                    current_attempt_id = ev_att or manifest_data.get("attempt_id") or "0001"
                    current_sequence = 1
                else:
                    current_sequence += 1

                ev["sequence"] = current_sequence
                if ev.get("attempt_id") is None:
                    ev["attempt_id"] = current_attempt_id
                if ev.get("run_id") is None and manifest_data.get("run_id"):
                    ev["run_id"] = manifest_data.get("run_id")

                serialized = (json.dumps(ev, allow_nan=False) + "\n").encode("utf-8")
                dst.write(serialized)
                digest.update(serialized)
                new_size += len(serialized)
                last_event = ev

            dst.flush()
            os.fsync(dst.fileno())

        os.replace(temp_path, events_path)
    finally:
        temp_path.unlink(missing_ok=True)

    # Invalidate stale views to prevent cursor mismatch
    views_dir = run_dir / "views"
    if views_dir.is_dir():
        shutil.rmtree(views_dir, ignore_errors=True)

    # Update manifest with boundary and envelope SHA256
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["events_compacted"] = True
            manifest["envelope_sha256"] = digest.hexdigest()
            if last_event and isinstance(last_event, dict):
                manifest["durable_event_boundary"] = {
                    "schema_version": 1,
                    "offset": new_size,
                    "sha256": digest.hexdigest(),
                    "run_id": manifest.get("run_id") or last_event.get("run_id"),
                    "attempt_id": last_event.get("attempt_id"),
                    "step": last_event.get("step", 0),
                    "sequence": last_event.get("sequence", 1),
                }
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass

    return max(0, before_size - new_size)


def gc_run(
    run_dir: Optional[Union[Path, str]] = None,
    dry_run: bool = False,
    stride: int = 10,
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Clean rebuildable views and compact events on a finished run."""
    run_dir = _resolve_run_dir(run_dir, cwd=cwd)
    is_active, _ = check_run_active(run_dir)
    if is_active:
        raise RuntimeError(f"Cannot perform GC on active training run: {run_dir}")

    # 1. Views directory
    views_dir = run_dir / "views"
    views_reclaimed = 0
    if views_dir.is_dir():
        for p in views_dir.rglob("*"):
            if p.is_file():
                try:
                    views_reclaimed += p.stat().st_size
                except OSError:
                    pass
        if not dry_run:
            shutil.rmtree(views_dir, ignore_errors=True)

    # 2. Events compaction
    events_reclaimed = 0
    events_file = run_dir / "events.jsonl"
    if events_file.is_file():
        if dry_run:
            # Estimate compaction size
            before_size = events_file.stat().st_size
            events_reclaimed = int(before_size * 0.85)  # conservative dry-run estimate
        else:
            events_reclaimed = compact_events(run_dir, stride=stride)

    total_reclaimed = views_reclaimed + events_reclaimed
    return {
        "run_dir": str(run_dir),
        "views_reclaimed": views_reclaimed,
        "events_reclaimed": events_reclaimed,
        "total_reclaimed": total_reclaimed,
        "reclaimed_bytes": total_reclaimed,
        "views_reclaimed_bytes": views_reclaimed,
        "events_reclaimed_bytes": events_reclaimed,
        "human_reclaimed": format_bytes(total_reclaimed),
        "dry_run": dry_run,
    }


def gc_all(
    root: Optional[Union[Path, str]] = None,
    dry_run: bool = False,
    stride: int = 10,
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Clean all finished runs under root (or across workspace runs_root + registered runs when root is None)."""
    from .registry import _is_isolated_test_env, find_workspace_runs_root, is_run_dir, list_registered_runs

    base_cwd = Path(cwd).resolve() if cwd is not None else (_is_isolated_test_env() or Path.cwd().resolve())

    if root is None:
        candidates: List[Path] = []
        seen: set[str] = set()
        ws_file = find_workspace(base_cwd)
        ws_dir = ws_file.parent.resolve() if ws_file is not None else None
        ws_root = find_workspace_runs_root(base_cwd)
        if ws_root and ws_root.is_dir():
            for c in sorted(ws_root.iterdir()):
                if c.is_dir() and not c.name.startswith(".") and is_run_dir(c):
                    rp = c.resolve()
                    if str(rp) not in seen:
                        seen.add(str(rp))
                        candidates.append(rp)
        for e in list_registered_runs(cwd=base_cwd):
            rdir = e.get("run_dir")
            if rdir:
                rp = Path(rdir).resolve()
                if rp.is_dir() and is_run_dir(rp) and str(rp) not in seen:
                    if ws_dir is not None:
                        in_ws = (
                            (ws_root is not None and rp.is_relative_to(ws_root))
                            or rp.is_relative_to(ws_dir)
                            or e.get("workspace") == str(ws_dir)
                            or e.get("workspace_root") == str(ws_dir)
                        )
                        if not in_ws:
                            continue
                    seen.add(str(rp))
                    candidates.append(rp)
    else:
        target_root = Path(root).resolve()
        if not target_root.exists():
            return []
        if is_run_dir(target_root):
            candidates = [target_root]
        else:
            candidates = [
                c.resolve()
                for c in target_root.iterdir()
                if c.is_dir() and not c.name.startswith(".") and is_run_dir(c)
            ]

    results = []
    for cand in candidates:
        is_active, _ = check_run_active(cand)
        if is_active:
            results.append({
                "run_dir": str(cand),
                "skipped": True,
                "reason": "active",
                "total_reclaimed": 0,
                "human_reclaimed": "0 B",
            })
            continue
        try:
            res = gc_run(cand, dry_run=dry_run, stride=stride)
            results.append(res)
        except Exception as exc:
            results.append({
                "run_dir": str(cand),
                "skipped": True,
                "reason": str(exc),
                "total_reclaimed": 0,
                "human_reclaimed": "0 B",
            })
    return results


def ps_runs(*, all_runs: bool = False, cwd: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """List active training runs (or all runs if all_runs=True) enriched with live GPU VRAM and queue status."""
    all_items = list_runs(root=None, sort="updated", status_filter=None, cwd=cwd)
    items = all_items if all_runs else [r for r in all_items if r.get("active")]

    # Enrich active runs with GPU VRAM and device info via GPU inventory provider (if available)
    gpu_procs: List[Dict[str, Any]] = []
    gpu_hw: Dict[str, Dict[str, Any]] = {}
    if any(r.get("active") for r in items):
        try:
            from .gpu_inventory import inspect_gpu_hardware, inspect_gpu_processes
            for g in inspect_gpu_hardware():
                if g.get("uuid"):
                    gpu_hw[str(g["uuid"])] = g
            gpu_procs = inspect_gpu_processes()
        except Exception:
            gpu_procs = []

    for r in items:
        r_dir = Path(r["run_dir"])
        if not r.get("device"):
            r["device"] = "cuda"
        r["vram_mb"] = None

        if r.get("active"):
            pid = r.get("pid")
            r_dir_str = str(r_dir)
            r_name = r_dir.name
            vram_total = 0.0
            matched_gpus = []
            for gp in gpu_procs:
                gp_pid = gp.get("pid")
                gp_cmd = gp.get("cmdline", "")
                if (pid and gp_pid == pid) or (r_dir_str in gp_cmd) or (r_name and r_name in gp_cmd):
                    vram_total += float(gp.get("used_memory") or 0.0)
                    uuid_str = str(gp.get("gpu_uuid", ""))
                    if uuid_str in gpu_hw:
                        idx = gpu_hw[uuid_str].get("index")
                        if idx is not None and f"cuda:{idx}" not in matched_gpus:
                            matched_gpus.append(f"cuda:{idx}")
            if vram_total > 0:
                r["vram_mb"] = round(vram_total, 1)
            if matched_gpus:
                r["device"] = ",".join(matched_gpus)

    # Also check queued jobs from queue.json if present
    queued_jobs: List[Dict[str, Any]] = []
    try:
        from .queue import find_queue_file, load_queue
        q_file = find_queue_file(cwd)
        if q_file.is_file():
            q_data = load_queue(q_file)
            for j in q_data.get("jobs", []):
                if j.get("status") == "queued":
                    queued_jobs.append(j)
    except Exception:
        pass

    return {
        "runs": items,
        "queued": queued_jobs,
        "active_count": sum(1 for r in items if r.get("active")),
        "total_known": len(all_items),
    }


def _truncate_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    half = (max_len - 3) // 2
    return text[:half] + "..." + text[-(max_len - 3 - half):]


def format_ps_table(ps_data: Dict[str, Any], *, all_runs: bool = False, quiet: bool = False) -> str:
    """Format `hypergan ps` output like `docker ps`."""
    runs = ps_data.get("runs", [])
    queued = ps_data.get("queued", [])

    if quiet:
        return "\n".join(str(r.get("short_id") or str(r.get("run_id", ""))[:8]) for r in runs)

    if not runs and not queued:
        if all_runs:
            return "No runs found."
        total = ps_data.get("total_known", 0)
        if total > 0:
            return f"No active training runs ({total} historical runs available; use `hypergan ps -a` or `hypergan runs`)."
        return "No active training runs."

    headers = ["RUN ID", "SLUG", "EXPERIMENT", "STATUS", "PID", "DEVICE", "VRAM", "STEP", "SPEED", "BEST FID"]
    rows = []
    for r in runs:
        short_id = str(r.get("short_id") or str(r.get("run_id", ""))[:8])
        slug = _truncate_middle(str(r.get("slug", "--")), 16)
        exp = _truncate_middle(str(r.get("name", "")), 26)
        status = "RUNNING" if r.get("active") else str(r.get("status", "unknown")).upper()
        pid_str = str(r.get("pid")) if r.get("pid") else "--"
        dev_str = str(r.get("device", "--"))
        vram = r.get("vram_mb")
        vram_str = f"{vram / 1024:.1f} GB" if (vram and vram >= 1024) else (f"{int(vram)} MB" if vram else "--")
        step_val = r.get("step", 0)
        total_val = r.get("total_steps")
        step_str = f"{step_val:,} / {total_val:,}" if total_val else f"{step_val:,}"
        sps = r.get("steps_per_sec")
        sps_str = f"{sps:.1f} it/s" if sps is not None else "--"
        bfid = r.get("best_fid")
        bfid_str = f"{bfid:.2f}" if bfid is not None else "--"
        rows.append([short_id, slug, exp, status, pid_str, dev_str, vram_str, step_str, sps_str, bfid_str])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    right_align = {4, 6, 7, 8, 9}

    def make_line(cells: List[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i in right_align:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "  ".join(parts)

    lines = [
        make_line(headers),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(make_line(row))

    if queued:
        lines.append("")
        lines.append(f"Queued jobs ({len(queued)}):")
        for j in queued[:10]:
            lines.append(f"  [{j.get('id')}] {j.get('name')} (devices: {j.get('devices') or 'default'})")

    return "\n".join(lines)


def stop_run(
    target: Optional[Union[Path, str]] = None,
    *,
    stop_all: bool = False,
    force: bool = False,
    timeout: float = 15.0,
    request_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    cwd: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Gracefully stop an active training run (or all active runs) via file request and/or SIGINT."""
    import signal
    import time
    from .registry import find_live_pid_for_run, resolve_run_target, sync_registry
    from .run_requests import request_status as _req_status, submit_stop_request

    if stop_all:
        entries = sync_registry(cwd=cwd)
        active_entries = [e for e in entries if e.get("active")]
        if not active_entries:
            return {"status": "idle", "stopped": [], "message": "No active training runs to stop."}
        stopped_list = []
        for e in active_entries:
            res = stop_run(e["run_dir"], stop_all=False, force=force, timeout=timeout, cwd=cwd)
            stopped_list.append(res)
        return {
            "status": "stopped",
            "stopped": stopped_list,
            "message": f"Stopped {len(stopped_list)} active training run(s).",
        }

    run_dir = resolve_run_target(target, cwd=cwd, allow_latest=False, require_active=(target is None))
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    summary = read_summary_json(run_dir)
    manifest = _read_manifest_safe(run_dir)
    host_info = inspect_run_host(run_dir)
    is_active, pid = check_run_active(run_dir)

    if not is_active and not force:
        if request_id is not None:
            try:
                existing_receipt = _req_status(run_dir, request_id)
                target_attempt = attempt_id or existing_receipt["request"]["attempt_id"]
                if (
                    existing_receipt["request"].get("action") == "stop"
                    and existing_receipt["request"].get("attempt_id") == target_attempt
                ):
                    run_id_val = manifest.get("run_id") or run_dir.name
                    return {
                        "status": "stopped",
                        "run_id": str(run_id_val),
                        "short_id": str(run_id_val)[:8],
                        "slug": str(manifest.get("slug") or ""),
                        "name": str(manifest.get("name") or run_dir.name),
                        "run_dir": str(run_dir),
                        "pid": pid,
                        "hostname": host_info["hostname"],
                        "remote": host_info["remote"],
                        "request_id": existing_receipt["request"]["request_id"],
                        "receipt": existing_receipt,
                        "message": f"Run {str(run_id_val)[:8]} is already stopped.",
                    }
            except FileNotFoundError:
                pass
        raise ValueError(f"Run '{run_dir.name}' is not currently running.")

    receipt = None
    if manifest.get("run_id") and (attempt_id or manifest.get("attempt_id")):
        try:
            receipt = submit_stop_request(
                run_dir,
                run_id=str(manifest["run_id"]),
                attempt_id=str(attempt_id or manifest["attempt_id"]),
                request_id=request_id,
            )
        except RuntimeError:
            pass

    if pid is None and not host_info["remote"]:
        pid = find_live_pid_for_run(run_dir)
    if pid is None and not force and receipt is None and not host_info["remote"]:
        raise RuntimeError(f"Run '{run_dir.name}' holds .run.lock, but its live process PID could not be resolved.")

    run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or run_dir.name
    slug = (summary.get("slug") if summary else None) or manifest.get("slug") or ""
    name = (
        (summary.get("name") if summary else None)
        or manifest.get("name")
        or (manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}).get("name")
        or run_dir.name
    )

    stopped_cleanly = not is_active
    signaled = False
    if not host_info["remote"] and pid is not None and int(pid) != os.getpid():
        can_signal = True
        proc_cmd = Path(f"/proc/{int(pid)}/cmdline")
        if Path("/proc").is_dir():
            if not proc_cmd.is_file():
                can_signal = False
            else:
                try:
                    raw_cmd = proc_cmd.read_bytes().decode("utf-8", errors="ignore")
                    if "hypergan" not in raw_cmd and "python" not in raw_cmd:
                        can_signal = False
                except OSError:
                    can_signal = False
        if can_signal:
            try:
                os.kill(int(pid), signal.SIGINT)
                signaled = True
            except (ProcessLookupError, PermissionError):
                pass

            wait_sec = min(float(timeout), 2.0) if force else max(0.5, float(timeout))
            deadline = time.monotonic() + wait_sec
            while time.monotonic() < deadline:
                still_active, _ = check_run_active(run_dir)
                if not still_active:
                    stopped_cleanly = True
                    break
                time.sleep(0.1)

            if force and not stopped_cleanly:
                try:
                    os.kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
                    signaled = True
                except (ProcessLookupError, PermissionError):
                    pass
                stopped_cleanly = True
        elif force:
            stopped_cleanly = True
    elif force and not host_info["remote"]:
        stopped_cleanly = True

    # Refresh summary and registry state
    sync_registry(cwd=cwd or run_dir)
    state_str = "stopped" if stopped_cleanly else "stopping"
    if stopped_cleanly:
        msg = f"Stopped {str(run_id)[:8]} ({slug or name}) [PID {pid}] cleanly."
    elif host_info["remote"]:
        msg = f"Submitted durable stop request for remote run {str(run_id)[:8]} on {host_info['hostname']}."
    else:
        msg = f"Sent graceful stop signal (SIGINT) to {str(run_id)[:8]} ({slug or name}) [PID {pid}]; waiting for checkpoint to finish."
    result: Dict[str, Any] = {
        "status": state_str,
        "run_id": str(run_id),
        "short_id": str(run_id)[:8],
        "slug": str(slug),
        "name": str(name),
        "run_dir": str(run_dir),
        "pid": int(pid) if pid is not None else None,
        "hostname": host_info["hostname"],
        "remote": host_info["remote"],
        "signaled": signaled,
        "message": msg,
    }
    if receipt is not None:
        result["request_id"] = receipt["request"]["request_id"]
        result["receipt"] = receipt
    return result


def tag_run(
    target: Union[Path, str],
    new_name: str,
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Rename/re-tag a run in summary.json, manifest.json, and the run registry."""
    from .registry import extract_or_derive_slug, register_run

    clean_name = str(new_name).strip()
    if not clean_name:
        raise ValueError("New run name must not be empty.")

    run_dir = _resolve_run_dir(target, cwd=cwd)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {target}")

    summary = read_summary_json(run_dir)
    if isinstance(summary, dict):
        summary["name"] = clean_name
        atomic_write_json(run_dir / "summary.json", summary)

    manifest_path = run_dir / "manifest.json"
    manifest = _read_manifest_safe(run_dir)
    if manifest_path.is_file() and manifest:
        manifest["name"] = clean_name
        if isinstance(manifest.get("config"), dict):
            manifest["config"]["name"] = clean_name
        atomic_write_json(manifest_path, manifest)

    entry = register_run(run_dir, name=clean_name)
    run_id = str(entry.get("run_id") or (summary.get("run_id") if summary else None) or manifest.get("run_id") or run_dir.name)
    slug = str(
        entry.get("slug")
        or (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or extract_or_derive_slug(run_id, name=clean_name, run_dir_name=run_dir.name)
    )
    return {
        "run_id": run_id,
        "slug": slug,
        "name": clean_name,
        "run_dir": str(run_dir),
    }


def remove_run(
    target: Union[Path, str],
    *,
    force: bool = False,
    cwd: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Remove a run directory and unregister it from runs.json, refusing active runs unless force=True."""
    from .registry import _read_registry_unlocked, extract_or_derive_slug, get_registry_dir, unregister_run

    run_dir = _resolve_run_dir(target, cwd=cwd)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {target}")

    active, pid = check_run_active(run_dir)
    if active and not force:
        raise RuntimeError(f"Run '{run_dir.name}' is currently active (PID {pid}). Use --force / -f to stop and remove it.")

    summary = read_summary_json(run_dir)
    manifest = _read_manifest_safe(run_dir)
    reg_entry: Optional[Dict[str, Any]] = None
    try:
        reg_data = _read_registry_unlocked(get_registry_dir(cwd or run_dir))
        reg_entry = reg_data.get("runs", {}).get(str(run_dir))
    except Exception:
        reg_entry = None

    run_id = str(
        (summary.get("run_id") if summary else None)
        or manifest.get("run_id")
        or (reg_entry.get("run_id") if reg_entry else None)
        or run_dir.name
    )
    name = str(
        (summary.get("name") if summary else None)
        or manifest.get("name")
        or (manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}).get("name")
        or (reg_entry.get("name") if reg_entry else None)
        or run_dir.name
    )
    slug = str(
        (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or (reg_entry.get("slug") if reg_entry else None)
        or extract_or_derive_slug(run_id, name=name, run_dir_name=run_dir.name)
    )

    if active and force:
        stop_run(run_dir, force=True, cwd=cwd)

    unregister_run(run_dir, cwd=cwd)
    shutil.rmtree(run_dir)

    return {
        "removed": True,
        "run_id": run_id,
        "slug": slug,
        "name": name,
        "run_dir": str(run_dir),
    }


def _format_event_log_line(ev: Dict[str, Any], total_steps: Optional[int] = None) -> Optional[str]:
    kind = ev.get("event")
    step = ev.get("step")
    if kind == "start":
        att = ev.get("attempt_id", "0001")
        return f"[start] Run started (attempt {att})"
    if kind == "train":
        s_val = int(step or 0)
        if total_steps and total_steps > 0:
            pct = s_val / total_steps * 100
            step_part = f"[{s_val:7,d}/{total_steps:,d}] {pct:5.1f}%"
        else:
            step_part = f"[step {s_val:7,d}]"
        sps = ev.get("steps_per_second")
        sps_part = f"{sps:5.1f} it/s" if isinstance(sps, (int, float)) else "   -- it/s"
        metrics = ev.get("metrics") or {}
        m_parts = []
        for k in ("loss/g_total", "loss_g", "loss/d_total", "loss_d", "loss/total"):
            if k in metrics and isinstance(metrics[k], (int, float)):
                m_parts.append(f"{k}: {metrics[k]:.4f}")
        if not m_parts:
            for k, v in list(metrics.items())[:4]:
                if isinstance(v, (int, float)):
                    m_parts.append(f"{k}: {v:.4f}")
        metrics_str = "  ".join(m_parts) if m_parts else ""
        return f"{step_part} | {sps_part}" + (f" | {metrics_str}" if metrics_str else "")
    if kind == "checkpoint":
        ckpt = ev.get("checkpoint_path") or ev.get("path") or ""
        return f"[checkpoint] step {int(step or 0):,d} -> {ckpt}"
    if kind in ("evaluation", "evaluation_complete"):
        m_id = ev.get("metric_id") or ev.get("metric") or "eval"
        val = ev.get("value")
        if val is None and isinstance(ev.get("result"), dict):
            val = ev["result"].get("value")
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val if val is not None else "complete")
        return f"[eval] step {int(step or 0):,d} | {m_id} = {val_str}"
    if kind in ("complete", "completed", "stopped", "failed"):
        return f"[{kind}] step {int(step or 0):,d}"
    return None


def read_run_logs(
    target: Optional[Union[Path, str]] = None,
    *,
    tail: int = 50,
    follow: bool = False,
    stream=None,
    cwd: Optional[Union[Path, str]] = None,
) -> List[str]:
    """Read or follow formatted training events from a run's events.jsonl."""
    import time

    out = stream or sys.stdout
    resolved_dir = _resolve_run_dir(target, cwd=cwd)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {resolved_dir}")

    summary = read_summary_json(resolved_dir)
    manifest = _read_manifest_safe(resolved_dir)
    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    total_steps = (
        (summary.get("total_steps") if summary else None)
        or (summary.get("max_steps") if summary else None)
        or config.get("training", {}).get("steps")
        or config.get("train", {}).get("steps")
        or manifest.get("total_steps")
    )

    events_file = resolved_dir / "events.jsonl"
    formatted_lines: List[str] = []
    offset = 0

    if events_file.is_file():
        try:
            with events_file.open("rb") as f:
                raw_bytes = f.read()
                offset = len(raw_bytes)
            for raw_line in raw_bytes.decode("utf-8", errors="ignore").splitlines():
                if not raw_line.strip():
                    continue
                try:
                    ev = json.loads(raw_line)
                    if isinstance(ev, dict):
                        line = _format_event_log_line(ev, total_steps=total_steps)
                        if line:
                            formatted_lines.append(line)
                except Exception:
                    continue
        except OSError:
            pass

    if tail and tail > 0 and len(formatted_lines) > tail:
        formatted_lines = formatted_lines[-tail:]

    for line in formatted_lines:
        print(line, file=out, flush=True)

    if not follow:
        return formatted_lines

    try:
        while True:
            if events_file.is_file():
                try:
                    with events_file.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read()
                        if chunk:
                            offset += len(chunk)
                            for raw_line in chunk.decode("utf-8", errors="ignore").splitlines():
                                if not raw_line.strip():
                                    continue
                                try:
                                    ev = json.loads(raw_line)
                                    if isinstance(ev, dict):
                                        line = _format_event_log_line(ev, total_steps=total_steps)
                                        if line:
                                            formatted_lines.append(line)
                                            print(line, file=out, flush=True)
                                        if ev.get("event") in ("complete", "completed", "stopped", "failed"):
                                            return formatted_lines
                                except Exception:
                                    continue
                except OSError:
                    pass
            is_active, _ = check_run_active(resolved_dir)
            if not is_active:
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nDetached from logs (training continues in background).", file=sys.stderr, flush=True)

    return formatted_lines

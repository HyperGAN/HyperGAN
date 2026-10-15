"""Run inspection, discovery, comparison, and garbage collection.

Operates strictly on run metadata (manifest.json, summary.json, events.jsonl).
Never imports torch, particlegan, or training execution dependencies.
"""
import csv
from datetime import datetime, timezone
import fcntl
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


def check_run_active(run_dir: Union[Path, str]) -> Tuple[bool, Optional[int]]:
    """Test non-blocking flock on .run.lock to check if training process is active."""
    run_dir = Path(run_dir).resolve()
    lock_file = run_dir / ".run.lock"
    is_active = False
    if lock_file.exists():
        try:
            with lock_file.open("a+b") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    is_active = False
                except (BlockingIOError, OSError):
                    is_active = True
        except Exception:
            pass

    pid = None
    launch_dir = run_dir / "launch"
    if launch_dir.is_dir():
        attempts = sorted(launch_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        for att_file in attempts:
            try:
                data = json.loads(att_file.read_text(encoding="utf-8"))
                p = data.get("pid") or data.get("supervisor_pid")
                if isinstance(p, int):
                    pid = p
                    break
            except Exception:
                pass

    return is_active, pid


def _resolve_run_dir(run_target: Union[Path, str]) -> Path:
    target = Path(run_target)
    if target.is_dir() and ((target / "manifest.json").is_file() or (target / "summary.json").is_file()):
        return target.resolve()
    # Try resolving relative to workspace runs_root
    ws = find_workspace(Path.cwd())
    if ws and ws.runs_root:
        cand = ws.runs_root / target
        if cand.is_dir() and ((cand / "manifest.json").is_file() or (cand / "summary.json").is_file()):
            return cand.resolve()
    return target.resolve()


def run_status(run_dir: Union[Path, str]) -> Dict[str, Any]:
    """Compile human-friendly status summary for a run (fast-path summary.json or manifest fallback)."""
    resolved_dir = _resolve_run_dir(run_dir)
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    is_active, pid = check_run_active(resolved_dir)
    summary = read_summary_json(resolved_dir)

    manifest_path = resolved_dir / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or resolved_dir.name
    name = (summary.get("name") if summary else None) or config.get("name") or run_id

    recorded_status = (summary.get("status") if summary else None) or manifest.get("status", "unknown")
    if is_active:
        status_str = "RUNNING"
    else:
        status_str = recorded_status.upper()

    step = (summary.get("step") if summary else None)
    if step is None:
        step = manifest.get("steps", 0)

    total_steps = (summary.get("total_steps") if summary else None) or config.get("training", {}).get("steps") or manifest.get("total_steps")
    percent = round((step / total_steps * 100), 1) if (total_steps and total_steps > 0) else None

    sps = (summary.get("steps_per_sec") or summary.get("steps_per_second") or summary.get("speed_its")) if summary else None
    if sps is None:
        sps = manifest.get("steps_per_second")

    eta_seconds = summary.get("eta_seconds") if summary else None
    if eta_seconds is None and sps and sps > 0 and total_steps and total_steps > step:
        eta_seconds = round((total_steps - step) / sps, 1)

    elapsed_seconds = manifest.get("training_seconds") or manifest.get("seconds")

    devices_str = config.get("training", {}).get("device") or manifest.get("device") or "unknown"
    launch_dir = resolved_dir / "launch"
    if launch_dir.is_dir():
        attempts = sorted(launch_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        if attempts:
            try:
                att_data = json.loads(attempts[0].read_text(encoding="utf-8"))
                cuda_devs = att_data.get("env", {}).get("CUDA_VISIBLE_DEVICES")
                if cuda_devs:
                    devices_str = f"cuda:{cuda_devs}"
            except Exception:
                pass

    latest_metrics = dict(summary.get("latest", {}) if summary else {})
    if not latest_metrics and (resolved_dir / "events.jsonl").is_file():
        # Fallback reading events tail
        try:
            with (resolved_dir / "events.jsonl").open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 65536))
                tail = f.read().decode("utf-8", errors="ignore")
                for line in reversed(tail.splitlines()):
                    if line.strip():
                        try:
                            ev = json.loads(line)
                            if ev.get("event") == "train" and "metrics" in ev:
                                latest_metrics = ev["metrics"]
                                break
                        except Exception:
                            continue
        except Exception:
            pass

    best = summary.get("best") if summary else None
    best_fid = summary.get("best_fid") if summary else None
    best_step = summary.get("best_step") if summary else None
    best_checkpoint = None

    if best:
        best_fid = best.get("value")
        best_step = best.get("step")
        best_checkpoint = best.get("checkpoint")
    elif not best:
        # Fallback scan evaluations
        eval_root = resolved_dir / "metrics" / "evaluations"
        best_val = None
        best_s = None
        if eval_root.is_dir():
            for rec in eval_root.glob("*/receipt.json"):
                try:
                    data = json.loads(rec.read_text(encoding="utf-8"))
                    if data.get("status") == "complete" and "result" in data:
                        val = data["result"].get("value")
                        s = data.get("source_step") or data["result"].get("step")
                        if val is not None and (best_val is None or val < best_val):
                            best_val = val
                            best_s = s
                except Exception:
                    continue
        if best_val is not None:
            best_fid = best_val
            best_step = best_s
            best = {"metric": "fid50k_train", "value": best_val, "step": best_s, "checkpoint": None}

    # Recent checkpoints
    checkpoints = summary.get("checkpoints") if summary else None
    if checkpoints is None:
        ckpt_root = resolved_dir / "checkpoints"
        ckpts = []
        if ckpt_root.is_dir():
            for cm in ckpt_root.glob("*/manifest.json"):
                try:
                    cdata = json.loads(cm.read_text(encoding="utf-8"))
                    ckpts.append({
                        "step": cdata.get("step", 0),
                        "name": cm.parent.name,
                        "path": f"checkpoints/{cm.parent.name}",
                        "metrics": {},
                    })
                except Exception:
                    continue
        ckpts.sort(key=lambda c: c["step"])
        checkpoints = ckpts

    recent_ckpts = list(reversed(checkpoints[-5:])) if checkpoints else []

    warnings = list(manifest.get("warnings", [])) + list(manifest.get("resume_warnings", []))

    return {
        "run_id": run_id,
        "name": name,
        "run_dir": str(resolved_dir),
        "status": status_str,
        "active": is_active,
        "pid": pid,
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


def format_run_status(info: Dict[str, Any]) -> str:
    """Format status dict into single-screen terminal summary."""
    sps = info.get("steps_per_sec")
    speed_str = f"{sps:.1f} it/s" if (sps is not None and isinstance(sps, (int, float))) else "-- it/s"
    lines = [
        "=" * 80,
        f"HyperGAN Run Status: {info['run_id']}",
        "=" * 80,
        f"Experiment:  {info['name']}",
        f"Run Dir:     {info['run_dir']}",
        f"Status:      {info['status']}" + (" (active)" if info["active"] else " (stopped/idle)"),
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
        if k in latest:
            metric_pairs.append(f"{k}: {latest[k]:.4f}")
    if metric_pairs:
        lines.append("  " + "    ".join(metric_pairs[:4]))
    else:
        lines.append("  (No intermediate scalar metrics logged)")

    if info.get("best_fid") is not None:
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
            fid_val = c.get("metrics", {}).get("fid50k_train")
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


def list_runs(root: Optional[Union[Path, str]] = None,
              sort: str = "fid",
              status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover and list all runs under root with filtering and sorting."""
    if root is None:
        ws = find_workspace(Path.cwd())
        if ws and ws.runs_root and ws.runs_root.is_dir():
            target_root = ws.runs_root
        else:
            target_root = Path.cwd()
    else:
        target_root = Path(root).resolve()

    if not target_root.exists():
        return []

    # If target_root is itself a run
    if (target_root / "manifest.json").is_file() or (target_root / "summary.json").is_file():
        candidates = [target_root]
    else:
        candidates = []
        try:
            for child in target_root.iterdir():
                if child.name.startswith(".") or not child.is_dir():
                    continue
                if (child / "manifest.json").is_file() or (child / "summary.json").is_file():
                    candidates.append(child)
        except OSError:
            return []

    runs = []
    for run_dir in candidates:
        try:
            summary = read_summary_json(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}

            config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
            run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or run_dir.name
            name = (summary.get("name") if summary else None) or config.get("name") or run_id

            is_active, _ = check_run_active(run_dir)
            status = "running" if is_active else ((summary.get("status") if summary else None) or manifest.get("status", "unknown")).lower()
            if status == "complete":
                status = "completed"

            step = (summary.get("step") if summary else None)
            if step is None:
                step = manifest.get("steps", 0)

            total_steps = (summary.get("total_steps") if summary else None) or config.get("training", {}).get("steps") or manifest.get("total_steps")
            sps = (summary.get("steps_per_sec") or summary.get("steps_per_second") or summary.get("speed_its")) if summary else None
            if sps is None:
                sps = manifest.get("steps_per_second")

            best = summary.get("best") if summary else None
            best_fid = (best.get("value") if best else None) or (summary.get("best_fid") if summary else None)
            if best_fid is None and (run_dir / "metrics" / "evaluations").is_dir():
                for rec in (run_dir / "metrics" / "evaluations").glob("*/receipt.json"):
                    try:
                        rdata = json.loads(rec.read_text(encoding="utf-8"))
                        if rdata.get("status") == "complete" and "result" in rdata:
                            val = rdata["result"].get("value")
                            if val is not None and (best_fid is None or val < best_fid):
                                best_fid = val
                    except Exception:
                        continue

            latest = summary.get("latest", {}) if summary else {}
            latest_metric_str = "--"
            for k in ("loss/g_total", "loss_g", "loss/total"):
                if k in latest:
                    latest_metric_str = f"loss_g={latest[k]:.2f}"
                    break

            updated_at = (summary.get("updated_at") if summary else None) or manifest.get("updated_at")
            if not updated_at:
                try:
                    target_file = run_dir / "summary.json" if (run_dir / "summary.json").is_file() else manifest_path
                    mtime = target_file.stat().st_mtime
                    updated_at = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                except OSError:
                    updated_at = "--"

            runs.append({
                "run_id": run_id,
                "run_dir": str(run_dir),
                "name": name,
                "status": status,
                "step": step,
                "total_steps": total_steps,
                "steps_per_sec": sps,
                "latest_metric": latest_metric_str,
                "best_fid": best_fid,
                "updated_at": updated_at,
            })
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
    else:  # updated
        runs.sort(key=lambda r: str(r["updated_at"]), reverse=True)

    return runs


def format_runs_table(runs: List[Dict[str, Any]]) -> str:
    """Format runs list into dynamic terminal table."""
    if not runs:
        return "No runs found."

    headers = ["Run ID", "Experiment", "Status", "Step", "it/s", "Latest Metric", "Best FID", "Updated"]
    rows = []
    for r in runs:
        run_id_short = r["run_id"][:8]
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
        rows.append([run_id_short, name, status, step_str, sps_str, latest, fid_str, updated])

    # Dynamic column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def make_line(items: List[str]) -> str:
        return "  ".join(items[i].ljust(widths[i]) if i != 3 and i != 4 and i != 6 else items[i].rjust(widths[i]) for i in range(len(items)))

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
                if ev.get("metric") == metric or ev.get("metric", "").lower() == metric.lower():
                    evals_by_step[ev["step"]] = ev["value"]
        if not evals_by_step and (resolved / "metrics" / "evaluations").is_dir():
            for receipt_file in (resolved / "metrics" / "evaluations").glob("*/receipt.json"):
                try:
                    rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
                    if rdata.get("status") == "complete" and "result" in rdata:
                        m_id = rdata.get("metric_id") or rdata.get("metric", "")
                        if m_id == metric or m_id.lower() == metric.lower():
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
                            if ev.get("event") == "train" and "metrics" in ev:
                                s = ev.get("step")
                                m_dict = ev["metrics"]
                                val = m_dict.get(metric) or m_dict.get(metric.replace("/", "_"))
                                if s is not None and val is not None:
                                    evals_by_step[int(s)] = float(val)
                        except Exception:
                            continue
            except Exception:
                pass

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
    writer = csv.writer(output)
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
    run_dir = Path(run_dir).resolve()
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return 0

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

    temp_path = run_dir / f".events.jsonl.compact.{os.getpid()}.tmp"
    digest = hashlib.sha256()
    new_size = 0
    last_event = None

    with events_path.open("rb") as src, temp_path.open("wb") as dst:
        for line in src:
            if not line.strip():
                continue
            keep = True
            try:
                ev = json.loads(line)
                if ev.get("event") == "train":
                    step = ev.get("step")
                    if step not in pinned_steps and (step % stride != 0):
                        keep = False
            except Exception:
                keep = True

            if keep:
                dst.write(line)
                digest.update(line)
                new_size += len(line)
                try:
                    last_event = json.loads(line)
                except Exception:
                    pass

        dst.flush()
        os.fsync(dst.fileno())

    os.replace(temp_path, events_path)

    # Update manifest
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["events_compacted"] = True
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


def gc_run(run_dir: Union[Path, str], dry_run: bool = False, stride: int = 10) -> Dict[str, Any]:
    """Clean rebuildable views and compact events on a finished run."""
    run_dir = Path(run_dir).resolve()
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


def gc_all(root: Optional[Union[Path, str]] = None,
           dry_run: bool = False,
           stride: int = 10) -> List[Dict[str, Any]]:
    """Clean all finished runs under root, cleanly skipping active runs."""
    if root is None:
        ws = find_workspace(Path.cwd())
        target_root = ws.runs_root if ws and ws.runs_root else Path.cwd()
    else:
        target_root = Path(root).resolve()

    if not target_root.exists():
        return []

    if (target_root / "manifest.json").is_file():
        candidates = [target_root]
    else:
        candidates = [c for c in target_root.iterdir() if c.is_dir() and not c.name.startswith(".") and (c / "manifest.json").is_file()]

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

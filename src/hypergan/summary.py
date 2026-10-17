"""Atomic summary.json persistence, checkpoint/evaluation collection, and run indexing."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from .run_state import sync_directory


def _replace_file(source, destination):
    # Windows can deny replacement while another handle briefly inspects the
    # destination. Keep the staged bytes and atomic replacement; retry only
    # access/sharing/lock violations, with at most .75 seconds of backoff.
    for attempt in range(11):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33)
                    or attempt == 10):
                raise
            time.sleep(min(.01 * 2 ** attempt, .1))


def atomic_write_json(path: Path | str, data: dict) -> None:
    """Atomically write data as JSON to path with fsync and directory sync."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_file = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_file(temp_file, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass


def read_summary_json(run_dir: Path | str) -> dict | None:
    """Read summary.json from a run directory if present and valid."""
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError, UnicodeError):
        pass
    return None


def collect_evaluations(run_dir: Path) -> list[dict]:
    """Scan metrics/evaluations/*/receipt.json and collect completed evaluation records."""
    eval_root = Path(run_dir) / "metrics" / "evaluations"
    results = []
    if not eval_root.is_dir():
        return results

    for receipt_path in eval_root.glob("*/receipt.json"):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "complete" or "result" not in receipt:
                continue
            val = receipt["result"].get("value")
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            step = receipt.get("source_step")
            if step is None:
                step = receipt.get("result", {}).get("step")
            if step is None:
                step = receipt.get("step")
            if step is None:
                continue

            metric_id = receipt.get("metric_id") or receipt.get("metric", "")
            ts = receipt.get("timestamp")
            if ts is None:
                try:
                    ts = datetime.fromtimestamp(receipt_path.stat().st_mtime, timezone.utc).isoformat()
                except OSError:
                    ts = None

            results.append({
                "step": int(step),
                "metric": str(metric_id),
                "value": float(val),
                "dir": receipt_path.parent.name,
                "timestamp": ts,
            })
        except Exception:
            continue

    results.sort(key=lambda r: (r["step"], r["metric"], r["dir"]))
    return results


def collect_checkpoints(run_dir: Path, evaluations_by_step: dict | None = None) -> list[dict]:
    """Scan checkpoints/*/manifest.json and collect active checkpoint records."""
    ckpt_root = Path(run_dir) / "checkpoints"
    results = []
    if not ckpt_root.is_dir():
        return results

    eval_by_step = evaluations_by_step or {}

    for manifest_path in ckpt_root.glob("*/manifest.json"):
        dir_name = manifest_path.parent.name
        if dir_name.startswith("."):
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            step = data.get("step")
            if step is None or not isinstance(step, int):
                continue

            ts = data.get("created_at")
            if ts is None:
                try:
                    ts = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc).isoformat()
                except OSError:
                    ts = None

            metrics_at_step = dict(eval_by_step.get(step, {}))

            results.append({
                "step": step,
                "name": dir_name,
                "path": f"checkpoints/{dir_name}",
                "timestamp": ts,
                "metrics": metrics_at_step,
            })
        except Exception:
            continue

    results.sort(key=lambda c: (c["step"], c["name"]))
    return results


def compute_best_evaluation(evaluations: list[dict], checkpoints_by_step: dict | None = None, policy: dict | None = None) -> dict | None:
    """Identify the best evaluation record according to retention policy or FID priority."""
    if not evaluations:
        return None

    policy = policy or {}
    keep_best = policy.get("keep_best") if isinstance(policy, dict) else None
    target_metric = None
    target_mode = None
    if isinstance(keep_best, dict):
        target_metric = keep_best.get("metric")
        target_mode = keep_best.get("mode")

    if target_metric:
        matched = [r for r in evaluations if r["metric"] == target_metric or r["metric"].lower() == target_metric.lower()]
        if not matched and "fid" in target_metric.lower():
            matched = [r for r in evaluations if "fid" in r["metric"].lower()]
        records = matched if matched else evaluations
    else:
        fid_recs = [r for r in evaluations if "fid" in r["metric"].lower()]
        records = fid_recs if fid_recs else evaluations

    if not records:
        return None

    if target_mode in ("min", "max"):
        mode_min = (target_mode == "min")
    else:
        sample_metric = records[0]["metric"].lower()
        mode_min = not (sample_metric in ("is", "inception_score", "score", "acc", "accuracy")
                        or sample_metric.endswith(("_is", "_score", "_acc", "_accuracy")))

    sorted_records = sorted(
        records,
        key=lambda r: (r["value"] if mode_min else -r["value"], -r["step"])
    )
    best_record = sorted_records[0]

    step = best_record["step"]
    ckpt_name = None
    if checkpoints_by_step and step in checkpoints_by_step:
        ckpt_name = checkpoints_by_step[step]["name"]

    return {
        "metric": best_record["metric"],
        "value": best_record["value"],
        "step": step,
        "checkpoint": ckpt_name,
    }


def write_summary_json(run_dir: Path | str, manifest: dict | None = None, metrics: dict | None = None, status: str | None = None) -> Path:
    """Atomically write canonical summary.json for run_dir."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"

    if manifest is None and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest = manifest or {}

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = manifest.get("run_id") or run_dir.name
    name = config.get("name") or run_id

    resolved_status = status or manifest.get("status", "unknown")
    canonical_status = resolved_status

    step = manifest.get("steps", 0)
    total_steps = config.get("training", {}).get("steps") or manifest.get("total_steps")

    sps = manifest.get("steps_per_second")
    if sps is not None:
        try:
            sps = float(sps)
        except (ValueError, TypeError):
            sps = None

    eta = None
    if sps and sps > 0 and total_steps is not None and total_steps > step:
        eta = round((total_steps - step) / sps, 2)

    # Collect evaluations
    eval_list = collect_evaluations(run_dir)
    eval_by_step = {}
    for ev in eval_list:
        eval_by_step.setdefault(ev["step"], {})[ev["metric"]] = ev["value"]

    # Collect checkpoints
    ckpt_list = collect_checkpoints(run_dir, eval_by_step)
    ckpt_by_step = {c["step"]: c for c in ckpt_list}

    # Best evaluation
    ckpt_policy = config.get("checkpoints", {})
    best = compute_best_evaluation(eval_list, ckpt_by_step, policy=ckpt_policy)

    # Latest metrics
    existing_summary = read_summary_json(run_dir) or {}
    latest_metrics = dict(existing_summary.get("latest", {}))
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                latest_metrics[k] = float(v)

    # Attach evaluation values to latest metrics if evaluated at or after step
    for ev in eval_list:
        if ev["step"] >= step or ev["metric"] not in latest_metrics:
            latest_metrics[ev["metric"]] = ev["value"]

    latest_fid = latest_metrics.get("fid50k_train") or latest_metrics.get("fid")
    best_fid = best["value"] if best and "fid" in best["metric"].lower() else (best["value"] if best else None)
    best_step = best["step"] if best else None

    from .registry import extract_or_derive_slug

    slug = (
        manifest.get("slug")
        or existing_summary.get("slug")
        or extract_or_derive_slug(run_id, name=str(name), run_dir_name=run_dir.name)
    )
    pid = manifest.get("pid") or (os.getpid() if canonical_status in ("running", "initializing") else existing_summary.get("pid"))

    summary_doc = {
        "run_id": run_id,
        "short_id": str(run_id)[:8],
        "slug": slug,
        "name": name,
        "status": canonical_status,
        "pid": pid,
        "step": step,
        "total_steps": total_steps,
        "steps_per_sec": sps,
        "steps_per_second": sps,
        "speed_its": sps,
        "eta_seconds": eta,
        "latest": latest_metrics,
        "best": best,
        "best_fid": best_fid,
        "best_step": best_step,
        "latest_fid": latest_fid,
        "checkpoints": ckpt_list,
        "evaluations": eval_list,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": manifest.get("error"),
    }

    target = run_dir / "summary.json"
    atomic_write_json(target, summary_doc)
    if metrics is None:
        try:
            from .registry import register_run
            register_run(
                run_dir,
                run_id=str(run_id),
                name=str(name),
                slug=str(slug),
                status=str(canonical_status),
                pid=pid if isinstance(pid, int) else None,
            )
        except Exception:
            pass
    return target

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


def _catalog_definitions(run_dir, revision, cache):
    """Metric definitions of one catalog revision; empty when unreadable."""
    if revision not in cache:
        try:
            from .metrics import read_catalog
            cache[revision] = read_catalog(run_dir, revision)['metrics']
        except Exception:
            cache[revision] = {}
    return cache[revision]


def collect_evaluations(run_dir: Path) -> list[dict]:
    """Scan metrics/evaluations/*/receipt.json and collect completed evaluation records.

    Each record carries the declared ``direction`` and ``unit`` from the
    catalog revision its receipt names, or None when that catalog is missing.
    """
    eval_root = Path(run_dir) / "metrics" / "evaluations"
    results = []
    if not eval_root.is_dir():
        return results
    catalogs = {}

    for receipt_path in eval_root.glob("*/receipt.json"):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "complete" or "result" not in receipt:
                continue
            val = receipt["result"].get("value")
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
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
            definition = {}
            if isinstance(receipt.get("catalog"), str):
                definition = _catalog_definitions(run_dir, receipt["catalog"], catalogs).get(metric_id) or {}

            results.append({
                "step": int(step),
                "metric": str(metric_id),
                "value": float(val),
                "dir": receipt_path.parent.name,
                "timestamp": ts,
                "attempt_id": receipt.get("attempt_id"),
                "direction": definition.get("direction"),
                "unit": definition.get("unit"),
            })
        except Exception:
            continue

    results.sort(key=lambda r: (r["step"], r["metric"], r["dir"]))
    return results


_FID_ID = re.compile(r'(?:^|[^a-z0-9])fid(?:$|[^a-z]|50k)', re.IGNORECASE)


def is_fid_metric(record: dict) -> bool:
    """Whether an evaluation record measures FID.

    The catalog's declared unit decides. Only legacy receipts without a
    readable catalog fall back to an ``fid`` token in the metric ID.
    """
    if record.get("unit") is not None:
        return record["unit"] == "FID"
    return bool(_FID_ID.search(str(record.get("metric", ""))))


def primary_metric(evaluations: list[dict], policy: dict | None = None) -> tuple[str, str] | tuple[None, None]:
    """The single metric that defines "best" for a run, and its direction.

    ``keep_best.metric`` when it has results, otherwise the only evaluated
    metric whose catalog declares a direction. Unrelated metrics are never
    pooled. The catalog's ``direction`` wins; ``keep_best.mode`` applies only
    when the configured metric declares none.
    """
    from .model_file import normalize_checkpoint_policy
    try:
        keep_best = normalize_checkpoint_policy(policy if isinstance(policy, dict) else None)['keep_best']
    except ValueError:
        keep_best = None
    present = {r["metric"] for r in evaluations}
    if keep_best and keep_best["metric"] in present:
        metric = keep_best["metric"]
    else:
        directional = {r["metric"] for r in evaluations if r.get("direction") in ("minimize", "maximize")}
        if len(directional) != 1:
            return None, None
        metric = directional.pop()
    declared = [r.get("direction") for r in evaluations if r["metric"] == metric and r.get("direction")]
    direction = declared[-1] if declared else None
    if direction not in ("minimize", "maximize"):
        if not keep_best or metric != keep_best["metric"]:
            return None, None
        direction = "minimize" if keep_best["mode"] == "min" else "maximize"
    return metric, direction


def rank_evaluations(evaluations: list[dict], policy: dict | None = None) -> tuple[str | None, str | None, list[dict]]:
    """Records of the primary metric ordered best first; ties prefer the later step.

    This one ranking backs summary ``best``, ``resolve_checkpoint('best')``
    and ``keep_best`` retention.
    """
    metric, direction = primary_metric(evaluations, policy)
    if metric is None:
        return None, None, []
    sign = 1 if direction == "minimize" else -1
    records = [r for r in evaluations if r["metric"] == metric]
    return metric, direction, sorted(records, key=lambda r: (sign * r["value"], -r["step"]))


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
    """The best record of the run's primary metric (see ``primary_metric``)."""
    metric, direction, ranked = rank_evaluations(evaluations, policy)
    if not ranked:
        return None
    best_record = ranked[0]
    step = best_record["step"]
    ckpt_name = None
    if checkpoints_by_step and step in checkpoints_by_step:
        ckpt_name = checkpoints_by_step[step]["name"]

    return {
        "metric": metric,
        "value": best_record["value"],
        "step": step,
        "checkpoint": ckpt_name,
        "direction": direction,
        "unit": best_record.get("unit"),
    }


def write_summary_json(run_dir: Path | str, manifest: dict | None = None, metrics: dict | None = None, status: str | None = None) -> Path:
    """Atomically write canonical summary.json for run_dir."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    existing_summary = read_summary_json(run_dir) or {}

    disk_manifest = {}
    if manifest_path.is_file() and (
        manifest is None
        or not isinstance(manifest, dict)
        or any(k not in manifest for k in ("config", "status", "steps"))
    ):
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                disk_manifest = loaded
        except Exception:
            disk_manifest = {}

    if manifest is None or not isinstance(manifest, dict):
        manifest = disk_manifest
    elif disk_manifest:
        manifest = {**disk_manifest, **manifest}

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = manifest.get("run_id") or existing_summary.get("run_id") or run_dir.name
    from .registry import read_run_tag

    # `hypergan tag` writes tag.json because the trainer republishes its own manifest.
    name = read_run_tag(run_dir) or manifest.get("name") or config.get("name") or existing_summary.get("name") or run_id

    resolved_status = status or manifest.get("status") or existing_summary.get("status") or "unknown"
    canonical_status = resolved_status

    step = manifest.get("steps")
    if step is None:
        step = existing_summary.get("step", 0)
    total_steps = config.get("training", {}).get("steps") or manifest.get("total_steps") or existing_summary.get("total_steps")

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
    if best is None and isinstance(manifest.get("best_checkpoint"), dict):
        bc = manifest["best_checkpoint"]
        if "metric" in bc and "value" in bc:
            best = {
                "metric": bc["metric"],
                "value": bc["value"],
                "step": bc.get("step"),
                "checkpoint": bc.get("checkpoint"),
            }

    # Latest metrics
    latest_metrics = dict(existing_summary.get("latest", {}))
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                latest_metrics[k] = float(v)

    # Evaluations finish after their source step, so the newest source step of
    # each metric is its latest value regardless of the current training step.
    latest_eval = {}
    for ev in eval_list:
        latest_eval[ev["metric"]] = ev
        latest_metrics[ev["metric"]] = ev["value"]

    # FID fields describe the same primary metric as "best", never a pool.
    best_is_fid = best is not None and best["metric"] in latest_eval and is_fid_metric(latest_eval[best["metric"]])
    latest_fid = latest_eval[best["metric"]]["value"] if best_is_fid else None
    best_fid = best["value"] if best_is_fid else None
    best_step = best["step"] if best else None

    from .registry import extract_or_derive_slug

    slug = (
        manifest.get("slug")
        or existing_summary.get("slug")
        or extract_or_derive_slug(run_id, name=str(name), run_dir_name=run_dir.name)
    )
    pid = manifest.get("pid") or (os.getpid() if canonical_status in ("running", "initializing") else existing_summary.get("pid"))

    # Update telemetry trends (rolling EMA alpha=0.2)
    telemetry_trends = dict(existing_summary.get("telemetry_trends", {}))
    for k, v in latest_metrics.items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            old_trend = telemetry_trends.get(k)
            if old_trend is not None and isinstance(old_trend, (int, float)):
                telemetry_trends[k] = round(0.2 * float(v) + 0.8 * float(old_trend), 6)
            else:
                telemetry_trends[k] = round(float(v), 6)

    # Advisory health alerts
    health_alerts = []
    gn_critic = latest_metrics.get("grad_norm/critic", latest_metrics.get("grad_norm_critic"))
    if gn_critic is not None and isinstance(gn_critic, (int, float)) and gn_critic < 1e-5:
        health_alerts.append("Discriminator vanishing gradient (grad_norm/critic < 1e-5)")

    gn_gen = latest_metrics.get("grad_norm/generator", latest_metrics.get("grad_norm_generator"))
    if gn_gen is not None and isinstance(gn_gen, (int, float)) and gn_gen < 1e-5:
        health_alerts.append("Generator vanishing gradient (grad_norm/generator < 1e-5)")

    div_ratio = latest_metrics.get("diversity/ratio", latest_metrics.get("diversity_ratio"))
    if div_ratio is not None and isinstance(div_ratio, (int, float)) and div_ratio < 0.2:
        health_alerts.append("Mode collapse risk (diversity/ratio < 0.2)")

    if step is not None and step >= 20:
        c_acc = latest_metrics.get("critic/accuracy", latest_metrics.get("critic_accuracy"))
        if c_acc is not None and isinstance(c_acc, (int, float)):
            if c_acc > 0.98:
                health_alerts.append("Discriminator overpowering generator (critic/accuracy > 0.98)")
            elif c_acc < 0.05:
                health_alerts.append("Critic collapsed (critic/accuracy < 0.05)")

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
        "health_alerts": health_alerts,
        "telemetry_trends": telemetry_trends,
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

"""Canonical typed reader for HyperGAN run metadata and artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .registry import resolve_run_target, read_run_tag
from .summary import read_summary_json, collect_checkpoints, collect_evaluations


@dataclass
class RunInfo:
    run_id: str
    run_dir: Path
    created_at: Optional[str] = None
    status: str = "unknown"
    step: int = 0
    total_steps: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    best_metric: Optional[Dict[str, Any]] = None
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    health_alerts: List[str] = field(default_factory=list)
    telemetry_trends: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    slug: Optional[str] = None
    short_id: Optional[str] = None
    pid: Optional[int] = None
    steps_per_second: Optional[float] = None
    eta_seconds: Optional[float] = None
    evaluations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status in ("running", "initializing")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "created_at": self.created_at,
            "status": self.status,
            "step": self.step,
            "total_steps": self.total_steps,
            "config": self.config,
            "latest_metrics": self.latest_metrics,
            "best_metric": self.best_metric,
            "checkpoints": self.checkpoints,
            "health_alerts": self.health_alerts,
            "telemetry_trends": self.telemetry_trends,
            "name": self.name,
            "slug": self.slug,
            "short_id": self.short_id,
            "pid": self.pid,
            "steps_per_second": self.steps_per_second,
            "eta_seconds": self.eta_seconds,
            "evaluations": self.evaluations,
        }


def read_run_info(run_dir: Union[Path, str, None] = None) -> RunInfo:
    """Canonical loader for run metadata from manifest.json, summary.json, and tag.json."""
    if run_dir is None:
        target_path = resolve_run_target(None)
    else:
        target_path = resolve_run_target(run_dir)
    target_path = Path(target_path).resolve()

    manifest_path = target_path / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    summary = read_summary_json(target_path) or {}

    raw_status = (
        summary.get("status")
        or manifest.get("status")
        or "unknown"
    )
    # Normalize "complete" -> "completed" for canonical RunInfo status contract
    if raw_status in ("complete", "completed"):
        status = "completed"
    else:
        status = raw_status

    run_id = str(
        manifest.get("run_id")
        or summary.get("run_id")
        or target_path.name
    )

    created_at = manifest.get("created_at") or summary.get("created_at")
    step = int(summary.get("step") if summary.get("step") is not None else manifest.get("steps", 0))

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    total_steps = (
        summary.get("total_steps")
        or config.get("training", {}).get("steps")
        or manifest.get("total_steps")
    )

    latest_metrics = summary.get("latest") or {}
    best_metric = summary.get("best")
    checkpoints = summary.get("checkpoints") or collect_checkpoints(target_path)
    evaluations = summary.get("evaluations") or collect_evaluations(target_path)
    health_alerts = summary.get("health_alerts") or []
    telemetry_trends = summary.get("telemetry_trends") or {}

    tag_name = read_run_tag(target_path)
    name = tag_name or manifest.get("name") or summary.get("name") or config.get("name") or run_id
    slug = summary.get("slug") or manifest.get("slug") or str(run_id)[:8]
    short_id = summary.get("short_id") or str(run_id)[:8]
    pid = summary.get("pid") or manifest.get("pid")
    steps_per_sec = summary.get("steps_per_second") or summary.get("steps_per_sec") or manifest.get("steps_per_second")
    eta_seconds = summary.get("eta_seconds")

    return RunInfo(
        run_id=run_id,
        run_dir=target_path,
        created_at=created_at,
        status=status,
        step=step,
        total_steps=total_steps,
        config=config,
        latest_metrics=latest_metrics,
        best_metric=best_metric,
        checkpoints=checkpoints,
        health_alerts=health_alerts,
        telemetry_trends=telemetry_trends,
        name=name,
        slug=slug,
        short_id=short_id,
        pid=pid,
        steps_per_second=steps_per_sec,
        eta_seconds=eta_seconds,
        evaluations=evaluations,
    )

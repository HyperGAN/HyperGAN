import json
import os
from pathlib import Path
import time

import pytest
from starlette.testclient import TestClient

from hypergan import api, checkpoints, inspection, registry
from hypergan.event_views import MapSpec, Projector as EventProjector
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_autostart import (
    Projector,
    ensure_run_projected,
    is_projection_stale_or_missing,
    project_run,
)
from hypergan.web_server import (
    ServiceRegistry,
    create_app,
    read_run_summary,
)
from hypergan.web_session import LocalSession


def _make_run(
    root: Path,
    run_id: str,
    *,
    status: str = "stopped",
    step: int = 50,
    max_steps: int = 200,
    slug: str | None = None,
    project: bool = True,
    with_summary: bool = True,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    definition = {"kind": "scalar", "source": "g_loss", "label": "Generator loss"}
    definition["definition_hash"] = digest(definition)
    catalog = {"schema_version": 1, "metrics": {"loss/g_total": definition}}
    revision = digest(catalog)
    atomic_json(run_dir / "metrics" / f"catalog-{revision}.json", catalog)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "slug": slug or f"slug-{run_id}",
        "attempt_id": "att-1",
        "steps": step,
        "total_steps": max_steps,
        "max_steps": max_steps,
        "status": status,
        "metrics_catalog": revision,
    }
    atomic_json(run_dir / "manifest.json", manifest)

    start_row = {
        "schema_version": 2,
        "run_id": run_id,
        "attempt_id": "att-1",
        "stream_id": "training",
        "stream_generation": "run",
        "sequence": 1,
        "step": 0,
        "event": "start",
        "catalog": revision,
        "parent_attempt_id": None,
        "restored_step": 0,
    }
    train_row = {
        "schema_version": 2,
        "run_id": run_id,
        "attempt_id": "att-1",
        "stream_id": "training",
        "stream_generation": "run",
        "sequence": 2,
        "step": step,
        "event": "train",
        "catalog": revision,
        "metrics": {"loss/g_total": 0.75},
    }
    (run_dir / "events.jsonl").write_text(
        json.dumps(start_row) + "\n" + json.dumps(train_row) + "\n",
        encoding="utf-8",
    )

    if project:
        with EventProjector(run_dir) as proj:
            proj.project(limit=100)

    if with_summary:
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "slug": slug or f"slug-{run_id}",
            "name": f"Exp {run_id}",
            "status": status,
            "step": step,
            "total_steps": max_steps,
            "max_steps": max_steps,
            "speed_its": 12.5,
            "best_fid": 14.2,
            "updated_at": "2026-10-17T00:00:00Z",
        }
        atomic_json(run_dir / "summary.json", summary)

    return run_dir


def test_service_registry_lru_eviction(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for i in range(5):
        _make_run(runs_root, f"run_{i}", status="stopped")

    default_reg = ServiceRegistry(runs_root)
    assert default_reg.max_services == 16

    reg = ServiceRegistry(runs_root, max_services=3)
    s0 = reg.get_service("run_0")
    s1 = reg.get_service("run_1")
    s2 = reg.get_service("run_2")
    assert list(reg._services.keys()) == ["run_0", "run_1", "run_2"]

    # Touch run_0 so it becomes MRU; run_1 is now LRU
    assert reg.get_service("run_0") is s0
    assert list(reg._services.keys()) == ["run_1", "run_2", "run_0"]

    # Adding run_3 should evict run_1
    s3 = reg.get_service("run_3")
    assert len(reg._services) == 3
    assert "run_1" not in reg._services
    assert list(reg._services.keys()) == ["run_2", "run_0", "run_3"]

    # Adding run_4 should evict run_2
    s4 = reg.get_service("run_4")
    assert len(reg._services) == 3
    assert "run_2" not in reg._services
    assert list(reg._services.keys()) == ["run_0", "run_3", "run_4"]
    assert s1 is not s3 and s2 is not s4


def test_discover_runs_watermark_cache_and_registry_integration(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    r1_dir = _make_run(runs_root, "run_hist_1", status="stopped", step=100)
    r2_dir = _make_run(runs_root, "run_hist_2", status="complete", step=200)

    # Also create an external run outside runs_root and register it
    ext_root = tmp_path / "external_runs"
    ext_root.mkdir()
    ext_dir = _make_run(ext_root, "run_external", status="stopped", step=300)
    registry.register_run(ext_dir, status="stopped")

    svc_reg = ServiceRegistry(runs_root)

    read_calls: list[Path] = []
    orig_read_summary = read_run_summary

    def spy_read_summary(run_dir: Path) -> dict:
        read_calls.append(Path(run_dir).resolve())
        return orig_read_summary(run_dir)

    monkeypatch.setattr("hypergan.web_server.read_run_summary", spy_read_summary)

    # First discovery reads all 3 runs (2 in runs_root + 1 in global registry)
    first = svc_reg._discover_runs()
    assert {r["run_id"] for r in first} == {"run_hist_1", "run_hist_2", "run_external"}
    assert len(read_calls) == 3

    # Second discovery with unchanged summary.json/manifest.json mtimes does ZERO reads
    read_calls.clear()
    second = svc_reg._discover_runs()
    assert len(read_calls) == 0
    assert {r["run_id"] for r in second} == {"run_hist_1", "run_hist_2", "run_external"}

    # Modify summary.json on run_hist_1 (bumping step and mtime)
    summary_path = r1_dir / "summary.json"
    sdata = json.loads(summary_path.read_text(encoding="utf-8"))
    sdata["step"] = 150
    summary_path.write_text(json.dumps(sdata), encoding="utf-8")
    st = summary_path.stat()
    os.utime(summary_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    read_calls.clear()
    third = svc_reg._discover_runs()
    assert read_calls == [r1_dir.resolve()]
    r1_row = next(r for r in third if r["run_id"] == "run_hist_1")
    assert r1_row["step"] == 150

    # Verify deleted registered folders are skipped gracefully
    missing_dir = tmp_path / "deleted_run"
    monkeypatch.setattr(
        registry,
        "list_registered_runs",
        lambda backfill_cwd=False: [
            {"run_dir": str(ext_dir)},
            {"run_dir": str(missing_dir)},
        ],
    )
    fourth = svc_reg._discover_runs()
    assert {r["run_id"] for r in fourth} == {"run_hist_1", "run_hist_2", "run_external"}


def test_lazy_on_demand_projection_on_unprojected_and_stale_runs(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    unprojected_dir = _make_run(runs_root, "run_lazy", status="stopped", step=25, project=False)

    map_rev = MapSpec().revision
    contrib_file = unprojected_dir / "views" / map_rev / "contributions.jsonl"
    proj_file = unprojected_dir / "views" / map_rev / "projection.json"
    assert not contrib_file.exists()
    assert is_projection_stale_or_missing(unprojected_dir) is True

    svc_reg = ServiceRegistry(runs_root)
    svc = svc_reg.get_service("run_lazy")
    assert svc.root == unprojected_dir.resolve()
    assert contrib_file.is_file()
    assert proj_file.is_file()
    assert is_projection_stale_or_missing(unprojected_dir) is False

    # Append a new training event with newer mtime so projection becomes stale
    manifest = json.loads((unprojected_dir / "manifest.json").read_text(encoding="utf-8"))
    events_path = unprojected_dir / "events.jsonl"
    new_event = {
        "schema_version": 2,
        "run_id": "run_lazy",
        "attempt_id": "att-1",
        "stream_id": "training",
        "stream_generation": "run",
        "sequence": 3,
        "step": 30,
        "event": "train",
        "catalog": manifest["metrics_catalog"],
        "metrics": {"loss/g_total": 0.5},
    }
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(new_event) + "\n")
    proj_mtime = max(proj_file.stat().st_mtime_ns, contrib_file.stat().st_mtime_ns)
    os.utime(events_path, ns=(proj_mtime + 10_000_000, proj_mtime + 10_000_000))
    assert is_projection_stale_or_missing(unprojected_dir) is True

    # Requesting /api/runs/run_lazy/contributions lazily updates the projection
    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        res = client.get("/api/runs/run_lazy/contributions")
        assert res.status_code == 200
        items = res.json()["contributions"]
        assert len(items) >= 2
    assert is_projection_stale_or_missing(unprojected_dir) is False


def test_projector_skips_inactive_runs_with_unchanged_watermark(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    inactive_1 = _make_run(runs_root, "inactive_1", status="stopped", project=True)
    inactive_2 = _make_run(runs_root, "inactive_2", status="complete", project=True)
    active_run = _make_run(runs_root, "active_1", status="running", project=True)

    monkeypatch.setattr(
        inspection,
        "check_run_active",
        lambda run_dir: (Path(run_dir).resolve() == active_run.resolve(), 4321 if Path(run_dir).resolve() == active_run.resolve() else None),
    )

    projected_dirs: list[Path] = []
    orig_project_run = project_run

    def spy_project_run(run_dir: Path, *, limit: int = 100) -> int:
        projected_dirs.append(Path(run_dir).resolve())
        return orig_project_run(run_dir, limit=limit)

    monkeypatch.setattr("hypergan.web_autostart.project_run", spy_project_run)

    proj = Projector(runs_root, poll_interval=0.05)

    # Inactive runs are already projected (events.jsonl mtime <= state.json mtime),
    # so poll_once ONLY projects active_run!
    proj.poll_once()
    assert projected_dirs == [active_run.resolve()]

    projected_dirs.clear()
    proj.poll_once()
    assert projected_dirs == [active_run.resolve()]

    # Now update inactive_1's events.jsonl so its watermark changes
    manifest = json.loads((inactive_1 / "manifest.json").read_text(encoding="utf-8"))
    ev_path = inactive_1 / "events.jsonl"
    extra_ev = {
        "schema_version": 2,
        "run_id": "inactive_1",
        "attempt_id": "att-1",
        "stream_id": "training",
        "stream_generation": "run",
        "sequence": 3,
        "step": 60,
        "event": "train",
        "catalog": manifest["metrics_catalog"],
        "metrics": {"loss/g_total": 0.42},
    }
    with ev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(extra_ev) + "\n")
    now_ns = time.time_ns() + 50_000_000
    os.utime(ev_path, ns=(now_ns, now_ns))

    projected_dirs.clear()
    proj.poll_once()
    assert set(projected_dirs) == {active_run.resolve(), inactive_1.resolve()}

    # Next poll skips inactive_1 again because its new watermark is now recorded
    projected_dirs.clear()
    proj.poll_once()
    assert projected_dirs == [active_run.resolve()]


def test_api_runs_and_live_stop_checkpoint_endpoints_and_ui(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    active_dir = _make_run(runs_root, "run_live", status="running", step=80, max_steps=400, slug="amber-falcon")
    stopped_dir = _make_run(runs_root, "run_done", status="stopped", step=400, max_steps=400, slug="cobalt-otter")

    monkeypatch.setattr(
        inspection,
        "check_run_active",
        lambda run_dir: (Path(run_dir).resolve() == active_dir.resolve(), 9999 if Path(run_dir).resolve() == active_dir.resolve() else None),
    )

    stopped_calls: list[Path] = []
    ckpt_calls: list[Path] = []

    def fake_stop_run(target_dir, **kwargs):
        stopped_calls.append(Path(target_dir).resolve())
        return {"status": "stopped", "pid": 9999}

    def fake_request_checkpoint(target_dir):
        ckpt_calls.append(Path(target_dir).resolve())
        return {"request_id": "req-1", "status": "pending"}

    monkeypatch.setattr(inspection, "stop_run", fake_stop_run)
    monkeypatch.setattr(api, "request_checkpoint", fake_request_checkpoint)
    monkeypatch.setattr(checkpoints, "request_checkpoint", fake_request_checkpoint, raising=False)

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        # 1. GET /api/runs
        res = client.get("/api/runs")
        assert res.status_code == 200
        runs = res.json()
        by_id = {r["run_id"]: r for r in runs}
        assert "run_live" in by_id and "run_done" in by_id
        live_row = by_id["run_live"]
        for key in ("run_id", "slug", "name", "status", "active", "pid", "step", "max_steps", "best_fid", "updated_at"):
            assert key in live_row
        assert live_row["slug"] == "amber-falcon"
        assert live_row["active"] is True
        assert live_row["pid"] == 9999
        assert by_id["run_done"]["active"] is False

        # 2. POST /api/runs/<id>/checkpoint on active run
        ckpt_res = client.post("/api/runs/run_live/checkpoint")
        assert ckpt_res.status_code == 200
        ckpt_data = ckpt_res.json()
        assert ckpt_data["ok"] is True
        assert ckpt_data["action"] == "checkpoint"
        assert ckpt_calls == [active_dir.resolve()]

        # 3. POST /api/runs/<id>/stop on active run
        stop_res = client.post("/api/runs/run_live/stop")
        assert stop_res.status_code == 200
        stop_data = stop_res.json()
        assert stop_data["ok"] is True
        assert stop_data["action"] == "stop"
        assert stopped_calls == [active_dir.resolve()]

        # 4. POST /api/runs/<id>/stop & /checkpoint on inactive run -> 400
        bad_stop = client.post("/api/runs/run_done/stop")
        assert bad_stop.status_code == 400
        assert bad_stop.json()["ok"] is False

        bad_ckpt = client.post("/api/runs/run_done/checkpoint")
        assert bad_ckpt.status_code == 400
        assert bad_ckpt.json()["ok"] is False

        # 5. GET / HTML index contains Slug/ID badges, Stop/Checkpoint buttons, and 3s auto-refresh
        index_res = client.get("/")
        assert index_res.status_code == 200
        html_text = index_res.text
        assert "amber-falcon" in html_text
        assert "cobalt-otter" in html_text
        assert "/api/runs/run_live/checkpoint" in html_text
        assert "/api/runs/run_live/stop" in html_text
        assert "/api/runs/run_done/stop" not in html_text
        assert "fetch('/api/runs'" in html_text
        assert "3000" in html_text

        # 6. GET /runs/<id> contains '← All Runs' link back to '/'
        detail_res = client.get("/runs/run_live")
        assert detail_res.status_code == 200
        assert "← All Runs" in detail_res.text
        assert 'href="/"' in detail_res.text

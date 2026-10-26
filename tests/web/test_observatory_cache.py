import json
import os
from pathlib import Path
import time

import pytest
from starlette.testclient import TestClient

from hypergan import api, inspection, registry
from hypergan.event_views import MapSpec, Projector as EventProjector
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_autostart import (
    Projector,
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
        "stream_generation": run_id,
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
        "stream_generation": run_id,
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

    # A registered run nested deeper under the root is served; one registered
    # elsewhere on the machine belongs to another root and is never listed.
    nested_dir = _make_run(runs_root / "group" / "inner", "run_nested", status="stopped", step=50)
    registry.register_run(nested_dir, status="stopped")
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

    first = svc_reg._discover_runs()
    assert {r["run_id"] for r in first} == {"run_hist_1", "run_hist_2", "run_nested"}
    assert len(read_calls) == 3
    assert svc_reg.resolve_run_dir("run_external") is None
    assert svc_reg.resolve_run_dir("run_nested") == nested_dir.resolve()

    # Second discovery with unchanged summary.json/manifest.json mtimes does ZERO reads
    read_calls.clear()
    second = svc_reg._discover_runs()
    assert len(read_calls) == 0
    assert {r["run_id"] for r in second} == {"run_hist_1", "run_hist_2", "run_nested"}

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

    # Deleted registered folders are skipped gracefully
    import shutil
    shutil.rmtree(nested_dir)
    fourth = svc_reg._discover_runs()
    assert {r["run_id"] for r in fourth} == {"run_hist_1", "run_hist_2"}


def test_requests_never_project_and_contributions_are_paged(tmp_path, monkeypatch):
    """Projection belongs to the viewer's projector: no request blocks the event loop on it."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    unprojected_dir = _make_run(runs_root, "run_lazy", status="stopped", step=25, project=False)

    map_rev = MapSpec().revision
    contrib_file = unprojected_dir / "views" / map_rev / "contributions.jsonl"
    assert is_projection_stale_or_missing(unprojected_dir) is True

    import hypergan.web_autostart as autostart
    original_project_run = autostart.project_run
    monkeypatch.setattr(autostart, "project_run", lambda *a, **k: pytest.fail("a request projected"))
    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        assert client.get("/").status_code == 200
        assert client.get("/runs/run_lazy").status_code == 200
        assert client.get("/api/runs/run_lazy").status_code == 200
        # Missing projection: the history waits for the projector instead of failing.
        bootstrap = client.get(f"/api/v1/runs/run_lazy/views/{map_rev}/bootstrap?series=loss/g_total")
        assert bootstrap.status_code == 202
        assert not contrib_file.exists()

        monkeypatch.setattr(autostart, "project_run", original_project_run)
        assert Projector(runs_root).poll_once() == [unprojected_dir.resolve()]
        assert contrib_file.is_file()

        first = client.get("/api/runs/run_lazy/contributions", params={"limit": 1}).json()
        assert len(first["contributions"]) == 1 and first["has_more"] is True
        rest = client.get("/api/runs/run_lazy/contributions", params={"cursor": first["cursor"]}).json()
        assert rest["contributions"] and rest["has_more"] is False
        assert client.get("/api/runs/run_lazy/contributions", params={"limit": 5000}).status_code == 400


def _bootstrap_until_done(client, run_id, map_rev, *, attempts=500):
    url = f"/api/v1/runs/{run_id}/views/{map_rev}/bootstrap?series=loss/g_total"
    for _ in range(attempts):
        response = client.get(url)
        if response.status_code != 202:
            return response
        time.sleep(.01)
    return response


def test_nested_registered_run_is_projected_by_the_viewers_projector(tmp_path):
    """Every run the dashboard serves has a projector, since requests never project."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    direct = _make_run(runs_root, "direct", project=False)
    nested = _make_run(runs_root / "group" / "inner", "run_nested", project=False)
    registry.register_run(nested, status="stopped")

    projector = Projector(runs_root)
    assert nested.resolve() in projector._candidate_runs()
    map_rev = MapSpec().revision
    session = LocalSession(8765)
    with TestClient(create_app(runs_root, session=session), base_url=session.origin) as client:
        assert set(projector.poll_once()) == {direct.resolve(), nested.resolve()}
        for run_id in ("direct", "run_nested"):
            response = _bootstrap_until_done(client, run_id, map_rev)
            assert response.status_code == 200, (run_id, response.text)


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs a directory mode that denies writes (not Windows, not root)")
def test_projection_failure_in_another_process_reaches_the_server(tmp_path):
    """The broker's projector and server are separate processes; the failure still shows."""
    import subprocess
    import sys
    import hypergan.web_autostart as autostart
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    read_only = _make_run(runs_root, "read_only", project=False)
    read_only.chmod(0o555)

    # The projector runs in its own interpreter, as in the train-launched broker.
    code = "import sys; from hypergan.web_autostart import Projector; print(Projector(sys.argv[1]).poll_once())"
    try:
        done = subprocess.run([sys.executable, "-c", code, str(runs_root)],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert read_only.resolve() not in autostart._PROJECTION_FAILURES
        assert "PermissionError" in (autostart.projection_failure(read_only) or "")

        map_rev = MapSpec().revision
        session = LocalSession(8765)
        with TestClient(create_app(runs_root, session=session), base_url=session.origin) as client:
            response = _bootstrap_until_done(client, "read_only", map_rev)
            assert response.status_code == 404, response.text
            assert "could not be built: PermissionError" in response.text
    finally:
        read_only.chmod(0o755)

    # Once a projection succeeds, the projector clears the record for every process.
    done = subprocess.run([sys.executable, "-c", code, str(runs_root)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert autostart.projection_failure(read_only) is None


def _append_train_event(run_dir, sequence, step):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    row = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "attempt_id": "att-1",
        "stream_id": "training",
        "stream_generation": run_dir.name,
        "sequence": sequence,
        "step": step,
        "event": "train",
        "catalog": manifest["metrics_catalog"],
        "metrics": {"loss/g_total": 0.42},
    }
    ev_path = run_dir / "events.jsonl"
    with ev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    now_ns = time.time_ns() + 50_000_000
    os.utime(ev_path, ns=(now_ns, now_ns))


def test_projector_projects_only_runs_whose_events_changed(tmp_path, monkeypatch):
    """An idle dashboard opens no projector, even for a run that is still training."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    inactive_1 = _make_run(runs_root, "inactive_1", status="stopped", project=True)
    _make_run(runs_root, "inactive_2", status="complete", project=True)
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

    # Every projection is current, so polling does nothing, active run included.
    proj.poll_once()
    proj.poll_once()
    assert projected_dirs == []

    # New events on the active run, then on an inactive one, are each projected once.
    _append_train_event(active_run, 3, 60)
    proj.poll_once()
    assert projected_dirs == [active_run.resolve()]

    _append_train_event(inactive_1, 3, 60)
    projected_dirs.clear()
    proj.poll_once()
    assert projected_dirs == [inactive_1.resolve()]

    projected_dirs.clear()
    proj.poll_once()
    assert projected_dirs == []


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

    def fake_request_checkpoint(target_dir, **kwargs):
        ckpt_calls.append(Path(target_dir).resolve())
        return {"request_id": "req-1", "status": "pending"}

    monkeypatch.setattr(inspection, "stop_run", fake_stop_run)
    monkeypatch.setattr(api, "request_checkpoint", fake_request_checkpoint)

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

        # 5. The index exposes controls through a CSP-compatible external script.
        index_res = client.get("/")
        assert index_res.status_code == 200
        html_text = index_res.text
        assert "amber-falcon" in html_text
        assert "cobalt-otter" in html_text
        assert 'data-action="checkpoint"' in html_text
        assert 'data-action="stop"' in html_text
        assert '<script src="/assets/runs.js" defer></script>' in html_text
        assert client.get('/assets/runs.js').status_code == 200

        # 6. GET /runs/<id> contains '← All Runs' link back to '/'
        detail_res = client.get("/runs/run_live")
        assert detail_res.status_code == 200
        assert "← All Runs" in detail_res.text
        assert 'href="/"' in detail_res.text

import json
from pathlib import Path
import time
import pytest
from starlette.testclient import TestClient

from hypergan.event_views import MapSpec, Projector
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_server import (
    create_app,
    get_run_disk_bytes,
    read_run_summary,
    _disk_cache,
)
from hypergan.web_session import LocalSession
from hypergan import __version__


def create_mock_run(runs_root: Path, run_id: str, *, status: str = "running",
                    step: int = 100, speed: float | None = 15.5,
                    fid_latest: float | None = 22.4, fid_best: float | None = 18.2,
                    with_summary: bool = True):
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    definition = {'kind': 'scalar', 'source': 'g_loss', 'label': 'Generator loss'}
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {'loss/g_total': definition}}
    revision = digest(catalog)
    atomic_json(run_dir / 'metrics' / f'catalog-{revision}.json', catalog)

    manifest = {
        'schema_version': 1,
        'run_id': run_id,
        'attempt_id': 'att-1',
        'steps': step,
        'target_steps': step * 2,
        'status': status,
        'metrics_catalog': revision,
    }
    atomic_json(run_dir / 'manifest.json', manifest)

    # Add a minimal event row so projector/observation service can initialize
    row = {
        'schema_version': 2,
        'run_id': run_id,
        'attempt_id': 'att-1',
        'stream_id': 'training',
        'stream_generation': 'run',
        'sequence': 1,
        'step': 0,
        'event': 'start',
        'catalog': revision,
        'parent_attempt_id': None,
        'restored_step': 0,
    }
    (run_dir / 'events.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')

    with Projector(run_dir) as projector:
        projector.project(limit=10)

    if with_summary:
        summary = {
            'schema_version': 1,
            'run_id': run_id,
            'status': status,
            'step': step,
            'target_steps': step * 2,
            'speed_its': speed,
            'fid_latest': fid_latest,
            'fid_best': fid_best,
            'updated_at': time.time(),
        }
        atomic_json(run_dir / 'summary.json', summary)

    return run_dir


def test_root_index_renders_html_table(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    create_mock_run(runs_root, "run_alpha", status="running", step=120, speed=14.2, fid_latest=19.5, fid_best=17.1)
    create_mock_run(runs_root, "run_beta", status="complete", step=250, with_summary=False)

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert res.headers["cache-control"] == "no-store"
        assert "etag" not in res.headers
        assert "last-modified" not in res.headers

        html = res.text
        assert "HyperGAN Runs" in html
        assert "run_alpha" in html
        assert "run_beta" in html
        assert "/runs/run_alpha" in html
        assert "/runs/run_beta" in html
        assert "14.2 it/s" in html
        assert "19.50" in html
        assert "17.10" in html
        assert "running" in html
        assert "complete" in html


def test_runs_route_serves_frontend_spa(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    create_mock_run(runs_root, "run_exp1")

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        res = client.get("/runs/run_exp1")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert res.headers["cache-control"] == "no-store"
        assert "etag" not in res.headers
        assert "last-modified" not in res.headers
        assert "<div id=" in res.text or "<!doctype html>" in res.text.lower()


def test_api_v1_runs_json_index(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    create_mock_run(runs_root, "run_a", status="running", step=80, speed=10.0)
    create_mock_run(runs_root, "run_b", status="stopped", step=200, with_summary=False)

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        res = client.get("/api/v1/runs")
        assert res.status_code == 200
        assert res.headers["cache-control"] == "no-store"
        assert "etag" not in res.headers

        runs = res.json()
        assert isinstance(runs, list)
        assert len(runs) == 2
        run_map = {r["run_id"]: r for r in runs}
        assert "run_a" in run_map
        assert "run_b" in run_map
        assert run_map["run_a"]["status"] == "running"
        assert run_map["run_a"]["speed_its"] == 10.0
        assert run_map["run_b"]["status"] == "stopped"
        assert run_map["run_b"]["step"] == 200
        assert "disk_bytes" in run_map["run_a"]
        assert run_map["run_a"]["disk_bytes"] > 0


def test_api_v1_runs_dynamic_dispatch_and_404(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    create_mock_run(runs_root, "run_active")

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        # Dynamic dispatch to known run
        res = client.get("/api/v1/runs/run_active/status")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == "run_active"
        assert data["status"] == "running"

        # 404 on unknown run
        res404 = client.get("/api/v1/runs/nonexistent_run/status")
        assert res404.status_code == 404
        assert res404.headers["cache-control"] == "no-store"
        assert "Unknown run identity: nonexistent_run" in res404.json()["error"]

        res404_points = client.get("/api/v1/runs/nonexistent_run/events")
        assert res404_points.status_code == 404
        assert "Unknown run identity: nonexistent_run" in res404_points.json()["error"]


def test_capabilities_enriched(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    create_mock_run(runs_root, "run_1")
    create_mock_run(runs_root, "run_2")

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        res = client.get("/api/v1/capabilities")
        assert res.status_code == 200
        cap = res.json()
        assert cap["version"] == __version__
        assert "git_commit" in cap
        assert "asset_hash" in cap
        assert cap["asset_hash"] is not None
        assert cap["runs_root"] == str(runs_root.resolve())
        assert cap["run_count"] == 2
        assert "sse" in cap["transports"]
        assert "limits" in cap


def test_disk_usage_calculation_and_ttl_cache(tmp_path):
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    file_a = run_dir / "file_a.bin"
    file_a.write_bytes(b"x" * 1024)
    file_b = run_dir / "subdir" / "file_b.bin"
    file_b.parent.mkdir()
    file_b.write_bytes(b"y" * 2048)

    _disk_cache.clear()
    bytes1 = get_run_disk_bytes(run_dir)
    assert bytes1 == 3072

    # Add a file, but verify cache still returns bytes1 within 20s TTL
    file_c = run_dir / "file_c.bin"
    file_c.write_bytes(b"z" * 4096)
    bytes2 = get_run_disk_bytes(run_dir)
    assert bytes2 == 3072

    # Expire cache entry manually and verify updated bytes
    _disk_cache[run_dir] = (time.monotonic() - 25.0, bytes1)
    bytes3 = get_run_disk_bytes(run_dir)
    assert bytes3 == 7168


def test_read_run_summary_atomic_fallback(tmp_path):
    run_dir = tmp_path / "fallback_run"
    run_dir.mkdir()
    manifest = {
        'schema_version': 1,
        'run_id': 'fallback_run',
        'status': 'stopped',
        'steps': 42,
    }
    atomic_json(run_dir / 'manifest.json', manifest)

    # 1. No summary.json exists: falls back to manifest
    summary = read_run_summary(run_dir)
    assert summary is not None
    assert summary['run_id'] == 'fallback_run'
    assert summary['status'] == 'stopped'
    assert summary['step'] == 42
    assert summary.get('speed_its') is None

    # 2. Corrupted summary.json exists: gracefully falls back to manifest
    (run_dir / 'summary.json').write_text('{incomplete json: ', encoding='utf-8')
    summary_corrupt = read_run_summary(run_dir)
    assert summary_corrupt is not None
    assert summary_corrupt['run_id'] == 'fallback_run'
    assert summary_corrupt['step'] == 42

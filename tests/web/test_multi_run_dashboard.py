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


def create_mock_run(runs_root: Path, run_id: str, *, dir_name: str | None = None,
                    status: str = "running", step: int = 100, speed: float | None = 15.5,
                    fid_latest: float | None = 22.4, fid_best: float | None = 18.2,
                    with_summary: bool = True):
    run_dir = runs_root / (dir_name or run_id)
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
        'created_at': '2026-10-17T12:30:00Z',
    }
    atomic_json(run_dir / 'manifest.json', manifest)

    # Add a minimal event row so projector/observation service can initialize
    row = {
        'schema_version': 2,
        'run_id': run_id,
        'attempt_id': 'att-1',
        'stream_id': 'training',
        'stream_generation': run_id,
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
    create_mock_run(runs_root, "run_beta", status="complete", step=250, speed=19.8, with_summary=True)

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
        assert "WORKSPACE OBSERVATORY" in html
        assert '<header class="masthead">' in html
        assert "EXPERIMENTS" in html
        assert "run_alpha" in html
        assert "run_beta" in html
        assert "/runs/run_alpha" in html
        assert "/runs/run_beta" in html
        assert 'class="run-row"' in html
        assert 'data-url="/runs/run_alpha"' in html
        assert 'data-url="/runs/run_beta"' in html
        assert "14.2 steps/s" in html
        assert "19.8" not in html
        assert "it/s" not in html
        assert "Topology & Batch" in html
        assert "Created at" in html
        assert "2026-10-17T12:30:00+00:00" in html
        assert "FID" in html
        assert "Preview" in html
        assert "120 / 240" in html
        assert 'data-action=' not in html
        assert 'Actions</th>' not in html
        assert '<script src="/assets/runs.js" defer></script>' in html
        assert 'onchange=' not in html
        assert 'onclick=' not in html
        # Assets must be served through the same policy that blocked inline code.
        assert "'unsafe-inline'" not in res.headers['content-security-policy'].split('script-src')[1].split(';')[0]
        assert client.get('/assets/runs.js').status_code == 200
        assert client.get('/assets/runs.css').status_code == 200
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
        assert run_map["run_a"]["created_at"] == "2026-10-17T12:30:00+00:00"
        assert run_map["run_a"]["total_steps"] == 160
        # FID remains available to API clients even though it is absent from the index.
        assert run_map["run_a"]["latest_fid"] == 22.4
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


def test_render_run_index_step_formatting_and_escaping(tmp_path):
    from hypergan.web_server import render_run_index_html
    runs = [
        {"run_id": 'r1"', "name": '<script>alert("test")</script>', "status": "running", "step": 120, "total_steps": 500,
         "best_fid": 15.2, "best_step": 120},
        {"run_id": "r2", "name": "Run 2", "status": "running", "step": 50000, "total_steps": 100000,
         "best_fid": 12.4, "best_step": 50000},
    ]
    html_out = render_run_index_html(runs, tmp_path)
    assert "120 / 500" in html_out
    assert "50,000 / 100,000" in html_out
    assert "FID" in html_out
    assert "15.20" in html_out
    assert "12.40" in html_out
    assert '<script>alert' not in html_out
    assert '&lt;script&gt;' in html_out
    assert '/runs/r1%22' in html_out


@pytest.mark.parametrize(('value', 'expected'), [
    ('2026-10-17T12:30:00Z', '2026-10-17T12:30:00+00:00'),
    ('2026-10-17T06:30:00-06:00', '2026-10-17T12:30:00+00:00'),
    (0, '1970-01-01T00:00:00+00:00'),
    ('invalid', None),
    (True, None),
])
def test_run_creation_timestamp(tmp_path, value, expected):
    atomic_json(tmp_path / 'manifest.json', {'run_id': 'created', 'created_at': value})
    assert read_run_summary(tmp_path)['created_at'] == expected


def test_creation_time_uses_first_launch_and_survives_progress_and_resume(tmp_path):
    atomic_json(tmp_path / 'manifest.json', {'run_id': 'created', 'status': 'stopped'})
    atomic_json(tmp_path / 'launch' / '0001.json', {'timestamp': '2026-10-15T01:00:00Z'})
    atomic_json(tmp_path / 'launch' / '0002.json', {'timestamp': '2026-10-17T02:00:00Z'})
    assert read_run_summary(tmp_path)['created_at'] == '2026-10-15T01:00:00+00:00'
    atomic_json(tmp_path / 'summary.json', {'step': 100, 'updated_at': time.time()})
    assert read_run_summary(tmp_path)['created_at'] == '2026-10-15T01:00:00+00:00'
    # Recorded manifest data takes precedence over summary and launch data.
    atomic_json(tmp_path / 'manifest.json', {'run_id': 'created', 'created_at': '2026-10-14T00:00:00Z'})
    atomic_json(tmp_path / 'summary.json', {'created_at': '2026-10-16T00:00:00Z'})
    assert read_run_summary(tmp_path)['created_at'] == '2026-10-14T00:00:00+00:00'


def test_creation_time_handles_missing_or_corrupt_provenance(tmp_path):
    atomic_json(tmp_path / 'manifest.json', {'run_id': 'created'})
    assert read_run_summary(tmp_path)['created_at'] is None
    launch = tmp_path / 'launch' / '0001.json'
    launch.parent.mkdir()
    launch.write_text('{incomplete', encoding='utf-8')
    assert read_run_summary(tmp_path)['created_at'] is None
    atomic_json(tmp_path / 'summary.json', {'created_at': '2026-10-16T00:00:00Z'})
    assert read_run_summary(tmp_path)['created_at'] == '2026-10-16T00:00:00+00:00'


@pytest.mark.parametrize('name', ['runs.js', 'runs.css'])
def test_dashboard_edits_change_server_compatibility_hash(tmp_path, name):
    from importlib.resources import files
    from hypergan.web_assets import CORE_ASSETS, compute_asset_hash

    for asset in CORE_ASSETS:
        (tmp_path / asset).write_bytes(files('hypergan.web_assets').joinpath(asset).read_bytes())
    before = compute_asset_hash(tmp_path)
    (tmp_path / name).write_text('/* dashboard edit */', encoding='utf-8')
    assert compute_asset_hash(tmp_path) != before


def test_service_registry_path_traversal_guard(tmp_path):
    from hypergan.web_server import ServiceRegistry
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    create_mock_run(runs_root, "legit_run")

    registry = ServiceRegistry(runs_root)
    # Direct candidate traversal outside runs_root must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Unknown run identity"):
        registry.resolve_run_dir("../../etc/passwd")
    with pytest.raises(FileNotFoundError, match="Unknown run identity"):
        registry.resolve_run_dir("../sibling_dir")

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        # Unknown run returns 404 JSON with no-store
        res = client.get("/api/v1/runs/nonexistent_run_id/status")
        assert res.status_code == 404
        assert res.headers["cache-control"] == "no-store"
        assert "Unknown run identity: nonexistent_run_id" in res.json().get("error", "")


def test_replaced_run_directory_evicts_stale_service_and_resolves_new_run_id(tmp_path):
    import shutil
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    dir_name = "imajev-apples-dcgan-hidden-bispf"
    old_run_id = "6253a54d32684da1862a58edb9697d02"
    new_run_id = "3261e90023c046ef9301427adc55f9b3"

    run_dir = create_mock_run(runs_root, old_run_id, dir_name=dir_name, status="stopped", step=50)

    session = LocalSession(8765, auth="none")
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        # 1. Populate registry cache under old_run_id
        res_old = client.get(f"/api/v1/runs/{old_run_id}/status")
        assert res_old.status_code == 200
        assert res_old.json()["run_id"] == old_run_id
        assert res_old.json()["steps"] == 50

        res_old_views = client.get(f"/api/v1/runs/{old_run_id}/views")
        assert res_old_views.status_code == 200
        assert res_old_views.json()["status"] == "available"

        # 2. Remove and recreate the same run directory with a new run_id
        shutil.rmtree(run_dir)
        create_mock_run(runs_root, new_run_id, dir_name=dir_name, status="running", step=200)

        # 3. Query by new_run_id: must return new_run_id (not stale old_run_id!)
        res_new = client.get(f"/api/v1/runs/{new_run_id}/status")
        assert res_new.status_code == 200
        assert res_new.json()["run_id"] == new_run_id
        assert res_new.json()["steps"] == 200

        res_new_views = client.get(f"/api/v1/runs/{new_run_id}/views")
        assert res_new_views.status_code == 200
        views_payload = res_new_views.json()
        assert views_payload["status"] == "available"
        assert views_payload["discovery_error"] is None
        for stream_info in views_payload["streams"]:
            assert stream_info["error"] is None

        res_new_events = client.get(f"/api/v1/runs/{new_run_id}/events")
        assert res_new_events.status_code == 200
        assert res_new_events.json()["events"][0]["run_id"] == new_run_id

        # 4. Query by directory name alias: must also return new_run_id
        res_alias = client.get(f"/api/v1/runs/{dir_name}/status")
        assert res_alias.status_code == 200
        assert res_alias.json()["run_id"] == new_run_id

        # 5. Query by old_run_id: must return 404
        res_stale = client.get(f"/api/v1/runs/{old_run_id}/status")
        assert res_stale.status_code == 404
        assert f"Unknown run identity: {old_run_id}" in res_stale.json()["error"]


def test_workspace_runs_subdir_discovery_and_projection(tmp_path):
    from hypergan.web_autostart import Projector as BgProjector, _manifest_roots
    from hypergan.web_server import ServiceRegistry

    workspace = tmp_path / "training-runs"
    workspace.mkdir()
    (workspace / "hypergan.toml").write_text('[workspace]\nruns_root = "runs"\n', encoding="utf-8")
    nested_runs = workspace / "runs"
    nested_runs.mkdir()

    # Create one top-level run and one nested run inside <workspace>/runs/
    create_mock_run(
        workspace,
        "top_run_01",
        dir_name="top-run",
        status="complete",
        step=100,
    )
    nested_dir = create_mock_run(
        nested_runs,
        "nested_run_02",
        dir_name="imajev-apples-joint-64-bispf-600",
        status="running",
        step=50,
    )

    registry = ServiceRegistry(workspace)
    assert registry.is_single() is False

    discovered = registry._discover_runs()
    run_ids = [r["run_id"] for r in discovered]
    assert run_ids == ["nested_run_02", "top_run_01"]

    # Resolve by run_id and by directory name inside <workspace>/runs/
    assert registry.resolve_run_dir("nested_run_02") == nested_dir.resolve()
    assert registry.resolve_run_dir("imajev-apples-joint-64-bispf-600") == nested_dir.resolve()

    # Projector and _manifest_roots also include <workspace>/runs/
    projector = BgProjector(workspace)
    cand_paths = projector._candidate_runs()
    assert nested_dir.resolve() in cand_paths
    assert nested_dir.resolve() in {p.resolve() for p in _manifest_roots(workspace)}

    session = LocalSession(8765, auth="none")
    app = create_app(workspace, session=session)
    with TestClient(app, base_url=session.origin) as client:
        res_runs = client.get("/api/runs")
        assert res_runs.status_code == 200
        api_ids = [r["run_id"] for r in res_runs.json()]
        assert api_ids == ["nested_run_02", "top_run_01"]

        res_detail = client.get("/runs/nested_run_02")
        assert res_detail.status_code == 200

        res_status = client.get("/api/v1/runs/nested_run_02/status")
        assert res_status.status_code == 200
        assert res_status.json()["run_id"] == "nested_run_02"


def test_crashed_unlocked_run_lock_resolves_to_stopped_in_dashboard(tmp_path):
    import os
    from hypergan.inspection import check_run_active

    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Run with unlocked .run.lock file and raw status='running' (crashed run)
    crashed_dir = create_mock_run(
        runs_root,
        "crashed_run_01",
        dir_name="crashed-run",
        status="running",
        step=42,
    )
    lock_file = crashed_dir / ".run.lock"
    lock_file.write_text(json.dumps({"pid": 999999, "hostname": "old-hostname"}), encoding="utf-8")
    stale_time = time.time() - 7200
    for f in (lock_file, crashed_dir / "summary.json", crashed_dir / "manifest.json", crashed_dir / "events.jsonl"):
        if f.exists():
            os.utime(f, (stale_time, stale_time))

    is_active, pid = check_run_active(crashed_dir)
    assert is_active is False
    assert pid is None

    summary = read_run_summary(crashed_dir)
    assert summary["status"] == "stopped"
    assert summary["active"] is False
    assert summary["running"] is False


class TestThumbnailEndpointHardening:
    """Tests for thumbnail_endpoint path containment, fallback, and inspect_png validation."""

    def test_thumbnail_success(self, tmp_path):
        from hypergan.image_grids import encode_png
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        run_dir = create_mock_run(runs_root, "run_thumb_ok")
        previews_dir = run_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        img_bytes = encode_png(b"\xff\x00\x00", 1, 1, 3)
        img_file = previews_dir / "preview_001.png"
        img_file.write_bytes(img_bytes)

        idx = {
            "previews": [
                {"image_grid": {"path": "previews/preview_001.png"}}
            ]
        }
        (previews_dir / "index.json").write_text(json.dumps(idx), encoding="utf-8")

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_thumb_ok/thumbnail")
            assert res.status_code == 200
            assert res.headers["content-type"] == "image/png"
            assert res.content == img_bytes

    def test_thumbnail_fallback_png(self, tmp_path):
        from hypergan.image_grids import encode_png
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        run_dir = create_mock_run(runs_root, "run_fallback")
        previews_dir = run_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        img_bytes = encode_png(b"\x00\xff\x00", 1, 1, 3)
        img_file = previews_dir / "preview_standalone.png"
        img_file.write_bytes(img_bytes)

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_fallback/thumbnail")
            assert res.status_code == 200
            assert res.headers["content-type"] == "image/png"
            assert res.content == img_bytes

    def test_thumbnail_missing(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        create_mock_run(runs_root, "run_empty")

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_empty/thumbnail")
            assert res.status_code == 404
            assert "No preview thumbnail available" in res.json().get("error", "")

    def test_thumbnail_path_traversal_dotdot_blocked(self, tmp_path):
        from hypergan.image_grids import encode_png
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        run_dir = create_mock_run(runs_root, "run_escape")
        secret_bytes = encode_png(b"\x00\x00\xff", 1, 1, 3)
        external_file = runs_root / "secret.png"
        external_file.write_bytes(secret_bytes)

        previews_dir = run_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        idx = {
            "previews": [
                {"image_grid": {"path": "../secret.png"}}
            ]
        }
        (previews_dir / "index.json").write_text(json.dumps(idx), encoding="utf-8")

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_escape/thumbnail")
            assert res.status_code == 404

    def test_thumbnail_absolute_path_blocked(self, tmp_path):
        from hypergan.image_grids import encode_png
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        run_dir = create_mock_run(runs_root, "run_abs")
        external_file = tmp_path / "external.png"
        external_file.write_bytes(encode_png(b"\x11\x22\x33", 1, 1, 3))

        previews_dir = run_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        idx = {
            "previews": [
                {"image_grid": {"path": str(external_file.resolve())}}
            ]
        }
        (previews_dir / "index.json").write_text(json.dumps(idx), encoding="utf-8")

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_abs/thumbnail")
            assert res.status_code == 404

    def test_thumbnail_symlink_escape_blocked(self, tmp_path):
        from hypergan.image_grids import encode_png
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        run_dir = create_mock_run(runs_root, "run_symlink")
        external_target = tmp_path / "outside_target.png"
        external_target.write_bytes(encode_png(b"\x44\x55\x66", 1, 1, 3))

        previews_dir = run_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        link = previews_dir / "symlink_escape.png"
        link.symlink_to(external_target)

        idx = {
            "previews": [
                {"image_grid": {"path": "previews/symlink_escape.png"}}
            ]
        }
        (previews_dir / "index.json").write_text(json.dumps(idx), encoding="utf-8")

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_symlink/thumbnail")
            assert res.status_code == 404

    def test_thumbnail_non_png_blocked(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        run_dir = create_mock_run(runs_root, "run_non_png")
        previews_dir = run_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        idx = {
            "previews": [
                {"image_grid": {"path": "manifest.json"}}
            ]
        }
        (previews_dir / "index.json").write_text(json.dumps(idx), encoding="utf-8")

        session = LocalSession(8765)
        app = create_app(runs_root, session=session)
        with TestClient(app, base_url=session.origin) as client:
            res = client.get("/api/v1/runs/run_non_png/thumbnail")
            assert res.status_code == 404


def test_index_active_vs_inactive_speed_eta_and_topology_rendering(tmp_path):
    import re
    from hypergan.web_server import render_run_index_html

    runs = [
        {
            "run_id": "run_2gpu_live",
            "name": "TransGAN CIFAR-10",
            "slug": "golden-lynx",
            "status": "running",
            "active": True,
            "running": True,
            "remote": True,
            "hostname": "gpu-node-02",
            "topology": {
                "device_kind": "cuda",
                "world_size": 2,
                "nodes": 1,
                "ranks_per_node": 2,
                "gpu_count": 2,
                "gpu_name": "RTX A6000",
                "label": "2 GPUs",
                "detail": "2× RTX A6000",
            },
            "batch_label": "Batch 256",
            "batch_detail": "128 / GPU",
            "step": 120,
            "total_steps": 10400,
            "steps_per_second": 14.2,
            "samples_per_second": 3635.2,
            "eta_seconds": 725.0,
            "best_fid": 17.1,
            "latest_fid": 19.5,
            "disk_bytes": 104857600,
            "disk_formatted": "100.0 MB",
            "created_at": "2026-10-18T10:00:00Z",
        },
        {
            "run_id": "run_stopped_stale_speed",
            "name": "Stopped Run",
            "slug": "silent-owl",
            "status": "stopped",
            "active": False,
            "running": False,
            "topology": {"label": "CPU", "detail": None},
            "batch_label": "Batch 16",
            "batch_detail": None,
            "step": 400,
            "total_steps": 1000,
            "steps_per_second": 21.6,
            "samples_per_second": 345.6,
            "eta_seconds": 27.7,
            "best_fid": None,
            "latest_fid": None,
            "disk_bytes": 4096,
            "disk_formatted": "4.0 KB",
            "created_at": "2026-10-17T10:00:00Z",
        },
        {
            "run_id": "run_unrecorded",
            "name": "Bare Run",
            "slug": "plain-fox",
            "status": "complete",
            "active": False,
            "running": False,
            "topology": {"label": None, "detail": None},
            "batch_label": None,
            "batch_detail": None,
            "step": 500,
            "total_steps": 500,
            "steps_per_second": 33.9,
            "created_at": "2026-10-16T10:00:00Z",
        },
    ]

    html_out = render_run_index_html(runs, tmp_path)

    # Row action buttons and Actions column are completely absent
    assert "data-action=" not in html_out
    assert "Open run" not in html_out
    assert "Actions" not in html_out

    # Multi-column sorting options & sortable column headers are present
    for sort_opt in ("newest", "oldest", "name", "fid", "progress", "speed", "disk"):
        assert f'<option value="{sort_opt}">' in html_out
    for sort_col in ("name", "fid", "newest", "progress", "speed", "disk"):
        assert f'data-sort="{sort_col}"' in html_out

    tbody = html_out.split('<tbody id="runs-body">', 1)[1].split("</tbody>", 1)[0]
    row_blocks = {
        m.group(1): m.group(0)
        for m in re.finditer(r'<tr class="run-row"[^>]*data-id="([^"]+)"[^>]*>.*?</tr>', tbody, re.DOTALL)
    }
    assert set(row_blocks) == {"run_2gpu_live", "run_stopped_stale_speed", "run_unrecorded"}

    # 1. Active 2-GPU run renders topology, batch, speed in steps/s, samples/s, ETA, and remote host
    live_html = row_blocks["run_2gpu_live"]
    assert 'data-url="/runs/run_2gpu_live"' in live_html
    assert 'data-active="true"' in live_html
    assert 'data-speed="14.2"' in live_html
    assert 'data-fid="17.1"' in live_html
    assert 'data-disk-bytes="104857600"' in live_html
    assert "gpu-node-02" in live_html
    assert "2 GPUs · Batch 256" in live_html
    assert "2× RTX A6000 · 128 / GPU" in live_html
    assert "14.2 steps/s" in live_html
    assert "3635 samples/s" in live_html
    assert "ETA 12m 05s" in live_html

    # 2. Stopped run with historical speed suppresses speed & ETA and renders CPU topology
    stopped_html = row_blocks["run_stopped_stale_speed"]
    assert 'data-url="/runs/run_stopped_stale_speed"' in stopped_html
    assert 'data-active="false"' in stopped_html
    assert 'data-speed=""' in stopped_html
    assert "CPU · Batch 16" in stopped_html
    assert "21.6" not in stopped_html
    assert "346 samples/s" not in stopped_html
    assert "steps/s" not in stopped_html
    assert "ETA" not in stopped_html
    assert '<span class="muted" data-field="speed-primary">—</span>' in stopped_html

    # 3. Complete run without recorded topology/batch renders — in Topology & Batch and Speed
    bare_html = row_blocks["run_unrecorded"]
    assert 'data-url="/runs/run_unrecorded"' in bare_html
    assert 'data-speed=""' in bare_html
    assert "33.9" not in bare_html
    assert '<span class="topo-primary muted" data-field="topology-primary">—</span>' in bare_html
    assert '<span class="muted" data-field="speed-primary">—</span>' in bare_html



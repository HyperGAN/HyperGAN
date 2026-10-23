"""Playbook 3: The Web Observatory & Python API Developer.

Covers:
1. Top-level ``import hypergan as hg`` exports (`ps`, `list_runs`, `status`,
   `logs`, `inspect`, `tag`, `rm`, `prune`, `new`, `load`, `load_bundle`).
2. ``hg.inspect(slug)`` returning a run inspection dict (not the Python stdlib
   ``inspect`` module) and ``hg.logs(slug, tail=5)`` returning ``list[str]``
   without writing to ``stdout``.
3. ``hg.tag(slug, "custom-tag")`` and resolution by custom tag across Python
   API and Web Observatory routes.
4. Web Observatory HTTP routes (`GET /`, `GET /api/runs`, `GET /runs/<slug>`,
   `GET /runs/<custom_tag>`, `/api/v1/openapi.json`) and route parity between
   `/api/runs/<ident>/*` and `/api/v1/runs/<ident>/*` for `/summary`,
   `/metrics`, `/contributions`, `/model`, `/architecture`, `/events`,
   `/views`, and `/artifacts`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import hypergan as hg
import hypergan.api  # noqa: F401
import hypergan.cli as cli
from hypergan.web_server import create_app
from hypergan.web_session import LocalSession


def _invoke_cli(capsys, *args: str) -> tuple[int, str, str]:
    capsys.readouterr()
    try:
        rc = cli.main([str(a) for a in args])
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return rc or 0, captured.out, captured.err


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _seed_trained_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    """Train a 2-step gaussian-grid run in an isolated state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / "obs-gaussian-run"

    model = hg.override(
        hg.packaged("gaussian-grid"),
        {"train.steps": 2, "train.device": "cpu"},
    )
    hg.train(model, run_dir, steps=2, checkpoint_every=2, previews=2, devices="cpu")
    st_init = hg.status(run_dir)
    slug = st_init.get("slug") or run_dir.name
    return runs_root, run_dir, slug


def test_top_level_python_api_exports():
    """Verify top-level `import hypergan as hg` exposes all lifecycle & model helpers."""
    required_callables = (
        "ps",
        "list_runs",
        "status",
        "logs",
        "inspect",
        "tag",
        "rm",
        "prune",
        "new",
        "load",
        "load_bundle",
    )
    for attr in required_callables:
        assert hasattr(hg, attr), f"hypergan top-level package missing hg.{attr}"
        val = getattr(hg, attr)
        assert not isinstance(val, types.ModuleType), (
            f"hg.{attr} resolved to module {val!r} instead of callable function"
        )
        assert callable(val), f"hg.{attr} is not callable: {val!r}"


def test_python_api_inspect_logs_new_load_and_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Verify hg.new, hg.load, hg.inspect, hg.logs (silent stdout), hg.tag, hg.load_bundle, hg.prune, hg.rm."""
    runs_root, run_dir, slug = _seed_trained_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    # 1. hg.new scaffolds a project or config and hg.load reads it back
    proj_dir = tmp_path / "scaffolded_proj"
    created = hg.new(project=proj_dir, template="gaussian-grid", device="cpu")
    created_cfg = Path(created.get("recipe_path") or created.get("config")) if isinstance(created, dict) else Path(created)
    assert created_cfg.exists()
    loaded_model = hg.load(created_cfg)
    assert isinstance(loaded_model, hg.Model)
    assert loaded_model.spec.get("format") == "hypergan-model/1"

    single_cfg = tmp_path / "single_model.toml"
    created_single = hg.new(output_path=single_cfg, template="gaussian-grid", device="cpu")
    assert Path(created_single).exists()
    assert isinstance(hg.load(Path(created_single)), hg.Model)

    # 2. hg.ps and hg.list_runs return run entries
    ps_res = hg.ps(all_runs=True, cwd=tmp_path)
    ps_rows = ps_res["runs"] if isinstance(ps_res, dict) and "runs" in ps_res else ps_res
    assert isinstance(ps_rows, list)
    assert any(r.get("slug") == slug or r.get("run_dir") == str(run_dir) for r in ps_rows)

    listed_res = hg.list_runs(runs_root)
    listed_rows = listed_res["runs"] if isinstance(listed_res, dict) and "runs" in listed_res else listed_res
    assert isinstance(listed_rows, list)
    assert any(r.get("slug") == slug or r.get("run_dir") == str(run_dir) for r in listed_rows)

    # 3. hg.status(slug) and hg.inspect(slug) return dicts (not stdlib inspect module)
    st = hg.status(slug)
    assert isinstance(st, dict)
    assert st.get("status") in ("completed", "stopped", "running")

    inspected = hg.inspect(slug)
    assert isinstance(inspected, dict)
    assert not isinstance(inspected, types.ModuleType)
    assert str(run_dir) in str(inspected.get("run_dir", "")) or inspected.get("slug") == slug

    # 4. hg.logs(slug, tail=5) returns list[str] without printing to stdout
    capsys.readouterr()  # clear prior training stdout
    log_lines = hg.logs(slug, tail=5)
    captured = capsys.readouterr()
    assert captured.out == "", f"hg.logs wrote to stdout unexpectedly: {captured.out!r}"
    assert isinstance(log_lines, list)
    assert len(log_lines) >= 1
    assert all(isinstance(line, str) for line in log_lines)

    # 5. hg.tag(slug, "custom-tag") and resolution by tag
    tag_res = hg.tag(slug, "custom-tag")
    assert isinstance(tag_res, dict)
    assert tag_res.get("name") == "custom-tag"

    inspected_by_tag = hg.inspect("custom-tag")
    assert isinstance(inspected_by_tag, dict)
    assert inspected_by_tag.get("name") == "custom-tag"

    # 6. hg.export + hg.load_bundle
    bundle_dir = tmp_path / "exported_bundle"
    exported_pt = hg.export("custom-tag", output=bundle_dir)
    assert Path(exported_pt).is_file()
    bundle = hg.load_bundle(exported_pt)
    assert isinstance(bundle, dict)
    assert "model_states" in bundle
    bundle_from_run = hg.load_bundle("custom-tag")
    assert isinstance(bundle_from_run, dict)
    assert "model_states" in bundle_from_run

    # 7. hg.prune(dry_run=True) and hg.rm
    prune_res = hg.prune("custom-tag", dry_run=True)
    assert isinstance(prune_res, dict)

    rm_res = hg.rm("custom-tag")
    assert isinstance(rm_res, dict)
    assert not run_dir.exists()


def test_observatory_routes_tag_resolution_and_api_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify Web Observatory routes, custom tag resolution, and /api/runs vs /api/v1/runs parity."""
    runs_root, run_dir, slug = _seed_trained_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    # Tag the run with a custom tag so we can test both slug and custom tag resolution
    hg.tag(slug, "custom-tag")

    port = _free_port()
    session = LocalSession(port, host="127.0.0.1", auth="none")
    app = create_app(runs_root, session)

    with TestClient(app, base_url=session.origin) as client:
        # 1. Global dashboard index
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        assert "text/html" in resp_root.headers.get("content-type", "")
        assert slug in resp_root.text or "custom-tag" in resp_root.text

        # 2. /api/runs and /api/v1/runs parity
        resp_runs = client.get("/api/runs")
        resp_v1_runs = client.get("/api/v1/runs")
        assert resp_runs.status_code == 200
        assert resp_v1_runs.status_code == 200
        runs_data = resp_runs.json()
        v1_runs_data = resp_v1_runs.json()
        assert isinstance(runs_data, list) and len(runs_data) >= 1
        assert isinstance(v1_runs_data, list) and len(v1_runs_data) == len(runs_data)
        assert {r["run_id"] for r in runs_data} == {r["run_id"] for r in v1_runs_data}

        # 3. Run detail HTML route by slug and by custom tag
        resp_slug_page = client.get(f"/runs/{slug}")
        assert resp_slug_page.status_code == 200, resp_slug_page.text

        resp_tag_page = client.get("/runs/custom-tag")
        assert resp_tag_page.status_code == 200, (
            f"Expected /runs/custom-tag to resolve tagged run, got {resp_tag_page.status_code}: {resp_tag_page.text}"
        )

        # 4. OpenAPI schema endpoint
        resp_openapi = client.get("/api/v1/openapi.json")
        assert resp_openapi.status_code == 200
        schema = resp_openapi.json()
        assert schema.get("openapi", "").startswith("3.")
        assert "/api/v1/runs" in schema.get("paths", {})

        # 5. Route parity between /api/runs/<ident>/* and /api/v1/runs/<ident>/*
        parity_suffixes = (
            "",
            "/summary",
            "/metrics",
            "/contributions",
            "/model",
            "/architecture",
            "/events",
            "/views",
            "/artifacts",
        )
        for ident in (slug, "custom-tag"):
            for suffix in parity_suffixes:
                url_short = f"/api/runs/{ident}{suffix}"
                url_v1 = f"/api/v1/runs/{ident}{suffix}"
                r_short = client.get(url_short)
                r_v1 = client.get(url_v1)
                assert r_short.status_code == 200, (
                    f"GET {url_short} failed with {r_short.status_code}: {r_short.text}"
                )
                assert r_v1.status_code == 200, (
                    f"GET {url_v1} failed with {r_v1.status_code}: {r_v1.text}"
                )
                body_short = r_short.json()
                body_v1 = r_v1.json()
                assert type(body_short) is type(body_v1), (
                    f"Type mismatch between {url_short} and {url_v1}"
                )
                if isinstance(body_short, dict):
                    assert set(body_short.keys()) == set(body_v1.keys()), (
                        f"Key mismatch between {url_short} ({set(body_short.keys())}) "
                        f"and {url_v1} ({set(body_v1.keys())})"
                    )


def test_torch_free_import_and_clean_namespace():
    """Verify `import hypergan as hg` does not load torch, defines explicit __all__, and does not leak stdlib symbols."""
    check_code = (
        "import sys\n"
        "import hypergan as hg\n"
        "assert 'torch' not in sys.modules, f'torch leaked into sys.modules: {sys.modules.get(\"torch\")}'\n"
        "assert hasattr(hg, '__all__') and len(hg.__all__) > 0, '__all__ is missing or empty'\n"
        "leaked = {'Path', 'dataclass', 'json', 're', 'tomllib'} & set(dir(hg))\n"
        "assert not leaked, f'Leaked stdlib symbols in dir(hg): {leaked}'\n"
    )
    env = os.environ.copy()
    res = subprocess.run([sys.executable, "-c", check_code], text=True, capture_output=True, env=env)
    assert res.returncode == 0, res.stderr


def test_public_sdk_docstrings():
    """Verify all public callable functions in hg.__all__ have non-empty docstrings."""
    missing = []
    for name in hg.__all__:
        obj = getattr(hg, name, None)
        if callable(obj) and not isinstance(obj, type):
            doc = getattr(obj, "__doc__", None)
            if not doc or not doc.strip():
                missing.append(name)
    assert not missing, f"Missing docstrings on public SDK functions: {missing}"


def test_callable_metrics_module_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify hg.metrics remains callable before and after `import hypergan.metrics`."""
    runs_root, run_dir, slug = _seed_trained_run(tmp_path, monkeypatch)
    import hypergan.metrics
    assert callable(hg.metrics), "hg.metrics is not callable after import hypergan.metrics"
    data = hg.metrics(slug)
    assert isinstance(data, dict)

    from hypergan import metrics
    assert callable(metrics), "from hypergan import metrics is not callable"
    assert metrics(slug) == data
    assert hasattr(hg.metrics, "validate_metrics")
    assert hasattr(hg.metrics, "compare")
    assert hasattr(hg.metrics, "get_metric")


def test_run_dataclass_properties(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify .slug, .short_id, .run_id, and .name properties on Run dataclass."""
    runs_root, run_dir, slug = _seed_trained_run(tmp_path, monkeypatch)
    run = hg.open_run(slug)
    assert run.slug == slug
    assert isinstance(run.run_id, str) and len(run.run_id) >= 8
    assert run.short_id == run.run_id[:8]
    assert run.name
    assert run.status in ("complete", "completed", "stopped", "running")


def test_train_optional_run_dir_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify `hg.train(model)` with `run_dir=None` creates and saves into `<runs_root>/<slug>`."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.chdir(tmp_path)

    model = hg.override(hg.packaged("gaussian-grid"), {"train.steps": 1, "train.device": "cpu"})
    run = hg.train(model, steps=1, devices="cpu", previews=0)
    assert run.path.is_dir()
    assert run.status in ("complete", "completed")
    assert run.slug
    assert (run.path / "manifest.json").is_file()


def test_top_level_sdk_exports_and_methods_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify top-level SDK convenience functions and enhanced arguments."""
    runs_root, run_dir, slug = _seed_trained_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    # 1. hg.list_templates
    templates = hg.list_templates()
    assert isinstance(templates, (list, tuple))
    template_names = [t["name"] if isinstance(t, dict) else t for t in templates]
    assert "gaussian-grid" in template_names

    # 2. hg.describe_model with Model instance
    m = hg.packaged("gaussian-grid")
    desc = hg.describe_model(m)
    assert isinstance(desc, dict)
    assert "networks" in desc or "formulation" in desc or "config" in desc

    # 3. hg.diff between two models/configs
    diff_res = hg.diff(m, m)
    assert isinstance(diff_res, dict)
    assert diff_res.get("identical") is True

    # 4. hg.compare with runs
    comp_res = hg.compare([slug, slug])
    assert isinstance(comp_res, dict)
    assert "matrix" in comp_res or "runs" in comp_res

    # 5. hg.prepare_resume by slug
    prep = hg.prepare_resume(slug)
    assert prep is not None and prep.operation == "resume"

    # 6. hg.contributions auto-projecting
    page = hg.contributions(slug)
    assert isinstance(page, dict)
    assert "frames" in page or "cursor" in page

    # 7. hg.load_inference by slug
    inf = hg.load_inference(slug)
    assert "model_states" in inf

    # 8. hg.sample by slug
    s = hg.sample(slug, count=2)
    assert s is not None
    assert isinstance(s, hg.Sample)


def test_view_cli_tag_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Verify `hypergan view <tag> --status` and `hypergan view <tag> --stop` resolve custom tags cleanly."""
    runs_root, run_dir, slug = _seed_trained_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    hg.tag(slug, "release-candidate")

    rc_status, out_status, err_status = _invoke_cli(capsys, "view", "release-candidate", "--status")
    assert rc_status == 0, err_status
    st_json = json.loads(out_status)
    assert "status" in st_json

    rc_stop, out_stop, err_stop = _invoke_cli(capsys, "view", "release-candidate", "--stop")
    assert rc_stop == 0, err_stop
    stop_json = json.loads(out_stop)
    assert "status" in stop_json

    # Also resolve by slug
    rc_slug, out_slug, err_slug = _invoke_cli(capsys, "view", slug, "--status")
    assert rc_slug == 0, err_slug
    st_slug_json = json.loads(out_slug)
    assert "status" in st_slug_json

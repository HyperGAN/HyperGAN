"""Every run reader reports the same FID, slug and tag.

Code review 2026-10-25, cross-cutting recommendation 3 and slice 6: the
registry, `ps`, `status` and the dashboard each guessed "is this FID?" from
metric names and pooled FID protocols, and the dashboard derived its own slug
and name. They now read summary.json's FID fields (which use the metric
catalog), registry.canonical_slug and registry.read_run_tag.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from hypergan import registry
from hypergan.inspection import format_run_status, list_runs, run_status, tag_run
from hypergan.metrics import digest
from hypergan.queue import _find_run_dir_from_registry, queue_add, queue_ls, queue_run
from hypergan.registry import get_registry_dir, list_registered_runs, read_registry, read_run_fid, reserve_run_slug
from hypergan.summary import read_summary_json, write_summary_json
from hypergan.web_server import read_run_summary


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERGAN_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("HYPERGAN_STATE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


def _write_catalog(run_dir, definitions):
    metrics = {}
    for name, fields in definitions.items():
        definition = {"kind": "scalar", "source": "custom:" + name, **fields}
        metrics[name] = dict(definition, definition_hash=digest(definition))
    catalog = {"schema_version": 1, "metrics": metrics}
    revision = digest(catalog)
    path = Path(run_dir) / "metrics" / f"catalog-{revision}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return revision


def _write_receipt(run_dir, metric, step, value, catalog=None):
    directory = Path(run_dir) / "metrics" / "evaluations" / f"step-{step:08d}-{metric}-{step:08x}"
    directory.mkdir(parents=True)
    receipt = {"schema_version": 1, "status": "complete", "metric_id": metric, "source_step": step,
               "attempt_id": "attempt-1", "result": {"value": value, "step": step}}
    if catalog is not None:
        receipt["catalog"] = catalog
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def _make_run(run_dir, *, run_id, keep_best=None, steps=10000, name="logos", slug=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"name": name, "training": {"steps": 20000}}
    if keep_best:
        config["checkpoints"] = {"keep_best": {"metric": keep_best, "mode": "min", "count": 1}}
    manifest = {"schema_version": 1, "run_id": run_id, "status": "completed", "steps": steps, "config": config}
    if slug:
        manifest["slug"] = slug
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _readers(run_dir, tmp_path):
    """best_fid as the registry, `ps`, `status` and the dashboard report it."""
    run_dir = run_dir.resolve()
    entry = {Path(e["run_dir"]): e for e in list_registered_runs(cwd=tmp_path)}[run_dir]
    row = {Path(r["run_dir"]): r for r in list_runs(root=run_dir.parent)}[run_dir]
    return {
        "registry": entry["best_fid"],
        "ps": row["best_fid"],
        "status": run_status(run_dir)["best_fid"],
        "web": read_run_summary(run_dir)["best_fid"],
    }


CATALOG = {
    "chroma": {"direction": "minimize", "unit": "normalized_chroma"},
    "fid_smoke": {"direction": "minimize", "unit": "FID"},
    "fid10k_heldout": {"direction": "minimize", "unit": "FID"},
}


def test_non_fid_primary_metric_reports_no_fid_anywhere(tmp_path):
    run = _make_run(tmp_path / "runs" / "logos-calm-otter", run_id="a" * 32, keep_best="chroma")
    revision = _write_catalog(run, CATALOG)
    _write_receipt(run, "chroma", 4000, 0.05, revision)
    _write_receipt(run, "fid10k_heldout", 5000, 61.0, revision)
    write_summary_json(run, status="completed")
    summary = read_summary_json(run)
    assert summary["best"]["metric"] == "chroma" and summary["best_fid"] is None

    # The dashboard used to show the chroma distance as "best FID"; ps scanned
    # receipts for any FID-like name.
    assert _readers(run, tmp_path) == {"registry": None, "ps": None, "status": None, "web": None}
    assert read_run_summary(run)["latest_fid"] is None
    text = format_run_status(run_status(run))
    assert "Best chroma:  0.05 at step 4,000" in text and "Best FID" not in text


def test_two_fid_protocols_are_never_pooled(tmp_path):
    # fid_smoke (128 samples) is lower but not comparable with fid10k_heldout.
    run = _make_run(tmp_path / "runs" / "logos-bold-hawk", run_id="b" * 32, keep_best="fid10k_heldout")
    revision = _write_catalog(run, CATALOG)
    for metric, step, value in [("fid_smoke", 2000, 20.0), ("fid_smoke", 4000, 18.0),
                                ("fid10k_heldout", 5000, 70.0), ("fid10k_heldout", 10000, 64.0)]:
        _write_receipt(run, metric, step, value, revision)
    write_summary_json(run, status="completed")
    assert read_summary_json(run)["best_fid"] == 64.0

    assert _readers(run, tmp_path) == {"registry": 64.0, "ps": 64.0, "status": 64.0, "web": 64.0}
    web = read_run_summary(run)
    assert (web["latest_fid"], web["best_step"]) == (64.0, 10000)


def test_legacy_receipts_with_two_fid_protocols_report_no_pooled_minimum(tmp_path):
    # Receipts without a catalog and no fid50k_train (the default keep_best):
    # no primary metric, so summary.json has no best. `ps` and `status` used to
    # take the minimum over every FID-like name (18.0 here).
    run = _make_run(tmp_path / "runs" / "legacy-amber-fox", run_id="c" * 32)
    for metric, step, value in [("fid_smoke", 4000, 18.0), ("fid10k_heldout", 5000, 40.0)]:
        _write_receipt(run, metric, step, value)
    events = [{"event": "evaluation", "step": 4000, "metric_id": "fid_smoke", "value": 18.0},
              {"event": "evaluation", "step": 5000, "metric_id": "fid10k_heldout", "value": 40.0}]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    write_summary_json(run, status="completed")
    assert read_summary_json(run)["best"] is None
    assert _readers(run, tmp_path) == {"registry": None, "ps": None, "status": None, "web": None}

    # Without summary.json or receipts, the events tail is not pooled either.
    (run / "summary.json").unlink()
    (run / "manifest.json").unlink()
    for receipt in (run / "metrics" / "evaluations").glob("*/receipt.json"):
        receipt.unlink()
    assert _readers(run, tmp_path) == {"registry": None, "ps": None, "status": None, "web": None}


def test_legacy_single_fid_protocol_and_missing_summary_fall_back_to_receipts(tmp_path):
    run = _make_run(tmp_path / "runs" / "legacy-noble-wren", run_id="d" * 32)
    for step, value in [(5000, 40.0), (10000, 35.0), (15000, 38.0)]:
        _write_receipt(run, "fid10k_heldout", step, value)
    write_summary_json(run, status="completed")
    assert read_summary_json(run)["best"] is None
    assert _readers(run, tmp_path) == {"registry": 35.0, "ps": 35.0, "status": 35.0, "web": 35.0}
    assert read_run_fid(run, read_summary_json(run)) == {
        "metric": "fid10k_heldout", "best_fid": 35.0, "best_step": 10000, "latest_fid": 38.0}
    assert read_run_summary(run)["latest_fid"] == 38.0
    assert run_status(run)["fid_metric"] == "fid10k_heldout"

    # Without summary.json the catalog-backed ranking runs over the receipts.
    cataloged = _make_run(tmp_path / "runs" / "cat-quiet-lynx", run_id="e" * 32, keep_best="fid10k_heldout")
    revision = _write_catalog(cataloged, CATALOG)
    _write_receipt(cataloged, "fid_smoke", 1000, 10.0, revision)
    _write_receipt(cataloged, "fid10k_heldout", 2000, 50.0, revision)
    assert read_run_fid(cataloged, None, json.loads((cataloged / "manifest.json").read_text())) == {
        "metric": "fid10k_heldout", "best_fid": 50.0, "best_step": 2000, "latest_fid": 50.0}


def test_reserved_slug_and_tag_match_between_dashboard_and_ps(tmp_path):
    runs = tmp_path / "runs"
    older = _make_run(runs / "dcgan-amber-fox", run_id="1" * 32, name="dcgan")
    write_summary_json(older, status="completed")
    # A run whose directory suffix collides gets another slug from the registry.
    run = runs / "wgan-amber-fox"
    run.mkdir()
    reserved = reserve_run_slug(run, "2" * 32)
    assert reserved != "amber-fox"
    # An older trainer did not store the slug in its manifest, so summary.json
    # carries the slug derived from the directory name.
    _make_run(run, run_id="2" * 32, name="wgan")
    write_summary_json(run, status="completed")
    assert read_summary_json(run)["slug"] == "amber-fox"
    tag_run(run, "wgan-baseline", cwd=tmp_path)

    ps_row = {Path(r["run_dir"]): r for r in list_runs(cwd=tmp_path, sort="updated")}[run.resolve()]
    web = read_run_summary(run)
    assert (ps_row["slug"], ps_row["name"]) == (reserved, "wgan-baseline")
    assert (web["slug"], web["name"], web["tag"]) == (reserved, "wgan-baseline", "wgan-baseline")
    status = run_status(run)
    assert (status["slug"], status["name"]) == (reserved, "wgan-baseline")


def test_dashboard_cache_refreshes_when_a_run_is_tagged(tmp_path):
    from hypergan.web_server import ServiceRegistry

    run = _make_run(tmp_path / "runs" / "gan-crisp-heron", run_id="3" * 32, name="gan")
    write_summary_json(run, status="completed")
    service = ServiceRegistry(tmp_path / "runs")
    assert service._read_cached_summary(run)["name"] == "gan"
    registry.write_run_tag(run, "renamed")
    assert service._read_cached_summary(run)["name"] == "renamed"


def _register(run_dir, *, run_id, name, pid=None, updated_at=None):
    _make_run(run_dir, run_id=run_id, name=name)
    write_summary_json(run_dir, status="completed")
    entry = registry.register_run(run_dir, pid=pid)
    if updated_at is not None:
        reg_file = get_registry_dir(run_dir) / "runs.json"
        data = json.loads(reg_file.read_text(encoding="utf-8"))
        data["runs"][str(run_dir.resolve())]["updated_at"] = updated_at
        reg_file.write_text(json.dumps(data), encoding="utf-8")
    return entry


def test_queue_never_records_an_older_run_that_shares_the_job_name(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    # Earlier runs of the same model share its stem, which is the default job name.
    _register(runs / "model-amber-fox", run_id="4" * 32, name="model")
    _register(runs / "model-bold-hawk", run_id="5" * 32, name="model")
    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    queue_file = tmp_path / ".hypergan" / "queue.json"
    job = queue_add(model_file, queue_file=queue_file)
    assert job["name"] == "model"

    class TrainerFailsBeforeCreatingARun:
        def __init__(self, cmd, **kwargs):
            self.pid = 424242

        def wait(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", TrainerFailsBeforeCreatingARun)
    queue_run(once=True, queue_file=queue_file)
    assert queue_ls(queue_file=queue_file)[0]["run_dir"] is None


def test_queue_finds_the_run_its_trainer_registered_by_pid(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    started = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    stale = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    # An entry left by an earlier process that had the same PID is not this job's run.
    _register(runs / "model-quiet-lynx", run_id="6" * 32, name="model", pid=424242, updated_at=stale)
    assert _find_run_dir_from_registry(424242, since=started) is None

    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    queue_file = tmp_path / ".hypergan" / "queue.json"
    queue_add(model_file, queue_file=queue_file)
    launched = runs / "model-noble-wren"

    class Trainer:
        def __init__(self, cmd, **kwargs):
            self.pid = 424242
            _register(launched, run_id="7" * 32, name="model", pid=self.pid)

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Trainer)
    queue_run(once=True, queue_file=queue_file)
    assert queue_ls(queue_file=queue_file)[0]["run_dir"] == str(launched.resolve())


def test_read_registry_is_the_public_unlocked_snapshot(tmp_path):
    run = _make_run(tmp_path / "runs" / "gan-calm-otter", run_id="8" * 32, name="gan")
    write_summary_json(run, status="completed")
    data = read_registry()
    assert str(run.resolve()) in data["runs"]
    assert read_registry(get_registry_dir(tmp_path)) == data
    # api.py and web_autostart.py still import the private name.
    assert registry._read_registry_unlocked is registry.read_registry
    for alias in ("_get_state_dir", "_load_registry_unlocked", "discover_runs_on_host", "_inspect_run_entry"):
        assert not hasattr(registry, alias)

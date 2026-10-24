"""Unit tests for filesystem resilience, registry self-healing, Tier 1 mtime cache,
lifecycle functions (tag_run, remove_run, gc_all), and workspace variable interpolation.
"""

from contextlib import ExitStack
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hypergan.inspection import (
    _collect_ps_row,
    gc_all,
    ps_runs,
    read_run_logs,
    remove_run,
    run_status,
    status,
    tag_run,
)
from hypergan.registry import (
    get_registry_dir,
    list_registered_runs,
    register_run,
    relocate_missing_run,
    resolve_run_target,
    unregister_run,
)
from hypergan.run_state import run_lock
from hypergan.summary import write_summary_json
from hypergan.workspace import interpolate_variables, load_workspace


def _create_mock_run(
    run_dir: Path,
    *,
    run_id: str | None = None,
    name: str | None = None,
    slug: str | None = None,
    status_val: str = "completed",
    steps: int = 50000,
    total_steps: int = 100000,
    sps: float = 35.5,
    best_fid: float = 14.25,
    device: str = "cuda:0",
    with_views: bool = False,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id or run_dir.name
    exp_name = name or run_dir.name

    manifest = {
        "schema_version": 1,
        "run_id": rid,
        "name": exp_name,
        "status": status_val,
        "steps": steps,
        "total_steps": total_steps,
        "steps_per_second": sps,
        "device": device,
        "config": {
            "name": exp_name,
            "train": {"device": device, "steps": total_steps},
            "training": {"device": device, "steps": total_steps},
        },
    }
    if slug:
        manifest["slug"] = slug
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    events = [
        {"event": "start", "step": 0, "attempt_id": "0001", "run_id": rid},
        {"event": "train", "step": steps // 2, "steps_per_second": sps, "metrics": {"loss/g_total": 1.75, "loss/d_total": 0.95}},
        {"event": "evaluation", "step": steps, "metric_id": "fid50k_train", "value": best_fid},
        {"event": "train", "step": steps, "steps_per_second": sps, "metrics": {"loss/g_total": 1.15, "loss/d_total": 0.65}},
        {"event": status_val, "step": steps},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    eval_dir = run_dir / "metrics" / "evaluations" / f"step-{steps:07d}-fid50k_train"
    eval_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "metric_id": "fid50k_train",
        "source_step": steps,
        "result": {"value": best_fid, "step": steps},
    }
    (eval_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    if with_views:
        views_dir = run_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        (views_dir / "contributions.jsonl").write_bytes(b"view_payload\n" * 200)

    write_summary_json(run_dir, status=status_val)
    return run_dir


def test_deleted_run_pruned_and_friendly_error(tmp_path):
    """Deleting a run folder (rm -rf) prunes it from list_registered_runs() and gives a friendly error when targeted."""
    r1 = _create_mock_run(
        tmp_path / "cifar-amber-fox",
        run_id="a1b2c3d4e5f64789a1b2c3d4e5f64789",
        name="cifar-amber-fox",
        slug="amber-fox",
    )
    r2 = _create_mock_run(
        tmp_path / "cifar-cobalt-hawk",
        run_id="f9e8d7c6b5a44321f9e8d7c6b5a44321",
        name="cifar-cobalt-hawk",
        slug="cobalt-hawk",
    )
    register_run(r1)
    register_run(r2)

    # Delete r1 on disk via rm -rf equivalent
    shutil.rmtree(r1)

    # Targeting deleted run by slug raises friendly FileNotFoundError and removes stale entry
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_run_target("amber-fox", cwd=tmp_path)
    msg = str(exc_info.value)
    assert "Run 'amber-fox' (a1b2c3d4) was at" in msg
    assert "which no longer exists (removed stale entry from registry)." in msg

    # Targeting by short ID also raises friendly FileNotFoundError
    with pytest.raises(FileNotFoundError, match=r"Run 'amber-fox' \(a1b2c3d4\) was at .* which no longer exists"):
        resolve_run_target("a1b2c3d4", cwd=tmp_path)

    # Verify pruned from list_registered_runs() and raw runs.json
    remaining = list_registered_runs(cwd=tmp_path)
    assert len(remaining) == 1
    assert remaining[0]["slug"] == "cobalt-hawk"

    reg_file = get_registry_dir(tmp_path) / "runs.json"
    raw_runs = json.loads(reg_file.read_text(encoding="utf-8"))["runs"]
    assert str(r1.resolve()) not in raw_runs
    assert str(r2.resolve()) in raw_runs


def test_renamed_or_moved_run_auto_healed_in_registry(tmp_path):
    """Renaming or moving a run folder (mv run_a run_b) auto-heals runs.json when targeted or listed."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_a = _create_mock_run(
        runs_root / "run_a",
        run_id="11223344556677889900aabbccddeeff",
        name="run_a",
        slug="crystal-falcon",
    )
    register_run(run_a)

    # 1. Rename run_a -> run_b and resolve by slug
    run_b = runs_root / "run_b"
    run_a.rename(run_b)

    resolved = resolve_run_target("crystal-falcon", cwd=tmp_path)
    assert resolved == run_b.resolve()

    # Check runs.json was healed in-place with new path and updated name
    reg_file = get_registry_dir(tmp_path) / "runs.json"
    reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
    healed_entry = reg_data["runs"].get(str(run_b.resolve()))
    assert healed_entry is not None
    assert healed_entry["run_dir"] == str(run_b.resolve())
    assert healed_entry["name"] == "run_b"
    assert str(run_a.resolve()) not in reg_data["runs"]

    # 2. Rename run_b -> run_c and call list_registered_runs() directly
    run_c = runs_root / "run_c"
    run_b.rename(run_c)

    listed = list_registered_runs(cwd=tmp_path)
    assert len(listed) == 1
    assert listed[0]["run_dir"] == str(run_c.resolve())
    assert listed[0]["name"] == "run_c"
    assert listed[0]["run_id"] == "11223344556677889900aabbccddeeff"

    # 3. Direct relocate_missing_run test when moved into workspace runs_root
    ws_runs = tmp_path / "ws_runs"
    ws_runs.mkdir()
    (tmp_path / "hypergan.toml").write_text(
        f'[workspace]\nroot = "{tmp_path}"\nruns_root = "{ws_runs}"\n',
        encoding="utf-8",
    )
    run_d = ws_runs / "run_d"
    run_c.rename(run_d)
    entry_copy = dict(listed[0])
    entry_copy["workspace"] = str(tmp_path)
    found = relocate_missing_run(entry_copy, cwd=tmp_path)
    assert found == run_d.resolve()
    assert entry_copy["run_dir"] == str(run_d.resolve())
    assert entry_copy["name"] == "run_d"


def test_corrupted_or_missing_metadata_fallback(tmp_path):
    """Corrupted/truncated summary.json and manifest.json fall back gracefully without crashing."""
    run_dir = _create_mock_run(
        tmp_path / "corrupt-test-run",
        run_id="deadbeefcafebabe1234567890abcdef",
        name="resilient-exp",
        slug="neon-nexus",
        status_val="completed",
        steps=40000,
        total_steps=80000,
        best_fid=12.34,
    )

    # Case 1: Truncated/corrupted summary.json -> falls back to manifest.json and events.jsonl
    (run_dir / "summary.json").write_text("{corrupted_json: [1, 2", encoding="utf-8")

    st = status(run_dir, cwd=tmp_path)
    assert st["run_id"] == "deadbeefcafebabe1234567890abcdef"
    assert st["name"] == "resilient-exp"
    assert st["status"] == "completed"
    assert st["step"] == 40000
    assert st["best_fid"] == pytest.approx(12.34)
    assert st["latest_metrics"]["loss/g_total"] == pytest.approx(1.15)

    ps_data = ps_runs(all_runs=True, cwd=tmp_path)
    matched = [r for r in ps_data["runs"] if r["run_id"] == "deadbeefcafebabe1234567890abcdef"]
    assert len(matched) == 1
    assert matched[0]["status"] == "completed"
    assert matched[0]["step"] == 40000

    # Case 2: Both summary.json AND manifest.json are corrupted -> falls back to events.jsonl and marks status 'incomplete'
    (run_dir / "manifest.json").write_text("<<<NOT VALID JSON>>>", encoding="utf-8")
    # Also append a truncated line to events.jsonl to ensure read_run_logs and status ignore bad lines
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"event": "train", "step": 40001, "metrics": {"loss/g_total": 0.99\n')

    st_both_corrupt = run_status(run_dir, cwd=tmp_path)
    assert st_both_corrupt["status"] == "incomplete"
    assert st_both_corrupt["step"] == 40000
    assert st_both_corrupt["latest_metrics"]["loss/g_total"] == pytest.approx(1.15)

    row = _collect_ps_row(run_dir)
    assert row["status"] == "incomplete"
    assert row["step"] == 40000

    buf = StringIO()
    log_lines = read_run_logs(run_dir, tail=10, stream=buf, cwd=tmp_path)
    assert len(log_lines) >= 3
    assert any("40,000" in ln for ln in log_lines)


def test_tag_run_and_remove_run_lifecycle(tmp_path, monkeypatch):
    """Verify tag_run updates summary, manifest, and registry, and remove_run respects active lock and force=True."""
    run_dir = _create_mock_run(
        tmp_path / "lifecycle-run",
        run_id="99887766554433221100ffeeddccbbaa",
        name="initial-name",
        slug="jade-orbit",
        status_val="completed",
    )

    # 1. tag_run updates summary.json, manifest.json, and registry
    tagged = tag_run("jade-orbit", "renamed-experiment-v2", cwd=tmp_path)
    assert tagged == {
        "run_id": "99887766554433221100ffeeddccbbaa",
        "slug": "jade-orbit",
        "name": "renamed-experiment-v2",
        "run_dir": str(run_dir.resolve()),
    }
    sum_data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    man_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sum_data["name"] == "renamed-experiment-v2"
    assert man_data["name"] == "renamed-experiment-v2"
    assert man_data["config"]["name"] == "renamed-experiment-v2"

    # Can now resolve by the new tag name
    assert resolve_run_target("renamed-experiment-v2", cwd=tmp_path) == run_dir.resolve()

    # 2. remove_run refuses to remove an active run when force=False
    with ExitStack() as active_run:
        active_run.enter_context(run_lock(run_dir))
        with pytest.raises(RuntimeError, match=r"is currently active .* Use --force / -f to stop and remove it"):
            remove_run("jade-orbit", force=False, cwd=tmp_path)
        assert run_dir.exists()

        def stop_active_run(target, *, force, cwd):
            assert target == run_dir.resolve() and force and cwd == tmp_path
            # A stopped worker closes its handles before the directory is
            # removed. Windows cannot unlink this fixture's open lock file.
            active_run.close()

        monkeypatch.setattr("hypergan.inspection.stop_run", stop_active_run)
        # 3. remove_run with force=True stops and removes the active run
        rem = remove_run("jade-orbit", force=True, cwd=tmp_path)
        assert rem == {
            "removed": True,
            "run_id": "99887766554433221100ffeeddccbbaa",
            "slug": "jade-orbit",
            "name": "renamed-experiment-v2",
            "run_dir": str(run_dir.resolve()),
        }
        assert not run_dir.exists()

    # Verify removed from registry
    assert list_registered_runs(cwd=tmp_path) == []
    assert unregister_run(run_dir, cwd=tmp_path) is False


def test_tier1_summary_mtime_ns_cache_avoids_reparse(tmp_path, monkeypatch):
    """Tier 1 summary_mtime_ns cache in runs.json avoids re-reading summary.json/manifest.json when mtime is unchanged."""
    run_dir = _create_mock_run(
        tmp_path / "cached-run",
        run_id="abcd1234abcd1234abcd1234abcd1234",
        name="cached-exp",
        slug="golden-prism",
        status_val="completed",
        steps=75000,
        total_steps=150000,
        sps=50.0,
        best_fid=10.5,
        device="cuda:1",
    )

    entry = register_run(run_dir)
    summary_file = run_dir / "summary.json"
    expected_mtime_ns = summary_file.stat().st_mtime_ns

    assert entry["summary_mtime_ns"] == expected_mtime_ns
    assert entry["step"] == 75000
    assert entry["max_steps"] == 150000
    assert entry["speed"] == pytest.approx(50.0)
    assert entry["best_fid"] == pytest.approx(10.5)
    assert entry["device"] == "cuda:1"
    assert entry["updated_at"]

    # Monkeypatch read_summary_json in both registry and inspection to fail if called during cache hit
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("read_summary_json should not be called on Tier 1 summary_mtime_ns cache hit!")

    monkeypatch.setattr("hypergan.registry.read_summary_json", _fail_if_called)
    monkeypatch.setattr("hypergan.inspection.read_summary_json", _fail_if_called)

    # Both list_registered_runs() and ps_runs(all_runs=True) must hit Tier 1 cache without reading summary.json
    reg_runs = list_registered_runs(cwd=tmp_path)
    assert len(reg_runs) == 1
    assert reg_runs[0]["step"] == 75000
    assert reg_runs[0]["max_steps"] == 150000
    assert reg_runs[0]["speed"] == pytest.approx(50.0)
    assert reg_runs[0]["best_fid"] == pytest.approx(10.5)
    assert reg_runs[0]["device"] == "cuda:1"

    ps_out = ps_runs(all_runs=True, cwd=tmp_path)
    assert len(ps_out["runs"]) == 1
    assert ps_out["runs"][0]["step"] == 75000
    assert ps_out["runs"][0]["device"] == "cuda:1"

    # Now restore read_summary_json, update summary.json with a new step & mtime, and verify cache invalidates
    monkeypatch.undo()
    sum_doc = json.loads(summary_file.read_text(encoding="utf-8"))
    sum_doc["step"] = 90000
    summary_file.write_text(json.dumps(sum_doc), encoding="utf-8")
    os.utime(summary_file, ns=(expected_mtime_ns + 1_000_000, expected_mtime_ns + 1_000_000))

    refreshed = list_registered_runs(cwd=tmp_path)
    assert refreshed[0]["step"] == 90000
    assert refreshed[0]["summary_mtime_ns"] == expected_mtime_ns + 1_000_000


def test_workspace_bare_and_dollar_variable_interpolation(tmp_path):
    """Both {runs_root} and ${runs_root} (plus {stem}, {root}, etc.) are interpolated in workspace.py."""
    ws_file = tmp_path / "hypergan.toml"
    ws_file.write_text(
        '[workspace]\n'
        'runs_root = "{root}/my_runs"\n'
        'run_dir = "{runs_root}/train-{stem}"\n'
        '\n'
        '[paths]\n'
        'datasets = "{root}/datasets"\n'
        'checkpoints = "${runs_root}/ckpts"\n',
        encoding="utf-8",
    )

    ws = load_workspace(ws_file)
    expected_runs_root = str((tmp_path / "my_runs").resolve())
    assert ws["workspace"]["runs_root"] == expected_runs_root
    assert ws["paths"]["datasets"] == f"{tmp_path}/datasets"
    assert ws["paths"]["checkpoints"] == f"{expected_runs_root}/ckpts"

    # Interpolating run_dir with stem replaces both {runs_root} and {stem} (no literal '{runs_root}' folder)
    interpolated_bare = interpolate_variables(
        "{runs_root}/train-{stem}",
        ws,
        {"stem": "cifar10"},
    )
    assert interpolated_bare == f"{expected_runs_root}/train-cifar10"
    assert "{runs_root}" not in interpolated_bare

    interpolated_dollar = interpolate_variables(
        "${runs_root}/train-${stem}",
        ws,
        {"stem": "cifar10"},
    )
    assert interpolated_dollar == f"{expected_runs_root}/train-cifar10"

    # Unknown bare placeholders (not in lookup dict) remain intact while known ones interpolate
    mixed = interpolate_variables("{runs_root}/epoch-{epoch_num:04d}", ws)
    assert mixed == f"{expected_runs_root}/epoch-{{epoch_num:04d}}"


def test_gc_all_includes_registered_runs_outside_workspace_root(tmp_path):
    """gc_all(root=None) cleans registered runs in addition to workspace runs_root."""
    ws_runs = tmp_path / "ws_runs"
    ws_runs.mkdir()
    (tmp_path / "hypergan.toml").write_text(
        f'[workspace]\nroot = "{tmp_path}"\nruns_root = "{ws_runs}"\n',
        encoding="utf-8",
    )

    r_ws = _create_mock_run(ws_runs / "ws_run_1", with_views=True)
    external_dir = tmp_path / "external_location" / "ext_run_2"
    r_ext = _create_mock_run(external_dir, with_views=True)
    register_run(r_ext)

    assert (r_ws / "views").is_dir()
    assert (r_ext / "views").is_dir()

    results = gc_all(root=None, cwd=tmp_path)
    cleaned_dirs = {Path(r["run_dir"]).resolve() for r in results if not r.get("skipped")}
    assert r_ws.resolve() in cleaned_dirs
    assert r_ext.resolve() in cleaned_dirs
    assert not (r_ws / "views").exists()
    assert not (r_ext / "views").exists()


def test_torch_free_invariant_on_registry_inspection_workspace(tmp_path):
    """Ensure registry.py, inspection.py, and workspace.py never import torch."""
    code = """
import sys
import hypergan.workspace
import hypergan.registry
import hypergan.inspection

assert "torch" not in sys.modules, f"torch imported: {[m for m in sys.modules if 'torch' in m]}"
assert "particlegan" not in sys.modules
"""
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert res.returncode == 0, res.stderr

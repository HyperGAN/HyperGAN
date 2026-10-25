"""Negative and concurrency tests for the destructive run commands (rm, gc/prune, stop, tag).

Code review 2026-10-25, fix-first #1-#3 and the slice-6 registry/inspection
findings: every destructive command resolves its target strictly and only acts
on a real run directory, and the registry never re-derives which run a slug names.
"""

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from hypergan import inspection
from hypergan import registry
from hypergan._file_lock import retry_lock
from hypergan.inspection import (
    check_run_active,
    compact_events,
    gc_all,
    gc_run,
    list_runs,
    read_run_logs,
    remove_run,
    stop_run,
    tag_run,
)
from hypergan.registry import (
    find_live_pid_for_run,
    get_registry_dir,
    list_registered_runs,
    register_run,
    reserve_run_slug,
    resolve_destructive_target,
)
from hypergan.run_state import run_lock
from hypergan.summary import write_summary_json

needs_proc = pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs Linux /proc")


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERGAN_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("HYPERGAN_STATE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


def _make_run(run_dir: Path, *, run_id: str, name: str = "exp", slug: str | None = None,
              train_steps: int = 0, views: bool = False) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "run_id": run_id, "status": "completed", "steps": max(train_steps, 1),
                "config": {"name": name, "training": {"steps": max(train_steps, 1)}}}
    if slug:
        manifest["slug"] = slug
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    events = [{"event": "start", "step": 0, "attempt_id": "0001", "run_id": run_id, "sequence": 1}]
    events += [{"event": "train", "step": s, "attempt_id": "0001", "run_id": run_id, "sequence": s + 1,
                "metrics": {"loss/g_total": 1.0}} for s in range(1, train_steps + 1)]
    events.append({"event": "complete", "step": train_steps, "attempt_id": "0001", "run_id": run_id})
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    if views:
        (run_dir / "views").mkdir()
        (run_dir / "views" / "cache.jsonl").write_bytes(b"x" * 500)
    write_summary_json(run_dir, status="completed")
    return run_dir


def _lock_holder(run_dir: Path, *, ignore_sigint: bool = False) -> subprocess.Popen:
    """A stand-in trainer: another process that holds run_dir/.run.lock."""
    code = (
        "import signal, sys, time\n"
        + ("signal.signal(signal.SIGINT, signal.SIG_IGN)\n" if ignore_sigint else "")
        + "from hypergan.run_state import run_lock\n"
        "with run_lock(sys.argv[1]):\n"
        "    print('locked', flush=True)\n"
        "    time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(run_dir)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "locked"
    return proc


def _reap(*procs):
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


# --------------------------------------------------------------------------- strict resolution (H1, H2, H3)

def test_rm_refuses_directories_that_are_not_runs(tmp_path, monkeypatch):
    """`rm notes`, `rm runs`, `rm .`, `rm <workspace>`, `rm ~` and `rm <copy of a dataset>` delete nothing."""
    (tmp_path / "hypergan.toml").write_text(f'[workspace]\nruns_root = "{tmp_path / "runs"}"\n', encoding="utf-8")
    run = _make_run(tmp_path / "runs" / "dcgan-amber-fox", run_id="a" * 32)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "precious.txt").write_text("keep", encoding="utf-8")
    victim = tmp_path / "victim"
    (victim / "images").mkdir(parents=True)
    # Run-like children are not a run identity: is_run_dir() accepts this, rm must not.
    (victim / "checkpoints").mkdir()
    (victim / "events.jsonl").write_text("{}\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    (home / "manifest.json").write_text(json.dumps({"run_id": "h" * 32}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    for target in ("notes", "runs", ".", str(tmp_path), str(victim), "victim", str(home)):
        with pytest.raises(ValueError, match="Refusing to act on"):
            remove_run(target, cwd=tmp_path)
    with pytest.raises(FileNotFoundError):
        remove_run("missing/dir", cwd=tmp_path)

    assert (notes / "precious.txt").read_text(encoding="utf-8") == "keep"
    assert (victim / "images").is_dir() and home.is_dir()
    assert (run / "manifest.json").is_file()


def test_rm_refuses_symlinks_and_a_runs_root_with_a_manifest(tmp_path):
    runs = tmp_path / "runs"
    run = _make_run(runs / "wgan-calm-otter", run_id="b" * 32)
    link = tmp_path / "link-to-run"
    link.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        remove_run(str(link), cwd=tmp_path)
    with pytest.raises(ValueError, match="symbolic link"):
        gc_run(str(link), cwd=tmp_path)

    # A registered runs root that happens to carry a manifest is still a runs root.
    (runs / "manifest.json").write_text(json.dumps({"run_id": "r" * 32}), encoding="utf-8")
    with pytest.raises(ValueError, match="registered runs root"):
        remove_run(str(runs), cwd=tmp_path)
    # So is an unregistered directory with a manifest that holds other runs.
    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    for d, rid in ((outer, "o" * 32), (outer / "inner", "i" * 32)):
        (d / "manifest.json").write_text(json.dumps({"run_id": rid}), encoding="utf-8")
    with pytest.raises(ValueError, match="contains other runs"):
        remove_run(str(outer), cwd=tmp_path)
    assert link.is_symlink() and (run / "manifest.json").is_file() and (outer / "inner").is_dir()


def test_rm_never_picks_one_of_several_or_a_fuzzy_match(tmp_path):
    """Shared config names, prefixes and substrings do not resolve; each error lists the candidates."""
    runs = tmp_path / "runs"
    a = _make_run(runs / "dcgan-amber-fox", run_id="1" * 32, name="dcgan")
    b = _make_run(runs / "dcgan-bold-hawk", run_id="2" * 32, name="dcgan")
    c = _make_run(runs / "wgan-calm-otter", run_id="3" * 32, name="wgan")
    for r in (a, b, c):
        register_run(r)

    with pytest.raises(FileNotFoundError, match="No run has the ID") as exc:
        remove_run("dcgan", cwd=tmp_path)
    assert "11111111" in str(exc.value) and "22222222" in str(exc.value)
    for fuzzy in ("w", "otter", "dcgan-amber", "1111"):
        with pytest.raises(FileNotFoundError):
            remove_run(fuzzy, cwd=tmp_path)
    with pytest.raises(FileNotFoundError):
        tag_run("dcgan", "renamed", cwd=tmp_path)
    with pytest.raises(FileNotFoundError):
        gc_run("otter", cwd=tmp_path)
    assert a.is_dir() and b.is_dir() and c.is_dir()

    # Exact identifiers do resolve: full id, short id, slug, directory name, path.
    assert resolve_destructive_target("3" * 32, cwd=tmp_path) == c.resolve()
    assert resolve_destructive_target("33333333", cwd=tmp_path) == c.resolve()
    assert resolve_destructive_target("calm-otter", cwd=tmp_path) == c.resolve()
    assert resolve_destructive_target("wgan-calm-otter", cwd=tmp_path) == c.resolve()
    assert resolve_destructive_target("runs/wgan-calm-otter/manifest.json", cwd=tmp_path) == c.resolve()


def test_ps_and_rm_agree_on_a_shared_legacy_slug(tmp_path):
    """Review H3: two pre-2.0 runs whose summaries both say amber-fox are never silently renamed."""
    runs = tmp_path / "runs"
    old = _make_run(runs / "wgan-amber-fox", run_id="4" * 32, name="wgan")
    live = _make_run(runs / "dcgan-amber-fox", run_id="5" * 32, name="dcgan")
    register_run(old)
    with run_lock(live):
        rows = list_runs(cwd=tmp_path, sort="updated")
        assert {r["slug"] for r in rows} == {"amber-fox"}
        with pytest.raises(ValueError, match="Ambiguous run reference 'amber-fox' matches 2 runs"):
            remove_run("amber-fox", cwd=tmp_path)
    assert old.is_dir() and live.is_dir()
    # Every unique slug ps shows resolves to the run ps showed it for.
    reserved = _make_run(runs / "gan-crisp-heron", run_id="6" * 32, slug="crisp-heron")
    for row in list_runs(cwd=tmp_path, sort="updated"):
        if row["slug"] == "crisp-heron":
            assert resolve_destructive_target(row["slug"], cwd=tmp_path) == Path(row["run_dir"]) == reserved.resolve()


def test_reserve_run_slug_is_unique_and_persisted_once(tmp_path):
    runs = tmp_path / "runs"
    taken = _make_run(runs / "dcgan-amber-fox", run_id="7" * 32)
    register_run(taken)
    new_dir = runs / "wgan-amber-fox"
    new_dir.mkdir()
    slug = reserve_run_slug(new_dir, "8" * 32)
    assert slug != "amber-fox"
    # The reservation survives a scan before the trainer writes its manifest.
    list_registered_runs(cwd=tmp_path)
    data = json.loads((get_registry_dir(tmp_path) / "runs.json").read_text(encoding="utf-8"))
    assert data["runs"][str(new_dir.resolve())]["slug"] == slug
    assert reserve_run_slug(new_dir, "8" * 32) == slug


def test_generate_run_slug_avoids_registered_slugs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    register_run(_make_run(runs / "dcgan-amber-fox", run_id="9" * 32))
    draws = iter([bytes([0, 27]), bytes([1, 3])])  # amber-fox (taken), then arctic-aurora
    monkeypatch.setattr(registry.os, "urandom", lambda n: next(draws))
    assert registry.generate_run_slug("wgan", parent_dir=runs) == ("wgan-arctic-aurora", "arctic-aurora")


# --------------------------------------------------------------------------- gc / prune (#3, M4, M5)

def test_gc_dir_only_touches_real_runs(tmp_path):
    notrun = tmp_path / "notrun"
    (notrun / "views").mkdir(parents=True)
    (notrun / "views" / "important.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not a HyperGAN run directory"):
        gc_run(str(notrun), cwd=tmp_path)
    with pytest.raises(ValueError, match="Name the run"):
        gc_run(None, cwd=tmp_path)
    assert (notrun / "views" / "important.txt").is_file()


def test_gc_all_needs_a_workspace_and_stays_inside_it(tmp_path):
    ws = tmp_path / "project"
    other = tmp_path / "other-project" / "runs"
    inside = _make_run(ws / "runs" / "in-ws", run_id="c" * 32, train_steps=30, views=True)
    outside = _make_run(other / "elsewhere", run_id="d" * 32, train_steps=30, views=True)
    register_run(inside)
    register_run(outside)
    before = (outside / "events.jsonl").read_bytes()

    home = tmp_path / "empty-home"
    home.mkdir()
    with pytest.raises(ValueError, match="needs a workspace"):
        gc_all(root=None, cwd=home)

    (ws / "hypergan.toml").write_text("[workspace]\n", encoding="utf-8")
    cleaned = {Path(r["run_dir"]) for r in gc_all(root=None, cwd=ws) if not r.get("skipped")}
    assert cleaned == {inside.resolve()}
    assert not (inside / "views").exists()
    assert (outside / "views" / "cache.jsonl").is_file()
    assert (outside / "events.jsonl").read_bytes() == before


def test_gc_dry_run_reports_the_measured_saving(tmp_path):
    run = _make_run(tmp_path / "runs" / "measured", run_id="e" * 32, train_steps=200, views=True)
    dry = gc_run(str(run), dry_run=True, stride=10, cwd=tmp_path)
    real = gc_run(str(run), stride=10, cwd=tmp_path)
    assert dry["events_reclaimed"] == real["events_reclaimed"] > 0
    assert dry["views_reclaimed"] == real["views_reclaimed"] == 500


def test_compaction_holds_the_run_lock(tmp_path, monkeypatch):
    """A trainer cannot start (and append to the replaced inode) while events are being compacted."""
    run = _make_run(tmp_path / "runs" / "locked", run_id="f" * 32, train_steps=40)
    real = inspection._compact_events_locked
    attempts = []

    def compact_while_probing(run_dir, stride, **kwargs):
        with pytest.raises(RuntimeError, match="already locked"):
            with run_lock(run_dir):
                pass
        attempts.append(run_dir)
        return real(run_dir, stride, **kwargs)

    monkeypatch.setattr(inspection, "_compact_events_locked", compact_while_probing)
    assert compact_events(str(run), stride=10, cwd=tmp_path) > 0
    assert gc_run(str(run), stride=10, cwd=tmp_path)["dry_run"] is False
    assert attempts == [run.resolve(), run.resolve()]


def test_compaction_keeps_train_events_without_a_step(tmp_path):
    run = _make_run(tmp_path / "runs" / "stepless", run_id="0" * 32, train_steps=20)
    with (run / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "train", "attempt_id": "0001", "metrics": {}}) + "\n")
    compact_events(str(run), stride=10, cwd=tmp_path)
    kept = [json.loads(ln) for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(ev.get("event") == "train" and "step" not in ev for ev in kept)


# --------------------------------------------------------------------------- liveness and signalling (M6, L1, L8)

@needs_proc
def test_find_live_pid_matches_exact_run_paths_only(tmp_path):
    runs = tmp_path / "runs"
    cifar = _make_run(runs / "cifar", run_id="1a" * 16)
    cifar_v2 = _make_run(runs / "cifar-v2", run_id="2b" * 16)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "hypergan", "train", str(cifar_v2)])
    holder = None
    try:
        deadline = time.monotonic() + 5
        while find_live_pid_for_run(cifar_v2) != sleeper.pid and time.monotonic() < deadline:
            time.sleep(0.02)
        assert find_live_pid_for_run(cifar_v2) == sleeper.pid
        assert find_live_pid_for_run(cifar) is None
        holder = _lock_holder(cifar)
        assert find_live_pid_for_run(cifar) == holder.pid
    finally:
        _reap(sleeper, *([holder] if holder else []))


@needs_proc
def test_stop_signals_the_lock_holder_not_a_stale_recorded_pid(tmp_path):
    run = _make_run(tmp_path / "runs" / "stale-pid", run_id="3c" * 16)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    holder = _lock_holder(run)
    try:
        (run / "launch").mkdir()
        (run / "launch" / "0001.json").write_text(json.dumps({"pid": bystander.pid}), encoding="utf-8")
        result = stop_run(str(run), timeout=5.0, cwd=tmp_path)
        assert result["pid"] == holder.pid and result["status"] == "stopped"
        assert holder.wait(timeout=5) != 0
        assert bystander.poll() is None
    finally:
        _reap(bystander, holder)


@needs_proc
def test_wait_for_exit_waits_for_a_killed_trainer(tmp_path):
    run = _make_run(tmp_path / "runs" / "killed", run_id="4d" * 16)
    holder = _lock_holder(run, ignore_sigint=True)
    try:
        assert inspection._wait_for_exit(holder.pid, run, timeout=0.2) is False
        os.kill(holder.pid, signal.SIGKILL)
        assert inspection._wait_for_exit(holder.pid, run, timeout=5.0) is True
        assert check_run_active(run)[0] is False
    finally:
        _reap(holder)


def test_rm_force_does_not_delete_while_the_process_is_still_alive(tmp_path, monkeypatch):
    run = _make_run(tmp_path / "runs" / "survivor", run_id="5e" * 16)
    calls = []
    monkeypatch.setattr(inspection, "stop_run", lambda target, *, force, cwd: calls.append(target) or {"status": "stopping"})
    with run_lock(run):
        with pytest.raises(RuntimeError, match="still running after a forced stop"):
            remove_run(str(run), force=True, cwd=tmp_path)
    assert calls == [run.resolve()] and (run / "manifest.json").is_file()


def test_rm_force_refuses_an_active_remote_run(tmp_path, monkeypatch):
    run = _make_run(tmp_path / "runs" / "remote", run_id="6f" * 16)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(status="running", hostname="some-other-node")
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(inspection, "stop_run", lambda *a, **k: pytest.fail("must not stop and delete a remote run"))
    with pytest.raises(RuntimeError, match="active on some-other-node"):
        remove_run(str(run), force=True, cwd=tmp_path)
    assert run.is_dir()


def test_retry_lock_survives_a_concurrent_liveness_probe(tmp_path):
    """Review L1: a probe holding .run.lock for an instant must not fail a trainer start."""
    run = tmp_path / "probed"
    run.mkdir()
    probing = threading.Event()

    def probe():
        with run_lock(run):
            probing.set()
            time.sleep(0.15)

    t = threading.Thread(target=probe)
    t.start()
    probing.wait(5)
    with pytest.raises(RuntimeError, match="already locked"):
        with run_lock(run):
            pass
    with retry_lock(lambda: run_lock(run), seconds=2.0):
        assert check_run_active(run)[0] is True
    t.join()


# --------------------------------------------------------------------------- tag (M7)

def test_tag_on_an_active_run_survives_the_trainer_republishing_its_manifest(tmp_path):
    run = _make_run(tmp_path / "runs" / "active-tag", run_id="7a" * 16, name="dcgan", slug="lucid-heron")
    manifest_before = (run / "manifest.json").read_bytes()
    with run_lock(run):
        result = tag_run("lucid-heron", "baseline-v1", cwd=tmp_path)
        assert result["name"] == "baseline-v1"
        # The trainer owns manifest.json while it runs: tag never rewrites it.
        assert (run / "manifest.json").read_bytes() == manifest_before
        # The trainer republishes its in-memory manifest, then its summary (and registry entry).
        (run / "manifest.json").write_bytes(manifest_before)
        write_summary_json(run)
        assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["name"] == "baseline-v1"
        listed = {e["run_dir"]: e for e in list_registered_runs(cwd=tmp_path)}
        assert listed[str(run.resolve())]["name"] == "baseline-v1"
        assert resolve_destructive_target("baseline-v1", cwd=tmp_path) == run.resolve()


# --------------------------------------------------------------------------- logs -f (M8)

def test_logs_follow_keeps_an_event_that_was_mid_write(tmp_path, monkeypatch):
    run = _make_run(tmp_path / "runs" / "following", run_id="8b" * 16)
    events = run / "events.jsonl"
    events.write_text(json.dumps({"event": "train", "step": 1, "metrics": {"loss/g_total": 1.0}}) + "\n", encoding="utf-8")
    step2 = json.dumps({"event": "train", "step": 2, "metrics": {"loss/g_total": 2.0}}) + "\n"
    with events.open("a", encoding="utf-8") as f:
        f.write(step2[:20])  # the trainer is in the middle of writing step 2
    rest = step2[20:] + json.dumps({"event": "train", "step": 3}) + "\n" + json.dumps({"event": "complete", "step": 3}) + "\n"

    real_sleep = time.sleep
    appended = []

    def sleep_then_finish_write(seconds):
        if not appended:
            with events.open("a", encoding="utf-8") as f:
                f.write(rest)
            appended.append(True)
        real_sleep(0)

    monkeypatch.setattr(time, "sleep", sleep_then_finish_write)
    for raw in (False, True):
        events.write_text(events.read_text(encoding="utf-8").split("\n")[0] + "\n" + step2[:20], encoding="utf-8")
        appended.clear()
        out = StringIO()
        with run_lock(run):
            lines = read_run_logs(str(run), follow=True, stream=out, cwd=tmp_path, **({"raw": True} if raw else {}))
        if raw:
            assert [json.loads(ln)["step"] for ln in lines] == [1, 2, 3, 3]
        else:
            assert any("[      2/" in ln or "step       2" in ln for ln in lines), lines
            assert lines[-1].startswith("[complete]")


# --------------------------------------------------------------------------- registry state (M9, L3, L4)

def test_scans_do_not_hold_the_registry_lock_and_never_dedupe_slugs(tmp_path, monkeypatch):
    for i in range(3):
        register_run(_make_run(tmp_path / "runs" / f"run-{i}", run_id=f"{i:x}" * 32))
    lock_path = get_registry_dir(tmp_path) / "registry.lock"
    real_extract = registry._extract_run_entry
    probes = []

    def extract_and_probe(*args, **kwargs):
        with registry.file_lock(lock_path, timeout=0):  # raises TimeoutError if the scan held it
            probes.append(True)
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(registry, "_extract_run_entry", extract_and_probe)
    monkeypatch.setattr(registry, "_ensure_unique_slug", lambda *a, **k: pytest.fail("slugs are not re-deduplicated on read"))
    (tmp_path / "runs" / "run-0" / "summary.json").touch()
    assert len(list_registered_runs(cwd=tmp_path)) == 3
    assert len(list_runs(cwd=tmp_path)) == 3
    assert probes


def test_corrupt_registry_is_moved_aside_with_a_warning(tmp_path):
    run = _make_run(tmp_path / "runs" / "survives", run_id="9c" * 16)
    register_run(run)
    reg_file = get_registry_dir(tmp_path) / "runs.json"
    reg_file.write_text('{"runs": {', encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="was corrupt; moved it to"):
        listed = list_registered_runs(cwd=tmp_path)
    backups = list(reg_file.parent.glob("runs.json.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == '{"runs": {'
    assert [e["run_id"] for e in listed] == ["9c" * 16]
    assert json.loads(reg_file.read_text(encoding="utf-8"))["runs"]


def test_roots_on_a_missing_mount_are_kept_until_they_expire(tmp_path):
    mount = tmp_path / "external"
    run = _make_run(mount / "runs" / "offline-run", run_id="ad" * 16, slug="polar-reef")
    register_run(run)
    reg_file = get_registry_dir(tmp_path) / "runs.json"
    root = str((mount / "runs").resolve())
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()

    unplugged = tmp_path / "unplugged"
    mount.rename(unplugged)  # the drive is unmounted
    assert list_registered_runs(cwd=elsewhere) == []
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    assert root in data["roots"] and root in data["missing_roots"]
    assert str(run.resolve()) in data["runs"]

    unplugged.rename(mount)  # and plugged back in
    listed = list_registered_runs(cwd=elsewhere)
    assert [(e["run_dir"], e["slug"]) for e in listed] == [(str(run.resolve()), "polar-reef")]
    assert "missing_roots" not in json.loads(reg_file.read_text(encoding="utf-8"))

    mount.rename(unplugged)
    list_registered_runs(cwd=elsewhere)
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["missing_roots"][root] = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    reg_file.write_text(json.dumps(data), encoding="utf-8")
    list_registered_runs(cwd=elsewhere)
    assert root not in json.loads(reg_file.read_text(encoding="utf-8"))["roots"]


def test_status_ps_and_registry_agree_on_an_unrecorded_stop(tmp_path):
    """Review L7: a run recorded 'running', inactive, copied without its .run.lock, is 'stopped' everywhere."""
    run = _make_run(tmp_path / "runs" / "copied", run_id="da" * 16)
    for name in ("manifest.json", "summary.json"):
        doc = json.loads((run / name).read_text(encoding="utf-8"))
        doc["status"] = "running"
        (run / name).write_text(json.dumps(doc), encoding="utf-8")
    assert not (run / ".run.lock").exists()
    assert inspection.run_status(str(run), cwd=tmp_path)["status"] == "stopped"
    assert [r["status"] for r in list_runs(cwd=tmp_path)] == ["stopped"]
    assert [e["status"] for e in list_registered_runs(cwd=tmp_path)] == ["stopped"]


def test_path_containment_is_by_component():
    assert registry._is_within(Path("/data/runs"), Path("/data"))
    assert not registry._is_within(Path("/data2/runs"), Path("/data"))


def test_unregister_run_matches_exact_paths_and_ids_only(tmp_path):
    a = _make_run(tmp_path / "runs" / "dcgan-amber-fox", run_id="be" * 16, name="dcgan")
    b = _make_run(tmp_path / "runs" / "dcgan-bold-hawk", run_id="cf" * 16, name="dcgan")
    register_run(a)
    register_run(b)
    for loose in ("dcgan", "amber-fox", "bebe", "dcgan-amber-fox"):
        assert registry.unregister_run(loose, cwd=tmp_path) is False
    assert registry.unregister_run(a, cwd=tmp_path) is True
    data = json.loads((get_registry_dir(tmp_path) / "runs.json").read_text(encoding="utf-8"))
    assert list(data["runs"]) == [str(b.resolve())]

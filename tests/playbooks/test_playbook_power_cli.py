"""UX Playbook 2: Power CLI & AI Developer (Isolation, Overrides, Compare, Queue & Chaos).

Covers:
- State isolation with `HYPERGAN_STATE_DIR` in an empty folder (`ps -a --json` returns 0 runs and does not import `../training-runs`)
- `hypergan validate <config> -s train.batch_size=32 -s optimizer.lr=0.0002 --json`
- `hypergan runs compare` with default `--metric` / `--at`, `--metric loss_g` alias, and `--json`
  (both `runs compare --json` and `runs --json compare`)
- `hypergan queue` (`queue add -n job1 --steps 10 -s train.lr=0.0005`, `queue --json ls`,
  `queue stop` clean error message, `queue stop --all`)
- Filesystem chaos self-healing: corrupted `summary.json` (`{bad`), renamed run directory (`mv`),
  and deleted run directory (`rm -rf`) targeted by slug
- Stopped/completed runs have `"pid": None` in `ps -a --json` and slug collisions are disambiguated.
"""

import json
from pathlib import Path
import shutil

import pytest

import hypergan.api  # noqa: F401
import hypergan.cli as cli
from hypergan.registry import register_run
from hypergan.summary import write_summary_json


def _invoke_cli(capsys, *args: str) -> tuple[int, str, str]:
    capsys.readouterr()
    try:
        rc = cli.main([str(a) for a in args])
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return rc or 0, captured.out, captured.err


def _make_mock_run(
    run_dir: Path,
    *,
    run_id: str,
    name: str,
    slug: str | None = None,
    status_val: str = "completed",
    steps: int = 20,
    total_steps: int = 20,
    loss_g: float = 0.85,
    loss_d: float = 0.45,
    stale_pid: int | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "name": name,
        "status": status_val,
        "steps": steps,
        "total_steps": total_steps,
        "steps_per_second": 25.0,
        "device": "cpu",
        "config": {
            "name": name,
            "train": {"device": "cpu", "steps": total_steps},
            "training": {"device": "cpu", "steps": total_steps},
        },
    }
    if slug:
        manifest["slug"] = slug
    if stale_pid is not None:
        manifest["pid"] = stale_pid
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    events = [
        {"event": "start", "step": 0, "attempt_id": "0001", "run_id": run_id},
        {
            "event": "train",
            "step": max(1, steps // 2),
            "steps_per_second": 25.0,
            "metrics": {"loss/g_total": loss_g + 0.2, "loss/d_total": loss_d + 0.1},
        },
        {
            "event": "train",
            "step": steps,
            "steps_per_second": 25.0,
            "metrics": {"loss/g_total": loss_g, "loss/d_total": loss_d},
        },
        {"event": status_val, "step": steps},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    write_summary_json(run_dir, status=status_val)
    if stale_pid is not None and (run_dir / "summary.json").is_file():
        sdata = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        sdata["pid"] = stale_pid
        (run_dir / "summary.json").write_text(json.dumps(sdata, indent=2), encoding="utf-8")
    return run_dir


def test_state_isolation_with_hypergan_state_dir(tmp_path, monkeypatch, capsys):
    """`HYPERGAN_STATE_DIR` isolates state in an empty folder without importing `../training-runs`."""
    state_dir = tmp_path / "isolated_state_dir"
    state_dir.mkdir(parents=True, exist_ok=True)
    empty_cwd = tmp_path / "empty_workspace"
    empty_cwd.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("HYPERGAN_HOME", raising=False)
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(empty_cwd)

    rc, out, err = _invoke_cli(capsys, "ps", "-a", "--json")
    assert rc == 0, err
    ps_data = json.loads(out)
    assert ps_data["runs"] == []
    assert ps_data["total_known"] == 0


def test_validate_with_set_overrides_and_optimizer_alias(tmp_path, monkeypatch, capsys):
    """`hypergan validate <config> -s train.batch_size=32 -s optimizer.lr=0.0002 --json` succeeds."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    cfg_path = tmp_path / "toy.toml"
    rc_new, _, err_new = _invoke_cli(capsys, "new", str(cfg_path), "--device", "cpu")
    assert rc_new == 0, err_new

    rc_val, out_val, err_val = _invoke_cli(
        capsys,
        "validate",
        str(cfg_path),
        "-s",
        "train.batch_size=32",
        "-s",
        "optimizer.lr=0.0002",
        "--json",
    )
    assert rc_val == 0, err_val
    resolved = json.loads(out_val)
    assert resolved["training"]["batch_size"] == 32
    assert resolved["optimizer"]["lr"] == pytest.approx(0.0002)


def test_runs_compare_defaults_aliases_and_json(tmp_path, monkeypatch, capsys):
    """`hypergan runs compare` works with default `--metric`/`--at`, `--metric loss_g` alias, and `--json`."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    runs_root = tmp_path / "runs"
    r1 = _make_mock_run(
        runs_root / "exp-alpha",
        run_id="11111111222222223333333344444444",
        name="exp-alpha",
        slug="amber-anchor",
        steps=20,
        loss_g=0.75,
    )
    r2 = _make_mock_run(
        runs_root / "exp-beta",
        run_id="55555555666666667777777788888888",
        name="exp-beta",
        slug="arctic-atlas",
        steps=20,
        loss_g=0.55,
    )
    register_run(r1)
    register_run(r2)

    # 1. Default --metric and --at
    rc_def, out_def, err_def = _invoke_cli(capsys, "runs", "compare", "amber-anchor", "arctic-atlas")
    assert rc_def == 0, err_def
    assert "exp-alpha" in out_def
    assert "exp-beta" in out_def

    # 2. `runs compare --metric loss_g --json`
    rc_json1, out_json1, err_json1 = _invoke_cli(
        capsys, "runs", "compare", "amber-anchor", "arctic-atlas", "--metric", "loss_g", "--json"
    )
    assert rc_json1 == 0, err_json1
    comp1 = json.loads(out_json1)
    assert len(comp1["runs"]) == 2
    latest_row = comp1["matrix"].get("latest", {})
    assert latest_row["11111111222222223333333344444444"]["value"] == pytest.approx(0.75)
    assert latest_row["55555555666666667777777788888888"]["value"] == pytest.approx(0.55)

    # 3. `runs --json compare` (parent --json flag before subcommand)
    rc_json2, out_json2, err_json2 = _invoke_cli(
        capsys, "runs", "--json", "compare", "amber-anchor", "arctic-atlas", "--metric", "loss_g"
    )
    assert rc_json2 == 0, err_json2
    comp2 = json.loads(out_json2)
    assert len(comp2["runs"]) == 2


def test_queue_cli_add_ls_and_stop(tmp_path, monkeypatch, capsys):
    """`hypergan queue` supports `-n`, `--steps`, `-s`, `queue --json ls`, clean `queue stop` error, and `queue stop --all`."""
    state_dir = tmp_path / "state"
    queue_file = tmp_path / "queue.json"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("HYPERGAN_QUEUE_FILE", str(queue_file))
    monkeypatch.chdir(tmp_path)

    cfg_path = tmp_path / "job_model.toml"
    rc_new, _, err_new = _invoke_cli(capsys, "new", str(cfg_path), "--device", "cpu")
    assert rc_new == 0, err_new

    # Enqueue job with -n, --steps, -s
    rc_add, out_add, err_add = _invoke_cli(
        capsys,
        "queue",
        "add",
        str(cfg_path),
        "-n",
        "job1",
        "--steps",
        "10",
        "-s",
        "train.lr=0.0005",
    )
    assert rc_add == 0, err_add
    assert "job1" in out_add

    # `queue --json ls` (parent --json flag)
    rc_ls, out_ls, err_ls = _invoke_cli(capsys, "queue", "--json", "ls")
    assert rc_ls == 0, err_ls
    jobs = json.loads(out_ls)
    assert isinstance(jobs, list) and len(jobs) == 1
    assert jobs[0]["name"] == "job1"
    assert jobs[0]["status"] == "queued"

    # `queue stop` without arguments gives a clean CLI error message (not "stop_all=True")
    rc_stop_err, out_stop_err, err_stop_err = _invoke_cli(capsys, "queue", "stop")
    assert rc_stop_err != 0
    combined_err = (out_stop_err + err_stop_err)
    assert "stop_all=True" not in combined_err
    assert "--all" in combined_err or "job_id" in combined_err.lower()

    # `queue stop --all` cancels queued jobs cleanly
    rc_stop_all, out_stop_all, err_stop_all = _invoke_cli(capsys, "queue", "stop", "--all")
    assert rc_stop_all == 0, err_stop_all

    rc_ls2, out_ls2, _ = _invoke_cli(capsys, "queue", "ls", "--json")
    assert rc_ls2 == 0
    jobs_after = json.loads(out_ls2)
    assert jobs_after[0]["status"] == "cancelled"


def test_filesystem_chaos_self_healing(tmp_path, monkeypatch, capsys):
    """Corrupted summary.json, renamed directory (`mv`), and deleted directory (`rm -rf`) self-heal cleanly."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    runs_root = tmp_path / "runs"
    r1 = _make_mock_run(
        runs_root / "exp1",
        run_id="aaaa1111bbbb2222cccc3333dddd4444",
        name="exp1",
        slug="cobalt-comet",
        steps=30,
        loss_g=0.62,
    )
    r2 = _make_mock_run(
        runs_root / "exp2",
        run_id="eeee5555ffff66660000777788889999",
        name="exp2",
        slug="crimson-crane",
        steps=40,
        loss_g=0.48,
    )
    register_run(r1)
    register_run(r2)

    # 1. Corrupt summary.json on r1
    (r1 / "summary.json").write_text("{bad", encoding="utf-8")
    rc_st, out_st, err_st = _invoke_cli(capsys, "status", "cobalt-comet", "--json")
    assert rc_st == 0, err_st
    st_data = json.loads(out_st)
    assert st_data["step"] == 30
    assert st_data["status"] == "completed"

    # 2. Rename r1 directory (`mv runs/exp1 runs/exp1_renamed`)
    r1_renamed = runs_root / "exp1_renamed"
    r1.rename(r1_renamed)
    rc_mv, out_mv, err_mv = _invoke_cli(capsys, "status", "cobalt-comet", "--json")
    assert rc_mv == 0, err_mv
    mv_data = json.loads(out_mv)
    assert Path(mv_data["run_dir"]).resolve() == r1_renamed.resolve()

    # 3. Delete r2 directory (`rm -rf runs/exp2`)
    shutil.rmtree(r2)
    rc_del, out_del, err_del = _invoke_cli(capsys, "status", "crimson-crane")
    assert rc_del != 0
    assert "no longer exists" in (out_del + err_del) or "not found" in (out_del + err_del).lower()

    # Verify r2 was pruned from `ps -a --json`
    rc_ps, out_ps, err_ps = _invoke_cli(capsys, "ps", "-a", "--json")
    assert rc_ps == 0, err_ps
    ps_runs = json.loads(out_ps)["runs"]
    assert [r["slug"] for r in ps_runs] == ["cobalt-comet"]


def test_inactive_pid_is_none_and_slug_collisions_disambiguated(tmp_path, monkeypatch, capsys):
    """Stopped/completed runs report `pid: None` in `ps -a --json` and colliding slugs are disambiguated."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    runs_root = tmp_path / "runs"
    r1 = _make_mock_run(
        runs_root / "collision-a",
        run_id="1234567890abcdef1234567890abcdef",
        name="collision-a",
        slug="keen-breeze",
        status_val="stopped",
        stale_pid=999991,
    )
    r2 = _make_mock_run(
        runs_root / "collision-b",
        run_id="fedcba0987654321fedcba0987654321",
        name="collision-b",
        slug="keen-breeze",
        status_val="completed",
        stale_pid=999992,
    )
    e1 = register_run(r1, slug="keen-breeze")
    e2 = register_run(r2, slug="keen-breeze")
    assert e1["slug"] != e2["slug"]

    rc_ps, out_ps, err_ps = _invoke_cli(capsys, "ps", "-a", "--json")
    assert rc_ps == 0, err_ps
    runs = json.loads(out_ps)["runs"]
    assert len(runs) == 2
    slugs = {r["slug"] for r in runs}
    assert len(slugs) == 2
    for r in runs:
        assert r["active"] is False
        assert r["pid"] is None


def test_top_level_cli_subcommands_and_grouped_help(tmp_path, monkeypatch, capsys):
    """Verify top-level `compare`, `diff`, `doctor`, `report`, `queue clear`, and zero-arg help."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    # 1. Zero-arg help prints usage and returns 0
    rc_zero, out_zero, _ = _invoke_cli(capsys)
    assert rc_zero == 0
    assert "usage: hypergan" in out_zero
    assert "compare" in out_zero
    assert "diff" in out_zero
    assert "doctor" in out_zero
    assert "report" in out_zero

    # Setup two mock runs
    runs_root = tmp_path / "runs"
    r1 = _make_mock_run(
        runs_root / "exp-a",
        run_id="aaaa0001bbbb0002cccc0003dddd0004",
        name="exp-a",
        slug="amber-anchor",
        steps=10,
        loss_g=0.6,
    )
    r2 = _make_mock_run(
        runs_root / "exp-b",
        run_id="eeee0005ffff0006aaaa0007bbbb0008",
        name="exp-b",
        slug="arctic-atlas",
        steps=10,
        loss_g=0.4,
    )
    register_run(r1)
    register_run(r2)

    # 2. Top-level compare
    rc_comp, out_comp, err_comp = _invoke_cli(capsys, "compare", "amber-anchor", "arctic-atlas")
    assert rc_comp == 0, err_comp
    assert "exp-a" in out_comp
    assert "exp-b" in out_comp

    # Missing runs exit non-zero
    rc_miss, out_miss, err_miss = _invoke_cli(capsys, "compare", "nonexistent-1", "nonexistent-2")
    assert rc_miss != 0

    # 3. Top-level diff between two configs
    cfg1 = tmp_path / "model1.toml"
    cfg2 = tmp_path / "model2.toml"
    rc_n1, _, err_n1 = _invoke_cli(capsys, "new", str(cfg1), "--device", "cpu")
    assert rc_n1 == 0, err_n1
    rc_n2, _, err_n2 = _invoke_cli(capsys, "new", str(cfg2), "--device", "cpu")
    assert rc_n2 == 0, err_n2

    rc_diff, out_diff, err_diff = _invoke_cli(capsys, "diff", str(cfg1), str(cfg2), "--json")
    assert rc_diff == 0, err_diff
    diff_data = json.loads(out_diff)
    assert "identical" in diff_data or "differences" in diff_data

    # 4. Top-level doctor
    rc_doc, out_doc, err_doc = _invoke_cli(capsys, "doctor", "--json")
    assert rc_doc == 0, err_doc
    doc_data = json.loads(out_doc)
    assert "python" in doc_data or "platform" in doc_data or "system" in doc_data

    # 5. Top-level report
    rc_rep, out_rep, err_rep = _invoke_cli(capsys, "report", "amber-anchor", "arctic-atlas", "--json")
    assert rc_rep == 0, err_rep
    rep_data = json.loads(out_rep)
    assert isinstance(rep_data, list) and len(rep_data) == 2

    # 6. queue clear cancels queued jobs (alias for queue stop --all)
    q_file = tmp_path / "queue.json"
    monkeypatch.setenv("HYPERGAN_QUEUE_FILE", str(q_file))
    rc_add, _, _ = _invoke_cli(capsys, "queue", "add", str(cfg1), "-n", "clear_job", "--steps", "5")
    assert rc_add == 0
    rc_clr, out_clr, err_clr = _invoke_cli(capsys, "queue", "clear")
    assert rc_clr == 0, err_clr
    rc_ls, out_ls, _ = _invoke_cli(capsys, "queue", "ls", "--json")
    assert rc_ls == 0
    jobs_after = json.loads(out_ls)
    assert jobs_after[0]["status"] == "cancelled"


def test_queue_run_execution_and_persistence(tmp_path, monkeypatch, capsys):
    """Verify `queue run --once` executes all queued jobs sequentially, respects -n, --steps, -s, --no-web, and records run_dir."""
    state_dir = tmp_path / "state"
    queue_file = tmp_path / "queue.json"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("HYPERGAN_QUEUE_FILE", str(queue_file))
    monkeypatch.chdir(tmp_path)

    cfg_path = tmp_path / "job_model.toml"
    rc_new, _, err_new = _invoke_cli(capsys, "new", str(cfg_path), "--template", "gaussian-grid", "--device", "cpu")
    assert rc_new == 0, err_new

    # Enqueue two jobs with overrides and --no-web
    rc_add1, _, err_add1 = _invoke_cli(
        capsys,
        "queue",
        "add",
        str(cfg_path),
        "-n",
        "worker_job_1",
        "--steps",
        "1",
        "-s",
        "train.device=cpu",
        "--no-web",
    )
    assert rc_add1 == 0, err_add1

    rc_add2, _, err_add2 = _invoke_cli(
        capsys,
        "queue",
        "add",
        str(cfg_path),
        "-n",
        "worker_job_2",
        "--steps",
        "1",
        "-s",
        "train.device=cpu",
        "--no-web",
    )
    assert rc_add2 == 0, err_add2

    # Execute queue with --once
    rc_run, out_run, err_run = _invoke_cli(capsys, "queue", "run", "--once", "-d", "cpu", "--no-web")
    assert rc_run == 0, f"queue run failed: {err_run}"

    # Verify queue.json records execution and valid run_dir
    assert queue_file.is_file()
    q_data = json.loads(queue_file.read_text(encoding="utf-8"))
    jobs = q_data.get("jobs", [])
    assert len(jobs) == 2
    for job in jobs:
        assert job["status"] in ("completed", "stopped")
        assert job.get("run_dir") is not None
        assert Path(job["run_dir"]).is_dir()
        manifest_p = Path(job["run_dir"]) / "manifest.json"
        assert manifest_p.is_file()


def test_event_compaction_integrity_and_metrics_recovery(tmp_path, monkeypatch, capsys):
    """Verify `hypergan gc <slug>` renumbers sequence contiguously per attempt and restores metrics/events/observations."""
    import hypergan as hg

    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "non-contiguous-run"
    model = hg.override(hg.packaged("gaussian-grid"), {"train.steps": 2, "train.device": "cpu"})
    hg.train(model, run_dir, steps=2, checkpoint_every=2, previews=0, devices="cpu")
    st = hg.status(run_dir)
    slug = st["slug"]

    # Read events.jsonl and intentionally corrupt sequence numbers to non-contiguous
    events_p = run_dir / "events.jsonl"
    lines = [ln for ln in events_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]
    seq_offsets = [1, 5, 8, 12, 15, 20]
    for idx, ev in enumerate(events):
        ev["sequence"] = seq_offsets[idx] if idx < len(seq_offsets) else (ev.get("sequence", 1) + 10)
    events_p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    # Run hypergan gc
    rc_gc, out_gc, err_gc = _invoke_cli(capsys, "gc", slug)
    assert rc_gc == 0, err_gc

    # Verify events are renumbered contiguously 1, 2, 3, ...
    compacted_lines = [ln for ln in events_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    compacted_events = [json.loads(line) for line in compacted_lines]
    sequences = [ev["sequence"] for ev in compacted_events]
    assert sequences == list(range(1, len(compacted_events) + 1))

    # Verify metrics, events, observations, and project succeed without sequence error
    m_data = hg.metrics(slug)
    assert isinstance(m_data, dict)
    ev_data = hg.events(slug)
    ev_list = ev_data.get("events") if isinstance(ev_data, dict) else ev_data
    assert isinstance(ev_list, list) and len(ev_list) == len(compacted_events)
    obs_data = hg.observations(slug)
    obs_list = obs_data.get("observations") if isinstance(obs_data, dict) else obs_data
    assert isinstance(obs_list, list)
    proj_res = hg.project(slug)
    assert proj_res is not None


def test_ps_best_fid_filter_and_corrupt_metadata_warning(tmp_path, monkeypatch, capsys):
    """Verify ps -a --json does not report best_fid for non-FID metrics, warns on corrupt metadata, and scans /tmp/runs."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.chdir(tmp_path)

    runs_root = tmp_path / "runs"
    # Run 1: best metric is sample_diversity, NOT fid
    r1 = _make_mock_run(
        runs_root / "div-run",
        run_id="dddd1111222233334444555566667777",
        name="div-run",
        slug="golden-gecko",
        steps=10,
    )
    s1 = json.loads((r1 / "summary.json").read_text(encoding="utf-8"))
    s1["best"] = {"metric": "sample_diversity", "value": 0.95, "step": 10}
    (r1 / "summary.json").write_text(json.dumps(s1, indent=2), encoding="utf-8")
    register_run(r1, slug="golden-gecko")

    # 1. ps -a --json reports best_fid as None for non-FID metrics
    rc_ps, out_ps, err_ps = _invoke_cli(capsys, "ps", "-a", "--json")
    assert rc_ps == 0, err_ps
    runs = json.loads(out_ps)["runs"]
    gecko = next(r for r in runs if r["slug"] == "golden-gecko")
    assert gecko["best_fid"] is None

    # Run 2: corrupt both manifest.json and summary.json, leaving events.jsonl
    r2 = _make_mock_run(
        runs_root / "corrupt-run",
        run_id="eeee1111222233334444555566667777",
        name="corrupt-run",
        slug="silver-swan",
        steps=5,
    )
    register_run(r2, slug="silver-swan")
    (r2 / "manifest.json").write_text("{corrupt json", encoding="utf-8")
    (r2 / "summary.json").write_text("{corrupt json", encoding="utf-8")

    # 2. hypergan status silver-swan emits warning in human output
    rc_st, out_st, err_st = _invoke_cli(capsys, "status", "silver-swan")
    assert rc_st == 0, err_st
    assert "Warnings:" in out_st
    assert "missing or corrupted" in out_st or "events.jsonl" in out_st

    # 3. Scanning in non-workspace /tmp only scans <cwd>/runs
    tmp_dir = Path("/tmp")
    from hypergan.inspection import list_runs
    runs_in_tmp = list_runs(root=None, cwd=tmp_dir)
    assert isinstance(runs_in_tmp, list)

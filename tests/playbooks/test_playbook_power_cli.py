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

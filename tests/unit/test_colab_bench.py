"""CPU-only unit tests for scripts/colab_bench.py (all Colab CLI calls mocked)."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "colab_bench.py"


def _load_colab_bench():
    spec = importlib.util.spec_from_file_location("colab_bench", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["colab_bench"] = module
    spec.loader.exec_module(module)
    return module


cb = _load_colab_bench()


def _make_run_tarball(
    tar_path: Path,
    *,
    run_name: str,
    step: int,
    status: str = "stopped",
    total_steps: int = 100,
    remote_prefix: str = "/content/runs",
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    remote_run_dir = f"{remote_prefix}/{run_name}"
    ckpt_name = f"0001-attempt-step-{step:08d}-deadbeef"
    manifest = {
        "schema_version": 1,
        "run_id": f"run-{run_name}",
        "status": status,
        "steps": step,
        "run_dir": remote_run_dir,
        "checkpoint_path": f"{remote_run_dir}/checkpoints/{ckpt_name}",
        "attempt_dir": f"{remote_run_dir}/attempts/0001-attempt",
        "exported_bundle_path": f"{remote_run_dir}/model.pt",
        "config": {"training": {"steps": total_steps}},
    }
    summary = {
        "run_id": f"run-{run_name}",
        "name": run_name,
        "status": status,
        "step": step,
        "total_steps": total_steps,
        "best_fid": 12.34,
        "best_step": step,
        "latest_fid": 12.34,
        "latest": {"loss/d_total": 0.42, "loss/g_total": 1.15, "diversity/ratio": 0.95},
        "evaluations": [{"step": step, "metric": "fid50k_train", "value": 12.34}],
        "health_alerts": [],
        "error": None,
    }
    files: dict[str, bytes] = {
        "manifest.json": (json.dumps(manifest) + "\n").encode("utf-8"),
        "summary.json": (json.dumps(summary) + "\n").encode("utf-8"),
        "model.json": b'{"networks": {}}\n',
        "events.jsonl": (json.dumps({"event": "train", "step": step}) + "\n").encode("utf-8"),
        "train.log": f"step {step}/{total_steps} loss=0.42\n".encode("utf-8"),
        "model.pt": b"fake-ema-bundle-bytes",
        "metrics/evaluations/step-eval/receipt.json": b'{"status":"complete"}\n',
        "previews/index.json": b'{"previews": []}\n',
        "attempts/0001-attempt/manifest.json": (json.dumps(manifest) + "\n").encode("utf-8"),
        "checkpoints/latest.json": (json.dumps({"checkpoint": ckpt_name}) + "\n").encode("utf-8"),
        f"checkpoints/{ckpt_name}/manifest.json": (json.dumps({"step": step}) + "\n").encode("utf-8"),
        f"checkpoints/{ckpt_name}/state.pt": b"checkpoint-state",
    }
    if extra_files:
        files.update(extra_files)

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel, data in files.items():
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tf.addfile(info, fileobj=__import__("io").BytesIO(data))
    return tar_path


# ---------------------------------------------------------------------------
# 1. Worker cap, GPU catalog & cost estimation
# ---------------------------------------------------------------------------


def test_worker_cap_enforcement():
    assert cb.validate_worker_count(1) == 1
    assert cb.validate_worker_count(3) == 3
    assert cb.validate_worker_count(2, max_workers=2) == 2

    with pytest.raises(ValueError, match="exceeds"):
        cb.validate_worker_count(4)
    with pytest.raises(ValueError, match="hard safety cap"):
        cb.validate_worker_count(4, max_workers=4, allow_over_cap=False)
    with pytest.raises(ValueError, match="exceeds --max-workers=2"):
        cb.validate_worker_count(3, max_workers=2)
    with pytest.raises(ValueError, match="positive integer"):
        cb.validate_worker_count(0)

    # Explicit override permits > 3 only when BOTH max_workers and allow_over_cap are set.
    assert cb.validate_worker_count(4, max_workers=4, allow_over_cap=True) == 4


def test_gpu_normalization_and_cost_estimation():
    assert cb.normalize_gpu("l4") == "L4"
    assert cb.normalize_gpu("A100") == "A100"
    assert cb.normalize_gpu("t4") == "T4"
    assert cb.normalize_gpu("h100") == "H100"
    assert cb.normalize_gpu("CPU") == "cpu"

    with pytest.raises(ValueError, match="Unsupported --gpu"):
        cb.normalize_gpu("V100")

    cost = cb.estimate_cost("L4", 1800.0)
    assert cost["gpu"] == "L4"
    assert cost["gpu_hours"] == pytest.approx(0.5)
    assert cost["estimated_cu"] == pytest.approx(0.97)

    a100_cost = cb.estimate_cost("A100", 3600.0, actual_cuda_model="NVIDIA A100-SXM4-40GB")
    assert a100_cost["cuda_model"] == "NVIDIA A100-SXM4-40GB"
    assert a100_cost["estimated_cu"] == pytest.approx(6.80)


# ---------------------------------------------------------------------------
# 2. OS process-group hard timeout (SIGTERM -> SIGKILL)
# ---------------------------------------------------------------------------


def test_run_with_hard_timeout_captures_output_and_logs(tmp_path):
    log_file = tmp_path / "cmd.log"
    rc, out = cb.run_with_hard_timeout(
        [sys.executable, "-c", "print('hello-colab')"],
        timeout_seconds=5.0,
        log_file=log_file,
        capture=True,
    )
    assert rc == 0
    assert "hello-colab" in out
    assert "hello-colab" in log_file.read_text(encoding="utf-8")


def test_run_with_hard_timeout_kills_sigterm_ignoring_process_group(tmp_path):
    # Child ignores SIGTERM and sleeps forever; must be hard-killed via SIGKILL -> rc=124.
    log_file = tmp_path / "hung.log"
    stubborn = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('started', flush=True); "
        "time.sleep(60)"
    )
    rc, out = cb.run_with_hard_timeout(
        [sys.executable, "-c", stubborn],
        timeout_seconds=0.35,
        kill_grace=0.25,
        log_file=log_file,
        capture=True,
    )
    assert rc == 124
    assert "[timeout] Hard-killing hung command" in out
    assert "[timeout] Hard-killing hung command" in log_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. VM-side watchdog script generation
# ---------------------------------------------------------------------------


def test_watchdog_arm_script_valid_ast_and_endpoints():
    script = cb.build_watchdog_arm_script(
        max_seconds=3600,
        idle_grace_seconds=300,
        stall_seconds=900,
    )
    ast.parse(script)
    assert "TBE_RUNTIME_ADDR" in script
    assert "/unassign" in script
    assert "CB_MAX_SECONDS" in script
    assert "CB_IDLE_GRACE_SECONDS" in script
    assert "CB_STALL_SECONDS" in script
    assert "events.jsonl" in script
    assert "summary.json" in script
    assert "start_new_session=True" in script

    # Extract embedded watchdog.py source and confirm it also parses cleanly as valid Python AST.
    start_marker = "watchdog_code = textwrap.dedent('''\n"
    end_marker = "''').strip()"
    assert start_marker in script and end_marker in script
    inner = script.split(start_marker, 1)[1].split(end_marker, 1)[0]
    ast.parse(__import__("textwrap").dedent(inner))

    with pytest.raises(ValueError):
        cb.build_watchdog_arm_script(max_seconds=0, idle_grace_seconds=300, stall_seconds=900)


# ---------------------------------------------------------------------------
# 4. Manifest parsing (JSON & TOML) and local run status inspection
# ---------------------------------------------------------------------------


def test_load_manifest_toml_and_json(tmp_path):
    cfg1 = tmp_path / "m1.toml"
    cfg2 = tmp_path / "m2.toml"
    cfg1.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    cfg2.write_text('format = "hypergan-model/1"\n', encoding="utf-8")

    toml_manifest = tmp_path / "queue.toml"
    toml_manifest.write_text(
        """
[defaults]
gpu = "A100"
stop_after_steps = 50000
chunk_steps = 10000
checkpoint_every = 5000
set_overrides = ["train.batch_size=256"]

[[jobs]]
run_name = "job-v12"
config = "m1.toml"

[[jobs]]
run_name = "job-adam"
config = "m2.toml"
gpu = "L4"
set_overrides = ["train.lr=0.0002"]
""",
        encoding="utf-8",
    )
    jobs = cb.load_manifest(toml_manifest, repo_root=tmp_path)
    assert [j.run_name for j in jobs] == ["job-v12", "job-adam"]
    assert jobs[0].gpu == "A100"
    assert jobs[1].gpu == "L4"
    assert jobs[0].stop_after_steps == 50000
    assert jobs[0].chunk_steps == 10000
    assert jobs[0].config == str(cfg1.resolve())
    assert jobs[1].set_overrides == ["train.lr=0.0002"]

    json_manifest = tmp_path / "queue.json"
    json_manifest.write_text(
        json.dumps([{"run_name": "j1", "config": "m1.toml", "stop_after_steps": 100}]),
        encoding="utf-8",
    )
    j_jobs = cb.load_manifest(json_manifest, repo_root=tmp_path, default_gpu="T4")
    assert len(j_jobs) == 1
    assert j_jobs[0].gpu == "T4"
    assert j_jobs[0].target_step() == 100

    # Duplicate run_name rejected
    dup_manifest = tmp_path / "dup.json"
    dup_manifest.write_text(
        json.dumps({"jobs": [{"run_name": "same", "config": "m1.toml"}, {"run_name": "same", "config": "m2.toml"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate run_name"):
        cb.load_manifest(dup_manifest, repo_root=tmp_path)


def test_inspect_local_run_skip_completed_and_resumable(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "stopped", "step": 50000, "total_steps": 200000, "best_fid": 8.4}),
        encoding="utf-8",
    )
    st = cb.inspect_local_run(run_dir, target_step=50000)
    assert st["completed"] is True
    assert st["resumable"] is False
    assert st["best_fid"] == pytest.approx(8.4)

    # Not yet at 100k target -> not completed; with checkpoint + manifest -> resumable
    st2 = cb.inspect_local_run(run_dir, target_step=100000)
    assert st2["completed"] is False
    assert st2["resumable"] is False

    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "abc"}), encoding="utf-8")
    ckpt_dir = run_dir / "checkpoints" / "0001-step-00050000"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "state.pt").write_bytes(b"ckpt")
    st3 = cb.inspect_local_run(run_dir, target_step=100000)
    assert st3["resumable"] is True
    assert st3["step"] == 50000


# ---------------------------------------------------------------------------
# 5. Artifact unpacking, path relocation, resume packing & telemetry analysis
# ---------------------------------------------------------------------------


def test_unpack_run_tarball_relocates_paths_and_rejects_traversal(tmp_path):
    tar_path = _make_run_tarball(tmp_path / "out.tar.gz", run_name="demo", step=50)
    dest = tmp_path / "runs" / "demo"
    info = cb.unpack_run_tarball(tar_path, dest, remote_run_dir="/content/runs/demo")
    assert info["has_summary"] and info["has_manifest"] and info["has_bundle"] and info["has_train_log"]
    assert info["manifests_relocated"] == 2

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_dir"] == str(dest.resolve())
    assert manifest["checkpoint_path"].startswith(str(dest.resolve()) + "/checkpoints/")
    assert manifest["exported_bundle_path"] == str(dest.resolve() / "model.pt")

    # Pack local run back for Colab resume and verify remote prefix restoration.
    resume_tar = tmp_path / "resume.tar.gz"
    cb.pack_local_run_for_resume(dest, resume_tar, remote_run_dir="/content/runs/demo")
    with tarfile.open(resume_tar, "r:gz") as tf:
        names = set(tf.getnames())
        assert "manifest.json" in names
        assert "checkpoints/latest.json" in names
        manifest_bytes = tf.extractfile("manifest.json").read()
        rm = json.loads(manifest_bytes.decode("utf-8"))
        assert rm["run_dir"] == "/content/runs/demo"
        assert rm["checkpoint_path"].startswith("/content/runs/demo/checkpoints/")

    # Unsafe path traversal archive MUST be rejected.
    evil_tar = tmp_path / "evil.tar.gz"
    with tarfile.open(evil_tar, "w:gz") as tf:
        ti = tarfile.TarInfo(name="../escape.txt")
        ti.size = 4
        tf.addfile(ti, fileobj=__import__("io").BytesIO(b"nope"))
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        cb.unpack_run_tarball(evil_tar, tmp_path / "safe-dest")

    # Symlink member MUST be rejected.
    sym_tar = tmp_path / "sym.tar.gz"
    with tarfile.open(sym_tar, "w:gz") as tf:
        ti = tarfile.TarInfo(name="link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    with pytest.raises(ValueError, match="Refusing unsafe link"):
        cb.unpack_run_tarball(sym_tar, tmp_path / "safe-dest-2")


def test_analyze_run_telemetry_detects_nan_and_collapse(tmp_path):
    run_dir = tmp_path / "adam-collapse"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "step": 420,
                "total_steps": 50000,
                "latest": {"loss/d_total": 0.01, "loss/g_total": 99.0, "diversity/ratio": 0.02},
                "health_alerts": ["Discriminator overpowering generator (critic/accuracy > 0.98)"],
                "error": "FloatingPointError: NaN detected in generator loss",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text("step 419 ok\nstep 420 loss=nan\n", encoding="utf-8")

    tel = cb.analyze_run_telemetry(run_dir)
    assert tel["nan_detected"] is True
    assert tel["collapse_detected"] is True
    assert tel["status"] == "failed"
    assert tel["step"] == 420
    assert len(tel["log_warnings"]) >= 1


# ---------------------------------------------------------------------------
# 6. Mocked Colab CLI orchestration: multi-worker scheduling, teardown, resume
# ---------------------------------------------------------------------------


def test_execute_benchmark_jobs_warm_pool_isolated_configs_and_teardown(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    cfg_file = tmp_path / "dummy.toml"
    cfg_file.write_text('format = "hypergan-model/1"\n[data]\nbuiltin = "cifar10"\n', encoding="utf-8")

    # Pre-seed one already-completed job so skip-completed is exercised alongside 4 pending jobs.
    done_dir = runs_dir / "job-already-done"
    done_dir.mkdir(parents=True)
    (done_dir / "summary.json").write_text(
        json.dumps({"status": "stopped", "step": 50, "total_steps": 200, "best_fid": 9.9}),
        encoding="utf-8",
    )

    calls: list[list[str]] = []
    active_sessions: set[str] = set()
    max_concurrent = 0
    seen_configs: dict[str, str] = {}
    lock = __import__("threading").Lock()

    def fake_run_timeout(cmd, *, timeout_seconds, stdin_text=None, log_file=None, capture=False, **kwargs):
        nonlocal max_concurrent
        with lock:
            calls.append(list(cmd))
        # cmd looks like: [colab_bin, "--config", cfg, subcmd, ...]
        cfg_idx = cmd.index("--config") + 1
        cfg_path = cmd[cfg_idx]
        subcmd = cmd[cfg_idx + 1]
        sess_idx = cmd.index("-s") + 1 if "-s" in cmd else -1
        sess = cmd[sess_idx] if sess_idx > 0 else ""

        if subcmd == "new":
            with lock:
                seen_configs[sess] = cfg_path
                active_sessions.add(sess)
                max_concurrent = max(max_concurrent, len(active_sessions))
            return 0, f"created {sess}\n"
        if subcmd == "exec":
            return 0, "WATCHDOG_OK\n"
        if subcmd == "upload":
            return 0, "uploaded\n"
        if subcmd == "ssh":
            script_text = cmd[-1]
            if "nvidia-smi" in script_text:
                return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
            return 0, "ok\n"
        if subcmd == "download":
            remote_path = cmd[-2]
            local_dest = Path(cmd[-1])
            # Extract run_name from /content/out-<run_name>-chunk1.tar.gz
            fname = Path(remote_path).name
            run_name = fname.removeprefix("out-").rsplit("-", 1)[0]
            _make_run_tarball(local_dest, run_name=run_name, step=50, status="stopped")
            return 0, "downloaded\n"
        if subcmd == "stop":
            with lock:
                active_sessions.discard(sess)
            return 0, f"stopped {sess}\n"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_run_timeout)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc123456789"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    jobs = [
        cb.BenchmarkJob(run_name="job-already-done", config=str(cfg_file), gpu="L4", stop_after_steps=50),
        cb.BenchmarkJob(run_name="job-1", config=str(cfg_file), gpu="L4", stop_after_steps=50),
        cb.BenchmarkJob(run_name="job-2", config=str(cfg_file), gpu="L4", stop_after_steps=50),
        cb.BenchmarkJob(run_name="job-3", config=str(cfg_file), gpu="L4", stop_after_steps=50),
        cb.BenchmarkJob(run_name="job-4", config=str(cfg_file), gpu="L4", stop_after_steps=50),
    ]
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=jobs,
        workers=3,
        max_workers=3,
        stagger_seconds=0.0,
        register_runs=False,
    )

    assert summary["status"] == "complete"
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0]["run_name"] == "job-already-done"
    assert len(summary["results"]) == 4
    # Exactly 3 warm workers booted, isolated config per worker, all torn down leaving 0 active!
    assert len(seen_configs) == 3
    assert len(set(seen_configs.values())) == 3
    assert max_concurrent <= 3
    assert len(active_sessions) == 0

    for jname in ("job-1", "job-2", "job-3", "job-4"):
        assert (runs_dir / jname / "summary.json").is_file()
        assert (runs_dir / jname / "model.pt").is_file()
        assert (runs_dir / jname / "train.log").is_file()

    # Durable session records all transitioned to stopped
    sess_files = list((state_dir / "sessions").glob("*.json"))
    assert len(sess_files) == 3
    for sf in sess_files:
        assert json.loads(sf.read_text(encoding="utf-8"))["status"] == "stopped"


def test_teardown_guaranteed_on_watchdog_or_training_exception(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    stopped: list[str] = []

    def fake_fail_watchdog(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        sess = cmd[cmd.index("-s") + 1]
        if subcmd == "new":
            return 0, "created"
        if subcmd == "exec":
            return 1, "watchdog arm error"
        if subcmd == "stop":
            stopped.append(sess)
            return 0, "stopped"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_fail_watchdog)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc123456789"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    job = cb.BenchmarkJob(run_name="watchdog-fail", config="cifar10-tiny-transformer", gpu="L4", stop_after_steps=10)
    with pytest.raises(RuntimeError, match="All Colab workers failed"):
        cb.execute_benchmark_jobs(
            repo_root=tmp_path,
            runs_dir=runs_dir,
            state_dir=state_dir,
            jobs=[job],
            workers=1,
            stagger_seconds=0.0,
            register_runs=False,
        )
    assert len(stopped) == 1


def test_disconnect_triggers_automatic_reprovision_and_resume(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    ssh_train_calls: list[str] = []
    booted_workers: list[str] = []
    stopped_workers: list[str] = []
    attempt_counter = {"n": 0}

    def fake_disconnect_then_resume(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        sess = cmd[cmd.index("-s") + 1]
        if subcmd == "new":
            booted_workers.append(sess)
            return 0, "created"
        if subcmd == "exec":
            return 0, "WATCHDOG_OK"
        if subcmd == "upload":
            return 0, "uploaded"
        if subcmd == "stop":
            stopped_workers.append(sess)
            return 0, "stopped"
        if subcmd == "ssh":
            body = cmd[-1]
            if "hypergan train" in body or "hypergan resume" in body:
                ssh_train_calls.append(body)
                attempt_counter["n"] += 1
                if attempt_counter["n"] == 1:
                    return 124, "RuntimeError: Connection was lost."
                return 0, "finished ok"
            return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
        if subcmd == "download":
            local_dest = Path(cmd[-1])
            step_now = 25 if attempt_counter["n"] == 1 else 50
            _make_run_tarball(local_dest, run_name="resumable-run", step=step_now, status="stopped")
            return 0, "downloaded"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_disconnect_then_resume)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc123456789"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    job = cb.BenchmarkJob(
        run_name="resumable-run",
        config="cifar10-tiny-transformer",
        gpu="L4",
        stop_after_steps=50,
        max_retries=2,
    )
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=[job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert len(booted_workers) == 2
    assert len(stopped_workers) == 2
    assert "hypergan train" in ssh_train_calls[0]
    assert "hypergan resume" in ssh_train_calls[1]
    assert "--stop-after-steps 25" in ssh_train_calls[1]


# ---------------------------------------------------------------------------
# 7. eval-bundles, status, and cleanup subcommands
# ---------------------------------------------------------------------------


def test_eval_bundles_downloads_receipts_and_tears_down(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "runs" / "my-run"
    bundle_dir.mkdir(parents=True)
    bundle_path = bundle_dir / "model.pt"
    bundle_path.write_bytes(b"ema-bundle")
    out_dir = tmp_path / "eval_out"
    state_dir = tmp_path / "colab_state"
    stopped: list[str] = []

    def fake_eval_colab(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        sess = cmd[cmd.index("-s") + 1]
        if subcmd == "new":
            return 0, "created"
        if subcmd == "exec":
            return 0, "WATCHDOG_OK"
        if subcmd == "upload":
            return 0, "uploaded"
        if subcmd == "ssh":
            return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
        if subcmd == "download":
            dest = Path(cmd[-1])
            receipts = [
                {
                    "schema_version": 1,
                    "bundle_name": "my-run",
                    "metric_id": "inceptionsuite",
                    "status": "complete",
                    "step": 50000,
                    "values": {"fid": 7.2, "precision": 0.71, "recall": 0.62, "density": 0.98, "coverage": 0.91},
                }
            ]
            with tarfile.open(dest, "w:gz") as tf:
                payload = (json.dumps(receipts) + "\n").encode("utf-8")
                ti = tarfile.TarInfo(name="all-receipts.json")
                ti.size = len(payload)
                tf.addfile(ti, fileobj=__import__("io").BytesIO(payload))
            return 0, "downloaded"
        if subcmd == "stop":
            stopped.append(sess)
            return 0, "stopped"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_eval_colab)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc123456789"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    summary = cb.execute_eval_bundles(
        repo_root=tmp_path,
        state_dir=state_dir,
        bundles=[{"name": "my-run", "bundle_path": str(bundle_path), "run_dir": str(bundle_dir)}],
        metrics=["InceptionSuite", "DINOv2Suite"],
        dataset_config={"builtin": "cifar10", "args": {"root": "~/.hypergan/cache/cifar10", "split": "train"}},
        gpu="L4",
        output_dir=out_dir,
        import_to_run=False,
    )
    assert summary["run_rc"] == 0
    assert len(summary["receipts"]) == 1
    assert summary["receipts"][0]["values"]["fid"] == pytest.approx(7.2)
    assert len(stopped) == 1
    assert (out_dir / "all-receipts.json").is_file()


def test_cleanup_stops_tracked_and_server_stragglers(tmp_path, monkeypatch, capsys):
    state_dir = tmp_path / "runs" / "colab"
    sess_dir = state_dir / "sessions"
    sess_dir.mkdir(parents=True)
    cfg_path = state_dir / "hg-l4-w1.colab.json"
    cfg_path.write_text("{}", encoding="utf-8")
    rec_path = sess_dir / "hg-l4-w1.json"
    rec_path.write_text(
        json.dumps({"session": "hg-l4-w1", "config_path": str(cfg_path), "gpu": "L4", "status": "active"}),
        encoding="utf-8",
    )

    stopped_cmds: list[list[str]] = []
    sessions_calls = {"n": 0}

    def fake_cleanup_colab(cmd, *, timeout_seconds, **kwargs):
        if "sessions" in cmd:
            sessions_calls["n"] += 1
            if sessions_calls["n"] == 1:
                return 0, "hg-l4-w1 | Hardware: L4\norphan-vm | Hardware: A100\n"
            return 0, "[colab] No active sessions found on server.\n"
        if "stop" in cmd:
            stopped_cmds.append(list(cmd))
            return 0, "stopped"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_cleanup_colab)

    rc = cb.main(["--root", str(tmp_path), "--state-dir", str(state_dir), "cleanup"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert sorted(item["session"] for item in out["stopped"]) == ["hg-l4-w1", "orphan-vm"]
    assert out["remaining_sessions"] == []
    assert any("--config" in c and "hg-l4-w1" in c for c in stopped_cmds)
    assert json.loads(rec_path.read_text(encoding="utf-8"))["status"] == "stopped"


def test_dry_run_and_status_json_never_boot_vms(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "runs" / "colab"
    cfg_file = tmp_path / "m.toml"
    cfg_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")

    called: list[list[str]] = []

    def fake_colab(cmd, *, timeout_seconds, **kwargs):
        called.append(list(cmd))
        if "sessions" in cmd:
            return 0, "[colab] No active sessions found on server.\n"
        if "usage" in cmd:
            return 0, "Current balance: 100.0 compute units\nActive assignments: 0\n"
        raise AssertionError(f"Unexpected colab call during dry-run/status: {cmd}")

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_colab)

    rc_dry = cb.main(
        [
            "--root",
            str(tmp_path),
            "--state-dir",
            str(state_dir),
            "run",
            "--config",
            str(cfg_file),
            "--run-name",
            "dry-job",
            "--runs-dir",
            str(runs_dir),
            "--gpu",
            "L4",
            "--workers",
            "3",
            "--dry-run",
        ]
    )
    assert rc_dry == 0
    dry_out = json.loads(capsys.readouterr().out)
    assert dry_out["status"] == "dry_run"
    assert len(dry_out["worker_plan"]) == 1  # Clamped to pending job count
    assert called == []

    # Over-cap refused by CLI
    with pytest.raises(ValueError, match="hard safety cap"):
        cb.main(
            [
                "--root",
                str(tmp_path),
                "run",
                "--config",
                str(cfg_file),
                "--workers",
                "4",
                "--max-workers",
                "4",
                "--dry-run",
            ]
        )

    rc_st = cb.main(["--root", str(tmp_path), "--state-dir", str(state_dir), "status", "--json"])
    assert rc_st == 0
    st_out = json.loads(capsys.readouterr().out)
    assert st_out["server_sessions"] == []
    assert st_out["tracked_total"] == 0


def test_prepare_source_bundle_tarball_and_eval_import_to_run(tmp_path, monkeypatch):
    # Verify tarball source mode creates an archive containing src/ & pyproject.toml
    (tmp_path / "pyproject.toml").write_text("[project]\nname='hypergan'\n", encoding="utf-8")
    (tmp_path / "src" / "hypergan").mkdir(parents=True)
    (tmp_path / "src" / "hypergan" / "__init__.py").write_text("__version__ = '2.2.0'\n", encoding="utf-8")
    bundle = cb.prepare_source_bundle(tmp_path, tmp_path / "out", source_mode="tarball")
    assert bundle["mode"] == "tarball"
    with tarfile.open(bundle["tarball_path"], "r:gz") as tf:
        names = set(tf.getnames())
        assert "pyproject.toml" in names
        assert "src/hypergan/__init__.py" in names

    # Verify eval-bundles --import-to-run invokes hypergan.evaluation_results.import_scores cleanly
    run_dir = tmp_path / "runs" / "eval-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "run12345", "status": "stopped", "steps": 500, "config": {}}),
        encoding="utf-8",
    )
    bundle_path = run_dir / "model.pt"
    bundle_path.write_bytes(b"ema-weights")
    bsha = __import__("hashlib").sha256(b"ema-weights").hexdigest()

    uploaded_runner: dict[str, str] = {}

    def fake_eval_colab(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        if subcmd == "upload" and cmd[-1].endswith("/eval_runner.py"):
            uploaded_runner["code"] = Path(cmd[-2]).read_text(encoding="utf-8")
            return 0, "uploaded"
        if subcmd == "ssh" and "eval_runner.py" in cmd[-1]:
            assert "cat <<'PY'" not in cmd[-1]
        if subcmd == "download":
            dest = Path(cmd[-1])
            receipts = [
                {
                    "schema_version": 1,
                    "bundle_name": "eval-run",
                    "local_run_dir": str(run_dir),
                    "bundle_sha256": bsha,
                    "metric_id": "dinov2suite",
                    "status": "complete",
                    "step": 500,
                    "attempt_id": "0001-attempt",
                    "values": {"fd_dinov2": 42.5, "precision": 0.81, "recall": 0.74},
                    "outputs": {
                        "fd_dinov2": {"kind": "scalar", "label": "FD DINOv2", "unit": "FD", "direction": "minimize"},
                        "precision": {"kind": "scalar", "label": "Precision", "unit": "", "direction": "maximize"},
                        "recall": {"kind": "scalar", "label": "Recall", "unit": "", "direction": "maximize"},
                    },
                    "snapshot_identity": {"run_id": "run12345", "source_step": 500, "attempt_id": "0001-attempt"},
                    "protocol": {"schema_version": 1, "samples": 50000},
                }
            ]
            with tarfile.open(dest, "w:gz") as tf:
                payload = (json.dumps(receipts) + "\n").encode("utf-8")
                ti = tarfile.TarInfo(name="all-receipts.json")
                ti.size = len(payload)
                tf.addfile(ti, fileobj=__import__("io").BytesIO(payload))
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_eval_colab)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    summary = cb.execute_eval_bundles(
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        bundles=[{"name": "eval-run", "bundle_path": str(bundle_path), "run_dir": str(run_dir)}],
        metrics=["DINOv2Suite"],
        dataset_config={"builtin": "cifar10", "args": {"root": "~/.hypergan/cache/cifar10", "split": "train"}},
        gpu="L4",
        output_dir=tmp_path / "receipts",
        import_to_run=True,
    )
    assert len(summary["imported"]) == 1
    assert "evaluation_id" in summary["imported"][0]
    assert "code" in uploaded_runner
    __import__("ast").parse(uploaded_runner["code"])
    assert (run_dir / "summary.json").is_file()
    imported_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    eval_metrics = {e["metric"]: e["value"] for e in imported_summary.get("evaluations", [])}
    assert eval_metrics["dinov2suite/fd_dinov2"] == pytest.approx(42.5)
    assert eval_metrics["dinov2suite/precision"] == pytest.approx(0.81)


def test_eval_runner_dataset_lowering_uses_real_hypergan_factory():
    # Confirm scripts/colab_bench.py routes publication suites to `python -m hypergan.image_metrics`
    # and does not reference nonexistent private model_file helpers.
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "_lower_data_builtin" not in src
    assert "hypergan.image_data:CIFAR10Data" in src
    assert "hypergan.image_metrics" in src
    assert cb.IMAGE_METRICS_CLI_ALIASES["InceptionSuite"] == "inception"
    assert cb.IMAGE_METRICS_CLI_ALIASES["DINOv2Suite"] == "dinov2"
    assert cb.IMAGE_METRICS_CLI_ALIASES["InceptionFID"] == "inception_fid"
    assert cb.IMAGE_METRICS_CLI_ALIASES["DINOv2FD"] == "dinov2_fd"


def test_boot_retries_recover_from_transient_503(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    boot_calls = {"n": 0}
    eval_spec_seen: dict[str, Any] = {}

    def fake_503_then_ok(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        if subcmd == "new":
            boot_calls["n"] += 1
            if boot_calls["n"] == 1:
                return 1, "ColabRequestError: 503 Service Unavailable"
            return 0, "Session READY."
        if subcmd == "upload" and cmd[-1].endswith("eval-spec.json"):
            eval_spec_seen.update(json.loads(Path(cmd[-2]).read_text(encoding="utf-8")))
            return 0, "uploaded"
        if subcmd == "download":
            dest = Path(cmd[-1])
            _make_run_tarball(dest, run_name="retry-job", step=20, status="stopped")
            return 0, "downloaded"
        return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 580.82.07\n"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_503_then_ok)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    job = cb.BenchmarkJob(run_name="retry-job", config="cifar10-tiny-transformer", gpu="L4", stop_after_steps=20)
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=[job],
        workers=1,
        boot_retries=2,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert boot_calls["n"] == 2


def test_ffhq64_auto_dataset_prep_and_fallback_gpu(tmp_path, monkeypatch):
    # Real whitepaper FFHQ-64 config must auto-detect `ffhq64`.
    ffhq_cfg = REPO_ROOT / "benchmarks" / "whitepaper" / "ffhq64-resnet-v12.toml"
    assert ffhq_cfg.is_file()
    job = cb.BenchmarkJob(run_name="wp-ffhq64", config=str(ffhq_cfg), gpu="L4", stop_after_steps=10)
    assert cb.detect_job_dataset(job) == "ffhq64"

    ssh_cmds: list[str] = []
    boot_gpus: list[str] = []

    def fake_fallback_and_ffhq(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        if subcmd == "new":
            gpu_arg = cmd[cmd.index("--gpu") + 1]
            boot_gpus.append(gpu_arg)
            if gpu_arg in ("L4", "A100"):
                return 1, "503 Service Unavailable"
            return 0, "Session READY."
        if subcmd == "ssh":
            ssh_cmds.append(cmd[-1])
            return 0, "COLAB_GPU_INFO: Tesla T4, 15360 MiB, 580.82.07\n"
        if subcmd == "download":
            dest = Path(cmd[-1])
            _make_run_tarball(dest, run_name="wp-ffhq64", step=10, status="stopped")
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_fallback_and_ffhq)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    summary = cb.execute_benchmark_jobs(
        repo_root=REPO_ROOT,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
        jobs=[job],
        workers=1,
        boot_retries=1,
        fallback_gpus="A100,T4",
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert boot_gpus == ["L4", "A100", "T4"]
    assert summary["results"][0]["requested_gpu"] == "L4"
    assert summary["results"][0]["gpu"] == "T4"
    assert summary["workers_cost"][0]["requested_gpu"] == "L4"
    assert summary["workers_cost"][0]["gpu"] == "T4"
    assert summary["workers_cost"][0]["cu_per_hr"] == pytest.approx(1.15)
    assert any("scripts/prepare_benchmark_dataset.py --dataset ffhq64" in c for c in ssh_cmds)
    # Fresh hypergan train must NOT pre-populate train.log inside remote_run_dir before startup.
    train_sh = [c for c in ssh_cmds if "hypergan train" in c]
    assert train_sh
    assert " -d 0 " in train_sh[0]
    assert " -d cuda " not in train_sh[0]
    assert "/content/logs/wp-ffhq64.train.log" in train_sh[0]
    assert "/content/runs/wp-ffhq64/train.log" not in train_sh[0]


def test_cpu_fallback_detection_in_telemetry(tmp_path):
    run_dir = tmp_path / "fallback-run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"status": "stopped", "step": 10}), encoding="utf-8")
    (run_dir / "train.log").write_text(
        "HyperGANHardwareWarning: CUDA device requested ('cuda') but CUDA is unavailable on this host; falling back to cpu\n",
        encoding="utf-8",
    )
    tel = cb.analyze_run_telemetry(run_dir)
    assert tel["cpu_fallback_detected"] is True


def test_ssh_broken_pipe_rc255_triggers_clean_reprovision_and_resume(tmp_path, monkeypatch):
    assert cb._is_colab_disconnect(255, "client_loop: send disconnect: Broken pipe")
    assert cb._is_colab_disconnect(1, "Connection to colab-runtime closed by remote host.")
    assert cb._is_colab_disconnect(1, "Already-active SSH session (HTTP 429)")
    assert not cb._is_colab_disconnect(1, "CUDA out of memory")

    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    booted: list[str] = []
    ssh_calls: list[tuple[str, str]] = []
    attempt = {"n": 0}

    def fake_broken_pipe_then_resume(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        sess = cmd[cmd.index("-s") + 1]
        if subcmd == "new":
            booted.append(sess)
            return 0, "Session READY."
        if subcmd == "ssh":
            body = cmd[-1]
            ssh_calls.append((sess, body))
            if "hypergan train" in body or "hypergan resume" in body:
                attempt["n"] += 1
                if attempt["n"] == 1:
                    return 255, "Connection to colab-runtime closed by remote host.\nclient_loop: send disconnect: Broken pipe"
                return 0, "finished ok"
            return 0, "COLAB_GPU_INFO: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07\n"
        if subcmd == "download":
            dest = Path(cmd[-1])
            _make_run_tarball(dest, run_name="bp-run", step=50, status="stopped")
            return 0, "downloaded"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_broken_pipe_then_resume)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    job = cb.BenchmarkJob(
        run_name="bp-run",
        config="cifar10-tiny-transformer",
        gpu="A100",
        stop_after_steps=50,
        chunk_steps=25,
        max_retries=2,
    )
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=[job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert len(booted) == 2
    assert booted[1].endswith("-r1")
    # Broken pipe on attempt 1 (booted[0]) must NOT invoke tar/pull on the dead SSH tunnel before reprovisioning.
    assert not any("chunk1" in body for sess, body in ssh_calls if sess == booted[0])
    assert any("chunk1" in body for sess, body in ssh_calls if sess == booted[1])


def test_detached_training_survives_transient_poll_blips_and_runs_serialized_sync(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    uploaded_scripts: dict[str, str] = {}
    poll_count = {"n": 0}
    sync_downloads: list[str] = []
    fake_now = {"t": 1000.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(secs):
        fake_now["t"] += max(1.0, float(secs))

    def fake_detached_colab(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        if subcmd == "new":
            return 0, "Session READY."
        if subcmd == "exec":
            return 0, "WATCHDOG_OK"
        if subcmd == "upload":
            remote = cmd[-1]
            if remote.endswith(".sh"):
                uploaded_scripts[remote] = Path(cmd[-2]).read_text(encoding="utf-8")
            return 0, "uploaded"
        if subcmd == "ssh":
            body = cmd[-1]
            if "DETACHED_LAUNCHED" in body:
                return 0, "DETACHED_LAUNCHED pid=4242\n"
            if "DETACHED_DONE:" in body:
                poll_count["n"] += 1
                if poll_count["n"] == 1:
                    return 0, "DETACHED_RUNNING\nDETACHED_TAIL:step 500 loss=0.4\n"
                if poll_count["n"] == 2:
                    # Transient SSH blip must NOT abort detached training!
                    return 255, "client_loop: send disconnect: Broken pipe"
                if poll_count["n"] == 3:
                    # Advance clock past sync_interval_minutes=1.0 -> triggers serialized periodic sync.
                    fake_now["t"] += 120.0
                    return 0, "DETACHED_RUNNING\nDETACHED_TAIL:step 2500 loss=0.3\n"
                return 0, "DETACHED_DONE:0\nDETACHED_TAIL:step 5000 done\n"
            return 0, "COLAB_GPU_INFO: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07\n"
        if subcmd == "download":
            remote_path = cmd[-2]
            dest = Path(cmd[-1])
            sync_downloads.append(Path(remote_path).name)
            step_now = 2500 if "sync" in remote_path else 5000
            _make_run_tarball(dest, run_name="detached-run", step=step_now, status="stopped", total_steps=5000)
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_detached_colab)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(cb.time, "sleep", fake_sleep)

    job = cb.BenchmarkJob(
        run_name="detached-run",
        config="cifar10-tiny-transformer",
        gpu="A100",
        stop_after_steps=5000,
        chunk_steps=5000,
        sync_interval_minutes=1.0,
        detached=True,
        poll_seconds=10.0,
        max_retries=2,
    )
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=[job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert poll_count["n"] == 4
    assert len(uploaded_scripts) == 1
    script_text = next(iter(uploaded_scripts.values()))
    assert "export PYTHONUNBUFFERED=1" in script_text
    assert "hypergan train" in script_text
    assert any("sync1" in name for name in sync_downloads)
    assert any("chunk1" in name for name in sync_downloads)
    assert summary["results"][0]["disconnected"] is False
    assert summary["results"][0]["telemetry"]["step"] == 5000


def test_detached_repeated_poll_failures_reprovision_and_resume(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "colab_state"
    booted: list[str] = []
    poll_attempt = {"first_vm_polls": 0, "second_vm_polls": 0}
    fake_now = {"t": 1000.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(secs):
        fake_now["t"] += max(1.0, float(secs))

    def fake_vm_dies_mid_detached(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        sess = cmd[cmd.index("-s") + 1]
        if subcmd == "new":
            booted.append(sess)
            return 0, "Session READY."
        if subcmd == "ssh":
            body = cmd[-1]
            if "DETACHED_LAUNCHED" in body:
                return 0, "DETACHED_LAUNCHED pid=777\n"
            if "DETACHED_DONE:" in body:
                if sess == booted[0]:
                    poll_attempt["first_vm_polls"] += 1
                    if poll_attempt["first_vm_polls"] == 1:
                        fake_now["t"] += 120.0
                        return 0, "DETACHED_RUNNING\nDETACHED_TAIL:step 2500\n"
                    # VM reclaimed: 5 consecutive failures -> reprovision!
                    return 255, "Connection closed by remote host"
                poll_attempt["second_vm_polls"] += 1
                return 0, "DETACHED_DONE:0\nDETACHED_TAIL:step 5000\n"
            return 0, "COLAB_GPU_INFO: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07\n"
        if subcmd == "download":
            remote_path = cmd[-2]
            dest = Path(cmd[-1])
            step_now = 2500 if "sync" in remote_path else 5000
            _make_run_tarball(dest, run_name="detached-resume", step=step_now, status="stopped", total_steps=5000)
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_vm_dies_mid_detached)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(cb.time, "sleep", fake_sleep)

    job = cb.BenchmarkJob(
        run_name="detached-resume",
        config="cifar10-tiny-transformer",
        gpu="A100",
        stop_after_steps=5000,
        chunk_steps=5000,
        sync_interval_minutes=1.0,
        detached=True,
        poll_seconds=10.0,
        max_retries=2,
    )
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=runs_dir,
        state_dir=state_dir,
        jobs=[job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert len(booted) == 2
    assert poll_attempt["first_vm_polls"] == 6  # 1 healthy poll (triggers sync) + 5 consecutive failures
    assert summary["results"][0]["telemetry"]["step"] == 5000


def test_manifest_and_cli_detached_flags(tmp_path):
    cfg = tmp_path / "m.toml"
    cfg.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    man = tmp_path / "q.toml"
    man.write_text(
        """
[defaults]
gpu = "A100"
detached = true
poll_seconds = 30.0
max_retries = 5

[[jobs]]
run_name = "j-det"
config = "m.toml"
""",
        encoding="utf-8",
    )
    jobs = cb.load_manifest(man, repo_root=tmp_path)
    assert jobs[0].detached is True
    assert jobs[0].poll_seconds == pytest.approx(30.0)
    assert jobs[0].max_retries == 5

    # Omitting --detached / --max-retries on CLI preserves manifest settings
    parsed = cb.build_parser().parse_args(["run", "--manifest", str(man)])
    assert parsed.detached is False
    assert parsed.poll_seconds is None
    assert parsed.max_retries is None


def test_disconnect_short_circuits_on_rc_zero_and_eval_honors_max_minutes(tmp_path, monkeypatch):
    # rc == 0 must NEVER be misclassified as a disconnect even if logs contain numbers like 429 or marker strings.
    assert cb._is_colab_disconnect(0, "step 42900 loss=0.429 websocket") is False
    # Bare digits '429' inside non-zero training errors must not match HTTP 429.
    assert cb._is_colab_disconnect(1, "step 42900 CUDA out of memory") is False
    assert cb._is_colab_disconnect(1, "HTTP 429 Too Many Requests") is True

    bundle_dir = tmp_path / "runs" / "b1"
    bundle_dir.mkdir(parents=True)
    bundle_path = bundle_dir / "model.pt"
    bundle_path.write_bytes(b"ema")
    eval_timeouts: list[float] = []

    def fake_eval_timeout(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        if subcmd == "ssh" and "eval_runner.py" in cmd[-1]:
            eval_timeouts.append(float(timeout_seconds))
            return 0, "ok"
        if subcmd == "download":
            dest = Path(cmd[-1])
            with tarfile.open(dest, "w:gz") as tf:
                payload = b"[]\n"
                ti = tarfile.TarInfo(name="all-receipts.json")
                ti.size = len(payload)
                tf.addfile(ti, fileobj=__import__("io").BytesIO(payload))
            return 0, "downloaded"
        return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 580.82.07\n"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_eval_timeout)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    cb.execute_eval_bundles(
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        bundles=[{"name": "b1", "bundle_path": str(bundle_path), "run_dir": str(bundle_dir)}],
        metrics=["InceptionSuite", "DINOv2Suite"],
        dataset_config={"builtin": "cifar10", "args": {"root": "~/.hypergan/cache/cifar10", "split": "train"}},
        gpu="L4",
        max_minutes=180,
        output_dir=tmp_path / "out",
    )
    assert eval_timeouts == [10800.0]


def test_chunked_upload_splits_large_payloads_and_reassembles(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir(parents=True)
    big_file = tmp_path / "resume-big.tar.gz"
    payload = b"0123456789ABCDEF" * 64 + b"TAIL"  # 1028 bytes
    big_file.write_bytes(payload)

    uploaded_parts: list[tuple[str, bytes]] = []
    ssh_cmds: list[str] = []

    def fake_chunk_colab(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
        if subcmd == "upload":
            local_p = Path(cmd[-2])
            remote_p = cmd[-1]
            uploaded_parts.append((remote_p, local_p.read_bytes()))
            return 0, "uploaded"
        if subcmd == "ssh":
            ssh_cmds.append(cmd[-1])
            return 0, "ok"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_chunk_colab)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    worker = cb.ColabBenchWorker(
        worker_name="hg-test-chunk",
        gpu="A100",
        batch_dir=batch_dir,
        state_dir=tmp_path / "state",
        repo_root=tmp_path,
        repo_url=cb.DEFAULT_REPO_URL,
        source_sync={"mode": "git", "ref": "develop", "sha": "abc"},
    )
    rc, _ = worker._upload_file(big_file, "/content/resume-big.tar.gz", chunk_bytes=300)
    assert rc == 0
    assert len(uploaded_parts) == 4  # ceil(1028 / 300) == 4
    assert [name for name, _ in uploaded_parts] == [
        "/content/resume-big.tar.gz.part0000",
        "/content/resume-big.tar.gz.part0001",
        "/content/resume-big.tar.gz.part0002",
        "/content/resume-big.tar.gz.part0003",
    ]
    assert b"".join(chunk for _, chunk in uploaded_parts) == payload
    assert len(ssh_cmds) == 1
    assert "cat " in ssh_cmds[0]
    assert f"-ne {len(payload)}" in ssh_cmds[0]


# ---------------------------------------------------------------------------
# 8. Audit regression tests (detached OOM, watchdog stall, signals, teardown,
#    git bundle/overlay, chunked resume, foreground sync race, edge cases)
# ---------------------------------------------------------------------------


def test_detached_poll_script_detects_oom_via_real_bash_lc(tmp_path, monkeypatch):
    """Empirically verify `bash -lc <poll_sh>` never self-matches `pgrep -f` and reaches DETACHED_EXITED_NO_RC."""
    import signal
    import subprocess
    import time

    remote_script = str(tmp_path / "my-run_1.chunk0.sh")
    remote_pid = str(tmp_path / "my-run_1.chunk0.pid")
    remote_rc = str(tmp_path / "my-run_1.chunk0.rc")
    remote_log = str(tmp_path / "my-run_1.train.log")
    hb_file = tmp_path / "hb"

    Path(remote_script).write_text("#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")
    os.chmod(remote_script, 0o755)
    Path(remote_log).write_text("step 10 loss=0.5\n", encoding="utf-8")

    poll_sh = cb._build_detached_poll_script(
        remote_script=remote_script,
        remote_pid=remote_pid,
        remote_rc=remote_rc,
        remote_train_log=remote_log,
    )
    # Redirect /content/bench_heartbeat touch in the local test sandbox so it doesn't require /content.
    local_poll_sh = poll_sh.replace("/content/bench_heartbeat", str(hb_file))
    assert remote_script not in local_poll_sh

    # 1. No running process and no rc file -> MUST report DETACHED_EXITED_NO_RC (no pgrep self-match).
    out_before = subprocess.check_output(["bash", "-lc", local_poll_sh], text=True)
    assert "DETACHED_EXITED_NO_RC" in out_before
    assert "DETACHED_RUNNING" not in out_before
    assert "DETACHED_TAIL:step 10 loss=0.5" in out_before

    # 2. Running detached process -> MUST report DETACHED_RUNNING both with and without pidfile.
    proc = subprocess.Popen(["bash", remote_script], start_new_session=True)
    try:
        Path(remote_pid).write_text(f"{proc.pid}\n", encoding="utf-8")
        time.sleep(0.15)
        out_running_pid = subprocess.check_output(["bash", "-lc", local_poll_sh], text=True)
        assert "DETACHED_RUNNING" in out_running_pid

        # Remove pidfile so pgrep -f fallback branch is exercised directly.
        Path(remote_pid).unlink()
        out_running_pgrep = subprocess.check_output(["bash", "-lc", local_poll_sh], text=True)
        assert "DETACHED_RUNNING" in out_running_pgrep

        # 3. SIGKILL (OOM simulation) leaving stale pidfile and no rc file -> DETACHED_EXITED_NO_RC.
        Path(remote_pid).write_text(f"{proc.pid}\n", encoding="utf-8")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)

    out_oom = subprocess.check_output(["bash", "-lc", local_poll_sh], text=True)
    assert "DETACHED_EXITED_NO_RC" in out_oom
    assert "DETACHED_RUNNING" not in out_oom

    # 4. Written rc file -> DETACHED_DONE:<rc>.
    Path(remote_rc).write_text("0\n", encoding="utf-8")
    out_done = subprocess.check_output(["bash", "-lc", local_poll_sh], text=True)
    assert "DETACHED_DONE:0" in out_done

    # 5. End-to-end `_run_chunk_detached` translates DETACHED_EXITED_NO_RC into rc=137, disconnected=False.
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    worker = cb.ColabBenchWorker(
        worker_name="hg-oom-test",
        gpu="L4",
        batch_dir=batch_dir,
        state_dir=tmp_path / "state",
        repo_root=tmp_path,
        repo_url=cb.DEFAULT_REPO_URL,
        source_sync={"mode": "git", "ref": "develop", "sha": "abc"},
    )
    job = cb.BenchmarkJob(run_name="oom-job", config="cifar10-tiny-transformer", gpu="L4", detached=True, poll_seconds=5.0)

    def fake_oom_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "ssh" and "DETACHED_DONE:" in cmd[-1]:
            return 0, "DETACHED_EXITED_NO_RC\nDETACHED_TAIL:CUDA out of memory\n"
        if subcmd == "ssh" and "DETACHED_LAUNCHED" in cmd[-1]:
            return 0, "DETACHED_LAUNCHED pid=999\n"
        return 0, "ok\n"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_oom_colab)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    rc, _out, disconnected, _unpack = worker._run_chunk_detached(
        job=job,
        chunk_index=0,
        quoted_hg="hypergan train cfg",
        remote_run_dir="/content/runs/oom-job",
        remote_train_log="/content/logs/oom-job.train.log",
        drive_sync_sh="",
        local_run_dir=tmp_path / "runs" / "oom-job",
        ssh_timeout=600,
        register_run_local=False,
    )
    assert rc == 137
    assert disconnected is False


def test_watchdog_separates_heartbeat_from_training_stall_activity(tmp_path):
    """Touching controller heartbeat MUST NOT defeat `--stall-minutes` for a stalled detached job."""
    import http.server
    import subprocess
    import textwrap
    import threading
    import time

    hb_file = tmp_path / "bench_heartbeat"
    done_file = tmp_path / "bench_job.done"
    job_log = tmp_path / "bench_job.log"
    runs_root = tmp_path / "runs"
    logs_dir = tmp_path / "logs"
    watchdog_log = tmp_path / "watchdog.log"
    run_dir = runs_root / "stalled-run"
    (run_dir / "checkpoints").mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    events_file = run_dir / "events.jsonl"
    train_log = logs_dir / "stalled-run.train.log"
    ckpt_latest = run_dir / "checkpoints" / "latest.json"

    script = cb.build_watchdog_arm_script(
        max_seconds=3600,
        idle_grace_seconds=5,
        stall_seconds=10,
        heartbeat_file=str(hb_file),
        done_file=str(done_file),
        job_log=str(job_log),
        runs_root=str(runs_root),
        logs_dir=str(logs_dir),
    )
    start_marker = "watchdog_code = textwrap.dedent('''\n"
    end_marker = "''').strip()"
    inner = textwrap.dedent(script.split(start_marker, 1)[1].split(end_marker, 1)[0])
    # Point logf at tmp_path and shorten poll interval to 0.05s for fast real subprocess testing.
    test_watchdog = (
        inner.replace('logf = "/content/watchdog.log"', f"logf = {str(watchdog_log)!r}")
        .replace("time.sleep(15)", "time.sleep(0.05)")
    )
    watchdog_py = tmp_path / "watchdog.py"
    watchdog_py.write_text(test_watchdog + "\n", encoding="utf-8")

    # 1. Run a real local HTTP server to capture POST /unassign calls from watchdog.py.
    unassign_hits: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            unassign_hits.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    addr = f"127.0.0.1:{server.server_port}"
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        # Backdate ALL training files beyond stall_seconds=10, while keeping hb_file fresh!
        old_ts = time.time() - 120.0
        for fpath in (job_log, events_file, train_log, ckpt_latest):
            fpath.write_text("stale\n", encoding="utf-8")
            os.utime(fpath, (old_ts, old_ts))

        # Repeatedly touch hb_file (simulating detached poll_sh) while watchdog runs.
        proc = subprocess.Popen([sys.executable, str(watchdog_py), addr, "3600", "5", "10"])
        try:
            deadline = time.monotonic() + 5.0
            while proc.poll() is None and time.monotonic() < deadline:
                hb_file.write_text(str(time.time()), encoding="utf-8")
                time.sleep(0.05)
            assert proc.wait(timeout=2) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

        assert unassign_hits == ["/unassign"]
        log_text = watchdog_log.read_text(encoding="utf-8")
        assert "training activity stall timeout 10s exceeded" in log_text

        # 2. Fresh detached train log or checkpoint prevents stall trigger while active.
        unassign_hits.clear()
        now_ts = time.time()
        os.utime(train_log, (now_ts, now_ts))
        proc2 = subprocess.Popen([sys.executable, str(watchdog_py), addr, "3600", "5", "10"])
        try:
            time.sleep(0.25)
            assert proc2.poll() is None  # Still alive because detached train_log is fresh
            assert unassign_hits == []
        finally:
            proc2.kill()
            proc2.wait(timeout=2)

        # 3. Post-job idle grace: stale done_file + stale hb_file -> unassigns; fresh hb_file keeps alive.
        done_file.write_text("done\n", encoding="utf-8")
        os.utime(done_file, (old_ts, old_ts))
        os.utime(train_log, (now_ts, now_ts))  # Keep training activity fresh so only idle grace is tested
        proc3 = subprocess.Popen([sys.executable, str(watchdog_py), addr, "3600", "5", "120"])
        try:
            # First keep hb_file fresh -> must stay alive past 0.25s.
            for _ in range(5):
                hb_file.write_text(str(time.time()), encoding="utf-8")
                time.sleep(0.05)
            assert proc3.poll() is None
            assert unassign_hits == []
            # Now backdate hb_file (controller disconnected) -> MUST trigger post-job idle unassign!
            os.utime(hb_file, (old_ts, old_ts))
            assert proc3.wait(timeout=3) == 0
        finally:
            if proc3.poll() is None:
                proc3.kill()
                proc3.wait(timeout=2)
        assert unassign_hits == ["/unassign"]
        assert "post-job idle grace 5s exceeded" in watchdog_log.read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()


def test_signal_handler_no_deadlock_and_no_reprovisioning_on_shutdown(tmp_path, monkeypatch):
    """SIGINT/SIGTERM must neither deadlock on registry lock nor reprovision paid VMs."""
    import signal
    import threading
    import time

    real_run_with_hard_timeout = cb.run_with_hard_timeout

    # 1. Holding `_GLOBAL_REGISTRY.lock` when SIGTERM arrives must NOT deadlock; teardown runs cleanly after unwind.
    state_dir = tmp_path / "sig_state"
    stopped_via_sig: list[str] = []

    def fake_sig_colab(cmd, *, timeout_seconds, **kwargs):
        if "stop" in cmd:
            stopped_via_sig.append(cmd[cmd.index("-s") + 1])
            return 0, "stopped"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_sig_colab)
    orig_state_dir = cb._GLOBAL_REGISTRY.state_dir
    orig_colab_bin = cb._GLOBAL_REGISTRY.colab_bin

    def fake_cmd_status(_args):
        cb._GLOBAL_REGISTRY.state_dir = state_dir
        cb._GLOBAL_REGISTRY.colab_bin = "colab"
        cfg = state_dir / "w-sig.colab.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("{}", encoding="utf-8")
        cb._GLOBAL_REGISTRY.register("w-sig", cfg, gpu="A100")
        # Acquire the non-reentrant registry lock on the main thread when SIGTERM fires!
        with cb._GLOBAL_REGISTRY.lock:
            os.kill(os.getpid(), signal.SIGTERM)
        return 0

    monkeypatch.setattr(cb, "cmd_status", fake_cmd_status)
    try:
        rc = cb.main(["status"])
        assert rc == 128 + int(signal.SIGTERM)
        assert stopped_via_sig == ["w-sig"]
        assert "w-sig" not in cb._GLOBAL_REGISTRY.sessions
    finally:
        cb._reset_shutdown_state()
        cb._GLOBAL_REGISTRY.sessions.clear()
        cb._GLOBAL_REGISTRY.state_dir = orig_state_dir
        cb._GLOBAL_REGISTRY.colab_bin = orig_colab_bin

    # 2. Setting `_SHUTDOWN_EVENT` mid-SSH (simulating Ctrl-C causing rc=255) MUST NOT reprovision a new VM.
    runs_dir = tmp_path / "runs"
    batch_state = tmp_path / "batch_state"
    booted: list[str] = []
    stopped: list[str] = []

    def fake_ctrl_c_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        sess = cmd[cmd.index("-s") + 1]
        if subcmd == "new":
            booted.append(sess)
            return 0, "Session READY."
        if subcmd == "stop":
            stopped.append(sess)
            return 0, "stopped"
        if subcmd == "ssh":
            body = cmd[-1]
            if "hypergan train" in body:
                # Simulate Ctrl-C arriving during training SSH -> tunnel drops with rc=255.
                cb._SHUTDOWN_EVENT.set()
                return 255, "Connection closed by remote host.\nBroken pipe"
            return 0, "COLAB_GPU_INFO: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07\n"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_ctrl_c_colab)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    jobs = [
        cb.BenchmarkJob(run_name="ctrlc-job-1", config="cifar10-tiny-transformer", gpu="A100", stop_after_steps=50, max_retries=3),
        cb.BenchmarkJob(run_name="ctrlc-job-2", config="cifar10-tiny-transformer", gpu="A100", stop_after_steps=50, max_retries=3),
    ]
    try:
        cb._reset_shutdown_state()
        summary = cb.execute_benchmark_jobs(
            repo_root=tmp_path,
            runs_dir=runs_dir,
            state_dir=batch_state,
            jobs=jobs,
            workers=1,
            stagger_seconds=0.0,
            register_runs=False,
        )
        assert summary["status"] == "interrupted"
        assert len(booted) == 1  # NEVER reprovisioned!
        assert stopped == booted  # Original VM cleanly stopped!
    finally:
        cb._reset_shutdown_state()
        cb._GLOBAL_REGISTRY.sessions.clear()

    # 3. `run_with_hard_timeout` aborts a sleeping child promptly when `cancel_event` is set.
    cancel_ev = threading.Event()
    timer = threading.Timer(0.15, cancel_ev.set)
    timer.start()
    t0 = time.monotonic()
    try:
        rc_c, out_c = real_run_with_hard_timeout(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=15.0,
            capture=True,
            cancel_event=cancel_ev,
        )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - t0
    assert rc_c == 130
    assert "[shutdown]" in out_c
    assert elapsed < 3.0

    # 4. Holding `_SHUTDOWN_EVENT._cond._lock` on main thread when SIGTERM arrives MUST NOT deadlock.
    stopped_via_cond_sig: list[str] = []

    def fake_cond_sig_colab(cmd, *, timeout_seconds, **kwargs):
        if "stop" in cmd:
            stopped_via_cond_sig.append(cmd[cmd.index("-s") + 1])
            return 0, "stopped"
        return 0, ""

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_cond_sig_colab)

    def fake_cmd_status_holding_event_lock(_args):
        cb._GLOBAL_REGISTRY.state_dir = state_dir
        cb._GLOBAL_REGISTRY.colab_bin = "colab"
        cfg = state_dir / "w-evlock.colab.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("{}", encoding="utf-8")
        cb._GLOBAL_REGISTRY.register("w-evlock", cfg, gpu="L4")
        # Hold Event's internal non-reentrant lock when SIGTERM arrives (as during Event.wait()).
        with cb._SHUTDOWN_EVENT._cond._lock:
            os.kill(os.getpid(), signal.SIGTERM)
        return 0

    monkeypatch.setattr(cb, "cmd_status", fake_cmd_status_holding_event_lock)
    try:
        rc_ev = cb.main(["status"])
        assert rc_ev == 128 + int(signal.SIGTERM)
        assert cb._SHUTDOWN_EVENT.is_set()
        assert stopped_via_cond_sig == ["w-evlock"]
    finally:
        cb._reset_shutdown_state()
        cb._GLOBAL_REGISTRY.sessions.clear()
        cb._GLOBAL_REGISTRY.state_dir = orig_state_dir
        cb._GLOBAL_REGISTRY.colab_bin = orig_colab_bin

    # 5. KeyboardInterrupt interrupting `run_with_hard_timeout` MUST terminate the child process group (no orphans).
    pid_file = tmp_path / "child.pid"
    child_code = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "print('READY', flush=True); time.sleep(60)"
    )

    def _send_sigint_when_ready() -> None:
        deadline = time.monotonic() + 5.0
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=_send_sigint_when_ready, daemon=True)
    interrupter.start()
    log_path = tmp_path / "kbd.log"
    with pytest.raises(KeyboardInterrupt):
        real_run_with_hard_timeout(
            [sys.executable, "-c", child_code],
            timeout_seconds=15.0,
            kill_grace=1.0,
            log_file=log_path,
            capture=True,
        )
    interrupter.join(timeout=2.0)
    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    # Child process group MUST be dead (ProcessLookupError from os.kill(pid, 0)).
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

    # 6. `_run_chunk_detached` unrecognized-status retry sleep MUST wake immediately on shutdown.
    batch_dir = tmp_path / "unrec_batch"
    batch_dir.mkdir()
    worker = cb.ColabBenchWorker(
        worker_name="hg-unrec-sig",
        gpu="L4",
        batch_dir=batch_dir,
        state_dir=tmp_path / "unrec_state",
        repo_root=tmp_path,
        repo_url=cb.DEFAULT_REPO_URL,
        source_sync={"mode": "git", "ref": "develop", "sha": "abc"},
    )
    job = cb.BenchmarkJob(run_name="unrec-job", config="cifar10-tiny-transformer", gpu="L4", detached=True, poll_seconds=5.0)

    def fake_unrec_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "ssh" and "DETACHED_LAUNCHED" in cmd[-1]:
            return 0, "DETACHED_LAUNCHED pid=1234\n"
        if subcmd == "ssh" and "DETACHED_DONE:" in cmd[-1]:
            return 0, "transient garbage without status token\n"
        return 0, "ok\n"

    sleep_calls: list[float] = []

    def fake_unrec_sleep(secs):
        sleep_calls.append(float(secs))
        if len(sleep_calls) >= 2:
            # Second sleep is the unrecognized-status backoff sleep; requesting shutdown must exit immediately!
            cb._request_shutdown()

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_unrec_colab)
    monkeypatch.setattr(cb.time, "sleep", fake_unrec_sleep)
    try:
        cb._reset_shutdown_state()
        rc_u, out_u, disc_u, _ = worker._run_chunk_detached(
            job=job,
            chunk_index=0,
            quoted_hg="hypergan train cfg",
            remote_run_dir="/content/runs/unrec-job",
            remote_train_log="/content/logs/unrec-job.train.log",
            drive_sync_sh="",
            local_run_dir=tmp_path / "runs" / "unrec-job",
            ssh_timeout=600,
            register_run_local=False,
        )
        assert rc_u == 130
        assert disc_u is False
        assert "Shutdown requested" in out_u
        assert len(sleep_calls) == 2
    finally:
        cb._reset_shutdown_state()


def test_teardown_preserves_stop_failed_sessions_and_cleanup_stops_them(tmp_path, monkeypatch, capsys):
    """Failed `colab stop` must keep sessions tracked as `stop_failed` with isolated config for `cleanup`."""
    state_dir = tmp_path / "runs" / "colab"
    batch_dir = state_dir / "bench-1"
    batch_dir.mkdir(parents=True)

    stop_should_succeed = {"w1": False}
    stop_cmds: list[list[str]] = []

    def fake_flaky_stop(cmd, *, timeout_seconds, **kwargs):
        if "sessions" in cmd:
            return 0, "[colab] No active sessions found on server.\n"
        if "stop" in cmd:
            stop_cmds.append(list(cmd))
            sess = cmd[cmd.index("-s") + 1]
            if stop_should_succeed.get(sess, True):
                return 0, f"stopped {sess}\n"
            return 124, "timeout stopping session\n"
        return 0, "ok\n"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_flaky_stop)

    reg = cb.ActiveSessionRegistry(colab_bin="colab", state_dir=state_dir)
    worker = cb.ColabBenchWorker(
        worker_name="w1",
        gpu="A100",
        batch_dir=batch_dir,
        state_dir=state_dir,
        repo_root=tmp_path,
        repo_url=cb.DEFAULT_REPO_URL,
        source_sync={"mode": "git", "ref": "develop", "sha": "abc"},
        registry=reg,
    )
    worker.cfg.write_text('{"isolated": true}\n', encoding="utf-8")
    worker.alive = True
    reg.register("w1", worker.cfg, gpu="A100", batch_dir=batch_dir)

    # 1. Transient failure in worker.stop() must NOT unregister or mark "stopped".
    assert worker.stop() is False
    assert worker.alive is True
    assert "w1" in reg.sessions
    assert reg.sessions["w1"]["status"] == "stop_failed"
    rec_path = state_dir / "sessions" / "w1.json"
    rec_doc = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec_doc["status"] == "stop_failed"
    assert rec_doc["config_path"] == str(worker.cfg.resolve())

    # 2. Failed stop_all() also keeps session tracked as stop_failed.
    outcomes = reg.stop_all()
    assert outcomes[0]["rc"] == 124
    assert "w1" in reg.sessions
    assert json.loads(rec_path.read_text(encoding="utf-8"))["status"] == "stop_failed"

    # 3. `cleanup` while stop still fails returns non-zero and preserves stop_failed record.
    capsys.readouterr()  # Clear worker.stop() log output from stdout
    rc_fail = cb.main(["--root", str(tmp_path), "--state-dir", str(state_dir), "cleanup"])
    assert rc_fail == 1
    out_fail = json.loads(capsys.readouterr().out)
    assert out_fail["stopped"][0]["session"] == "w1"
    assert out_fail["stopped"][0]["config"] == str(worker.cfg.resolve())
    assert out_fail["stopped"][0]["rc"] == 124
    assert json.loads(rec_path.read_text(encoding="utf-8"))["status"] == "stop_failed"

    # 4. Once transient error clears, `cleanup` finds `stop_failed`, uses isolated `--config`, and marks `stopped`.
    stop_should_succeed["w1"] = True
    stop_cmds.clear()
    rc_ok = cb.main(["--root", str(tmp_path), "--state-dir", str(state_dir), "cleanup"])
    assert rc_ok == 0
    out_ok = json.loads(capsys.readouterr().out)
    assert out_ok["stopped"][0]["session"] == "w1"
    assert out_ok["stopped"][0]["rc"] == 0
    assert any("--config" in c and str(worker.cfg.resolve()) in c and "w1" in c for c in stop_cmds)
    assert json.loads(rec_path.read_text(encoding="utf-8"))["status"] == "stopped"

    # 5. Succeeding `stop_all()` removes session from memory and marks stopped on disk.
    reg.register("w1", worker.cfg, gpu="A100", batch_dir=batch_dir)
    outcomes_ok = reg.stop_all()
    assert outcomes_ok[0]["rc"] == 0
    assert "w1" not in reg.sessions
    assert json.loads(rec_path.read_text(encoding="utf-8"))["status"] == "stopped"


def test_prepare_source_bundle_real_git_bundle_and_dirty_overlay(tmp_path):
    """Real `git bundle create` must succeed for unpushed HEAD & raw SHA, plus handle deletions/spaces/non-ASCII."""
    import subprocess

    def _git(cwd: Path, *args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    origin = tmp_path / "origin.git"
    subprocess.check_call(["git", "init", "--bare", "-b", "develop", str(origin)])

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "develop")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "src" / "hypergan").mkdir(parents=True)
    (repo / "examples").mkdir(parents=True)
    (repo / "src" / "hypergan" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "src" / "hypergan" / "to_delete.py").write_text("old = True\n", encoding="utf-8")
    (repo / "src" / "hypergan" / "old name.py").write_text("rename_me = True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-u", "origin", "develop")

    # Add an unpushed commit on local develop.
    (repo / "src" / "hypergan" / "keep.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "src/hypergan/keep.py")
    _git(repo, "commit", "-m", "unpushed feature")
    unpushed_sha = _git(repo, "rev-parse", "HEAD")

    # 1. Auto mode on unpushed HEAD creates a real verifiable git bundle that fetches into a fresh clone!
    out_auto = tmp_path / "out_auto"
    res_auto = cb.prepare_source_bundle(repo, out_auto, ref="HEAD", source_mode="auto")
    assert res_auto["mode"] == "bundle"
    assert res_auto["bundle_path"] is not None
    assert Path(res_auto["bundle_path"]).is_file()
    assert res_auto["bundle_ref"] == "HEAD"

    # Fetch the bundle into a fresh shallow clone of origin/develop to prove end-to-end validity.
    fresh = tmp_path / "fresh"
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "develop", str(origin), str(fresh)])
    _git(fresh, "fetch", res_auto["bundle_path"], f"{res_auto['bundle_ref']}:refs/heads/colab-branch")
    _git(fresh, "checkout", unpushed_sha)
    assert (fresh / "src" / "hypergan" / "keep.py").read_text(encoding="utf-8") == "x = 2\n"

    # 2. Passing a raw 40-char commit SHA in `bundle` mode ALSO succeeds and cleans up temporary refs.
    out_sha = tmp_path / "out_sha"
    res_sha = cb.prepare_source_bundle(repo, out_sha, ref=unpushed_sha, source_mode="bundle")
    assert res_sha["bundle_path"] is not None and Path(res_sha["bundle_path"]).is_file()
    assert "refs/colab-bench-tmp/" not in _git(repo, "show-ref")
    fresh2 = tmp_path / "fresh2"
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "develop", str(origin), str(fresh2)])
    _git(fresh2, "fetch", res_sha["bundle_path"], f"{res_sha['bundle_ref']}:refs/heads/colab-branch")
    _git(fresh2, "checkout", unpushed_sha)
    assert (fresh2 / "src" / "hypergan" / "keep.py").read_text(encoding="utf-8") == "x = 2\n"

    # 3. Dirty deleted tracked files + git-quoted paths (spaces & non-ASCII) + rename.
    (repo / "src" / "hypergan" / "to_delete.py").unlink()
    _git(repo, "mv", "src/hypergan/old name.py", "src/hypergan/new café.py")
    (repo / "examples" / "my config café.toml").write_text('format = "hypergan-model/1"\n', encoding="utf-8")

    out_dirty = tmp_path / "out_dirty"
    res_dirty = cb.prepare_source_bundle(repo, out_dirty, ref="HEAD", source_mode="auto")
    assert res_dirty["mode"] == "bundle"
    assert res_dirty["overlay_path"] is not None
    with tarfile.open(res_dirty["overlay_path"], "r:gz") as tf:
        overlay_names = set(tf.getnames())
    assert "src/hypergan/new café.py" in overlay_names
    assert "examples/my config café.toml" in overlay_names
    assert res_dirty["deleted_files"] == ["src/hypergan/old name.py", "src/hypergan/to_delete.py"]
    assert res_dirty["deleted_file_count"] == 2

    # 4. Deletion-only dirty worktree on a pushed commit still switches to `bundle` mode with deleted_files.
    _git(repo, "reset", "--hard", "origin/develop")
    (repo / "examples" / "my config café.toml").unlink(missing_ok=True)
    (repo / "src" / "hypergan" / "to_delete.py").unlink()
    res_del_only = cb.prepare_source_bundle(repo, tmp_path / "out_del_only", ref="HEAD", source_mode="auto")
    assert res_del_only["mode"] == "bundle"
    assert res_del_only["overlay_path"] is None
    assert res_del_only["deleted_files"] == ["src/hypergan/to_delete.py"]

    # 5. `_bootstrap_vm` emits properly shell-quoted `rm -f --` for deleted tracked files and `bundle_ref`.
    captured_bootstrap: list[str] = []
    orig_runner = cb.run_with_hard_timeout
    try:
        def _cap_boot(cmd, *, timeout_seconds, **kwargs):
            captured_bootstrap.append(cmd[-1])
            return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"

        cb.run_with_hard_timeout = _cap_boot
        w = cb.ColabBenchWorker(
            worker_name="hg-boot-check",
            gpu="L4",
            batch_dir=tmp_path / "boot_batch",
            state_dir=tmp_path / "boot_state",
            repo_root=repo,
            repo_url=cb.DEFAULT_REPO_URL,
            source_sync=res_dirty,
        )
        w._bootstrap_vm()
    finally:
        cb.run_with_hard_timeout = orig_runner
    assert len(captured_bootstrap) == 1
    assert "rm -f -- 'src/hypergan/old name.py' src/hypergan/to_delete.py" in captured_bootstrap[0]
    assert "git fetch /content/branch.bundle HEAD:refs/heads/colab-branch" in captured_bootstrap[0]


def test_chunked_resume_extends_steps_uses_total_steps_progress_guard_and_schema_version(tmp_path, monkeypatch):
    """Verify `--steps` on resume extension, `latest.json` schema_version=1, `refreshed['total_steps']`, and progress guard."""
    # 1. Fallback latest.json written by pack_local_run_for_resume MUST include schema_version=1.
    run_dir = tmp_path / "runs" / "extend-run"
    ckpt_name = "0001-attempt-step-00000050-deadbeef"
    ckpt_dir = run_dir / "checkpoints" / ckpt_name
    ckpt_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "run-ext", "status": "complete", "steps": 50, "run_dir": str(run_dir)}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "complete", "step": 50, "total_steps": 50}),
        encoding="utf-8",
    )
    (ckpt_dir / "manifest.json").write_text(json.dumps({"step": 50}), encoding="utf-8")
    (ckpt_dir / "state.pt").write_bytes(b"ckpt-50")

    # Extending a completed run (step 50/50 -> target 100) must be recognized as resumable!
    st = cb.inspect_local_run(run_dir, target_step=100)
    assert st["completed"] is False
    assert st["resumable"] is True

    # Missing latest.json -> fallback written with schema_version=1 and step=50.
    resume_tar = tmp_path / "fallback-resume.tar.gz"
    cb.pack_local_run_for_resume(run_dir, resume_tar, remote_run_dir="/content/runs/extend-run")
    with tarfile.open(resume_tar, "r:gz") as tf:
        latest_ptr = json.loads(tf.extractfile("checkpoints/latest.json").read().decode("utf-8"))
    assert latest_ptr["schema_version"] == 1
    assert latest_ptr["checkpoint"] == ckpt_name
    assert latest_ptr["step"] == 50

    # 2. Extending run via execute_benchmark_jobs passes `--steps 100` to `hypergan resume`.
    resume_cmds: list[str] = []
    chunk_steps_seq = iter([75, 100])

    def fake_extend_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "ssh":
            body = cmd[-1]
            if "hypergan resume" in body or "hypergan train" in body:
                resume_cmds.append(body)
            return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
        if subcmd == "download":
            dest = Path(cmd[-1])
            step_now = next(chunk_steps_seq)
            _make_run_tarball(dest, run_name="extend-run", step=step_now, status="stopped", total_steps=100)
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_extend_colab)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    extend_job = cb.BenchmarkJob(
        run_name="extend-run",
        config="cifar10-tiny-transformer",
        gpu="L4",
        steps=100,
        chunk_steps=25,
    )
    summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state1",
        jobs=[extend_job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert summary["status"] == "complete"
    assert len(resume_cmds) == 2
    assert "hypergan resume" in resume_cmds[0]
    assert "--steps 100" in resume_cmds[0]
    assert "--stop-after-steps 25" in resume_cmds[0]
    assert summary["results"][0]["telemetry"]["step"] == 100

    # 3. `--chunk-steps` WITHOUT CLI `--steps`/`--stop-after-steps` uses `refreshed["total_steps"]` (doesn't stop at chunk 1).
    auto_target_steps = iter([25, 50, 75])
    auto_cmds: list[str] = []

    def fake_auto_target_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "ssh":
            body = cmd[-1]
            if "hypergan train" in body or "hypergan resume" in body:
                auto_cmds.append(body)
            return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
        if subcmd == "download":
            dest = Path(cmd[-1])
            step_now = next(auto_target_steps)
            status_now = "complete" if step_now == 75 else "stopped"
            _make_run_tarball(dest, run_name="auto-target", step=step_now, status=status_now, total_steps=75)
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_auto_target_colab)
    auto_job = cb.BenchmarkJob(
        run_name="auto-target",
        config="cifar10-tiny-transformer",
        gpu="L4",
        steps=None,
        stop_after_steps=None,
        chunk_steps=25,
    )
    auto_summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state2",
        jobs=[auto_job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert auto_summary["status"] == "complete"
    assert len(auto_cmds) == 3
    assert "hypergan train" in auto_cmds[0]
    assert "hypergan resume" in auto_cmds[1]
    assert "hypergan resume" in auto_cmds[2]
    assert auto_summary["results"][0]["chunks_completed"] == 3
    assert auto_summary["results"][0]["telemetry"]["step"] == 75

    # 4. Progress guard: if `new_step <= prev_step` after a completed chunk, break with error instead of infinite loop!
    stuck_chunks = {"n": 0}
    stuck_boots: list[str] = []

    def fake_stuck_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "new":
            stuck_boots.append(cmd[cmd.index("-s") + 1])
            return 0, "created"
        if subcmd == "ssh":
            body = cmd[-1]
            if "hypergan train" in body or "hypergan resume" in body:
                stuck_chunks["n"] += 1
            return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
        if subcmd == "download":
            dest = Path(cmd[-1])
            # Always stuck at step 25 even after chunk 2!
            _make_run_tarball(dest, run_name="stuck-run", step=25, status="stopped", total_steps=100)
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_stuck_colab)
    stuck_job = cb.BenchmarkJob(
        run_name="stuck-run",
        config="cifar10-tiny-transformer",
        gpu="L4",
        stop_after_steps=100,
        chunk_steps=25,
        max_retries=2,
    )
    stuck_summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state3",
        jobs=[stuck_job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert stuck_summary["status"] == "partial_or_failed"
    assert stuck_chunks["n"] == 2  # Chunk 1 reached 25, chunk 2 stayed at 25 -> broke immediately!
    assert len(stuck_boots) == 1  # Did not reprovision on deterministic stall
    assert stuck_summary["results"][0]["run_rc"] == 1
    assert "no step progress" in stuck_summary["results"][0]["error"]


def test_foreground_sync_no_concurrent_ssh_and_unique_unpack_tmp_files(tmp_path, monkeypatch):
    """Foreground mode must never overlap `colab ssh` sessions, and concurrent `unpack_run_tarball` must never collide."""
    import threading
    import time

    # 1. Foreground mode with sync_interval_minutes > 0 must keep max concurrent SSH == 1.
    ssh_active = {"n": 0, "max": 0}
    ssh_guard = threading.Lock()

    def fake_fg_colab(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "ssh":
            with ssh_guard:
                ssh_active["n"] += 1
                ssh_active["max"] = max(ssh_active["max"], ssh_active["n"])
            try:
                if "hypergan train" in cmd[-1]:
                    time.sleep(0.05)
                return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"
            finally:
                with ssh_guard:
                    ssh_active["n"] -= 1
        if subcmd == "download":
            dest = Path(cmd[-1])
            _make_run_tarball(dest, run_name="fg-sync", step=50, status="stopped")
            return 0, "downloaded"
        return 0, "ok"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_fg_colab)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    fg_job = cb.BenchmarkJob(
        run_name="fg-sync",
        config="cifar10-tiny-transformer",
        gpu="L4",
        stop_after_steps=50,
        chunk_steps=0,
        sync_interval_minutes=1.0,
        detached=False,
    )
    fg_summary = cb.execute_benchmark_jobs(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "fg_state",
        jobs=[fg_job],
        workers=1,
        stagger_seconds=0.0,
        register_runs=False,
    )
    assert fg_summary["status"] == "complete"
    assert ssh_active["max"] == 1

    # 2. Concurrent `unpack_run_tarball` calls from multiple threads into the same dir must not collide on temp files.
    tar_path = _make_run_tarball(tmp_path / "shared.tar.gz", run_name="race-run", step=50)
    dest_dir = tmp_path / "runs" / "race-run"
    errors: list[Exception] = []
    barrier = threading.Barrier(6)

    def _unpack_worker() -> None:
        try:
            barrier.wait()
            for _ in range(5):
                info = cb.unpack_run_tarball(tar_path, dest_dir, remote_run_dir="/content/runs/race-run")
                assert info["has_summary"] and info["has_manifest"]
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_unpack_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert errors == []
    assert json.loads((dest_dir / "summary.json").read_text(encoding="utf-8"))["step"] == 50


def test_edge_cases_particlegan_spec_bundle_collision_nan_re_and_sessions_parser(tmp_path, monkeypatch):
    """Cover edge cases: DEFAULT_PARTICLEGAN_SPEC, duplicate bundle IDs, tightened _NAN_LOG_RE, and session table headers."""
    # 1. DEFAULT_PARTICLEGAN_SPEC updated for current develop (no stale 0.12.0 pin).
    assert "0.12.0" not in cb.DEFAULT_PARTICLEGAN_SPEC
    assert "ParticleGAN.git@develop" in cb.DEFAULT_PARTICLEGAN_SPEC or "0.13" in cb.DEFAULT_PARTICLEGAN_SPEC

    # 2. Duplicate `--bundle` names in `run_bundle_evaluations` must upload to distinct remote filenames.
    dir_a = tmp_path / "exp_a" / "shared"
    dir_b = tmp_path / "exp_b" / "shared"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    b_a = dir_a / "model.pt"
    b_b = dir_b / "model.pt"
    b_a.write_bytes(b"weights-a")
    b_b.write_bytes(b"weights-b")

    uploaded_bundles: list[str] = []
    eval_spec_uploaded: dict[str, Any] = {}

    def fake_dup_eval(cmd, *, timeout_seconds, **kwargs):
        subcmd = cmd[cmd.index("--config") + 2]
        if subcmd == "upload":
            remote_p = cmd[-1]
            if remote_p.endswith(".pt"):
                uploaded_bundles.append(remote_p)
            elif remote_p.endswith("eval-spec.json"):
                eval_spec_uploaded.update(json.loads(Path(cmd[-2]).read_text(encoding="utf-8")))
            return 0, "uploaded"
        if subcmd == "download":
            dest = Path(cmd[-1])
            with tarfile.open(dest, "w:gz") as tf:
                payload = b"[]\n"
                ti = tarfile.TarInfo(name="all-receipts.json")
                ti.size = len(payload)
                tf.addfile(ti, fileobj=__import__("io").BytesIO(payload))
            return 0, "downloaded"
        return 0, "COLAB_GPU_INFO: NVIDIA L4, 23034 MiB, 550.54\n"

    monkeypatch.setattr(cb, "run_with_hard_timeout", fake_dup_eval)
    monkeypatch.setattr(cb, "prepare_source_bundle", lambda *a, **kw: {"mode": "git", "ref": "develop", "sha": "abc"})
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    cb.execute_eval_bundles(
        repo_root=tmp_path,
        state_dir=tmp_path / "eval_state",
        bundles=[
            {"name": "shared", "bundle_path": str(b_a), "run_dir": None},
            {"name": "shared", "bundle_path": str(b_b), "run_dir": None},
        ],
        metrics=["InceptionSuite"],
        dataset_config={"builtin": "cifar10", "args": {"root": "~/.hypergan/cache/cifar10", "split": "train"}},
        gpu="L4",
        output_dir=tmp_path / "eval_out",
    )
    assert len(uploaded_bundles) == 2
    assert len(set(uploaded_bundles)) == 2
    spec_names = [b["name"] for b in eval_spec_uploaded["bundles"]]
    assert len(set(spec_names)) == 2
    assert spec_names[0] == "shared"
    assert spec_names[1].startswith("shared-1-")

    # 3. Tightened `_NAN_LOG_RE`: benign text must NOT false-positive, while real NaN/Inf/divergence MUST match.
    benign_dir = tmp_path / "benign-run"
    benign_dir.mkdir()
    (benign_dir / "summary.json").write_text(
        json.dumps({"status": "complete", "step": 100, "latest": {"loss/d_total": 0.4, "diversity/ratio": 0.9}}),
        encoding="utf-8",
    )
    (benign_dir / "train.log").write_text(
        "\n".join(
            [
                "Note: non-finite gradient screening is enabled by default",
                "# Help: --abort-on-nan skips non-finite steps gracefully",
                "UserWarning: preview panel collapsed in narrow terminal",
                "git note: local branch diverged from upstream",
                "step 100 loss/d_total=0.412 loss/g_total=1.084 info=ok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benign_tel = cb.analyze_run_telemetry(benign_dir)
    assert benign_tel["nan_detected"] is False
    assert benign_tel["collapse_detected"] is False
    assert benign_tel["log_warnings"] == []

    for real_line in (
        "step 420 loss=nan",
        "step 420 d_loss=-inf g_loss=1.2",
        "loss/g_total: NaN",
        "FloatingPointError: NaN detected in generator loss",
        "RuntimeError: Non-finite loss encountered at step 500",
        "ERROR: detected non-finite gradients",
        "gradients exploded at step 100",
        "training diverged at step 200",
    ):
        assert cb._NAN_LOG_RE.search(real_line), f"Expected _NAN_LOG_RE to match: {real_line!r}"

    # 4. Tightened `parse_colab_sessions_output`: skips table header & divider rows, preserves real session names.
    table_out = "\n".join(
        [
            "[colab] Active sessions:",
            "Session | Hardware | Status",
            "--------+----------+-------",
            "hg-l4-w1 | Hardware: L4 | READY",
            "Name | GPU | State",
            "--- | --- | ---",
            "│ Session │ Hardware │ Status │",
            "├─────────┼──────────┼────────┤",
            "│ hg-a100-w2 │ A100 │ READY │",
            "* orphan-vm | Hardware: T4",
        ]
    )
    assert cb.parse_colab_sessions_output(table_out) == ["hg-l4-w1", "hg-a100-w2", "orphan-vm"]

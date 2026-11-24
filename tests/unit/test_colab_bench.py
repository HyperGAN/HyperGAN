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

    def fake_eval_colab(cmd, *, timeout_seconds, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        subcmd = cmd[cfg_idx + 1]
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
                        "fd_dinov2": {"label": "FD DINOv2", "unit": "FD", "direction": "minimize"},
                        "precision": {"label": "Precision", "unit": "", "direction": "maximize"},
                        "recall": {"label": "Recall", "unit": "", "direction": "maximize"},
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
    assert (run_dir / "summary.json").is_file()
    imported_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    eval_metrics = {e["metric"]: e["value"] for e in imported_summary.get("evaluations", [])}
    assert eval_metrics["dinov2suite/fd_dinov2"] == pytest.approx(42.5)
    assert eval_metrics["dinov2suite/precision"] == pytest.approx(0.81)


def test_eval_runner_dataset_lowering_uses_real_hypergan_factory():
    # Confirm scripts/colab_bench.py does not reference nonexistent private model_file helpers.
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "_lower_data_builtin" not in src
    assert "hypergan.image_data:CIFAR10Data" in src



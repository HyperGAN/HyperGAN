"""Empirical stress tests for hypergan.queue: Concurrency, Advisory Flock,
Collision Avoidance, Subprocess Failure Isolation, and Systemd Service Unit Generation.
"""
import configparser
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest import mock

import pytest

import hypergan as hg
from hypergan import cli
from hypergan.queue import (
    find_queue_file,
    queue_lock,
    load_queue,
    save_queue,
    inspect_gpu_hardware,
    inspect_gpu_processes,
    check_device_availability,
    queue_add,
    queue_ls,
    queue_status,
    queue_stop,
    queue_run,
    install_service,
    format_queue_table,
    format_queue_status,
)


@pytest.fixture
def temp_queue(tmp_path):
    q_dir = tmp_path / ".hypergan"
    q_dir.mkdir(parents=True, exist_ok=True)
    q_file = q_dir / "queue.json"
    return q_file


# Helper worker functions for multiprocessing stress testing
def _multiprocess_adder_worker(queue_path_str: str, num_jobs: int, worker_id: int):
    q_path = Path(queue_path_str)
    for i in range(num_jobs):
        model_name = f"model_w{worker_id}_{i}.toml"
        queue_add(model_name, devices="gpu0", name=f"job_w{worker_id}_{i}", queue_file=q_path)
        time.sleep(0.002)


def _multiprocess_reader_worker(queue_path_str: str, iterations: int):
    q_path = Path(queue_path_str)
    for _ in range(iterations):
        jobs = queue_ls(queue_file=q_path)
        assert isinstance(jobs, list)
        for j in jobs:
            assert "id" in j
            assert "status" in j
        time.sleep(0.003)


def _multiprocess_status_and_cancel_worker(queue_path_str: str, iterations: int):
    q_path = Path(queue_path_str)
    for _ in range(iterations):
        jobs = queue_ls(queue_file=q_path)
        if jobs:
            target = jobs[-1]
            st = queue_status(job_id=target["id"], queue_file=q_path)
            assert st["active_job"]["id"] == target["id"]
            if target.get("status") == "queued" and target["name"].endswith("_0"):
                queue_stop(job_id=target["id"], queue_file=q_path)
        time.sleep(0.005)


def test_concurrent_multiprocess_queue_operations(temp_queue, monkeypatch):
    """Stress-test advisory flock across independent parallel OS processes."""
    # Spawn must be able to import this test module outside the checkout.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    ctx = multiprocessing.get_context("spawn")
    processes = []

    # 4 Adder processes, each adding 15 jobs = 60 jobs total
    for w in range(4):
        p = ctx.Process(target=_multiprocess_adder_worker, args=(str(temp_queue), 15, w))
        processes.append(p)

    # 2 Concurrent Reader processes
    for _ in range(2):
        p = ctx.Process(target=_multiprocess_reader_worker, args=(str(temp_queue), 30))
        processes.append(p)

    # 2 Concurrent Status & Selective Canceller processes
    for _ in range(2):
        p = ctx.Process(target=_multiprocess_status_and_cancel_worker, args=(str(temp_queue), 20))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join(timeout=30)
        assert not p.is_alive(), "Worker process timed out during queue stress test"
        assert p.exitcode == 0, f"Worker process failed with exit code {p.exitcode}"

    # Verify final queue.json file integrity
    assert temp_queue.is_file(), "queue.json must exist after stress test"
    raw_content = temp_queue.read_text(encoding="utf-8")
    data = json.loads(raw_content)

    assert data.get("schema_version") == 1
    jobs = data.get("jobs", [])
    assert len(jobs) == 60, f"Expected 60 enqueued jobs, found {len(jobs)}"
    assert data.get("next_job_id") == 61, f"Expected next_job_id 61, got {data.get('next_job_id')}"

    # Verify all job IDs are unique and strictly monotonically sequential
    job_ids = [j["id"] for j in jobs]
    assert len(job_ids) == len(set(job_ids)), "Duplicate job IDs detected in queue under concurrency!"
    expected_ids = [f"job-{i:04d}" for i in range(1, 61)]
    assert job_ids == expected_ids, "Job IDs were not strictly sequential"

    # Verify no corrupted job entries
    for j in jobs:
        assert j["status"] in ("queued", "cancelled")
        if j["status"] == "cancelled":
            assert j["error"] == "Cancelled by user"
            assert j["ended_at"] is not None
        else:
            assert j["error"] is None


def test_concurrent_multithread_queue_operations(temp_queue):
    """Stress-test flock and atomic writes under high multithread contention."""
    errors = []

    def _thread_adder(tid: int, count: int):
        try:
            for i in range(count):
                queue_add(f"model_t{tid}_{i}.toml", name=f"t_{tid}_{i}", queue_file=temp_queue)
        except Exception as e:
            errors.append(e)

    def _thread_reader(count: int):
        try:
            for _ in range(count):
                jobs = queue_ls(queue_file=temp_queue)
                assert isinstance(jobs, list)
        except Exception as e:
            errors.append(e)

    threads = []
    # 8 Adder threads, each adding 10 jobs = 80 jobs
    for tid in range(8):
        t = threading.Thread(target=_thread_adder, args=(tid, 10))
        threads.append(t)
    # 4 Reader threads
    for _ in range(4):
        t = threading.Thread(target=_thread_reader, args=(30,))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive()

    assert not errors, f"Encountered thread exceptions: {errors}"

    jobs = queue_ls(queue_file=temp_queue)
    assert len(jobs) == 80
    assert len(set(j["id"] for j in jobs)) == 80


def test_hardware_preflight_collision_avoidance_low_memory(temp_queue, monkeypatch, capsys):
    """Verify runner detects low GPU memory, waits without colliding, and proceeds when memory frees."""
    model_file = temp_queue.parent / "test_model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    job = queue_add(model_file, devices="0", queue_file=temp_queue)

    state = {"memory_free": 1024.0, "ran_job": False}  # 1024 MiB < 2048 MiB threshold

    def _mock_hw():
        return [{
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "uuid": "GPU-001",
            "memory_total": 48000.0,
            "memory_free": state["memory_free"],
            "memory_used": 48000.0 - state["memory_free"],
            "utilization": 80.0,
        }]

    class _MockPopen:
        def __init__(self, cmd, **kwargs):
            state["ran_job"] = True
            self.pid = 12345
            self.cmd = cmd

        def wait(self):
            return 0

    monkeypatch.setattr("hypergan.queue.inspect_gpu_hardware", _mock_hw)
    monkeypatch.setattr("hypergan.queue.inspect_gpu_processes", lambda: [])
    monkeypatch.setattr(subprocess, "Popen", _MockPopen)

    # Background thread will wait 0.15s while runner detects busy, then free the memory
    def _free_memory_after_delay():
        time.sleep(0.15)
        # Verify job is still queued while memory is low
        j = queue_ls(queue_file=temp_queue)[0]
        assert j["status"] == "queued"
        assert state["ran_job"] is False
        state["memory_free"] = 32000.0  # Above 2048 MiB threshold

    helper = threading.Thread(target=_free_memory_after_delay)
    helper.start()

    # Run queue_run in main thread (so signal.signal succeeds)
    ret = queue_run(poll_interval=0.04, queue_file=temp_queue, once=True)
    helper.join(timeout=5.0)

    assert ret == 0
    assert state["ran_job"] is True

    final_jobs = queue_ls(queue_file=temp_queue)
    assert final_jobs[0]["status"] == "completed"
    assert final_jobs[0]["exit_code"] == 0

    err = capsys.readouterr().err
    assert "[queue] Hardware busy for job-0001" in err
    assert "free memory (1024 MiB) < threshold" in err


def test_hardware_preflight_collision_avoidance_conflicting_process(temp_queue, monkeypatch, capsys):
    """Verify runner detects active hypergan/train process on GPU and avoids collision."""
    model_file = temp_queue.parent / "test_model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    queue_add(model_file, devices="0", queue_file=temp_queue)

    state = {"active_process": True, "executed": False}

    def _mock_hw():
        return [{
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "uuid": "GPU-001",
            "memory_total": 48000.0,
            "memory_free": 30000.0,
            "memory_used": 18000.0,
            "utilization": 90.0,
        }]

    def _mock_procs():
        if state["active_process"]:
            return [{
                "gpu_uuid": "GPU-001",
                "pid": 5555,
                "process_name": "python",
                "used_memory": 15000.0,
                "cmdline": "python -m hypergan train existing_experiment.toml",
            }]
        return []

    class _MockPopen:
        def __init__(self, cmd, **kwargs):
            state["executed"] = True
            self.pid = 9999

        def wait(self):
            return 0

    monkeypatch.setattr("hypergan.queue.inspect_gpu_hardware", _mock_hw)
    monkeypatch.setattr("hypergan.queue.inspect_gpu_processes", _mock_procs)
    monkeypatch.setattr(subprocess, "Popen", _MockPopen)

    def _terminate_process_after_delay():
        time.sleep(0.15)
        assert state["executed"] is False
        assert queue_ls(queue_file=temp_queue)[0]["status"] == "queued"
        state["active_process"] = False

    helper = threading.Thread(target=_terminate_process_after_delay)
    helper.start()

    ret = queue_run(poll_interval=0.04, queue_file=temp_queue, once=True)
    helper.join(timeout=5.0)

    assert ret == 0
    assert state["executed"] is True
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "completed"

    err = capsys.readouterr().err
    assert "[queue] Hardware busy for job-0001" in err
    assert "is busy running process 5555" in err


def test_active_run_directory_lock_collision_avoidance(temp_queue, tmp_path, monkeypatch, capsys):
    """Verify runner detects active .run.lock in target run_dir and avoids colliding."""
    run_dir = tmp_path / "colliding_run"
    run_dir.mkdir(parents=True)
    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")

    job = queue_add(model_file, run_dir=run_dir, queue_file=temp_queue)
    state = {"executed": False}

    class _MockPopen:
        def __init__(self, cmd, **kwargs):
            state["executed"] = True
            self.pid = 8888

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _MockPopen)

    # Acquire an exclusive lock on .run.lock in the run_dir
    lock_file = run_dir / ".run.lock"
    lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
    fcntl = pytest.importorskip("fcntl", reason="POSIX advisory lock interoperability")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    def _release_lock_after_delay():
        time.sleep(0.15)
        assert state["executed"] is False
        assert queue_ls(queue_file=temp_queue)[0]["status"] == "queued"
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    helper = threading.Thread(target=_release_lock_after_delay)
    helper.start()

    ret = queue_run(poll_interval=0.04, queue_file=temp_queue, once=True)
    helper.join(timeout=5.0)

    assert ret == 0
    assert state["executed"] is True
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "completed"

    err = capsys.readouterr().err
    assert "locked by active process. Retrying" in err


def test_wait_kind_job_completion_flow(temp_queue, tmp_path):
    """Verify kind='wait' job pauses until manifest status reaches terminal state."""
    target_rd = tmp_path / "wait_target_run"
    target_rd.mkdir(parents=True)
    manifest_file = target_rd / "manifest.json"
    manifest_file.write_text(json.dumps({"status": "running"}), encoding="utf-8")

    wait_job = queue_add(target_rd, wait=True, queue_file=temp_queue)
    assert wait_job["kind"] == "wait"

    def _mark_terminal_after_delay():
        time.sleep(0.12)
        # While still running, verify job remains queued
        jobs = queue_ls(queue_file=temp_queue)
        assert jobs[0]["status"] == "queued"
        manifest_file.write_text(json.dumps({"status": "stopped"}), encoding="utf-8")

    helper = threading.Thread(target=_mark_terminal_after_delay)
    helper.start()

    ret = queue_run(poll_interval=0.04, queue_file=temp_queue, once=True)
    helper.join(timeout=5.0)

    assert ret == 0
    jobs = queue_ls(queue_file=temp_queue)
    assert jobs[0]["status"] == "completed"


def test_subprocess_failure_isolation_and_runner_continuity(temp_queue, tmp_path, monkeypatch):
    """Verify failing job records status='failed' with exit code and error, and runner cleanly continues to execute subsequent jobs."""
    m1 = tmp_path / "job1_failing.toml"
    m1.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    m2 = tmp_path / "job2_succeeding.toml"
    m2.write_text('format = "hypergan-model/1"\n', encoding="utf-8")

    j1 = queue_add(m1, name="failing_exp", queue_file=temp_queue)
    j2 = queue_add(m2, name="succeeding_exp", queue_file=temp_queue)

    launched_commands = []

    class _MockSequentialPopen:
        def __init__(self, cmd, **kwargs):
            launched_commands.append(cmd)
            self.cmd = cmd
            self.pid = 1000 + len(launched_commands)
            stdout = kwargs.get("stdout")
            if stdout:
                if "job1_failing.toml" in " ".join(cmd):
                    stdout.write(b"Error: Out of memory on rank 0\n")
                else:
                    stdout.write(b"Training successful step 100\n")
                stdout.flush()

        def wait(self):
            # Job 1 fails with code 1; Job 2 succeeds with code 0
            if "job1_failing.toml" in " ".join(self.cmd):
                return 1
            return 0

    monkeypatch.setattr(subprocess, "Popen", _MockSequentialPopen)

    def _monitor_and_stop():
        # Wait until job 2 finishes, then touch .stop file
        deadline = time.time() + 5.0
        while time.time() < deadline:
            jobs = queue_ls(queue_file=temp_queue)
            statuses = [j["status"] for j in jobs]
            if statuses == ["failed", "completed"]:
                stop_file = temp_queue.with_name(temp_queue.name + ".stop")
                stop_file.touch()
                return
            time.sleep(0.02)

    helper = threading.Thread(target=_monitor_and_stop)
    helper.start()

    # Run in main thread
    ret = queue_run(poll_interval=0.03, queue_file=temp_queue)
    helper.join(timeout=5.0)

    assert ret == 0

    final_jobs = queue_ls(queue_file=temp_queue)
    assert len(final_jobs) == 2

    # Job 1 assertions: failure isolation
    job1 = final_jobs[0]
    assert job1["id"] == j1["id"]
    assert job1["status"] == "failed"
    assert job1["exit_code"] == 1
    assert "Process exited with code 1" in job1["error"]
    assert job1["started_at"] is not None
    assert job1["ended_at"] is not None
    j1_log = Path(job1["log_file"]).read_text(encoding="utf-8")
    assert "Error: Out of memory on rank 0" in j1_log

    # Job 2 assertions: cleanly executed despite Job 1 failure
    job2 = final_jobs[1]
    assert job2["id"] == j2["id"]
    assert job2["status"] == "completed"
    assert job2["exit_code"] == 0
    assert job2["error"] is None
    assert job2["started_at"] is not None
    assert job2["ended_at"] is not None
    j2_log = Path(job2["log_file"]).read_text(encoding="utf-8")
    assert "Training successful step 100" in j2_log

    # Verify current.log symlink pointed to job2 log
    current_symlink = temp_queue.parent / "logs" / "current.log"
    assert current_symlink.is_symlink()
    assert current_symlink.resolve() == Path(job2["log_file"]).resolve()


def test_systemd_install_service_syntax_and_validation(tmp_path):
    """Stress-test systemd service generation across device topologies and validate INI syntax + systemd-analyze."""
    scenarios = [
        {"devices": None, "name": None, "expected_svc": "hypergan-queue.service"},
        {"devices": "gpu0", "name": None, "expected_svc": "hypergan-queue-gpu0.service"},
        {"devices": "gpu0,gpu1", "name": None, "expected_svc": "hypergan-queue-gpu0-gpu1.service"},
        {"devices": "0,1", "name": "custom-service", "expected_svc": "custom-service.service"},
    ]

    systemd_analyze_bin = shutil.which("systemd-analyze")

    for sc in scenarios:
        unit_path, instructions = install_service(
            devices=sc["devices"],
            name=sc["name"],
            dry_run=True,
        )

        assert unit_path.name == sc["expected_svc"]
        assert f"systemctl --user enable --now {unit_path.stem}" in instructions

        # Extract unit file content from instructions
        lines = instructions.splitlines()
        content_lines = []
        capturing = False
        for l in lines:
            if l.startswith("[Unit]"):
                capturing = True
            if capturing:
                if l.startswith("Installed systemd service:"):
                    break
                content_lines.append(l)

        unit_content = "\n".join(content_lines).strip()
        assert "[Unit]" in unit_content
        assert "[Service]" in unit_content
        assert "[Install]" in unit_content

        # INI Syntax parse check
        config = configparser.ConfigParser()
        config.read_string(unit_content)

        assert config.has_section("Unit")
        assert config.has_section("Service")
        assert config.has_section("Install")

        svc = config["Service"]
        assert svc.get("Type") == "simple"
        assert svc.get("Restart") == "on-failure"
        assert svc.get("RestartSec") == "10"
        assert "PYTHONUNBUFFERED=1" in svc.get("Environment", "")

        working_dir = Path(svc.get("WorkingDirectory"))
        assert working_dir.is_absolute()
        assert working_dir.exists()

        exec_start = svc.get("ExecStart")
        assert sys.executable in exec_start
        assert "-m hypergan queue run" in exec_start
        if sc["devices"]:
            assert f"--devices {sc['devices']}" in exec_start

        # Validate with systemd-analyze verify if present
        if systemd_analyze_bin:
            temp_unit = tmp_path / sc["expected_svc"]
            temp_unit.write_text(unit_content, encoding="utf-8")
            res = subprocess.run(
                [systemd_analyze_bin, "--user", "verify", str(temp_unit)],
                capture_output=True,
                text=True
            )
            # systemd-analyze verify prints warnings or errors; exit code 0 indicates valid syntax
            assert res.returncode == 0, f"systemd-analyze failed on {sc['expected_svc']}: {res.stderr}"


def test_cli_queue_e2e_stress(temp_queue, monkeypatch, capsys):
    """End-to-end stress of hypergan queue CLI commands with real flag combinations."""
    monkeypatch.setenv("HYPERGAN_QUEUE_FILE", str(temp_queue))

    # 1. Add multiple jobs
    assert cli.main(["queue", "add", "exp1.toml", "-d", "gpu0", "--name", "first"]) == 0
    assert cli.main(["queue", "add", "exp2.toml", "-d", "gpu1", "--name", "second"]) == 0
    assert cli.main(["queue", "add", "exp3.toml", "-d", "gpu0,gpu1", "--name", "third"]) == 0

    capsys.readouterr()

    # 2. List tabular and JSON
    assert cli.main(["queue", "ls"]) == 0
    out_table = capsys.readouterr().out
    assert "job-0001" in out_table
    assert "first" in out_table
    assert "job-0002" in out_table
    assert "job-0003" in out_table

    assert cli.main(["queue", "ls", "--json"]) == 0
    out_json = json.loads(capsys.readouterr().out)
    assert len(out_json) == 3
    assert [j["id"] for j in out_json] == ["job-0001", "job-0002", "job-0003"]

    # 3. Status checks
    assert cli.main(["queue", "status", "job-0002"]) == 0
    out_st = capsys.readouterr().out
    assert "Job ID:         job-0002" in out_st
    assert "Name:           second" in out_st

    assert cli.main(["queue", "status", "--json"]) == 0
    out_st_json = json.loads(capsys.readouterr().out)
    assert out_st_json["active_job"]["id"] == "job-0001"

    # 4. Stop single job
    assert cli.main(["queue", "stop", "job-0002"]) == 0
    out_stop = capsys.readouterr().out
    assert "Cancelled queued job job-0002" in out_stop

    jobs_after = queue_ls(queue_file=temp_queue)
    assert jobs_after[1]["status"] == "cancelled"
    assert jobs_after[0]["status"] == "queued"
    assert jobs_after[2]["status"] == "queued"

    # 5. Stop all remaining jobs
    assert cli.main(["queue", "stop", "--all"]) == 0
    out_stop_all = capsys.readouterr().out
    assert "Cancelled 2 queued job(s)" in out_stop_all

    all_stopped = queue_ls(queue_file=temp_queue)
    assert all(j["status"] == "cancelled" for j in all_stopped)

    # 6. Service generator dry run
    assert cli.main(["queue", "install-service", "--dry-run", "-d", "gpu0,gpu1"]) == 0
    svc_out = capsys.readouterr().out
    assert "hypergan-queue-gpu0-gpu1.service" in svc_out
    assert "ExecStart=" in svc_out
    assert "--devices gpu0,gpu1" in svc_out


def test_real_subprocess_failure_isolation_unmocked(temp_queue, tmp_path):
    """Empirical unmocked test: spawn real Python subprocess on broken model TOML and verify failure capture."""
    broken_toml = tmp_path / "syntax_error.toml"
    broken_toml.write_text("invalid [[[ toml syntax = true", encoding="utf-8")

    job = queue_add(broken_toml, queue_file=temp_queue)

    ret = queue_run(once=True, queue_file=temp_queue)
    # Runner returns 1 when once=True and job failed
    assert ret == 1

    jobs = queue_ls(queue_file=temp_queue)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["exit_code"] == 1
    assert "Process exited with code 1" in jobs[0]["error"]

    log_path = Path(jobs[0]["log_file"])
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "error:" in log_text.lower()


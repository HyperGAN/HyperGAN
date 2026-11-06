import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
    q_file = tmp_path / ".hypergan" / "queue.json"
    return q_file


def test_queue_add_and_ls(temp_queue):
    job1 = queue_add("model_a.toml", devices="gpu0", queue_file=temp_queue)
    assert job1["id"] == "job-0001"
    assert job1["name"] == "model_a"
    assert job1["status"] == "queued"
    assert job1["devices"] == "gpu0"

    job2 = queue_add("model_b.toml", devices="gpu0,gpu1", name="custom_b", queue_file=temp_queue)
    assert job2["id"] == "job-0002"
    assert job2["name"] == "custom_b"
    assert job2["status"] == "queued"

    all_jobs = queue_ls(queue_file=temp_queue)
    assert len(all_jobs) == 2
    assert all_jobs[0]["id"] == "job-0001"
    assert all_jobs[1]["id"] == "job-0002"

    queued_jobs = queue_ls(status_filter="queued", queue_file=temp_queue)
    assert len(queued_jobs) == 2

    running_jobs = queue_ls(status_filter="running", queue_file=temp_queue)
    assert len(running_jobs) == 0


def test_queue_wait_entry(temp_queue, tmp_path):
    run_dir = tmp_path / "train-experiment"
    run_dir.mkdir(parents=True)
    job = queue_add(run_dir, wait=True, queue_file=temp_queue)
    assert job["kind"] == "wait"
    assert job["run_dir"] == str(run_dir.resolve())
    assert job["name"] == f"wait-{run_dir.name}"


def test_queue_status_with_summary(temp_queue, tmp_path):
    run_dir = tmp_path / "train-test"
    run_dir.mkdir(parents=True)
    summary_data = {
        "schema_version": 1,
        "run_id": "test-run-123",
        "step": 450,
        "total_steps": 1000,
        "steps_per_sec": 12.5,
        "eta_seconds": 44.0,
        "latest": {"fid50k_train": 18.25, "loss_g": 0.42},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    job = queue_add("model.toml", run_dir=run_dir, queue_file=temp_queue)
    status_info = queue_status(job_id=job["id"], queue_file=temp_queue)

    assert status_info["jobs_total"] == 1
    assert status_info["active_job"]["id"] == job["id"]
    assert status_info["summary"]["step"] == 450
    assert status_info["summary"]["steps_per_sec"] == 12.5

    formatted = format_queue_status(status_info)
    assert "Step 450/1000 (12.50 it/s)" in formatted
    assert "fid50k_train=18.25" in formatted


def test_queue_status_log_tail(temp_queue, tmp_path):
    job = queue_add("model.toml", queue_file=temp_queue)
    log_file = Path(job["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")

    status_info = queue_status(job_id=job["id"], queue_file=temp_queue, lines=5)
    assert len(status_info["log_tail"]) == 5
    assert status_info["log_tail"][-1] == "line 49"
    assert status_info["log_tail"][0] == "line 45"


def test_queue_stop_specific(temp_queue):
    job = queue_add("model.toml", queue_file=temp_queue)
    res = queue_stop(job_id=job["id"], queue_file=temp_queue)
    assert res["status"] == "cancelled"

    jobs = queue_ls(queue_file=temp_queue)
    assert jobs[0]["status"] == "cancelled"
    assert jobs[0]["error"] == "Cancelled by user"


def test_queue_stop_all(temp_queue):
    queue_add("model_1.toml", queue_file=temp_queue)
    queue_add("model_2.toml", queue_file=temp_queue)

    res = queue_stop(stop_all=True, queue_file=temp_queue)
    assert res["stopped_all"] is True
    assert res["cancelled_count"] == 2

    # Runners halt on the generation bump; no stop file is left behind for later runners.
    assert res["stop_generation"] == 1
    assert load_queue(temp_queue)["stop_generation"] == 1
    assert not temp_queue.with_name(temp_queue.name + ".stop").exists()

    jobs = queue_ls(queue_file=temp_queue)
    assert all(j["status"] == "cancelled" for j in jobs)


def test_queue_runner_once_and_completion(temp_queue, tmp_path, monkeypatch):
    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    job = queue_add(model_file, queue_file=temp_queue)

    # Mock subprocess.Popen to simulate a successful training run
    class MockPopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 99999
            self.cmd = cmd
            stdout = kwargs.get("stdout")
            if stdout:
                stdout.write(b"Training completed step 10\n")
                stdout.flush()

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", MockPopen)
    code = queue_run(once=True, queue_file=temp_queue)
    assert code == 0

    jobs = queue_ls(queue_file=temp_queue)
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["exit_code"] == 0
    assert jobs[0]["started_at"] is not None
    assert jobs[0]["ended_at"] is not None

    log_path = Path(jobs[0]["log_file"])
    assert log_path.is_file()
    assert "Training completed step 10" in log_path.read_text(encoding="utf-8")


def test_queue_runner_failure_handling(temp_queue, tmp_path, monkeypatch):
    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    job = queue_add(model_file, queue_file=temp_queue)

    class MockFailingPopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 99998
            self.cmd = cmd
            stdout = kwargs.get("stdout")
            if stdout:
                stdout.write(b"CUDA Out of Memory\n")
                stdout.flush()

        def wait(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", MockFailingPopen)
    code = queue_run(once=True, queue_file=temp_queue)
    assert code == 1

    jobs = queue_ls(queue_file=temp_queue)
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["exit_code"] == 1
    assert "Process exited with code 1" in jobs[0]["error"]


def test_queue_runner_stop_file(temp_queue):
    stop_file = temp_queue.with_name(temp_queue.name + ".stop")
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()

    code = queue_run(queue_file=temp_queue)
    assert code == 0
    # The stop file is consumed so it does not halt every later runner.
    assert not stop_file.exists()


def test_hardware_preflight_busy_check(monkeypatch):
    mock_hw = [
        {
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "uuid": "GPU-aaa",
            "memory_total": 48000.0,
            "memory_free": 1024.0,  # Below 2048 threshold
            "memory_used": 46976.0,
            "utilization": 95.0,
        }
    ]
    monkeypatch.setattr("hypergan.queue.inspect_gpu_hardware", lambda: mock_hw)
    monkeypatch.setattr("hypergan.queue.inspect_gpu_processes", lambda: [])

    avail, reason = check_device_availability("0")
    assert avail is False
    assert "free memory (1024 MiB) < threshold" in reason

    # Check active compute process conflict
    mock_hw_ok = [
        {
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "uuid": "GPU-aaa",
            "memory_total": 48000.0,
            "memory_free": 40000.0,
            "memory_used": 8000.0,
            "utilization": 50.0,
        }
    ]
    mock_procs = [
        {
            "gpu_uuid": "GPU-aaa",
            "pid": 1234,
            "process_name": "python",
            "used_memory": 8000.0,
            "cmdline": "python -m hypergan train recipe.toml",
        }
    ]
    monkeypatch.setattr("hypergan.queue.inspect_gpu_hardware", lambda: mock_hw_ok)
    monkeypatch.setattr("hypergan.queue.inspect_gpu_processes", lambda: mock_procs)

    avail, reason = check_device_availability("0")
    assert avail is False
    assert "busy running process 1234" in reason


def test_systemd_install_service_dry_run():
    unit_path, instructions = install_service(devices="gpu0,gpu1", dry_run=True)
    assert unit_path.name == "hypergan-queue-gpu0-gpu1.service"
    assert "systemctl --user enable --now hypergan-queue-gpu0-gpu1" in instructions
    assert "--devices gpu0,gpu1" in instructions or "hypergan queue run --devices gpu0,gpu1" in instructions


def test_format_queue_table():
    empty_str = format_queue_table([])
    assert "empty" in empty_str.lower()

    jobs = [
        {
            "id": "job-0001",
            "name": "test_exp",
            "devices": "gpu0",
            "status": "queued",
            "created_at": "2026-10-15T12:00:00Z",
            "pid": None,
            "exit_code": None,
        }
    ]
    table = format_queue_table(jobs)
    assert "job-0001" in table
    assert "test_exp" in table
    assert "queued" in table


def test_cli_queue_commands(temp_queue, monkeypatch, capsys):
    monkeypatch.setenv("HYPERGAN_QUEUE_FILE", str(temp_queue))

    # Add via CLI
    ret = cli.main(["queue", "add", "dummy.toml", "-d", "gpu0"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Enqueued job-0001: dummy" in out

    # List via CLI
    ret = cli.main(["queue", "ls"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "job-0001" in out
    assert "dummy" in out

    # List JSON via CLI
    ret = cli.main(["queue", "ls", "--json"])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["id"] == "job-0001"

    # Status via CLI
    ret = cli.main(["queue", "status"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Job ID:         job-0001" in out

    # Stop via CLI
    ret = cli.main(["queue", "stop", "job-0001"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Cancelled queued job job-0001" in out

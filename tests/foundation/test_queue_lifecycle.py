"""Queue job lifecycle: claiming, stopping, runner death and recovery (review 2026-10-25, slice 6)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import hypergan.queue as queue_module
from hypergan.queue import (
    _identity_alive,
    _normalize_devices,
    _process_identity,
    format_queue_status,
    load_queue,
    queue_add,
    queue_lock,
    queue_ls,
    queue_run,
    queue_status,
    queue_stop,
    save_queue,
)


@pytest.fixture
def temp_queue(tmp_path):
    q_dir = tmp_path / ".hypergan"
    q_dir.mkdir(parents=True, exist_ok=True)
    return q_dir / "queue.json"


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.toml"
    path.write_text('format = "hypergan-model/1"\n', encoding="utf-8")
    return path


class _RecordingPopen:
    """Popen stand-in that exits immediately and records each launch."""
    launches: list = []

    def __init__(self, cmd, **kwargs):
        type(self).launches.append(cmd)
        self.pid = 4_000_000 + len(type(self).launches)
        self.signals = []

    def wait(self):
        return 0

    def send_signal(self, sig):
        self.signals.append(sig)

    def terminate(self):
        self.signals.append("terminate")


# How the runner stops its own trainer: Windows has no SIGINT for a child process.
_RUNNER_STOP = "terminate" if os.name == "nt" else signal.SIGINT


@pytest.fixture
def recording_popen(monkeypatch):
    _RecordingPopen.launches = []
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    return _RecordingPopen


def _edit_job(queue_path, job_id, **fields):
    with queue_lock(queue_path):
        q = load_queue(queue_path)
        for j in q["jobs"]:
            if j["id"] == job_id:
                j.update(fields)
        save_queue(queue_path, q)


def _dead_identity():
    """Identity of a process that has exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = _process_identity(proc.pid)
    proc.wait()
    return identity


# Claiming

def test_job_cancelled_during_preflight_is_not_launched(temp_queue, model_file, recording_popen, monkeypatch):
    job = queue_add(model_file, devices="0", queue_file=temp_queue)

    def _preflight(devices, ws=None):
        queue_stop(job_id=job["id"], queue_file=temp_queue)
        return True, None

    monkeypatch.setattr(queue_module, "check_device_availability", _preflight)
    assert queue_run(once=True, poll_interval=0.01, queue_file=temp_queue) == 0

    assert recording_popen.launches == []
    final = queue_ls(queue_file=temp_queue)[0]
    assert final["status"] == "cancelled"
    assert final["error"] == "Cancelled by user"


def test_two_runners_cannot_launch_the_same_job(temp_queue, model_file, recording_popen, monkeypatch):
    queue_add(model_file, queue_file=temp_queue)
    calls = []

    def _preflight(devices, ws=None):
        calls.append(devices)
        if len(calls) == 1:
            # A second runner (e.g. the `-d 1` service) picks the same job while
            # the first runner is still in its device preflight.
            assert queue_run(devices="1", once=True, poll_interval=0.01, queue_file=temp_queue) == 0
        return True, None

    monkeypatch.setattr(queue_module, "check_device_availability", _preflight)
    assert queue_run(devices="0", once=True, poll_interval=0.01, queue_file=temp_queue) == 0

    assert len(calls) == 2
    assert len(recording_popen.launches) == 1
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "completed"


def test_claim_records_runner_and_trainer_identity(temp_queue, model_file, monkeypatch):
    queue_add(model_file, queue_file=temp_queue)
    seen = {}

    class _InspectingPopen(_RecordingPopen):
        def wait(self):
            seen.update(queue_ls(queue_file=temp_queue)[0])
            return 0

    monkeypatch.setattr(subprocess, "Popen", _InspectingPopen)
    assert queue_run(once=True, queue_file=temp_queue) == 0

    assert seen["status"] == "running"
    assert seen["runner"]["pid"] == os.getpid()
    assert _identity_alive(seen["runner"]) is True
    assert seen["process"]["pid"] == seen["pid"]


def test_job_cancelled_between_claim_and_launch_is_signalled_once(temp_queue, model_file, monkeypatch):
    job = queue_add(model_file, queue_file=temp_queue)
    instances = []

    class _CancelOnLaunch(_RecordingPopen):
        def __init__(self, cmd, **kwargs):
            super().__init__(cmd, **kwargs)
            instances.append(self)
            # No pid is recorded yet, so `queue stop` cannot signal the trainer itself.
            queue_stop(job_id=job["id"], queue_file=temp_queue)

    monkeypatch.setattr(subprocess, "Popen", _CancelOnLaunch)
    assert queue_run(once=True, queue_file=temp_queue) == 0

    assert [p.signals for p in instances] == [[_RUNNER_STOP]]
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "cancelled"


def test_launch_failure_marks_job_failed(temp_queue, model_file, monkeypatch):
    queue_add(model_file, queue_file=temp_queue)

    def _broken_popen(cmd, **kwargs):
        raise FileNotFoundError("python vanished")

    monkeypatch.setattr(subprocess, "Popen", _broken_popen)
    assert queue_run(once=True, queue_file=temp_queue) == 1
    final = queue_ls(queue_file=temp_queue)[0]
    assert final["status"] == "failed"
    assert "python vanished" in final["error"]


# Stopping

_SIGINT_COUNTER = textwrap.dedent("""
    import pathlib, signal, sys, time
    count_path, ready_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    hits = []
    def _on_int(signum, frame):
        hits.append(time.monotonic())
        count_path.write_text(str(len(hits)))
    signal.signal(signal.SIGINT, _on_int)
    ready_path.write_text("ready")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        # Emulate a trainer finishing its final checkpoint after the first SIGINT.
        if hits and time.monotonic() - hits[0] > 1.5:
            sys.exit(0)
        time.sleep(0.02)
    sys.exit(3)
""")


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot deliver SIGINT to another process; queue stop terminates the trainer there")
def test_stop_all_sends_a_single_sigint_to_a_running_trainer(temp_queue, model_file, tmp_path, monkeypatch):
    script = tmp_path / "sigint_counter.py"
    script.write_text(_SIGINT_COUNTER, encoding="utf-8")
    count_path = tmp_path / "sigint_count"
    ready_path = tmp_path / "ready"
    real_popen = subprocess.Popen

    def _trainer(cmd, **kwargs):
        return real_popen([sys.executable, str(script), str(count_path), str(ready_path)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _trainer)
    queue_add(model_file, queue_file=temp_queue)
    queue_add(model_file, queue_file=temp_queue)
    results = {}

    def _stop_when_ready():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if ready_path.exists() and queue_ls(queue_file=temp_queue)[0].get("pid"):
                results["stop"] = queue_stop(stop_all=True, queue_file=temp_queue)
                return
            time.sleep(0.02)

    helper = threading.Thread(target=_stop_when_ready)
    helper.start()
    ret = queue_run(poll_interval=0.05, queue_file=temp_queue)
    helper.join(timeout=30)

    assert ret == 0
    assert results["stop"]["signalled_count"] == 1
    # A second SIGINT would turn the trainer's graceful stop into os._exit.
    assert count_path.read_text() == "1"
    jobs = queue_ls(queue_file=temp_queue)
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["exit_code"] == 0
    assert jobs[1]["status"] == "cancelled"


def test_stop_all_without_a_runner_does_not_block_later_runners(temp_queue, model_file, recording_popen):
    queue_add(model_file, queue_file=temp_queue)
    queue_stop(stop_all=True, queue_file=temp_queue)
    assert not temp_queue.with_name(temp_queue.name + ".stop").exists()

    job = queue_add(model_file, queue_file=temp_queue)
    assert queue_run(once=True, poll_interval=0.01, queue_file=temp_queue) == 0

    assert len(recording_popen.launches) == 1
    assert queue_ls(queue_file=temp_queue)[1]["id"] == job["id"]
    assert queue_ls(queue_file=temp_queue)[1]["status"] == "completed"


def test_stop_all_halts_a_runner_started_before_it(temp_queue, recording_popen):
    helper = threading.Timer(0.2, queue_stop, kwargs={"stop_all": True, "queue_file": temp_queue})
    helper.start()
    started = time.monotonic()
    assert queue_run(poll_interval=0.02, queue_file=temp_queue) == 0
    helper.join(timeout=5)

    assert time.monotonic() - started < 3.0
    assert recording_popen.launches == []


def test_stop_file_is_consumed_and_later_runs_work(temp_queue, model_file, recording_popen):
    stop_file = temp_queue.with_name(temp_queue.name + ".stop")
    stop_file.touch()
    queue_add(model_file, queue_file=temp_queue)

    assert queue_run(once=True, queue_file=temp_queue) == 0
    assert recording_popen.launches == []
    assert not stop_file.exists()

    assert queue_run(once=True, queue_file=temp_queue) == 0
    assert len(recording_popen.launches) == 1
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "completed"


# Runner death and process identity

def test_jobs_of_a_dead_runner_are_marked_interrupted(temp_queue, model_file):
    crashed = queue_add(model_file, queue_file=temp_queue)
    rebooted = queue_add(model_file, queue_file=temp_queue)
    alive = queue_add(model_file, queue_file=temp_queue)
    outlived = queue_add(model_file, queue_file=temp_queue)
    dead = _dead_identity()
    me = _process_identity(os.getpid())
    # Linux start times count from boot, so a reboot shows as a new boot id; macOS and
    # Windows record absolute creation times, which a PID recycled after a reboot never matches.
    if me["boot_id"]:
        before_reboot = dict(me, boot_id="boot-before-reboot")
    else:
        before_reboot = dict(me, start_time=me["start_time"] - 1)
    _edit_job(temp_queue, crashed["id"], status="running", runner=dead, pid=dead["pid"], process=dead)
    _edit_job(temp_queue, rebooted["id"], status="running", runner=before_reboot, pid=os.getpid(), process=before_reboot)
    _edit_job(temp_queue, alive["id"], status="running", runner=me, pid=None)
    # The runner died but its trainer still runs: the job is still really running.
    _edit_job(temp_queue, outlived["id"], status="running", runner=dead, pid=os.getpid(), process=me)

    jobs = {j["id"]: j for j in queue_ls(queue_file=temp_queue)}

    for job_id in (crashed["id"], rebooted["id"]):
        assert jobs[job_id]["status"] == "interrupted"
        assert jobs[job_id]["pid"] is None
        assert jobs[job_id]["ended_at"] is not None
        assert "exited while the job was running" in jobs[job_id]["error"]
    assert jobs[alive["id"]]["status"] == "running"
    assert jobs[outlived["id"]]["status"] == "running"
    # Persisted, so `queue status` no longer reports the dead job as active.
    assert queue_status(queue_file=temp_queue)["active_job"]["id"] == alive["id"]


def test_jobs_claimed_on_another_host_are_left_alone(temp_queue, model_file):
    job = queue_add(model_file, queue_file=temp_queue)
    remote = dict(_dead_identity(), host="some-other-host")
    _edit_job(temp_queue, job["id"], status="running", runner=remote, pid=remote["pid"], process=remote)
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "running"


def test_stop_never_signals_a_reused_pid(temp_queue, model_file, monkeypatch):
    job = queue_add(model_file, queue_file=temp_queue)
    me = _process_identity(os.getpid())
    # PID 1 now, but recorded with a different start time: a recycled PID, not our trainer.
    recycled = dict(me, pid=1, start_time=(queue_module._process_start_time(1) or 0) + 1)
    _edit_job(temp_queue, job["id"], status="running", runner=me, pid=1, process=recycled)
    kills = []
    monkeypatch.setattr(queue_module.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    res = queue_stop(job_id=job["id"], queue_file=temp_queue)

    # Signal 0 is only the liveness probe used where no start time is available.
    assert [k for k in kills if k[1] != 0] == []
    assert res["signalled"] is False
    assert queue_ls(queue_file=temp_queue)[0]["status"] == "cancelled"


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs a PID owned by another user")
@pytest.mark.parametrize("stop_all", [False, True])
def test_stop_survives_a_pid_owned_by_another_user(temp_queue, model_file, stop_all):
    job = queue_add(model_file, queue_file=temp_queue)
    me = _process_identity(os.getpid())
    # Legacy record without a trainer identity, pointing at a root-owned PID.
    _edit_job(temp_queue, job["id"], status="running", runner=me, pid=1)

    res = queue_stop(job_id=None if stop_all else job["id"], stop_all=stop_all, queue_file=temp_queue)

    assert res.get("signalled", False) is False
    assert res.get("signalled_count", 0) == 0


@pytest.mark.skipif(not (sys.platform.startswith("linux") or sys.platform == "darwin" or os.name == "nt"),
                    reason="process start times are read on Linux, macOS and Windows only")
def test_process_identity_is_checkable_on_every_supported_os():
    me = _process_identity(os.getpid())
    assert me["start_time"] is not None
    assert _identity_alive(me) is True
    dead = _dead_identity()
    assert dead["start_time"] is not None
    assert _identity_alive(dead) is False


_SLEEPING_TRAINER = textwrap.dedent("""
    import pathlib, signal, sys, time
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    pathlib.Path(sys.argv[1]).write_text("ready")
    time.sleep(60)
    sys.exit(3)
""")


@pytest.mark.parametrize("stop_all", [False, True])
def test_stop_ends_a_real_trainer(temp_queue, model_file, tmp_path, stop_all):
    ready = tmp_path / "ready"
    trainer = subprocess.Popen([sys.executable, "-c", _SLEEPING_TRAINER, str(ready)])
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        job = queue_add(model_file, queue_file=temp_queue)
        _edit_job(temp_queue, job["id"], status="running", runner=_process_identity(os.getpid()),
                  pid=trainer.pid, process=_process_identity(trainer.pid))

        res = queue_stop(job_id=None if stop_all else job["id"], stop_all=stop_all, queue_file=temp_queue)

        assert (res["signalled_count"] == 1) if stop_all else (res["signalled"] is True)
        # SIGINT on POSIX, TerminateProcess on Windows: either way the trainer exits early.
        assert trainer.wait(timeout=15) != 3
        assert queue_ls(queue_file=temp_queue)[0]["stop_signalled"] is True
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait()


@pytest.mark.parametrize("stop_all", [False, True])
def test_stop_leaves_an_unreachable_trainer_to_its_runner(temp_queue, model_file, stop_all):
    job = queue_add(model_file, queue_file=temp_queue)
    remote = dict(_dead_identity(), host="some-other-host")
    _edit_job(temp_queue, job["id"], status="running", runner=remote, pid=remote["pid"], process=remote)

    res = queue_stop(job_id=None if stop_all else job["id"], stop_all=stop_all, queue_file=temp_queue)

    if stop_all:
        assert res["signalled_count"] == 0
    else:
        assert res["signalled"] is False
        assert "another host" in res["message"]
    # Unflagged, so that host's runner still sends its trainer the one SIGINT.
    assert not queue_ls(queue_file=temp_queue)[0].get("stop_signalled")


def test_signal_errors_never_escape(monkeypatch):
    class _NoSignals:
        def send_signal(self, sig):
            raise ValueError("Unsupported signal")

        def terminate(self):
            raise ValueError("Unsupported signal")

    queue_module._interrupt_child(_NoSignals())

    def _kill(pid, sig):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(queue_module.os, "kill", _kill)
    # A legacy record without an identity is signalled by PID; any OS error means "not signalled".
    assert queue_module._signal_job_process({"pid": 12345}, signal.SIGINT) is False


# Wait jobs

def test_wait_job_does_not_hang_on_a_killed_run(temp_queue, tmp_path):
    run_dir = tmp_path / "killed-run"
    run_dir.mkdir()
    # A SIGKILLed trainer leaves status "running" and an unlocked .run.lock behind.
    (run_dir / "manifest.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (run_dir / ".run.lock").touch()
    queue_add(run_dir, wait=True, queue_file=temp_queue)
    stop_file = temp_queue.with_name(temp_queue.name + ".stop")
    watchdog = threading.Timer(5.0, stop_file.touch)
    watchdog.start()
    try:
        started = time.monotonic()
        assert queue_run(once=True, poll_interval=0.01, queue_file=temp_queue) == 0
        elapsed = time.monotonic() - started
    finally:
        watchdog.cancel()

    assert queue_ls(queue_file=temp_queue)[0]["status"] == "completed"
    assert elapsed < 4.0


# Corrupt queue file

def test_corrupt_queue_file_is_moved_aside_and_numbering_continues(temp_queue, model_file, capsys):
    logs = temp_queue.parent / "logs"
    logs.mkdir()
    (logs / "job-0007.log").write_text("old job output\n", encoding="utf-8")
    temp_queue.write_text('{"schema_version": 1, "jobs": [', encoding="utf-8")

    job = queue_add(model_file, queue_file=temp_queue)

    assert job["id"] == "job-0008"
    assert (logs / "job-0007.log").read_text(encoding="utf-8") == "old job output\n"
    aside = list(temp_queue.parent.glob("queue.json.corrupt-*"))
    assert len(aside) == 1
    assert aside[0].read_text(encoding="utf-8") == '{"schema_version": 1, "jobs": ['
    assert "unreadable" in capsys.readouterr().err


def test_queue_file_from_another_schema_is_not_replaced(temp_queue, model_file):
    original = json.dumps({"schema_version": 2, "jobs": [], "next_job_id": 40})
    temp_queue.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        queue_add(model_file, queue_file=temp_queue)
    assert temp_queue.read_text(encoding="utf-8") == original


def test_unlocked_reader_never_moves_the_queue_file(temp_queue):
    # `hypergan ps` reads the queue without the lock; it must not race a locked writer
    # that has already moved the corrupt file aside and saved a fresh queue.
    temp_queue.write_text("{", encoding="utf-8")
    assert load_queue(temp_queue)["jobs"] == []
    assert temp_queue.read_text(encoding="utf-8") == "{"
    assert list(temp_queue.parent.glob("queue.json.corrupt-*")) == []

    with queue_lock(temp_queue):
        load_queue(temp_queue)
    assert len(list(temp_queue.parent.glob("queue.json.corrupt-*"))) == 1


# Devices and status display

def test_device_spellings_are_normalised():
    assert _normalize_devices("0") == _normalize_devices("cuda:0") == _normalize_devices("gpu0")
    assert _normalize_devices("0,1") == _normalize_devices("cuda:1, cuda:0")
    assert _normalize_devices("0") != _normalize_devices("1")
    assert _normalize_devices("gpu-a", {"gpus": {"gpu-a": "cuda:2"}}) == _normalize_devices("2")
    assert _normalize_devices(None) is None
    assert _normalize_devices("CPU") == _normalize_devices("cpu")


def test_runner_matches_jobs_across_device_spellings(temp_queue, model_file, recording_popen, monkeypatch):
    monkeypatch.setattr(queue_module, "check_device_availability", lambda devices, ws=None: (True, None))
    a = queue_add(model_file, devices="cuda:0", queue_file=temp_queue)
    b = queue_add(model_file, devices="1", queue_file=temp_queue)
    c = queue_add(model_file, devices="gpu0", queue_file=temp_queue)

    assert queue_run(devices="0", once=True, queue_file=temp_queue) == 0

    statuses = {j["id"]: j["status"] for j in queue_ls(queue_file=temp_queue)}
    assert statuses == {a["id"]: "completed", b["id"]: "queued", c["id"]: "completed"}


def test_queue_status_shows_speed_and_metrics_from_a_real_summary(temp_queue, tmp_path):
    from hypergan.summary import atomic_write_json

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # The keys summary.write_summary_json produces.
    atomic_write_json(run_dir / "summary.json", {
        "run_id": "r1", "step": 450, "total_steps": 1000, "steps_per_sec": 12.5,
        "eta_seconds": 44.0, "latest": {"fid50k_train": 18.25},
    })
    job = queue_add(tmp_path / "model.toml", run_dir=run_dir, queue_file=temp_queue)

    formatted = format_queue_status(queue_status(job_id=job["id"], queue_file=temp_queue))

    assert "Step 450/1000 (12.50 it/s), ETA 0.7m" in formatted
    assert "fid50k_train=18.2500" in formatted

"""Unit tests for multi-GPU `hypergan ps`, `list_runs`, and `run_status` telemetry."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hypergan as hg
from hypergan.inspection import (
    _build_ppid_map,
    _is_pid_or_descendant,
    _parse_proc_stat_ppid,
    format_ps_table,
    list_runs,
    ps_runs,
    run_status,
)
from hypergan.registry import register_run
from hypergan.summary import write_summary_json


def _write_run(
    run_dir: Path,
    *,
    run_id: str,
    name: str,
    status: str = "completed",
    device: str = "cuda",
    runtime: dict | None = None,
    execution: dict | None = None,
    pid: int | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema_version": 1,
        "run_id": run_id,
        "name": name,
        "status": status,
        "steps": 1200,
        "total_steps": 10000,
        "steps_per_second": 4.5,
        "device": device,
        "config": {
            "name": name,
            "train": {"device": device, "steps": 10000},
            "training": {"device": device, "steps": 10000},
        },
    }
    if pid is not None:
        manifest["pid"] = pid
    if runtime is not None:
        manifest["runtime"] = runtime
    if execution is not None:
        manifest["execution"] = execution

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_json(run_dir, status=status)
    return run_dir


def test_ps_runs_aggregates_descendant_worker_vram_and_sorts_gpu_devices(tmp_path, monkeypatch):
    """Spawned `replicated_worker` child processes aggregate VRAM and ordered GPU devices onto parent PID."""
    run_dir = _write_run(
        tmp_path / "runs" / "transgan-2gpu",
        run_id="11223344556677889900aabbccddeeff",
        name="cifar10-transgan",
        status="running",
        device="cuda",
        pid=4000,
        execution={"name": "cuda-replicated-nccl", "world_size": 2, "nodes": 1, "ranks_per_node": 2},
        runtime={
            "device": "cuda",
            "world_size": 2,
            "backend": "nccl",
            "cuda": {
                "rank_devices": [
                    {"rank": 0, "device": "cuda:0", "uuid": "GPU-0000", "name": "NVIDIA RTX A6000"},
                    {"rank": 1, "device": "cuda:1", "uuid": "GPU-1111", "name": "NVIDIA RTX A6000"},
                ]
            },
        },
    )
    register_run(run_dir, status="running", pid=4000)

    # Simulate active flock held by parent launcher PID 4000
    monkeypatch.setattr(
        "hypergan.inspection.check_run_active",
        lambda p: (True, 4000) if Path(p).resolve() == run_dir.resolve() else (False, None),
    )

    # Build a synthetic /proc tree: 4000 (launcher) -> 4005 (spawn broker) -> {4010, 4011} (CUDA workers)
    fake_proc = tmp_path / "fake_proc"
    for proc_pid, comm, ppid in (
        (4000, "hypergan", 1000),
        (4005, "python3 (spawn broker)", 4000),
        (4010, "pt_main_thread", 4005),
        (4011, "pt_main_thread (rank 1)", 4005),
        (5000, "other_job", 1),
    ):
        pdir = fake_proc / str(proc_pid)
        pdir.mkdir(parents=True)
        (pdir / "stat").write_text(f"{proc_pid} ({comm}) S {ppid} {proc_pid} 0 0\n", encoding="utf-8")

    ppid_map = _build_ppid_map(fake_proc)
    monkeypatch.setattr("hypergan.inspection._build_ppid_map", lambda proc_root=None: ppid_map)

    # Return GPU 1 process BEFORE GPU 0 process, with worker cmdlines lacking `run_dir`
    monkeypatch.setattr(
        "hypergan.gpu_inventory.inspect_gpu_hardware",
        lambda: [
            {"index": 0, "uuid": "GPU-0000", "name": "NVIDIA RTX A6000"},
            {"index": 1, "uuid": "GPU-1111", "name": "NVIDIA RTX A6000"},
        ],
    )
    monkeypatch.setattr(
        "hypergan.gpu_inventory.inspect_gpu_processes",
        lambda: [
            {
                "gpu_uuid": "GPU-1111",
                "pid": 4011,
                "process_name": "python3.14",
                "used_memory": 25600.0,
                "cmdline": "python3.14 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=11)",
            },
            {
                "gpu_uuid": "GPU-0000",
                "pid": 4010,
                "process_name": "python3.14",
                "used_memory": 24576.0,
                "cmdline": "python3.14 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=9)",
            },
            {
                "gpu_uuid": "GPU-0000",
                "pid": 5000,
                "process_name": "python3.14",
                "used_memory": 8192.0,
                "cmdline": "python3.14 unrelated_train.py",
            },
        ],
    )

    ps_data = ps_runs(cwd=tmp_path)
    assert ps_data["active_count"] == 1
    assert len(ps_data["runs"]) == 1
    row = ps_data["runs"][0]
    assert row["pid"] == 4000
    assert row["device"] == "cuda:0,cuda:1"
    assert row["vram_mb"] == pytest.approx(50176.0)

    table = format_ps_table(ps_data)
    assert "cuda:0,cuda:1" in table
    assert "49.0 GB" in table

    snap = hg.ps(cwd=tmp_path)
    assert len(snap.runs) == 1
    assert snap.runs[0].device == "cuda:0,cuda:1"
    assert snap.runs[0]["vram_mb"] == pytest.approx(50176.0)


def test_ppid_parsing_posix_fallback_and_cycle_guard(tmp_path, monkeypatch):
    """Verify /proc/<pid>/stat comm edge cases, POSIX ps fallback, cycle guard, and max_depth."""
    assert _parse_proc_stat_ppid("12345 (comm (with) parens & spaces) R 9999 12345 0\n") == 9999
    assert _parse_proc_stat_ppid("malformed stat without parens") is None

    # POSIX `ps -eo pid=,ppid=` fallback when /proc directory is absent
    missing_proc = tmp_path / "no_such_proc"
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="  100   1\n  200 100\n  201 200\n"),
    )
    fallback_map = _build_ppid_map(missing_proc)
    assert fallback_map == {100: 1, 200: 100, 201: 200}
    assert _is_pid_or_descendant(201, 100, fallback_map) is True
    assert _is_pid_or_descendant(201, 999, fallback_map) is False
    # Container PID 1 launcher matches children whose PPID is 1
    assert _is_pid_or_descendant(100, 1, fallback_map) is True

    # Cycle guard terminates cleanly without infinite loop
    cyclic_map = {10: 20, 20: 30, 30: 10}
    assert _is_pid_or_descendant(10, 999, cyclic_map) is False

    # Max-depth cap terminates deep chains cleanly
    deep_map = {i: i - 1 for i in range(100, 50, -1)}
    assert _is_pid_or_descendant(100, 90, deep_map, max_depth=16) is True
    assert _is_pid_or_descendant(100, 70, deep_map, max_depth=16) is False


def test_static_multigpu_device_display_preserves_single_gpu_and_cpu(tmp_path):
    """Multi-GPU manifests upgrade bare 'cuda' to 'cuda:0,cuda:1' or '<N>x cuda' without altering 1-GPU/CPU runs."""
    runs_root = tmp_path / "runs"

    # 1. Completed 2-GPU run with rank_devices (also exercises Tier 1 registry cache)
    r_2gpu = _write_run(
        runs_root / "done-2gpu",
        run_id="aaaa1111bbbb2222cccc3333dddd4444",
        name="done-2gpu",
        status="completed",
        device="cuda",
        runtime={
            "device": "cuda",
            "world_size": 2,
            "cuda": {
                "rank_devices": [
                    {"rank": 1, "device": "cuda:1", "name": "NVIDIA RTX 4090"},
                    {"rank": 0, "device": "cuda:0", "name": "NVIDIA RTX 4090"},
                ]
            },
        },
    )
    register_run(r_2gpu, status="completed")

    # 2. Multi-GPU run with world_size > 1 and no rank_devices
    r_4gpu = _write_run(
        runs_root / "done-4gpu",
        run_id="bbbb1111cccc2222dddd3333eeee4444",
        name="done-4gpu",
        status="completed",
        device="cuda",
        execution={"world_size": 4},
    )
    register_run(r_4gpu, status="completed")

    # 3. Single-GPU bare 'cuda' run stays 'cuda'
    r_1gpu = _write_run(
        runs_root / "done-1gpu",
        run_id="cccc1111dddd2222eeee3333ffff4444",
        name="done-1gpu",
        status="completed",
        device="cuda",
        runtime={"device": "cuda:0", "world_size": 1, "cuda": {"name": "NVIDIA RTX 4090"}},
    )
    register_run(r_1gpu, status="completed")

    # 4. CPU 2-rank Gloo run stays 'cpu'
    r_cpu = _write_run(
        runs_root / "done-cpu-gloo",
        run_id="dddd1111eeee2222ffff333300004444",
        name="done-cpu-gloo",
        status="completed",
        device="cpu",
        execution={"world_size": 2},
        runtime={"device": "cpu", "world_size": 2, "backend": "gloo"},
    )
    register_run(r_cpu, status="completed")

    by_name = {r["name"]: r for r in list_runs(root=runs_root)}
    assert by_name["done-2gpu"]["device"] == "cuda:0,cuda:1"
    assert by_name["done-4gpu"]["device"] == "4x cuda"
    assert by_name["done-1gpu"]["device"] == "cuda"
    assert by_name["done-cpu-gloo"]["device"] == "cpu"

    ps_all = {r["name"]: r for r in ps_runs(all_runs=True, cwd=tmp_path)["runs"]}
    assert ps_all["done-2gpu"]["device"] == "cuda:0,cuda:1"
    assert ps_all["done-4gpu"]["device"] == "4x cuda"
    assert ps_all["done-1gpu"]["device"] == "cuda"
    assert ps_all["done-cpu-gloo"]["device"] == "cpu"

    assert run_status(r_2gpu, cwd=tmp_path)["devices"] == "cuda:0,cuda:1"
    assert run_status(r_4gpu, cwd=tmp_path)["devices"] == "4x cuda"
    assert run_status(r_1gpu, cwd=tmp_path)["devices"] == "cuda"
    assert run_status(r_cpu, cwd=tmp_path)["devices"] == "cpu"

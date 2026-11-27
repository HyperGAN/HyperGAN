"""Regression tests for Web Observatory security, CPU worker validation, CI/packaging, ROCm preflight, and docs site fixes."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Web Observatory security: stop_all_servers & stop_projector
# ---------------------------------------------------------------------------

def _make_proc_entry(proc_root: Path, pid: int, cmdline: list[str], ppid: int = 1) -> Path:
    entry = proc_root / str(pid)
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "cmdline").write_bytes(b"\0".join(tok.encode("utf-8") for tok in cmdline) + b"\0")
    (entry / "stat").write_text(f"{pid} (python) S {ppid} 0 0\n", encoding="utf-8")
    return entry


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership/symlink/permission checks")
def test_stop_all_servers_rejects_other_users_symlinks_traversal_and_negative_pids(tmp_path, monkeypatch):
    import hypergan.web_autostart as wa
    from hypergan.run_state import atomic_json, run_lock

    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(wa.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("HYPERGAN_PROC_ROOT", str(proc))
    monkeypatch.setattr(wa.time, "sleep", lambda _: None)

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(wa.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    # A. Another user's viewer directory must NEVER be touched or globbed.
    other_dir = tmp_path / "hypergan-viewers-999999" / "other-reg"
    (other_dir / "service").mkdir(parents=True)
    atomic_json(other_dir / "state.json", {
        "launch_id": "other1",
        "status": "ready",
        "supervisor_pid": 50001,
    })
    _make_proc_entry(proc, 50001, ["python3", "-m", "hypergan.web_autostart"])

    # B. Current user's own base dir (0o700) with adversarial registry entries:
    base = wa._viewer_base_dir()
    base.mkdir(mode=0o700)
    os.chmod(base, 0o700)

    # B1. Symlinked registry dir pointing to an outside directory must NOT be touched/unlinked.
    outside = tmp_path / "outside-victim"
    outside.mkdir(mode=0o700)
    outside_state = outside / "state.json"
    atomic_json(outside_state, {"launch_id": "outside", "status": "ready", "supervisor_pid": 50002})
    (base / "symlink-reg").symlink_to(outside)

    # B2. Non-private (0o755) registry dir must be ignored.
    loose_reg = base / "loose-reg"
    loose_reg.mkdir(mode=0o755)
    os.chmod(loose_reg, 0o755)
    (loose_reg / "service").mkdir(mode=0o700)
    atomic_json(loose_reg / "state.json", {"launch_id": "loose", "status": "ready", "supervisor_pid": 50003})

    # B3. Private owned registry dir with path-traversal launch_id and PID -1 / 0 / non-viewer PID!
    evil_reg = base / "evil-reg"
    evil_reg.mkdir(mode=0o700)
    os.chmod(evil_reg, 0o700)
    (evil_reg / "service").mkdir(mode=0o700)
    atomic_json(evil_reg / "state.json", {
        "launch_id": "../../pwned-escape",
        "status": "ready",
        "supervisor_pid": -1,
        "processes": {"server": 0, "projector": -42},
    })

    # B4. Private owned locked registry whose supervisor_pid is a NON-viewer training process!
    nonviewer_reg = base / "nonviewer-reg"
    nonviewer_reg.mkdir(mode=0o700)
    os.chmod(nonviewer_reg, 0o700)
    (nonviewer_reg / "service").mkdir(mode=0o700)
    _make_proc_entry(proc, 50004, ["python3", "-m", "hypergan", "train", "config.toml"])
    atomic_json(nonviewer_reg / "state.json", {
        "launch_id": "validhex1234",
        "status": "ready",
        "supervisor_pid": 50004,
        "processes": {"server": 50004},
    })

    # B5. Private owned locked registry with a genuine verified viewer supervisor + verified child + spoofed non-child!
    good_reg = base / "good-reg"
    good_reg.mkdir(mode=0o700)
    os.chmod(good_reg, 0o700)
    (good_reg / "service").mkdir(mode=0o700)
    _make_proc_entry(proc, 50010, ["python3", "-m", "hypergan.web_autostart", "/runs", "/priv", "goodid"])
    _make_proc_entry(proc, 50011, ["python3", "-c", "from multiprocessing.spawn import spawn_main"], ppid=50010)
    _make_proc_entry(proc, 50012, ["python3", "-c", "print('unrelated')"], ppid=1)
    atomic_json(good_reg / "state.json", {
        "launch_id": "goodid1234",
        "status": "ready",
        "supervisor_pid": 50010,
        "processes": {"server": 50011, "spoofed": 50012, "neg": -1},
    })

    with run_lock(evil_reg / "service"), run_lock(nonviewer_reg / "service"), run_lock(good_reg / "service"):
        killed = wa.stop_all_servers()

    # Negative/zero PIDs, non-viewer training PIDs, and spoofed non-child PIDs must NEVER be signalled.
    assert killed == [50001, 50010, 50011]  # 50001 discovered only via /proc scan of current UID viewer command
    assert all(pid > 0 for pid, _ in signalled)
    assert all(pid not in (50004, 50012) for pid, _ in signalled)
    # Path traversal stop file must NOT have been created outside evil_reg.
    assert not (tmp_path / "pwned-escape").exists()
    assert not (base / "pwned-escape").exists()
    # Other user's state.json and symlinked victim state.json must be completely untouched.
    assert (other_dir / "state.json").is_file()
    assert not (other_dir / "stop-other1").exists()
    assert outside_state.is_file()
    assert (loose_reg / "state.json").is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink checks")
def test_stop_all_servers_skips_symlinked_base_dir_and_rechecks_pid_before_sigkill(tmp_path, monkeypatch):
    import hypergan.web_autostart as wa
    from hypergan.run_state import atomic_json, run_lock

    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(wa.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("HYPERGAN_PROC_ROOT", str(proc))
    monkeypatch.setattr(wa.time, "sleep", lambda _: None)

    # 1. Symlinked base dir must be ignored completely.
    real_target = tmp_path / "real-target"
    real_target.mkdir(mode=0o700)
    (real_target / "reg").mkdir(mode=0o700)
    atomic_json(real_target / "reg" / "state.json", {"launch_id": "x", "supervisor_pid": 60001})
    wa._viewer_base_dir().symlink_to(real_target)
    assert wa.stop_all_servers() == []
    assert (real_target / "reg" / "state.json").is_file()
    wa._viewer_base_dir().unlink()

    # 2. PID reuse between SIGTERM and SIGKILL must NOT receive SIGKILL.
    base = wa._viewer_base_dir()
    base.mkdir(mode=0o700)
    os.chmod(base, 0o700)
    reg = base / "reused"
    reg.mkdir(mode=0o700)
    os.chmod(reg, 0o700)
    (reg / "service").mkdir(mode=0o700)
    _make_proc_entry(proc, 60010, ["python3", "-m", "hypergan.web_autostart"])
    atomic_json(reg / "state.json", {"launch_id": "reused1", "status": "ready", "supervisor_pid": 60010})

    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            # Simulate immediate PID reuse by an unrelated process before SIGKILL!
            _make_proc_entry(proc, 60010, ["python3", "-m", "hypergan", "train", "important.toml"])

    monkeypatch.setattr(wa.os, "kill", fake_kill)
    with run_lock(reg / "service"):
        wa.stop_all_servers()
    assert signals == [(60010, signal.SIGTERM)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX /proc ownership checks")
def test_stop_projector_requires_lock_positive_pid_and_proc_verification(tmp_path, monkeypatch):
    import hypergan.web_autostart as wa
    from hypergan.run_state import atomic_json, run_lock

    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(wa.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("HYPERGAN_PROC_ROOT", str(proc))

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(wa.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "manifest.json", {"run_id": "r1"})
    private = wa._registry(run_dir.parent)

    _make_proc_entry(proc, 70001, ["python3", "-m", "hypergan.web_autostart"])
    _make_proc_entry(proc, 70002, ["python3", "-c", "from multiprocessing.spawn import spawn_main"], ppid=70001)
    _make_proc_entry(proc, 70003, ["python3", "-m", "hypergan", "train", "x.toml"], ppid=1)

    # 1. Unlocked registry -> MUST NOT signal even valid projector PID.
    atomic_json(private / "state.json", {
        "launch_id": "p1", "status": "ready", "supervisor_pid": 70001, "processes": {"projector": 70002},
    })
    wa.stop_projector(run_dir)
    assert signalled == []

    # 2. Locked registry with PID -1 -> MUST NOT signal.
    atomic_json(private / "state.json", {
        "launch_id": "p1", "status": "ready", "supervisor_pid": 70001, "processes": {"projector": -1},
    })
    with run_lock(private / "service"):
        wa.stop_projector(run_dir)
    assert signalled == []

    # 3. Locked registry with unrelated non-child non-viewer PID -> MUST NOT signal.
    atomic_json(private / "state.json", {
        "launch_id": "p1", "status": "ready", "supervisor_pid": 70001, "processes": {"projector": 70003},
    })
    with run_lock(private / "service"):
        wa.stop_projector(run_dir)
    assert signalled == []

    # 4. Locked registry with verified child of verified viewer supervisor -> signals SIGTERM!
    atomic_json(private / "state.json", {
        "launch_id": "p1", "status": "ready", "supervisor_pid": 70001, "processes": {"projector": 70002},
    })
    with run_lock(private / "service"):
        wa.stop_projector(run_dir)
    assert signalled == [(70002, signal.SIGTERM)]


# ---------------------------------------------------------------------------
# 2. Dashboard reuse checks (host, public_origin, session_file) & bound host probing
# ---------------------------------------------------------------------------

def test_ensure_dashboard_running_checks_host_public_origin_and_probes_bound_host(tmp_path, monkeypatch):
    from hypergan import __version__
    import hypergan.web_autostart as wa

    root = tmp_path / "runs"
    root.mkdir()
    probed_hosts: list[str] = []

    monkeypatch.setattr(wa, "_registered_dashboard", lambda r: (
        {"launch_id": "L1", "server_instance_id": "inst1", "host": "100.64.0.7",
         "port": 8765, "public_origin": "https://box.ts.net"},
        None,
    ))
    monkeypatch.setattr(wa, "_dashboard_mismatch", lambda p: [])
    monkeypatch.setattr(wa, "launch_dashboard_daemon", lambda *a, **k: pytest.fail("should not launch"))

    def fake_probe(host, port, auth_token=None, **kwargs):
        probed_hosts.append(host)
        return {
            "version": __version__,
            "runs_root": str(root.resolve()),
            "server_instance_id": "inst1",
            "auth_mode": "none",
            "bind_host": "100.64.0.7",
            "public_origin": "https://box.ts.net",
        }

    monkeypatch.setattr(wa, "probe_dashboard", fake_probe)

    # Matching host + public_origin probes the actual bound host (not hardcoded 127.0.0.1) and reuses.
    url = wa.ensure_dashboard_running(root, port=8765, host="100.64.0.7", public_origin="https://box.ts.net")
    assert url == "http://100.64.0.7:8765"
    assert probed_hosts[-1] == "100.64.0.7"

    # Mismatched host refuses reuse.
    with pytest.raises(OSError, match="on host 100.64.0.7"):
        wa.ensure_dashboard_running(root, port=8765, host="0.0.0.0", public_origin="https://box.ts.net")
    assert probed_hosts[-1] == "127.0.0.1"  # 0.0.0.0 mapped to 127.0.0.1 for local probe

    # Mismatched public_origin refuses reuse.
    with pytest.raises(OSError, match="--public-origin"):
        wa.ensure_dashboard_running(root, port=8765, host="100.64.0.7", public_origin="https://other.ts.net")


def test_running_dashboard_checks_host_public_origin_and_session_file(tmp_path, monkeypatch):
    import hypergan.web_autostart as wa
    from hypergan.web_launch import running_dashboard

    root = tmp_path / "runs"
    root.mkdir()
    existing_sf = tmp_path / "sess-a.json"
    other_sf = tmp_path / "sess-b.json"

    probed_hosts: list[str] = []
    monkeypatch.setattr(wa, "_registered_dashboard", lambda r: (
        {"host": "192.168.1.20", "public_origin": "https://ml.example", "session_file": str(existing_sf)},
        "tok",
    ))

    def fake_probe(host, port, auth_token=None, **kwargs):
        probed_hosts.append(host)
        return {
            "runs_root": str(root.resolve()),
            "auth_mode": "token",
            "bind_host": "192.168.1.20",
            "public_origin": "https://ml.example",
        }

    monkeypatch.setattr(wa, "probe_dashboard", fake_probe)

    assert running_dashboard(
        root, 8765, "token", host="192.168.1.20", public_origin="https://ml.example", session_file=existing_sf
    ) == "http://192.168.1.20:8765"
    assert probed_hosts[-1] == "192.168.1.20"

    with pytest.raises(OSError, match="on host 192.168.1.20"):
        running_dashboard(root, 8765, "token", host="127.0.0.1", public_origin="https://ml.example")

    with pytest.raises(OSError, match="--public-origin"):
        running_dashboard(root, 8765, "token", host="192.168.1.20", public_origin="https://other.example")

    with pytest.raises(OSError, match="different session file"):
        running_dashboard(root, 8765, "token", host="192.168.1.20", public_origin="https://ml.example",
                          session_file=other_sf)


# ---------------------------------------------------------------------------
# 3. Unauthenticated wildcard / non-loopback control mutation refusal
# ---------------------------------------------------------------------------

def test_non_loopback_auth_none_refuses_control_mutations(tmp_path):
    from starlette.testclient import TestClient
    from hypergan.metrics import digest
    from hypergan.run_state import atomic_json
    from hypergan.web_server import create_app
    from hypergan.web_session import LocalSession, is_non_loopback_host

    assert is_non_loopback_host("0.0.0.0") is True
    assert is_non_loopback_host("::") is True
    assert is_non_loopback_host("192.168.1.20") is True
    assert is_non_loopback_host("127.0.0.1") is False
    assert is_non_loopback_host("localhost") is False
    assert is_non_loopback_host("::1") is False

    definition = {"kind": "scalar", "source": "g_loss", "label": "G"}
    definition["definition_hash"] = digest(definition)
    catalog = {"schema_version": 1, "metrics": {"loss/g_total": definition}}
    rev = digest(catalog)
    atomic_json(tmp_path / "metrics" / f"catalog-{rev}.json", catalog)
    atomic_json(tmp_path / "manifest.json", {
        "schema_version": 1, "run_id": "run", "attempt_id": "a",
        "steps": 1, "status": "running", "hostname": "remote.internal", "metrics_catalog": rev,
    })

    for bind_host in ("0.0.0.0", "::", "192.168.1.20"):
        session = LocalSession(8123, host=bind_host, auth="none")
        assert session.is_loopback_bind is False
        assert session.controls_allowed is False
        with TestClient(create_app(tmp_path, session, poll_seconds=0.01), base_url=session.origin) as client:
            caps = client.get("/api/v1/capabilities")
            assert caps.status_code == 200
            assert caps.json()["controls"] == []
            assert caps.json()["bind_host"] == bind_host
            for prefix in ("/api/v1/runs/run", "/api/runs/run"):
                for action in ("stop", "checkpoint", "sample"):
                    res = client.post(f"{prefix}/{action}", json={"request_id": f"unauth-{action}"})
                    assert res.status_code == 403, (bind_host, prefix, action, res.text)
                    assert "--auth token" in res.json()["error"]

    # With auth='token' on wildcard bind, authenticated control mutations succeed.
    authed_session = LocalSession(8123, host="0.0.0.0", auth="token")
    assert authed_session.controls_allowed is True
    with TestClient(create_app(tmp_path, authed_session, poll_seconds=0.01), base_url=authed_session.origin) as client:
        client.headers["Authorization"] = "Bearer " + authed_session._token
        assert client.get("/api/v1/capabilities").json()["controls"] == ["checkpoint", "stop", "sample"]
        res = client.post("/api/v1/runs/run/checkpoint", json={"request_id": "authed-ckpt"})
        assert res.status_code == 200, res.text


# ---------------------------------------------------------------------------
# 4. cpu_worker_service.py multi-node topology validation
# ---------------------------------------------------------------------------

def test_cpu_worker_service_validates_node_topology():
    from hypergan.cpu_worker_service import CPUWorkerService, _validate_node_topology

    def dummy_factory(rank, world_size):
        return None

    def dummy_handler(state, op, payload):
        return None

    common = dict(factory=dummy_factory, handler=dummy_handler, run_id="r1", attempt_id="a1")

    for bad_kwargs in [
        dict(world_size=2, nodes=0),
        dict(world_size=2, nodes=-1),
        dict(world_size=2, nodes=True),
        dict(world_size=2, nodes=1, node_rank=-1),
        dict(world_size=2, nodes=1, node_rank=1),
        dict(world_size=4, nodes=2, node_rank=2),
        dict(world_size=3, nodes=2),
        dict(world_size=4, nodes=2, ranks_per_node=3),
        dict(world_size=4, nodes=2, ranks_per_node=0),
        dict(world_size=4, nodes=2, ranks_per_node=-2),
    ]:
        with pytest.raises(ValueError):
            CPUWorkerService(**common, **bad_kwargs)

    svc = CPUWorkerService(**common, world_size=4, nodes=2, node_rank=1)
    assert svc._ranks_per_node == 2
    assert _validate_node_topology(4, 2, 0, None) == 2
    with pytest.raises(ValueError):
        _validate_node_topology(3, 2, 0, None)


# ---------------------------------------------------------------------------
# 5. CI, Release, Packaging & Docker fixes
# ---------------------------------------------------------------------------

def test_ci_pages_release_dockerfile_and_manifest_hardening(tmp_path, monkeypatch):
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "integration_only" not in ci
    assert "github.head_ref == 'develop'" not in ci
    assert "benchmarks/whitepaper" in ci
    # PARTICLEGAN_SOURCE install must come AFTER wheel[train,...] install.
    wheel_install_idx = ci.index("[train,image,cifar,fid,web,test]")
    pg_install_idx = ci.index("install_particlegan.py")
    assert pg_install_idx > wheel_install_idx
    # InceptionFID verification must actually import scipy.linalg & torch_fidelity and compute frechet_distance.
    assert "import scipy.linalg" in ci
    assert "fid_statistics_to_metric" in ci
    assert "frechet_distance" in ci

    pages = (REPO_ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "refs/heads/develop" not in pages
    assert "refs/heads/master" in pages
    assert "inputs.deploy == true" in pages

    release = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "direct-URL" in release

    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get install" in dockerfile and "git" in dockerfile
    assert "printf" not in dockerfile
    assert "json.dumps" in dockerfile

    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include benchmarks/whitepaper *.toml *.md" in manifest

    # Verify install_particlegan.py uses --force-reinstall when PARTICLEGAN_SOURCE is set,
    # and does not overwrite an existing install when PARTICLEGAN_SOURCE is unset.
    spec = importlib.util.spec_from_file_location(
        "install_particlegan", REPO_ROOT / ".github" / "scripts" / "install_particlegan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, check=True: calls.append(cmd))
    monkeypatch.setenv("PARTICLEGAN_SOURCE", "https://example.invalid/particlegan-0.13.0.whl")
    mod.main()
    assert calls == [[sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                      "https://example.invalid/particlegan-0.13.0.whl"]]

    calls.clear()
    monkeypatch.delenv("PARTICLEGAN_SOURCE", raising=False)
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: object())
    mod.main()
    assert calls == []


@pytest.mark.skipif(os.name == "nt" or shutil.which("sh") is None, reason="POSIX shell test")
def test_dockerfile_commit_json_handles_special_characters(tmp_path):
    from hypergan.provenance import hypergan_source

    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", " ", dockerfile)
    (step,) = [line.strip()[4:] for line in joined.splitlines() if line.strip().startswith("RUN ") and "HYPERGAN_COMMIT" in line]

    package = tmp_path / "build" / "src" / "hypergan"
    package.mkdir(parents=True)
    stub = tmp_path / "bin" / "python"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 0; fi\nexec "{sys.executable}" "$@"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

    tricky_commit = 'abc"def\\ghi'
    env = {**os.environ, "PATH": f"{stub.parent}{os.pathsep}{os.environ.get('PATH', '')}", "HYPERGAN_COMMIT": tricky_commit}
    subprocess.run(["sh", "-c", step], cwd=package.parent.parent, env=env, check=True, timeout=30)
    parsed = json.loads((package / "_build_provenance.json").read_text(encoding="utf-8"))
    assert parsed == {"commit": tricky_commit, "dirty": None, "schema_version": 1}

    # And a valid 40-char SHA is accepted end-to-end by hypergan_source().
    valid_sha = "deadbeef" * 5
    env_valid = {**env, "HYPERGAN_COMMIT": valid_sha}
    subprocess.run(["sh", "-c", step], cwd=package.parent.parent, env=env_valid, check=True, timeout=30)
    assert hypergan_source(package) == {
        "hypergan_commit": valid_sha,
        "hypergan_dirty": None,
        "hypergan_provenance": "build",
    }


# ---------------------------------------------------------------------------
# 6. ROCm preflight & utility scripts
# ---------------------------------------------------------------------------

def test_rocm_preflight_never_silently_falls_back_to_cpu_and_checks_uuid(tmp_path, monkeypatch):
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import rocm_preflight_helper as helper

    # 1. Sanity check must FAIL (not fall back to CPU) when CUDA is unavailable.
    with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=0):
        res = helper.run_sanity_check(device_name="cuda:0")
    assert res.status == "FAIL" and res.passed is False and "refusing silent CPU fallback" in res.details

    # 2. Short training must FAIL (not fall back to CPU) when CUDA is unavailable.
    with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=0):
        tres = helper.run_short_training(device="cuda:0", steps=2, output_dir=tmp_path)
    assert tres.status == "FAIL" and "refusing silent CPU fallback" in tres.details

    # 3. Short training must honor requested GPU index (not hardcoded 0).
    seen_indices: list[tuple[str, object]] = []
    fake_hg = SimpleNamespace(
        packaged=lambda name: {"name": name},
        override=lambda m, ov: m,
        train=lambda model, **kwargs: seen_indices.append(("devices", kwargs["devices"])),
        status=lambda run_dir: {"step": 2, "latest": {}},
    )
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.reset_peak_memory_stats", side_effect=lambda i: seen_indices.append(("reset", i))), \
         patch("torch.cuda.max_memory_allocated", side_effect=lambda i: (seen_indices.append(("max", i)) or 1024 * 1024)), \
         patch.dict(sys.modules, {"hypergan": fake_hg}):
        tres_gpu1 = helper.run_short_training(device="cuda:1", steps=2, output_dir=tmp_path)
    assert tres_gpu1.status == "PASS" and tres_gpu1.device == "cuda:1"
    assert ("reset", 1) in seen_indices and ("max", 1) in seen_indices and ("devices", "1") in seen_indices

    # 4. Device-order table MUST flag MISMATCH when names match but UUIDs differ.
    table = helper.format_device_order_table(
        [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0xaaaa", "pci_bus": "-"}],
        [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "uuid": "0xbbbb", "pci_bus": "0000:03:00.0"}],
    )
    assert "| MISMATCH |" in table

    # 5. Report must NEVER claim "ROCm Single-GPU Ready" when cuda_available is False.
    no_gpu_sanity = helper.SanityCheckResult(
        status="PASS", passed=True, hip_version=None, cuda_available=False, device_count=0,
        device_name="CPU", gcn_arch="none", matmul_fwd_max_diff=0.0, matmul_bwd_max_diff=0.0,
        conv2d_fwd_max_diff=0.0, conv2d_bwd_max_diff=0.0, details="cpu",
    )
    no_gpu_train = helper.TrainingRunResult(
        status="PASS", recipe="gaussian-grid", device="cpu", target_steps=2, completed_steps=2,
        duration_sec=1.0, steps_per_sec=2.0, peak_vram_mb=0.0, final_metrics={}, details="ok",
    )
    report = helper.format_preflight_report(
        env_data={}, install_data={}, sanity_res=no_gpu_sanity, device_order_md="",
        ref_suite_data={"total": 5, "passed": 5, "failed": 0, "skipped": 0, "duration_sec": 1.0, "status": "PASS"},
        single_gpu_tests=[], single_gpu_results=[], deferred_tests=[],
        train_res=no_gpu_train, inventory_md="", failures=[],
    )
    assert "ROCm Single-GPU Ready" not in report
    assert "NOT READY" in report

    # 6. parse_pytest_summary reports real counts.
    counts = helper.parse_pytest_summary("..sF\n1 failed, 118 passed, 2 skipped in 4.21s\n", returncode=1)
    assert counts == {"total": 121, "passed": 118, "failed": 1, "skipped": 2}

    # 7. Script version pin & --no-deps preservation + non-stale versions in helper scripts.
    sh_text = (REPO_ROOT / "scripts" / "rocm_preflight.sh").read_text(encoding="utf-8")
    assert "2.5.1" not in sh_text
    assert "--no-deps -e" in sh_text
    assert "torch.version, 'hip'" in sh_text

    cifar_text = (REPO_ROOT / "scripts" / "benchmark_particlegan_cifar.py").read_text(encoding="utf-8")
    assert "config['training']['batch_size']" in cifar_text
    assert "64 / statistics.median" not in cifar_text

    for script_name in ("modal_multinode_smoke.py", "modal_scaling.py"):
        text = (REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "particlegan==0.9.3" not in text
        assert ".apt_install('git')" in text

    transgan_sh = (REPO_ROOT / "scripts" / "benchmark_transgan_critics.sh").read_text(encoding="utf-8")
    assert "HyperGAN 2.0.0" not in transgan_sh

    # 8. Full run_preflight_pipeline exits non-zero when CUDA is unavailable, records real hypergan.__version__ & ref counts.
    import hypergan
    fake_ref = SimpleNamespace(returncode=0, stdout="12 passed, 1 skipped in 0.50s\n", stderr="")
    out_dir = tmp_path / "preflight_out"
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.device_count", return_value=0), \
         patch.object(helper, "discover_and_partition_cuda_tests", return_value=(["tests/cuda/test_one.py::test_a"], [])), \
         patch.object(helper, "dump_gpu_inventory", return_value=({"hardware": [], "processes": []}, "none")), \
         patch("hypergan.queue.inspect_gpu_hardware", return_value=[]), \
         patch.object(helper.subprocess, "run", return_value=fake_ref):
        rc = helper.run_preflight_pipeline(output_dir=out_dir, quick=True, device_str="cuda:0")
    assert rc == 1
    report_text = (out_dir / "rocm_preflight_report.md").read_text(encoding="utf-8")
    assert "ROCm Single-GPU Ready" not in report_text
    assert f"`{hypergan.__version__} (editable with --no-deps)`" in report_text
    assert "- **Total Tests Executed**: 13" in report_text
    assert "- **Passed**: 12" in report_text
    assert "- **Skipped**: 1" in report_text


# ---------------------------------------------------------------------------
# 7. Site XSS blocking & docs version/preset drift
# ---------------------------------------------------------------------------

def test_markdown_renderer_blocks_javascript_and_unsafe_data_urls():
    site_dir = str(REPO_ROOT / "site")
    if site_dir not in sys.path:
        sys.path.insert(0, site_dir)
    import markdown_render

    kwargs = dict(
        source_path="docs/sample.md",
        page_dir="docs/sample",
        route_map={},
        repo_url="https://github.com/HyperGAN/HyperGAN",
        commit="master",
        local_heading_ids=set(),
    )

    for dangerous in (
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "java\tscript:alert(1)",
        "  javascript:void(0)",
        "vbscript:msgbox(1)",
        "data:text/html,<script>alert(1)</script>",
        "data:image/svg+xml;base64,PHN2Zz4=",
        "data:image/png;base64,AAAA",  # data: blocked in hyperlinks
    ):
        assert markdown_render._rewrite_url(dangerous, is_image=False, **kwargs) == "", dangerous

    # Safe raster data URI allowed in images, SVG/HTML data URI blocked in images.
    assert markdown_render._rewrite_url("data:image/png;base64,AAAA", is_image=True, **kwargs) == "data:image/png;base64,AAAA"
    assert markdown_render._rewrite_url("data:image/svg+xml;base64,PHN2Zz4=", is_image=True, **kwargs) == ""
    assert markdown_render._rewrite_url("javascript:alert(1)", is_image=True, **kwargs) == ""

    # Legitimate URLs remain intact.
    assert markdown_render._rewrite_url("https://example.com/a", is_image=False, **kwargs) == "https://example.com/a"
    assert markdown_render._rewrite_url("mailto:hi@example.com", is_image=False, **kwargs) == "mailto:hi@example.com"

    # Full render check: markdown-it + _rewrite_url must emit zero javascript:/data: hrefs or unsafe data: srcs.
    rendered = markdown_render.render_markdown(
        "[bad](javascript:alert(1)) [bad2](data:text/html,foo) [datalink](data:image/png;base64,AAAA) "
        "![svg](data:image/svg+xml;base64,PHN2Zz4=) ![ok](data:image/png;base64,AAAA)",
        source_path="docs/sample.md",
        page_url_path="docs/sample/index.html",
        route_map={},
    )["html"]
    assert 'href="javascript:' not in rendered.lower()
    assert 'href="data:' not in rendered.lower()
    assert 'src="data:image/svg+xml' not in rendered.lower()
    assert 'href=""' in rendered  # datalink href neutralized
    assert 'src="data:image/png;base64,AAAA"' in rendered  # safe raster data URI preserved on <img>


def test_docs_and_site_version_preset_drift_resolved():
    build_py = (REPO_ROOT / "site" / "build.py").read_text(encoding="utf-8")
    assert "ParticleGAN unified_v12" not in build_py
    assert 'page_description="Install HyperGAN 2.2.0' not in build_py

    llms = (REPO_ROOT / "llms.txt").read_text(encoding="utf-8")
    assert "111 fields" not in llms
    assert "112 fields" in llms
    assert "ParticleGAN (`0.13.0`, `axiom`)" in llms

    readme_line20 = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()[19]
    assert "`axiom`" in readme_line20 and "`unified_v12`" not in readme_line20

    colab = (REPO_ROOT / "docs" / "colab.md").read_text(encoding="utf-8")
    assert "particlegan==0.12.0" not in colab

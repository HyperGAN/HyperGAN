"""Validate the OCI Dockerfile, .dockerignore, documentation, and entrypoint contract."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from hypergan.run_signals import GracefulStop

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
DOCKER_DOC = ROOT / "docs" / "docker.md"


def _dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    """Parse logical Dockerfile instructions after joining line continuations."""
    logical_lines: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            if not current:
                continue
        if stripped.endswith("\\"):
            current += stripped[:-1].strip() + " "
        else:
            current += stripped
            logical_lines.append(current.strip())
            current = ""
    if current.strip():
        logical_lines.append(current.strip())

    instructions: list[tuple[str, str]] = []
    for line in logical_lines:
        parts = line.split(None, 1)
        if len(parts) == 2:
            instructions.append((parts[0].upper(), parts[1]))
        elif len(parts) == 1:
            instructions.append((parts[0].upper(), ""))
    return instructions


def test_dockerfile_pinned_multistage_and_security_contract():
    assert DOCKERFILE.is_file(), "Dockerfile must exist at repository root"
    content = DOCKERFILE.read_text(encoding="utf-8")
    instructions = _dockerfile_instructions(content)

    from_stages = [arg for op, arg in instructions if op == "FROM"]
    assert len(from_stages) >= 2, "Dockerfile must use a multi-stage build"
    for stage in from_stages:
        assert stage.startswith("python:3.12-slim-bookworm"), (
            f"Base image must be pinned to python:3.12-slim-bookworm, got: {stage}"
        )
        assert not stage.endswith(":latest"), "Base image must not use unpinned :latest tag"

    # Must not contain retired 0.x artifacts reported in #224.
    assert "handler.py" not in content
    assert "python:3.4" not in content
    assert "ADD build" not in content

    # Builder stage must build a wheel into /dist and runtime stage must install the wheel.
    run_cmds = [arg for op, arg in instructions if op == "RUN"]
    assert any("pip wheel" in cmd and "/dist" in cmd for cmd in run_cmds), (
        "Builder stage must build a wheel into /dist"
    )
    assert any("pip install" in cmd and ".whl" in cmd for cmd in run_cmds), (
        "Runtime stage must install from the built wheel"
    )

    # Non-root user contract.
    users = [arg for op, arg in instructions if op == "USER"]
    assert users == ["hypergan:hypergan"], f"Runtime must switch to non-root user hypergan:hypergan, got {users}"
    assert any("useradd" in cmd and "1000" in cmd for cmd in run_cmds), (
        "Runtime must create uid 1000 hypergan user"
    )

    # Explicit mount directories, workdir, signal handling, and exec-form entrypoint.
    workdirs = [arg for op, arg in instructions if op == "WORKDIR"]
    assert workdirs[-1] == "/workspace"

    volumes = [json.loads(arg) for op, arg in instructions if op == "VOLUME"]
    assert ["/data", "/runs"] in volumes

    stopsignals = [arg for op, arg in instructions if op == "STOPSIGNAL"]
    assert stopsignals == ["SIGINT"]

    entrypoints = [json.loads(arg) for op, arg in instructions if op == "ENTRYPOINT"]
    assert entrypoints == [["python", "-m", "hypergan"]]

    cmds = [json.loads(arg) for op, arg in instructions if op == "CMD"]
    assert cmds == [["--help"]]


def test_dockerignore_excludes_local_state_and_non_package_trees():
    assert DOCKERIGNORE.is_file(), ".dockerignore must exist at repository root"
    lines = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required_entries = {
        ".git",
        ".claude",
        ".venv",
        "venv",
        "__pycache__/",
        "build/",
        "dist/",
        "runs/",
        "docs/",
        "reports/",
        "tests/",
        "frontend/node_modules/",
    }
    missing = required_entries - lines
    assert not missing, f".dockerignore is missing required exclusions: {sorted(missing)}"


def test_docker_documentation_covers_mounts_signals_and_profiles():
    assert DOCKER_DOC.is_file(), "docs/docker.md must exist"
    doc = DOCKER_DOC.read_text(encoding="utf-8")
    for required_snippet in (
        "docker build",
        "--build-arg EXTRAS=",
        "dst=/data,readonly",
        "dst=/runs",
        "--no-server",
        "8765:8765",
        "STOPSIGNAL SIGINT",
        "docker stop",
        "--gpus all",
    ):
        assert required_snippet in doc, f"docs/docker.md must document {required_snippet!r}"


def test_container_entrypoint_and_mount_layout_smoke(tmp_path):
    """Exercise the container entrypoint, /workspace + /runs mount layout, and SIGINT contract."""
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir()
    runs_dir.mkdir()
    workspace_dir.mkdir()

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HYPERGAN_STATE_DIR": str(runs_dir / ".hypergan"),
    }

    # 1. Default entrypoint + CMD (`python -m hypergan --help`) outside checkout.
    help_proc = subprocess.run(
        [sys.executable, "-I", "-m", "hypergan", "--help"],
        cwd=workspace_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "validate" in help_proc.stdout

    # 2. Scaffold a starter project in /workspace and validate it with JSON output.
    project_dir = workspace_dir / "toy_project"
    new_proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "hypergan",
            "new",
            "--project",
            str(project_dir),
            "--template",
            "gaussian-grid",
            "--device",
            "cpu",
            "--json",
        ],
        cwd=workspace_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert new_proc.returncode == 0, new_proc.stderr
    created_info = json.loads(new_proc.stdout)
    recipe_path = Path(created_info["recipe_path"])
    assert recipe_path.is_file()

    val_proc = subprocess.run(
        [sys.executable, "-I", "-m", "hypergan", "validate", str(recipe_path), "--json"],
        cwd=workspace_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert val_proc.returncode == 0, val_proc.stderr
    resolved = json.loads(val_proc.stdout)
    assert isinstance(resolved, dict)

    # 3. Verify STOPSIGNAL SIGINT contract with GracefulStop.
    with GracefulStop() as gs:
        signal.raise_signal(signal.SIGINT)
        assert gs.reason == "SIGINT"

    # 4. Optional live Docker daemon build-and-run check when explicitly enabled.
    if os.environ.get("HYPERGAN_TEST_DOCKER_BUILD") == "1" and shutil.which("docker"):
        info = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=15)
        if info.returncode == 0:
            tag = f"hypergan-smoke-test:{os.getpid()}"
            try:
                build = subprocess.run(
                    ["docker", "build", "-t", tag, str(ROOT)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                assert build.returncode == 0, build.stderr
                run = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--mount",
                        f"type=bind,src={data_dir},dst=/data,readonly",
                        "--mount",
                        f"type=bind,src={runs_dir},dst=/runs",
                        "--mount",
                        f"type=bind,src={workspace_dir},dst=/workspace",
                        tag,
                        "validate",
                        f"/workspace/toy_project/configs/{recipe_path.name}",
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                assert run.returncode == 0, run.stderr
                assert isinstance(json.loads(run.stdout), dict)
            finally:
                subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=30)

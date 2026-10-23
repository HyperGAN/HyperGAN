"""Installed distributions must exclude the retired runtime and dependencies."""

import importlib.metadata


def test_distribution_contains_only_supported_package():
    distribution = importlib.metadata.distribution("hypergan")
    files = {str(path).replace("\\", "/") for path in distribution.files}
    if any(path.startswith("__editable__.") for path in files):
        from pathlib import Path
        import hypergan
        pkg_root = Path(hypergan.__file__).resolve().parent
        src_root = pkg_root.parent
        files |= {p.relative_to(src_root).as_posix() for p in pkg_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    elif any(path.startswith("src/hypergan/") for path in files):
        files |= {path.removeprefix("src/") for path in files if path.startswith("src/hypergan/")}
    assert "hypergan/cli.py" in files
    assert "hypergan/__main__.py" in files
    assert not any("needs_pytorch" in path for path in files)
    assert not any(path.startswith(("hypergan/configurations/", "hypergan/backends/", "worktrees/")) for path in files)
    assert distribution.version == "2.0.0a1"


def test_base_requirements_do_not_pull_training_stack():
    from packaging.requirements import Requirement

    requirements = importlib.metadata.requires("hypergan") or []
    active = [Requirement(value) for value in requirements]
    assert not any(
        requirement.name in {"torch", "particlegan", "numpy"}
        and (requirement.marker is None or requirement.marker.evaluate({"extra": ""}))
        for requirement in active
    )

"""Installed distributions must exclude the retired runtime and dependencies."""

import importlib.metadata


def test_distribution_contains_only_supported_package():
    distribution = importlib.metadata.distribution("hypergan")
    files = {str(path).replace("\\", "/") for path in distribution.files}
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

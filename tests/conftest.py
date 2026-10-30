import os
from pathlib import Path
import sys

import pytest

# Ensure repository root is on sys.path and PYTHONPATH for multiprocessing spawn workers
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_cur_pythonpath = os.environ.get("PYTHONPATH")
if not _cur_pythonpath:
    os.environ["PYTHONPATH"] = _REPO_ROOT
elif _REPO_ROOT not in _cur_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{_cur_pythonpath}"



@pytest.fixture(autouse=True, scope="session")
def _isolated_hypergan_cache(tmp_path_factory):
    # ImageFolder caches per-file inspections under the user's cache by default.
    # Tests (and the CLI subprocesses they start) must not write there.
    previous = os.environ.get("HYPERGAN_CACHE_DIR")
    os.environ["HYPERGAN_CACHE_DIR"] = str(tmp_path_factory.mktemp("hypergan-cache"))
    yield
    if previous is None:
        os.environ.pop("HYPERGAN_CACHE_DIR", None)
    else:
        os.environ["HYPERGAN_CACHE_DIR"] = previous


@pytest.fixture(autouse=True)
def private_viewer_cache(tmp_path_factory, monkeypatch):
    """Viewer caches live outside runs; keep every test's out of the owner's home.

    The cache prunes its least recently used runs, so a test writing to the real
    default cache could delete the owner's viewer caches.
    """
    cache = tmp_path_factory.mktemp('viewer-cache')
    monkeypatch.setenv('HYPERGAN_VIEWER_CACHE', str(cache))
    return cache


@pytest.fixture(autouse=True)
def isolate_hypergan_state(tmp_path):
    """Isolate HyperGAN state directory (registry, locks, runs.json) per test."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("HYPERGAN_STATE_DIR")
    os.environ["HYPERGAN_STATE_DIR"] = str(state_dir)
    yield state_dir
    if previous is None:
        os.environ.pop("HYPERGAN_STATE_DIR", None)
    else:
        os.environ["HYPERGAN_STATE_DIR"] = previous


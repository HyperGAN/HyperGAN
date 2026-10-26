"""Suite-wide isolation that every test directory shares."""
import os

import pytest


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

import pytest


@pytest.fixture(autouse=True)
def private_viewer_cache(tmp_path_factory, monkeypatch):
    """Viewer caches live outside runs; keep each test's out of the owner's home."""
    cache = tmp_path_factory.mktemp('viewer-cache')
    monkeypatch.setenv('HYPERGAN_VIEWER_CACHE', str(cache))
    return cache

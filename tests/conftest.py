import pytest


@pytest.fixture(autouse=True)
def private_viewer_cache(tmp_path_factory, monkeypatch):
    """Viewer caches live outside runs; keep every test's out of the owner's home.

    The cache prunes its least recently used runs, so a test writing to the real
    default cache could delete the owner's viewer caches.
    """
    cache = tmp_path_factory.mktemp('viewer-cache')
    monkeypatch.setenv('HYPERGAN_VIEWER_CACHE', str(cache))
    return cache

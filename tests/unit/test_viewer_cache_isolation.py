import os
from pathlib import Path

from hypergan.web_service import view_cache_root


def test_no_test_writes_the_owners_viewer_cache():
    """tests/conftest.py points every test's viewer cache at a temporary directory.

    The cache prunes its least recently used runs, so a test using the real
    default cache could delete the owner's caches.
    """
    root = view_cache_root().resolve()
    assert os.environ.get('HYPERGAN_VIEWER_CACHE')
    assert not root.is_relative_to((Path.home() / '.cache').resolve())

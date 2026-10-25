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

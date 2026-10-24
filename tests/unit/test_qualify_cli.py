"""Alias import proxy for tests/unit/test_qualify.py."""
try:
    from .test_qualify import *  # noqa: F401, F403
except ImportError:
    try:
        from unit.test_qualify import *  # noqa: F401, F403
    except ImportError:
        from tests.unit.test_qualify import *  # noqa: F401, F403

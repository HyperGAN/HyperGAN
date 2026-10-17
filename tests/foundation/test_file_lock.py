"""Exercise Windows lock contention without requiring a Windows test host."""
import errno
import sys
from types import SimpleNamespace

import pytest

from hypergan import _file_lock


@pytest.mark.parametrize('blocking, error', [(False, errno.EACCES), (True, errno.EINVAL)])
def test_windows_lock_propagates_nonblocking_and_noncontention_errors(monkeypatch, blocking, error):
    def fail(*args):
        raise OSError(error, 'injected lock failure')
    backend = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=fail)
    monkeypatch.setitem(sys.modules, 'msvcrt', backend)
    monkeypatch.setattr(_file_lock, 'os', SimpleNamespace(name='nt', SEEK_SET=0, lseek=lambda *args: None))
    monkeypatch.setattr(_file_lock.time, 'sleep', lambda _: pytest.fail('unexpected lock retry'))
    with pytest.raises(OSError) as raised:
        _file_lock.lock_file(7, blocking=blocking)
    assert raised.value.errno == error


def test_windows_blocking_lock_retries_contention_and_unlocks_first_byte(monkeypatch):
    calls = []
    busy = True
    def locking(fd, operation, length):
        nonlocal busy
        calls.append(('lock', fd, operation, length))
        if busy:
            busy = False
            raise OSError(errno.EACCES, 'already locked')
    backend = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
    monkeypatch.setitem(sys.modules, 'msvcrt', backend)
    monkeypatch.setattr(_file_lock, 'os', SimpleNamespace(
        name='nt', SEEK_SET=0, lseek=lambda *args: calls.append(('seek', *args))))
    monkeypatch.setattr(_file_lock.time, 'sleep', lambda delay: calls.append(('retry', delay)))
    _file_lock.lock_file(7, blocking=True)
    _file_lock.unlock_file(7)
    assert calls == [('seek', 7, 0, 0), ('lock', 7, 1, 1), ('retry', .01),
                     ('seek', 7, 0, 0), ('lock', 7, 1, 1),
                     ('seek', 7, 0, 0), ('lock', 7, 2, 1)]

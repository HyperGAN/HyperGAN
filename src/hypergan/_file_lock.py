"""Exclusive descriptor locks shared by queue writers and run-lock probes."""
from contextlib import ExitStack, contextmanager
import errno
import os
import time


def lock_file(fd, *, blocking=False):
    if os.name == 'nt':
        import msvcrt
        while True:
            # Match run_state.run_lock: the first byte can lie beyond EOF.
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if not blocking or error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(.01)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def unlock_file(fd):
    if os.name == 'nt':
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def retry_lock(acquire, *, seconds=1.0, interval=.02):
    """Enter the lock context ``acquire()`` returns, retrying while it raises RuntimeError.

    Liveness probes (`ps`, `logs -f`, the queue) hold a run's lock for a few
    microseconds to test it. A trainer or maintenance command that retries
    briefly is not refused just because a probe landed at the same instant.
    """
    deadline = time.monotonic() + seconds
    stack = ExitStack()
    while True:
        try:
            stack.enter_context(acquire())
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)
    with stack:
        yield

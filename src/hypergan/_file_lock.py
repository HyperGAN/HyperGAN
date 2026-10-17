"""Exclusive descriptor locks shared by queue writers and run-lock probes."""
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

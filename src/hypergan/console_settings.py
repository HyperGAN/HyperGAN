"""Run-local console policy, independent from numerical and metric identity."""
import json
import os
import stat
from pathlib import Path
import time

from .run_state import atomic_json

DEFAULT_PROGRESS_EVERY = 100
MAX_PROGRESS_EVERY = 1000000000


def validate_settings(value):
    if (type(value) is not dict or set(value) != {'progress_every'} or
            type(value['progress_every']) is not int or
            not 1 <= value['progress_every'] <= MAX_PROGRESS_EVERY):
        raise ValueError('Expected progress_every as an integer in 1..1000000000')
    return value


def read_settings(run_dir):
    path = Path(run_dir) / 'console.json'
    descriptor = None
    try:
        # Reject devices and named pipes before opening, including on Windows.
        # O_NONBLOCK/O_NOFOLLOW also protect POSIX against replacement between
        # this check and open; recheck the actual descriptor before reading.
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Console settings must be a regular file without links')
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
        flags |= getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Console settings must be a regular file')
        if info.st_size > 4096:
            raise ValueError('Console settings exceed 4096 bytes')
        with os.fdopen(descriptor, 'rb') as stream:
            descriptor = None  # fdopen owns the descriptor, including failures.
            raw = stream.read(4097)
    except FileNotFoundError:
        return {'progress_every': DEFAULT_PROGRESS_EVERY}
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > 4096:
        raise ValueError('Console settings exceed 4096 bytes')
    return validate_settings(json.loads(raw))


def write_settings(run_dir, value):
    value = validate_settings(value)
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError('Run directory must exist before changing console settings')
    atomic_json(root / 'console.json', value)
    return value


class ConsolePolicy:
    """Bound reads to four per second; UI changes apply at update boundaries."""
    def __init__(self, run_dir=None, *, progress_every=None):
        self.run_dir = run_dir
        self.override = progress_every
        self.every = DEFAULT_PROGRESS_EVERY if progress_every is None else validate_settings({'progress_every': progress_every})['progress_every']
        self.next_check = 0.
        self.warned = False

    def refresh(self, stderr):
        if self.run_dir is None or time.monotonic() < self.next_check:
            return
        self.next_check = time.monotonic() + .25
        try:
            if self.override is not None:
                write_settings(self.run_dir, {'progress_every': self.override})
                self.override = None
            self.every = read_settings(self.run_dir)['progress_every']
            self.warned = False
        except (OSError, ValueError) as error:
            if not self.warned:
                stderr.write(f'warning: console settings unavailable; retaining interval {self.every}: {error}\n')
                self.warned = True

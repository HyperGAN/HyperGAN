"""Bounded background persistence for live observation.

Only explicit durability boundaries wait for storage. Accepted event rows remain
ordered and lossless; the controller reports training rows refused by the queue.
Status snapshots coalesce because only their latest observed state is meaningful.
"""
from collections import deque
from copy import deepcopy
import math
from threading import Condition, Thread

from .run_state import EventJournal, atomic_json


def _event_snapshot(value, *, max_bytes=1024 * 1024):
    """Copy only bounded JSON values without serializing on the update thread."""
    budget = [max_bytes, 16384]
    def copy(item, depth=0):
        budget[1] -= 1
        if budget[1] < 0 or depth > 32:
            raise ValueError('Observation event exceeds its structure bound')
        if item is None or type(item) in (bool, int, float):
            if type(item) is float and not math.isfinite(item):
                raise ValueError('Observation events require finite values')
            # bit_length / 3 conservatively bounds decimal integer digits.
            budget[0] -= max(32, item.bit_length() // 3 + 3) if type(item) is int else 32
            result = item
        elif type(item) is str:
            # Non-BMP characters require two six-byte UTF-16 escapes.
            budget[0] -= len(item) * 12 + 2
            result = item
        elif type(item) in (list, tuple):
            budget[0] -= 2 + len(item)
            result = [copy(child, depth + 1) for child in item]
        elif type(item) is dict:
            budget[0] -= 2 + len(item) * 2
            result = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError('Observation event keys must be strings')
                result[copy(key, depth + 1)] = copy(child, depth + 1)
        else:
            raise ValueError('Observation events require JSON values')
        if budget[0] < 0:
            raise ValueError('Observation event exceeds 1 MiB byte bound')
        return result
    result = copy(value)
    return result, max_bytes - budget[0]


class ObservationIO:
    def __init__(self, run_dir, attempt_dir, *, capacity=256, max_bytes=8 * 1024 * 1024):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Observation queue capacity must be positive')
        self.journal = EventJournal(run_dir)
        self.paths = (run_dir / 'manifest.json', attempt_dir / 'manifest.json')
        self.capacity = capacity
        if type(max_bytes) is not int or max_bytes < 1024 * 1024:
            raise ValueError('Observation byte capacity must be at least 1 MiB')
        self.max_bytes = max_bytes
        self.queued_bytes = 0
        self.condition = Condition()
        self.events = deque()
        self.status = None
        self.error = None
        self.active = False
        self.stopping = False
        self.worker = Thread(target=self._run, name='hypergan-observation-io', daemon=True)
        self.worker.start()

    @property
    def failed(self):
        return self.error is not None or self.journal.failed

    def check(self):
        if self.error is not None:
            raise OSError(f'Observation storage failed: {self.error}') from self.error

    def append(self, row, *, wait=False):
        row, size = _event_snapshot(row)
        with self.condition:
            self.check()
            while len(self.events) >= self.capacity or self.queued_bytes + size > self.max_bytes:
                if not wait:
                    return False
                self.condition.wait()
                self.check()
            self.events.append((row, size))
            self.queued_bytes += size
            self.condition.notify_all()
        return True

    def publish(self, manifest, *, wait=False):
        with self.condition:
            self.check()
            self.status = deepcopy(manifest)
            self.condition.notify_all()
        if wait:
            self.flush()

    def flush(self):
        with self.condition:
            while self.events or self.status is not None or self.active:
                self.check()
                self.condition.wait()
            self.check()

    def commit_boundary(self):
        self.flush()
        # The controller is the sole producer; the idle worker cannot mutate the
        # journal while this durable checkpoint prefix is committed.
        return self.journal.commit_boundary()

    def close(self):
        try:
            self.flush()
        finally:
            with self.condition:
                self.stopping = True
                self.condition.notify_all()
            self.worker.join()

    def _run(self):
        try:
            while True:
                with self.condition:
                    while not self.events and self.status is None and not self.stopping:
                        self.condition.wait()
                    if self.stopping:
                        return
                    # Never hold the producer lock during storage I/O.
                    event = self.events.popleft() if self.events else None
                    if event is not None:
                        self.queued_bytes -= event[1]
                    status, self.status = self.status, None
                    self.active = True
                    self.condition.notify_all()
                if event is not None:
                    self.journal.append(event[0])
                if status is not None:
                    for path in self.paths:
                        atomic_json(path, status)
                with self.condition:
                    self.active = False
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.active = False
                self.condition.notify_all()

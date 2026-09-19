"""Bounded optional progress callbacks, isolated from the numerical parent.

A delivery owns a fresh, group-free worker and independent reaping broker. Only
an importable function reference and a finite JSON snapshot cross the boundary;
callbacks receive ``callback(event)`` and their return value is ignored. Callback
module globals do not persist between deliveries. The caller is sequential: one
delivery can be outstanding, with no queue and no retry.

The total deadline covers startup, callback and shutdown; broker cleanup grace
is additional. Slow stdout/stderr is subject to the same deadline. As with the
worker service, guarded importable main-module bootstrap must not block, custom
subprocess descendants are unmanaged, and broker/host failure is outside the
parent-death guarantee. Encoded size limits do not bound custom allocations.
"""
import importlib
import inspect
import json
import math
import sys
import threading
from pathlib import Path

from .cpu_worker_service import CPUWorkerService, _json

MAX_EVENT_BYTES = 65536


class ObserverError(RuntimeError):
    """Optional callback failed; this observer is now explicitly disabled."""


def callback_reference(callback):
    """Validate an already-imported function without invoking user code."""
    if (not inspect.isfunction(callback) or '<locals>' in callback.__qualname__
            or callback.__name__ == '<lambda>'):
        raise ValueError('Progress callback must be an importable module-level function; closures, lambdas and callable instances are unsupported')
    module_name, qualname = callback.__module__, callback.__qualname__
    module = sys.modules.get(module_name)
    if module is None or not getattr(module, '__file__', None) or not Path(module.__file__).is_file():
        raise ValueError('Progress callback requires an importable Python file and a guarded main entry point')
    value = module
    for part in qualname.split('.'):
        # Inspect dictionaries instead of invoking module/class __getattr__.
        value = vars(value).get(part) if hasattr(value, '__dict__') else None
    if value is not callback:
        raise ValueError('Progress callback must be available unchanged at its module:qualname reference')
    return module_name + ':' + qualname


def _create_callback(rank, world_size, reference, event):
    module_name, qualname = reference.split(':', 1)
    callback = importlib.import_module(module_name)
    for part in qualname.split('.'):
        callback = getattr(callback, part)
    if not inspect.isfunction(callback):
        raise ValueError('Progress callback reference no longer resolves to a function')
    return callback, json.loads(event)


def _deliver_callback(state, operation, payload):
    if operation != 'deliver' or payload is not None:
        raise ValueError('Invalid progress callback operation')
    callback, event = state
    callback(event)
    return None


class BoundedObserver:
    """Synchronous optional observer with no process or event queue at rest.

    ``deliver`` returns True after callback completion and worker reaping. The
    first runtime failure raises ObserverError and disables future delivery;
    later calls return False, with ``disabled`` True. Invalid caller event data
    raises ValueError before spawning and does not disable the observer. Parent
    KeyboardInterrupt/SystemExit propagate after cleanup and also disable it.
    ``close`` is final; delivery after close is an explicit error. Construction
    performs validation only and can precede run/attempt filesystem mutation.
    """
    def __init__(self, callback, *, timeout=5.0, run_id='observer', attempt_id='observer'):
        self.reference = callback_reference(callback)
        try:
            valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('Progress timeout must be finite positive seconds')
        self.timeout = float(timeout)
        self.run_id, self.attempt_id = run_id, attempt_id
        self._disabled = self._closed = False
        self._delivery_lock = threading.Lock()  # No callback thread is created.
        self.broker_pid = None
        self.worker_pids = []
        # Validate IDs and service limits before any filesystem or process work.
        self._service(b'{}')

    @property
    def disabled(self):
        return self._disabled

    def _service(self, encoded):
        return CPUWorkerService(_create_callback, _deliver_callback,
            args=(self.reference, encoded), run_id=self.run_id, attempt_id=self.attempt_id,
            world_size=1, initialize_process_group=False,
            startup_timeout=self.timeout, command_timeout=self.timeout,
            collective_timeout=self.timeout, total_timeout=self.timeout)

    def deliver(self, event):
        if not self._delivery_lock.acquire(blocking=False):
            raise RuntimeError('Only one progress callback delivery can be outstanding')
        try:
            if self._closed:
                raise RuntimeError('Progress observer is closed')
            if self._disabled:
                return False
            if type(event) is not dict:
                raise ValueError('Progress event must be a JSON object')
            encoded = _json(event)  # Finite snapshot, maximum MAX_EVENT_BYTES.
            service = self._service(encoded)
            try:
                with service:
                    self.broker_pid, self.worker_pids = service.broker_pid, list(service.worker_pids)
                    service.command('deliver')
                return True
            except BaseException as error:
                self.broker_pid = service.broker_pid
                self.worker_pids = list(service.worker_pids)
                self._disabled = True
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ObserverError(f'Progress callback {self.reference} failed and was disabled: {error}') from error
        finally:
            self._delivery_lock.release()

    def close(self):
        if not self._delivery_lock.acquire(blocking=False):
            raise RuntimeError('Cannot close an observer during progress delivery')
        try:
            self._closed = True
        finally:
            self._delivery_lock.release()

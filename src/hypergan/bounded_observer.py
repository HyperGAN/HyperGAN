"""Bounded optional progress callbacks, isolated from the numerical parent.

A delivery owns a fresh, group-free worker and independent reaping broker. Only
an importable function reference and a finite JSON snapshot cross the boundary;
callbacks receive ``callback(event)`` and their return value is ignored. Callback
module globals do not persist between deliveries. The caller is sequential: one
delivery can be outstanding, with no queue and no retry.

``AsyncBoundedObserver`` keeps that synchronous primitive on a dispatcher thread
and drops busy notifications instead of waiting on the training thread.

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
import queue
import sys
import threading
import time
from pathlib import Path

from .cpu_worker_service import CPUWorkerService, _json
from .run_controller import ObserverError

MAX_EVENT_BYTES = 65536


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
    from .metric_plugins import cpu_observation_resources
    cpu_observation_resources()
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
    def __init__(self, callback, *, timeout=5.0, run_id='observer', attempt_id='observer', cancellation_event=None):
        self.reference = callback_reference(callback)
        try:
            valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('Progress timeout must be finite positive seconds')
        self.timeout = float(timeout)
        self.run_id, self.attempt_id = run_id, attempt_id
        self._cancellation_event = cancellation_event
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
            collective_timeout=self.timeout, total_timeout=self.timeout,
            cancellation_event=self._cancellation_event)

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


class AsyncBoundedObserver:
    """One accepted callback at a time; busy progress notifications are dropped.

    Only the dispatcher starts or joins workers. The host snapshots finite JSON
    and polls results without waiting. Terminal delivery drains prior work and
    its own callback; exceptional cleanup cancels/reaps the active worker.
    """
    def __init__(self, callback, **options):
        self._cancel = threading.Event()
        self._observer = BoundedObserver(callback, cancellation_event=self._cancel, **options)
        self._timeout = self._observer.timeout
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=1)
        self._thread = None
        self._pending = self._closed = self._disabled = False
        self._deadline = 0.0
        self._status = {'accepted': 0, 'completed': 0, 'dropped': 0, 'failed': 0}

    @property
    def disabled(self):
        return self._disabled

    def start(self):
        if self._closed:
            raise RuntimeError('Progress observer is closed')
        if self._thread is None:
            self._thread = threading.Thread(target=self._dispatch,
                name='hypergan-progress-observer', daemon=True)
            self._thread.start()

    def _dispatch(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            encoded, deadline = job
            error = None
            try:
                self._observer.timeout = max(1e-9, deadline - time.monotonic())
                self._observer.deliver(json.loads(encoded))
            except BaseException as exc:
                error = exc
            self._results.put_nowait((json.loads(encoded).get('step'), error))

    def poll(self):
        try:
            step, error = self._results.get_nowait()
        except queue.Empty:
            return
        self._pending = False
        if error is not None:
            self._disabled = True
            self._status['failed'] += 1
            self._status['last_failed_step'] = step
            failure = ObserverError(str(error))
            failure.observation_step = step
            raise failure from error
        self._status['completed'] += 1
        self._status['last_completed_step'] = step

    def deliver(self, event):
        if self._closed:
            raise RuntimeError('Progress observer is closed')
        if type(event) is not dict:
            raise ValueError('Progress event must be a JSON object')
        self.poll()
        if self._disabled:
            return False
        if self._pending:
            self._status['dropped'] += 1
            self._status['last_dropped_step'] = event.get('step')
            return False
        if self._thread is None:
            raise RuntimeError('Progress observer must be started before training')
        encoded = _json(event)
        self._deadline = time.monotonic() + self._timeout
        self._jobs.put_nowait((encoded, self._deadline))
        self._pending = True
        self._status['accepted'] += 1
        return True

    def _wait(self):
        if self._pending:
            # Only used at terminal boundaries, never from deliver or poll.
            timeout = max(0.0, self._deadline - time.monotonic()) + 8.0
            try:
                step, error = self._results.get(timeout=timeout)
            except queue.Empty as exc:
                raise ObserverError('Progress worker exceeded its deadline and cleanup grace') from exc
            self._results.put_nowait((step, error))
            self.poll()

    def finish(self, event):
        if self._closed:
            return
        try:
            self._wait()
            if not self._disabled:
                self.deliver(event)
                self._wait()
        finally:
            self.close()

    def statistics(self):
        return dict(self._status, pending=self._pending)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(8.0)
            if self._thread.is_alive():
                raise RuntimeError('Progress dispatcher has not completed worker cleanup')
        if self._pending:
            self._status['cancelled'] = self._status.get('cancelled', 0) + 1
            self._pending = False
        self._observer.close()

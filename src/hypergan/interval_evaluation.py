"""Bounded asynchronous snapshot evaluation, supervised outside the training loop."""
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
import tempfile
import time
import uuid

from .cpu_worker_service import CPUServiceCancelled
from .metric_evaluation import _evaluate_pinned, _write_catalog, _recover_abandoned
from .metric_plugins import enabled_custom
from .metrics import metric_catalog
from .run_state import atomic_json


class EvaluationWorker:
    """One immutable snapshot in flight, with no live numerical state in the thread."""
    def __init__(self):
        self.future = None

        self.cancel = Event()

    @property
    def busy(self):
        return self.future is not None

    def submit(self, root, selected, metric_id, identity, snapshot):
        if self.busy:
            raise RuntimeError('Snapshot evaluator is busy')
        self.cancel.clear()
        self.future = Future()
        self.source = dict(identity, metric_id=metric_id)
        self.deadline = time.monotonic() + selected['metrics']['custom'][metric_id]['timeout']
        future = self.future

        def evaluate():
            temporary = snapshot.get('temporary')
            directory = None
            try:
                if temporary is None:
                    temporary = tempfile.TemporaryDirectory(prefix='.evaluation-', dir=root)
                    from .preview_snapshot import write_snapshot
                    try:
                        descriptor = write_snapshot(snapshot.pop('state'), Path(temporary.name) / 'snapshot.pt',
                            cancellation_event=self.cancel, deadline=self.deadline)
                    except RuntimeError as error:
                        if self.cancel.is_set() and str(error) == 'Preview cancelled during snapshot persistence':
                            raise CPUServiceCancelled('Evaluation cancelled during snapshot persistence') from error
                        raise
                else:
                    descriptor = snapshot['descriptor']
                if self.cancel.is_set():
                    raise CPUServiceCancelled('Evaluation cancelled before worker startup')
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Evaluation snapshot persistence deadline exceeded')
                directory = Path(root) / 'metrics' / 'evaluations' / identity['evaluation_id']
                directory.mkdir(parents=True, exist_ok=False)
                revision = _write_catalog(Path(root), metric_catalog(selected))
                atomic_json(directory / 'receipt.json', {'schema_version': 1, 'status': 'running',
                    **identity, 'snapshot_sha256': descriptor['sha256'], 'metric_id': metric_id,
                    'catalog': revision})
                # The numerical protocol retains configured timeout. The service
                # receives only the remaining wall-time budget after persistence.
                result = _evaluate_pinned(Path(root), selected, metric_id, directory,
                    Path(temporary.name) / 'snapshot.pt', descriptor['sha256'], identity,
                    cancellation_event=self.cancel, timeout=remaining)
            except BaseException as error:
                failure = error
            else:
                failure = None
            try:
                if temporary is not None:
                    temporary.cleanup()
            except BaseException as error:
                if failure is None:
                    failure = error
                elif hasattr(failure, 'add_note'):
                    failure.add_note(f'Evaluation snapshot cleanup also failed: {error}')
            if failure is not None:
                future.set_exception(failure)
            else:
                future.set_result(result)

        self.thread = Thread(target=evaluate, name='hypergan-evaluation-supervisor', daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.future = None
            temporary = snapshot.get('temporary')
            if temporary is not None:
                temporary.cleanup()
            raise

    def poll(self):
        if self.future is None:
            return None
        if not self.future.done():
            if time.monotonic() > self.deadline + 8:
                self.cancel.set()
                raise TimeoutError('Evaluation exceeded its deadline and cleanup grace')
            return None
        future, self.future = self.future, None
        return future.result()

    def abort(self, *, preserve_outcome=False):
        if self.future is None:
            return
        self.cancel.set()
        self.thread.join(8)
        if self.thread.is_alive():
            raise RuntimeError('Evaluation dispatcher has not completed worker cleanup')
        future, self.future = self.future, None
        if preserve_outcome:
            return future.result()


class IntervalEvaluations:
    """Attempt-scoped cadence; recovery starts strictly after the restored step."""
    def __init__(self, config, root, manifest, execution, emit):
        self.config, self.root, self.manifest = config, Path(root), manifest
        self.execution, self.emit = execution, emit
        self.specs = {key: value for key, value in enabled_custom(config).items()
                      if value['mode'] == 'snapshot' and value['trigger'] == 'interval'}
        self.worker = EvaluationWorker()
        self.disabled = set()
        self.rotation = 0

    def start(self):
        _recover_abandoned(self.root, self.manifest['run_id'])
        _clean_abandoned_snapshots(self.root)
        if self.specs and not callable(getattr(self.execution, 'evaluation_snapshot', None)):
            raise ValueError('Execution adapter does not support interval evaluation snapshots')
        step = self.manifest['steps']
        self.manifest['evaluation_schedule'] = {
            key: {'status': 'pending', 'next_step': (step // spec['every_steps'] + 1) * spec['every_steps'],
                  'skipped_busy': 0} for key, spec in self.specs.items()}

    def _event(self, key, status, step, **values):
        self.emit('evaluation_' + status, _observe=False, _step=step, metric_id=key, **values)

    def _failure(self, key, source, error):
        record = self.manifest['evaluation_schedule'][key]
        record.update(status='failed', source_step=source['source_step'], reason=str(error)[:1000])
        self._event(key, 'failed', source['source_step'], reason=record['reason'],
                    evaluation_id=source['evaluation_id'])
        if self.specs[key]['on_error'] == 'fail':
            raise RuntimeError(f'Interval metric {key} failed: {error}') from error
        self.disabled.add(key)
        record.update(status='disabled', next_step=None)

    def poll(self):
        if not self.worker.busy:
            return
        source = self.worker.source
        key = source['metric_id']
        try:
            receipt = self.worker.poll()
            if receipt is None:
                return
            if receipt['status'] == 'cancelled':
                self._cancelled(key, source)
                return
            if receipt['status'] != 'complete':
                raise RuntimeError(receipt['result']['error'])
        except CPUServiceCancelled:
            self.worker.abort()
            self._cancelled(key, source)
            return
        except Exception as error:
            # A persistence timeout may still own its slot; cleanup must finish
            # before disabling a metric or considering another submission.
            self.worker.abort()
            self._failure(key, source, error)
            return
        record = self.manifest['evaluation_schedule'][key]
        record.update(status='complete', source_step=source['source_step'],
                      evaluation_id=receipt['evaluation_id'])
        record.pop('reason', None)
        self._event(key, 'complete', source['source_step'], evaluation_id=receipt['evaluation_id'])

    def schedule(self):
        step = self.manifest['steps']
        keys = list(self.specs)
        if not keys:
            return
        self.poll()
        ordered = keys[self.rotation:] + keys[:self.rotation]
        for key in ordered:
            spec = self.specs[key]
            record = self.manifest['evaluation_schedule'][key]
            if key in self.disabled or step % spec['every_steps']:
                continue
            record['next_step'] = step + spec['every_steps']
            if self.worker.busy:
                record['skipped_busy'] += 1
                record.update(last_skipped_step=step, reason='worker_busy')
                if record['status'] != 'running':
                    record['status'] = 'skipped'
                self._event(key, 'skipped', step, reason='worker_busy')
                continue
            identity = {name: self.manifest[name] for name in ('run_id', 'attempt_id', 'attempt_index')}
            identity.update(evaluation_id=uuid.uuid4().hex, source_step=step)
            record.update(status='running', source_step=step, evaluation_id=identity['evaluation_id'])
            record.pop('reason', None)
            try:
                snapshot = self.execution.evaluation_snapshot(self.root, identity)
                selected = dict(self.config)
                selected['metrics'] = dict(self.config['metrics'], preset='none', custom={key: spec},
                                           disable=[], overrides={})
                self.worker.submit(self.root, selected, key, identity, snapshot)
            except Exception as error:
                from .run_controller import FatalExecutionError
                if isinstance(error, FatalExecutionError):
                    record.update(status='failed', reason=str(error)[:1000])
                    self._event(key, 'failed', step, reason=record['reason'], evaluation_id=identity['evaluation_id'])
                    raise
                self._failure(key, identity, error)
                continue
            self.rotation = (keys.index(key) + 1) % len(keys)
            self._event(key, 'scheduled', step, evaluation_id=identity['evaluation_id'])

    def _cancelled(self, key, source):
        self.manifest['evaluation_schedule'][key].update(status='cancelled', reason='attempt_stopped')
        self._event(key, 'cancelled', source['source_step'], evaluation_id=source['evaluation_id'])

    def close(self, *, cancel=False, stop_requested=lambda: False):
        while self.worker.busy:
            # A completed result or actual failure keeps its meaning even when
            # a signal arrived before the training loop collected the future.
            if self.worker.future.done():
                self.poll()
                continue
            if cancel or stop_requested():
                source = self.worker.source
                key = source['metric_id']
                try:
                    receipt = self.worker.abort(preserve_outcome=True)
                    if receipt['status'] == 'failed':
                        raise RuntimeError(receipt['result']['error'])
                except CPUServiceCancelled:
                    self._cancelled(key, source)
                except Exception as error:
                    # A failed reap still owns a live worker. Never recover its
                    # receipt or treat infrastructure cleanup as optional.
                    if self.worker.busy:
                        raise
                    self._failure(key, source, error)
                else:
                    if receipt['status'] == 'cancelled':
                        self._cancelled(key, source)
                    else:
                        record = self.manifest['evaluation_schedule'][key]
                        record.update(status='complete', source_step=source['source_step'],
                                      evaluation_id=receipt['evaluation_id'])
                        record.pop('reason', None)
                        self._event(key, 'complete', source['source_step'], evaluation_id=receipt['evaluation_id'])
                break
            self.poll()
            if self.worker.busy:
                time.sleep(.01)
        _recover_abandoned(self.root, self.manifest['run_id'])


def _clean_abandoned_snapshots(root):
    """Only the run-lock owner may reclaim these reserved transport directories."""
    parents = [root]
    attempts = root / 'attempts'
    if attempts.is_dir():
        parents.extend(path for path in attempts.iterdir() if path.is_dir() and not path.is_symlink())
    for parent in parents:
        for path in parent.glob('.evaluation-*'):
            if path.is_symlink() or not path.is_dir():
                raise ValueError('Abandoned evaluation transport must be an ordinary directory')
            children = list(path.iterdir())
            if any(child.name != 'snapshot.pt' or child.is_symlink() or not child.is_file() for child in children):
                raise ValueError('Abandoned evaluation transport contains unexpected files')
            for child in children:
                child.unlink()
            path.rmdir()

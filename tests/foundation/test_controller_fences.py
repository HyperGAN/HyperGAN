"""Attempt identity, execution routing and fatal group failures without torch."""
from copy import deepcopy
import json

import pytest

from hypergan.config import write_default
from hypergan.run_controller import (
    CompletedUpdate, ExecutionInfo, FatalExecutionError, ObserverError, Restored, run_resume, run_train,
)
from hypergan.run_requests import checkpoint_request_status, submit_checkpoint_request


class FencedExecution:
    def __init__(self, config, root, options, trace):
        self.root, self.options, self.trace = root, options, trace
        self.context = None
        self.step = self.count = 0
        self.closed = False

    @staticmethod
    def environment():
        return {'runtime': {'runtime_checked': False}, 'source': {}}

    def configure_attempt(self, context, *, preview_every, on_event):
        self.context = context
        self.trace.append(('configure', context.attempt_id))
        assert not context.attempt_dir.exists()
        if self.options.get('reject_controls') and (preview_every or on_event):
            raise ValueError('Observers unsupported by this bounded adapter')
        return {'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2,
                              'accumulation_steps': self.options.get('accumulation', 1)},
                'service_policy': {'command_timeout': self.options.get('timeout', 30)}}

    def restore(self, run_dir, checkpoint, run_id, config_sha256):
        if self.context is not None:
            assert self.context.run_id == run_id
            assert not self.context.attempt_dir.exists()
        self.trace.append(('restore', self.context.attempt_id if self.context else None))
        target = checkpoint or json.loads((run_dir / 'manifest.json').read_text())['checkpoint_path']
        from pathlib import Path
        target = Path(target)
        self.step = json.loads((target / 'manifest.json').read_text())['step']
        if self.options.get('fail_restore'):
            raise ValueError('Strict worker restore refused the checkpoint')
        return Restored(target, self.step)

    def start(self):
        if self.context is not None:
            assert self.context.attempt_dir.is_dir()
        self.trace.append(('start', self.context.attempt_id if self.context else None))
        return ExecutionInfo(self.step, {'dataset': 'fixture'}, [], {},
                             environment={'runtime': {'runtime_checked': True}, 'source': {'fixture': 'actual'}})

    def update(self):
        self.step += 1
        if self.options.get('manual') and self.step == 1:
            submit_checkpoint_request(self.root, run_id=self.context.run_id,
                                      attempt_id=self.context.attempt_id, request_id='manual-one')
        return CompletedUpdate(self.step, {'loss': 0.25})

    def checkpoint(self, run_dir, metadata):
        if metadata.get('request_ids') and self.options.get('save_failure'):
            failure = FatalExecutionError if self.options['save_failure'] == 'fatal' else OSError
            raise failure('Checkpoint worker group or volume failed')
        self.count += 1
        path = run_dir / 'fixture-checkpoints' / f"{metadata['attempt_id']}-{self.step}-{self.count}"
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text(json.dumps(dict(metadata, step=self.step)))
        return path

    def preview(self, *args, **kwargs):
        failure = FatalExecutionError if self.options.get('preview_failure') == 'fatal' else OSError
        raise failure('Preview worker or renderer failed')

    @property
    def inference_available(self):
        return False

    def observe(self, callback, event):
        callback(event)

    def shutdown(self):
        if not self.closed:
            self.trace.append(('shutdown', self.context.attempt_id if self.context else None))
            self.closed = True


class NativeExecution(FencedExecution):
    configure_attempt = None


def setup(tmp_path, *, native=False, **options):
    path = write_default(tmp_path / 'config', device="cpu")
    root, trace, instances = tmp_path / 'run', [], []
    def factory(config):
        execution = (NativeExecution if native else FencedExecution)(config, root, options, trace)
        instances.append(execution)
        return execution
    factory.environment = FencedExecution.environment
    return path, root, factory, options, trace, instances


def tree(root):
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in root.rglob('*')}


def test_candidate_identity_survives_restore_and_mutable_policy_change(tmp_path):
    path, root, factory, options, trace, instances = setup(tmp_path)
    first = run_train(path, root, steps=3, stop_after_steps=1, execution_factory=factory)
    identity = deepcopy(first['execution'])
    options['timeout'] = 90
    second = run_resume(root, stop_after_steps=1, execution_factory=factory)
    candidate = instances[-1].context
    assert candidate.attempt_id == second['attempt_id']
    assert candidate.attempt_index == second['attempt_index'] == 2
    assert second['execution'] == identity
    assert second['service_policy'] == {'command_timeout': 90}
    assert ('restore', candidate.attempt_id) in trace and ('start', candidate.attempt_id) in trace
    assert second['runtime'] == {'runtime_checked': True}
    assert second['source'] == {'fixture': 'actual'}
    assert second['qualification']['status'] == 'unqualified'
    assert 'rng_streams' not in second


@pytest.mark.parametrize('failure', ['restore', 'configure', 'identity', 'identity-type', 'identity-float'])
def test_failed_resume_does_not_change_any_run_path(tmp_path, failure):
    path, root, factory, options, trace, instances = setup(tmp_path)
    run_train(path, root, steps=3, stop_after_steps=1, execution_factory=factory)
    before = tree(root)
    if failure == 'restore':
        options['fail_restore'] = True
    elif failure == 'configure':
        options['reject_controls'] = True
    else:
        options['accumulation'] = True if failure == 'identity-type' else 1.0 if failure == 'identity-float' else 2
    trace.clear()
    with pytest.raises(ValueError):
        run_resume(root, execution_factory=factory, preview_every=1 if failure == 'configure' else 0)
    assert tree(root) == before
    assert instances[-1].closed
    assert not instances[-1].context.attempt_dir.exists()
    assert sum(name == 'shutdown' for name, _ in trace) == 1
    assert not any(name == 'start' for name, _ in trace)
    if failure != 'restore':
        assert not any(name == 'restore' for name, _ in trace)


@pytest.mark.parametrize('native_first', [False, True])
def test_native_and_explicit_execution_cannot_cross_resume_routes(tmp_path, native_first):
    path, root, first_factory, _, _, _ = setup(tmp_path, native=native_first)
    run_train(path, root, steps=3, stop_after_steps=1, execution_factory=first_factory)
    before = tree(root)
    instances = []
    def other(config):
        instance = (FencedExecution if native_first else NativeExecution)(config, root, {}, [])
        instances.append(instance)
        return instance
    with pytest.raises(ValueError, match='numerical execution identity differs'):
        run_resume(root, execution_factory=other)
    assert tree(root) == before and instances[0].closed


def test_historical_native_manifest_without_execution_stays_native(tmp_path):
    path, root, factory, _, _, _ = setup(tmp_path, native=True)
    first = run_train(path, root, steps=2, stop_after_steps=1, execution_factory=factory)
    assert 'execution' not in first and 'service_policy' not in first and 'rng_streams' in first
    second = run_resume(root, execution_factory=factory)
    assert second['status'] == 'complete' and second['attempt_index'] == 2
    assert 'execution' not in second


@pytest.mark.parametrize('option', ['preview', 'callback'])
def test_unsupported_observers_rejected_before_new_run_artifacts(tmp_path, option):
    path, root, factory, _, _, instances = setup(tmp_path, reject_controls=True)
    with pytest.raises(ValueError, match='Observers unsupported'):
        run_train(path, root, execution_factory=factory,
                  preview_every=1 if option == 'preview' else 0,
                  on_event=(lambda row: None) if option == 'callback' else None)
    assert not root.exists() and instances[0].closed


def test_adapter_environment_failure_closes_configured_adapter_without_artifacts(tmp_path, monkeypatch):
    path, root, factory, _, _, instances = setup(tmp_path)
    def failed_environment():
        raise OSError('Runtime environment could not be inspected')
    monkeypatch.setattr(FencedExecution, "environment", staticmethod(failed_environment))
    with pytest.raises(OSError, match='environment could not be inspected'):
        run_train(path, root, execution_factory=factory)
    assert not root.exists() and instances[0].closed


@pytest.mark.parametrize('fatal', [False, True])
def test_manual_save_is_optional_only_while_execution_remains_usable(tmp_path, fatal):
    path, root, factory, _, _, instances = setup(tmp_path, manual=True, save_failure='fatal' if fatal else 'io')
    if fatal:
        with pytest.raises(FatalExecutionError):
            run_train(path, root, steps=2, execution_factory=factory)
    else:
        run_train(path, root, steps=2, execution_factory=factory)
    manifest = json.loads((root / 'manifest.json').read_text())
    receipt = checkpoint_request_status(root, 'manual-one')
    assert manifest['status'] == ('failed' if fatal else 'complete')
    assert receipt['status'] == ('pending' if fatal else 'rejected')
    assert instances[0].closed
    if fatal:
        # Fast updates may finish before the 250 ms request poll. The forced
        # terminal poll still executes the request before final checkpointing.
        assert 1 <= manifest['steps'] <= 2 and manifest['last_durable_step'] == 0
        assert not manifest['observation_errors']


@pytest.mark.parametrize('fatal', [False, True])
def test_preview_cannot_swallow_a_fatal_execution_failure(tmp_path, fatal):
    path, root, factory, _, _, _ = setup(tmp_path, preview_failure='fatal' if fatal else 'io')
    if fatal:
        with pytest.raises(FatalExecutionError):
            run_train(path, root, steps=2, preview_every=1, execution_factory=factory)
    else:
        run_train(path, root, steps=2, preview_every=1, execution_factory=factory)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == ('failed' if fatal else 'complete')
    assert bool(manifest['observation_errors']) is not fatal


@pytest.mark.parametrize('fatal', [False, True])
def test_supervised_progress_failure_is_recorded_without_recursive_delivery(tmp_path, fatal):
    path, root, factory, _, _, instances = setup(tmp_path)
    original = factory
    delivered = []
    def supervised(config):
        execution = original(config)
        def observe(callback, event):
            delivered.append(event['event'])
            if event['event'] == 'train':
                failure = FatalExecutionError if fatal else ObserverError
                raise failure('Observer deadline exceeded')
        execution.observe = observe
        return execution
    supervised.environment = original.environment
    if fatal:
        with pytest.raises(FatalExecutionError, match='deadline'):
            run_train(path, root, steps=2, execution_factory=supervised, on_event=lambda _: None)
    else:
        run_train(path, root, steps=2, execution_factory=supervised, on_event=lambda _: None)
    manifest = json.loads((root / 'manifest.json').read_text())
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    assert manifest['status'] == ('failed' if fatal else 'complete')
    assert instances[0].closed
    assert 'observer_error' not in delivered
    assert [event['sequence'] for event in events] == list(range(1, len(events) + 1))
    assert bool(manifest['observation_errors']) is not fatal
    if not fatal:
        assert len(manifest['observation_errors']) == 2
        assert all(error['source'] == 'progress' for error in manifest['observation_errors'])
        assert sum(event['event'] == 'observer_error' for event in events) == 2


def test_unclassified_observer_execution_failure_remains_fatal(tmp_path):
    path, root, factory, _, _, instances = setup(tmp_path)
    original = factory
    def broken_restore(config):
        execution = original(config)
        def observe(callback, event):
            if event['event'] == 'train':
                raise RuntimeError('Numerical observer state restoration failed')
        execution.observe = observe
        return execution
    broken_restore.environment = original.environment
    with pytest.raises(RuntimeError, match='restoration failed'):
        run_train(path, root, steps=3, execution_factory=broken_restore, on_event=lambda _: None)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['steps'] == 1 and manifest['last_durable_step'] == 0
    assert manifest['possible_lost_steps'] == 1 and not manifest['observation_errors']
    assert instances[0].closed


def test_training_continues_while_event_disk_is_stalled_and_reports_queue_loss(tmp_path, monkeypatch):
    from threading import Event
    import hypergan.run_controller as controller
    from hypergan.observation_io import ObservationIO
    from hypergan.run_events import read_event_page
    from hypergan.run_state import EventJournal, validate_event_boundary

    entered, release = Event(), Event()
    append = EventJournal.append
    def slow_append(self, event):
        if event['event'] == 'train':
            entered.set()
            assert release.wait(5)
        return append(self, event)
    class SmallQueue(ObservationIO):
        def __init__(self, run_dir, attempt_dir):
            super().__init__(run_dir, attempt_dir, capacity=2)
    monkeypatch.setattr(EventJournal, 'append', slow_append)
    monkeypatch.setattr(controller, 'ObservationIO', SmallQueue)
    path, root, factory, _, _, instances = setup(tmp_path)
    original = factory
    def execution_factory(config):
        execution = original(config)
        update = execution.update
        def advance():
            if execution.step == 1:
                assert entered.wait(5)
            value = update()
            if value.step == 10:
                release.set()
            return value
        execution.update = advance
        return execution
    execution_factory.environment = original.environment
    try:
        result = run_train(path, root, steps=10, execution_factory=execution_factory)
    finally:
        release.set()
    assert instances[0].step == 10
    assert result['dropped_train_events'] >= 1
    events = read_event_page(root)['events']
    assert [event['sequence'] for event in events] == list(range(1, len(events) + 1))
    assert sum(event.get('observation_gap', {}).get('dropped_train_events', 0) for event in events) == result['dropped_train_events']
    checkpoint = json.loads((__import__('pathlib').Path(result['checkpoint_path']) / 'manifest.json').read_text())
    assert checkpoint['step'] == checkpoint['event_boundary']['step'] == 10
    validate_event_boundary(root, checkpoint)


@pytest.mark.parametrize('required_failure', [False, True])
def test_final_delayed_metrics_keep_source_step_and_current_checkpoint_frontier(tmp_path, monkeypatch, required_failure):
    import hypergan.run_controller as controller
    from pathlib import Path
    from hypergan.run_events import read_event_page
    from hypergan.run_state import validate_event_boundary

    class DelayedMetrics:
        def __init__(self, config):
            self.source = None
        def start(self):
            pass
        def evaluate(self, values, context):
            if self.source is None:
                self.source = dict(context)
            return {}, {}
        def poll(self):
            return []
        def close(self, *, drain=True, stop_requested=None):
            if not drain:
                return []
            outcome = {'context': self.source, 'metrics': {'loss/g_total': 1.5},
                       'measurement_status': {}}
            if required_failure:
                error = RuntimeError('Required metric delayed failed at terminal drain')
                outcome.update(metrics={}, measurement_status={'loss/g_total': {'status': 'failed'}})
                error.metric_outcomes = [outcome]
                raise error
            return [outcome]

    monkeypatch.setattr(controller, 'ScalarMetrics', DelayedMetrics)
    path, root, factory, _, _, instances = setup(tmp_path)
    if required_failure:
        with pytest.raises(RuntimeError, match='terminal drain'):
            run_train(path, root, steps=2, execution_factory=factory)
        manifest = json.loads((root / 'manifest.json').read_text())
        assert manifest['status'] == 'failed' and manifest['last_durable_step'] == 0
        assert instances[0].closed
    else:
        manifest = run_train(path, root, steps=2, checkpoint_every=2, execution_factory=factory)
        # The final periodic save preceded metric completion. Refresh its event
        # boundary with a second same-step save without changing trainer state.
        assert instances[0].count == 3
        info = json.loads((Path(manifest['checkpoint_path']) / 'manifest.json').read_text())
        validate_event_boundary(root, info)
        prefix = (root / 'events.jsonl').read_bytes()[:info['event_boundary']['offset']]
        saved_events = [json.loads(line) for line in prefix.splitlines()]
        assert saved_events[-1]['step'] == info['step'] == 2
        assert any(event['event'] == 'metric' and event['step'] == 1 for event in saved_events)
    events = read_event_page(root)['events']
    measured = [event for event in events if event['event'] == 'metric']
    assert len(measured) == 1 and measured[0]['step'] == 1
    assert [event['sequence'] for event in events] == list(range(1, len(events) + 1))

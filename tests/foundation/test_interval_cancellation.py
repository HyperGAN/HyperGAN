"""Signals preserve completed snapshot outcomes and publish real cancellation."""
from concurrent.futures import Future
import json
from threading import Thread
import time
from types import SimpleNamespace

import pytest

from hypergan.config import resolve_config
from hypergan.cpu_worker_service import CPUServiceCancelled
from hypergan.interval_evaluation import IntervalEvaluations
from hypergan.metric_evaluation import _evaluate_pinned, _recover_abandoned


@pytest.mark.parametrize('outcome', ['complete', 'failed', 'cancelled'])
@pytest.mark.parametrize('already_done', [False, True])
@pytest.mark.parametrize('on_error', ['fail', 'disable'])
def test_signal_preserves_worker_outcome(tmp_path, outcome, already_done, on_error):
    config = resolve_config({'metrics': {'custom': {'fid': {
        'factory': 'uninstalled.metric:FID', 'mode': 'snapshot', 'trigger': 'interval',
        'every_steps': 10, 'on_error': on_error,
        'inputs': {'generated': 'evaluation.generated'},
        'evaluation': {'device': 'cpu', 'sample_count': 8, 'batch_size': 4, 'seed': 1,
                       'data': {'factory': 'gaussian_grid', 'args': {}}}}}}})
    manifest = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 0, 'steps': 10}
    events = []
    scheduler = IntervalEvaluations(config, tmp_path, manifest,
        SimpleNamespace(evaluation_snapshot=lambda *args: None),
        lambda event, **values: events.append((event, values)))
    scheduler.start()
    worker = scheduler.worker
    worker.source = {'metric_id': 'fid', 'source_step': 10, 'evaluation_id': 'a' * 32}
    worker.future = Future()
    worker.deadline = time.monotonic() + 10
    future = worker.future
    def finish():
        if not already_done:
            assert worker.cancel.wait(5)
        if outcome == 'complete':
            future.set_result({'status': 'complete', 'evaluation_id': 'a' * 32})
        elif outcome == 'cancelled':
            future.set_exception(CPUServiceCancelled('explicit cancellation'))
        else:
            future.set_exception(ValueError('real evaluator failure'))
    worker.thread = Thread(target=finish)
    worker.thread.start()
    if already_done:
        worker.thread.join(5)
    if outcome == 'failed' and on_error == 'fail':
        with pytest.raises(RuntimeError, match='real evaluator failure'):
            scheduler.close(stop_requested=lambda: True)
    else:
        scheduler.close(stop_requested=lambda: True)
    assert not worker.busy
    assert not worker.thread.is_alive()
    assert events[-1][0] == 'evaluation_' + outcome
    status = manifest['evaluation_schedule']['fid']['status']
    assert status == ('disabled' if outcome == 'failed' and on_error == 'disable' else outcome)


@pytest.mark.parametrize('on_error', ['disable', 'fail'])
def test_cancelled_worker_publishes_cancelled_receipt_and_recovers_registration(tmp_path, monkeypatch, on_error):
    from hypergan import cpu_worker_service, metric_evaluation
    class CancelledService:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            raise CPUServiceCancelled('owner stopped the attempt')
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(cpu_worker_service, 'CPUWorkerService', CancelledService)
    monkeypatch.setattr(metric_evaluation, 'metric_catalog', lambda _: {
        'schema_version': 1, 'metrics': {'fid': {'kind': 'scalar', 'definition_hash': 'b' * 64}}})
    evaluation_id = 'a' * 32
    directory = tmp_path / 'metrics' / 'evaluations' / evaluation_id
    directory.mkdir(parents=True)
    selected = {'metrics': {'custom': {'fid': {'timeout': 10, 'on_error': on_error}}},
                '_metric_runtime': {'fid': {}}}
    receipt = _evaluate_pinned(tmp_path, selected, 'fid', directory, directory / 'snapshot.pt',
        'c' * 64, {'run_id': 'run', 'evaluation_id': evaluation_id})
    event = json.loads((directory / 'events.jsonl').read_text())
    assert receipt['status'] == event['status'] == 'cancelled'
    assert event['measurement_status']['fid']['status'] == 'cancelled'
    assert event['metrics'] == {}
    (directory / 'stream.json').unlink()
    _recover_abandoned(tmp_path, 'run')
    assert (directory / 'stream.json').is_file()
    assert json.loads((directory / 'receipt.json').read_text())['status'] == 'cancelled'


def test_failed_reap_is_not_optional_or_reconciled_as_abandoned(tmp_path, monkeypatch):
    from hypergan import interval_evaluation
    scheduler = object.__new__(IntervalEvaluations)
    worker = SimpleNamespace(busy=True, future=Future(),
        source={'metric_id': 'fid', 'source_step': 10, 'evaluation_id': 'a' * 32})
    def abort(**kwargs):
        raise RuntimeError('worker cleanup is still in progress')
    worker.abort = abort
    scheduler.worker = worker
    scheduler.root = tmp_path
    scheduler.manifest = {'run_id': 'run'}
    monkeypatch.setattr(interval_evaluation, '_recover_abandoned',
        lambda *args: pytest.fail('must not reconcile a potentially active evaluator'))
    with pytest.raises(RuntimeError, match='cleanup is still in progress'):
        scheduler.close(cancel=True)

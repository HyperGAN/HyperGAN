"""Custom factories are structurally validated without importing their code."""
from copy import deepcopy
import subprocess
import sys
import threading
import time

import pytest

from hypergan.config import fingerprint, observation_fingerprint, resolve_config


def scalar():
    return {'factory': 'uninstalled.metrics:Ratio', 'inputs': {'numerator': 'update.g_loss', 'denominator': 'update.d_loss'}}


def snapshot():
    return {'factory': 'uninstalled.metrics:Color', 'mode': 'snapshot', 'trigger': 'manual',
            'inputs': {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'},
            'evaluation': {'data': {'factory': 'gaussian_grid', 'args': {}},
                           'sample_count': 8, 'batch_size': 4, 'seed': 5}}


def test_custom_structural_configuration_never_imports_factory():
    config = resolve_config({'metrics': {'custom': {'ratio': scalar(), 'color': snapshot()}}})
    assert 'uninstalled.metrics' not in sys.modules
    assert config['metrics']['custom']['color']['evaluation']['device'] == 'cuda'
    assert fingerprint(config) == fingerprint(resolve_config({}))
    code = 'import sys; from hypergan.config import resolve_config; resolve_config({"metrics":{"custom":{"custom":{"factory":"nonexistent.module:C","inputs":{"value":"update.g_loss"}}}}}); assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('changes', [
    {'factory': 'missingcolon'}, {'inputs': {'tensor': 'trainer.latent'}}, {'inputs': {}},
    {'inputs': {'context': 'update.g_loss'}}, {'mode': 'stream-stateful'}, {'every_steps': 0},
    {'timeout': float('inf')}, {'on_error': 'ignore'}, {'args': {'bad': float('nan')}},
    {'unknown': True}, {'evaluation': {}}, {'every': 2}, {'on_busy': 'skip'},
])
def test_invalid_scalar_modes_and_bindings_fail(changes):
    value = scalar()
    value.update(changes)
    with pytest.raises(ValueError):
        resolve_config({'metrics': {'custom': {'ratio': value}}})


def test_snapshot_rejects_unknown_schedules_and_requires_evaluation_data():
    for changes in ({'trigger': 'scheduled'}, {'every_steps': 2}, {'evaluation': {}}, {'inputs': {'x': 'update.g_loss'}}):
        spec = snapshot()
        spec.update(changes)
        with pytest.raises(ValueError):
            resolve_config({'metrics': {'custom': {'color': spec}}})


def test_custom_cannot_replace_builtin_but_can_be_disabled():
    with pytest.raises(ValueError, match='replace'):
        resolve_config({'metrics': {'custom': {'loss/total': scalar()}}})
    config = resolve_config({'metrics': {'preset': 'none', 'custom': {'ratio': scalar()}, 'disable': ['ratio']}})
    from hypergan.metric_plugins import prepare_custom
    prepare_custom(config)
    assert config['_metric_runtime'] == {}


def test_manual_only_catalog_does_not_claim_sampled_training_values():
    from hypergan.metrics import metric_catalog, select_metrics
    config=resolve_config({'metrics':{'preset':'none','custom':{'color':snapshot()}}})
    config['_metric_runtime']={'color':{'descriptor':{'kind':'scalar'},'factory_sources':{'module':'a'*64},'protocol':'hypergan-metric/v1'}}
    assert select_metrics(config,metric_catalog(config),{},1,.1)==({}, {}, 'disabled')


def _async_metrics(monkeypatch, invoke, *, on_error='disable', count=1):
    from hypergan import metric_plugins
    monkeypatch.setattr(metric_plugins, 'invoke', invoke)
    config = resolve_config({'metrics': {'custom': {
        f'ratio{i}': dict(scalar(), on_error=on_error) for i in range(count)}}})
    config['_metric_runtime'] = {name: {} for name in config['metrics']['custom']}
    metrics = metric_plugins.ScalarMetrics(config)
    metrics.start()
    return metrics


def test_slow_metric_never_blocks_submit_poll_or_backpressure(monkeypatch):
    running, release = threading.Event(), threading.Event()
    def invoke(spec, operation, **kwargs):
        running.set()
        assert release.wait(5)
        return {'value': 3}
    metrics = _async_metrics(monkeypatch, invoke)
    try:
        assert metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 1})[1] == {'ratio0': {'status': 'queued'}}
        assert running.wait(2)
        for step in range(2, 1002):
            assert metrics.poll() == []
            assert metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': step})[1]['ratio0']['status'] == 'dropped'
        assert metrics._pending == {'ratio0'}
        assert metrics._jobs.qsize() == 0
    finally:
        release.set()
    assert metrics.close() == [{'context': {'step': 1}, 'metrics': {'ratio0': 3}, 'measurement_status': {}}]
    assert not metrics._thread.is_alive()


def test_async_failure_is_optional_or_required_at_terminal_drain(monkeypatch):
    def invoke(*args, **kwargs):
        raise ValueError('explicit observation failure')
    metrics = _async_metrics(monkeypatch, invoke)
    metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 7})
    outcome, = metrics.close()
    assert outcome['context']['step'] == 7
    assert outcome['measurement_status']['ratio0']['status'] == 'disabled'
    metrics = _async_metrics(monkeypatch, invoke, on_error='fail')
    metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 8})
    with pytest.raises(RuntimeError, match='Required metric ratio0 failed') as error:
        metrics.close()
    assert error.value.metric_outcomes[0]['context']['step'] == 8


def test_all_metric_slots_and_results_are_bounded_and_cancel_reaps_dispatcher(monkeypatch):
    running = threading.Event()
    calls = []
    def invoke(spec, operation, *, cancellation_event, **kwargs):
        calls.append(kwargs)
        running.set()
        assert cancellation_event.wait(5)
        raise RuntimeError('cancelled')
    metrics = _async_metrics(monkeypatch, invoke, count=32)
    metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 1})
    assert running.wait(2)
    assert len(metrics._pending) == 32
    assert metrics._jobs.qsize() == 31
    statuses = metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 2})[1]
    assert all(row['status'] == 'dropped' for row in statuses.values())
    assert metrics.close(drain=False) == []
    assert len(calls) == 1
    assert metrics._results.qsize() == 32
    assert not metrics._thread.is_alive()


def test_queue_wait_is_included_in_metric_deadline(monkeypatch):
    def invoke(spec, operation, **kwargs):
        time.sleep(.03)
        return {'value': 1}
    metrics = _async_metrics(monkeypatch, invoke, count=2)
    metrics.specs['ratio1']['timeout'] = .01
    metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 1})
    first, second = metrics.close()
    assert first['metrics'] == {'ratio0': 1}
    assert 'expired while queued' in second['measurement_status']['ratio1']['reason']


def test_shutdown_cancellation_does_not_relabel_an_actual_worker_failure(monkeypatch):
    def invoke(spec, operation, *, cancellation_event, **kwargs):
        assert cancellation_event.wait(5)
        raise ValueError('actual worker failure during cancellation')
    metrics = _async_metrics(monkeypatch, invoke)
    metrics.evaluate({'g_loss': 6, 'd_loss': 2}, {'step': 1})
    # Give the dispatcher ownership before requesting cancellation, so this
    # exercises an actual in-flight error rather than a cancelled queued job.
    deadline=time.monotonic()+2
    while metrics._jobs.qsize() and time.monotonic()<deadline:
        time.sleep(.001)
    outcome, = metrics.close(stop_requested=lambda: True)
    assert outcome['measurement_status']['ratio0']['status']=='disabled'
    assert 'actual worker failure' in outcome['measurement_status']['ratio0']['reason']


def test_cancellation_does_not_discard_received_worker_failure():
    from hypergan.cpu_worker_service import CPUServiceCancelled, _receive
    cancel=threading.Event()
    cancel.set()
    class Channel:
        closed=False
        def pump(self):
            return [{'kind':'error','error':'actual worker failure'}]
    assert _receive(Channel(), cancellation_event=cancel)['error']=='actual worker failure'
    channel=Channel()
    channel.pump=lambda: []
    with pytest.raises(CPUServiceCancelled):
        _receive(channel, cancellation_event=cancel)


def interval_snapshot():
    spec = snapshot()
    spec.update(trigger='interval', every_steps=10000)
    spec['evaluation']['device'] = 'cuda:1'
    return spec


def test_interval_snapshot_configuration_is_explicit_and_observation_only():
    supplied = interval_snapshot()
    original = deepcopy(supplied)
    config = resolve_config({'metrics': {'custom': {'fid': supplied}}})
    spec = config['metrics']['custom']['fid']
    assert spec['every_steps'] == 10000 and spec['on_busy'] == 'skip'
    assert spec['evaluation']['device'] == 'cuda:1'
    assert supplied == original
    assert 'uninstalled.metrics' not in sys.modules
    changed = deepcopy(config)
    changed['metrics']['custom']['fid']['every_steps'] = 20000
    assert fingerprint(config) == fingerprint(changed) == fingerprint(resolve_config({}))
    assert observation_fingerprint(config) != observation_fingerprint(changed)


@pytest.mark.parametrize('changes, message', [
    ({'every_steps': None}, 'every_steps'), ({'every_steps': True}, 'every_steps'),
    ({'every_steps': 0}, 'every_steps'), ({'every_steps': -1}, 'every_steps'),
    ({'every_steps': 1.5}, 'every_steps'), ({'every_steps': '10000'}, 'every_steps'),
    ({'on_busy': 'queue'}, 'on_busy'), ({'on_busy': 'wait'}, 'on_busy'),
    ({'on_busy': None}, 'on_busy'), ({'every': 10}, 'fields'),
])
def test_interval_snapshot_rejects_invalid_cadence_and_busy_policy(changes, message):
    spec = interval_snapshot()
    spec.update(changes)
    with pytest.raises(ValueError, match=message):
        resolve_config({'metrics': {'custom': {'fid': spec}}})


@pytest.mark.parametrize('field', ['every_steps', 'device'])
def test_interval_snapshot_requires_cadence_and_explicit_device(field):
    spec = interval_snapshot()
    if field == 'device':
        del spec['evaluation']['device']
    else:
        del spec[field]
    with pytest.raises(ValueError, match=field):
        resolve_config({'metrics': {'custom': {'fid': spec}}})


@pytest.mark.parametrize('changes', [{'every_steps': 10000}, {'on_busy': 'skip'}])
def test_manual_snapshot_rejects_interval_options(changes):
    spec = snapshot()
    spec.update(changes)
    with pytest.raises(ValueError, match='manual snapshot'):
        resolve_config({'metrics': {'custom': {'fid': spec}}})


def test_interval_cpu_fixture_is_explicit():
    spec = interval_snapshot()
    spec['evaluation']['device'] = 'cpu'
    config = resolve_config({'metrics': {'custom': {'fid': spec}}})
    assert config['metrics']['custom']['fid']['evaluation']['device'] == 'cpu'

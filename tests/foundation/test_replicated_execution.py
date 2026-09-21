"""Parent adapter validation without optional numerical dependencies."""
import copy
import json
import subprocess
import sys

import pytest

from hypergan.config import config_values, fingerprint, resolve_config, write_default
from hypergan.replicated_execution import ReplicatedExecution, run_train
from hypergan.run_controller import AttemptContext, FatalExecutionError


PROFILE = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo'}}


def test_parent_import_is_numerical_dependency_free():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'particlegan', 'PIL'):
            raise AssertionError('Parent imported numerical dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import hypergan.replicated_execution
'''
    subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', code],
                   check=True, capture_output=True, text=True)


@pytest.mark.parametrize('kwargs', [{'service_policy': {'preview_timeout': False}}, {'on_event': lambda _: None},
    {'service_policy': {'command_timeout': True}}, {'service_policy': {'collective_timeout': 1000}},
    {'profile': {'schema_version': 1, 'execution': {'name': 'cpu-single'}}}])
def test_unsupported_controls_fail_before_run_mutation(tmp_path, kwargs):
    config = write_default(tmp_path / 'project', device="cpu")
    with pytest.raises(ValueError):
        run_train(config, tmp_path / 'run', **{'profile': PROFILE, **kwargs})
    assert not (tmp_path / 'run').exists()


def test_sample_limit_is_actionable_before_worker_start():
    config = resolve_config({'sampling': {'count': 1025}})
    with pytest.raises(ValueError, match='sample count exceeds 1024'):
        ReplicatedExecution(config, PROFILE)


def test_attempt_identity_cannot_be_reconfigured(tmp_path):
    execution = ReplicatedExecution(resolve_config({}), PROFILE)
    context = AttemptContext('run', 'attempt', 1, tmp_path, tmp_path / 'attempts/attempt')
    descriptor = execution.configure_attempt(context, preview_every=0, on_event=None)
    assert descriptor['execution']['world_size'] == 2
    assert execution.service is None and not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='immutable'):
        execution.configure_attempt(context, preview_every=0, on_event=None)


class FakeService:
    def __init__(self, result):
        self.result, self.aborted = result, False
    def command(self, *args):
        return self.result
    def _abort_preserving(self, error):
        self.aborted = True


@pytest.mark.parametrize('lr_floor', [1.0, .05])
def test_restore_extended_steps_checks_manifest_before_starting_workers(tmp_path, monkeypatch, lr_floor):
    original = resolve_config({'training': {'steps': 4, 'lr_floor': lr_floor}})
    current = resolve_config({'training': {'steps': 8, 'lr_floor': lr_floor}})
    (tmp_path / 'manifest.json').write_text(json.dumps({'config': config_values(original)}))
    execution = ReplicatedExecution(current, PROFILE)
    execution.configure_attempt(AttemptContext('run', 'attempt', 1, tmp_path, tmp_path / 'attempts/attempt'),
                                preview_every=0, on_event=None)
    started = []
    def opened():
        started.append(True)
        execution.information = {'recovery_reasons': []}
    monkeypatch.setattr(execution, '_open', opened)
    restored = {'ready': True, 'step': 4, 'saved_step': 4,
                'inference_available': True, 'checkpoint_path': str(tmp_path / 'checkpoint')}
    execution.service = FakeService({'results': [restored, copy.deepcopy(restored)]})
    if lr_floor == 1:
        assert execution.restore(tmp_path, None, 'run', fingerprint(original)).step == 4
        assert started == [True]
    else:
        with pytest.raises(ValueError, match='configuration differs'):
            execution.restore(tmp_path, None, 'run', fingerprint(original))
        assert started == []


@pytest.mark.parametrize('metrics_preset', ['standard', 'none'])
@pytest.mark.parametrize('failure', ['step', 'readiness', 'metrics-disagree', 'nonfinite', 'batch-bool'])
def test_malformed_complete_update_poisons_execution(failure, metrics_preset):
    execution = ReplicatedExecution(resolve_config({'metrics': {'preset': metrics_preset}}), PROFILE)
    metrics = {'event': 'train', 'step': 1, 'global_batch_size': 16, 'local_batch_size': 8,
        'world_size': 2, 'd_loss': 0.2, 'g_loss': 0.3, 'd_adversarial': 0.2, 'd_adversarial_weighted': 0.2, 'g_adversarial_weighted': 0.1, 'g_adversarial': 0.1,
        'prior_loss': 0.2, 'gradient_penalty': 0.0, 'lr_scale': 1.0, 'objectives': []}
    row = {'ready': True, 'step': 1, 'inference_available': True, 'metrics': metrics}
    results = [copy.deepcopy(row), copy.deepcopy(row)]
    if failure == 'step':
        results[1]['step'] = 2
    elif failure == 'readiness':
        results[1]['ready'] = False
    elif failure == 'metrics-disagree':
        results[1]['metrics']['g_loss'] = 0.9
    elif failure == 'nonfinite':
        for item in results:
            item['metrics']['g_loss'] = float('inf')
    else:
        for item in results:
            item['metrics']['world_size'] = True
    execution.service = FakeService({'results': results})
    with pytest.raises(FatalExecutionError):
        execution.update()
    assert execution._poisoned and execution.service.aborted and execution.step == 0


class SnapshotService(FakeService):
    def __init__(self, failure=None):
        super().__init__(None)
        self.failure = failure
        self.calls = []

    def assert_healthy(self):
        if self.aborted:
            raise RuntimeError('worker group was aborted')

    def command(self, operation, payload):
        import hashlib
        from pathlib import Path
        self.calls.append((operation, payload))
        path = Path(payload['path'])
        path.write_bytes(b'frozen EMA state')
        descriptor = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        results = [{'ready': True, 'step': 3, 'inference_available': True} for _ in range(2)]
        results[0]['snapshot'] = descriptor
        if self.failure == 'step':
            results[1]['step'] = 4
        elif self.failure == 'descriptor':
            descriptor['bytes'] = True
        elif self.failure == 'missing':
            path.unlink()
        return {'results': results}


def _snapshot_execution(tmp_path, failure=None):
    execution = ReplicatedExecution(resolve_config({}), PROFILE)
    directory = tmp_path / 'attempts' / 'attempt'
    directory.mkdir(parents=True)
    execution.configure_attempt(AttemptContext('run', 'attempt', 1, tmp_path, directory),
                                preview_every=0, on_event=None)
    execution.service = SnapshotService(failure)
    execution.step = 3
    return execution, {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1, 'evaluation_id': 'eval-1'}


def test_evaluation_snapshot_hands_directory_ownership_to_scheduler(tmp_path):
    from pathlib import Path
    execution, identity = _snapshot_execution(tmp_path)
    result = execution.evaluation_snapshot(tmp_path, identity)
    directory = Path(result['temporary'].name)
    assert directory.parent == tmp_path / 'attempts' / 'attempt'
    assert directory.name.startswith('.evaluation-')
    assert result['descriptor']['bytes'] == len(b'frozen EMA state')
    assert execution.service.calls[0][0] == 'evaluation-snapshot'
    assert execution.service.calls[0][1]['identity'] == identity
    result['temporary'].cleanup()
    assert not directory.exists()


@pytest.mark.parametrize('failure', ['step', 'descriptor', 'missing'])
def test_failed_evaluation_snapshot_cleans_and_poisons_invalid_worker_result(tmp_path, failure):
    execution, identity = _snapshot_execution(tmp_path, failure)
    with pytest.raises(FatalExecutionError):
        execution.evaluation_snapshot(tmp_path, identity)
    assert execution._poisoned and execution.service.aborted
    assert not list((tmp_path / 'attempts' / 'attempt').iterdir())


@pytest.mark.parametrize('change', [{'run_id': 'other'}, {'attempt_id': 'other'},
    {'attempt_index': True}, {'evaluation_id': ''}, {'evaluation_id': None}])
def test_evaluation_snapshot_rejects_wrong_identity_before_worker_command(tmp_path, change):
    execution, identity = _snapshot_execution(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        execution.evaluation_snapshot(tmp_path, {**identity, **change})
    assert not execution.service.calls and not execution._poisoned
    assert not list((tmp_path / 'attempts' / 'attempt').iterdir())

"""Parent adapter validation without optional numerical dependencies."""
import copy
import subprocess
import sys

import pytest

from hypergan.config import resolve_config, write_default
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
    config = write_default(tmp_path / 'project')
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


@pytest.mark.parametrize('failure', ['step', 'readiness', 'metrics-disagree', 'nonfinite', 'batch-bool'])
def test_malformed_complete_update_poisons_execution(failure):
    execution = ReplicatedExecution(resolve_config({}), PROFILE)
    metrics = {'event': 'train', 'step': 1, 'global_batch_size': 16, 'local_batch_size': 8,
        'world_size': 2, 'd_loss': 0.2, 'g_loss': 0.3, 'g_adversarial': 0.1,
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

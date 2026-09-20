"""Actual CPU worker-group/controller interval evaluation and recovery acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys

from hypergan.config import write_default


METRIC = '''
import os
from pathlib import Path
import time
class Distance:
    def __init__(self, entered=None, release=None):
        self.entered, self.release = entered, release
    def describe(self):
        return {'kind': 'scalar', 'label': 'Fixture distance'}
    def evaluate(self, *, batches, context):
        import torch.distributed as dist
        assert not dist.is_initialized(), 'evaluation inherited a training group'
        if self.entered:
            Path(self.entered).write_text(str(os.getpid()))
            deadline = time.monotonic() + 30
            while not Path(self.release).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('training did not progress while evaluation was busy')
                time.sleep(.01)
        values = [(b['generated'] - b['reference']).square().mean().item() for b in batches]
        return sum(values) / len(values)
'''

DRIVER = '''
import json
from pathlib import Path
import sys
import time
from hypergan import run_controller
from hypergan.replicated_execution import ReplicatedExecution, run_train, run_resume

PROFILE = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo',
    'world_size': 2, 'accumulation_steps': 2}}

class ObservedExecution(ReplicatedExecution):
    def update(self):
        result = super().update()
        if result.step == 5:
            deadline = time.monotonic() + 30
            while not (root / 'entered').exists():
                assert time.monotonic() < deadline, 'isolated evaluator did not start'
                time.sleep(.01)
            (root / 'release').touch()
        return result

if __name__ == '__main__':
    root, mode = Path(sys.argv[1]), sys.argv[2]
    run_train(root / 'manual.toml', root / 'plain', profile=PROFILE, checkpoint_every=1)
    if mode == 'busy':
        result = run_controller.run_train(root / 'interval.toml', root / 'interval', checkpoint_every=1,
            execution_factory=lambda config: ObservedExecution(config, PROFILE))
        (root / 'results.json').write_text(json.dumps({'result': result}))
    else:
        stopped = run_train(root / 'interval.toml', root / 'interval', profile=PROFILE,
                            checkpoint_every=1, stop_after_steps=2)
        resumed = run_resume(root / 'interval')
        replayed = run_resume(root / 'interval', checkpoint=stopped['checkpoint_path'])
        (root / 'results.json').write_text(json.dumps({'stopped': stopped, 'resumed': resumed, 'replayed': replayed}))
    assert 'torch' not in sys.modules, 'replicated coordinator imported numerical runtime'
'''


def _run(tmp_path, mode):
    (tmp_path / 'replicated_interval_metric.py').write_text(METRIC)
    config = write_default(tmp_path / 'manual.toml', device='cpu')
    text = config.read_text().replace('num_particles = 20000', 'num_particles = 32').replace('count = 256', 'count = 4')
    spec = '''
[metrics.custom.distance]
factory = "replicated_interval_metric:Distance"
mode = "snapshot"
trigger = "manual"
on_error = "fail"
timeout = 40
inputs = {generated="evaluation.generated",reference="evaluation.reference"}
'''
    if mode == 'busy':
        spec += '[metrics.custom.distance.args]\n'
        spec += 'entered = ' + json.dumps(str(tmp_path / 'entered')) + '\n'
        spec += 'release = ' + json.dumps(str(tmp_path / 'release')) + '\n'
    spec += '''
[metrics.custom.distance.evaluation]
device = "cpu"
sample_count = 8
batch_size = 4
seed = 88
[metrics.custom.distance.evaluation.data]
factory = "gaussian_grid"
args = {side=10,noise=0.015}
'''
    config.write_text(text + spec)
    interval = 1 if mode == 'busy' else 2
    (tmp_path / 'interval.toml').write_text((text + spec).replace('trigger = "manual"',
        f'trigger = "interval"\nevery_steps = {interval}\non_busy = "skip"'))
    driver = tmp_path / 'replicated_interval_driver.py'
    driver.write_text(DRIVER)
    result = subprocess.run([sys.executable, str(driver), str(tmp_path), mode],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((tmp_path / 'results.json').read_text())


def _states(root, checkpoint=None):
    import torch
    if checkpoint is None:
        directory = root / 'distributed-checkpoints'
        latest = json.loads((directory / 'latest.json').read_text())
        checkpoint = directory / latest['checkpoint']
    else:
        checkpoint = Path(checkpoint)
    return [torch.load(checkpoint / f'rank-{rank:05d}.pt', weights_only=True) for rank in range(2)]


def _receipts(root):
    return [json.loads(path.read_text()) for path in (root / 'metrics/evaluations').glob('*/receipt.json')]


def _assert_clean_and_registered(root):
    assert not list(root.glob('.evaluation-*'))
    assert not list((root / 'attempts').glob('*/.evaluation-*'))
    for receipt_path in (root / 'metrics/evaluations').glob('*/receipt.json'):
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'complete'
        assert receipt['source_step'] == receipt['result']['step']
        assert receipt['attempt_id'] == receipt['result']['snapshot_identity']['attempt_id']
        event = json.loads(receipt_path.with_name('events.jsonl').read_text())
        assert event['step'] == receipt['source_step']
        assert event['attempt_id'] == receipt['attempt_id']
        assert set(event['metrics']) == {'distance'}
        assert receipt_path.with_name('stream.json').is_file()


def test_replicated_interval_evaluation_keeps_training_live_and_complete_state_equal(tmp_path):
    from hypergan.distributed_checkpoints import _digest
    result = _run(tmp_path, 'busy')['result']
    assert result['status'] == 'complete'
    assert result['evaluation_schedule']['distance']['skipped_busy'] >= 3
    measured = _receipts(tmp_path / 'interval')
    assert measured and min(item['source_step'] for item in measured) == 1
    assert _digest(_states(tmp_path / 'plain')) == _digest(_states(tmp_path / 'interval'))
    events = [json.loads(line) for line in (tmp_path / 'interval/events.jsonl').read_text().splitlines()]
    assert any(event['event'] == 'evaluation_skipped' and event['step'] == 4 for event in events)
    assert any(event['event'] == 'evaluation_complete' and event['step'] == 1 for event in events)
    _assert_clean_and_registered(tmp_path / 'interval')


def test_replicated_interval_resume_and_older_checkpoint_replay_preserve_state_and_attempts(tmp_path):
    from hypergan.distributed_checkpoints import _digest
    results = _run(tmp_path, 'recovery')
    measured = _receipts(tmp_path / 'interval')
    assert sorted(item['source_step'] for item in measured) == [2, 4, 4]
    assert {(item['attempt_id'], item['source_step']) for item in measured} == {
        (results['stopped']['attempt_id'], 2), (results['resumed']['attempt_id'], 4),
        (results['replayed']['attempt_id'], 4)}
    baseline = _digest(_states(tmp_path / 'plain'))
    assert baseline == _digest(_states(tmp_path / 'interval', results['resumed']['checkpoint_path']))
    assert baseline == _digest(_states(tmp_path / 'interval'))
    assert results['replayed']['evaluation_schedule']['distance']['next_step'] == 6
    assert results['replayed']['status'] == 'complete'
    _assert_clean_and_registered(tmp_path / 'interval')

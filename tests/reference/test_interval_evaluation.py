"""Real snapshot workers: nonblocking training, source provenance and recovery."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tests.config_fixtures import write_default
from hypergan.training import train, resume
from hypergan.checkpoints import read_checkpoint
from hypergan.distributed_checkpoints import _digest


METRIC = '''
import os
from pathlib import Path
import time
class Distance:
    def __init__(self, entered=None, release=None, fail=False):
        self.entered,self.release,self.fail=entered,release,fail
    def describe(self): return {'kind':'scalar','label':'Fixture distance'}
    def evaluate(self,*,batches,context):
        if self.entered:
            Path(self.entered).write_text(str(os.getpid()))
            deadline=time.monotonic()+30
            while not Path(self.release).exists():
                if time.monotonic()>deadline: raise TimeoutError('training did not release evaluation')
                time.sleep(.01)
        if self.fail: raise RuntimeError('intentional evaluation failure')
        values=[(b['generated']-b['reference']).square().mean().item() for b in batches]
        return sum(values)/len(values)
'''


def config(tmp_path, monkeypatch, *, interval=1, args=None, on_error='fail'):
    (tmp_path / 'interval_fixture.py').write_text(METRIC)
    monkeypatch.syspath_prepend(str(tmp_path))
    path = write_default(tmp_path / 'config.toml', device='cpu', defaults='hypergan')
    text = path.read_text().replace('num_particles = 20000', 'num_particles = 32').replace('count = 256', 'count = 4')
    spec = f'''
[metrics.custom.distance]
factory = "interval_fixture:Distance"
mode = "snapshot"
trigger = "interval"
every_steps = {interval}
on_busy = "skip"
on_error = "{on_error}"
timeout = 30
inputs = {{generated="evaluation.generated",reference="evaluation.reference"}}
'''
    if args:
        spec += '[metrics.custom.distance.args]\n'
        for key, value in args.items():
            spec += f'{key} = {json.dumps(value)}\n'
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
    path.write_text(text + spec)
    return path


def receipts(root):
    return [json.loads(path.read_text()) for path in (root / 'metrics/evaluations').glob('*/receipt.json')]


@pytest.mark.heavy
def test_interval_worker_allows_updates_while_busy_and_preserves_complete_state(tmp_path, monkeypatch):
    entered, release = tmp_path / 'entered', tmp_path / 'release'
    path = config(tmp_path, monkeypatch, args={'entered': str(entered), 'release': str(release)})
    plain = tmp_path / 'plain.toml'
    plain.write_text(path.read_text().replace('trigger = "interval"\nevery_steps = 1\non_busy = "skip"', 'trigger = "manual"'))
    train(plain, tmp_path / 'plain', checkpoint_every=1)

    def observe(row):
        if row['event'] == 'train' and row['step'] == 5:
            deadline = time.monotonic() + 20
            while not entered.exists():
                assert time.monotonic() < deadline, 'evaluator did not start'
                time.sleep(.01)
            release.touch()

    result = train(path, tmp_path / 'scheduled', checkpoint_every=1, on_event=observe)
    assert result['status'] == 'complete'
    assert result['evaluation_schedule']['distance']['skipped_busy'] >= 3
    measured = receipts(tmp_path / 'scheduled')
    assert measured and measured[0]['status'] == 'complete'
    assert min(item['result']['step'] for item in measured) == 1
    assert _digest(read_checkpoint(tmp_path / 'plain')[2]) == _digest(read_checkpoint(tmp_path / 'scheduled')[2])
    assert not list((tmp_path / 'scheduled').glob('.evaluation-*'))
    for receipt in measured:
        assert receipt['result']['snapshot_identity']['attempt_id'] == result['attempt_id']
        assert receipt['result']['step'] == receipt['source_step']


@pytest.mark.heavy
def test_interval_resume_and_older_snapshot_recovery_keep_attempt_positions(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch, interval=2)
    stopped = train(path, tmp_path / 'run', checkpoint_every=1, stop_after_steps=2)
    assert [item['result']['step'] for item in receipts(tmp_path / 'run')] == [2]
    first = stopped['attempt_id']
    resumed = resume(tmp_path / 'run')
    # Replay may prune the abandoned branch; capture its final state first.
    resumed_digest = _digest(read_checkpoint(tmp_path / 'run')[2])
    recovered = resume(tmp_path / 'run', checkpoint=stopped['checkpoint_path'])
    measured = receipts(tmp_path / 'run')
    assert sorted(item['result']['step'] for item in measured) == [2, 4, 4]
    assert {item['result']['snapshot_identity']['attempt_id'] for item in measured} == {
        first, resumed['attempt_id'], recovered['attempt_id']}
    assert resumed_digest == _digest(read_checkpoint(tmp_path / 'run')[2])
    assert recovered['evaluation_schedule']['distance']['next_step'] == 6


@pytest.mark.heavy
@pytest.mark.parametrize('policy', ['fail', 'disable'])
def test_interval_failure_is_visible_and_honors_policy(tmp_path, monkeypatch, policy):
    path = config(tmp_path, monkeypatch, interval=2, args={'fail': True}, on_error=policy)
    if policy == 'fail':
        with pytest.raises(RuntimeError, match='intentional evaluation failure'):
            train(path, tmp_path / 'run', checkpoint_every=1)
    else:
        assert train(path, tmp_path / 'run')['status'] == 'complete'
    manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert manifest['evaluation_schedule']['distance']['status'] == ('failed' if policy == 'fail' else 'disabled')
    assert receipts(tmp_path / 'run')[0]['status'] == 'failed'
    assert 'intentional evaluation failure' in receipts(tmp_path / 'run')[0]['result']['error']


@pytest.mark.parametrize('field,value', [('source_step', 3), ('attempt_id', 'other'),
                                        ('attempt_index', 2), ('evaluation_id', 'other')])
def test_interval_worker_rejects_mismatched_snapshot_source(tmp_path, field, value):
    import torch
    from hypergan.metric_evaluation import _sha256
    from hypergan.metric_evaluation_worker import evaluate_snapshot
    identity = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1,
                'evaluation_id': 'evaluation', 'source_step': 2}
    path = tmp_path / 'snapshot.pt'
    torch.save({'schema_version': 1, 'kind': 'ema-inference', 'step': 2, 'identity': identity}, path)
    requested = dict(identity, **{field: value})
    with pytest.raises(ValueError, match='requested source step or attempt identity'):
        evaluate_snapshot({'evaluation': {'seed': 1}}, {}, path, _sha256(path), requested)


@pytest.mark.heavy
@pytest.mark.skipif(os.name != 'posix', reason='POSIX signal and process-death recovery contract')
@pytest.mark.parametrize('abrupt', [False, True])
def test_signal_reaps_evaluator_and_recovers_abandoned_snapshot(tmp_path, monkeypatch, abrupt):
    entered, release = tmp_path / 'entered', tmp_path / 'release'
    path = config(tmp_path, monkeypatch, args={'entered': str(entered), 'release': str(release)})
    path.write_text(path.read_text().replace('timeout = 30', 'timeout = 3600'))
    driver = tmp_path / 'driver.py'
    driver.write_text('''from pathlib import Path
from hypergan.training import train
if __name__ == '__main__':
    root = Path(__file__).parent
    train(root/'config.toml', root/'run', checkpoint_every=1)
''')
    process = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 25
        while not entered.exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, 'snapshot worker did not start'
            time.sleep(.02)
        worker_pid = int(entered.read_text())
        started = time.monotonic()
        process.send_signal(signal.SIGKILL if abrupt else signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=12)
        assert time.monotonic() - started < 12
        assert process.returncode == (-signal.SIGKILL if abrupt else 0), stdout + stderr
        deadline = time.monotonic() + 8
        while True:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                break
            assert time.monotonic() < deadline, 'evaluation worker survived coordinator shutdown'
            time.sleep(.02)
        release.touch()
        if abrupt:
            # Retry from a current-run checkpoint; reconciliation publishes the
            # lost observation and removes both native/rank transport snapshots.
            restored = resume(tmp_path / 'run')
            assert restored['status'] == 'complete'
            assert receipts(tmp_path / 'run')[0]['status'] == 'failed'
        else:
            manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
            assert manifest['stop_reason'] == 'SIGTERM'
            assert manifest['evaluation_schedule']['distance']['status'] == 'cancelled'
            event_path = next((tmp_path / 'run/metrics/evaluations').glob('*/events.jsonl'))
            event = json.loads(event_path.read_text())
            assert event['status'] == 'cancelled' and event['source_position_known']
            assert event['step'] == 1
        assert not list((tmp_path / 'run').glob('.evaluation-*'))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.heavy
def test_interval_completion_prunes_non_best_checkpoint_and_exports_bundle(tmp_path, monkeypatch):
    entered, release = tmp_path / 'entered_step2', tmp_path / 'release_step2'
    (tmp_path / 'ranked_interval_fixture.py').write_text(f'''
from pathlib import Path
import time
class RankedDistance:
    def describe(self): return {{'kind': 'scalar', 'label': 'Ranked distance'}}
    def evaluate(self, *, batches, context):
        for _ in batches:
            pass
        step = context['step']
        if step == 2:
            Path({str(entered)!r}).write_text('entered')
            deadline = time.monotonic() + 30
            while not Path({str(release)!r}).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('step 2 evaluation was not released')
                time.sleep(0.01)
            return 5.0
        return 0.1 if step == 1 else 9.0
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    path = config(tmp_path, monkeypatch, interval=1)
    text = (
        path.read_text()
        .replace('steps = 5', 'steps = 3')
        .replace('interval_fixture:Distance', 'ranked_interval_fixture:RankedDistance')
        + '''
[checkpoints]
keep_last = 1
keep_every = 0
keep_best = { metric = "distance", mode = "min", count = 1 }
export_bundle = "on_eval"
'''
    )
    path.write_text(text)
    run_dir = tmp_path / 'ranked_run'
    pinned_during_step3 = []

    def observe(row):
        if row['event'] == 'train' and row['step'] == 2:
            deadline = time.monotonic() + 20
            while not any(r.get('status') == 'complete' and r.get('source_step') == 1 for r in receipts(run_dir)):
                assert time.monotonic() < deadline, 'step 1 evaluation did not complete before step 2 schedule'
                time.sleep(0.01)
        elif row['event'] == 'checkpoint' and row['step'] == 3:
            deadline = time.monotonic() + 20
            while not entered.exists():
                assert time.monotonic() < deadline, 'step 2 evaluation did not enter before step 3 release'
                time.sleep(0.01)
            names = [p.name for p in (run_dir / 'checkpoints').iterdir() if p.is_dir() and not p.name.startswith('.')]
            pinned_during_step3.extend(names)
            release.touch()

    result = train(path, run_dir, checkpoint_every=1, on_event=observe)
    assert result['status'] == 'complete'
    # During step 3's final training checkpoint, step 2 was still pinned by S2 (active_evaluation).
    assert any('-step-00000002-' in name for name in pinned_during_step3)
    # Once step 2's evaluation finished with score 5.0 > 0.1, IntervalEvaluations pruned step 2 immediately.
    remaining = [p.name for p in (run_dir / 'checkpoints').iterdir() if p.is_dir() and not p.name.startswith('.')]
    assert any('-step-00000000-' in name for name in remaining)
    assert any('-step-00000001-' in name for name in remaining)
    assert not any('-step-00000002-' in name for name in remaining)
    assert any('-step-00000003-' in name for name in remaining)
    assert (run_dir / 'model.pt').is_file()
    assert (run_dir / 'model.sha256').is_file()
    assert result['exported_bundle_path'] == str(run_dir / 'model.pt')


@pytest.mark.heavy
def test_replicated_export_bundle_on_checkpoint_and_on_eval_stays_torch_free(tmp_path, monkeypatch):
    import hashlib
    import hypergan as hg

    base = config(tmp_path, monkeypatch, interval=2).read_text().replace('steps = 5', 'steps = 3')
    (tmp_path / 'rep_eval.toml').write_text(base + '\n[checkpoints]\nexport_bundle = "on_eval"\n')
    (tmp_path / 'rep_ckpt.toml').write_text(base + '\n[checkpoints]\nexport_bundle = "on_checkpoint"\n')

    driver = tmp_path / 'rep_export_driver.py'
    driver.write_text('''from pathlib import Path
import sys
from hypergan.replicated_execution import run_train

PROFILE = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2}}

if __name__ == '__main__':
    root = Path(sys.argv[1])
    run_train(root / 'rep_eval.toml', root / 'run_eval', profile=PROFILE, checkpoint_every=3)
    run_train(root / 'rep_ckpt.toml', root / 'run_ckpt', profile=PROFILE, checkpoint_every=2)
    assert 'torch' not in sys.modules, 'replicated coordinator imported torch during bundle export'
''')
    env = dict(os.environ)
    env['PYTHONPATH'] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = subprocess.run([sys.executable, str(driver), str(tmp_path)], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    run_eval = tmp_path / 'run_eval'
    ckpt_dirs = [
        p.name
        for d in ('checkpoints', 'distributed-checkpoints')
        if (run_eval / d).is_dir()
        for p in (run_eval / d).iterdir()
        if p.is_dir()
    ]
    assert ckpt_dirs and not any('-step-00000002-' in name for name in ckpt_dirs)
    assert (run_eval / 'model.pt').is_file() and (run_eval / 'model.sha256').is_file()
    assert hashlib.sha256((run_eval / 'model.pt').read_bytes()).hexdigest() == (run_eval / 'model.sha256').read_text().strip()
    bundle_eval = hg.load_bundle(run_eval / 'model.pt', device='cpu')
    assert bundle_eval['step'] == 2
    assert bundle_eval.sample(count=2, seed=3).shape == (2, 2)

    run_ckpt = tmp_path / 'run_ckpt'
    assert (run_ckpt / 'model.pt').is_file() and (run_ckpt / 'model.sha256').is_file()
    assert hashlib.sha256((run_ckpt / 'model.pt').read_bytes()).hexdigest() == (run_ckpt / 'model.sha256').read_text().strip()
    bundle_ckpt = hg.load_bundle(run_ckpt / 'model.pt', device='cpu')
    assert bundle_ckpt['step'] == 3
    assert bundle_ckpt.sample(count=2, seed=5).shape == (2, 2)

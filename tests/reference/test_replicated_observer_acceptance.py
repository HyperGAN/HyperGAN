"""Whole-job CPU evidence that optional isolated observers cannot alter training."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


_spec = importlib.util.spec_from_file_location(
    '_replicated_acceptance_fixtures', Path(__file__).with_name('test_replicated_job_acceptance.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


MODELS = _base.MODELS.replace(
    "        rank = dist.get_rank() if dist.is_initialized() else 0\n"
    "        with (root / 'all-ranks.txt').open('a') as output:\n"
    "            output.write(str(os.getpid()) + '\\n')\n"
    "        (root / f'rank-{rank}.pid').write_text(str(os.getpid()))\n"
    "        (root / 'broker.pid').write_text(str(multiprocessing.parent_process().pid))",
    "        role = 'rank' if dist.is_initialized() else 'renderer'\n"
    "        with (root / 'managed.jsonl').open('a') as output:\n"
    "            import json\n"
    "            output.write(json.dumps({'pid': os.getpid(), 'role': role, 'broker': multiprocessing.parent_process().pid}) + '\\n')"
).replace(
    "    def forward(self, x):\n        if os.environ.get('HG_ACCEPTANCE_MODE') == 'fatal-inference'",
    "    def forward(self, x):\n"
    "        if not dist.is_initialized():\n"
    "            mode = os.environ.get('HG_ACCEPTANCE_MODE', '')\n"
    "            root = Path(os.environ['HG_ACCEPTANCE_MARKERS'])\n"
    "            (root / 'render-entered').write_text(str(os.getpid()))\n"
    "            if mode in ('render-hang', 'render-death'):\n"
    "                ctypes.PyDLL(None).sleep(120)\n"
    "            if mode == 'render-collective':\n"
    "                dist.all_reduce(torch.zeros(1))\n"
    "        if os.environ.get('HG_ACCEPTANCE_MODE') == 'fatal-inference'"
)

CALLBACK = '''
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import random


def observe(event):
    root = Path(os.environ['HG_ACCEPTANCE_MARKERS'])
    with (root / 'managed.jsonl').open('a') as output:
        output.write(json.dumps({'pid': os.getpid(), 'role': 'callback', 'broker': multiprocessing.parent_process().pid}) + '\\n')
    with (root / 'delivered.jsonl').open('a') as output:
        output.write(json.dumps(event) + '\\n')
    # Only child code imports the numerical runtime, never the coordinator.
    if event['event'] == 'start':
        import numpy as np
        import torch
        import torch.distributed as dist
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert torch.get_num_threads() == 1
        assert not dist.is_initialized(), 'callback inherited the training process group'
        random.random(), np.random.random(), torch.rand(20)
        mode = os.environ.get('HG_ACCEPTANCE_MODE', '')
        if mode in ('callback-hang', 'callback-error'):
            (root / 'callback-fault-entered').write_text(mode)
        if mode == 'callback-hang':
            ctypes.PyDLL(None).sleep(120)
        if mode == 'callback-error':
            raise RuntimeError('injected isolated callback failure')
    event.clear()  # The private observer payload must not alias the live event.
'''

DRIVER = '''
import json
import os
from pathlib import Path
import sys
from hypergan import run_controller
from hypergan.replicated_execution import run_train, run_resume
from observer_callback import observe


def reaped(markers):
    rows = [json.loads(line) for line in (markers / 'managed.jsonl').read_text().splitlines()]
    assert any(row['role'] == 'rank' for row in rows)
    for pid in {row['pid'] for row in rows}:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError('managed process not reaped: ' + str(pid))


if __name__ == '__main__':
    config, run, markers, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    markers.mkdir(exist_ok=True)
    os.environ['HG_ACCEPTANCE_MARKERS'] = str(markers)
    os.environ['HG_ACCEPTANCE_MODE'] = mode
    assert 'torch' not in sys.modules
    original_atomic = run_controller.atomic_json
    def audit(path, value):
        if Path(path).name == 'manifest.json' and value.get('status') in ('complete', 'stopped', 'failed', 'interrupted'):
            reaped(markers)
        return original_atomic(path, value)
    run_controller.atomic_json = audit
    policy = {'startup_timeout': 40, 'command_timeout': 25, 'collective_timeout': 15,
              'total_timeout': 180, 'preview_timeout': 5 if mode == 'render-hang' else 30}
    if mode == 'render-death':
        policy['preview_timeout'] = 90
    callback = observe if mode in ('observed', 'split', 'resume', 'replay', 'callback-hang', 'callback-error') else None
    if callback is not None:
        policy['observer_timeout'] = 8 if mode == 'callback-hang' else 20
    options = {'service_policy': policy, 'on_event': callback}
    if mode in ('resume', 'replay'):
        if mode == 'replay':
            options.update(checkpoint=json.loads((run / 'saved-stop.json').read_text())['checkpoint_path'], max_seconds=1e-9)
        result = run_resume(run, **options)
    else:
        result = run_train(config, run,
            profile={'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'accumulation_steps': 2}},
            checkpoint_every=1, preview_every=0 if mode in ('plain', 'callback-hang', 'callback-error') else 1,
            preview_keep=2, stop_after_steps=2 if mode == 'split' else None, **options)
    reaped(markers)  # Includes terminal-event callbacks, which run after manifest publication.
    assert 'torch' not in sys.modules
    if mode == 'split':
        (run / 'saved-stop.json').write_text(json.dumps(result))
    (markers / 'result.json').write_text(json.dumps(result))
'''


def _setup(tmp_path):
    driver, config = _base._setup(tmp_path)
    (tmp_path / 'job_fixture.py').write_text(MODELS, encoding='utf-8')
    (tmp_path / 'observer_callback.py').write_text(CALLBACK, encoding='utf-8')
    driver.write_text(DRIVER, encoding='utf-8')
    return driver, config


def _run(driver, config, run, markers, mode):
    process = subprocess.Popen(_base._command(driver, config, run, markers, mode),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('observer whole-job test exceeded independent deadline')
    assert process.returncode == 0, stdout + stderr
    return json.loads((markers / 'result.json').read_text())


def _equal_runs(left, right):
    for actual, expected in zip(_base._states(left), _base._states(right)):
        _base._same(actual, expected)


def _index(run):
    value = json.loads((run / 'previews/index.json').read_text())
    assert value['keep'] == 2 and len(value['previews']) <= 2
    for record in value['previews']:
        path = Path(record['path'])
        payload = json.loads(path.read_text())
        assert payload['identity'] == record['identity']
        assert payload['step'] == record['step']
        assert path.stat().st_size == record['bytes']
    generations = [path for path in (run / 'previews').iterdir() if path.is_dir()]
    assert len(generations) <= 2
    return value['previews']


def test_observation_preserves_full_state_and_resume_preview_identity(tmp_path):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'plain', tmp_path / 'plain-pids', 'plain')
    full = _run(driver, config, tmp_path / 'observed', tmp_path / 'observed-pids', 'observed')
    assert not full['observation_errors']
    assert full['progress_observation']['accepted'] == full['progress_observation']['completed']
    assert not full['progress_observation']['pending']
    events = [json.loads(line) for line in (tmp_path/'observed/events.jsonl').read_text().splitlines()]
    assert [row['delivery'] for row in events if row['event']=='observer_status'] == [full['progress_observation']]
    assert [row['step'] for row in _index(tmp_path / 'observed')] == [3, 4]
    _equal_runs(tmp_path / 'plain', tmp_path / 'observed')
    stopped = _run(driver, config, tmp_path / 'split', tmp_path / 'stop-pids', 'split')
    prior = _index(tmp_path / 'split')
    assert [row['step'] for row in prior] == [1, 2]
    first_sequence = max(row['identity']['sample_sequence'] for row in prior)
    old_final = _base._artifacts(stopped)
    resumed = _run(driver, config, tmp_path / 'split', tmp_path / 'resume-pids', 'resume')
    retained = _index(tmp_path / 'split')
    assert [row['step'] for row in retained] == [3, 4]
    assert min(row['identity']['sample_sequence'] for row in retained) > first_sequence
    _equal_runs(tmp_path / 'plain', tmp_path / 'split')
    replayed = _run(driver, config, tmp_path / 'split', tmp_path / 'replay-pids', 'replay')
    assert replayed['steps'] == 2 and replayed['status'] == 'stopped'
    assert _index(tmp_path / 'split') == retained  # Zero updates reserve final artifact only.
    final = _run(driver, config, tmp_path / 'split', tmp_path / 'final-pids', 'resume')
    newest = _index(tmp_path / 'split')
    assert min(row['identity']['sample_sequence'] for row in newest) > max(row['identity']['sample_sequence'] for row in retained)
    assert all(row['identity']['attempt_id'] == final['attempt_id'] for row in newest)
    assert Path(stopped['bundle_path']).read_bytes() == old_final[0]
    assert Path(stopped['sample_path']).read_bytes() == old_final[1]
    assert stopped['next_sample_sequence'] < resumed['next_sample_sequence'] < replayed['next_sample_sequence'] < final['next_sample_sequence']
    _equal_runs(tmp_path / 'plain', tmp_path / 'split')


@pytest.mark.parametrize('mode', ['render-collective', 'render-hang', 'callback-error', 'callback-hang'])
def test_optional_observer_failures_are_bounded_and_do_not_change_updates(tmp_path, mode):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'plain', tmp_path / 'plain-pids', 'plain')
    result = _run(driver, config, tmp_path / 'fault', tmp_path / 'fault-pids', mode)
    assert result['status'] == 'complete' and result['steps'] == result['last_durable_step'] == 4
    _base._artifacts(result)
    _equal_runs(tmp_path / 'plain', tmp_path / 'fault')
    events = [json.loads(line) for line in (tmp_path / 'fault/events.jsonl').read_text().splitlines()]
    assert [row['step'] for row in events if row['event'] == 'train'] == [1, 2, 3, 4]
    if mode.startswith('render'):
        assert result['observation_errors'] and not result['previews']
        assert (tmp_path / 'fault-pids/render-entered').exists()
    else:
        assert (tmp_path / 'fault-pids/callback-fault-entered').read_text() == mode
        errors = [row for row in result['observation_errors'] if row['source'] == 'progress']
        assert len(errors) == 1 and errors[0]['step'] == 0
        assert len([row for row in events if row['event'] == 'observer_error' and row['source'] == 'progress']) == 1
        delivered = [json.loads(line) for line in (tmp_path / 'fault-pids/delivered.jsonl').read_text().splitlines()]
        assert any(row['event'] == 'start' and row['step'] == 0 for row in delivered)
        assert not any(row['event'] == 'train' for row in delivered)
        assert result['progress_observation']['failed'] == 1
        assert not result['progress_observation']['pending']


DEATH_HARNESS = '''
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from hypergan.run_state import run_lock


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


if __name__ == '__main__':
    assert sys.platform == 'linux'
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    driver, config, run, markers = map(Path, sys.argv[1:])
    markers.mkdir()
    bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', bootstrap,
               str(driver.parent), str(driver), str(config), str(run), str(markers), 'render-death']
    rows, brokers = [], set()
    with (markers / 'coordinator.log').open('w') as log:
        parent = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not (markers / 'render-entered').exists():
                assert parent.poll() is None, (markers / 'coordinator.log').read_text()
                assert time.monotonic() < deadline, 'renderer did not enter native call'
                time.sleep(.02)
            rows = [json.loads(line) for line in (markers / 'managed.jsonl').read_text().splitlines()]
            assert {'rank', 'renderer'} <= {row['role'] for row in rows}
            brokers = {row['broker'] for row in rows}
            manifest = json.loads((run / 'manifest.json').read_text())
            assert manifest['status'] == 'running' and manifest['last_durable_step'] == 1
            pointer = (run / 'distributed-checkpoints/latest.json').read_bytes()
            parent.kill()
            parent.wait(timeout=5)
            deadline = time.monotonic() + 15
            while any(alive(row['pid']) for row in rows):
                assert time.monotonic() < deadline, 'orphan training or renderer process survived coordinator death'
                time.sleep(.02)
            deadline = time.monotonic() + 10
            while brokers:
                for pid in list(brokers):
                    if os.waitpid(pid, os.WNOHANG)[0] == pid:
                        brokers.remove(pid)
                assert time.monotonic() < deadline, 'adopted guardian did not exit'
                time.sleep(.02)
            assert (run / 'distributed-checkpoints/latest.json').read_bytes() == pointer
            with run_lock(run):
                pass
            (markers / 'proof.json').write_text(json.dumps({'attempt_id': manifest['attempt_id'], 'roles': [row['role'] for row in rows]}))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if (markers / 'managed.jsonl').exists():
                rows = [json.loads(line) for line in (markers / 'managed.jsonl').read_text().splitlines()]
            for pid in {row['pid'] for row in rows} | brokers:
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not pid:
                    time.sleep(.02)
'''


def test_coordinator_death_during_native_hung_preview_reaps_both_groups(tmp_path):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'plain', tmp_path / 'plain-pids', 'plain')
    harness = tmp_path / 'observer_death.py'
    harness.write_text(DEATH_HARNESS, encoding='utf-8')
    process = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []), str(harness),
                                str(driver), str(config), str(tmp_path / 'fault'), str(tmp_path / 'death-pids')],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=95)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('observer parent-death harness exceeded independent deadline')
    assert process.returncode == 0, stdout + stderr
    proof = json.loads((tmp_path / 'death-pids/proof.json').read_text())
    result = _run(driver, config, tmp_path / 'fault', tmp_path / 'takeover-pids', 'resume')
    assert result['status'] == 'complete' and result['attempt_index'] == 2
    assert result['attempt_id'] != proof['attempt_id']
    _base._artifacts(result)
    _equal_runs(tmp_path / 'plain', tmp_path / 'fault')

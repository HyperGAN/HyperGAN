"""Independent Linux CPU whole-job acceptance through the shared controller."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch

from hypergan.config import write_default


MODELS = '''
import ctypes
import multiprocessing
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from hypergan.recipes import MLP

class Generator(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = Path(os.environ['HG_ACCEPTANCE_MARKERS'])
        rank = dist.get_rank() if dist.is_initialized() else 0
        with (root / 'all-ranks.txt').open('a') as output:
            output.write(str(os.getpid()) + '\\n')
        (root / f'rank-{rank}.pid').write_text(str(os.getpid()))
        (root / 'broker.pid').write_text(str(multiprocessing.parent_process().pid))

    def forward(self, x):
        if os.environ.get('HG_ACCEPTANCE_MODE') == 'fatal-inference' and not self.training:
            raise RuntimeError('injected required inference failure')
        return super().forward(x) + torch.randn(len(x), 2) * .01 + (random.random() + float(np.random.random())) * .001

class ImageGenerator(Generator):
    def forward(self, x):
        return super().forward(x).reshape(-1, 1, 1, 2)

class ImageDiscriminator(MLP):
    def forward(self, x):
        return super().forward(x.flatten(1))

class Data:
    def __init__(self):
        self.order, self.cursor, self.epoch, self.seen = None, 0, 0, 0

    def __call__(self, batch_size, *, generator):
        mode = os.environ.get('HG_ACCEPTANCE_MODE', '')
        root = Path(os.environ['HG_ACCEPTANCE_MARKERS'])
        if (mode in ('requests', 'fatal') and self.seen == 0) or (mode == 'orphan' and self.seen == 16):
            (root / f'gate-{dist.get_rank()}').write_text('entered')
            if mode == 'orphan':
                ctypes.PyDLL(None).sleep(120)  # Hold the worker GIL during abrupt coordinator death.
            else:
                deadline = time.monotonic() + 30
                while not (root / 'release').exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError('acceptance request submission missed its deadline')
                    time.sleep(.01)
        rows = []
        while len(rows) < batch_size:
            if self.order is None or self.cursor == len(self.order):
                self.order = torch.randperm(17, generator=generator)
                self.cursor = 0
                self.epoch += 1
            rows.append(int(self.order[self.cursor]))
            self.cursor += 1
        self.seen += batch_size
        rows = torch.tensor(rows)
        return {'real': torch.stack((rows / 17, -rows / 17), dim=1) + torch.randn(batch_size, 2, generator=generator) * .01}

    def resume_identity(self):
        if os.environ.get('HG_ACCEPTANCE_MODE') == 'fatal' and self.seen and dist.get_rank() == 1:
            (Path(os.environ['HG_ACCEPTANCE_MARKERS']) / 'injected-rank-1-exit').write_text('31')
            os._exit(31)  # Fail the optional checkpoint's worker group, not ordinary serialization.
        return {'dataset': 'seventeen-shuffled-items', 'revision': 2 if os.environ.get('HG_ACCEPTANCE_MODE') == 'mismatch-data' else 1}

    def state_dict(self):
        return {'order': self.order, 'cursor': self.cursor, 'epoch': self.epoch, 'seen': self.seen}

    def load_state_dict(self, state):
        self.order, self.cursor, self.epoch, self.seen = (state[key] for key in ('order', 'cursor', 'epoch', 'seen'))
'''

DRIVER = '''
import json
import os
from pathlib import Path
import sys
import threading
import time
from hypergan import run_controller
from hypergan.replicated_execution import run_train, run_resume
from hypergan.run_requests import submit_checkpoint_request, checkpoint_request_status


def profile(accumulation=2):
    return {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2,
                                              'accumulation_steps': accumulation}}


def reaped(markers):
    paths = list(markers.glob('rank-*.pid'))
    assert len(paths) == 2
    for pid in set(map(int, (markers / 'all-ranks.txt').read_text().splitlines())):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError('terminal status published before rank exit: ' + str(pid))


def submit_requests(run, markers, identifiers, errors):
    try:
        deadline = time.monotonic() + 40
        while not all((markers / f'gate-{rank}').exists() for rank in range(2)):
            assert time.monotonic() < deadline, 'workers never entered gated first update'
            time.sleep(.01)
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['status'] == 'running'
        for _ in range(2):
            value = submit_checkpoint_request(run, run_id=manifest['run_id'], attempt_id=manifest['attempt_id'])
            identifiers.append(value['request']['request_id'])
    except BaseException as error:
        errors.append(repr(error))
    finally:
        (markers / 'release').write_text('go')


if __name__ == '__main__':
    config, run, markers, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    markers.mkdir(exist_ok=True)
    os.environ['HG_ACCEPTANCE_MARKERS'] = str(markers)
    os.environ['HG_ACCEPTANCE_MODE'] = mode
    assert 'torch' not in sys.modules, 'replicated coordinator imported the numerical runtime'
    original_atomic = run_controller.atomic_json
    def audited_publish(path, value):
        if Path(path).name == 'manifest.json' and value.get('status') in ('complete', 'stopped', 'failed', 'interrupted'):
            reaped(markers)
        return original_atomic(path, value)
    run_controller.atomic_json = audited_publish
    policy = {'startup_timeout': 40, 'command_timeout': 25, 'collective_timeout': 15, 'total_timeout': 120}
    requests, errors, thread = [], [], None
    if mode in ('requests', 'fatal'):
        thread = threading.Thread(target=submit_requests, args=(run, markers, requests, errors))
        thread.start()
    if mode == 'requests':
        import hypergan.run_requests as request_api
        original_ack, calls = request_api.acknowledge_request, []
        def lost_ack(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise OSError('injected acknowledgement write failure')
            return original_ack(*args, **kwargs)
        request_api.acknowledge_request = lost_ack
    try:
        if mode in ('full', 'split', 'requests', 'fatal', 'fatal-inference'):
            result = run_train(config, run, profile=profile(), service_policy=policy,
                               checkpoint_every=100, stop_after_steps=2 if mode == 'split' else None)
        elif mode == 'replay':
            old = json.loads((run / 'saved-stop.json').read_text())
            result = run_resume(run, checkpoint=old['checkpoint_path'], profile=profile(),
                                service_policy=policy, max_seconds=1e-9)
        elif mode in ('mismatch-profile', 'mismatch-config', 'mismatch-data', 'preview', 'observer'):
            before = (run / 'manifest.json').read_bytes()
            pointer = (run / 'distributed-checkpoints/latest.json').read_bytes()
            attempts = sorted(path.name for path in (run / 'attempts').iterdir())
            options = {'profile': profile(1 if mode == 'mismatch-profile' else 2), 'service_policy': policy}
            if mode == 'mismatch-config':
                bad = config.with_name('changed.toml')
                bad.write_text(config.read_text().replace('steps = 4', 'steps = 5'))
                options['config_path'] = bad
            if mode == 'preview':
                options['preview_every'] = -1
            if mode == 'observer':
                options['on_event'] = lambda row: None
            try:
                run_resume(run, **options)
            except (ValueError, RuntimeError) as error:
                (markers / 'rejection.txt').write_text(str(error))
            else:
                raise AssertionError('unsupported or incompatible resume unexpectedly succeeded')
            assert (run / 'manifest.json').read_bytes() == before
            assert (run / 'distributed-checkpoints/latest.json').read_bytes() == pointer
            assert sorted(path.name for path in (run / 'attempts').iterdir()) == attempts
            result = {'rejected': True}
        else:
            policy.update(startup_timeout=45, command_timeout=30, collective_timeout=20, total_timeout=140)
            result = run_resume(run, service_policy=policy)
    except BaseException as error:
        if mode not in ('fatal', 'fatal-inference'):
            raise
        result = json.loads((run / 'manifest.json').read_text())
        assert result['status'] == 'failed'
        assert not result.get('sample_path') and not result.get('bundle_path')
        if mode == 'fatal':
            assert result['steps'] == 1 and result['last_durable_step'] == 0
            assert result['possible_lost_steps'] == 1
            assert (markers / 'injected-rank-1-exit').read_text() == '31'
            assert 'rank ' in str(error) and 'operation=prepare' in str(error)
            assert result['attempt_id'] in str(error)
        else:
            assert result['steps'] == result['last_durable_step'] == 4
            assert result['possible_lost_steps'] == 0
            assert 'injected required inference failure' in str(error)
    finally:
        if thread is not None:
            thread.join(timeout=45)
            assert not thread.is_alive() and not errors, errors
    if mode == 'split':
        (run / 'saved-stop.json').write_text(json.dumps(result))
    if requests:
        receipts = [checkpoint_request_status(run, identifier) for identifier in requests]
        if mode == 'requests':
            assert all(row['status'] == 'succeeded' and row['step'] == 1 for row in receipts)
            assert receipts[0]['checkpoint_path'] == receipts[1]['checkpoint_path']
            saved = json.loads((Path(receipts[0]['checkpoint_path']) / 'manifest.json').read_text())
            assert set(saved['request_ids']) == set(requests)
            events = [json.loads(row) for row in (run / 'events.jsonl').read_text().splitlines()]
            assert len([row for row in events if row['event'] == 'checkpoint' and set(row.get('request_ids', [])) == set(requests)]) == 1
        else:
            assert all(row['status'] != 'succeeded' for row in receipts)
        (markers / 'receipts.json').write_text(json.dumps(receipts))
    assert 'torch' not in sys.modules, 'replicated lifecycle or parent artifact validation imported Torch'
    (markers / 'result.json').write_text(json.dumps(result))
'''


def _setup(tmp_path):
    (tmp_path / 'job_fixture.py').write_text(MODELS, encoding='utf-8')
    driver = tmp_path / 'replicated_job.py'
    driver.write_text(DRIVER, encoding='utf-8')
    config = write_default(tmp_path / 'project', device="cpu")
    config.write_text(config.read_text().replace('steps = 5', 'steps = 4').replace('batch_size = 16', 'batch_size = 8')
                      .replace('num_particles = 20000', 'num_particles = 32')
                      .replace('factory = "mlp"', 'factory = "job_fixture:Generator"', 1)
                      .replace('factory = "gaussian_grid"', 'factory = "job_fixture:Data"')
                      .replace('side = 10\nnoise = 0.015\n', ''), encoding='utf-8')
    return driver, config


def _command(driver, config, run, markers, mode):
    bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    return [sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', bootstrap,
            str(driver.parent), str(driver), str(config), str(run), str(markers), mode]


def _run(driver, config, run, markers, mode):
    process = subprocess.Popen(_command(driver, config, run, markers, mode), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=80)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('whole-job acceptance exceeded its independent deadline')
    assert process.returncode == 0, stdout + stderr
    return json.loads((markers / 'result.json').read_text())


def _same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _same(left, right)
    else:
        assert actual == expected


def _states(run):
    pointer = json.loads((run / 'distributed-checkpoints/latest.json').read_text())
    root = run / 'distributed-checkpoints' / pointer['checkpoint']
    return [torch.load(root / f'rank-{rank:05d}.pt', weights_only=True) for rank in range(2)]


def _artifacts(manifest):
    bundle, sample = Path(manifest['bundle_path']), Path(manifest['sample_path'])
    assert bundle.is_file() and sample.is_file()
    model = torch.load(bundle, weights_only=True)
    payload = json.loads(sample.read_text())
    assert model['kind'] == 'ema-inference' and model['step'] == payload['step'] == manifest['steps']
    assert model['identity'] == payload['identity']
    assert payload['identity']['run_id'] == manifest['run_id']
    assert payload['identity']['attempt_id'] == manifest['attempt_id']
    assert payload['identity']['sample_sequence'] < manifest['next_sample_sequence']
    return bundle.read_bytes(), sample.read_bytes(), payload['identity']['sample_sequence']


def test_replicated_job_exact_recovery_old_selection_and_required_artifacts(tmp_path):
    driver, config = _setup(tmp_path)
    full = _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    stopped = _run(driver, config, tmp_path / 'split', tmp_path / 'stop-pids', 'split')
    old_bundle, old_sample, first_sequence = _artifacts(stopped)
    resumed = _run(driver, config, tmp_path / 'split', tmp_path / 'resume-pids', 'resume')
    assert full['status'] == resumed['status'] == 'complete' and stopped['status'] == 'stopped'
    assert resumed['execution'] == stopped['execution']
    assert resumed['service_policy'] != stopped['service_policy']
    for a, b in zip(_states(tmp_path / 'full'), _states(tmp_path / 'split')):
        _same(a, b)
    _, _, second_sequence = _artifacts(resumed)
    replayed = _run(driver, config, tmp_path / 'split', tmp_path / 'replay-pids', 'replay')
    assert replayed['status'] == 'stopped' and replayed['steps'] == replayed['last_durable_step'] == 2
    assert json.loads((tmp_path / 'split/distributed-checkpoints/latest.json').read_text())['step'] == 2
    _, _, third_sequence = _artifacts(replayed)
    final = _run(driver, config, tmp_path / 'split', tmp_path / 'final-pids', 'resume')
    _, _, fourth_sequence = _artifacts(final)
    assert first_sequence < second_sequence < third_sequence < fourth_sequence
    assert len({stopped['sample_path'], resumed['sample_path'], replayed['sample_path'], final['sample_path']}) == 4
    assert Path(stopped['bundle_path']).read_bytes() == old_bundle
    assert Path(stopped['sample_path']).read_bytes() == old_sample
    for a, b in zip(_states(tmp_path / 'full'), _states(tmp_path / 'split')):
        _same(a, b)


def test_strict_rejections_leave_run_and_attempts_untouched(tmp_path):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'run', tmp_path / 'stop-pids', 'split')
    for mode in ('mismatch-profile', 'mismatch-config', 'mismatch-data', 'preview', 'observer'):
        result = _run(driver, config, tmp_path / 'run', tmp_path / mode, mode)
        assert result['rejected'] is True


def test_manual_requests_reconcile_and_failed_save_group_is_fatal(tmp_path):
    driver, config = _setup(tmp_path)
    full = _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    observed = _run(driver, config, tmp_path / 'observed', tmp_path / 'request-pids', 'requests')
    assert full['status'] == observed['status'] == 'complete'
    for a, b in zip(_states(tmp_path / 'full'), _states(tmp_path / 'observed')):
        _same(a, b)
    failed = _run(driver, config, tmp_path / 'failed', tmp_path / 'failure-pids', 'fatal')
    assert failed['status'] == 'failed' and failed['last_durable_step'] == 0
    recovered = _run(driver, config, tmp_path / 'failed', tmp_path / 'recover-pids', 'resume')
    assert recovered['status'] == 'complete'
    for a, b in zip(_states(tmp_path / 'full'), _states(tmp_path / 'failed')):
        _same(a, b)


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
    if sys.platform == 'linux':
        assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    driver, config, run, markers = map(Path, sys.argv[1:])
    markers.mkdir()
    bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', bootstrap,
               str(driver.parent), str(driver), str(config), str(run), str(markers), 'orphan']
    ranks, broker = [], None
    with (markers / 'coordinator.log').open('w') as log:
        parent = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 50
            while not all((markers / f'gate-{rank}').exists() for rank in range(2)):
                assert parent.poll() is None, (markers / 'coordinator.log').read_text()
                assert time.monotonic() < deadline, 'resumed update never entered native call'
                time.sleep(.02)
            ranks = list(set(map(int, (markers / 'all-ranks.txt').read_text().splitlines())))
            broker = int((markers / 'broker.pid').read_text())
            manifest = json.loads((run / 'manifest.json').read_text())
            assert manifest['status'] == 'running' and manifest['steps'] == manifest['last_durable_step'] == 2
            pointer = (run / 'distributed-checkpoints/latest.json').read_bytes()
            parent.kill()
            parent.wait(timeout=5)
            deadline = time.monotonic() + 15
            while any(alive(pid) for pid in ranks):
                assert time.monotonic() < deadline, 'orphan ranks were not killed and reaped'
                time.sleep(.02)
            if sys.platform == 'linux':
                deadline = time.monotonic() + 10
                while os.waitpid(broker, os.WNOHANG)[0] != broker:
                    assert time.monotonic() < deadline, 'adopted broker did not exit'
                    time.sleep(.02)
            assert (run / 'distributed-checkpoints/latest.json').read_bytes() == pointer
            with run_lock(run):
                pass
            (markers / 'death-proof.json').write_text(json.dumps({'ranks': ranks, 'attempt_id': manifest['attempt_id']}))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if (markers / 'all-ranks.txt').exists():
                ranks = list(set(ranks) | set(map(int, (markers / 'all-ranks.txt').read_text().splitlines())))
            if broker is None and (markers / 'broker.pid').exists():
                broker = int((markers / 'broker.pid').read_text())
            for pid in [*ranks, *([broker] if broker else [])]:
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)
            if sys.platform == 'linux':
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        pid, _ = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if not pid:
                        time.sleep(.02)
'''


def test_abrupt_coordinator_death_and_fresh_controller_takeover(tmp_path):
    driver, config = _setup(tmp_path)
    _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    _run(driver, config, tmp_path / 'split', tmp_path / 'stop-pids', 'split')
    harness = tmp_path / 'death_harness.py'
    harness.write_text(DEATH_HARNESS, encoding='utf-8')
    process = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []),
                                str(harness), str(driver), str(config), str(tmp_path / 'split'),
                                str(tmp_path / 'orphan-pids')], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=85)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AssertionError('coordinator-death acceptance exceeded its independent deadline')
    assert process.returncode == 0, stdout + stderr
    proof = json.loads((tmp_path / 'orphan-pids/death-proof.json').read_text())
    resumed = _run(driver, config, tmp_path / 'split', tmp_path / 'takeover-pids', 'resume')
    assert resumed['status'] == 'complete' and resumed['attempt_index'] == 3
    assert resumed['attempt_id'] != proof['attempt_id']
    _artifacts(resumed)
    for a, b in zip(_states(tmp_path / 'full'), _states(tmp_path / 'split')):
        _same(a, b)


def test_actual_image_folder_whole_job_accumulated_recovery(tmp_path):
    from PIL import Image
    driver, config = _setup(tmp_path)
    images = tmp_path / 'images'
    images.mkdir()
    for index in range(5):
        Image.frombytes('L', (2, 1), bytes([index * 40, 255 - index * 30])).save(images / f'{index}.png')
    text = config.read_text().replace('job_fixture:Generator', 'job_fixture:ImageGenerator')
    text = text.replace('factory = "mlp"', 'factory = "job_fixture:ImageDiscriminator"')
    text = text.replace('factory = "job_fixture:Data"', 'factory = "image_folder"')
    text = text.replace('[data.args]', '[data.args]\nroot = ' + json.dumps(str(images)) + '\nheight = 1\nwidth = 2\nmode = "L"\nshuffle = true')
    config.write_text(text, encoding='utf-8')
    _run(driver, config, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    _run(driver, config, tmp_path / 'split', tmp_path / 'stop-pids', 'split')
    result = _run(driver, config, tmp_path / 'split', tmp_path / 'resume-pids', 'resume')
    assert result['status'] == 'complete'
    assert json.loads(Path(result['sample_path']).read_text())['shape'][1:] == [1, 1, 2]
    for a, b in zip(_states(tmp_path / 'full'), _states(tmp_path / 'split')):
        _same(a, b)


def test_required_inference_failure_recovers_without_repeating_updates(tmp_path):
    driver, config = _setup(tmp_path)
    failed = _run(driver, config, tmp_path / 'run', tmp_path / 'failure-pids', 'fatal-inference')
    before = _states(tmp_path / 'run')
    recovered = _run(driver, config, tmp_path / 'run', tmp_path / 'resume-pids', 'resume')
    assert failed['status'] == 'failed' and recovered['status'] == 'complete'
    assert recovered['steps'] == recovered['last_durable_step'] == 4
    assert recovered['attempt_index'] == 2 and recovered['attempt_id'] != failed['attempt_id']
    _artifacts(recovered)
    for actual, expected in zip(_states(tmp_path / 'run'), before):
        _same(actual, expected)

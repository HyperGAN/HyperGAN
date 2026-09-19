"""Linux CPU whole-process command, parent commit and abrupt-death acceptance."""
import json
import os
import signal
import subprocess
import sys

import torch


MODELS = '''
import random
import numpy as np
import torch
from hypergan.recipes import MLP

class StochasticGenerator(MLP):
    def forward(self, x):
        return super().forward(x) + torch.randn(len(x), 2) * 0.01 + (random.random() + float(np.random.random())) * 0.001

class OrderedData:
    def __init__(self):
        self.order = None
        self.cursor = 0
        self.epoch = 0

    def __call__(self, batch_size, *, generator):
        rows = []
        while len(rows) < batch_size:
            if self.order is None or self.cursor == len(self.order):
                self.order = torch.randperm(17, generator=generator)
                self.cursor = 0
                self.epoch += 1
            rows.append(int(self.order[self.cursor]))
            self.cursor += 1
        rows = torch.tensor(rows)
        return {'real': torch.stack((rows / 17, -rows / 17), dim=1) + torch.randn(batch_size, 2, generator=generator) * 0.01}

    def resume_identity(self):
        return {'dataset': 'seventeen-shuffled-items', 'revision': 1}

    def state_dict(self):
        return {'cursor': self.cursor, 'order': self.order, 'epoch': self.epoch}

    def load_state_dict(self, state):
        self.cursor, self.order, self.epoch = state['cursor'], state['order'], state['epoch']
'''

DRIVER = '''
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from hypergan.cpu_worker_service import CPUWorkerService
from hypergan.distributed_commit import CheckpointCommitAuthority
from hypergan.run_state import run_lock


def factory(rank, world_size, run, markers):
    import copy
    from hypergan.config import DEFAULT, resolve_config
    from hypergan.distributed_training import ReplicatedCPUTrainer
    Path(markers, f'rank-{rank}.pid').write_text(str(os.getpid()))
    raw = copy.deepcopy(DEFAULT)
    raw['training'].update(steps=4, batch_size=8)
    raw['prior']['args'] = {'num_particles': 32, 'z_dim': 4}
    raw['components']['generator']['factory'] = 'worker_fixture:StochasticGenerator'
    raw['data'] = {'factory': 'worker_fixture:OrderedData', 'args': {}}
    config = resolve_config(raw)
    trainer = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=2)
    return {'trainer': trainer, 'batch': None, 'run': run, 'markers': markers, 'rank': rank}


def handler(state, operation, payload):
    from hypergan.distributed_checkpoints import distributed_checkpoint_identity, prepare_distributed_checkpoint, restore_distributed_checkpoint
    trainer = state['trainer']
    if operation == 'identity':
        return distributed_checkpoint_identity(trainer)
    if operation == 'update':
        _, state['batch'] = trainer.update()
        return {'step': trainer.step, 'ready': trainer.checkpoint_ready}
    if operation in ('prepare', 'prepare-fail'):
        receipt = prepare_distributed_checkpoint(state['run'], trainer, state['batch'], payload['metadata'],
            command_sequence=payload['command_sequence'], controller_id=payload['controller_id'])
        if operation == 'prepare-fail':
            import torch.distributed as dist
            if state['rank'] == 0:
                Path(state['run'], 'uncommitted-receipt.json').write_text(json.dumps({
                    'receipt': receipt, 'sequence': payload['command_sequence']}))
            dist.barrier()  # Receipt really exists before the command result fails.
            if state['rank'] == 1:
                raise RuntimeError('rank-one failure after complete preparation')
        return receipt
    if operation == 'restore':
        _, info, state['batch'] = restore_distributed_checkpoint(state['run'], trainer, {'run_id': 'acceptance'})
        return {'step': trainer.step, 'saved_step': info['step']}
    if operation == 'native-hang':
        Path(state['markers'], f'inside-native-{state["rank"]}').write_text('entered')
        # PyDLL deliberately keeps the worker GIL held: a Python thread watchdog
        # cannot rescue this operation. The external owning broker must do so.
        ctypes.PyDLL(None).sleep(120)
        raise AssertionError('native hang unexpectedly returned')
    raise ValueError('unknown fixture command')


def service(run, markers, attempt):
    return CPUWorkerService(factory, handler, args=(str(run), str(markers)), run_id='acceptance',
        attempt_id=attempt, world_size=2, startup_timeout=40, command_timeout=25,
        collective_timeout=15, total_timeout=120)


def prepare(group, authority, attempt, operation='prepare'):
    sequence = group.next_sequence
    response = group.command(operation, {'metadata': {'run_id': 'acceptance', 'attempt_id': attempt},
        'controller_id': authority.controller_id, 'command_sequence': sequence})
    assert response['sequence'] == sequence
    assert len(response['results']) == 2 and response['results'][0] == response['results'][1]
    return response['results'][0], sequence


def execute(run, markers, mode):
    assert 'torch' not in sys.modules, 'parent service imported the numerical runtime'
    run.mkdir(exist_ok=True)
    markers.mkdir(exist_ok=True)
    with run_lock(run):
        with service(run, markers, mode) as group:
            identity_response = group.command('identity')
            assert identity_response['results'][0] == identity_response['results'][1]
            identity = identity_response['results'][0]
            with CheckpointCommitAuthority(run, run_id='acceptance', attempt_id=mode, identity=identity) as authority:
                sequence = identity_response['sequence']
                if mode in ('resume', 'orphan', 'takeover', 'partial-result'):
                    response = group.command('restore')
                    sequence = response['sequence']
                    assert all(row['step'] == row['saved_step'] == 2 for row in response['results'])
                if mode == 'takeover':
                    pointer = (run / 'distributed-checkpoints/latest.json').read_bytes()
                    old = json.loads((run / 'uncommitted-receipt.json').read_text())
                    try:
                        group.assert_healthy()
                        authority.commit(old['receipt'], expected_command_sequence=old['sequence'])
                    except (ValueError, RuntimeError):
                        pass
                    else:
                        raise AssertionError('new controller accepted prior controller receipt')
                    assert (run / 'distributed-checkpoints/latest.json').read_bytes() == pointer
                    receipt, accepted_sequence = prepare(group, authority, mode)
                    group.assert_healthy()
                    authority.commit(receipt, expected_command_sequence=accepted_sequence)
                    assert json.loads((run / 'distributed-checkpoints/latest.json').read_text())['step'] == 2
                count = 4 if mode == 'full' else 1 if mode in ('orphan', 'partial-result') else 2
                step = 2 if mode in ('resume', 'orphan', 'takeover', 'partial-result') else 0
                for _ in range(count):
                    expected_sequence = group.next_sequence
                    response = group.command('update')
                    step += 1
                    assert response['sequence'] == expected_sequence
                    assert all(row == {'ready': True, 'step': step} for row in response['results'])
                pointer_path = run / 'distributed-checkpoints/latest.json'
                before = pointer_path.read_bytes() if pointer_path.exists() else None
                if mode == 'partial-result':
                    try:
                        prepare(group, authority, mode, operation='prepare-fail')
                    except RuntimeError as error:
                        detail = str(error)
                        assert 'rank 1' in detail and 'prepare-fail' in detail and 'partial-result' in detail
                        assert 'rank-one failure after complete preparation' in detail
                        (markers / 'failure.txt').write_text(detail)
                    else:
                        raise AssertionError('failed rank command returned a usable aggregate')
                    try:
                        group.assert_healthy()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('failed command group still claims healthy')
                    staged = json.loads((run / 'uncommitted-receipt.json').read_text())
                    assert (run / staged['receipt']['staging']).is_dir()
                    assert pointer_path.read_bytes() == before
                    assert 'torch' not in sys.modules
                    return
                receipt, sequence = prepare(group, authority, mode)
                assert (pointer_path.read_bytes() if pointer_path.exists() else None) == before, 'workers published canonical checkpoint'
                if mode == 'orphan':
                    (run / 'uncommitted-receipt.json').write_text(json.dumps({'receipt': receipt, 'sequence': sequence}))
                    (markers / 'broker.pid').write_text(str(group.broker_pid))
                    group.command('native-hang')
                group.assert_healthy()
                target = authority.commit(receipt, expected_command_sequence=sequence)
                assert target.is_dir()
                committed = pointer_path.read_bytes()
                try:
                    authority.commit(receipt, expected_command_sequence=sequence)
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError('same prepare receipt committed twice')
                assert pointer_path.read_bytes() == committed
    assert 'torch' not in sys.modules, 'parent commit loaded numerical state'


if __name__ == '__main__':
    execute(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
'''


def _script(tmp_path):
    (tmp_path / 'worker_fixture.py').write_text(MODELS, encoding='utf-8')
    script = tmp_path / 'worker_service_case.py'
    script.write_text(DRIVER, encoding='utf-8')
    return script


def _command(script, run, markers, mode):
    bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    return [sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', bootstrap,
            str(script.parent), str(script), str(run), str(markers), mode]


def _run(script, run, markers, mode):
    process = subprocess.Popen(_command(script, run, markers, mode), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=os.name == 'posix')
    try:
        stdout, stderr = process.communicate(timeout=65)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise AssertionError('worker service exceeded independent acceptance deadline')
    assert process.returncode == 0, stdout + stderr
    _ranks_reaped(markers)


def _ranks_reaped(markers):
    pids = sorted(markers.glob('rank-*.pid'))
    assert len(pids) == 2
    for path in pids:
        try:
            os.kill(int(path.read_text()), 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f'worker remains alive or unreaped: {path.read_text()}')


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
    root = run / 'distributed-checkpoints'
    pointer = json.loads((root / 'latest.json').read_text())
    assert pointer['step'] == 4
    return [torch.load(root / pointer['checkpoint'] / f'rank-{rank:05d}.pt', weights_only=True)
            for rank in range(2)]


def test_persistent_commands_parent_commit_and_fresh_group_accumulated_recovery(tmp_path):
    script = _script(tmp_path)
    for run, attempt in [('full', 'full'), ('resumed', 'split'), ('resumed', 'resume')]:
        _run(script, tmp_path / run, tmp_path / (attempt + '-pids'), attempt)
    for full, resumed in zip(_states(tmp_path / 'full'), _states(tmp_path / 'resumed')):
        _same(full, resumed)


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


def exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


if __name__ == '__main__':
    # Keep reaping responsibilities local even under a container PID 1 that
    # leaves orphan zombies. The service broker still must reap its own ranks.
    if sys.platform == 'linux':
        assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    driver, run, markers = map(Path, sys.argv[1:])
    markers.mkdir()
    pointer = run / 'distributed-checkpoints/latest.json'
    before = pointer.read_bytes()
    bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    command = [sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', bootstrap,
               str(driver.parent), str(driver), str(run), str(markers), 'orphan']
    rank_pids, broker_pid = [], None
    with (markers / 'coordinator.log').open('w') as log:
        parent = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while not all((markers / f'inside-native-{rank}').exists() for rank in range(2)):
                assert parent.poll() is None, (markers / 'coordinator.log').read_text()
                assert time.monotonic() < deadline, 'coordinator never reached native in-flight work'
                time.sleep(0.02)
            rank_pids = [int((markers / f'rank-{rank}.pid').read_text()) for rank in range(2)]
            broker_pid = int((markers / 'broker.pid').read_text())
            assert pointer.read_bytes() == before
            parent.kill()  # No finally, abort, close, or cooperative Python handler.
            parent.wait(timeout=5)
            deadline = time.monotonic() + 15
            while any(exists(pid) for pid in rank_pids):
                assert time.monotonic() < deadline, 'guardian failed to kill and reap old ranks'
                time.sleep(0.02)
            if sys.platform == 'linux':
                deadline = time.monotonic() + 10
                while True:
                    waited, status = os.waitpid(broker_pid, os.WNOHANG)
                    if waited == broker_pid:
                        assert os.WIFEXITED(status) or os.WIFSIGNALED(status)
                        break
                    assert time.monotonic() < deadline, 'guardian did not exit after reaping ranks'
                    time.sleep(0.02)
            assert pointer.read_bytes() == before, 'orphan preparation changed canonical latest'
            with run_lock(run):
                pass  # The old owner is gone and the preserved run is available.
            (markers / 'death-proof.json').write_text(json.dumps({'parent': parent.pid,
                'broker': broker_pid, 'ranks': rank_pids, 'ranks_reaped': True}))
        finally:
            rank_pids = list(set(rank_pids) | {int(path.read_text()) for path in markers.glob('rank-*.pid')})
            if broker_pid is None and (markers / 'broker.pid').exists():
                broker_pid = int((markers / 'broker.pid').read_text())
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            for pid in [*rank_pids, *([broker_pid] if broker_pid else [])]:
                if exists(pid):
                    os.kill(pid, signal.SIGKILL)
            if sys.platform == 'linux':
                # Reap any adopted processes on failure too; no live work leaks
                # from an unsuccessful acceptance fixture.
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        waited, _ = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if not waited:
                        time.sleep(0.02)
'''


def test_abrupt_parent_death_during_native_work_preserves_commit_and_allows_fenced_takeover(tmp_path):
    script = _script(tmp_path)
    _run(script, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    _run(script, tmp_path / 'recovery', tmp_path / 'split-pids', 'split')
    harness = tmp_path / 'death_harness.py'
    harness.write_text(DEATH_HARNESS, encoding='utf-8')
    process = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []),
                                str(harness), str(script), str(tmp_path / 'recovery'),
                                str(tmp_path / 'orphan-pids')], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=os.name == 'posix')
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise AssertionError('abrupt-death harness exceeded its independent deadline')
    assert process.returncode == 0, stdout + stderr
    proof = json.loads((tmp_path / 'orphan-pids/death-proof.json').read_text())
    assert proof['ranks_reaped'] is True
    _ranks_reaped(tmp_path / 'orphan-pids')
    _run(script, tmp_path / 'recovery', tmp_path / 'takeover-pids', 'takeover')
    for full, recovered in zip(_states(tmp_path / 'full'), _states(tmp_path / 'recovery')):
        _same(full, recovered)


def test_prepared_receipt_without_every_rank_command_result_cannot_advance_latest(tmp_path):
    script = _script(tmp_path)
    _run(script, tmp_path / 'full', tmp_path / 'full-pids', 'full')
    _run(script, tmp_path / 'recovery', tmp_path / 'split-pids', 'split')
    pointer = tmp_path / 'recovery/distributed-checkpoints/latest.json'
    before = pointer.read_bytes()
    _run(script, tmp_path / 'recovery', tmp_path / 'failure-pids', 'partial-result')
    assert pointer.read_bytes() == before
    assert (tmp_path / 'failure-pids/failure.txt').is_file()
    _run(script, tmp_path / 'recovery', tmp_path / 'takeover-pids', 'takeover')
    for full, recovered in zip(_states(tmp_path / 'full'), _states(tmp_path / 'recovery')):
        _same(full, recovered)

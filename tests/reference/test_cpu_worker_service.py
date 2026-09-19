"""Real persistent Gloo workers: idle/control boundaries and broker cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


DRIVER = '''
import json, os, sys, time
from pathlib import Path
from hypergan.cpu_worker_service import CPUWorkerService

def factory(rank, world, root, mode):
    if rank==0:
        Path(root,'broker.pid').write_text(str(os.getppid()))
    Path(root, f'rank-{rank}.pid').write_text(str(os.getpid()))
    if mode=='startup-hang':
        import ctypes
        ctypes.PyDLL(None).sleep(120)
    if mode == 'startup' and rank == 1:
        raise ValueError('fixture startup refusal')
    if mode in ('mismatch','missing') and rank == 1:
        import hypergan.cpu_worker_service as implementation
        receive=implementation._receive
        def altered(channel, timeout=None):
            row=receive(channel,timeout)
            if row.get('kind')=='command':
                if mode=='missing':
                    import ctypes
                    ctypes.PyDLL(None).sleep(120)
                row['payload']={'different':'rank-one-command'}
            return row
        implementation._receive=altered
    return {'rank':rank,'value':0,'root':root}

def handler(state, operation, payload):
    import torch
    import torch.distributed as dist
    Path(state['root'],f'handler-{state["rank"]}').touch()
    if operation == 'hang':
        import ctypes
        ctypes.PyDLL(None).sleep(120)
    if operation == 'fail' and state['rank'] == 1:
        raise ValueError('fixture rank one refusal')
    if operation == 'exit':
        os._exit(7)
    if operation == 'add':
        value=torch.tensor(payload)
        dist.all_reduce(value)
        state['value'] += value.item()
    return {'rank':state['rank'],'value':state['value']}

if __name__ == '__main__':
    root, mode = sys.argv[1:]
    service=CPUWorkerService(factory,handler,args=(root,mode),run_id='run-one',attempt_id='attempt-one',
             startup_timeout=20,command_timeout=4,collective_timeout=2,total_timeout=30 if mode!='total' else 5)
    error=None
    try:
        with service:
            if mode=='happy':
                first=service.command('add',2)
                assert first['sequence']==1 and [r['value'] for r in first['results']]==[4,4]
                assert service.sequence==1 and service.next_sequence==2
                # Longer than collective timeout: no collective may run while idle.
                time.sleep(2.5)
                service.assert_healthy()
                second=service.command('add',3)
                assert second['sequence']==2 and [r['value'] for r in second['results']]==[10,10]
            elif mode=='stale':
                service._channel.send({'run_id':'run-one','attempt_id':'old-attempt','sequence':1,
                                       'operation':'add','kind':'command','payload':2})
                service._reply('add',8)
            elif mode=='total':
                time.sleep(6)
                service.assert_healthy()
            elif mode=='idle-exit':
                service.command('add',1)
                os.kill(service.worker_pids[1],9)
                time.sleep(0.1)
                service.assert_healthy()
            else:
                service.command(mode)
    except (RuntimeError, TimeoutError, EOFError, OSError) as exc:
        error=str(exc)
    assert (error is None)==(mode=='happy'),error
    if mode in ('fail','startup'):
        assert 'rank 1' in error,error
        assert 'fixture' in error,error
    if mode=='stale':
        assert 'Stale or invalid' in error,error
    if mode=='total':
        assert 'total deadline exceeded' in error,error
    if mode=='idle-exit':
        assert 'rank 1' in error.lower(),error
    if mode in ('mismatch','missing'):
        assert not list(Path(root).glob('handler-*')),'handler ran without all-rank command agreement'
    if mode=='mismatch':
        assert 'disagree' in error,error
    for path in Path(root).glob('rank-*.pid'):
        try: os.kill(int(path.read_text()),0)
        except ProcessLookupError: pass
        else: raise AssertionError('rank leaked after service return')
    import multiprocessing
    assert not multiprocessing.active_children()
    print(json.dumps({'error':error,'broker_pid':service.broker_pid}))
'''


@pytest.mark.parametrize('mode', ['happy', 'fail', 'hang', 'exit', 'startup', 'stale', 'total', 'idle-exit', 'mismatch', 'missing'])
def test_persistent_commands_and_cleanup(tmp_path, mode):
    script = tmp_path / 'service_driver.py'
    script.write_text(DRIVER, encoding='utf-8')
    result = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []), str(script), str(tmp_path), mode],
                            cwd=tmp_path, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert isinstance(json.loads(result.stdout)['broker_pid'], int)


def test_coordinator_death_during_native_startup_reaps_ranks(tmp_path):
    """Linux CPU acceptance uses a subreaper because container PID 1 may not reap."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    script = tmp_path / 'service_driver.py'
    script.write_text(DRIVER, encoding='utf-8')
    process = subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []), str(script), str(tmp_path), 'startup-hang'],
                               cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    broker = None
    ranks = []
    try:
        deadline = time.monotonic() + 15
        while not (len(list(tmp_path.glob('rank-*.pid'))) == 2
                   and all(path.stat().st_size for path in tmp_path.glob('rank-*.pid'))):
            assert process.poll() is None, 'coordinator exited before startup fixture'
            assert time.monotonic() < deadline, 'ranks did not enter startup fixture'
            time.sleep(0.02)
        ranks = [int(path.read_text()) for path in tmp_path.glob('rank-*.pid')]
        broker = int((tmp_path / 'broker.pid').read_text())
        process.kill()
        process.wait(timeout=5)
        deadline = time.monotonic() + 8
        while True:
            alive = []
            for pid in ranks:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                alive.append(pid)
            if not alive:
                break
            assert time.monotonic() < deadline, f'guardian left ranks alive or zombies: {alive}'
            time.sleep(0.02)
        while True:
            reaped, status = os.waitpid(broker, os.WNOHANG)
            if reaped:
                assert os.waitstatus_to_exitcode(status) == 0
                broker = None
                break
            assert time.monotonic() < deadline, 'broker did not exit after reaping ranks'
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pid in ranks:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if broker is not None:
            try:
                os.kill(broker, 9)
                os.waitpid(broker, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        # A failing guardian assertion can force broker termination above. Any
        # unreaped rank then belongs to this temporary subreaper, not PID 1.
        deadline = time.monotonic() + 2
        for pid in ranks:
            while True:
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if reaped or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0

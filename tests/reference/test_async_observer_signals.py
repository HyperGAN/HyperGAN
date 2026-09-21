"""Signals cancel terminal progress workers without hiding numerical failures."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from hypergan.config import write_default


# Heavy: every test here starts real subprocesses or multi-rank jobs and
# measured at a second or more; see reports/test-durations-2026-09-20.txt.
pytestmark = pytest.mark.heavy


@pytest.mark.skipif(os.name == 'nt', reason='POSIX cooperative SIGTERM contract')
@pytest.mark.parametrize('fatal', [False, True])
def test_signal_during_terminal_callback_cancels_hour_timeout_and_reaps(tmp_path, fatal):
    config = write_default(tmp_path/'config', device='cpu')
    config.write_text(config.read_text().replace('steps = 5', 'steps = 2')
                      .replace('num_particles = 20000', 'num_particles = 32'))
    marker = tmp_path/'callback-pid'
    (tmp_path/'signal_callback.py').write_text('''
import ctypes
import os
from pathlib import Path

def observe(event):
    if event['event'] in ('complete', 'failed'):
        Path(os.environ['HG_CALLBACK_SIGNAL_MARKER']).write_text(str(os.getpid()))
        ctypes.PyDLL(None).sleep(120)
''')
    driver = tmp_path/'driver.py'
    driver.write_text('''
from pathlib import Path
import sys
from hypergan.replicated_execution import ReplicatedExecution, run_train
from hypergan.run_controller import FatalExecutionError
from signal_callback import observe

if __name__ == '__main__':
    root=Path(sys.argv[1])
    if sys.argv[2]=='fatal':
        def failed_update(self):
            raise FatalExecutionError('primary numerical failure')
        ReplicatedExecution.update=failed_update
    run_train(root/'config', root/'run',
        profile={'schema_version':1,'execution':{'name':'cpu-replicated-gloo','world_size':2}},
        service_policy={'observer_timeout':3600}, on_event=observe)
''')
    bootstrap='import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    command=[sys.executable, *(['-I'] if sys.flags.isolated else []), '-c', bootstrap,
             str(tmp_path),str(driver),str(tmp_path),'fatal' if fatal else 'normal']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        env=dict(os.environ,HG_CALLBACK_SIGNAL_MARKER=str(marker)))
    try:
        deadline=time.monotonic()+30
        while not marker.exists() and time.monotonic()<deadline:
            assert process.poll() is None
            time.sleep(.01)
        assert marker.exists(), 'Terminal callback did not start'
        started=time.monotonic()
        process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=12)
        assert time.monotonic()-started<12
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    saved=json.loads((tmp_path/'run/manifest.json').read_text())
    assert saved['stop_reason']=='SIGTERM'
    assert saved['status']==('failed' if fatal else 'complete')
    assert saved['last_durable_step']==(0 if fatal else 2)
    assert (process.returncode!=0)==fatal, stdout+stderr
    if fatal:
        assert 'primary numerical failure' in saved['error']
        assert 'primary numerical failure' in stderr
    status=saved['progress_observation']
    assert status['cancelled']==1 and status['failed']==0 and not status['pending']
    assert status['last_cancelled_step']==(0 if fatal else 2)
    events=[json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    assert [event['delivery'] for event in events if event['event']=='observer_status']==[status]
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()),0)

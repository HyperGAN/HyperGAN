"""Real process signals never publish a discriminator-only update."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from hypergan.checkpoints import read_checkpoint
from hypergan.config import write_default
from hypergan.run_state import validate_event_boundary
from hypergan.training import train, resume
from .test_checkpoint_event_commit import equal


SCRIPT = '''
import os, signal, sys, torch
from pathlib import Path
from hypergan.cli import main
from hypergan.training import DeviceAdam
import hypergan.checkpoints as checkpoints
original_step = DeviceAdam.step
original_atomic = checkpoints.atomic_json
original_save = torch.save
calls = 0
mode = sys.argv[3]
def update(self, *args, **kwargs):
    global calls
    result = original_step(self, *args, **kwargs)
    calls += 1
    if calls == 1 and mode in ('SIGTERM', 'SIGINT', 'SIGKILL'):
        os.kill(os.getpid(), getattr(signal, mode))
    return result
def publish(path, value):
    selected = Path(path).name == 'latest.json' and value.get('step') == 1
    if selected and mode in ('publication', 'kill-pointer'):
        os.kill(os.getpid(), signal.SIGTERM if mode == 'publication' else signal.SIGKILL)
    original_atomic(path, value)
    if selected and mode == 'kill-after-pointer':
        os.kill(os.getpid(), signal.SIGKILL)
def save(value, destination, *args, **kwargs):
    if mode == 'kill-payload' and value.get('step') == 1:
        destination.write(b'partial')
        destination.flush()
        os.kill(os.getpid(), signal.SIGKILL)
    return original_save(value, destination, *args, **kwargs)
from hypergan.run_state import EventJournal
original_append = EventJournal.append
def append(journal, row):
    if mode == 'kill-event' and row['step'] == 1:
        with journal.path.open('ab') as output:
            output.write(b'{"incomplete":')
            output.flush()
        os.kill(os.getpid(), signal.SIGKILL)
    return original_append(journal, row)
EventJournal.append = append
torch.save = save
DeviceAdam.step = update
checkpoints.atomic_json = publish
raise SystemExit(main(['train', sys.argv[1], '--run-dir', sys.argv[2], '--no-server', '--checkpoint-every', '1']))
'''


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signals; Windows TerminateProcess is forced termination')
@pytest.mark.parametrize('mode', ['SIGTERM', 'SIGINT', 'SIGKILL', 'publication',
                                 'kill-event', 'kill-payload', 'kill-pointer', 'kill-after-pointer'])
def test_real_cli_signal_during_half_update_or_pointer_publish_recovers_exactly(tmp_path, mode):
    config = write_default(tmp_path / 'config', device='cpu')
    train(config, tmp_path / 'full')
    result = subprocess.run([sys.executable, '-c', SCRIPT, str(config), str(tmp_path / 'run'), mode],
                            capture_output=True, text=True, timeout=30)
    killed = mode == 'SIGKILL' or mode.startswith('kill-')
    assert result.returncode == (-signal.SIGKILL if killed else 0), result.stderr
    _, info, state = read_checkpoint(tmp_path / 'run')
    assert info['step'] == state['step'] == (0 if killed and mode != 'kill-after-pointer' else 1)
    validate_event_boundary(tmp_path / 'run', info)
    if not killed:
        manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
        assert manifest['status'] == 'stopped'
        assert manifest['stop_reason'] == ('SIGTERM' if mode == 'publication' else mode)
        assert 'bundle_path' not in manifest
    resume(tmp_path / 'run')
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'run')[2])

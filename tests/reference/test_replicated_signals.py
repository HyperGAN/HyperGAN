"""Terminal group signals leave ranks alive through the controller's save."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from hypergan.config import write_default
from hypergan.run_state import validate_event_boundary
from .test_checkpoint_event_commit import equal


SCRIPT = '''
import os, signal, sys
from hypergan.cli import main
from hypergan.replicated_execution import ReplicatedExecution
update = ReplicatedExecution.update
mode = sys.argv[1]
sent = False
def stop(self):
    global sent
    if not sent and mode != 'none':
        sent = True
        os.killpg(os.getpgrp(), getattr(signal, mode))
    return update(self)
ReplicatedExecution.update = stop
raise SystemExit(main(sys.argv[2:]))
'''


def states(run):
    root = run / 'distributed-checkpoints'
    path = root / json.loads((root / 'latest.json').read_text())['checkpoint']
    info = json.loads((path / 'manifest.json').read_text())
    validate_event_boundary(run, info)
    return [torch.load(path / record['file'], map_location='cpu', weights_only=True) for record in info['ranks']]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal process-group signal contract')
@pytest.mark.parametrize('mode', ['SIGINT', 'SIGTERM'])
def test_group_signal_during_update_commits_and_resumes_all_ranks(tmp_path, mode):
    config = write_default(tmp_path / 'config', device='cpu')
    config.write_text(config.read_text().replace('steps = 5', 'steps = 2'))
    full, split = tmp_path / 'full', tmp_path / 'split'
    def invoke(mode, *arguments):
        result = subprocess.run([sys.executable, '-c', SCRIPT, mode, *map(str, arguments), '--no-server'],
                                start_new_session=True, capture_output=True, text=True, timeout=75)
        assert result.returncode == 0, result.stdout + result.stderr
        # Terminal output is a bounded observer; recovery reads authoritative files.
        run = arguments[arguments.index('--run-dir') + 1] if arguments[0] == 'train' else arguments[1]
        return json.loads((Path(run) / 'manifest.json').read_text())
    invoke('none', 'train', config, '--run-dir', full, '--profile', 'cpu-replicated-gloo')
    stopped = invoke(mode, 'train', config, '--run-dir', split, '--profile', 'cpu-replicated-gloo')
    assert stopped['steps'] == stopped['last_durable_step'] == 1
    assert stopped['status'] == 'stopped' and stopped['stop_reason'] == mode
    invoke('none', 'resume', split)
    for expected, actual in zip(states(full), states(split)):
        equal(expected, actual)

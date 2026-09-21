"""Production supervisor, designated writer lock and fresh-group resume together."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


# Heavy: every test here starts real subprocesses or multi-rank jobs and
# measured at a second or more; see reports/test-durations-2026-09-20.txt.
pytestmark = pytest.mark.heavy


SCRIPT = '''
from contextlib import nullcontext
from pathlib import Path
import sys

from hypergan.cpu_workers import launch_cpu_workers

def worker(rank, world_size, directory, mode):
    from hypergan.config import resolve_config
    from hypergan.distributed_training import ReplicatedCPUTrainer
    from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint
    from hypergan.run_state import run_lock

    with run_lock(directory) if rank == 0 else nullcontext():
        config = resolve_config({'training': {'steps': 3}, 'prior': {'args': {'num_particles': 32, 'z_dim': 4}}})
        trainer = ReplicatedCPUTrainer(config, world_size=world_size)
        batch = None
        if mode == 'resume':
            _, info, batch = restore_distributed_checkpoint(directory, trainer, {'run_id': 'supervised'})
            assert info['step'] == trainer.step == 1
        target = 1 if mode == 'split' else 3
        while trainer.step < target:
            _, batch = trainer.update()
        save_distributed_checkpoint(directory, trainer, batch,
            {'run_id': 'supervised', 'attempt_id': mode})

if __name__ == '__main__':
    directory, mode = sys.argv[1:]
    if mode != 'resume':
        Path(directory).mkdir()
    launch_cpu_workers(worker, args=(directory, mode), timeout=30, collective_timeout=15)
'''


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
    assert pointer['step'] == 3
    generation = root / pointer['checkpoint']
    return [torch.load(generation / f'rank-{rank:05d}.pt', weights_only=True) for rank in range(2)]


def test_supervised_training_and_fresh_group_resume(tmp_path):
    script = tmp_path / 'cpu_job.py'
    script.write_text(SCRIPT)
    for directory, mode in [('full', 'full'), ('resumed', 'split'), ('resumed', 'resume')]:
        completed = subprocess.run(
            [sys.executable, *(['-I'] if sys.flags.isolated else []), str(script), str(tmp_path / directory), mode],
            capture_output=True, text=True, timeout=40)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    for full, resumed in zip(_states(tmp_path / 'full'), _states(tmp_path / 'resumed')):
        _same(full, resumed)

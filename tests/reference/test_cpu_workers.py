"""Exercise the production CPU supervisor from fresh importable worker files."""
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hypergan.cpu_workers import launch_cpu_workers


@pytest.mark.parametrize("arguments", [
    {"world_size": True}, {"world_size": 0}, {"world_size": 65},
    {"timeout": float("nan")}, {"timeout": 0}, {"collective_timeout": True},
    {"timeout": 1, "collective_timeout": 2}, {"args": []},
])
def test_invalid_worker_controls(arguments):
    with pytest.raises(ValueError):
        launch_cpu_workers(print, **arguments)


def test_parent_interrupt_reaps_started_workers(monkeypatch):
    before = {p.pid for p in multiprocessing.active_children()}

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("hypergan.cpu_workers.time.sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        launch_cpu_workers(print)
    assert {p.pid for p in multiprocessing.active_children()} == before


SCRIPT = '''
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import torch
import torch.distributed as dist
from hypergan.cpu_workers import launch_cpu_workers

def worker(rank, world_size, directory, mode):
    root = Path(directory)
    (root / (str(rank) + '.pid')).write_text(str(os.getpid()))
    if mode == 'crash' and rank == 1:
        os._exit(17)
    if mode == 'error' and rank == 1:
        raise ValueError('deliberate callback failure')
    if mode in ('crash', 'error', 'stall'):
        time.sleep(120)
    value = torch.tensor(rank + 1)
    dist.all_reduce(value)
    assert value.item() == 3
    (root / (str(rank) + '.json')).write_text(json.dumps({'sum': value.item(), 'threads': torch.get_num_threads()}))

if __name__ == '__main__':
    root, mode = sys.argv[1:]
    try:
        launch_cpu_workers(worker, args=(root, mode), timeout=5 if mode == 'stall' else 30,
                           collective_timeout=3 if mode == 'stall' else 15)
        assert mode == 'success'
    except (RuntimeError, TimeoutError) as exc:
        assert mode != 'success'
        Path(root, 'failure.txt').write_text(str(exc))
    assert not multiprocessing.active_children(), 'supervisor leaked workers'
'''


@pytest.mark.parametrize("mode", ["success", "crash", "error", "stall"])
def test_cpu_worker_lifecycle(tmp_path, mode):
    script = tmp_path / "worker_case.py"
    script.write_text(SCRIPT)
    command = [sys.executable, *(["-I"] if sys.flags.isolated else []), str(script), str(tmp_path), mode]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if mode == "success":
        for rank in (0, 1):
            assert json.loads((tmp_path / f"{rank}.json").read_text()) == {"sum": 3, "threads": 1}
    else:
        error = (tmp_path / "failure.txt").read_text()
        assert {"crash": "code 17", "error": "deliberate callback failure", "stall": "exceeded 5 seconds"}[mode] in error
    if os.name == "posix":
        for pid_file in tmp_path.glob("*.pid"):
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

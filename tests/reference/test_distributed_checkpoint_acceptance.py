"""Independent checkpoint observation and atomic-publication regressions."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from hypergan.checkpoints import capture_rng
from hypergan.config import DEFAULT, resolve_config
from hypergan.recipes import GaussianGrid


class DescriptorRngData(GaussianGrid):
    """Deterministic identity output, deliberately impure observer implementation."""
    resume_stateless = True

    def resume_identity(self):
        random.random()
        np.random.random()
        torch.rand(3)
        return {'fixture': 'deterministic-descriptor-with-observer-rng'}


def _equal_rng(actual, expected):
    assert torch.equal(actual['torch'], expected['torch'])
    assert actual['python'] == expected['python']
    assert actual['numpy'] == expected['numpy']


def _worker(rank, root):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    import hypergan.distributed_checkpoints as checkpoint_api

    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method=(root / 'rendezvous').as_uri(), timeout=timedelta(seconds=12))
    try:
        raw = copy.deepcopy(DEFAULT)
        raw['prior']['args']['num_particles'] = 16
        raw['data'] = {'factory': f'{__name__}:DescriptorRngData', 'args': {'side': 3}}
        trainer = ReplicatedCPUTrainer(resolve_config(raw), world_size=2)
        before = capture_rng()
        run = root / 'run'
        selected = checkpoint_api.save_distributed_checkpoint(run, trainer, None,
            {'run_id': 'rng-test', 'attempt_id': 'attempt-one'})
        _equal_rng(capture_rng(), before)
        saved = torch.load(selected / f'rank-{rank:05d}.pt', weights_only=True)
        _equal_rng(saved['rng'], before)
        pointer = run / 'distributed-checkpoints/latest.json'
        prior_pointer = pointer.read_bytes()

        # The writer must not publish metadata that its own bounded reader refuses.
        checkpoint_api.MAX_METADATA_BYTES = 64
        try:
            checkpoint_api.save_distributed_checkpoint(run, trainer, None,
                {'run_id': 'rng-test', 'attempt_id': 'attempt-two'})
        except ValueError as error:
            assert 'metadata' in str(error).lower(), str(error)
        else:
            raise AssertionError('Writer published metadata beyond its reader byte cap')
        _equal_rng(capture_rng(), before)
        dist.barrier()
        assert pointer.read_bytes() == prior_pointer
        assert not list((run / 'distributed-checkpoints').glob('.pending-*'))
        assert not list((run / 'distributed-checkpoints' / '.prepared').rglob('command-*'))
        (root / f'rank-{rank}.json').write_text(json.dumps({'passed': True}))
    finally:
        dist.destroy_process_group()


def test_checkpoint_identity_observation_and_metadata_limit_preserve_durable_state(tmp_path):
    (tmp_path / 'run').mkdir()
    processes = []
    deadline = time.monotonic() + 45
    try:
        for rank in range(2):
            processes.append(subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []),
                str(Path(__file__).resolve()), str(rank), str(tmp_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            output, error = process.communicate(timeout=max(.1, deadline - time.monotonic()))
            assert process.returncode == 0, output + error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    assert [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)] == [
        {'passed': True}, {'passed': True}]


if __name__ == '__main__':
    _worker(int(sys.argv[1]), sys.argv[2])

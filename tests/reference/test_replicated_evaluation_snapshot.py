"""Immutable interval evaluation handoff at an actual two-rank CPU boundary."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


DRIVER = '''
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hypergan.checkpoints import capture_rng
from hypergan.config import resolve_config
from hypergan.distributed_checkpoints import _digest
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.replicated_worker import handle_command


def digest(trainer, batch):
    return _digest({'rng': capture_rng(), 'step': trainer.step, 'batch': batch,
        'streams': {name: stream.get_state() for name, stream in trainer.streams.items()},
        'data': vars(trainer.data), 'base_lrs': trainer.base_lrs,
        'optimizers': [trainer.opt_g.state_dict(), trainer.opt_d.state_dict()],
        'models': {name: {'state': getattr(trainer, name).state_dict(),
            'buffers': dict(getattr(trainer, name).named_buffers()),
            'grads': {key: value.grad for key, value in getattr(trainer, name).named_parameters()},
            'modes': {key: value.training for key, value in getattr(trainer, name).named_modules()}}
            for name in ('graph', 'prior', 'ema_graph', 'ema_prior')}})


def worker(rank, root, mode):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method='file://' + str(root / 'rendezvous'), timeout=timedelta(seconds=30))
    try:
        config = resolve_config({'prior': {'args': {'num_particles': 32, 'z_dim': 4}},
            'training': {'device': 'cpu', 'batch_size': 8}, 'sampling': {'count': 4}})
        trainer = ReplicatedTrainer(config, world_size=2)
        _, batch = trainer.update()
        context = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1,
                   'attempt_dir': str(root)}
        identity = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1,
                    'evaluation_id': 'eval-1'}
        state = {'rank': rank, 'trainer': trainer, 'batch': batch, 'context': context}
        before = digest(trainer, batch)
        path = root / '.evaluation-test' / 'snapshot.pt'
        if mode == 'capture-failure':
            if rank == 0:
                import hypergan.evaluation_snapshot as snapshots
                def fail(*args):
                    raise ValueError('injected snapshot serialization failure')
                snapshots.capture_evaluation_state = fail
            try:
                handle_command(state, 'evaluation-snapshot', {'path': str(path), 'identity': identity})
            except RuntimeError as error:
                assert 'injected snapshot serialization failure' in str(error)
            else:
                raise AssertionError('capture failure did not propagate to every rank')
            assert trainer._poisoned and not trainer.checkpoint_ready
            assert not path.exists()
            (root / f'rank-{rank}.json').write_text(json.dumps({'poisoned': True}))
            return
        result = handle_command(state, 'evaluation-snapshot', {'path': str(path), 'identity': identity})
        assert digest(trainer, batch) == before
        assert result['step'] == 1 and result['ready'] is True
        if rank == 0:
            saved = torch.load(path, weights_only=True)
            assert saved['kind'] == 'ema-inference' and saved['step'] == 1
            assert saved['identity'] == identity
            assert 'discriminator' not in saved['model_states']
            assert result['snapshot']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            frozen_digest = _digest(saved)
        _, state['batch'] = trainer.update()
        if rank == 0:
            assert _digest(torch.load(path, weights_only=True)) == frozen_digest
            assert trainer.step == 2
        (root / f'rank-{rank}.json').write_text(json.dumps({'step': trainer.step, 'preserved': True}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(sys.argv[1], sys.argv[2]), nprocs=2, join=True)
'''


@pytest.mark.heavy
@pytest.mark.parametrize('mode', ['preserve-state', 'capture-failure'])
def test_two_rank_evaluation_snapshot_preserves_state_or_propagates_failure(tmp_path, mode):
    (tmp_path / '.evaluation-test').mkdir()
    driver = tmp_path / 'snapshot_driver.py'
    driver.write_text(DRIVER)
    result = subprocess.run([sys.executable, str(driver), str(tmp_path), mode],
                            capture_output=True, text=True, timeout=90,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
    assert all((tmp_path / f'rank-{rank}.json').is_file() for rank in range(2))


@pytest.mark.parametrize('failure', ['identity', 'empty-id', 'wrong-prefix', 'outside', 'symlink', 'existing'])
def test_worker_rejects_invalid_evaluation_snapshot_and_poisons_boundary(tmp_path, failure):
    from types import SimpleNamespace
    from hypergan.replicated_worker import handle_command
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        directory = tmp_path / '.evaluation-test'
        directory.mkdir()
        path = directory / 'snapshot.pt'
        identity = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1, 'evaluation_id': 'eval'}
        context = {**identity, 'attempt_dir': str(tmp_path)}
        if failure == 'identity':
            identity['attempt_index'] = True
        elif failure == 'empty-id':
            identity['evaluation_id'] = ''
        elif failure == 'wrong-prefix':
            path = tmp_path / 'plain' / 'snapshot.pt'
            path.parent.mkdir()
        elif failure == 'outside':
            path = tmp_path.parent / '.evaluation-outside' / 'snapshot.pt'
        elif failure == 'symlink':
            link = tmp_path / '.evaluation-link'
            link.symlink_to(directory, target_is_directory=True)
            path = link / 'snapshot.pt'
        elif failure == 'existing':
            path.write_bytes(b'preserved')
        trainer = SimpleNamespace(checkpoint_ready=True, _poisoned=False, device=torch.device('cpu'), step=1,
                                  _phase=lambda name, function: function())
        state = {'rank': 0, 'trainer': trainer, 'batch': {}, 'context': context}
        with pytest.raises(ValueError, match='destination or identity'):
            handle_command(state, 'evaluation-snapshot', {'identity': identity, 'path': str(path)})
        assert trainer._poisoned and not trainer.checkpoint_ready
        if failure == 'existing':
            assert path.read_bytes() == b'preserved'
    finally:
        torch.set_num_threads(previous)

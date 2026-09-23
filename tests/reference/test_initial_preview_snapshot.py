"""Step-zero previews observe calibrated EMA without consuming training data."""
import copy
import random

import numpy as np
import pytest
import torch

from hypergan.checkpoints import capture_rng
from hypergan.config import DEFAULT, resolve_config
from hypergan.distributed_checkpoints import _digest
from hypergan.preview_snapshot import capture_snapshot_state
from hypergan.training import ReferenceTrainer


class StatefulData:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def state_dict(self):
        return {'calls': self.calls}

    def load_state_dict(self, state):
        self.calls = state['calls']
        # A legal custom restore hook may consume global randomness.
        random.random()
        np.random.random()
        torch.rand(())

    def __call__(self, count, *, generator):
        self.calls += 1
        offset = random.random() + np.random.random() + torch.rand(()).item()
        real = torch.randn(count, 2, generator=generator) + offset + self.calls
        if self.fail:
            raise RuntimeError('initial data draw failed')
        return {'real': real}


class HiddenData:
    def __call__(self, *args, **kwargs):
        pytest.fail('unsupported hidden state must be rejected before data draw')


def _trainer(data=None):
    values = copy.deepcopy(DEFAULT)
    values['training']['device'] = 'cpu'
    trainer = ReferenceTrainer(resolve_config(values))
    if data is not None:
        trainer.data = data
        trainer.config['data'] = {'factory': f'{__name__}:{type(data).__name__}', 'args': {}}
    return trainer


def _state(trainer):
    data = trainer.data.state_dict() if hasattr(trainer.data, 'state_dict') else None
    return _digest({'rng': capture_rng(), 'data': data,
        'streams': {name: stream.get_state() for name, stream in trainer.streams.items()},
        'models': {name: getattr(trainer, name).state_dict()
                   for name in ('graph', 'prior', 'ema_graph', 'ema_prior')},
        'optimizers': [trainer.opt_g.state_dict(), trainer.opt_d.state_dict()], 'step': trainer.step})


def test_initial_preview_snapshots_calibrated_ema_and_preserves_next_training_batch():
    trainer = _trainer(StatefulData())
    # Simulate accepted calibration copied into initial EMA. Distinct online
    # weights ensure capture demonstrably uses EMA rather than the live graph.
    with torch.no_grad():
        for value in trainer.ema_graph.models['generator'].parameters():
            value.mul_(.75)
    before = _state(trainer)
    snapshot = capture_snapshot_state(trainer, None, {'sample_sequence': 1})
    assert _state(trainer) == before
    assert snapshot['step'] == 0
    assert set(snapshot['model_states']) == {'generator'}
    for name, value in trainer.ema_graph.models['generator'].state_dict().items():
        assert torch.equal(snapshot['model_states']['generator'][name], value.cpu())
    assert any(not torch.equal(snapshot['model_states']['generator'][name], value.cpu())
               for name, value in trainer.graph.models['generator'].state_dict().items())
    next_batch = trainer.batch()
    count = len(snapshot['batch']['real'])
    assert torch.equal(snapshot['batch']['real'], next_batch['real'][:count])
    assert trainer.data.calls == 1 and trainer.step == 0


def test_initial_builtin_stateless_preview_preserves_rng_and_first_batch():
    trainer = _trainer()
    before = _state(trainer)
    snapshot = capture_snapshot_state(trainer, None, {})
    assert _state(trainer) == before
    count = len(snapshot['batch']['real'])
    assert torch.equal(snapshot['batch']['real'], trainer.batch()['real'][:count])


@pytest.mark.parametrize('failure', ['draw', 'snapshot'])
def test_failed_initial_preview_restores_data_and_rng(monkeypatch, failure):
    import hypergan.preview_snapshot as module
    trainer = _trainer(StatefulData(fail=failure == 'draw'))
    before = _state(trainer)
    if failure == 'snapshot':
        monkeypatch.setattr(module, 'MAX_SNAPSHOT_BYTES', 1)
    with pytest.raises((RuntimeError, ValueError)):
        capture_snapshot_state(trainer, None, {})
    assert _state(trainer) == before


def test_initial_preview_rejects_unrestorable_data_and_late_missing_batches():
    trainer = _trainer(HiddenData())
    before = _state(trainer)
    with pytest.raises(ValueError, match='stateless data or paired'):
        capture_snapshot_state(trainer, None, {})
    assert _state(trainer) == before
    trainer.step = 1
    with pytest.raises(ValueError, match='only before the first update'):
        capture_snapshot_state(trainer, None, {})


def test_replicated_initial_preview_draws_rank_zero_shard_without_collectives():
    trainer = _trainer(StatefulData())
    trainer.world_size, trainer.rank = 2, 0
    trainer.global_batch_size = trainer.config['training']['batch_size']
    trainer.local_batch_size = trainer.global_batch_size // 2
    trainer.batch = lambda: pytest.fail('rank-zero preview must not enter collective batch method')
    before = _state(trainer)
    snapshot = capture_snapshot_state(trainer, None, {})
    assert _state(trainer) == before
    expected = trainer.data(trainer.global_batch_size, generator=trainer.streams['data'])['real']
    assert len(snapshot['batch']['real']) <= trainer.local_batch_size
    assert torch.equal(snapshot['batch']['real'], expected[:len(snapshot['batch']['real'])])


@pytest.mark.heavy
def test_two_rank_initial_preview_preserves_boundary_and_next_update(tmp_path):
    import os
    import subprocess
    import sys
    driver = tmp_path / 'initial_preview_driver.py'
    driver.write_text('''
from datetime import timedelta
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


def worker(rank, root):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method='file://' + str(root / 'rendezvous'), timeout=timedelta(seconds=20))
    try:
        trainer = ReplicatedTrainer(resolve_config({'training': {'device': 'cpu', 'batch_size': 8},
            'prior': {'args': {'num_particles': 32, 'z_dim': 4}}, 'sampling': {'count': 4}}), world_size=2)
        context = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1, 'attempt_dir': str(root)}
        identity = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1, 'sample_sequence': 1}
        state = {'trainer': trainer, 'rank': rank, 'batch': None, 'context': context}
        def digest():
            return _digest({'rng': capture_rng(), 'step': trainer.step,
                'streams': {name: stream.get_state() for name, stream in trainer.streams.items()},
                'modules': {name: getattr(trainer, name).state_dict()
                            for name in ('graph', 'prior', 'ema_graph', 'ema_prior')}})
        before = digest()
        path = root / '.preview-initial' / 'snapshot.pt'
        result = handle_command(state, 'preview-snapshot', {'identity': identity, 'path': str(path)})
        assert result['step'] == 0 and result['ready'] and digest() == before
        if rank == 0:
            snapshot = torch.load(path, weights_only=True)
            assert snapshot['step'] == 0
            for name, value in trainer.ema_graph.models['generator'].state_dict().items():
                assert torch.equal(snapshot['model_states']['generator'][name], value)
        dist.barrier()
        trainer.update()
        assert trainer.step == 1
        (root / ('rank-' + str(rank))).write_text('preserved')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(sys.argv[1],), nprocs=2, join=True)
''')
    (tmp_path / '.preview-initial').mkdir()
    result = subprocess.run([sys.executable, str(driver), str(tmp_path)],
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
    assert all((tmp_path / f'rank-{rank}').read_text() == 'preserved' for rank in range(2))


def test_failed_data_restoration_is_fatal_instead_of_training_after_a_consumed_batch():
    from hypergan.run_controller import FatalExecutionError
    trainer = _trainer(StatefulData())
    def fail(state):
        raise RuntimeError('restore unavailable')
    trainer.data.load_state_dict = fail
    streams = _digest({name: stream.get_state() for name, stream in trainer.streams.items()})
    rng = _digest(capture_rng())
    with pytest.raises(FatalExecutionError, match='failed to restore training data'):
        capture_snapshot_state(trainer, None, {})
    assert _digest({name: stream.get_state() for name, stream in trainer.streams.items()}) == streams
    assert _digest(capture_rng()) == rng

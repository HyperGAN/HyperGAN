"""Snapshot capture insulates live state from custom serialization hooks."""
import copy
import random

import numpy as np
import pytest
import torch

from hypergan.checkpoints import capture_rng
from hypergan.config import DEFAULT, resolve_config
from hypergan.distributed_checkpoints import _digest
from hypergan.preview_snapshot import capture_snapshot, renderer_command, renderer_factory
from hypergan.recipes import MLP
from hypergan.training import ReferenceTrainer


class SerializationHooks(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('serialization_calls', torch.zeros(()), persistent=False)

    def get_extra_state(self):
        self.serialization_calls.add_(1)
        random.random()
        np.random.random()
        torch.rand(())
        return {}

    def set_extra_state(self, state):
        pass


def _trainer():
    raw = copy.deepcopy(DEFAULT)
    raw['components']['generator']['factory'] = f'{__name__}:SerializationHooks'
    raw['prior']['args']['num_particles'] = 32
    trainer = ReferenceTrainer(resolve_config(raw))
    _, batch = trainer.update()
    return trainer, batch


def _state(trainer, batch):
    return _digest({'rng': capture_rng(), 'batch': batch,
        'streams': {key: stream.get_state() for key, stream in trainer.streams.items()},
        'modules': {key: {'parameters': dict(getattr(trainer, key).named_parameters()),
                         'buffers': dict(getattr(trainer, key).named_buffers()),
                         'modes': {name: module.training for name, module in getattr(trainer, key).named_modules()}}
                    for key in ('graph', 'prior', 'ema_graph', 'ema_prior')}})


def test_snapshot_serialization_preserves_live_state_rng_and_excludes_discriminator(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer, batch = _trainer()
        before = _state(trainer, batch)
        path = tmp_path / 'snapshot.pt'
        identity = {'run_id': 'run', 'attempt_id': 'attempt', 'sample_sequence': 1}
        receipt = capture_snapshot(trainer, batch, identity, path)
        assert _state(trainer, batch) == before
        saved = torch.load(path, weights_only=True)
        assert set(saved['model_states']) == {'generator'}
        assert saved['model_buffers']['generator']['serialization_calls'].item() == 1
        assert saved['batch']['real'].shape[0] == 1
        # Actual reconstruction must work without a training process group.
        assert not torch.distributed.is_initialized()
        renderer = renderer_factory(0, 1, str(path), receipt, identity, trainer.step, str(tmp_path / 'preview.json'))
        result = renderer_command(renderer, 'render', None)
        assert result['step'] == trainer.step and result['bytes'] > 0
    finally:
        torch.set_num_threads(old_threads)


def test_failed_bounded_serialization_still_preserves_live_state(tmp_path, monkeypatch):
    import hypergan.preview_snapshot as snapshot
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer, batch = _trainer()
        before = _state(trainer, batch)
        monkeypatch.setattr(snapshot, 'MAX_SNAPSHOT_BYTES', 100)
        with pytest.raises(ValueError, match='snapshot exceeds'):
            capture_snapshot(trainer, batch, {'sample_sequence': 1}, tmp_path / 'snapshot.pt')
        assert _state(trainer, batch) == before
        assert (tmp_path / 'snapshot.pt').stat().st_size <= 100
    finally:
        torch.set_num_threads(old_threads)

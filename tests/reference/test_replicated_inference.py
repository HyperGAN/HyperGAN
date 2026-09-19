"""Final worker inference operates on copies and restores training randomness."""
import copy
import json
import random

import numpy as np
import torch

from hypergan.checkpoints import capture_rng
from hypergan.config import DEFAULT, resolve_config
from hypergan.distributed_checkpoints import _digest
from hypergan.recipes import MLP
from hypergan.replicated_worker import _inference
from hypergan.training import ReferenceTrainer


class MutatingInferenceMLP(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('observed', torch.zeros(()))

    def get_extra_state(self):
        self.observed.add_(1)
        random.random()
        np.random.random()
        torch.rand(())
        return {}

    def set_extra_state(self, state):
        pass

    def forward(self, x):
        if not self.training:
            self.observed.add_(1)
        return super().forward(x)


def test_final_inference_copies_registered_state_and_restores_all_rng(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = copy.deepcopy(DEFAULT)
        raw['components']['generator']['factory'] = f'{__name__}:MutatingInferenceMLP'
        raw['sampling']['count'] = 8
        raw['prior']['args'] = {'num_particles': 32, 'z_dim': 4}
        config = resolve_config(raw)
        trainer = ReferenceTrainer(config)
        _, batch = trainer.update()
        trainer.checkpoint_ready, trainer._poisoned = True, False
        context = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_dir': str(tmp_path)}
        directory = tmp_path / 'inference'
        directory.mkdir()
        state = {'rank': 0, 'trainer': trainer, 'batch': batch, 'context': context}
        before = _digest({'rng': capture_rng(), 'streams': {key: value.get_state() for key, value in trainer.streams.items()},
            'buffers': {key: dict(getattr(trainer, key).named_buffers()) for key in ('graph', 'ema_graph', 'prior', 'ema_prior')},
            'batch': batch})
        result = _inference(state, {'bundle_dir': str(directory), 'identity': dict(context, sample_sequence=1)})
        after = _digest({'rng': capture_rng(), 'streams': {key: value.get_state() for key, value in trainer.streams.items()},
            'buffers': {key: dict(getattr(trainer, key).named_buffers()) for key in ('graph', 'ema_graph', 'prior', 'ema_prior')},
            'batch': batch})
        assert after == before
        assert result['ready'] and result['step'] == trainer.step
        payload = json.loads(open(result['sample_path']).read())
        assert payload['count'] == 8 and payload['shape'] == [8, 2]
        assert payload['identity']['sample_sequence'] == 1
    finally:
        torch.set_num_threads(previous)

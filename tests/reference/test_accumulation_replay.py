"""Replay contracts: canonical RNG, separable sums and concrete incompatibilities."""
from datetime import timedelta
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist

from hypergan.checkpoints import capture_rng, restore_rng
from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedCPUTrainer


class RandomGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 2)

    def forward(self, x):
        return self.project(x) + .01 * (torch.rand(len(x), 2) + random.random() + np.random.random())


class ImpureGenerator(RandomGenerator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x) + .01 * self.calls


class BufferedGenerator(RandomGenerator):
    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.zeros(()), persistent=False)

    def forward(self, x):
        self.counter.add_(1)
        return super().forward(x)


class SumSquared(nn.Module):
    accumulation_reduction = 'sum'

    def forward(self, input, target):
        return (input - target).square().sum()


class UnknownReduction(nn.Module):
    def forward(self, input, target):
        return (input - target).square().mean()


class HugeMean(nn.Module):
    accumulation_reduction = 'mean'

    def forward(self, input, target):
        return input.sum() * 0 + input.new_tensor(2e38)


def _config(generator='mlp', objective=None):
    raw = {'components': {
        'generator': {'factory': generator if generator == 'mlp' else f'{__name__}:{generator}',
                      'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]} if generator == 'mlp' else {},
                      'inputs': {'x': 'latent'}},
        'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}}},
        'prior': {'args': {'num_particles': 20, 'z_dim': 4}},
        'training': {'batch_size': 8, 'steps': 3}, 'gradient_penalty': {'lazy_k': 2}, 'sampling': {'count': 4}}
    if objective:
        raw['objectives'] = [{'factory': f'{__name__}:{objective}', 'inputs': {'input': 'generated', 'target': 'batch.real'}, 'weight': .2, 'detach': ['target']}]
    return resolve_config(raw)


def _draw(trainer, rank):
    ids = torch.tensor([0, 0, 2, 3, 0, 4, 3, 3]).chunk(2)[rank]
    return {'real': torch.linspace(-1, 1, 16).reshape(8, 2).chunk(2)[rank]}, (trainer.prior.z[ids], ids)


def _state(trainer):
    return {name: getattr(trainer, name).state_dict() for name in ('graph', 'prior', 'ema_graph', 'ema_prior', 'opt_g', 'opt_d')}


def _close(actual, expected, path='state'):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6, msg=lambda msg: f'{path}: {msg}')
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _close(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            _close(a, b, f'{path}[{index}]')
    else:
        assert actual == expected, path


def _rng(rank):
    trainer = ReplicatedCPUTrainer(_config('RandomGenerator'), accumulation_steps=2)
    before = capture_rng()
    for _ in range(2):
        torch.rand(2, 2)
        random.random()
        np.random.random()
    expected = capture_rng()
    restore_rng(before)
    streams = {name: stream.get_state() for name, stream in trainer.streams.items()}
    batch, draw = _draw(trainer, rank)
    trainer.update(batch, draw)
    _close(capture_rng(), expected, 'canonical RNG')
    for name, value in streams.items():
        assert torch.equal(trainer.streams[name].get_state(), value), name
    assert trainer.checkpoint_ready


def _sum(rank):
    config = _config(objective='SumSquared')
    reference = ReplicatedCPUTrainer(config)
    accumulated = ReplicatedCPUTrainer(config, accumulation_steps=2)
    for step in range(3):
        expected, _ = reference.update(*_draw(reference, rank))
        actual, _ = accumulated.update(*_draw(accumulated, rank))
        _close(_state(accumulated), _state(reference), f'sum step{step}')
        assert actual['objectives'] == pytest.approx(expected['objectives'], rel=3e-5, abs=3e-6)


def _failures(rank):
    for factory, match in [('ImpureGenerator', 'replay differs'), ('BufferedGenerator', 'mutated registered')]:
        trainer = ReplicatedCPUTrainer(_config(factory), accumulation_steps=2)
        with pytest.raises(RuntimeError, match=match):
            trainer.update(*_draw(trainer, rank))
        assert not trainer.checkpoint_ready and trainer._poisoned
        assert not trainer.opt_d.state and not trainer.opt_g.state
        with pytest.raises(RuntimeError, match='complete boundary'):
            trainer.update(*_draw(trainer, rank))
    with pytest.raises(RuntimeError, match='accumulation_reduction'):
        ReplicatedCPUTrainer(_config(objective='UnknownReduction'), accumulation_steps=2)
    for value in (True, 2.0):
        with pytest.raises(RuntimeError, match='positive integer'):
            ReplicatedCPUTrainer(_config(), accumulation_steps=value if rank else int(value))
    config = _config(objective='HugeMean')
    config['objectives'][0]['weight'] = 1.
    trainer = ReplicatedCPUTrainer(config, accumulation_steps=2)
    metrics, _ = trainer.update(*_draw(trainer, rank))
    assert trainer.checkpoint_ready and metrics['objectives'][0] == pytest.approx(2e38)
    config['objectives'].append(dict(config['objectives'][0]))
    trainer = ReplicatedCPUTrainer(config, accumulation_steps=2)
    with pytest.raises(RuntimeError, match='Nonfinite loss'):
        trainer.update(*_draw(trainer, rank))
    assert not trainer.checkpoint_ready and trainer._poisoned
    assert trainer.opt_d.state and not trainer.opt_g.state
    assert all(p.requires_grad for p in trainer.graph.models['discriminator'].parameters())


def _worker(mode, rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        {'rng': _rng, 'sum': _sum, 'failures': _failures}[mode](rank)
        Path(output).write_text(json.dumps({'passed': True}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['rng', 'sum', 'failures'])
def test_accumulation_replay_contract(tmp_path, mode):
    processes = []
    deadline = time.monotonic() + 60
    try:
        for rank in range(2):
            processes.append(subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve()),
                mode, str(rank), str(tmp_path / 'store'), str(tmp_path / f'{rank}.json')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            stdout, stderr = process.communicate(timeout=max(.1, deadline - time.monotonic()))
            assert process.returncode == 0, stdout + stderr
        assert all(json.loads((tmp_path / f'{rank}.json').read_text())['passed'] for rank in range(2))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)


if __name__ == '__main__':
    _worker(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])

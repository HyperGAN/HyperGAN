"""Independent accumulation numerics, saved-activation bounds and fresh-group recovery."""
import copy
from contextlib import nullcontext
from datetime import timedelta
import gc
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

from hypergan.checkpoints import trainer_state
from hypergan.config import DEFAULT, resolve_config
from hypergan.run_state import run_lock
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint

# Heavy: every test here starts real subprocesses or multi-rank jobs and
# measured at a second or more; see reports/test-durations-2026-09-20.txt.
pytestmark = pytest.mark.heavy


_FORWARD_LIMIT = None
_MAX_FORWARD_ROWS = 0
_BACKWARD_COUNT = 0


def _observe(value):
    global _MAX_FORWARD_ROWS
    _MAX_FORWARD_ROWS = max(_MAX_FORWARD_ROWS, len(value))
    if _FORWARD_LIMIT is not None:
        assert len(value) <= _FORWARD_LIMIT, f'Forward batch {len(value)} exceeds microbatch {_FORWARD_LIMIT}'


class Generator(nn.Module):
    def __init__(self, width=12, depth=2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, width), nn.LeakyReLU(.2),
            *[layer for _ in range(depth - 1) for layer in (nn.Linear(width, width), nn.LeakyReLU(.2))],
            nn.Linear(width, 2))

    def forward(self, x, condition):
        _observe(x)
        value = self.layers(torch.cat((x, condition), dim=1))
        # A1 and accumulated discovery must use the same grad mode.
        return value if torch.is_grad_enabled() else value + .125


class Encoder(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)

    def forward(self, input):
        _observe(input)
        return super().forward(input)


class CubicCritic(nn.Module):
    """No additive relativistic bias null direction; genuine nonlinear b-cap."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.8, -.6]))

    def forward(self, x):
        _observe(x)
        return (x @ self.weight).pow(3).unsqueeze(1)


class MeanObjective(nn.Module):
    resume_stateless = True
    accumulation_reduction = 'mean'

    def forward(self, input, target):
        _observe(input)
        return ((input - target).square().sum(dim=1) + .03 * input.abs().sum(dim=1)).mean()


class StochasticGenerator(Generator):
    def get_extra_state(self):
        # Serializing an unchanged descriptor must not perturb replay randomness.
        random.random()
        np.random.random()
        torch.rand(1)
        return {'fixture': 'stochastic-replay'}

    def set_extra_state(self, state):
        assert state == {'fixture': 'stochastic-replay'}

    def forward(self, x, condition):
        result = super().forward(x, condition)
        return result + .001 * (torch.rand_like(result) + random.random() + float(np.random.random()))


class MutableGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.register_buffer('forward_count', torch.zeros(()), persistent=False)

    def forward(self, x, condition):
        self.forward_count.add_(1)
        return super().forward(x, condition)


class _FailSecondMicroBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        global _BACKWARD_COUNT
        _BACKWARD_COUNT += 1
        if dist.get_rank() == 1 and _BACKWARD_COUNT == 2:
            raise RuntimeError('injected rank1 second-micro backward failure')
        return gradient


class BackwardFailureGenerator(Generator):
    def forward(self, x, condition):
        # Forward is pure: only a real G backward increments the failure counter.
        return _FailSecondMicroBackward.apply(super().forward(x, condition))


class OrderedData:
    def __init__(self):
        self.order, self.cursor = [], 0

    def __call__(self, batch_size, *, generator):
        ids = []
        for _ in range(batch_size):
            if self.cursor == len(self.order):
                self.order = torch.randperm(7, generator=generator).tolist()
                self.cursor = 0
            ids.append(self.order[self.cursor])
            self.cursor += 1
        values = torch.tensor(ids, dtype=torch.float32) / 7
        condition = torch.stack((values + .1, values.square() - .4), dim=1)
        return {'condition': condition, 'real': condition * 1.3 + .2, 'ids': torch.tensor(ids)}

    def state_dict(self):
        return {'order': list(self.order), 'cursor': self.cursor, 'rank': dist.get_rank()}

    def load_state_dict(self, state):
        assert state['rank'] == dist.get_rank()
        self.order, self.cursor = list(state['order']), state['cursor']

    def resume_identity(self):
        return {'fixture': 'seven-deterministic-paired-points'}


def _config(mode='ra', kernel='logistic', rows='sampled_unique', *, memory=False, stochastic=False, mutable=False, backward_failure=False):
    raw = copy.deepcopy(DEFAULT)
    raw['name'] = 'independent/accumulation'
    raw['training'].update(steps=3, batch_size=64 if memory else 16, seed=421, lr_anneal_start=.3)
    raw['data'] = {'factory': f'{__name__}:OrderedData', 'args': {}}
    raw['adversarial'].update(mode=mode, loss_type=kernel)
    raw['prior'] = {'kind': 'mog', 'args': {'num_particles': 12, 'z_dim': 4, 'sigma_rel': .03}}
    raw['prior_regularizer']['rows'] = rows
    raw['gradient_penalty'].update(arm='f_none' if memory else 'b_cap', lazy_k=2, kappa=.03)
    generator = 'BackwardFailureGenerator' if backward_failure else 'MutableGenerator' if mutable else 'StochasticGenerator' if stochastic else 'Generator'
    raw['components'] = {
        'encoder': {'factory': f'{__name__}:Encoder', 'inputs': {'input': 'batch.condition'}},
        'generator': {'factory': f'{__name__}:{generator}',
                      'args': {'width': 384, 'depth': 4} if memory else {},
                      'inputs': {'x': 'latent', 'condition': 'components.encoder'}},
        'discriminator': {'factory': f'{__name__}:CubicCritic', 'inputs': {'x': 'candidate'}},
    }
    raw['objectives'] = [{'factory': f'{__name__}:MeanObjective', 'inputs': {'input': 'generated', 'target': 'batch.real'},
                          'weight': .4, 'detach': ['target']}]
    return resolve_config(raw)


def _draw(trainer, step):
    local = trainer.local_batch_size
    total = trainer.global_batch_size
    values = torch.linspace(-.8, 1.3, total)
    condition = torch.stack((values, values.square() - .2), dim=1).chunk(2)[trainer.rank]
    ids = torch.tensor(([0, 0, 2, 4, 2, 6, 6, 0] * ((total + 7) // 8))[:total]).chunk(2)[trainer.rank]
    eps = torch.linspace(-.3, .5, total * 4).reshape(total, 4).chunk(2)[trainer.rank]
    return {'condition': condition, 'real': condition * (1.3 + step * .05) + .2}, (trainer.prior(ids, eps=eps), ids)


def _equal(actual, expected, *, exact=False, path='state'):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 3e-5, atol=0 if exact else 3e-6,
                                   msg=lambda message: f'{path}: {message}')
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _equal(actual[key], expected[key], exact=exact, path=f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected)):
            _equal(left, right, exact=exact, path=f'{path}[{i}]')
    elif isinstance(expected, float) and not exact:
        assert actual == pytest.approx(expected, rel=3e-5, abs=3e-6), path
    else:
        assert actual == expected, path


def _numeric_case():
    global _FORWARD_LIMIT, _MAX_FORWARD_ROWS
    cases = [(mode, kernel) for mode in ('vanilla', 'rp', 'ra') for kernel in ('logistic', 'hinge', 'wasserstein', 'lsgan')]
    for index, (mode, kernel) in enumerate(cases):
        config = _config(mode, kernel, rows='full' if index % 2 else 'sampled_unique')
        plain = ReplicatedCPUTrainer(config, world_size=2, accumulation_steps=1)
        accumulated = ReplicatedCPUTrainer(config, world_size=2, accumulation_steps=2)
        for step in range(1, 4):
            _FORWARD_LIMIT = plain.local_batch_size
            batch, draw = _draw(plain, step)
            expected_row, expected_batch = plain.update(batch, draw)
            _FORWARD_LIMIT = accumulated.local_batch_size // 2
            _MAX_FORWARD_ROWS = 0
            batch, draw = _draw(accumulated, step)
            actual_row, actual_batch = accumulated.update(batch, draw)
            assert _MAX_FORWARD_ROWS == accumulated.local_batch_size // 2
            expected, actual = trainer_state(plain, expected_batch), trainer_state(accumulated, actual_batch)
            for key in ('graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs', 'step', 'modes', 'trainable', 'buffers'):
                _equal(actual[key], expected[key], path=f'{mode}/{kernel}/step{step}/{key}')
            for key in ('d_loss', 'g_loss', 'prior_loss', 'gradient_penalty', 'objectives'):
                _equal(actual_row[key], expected_row[key], path=f'{mode}/{kernel}/{key}')
            assert expected_row['gradient_penalty'] > 0 if step == 2 else expected_row['gradient_penalty'] == 0
            assert accumulated.checkpoint_ready
    _FORWARD_LIMIT = None


class _HeldTensor:
    def __init__(self, tensor, tracker):
        self.tensor, self.tracker = tensor.detach(), tracker
        self.key = tensor.untyped_storage().data_ptr()
        self.counted = tensor.numel() > 0 and self.key not in tracker.excluded
        if self.counted:
            size, count = tracker.live.get(self.key, (tensor.untyped_storage().nbytes(), 0))
            tracker.live[self.key] = (size, count + 1)
            if count == 0:
                tracker.bytes += size
                tracker.peak = max(tracker.peak, tracker.bytes)

    def __del__(self):
        if self.counted:
            size, count = self.tracker.live[self.key]
            if count == 1:
                del self.tracker.live[self.key]
                self.tracker.bytes -= size
            else:
                self.tracker.live[self.key] = (size, count - 1)


class _ActivationTracker:
    def __init__(self, trainer, batch, draw):
        self.live, self.bytes, self.peak = {}, 0, 0
        tensors = [*trainer.graph.parameters(), *trainer.graph.buffers(), *trainer.prior.parameters(), *trainer.prior.buffers(),
                   *batch.values(), *draw]
        self.excluded = {value.untyped_storage().data_ptr() for value in tensors if isinstance(value, torch.Tensor)}

    def pack(self, tensor):
        return _HeldTensor(tensor, self)

    @staticmethod
    def unpack(held):
        return held.tensor


def _memory_case():
    global _FORWARD_LIMIT
    config = _config(memory=True)
    peaks = []
    for accumulation in (1, 4):
        trainer = ReplicatedCPUTrainer(config, world_size=2, accumulation_steps=accumulation)
        batch, draw = _draw(trainer, 1)
        tracker = _ActivationTracker(trainer, batch, draw)
        _FORWARD_LIMIT = trainer.local_batch_size // accumulation
        with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
            trainer.update(batch, draw)
        gc.collect()
        assert tracker.bytes == 0, f'Saved activation storage survived update: {tracker.bytes}'
        peaks.append(tracker.peak)
    _FORWARD_LIMIT = None
    assert peaks[0] > 100000, peaks
    assert peaks[1] < peaks[0] * .65, f'Accumulation retained too many saved activations: A1={peaks[0]}, A4={peaks[1]}'
    return peaks


def _recovery_case(mode, root):
    trainer = ReplicatedCPUTrainer(_config(stochastic=True), world_size=2, accumulation_steps=2)
    run = root / ('full' if mode == 'full' else 'split')
    with run_lock(run) if trainer.rank == 0 else nullcontext():
        last = None
        if mode == 'resume':
            _, info, last = restore_distributed_checkpoint(run, trainer, {'run_id': 'accumulation'})
            assert info['step'] == trainer.step == 1
        while trainer.step < (1 if mode == 'split' else 3):
            _, last = trainer.update()
        path = save_distributed_checkpoint(run, trainer, last, {'run_id': 'accumulation', 'attempt_id': mode})
        torch.save(trainer_state(trainer, last), root / f'{mode}-rank{trainer.rank}.pt')
        return str(path)


def _second_micro_failure(root):
    global _BACKWARD_COUNT
    trainer = ReplicatedCPUTrainer(_config(backward_failure=True), world_size=2, accumulation_steps=2)
    run = root / 'failure'
    with run_lock(run) if trainer.rank == 0 else nullcontext():
        before = copy.deepcopy(trainer_state(trainer, None))
        saved = save_distributed_checkpoint(run, trainer, None,
            {'run_id': 'backward-failure', 'attempt_id': 'initial'})
        pointer = run / 'distributed-checkpoints/latest.json'
        pointer_before = pointer.read_bytes()
        _BACKWARD_COUNT = 0
        batch, draw = _draw(trainer, 1)
        with pytest.raises(RuntimeError, match='rank1 second-micro backward failure'):
            trainer.update(batch, draw)
        assert _BACKWARD_COUNT == 2, 'Failure must occur during the second real microbatch backward'
        assert trainer._poisoned and not trainer.checkpoint_ready and trainer.step == 0
        after = trainer_state(trainer, None)
        # D completed first; neither the G/prior/aux optimizer nor either EMA committed.
        assert trainer.opt_d.state and all(int(state['step']) == 1 for state in trainer.opt_d.state.values())
        assert any(not torch.equal(value, before['graph'][name]) for name, value in after['graph'].items()
                   if name.startswith('models.discriminator.') and isinstance(value, torch.Tensor))
        _equal(after['optimizers'][0], before['optimizers'][0], exact=True)
        for key in ('prior', 'ema_graph', 'ema_prior', 'base_lrs', 'step'):
            _equal(after[key], before[key], exact=True, path=key)
        for name, value in before['graph'].items():
            if not name.startswith('models.discriminator.'):
                _equal(after['graph'][name], value, exact=True, path=name)
        assert all(parameter.requires_grad for parameter in trainer.graph.models['discriminator'].parameters())
        with pytest.raises(ValueError, match='complete update boundary|half/failed'):
            save_distributed_checkpoint(run, trainer, batch,
                {'run_id': 'backward-failure', 'attempt_id': 'must-not-commit'})
        dist.barrier()
        assert pointer.read_bytes() == pointer_before
        assert json.loads((saved / 'manifest.json').read_text())['step'] == 0
        checkpoint_root = run / 'distributed-checkpoints'
        assert [path for path in checkpoint_root.iterdir() if path.is_dir() and not path.name.startswith('.')] == [saved]
        assert not list((checkpoint_root / '.prepared').rglob('command-*'))


def _worker(mode, rank, root):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root / f'{mode}-rendezvous').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    try:
        result = {'passed': True}
        if mode == 'numerics':
            _numeric_case()
        elif mode == 'memory':
            result['peaks'] = _memory_case()
        elif mode in ('full', 'split', 'resume'):
            result['checkpoint'] = _recovery_case(mode, root)
        elif mode == 'wrong-accumulation':
            trainer = ReplicatedCPUTrainer(_config(stochastic=True), world_size=2, accumulation_steps=4)
            before = copy.deepcopy(trainer_state(trainer, None))
            with pytest.raises(ValueError, match='identity|topology'):
                restore_distributed_checkpoint(root / 'split', trainer, {'run_id': 'accumulation'})
            _equal(trainer_state(trainer, None), before, exact=True)
        elif mode == 'second-micro-failure':
            _second_micro_failure(root)
        elif mode == 'mutation':
            trainer = ReplicatedCPUTrainer(_config(mutable=True), world_size=2, accumulation_steps=2)
            batch, draw = _draw(trainer, 1)
            with pytest.raises((ValueError, RuntimeError)):
                trainer.update(batch, draw)
            assert not trainer.checkpoint_ready and trainer._poisoned
            assert not trainer.opt_d.state and not trainer.opt_g.state
        else:
            raise AssertionError(mode)
        (root / f'{mode}-rank{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _launch(root, mode):
    processes = []
    deadline = time.monotonic() + 90
    try:
        for rank in range(2):
            processes.append(subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []),
                str(Path(__file__).resolve()), mode, str(rank), str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            output, error = process.communicate(timeout=max(.1, deadline - time.monotonic()))
            assert process.returncode == 0, output + error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    return [json.loads((root / f'{mode}-rank{rank}.json').read_text()) for rank in range(2)]


def test_accumulation_complete_updates_match_unaccumulated_global_objectives(tmp_path):
    assert _launch(tmp_path, 'numerics') == [{'passed': True}, {'passed': True}]


def test_accumulation_bounds_forward_batch_and_peak_saved_activation_storage(tmp_path):
    assert all(result['passed'] for result in _launch(tmp_path, 'memory'))


def test_accumulated_checkpoint_resumes_exactly_in_fresh_worker_group(tmp_path):
    (tmp_path / 'full').mkdir()
    (tmp_path / 'split').mkdir()
    _launch(tmp_path, 'full')
    _launch(tmp_path, 'split')
    _launch(tmp_path, 'resume')
    _launch(tmp_path, 'wrong-accumulation')
    for rank in range(2):
        full = torch.load(tmp_path / f'full-rank{rank}.pt', weights_only=True)
        resumed = torch.load(tmp_path / f'resume-rank{rank}.pt', weights_only=True)
        _equal(resumed, full, exact=True)


def test_mutating_forward_state_cannot_be_replayed_or_checkpointed(tmp_path):
    assert _launch(tmp_path, 'mutation') == [{'passed': True}, {'passed': True}]


def test_second_micro_backward_failure_keeps_prior_durable_checkpoint(tmp_path):
    (tmp_path / 'failure').mkdir()
    assert _launch(tmp_path, 'second-micro-failure') == [{'passed': True}, {'passed': True}]


if __name__ == '__main__':
    _worker(sys.argv[1], int(sys.argv[2]), sys.argv[3])

"""Independent full-update CPU oracles and adversarial collective failures.

The oracle is the established single-process ReferenceTrainer; it never calls the
replicated strategy. Fixed batches and prior noise separate numerical correctness
from rank-owned random streams, which coordinated recovery tests cover separately.
"""
import copy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
from torch import nn
import torch.distributed as dist

from hypergan.checkpoints import trainer_state
from hypergan.config import DEFAULT, resolve_config
from hypergan.training import ReferenceTrainer


class ConditionalExperts(nn.Module):
    """Each rank uses one expert; the global oracle uses both parameters."""
    def __init__(self):
        super().__init__()
        self.positive = nn.Linear(4, 2)
        self.negative = nn.Linear(4, 2)

    def forward(self, x, condition):
        result = x.new_zeros((len(x), 2))
        selected = condition[:, 0] > 0
        if selected.any():
            result[selected] = self.positive(x[selected])
        if (~selected).any():
            result[~selected] = self.negative(x[~selected])
        return result


class _RankNanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float('nan') if dist.get_rank() == 1 else gradient


class PoisonedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 2)

    def forward(self, x):
        return _RankNanGradient.apply(self.project(x))


class DivergentBufferGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 2)
        self.register_buffer('counter', torch.zeros(()), persistent=False)

    def forward(self, x, condition):
        self.counter.add_(condition[:, 0].mean())
        return self.project(x) + self.counter * .001



class DivergentExtraStateGenerator(nn.Module):
    """Portable module extra state is part of replica/checkpoint identity."""
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 2)
        self.counter = 0.0

    def get_extra_state(self):
        return {'counter': self.counter}

    def set_extra_state(self, state):
        self.counter = state['counter']

    def forward(self, x, condition):
        self.counter += float(condition[:, 0].mean())
        return self.project(x) + self.counter * .001


def _config(case='ra'):
    raw = copy.deepcopy(DEFAULT)
    raw['training'].update(steps=3, batch_size=8, seed=817, lr_anneal_start=.3)
    raw['prior'] = {'kind': 'mog', 'args': {'num_particles': 12, 'z_dim': 4, 'sigma_rel': .03}}
    raw['gradient_penalty'].update(lazy_k=2, kappa=.05)
    raw['adversarial']['mode'] = 'ra' if case == 'ra' else 'rp'
    raw['components'] = {
        'encoder': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 2},
                    'inputs': {'input': 'batch.condition'}},
        'generator': {'factory': 'mlp', 'args': {'input_dim': 6, 'output_dim': 2, 'hidden': [8]},
                      'inputs': {'x': 'latent', 'condition': 'components.encoder'}},
        # RA is invariant to additive critic bias. Adam amplifies near-zero
        # reduction-order residuals in that null direction; omit it in this oracle.
        'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                          'inputs': {'input': 'candidate'}},
    }
    raw['objectives'] = [{'factory': 'mse', 'inputs': {'input': 'generated', 'target': 'batch.real'},
                          'weight': .7, 'detach': ['target']}]
    if case in ('experts', 'poison', 'buffers', 'extra-state'):
        factory = {'experts': 'ConditionalExperts', 'poison': 'PoisonedGenerator', 'buffers': 'DivergentBufferGenerator', 'extra-state': 'DivergentExtraStateGenerator'}[case]
        inputs = {'x': 'latent'}
        if case != 'poison':
            inputs['condition'] = 'batch.condition'
        raw['components']['generator'] = {'factory': f'{__name__}:{factory}', 'inputs': inputs}
        raw['components'].pop('encoder')
    return resolve_config(raw)


def _draw(trainer, step, rank=None):
    condition = torch.tensor([[-1.4, .2], [-.7, .8], [-1.1, -.4], [-.3, 1.2],
                              [.2, -.8], [.9, .4], [1.3, -.6], [.6, 1.4]])
    real = condition * (1.2 + step * .1) + .15
    # Repeats both within and across ranks; row population must be globally unique.
    ids = torch.tensor([0, 0, 2, 4, 2, 6, 6, 0])
    eps = torch.linspace(-.4, .7, 32).reshape(8, 4) + step * .02
    if rank is not None:
        condition, real, ids, eps = (value.chunk(2)[rank] for value in (condition, real, ids, eps))
    return {'real': real, 'condition': condition}, (trainer.prior(ids, eps=eps), ids)


def _close(actual, expected, path='state'):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6, msg=lambda message: f'{path}: {message}')
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _close(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _close(left, right, f'{path}[{index}]')
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=3e-5, abs=3e-6), path
    else:
        assert actual == expected, path


def _numerics(rank):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    for case in ('ra', 'experts'):
        config = _config(case)
        reference = ReferenceTrainer(config)
        replica = ReplicatedCPUTrainer(config, world_size=2)
        initial_encoder = copy.deepcopy(reference.graph.models['encoder'].state_dict()) if case == 'ra' else None
        for step in range(1, 4):
            batch, draw = _draw(reference, step)
            expected_row, expected_batch = reference.update(batch, draw)
            local, local_draw = _draw(replica, step, rank)
            actual_row, actual_batch = replica.update(local, local_draw)
            assert replica.checkpoint_ready and replica.step == reference.step == step
            actual = trainer_state(replica, actual_batch)
            expected = trainer_state(reference, expected_batch)
            for key in ('graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs',
                        'buffers', 'modes', 'trainable', 'step'):
                _close(actual[key], expected[key], f'{case}.step{step}.{key}')
            for key in ('d_loss', 'g_loss', 'prior_loss', 'gradient_penalty'):
                _close(actual_row[key], expected_row[key], f'{case}.{key}')
            if step == 2:
                assert expected_row['gradient_penalty'] > 0
            else:
                assert expected_row['gradient_penalty'] == 0
        if initial_encoder is not None:
            assert any(not torch.equal(before, reference.graph.models['encoder'].state_dict()[name])
                       for name, before in initial_encoder.items())
        # Unsampled standardized MoG rows must receive adversarial gradients.
        assert replica.opt_g.state[replica.prior.z]['exp_avg'][[1, 3, 5, 7, 8, 9, 10, 11]].abs().sum() > 0


def _failure(mode, rank):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    config = _config(mode)
    replica = ReplicatedCPUTrainer(config, world_size=2)
    before_g = copy.deepcopy(replica.graph.models['generator'].state_dict())
    batch, draw = _draw(replica, 1, rank)
    try:
        replica.update(batch, draw)
    except (ValueError, RuntimeError) as error:
        assert not replica.checkpoint_ready
        assert all(parameter.requires_grad for parameter in replica.graph.models['discriminator'].parameters())
        if mode == 'poison':
            assert 'finite' in str(error).lower()
            _close(replica.graph.models['generator'].state_dict(), before_g)
            assert not replica.opt_g.state
            assert replica.opt_d.state  # Proves failure happened after the D half-step.
        return {'error': str(error), 'checkpoint_ready': replica.checkpoint_ready}
    raise AssertionError(f'{mode} unexpectedly committed a complete update')


def _worker(mode, rank, rendezvous, result):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=12))
    try:
        if mode == 'numerics':
            _numerics(rank)
            outcome = {'passed': True}
        elif mode in ('poison', 'buffers', 'extra-state'):
            outcome = _failure(mode, rank)
        elif mode == 'accumulation':
            from hypergan.distributed_training import ReplicatedCPUTrainer
            with pytest.raises(ValueError, match='accumulation'):
                ReplicatedCPUTrainer(_config(), world_size=2, accumulation_steps=3)
            outcome = {'rejected': True}
        else:
            raise AssertionError(mode)
        Path(result).write_text(json.dumps(outcome), encoding='utf-8')
    finally:
        dist.destroy_process_group()


def _launch(tmp_path, mode):
    processes = []
    deadline = time.monotonic() + 45
    try:
        for rank in range(2):
            processes.append(subprocess.Popen([sys.executable, *(['-I'] if sys.flags.isolated else []),
                str(Path(__file__).resolve()), mode, str(rank), str(tmp_path / 'rendezvous'),
                str(tmp_path / f'rank{rank}.json')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            output, error = process.communicate(timeout=max(.1, deadline - time.monotonic()))
            assert process.returncode == 0, output + error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    return [json.loads((tmp_path / f'rank{rank}.json').read_text()) for rank in range(2)]


def test_replicated_complete_updates_match_independent_global_oracle(tmp_path):
    assert _launch(tmp_path, 'numerics') == [{'passed': True}, {'passed': True}]


@pytest.mark.parametrize('mode', ['poison', 'buffers', 'extra-state'])
def test_rank_local_gradient_or_buffer_failure_never_commits_complete_step(tmp_path, mode):
    results = _launch(tmp_path, mode)
    assert all(not result['checkpoint_ready'] and result['error'] for result in results)


def test_accumulation_rejects_nondividing_microbatch_count(tmp_path):
    assert _launch(tmp_path, 'accumulation') == [{'rejected': True}, {'rejected': True}]


if __name__ == '__main__':
    _worker(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])

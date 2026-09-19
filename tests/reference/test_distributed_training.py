"""Complete CPU replica updates against the ordinary global-batch trainer."""
from datetime import timedelta
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
from torch import nn

from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.training import ReferenceTrainer


class CubicCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.4], [1.1]]))

    def forward(self, x):
        return (x @ self.weight).pow(3)


def _config(mode='rp', kernel='logistic', nonlinear=False):
    discriminator = {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}}
    if nonlinear:
        discriminator = {'factory': f'{__name__}:CubicCritic', 'inputs': {'x': 'candidate'}}
    return resolve_config({'components': {
        'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
        'discriminator': discriminator},
        'prior': {'args': {'num_particles': 20, 'z_dim': 4}},
        'adversarial': {'mode': mode, 'loss_type': kernel},
        'gradient_penalty': {'lazy_k': 2, 'kappa': .01},
        'training': {'steps': 3, 'batch_size': 8}, 'sampling': {'count': 4}})


def _close(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _close(a, b)
    else:
        assert left == right


def _state(trainer):
    return {'graph': trainer.graph.state_dict(), 'prior': trainer.prior.state_dict(),
            'ema_graph': trainer.ema_graph.state_dict(), 'ema_prior': trainer.ema_prior.state_dict(),
            'g_optimizer': trainer.opt_g.state_dict(), 'd_optimizer': trainer.opt_d.state_dict(),
            'step': trainer.step, 'base_lrs': trainer.base_lrs}


def _parity(rank):
    ids = torch.tensor([0, 0, 2, 2, 3, 4, 4, 5])
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    for mode in ('rp', 'ra', 'vanilla'):
        for kernel in ('logistic', 'hinge', 'wasserstein', 'lsgan'):
            config = _config(mode, kernel)
            replicated = ReplicatedCPUTrainer(config, world_size=2)
            reference = ReferenceTrainer(config)
            for step in range(3):
                local_ids = ids.chunk(2)[rank]
                row, _ = replicated.update({'real': data.chunk(2)[rank]},
                                            (replicated.prior.z[local_ids], local_ids))
                expected, _ = reference.update({'real': data}, (reference.prior.z[ids], ids))
                _close(_state(replicated), _state(reference))
                for key in ('d_loss', 'g_loss', 'prior_loss', 'gradient_penalty'):
                    assert row[key] == pytest.approx(expected[key], rel=3e-5, abs=3e-6)
                assert row['step'] == step + 1 and replicated.checkpoint_ready
    # Nonlinear active b-cap, controlling interpolation by real==fake at D time.
    config = _config(nonlinear=True)
    replicated, reference = ReplicatedCPUTrainer(config), ReferenceTrainer(config)
    for step in range(3):
        with torch.no_grad():
            real = reference.graph.generate(reference.prior.z[ids], {})['generated'].detach()
        local_ids = ids.chunk(2)[rank]
        row, _ = replicated.update({'real': real.chunk(2)[rank]}, (replicated.prior.z[local_ids], local_ids))
        expected, _ = reference.update({'real': real}, (reference.prior.z[ids], ids))
        _close(_state(replicated), _state(reference))
        assert row['gradient_penalty'] == pytest.approx(expected['gradient_penalty'], rel=3e-5, abs=3e-6)
        if step == 1:
            assert row['gradient_penalty'] > 0


def _data_and_none_gradients(rank):
    # Keep the real default architecture/prior in the exercised path. Its rank
    # state must match exactly even where null bias directions prevent tight
    # cross-reduction-order comparisons with the single-process oracle.
    default = ReplicatedCPUTrainer(resolve_config({}))
    for _ in range(2):
        metrics, _ = default.update()
        assert default.checkpoint_ready and all(torch.isfinite(torch.tensor(metrics[key])) for key in ('d_loss', 'g_loss'))
    config = _config()
    trainer = ReplicatedCPUTrainer(config)
    reference = ReferenceTrainer(config)
    full = reference.batch()
    local = trainer.batch()
    assert torch.equal(local['real'], full['real'].chunk(2)[rank])
    # Absent local contributions average as zero; globally absent grads remain
    # None so Adam does not advance an unused parameter's moments/step.
    parameters = trainer._parameters(trainer.opt_g)
    for parameter in parameters:
        parameter.grad = None
    parameters[0].grad = None if rank == 0 else torch.ones_like(parameters[0])
    trainer._reduce_gradients(trainer.opt_g, 'test partial participation')
    assert torch.equal(parameters[0].grad, torch.full_like(parameters[0], .5))
    assert all(parameter.grad is None for parameter in parameters[1:])
    # Gaussian priors have no z table even when configured rows='full'.
    gaussian = copy.deepcopy(config)
    gaussian['prior'] = {'kind': 'gaussian', 'args': {'z_dim': 4}}
    gaussian['prior_regularizer'].update(weight=0., rows='full')
    other = ReplicatedCPUTrainer(gaussian)
    row, _ = other.update()
    assert row['prior_loss'] == 0 and other.checkpoint_ready


def _errors(rank):
    config = _config()
    with pytest.raises(ValueError, match='accumulation_steps=1'):
        ReplicatedCPUTrainer(config, accumulation_steps=2)
    bad = copy.deepcopy(config)
    bad['training']['batch_size'] = 7
    with pytest.raises(ValueError, match='divide evenly'):
        ReplicatedCPUTrainer(bad)
    changed = copy.deepcopy(config)
    if rank:
        changed['training']['seed'] += 1
    with pytest.raises(ValueError, match='configuration'):
        ReplicatedCPUTrainer(changed)
    mismatched = ReplicatedCPUTrainer(config)
    with pytest.raises(ValueError, match='input source'):
        mismatched.update(None if rank == 0 else {'real': torch.ones(4, 2)})
    assert not mismatched.checkpoint_ready
    # Named metadata prevents two different control operations from accidentally
    # accepting compatible-looking payloads on the same process group.
    control = ReplicatedCPUTrainer(config)
    with pytest.raises(ValueError, match='different trainer operations'):
        control._agree('one' if rank == 0 else 'two', None)
    trainer = ReplicatedCPUTrainer(config)
    initial = copy.deepcopy(_state(trainer))
    bad_batch = torch.ones(4, 2)
    if rank:
        bad_batch[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite real data'):
        trainer.update({'real': bad_batch})
    assert not trainer.checkpoint_ready
    _close(initial, _state(trainer))
    with pytest.raises(RuntimeError, match='complete boundary'):
        trainer.update({'real': torch.ones(4, 2)})


def _worker(mode, rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        {'parity': _parity, 'data': _data_and_none_gradients, 'errors': _errors}[mode](rank)
        Path(output).write_text(json.dumps({'passed': True}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['parity', 'data', 'errors'])
def test_complete_replicated_cpu_updates(tmp_path, mode):
    processes = []
    deadline = time.monotonic() + 60
    try:
        for rank in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve()),
                 mode, str(rank), str(tmp_path / 'store'), str(tmp_path / f'{rank}.json')],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
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

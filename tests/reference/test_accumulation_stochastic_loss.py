"""Global RNG and prior controls survive activation replay."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


# Heavy: every test here starts real subprocesses or multi-rank jobs and
# measured at a second or more; see reports/test-durations-2026-09-20.txt.
pytestmark = pytest.mark.heavy


SCRIPT = '''
import copy
from pathlib import Path
import sys
import torch

from hypergan.cpu_workers import launch_cpu_workers


def close(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            close(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            close(a, b)
    else:
        assert left == right


def state(trainer):
    return {name: copy.deepcopy(getattr(trainer, name).state_dict())
            for name in ('graph', 'prior', 'ema_graph', 'ema_prior', 'opt_g', 'opt_d')}


def worker(rank, world_size, directory):
    from hypergan.checkpoints import capture_rng, restore_rng
    from hypergan.config import resolve_config
    from hypergan.distributed_training import ReplicatedCPUTrainer

    for prior_kind in ('learned', 'frozen', 'gaussian'):
        prior = {'kind': 'particles', 'args': {'num_particles': 16, 'z_dim': 4, 'learnable': prior_kind == 'learned'}}
        if prior_kind == 'gaussian':
            prior = {'kind': 'gaussian', 'args': {'z_dim': 4}}
        config = resolve_config({
            'prior': prior,
            'components': {
                'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
                'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}}},
            'gradient_penalty': {'coeff': 0.0},
            'prior_regularizer': {'weight': 0.0, 'rows': 'full'},
            'training': {'batch_size': 8, 'steps': 2}})
        for factor in (2, 4):
            ordinary = ReplicatedCPUTrainer(config, world_size=world_size)
            accumulated = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=factor)
            initial_prior = copy.deepcopy(ordinary.prior.state_dict())
            for step in range(2):
                # Replay must not consume additional global RNG draws.
                start_rng = capture_rng()
                row, batch = ordinary.update()
                end_rng = capture_rng()
                restore_rng(start_rng)
                actual, other_batch = accumulated.update()
                after_rng = capture_rng()
                assert torch.equal(after_rng['torch'], end_rng['torch'])
                assert after_rng['python'] == end_rng['python']
                assert after_rng['numpy'] == end_rng['numpy']
                close(state(accumulated), state(ordinary))
                close(other_batch, batch)
                for key in ('d_loss', 'g_loss', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'lr_scale'):
                    torch.testing.assert_close(torch.tensor(actual[key]), torch.tensor(row[key]), rtol=3e-5, atol=3e-6)
                for name in ordinary.streams:
                    assert torch.equal(ordinary.streams[name].get_state(), accumulated.streams[name].get_state())
                assert accumulated.step == step + 1 and accumulated.checkpoint_ready
            if prior_kind != 'learned':
                close(accumulated.prior.state_dict(), initial_prior)
                assert len(accumulated.opt_g.param_groups) == 1
    # A disconnected explicit latent graph must retain absent gradients, so
    # Adam does not create moments or advance a prior the recipe did not use.
    config = resolve_config({
        'prior': {'kind': 'particles', 'args': {'num_particles': 16, 'z_dim': 4}},
        'components': {
            'generator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 2}, 'inputs': {'input': 'batch.real'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}}},
        'gradient_penalty': {'coeff': 0.0},
        'training': {'batch_size': 8, 'steps': 1}})
    ordinary = ReplicatedCPUTrainer(config, world_size=world_size)
    accumulated = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=2)
    batch = {'real': torch.linspace(-1, 1, 8).reshape(4, 2) + rank * .1}
    ordinary.update(batch, (ordinary.prior.z[:4].square(), None))
    accumulated.update(batch, (accumulated.prior.z[:4].square(), None))
    close(state(accumulated), state(ordinary))
    assert accumulated.prior.z.grad is None
    assert accumulated.prior.z not in accumulated.opt_g.state
    (Path(directory) / f'rank-{rank}.json').write_text('{"passed": true}')


if __name__ == '__main__':
    launch_cpu_workers(worker, args=(sys.argv[1],), timeout=45, collective_timeout=15)
'''


def test_accumulation_preserves_logical_label_randomness_and_prior_controls(tmp_path):
    script = tmp_path / 'stochastic_loss.py'
    script.write_text(SCRIPT)
    result = subprocess.run(
        [sys.executable, *(['-I'] if sys.flags.isolated else []), str(script), str(tmp_path)],
        capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)] == [
        {'passed': True}, {'passed': True}]

"""Factory-owned frozen feature paths survive adversarial updates and recovery."""
import copy
from datetime import timedelta
from pathlib import Path
import subprocess
import sys

import pytest
import torch
# Direct worker entry points need the repository fixture package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.hndl_fixtures import fixture_linear, fixture_network
import torch.distributed as dist

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer


class FeatureCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = fixture_network('features', (2,), (4,))
        self.features.eval().requires_grad_(False)
        self.head = fixture_linear(4, 1)

    def forward(self, x):
        return self.head(self.features(x))


def config():
    return resolve_config({'components': {
        'generator': {'factory': 'linear', 'args': {'in_features': 4, 'out_features': 2}, 'inputs': {'input': 'latent'}},
        'discriminator': {'factory': f'{__name__}:FeatureCritic', 'inputs': {'x': 'candidate'}}},
        'prior': {'args': {'num_particles': 12, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': .001},
        'training': {'steps': 3, 'batch_size': 8}})


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def check(trainer, frozen):
    d = trainer.graph.models['discriminator']
    equal(d.features.state_dict(), frozen)
    assert not d.features.training and not d.features.nodes.n_normalization.training
    assert all(not p.requires_grad and p.grad is None for p in d.features.parameters())
    assert all(p.requires_grad for p in d.head.parameters())
    owned = {id(p) for group in trainer.opt_d.param_groups for p in group['params']}
    assert owned == {id(p) for p in d.head.parameters()}
    # Frozen parameters must not detach candidate gradients or b-cap's second backward.
    x = torch.ones(3, 2, requires_grad=True)
    gradient = torch.autograd.grad(d(x).sum(), x, create_graph=True)[0]
    assert gradient.abs().sum() > 0
    second = torch.autograd.grad(gradient.square().sum(), d.head.weight)[0]
    assert torch.isfinite(second).all() and second.abs().sum() > 0


def test_native_factory_mask_gradients_and_exact_recovery():
    torch.set_num_threads(1)
    trainer = ReferenceTrainer(config())
    frozen = copy.deepcopy(trainer.graph.models['discriminator'].features.state_dict())
    initial_head = trainer.graph.models['discriminator'].head.weight.detach().clone()
    check(trainer, frozen)
    _, batch = trainer.update()
    state = copy.deepcopy(trainer_state(trainer, batch))
    row, batch = trainer.update()
    assert row['gradient_penalty'] > 0
    _, batch = trainer.update()
    expected = copy.deepcopy(trainer_state(trainer, batch))
    restored = ReferenceTrainer(config())
    restore_trainer(restored, state)
    _, batch = restored.update()
    _, batch = restored.update()
    equal(trainer_state(restored, batch), expected)
    check(restored, frozen)
    assert not torch.equal(initial_head, restored.graph.models['discriminator'].head.weight)


def worker(rank, root):
    from hypergan.distributed_training import ReplicatedCPUTrainer
    from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root / 'rendezvous').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    try:
        trainer = ReplicatedCPUTrainer(config())
        frozen = copy.deepcopy(trainer.graph.models['discriminator'].features.state_dict())
        check(trainer, frozen)
        _, batch = trainer.update()
        save_distributed_checkpoint(root / 'run', trainer, batch, {'run_id': 'frozen-fixture', 'attempt_id': 'first'})
        for _ in range(2):
            _, batch = trainer.update()
        expected = copy.deepcopy(trainer_state(trainer, batch))
        restored = ReplicatedCPUTrainer(config())
        restore_distributed_checkpoint(root / 'run', restored, {'run_id': 'frozen-fixture'})
        for _ in range(2):
            _, batch = restored.update()
        equal(trainer_state(restored, batch), expected)
        check(restored, frozen)
    finally:
        dist.destroy_process_group()


@pytest.mark.heavy
def test_replicated_factory_mask_and_complete_checkpoint_recovery(tmp_path):
    (tmp_path / 'run').mkdir()
    processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(rank), str(tmp_path)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for rank in range(2)]
    try:
        for process in processes:
            out, err = process.communicate(timeout=45)
            assert process.returncode == 0, out + err
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


if __name__ == '__main__':
    worker(int(sys.argv[1]), sys.argv[2])

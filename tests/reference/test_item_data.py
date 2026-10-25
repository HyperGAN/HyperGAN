"""ItemData sampler cost and shared helpers."""
import copy

import torch

import hypergan.data
import hypergan.item_data
from hypergan.item_data import ItemData


class Points:
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return torch.tensor([float(index)])


def test_epoch_permutation_is_built_once_per_epoch(monkeypatch):
    calls = []
    original = torch.randperm

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'randperm', counting)
    data = ItemData(f'{__name__}:Points')
    generator = torch.Generator().manual_seed(4)
    drawn = [data(2, generator=generator)['real'].view(-1) for _ in range(10)]
    assert len(calls) == 1  # Formerly one randperm per batch.
    assert sorted(torch.cat(drawn).tolist()) == [float(i) for i in range(20)]
    # A restored sampler rebuilds the same permutation from its epoch seed.
    state, rng = data.state_dict(), generator.get_state()
    expected = data(6, generator=generator)['real']
    restored = ItemData(f'{__name__}:Points')
    restored.load_state_dict(state)
    generator.set_state(rng)
    assert torch.equal(restored(6, generator=generator)['real'], expected)
    # Checkpoint candidates deep-copy the sampler; the cache is derived state.
    assert copy.deepcopy(data)._order_cache is None


def test_loaders_share_one_identity_digest():
    assert hypergan.item_data._digest is hypergan.data._digest

import hashlib
import pickle

import numpy as np
import pytest
import torch

from hypergan import image_data


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # Synthetic bytes with test-only pins exercise the real parser and sampler.
    images = np.arange(12 * 3072, dtype=np.int64).astype(np.uint8).reshape(12, 3072)
    labels = list(range(12))
    path = tmp_path / 'test_batch'
    payload = pickle.dumps({'data': images, 'labels': labels}, protocol=2)
    path.write_bytes(payload)
    monkeypatch.setitem(image_data.CIFAR10_SHA256, 'test_batch', hashlib.sha256(payload).hexdigest())
    return tmp_path, torch.from_numpy(images.reshape(12, 3, 32, 32)).contiguous()


def test_source_sampling_order_and_exact_recovery(dataset):
    root, images = dataset
    data = image_data.CIFAR10Data(root, split='test')
    actual_rng = torch.Generator().manual_seed(24)
    oracle_rng = torch.Generator().manual_seed(24)
    ids = torch.randint(len(images), (7,), generator=oracle_rng)
    expected = images[ids].float() / 127.5 - 1
    flip = torch.rand((7, 1, 1, 1), generator=oracle_rng) < .5
    expected = torch.where(flip, expected.flip(-1), expected)
    actual = data(7, generator=actual_rng)
    assert torch.equal(actual['real'], expected)
    assert torch.equal(actual['labels'], ids)
    assert torch.equal(actual_rng.get_state(), oracle_rng.get_state())
    saved_data, saved_rng = data.state_dict(), actual_rng.get_state()
    following = data(9, generator=actual_rng)
    restored = image_data.CIFAR10Data(root, split='test')
    restored.load_state_dict(saved_data)
    actual_rng.set_state(saved_rng)
    repeated = restored(9, generator=actual_rng)
    for key in following:
        assert torch.equal(following[key], repeated[key])
    assert data.state_dict() == restored.state_dict()


def test_evaluation_no_replacement_augmentation_or_rng_draws(dataset):
    root, images = dataset
    data = image_data.CIFAR10Data(root, split='test', sampling='sequential', horizontal_flip=False)
    rng = torch.Generator().manual_seed(44)
    state = rng.get_state()
    first = data(5, generator=rng)
    saved = data.state_dict()
    rest = data(7, generator=rng)
    assert torch.equal(torch.cat([first['real'], rest['real']]), images.float() / 127.5 - 1)
    assert torch.equal(torch.cat([first['labels'], rest['labels']]), torch.arange(12))
    assert torch.equal(state, rng.get_state())
    with pytest.raises(StopIteration, match='never wraps'):
        data(1, generator=rng)
    data.load_state_dict(saved)
    assert torch.equal(data(7, generator=rng)['real'], rest['real'])


def test_changed_dataset_rejected_before_deserialization(dataset, monkeypatch):
    root, _ = dataset
    (root / 'test_batch').write_bytes(b'changed bytes')
    def forbidden(*args, **kwargs):
        pytest.fail('Changed dataset must not be unpickled')
    monkeypatch.setattr(pickle, 'load', forbidden)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        image_data.CIFAR10Data(root, split='test')


@pytest.mark.parametrize('state', [{}, {'cursor': -1}, {'cursor': True}, {'cursor': 13}, {'cursor': 2, 'extra': 0}])
def test_invalid_sequential_recovery_state(dataset, state):
    data = image_data.CIFAR10Data(dataset[0], split='test', sampling='sequential', horizontal_flip=False)
    with pytest.raises(ValueError, match='recovery cursor'):
        data.load_state_dict(state)

"""Worker completion order and speculative reads must not affect training state."""
import copy
import sys
import threading
import types
import warnings
from concurrent.futures import wait

import pytest
import torch
from PIL import Image

from hypergan.data import ImageFolder


def folder(tmp_path, **kwargs):
    for i in range(11):
        path = tmp_path / ('cats' if i % 2 else 'dogs') / f'{i:02}.png'
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            image = Image.new('RGB', (5, 3))
            image.putdata([(i * 17, x * 13, 255 - x * 11) for x in range(15)])
            image.save(path)
    return ImageFolder(tmp_path, height=8, width=8, resize='pad', labels=True, **kwargs)


@pytest.mark.parametrize('workers,prefetch', [(0, 1), (1, 0), (1, 1), (4, 2)])
def test_exact_order_pixels_rng_and_restore_with_different_batch_sizes(tmp_path, workers, prefetch):
    serial = folder(tmp_path, workers=0)
    threaded = folder(tmp_path, workers=workers, prefetch_batches=prefetch)
    rngs = [torch.Generator().manual_seed(17) for _ in range(2)]
    try:
        assert serial.resume_identity() == threaded.resume_identity()
        for size in (3, 1, 15, 2, 7):
            expected, actual = [data(size, generator=rng) for data, rng in zip((serial, threaded), rngs)]
            for key in expected:
                assert torch.equal(expected[key], actual[key])
            assert actual['real'].is_contiguous()
            assert serial.state_dict() == threaded.state_dict()
            assert torch.equal(rngs[0].get_state(), rngs[1].get_state())
        state, rng_state = threaded.state_dict(), rngs[1].get_state()
        expected = threaded(19, generator=rngs[1])
        threaded.load_state_dict(state)
        assert not threaded._pending
        rngs[1].set_state(rng_state)
        actual = threaded(19, generator=rngs[1])
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
    finally:
        serial.close()
        threaded.close()


def test_workers_decode_concurrently_and_prefetch_without_rng_advance(tmp_path, monkeypatch):
    data = folder(tmp_path, workers=4, shuffle=False)
    original = data._pixels
    barrier = threading.Barrier(4, timeout=5)
    threads = set()

    def pixels(entry):
        threads.add(threading.get_ident())
        barrier.wait()
        return original(entry)

    monkeypatch.setattr(data, '_pixels', pixels)
    rng = torch.Generator().manual_seed(17)
    before = rng.get_state()
    try:
        data(4, generator=rng)
        wait(data._pending.values(), timeout=5)
        assert len(threads) == 4
        assert threading.get_ident() not in threads
        assert len(data._pending) == 4
        assert all(f.done() and f.exception() is None for f in data._pending.values())
        assert data.state_dict()['cursor'] == 4
        assert torch.equal(rng.get_state(), before)
    finally:
        data.close()
    assert data._pool is None and not data._pending


@pytest.mark.parametrize('mutation', ['content', 'symlink'])
def test_prefetched_files_rechecked_and_failed_batch_rolls_back(tmp_path, mutation):
    data = folder(tmp_path, workers=4)
    rng = torch.Generator().manual_seed(17)
    try:
        data(3, generator=rng)
        wait(data._pending.values())
        state, rng_state = data.state_dict(), rng.get_state()
        path = tmp_path / data.entries[state['permutation'][state['cursor']]]['path']
        original = path.read_bytes()
        if mutation == 'content':
            path.write_bytes(b'changed after prefetch')
        else:
            target = tmp_path / 'replacement.bin'
            target.write_bytes(original)
            path.unlink()
            path.symlink_to(target)
        with pytest.raises(ValueError, match='content changed|symlink'):
            data(12, generator=rng)  # Also speculates across an epoch boundary.
        assert data.state_dict() == state
        assert torch.equal(rng.get_state(), rng_state)
        assert not data._pending
        path.unlink()
        path.write_bytes(original)
        resumed = folder(tmp_path, workers=0)
        resumed.load_state_dict(state)
        other_rng = torch.Generator().set_state(rng_state)
        actual, expected = data(12, generator=rng), resumed(12, generator=other_rng)
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
        resumed.close()
    finally:
        data.close()


def test_checkpoint_candidate_copy_drops_live_workers(tmp_path):
    data = folder(tmp_path, workers=4)
    clone = None
    try:
        data(3, generator=torch.Generator())
        memo = {id(m): m for m in list(sys.modules.values()) if isinstance(m, types.ModuleType)}
        clone = copy.deepcopy(data, memo)
        assert clone._pool is None and not clone._pending
        assert clone.state_dict() == data.state_dict()
        assert torch.equal(clone(3, generator=torch.Generator())['real'],
                           data(3, generator=torch.Generator())['real'])
    finally:
        data.close()
        if clone is not None:
            clone.close()


def test_prefetch_failure_is_reported_when_consumed(tmp_path):
    data = folder(tmp_path, workers=4, shuffle=False)
    rng = torch.Generator()
    try:
        # The first request does not consume the bad file. Its speculative
        # failure must not advance/poison the sampler or fail an earlier batch.
        (tmp_path / data.entries[3]['path']).write_bytes(b'changed')
        data(3, generator=rng)
        wait(data._pending.values())
        state, rng_state = data.state_dict(), rng.get_state()
        with pytest.raises(ValueError, match='content changed'):
            data(3, generator=rng)
        assert data.state_dict() == state
        assert torch.equal(rng.get_state(), rng_state)
    finally:
        data.close()


def test_worker_decoding_keeps_warning_filters_and_pillow_limits(tmp_path, monkeypatch):
    data = folder(tmp_path, workers=4)
    rng = torch.Generator()
    try:
        before = list(warnings.filters)
        data(3, generator=rng)
        wait(data._pending.values())
        assert warnings.filters == before
        data.close()
        # Each source is 15 pixels, between Pillow's warning and error limits.
        monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 8)
        with pytest.warns(Image.DecompressionBombWarning), pytest.raises(ValueError, match='Pillow limit'):
            data(1, generator=rng)
    finally:
        data.close()


@pytest.mark.parametrize('kwargs', [{'workers': -1}, {'workers': True}, {'workers': 1.5},
                                  {'prefetch_batches': -1}, {'prefetch_batches': False}])
def test_worker_settings_reject_invalid_values(tmp_path, kwargs):
    with pytest.raises(ValueError, match='nonnegative integer'):
        folder(tmp_path, **kwargs)

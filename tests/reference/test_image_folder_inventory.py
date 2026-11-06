"""Cached ImageFolder inspections and per-rank sharded reads."""
from concurrent.futures import wait
import json
import logging
import os
import time

from PIL import Image
import pytest
import torch

import hypergan.data as data_module
from hypergan.colorization_data import ColorizationData, prepare_manifest
from hypergan.data import ImageFolder


def save(root, name, value=0, size=(5, 3)):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value,) * 3).save(path)
    return path


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "images"
    for i in range(6):
        save(root, f"{i}.png", i * 40)
    return root


@pytest.fixture
def decodes(monkeypatch):
    calls = []
    original = ImageFolder._decode

    def counting(self, content, name):
        calls.append(name)
        return original(self, content, name)
    monkeypatch.setattr(ImageFolder, "_decode", counting)
    return calls


@pytest.fixture
def settled(monkeypatch):
    # Files written by a test are always within the racy window of a scan.
    # Treat timestamps from before the scan as settled instead of sleeping.
    monkeypatch.setattr(data_module, "_RACY_NS", 0)
    time.sleep(0.01)


def loader(root, **kwargs):
    return ImageFolder(root, height=3, width=5, workers=0, **kwargs)


def test_unchanged_files_are_not_decoded_again(folder, tmp_path, decodes, settled):
    cache = tmp_path / "cache"
    uncached = loader(folder, manifest_cache=False)
    first = loader(folder, manifest_cache=cache)
    assert len(decodes) == 12 and len(list(cache.glob("*.json"))) == 1
    decodes.clear()
    warm = loader(folder, manifest_cache=cache)
    assert decodes == []
    # The cache changes neither the pinned identity nor the pixels.
    assert warm.resume_identity() == first.resume_identity() == uncached.resume_identity()
    assert warm._identity_sha256 == uncached._identity_sha256
    warm.load_state_dict(uncached.state_dict())
    assert torch.equal(warm(6, generator=torch.Generator().manual_seed(3))["real"],
                       uncached(6, generator=torch.Generator().manual_seed(3))["real"])


def test_changed_and_added_files_are_inspected_again(folder, tmp_path, decodes, settled):
    cache = tmp_path / "cache"
    original = loader(folder, manifest_cache=cache)
    path = folder / "2.png"
    before = path.stat()
    # Same size and restored mtime: ctime/inode still expose the rewrite.
    save(folder, "2.png", 255)
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    save(folder, "new.png", 7)
    time.sleep(0.01)
    decodes.clear()
    changed = loader(folder, manifest_cache=cache)
    assert sorted(decodes) == ["2.png", "new.png"]
    assert changed.resume_identity() != original.resume_identity()
    assert changed.resume_identity() == loader(folder, manifest_cache=False).resume_identity()


def test_racy_records_are_not_trusted(folder, tmp_path, decodes):
    # Without settling, freshly written files fall inside the racy window.
    loader(folder, manifest_cache=tmp_path / "cache")
    decodes.clear()
    loader(folder, manifest_cache=tmp_path / "cache")
    assert len(decodes) == 6


def test_cached_inventory_is_still_verified_on_every_read(folder, tmp_path, settled):
    cache = tmp_path / "cache"
    loader(folder, manifest_cache=cache)
    path, = cache.glob("*.json")
    content = json.loads(path.read_text())
    content["files"]["0.png"]["entry"]["sha256"] = "0" * 64
    path.write_text(json.dumps(content))
    forged = loader(folder, manifest_cache=cache, shuffle=False)
    with pytest.raises(ValueError, match="content changed since inventory"):
        forged(1, generator=torch.Generator())


def test_damaged_cache_is_ignored_and_rewritten(folder, tmp_path, decodes, settled):
    cache = tmp_path / "cache"
    expected = loader(folder, manifest_cache=cache).resume_identity()
    path, = cache.glob("*.json")
    for damage in (b"{not json", b"[]", json.dumps({"schema_version": 1, "files": {"0.png": {"entry": 1}}}).encode()):
        path.write_bytes(damage)
        decodes.clear()
        assert loader(folder, manifest_cache=cache).resume_identity() == expected
        assert decodes
        assert json.loads(path.read_text())["schema_version"] == 1


def test_cache_location_settings(folder, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(tmp_path / "env"))
    loader(folder)
    assert len(list((tmp_path / "env" / "image-folder").glob("*.json"))) == 1
    loader(folder, manifest_cache=False, mode="L")
    assert len(list((tmp_path / "env" / "image-folder").glob("*.json"))) == 1
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory")
    with caplog.at_level(logging.WARNING):
        assert loader(folder, manifest_cache=blocked).inventory["accepted"] == 6
    assert "cache not written" in caplog.text
    for value in ("", "  ", 3, True):
        with pytest.raises(ValueError, match="manifest_cache"):
            loader(folder, manifest_cache=value)


def test_inspection_uses_the_worker_pool(folder, tmp_path, monkeypatch):
    threads = set()
    original = ImageFolder._inspect

    def recording(self, relative):
        import threading
        threads.add(threading.current_thread().name)
        return original(self, relative)
    monkeypatch.setattr(ImageFolder, "_inspect", recording)
    data = ImageFolder(folder, height=3, width=5, workers=2, manifest_cache=False)
    assert threads and all(name.startswith("image-loader") for name in threads)
    assert [entry["path"] for entry in data.entries] == [f"{i}.png" for i in range(6)]
    data.close()


@pytest.mark.parametrize("prefetch_batches", [1, 2])
def test_sharded_ranks_prefetch_their_own_slice(tmp_path, prefetch_batches):
    root = tmp_path / "images"
    for i in range(32):
        save(root, f"{i:02}.png", i)
    for rank in (0, 1):
        data = ImageFolder(root, height=3, width=5, shuffle=False, workers=2,
                           prefetch_batches=prefetch_batches, manifest_cache=False)
        data(8, generator=torch.Generator(), rank=rank, world_size=2)
        wait(data._pending.values())
        # The next global batches start at 8 and 16; each rank owns 4 of each.
        expected = {start + 4 * rank + i for start in (8, 16)[:prefetch_batches] for i in range(4)}
        assert set(data._pending) == expected
        data.close()


def test_sharded_rank_rejects_invalid_layout(folder):
    data = loader(folder)
    for rank, world_size, batch in ((2, 2, 4), (-1, 2, 4), (0, 2, 3), (True, 2, 4)):
        with pytest.raises(ValueError, match="rank < world_size"):
            data(batch, generator=torch.Generator(), rank=rank, world_size=world_size)


@pytest.fixture
def pinned(tmp_path):
    root = tmp_path / "logos"
    root.mkdir()
    for i in range(24):
        Image.new("RGB", (2, 2), (i * 9, i, 255 - i)).save(root / f"{i:02}.png")
    manifest = tmp_path / "inventory.json"
    info = prepare_manifest(root, manifest, height=2, width=2, workers=2)
    return {"root": root, "manifest": manifest, "manifest_sha256": info["sha256"]}


@pytest.mark.parametrize("workers", [0, 2])
def test_sharded_skip_agrees_across_ranks(pinned, workers):
    # Each rank must exclude the same bad images and refill the same slots,
    # or samplers diverge; formerly the sharded path crashed on reshape.
    def make():
        return ColorizationData(**pinned, workers=workers, bad_image_policy="skip")
    unsharded, ranks = make(), [make(), make()]
    order = list(torch.randperm(len(unsharded.entries), generator=torch.Generator().manual_seed(5)))
    (pinned["root"] / unsharded.entries[int(order[1])]["path"]).unlink()  # In rank 0's slice.
    (pinned["root"] / unsharded.entries[int(order[6])]["path"]).unlink()  # In rank 1's slice.
    try:
        for _ in range(3):
            expected = unsharded(8, generator=torch.Generator().manual_seed(5))
            parts = [data(8, generator=torch.Generator().manual_seed(5), rank=rank, world_size=2)
                     for rank, data in enumerate(ranks)]
            for key in ("real", "gray"):
                assert torch.equal(torch.cat([part[key] for part in parts]), expected[key])
            assert all(len(part["real"]) == 4 for part in parts)
            assert ranks[0].state_dict() == ranks[1].state_dict() == unsharded.state_dict()
        assert len(unsharded.state_dict()["excluded"]) == 2
    finally:
        for data in (unsharded, *ranks):
            data.close()


def test_colorization_ranks_read_only_their_slice(pinned, monkeypatch):
    reads = []
    original = ColorizationData._pixels

    def recording(self, entry):
        reads.append(entry["path"])
        return original(self, entry)
    monkeypatch.setattr(ColorizationData, "_pixels", recording)
    unsharded, stream = ColorizationData(**pinned, workers=0), torch.Generator().manual_seed(2)
    expected = unsharded(8, generator=stream)
    reads.clear()
    parts = []
    for rank in (0, 1):
        data = ColorizationData(**pinned, workers=0)
        generator = torch.Generator().manual_seed(2)
        parts.append(data(8, generator=generator, rank=rank, world_size=2))
        assert torch.equal(generator.get_state(), stream.get_state())
        assert data.state_dict() == unsharded.state_dict()
    assert len(reads) == 8  # Four per rank, not the whole global batch twice.
    for key in ("real", "gray"):
        assert torch.equal(torch.cat([part[key] for part in parts]), expected[key])

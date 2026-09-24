"""Derived pixels accelerate loading without changing source or recovery rules."""
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
import torch
from PIL import Image

from hypergan.checkpoints import read_checkpoint
from hypergan.colorization_data import ColorizationData, prepare_manifest
from hypergan.config import resolve_config, resume_compatible
from hypergan.image_cache import PixelCache
from hypergan.training import resume, train
from tests.reference.test_image_recovery import _equal, _project


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    for i in range(30):
        image = Image.new('RGBA', (7, 11))
        image.putdata([((i * 9 + x) % 256, x * 3 % 256, 255 - i, x * 7 % 256)
                       for x in range(77)])
        mode = ('RGBA', 'RGB', 'L', 'LA', 'P', '1')[i % 6]
        image = image.convert(mode)
        exif = Image.Exif()
        exif[274] = 6
        image.save(root / f'{i:02}.png', exif=exif)
    manifest = tmp_path / 'manifest.json'
    info = prepare_manifest(root, manifest, height=8, width=8, workers=2)
    return {'root': root, 'manifest': manifest, 'manifest_sha256': info['sha256']}


@pytest.mark.parametrize('workers', [0, 4])
def test_cache_is_exact_across_modes_epochs_and_restore(sources, tmp_path, workers, monkeypatch):
    uncached = ColorizationData(**sources, workers=0)
    cached = ColorizationData(**sources, workers=workers, cache_dir=tmp_path / 'cache')
    rngs = [torch.Generator().manual_seed(71) for _ in range(2)]
    try:
        assert uncached.resume_identity() == cached.resume_identity()
        n = len(cached.entries)
        for size in (n, 3, n + 7):
            expected, actual = [d(size, generator=r) for d, r in zip((uncached, cached), rngs)]
            _equal(actual, expected)
            _equal(cached.state_dict(), uncached.state_dict())
            assert torch.equal(rngs[0].get_state(), rngs[1].get_state())
        cached.close()
        # A new loader has no in-memory pixels and must use the persisted cache.
        cached = ColorizationData(**sources, workers=workers, cache_dir=tmp_path / 'cache')
        cached.load_state_dict(uncached.state_dict())
        def no_decode(*args):
            pytest.fail('Warm cache decoded a source')
        monkeypatch.setattr(cached, '_decode', no_decode)
        _equal(cached(n + 5, generator=rngs[1]), uncached(n + 5, generator=rngs[0]))
        _equal(cached.state_dict(), uncached.state_dict())
    finally:
        uncached.close()
        cached.close()


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'bitflip', 'oversized', 'swapped'])
def test_bad_cache_entries_are_regenerated(sources, tmp_path, damage, monkeypatch):
    data = ColorizationData(**sources, workers=0, cache_dir=tmp_path / 'cache')
    entry, other = data.entries[:2]
    expected = data._pixels(entry)
    data._pixels(other)
    path = data._pixel_cache._path(entry['sha256'])
    content = path.read_bytes()
    if damage == 'missing':
        path.unlink()
    elif damage == 'truncated':
        path.write_bytes(content[:41])
    elif damage == 'bitflip':
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif damage == 'oversized':
        path.write_bytes(content + b'x')
    else:
        path.write_bytes(data._pixel_cache._path(other['sha256']).read_bytes())
    original = data._decode
    calls = []
    def decode(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(data, '_decode', decode)
    assert data._pixels(entry) == expected
    assert calls == [1]
    assert path.read_bytes() == content
    assert data._pixels(entry) == expected
    assert calls == [1]


@pytest.mark.parametrize('damage', ['changed', 'missing', 'symlink', 'parent_symlink'])
def test_cache_never_hides_damaged_sources(sources, tmp_path, damage):
    data = ColorizationData(**sources, workers=0, cache_dir=tmp_path / 'cache')
    entry = data.entries[0]
    data._pixels(entry)
    path = sources['root'] / entry['path']
    if damage == 'changed':
        path.write_bytes(b'changed')
    elif damage == 'missing':
        path.unlink()
    elif damage == 'symlink':
        target = tmp_path / 'copy.png'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        # The resolved root itself is replaced, so containment must still fail.
        target = tmp_path / 'moved'
        sources['root'].rename(target)
        sources['root'].symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='Cannot read image'):
        data._pixels(entry)


def test_write_failure_falls_back_without_excluding_sources(sources, tmp_path, monkeypatch, caplog):
    data = ColorizationData(**sources, workers=0, cache_dir=tmp_path / 'cache',
                            bad_image_policy='skip')
    baseline = ColorizationData(**sources, workers=0)
    def full(*args):
        raise OSError('No space left on device')
    monkeypatch.setattr('hypergan.image_cache.os.replace', full)
    _equal(data(8, generator=torch.Generator()), baseline(8, generator=torch.Generator()))
    assert not data._excluded
    assert caplog.text.count('Pixel cache writes disabled') == 1
    assert not list((tmp_path / 'cache').rglob('.pixels-*'))


def test_atomic_cache_sharing_and_namespace_binding(tmp_path):
    caches = [PixelCache(tmp_path, 'namespace', 32) for _ in range(2)]
    sha, pixels = 'f' * 64, bytes(range(32))
    def access(cache):
        for _ in range(30):
            cache.put(sha, pixels)
            assert cache.get(sha) == pixels
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(access, caches))
    other = PixelCache(tmp_path, 'different-preprocessing', 32)
    other._path(sha).parent.mkdir(parents=True)
    other._path(sha).write_bytes(caches[0]._path(sha).read_bytes())
    assert other.get(sha) is None


@pytest.mark.parametrize('value', ['', '  ', 42, False])
def test_invalid_cache_path(sources, value):
    with pytest.raises(ValueError, match='cache_dir'):
        ColorizationData(**sources, cache_dir=value)


def test_cache_resume_exception_keeps_recipe_exact():
    original = resolve_config({'data': {'factory': 'hypergan.colorization_data:ColorizationData'}})
    before = deepcopy(original)
    current = deepcopy(original)
    current['data']['args']['cache_dir'] = '/tmp/pixels'
    assert resume_compatible(current, original, include_observation=True)
    assert resume_compatible(original, current, include_observation=True)
    assert original == before
    changed = deepcopy(current)
    changed['data']['args']['cache_dir'] = '/elsewhere'
    assert resume_compatible(changed, current)
    for key, value in [('manifest_sha256', 'changed'), ('root', '/elsewhere'),
                       ('shuffle', False), ('cache_dir', True)]:
        changed = deepcopy(current)
        changed['data']['args'][key] = value
        assert not resume_compatible(changed, original)
    changed = deepcopy(current)
    changed['training']['lr'] = .5
    assert not resume_compatible(changed, original)


def test_cache_can_be_enabled_and_removed_during_exact_training_resume(tmp_path):
    config, root = _project(tmp_path)
    manifest = tmp_path / 'inventory.json'
    info = prepare_manifest(root, manifest, height=2, width=2, workers=2)
    source = config.read_text()
    start, end = source.index('[data]'), source.index('[components.generator]')
    source = source[:start] + f'''[data]
factory = "hypergan.colorization_data:ColorizationData"
[data.args]
root = {json.dumps(str(root))}
manifest = {json.dumps(str(manifest))}
manifest_sha256 = "{info['sha256']}"
''' + source[end:]
    config.write_text(source)
    baseline, run = tmp_path / 'baseline', tmp_path / 'resumed'
    train(config, baseline)
    train(config, run, stop_after_steps=2)
    config.write_text(source.replace('[data.args]', '[data.args]\ncache_dir = '
                                    + json.dumps(str(tmp_path / 'cache'))))
    stopped = resume(run, config_path=config, require_same_config=True, stop_after_steps=2)
    assert stopped['last_durable_step'] == 4
    config.write_text(source)
    finished = resume(run, config_path=config, require_same_config=True)
    assert finished['status'] == 'complete'
    _, _, expected = read_checkpoint(baseline)
    _, _, actual = read_checkpoint(run)
    _equal(actual, expected)

"""The data bridge preserves CIFAR draw semantics and exact rollback."""
from pathlib import Path

import pytest
import torch
from PIL import Image

from hypergan.colorization_data import prepare_manifest


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[2] / 'research/startup_tuning'
    monkeypatch.syspath_prepend(str(scripts))
    from bridge_data import Logos32Data
    root = tmp_path / 'images'
    root.mkdir()
    originals = {}
    for i in range(3):
        raw = bytes(value for y in range(32) for x in range(32)
                    for value in (x * 8, y * 8, (x + y + i * 10) % 256))
        name = f'{i}.png'
        Image.frombytes('RGB', (32, 32), raw).save(root / name)
        originals[name] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(32, 32, 3).permute(2, 0, 1)
    manifest = tmp_path / 'manifest.json'
    result = prepare_manifest(root, manifest, height=128, width=128, workers=1)
    args = dict(root=str(root), manifest=str(manifest), manifest_sha256=result['sha256'])
    return Logos32Data, args, originals


def test_random_draws_match_cifar_ids_flips_normalization_and_restore(bridge):
    cls, args, originals = bridge
    data = cls(**args)
    ordered = torch.stack([originals[e['path']] for e in data.decoder.entries])
    generator = torch.Generator().manual_seed(31)
    reference = torch.Generator().set_state(generator.get_state())
    state, rng = data.state_dict(), generator.get_state()
    ids = torch.randint(len(ordered), (20,), generator=reference)
    expected = ordered[ids].float() / 127.5 - 1
    flip = torch.rand((20, 1, 1, 1), generator=reference) < .5
    expected = torch.where(flip, expected.flip(-1), expected)
    actual = data(20, generator=generator)['real']
    assert torch.equal(actual, expected)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert data.cursor == 20
    identity = data.resume_identity()
    assert identity['source_inventory']['preprocessing']['height'] == 128
    assert identity['effective_preprocessing']['height'] == 32
    data.load_state_dict(state)
    generator.set_state(rng)
    assert torch.equal(data(20, generator=generator)['real'], expected)
    with pytest.raises(ValueError, match='identity'):
        cls(**args, horizontal_flip=False).load_state_dict(data.state_dict())


def test_sequential_exhaustion_and_decode_failure_restore_rng(bridge, monkeypatch):
    cls, args, originals = bridge
    data = cls(**args, sampling='sequential', horizontal_flip=False)
    generator = torch.Generator().manual_seed(31)
    rng = generator.get_state()
    expected = torch.stack([originals[e['path']] for e in data.decoder.entries]).float() / 127.5 - 1
    assert torch.equal(data(len(expected), generator=generator)['real'], expected)
    assert torch.equal(generator.get_state(), rng)
    state = data.state_dict()
    with pytest.raises(StopIteration):
        data(1, generator=generator)
    assert data.state_dict() == state
    random_data = cls(**args)
    def fail(entry):
        raise OSError('controlled read failure')
    monkeypatch.setattr(random_data.decoder, '_read_bytes', fail)
    with pytest.raises(OSError, match='controlled read failure'):
        random_data(4, generator=generator)
    assert random_data.cursor == 0
    assert torch.equal(generator.get_state(), rng)


def test_bridge_recipe_changes_only_data_bindings(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'research/startup_tuning'
    monkeypatch.syspath_prepend(str(scripts))
    from logos_data_bridge_screen import verify_data_only_config
    config = verify_data_only_config()
    assert config['data']['factory'] == 'bridge_data:Logos32Data'

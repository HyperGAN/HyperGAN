"""Real image fixtures for preprocessing and exact caller-owned sampler recovery."""
import copy
import json
import shutil
import subprocess
import sys

from PIL import Image
import pytest
import torch

from hypergan.config import resolve_config
from hypergan.data import ImageFolder
from hypergan.recipes import construct


def save(root, name, value=0, size=(5, 3), mode="RGB"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    color = value if mode == "L" else (value,) * len(mode)
    Image.new(mode, size, color).save(path)
    return path


def loader(root, **kwargs):
    return ImageFolder(root, height=3, width=5, **kwargs)


def rng(seed=17):
    return torch.Generator().manual_seed(seed)


def test_flat_and_recursive_sorted_inventory_and_exact_rgb_range(tmp_path):
    save(tmp_path, "z.png", 255)
    save(tmp_path, "a.png", 0)
    save(tmp_path, "nested/b.PNG", 127)
    (tmp_path / "README.txt").write_text("not an image")
    data = loader(tmp_path, shuffle=False)
    assert [entry["path"] for entry in data.resume_identity()["entries"]] == ["a.png", "nested/b.PNG", "z.png"]
    assert data.inventory["accepted"] == 3
    assert data.inventory["ignored_paths"] == ["README.txt"]
    batch = data(3, generator=rng())
    assert set(batch) == {"real"}
    assert batch["real"].shape == (3, 3, 3, 5)
    assert batch["real"].dtype == torch.float32
    assert batch["real"].is_contiguous()
    torch.testing.assert_close(batch["real"][:, 0, 0, 0], torch.tensor([-1., 127 / 127.5 - 1, 1.]), atol=1e-7, rtol=0)
    assert loader(tmp_path, recursive=False).inventory["accepted"] == 2
    json.dumps(data.resume_identity(), allow_nan=False)


def test_nonnumeric_labels_stable_map_and_grayscale_rectangles(tmp_path):
    save(tmp_path, "zebras/nested/a.png", 255, mode="L")
    save(tmp_path, "cats/b.png", 0, mode="L")
    data = loader(tmp_path, mode="L", labels=True, shuffle=False)
    assert data.class_map == {"cats": 0, "zebras": 1}
    batch = data(2, generator=rng())
    assert batch["real"].shape == (2, 1, 3, 5)
    assert batch["labels"].dtype == torch.int64
    assert batch["labels"].tolist() == [0, 1]
    assert data.resume_identity()["class_map"] == data.class_map


@pytest.mark.parametrize("source", ["RGB", "L"])
def test_channel_conversion_is_explicit(tmp_path, source):
    save(tmp_path, "a.png", 255, mode=source)
    for mode, channels in (("RGB", 3), ("L", 1)):
        batch = loader(tmp_path, mode=mode)(1, generator=rng())
        assert batch["real"].shape == (1, channels, 3, 5)
        assert torch.equal(batch["real"], torch.ones_like(batch["real"]))


def test_explicit_stretch_crop_and_pad_pixel_semantics(tmp_path):
    image = Image.new("L", (4, 2))
    image.putdata([0, 64, 128, 255] * 2)
    image.save(tmp_path / "a.png")
    expected = {
        "stretch": [[64, 255], [64, 255]],
        "center_crop": [[64, 128], [64, 128]],
        "pad": [[64, 255], [0, 0]],
    }
    for policy, pixels in expected.items():
        data = ImageFolder(tmp_path, height=2, width=2, mode="L", resize=policy, interpolation="nearest")
        actual = data(1, generator=rng())["real"][0, 0]
        torch.testing.assert_close(actual, torch.tensor(pixels, dtype=torch.float32) / 127.5 - 1, rtol=0, atol=0)
    padded = ImageFolder(tmp_path, height=2, width=2, mode="L", resize="pad", interpolation="nearest", fill=255)
    assert torch.equal(padded(1, generator=rng())["real"][0, 0, 1], torch.ones(2))
    with pytest.raises(ValueError, match="explicit resize policy"):
        ImageFolder(tmp_path, height=2, width=2)


@pytest.mark.parametrize("size", [(1, 2), (2, 1), (8, 3), (3, 8)])
@pytest.mark.parametrize("policy", ["pad", "stretch", "center_crop"])
def test_portrait_landscape_and_undersized_inputs(tmp_path, size, policy):
    save(tmp_path, "a.png", 255, size=size)
    result = loader(tmp_path, resize=policy)(1, generator=rng())["real"]
    assert result.shape == (1, 3, 3, 5)
    assert torch.isfinite(result).all() and result.min() >= -1 and result.max() <= 1


def test_exif_orientation_is_applied_before_shape_validation(tmp_path):
    image = Image.new("RGB", (3, 5), (255, 255, 255))
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "a.jpg", exif=exif)
    data = loader(tmp_path)
    assert data.resume_identity()["entries"][0]["source_width"] == 3
    assert data(1, generator=rng())["real"].shape == (1, 3, 3, 5)


@pytest.mark.parametrize("bad", [b"", b"not an image"])
def test_corrupt_file_alongside_valid_image_fails_preflight(tmp_path, bad):
    save(tmp_path, "good.png")
    (tmp_path / "bad.jpg").write_bytes(bad)
    with pytest.raises(ValueError, match="bad.jpg"):
        loader(tmp_path)


def test_truncated_image_zero_images_wrong_path_and_alpha_errors(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        loader(tmp_path / "missing")
    (tmp_path / "readme.txt").write_text("hello")
    with pytest.raises(ValueError, match="accepted=0, ignored_extensions=1"):
        loader(tmp_path)
    image = save(tmp_path, "a.png", 127, size=(100, 100))
    image.write_bytes(image.read_bytes()[:60])
    with pytest.raises(ValueError, match="Cannot decode image"):
        loader(tmp_path, resize="stretch")
    image.unlink()
    save(tmp_path, "a.png", 255, mode="RGBA")
    with pytest.raises(ValueError, match="composite alpha"):
        loader(tmp_path)


def test_label_layout_size_and_file_limits_are_actionable(tmp_path):
    save(tmp_path, "a.png")
    with pytest.raises(ValueError, match="top-level class"):
        loader(tmp_path, labels=True)
    with pytest.raises(ValueError, match="requires recursive"):
        loader(tmp_path, labels=True, recursive=False)
    with pytest.raises(ValueError, match="exceeds max_pixels"):
        ImageFolder(tmp_path, height=1, width=1, resize="stretch", max_pixels=1)
    with pytest.raises(ValueError, match="max_file_bytes"):
        loader(tmp_path, max_file_bytes=1)
    with pytest.raises(ValueError, match="fill only applies"):
        loader(tmp_path, fill=7)


@pytest.mark.parametrize("shuffle", [False, True])
def test_exact_resume_including_epoch_crossing_and_global_rng_isolation(tmp_path, shuffle):
    for i in range(5):
        save(tmp_path, f"{i}.png", i * 50)
    continuous = loader(tmp_path, shuffle=shuffle)
    stream = rng()
    global_state = torch.random.get_rng_state().clone()
    continuous(3, generator=stream)
    saved_data = json.loads(json.dumps(continuous.state_dict()))
    saved_rng = stream.get_state().clone()
    expected = [continuous(n, generator=stream)["real"] for n in (4, 6, 2)]
    restored = loader(tmp_path, shuffle=shuffle)
    restored.load_state_dict(saved_data)
    restored_rng = rng(999)
    restored_rng.set_state(saved_rng)
    for n, batch in zip((4, 6, 2), expected):
        torch.testing.assert_close(restored(n, generator=restored_rng)["real"], batch, rtol=0, atol=0)
    assert restored.state_dict() == continuous.state_dict()
    assert torch.equal(restored_rng.get_state(), stream.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_content_and_class_changes_reject_resume_but_root_relocation_does_not(tmp_path):
    root = tmp_path / "original"
    save(root, "cats/a.png", 0)
    original = loader(root, labels=True)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert loader(relocated, labels=True).resume_identity() == original.resume_identity()
    save(root, "cats/a.png", 255)
    changed = loader(root, labels=True)
    assert changed.resume_identity() != original.resume_identity()
    with pytest.raises(ValueError, match="identity mismatch"):
        changed.load_state_dict(original.state_dict())
    save(relocated, "dogs/a.png", 0)
    assert loader(relocated, labels=True).class_map == {"cats": 0, "dogs": 1}
    with pytest.raises(ValueError, match="identity mismatch"):
        loader(relocated, labels=True).load_state_dict(original.state_dict())
    with pytest.raises(ValueError, match="identity mismatch"):
        loader(tmp_path / "original", labels=True, mode="L").load_state_dict(changed.state_dict())


@pytest.mark.parametrize("shuffle", [False, True])
def test_bad_file_during_batch_restores_sampler_and_rng_transaction(tmp_path, shuffle):
    save(tmp_path, "a.png", 0)
    path = save(tmp_path, "b.png", 255)
    data = loader(tmp_path, shuffle=shuffle)
    stream = rng()
    before_data, before_rng = data.state_dict(), stream.get_state().clone()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        data(3, generator=stream)
    assert data.state_dict() == before_data
    assert torch.equal(stream.get_state(), before_rng)
    save(tmp_path, "b.png", 255)
    assert data(3, generator=stream)["real"].shape == (3, 3, 3, 5)


def test_invalid_sampler_state_rejected_without_mutation(tmp_path):
    save(tmp_path, "a.png")
    save(tmp_path, "b.png")
    data = loader(tmp_path)
    data(1, generator=rng())
    before = data.state_dict()
    changes = [{"permutation": [0, 0]}, {"permutation": [True, 0]}, {"cursor": 3}, {"epoch": 0}, {"schema_version": True}, {"unexpected": 1}]
    for change in changes:
        state = copy.deepcopy(before)
        state.update(change)
        with pytest.raises(ValueError):
            data.load_state_dict(state)
        assert data.state_dict() == before


def test_symlinks_rejected_initially_and_after_preflight(tmp_path):
    save(tmp_path, "a.png")
    data = loader(tmp_path)
    link = tmp_path / "linked.png"
    link.symlink_to(tmp_path / "a.png")
    with pytest.raises(ValueError, match="symlink"):
        loader(tmp_path)
    link.unlink()
    (tmp_path / "a.png").rename(tmp_path / "original.png")
    (tmp_path / "a.png").symlink_to(tmp_path / "original.png")
    with pytest.raises(ValueError, match="symlink"):
        data(1, generator=rng())


def test_registered_factory_and_optional_import_boundaries(tmp_path):
    save(tmp_path, "a.png")
    spec = {"factory": "image_folder", "args": {"root": str(tmp_path), "height": 3, "width": 5}}
    assert resolve_config({"data": spec})["data"] == spec
    assert isinstance(construct(spec), ImageFolder)
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'PIL'}:
            raise ImportError('optional dependencies intentionally absent')
sys.meta_path.insert(0, Block())
from hypergan.config import resolve_config
from hypergan.data import ImageFolder
resolve_config({'data': {'factory':'image_folder', 'args':{'root':sys.argv[1], 'height':3, 'width':5}}})
try:
    ImageFolder(sys.argv[1], height=3, width=5)
except ValueError as error:
    assert 'hypergan[train,image]' in str(error)
else:
    raise AssertionError('missing Pillow must be actionable')
assert 'torch' not in sys.modules and 'PIL' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, cwd=tmp_path)

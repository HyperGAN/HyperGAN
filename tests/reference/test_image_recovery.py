"""CPU image recovery contracts; these tiny modules make no image-quality claim."""
import json
from pathlib import Path
import random

import numpy as np

from PIL import Image
import pytest
import torch
from torch import nn

from hypergan.checkpoints import read_checkpoint
from hypergan.training import resume, train


class ImageGenerator(nn.Module):
    """Registered buffers and stochastic training expose incomplete recovery."""

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(4, 12)
        self.norm = nn.BatchNorm1d(12)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        value = self.dropout(self.norm(self.project(x)))
        # Also draw in eval mode: previews must isolate all supported global RNGs.
        noise = torch.rand_like(value) + random.random() + float(np.random.random())
        return (value + 0.001 * noise).tanh().reshape(-1, 3, 2, 2)


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Linear(12, 1)

    def forward(self, x):
        return self.score(x.flatten(1))


def _image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (value, 255 - value, value // 2)).save(path)


def _project(tmp_path):
    root = tmp_path / "images"
    for index in range(5):
        _image(root / ("cats" if index < 3 else "dogs") / f"{index}.png", index * 45)
    config = tmp_path / "image.toml"
    config.write_text(f'''
name = "test/image-recovery-contract"
[data]
factory = "image_folder"
[data.args]
root = {json.dumps(str(root))}
height = 2
width = 2
labels = true
shuffle = true

[components.generator]
factory = "{__name__}:ImageGenerator"
inputs = {{ x = "latent" }}
[components.discriminator]
factory = "{__name__}:ImageDiscriminator"
inputs = {{ x = "candidate" }}

[prior.args]
num_particles = 20
z_dim = 4
[gradient_penalty]
lazy_k = 2
[training]
steps = 6
batch_size = 3
seed = 71
lr_anneal_start = 0.5
[sampling]
count = 3
seed = 19
''')
    return config, root


def _equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _equal(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path


def test_image_training_resume_matches_buffers_optimizers_rng_and_data(tmp_path):
    config, _ = _project(tmp_path)
    continuous_dir, split_dir = tmp_path / "continuous", tmp_path / "split"
    continuous = train(config, continuous_dir, checkpoint_every=2)
    stopped = train(config, split_dir, checkpoint_every=2, stop_after_steps=3)
    assert stopped["last_durable_step"] == 3
    old_sample = Path(stopped["sample_path"])
    old_bytes = old_sample.read_bytes()
    finished = resume(split_dir)
    assert continuous["status"] == finished["status"] == "complete"
    assert finished["steps"] == 6
    assert old_sample.read_bytes() == old_bytes
    # Includes Adam moments/groups, BN buffers, all priors/EMA, RNG streams,
    # shuffled epoch permutation/cursor and last batch, not just model weights.
    _, _, expected = read_checkpoint(continuous_dir)
    _, _, actual = read_checkpoint(split_dir)
    _equal(actual, expected)
    left = json.loads(Path(continuous["sample_path"]).read_text())
    right = json.loads(Path(finished["sample_path"]).read_text())
    assert left["shape"] == right["shape"] == [3, 3, 2, 2]
    assert left["samples"] == right["samples"]


@pytest.mark.parametrize("change", ["content", "class-map"])
def test_image_resume_rejects_changed_content_or_class_map(tmp_path, change):
    config, root = _project(tmp_path)
    run = tmp_path / "run"
    stopped = train(config, run, stop_after_steps=2, checkpoint_every=1)
    checkpoint, _, before = read_checkpoint(run)
    before_bytes = (checkpoint / "state.pt").read_bytes()
    if change == "content":
        _image(root / "cats/0.png", 255)
    else:
        (root / "cats").rename(root / "aardvarks")
    with pytest.raises(ValueError, match="(?i)(data|identity|dataset)"):
        resume(run)
    assert (checkpoint / "state.pt").read_bytes() == before_bytes
    _, _, after = read_checkpoint(run)
    _equal(after, before)
    assert stopped["last_durable_step"] == 2

"""Real PNG pixels, fresh-process sampling and unchanged CPU training state."""
import base64
import copy
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image
import torch
from torch import nn

from hypergan.artifacts import sample
from hypergan.checkpoints import read_checkpoint
from hypergan.distributed_checkpoints import _digest
from hypergan.image_grids import tensor_grid
from hypergan.previews import publish_preview_payload, render_preview
from hypergan.training import ReferenceTrainer, train, resume
from hypergan.config import DEFAULT, resolve_config


class ImageGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(2, 12)

    def forward(self, x):
        return self.project(x).tanh().reshape(-1, 3, 2, 2)


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(12, 1)

    def forward(self, x):
        return self.project(x.flatten(1))


class ImageData:
    resume_stateless = True

    def __call__(self, count, generator=None):
        return {'real': torch.rand(count, 3, 2, 2, generator=generator).mul(2).sub(1)}


def recipe():
    config = copy.deepcopy(DEFAULT)
    config['components']['generator'].update(factory=__name__ + ':ImageGenerator', args={})
    config['components']['discriminator'].update(factory=__name__ + ':ImageDiscriminator', args={})
    config['data'] = {'factory': __name__ + ':ImageData', 'args': {}}
    config['prior']['args'].update(num_particles=16, z_dim=2)
    config['sampling'].update(count=3, seed=91)
    config['training'].update(steps=4, batch_size=3, device='cpu')
    config['metrics'] = {'preset': 'none'}
    return resolve_config(config)


@pytest.mark.parametrize('channels', [1, 3])
def test_grid_pixels_clamping_rounding_order_and_no_tensor_mutation(channels):
    values = torch.tensor([-2., 0., 2.]).view(3, 1, 1, 1).expand(3, channels, 2, 2).clone()
    before = values.clone()
    encoded, grid = tensor_grid(values, {'step': 2})
    assert torch.equal(values, before)
    assert grid['columns'] == grid['rows'] == 2
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.mode == ('L' if channels == 1 else 'RGB') and image.size == (4, 4)
        for point, value in [((0, 0), 0), ((2, 0), 128), ((0, 2), 255), ((2, 2), 0)]:
            assert image.getpixel(point) == (value if channels == 1 else (value,) * 3)
        assert json.loads(image.info['hypergan'])['step'] == 2


def test_image_grid_limits_and_unsupported_shapes():
    for values in (torch.zeros(1, 2), torch.zeros(1, 4, 2, 2), torch.zeros(65, 1, 1, 1),
                   torch.zeros(1, 3, 2048, 1024), torch.full((1, 1, 1, 1), float('nan'))):
        with pytest.raises(ValueError):
            tensor_grid(values)


def test_fixed_grid_publication_retention_and_tampered_payload_cleanup(tmp_path):
    trainer = ReferenceTrainer(recipe())
    _, batch = trainer.update()
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    first = render_preview(trainer, batch, identity)
    assert first == render_preview(trainer, batch, identity)
    record, _, _ = publish_preview_payload(tmp_path, first, identity, trainer.step, keep=1)
    original = Path(record['image_grid']['path'])
    assert original.read_bytes() == base64.b64decode(first['image_grid']['png_base64'])
    payload = json.loads(Path(record['path']).read_text())
    assert 'png_base64' not in payload['image_grid'] and payload['samples'] == first['samples']
    invalid = copy.deepcopy(first)
    invalid['image_grid']['width'] += 1
    with pytest.raises(ValueError, match='dimensions'):
        publish_preview_payload(tmp_path, invalid, identity, trainer.step, keep=1)
    assert original.exists()
    assert not list((tmp_path / 'previews').glob('.pending-*'))
    identity['sample_sequence'] = 2
    second = render_preview(trainer, batch, identity)
    record, _, _ = publish_preview_payload(tmp_path, second, identity, trainer.step, keep=1)
    assert not original.exists() and Path(record['image_grid']['path']).exists()


def test_image_previews_metrics_disabled_resume_and_fresh_process_png(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = tmp_path / 'images.toml'
        config.write_text(f'''[components.generator]
factory = "{__name__}:ImageGenerator"
args = {{}}
inputs = {{x = "latent"}}
[components.discriminator]
factory = "{__name__}:ImageDiscriminator"
args = {{}}
inputs = {{x = "candidate"}}
[data]
factory = "{__name__}:ImageData"
args = {{}}
[prior.args]
num_particles = 16
z_dim = 2
[sampling]
count = 3
seed = 91
[training]
device = "cpu"
steps = 4
batch_size = 3
[metrics]
preset = "none"
''')
        train(config, tmp_path / 'plain', checkpoint_every=1)
        stopped = train(config, tmp_path / 'viewed', preview_every=1, preview_keep=4,
                        checkpoint_every=1, stop_after_steps=2)
        old_grid = Path(stopped['previews'][0]['image_grid']['path'])
        old_bytes = old_grid.read_bytes()
        finished = resume(tmp_path / 'viewed')
        assert finished['status'] == 'complete' and old_grid.read_bytes() == old_bytes
        assert _digest(read_checkpoint(tmp_path / 'plain')[2]) == _digest(read_checkpoint(tmp_path / 'viewed')[2])
        assert len(finished['previews']) == 4
        output = tmp_path / 'fresh.png'
        # Install the trusted fixture factory in the fresh interpreter, then invoke
        # the actual CLI. hypergan itself resolves from the installed distribution.
        script = f'''import importlib.util, sys
spec = importlib.util.spec_from_file_location({__name__!r}, {str(Path(__file__).resolve())!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from hypergan.cli import main
raise SystemExit(main(["sample", {str(tmp_path / 'viewed')!r}, "--count", "3", "--seed", "123", "--output", {str(output)!r}]))
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        with Image.open(output) as image:
            metadata = json.loads(image.info['hypergan'])
            assert image.size == (4, 4) and metadata['seed'] == 123 and metadata['count'] == 3
            assert metadata['step'] == 4 and metadata['bundle_sha256']
        before = output.read_bytes()
        with pytest.raises(FileExistsError):
            sample(tmp_path / 'viewed', count=3, seed=456, output=output)
        assert output.read_bytes() == before
        other = sample(tmp_path / 'viewed', count=3, seed=456, output=tmp_path / 'other.png')
        assert other.read_bytes() != before
        with pytest.raises(ValueError, match='at most'):
            sample(tmp_path / 'viewed', count=65, output=tmp_path / 'oversized.png')
    finally:
        torch.set_num_threads(old_threads)

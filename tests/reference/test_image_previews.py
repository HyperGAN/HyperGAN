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


def test_isolated_renderer_transports_png_without_touching_parent_state(tmp_path, monkeypatch):
    import importlib
    import shutil
    from hypergan.checkpoints import capture_rng
    from hypergan.preview_snapshot import capture_snapshot
    from hypergan.snapshot_renderer import render_snapshot
    # Spawned workers import this ordinary local factory module independently.
    shutil.copyfile(__file__, tmp_path / 'png_fixture.py')
    monkeypatch.syspath_prepend(str(tmp_path))
    fixture = importlib.import_module('png_fixture')
    trainer = ReferenceTrainer(fixture.recipe())
    _, batch = trainer.update()
    def digest():
        return _digest({'rng': capture_rng(), 'graph': trainer.graph.state_dict(),
                        'ema': trainer.ema_graph.state_dict(), 'prior': trainer.prior.state_dict(),
                        'g_adam': trainer.opt_g.state_dict(), 'd_adam': trainer.opt_d.state_dict(),
                        'streams': {key: value.get_state() for key, value in trainer.streams.items()}})
    before = digest()
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    snapshot = tmp_path / 'snapshot.pt'
    receipt = capture_snapshot(trainer, batch, identity, snapshot)
    payload = render_snapshot(snapshot, receipt, identity, trainer.step, tmp_path / 'render.json', timeout=30)
    assert digest() == before
    assert payload == render_preview(trainer, batch, identity)
    record = publish_preview_payload(tmp_path, payload, identity, trainer.step)[0]
    with Image.open(record['image_grid']['path']) as image:
        assert image.size == (4, 4)


def test_image_previews_metrics_disabled_resume_and_fresh_process_png(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
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
        assert 2 <= len(finished['previews']) <= 4
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


def test_named_samples_publish_real_grid_and_bounded_retention(tmp_path):
    """Samples carry a short stable name; the real batch publishes beside them."""
    from hypergan.previews import DEFAULT_KEEP, KEEP_ALL, sample_name
    assert DEFAULT_KEEP == KEEP_ALL == 0
    trainer = ReferenceTrainer(recipe())
    _, batch = trainer.update()
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1, 'name': 'g'}
    payload = render_preview(trainer, batch, identity)
    assert payload['name'] == 'g' and payload['image_grid']['name'] == 'g'
    assert payload['real_image_grid']['name'] == 'x'
    assert payload['real_image_grid']['source'] == 'batch.real'
    record, index, errors = publish_preview_payload(tmp_path, payload, identity, trainer.step, keep=2)
    assert not errors and record['name'] == 'g' and index['names'] == ['g'] and index['keep'] == 2
    assert index['previews'][-1]['name'] == 'g'
    grid, real = Path(record['image_grid']['path']), Path(record['real_image_grid']['path'])
    assert grid.name == 'grid.png' and real.name == 'real.png' and grid.parent == real.parent
    assert real.read_bytes() == base64.b64decode(payload['real_image_grid']['png_base64'])
    with Image.open(real) as image:
        assert image.size == (4, 4)
        metadata = json.loads(image.info['hypergan'])
        assert metadata['name'] == 'x' and metadata['source'] == 'batch.real'
        assert metadata['step'] == trainer.step
    stored = json.loads(Path(record['path']).read_text())
    assert 'png_base64' not in stored['real_image_grid'] and stored['name'] == 'g'

    for sequence in range(2, 7):
        moment = dict(identity, sample_sequence=sequence)
        record, index, errors = publish_preview_payload(
            tmp_path, render_preview(trainer, batch, moment), moment, trainer.step, keep=2)
        assert not errors
    generations = [entry for entry in (tmp_path / 'previews').iterdir() if entry.is_dir()]
    assert len(index['previews']) == len(generations) == 2
    assert [item['identity']['sample_sequence'] for item in index['previews']] == [5, 6]

    renamed = dict(identity, sample_sequence=7, name='ema:g')
    payload = render_preview(trainer, batch, renamed)
    assert payload['name'] == payload['image_grid']['name'] == 'ema:g'
    record, index, _ = publish_preview_payload(tmp_path, payload, renamed, trainer.step, keep=2)
    assert record['name'] == 'ema:g' and index['names'] == ['ema:g', 'g']

    assert sample_name(None) == 'g' and sample_name('x') == 'x'
    for invalid in ('has space', '', '-leading', 'x' * 17, 5):
        with pytest.raises(ValueError, match='Sample name'):
            sample_name(invalid)
    with pytest.raises(ValueError, match='Sample name'):
        render_preview(trainer, batch, dict(identity, name='not a name'))
    with pytest.raises(ValueError, match='identity, step, name or shape'):
        publish_preview_payload(tmp_path, payload, dict(renamed, name='g'), trainer.step, keep=2)
    for keep in (-1, 2.0, '2', None):
        with pytest.raises(ValueError, match='preview_keep'):
            publish_preview_payload(tmp_path, payload, renamed, trainer.step, keep=keep)


def test_retention_keeps_every_generation_until_a_bound_is_requested(tmp_path):
    """The default history spans the whole run; a bound is an explicit opt-in."""
    from hypergan.previews import DEFAULT_KEEP
    trainer = ReferenceTrainer(recipe())
    _, batch = trainer.update()
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1, 'name': 'g'}

    def publish(root, sequence, **kwargs):
        moment = dict(identity, sample_sequence=sequence)
        payload = render_preview(trainer, batch, moment)
        return publish_preview_payload(root, payload, moment, trainer.step, **kwargs)

    kept = tmp_path / 'kept'
    kept.mkdir()
    for sequence in range(1, 9):
        record, index, errors = publish(kept, sequence)
        assert not errors
    assert index['keep'] == DEFAULT_KEEP and index['retention'] == 'all'
    assert [item['identity']['sample_sequence'] for item in index['previews']] == list(range(1, 9))
    generations = [entry for entry in (kept / 'previews').iterdir() if entry.is_dir()]
    assert len(generations) == 8
    # Every image grid the slider can reach is still on disk, back to the first.
    for item in index['previews']:
        assert Path(item['path']).is_file()
        assert Path(item['image_grid']['path']).is_file()
        assert Path(item['real_image_grid']['path']).is_file()

    bounded = tmp_path / 'bounded'
    bounded.mkdir()
    for sequence in range(1, 9):
        record, index, errors = publish(bounded, sequence, keep=3)
        assert not errors
    assert index['keep'] == 3 and index['retention'] == 'bounded'
    assert [item['identity']['sample_sequence'] for item in index['previews']] == [6, 7, 8]
    generations = [entry for entry in (bounded / 'previews').iterdir() if entry.is_dir()]
    assert len(generations) == 3
    # A run that pruned under an earlier bound keeps publishing once it is lifted.
    record, index, errors = publish(bounded, 9)
    assert not errors and index['retention'] == 'all'
    assert [item['identity']['sample_sequence'] for item in index['previews']] == [6, 7, 8, 9]


def test_index_reuses_published_records_instead_of_rereading_manifests(tmp_path):
    """Publishing is O(new generations), not O(history), for a long run."""
    trainer = ReferenceTrainer(recipe())
    _, batch = trainer.update()
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1, 'name': 'g'}
    for sequence in range(1, 6):
        moment = dict(identity, sample_sequence=sequence)
        publish_preview_payload(tmp_path, render_preview(trainer, batch, moment), moment, trainer.step)
    root = tmp_path / 'previews'
    generations = sorted(entry for entry in root.iterdir() if entry.is_dir())
    reads = []
    original = Path.read_text

    def counted(self, *args, **kwargs):
        if self.name == 'manifest.json':
            reads.append(self)
        return original(self, *args, **kwargs)

    moment = dict(identity, sample_sequence=6)
    payload = render_preview(trainer, batch, moment)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, 'read_text', counted)
        _, index, _ = publish_preview_payload(tmp_path, payload, moment, trainer.step)
    assert len(index['previews']) == 6
    # Only the generation the index does not name yet is read from disk.
    assert [path.parent.name for path in reads] == [sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and entry not in generations)[0]]
    # A damaged index falls back to rereading every generation manifest.
    (root / 'index.json').write_text('{"schema_version": 1}')
    moment = dict(identity, sample_sequence=7)
    payload = render_preview(trainer, batch, moment)
    reads.clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, 'read_text', counted)
        _, index, _ = publish_preview_payload(tmp_path, payload, moment, trainer.step)
    assert len(reads) == 7 and len(index['previews']) == 7


def test_named_previews_reach_the_run_manifest_with_a_configurable_history(tmp_path, monkeypatch):
    """Training publishes named previews; resume inherits the name and retention."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
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
        stopped = train(config, tmp_path / 'named', preview_every=1, preview_keep=4,
                        preview_name='gen', checkpoint_every=1, stop_after_steps=2)
        assert stopped['preview_name'] == 'gen' and stopped['preview_keep'] == 4
        assert stopped['previews'] and not stopped['observation_errors']
        for preview in stopped['previews']:
            assert preview['name'] == preview['identity']['name'] == 'gen'
            assert preview['image_grid']['name'] == 'gen'
            assert preview['real_image_grid']['name'] == 'x'
        finished = resume(tmp_path / 'named', preview_every=1)
        assert finished['preview_name'] == 'gen' and finished['preview_keep'] == 4
        assert 0 < len(finished['previews']) <= 4
        assert all(item['name'] == 'gen' for item in finished['previews'])
    finally:
        torch.set_num_threads(old_threads)

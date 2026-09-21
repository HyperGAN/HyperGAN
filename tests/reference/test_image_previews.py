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
from tests.hndl_fixtures import fixture_network
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
        self.network = fixture_network('image_generator', (2,), (3, 2, 2))

    def forward(self, x):
        return self.network(x)


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('image_discriminator', (3, 2, 2), (1,))

    def forward(self, x):
        return self.network(x)


class ImageData:
    resume_stateless = True

    def __call__(self, count, generator=None):
        return {'real': torch.rand(count, 3, 2, 2, generator=generator).mul(2).sub(1)}


class ColorGenerator(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.network = fixture_network('color_generator', (2 + size * size,), (3, size, size), size=size)

    def forward(self, x, gray):
        return self.network(torch.cat((x, gray.flatten(1)), dim=1))


class ColorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('pooled_discriminator', (3, 2, 2), (1,))

    def forward(self, x):
        return self.network(x)


class RoutedColorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = fixture_network('pooled_discriminator', (1, 2, 2), (2,))

    def forward(self, gray):
        return {'latent': self.network(gray),
                'ids': torch.full((len(gray),), 3, dtype=torch.int64, device=gray.device)}


def test_preview_records_routed_particle_ids_and_disambiguates_input_names():
    config = recipe()
    config['components']['generator'].update(factory=__name__ + ':ColorGenerator',
        inputs={'x': 'components.encoder.latent', 'gray': 'batch.x'})
    config['components']['discriminator'].update(factory=__name__ + ':ColorDiscriminator')
    config['components']['encoder'] = dict(factory=__name__ + ':RoutedColorEncoder', args={},
        inputs={'gray': 'batch.x'}, trainable=True)
    config['sampling']['particle_ids'] = 'components.encoder.ids'
    trainer = ReferenceTrainer(config)
    batch = {'real': torch.zeros(3, 3, 2, 2), 'x': torch.zeros(3, 1, 2, 2)}
    payload = render_preview(trainer, batch, {'run_id': 'run'})
    assert payload['particle_ids'] == [3, 3, 3]
    assert payload['input_image_grid_0']['name'] == 'input:0'
    assert payload['input_image_grid_0']['source'] == 'batch.x'
    with Image.open(io.BytesIO(base64.b64decode(payload['image_grid']['png_base64']))) as image:
        assert json.loads(image.info['hypergan'])['particle_ids'] == [3, 3, 3]


def test_color256_snapshot_png_only_named_inputs_and_tensor_budget(tmp_path):
    from hypergan.preview_snapshot import capture_snapshot, renderer_command
    from hypergan.snapshot_renderer import _read_output
    from hypergan.previews import MAX_BYTES, preview_budget
    config = recipe()
    config['components']['generator'].update(factory=__name__ + ':ColorGenerator',
        inputs={'x': 'latent', 'gray': 'batch.gray'}, args={'size': 256})
    config['components']['discriminator'].update(factory=__name__ + ':ColorDiscriminator')
    config['sampling']['count'] = 8
    trainer = ReferenceTrainer(config)
    batch = {'real': torch.rand(8, 3, 256, 256).mul(2).sub(1),
             'gray': torch.rand(8, 1, 256, 256).mul(2).sub(1)}
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    snapshot, output = tmp_path / 'snapshot.pt', tmp_path / 'render.json'
    descriptor = capture_snapshot(trainer, batch, identity, snapshot)
    receipt = renderer_command((str(snapshot), descriptor, identity, trainer.step, str(output)), 'render', None)
    payload = _read_output(output, receipt, identity, trainer.step)
    assert payload['count'] == 8 and payload['shape'] == [8, 3, 256, 256]
    assert payload['representation'] == 'png' and payload['samples'] is None
    assert payload['inputs']['gray'] == {'shape': [8, 1, 256, 256], 'representation': 'png'}
    assert output.stat().st_size > MAX_BYTES  # Separate bounded PNG transport.
    record, _, _ = publish_preview_payload(tmp_path, payload, identity, trainer.step)
    assert record['bytes'] < 8192 and record['representation'] == 'png'
    for field, name, mode in [('image_grid', 'g', 'RGB'), ('real_image_grid', 'x', 'RGB'),
                              ('input_image_grid_0', 'gray', 'L')]:
        assert record[field]['name'] == name
        with Image.open(record[field]['path']) as image:
            assert image.size == (768, 768) and image.mode == mode
    with pytest.raises(ValueError, match='element'):
        preview_budget(trainer, {'real': torch.zeros(1, 65537)}, {})
    with pytest.raises(ValueError, match='element'):
        preview_budget(trainer, {'real': torch.zeros(1, 3, 2048, 2048)}, {})


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


@pytest.mark.heavy
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


@pytest.mark.heavy
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
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
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
    from hypergan.previews import DEFAULT_KEEP, KEEP_ALL, drain_pruning, sample_name
    assert DEFAULT_KEEP == 128 and KEEP_ALL == 0
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
    assert drain_pruning(30)
    generations = [entry for entry in (tmp_path / 'previews').iterdir() if entry.is_dir()]
    # A bound of two keeps the beginning of the run and its latest sample.
    assert len(index['previews']) == len(generations) == 2
    assert [item['identity']['sample_sequence'] for item in index['previews']] == [1, 6]

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


def test_retention_thins_the_history_instead_of_dropping_its_beginning(tmp_path):
    """A bounded run still reaches back to its first sample; 'all' keeps every one."""
    from hypergan.previews import DEFAULT_KEEP, drain_pruning
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
        record, index, errors = publish(kept, sequence, keep=0)
        assert not errors
    assert index['keep'] == 0 and index['retention'] == 'all'
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
    counts = []
    for sequence in range(1, 25):
        record, index, errors = publish(bounded, sequence, keep=8)
        assert not errors and len(index['previews']) <= 8
        counts.append(len(index['previews']))
    assert index['keep'] == 8 and index['retention'] == 'thinned'
    # The first sample and the latest survive, the middle is thinned to every
    # second sample, and the newest window stays dense.
    sequences = [item['identity']['sample_sequence'] for item in index['previews']]
    assert sequences == [1, 15, 17, 19, 21, 22, 23, 24]
    # Pruning happens in chunks, not one generation per publication.
    assert max(before + 1 - after for before, after in zip(counts, counts[1:])) > 1
    assert drain_pruning(30)
    generations = [entry for entry in (bounded / 'previews').iterdir() if entry.is_dir()]
    assert len(generations) == len(sequences)
    for item in index['previews']:
        assert Path(item['image_grid']['path']).is_file()

    # A run that thinned under a bound keeps publishing once the bound is lifted.
    record, index, errors = publish(bounded, 25, keep=0)
    assert not errors and index['retention'] == 'all'
    assert [item['identity']['sample_sequence'] for item in index['previews']] == [*sequences, 25]
    assert DEFAULT_KEEP == 128


def test_thinning_halves_spacing_and_holds_any_bound(tmp_path):
    """The retained set is nested, bounded and dense at the end of the run."""
    from hypergan.previews import DENSE_WINDOW, KEEP_ALL, thin
    assert thin([], KEEP_ALL) == set() and thin([4, 1], KEEP_ALL) == {1, 4}
    assert thin([1, 2, 3], 5) == {1, 2, 3}
    for keep in (1, 2, 3, 4, 8, 16, 20, DENSE_WINDOW * 8, 128):
        history, previous = set(), set()
        chunk = 0
        for sequence in range(1, 400):
            history.add(sequence)
            retained = thin(history, keep)
            assert len(retained) <= keep
            assert sequence in retained and (keep == 1 or 1 in retained)
            # Nested: a later prune only ever removes, so nothing is re-admitted.
            assert retained <= previous | {sequence}
            chunk = max(chunk, len(previous) + 1 - len(retained))
            history, previous = set(retained), retained
        # Beyond the smallest bounds there is room to prune in chunks.
        assert chunk > 1 or keep <= 3
    # Spacing doubles: a run publishing every sample is thinned to every second,
    # then every fourth, as it outgrows the bound.
    assert sorted(thin(range(1, 10), 5)) == [1, 3, 5, 7, 9]
    assert sorted(thin(range(1, 19), 8))[:4] == [1, 5, 9, 13]
    # A history an older release already thinned is bounded from what is left.
    assert sorted(thin(range(31, 51), 4)) == [31, 47, 49, 50]


def test_pruning_runs_in_the_background_and_never_indexes_what_it_deletes(tmp_path, monkeypatch):
    """A prune rewrites the index at once and deletes off the publishing path."""
    import shutil
    import threading
    from hypergan import previews as module
    trainer = ReferenceTrainer(recipe())
    _, batch = trainer.update()
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1, 'name': 'g'}
    root = tmp_path / 'previews'

    def publish(sequence):
        moment = dict(identity, sample_sequence=sequence)
        return publish_preview_payload(tmp_path, render_preview(trainer, batch, moment),
                                       moment, trainer.step, keep=8)

    for sequence in range(1, 9):
        _, index, errors = publish(sequence)
        assert not errors
    assert module.drain_pruning(30)

    released, removals = threading.Event(), []
    original = shutil.rmtree

    def slow(path, *args, **kwargs):
        removals.append(Path(path))
        assert released.wait(30)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, 'rmtree', slow)
    try:
        # The publication that trips the bound returns without waiting on rmtree.
        _, index, errors = publish(9)
        assert not errors and len(index['previews']) < 9
        assert not module.drain_pruning(0.2)
        expired = sorted(entry for entry in root.iterdir() if entry.name.startswith('.expired-'))
        assert expired, 'a prune should retire directories before deleting them'
        indexed = {Path(item['path']).parent.name for item in index['previews']}
        # The index never names a directory the worker is about to delete.
        assert all(entry.name.removeprefix('.expired-') not in indexed for entry in expired)
        # A publication is not blocked by a prune still in flight, and the scan
        # does not re-admit a retired directory.
        _, later, errors = publish(10)
        assert not errors and 10 in {item['identity']['sample_sequence'] for item in later['previews']}
        assert all(Path(item['path']).parent.name not in
                   {entry.name.removeprefix('.expired-') for entry in expired}
                   for item in later['previews'])
    finally:
        released.set()
    assert module.drain_pruning(30)
    monkeypatch.undo()
    assert removals and not [entry for entry in root.iterdir() if entry.name.startswith('.expired-')]
    # Whatever survives the prune is still readable, back to the first sample.
    _, final, errors = publish(11)
    assert not errors and final['previews'][0]['identity']['sample_sequence'] == 1
    for item in final['previews']:
        assert Path(item['path']).is_file() and Path(item['image_grid']['path']).is_file()


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


@pytest.mark.heavy
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


def test_random_prior_preview_and_paired_reconstruction_keep_distinct_bindings(tmp_path):
    from hypergan.artifacts import save_bundle
    from hypergan.checkpoints import capture_rng
    from hypergan.preview_snapshot import capture_snapshot, renderer_command
    from hypergan.recipes import generation_output
    config = recipe()
    config['components']['encoder'] = dict(factory=__name__ + ':RoutedColorEncoder', args={},
        inputs={'gray': 'batch.gray'}, trainable=True)
    config['components']['reconstruction'] = dict(reuse='generator',
        inputs={'x': 'components.encoder.latent'}, trainable=True, freeze_parameters=True)
    config['sampling'].update(generated='components.reconstruction',
        particle_ids='components.encoder.ids', views={'random': 'components.generator'},
        comparison=[{'label': 'X', 'binding': 'batch.real'}, {'label': 'B', 'binding': 'batch.gray'},
                    {'label': 'X_hat', 'binding': 'components.reconstruction'}])
    config['objectives'] = [dict(factory='mse', inputs={'prediction': 'components.reconstruction', 'target': 'batch.real'})]
    from hypergan.config import config_values
    trainer = ReferenceTrainer(resolve_config(config_values(config)))
    batch = {'real': torch.tensor([-1., 0., 1.]).reshape(3, 1, 1, 1).expand(3, 3, 2, 2),
             'gray': torch.tensor([1., 0., -1.]).reshape(3, 1, 1, 1).expand(3, 1, 2, 2)}
    context = trainer.ema_graph.generate(torch.zeros(3, 2), batch, prior=trainer.ema_prior)
    random_output = context['generated'].clone()
    conditional = generation_output(trainer.ema_graph, context, config['sampling'])
    assert torch.equal(context['generated'], random_output)
    assert not torch.equal(conditional, random_output)
    identity = {'run_id': 'run', 'attempt_id': '0001-' + 'a' * 32, 'sample_sequence': 1}
    before = _digest({'rng': capture_rng(), 'ema': trainer.ema_graph.state_dict()})
    payload = render_preview(trainer, batch, identity)
    assert _digest({'rng': capture_rng(), 'ema': trainer.ema_graph.state_dict()}) == before
    assert payload == render_preview(trainer, batch, identity)
    assert payload['particle_ids'] == [3, 3, 3]
    assert payload['routing'] == {'unique_particles': 1, 'top_particle_share': 1.0}
    assert payload['extra_image_grid_0']['name'] == 'random'
    assert payload['image_grid']['png_base64'] != payload['extra_image_grid_0']['png_base64']
    with Image.open(io.BytesIO(base64.b64decode(payload['extra_image_grid_0']['png_base64']))) as image:
        assert json.loads(image.info['hypergan'])['particle_ids'] is None
        assert json.loads(image.info['hypergan'])['conditioning'] == 'unconditional'
    with Image.open(io.BytesIO(base64.b64decode(payload['comparison_image_grid']['png_base64']))) as image:
        assert image.size == (6, 30)
        assert json.loads(image.info['hypergan'])['grid']['column_labels'] == ['X', 'B', 'X_hat']
        for row, value in enumerate([0, 128, 255]):
            assert image.getpixel((0, 24 + row * 2)) == (value,) * 3
            assert image.getpixel((2, 24 + row * 2)) == ([255, 128, 0][row],) * 3
            expected = conditional[row, :, 0, 0].detach().clamp(-1, 1).add(1).mul(127.5).round()
            assert image.getpixel((4, 24 + row * 2)) == tuple(expected.int().tolist())
    record, _, _ = publish_preview_payload(tmp_path, payload, identity, trainer.step)
    assert Path(record['extra_image_grid_0']['path']).is_file()
    assert Path(record['comparison_image_grid']['path']).is_file()
    snapshot, output = tmp_path / 'snapshot.pt', tmp_path / 'render.json'
    receipt = capture_snapshot(trainer, batch, identity, snapshot)
    renderer_command((str(snapshot), receipt, identity, trainer.step, str(output)), 'render', None)
    assert json.loads(output.read_text()) == payload
    save_bundle(tmp_path, trainer, batch)
    sampled = json.loads(sample(tmp_path, count=3, seed=config['sampling']['seed']).read_text())
    assert sampled['particle_ids'] == [3, 3, 3]
    assert torch.equal(torch.tensor(sampled['samples']), conditional)


def test_comparison_256_columns_fit_and_preview_budget_caps_tall_layout():
    from types import SimpleNamespace
    from hypergan.image_grids import comparison_grid
    from hypergan.previews import preview_budget
    real = torch.zeros(16, 3, 256, 256)
    columns = [{'label': 'X', 'binding': 'batch.real'}, {'label': 'B', 'binding': 'batch.gray'},
               {'label': 'X_hat', 'binding': 'components.reconstruction'}]
    config = {'sampling': {'count': 16, 'comparison': columns}}
    count, _, _, _ = preview_budget(SimpleNamespace(config=config), {'real': real}, {})
    assert count == 15  # A 24px header must fit below the 4096px image bound.
    encoded, grid = comparison_grid([('X', real[:8]), ('B', real[:8, :1]), ('X_hat', real[:8])])
    assert grid['width'] == 768 and grid['height'] == 2072
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.size == (768, 2072)

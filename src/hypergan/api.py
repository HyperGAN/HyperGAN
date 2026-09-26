"""HyperGAN's Python API: a thin, functional layer over model files.

The model file is the source of truth (see ``hypergan.model_file``). This module
loads one, lets a script change values, trains it and reads the run back::

    import hypergan.api as hg

    model = hg.load("model.toml")
    model = hg.override(model, {"train.steps": 200, "losses.reconstruction.weight": 0.5})
    run = hg.train(model, "runs/demo", previews=50)
    hg.metrics(run)["loss/d_total"]        # [(step, value), ...]

Everything here is a plain function over two data objects, ``Model`` (the
model file as a dictionary plus the directory its relative paths use) and
``Run`` (a run directory and its manifest). Nothing is copied into a run
directory: the run records the lowered recipe, and user code stays where the
model file's import paths point.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from .model_file import FORMAT, ModelFileError, is_model_file, lower as _lower, model_warnings
from .toml_writer import dumps as _dumps

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PACKAGED = Path(__file__).with_name('models')

__all__ = ['Model', 'Run', 'Sample', 'Image', 'Audio', 'ModelFileError',
           'load', 'packaged', 'from_dict', 'override', 'save', 'dumps', 'lower', 'resolve', 'validate',
           'fingerprint', 'hndl', 'module', 'adversarial', 'reconstruction', 'objective', 'prior_loss',
           'train', 'resume', 'open_run', 'metrics', 'evaluations', 'samples', 'sample', 'view',
           'evaluate', 'image', 'audio']


# --------------------------------------------------------------------------- model files

@dataclass(frozen=True)
class Model:
    """A model file's contents. ``base`` resolves relative ``hndl`` paths."""
    spec: dict
    base: Path | None = None
    path: Path | None = None

    @property
    def name(self):
        return self.spec.get('name', 'custom/model')


def _model(value):
    if isinstance(value, Model):
        return value
    if isinstance(value, (str, Path)):
        return load(value)
    if isinstance(value, dict):
        return from_dict(value)
    raise TypeError(f'Expected a Model, model file path or dictionary, got {type(value).__name__}')


def load(path):
    """Load a model file (or an engine recipe TOML, which loads unchanged)."""
    path = Path(path)
    if path.is_dir():
        path = path / 'model.toml' if (path / 'model.toml').is_file() else path / 'config.toml'
    with path.open('rb') as stream:
        spec = tomllib.load(stream)
    return Model(spec, base=path.parent.resolve(), path=path.resolve())


def packaged(name):
    """A model file shipped with HyperGAN, e.g. ``packaged("gaussian-grid")``."""
    path = PACKAGED / f'{name}.toml'
    if re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) is None or not path.is_file():
        choices = sorted(p.stem for p in PACKAGED.glob('*.toml'))
        raise ValueError(f'Unknown packaged model {name!r}; choose one of {", ".join(choices)}')
    return load(path)


def from_dict(spec, base=None):
    """A model built in Python; ``base`` resolves any relative ``hndl`` paths."""
    if not isinstance(spec, dict):
        raise TypeError('A model is a dictionary shaped like a model file')
    return Model(deepcopy(spec), base=Path(base).resolve() if base is not None else None)


def _segment(container, key, location):
    if isinstance(container, list):
        if key.isdigit() and int(key) < len(container):
            return int(key)
        # Losses, like other lists of tables, can be addressed by id or critic.
        for index, item in enumerate(container):
            if isinstance(item, dict) and key in (item.get('id'), item.get('critic')):
                return index
        raise KeyError(f'{location}: no list element {key!r} (use an index, id or critic name)')
    return key


def override(model, changes=None, /, **sections):
    """Return a copy with dotted-path values set, e.g. ``{"train.steps": 10}``.

    List elements are addressed by index, ``id`` or ``critic``:
    ``"losses.reconstruction.weight"``. Keyword sections deep-merge tables:
    ``override(model, train={"steps": 10})``.
    """
    model = _model(model)
    spec = deepcopy(model.spec)
    for dotted, value in (changes or {}).items():
        parts = dotted.split('.')
        target = spec
        for depth, part in enumerate(parts[:-1]):
            key = _segment(target, part, '.'.join(parts[:depth + 1]))
            if isinstance(target, dict) and key not in target:
                target[key] = {}
            target = target[key]
        target[_segment(target, parts[-1], dotted) if isinstance(target, list) else parts[-1]] = deepcopy(value)

    def merge(into, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(into.get(key), dict):
                merge(into[key], value)
            else:
                into[key] = deepcopy(value)
    merge(spec, sections)
    return Model(spec, base=model.base, path=None)


def _relocated(spec, base, destination):
    """Rewrite relative hndl paths so a saved copy still reads the same files."""
    spec = deepcopy(spec)
    for network in (spec.get('networks') or {}).values():
        file = network.get('hndl') if isinstance(network, dict) else None
        if isinstance(file, str) and not Path(file).is_absolute():
            if base is None:
                raise ModelFileError('A model with relative hndl paths needs a base directory to be saved elsewhere')
            source = (base / file).resolve()
            try:
                network['hndl'] = Path(source).relative_to(destination).as_posix()
            except ValueError:
                import os
                network['hndl'] = Path(os.path.relpath(source, destination)).as_posix()
    return spec


def dumps(model):
    """The model file text (TOML)."""
    model = _model(model)
    return _dumps(model.spec, comment=None)


def save(model, path):
    """Write the model file; referenced .hndl files and Python are not copied."""
    model = _model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = _relocated(model.spec, model.base, path.parent.resolve())
    if is_model_file(spec):
        spec = {'format': FORMAT, **{k: v for k, v in spec.items() if k != 'format'}}
    path.write_text(_dumps(spec))
    return path


def lower(model):
    """The engine recipe for this model (torch-free, no user code imported)."""
    model = _model(model)
    return _lower(model.spec, model.base) if is_model_file(model.spec) else deepcopy(model.spec)


def resolve(model):
    """The complete resolved configuration a run would record."""
    from .config import load_config
    return load_config(lower(model))


def validate(model):
    """Check the model without Torch or user code; return warnings, raise on errors."""
    model = _model(model)
    resolved = resolve(model)
    extra = model_warnings(model.spec) if is_model_file(model.spec) else []
    return extra + list(resolved['warnings'])


def fingerprint(model):
    """Numerical identity: equal fingerprints train the same recipe."""
    from .config import fingerprint as recipe_fingerprint
    return recipe_fingerprint(resolve(model))


# --------------------------------------------------------------------------- building in Python

def hndl(source=None, *, file=None, input_shape, output_shape, inputs, role, **args):
    """A network written in HNDL, inline (``source``) or in a file (``file``)."""
    if (source is None) == (file is None):
        raise ValueError('Pass exactly one of source or file')
    network = {'role': role, 'inputs': dict(inputs), 'input_shape': input_shape, 'output_shape': output_shape}
    network.update({'source': source} if source is not None else {'hndl': str(file)})
    if args:
        network['args'] = args
    return network


def module(path, *, inputs, role, trainable=None, **args):
    """A network implemented as an importable ``torch.nn.Module`` class or factory."""
    network = {'role': role, 'module': path, 'inputs': dict(inputs)}
    if args:
        network['args'] = args
    if trainable is not None:
        network['trainable'] = trainable
    return network


def adversarial(critic, *, weight=1.0, penalty=1.0, real=None, fake=None, inputs=None, id=None):
    loss = {'type': 'adversarial', 'critic': critic, 'weight': weight, 'penalty': penalty}
    for key, value in (('real', real), ('fake', fake), ('inputs', inputs), ('id', id)):
        if value is not None:
            loss[key] = value
    return loss


def objective(id, fn='mse', *, input=None, target=None, inputs=None, weight=1.0, args=None, type='objective'):
    loss = {'type': type, 'id': id, 'fn': fn, 'weight': weight}
    if inputs is not None:
        loss['inputs'] = inputs
    else:
        loss['input'] = input
        if target is not None:
            loss['target'] = target
    if args:
        loss['args'] = args
    return loss


def reconstruction(id='reconstruction', fn='mse', *, input='generated', target='batch.real', weight=1.0):
    return objective(id, fn, input=input, target=target, weight=weight, type='reconstruction')


def prior_loss(weight=1.0, **settings):
    return {'type': 'prior', 'weight': weight, **settings}


# --------------------------------------------------------------------------- training

@dataclass(frozen=True)
class Run:
    """A run directory and the manifest it recorded."""
    path: Path
    manifest: dict = field(repr=False)

    @property
    def status(self):
        return self.manifest.get('status')

    @property
    def step(self):
        return self.manifest.get('steps')

    @property
    def config(self):
        return self.manifest.get('config')

    @property
    def fingerprint(self):
        return self.manifest.get('config_sha256')


def open_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    return Run(run_dir, manifest)


def _run(value):
    return value if isinstance(value, Run) else open_run(value)


def train(model, run_dir, *, steps=None, previews=None, checkpoint_every=None, profile=None,
          on_event=None, **controls):
    """Train (or continue, if ``run_dir`` exists) and return the ``Run``.

    The engine receives the lowered recipe, exactly what the CLI builds from the
    same model file, so a run started here and one started with
    ``hypergan train model.toml`` have the same fingerprint. ``profile`` selects
    replicated execution (for example ``"cpu-replicated-gloo"``); call it under
    ``if __name__ == "__main__":`` then.
    """
    from .execution import train as execute
    execute(lower(model), run_dir, steps, profile=profile, on_event=on_event,
            preview_every=previews, checkpoint_every=checkpoint_every, **controls)
    return open_run(run_dir)


def resume(run, model=None, *, steps=None, on_event=None, **controls):
    """Continue a run from its latest checkpoint (optionally with changed observation settings)."""
    from .execution import resume as execute
    run = _run(run)
    execute(run.path, None, lower(model) if model is not None else None, steps=steps, on_event=on_event, **controls)
    return open_run(run.path)


# --------------------------------------------------------------------------- reading runs

def _events(path):
    from .run_events import read_event_page
    cursor = None
    while True:
        page = read_event_page(path, cursor, limit=1000)
        yield from page['events']
        cursor = page['cursor']
        if not page['has_more']:
            return


def metrics(run, names=None):
    """Published scalar metrics: ``{name: [(step, value), ...]}`` in step order."""
    run = _run(run)
    wanted = set(names) if names is not None else None
    series = {}
    for event in _events(run.path):
        for name, value in (event.get('metrics') or {}).items():
            if wanted is None or name in wanted:
                series.setdefault(name, []).append((event['step'], value))
    return series


def evaluations(run):
    """Snapshot evaluation results: ``{metric_id: [{step, value, status, ...}, ...]}``."""
    run = _run(run)
    results = {}
    for receipt in sorted((run.path / 'metrics' / 'evaluations').glob('*/receipt.json')):
        data = json.loads(receipt.read_text())
        result = data.get('result') or {}
        results.setdefault(data.get('metric_id'), []).append({
            'step': result.get('step', data.get('source_step')), 'value': result.get('value'),
            'status': data.get('status'), 'evaluation_id': data.get('evaluation_id'),
            'error': result.get('error')})
    for rows in results.values():
        rows.sort(key=lambda row: (row['step'] is None, row['step'] or 0))
    return results


@dataclass(frozen=True)
class Sample:
    """Generated output recorded by a run, in whatever structure the model produces.

    ``data`` holds the generated tensor (``None`` when a large image preview was
    stored only as a PNG grid, then in ``png``). ``inputs`` holds the batch
    conditioning it was generated from.
    """
    step: int
    name: str
    data: object
    inputs: dict
    particle_ids: list | None
    source: str
    path: Path
    png: bytes | None = None


def _tensor(value):
    import torch
    return None if value is None else torch.tensor(value)


def samples(run, *, name=None):
    """Periodic EMA previews, oldest first (enable with ``train(..., previews=N)``)."""
    import base64
    run = _run(run)
    index = run.path / 'previews' / 'index.json'
    if not index.is_file():
        return []
    result = []
    for entry in json.loads(index.read_text())['previews']:
        if name is not None and entry.get('name') != name:
            continue
        path = Path(entry['path'])
        payload = json.loads(path.read_text())
        grid = payload.get('image_grid')
        result.append(Sample(step=payload['step'], name=payload['name'], data=_tensor(payload.get('samples')),
                             inputs={key: _tensor(value) for key, value in payload.get('inputs', {}).items()
                                     if isinstance(value, list)},
                             particle_ids=payload.get('particle_ids'), source='preview', path=path,
                             png=base64.b64decode(grid['png_base64']) if grid else None))
    return sorted(result, key=lambda item: item.step)


def sample(run, count=16, seed=0, *, inputs=None, output=None):
    """Fresh samples from the run's final EMA model (a new file beside the run)."""
    from .artifacts import sample as draw
    run = _run(run)
    path = Path(draw(run.path, count=count, seed=seed, inputs=inputs, output=output))
    payload = json.loads(path.read_text())
    return Sample(step=payload['step'], name='sample', data=_tensor(payload['samples']),
                  inputs={key: _tensor(value) for key, value in payload.get('inputs', {}).items()},
                  particle_ids=payload.get('particle_ids'), source='inference', path=path)


def view(item, sampler, *, run=None):
    """Apply a sampler to a Sample: a name from the model's ``[samplers]``, a
    ``module:function`` path, or a callable. Returns what the sampler returns
    (``Image``/``Audio`` for the built-in helpers)."""
    from .item_data import import_object
    args = {}
    if isinstance(sampler, str) and ':' not in sampler:
        config = _run(run).config if run is not None else None
        declared = (config or {}).get('samplers', {})
        if sampler not in declared:
            raise KeyError(f'Sampler {sampler!r} is not declared in this run ({", ".join(declared) or "none"})')
        args = declared[sampler]['args']
        sampler = declared[sampler]['fn']
    fn = import_object(sampler) if isinstance(sampler, str) else sampler
    return fn(item, **args)


def evaluate(run, metric_id, *, model=None):
    """Run one snapshot evaluation now on a finished run; returns its receipt."""
    from .metric_evaluation import evaluate as run_evaluation
    run = _run(run)
    return run_evaluation(run.path, metric_id, config_path=lower(model) if model is not None else None)


# --------------------------------------------------------------------------- viewables

@dataclass(frozen=True)
class Image:
    """8-bit pixels, row-major HWC (1 or 3 channels)."""
    pixels: bytes
    width: int
    height: int
    channels: int

    def png(self):
        from .image_grids import encode_png
        return encode_png(self.pixels, self.width, self.height, self.channels)

    def save(self, path):
        Path(path).write_bytes(self.png())
        return Path(path)


@dataclass(frozen=True)
class Audio:
    """Mono or interleaved float samples in [-1, 1]."""
    samples: list
    rate: int
    channels: int = 1

    def save(self, path):
        import struct
        import wave
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(self.channels)
            stream.setsampwidth(2)
            stream.setframerate(self.rate)
            stream.writeframes(b''.join(struct.pack('<h', int(max(-1.0, min(1.0, x)) * 32767)) for x in self.samples))
        return Path(path)


def image(tensor, *, low=-1.0, high=1.0):
    """An ``Image`` from a CHW or HW tensor/array in [low, high]."""
    import torch
    value = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu()
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError('image() expects CHW with 1 or 3 channels, or HW')
    pixels = ((value - low) / (high - low)).clamp(0, 1).mul(255).round().to(torch.uint8)
    channels, height, width = pixels.shape
    return Image(pixels.permute(1, 2, 0).contiguous().numpy().tobytes(), width, height, channels)


def audio(tensor, *, rate):
    import torch
    value = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu()
    channels = 1 if value.ndim == 1 else value.shape[0]
    return Audio(value.T.reshape(-1).tolist() if value.ndim == 2 else value.tolist(), rate, channels)

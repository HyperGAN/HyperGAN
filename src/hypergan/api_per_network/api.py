"""Plain functions over recipes and runs. No trainer object is exposed.

A recipe lowers to one config file; a run directory is read back through its
manifest, event log, evaluation streams and previews.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

from .declare import Recipe, Sampler
from .lowering import PRIMARY_CRITIC, lift, lower, producer, upstream, relative_file
from . import toml_io

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _source(recipe):
    """A recipe, a raw config dict, or a config path -> something load_config accepts."""
    if isinstance(recipe, Recipe):
        return lower(recipe)
    if isinstance(recipe, dict):
        return recipe
    return str(Path(recipe).resolve())


def validate(recipe):
    """The resolved engine configuration (torch-free; user code is not imported)."""
    from ..config import load_config
    return load_config(_source(recipe))


def fingerprint(recipe):
    """Numerical identity; the same recipe has the same fingerprint in every process."""
    from ..config import fingerprint as engine_fingerprint
    return engine_fingerprint(validate(recipe))


def save(recipe, path):
    """Write the recipe as a plain HyperGAN config file (``hypergan train`` reads it).

    HNDL files stay references relative to the saved file; Python stays import paths.
    """
    path = Path(path)
    config = lower(recipe)
    for name, declaration in recipe.networks.items():
        if declaration.network is None:
            continue
        args = config['components'][name]['args']
        if declaration.network.file is not None:
            args['file'] = relative_file(declaration.network.file, path.parent)
        if isinstance(args.get('network_files'), dict):
            args['network_files'] = {key: relative_file(value, path.parent)
                                     for key, value in args['network_files'].items()}
    header = ('HyperGAN recipe written by hypergan.api_per_network.save().\n'
              'A plain config: `hypergan train THIS_FILE`, or load it with api_per_network.load().\n'
              'Python is referenced by import path (module:object) and never copied into runs.')
    text = toml_io.dumps(config, recipe, header=header)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    # The written file must mean exactly what the in-memory recipe means.
    written = fingerprint(path)
    if written != fingerprint(recipe):
        raise AssertionError('Saved config does not reproduce the recipe fingerprint')
    return path


def load(path):
    """Declarations for any HyperGAN config file (not only ones written by save())."""
    path = Path(path)
    if path.is_dir():
        path = path / 'config.toml'
    with path.open('rb') as stream:
        raw = tomllib.load(stream)
    validate(path)
    return lift(raw, base=path.parent)


def explain(recipe):
    """Per network: role, optimizer group, own losses, and gradient that reaches it."""
    config = validate(recipe)
    components = config['components']
    names = recipe.networks if isinstance(recipe, Recipe) else lift(config).networks
    reach = {name: [] for name in components}
    for term in config['objectives']:
        detached = set(term['detach'])
        seen = set()
        for key, path in term['inputs'].items():
            if key not in detached:
                upstream(path, components, seen=seen)
        label = term.get('id') or term['factory']
        for name in seen:
            reach[name].append(f'loss {label}')
    terms = [('discriminator', 'generated')] + [(t['component'], t['fake']) for t in config.get('adversarial_terms', ())]
    for critic_name, fake in terms:
        for name in upstream(fake, components):
            reach[name].append(f'fooling {critic_name}')
    lines = []
    for name, declaration in names.items():
        spec = components[name]
        if declaration.role == 'critic':
            group = 'critic Adam (shared by all critics), lr = optimizer.lr * d_lr_mult'
        elif declaration.role == 'frozen':
            group = 'none (frozen)'
        elif declaration.role == 'shared':
            group = f'parameters of {spec["reuse"]}'
        else:
            group = 'generator-side Adam, lr = optimizer.lr'
        factory = spec.get('factory', 'reuse')
        lines.append(f'{name} [{declaration.role}] {factory}')
        lines.append(f'  inputs: ' + ', '.join(f'{k} <- {v}' for k, v in spec['inputs'].items()))
        lines.append(f'  optimizer: {group}')
        if declaration.role == 'critic':
            for judge in declaration.judges:
                lines.append(f'  judges: real {judge.real} vs fake {judge.fake}'
                             + (f' (weight {judge.weight})' if judge.weight is not None else ''))
            penalty = declaration.penalty
            lines.append('  penalty: ' + ('none' if penalty is None else 'k3p(' + ', '.join(
                f'{key}={value}' for key, value in vars(penalty).items() if value is not None) + ')'))
        else:
            own = [l.id or l.factory if l.kind == 'objective' else f'adversarial({l.critic})'
                   for l in declaration.losses]
            lines.append('  own losses: ' + (', '.join(own) or '-'))
            received = sorted(set(reach.get(name, ())))
            lines.append('  gradient from: ' + (', '.join(received) or '-'))
    return '\n'.join(lines)


# ------------------------------------------------------------------ runs

@dataclass(frozen=True)
class Run:
    path: Path
    run_id: str
    status: str
    steps: int
    config_sha256: str


def run(path):
    manifest = json.loads((Path(path) / 'manifest.json').read_text())
    return Run(Path(path), manifest['run_id'], manifest['status'], manifest['steps'], manifest['config_sha256'])


def train(recipe, run_dir, *, steps=None, **controls):
    """Train (or continue) a run. ``recipe`` is a Recipe, raw config dict or config path.

    Controls pass through: checkpoint_every, preview_every, preview_keep,
    max_seconds, stop_after_steps, profile (e.g. 'cpu-replicated-gloo'), on_event.
    """
    from ..execution import train as engine_train
    on_event = controls.pop('on_event', None)
    engine_train(_source(recipe), run_dir, steps, on_event=on_event, **controls)
    return run(run_dir)


def resume(run_dir, recipe=None, **controls):
    from ..execution import resume as engine_resume
    on_event = controls.pop('on_event', None)
    engine_resume(run_dir, None, None if recipe is None else _source(recipe), on_event=on_event, **controls)
    return run(run_dir)


def _run_path(value):
    return value.path if isinstance(value, Run) else Path(value)


def _events(path):
    from ..run_events import read_event_page
    cursor = None
    while True:
        page = read_event_page(path, cursor, limit=1000)
        yield from page['events']
        if not page['has_more']:
            return
        cursor = page['cursor']


def metrics(run_or_path, names=None):
    """{metric id: [(step, value), ...]} for published update and preview scalars."""
    result = {}
    for event in _events(_run_path(run_or_path)):
        for name, number in (event.get('metrics') or {}).items():
            if names is None or name in names:
                result.setdefault(name, []).append((event['step'], number))
    return result


def observations(run_or_path):
    """Why a value is or is not there: [(step, id, status, reason)] for custom metrics,
    evaluations and previews (queued, dropped, scheduled, skipped, ...)."""
    rows = []
    for event in _events(_run_path(run_or_path)):
        kind = event.get('event')
        for name, status in (event.get('measurement_status') or {}).items():
            if status.get('status') not in ('available', 'unavailable'):
                rows.append((event['step'], name, status.get('status'), status.get('reason')))
        if kind in ('evaluation_scheduled', 'evaluation_skipped', 'evaluation_complete'):
            rows.append((event['step'], event.get('metric_id', 'evaluation'), kind.split('_', 1)[1],
                         event.get('reason')))
        elif kind in ('preview', 'preview_skipped'):
            rows.append((event['step'], 'preview', 'published' if kind == 'preview' else 'skipped',
                         event.get('reason')))
    return rows


def evaluations(run_or_path):
    """{metric id: [{'step', 'value', 'status', 'evaluation_id'}, ...]} ordered by step."""
    root = _run_path(run_or_path) / 'metrics' / 'evaluations'
    result = {}
    for events in sorted(root.glob('*/events.jsonl')) if root.is_dir() else ():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            values = {**event.get('metrics', {}), **event.get('distributions', {})}
            for name in set(values) | set(event.get('measurement_status', {})):
                result.setdefault(name, []).append({
                    'step': event['step'], 'value': values.get(name), 'status': event['status'],
                    'evaluation_id': event['evaluation_id']})
    for rows in result.values():
        rows.sort(key=lambda row: row['step'])
    return result


def evaluate(run_or_path, metric_id, recipe=None):
    """Run one snapshot evaluation now on a stopped run (manual or interval metric)."""
    from ..metric_evaluation import evaluate as engine_evaluate
    return engine_evaluate(str(_run_path(run_or_path)), metric_id,
                           config_path=None if recipe is None else _source(recipe))


def _apply(sampler, values, **context):
    if sampler is None:
        return values
    from .adapters import load_reference
    function = load_reference(sampler.function) if isinstance(sampler, Sampler) else sampler
    args = sampler.args if isinstance(sampler, Sampler) else {}
    return function(values, **args, **context)


def samples(run_or_path, count=16, *, seed=0, inputs=None, sampler=None):
    """Fresh samples from the run's EMA model; a sampler maps them to something viewable."""
    import torch
    from ..artifacts import sample
    with tempfile.TemporaryDirectory() as directory:
        output = sample(str(_run_path(run_or_path)), count, seed, Path(directory) / 'samples.json', inputs=inputs)
        payload = json.loads(output.read_text())
    values = torch.tensor(payload['samples'])
    return _apply(sampler, values, step=payload['step'])


@dataclass(frozen=True)
class Preview:
    step: int
    name: str
    shape: tuple
    value: object  # the tensor, or the sampler's view of it


def previews(run_or_path, *, name=None, sampler=None):
    """Previews the run published during training, optionally mapped by a sampler."""
    import torch
    root = _run_path(run_or_path) / 'previews'
    index = root / 'index.json'
    if not index.is_file():
        return []
    result = []
    for record in json.loads(index.read_text())['previews']:
        if name is not None and record['name'] != name:
            continue
        payload = json.loads(Path(record['path']).read_text())
        if 'samples' not in payload:
            continue  # image previews publish PNG grids; read those files directly
        values = torch.tensor(payload['samples'])
        result.append(Preview(record['step'], record['name'], tuple(values.shape),
                              _apply(sampler, values, step=record['step'])))
    return result


def catalog(run_or_path):
    """Definitions (label, unit, formula, owner) of every metric the run publishes."""
    from ..metrics import read_catalog
    return read_catalog(_run_path(run_or_path))['metrics']


def producer_of(binding):
    """The network a binding reads directly (engine path), for tools."""
    return producer(binding)


__all__ = ['validate', 'fingerprint', 'save', 'load', 'explain', 'train', 'resume', 'run', 'Run',
           'metrics', 'evaluations', 'evaluate', 'samples', 'previews', 'Preview', 'catalog',
           'lower', 'lift', 'PRIMARY_CRITIC']

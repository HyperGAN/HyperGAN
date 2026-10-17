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
import hashlib
import inspect
import json
from pathlib import Path
import re

from .model_file import FORMAT, ModelFileError, is_model_file, lower as _lower, model_warnings, upstream, producer
from .toml_writer import dumps as _dumps

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .workspace import find_workspace, load_workspace, interpolate_variables, resolve_device_aliases
from .model_file import resolve_inheritance, collect_source_files

PACKAGED = Path(__file__).with_name('models')

__all__ = ['Model', 'Run', 'Sample', 'Image', 'Audio', 'ModelFileError',
           'model', 'load', 'packaged', 'packaged_list', 'from_dict', 'override', 'save', 'dumps', 'lower', 'resolve', 'validate',
           'fingerprint', 'explain', 'import_path', 'plain', 'hndl', 'module', 'sampler', 'adversarial',
           'reconstruction', 'objective', 'prior_loss', 'prepare', 'prepare_resume',
           'train', 'resume', 'open_run', 'metrics', 'evaluations', 'observations', 'catalog', 'events',
           'samples', 'sample', 'load_inference', 'view', 'evaluate', 'image', 'audio',
           'lift', 'load_recipe', 'describe', 'data_check', 'preflight', 'serve', 'viewer_status',
           'stop_viewer', 'contributions', 'project', 'checkpoint_status', 'request_checkpoint',
           'export', 'prune_checkpoints', 'resolve_checkpoint', 'extract_ema_bundle',
           'ensure_dashboard_running', 'stop_all_servers', 'dashboard_status', 'stop_dashboard',
           'find_runs_root', 'stop_projector',
           'find_workspace', 'load_workspace', 'interpolate_variables', 'resolve_device_aliases',
           'resolve_inheritance', 'collect_source_files',
           'status', 'runs_ls', 'runs_compare', 'gc', 'gc_all', 'compact_events',
           'resolve_run_target', 'register_run', 'sync_registry', 'generate_run_slug',
           'queue_add', 'queue_ls', 'queue_status', 'queue_stop', 'queue_run', 'queue_install_service']




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

    if isinstance(spec, dict):
        ws_path = find_workspace(path)
        ws_dict = load_workspace(ws_path) if ws_path else None
        ws_root = ws_path.parent if ws_path else None

        if 'extends' in spec:
            spec = resolve_inheritance(spec, config_path=path, workspace_root=ws_root)
        elif is_model_file(spec) and 'name' not in spec:
            spec['name'] = path.stem

        if ws_dict is not None:
            spec = interpolate_variables(spec, ws_dict)

    return Model(spec, base=path.parent.resolve(), path=path.resolve())



def packaged(name):
    """A model file shipped with HyperGAN, e.g. ``packaged("gaussian-grid")``."""
    path = PACKAGED / f'{name}.toml'
    if re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) is None or not path.is_file():
        choices = sorted(p.stem for p in PACKAGED.glob('*.toml'))
        raise ValueError(f'Unknown packaged model {name!r}; choose one of {", ".join(choices)}')
    return load(path)


def packaged_list():
    """List available packaged model files."""
    results = []
    for path in sorted(PACKAGED.glob('*.toml')):
        try:
            m = load(path)
            results.append({
                'name': path.stem,
                'path': str(path),
                'description': m.spec.get('name', path.stem),
                'scope': 'numerical-reference' if path.stem == 'gaussian-grid' else 'packaged',
            })
        except Exception:
            results.append({'name': path.stem, 'path': str(path)})
    return results


def from_dict(spec, base=None):
    """A model built in Python; ``base`` resolves any relative ``hndl`` paths."""
    if not isinstance(spec, dict):
        raise TypeError('A model is a dictionary shaped like a model file')
    return Model(deepcopy(spec), base=Path(base).resolve() if base is not None else None)


def model(name='custom/model', *, data=None, prior=None, networks=None, losses=None,
          penalty=None, train=None, sampling=None, samplers=None, metrics=None,
          evaluations=None, defaults=None, base=None, **kwargs):
    """Keyword constructor for a Model (front end over from_dict)."""
    spec = {'format': kwargs.pop('format', FORMAT)}
    if name is not None:
        spec['name'] = name
    if defaults is not None:
        spec['defaults'] = defaults
    if data is not None:
        spec['data'] = data
    if prior is not None:
        spec['prior'] = prior
    if networks is not None:
        spec['networks'] = networks
    if losses is not None:
        spec['losses'] = losses
    if penalty is not None:
        spec['penalty'] = penalty
    if train is not None:
        spec['train'] = train
    if sampling is not None:
        spec['sampling'] = sampling
    if samplers is not None:
        spec['samplers'] = samplers
    if metrics is not None:
        spec['metrics'] = metrics
    if evaluations is not None:
        spec['evaluations'] = evaluations
    spec.update(kwargs)
    return from_dict(spec, base=base)


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
    from .model_file import OPTIMIZER_FIELDS, is_model_file
    model = _model(model)
    spec = deepcopy(model.spec)
    is_model = is_model_file(spec)

    norm_changes = {}
    for dotted, value in (changes or {}).items():
        if not is_model and (dotted.startswith('train.') or dotted.startswith('training.')):
            field = dotted.split('.', 1)[1]
            if field in OPTIMIZER_FIELDS and 'optimizer' in spec:
                norm_changes[f'optimizer.{field}'] = value
            elif 'training' in spec:
                norm_changes[f'training.{field}'] = value
            else:
                norm_changes[dotted] = value
        elif is_model and dotted.startswith('training.'):
            field = dotted.split('.', 1)[1]
            norm_changes[f'train.{field}'] = value
        else:
            norm_changes[dotted] = value

    for dotted, value in norm_changes.items():
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
    def relocated(file):
        if not isinstance(file, str) or Path(file).is_absolute():
            return file
        if base is None:
            raise ModelFileError('A model with relative hndl paths needs a base directory to be saved elsewhere')
        import os
        source = (base / file).resolve()
        try:
            shared = Path(os.path.commonpath([source, destination]))
        except ValueError:
            # Windows volumes have no shared directory or relative path.
            return source.as_posix()
        # Relative when the files share a project directory, else absolute.
        return (Path(os.path.relpath(source, destination)).as_posix()
                if shared != Path(shared.anchor) else source.as_posix())

    for network in (spec.get('networks') or {}).values():
        file = network.get('hndl') if isinstance(network, dict) else None
        if file is not None:
            network['hndl'] = relocated(file)
    for component in (spec.get('components') or {}).values():
        args = component.get('args', {})
        if component.get('factory') == 'hndl' and 'file' in args:
            args['file'] = relocated(args['file'])
        for name, file in args.get('network_files', {}).items():
            args['network_files'][name] = relocated(file)
    return spec


def dumps(model):
    """The model file text (TOML)."""
    model = _model(model)
    return _dumps(model.spec, table_depth=3)


_HASH = re.compile(r"^# body-sha256: ([0-9a-f]{64})$", re.M)


def save(model, path, *, overwrite=False):
    """Write the model file; referenced .hndl files and Python are not copied."""
    model = _model(model)
    path = Path(path)
    if path.suffix != '.toml':
        raise ValueError('Model files use the .toml suffix')
    spec = _relocated(model.spec, model.base, path.parent.resolve())
    if is_model_file(spec):
        spec = {'format': FORMAT, **{k: v for k, v in spec.items() if k != 'format'}}
    body = _dumps(spec, table_depth=3)
    digest = hashlib.sha256(body.encode()).hexdigest()
    header = (
        "# HyperGAN model written by hypergan.api (hg.save). Train it with `hypergan train` or hg.train.\n"
        "# Networks and data are referenced by import path; edit those modules, not copies.\n"
        "# hg.save replaces this file only while it is unedited; after editing it here, load it with hg.load.\n"
        f"# body-sha256: {digest}\n\n"
    )
    text = header + body
    if path.exists() and not overwrite:
        existing = path.read_text(encoding="utf-8")
        match = _HASH.search(existing)
        if existing != text:
            if match is None:
                raise FileExistsError(f"{path} exists and was not written by hg.save; pass overwrite=True or choose another path")
            current = existing.split("\n\n", 1)[1] if "\n\n" in existing else ""
            if hashlib.sha256(current.encode()).hexdigest() != match.group(1):
                raise FileExistsError(f"{path} was edited after hg.save wrote it; load it with hg.load(path) "
                                      "to train the edited version, or pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def lower(model):
    """The engine recipe for this model (torch-free, no user code imported)."""
    model = _model(model)
    if is_model_file(model.spec):
        return _lower(model.spec, model.base)
    if model.base is not None:
        from .config import inline_network_sources
        return inline_network_sources(model.spec, model.base)
    return deepcopy(model.spec)


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


def explain(model):
    """Per network: role, optimizer group, own losses, and gradient that reaches it."""
    model = _model(model)
    recipe = lower(model)
    components = recipe.get('components', {})
    name_map = recipe.get('name_map', {})
    reach = {name: [] for name in components}
    for term in recipe.get('objectives', []):
        detached = set(term.get('detach', ['target'] if 'target' in term.get('inputs', {}) else []))
        seen = set()
        for key, path in term.get('inputs', {}).items():
            if key not in detached:
                upstream(path, components, seen=seen)
        label = term.get('id') or term.get('factory')
        for name in seen:
            reach[name].append(f'loss {label}')
    terms = []
    if 'adversarial' in recipe:
        disc_inputs = components.get('discriminator', {}).get('inputs', {})
        if any(p == 'candidate' or p.startswith('candidate.') or producer(p) == 'generator' for p in disc_inputs.values()):
            terms.append(('discriminator', 'generated'))
    for t in recipe.get('adversarial_terms', ()):
        terms.append((t['component'], t['fake']))
    for critic_name, fake in terms:
        for name in upstream(fake, components):
            reach[name].append(f'fooling {critic_name}')

    lines = []
    networks = model.spec.get('networks') if is_model_file(model.spec) else None
    if networks:
        for orig_name, net_spec in networks.items():
            comp_name = name_map.get(orig_name, orig_name)
            role = net_spec.get('role', 'generator')
            spec = components.get(comp_name, {})
            factory = spec.get('factory', net_spec.get('hndl') or net_spec.get('module') or 'unknown')
            if role == 'critic':
                group = 'critic Adam (shared by all critics), lr = optimizer.lr * d_lr_mult'
            elif 'reuse' in net_spec or 'reuse' in spec:
                reuse_target = net_spec.get('reuse') or spec.get('reuse')
                group = f'parameters of {reuse_target}'
            elif net_spec.get('trainable') is False or spec.get('trainable') is False or net_spec.get('freeze_parameters') or spec.get('freeze_parameters') or role == 'frozen':
                group = 'none (frozen)'
            elif role == 'shared':
                group = f'parameters of {spec.get("reuse")}'
            else:
                group = 'generator-side Adam, lr = optimizer.lr'
            lines.append(f'{orig_name} [{role}] {factory}')
            inputs_str = ', '.join(f'{k} <- {v}' for k, v in net_spec.get('inputs', {}).items())
            lines.append(f'  inputs: {inputs_str}')
            lines.append(f'  optimizer: {group}')
            if role != 'critic':
                received = sorted(set(reach.get(comp_name, [])))
                lines.append('  gradient from: ' + (', '.join(received) or '-'))
    else:
        for name, spec in components.items():
            factory = spec.get('factory', 'reuse')
            role = 'critic' if name == 'discriminator' or any(t.get('component') == name for t in recipe.get('adversarial_terms', [])) else 'generator'
            if role == 'critic':
                group = 'critic Adam (shared by all critics), lr = optimizer.lr * d_lr_mult'
            else:
                group = 'generator-side Adam, lr = optimizer.lr'
            lines.append(f'{name} [{role}] {factory}')
            inputs_str = ', '.join(f'{k} <- {v}' for k, v in spec.get('inputs', {}).items())
            lines.append(f'  inputs: {inputs_str}')
            lines.append(f'  optimizer: {group}')
            received = sorted(set(reach.get(name, [])))
            lines.append('  gradient from: ' + (', '.join(received) or '-'))
    return '\n'.join(lines)


# --------------------------------------------------------------------------- building in Python

def plain(value, location):
    """Check that an argument survives the config file unchanged (no None, NaN or objects)."""
    if value is None:
        raise ValueError(f"{location} is None; the config file has no null, so omit the argument instead")
    if isinstance(value, bool) or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v, f"{location}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError(f"{location} keys must be strings")
        return {k: plain(v, f"{location}.{k}") for k, v in value.items()}
    raise TypeError(f"{location} is a {type(value).__name__}; config arguments must be numbers, strings, lists or dicts")


def import_path(obj, what):
    """``module:qualname`` for a class or function that other processes can import."""
    if isinstance(obj, str):
        if obj.count(":") != 1 or not all(obj.split(":")):
            raise ValueError(f"{what} must be an importable object or a 'module:object' string, got {obj!r}")
        return obj
    module_name, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not module_name or not qualname:
        raise TypeError(f"{what} must be a class or function, got {type(obj).__name__}")
    if module_name == "__main__":
        raise ValueError(
            f"{what} {qualname} is defined in the script being run (__main__). Every training process "
            f"rebuilds the model from the config file by import path, so move {qualname} into an "
            f"importable module (for example my_project/networks.py) and import it from there."
        )
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(f"{what} {qualname} is local to a function; define it at module level so it can be imported")
    return f"{module_name}:{qualname}"


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
    path_str = import_path(path, 'Network') if callable(path) else path
    if isinstance(path_str, str) and (path_str.count(':') != 1 or not all(path_str.split(':'))):
        raise ValueError(f"Network must be an importable object or a 'module:object' string, got {path_str!r}")
    if callable(path):
        try:
            sig = inspect.signature(path)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                for arg_name in args:
                    if arg_name not in sig.parameters:
                        raise TypeError(f"Network {getattr(path, '__name__', path)} got unexpected keyword argument {arg_name!r}")
        except (ValueError, TypeError) as exc:
            if "unexpected keyword argument" in str(exc):
                raise
    network = {'role': role, 'module': path_str, 'inputs': dict(inputs)}
    if args:
        network['args'] = args
    if trainable is not None:
        network['trainable'] = trainable
    return network


def sampler(fn, *, inputs=None, count=None, seed=None, **args):
    """Declare a sampler entry for [samplers.<name>]."""
    fn_str = import_path(fn, 'Sampler') if callable(fn) else fn
    entry = {'fn': fn_str}
    if inputs is not None:
        entry['inputs'] = dict(inputs)
    if count is not None:
        entry['count'] = count
    if seed is not None:
        entry['seed'] = seed
    if args:
        entry['args'] = args
    return entry


def adversarial(critic, *, weight=1.0, penalty=1.0, real=None, fake=None, inputs=None, id=None):
    loss = {'type': 'adversarial', 'critic': critic, 'weight': weight, 'penalty': penalty}
    for key, value in (('real', real), ('fake', fake), ('inputs', inputs), ('id', id)):
        if value is not None:
            loss[key] = value
    return loss


def objective(id, fn='mse', *, input=None, target=None, inputs=None, weight=1.0, args=None, type='objective'):
    if callable(fn):
        fn_str = import_path(fn, f'Objective {id}')
    else:
        fn_str = fn
    loss = {'type': type, 'id': id, 'fn': fn_str, 'weight': weight}
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


def open_run(run_dir=None):
    if run_dir is not None and Path(run_dir).suffix == ".json":
        manifest_path = Path(run_dir).resolve()
        run_path = manifest_path.parent
    else:
        from .registry import resolve_run_target
        run_path = resolve_run_target(run_dir)
        if run_path.is_file():
            manifest_path = run_path
            run_path = run_path.parent
        else:
            manifest_path = run_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must contain a JSON object")
    return Run(run_path, manifest)


def _run(value):
    return value if isinstance(value, Run) else open_run(value)


def prepare(model, run_dir, steps=None, *, profile=None, service_policy=None,
            devices=None,
            checkpoint_every=None, max_seconds=None, stop_after_steps=None,
            preview_every=None, previews=None, preview_keep=None, preview_keep_source=None,
            preview_name=None, **extra):
    """Validate and prepare a training run, returning a PreparedExecution."""
    from .execution import prepare_train as _prepare_train
    model = _model(model)
    pe = preview_every if preview_every is not None else previews
    lowered = lower(model)
    if getattr(model, 'path', None):
        lowered['_source_path'] = str(model.path)
    return _prepare_train(lowered, run_dir, steps=steps, profile=profile,
                          service_policy=service_policy, devices=devices,
                          checkpoint_every=checkpoint_every,
                          max_seconds=max_seconds, stop_after_steps=stop_after_steps,
                          preview_every=pe, preview_keep=preview_keep,
                          preview_keep_source=preview_keep_source, preview_name=preview_name,
                          **extra)


def prepare_resume(run_dir, checkpoint=None, model=None, *, steps=None,
                   profile=None, service_policy=None, devices=None,
                   checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                   preview_every=None, previews=None, preview_keep=None, preview_keep_source=None,
                   preview_name=None, sets=None, **extra):
    """Validate and prepare a run resumption, returning a PreparedExecution."""
    from .execution import prepare_resume as _prepare_resume
    run_path = Path(run_dir).resolve()
    if model is None and (sets or steps is not None):
        if (run_path / 'model.toml').is_file():
            model = load(run_path / 'model.toml')
        elif (run_path / 'recipe.toml').is_file():
            model = load(run_path / 'recipe.toml')
        elif (run_path / 'config.toml').is_file():
            model = load(run_path / 'config.toml')
        elif (run_path / 'manifest.json').is_file():
            with (run_path / 'manifest.json').open('rb') as f:
                manifest = json.load(f)
            saved_config = manifest.get('config')
            if not isinstance(saved_config, dict):
                raise ValueError(f"Run {run_dir} has no recorded configuration in manifest.json")
            model = from_dict(saved_config, base=run_path)
        else:
            raise ValueError(f"Run directory {run_dir} does not exist or has no configuration")
    if sets:
        normalized_sets = {}
        for k, v in sets.items():
            if k.startswith('training.') and hasattr(model, 'spec') and 'train' in model.spec:
                normalized_sets['train.' + k[len('training.'):]] = v
            elif k.startswith('train.') and hasattr(model, 'spec') and 'training' in model.spec and 'train' not in model.spec:
                normalized_sets['training.' + k[len('train.'):]] = v
            else:
                normalized_sets[k] = v
        model = override(model, normalized_sets)
        if steps is None:
            step_val = normalized_sets.get('train.steps', normalized_sets.get('training.steps'))
            if step_val is not None:
                steps = int(step_val)

    config_path = lower(model) if model is not None else None
    pe = preview_every if preview_every is not None else previews
    return _prepare_resume(run_dir, checkpoint=checkpoint, config_path=config_path, steps=steps,
                           profile=profile, service_policy=service_policy, devices=devices,
                           checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                           stop_after_steps=stop_after_steps, preview_every=pe,
                           preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                           preview_name=preview_name, **extra)


def train(model, run_dir, *, steps=None, previews=None, preview_every=None, checkpoint_every=None, profile=None,
          devices=None, on_event=None, **controls):
    """Train (or continue, if ``run_dir`` exists) and return the ``Run``."""
    pe = preview_every if preview_every is not None else previews
    prep = prepare(model, run_dir, steps=steps, profile=profile, devices=devices,
                   checkpoint_every=checkpoint_every, preview_every=pe, **controls)
    prep.run(on_event=on_event)
    return open_run(run_dir)


def resume(run, model=None, *, steps=None, devices=None, on_event=None, **controls):
    """Continue a run from its latest checkpoint (optionally with changed observation settings)."""
    run = _run(run)
    prep = prepare_resume(run.path, model=model, steps=steps, devices=devices, **controls)
    prep.run(on_event=on_event)
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


def observations(run):
    """Why a value is or is not there: [(step, id, status, reason)] for custom metrics,
    evaluations and previews (queued, dropped, scheduled, skipped, ...)."""
    run = _run(run)
    rows = []
    for event in _events(run.path):
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


def catalog(run, revision=None):
    """Immutable metric definitions recorded by a run."""
    from .metrics import read_catalog
    run = _run(run)
    return read_catalog(run.path, revision)


def events(run, cursor=None, limit=100, max_bytes=1048576):
    """Read a bounded page of run events without loading training dependencies."""
    from .run_events import read_event_page
    run = _run(run)
    return read_event_page(run.path, cursor=cursor, limit=limit, max_bytes=max_bytes)


def load_inference(run_dir):
    """Verify and load a run's EMA inference bundle on CPU in evaluation mode.

    Returns the raw state, its resolved configuration, the component graph, the
    prior and the verified bundle SHA256.
    """
    import torch
    from .recipes import ComponentGraph, make_prior
    from .config import resolve_config

    run_dir = Path(run_dir)
    bundle_dir = run_dir
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle_path"):
            path = Path(manifest["bundle_path"])
            bundle_dir = (path if path.is_absolute() else run_dir / path).parent
    path = bundle_dir / "model.pt"
    expected = (bundle_dir / "model.sha256").read_text().strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Inference artifact hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("kind") != "ema-inference":
        raise ValueError("Unsupported inference artifact schema")
    for key in ("config", "components", "model_states", "prior", "example_inputs"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"Invalid inference artifact: missing or invalid {key}")
    if "generator" not in state["components"] or set(state["model_states"]) != {name for name, spec in state["components"].items() if 'reuse' not in spec}:
        raise ValueError("Invalid inference artifact: component states must exactly match the generator graph")
    config = resolve_config(state["config"])
    graph = ComponentGraph(state["components"]).float().eval().requires_grad_(False)
    for name, weights in state["model_states"].items():
        graph.models[name].load_state_dict(weights)
        if "model_buffers" in state and name in state["model_buffers"]:
            buf = state["model_buffers"][name]
            for bname, bval in (buf or {}).items():
                if hasattr(graph.models[name], bname):
                    getattr(graph.models[name], bname).copy_(bval)
    prior = make_prior(config["prior"], device='cpu', recipe=config.get('recipe')).float().eval().requires_grad_(False)
    prior.load_state_dict(state["prior"])
    if "prior_buffers" in state:
        for bname, bval in (state["prior_buffers"] or {}).items():
            if hasattr(prior, bname):
                getattr(prior, bname).copy_(bval)
    return state, config, graph, prior, expected


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


def sample(run, count=16, seed=0, *, inputs=None, output=None, checkpoint=None):
    """Fresh samples from the run's final EMA model (or target checkpoint)."""
    from .artifacts import sample as draw
    run = _run(run)
    path = Path(draw(run.path, count=count, seed=seed, inputs=inputs, output=output, checkpoint=checkpoint))
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


def evaluate(run, metric_id=None, *, checkpoint=None, devices=None, model=None, config=None, bundle=None):
    """Run one snapshot evaluation now on a run (finished or running); returns its receipt."""
    from .metric_evaluation import evaluate as run_evaluation
    run = _run(run)
    config_p = lower(model) if model is not None else config
    return run_evaluation(run.path, metric_id, checkpoint=checkpoint, devices=devices,
                          config_path=config_p, bundle=bundle)


def _run_path(value=None):
    if isinstance(value, Run):
        return value.path
    from .registry import resolve_run_target
    return resolve_run_target(value)


def resolve_run_target(target=None, *, cwd=None, allow_latest=True, require_active=False):
    """Resolve a run reference (short ID, slug, name, path, @latest, or None for active/latest)."""
    from .registry import resolve_run_target as _resolve_target
    return _resolve_target(target, cwd=cwd, allow_latest=allow_latest, require_active=require_active)


def register_run(run_dir, **kwargs):
    from .registry import register_run as _register
    return _register(run_dir, **kwargs)


def sync_registry(cwd=None, extra_roots=None):
    from .registry import sync_registry as _sync
    return _sync(cwd=cwd, extra_roots=extra_roots)


def generate_run_slug(stem, parent_dir=None):
    from .registry import generate_run_slug as _gen_slug
    return _gen_slug(stem, parent_dir=parent_dir)


def export(run=None, checkpoint="latest", output=None):
    """Export a standalone EMA inference model bundle from a run checkpoint."""
    from .checkpoints import resolve_checkpoint as _resolve, export_checkpoint_bundle as _export
    run_dir = _run_path(run)
    ckpt_dir = _resolve(run_dir, target=checkpoint)
    dest, _ = _export(ckpt_dir, output_dir=output or run_dir)
    return dest


def prune_checkpoints(run=None, policy=None, dry_run=False):
    """Prune obsolete checkpoints in run under the retention policy."""
    from .checkpoints import prune_checkpoints as _prune
    return _prune(_run_path(run), policy=policy, dry_run=dry_run)


def resolve_checkpoint(run=None, target="latest"):
    """Resolve target ('latest', 'best', step int/str, or path) to a checkpoint directory."""
    from .checkpoints import resolve_checkpoint as _resolve
    return _resolve(_run_path(run), target=target)


def extract_ema_bundle(checkpoint_dir, manifest=None):
    """Extract EMA weights from checkpoint_dir into an in-memory bundle."""
    from .checkpoints import extract_ema_bundle as _extract
    return _extract(checkpoint_dir, manifest=manifest)


def status(run=None):
    """Compile human-friendly status summary for a run."""
    from .inspection import run_status as _status
    return _status(_run_path(run))


def runs_ls(root=None, sort="fid", status_filter=None):
    """Discover and list all runs under root with filtering and sorting."""
    from .inspection import list_runs as _list_runs
    return _list_runs(root, sort=sort, status_filter=status_filter)


def runs_compare(runs, metric="fid50k_train", at_steps=None):
    """Compare multiple runs for a given metric across step milestones."""
    from .inspection import compare_runs as _compare_runs
    return _compare_runs(runs, metric=metric, at_steps=at_steps)


def gc(run=None, dry_run=False, stride=10):
    """Clean rebuildable views and compact events on a finished run."""
    from .inspection import gc_run as _gc_run
    return _gc_run(_run_path(run), dry_run=dry_run, stride=stride)


def gc_all(root=None, dry_run=False, stride=10):
    """Clean all finished runs under root."""
    from .inspection import gc_all as _gc_all
    return _gc_all(root, dry_run=dry_run, stride=stride)


def compact_events(run, stride=10):
    """Compact redundant train events in events.jsonl, returning reclaimed bytes."""
    from .inspection import compact_events as _compact
    return _compact(_run_path(run), stride=stride)


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


# --------------------------------------------------------------------------- recipe lifting & CLI services

def load_recipe(path):
    """Read a recipe TOML file without lowering or resolving."""
    path = Path(path)
    with path.open('rb') as stream:
        return tomllib.load(stream)


def lift(raw, *, base=None):
    """Lift an engine recipe dictionary into a Model conforming to hypergan-model/1."""
    from .config import DEFAULT
    from .model_file import PENALTY_FIELDS
    raw = deepcopy(raw)
    base = Path.cwd() if base is None else Path(base)
    components = raw.get('components', {}) or deepcopy(DEFAULT.get('components', {}))
    networks = {}
    losses = []

    critics_in_terms = {term.get('component') for term in raw.get('adversarial_terms', [])}
    if 'discriminator' in components:
        critics_in_terms.add('discriminator')

    for name, spec in components.items():
        if name == 'discriminator' or name in critics_in_terms:
            role = 'critic'
        elif name == 'generator':
            role = 'generator'
        elif 'encoder' in name or spec.get('factory', '').endswith('Encoder'):
            role = 'encoder'
        else:
            role = 'auxiliary'

        net = {'role': role, 'inputs': dict(spec.get('inputs', {}))}
        if 'trainable' in spec:
            net['trainable'] = spec['trainable']
        if 'freeze_parameters' in spec:
            net['freeze_parameters'] = spec['freeze_parameters']
        if 'reuse' in spec:
            net['reuse'] = spec['reuse']
        elif spec.get('factory') == 'hndl':
            args = dict(spec.get('args', {}))
            if 'file' in args:
                net['hndl'] = args.pop('file')
            elif 'source' in args:
                net['source'] = args.pop('source')
            if 'input_shape' in args:
                net['input_shape'] = args.pop('input_shape')
            if 'output_shape' in args:
                net['output_shape'] = args.pop('output_shape')
            if args:
                net['args'] = args
        else:
            net['module'] = spec.get('factory')
            if spec.get('args'):
                net['args'] = dict(spec['args'])
        networks[name] = net

    adv_idx = None
    if 'adversarial' in raw:
        adv_idx = len(losses)
        losses.append({'type': 'adversarial', 'critic': 'discriminator', 'weight': raw['adversarial'].get('weight', 1.0)})
    elif 'discriminator' in components:
        adv_idx = len(losses)
        losses.append({'type': 'adversarial', 'critic': 'discriminator', 'weight': 1.0})

    if adv_idx is not None and 'gradient_penalty' in raw and 'coeff' in raw['gradient_penalty']:
        losses[adv_idx]['penalty'] = float(raw['gradient_penalty']['coeff'])

    for term in raw.get('adversarial_terms', []):
        entry = {'type': 'adversarial', 'critic': term['component'], 'weight': term.get('weight', 1.0)}
        for k in ('real', 'fake', 'inputs', 'id'):
            if k in term:
                entry[k] = term[k]
        if term.get('penalty') is False:
            entry['penalty'] = 0.0
        elif term.get('penalty_coeff') is not None:
            entry['penalty'] = float(term['penalty_coeff'])
        losses.append(entry)

    for term in raw.get('objectives', []):
        factory = term.get('factory', 'mse')
        args = dict(term.get('args', {}))
        if factory == 'hypergan.plugin_functions:FunctionLoss' and 'fn' in args:
            fn = args.pop('fn')
        else:
            fn = factory
        kind = 'reconstruction' if term.get('id') == 'reconstruction' else 'objective'
        entry = {'type': kind, 'id': term.get('id', 'loss'), 'fn': fn, 'weight': term.get('weight', 1.0)}
        if 'inputs' in term:
            entry['inputs'] = dict(term['inputs'])
        if 'detach' in term:
            entry['detach'] = list(term['detach'])
        if args:
            entry['args'] = args
        losses.append(entry)

    if 'prior_regularizer' in raw:
        losses.append({'type': 'prior', **deepcopy(raw['prior_regularizer'])})

    model_spec = {'format': FORMAT, 'networks': networks, 'losses': losses}
    if 'name' in raw:
        model_spec['name'] = raw['name']
    if 'defaults' in raw:
        model_spec['defaults'] = raw['defaults']
    if 'run' in raw:
        model_spec['run'] = deepcopy(raw['run'])
    if 'checkpoints' in raw:
        model_spec['checkpoints'] = deepcopy(raw['checkpoints'])
    if 'data' in raw:
        data = deepcopy(raw['data'])
        if data.get('factory') == 'hypergan.item_data:ItemData' and 'dataset' in data.get('args', {}):
            model_spec['data'] = {'dataset': data['args']['dataset'], **{k: v for k, v in data['args'].items() if k != 'dataset'}}
        else:
            model_spec['data'] = data
    if 'prior' in raw:
        model_spec['prior'] = deepcopy(raw['prior'])
    if 'training' in raw or 'optimizer' in raw:
        train_table = deepcopy(raw.get('training', {}))
        if 'optimizer' in raw:
            train_table.update(deepcopy(raw['optimizer']))
        model_spec['train'] = train_table
    gp = raw.get('penalty') or raw.get('gradient_penalty')
    if gp:
        penalty_table = {k: deepcopy(v) for k, v in gp.items() if k != 'coeff' and k in PENALTY_FIELDS}
        if penalty_table:
            model_spec['penalty'] = penalty_table
    if 'sampling' in raw:
        sampling_spec = deepcopy(raw['sampling'])
        sampling_spec.pop('output_noise', None)
        if 'generated' in sampling_spec:
            sampling_spec['output'] = sampling_spec.pop('generated')
        model_spec['sampling'] = sampling_spec
    if 'samplers' in raw:
        model_spec['samplers'] = deepcopy(raw['samplers'])


    if 'metrics' in raw:
        metrics_table = deepcopy(raw['metrics'])
        custom = metrics_table.pop('custom', {})
        scalar_custom = {}
        evaluations = {}
        for name, metric in custom.items():
            if 'evaluation' in metric or metric.get('mode') == 'snapshot' or metric.get('trigger') in ('interval', 'manual'):
                eval_entry = {}
                if 'factory' in metric:
                    eval_entry['factory'] = metric['factory']
                if 'fn' in metric:
                    eval_entry['fn'] = metric['fn']
                if 'args' in metric:
                    eval_entry['args'] = deepcopy(metric['args'])
                if 'inputs' in metric:
                    norm_inputs = {}
                    for ik, iv in metric['inputs'].items():
                        if iv == 'evaluation.generated':
                            norm_inputs[ik] = 'generated'
                        elif iv == 'evaluation.reference':
                            norm_inputs[ik] = 'reference'
                        else:
                            norm_inputs[ik] = iv
                    eval_entry['inputs'] = norm_inputs
                if metric.get('trigger') == 'interval' and 'every_steps' in metric:
                    eval_entry['every'] = metric['every_steps']
                elif 'every_steps' in metric and metric.get('trigger') != 'manual':
                    eval_entry['every'] = metric['every_steps']
                for k in ('timeout', 'on_error'):
                    if k in metric:
                        eval_entry[k] = metric[k]
                if 'factory' not in eval_entry and eval_entry.get('kind') != 'factory':
                    for k in ('label', 'unit', 'direction', 'description', 'reduce', 'kind'):
                        if k in metric:
                            eval_entry[k] = metric[k]
                proto = metric.get('evaluation', {})
                if isinstance(proto, str):
                    if 'factory' not in eval_entry and eval_entry.get('kind') != 'factory':
                        eval_entry.setdefault('kind', proto)
                    proto = {}
                elif not isinstance(proto, dict):
                    proto = {}
                if 'device' in proto:
                    eval_entry['device'] = proto['device']
                elif 'device' in metric:
                    eval_entry['device'] = metric['device']
                if 'sample_count' in proto:
                    eval_entry['samples'] = proto['sample_count']
                elif 'samples' in proto:
                    eval_entry['samples'] = proto['samples']
                elif 'sample_count' in metric:
                    eval_entry['samples'] = metric['sample_count']
                elif 'samples' in metric:
                    eval_entry['samples'] = metric['samples']
                if 'batch_size' in proto:
                    eval_entry['batch_size'] = proto['batch_size']
                elif 'batch_size' in metric:
                    eval_entry['batch_size'] = metric['batch_size']
                if 'seed' in proto:
                    eval_entry['seed'] = proto['seed']
                elif 'seed' in metric:
                    eval_entry['seed'] = metric['seed']
                proto_data = proto.get('data') or metric.get('data')
                if proto_data is not None:
                    d = deepcopy(proto_data)
                    if d.get('factory') == 'hypergan.item_data:ItemData' and 'dataset' in d.get('args', {}):
                        eval_entry['data'] = {'dataset': d['args']['dataset'], **{k: v for k, v in d['args'].items() if k != 'dataset'}}
                    else:
                        eval_entry['data'] = d
                if 'output' in proto:
                    eval_entry['output'] = proto['output']
                elif 'output' in metric:
                    eval_entry['output'] = metric['output']
                evaluations[name] = eval_entry
            else:
                scalar_entry = deepcopy(metric)
                scalar_entry.pop('mode', None)
                scalar_custom[name] = scalar_entry

        if scalar_custom:
            metrics_table['custom'] = scalar_custom
        if metrics_table:
            model_spec['metrics'] = metrics_table
        if evaluations:
            model_spec['evaluations'] = evaluations

    return Model(model_spec, base=base)


def describe(path=None, networks=False, write=False):
    from .model_description import MODEL_FILE, build_networks, describe_run, load as load_model_desc
    if path is None or (not Path(path).exists() and not str(path).endswith(".toml")):
        from .registry import resolve_run_target
        try:
            path = resolve_run_target(path)
        except Exception:
            if path is None:
                raise
    path = Path(path)
    manifest, config = load_model_desc(path)
    if write and not (path.is_dir() and manifest):
        raise ValueError("--write needs a run directory")
    if not isinstance(config, dict):
        raise ValueError("No model configuration recorded for this run")
    recorded = None
    if networks or write:
        recorded = build_networks(config, manifest.get("config_sha256"))
    if write:
        from .run_state import atomic_json
        atomic_json(path / MODEL_FILE, recorded)
    return describe_run(manifest if manifest else {"config": config}, recorded=recorded)


def data_check(config, output=None):
    from .data import ImageFolder
    from .config import load_config
    cfg = resolve(config) if isinstance(config, Model) or is_model_file(config) else load_config(config)
    if cfg["data"]["factory"] != "image_folder":
        raise ValueError("data-check currently supports data.factory='image_folder'")
    data = ImageFolder(**cfg["data"]["args"])
    manifest = {"schema_version": 1, "data": data.resume_identity(), "inventory": data.inventory}
    if output is not None:
        out_path = Path(output)
        with out_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return manifest


def preflight(config, profile=None, runtime=False):
    from .execution_profiles import load_execution_profile
    from .execution_preflight import _resolve_profile, preflight as run_preflight
    from .config import load_config
    cfg = resolve(config) if isinstance(config, Model) or is_model_file(config) else load_config(config)
    prof = (load_execution_profile(profile, cfg) if profile is not None
            else _resolve_profile(None, cfg))
    if runtime:
        return run_preflight(cfg, prof)
    return {"schema_version": 1, "stage": "structural", "profile": prof, "runtime_checked": False}


def serve(run_dir, port=None, host="0.0.0.0", auth="none", session_file=None, open_browser=False, public_origin=None, dev=False):
    if dev:
        from .web_dev import enable
        enable()
    from .web_launch import serve as launch_serve
    launch_serve(Path(run_dir), port=port, host=host, auth=auth,
                 session_file=session_file, open_browser=open_browser,
                 public_origin=public_origin)


def viewer_status(run_dir=None):
    from .web_autostart import viewer_status as autostart_status
    return autostart_status(_run_path(run_dir))


def stop_viewer(run_dir=None):
    from .web_autostart import stop_viewer as autostart_stop
    return autostart_stop(_run_path(run_dir))


def contributions(run_dir=None, map_revision=None, cursor=None, limit=100, max_bytes=1048576):
    from .event_views import MapSpec, read_projection_page
    return read_projection_page(_run_path(run_dir), map_revision or MapSpec().revision,
                                cursor, limit=limit, max_bytes=max_bytes)


def project(run_dir=None, map_spec=None, follow=False, limit=100, on_progress=None):
    from .event_views import MapSpec, Projector
    import time
    run_path = _run_path(run_dir)
    spec = MapSpec(**json.loads(Path(map_spec).read_text(encoding="utf-8"))) if map_spec else MapSpec()
    following = True
    last_progress = None
    while following:
        opened = time.monotonic()
        with Projector(run_path, spec) as projector:
            while True:
                progress = projector.project(limit=limit)
                last_progress = dict(progress, map_revision=spec.revision)
                if follow and progress["documents"] and on_progress:
                    on_progress(last_progress)
                if not follow and not progress["has_more"]:
                    following = False
                    return last_progress
                if time.monotonic() - opened >= 240:
                    break
                if not progress["has_more"]:
                    time.sleep(0.25)
    return last_progress


def checkpoint_status(run_dir=None, request_id=None):
    from .run_requests import checkpoint_request_status
    return checkpoint_request_status(_run_path(run_dir), request_id)


def request_checkpoint(run_dir=None, attempt_id=None, request_id=None):
    from .run_requests import submit_checkpoint_request
    run_dir = _run_path(run_dir)
    with (run_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not manifest.get("run_id") or not manifest.get("attempt_id"):
        raise ValueError("Run manifest has no run/attempt identity for a checkpoint request")
    if manifest.get("status") != "running":
        if request_id is None:
            raise ValueError("Run is not running; use resume to continue a stopped run")
        try:
            return checkpoint_status(run_dir, request_id)
        except FileNotFoundError as exc:
            raise ValueError("Run is not running; use resume to continue a stopped run") from exc
    return submit_checkpoint_request(run_dir, run_id=manifest["run_id"],
                                     attempt_id=attempt_id or manifest["attempt_id"],
                                     request_id=request_id)


def ensure_dashboard_running(runs_root, check_version=True, run_id=None):
    from .web_autostart import ensure_dashboard_running as autostart_ensure
    return autostart_ensure(Path(runs_root), check_version=check_version, run_id=run_id)


def stop_all_servers(force=False):
    from .web_autostart import stop_all_servers as autostart_stop_all
    return autostart_stop_all(force=force)


def dashboard_status(runs_root):
    from .web_autostart import dashboard_status as autostart_dstatus
    return autostart_dstatus(Path(runs_root))


def stop_dashboard(runs_root):
    from .web_autostart import stop_dashboard as autostart_sstop
    return autostart_sstop(Path(runs_root))


def find_runs_root(path=None):
    from .web_autostart import find_runs_root as autostart_find
    return autostart_find(path)


def stop_projector(run_dir):
    from .web_autostart import stop_projector as autostart_sproj
    return autostart_sproj(Path(run_dir))


# --------------------------------------------------------------------------- job queue

def queue_add(model_file, devices=None, run_dir=None, name=None, wait=False, queue_file=None):
    from .queue import queue_add as _add
    return _add(model_file, devices=devices, run_dir=run_dir, name=name, wait=wait, queue_file=queue_file)


def queue_ls(status=None, queue_file=None):
    from .queue import queue_ls as _ls
    return _ls(status_filter=status, queue_file=queue_file)


def queue_status(job_id=None, queue_file=None, lines=20):
    from .queue import queue_status as _status
    return _status(job_id=job_id, queue_file=queue_file, lines=lines)


def queue_stop(job_id=None, stop_all=False, queue_file=None):
    from .queue import queue_stop as _stop
    return _stop(job_id=job_id, stop_all=stop_all, queue_file=queue_file)


def queue_run(devices=None, poll_interval=5.0, once=False, queue_file=None):
    from .queue import queue_run as _run
    return _run(devices=devices, poll_interval=poll_interval, once=once, queue_file=queue_file)


def queue_install_service(user=True, devices=None, name=None, dry_run=False):
    from .queue import install_service as _install
    return _install(user=user, devices=devices, name=name, dry_run=dry_run)



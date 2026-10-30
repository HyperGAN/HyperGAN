"""HyperGAN's Python API: a thin, functional layer over model files.

The model file is the source of truth (see ``hypergan.model_file``). This module
loads one, lets a script change values, trains it and reads the run back::

    import hypergan.api as hg

    if __name__ == "__main__":
        model = hg.load("model.toml")
        model = hg.override(model, {"train.steps": 200, "losses.reconstruction.weight": 0.5})
        run = hg.train(model, "runs/demo", previews=50)
        hg.metrics(run)["loss/d_total"]        # [(step, value), ...]

Training starts worker processes that re-import the calling script, so a
script that trains keeps its top-level code under a ``__main__`` guard.

Everything here is a plain function over two data objects, ``Model`` (the
model file as a dictionary plus the directory its relative paths use) and
``Run`` (a run directory and its manifest). Nothing is copied into a run
directory: the run records the lowered recipe, and user code stays where the
model file's import paths point.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import inspect as _py_inspect
import json
from pathlib import Path
import re
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .model_file import FORMAT, ConfigError, ModelFileError, is_model_file, lower as _lower, model_warnings, upstream, producer
from .toml_writer import dumps as _dumps

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .workspace import find_workspace, load_workspace, interpolate_model_paths, interpolate_variables, resolve_device_aliases
from .model_file import resolve_inheritance, collect_source_files, require_networks
from .cache import get_cache_dir, ensure_cached_url, resolve_or_cache, OfflineError

PACKAGED = Path(__file__).with_name('models')

__all__ = ['Model', 'Run', 'Sample', 'Image', 'Audio', 'ModelFileError', 'ConfigError',
           'StopTraining', 'TrainResult', 'SampleResult', 'ModelDescription', 'ValidationWarning', 'RunStatus', 'PsSnapshot', 'RunHandle',
           'model', 'load', 'load_bundle', 'packaged', 'packaged_list', 'from_dict', 'override', 'save', 'dumps', 'lower', 'resolve', 'validate',
           'validate_config', 'init_project', 'new', 'list_templates',
           'fingerprint', 'explain', 'import_path', 'plain', 'hndl', 'module', 'sampler', 'adversarial',
           'reconstruction', 'objective', 'prior_loss', 'prepare', 'prepare_resume',
           'train', 'resume', 'launch', 'open_run', 'inspect', 'metrics', 'evaluations', 'observations', 'catalog', 'events',
           'samples', 'sample', 'load_inference', 'view', 'evaluate', 'image', 'audio',
           'lift', 'load_recipe', 'describe', 'describe_model', 'data_check', 'check_data',
           'download_dataset', 'dataset_info', 'list_datasets',
           'get_cache_dir', 'ensure_cached_url', 'resolve_or_cache', 'OfflineError',
           'preflight', 'preflight_check', 'serve', 'viewer_status',
           'stop_viewer', 'contributions', 'project', 'checkpoint_status', 'request_checkpoint',
           'request_status', 'request_sample', 'request_stop',
           'export', 'export_generator', 'interpolate', 'LoadedGenerator', 'prune_checkpoints', 'resolve_checkpoint', 'extract_ema_bundle',
           'ensure_dashboard_running', 'stop_all_servers', 'dashboard_status', 'stop_dashboard',
           'find_runs_root', 'stop_projector',
           'find_workspace', 'load_workspace', 'interpolate_variables', 'resolve_device_aliases',
           'resolve_inheritance', 'collect_source_files',
           'status', 'ps', 'stop', 'logs', 'tag', 'rm', 'list_runs', 'runs', 'runs_ls', 'runs_compare', 'compare', 'diff', 'gc', 'prune', 'gc_all', 'compact_events',
           'resolve_run_target', 'register_run', 'sync_registry', 'generate_run_slug',
           'queue_add', 'queue_ls', 'queue_status', 'queue_stop', 'queue_run', 'queue_install_service', 'queue_clear',
           'schema', 'operators', 'ops', 'RunInfo', 'read_run_info']


def __getattr__(name: str):
    if name == "LoadedGenerator":
        from .artifacts import LoadedGenerator
        return LoadedGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- core exceptions & typed results

class StopTraining(Exception):
    """Raised to request immediate graceful training termination."""
    pass


@dataclass
class TrainResult:
    """Encapsulates training execution results."""
    run: Any
    step: int = 0
    status: str = "completed"
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    stop_reason: str | None = None

    @property
    def run_dir(self) -> str:
        return str(getattr(self.run, "path", self.run))

    @property
    def run_id(self) -> str:
        return str(getattr(self.run, "run_id", getattr(self.run, "name", "")))

    @property
    def summary(self) -> dict[str, Any]:
        return self.metrics


@dataclass
class SampleResult:
    """Encapsulates batch sampling output."""
    path: Path
    count: int = 0
    step: int = 0
    png: bytes | None = None
    data: Any = None
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def images(self) -> Any:
        return self.data

    @property
    def output_path(self) -> Path:
        return self.path


@dataclass
class ModelDescription(dict):
    """Encapsulates structured model description from describe()."""
    name: str = ""
    family: str = ""
    components: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, Any] = field(default_factory=dict)
    optimizers: dict[str, Any] = field(default_factory=dict)
    prior: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        d = dict(self.raw) if self.raw else {}
        d.update({
            "name": self.name,
            "family": self.family,
            "components": self.components,
            "losses": self.losses,
            "optimizers": self.optimizers,
            "prior": self.prior,
            "training": self.training,
        })
        dict.__init__(self, d)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ModelDescription' object has no attribute {name!r}")


@dataclass
class ValidationWarning(str):
    """Encapsulates diagnostic validation entries."""
    message: str
    field: str | None = None
    level: str = "warning"

    def __new__(cls, message: str, field: str | None = None, level: str = "warning"):
        return str.__new__(cls, message)


@dataclass
class RunStatus(dict):
    """Encapsulates status() inspection summary."""
    run_id: str
    status: str
    step: int | None = None
    total_steps: int | None = None
    percent: float | None = None
    steps_per_sec: float | None = None
    eta_seconds: float | None = None
    devices: str = "cpu"
    best_metric: dict[str, Any] | None = None
    recent_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slug: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        d = dict(self.raw) if self.raw else {}
        if self.slug is None and "slug" in d:
            object.__setattr__(self, "slug", d["slug"])
        elif self.slug is not None:
            d["slug"] = self.slug
        d.update({
            "run_id": self.run_id,
            "status": self.status,
            "step": self.step,
            "total_steps": self.total_steps,
            "percent": self.percent,
            "steps_per_sec": self.steps_per_sec,
            "eta_seconds": self.eta_seconds,
            "devices": self.devices,
            "best_metric": self.best_metric,
            "recent_checkpoints": self.recent_checkpoints,
            "warnings": self.warnings,
            "slug": self.slug,
        })
        dict.__init__(self, d)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'RunStatus' object has no attribute {name!r}")

    @property
    def device(self) -> str:
        return str(self.get("device") or self.devices)

    @property
    def pid(self) -> int | None:
        return self.get("pid")


@dataclass
class PsSnapshot(dict):
    """Encapsulates ps() listing enriched with process metrics."""
    runs: list[RunStatus] = field(default_factory=list)
    queued: list[dict[str, Any]] = field(default_factory=list)
    active_count: int = 0
    total_known: int = 0

    def __post_init__(self):
        normalized_runs = [
            r if isinstance(r, RunStatus) else RunStatus(
                run_id=str(r.get("run_id", "")),
                status=str(r.get("status", "unknown")),
                step=r.get("step"),
                total_steps=r.get("total_steps") or r.get("max_steps"),
                percent=r.get("percent"),
                steps_per_sec=r.get("steps_per_sec"),
                eta_seconds=r.get("eta_seconds"),
                devices=str(r.get("device") or r.get("devices", "cpu")),
                slug=r.get("slug"),
                raw=r,
            ) if isinstance(r, dict) else r
            for r in self.runs
        ]
        object.__setattr__(self, "runs", normalized_runs)
        dict.__init__(self, runs=self.runs, queued=self.queued, active_count=self.active_count, total_known=self.total_known)

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def total_count(self) -> int:
        return self.total_known

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.runs if (getattr(r, "status", None) or r.get("status")) in ("completed", "complete"))

    def __iter__(self):
        return iter(super().keys())


class RunHandle:
    """Handle to a training run launched asynchronously in a background thread."""

    def __init__(self, run_dir: Path, thread: threading.Thread | None, run: Run | None = None):
        self.run_dir = Path(run_dir)
        self._thread = thread
        self._run = run
        self._exc: Exception | None = None

    @property
    def run(self) -> Run:
        """The Run instance for this execution (loaded once available)."""
        if self._run is None:
            self._run = open_run(self.run_dir)
        return self._run

    def is_alive(self) -> bool:
        """Check if the training thread/process is active."""
        if self._thread is None:
            return False
        return self._thread.is_alive()

    is_running = is_alive

    def wait(self, timeout: float | None = None) -> Run:
        """Wait for training to complete and return the final Run object."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError(f"Training did not complete within {timeout}s")
        if self._exc is not None:
            raise self._exc
        self._run = open_run(self.run_dir)
        return self._run

    join = wait

    def stop(self, timeout: float = 15.0) -> None:
        """Request graceful stop of the training run."""
        stop(self.run_dir, timeout=timeout)

    @property
    def run_info(self) -> RunInfo:
        """Load current RunInfo for the active run."""
        return read_run_info(self.run_dir)





# --------------------------------------------------------------------------- model files

@dataclass(frozen=True)
class Model:
    """A model file's contents. ``base`` resolves relative ``hndl`` paths.

    ``path`` is the file the spec was read from unchanged; ``origin`` is the
    file it came from even after ``override``, so a fresh run is placed and
    named as ``hypergan train <file>`` would.
    """
    spec: dict
    base: Path | None = None
    path: Path | None = None
    origin: Path | None = field(default=None, compare=False, repr=False)

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
    """Load a model file, an engine recipe TOML, a run's stored Model, or a .pt bundle."""
    if isinstance(path, Run):
        return _load_stored_run_model(path.path)
    raw = str(path)
    path = Path(path)
    if not path.exists() and path.parent.name == "examples" and (path.parent / "archive" / path.name).is_file():
        path = path.parent / "archive" / path.name
    if path.suffix.lower() == '.pt':
        return load_bundle(path)
    if not path.exists() and path.suffix.lower() not in ('.toml', '.pt'):
        try:
            resolved_run = resolve_run_target(raw, allow_latest=(raw == 'latest'))
            if resolved_run.is_dir():
                return _load_stored_run_model(resolved_run)
        except Exception:
            pass
    if not path.exists():
        from .config import resolve_packaged_model_path
        packaged_file = resolve_packaged_model_path(raw)
        if packaged_file is not None:
            path = packaged_file
    if path.is_dir():
        if (path / 'model.toml').is_file():
            path = path / 'model.toml'
        elif (path / 'config.toml').is_file():
            path = path / 'config.toml'
        elif any(
            (path / name).exists()
            for name in ('recipe.toml', 'manifest.json', 'summary.json', 'config-source')
        ):
            return _load_stored_run_model(path)
        else:
            path = path / 'config.toml'
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
            spec = interpolate_model_paths(spec, ws_dict)

    return Model(spec, base=path.parent.resolve(), path=path.resolve(), origin=path.resolve())



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
        # Losses, like other lists of tables, can be addressed by id, critic, or type.
        for index, item in enumerate(container):
            if isinstance(item, dict) and key in (item.get('id'), item.get('critic')):
                return index
        for index, item in enumerate(container):
            if isinstance(item, dict) and key == item.get('type'):
                return index
        raise KeyError(f'{location}: no list element {key!r} (use an index, id, critic, or type name)')
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
        if is_model and dotted in ('trainer', 'training', 'optimizer') and isinstance(value, dict):
            for k, v in value.items():
                norm_changes[f'train.{k}'] = v
            continue
        if not is_model and dotted in ('train', 'trainer') and isinstance(value, dict):
            for k, v in value.items():
                if k in OPTIMIZER_FIELDS and 'optimizer' in spec:
                    norm_changes[f'optimizer.{k}'] = v
                else:
                    norm_changes[f'training.{k}'] = v
            continue
        if dotted.startswith('trainer.'):
            dotted = 'train.' + dotted[len('trainer.'):]
        if not is_model and (dotted.startswith('train.') or dotted.startswith('training.')):
            field = dotted.split('.', 1)[1]
            if field in OPTIMIZER_FIELDS and 'optimizer' in spec:
                norm_changes[f'optimizer.{field}'] = value
            elif 'training' in spec:
                norm_changes[f'training.{field}'] = value
            else:
                norm_changes[dotted] = value
        elif is_model and (dotted.startswith('training.') or dotted.startswith('optimizer.')):
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
        last_key = _segment(target, parts[-1], dotted) if isinstance(target, list) else parts[-1]
        target[last_key] = deepcopy(value)
        if isinstance(target, dict) and last_key == 'id' and value is None:
            target['__id_required__'] = True

    norm_sections = {}
    for key, value in sections.items():
        if is_model and key in ('trainer', 'training', 'optimizer') and isinstance(value, dict):
            norm_sections.setdefault('train', {}).update(deepcopy(value))
        else:
            norm_sections[key] = value

    def merge(into, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(into.get(key), dict):
                merge(into[key], value)
            else:
                into[key] = deepcopy(value)
    merge(spec, norm_sections)
    return Model(spec, base=model.base, path=None, origin=model.path or model.origin)


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
        if isinstance(network, dict):
            for hndl_key in ('hndl', 'file'):
                if network.get(hndl_key) is not None:
                    network[hndl_key] = relocated(network[hndl_key])
            for name, file in (network.get('network_files') or {}).items():
                network['network_files'][name] = relocated(file)
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
    if model.path is not None:
        require_networks(model.spec)  # A file that declares no model never means the packaged default.
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
    raw = extra + list(resolved['warnings'])
    return [
        w if isinstance(w, ValidationWarning) else ValidationWarning(message=str(w))
        for w in raw
    ]


validate_config = validate


def init_project(project_dir, *, template=None, resolution=None, data_dir=None, output_config=None, device="cuda", overwrite=False):
    """Initialize a HyperGAN project directory with hypergan.toml, runs/, and configs/<template>.toml."""
    from .recipe_generator import init_project as _init_project
    return _init_project(project_dir, template=template, resolution=resolution, data_dir=data_dir,
                         output_config=output_config, device=device, overwrite=overwrite)


def new(data_dir=None, output_path=None, *, template=None, resolution=None, device="cuda",
        project=None, overwrite=False, name=None, dataset=None):
    """Scaffold a new HyperGAN model file (or initialize a project directory when project is set).

    ``dataset`` is another name for ``data_dir``, the directory of training images.
    """
    if dataset is not None:
        if data_dir is not None:
            raise TypeError("new() takes data_dir or dataset, not both")
        data_dir = dataset
    if project is not None:
        return init_project(
            project,
            template=template,
            resolution=resolution,
            data_dir=data_dir,
            output_config=output_path,
            device=device,
            overwrite=overwrite,
        )
    from .recipe_generator import generate_recipe as _generate_recipe
    ds_dir = data_dir
    out_p = output_path
    if out_p is None and ds_dir is not None:
        p = Path(ds_dir)
        if p.suffix.lower() == ".toml" or not p.is_dir():
            out_p = ds_dir
            ds_dir = None
    name_arg = str(out_p) if out_p is not None else name
    return _generate_recipe(
        name=name_arg,
        template=template or "dcgan",
        resolution=resolution,
        dataset_dir=ds_dir,
        device=device,
        overwrite=overwrite,
    )


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
    inv_name_map = {v: k for k, v in name_map.items()}
    opt_cfg = recipe.get('optimizer', {})
    rec_cfg = recipe.get('recipe', {})
    eff_lr = opt_cfg.get('lr', rec_cfg.get('lr', 0.0002))
    eff_d_mult = opt_cfg.get('d_lr_mult', rec_cfg.get('d_lr_mult', 1.0))
    eff_d_lr = eff_lr * eff_d_mult if isinstance(eff_lr, (int, float)) and isinstance(eff_d_mult, (int, float)) else None
    g_lr_str = f'optimizer.lr ({eff_lr:g})' if isinstance(eff_lr, (int, float)) else 'optimizer.lr'
    d_lr_str = f'optimizer.lr * d_lr_mult ({eff_d_lr:g})' if isinstance(eff_d_lr, (int, float)) else 'optimizer.lr * d_lr_mult'
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
        fake_path = recipe['adversarial'].get('fake', 'generated')
        if any(p == 'candidate' or p.startswith('candidate.') or producer(p) == 'generator' for p in disc_inputs.values()):
            terms.append(('discriminator', fake_path))
    for t in recipe.get('adversarial_terms', ()):
        terms.append((t['component'], t['fake']))
    for critic_name, fake in terms:
        orig_critic = inv_name_map.get(critic_name, critic_name)
        critic_label = f'{critic_name} ({orig_critic})' if orig_critic != critic_name else critic_name
        for name in upstream(fake, components):
            reach[name].append(f'fooling {critic_label}')

    lines = []
    networks = model.spec.get('networks') if is_model_file(model.spec) else None
    if networks:
        for orig_name, net_spec in networks.items():
            comp_name = name_map.get(orig_name, orig_name)
            role = net_spec.get('role', 'generator')
            spec = components.get(comp_name, {})
            factory = spec.get('factory', net_spec.get('hndl') or net_spec.get('file') or net_spec.get('module') or 'unknown')
            if role == 'critic':
                group = f'critic Adam (shared by all critics), lr = {d_lr_str}'
            elif 'reuse' in net_spec or 'reuse' in spec:
                reuse_target = net_spec.get('reuse') or spec.get('reuse')
                group = f'parameters of {reuse_target}'
            elif net_spec.get('trainable') is False or spec.get('trainable') is False or net_spec.get('freeze_parameters') or spec.get('freeze_parameters'):
                group = 'none (frozen)'
            else:
                group = f'generator-side Adam, lr = {g_lr_str}'
            lines.append(f'{orig_name} [{role}] {factory}')
            inputs_str = ', '.join(f'{k} <- {v}' for k, v in net_spec.get('inputs', {}).items())
            lines.append(f'  inputs: {inputs_str}')
            lines.append(f'  optimizer: {group}')
            if role != 'critic':
                received = sorted(set(reach.get(comp_name, [])))
                lines.append('  gradient from: ' + (', '.join(received) or '-'))
    else:
        from .model_description import roles
        inferred = roles({'components': components, 'adversarial_terms': recipe.get('adversarial_terms', [])},
                         aliases=False)
        for name, spec in components.items():
            orig_name = inv_name_map.get(name, name)
            factory = spec.get('factory', 'reuse')
            role = inferred[name]
            if role == 'critic':
                group = f'critic Adam (shared by all critics), lr = {d_lr_str}'
            else:
                group = f'generator-side Adam, lr = {g_lr_str}'
            lines.append(f'{orig_name} [{role}] {factory}')
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
            sig = _py_inspect.signature(path)
            if not any(p.kind == _py_inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
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
    """Define an adversarial loss term binding a critic network to real and generated inputs."""
    loss = {'type': 'adversarial', 'critic': critic, 'weight': weight, 'penalty': penalty}
    for key, value in (('real', real), ('fake', fake), ('inputs', inputs), ('id', id)):
        if value is not None:
            loss[key] = value
    return loss


def objective(id, fn='mse', *, input=None, target=None, inputs=None, weight=1.0, args=None, type='objective'):
    """Define a loss objective term computing a loss function over specified input tensors."""
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
    """Define a reconstruction loss term between generated output and a target tensor (default MSE)."""
    return objective(id, fn, input=input, target=target, weight=weight, type='reconstruction')


def prior_loss(weight=1.0, **settings):
    """Define a prior regularization loss term with an optional weight and settings."""
    return {'type': 'prior', 'weight': weight, **settings}


# --------------------------------------------------------------------------- training

@dataclass(frozen=True)
class Run:
    """A run directory and the manifest it recorded."""
    path: Path
    manifest: dict = field(repr=False)

    @property
    def status(self):
        """Current lifecycle status of the run (running, completed, stopped, etc.)."""
        return self.manifest.get('status')

    @property
    def step(self):
        """Latest recorded training step count."""
        return self.manifest.get('steps')

    @property
    def stop_reason(self) -> str | None:
        """Reason why training stopped (e.g. early_stop, max_seconds), if any."""
        sr = (self.manifest or {}).get('stop_reason')
        if sr is not None:
            return str(sr)
        summary_p = self.path / 'summary.json'
        if summary_p.is_file():
            try:
                data = json.loads(summary_p.read_text(encoding='utf-8'))
                if isinstance(data, dict) and data.get('stop_reason'):
                    return str(data['stop_reason'])
            except Exception:
                pass
        return None

    @property
    def config(self):
        """Resolved configuration recorded in the run manifest."""
        return self.manifest.get('config')

    @property
    def fingerprint(self):
        """Numerical recipe fingerprint (config_sha256)."""
        return self.manifest.get('config_sha256')

    def _identity(self, key, *, registry=True):
        """``key`` from the manifest, else summary.json, else the global registry entry."""
        value = self.manifest.get(key)
        if value:
            return str(value)
        summary_p = self.path / 'summary.json'
        if summary_p.is_file():
            try:
                data = json.loads(summary_p.read_text(encoding='utf-8'))
                if isinstance(data, dict) and data.get(key):
                    return str(data[key])
            except Exception:
                pass
        if registry:
            try:
                from .registry import _read_registry_unlocked, get_registry_dir
                reg_data = _read_registry_unlocked(get_registry_dir(self.path))
                entry = reg_data.get('runs', {}).get(str(self.path.resolve()))
                if entry and entry.get(key):
                    return str(entry[key])
            except Exception:
                pass
        return None

    @property
    def run_id(self) -> str:
        """Unique 32-character hexadecimal run identifier."""
        return self._identity('run_id') or self.path.name

    @property
    def short_id(self) -> str:
        """First 8 characters of the run identifier."""
        return self._identity('short_id', registry=False) or str(self.run_id)[:8]

    @property
    def name(self) -> str:
        """Human-readable experiment name or custom tag."""
        nm = self.manifest.get('name') or (self.manifest.get('config') or {}).get('name')
        return str(nm) if nm else (self._identity('name') or self.path.name)

    @property
    def slug(self) -> str:
        """Memorable '<adj>-<noun>' run slug."""
        found = self._identity('slug')
        if found:
            return found
        from .registry import extract_or_derive_slug
        return extract_or_derive_slug(str(self.run_id), name=str(self.name), run_dir_name=self.path.name)


def open_run(run_dir=None):
    """Open and load a Run instance and its manifest by directory, slug, tag, or short ID."""
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


def _run_path(value=None):
    if isinstance(value, Run):
        return value.path
    from .registry import resolve_run_target
    return resolve_run_target(value)


# A run directory holding a different configuration refuses to continue with one of these.
_CONFIG_DIFFERS = ("differs from the original run", "differs from the run manifest")


def _config_differs(error):
    return any(text in str(error) for text in _CONFIG_DIFFERS)


def _source_file(model_obj):
    """The model file behind ``model_obj`` (a path, or a Model loaded and maybe overridden)."""
    if isinstance(model_obj, (str, Path)):
        return model_obj
    return getattr(model_obj, "path", None) or getattr(model_obj, "origin", None)


def _run_stem(model_obj):
    """The config file stem a fresh run is named after (the model name for in-memory models)."""
    cfg_path = _source_file(model_obj)
    if cfg_path is not None:
        return Path(cfg_path).stem
    return (getattr(model_obj, "name", None) or "run").split("/")[-1] or "run"


def _default_run_dir(model_obj, name=None):
    """The fresh run directory ``hypergan train`` and ``hg.train`` use when none is given.

    ``model_obj`` is a Model or a model file path. The workspace is found from
    the model file, including the file an overridden Model came from (from the
    working directory for packaged and in-memory models); the rule itself is
    ``hypergan.workspace.default_run_dir``.
    """
    from .workspace import default_run_dir
    cfg_path = _source_file(model_obj)
    in_package = cfg_path is not None and PACKAGED.resolve() in Path(cfg_path).resolve().parents
    start = None if cfg_path is None or in_package else cfg_path
    return default_run_dir(_run_stem(model_obj), name=name or None, start=start)


def prepare(model, run_dir=None, steps=None, *, name=None, profile=None, service_policy=None,
            devices=None, device=None, nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
            checkpoint_every=None, max_seconds=None, stop_after_steps=None,
            preview_every=None, previews=None, preview_keep=None, preview_keep_source=None,
            preview_name=None, **extra):
    """Validate and prepare a training run, returning a PreparedExecution.

    Without ``run_dir`` the run gets a fresh directory as ``hypergan train``
    gives it (see ``_default_run_dir``); ``name`` names that directory and the
    run instead of a generated slug.
    """
    from .execution import prepare_train as _prepare_train
    model = _model(model)
    if name:
        model = override(model, {"name": name})
    defaulted = run_dir is None
    if defaulted:
        run_dir = _default_run_dir(model, name=name)
    pe = preview_every if preview_every is not None else previews
    lowered = lower(model)
    if getattr(model, 'path', None):
        lowered['_source_path'] = str(model.path)
    if device is not None:
        if isinstance(lowered, dict) and "training" in lowered:
            lowered["training"]["device"] = device
        elif isinstance(lowered, dict):
            lowered.setdefault("training", {})["device"] = device

    def attempt(target):
        return _prepare_train(lowered, target, steps=steps, profile=profile,
                              service_policy=service_policy, devices=devices, device=device,
                              nodes=nodes, node_rank=node_rank, rendezvous=rendezvous,
                              strategy=strategy, sync_every=sync_every,
                              checkpoint_every=checkpoint_every,
                              max_seconds=max_seconds, stop_after_steps=stop_after_steps,
                              preview_every=pe, preview_keep=preview_keep,
                              preview_keep_source=preview_keep_source, preview_name=preview_name,
                              **extra)
    try:
        return attempt(run_dir)
    except ValueError as error:
        # A named default directory that holds another configuration starts a
        # sibling run, as the CLI does, instead of refusing.
        if not (defaulted and _config_differs(error)):
            raise
        sibling, _ = generate_run_slug(name or _run_stem(model), parent_dir=Path(run_dir).parent)
        return attempt(Path(run_dir).parent / sibling)


def _load_stored_run_model(run_path: Path):
    """Load a run's stored Model from recipe.toml, model.toml, config-source/*.toml, or manifest.json."""
    if (run_path / 'model.toml').is_file():
        return load(run_path / 'model.toml')
    if (run_path / 'recipe.toml').is_file():
        return load(run_path / 'recipe.toml')
    if (run_path / 'config.toml').is_file():
        return load(run_path / 'config.toml')
    cs_dir = run_path / 'config-source'
    if cs_dir.is_dir():
        cs_manifest = cs_dir / 'manifest.json'
        if cs_manifest.is_file():
            try:
                cs_data = json.loads(cs_manifest.read_text(encoding='utf-8'))
                files = cs_data.get('files') or []
                for f_entry in files:
                    if isinstance(f_entry, dict) and f_entry.get('snapshot'):
                        snap = cs_dir / f_entry['snapshot']
                        if snap.is_file() and snap.name != 'hypergan.toml':
                            return load(snap)
            except Exception:
                pass
        tomls = [p for p in sorted(cs_dir.rglob('*.toml')) if p.name != 'hypergan.toml']
        if tomls:
            return load(tomls[0])
    if (run_path / 'manifest.json').is_file():
        with (run_path / 'manifest.json').open('rb') as f:
            manifest = json.load(f)
        saved_config = manifest.get('config')
        if not isinstance(saved_config, dict):
            raise ValueError(f"Run {run_path} has no recorded configuration in manifest.json")
        if 'components' in saved_config and 'networks' not in saved_config:
            try:
                return lift(saved_config, base=run_path)
            except Exception:
                pass
        return from_dict(saved_config, base=run_path)
    raise ValueError(f"Run directory {run_path} does not exist or has no configuration")


def prepare_resume(run_dir, checkpoint=None, model=None, *, steps=None,
                   profile=None, service_policy=None, devices=None, device=None,
                   nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
                   checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                   preview_every=None, previews=None, preview_keep=None, preview_keep_source=None,
                   preview_name=None, sets=None, overrides=None, **extra):
    """Validate and prepare a run resumption, returning a PreparedExecution."""
    from .execution import prepare_resume as _prepare_resume
    run_path = _run_path(run_dir)
    merged_sets = dict(sets or {})
    if overrides:
        merged_sets.update(overrides)
    sets = merged_sets or None
    if model is None and (sets or steps is not None):
        model = _load_stored_run_model(run_path)
    allow_batch_size_change = bool(extra.pop('_allow_batch_size_change', False))
    allow_device_change = bool(extra.pop('_allow_device_change', False))
    if sets:
        normalized_sets = {}
        for k, v in sets.items():
            if k.startswith('trainer.'):
                k = 'train.' + k[len('trainer.'):]
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
        if 'train.batch_size' in normalized_sets or 'training.batch_size' in normalized_sets:
            allow_batch_size_change = True
        dev_val = normalized_sets.get('train.device', normalized_sets.get('training.device'))
        if dev_val is not None:
            allow_device_change = True
            if device is None and devices is None:
                device = str(dev_val)

    config_path = lower(model) if model is not None else None
    if device is not None:
        if isinstance(config_path, dict) and "training" in config_path:
            config_path["training"]["device"] = device
        elif isinstance(config_path, dict):
            config_path.setdefault("training", {})["device"] = device
    pe = preview_every if preview_every is not None else previews
    return _prepare_resume(run_path, checkpoint=checkpoint, config_path=config_path, steps=steps,
                           _allow_device_change=allow_device_change,
                           _allow_batch_size_change=allow_batch_size_change,
                           profile=profile, service_policy=service_policy, devices=devices, device=device,
                           nodes=nodes, node_rank=node_rank, rendezvous=rendezvous,
                           strategy=strategy, sync_every=sync_every,
                           checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                           stop_after_steps=stop_after_steps, preview_every=pe,
                           preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                           preview_name=preview_name, **extra)


def train(model, run_dir=None, *, steps=None, name=None, previews=None, preview_every=None, checkpoint_every=None,
          profile=None, devices=None, device=None, nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
          on_event=None, **controls) -> Run:
    """Train (or continue, if ``run_dir`` exists) and return the ``Run``.

    If ``run_dir`` is omitted or None, each call starts a fresh run directory
    the way ``hypergan train`` does: ``<runs_root>/<stem>-<adj>-<noun>``, or
    ``<runs_root>/<name>`` when ``name`` is given.
    """
    pe = preview_every if preview_every is not None else previews
    prep = prepare(model, run_dir, steps=steps, name=name, profile=profile, devices=devices, device=device,
                   nodes=nodes, node_rank=node_rank, rendezvous=rendezvous, strategy=strategy, sync_every=sync_every,
                   checkpoint_every=checkpoint_every, preview_every=pe, **controls)
    prep.run(on_event=on_event)
    return open_run(prep.run_dir)


def launch(model, run_dir=None, *, steps=None, name=None, previews=None, preview_every=None, checkpoint_every=None,
           profile=None, devices=None, device=None, nodes=None, node_rank=None, rendezvous=None, strategy=None, sync_every=None,
           on_event=None, **controls) -> RunHandle:
    """Launch a training run asynchronously in a background thread and return a RunHandle."""
    pe = preview_every if preview_every is not None else previews
    prep = prepare(model, run_dir, steps=steps, name=name, profile=profile, devices=devices, device=device,
                   nodes=nodes, node_rank=node_rank, rendezvous=rendezvous, strategy=strategy, sync_every=sync_every,
                   checkpoint_every=checkpoint_every, preview_every=pe, **controls)
    handle = RunHandle(run_dir=prep.run_dir, thread=None)

    def _worker():
        try:
            prep.run(on_event=on_event)
        except Exception as exc:
            handle._exc = exc

    run_name = getattr(model, "name", "run")
    thread = threading.Thread(target=_worker, name=f"hypergan-train-{run_name}", daemon=True)
    handle._thread = thread
    thread.start()
    return handle


def resume(run, model=None, *, steps=None, devices=None, device=None, on_event=None, **controls) -> Run:
    """Continue a run from its latest checkpoint (optionally with changed observation settings)."""
    run = _run(run)
    prep = prepare_resume(run.path, model=model, steps=steps, devices=devices, device=device, **controls)
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


class InferenceBundle(tuple):
    """Loaded inference bundle supporting both 5-tuple unpacking and dictionary access."""

    def __contains__(self, key):
        if len(self) > 0 and isinstance(self[0], dict) and key in self[0]:
            return True
        return super().__contains__(key)

    def __getitem__(self, item):
        if isinstance(item, str) and len(self) > 0 and isinstance(self[0], dict) and item in self[0]:
            return self[0][item]
        return super().__getitem__(item)


def load_inference(target=None):
    """Verify and load a run's EMA inference bundle on CPU in evaluation mode.

    Accepts an exported .pt bundle file, a run directory, slug, tag, short ID, or Run instance.
    Returns (state_dict, config_dict, ComponentGraph, prior, bundle_sha256).
    """
    import torch
    from .recipes import ComponentGraph, make_prior
    from .config import resolve_config

    if isinstance(target, Run):
        run_dir = target.path
        is_direct_pt = False
    elif target is not None and (str(target).endswith(".pt") or (Path(target).is_file() and Path(target).suffix.lower() == ".pt")):
        is_direct_pt = True
        pt_path = Path(target).resolve()
    else:
        is_direct_pt = False
        run_dir = _run_path(target)

    if is_direct_pt:
        path = pt_path
        checksum_path = pt_path.with_name("model.sha256") if pt_path.name == "model.pt" else pt_path.with_suffix(".sha256")
        if checksum_path.is_file():
            expected = checksum_path.read_text(encoding="utf-8").strip()
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Inference artifact hash mismatch")
        else:
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        bundle_dir = run_dir
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("bundle_path"):
                    bp = Path(manifest["bundle_path"])
                    bundle_dir = (bp if bp.is_absolute() else run_dir / bp).parent
            except Exception:
                pass
        path = bundle_dir / "model.pt"
        if not path.is_file():
            try:
                from .checkpoints import resolve_checkpoint, export_checkpoint_bundle
                ckpt_dir = resolve_checkpoint(run_dir, target="latest")
                path, checksum_path = export_checkpoint_bundle(ckpt_dir, output_dir=run_dir)
                bundle_dir = run_dir
            except Exception:
                pass
        if not path.is_file():
            raise FileNotFoundError(f"No inference bundle or checkpoint found for {target or run_dir}")
        sha_file = bundle_dir / "model.sha256"
        if sha_file.is_file():
            expected = sha_file.read_text(encoding="utf-8").strip()
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Inference artifact hash mismatch")
        else:
            expected = hashlib.sha256(path.read_bytes()).hexdigest()

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
    return InferenceBundle((state, config, graph, prior, expected))


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
    from .previews import index_path
    run = _run(run)
    index = run.path / 'previews' / 'index.json'
    if not index.is_file():
        return []
    result = []
    for entry in json.loads(index.read_text())['previews']:
        if name is not None and entry.get('name') != name:
            continue
        path = index_path(run.path, entry['path'])
        payload = json.loads(path.read_text())
        grid = payload.get('image_grid')
        result.append(Sample(step=payload['step'], name=payload['name'], data=_tensor(payload.get('samples')),
                             inputs={key: _tensor(value) for key, value in payload.get('inputs', {}).items()
                                     if isinstance(value, list)},
                             particle_ids=payload.get('particle_ids'), source='preview', path=path,
                             png=base64.b64decode(grid['png_base64']) if grid else None))
    return sorted(result, key=lambda item: item.step)


def sample(run=None, count=16, seed=None, *, inputs=None, output=None, checkpoint=None, default_image_png=False, device=None, output_dir=None):
    """Fresh samples from the run's final EMA model (or target checkpoint or .pt bundle)."""
    from .artifacts import sample as draw
    if isinstance(run, (str, Path)) and (str(run).endswith('.pt') or (Path(run).is_file() and Path(run).suffix.lower() == '.pt')):
        target_path = Path(run)
    else:
        run_obj = _run(run)
        target_path = run_obj.path

    if seed is None:
        cfg = getattr(run, "config", None) if isinstance(run, Run) else None
        if cfg is None and target_path.is_dir() and (target_path / "manifest.json").is_file():
            try:
                m_data = json.loads((target_path / "manifest.json").read_text(encoding="utf-8"))
                cfg = m_data.get("config")
            except Exception:
                pass
        seed = (cfg or {}).get("sampling", {}).get("seed", 42) if isinstance(cfg, dict) else 42

    path = Path(draw(target_path, count=count, seed=seed, inputs=inputs, output=output,
                     checkpoint=checkpoint, default_image_png=default_image_png, device=device, output_dir=output_dir))
    if path.is_dir():
        return Sample(step=0, name='sample', data=None, inputs={}, particle_ids=None, source='inference', path=path)
    if path.suffix.lower() == '.png':
        from PIL import Image
        with Image.open(path, formats=["PNG"]) as img:
            raw_meta = img.info.get('hypergan', '{}')
            payload = json.loads(raw_meta) if raw_meta else {}
        return Sample(step=payload.get('step', 0), name='sample', data=None,
                      inputs={}, particle_ids=payload.get('particle_ids'),
                      source='inference', path=path, png=path.read_bytes())
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


def resolve_run_target(target=None, *, cwd=None, allow_latest=True, require_active=False):
    """Resolve a run reference (short ID, slug, name, path, @latest, or None for active/latest)."""
    from .registry import resolve_run_target as _resolve_target
    return _resolve_target(target, cwd=cwd, allow_latest=allow_latest, require_active=require_active)


def register_run(run_dir, **kwargs):
    """Register a run directory in the global runs registry (runs.json)."""
    from .registry import register_run as _register
    return _register(run_dir, **kwargs)


def sync_registry(cwd=None, extra_roots=None):
    """Synchronize runs in workspace and runs roots into the global registry."""
    from .registry import sync_registry as _sync
    return _sync(cwd=cwd, extra_roots=extra_roots)


def generate_run_slug(stem, parent_dir=None):
    """Generate a unique run name and short '<adj>-<noun>' slug for a model stem."""
    from .registry import generate_run_slug as _gen_slug
    return _gen_slug(stem, parent_dir=parent_dir)


def export(run=None, checkpoint="latest", output=None, format="bundle", include_example_inputs=False, device="cpu"):
    """Export a standalone EMA inference model bundle from a run checkpoint."""
    if output is None and isinstance(checkpoint, (str, Path)):
        ckpt_str = str(checkpoint).lower()
        if ckpt_str.endswith(('.pt', '.safetensors', '.onnx', '.pt2', '.bin')):
            output = checkpoint
            checkpoint = "latest"
    from .artifacts import export_generator as _export
    return _export(run, output_path=output, format=format, include_example_inputs=include_example_inputs, checkpoint=checkpoint, device=device)


def export_generator(run_dir=None, output_path=None, format="bundle", include_example_inputs=False, checkpoint="latest", device="cpu"):
    """Export standalone generator across bundle, safetensors, pt2, or onnx formats."""
    from .artifacts import export_generator as _export
    return _export(run_dir, output_path=output_path, format=format, include_example_inputs=include_example_inputs, checkpoint=checkpoint, device=device)


def interpolate(v0, v1, steps=10, *, mode="slerp", t=None):
    """Perform linear or spherical linear interpolation between two latent vectors."""
    from .artifacts import interpolate as _interpolate
    return _interpolate(v0, v1, steps=steps, mode=mode, t=t)


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


def load_bundle(path_or_run, *, checkpoint="latest", device="cpu"):
    """Load an exported .pt bundle file or extract the EMA bundle from a run checkpoint."""
    from .artifacts import LoadedGenerator
    if isinstance(path_or_run, Run):
        run_dir = path_or_run.path
    else:
        p = Path(path_or_run)
        if p.suffix.lower() == ".pt" or (p.is_file() and p.name.endswith(".pt")):
            return LoadedGenerator(p, device=device)
        run_dir = _run_path(path_or_run)
    try:
        ckpt_dir = resolve_checkpoint(run_dir, target=checkpoint)
        bundle_dict = extract_ema_bundle(ckpt_dir)
        return LoadedGenerator(bundle_dict, device=device)
    except FileNotFoundError:
        model_pt = run_dir / "model.pt"
        if model_pt.is_file():
            return LoadedGenerator(model_pt, device=device)
        raise


def inspect(run=None):
    """Inspect a run and return its parsed manifest and run identity dictionary."""
    if isinstance(run, Run):
        run_path = run.path
        data = dict(run.manifest)
    else:
        try:
            opened = open_run(run)
            run_path = opened.path
            data = dict(opened.manifest)
        except FileNotFoundError:
            run_path = _run_path(run)
            if run_path.is_dir():
                return status(run_path)
            raise
    data.setdefault("run_dir", str(run_path))
    summary_p = run_path / "summary.json"
    summary = {}
    if summary_p.is_file():
        try:
            loaded = json.loads(summary_p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}
    if "name" not in data:
        cfg_name = data.get("config", {}).get("name") if isinstance(data.get("config"), dict) else None
        data["name"] = summary.get("name") or cfg_name or run_path.name
    elif summary.get("name") and data.get("name") != summary.get("name"):
        data["name"] = summary["name"]
    if "slug" not in data:
        from .registry import extract_or_derive_slug
        run_id = str(data.get("run_id") or summary.get("run_id") or run_path.name)
        data["slug"] = summary.get("slug") or extract_or_derive_slug(run_id, name=str(data.get("name", "")), run_dir_name=run_path.name)
    if "short_id" not in data and (data.get("run_id") or summary.get("run_id")):
        data["short_id"] = str(data.get("run_id") or summary.get("run_id"))[:8]
    if "step" not in data:
        if "step" in summary:
            data["step"] = summary["step"]
        elif "steps" in data:
            data["step"] = data["steps"]
    return data


def status(run=None) -> RunStatus:
    """Compile human-friendly status summary for a run."""
    from .inspection import run_status as _status
    res = _status(_run_path(run))
    if isinstance(res, RunStatus):
        return res
    return RunStatus(
        run_id=str(res.get("run_id", "")),
        status=str(res.get("status", "unknown")),
        step=res.get("step"),
        total_steps=res.get("total_steps"),
        percent=res.get("percent"),
        steps_per_sec=res.get("steps_per_sec"),
        eta_seconds=res.get("eta_seconds"),
        devices=str(res.get("devices", "cpu")),
        best_metric=res.get("best"),
        recent_checkpoints=list(res.get("recent_checkpoints", [])),
        warnings=list(res.get("warnings", [])),
        raw=res,
    )


def ps(all_runs=False, cwd=None, *, all=None, root=None) -> PsSnapshot:
    """List active training runs (or all runs if all_runs=True) enriched with GPU VRAM and queue status."""
    if all is not None:
        all_runs = bool(all)
    target_cwd = root if root is not None else cwd
    from .inspection import ps_runs as _ps
    res = _ps(all_runs=all_runs, cwd=target_cwd)
    if isinstance(res, PsSnapshot):
        return res
    raw_runs = res.get("runs", [])
    runs = [
        r if isinstance(r, RunStatus) else RunStatus(
            run_id=str(r.get("run_id", "")),
            status=str(r.get("status", "unknown")),
            step=r.get("step"),
            total_steps=r.get("total_steps") or r.get("max_steps"),
            percent=r.get("percent"),
            steps_per_sec=r.get("steps_per_sec"),
            eta_seconds=r.get("eta_seconds"),
            devices=str(r.get("device") or r.get("devices", "cpu")),
            slug=r.get("slug"),
            raw=r,
        ) if isinstance(r, dict) else r
        for r in raw_runs
    ]
    return PsSnapshot(
        runs=runs,
        queued=list(res.get("queued", [])),
        active_count=int(res.get("active_count", 0)),
        total_known=int(res.get("total_known", 0)),
    )


def stop(run=None, *, stop_all=False, timeout=15.0):
    """Gracefully stop an active training run (or all active runs) via SIGINT."""
    from .inspection import stop_run as _stop
    return _stop(run, stop_all=stop_all, timeout=timeout)


def logs(run=None, *, tail=50, follow=False, raw=False, stream=None):
    """Read or follow a run's training log and return its lines.

    Without ``follow`` the last ``tail`` lines are returned, and written only
    when ``stream`` is given. With ``follow`` they are also written as they
    arrive, to ``stream`` or stdout, until the run ends or Ctrl+C. ``raw``
    gives the events.jsonl lines instead of formatted ones.
    """
    import io
    target_stream = stream if stream is not None or follow else io.StringIO()
    if raw:
        return _raw_logs(_run_path(run), tail=tail, follow=follow, stream=target_stream or sys.stdout)
    from .inspection import read_run_logs as _logs
    return _logs(run, tail=tail, follow=follow, stream=target_stream)


_TERMINAL_EVENTS = ("complete", "completed", "stopped", "failed")


def _raw_logs(run_dir, *, tail, follow, stream):
    import time
    from .inspection import check_run_active
    events_file = run_dir / "events.jsonl"
    lines, offset = [], 0
    if events_file.is_file():
        raw_bytes = events_file.read_bytes()
        if follow:
            # Hold back a partly written last line; the follow loop prints it whole.
            raw_bytes = raw_bytes[:raw_bytes.rfind(b"\n") + 1]
        offset = len(raw_bytes)
        lines = [ln for ln in raw_bytes.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        if tail and tail > 0:
            lines = lines[-tail:]
        for ln in lines:
            stream.write(ln + "\n")
        stream.flush()
    if not follow:
        return lines
    try:
        while True:
            chunk = b""
            if events_file.is_file():
                try:
                    with events_file.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read()
                except OSError:
                    pass
            # Only whole lines: a writer may be part way through the last one.
            complete = chunk[:chunk.rfind(b"\n") + 1]
            offset += len(complete)
            for ln in complete.decode("utf-8", errors="replace").splitlines():
                if not ln.strip():
                    continue
                stream.write(ln + "\n")
                stream.flush()
                lines.append(ln)
                try:
                    if json.loads(ln).get("event") in _TERMINAL_EVENTS:
                        return lines
                except (ValueError, AttributeError):
                    pass
            if not check_run_active(run_dir)[0]:
                return lines
            time.sleep(0.25)
    except KeyboardInterrupt:
        return lines


def tag(run, name):
    """Rename or tag a training run in its manifest, summary, and global registry."""
    from .inspection import tag_run
    res = dict(tag_run(run, name))
    run_id = str(res.get("run_id") or res.get("short_id") or "")
    r_dir = res.get("run_dir", "")
    r_name = res.get("name", name)
    res.setdefault("short_id", run_id[:8])
    res.setdefault("message", f"Tagged {run_id[:8]} as '{r_name}' ({r_dir})")
    return res


def rm(run, *, force=False):
    """Remove a run directory and its registry entry (stops active run if force=True)."""
    from .inspection import remove_run
    res = dict(remove_run(run, force=force))
    run_id = str(res.get("run_id") or res.get("short_id") or "")
    r_dir = res.get("run_dir", "")
    r_name = res.get("name", Path(r_dir).name if r_dir else "")
    res.setdefault("short_id", run_id[:8])
    res.setdefault("message", f"Removed run '{r_name}' ({run_id[:8]}) at {r_dir}")
    return res


def list_runs(root=None, *, sort="updated", status=None, status_filter=None, cwd=None):
    """Discover and list all runs under root with filtering and sorting."""
    from .inspection import list_runs as _list_runs
    filt = status if status is not None else status_filter
    return _list_runs(root, sort=sort, status_filter=filt, cwd=cwd)


runs = list_runs


def runs_ls(root=None, sort="updated", status_filter=None, *, status=None, cwd=None):
    """Discover and list all runs under root with filtering and sorting."""
    return list_runs(root, sort=sort, status=status, status_filter=status_filter, cwd=cwd)


def runs_compare(runs, metric="loss/g_total", at_steps=None):
    """Compare multiple runs for a given metric across step milestones."""
    from .inspection import compare_runs as _compare_runs
    return _compare_runs(runs, metric=metric, at_steps=at_steps)


def gc(run=None, dry_run=False, stride=10, *, all_runs=False, all=None):
    """Clean rebuildable views and compact events on a finished run (or all runs if all_runs=True)."""
    if all is not None:
        all_runs = bool(all)
    if all_runs:
        return gc_all(root=run, dry_run=dry_run, stride=stride)
    from .inspection import gc_run as _gc_run
    return _gc_run(run.path if isinstance(run, Run) else run, dry_run=dry_run, stride=stride)


prune = gc


def gc_all(root=None, dry_run=False, stride=10):
    """Clean all finished runs under root."""
    from .inspection import gc_all as _gc_all
    return _gc_all(root, dry_run=dry_run, stride=stride)


def compact_events(run, stride=10):
    """Compact redundant train events in events.jsonl, returning reclaimed bytes."""
    from .inspection import compact_events as _compact
    return _compact(run.path if isinstance(run, Run) else run, stride=stride)


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
    """Construct an Audio object from a 1D (mono) or 2D (channels, samples) tensor/array with sample rate."""
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
    if isinstance(raw, Model):
        return raw if is_model_file(raw.spec) else lift(raw.spec, base=base or raw.base)
    raw = deepcopy(raw)
    base = Path.cwd() if base is None else Path(base)
    if is_model_file(raw):
        return from_dict(raw, base=base)
    components = raw.get('components', {}) or deepcopy(DEFAULT.get('components', {}))
    networks = {}
    losses = []

    from .model_description import roles
    inferred = roles({'components': components, 'adversarial_terms': raw.get('adversarial_terms', [])}, aliases=False)

    for name, spec in components.items():
        net = {'role': inferred[name], 'inputs': dict(spec.get('inputs', {}))}
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
        prior_spec = deepcopy(raw['prior'])
        prior_args = prior_spec.pop('args', None)
        if isinstance(prior_args, dict):
            for k, v in prior_args.items():
                prior_spec.setdefault(k, v)
        model_spec['prior'] = prior_spec
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
                for k in ('timeout', 'on_error', 'on_busy'):
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


def describe(path=None, networks=False, write=False) -> ModelDescription:
    """Describe a model's formulation, losses, optimizers, and network architectures."""
    from .model_description import MODEL_FILE, build_networks, describe_run, load as load_model_desc
    manifest = {}
    config = None
    resolved_path = None

    if isinstance(path, Model):
        config = resolve(path)
        manifest = {"config": config, "config_sha256": fingerprint(path)}
    elif isinstance(path, Run):
        resolved_path = path.path
        manifest, config = load_model_desc(resolved_path)
    elif isinstance(path, dict):
        if "components" in path or "networks" in path:
            config = path
            manifest = {"config": config}
        else:
            manifest, config = load_model_desc(path)
    else:
        if path is None or (not Path(path).exists() and not str(path).endswith(".toml")):
            from .registry import resolve_run_target
            try:
                resolved_path = resolve_run_target(path)
            except Exception:
                if path is None:
                    raise
                resolved_path = Path(path)
        else:
            resolved_path = Path(path)
        if path is not None and not resolved_path.exists():
            from .config import resolve_packaged_model_path
            packaged_file = resolve_packaged_model_path(path)
            if packaged_file is not None:
                resolved_path = packaged_file
        manifest, config = load_model_desc(resolved_path)

    if write and (resolved_path is None or not (resolved_path.is_dir() and manifest)):
        raise ValueError("--write needs a run directory")
    if not isinstance(config, dict):
        raise ValueError("No model configuration recorded for this run")
    recorded = None
    if networks or write:
        recorded = build_networks(config, manifest.get("config_sha256"))
    if write and resolved_path is not None:
        from .run_state import atomic_json
        atomic_json(resolved_path / MODEL_FILE, recorded)
    res = describe_run(manifest if manifest else {"config": config}, recorded=recorded)
    if isinstance(res, ModelDescription):
        return res
    return ModelDescription(
        name=str(res.get("name", "")),
        family=str(res.get("family", "")),
        components=dict(res.get("components", {})),
        losses=dict(res.get("losses", {})),
        optimizers=dict(res.get("optimizers", {})),
        prior=dict(res.get("prior", {})),
        training=dict(res.get("training", {})),
        raw=res,
    )


describe_model = describe


def data_check(config, output=None):
    """Inspect and inventory a dataset folder defined in config, optionally writing a manifest."""
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


check_data = data_check


def dataset_info(name="cifar10"):
    """Return source URL, version, expected bytes, checksums, and usage notice for a qualified dataset."""
    from .dataset_download import dataset_info as _dataset_info
    return _dataset_info(name)


def list_datasets():
    """Return metadata for all qualified datasets available via explicit opt-in download."""
    from .dataset_download import list_datasets as _list_datasets
    return _list_datasets()


def download_dataset(name="cifar10", destination=None, *, cache_dir=None, url=None,
                     source_archive=None, verify_only=False, timeout=60.0):
    """Acquire, verify, and extract a qualified dataset into destination, writing dataset-manifest.json."""
    from .dataset_download import download_dataset as _download_dataset
    return _download_dataset(
        name,
        destination,
        cache_dir=cache_dir,
        url=url,
        source_archive=source_archive,
        verify_only=verify_only,
        timeout=timeout,
    )



def preflight(config, profile=None, runtime=False):
    """Validate execution environment, hardware, and configuration compatibility."""
    from .execution_profiles import load_execution_profile
    from .execution_preflight import _resolve_profile, preflight as run_preflight
    from .config import load_config
    cfg = resolve(config) if isinstance(config, Model) or is_model_file(config) else load_config(config)
    prof = (load_execution_profile(profile, cfg) if profile is not None
            else _resolve_profile(None, cfg))
    if runtime:
        return run_preflight(cfg, prof)
    return {"schema_version": 1, "stage": "structural", "profile": prof, "runtime_checked": False}


def preflight_check(config, profile=None, runtime=True):
    """Run preflight check on config (defaults to runtime=True)."""
    return preflight(config, profile=profile, runtime=runtime)


def serve(run_dir=None, port=None, host="127.0.0.1", auth="none", session_file=None, open_browser=False, public_origin=None, dev=False, bind=None):
    """Launch the Web Observatory server for a runs root or run directory."""
    if bind is not None:
        host = bind
    if dev:
        from .web_dev import enable
        enable()
    from .web_launch import serve as launch_serve
    target = find_runs_root(Path.cwd()) if run_dir is None else run_dir
    launch_serve(target, port=port, host=host, auth=auth,
                 session_file=session_file, open_browser=open_browser,
                 public_origin=public_origin)


def viewer_status(run_dir=None):
    """Check the running status and port of a viewer process for a run or runs root."""
    from .web_autostart import viewer_status as autostart_status
    return autostart_status(_run_path(run_dir))


def stop_viewer(run_dir=None):
    """Stop a running viewer process for a run directory."""
    from .web_autostart import stop_viewer as autostart_stop
    return autostart_stop(_run_path(run_dir))


def contributions(run_dir=None, map_revision=None, cursor=None, limit=100, max_bytes=1048576):
    """Read a page of mapped contribution events, auto-projecting if needed."""
    from .event_views import MapSpec, read_projection_page
    run_path = _run_path(run_dir)
    rev = map_revision or MapSpec().revision
    view_dir = run_path / 'views' / rev
    has_events = (run_path / 'events.jsonl').is_file()
    has_views = (view_dir / 'contributions.jsonl').is_file() and (view_dir / 'metadata.json').is_file()
    if has_events and not has_views:
        project(run_path)
    return read_projection_page(run_path, rev, cursor, limit=limit, max_bytes=max_bytes)


def project(run_dir=None, map_spec=None, follow=False, limit=100, on_progress=None):
    """Project run events into the mapped contribution views."""
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


def request_status(run_dir=None, request_id=None):
    """Check the status of an asynchronous control request (checkpoint, stop, or sample)."""
    from .run_requests import request_status as _req_status
    return _req_status(_run_path(run_dir), request_id)


checkpoint_status = request_status


def _submit_control_request(run_dir=None, *, action="checkpoint", attempt_id=None, request_id=None):
    from .run_requests import submit_request
    run_dir = _run_path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        from .inspection import check_run_active
        if check_run_active(run_dir)[0]:
            raise ValueError(f"Run is still initializing step 0; {action} will be available once step 1 starts")
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not manifest.get("run_id") or not manifest.get("attempt_id"):
        from .inspection import check_run_active
        if check_run_active(run_dir)[0] or (isinstance(manifest, dict) and manifest.get("status") == "initializing"):
            raise ValueError(f"Run is still initializing step 0; {action} will be available once step 1 starts")
        raise ValueError(f"Run manifest has no run/attempt identity for a {action} request")
    if manifest.get("status") == "initializing":
        raise ValueError(f"Run is still initializing step 0; {action} will be available once step 1 starts")
    if manifest.get("status") != "running":
        from .inspection import check_run_active
        if check_run_active(run_dir)[0] and int(manifest.get("steps", 0) or 0) == 0:
            raise ValueError(f"Run is still initializing step 0; {action} will be available once step 1 starts")
        if request_id is None:
            raise ValueError("Run is not running; use resume to continue a stopped run")
        try:
            existing = request_status(run_dir, request_id)
        except FileNotFoundError as exc:
            raise ValueError("Run is not running; use resume to continue a stopped run") from exc
        target_attempt = attempt_id or existing["request"]["attempt_id"]
        if (
            existing["request"]["run_id"] != manifest["run_id"]
            or existing["request"]["attempt_id"] != target_attempt
            or existing["request"]["action"] != action
        ):
            raise ValueError("Conflicting control request ID: reuse requires the same run, attempt and action")
        return existing
    return submit_request(
        run_dir,
        action=action,
        run_id=manifest["run_id"],
        attempt_id=attempt_id or manifest["attempt_id"],
        request_id=request_id,
    )


def request_checkpoint(run_dir=None, attempt_id=None, request_id=None):
    """Submit an asynchronous request for the training process to save a checkpoint."""
    return _submit_control_request(run_dir, action="checkpoint", attempt_id=attempt_id, request_id=request_id)


def request_sample(run_dir=None, attempt_id=None, request_id=None):
    """Submit an asynchronous request for the training process to emit a preview/sample."""
    return _submit_control_request(run_dir, action="sample", attempt_id=attempt_id, request_id=request_id)


def request_stop(run_dir=None, attempt_id=None, request_id=None):
    """Submit an asynchronous request for the training process to stop cleanly at a boundary."""
    return _submit_control_request(run_dir, action="stop", attempt_id=attempt_id, request_id=request_id)


def ensure_dashboard_running(runs_root=None, check_version=True, run_id=None):
    """Ensure the persistent root dashboard is running for runs_root, returning its URL."""
    from .web_autostart import ensure_dashboard_running as autostart_ensure
    target = Path(runs_root) if runs_root is not None else None
    return autostart_ensure(target, check_version=check_version, run_id=run_id)


def stop_all_servers(force=False):
    """Terminate all background viewer, dashboard, and supervisor processes on the host."""
    from .web_autostart import stop_all_servers as autostart_stop_all
    return autostart_stop_all(force=force)


def dashboard_status(runs_root=None):
    """Query the status of the persistent Web Observatory dashboard for a runs root or run."""
    from .web_autostart import dashboard_status as autostart_dstatus
    target = Path(runs_root) if runs_root is not None else None
    return autostart_dstatus(target)


def stop_dashboard(runs_root=None, timeout=8.0):
    """Stop the persistent Web Observatory dashboard for a runs root or run."""
    from .web_autostart import stop_dashboard as autostart_sstop
    target = Path(runs_root) if runs_root is not None else None
    return autostart_sstop(target, timeout=timeout)


def find_runs_root(path=None):
    """Locate the root directory containing training runs for a workspace or path."""
    from .web_autostart import find_runs_root as autostart_find
    return autostart_find(path)


def stop_projector(run_dir):
    """Stop the background projection worker for a run directory."""
    from .web_autostart import stop_projector as autostart_sproj
    return autostart_sproj(Path(run_dir))


# --------------------------------------------------------------------------- job queue

def queue_add(model_file, devices=None, run_dir=None, name=None, steps=None, sets=None, no_web=False, wait=False, queue_file=None, **kwargs):
    """Enqueue a training experiment or wait directive into the job queue."""
    from .queue import queue_add as _add
    return _add(model_file, devices=devices, run_dir=run_dir, name=name, steps=steps, sets=sets, no_web=no_web, wait=wait, queue_file=queue_file, **kwargs)


def queue_ls(status=None, queue_file=None):
    """List queued, running, and completed jobs from the job queue."""
    from .queue import queue_ls as _ls
    return _ls(status_filter=status, queue_file=queue_file)


def queue_status(job_id=None, queue_file=None, lines=20):
    """Check the execution status and recent logs of a queued job."""
    from .queue import queue_status as _status
    return _status(job_id=job_id, queue_file=queue_file, lines=lines)


def queue_stop(job_id=None, stop_all=False, queue_file=None):
    """Stop or cancel a queued job, or all jobs if stop_all=True."""
    from .queue import queue_stop as _stop
    return _stop(job_id=job_id, stop_all=stop_all, queue_file=queue_file)


def queue_run(devices=None, poll_interval=5.0, once=False, queue_file=None):
    """Process and run jobs sequentially from the queue until drained."""
    from .queue import queue_run as _run
    return _run(devices=devices, poll_interval=poll_interval, once=once, queue_file=queue_file)


def queue_install_service(user=True, devices=None, name=None, dry_run=False):
    """Generate or install a systemd service unit for the HyperGAN queue runner."""
    from .queue import install_service as _install
    return _install(user=user, devices=devices, name=name, dry_run=dry_run)


def queue_clear(queue_file=None):
    """Clear all queued and running jobs in the job queue."""
    from .queue import queue_clear as _clear
    return _clear(queue_file=queue_file)


# --------------------------------------------------------------------------- convenience exports

def compare(runs, metric="loss/g_total", at_steps=None):
    """Compare multiple runs for a given metric across step milestones."""
    return runs_compare(runs, metric=metric, at_steps=at_steps)


def _clean_diff_dict(cfg):
    if not isinstance(cfg, dict):
        return cfg
    cleaned = deepcopy(cfg)
    for k in ("warnings", "qualification", "_source_path", "name_map", "schema_version", "format"):
        cleaned.pop(k, None)
    if isinstance(cleaned.get("recipe"), dict):
        cleaned.pop("recipe", None)
    return cleaned


def _extract_diff_config(target):
    if isinstance(target, Model):
        return _clean_diff_dict(dict(resolve(target)))
    if isinstance(target, Run):
        if target.manifest.get("config") and isinstance(target.manifest["config"], dict):
            return _clean_diff_dict(target.manifest["config"])
        target = target.path
    if isinstance(target, dict):
        if "components" in target or "networks" in target:
            return _clean_diff_dict(dict(resolve(from_dict(target))))
        if "config" in target and isinstance(target["config"], dict):
            return _clean_diff_dict(target["config"])
        return _clean_diff_dict(target)
    target_str = str(target)
    target_p = Path(target_str)
    if target_str.endswith(".toml") or (target_p.is_file() and target_p.suffix == ".toml"):
        if not target_p.is_file():
            raise FileNotFoundError(f"Recipe target '{target_str}' not found")
        m = load(target_p)
        return _clean_diff_dict(dict(resolve(m)))
    if target_str.endswith(".pt") or (target_p.is_file() and target_p.suffix == ".pt"):
        if not target_p.is_file():
            raise FileNotFoundError(f"Checkpoint file '{target_str}' not found")
        try:
            import torch
            state = torch.load(target_p, map_location="cpu", weights_only=True)
        except ModuleNotFoundError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not load checkpoint '{target_str}': {exc}") from exc
        if isinstance(state, dict):
            if "config" in state and isinstance(state["config"], dict):
                return _clean_diff_dict(state["config"])
            if "model_state_dict" in state:
                return {k: list(v.shape) for k, v in state["model_state_dict"].items() if hasattr(v, "shape")}
        return {}
    try:
        resolved = resolve_run_target(target_str)
    except Exception as exc:
        raise FileNotFoundError(f"Run or recipe target '{target_str}' not found") from exc
    manifest_p = resolved / "manifest.json"
    if manifest_p.is_file():
        try:
            m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            if "config" in m_data and isinstance(m_data["config"], dict):
                return _clean_diff_dict(m_data["config"])
        except Exception:
            pass
    for fname in ("model.toml", "recipe.toml", "config.toml"):
        recipe_p = resolved / fname
        if recipe_p.is_file():
            m = load(recipe_p)
            return _clean_diff_dict(dict(resolve(m)))
    raise FileNotFoundError(f"Run or recipe target '{target_str}' not found or contains no configuration")


def _flatten_diff_dict(d, prefix=""):
    flat = {}
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                flat.update(_flatten_diff_dict(v, full))
            else:
                flat[full] = v
    else:
        flat[prefix] = d
    return flat


def _expand_override_shorthand(model, overrides):
    """Expand ``optimizer.x`` and ``training.x`` override keys for a hypergan-model/1 spec.

    ``optimizer.lr`` becomes ``optimizers.<each>.lr`` (or ``train.lr`` when the
    model has no ``[optimizers]``), and ``training.x`` becomes ``train.x``,
    unless the model itself uses those section names.
    """
    if not overrides or model is None:
        return dict(overrides or {})
    spec = getattr(model, "spec", model)
    spec = spec if isinstance(spec, dict) else {}
    optimizers = spec.get("optimizers")
    expanded = {}
    for k, v in overrides.items():
        if k.startswith("optimizer.") and "optimizer" not in spec:
            subkey = k[len("optimizer."):]
            if isinstance(optimizers, dict) and optimizers:
                for opt_name in optimizers:
                    expanded[f"optimizers.{opt_name}.{subkey}"] = v
            else:
                expanded[f"train.{subkey}"] = v
        elif k.startswith("training.") and "training" not in spec:
            expanded[f"train.{k[len('training.'):]}"] = v
        else:
            expanded[k] = v
    return expanded


def diff(target_a, target_b=None, *, sets=None, overrides=None):
    """Compare two configurations, models, runs, or checkpoints, returning differences.

    ``sets`` (or ``overrides``) are applied to ``target_b``, or to a copy of
    ``target_a`` when ``target_b`` is omitted, as ``hypergan diff -s`` does.
    """
    merged_sets = dict(sets or {})
    if overrides:
        merged_sets.update(overrides)
    if target_b is None:
        if not merged_sets:
            raise ValueError("diff requires a second target (target_b) or at least one -s/--set override")
        m_a = _model(target_a)
        m_b = override(m_a, _expand_override_shorthand(m_a, merged_sets))
        target_b_label = f"{getattr(target_a, 'path', target_a)} (with overrides)"
        cfg_b = _extract_diff_config(m_b)
    elif merged_sets:
        m_b = _model(target_b)
        m_b = override(m_b, _expand_override_shorthand(m_b, merged_sets))
        target_b_label = f"{getattr(target_b, 'path', target_b)} (with overrides)"
        cfg_b = _extract_diff_config(m_b)
    else:
        target_b_label = str(getattr(target_b, "path", target_b))
        cfg_b = _extract_diff_config(target_b)
    cfg_a = _extract_diff_config(target_a)
    flat_a = _flatten_diff_dict(cfg_a)
    flat_b = _flatten_diff_dict(cfg_b)
    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    differences = {}
    for k in all_keys:
        val_a = flat_a.get(k)
        val_b = flat_b.get(k)
        if val_a != val_b:
            differences[k] = {"a": val_a, "b": val_b}
    return {
        "target_a": str(getattr(target_a, "path", target_a)),
        "target_b": target_b_label,
        "identical": len(differences) == 0,
        "differences": differences,
    }


def list_templates():
    """List available starter templates (e.g. gaussian-grid, dcgan, resnet, tiny-transformer)."""
    from .recipe_generator import list_starter_templates
    return list_starter_templates()


def schema(section=None):
    """Return the hypergan-model/1 specification, optionally filtered to a section."""
    from .schema_info import get_schema
    return get_schema(section)


def operators(query=None):
    """Return catalog of all 85 HNDL operators, optionally filtered by search query."""
    from .ops_info import get_operators
    return get_operators(query)


ops = operators

from .run_info import RunInfo, read_run_info


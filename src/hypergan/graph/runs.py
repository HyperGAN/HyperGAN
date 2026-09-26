"""Train a model and read a run directory back: metrics, evaluations, samples."""
from dataclasses import dataclass
import json
from pathlib import Path

from .model import Model, load, model as build_model, save


@dataclass(frozen=True)
class Run:
    """A run directory. All state lives on disk; this is just its path."""
    path: Path
    config_path: Path = None

    @property
    def manifest(self):
        return json.loads((self.path / "manifest.json").read_text())

    @property
    def step(self):
        return self.manifest.get("steps")


def _as_model(target, losses, kwargs):
    if isinstance(target, Model):
        return target, target.path
    if isinstance(target, (str, Path)) and str(target).endswith(".toml"):
        loaded = load(target)
        return loaded, loaded.path
    return build_model(target, losses, **kwargs), None


def train(target, losses=None, *, run, config=None, checkpoint_every=None, preview_every=None, max_seconds=None,
          profile=None, on_event=None, overwrite_config=False, **model_kwargs):
    """Save the model's config file, then train it with the existing engine.

    ``target`` is a data source (with ``losses`` or ``generator=/discriminator=``),
    a ``Model``, or a path to a config file. A model built in Python is written
    to ``config`` (default ``<run>.toml`` beside the run directory) and training
    reads that file, exactly as ``hypergan train`` would. Running the same
    script again continues the run when the config is unchanged.
    ``profile`` selects replicated execution, e.g. ``"cpu-replicated-gloo"``.
    """
    from ..execution import train as execute
    run = Path(run)
    steps = model_kwargs.get("steps") if isinstance(target, (Model, str, Path)) else None
    if isinstance(target, (Model, str, Path)):
        extra = set(model_kwargs) - {"steps"}
        if extra:
            raise TypeError(f"A built model or config file takes no model arguments: {sorted(extra)}")
        model_kwargs = {}
    model, path = _as_model(target, losses, model_kwargs)
    if path is None or config is not None:
        path = Path(config) if config is not None else run.parent / (run.name + ".toml")
        save(model, path, overwrite=overwrite_config)
    execute(path, run, steps, profile=profile, checkpoint_every=checkpoint_every, preview_every=preview_every,
            max_seconds=max_seconds, on_event=on_event)
    return Run(run.resolve(), Path(path).resolve())


def resume(run, *, config=None, steps=None, **options):
    """Continue a run from its latest checkpoint (optionally with a longer ``steps``)."""
    from ..execution import resume as execute
    path = Path(getattr(run, "path", run))
    execute(path, None, config, steps=steps, **options)
    return Run(path.resolve(), Path(config).resolve() if config else None)


def _events(path):
    from ..run_events import read_event_page
    cursor = None
    while True:
        page = read_event_page(path, cursor, limit=1000)
        yield from page["events"]
        cursor = page["cursor"]
        if not page["has_more"]:
            return


def metrics(run, names=None):
    """Published scalar series: ``{metric_id: [(step, value), ...]}``.

    Includes the built-in update metrics (losses, penalty terms, objective
    contributions, throughput) and custom scalar metrics.
    """
    path = Path(getattr(run, "path", run))
    series = {}
    for event in _events(path):
        if event.get("event") not in ("train", "metric") or not isinstance(event.get("metrics"), dict):
            continue
        for key, value in event["metrics"].items():
            if names is None or key in names:
                series.setdefault(key, []).append((event["step"], value))
    return series


def last(run, names=None):
    """The last value of each metric: ``{metric_id: value}``."""
    return {key: values[-1][1] for key, values in metrics(run, names).items()}


def evaluations(run):
    """Snapshot evaluation results: ``{metric_id: [{"step", "value", "status", ...}]}``."""
    path = Path(getattr(run, "path", run))
    results, seen = {}, set()
    rows = []
    root = path / "metrics" / "evaluations"
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            stream = directory / "events.jsonl"
            if stream.is_file():
                rows += [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
    rows += [event for event in _events(path) if event.get("event") == "evaluation"]
    for event in rows:
        key = (event.get("evaluation_id"), event.get("step"))
        if event.get("event") != "evaluation" or key in seen:
            continue
        seen.add(key)
        for metric_id, value in {**event.get("metrics", {}), **event.get("distributions", {})}.items():
            results.setdefault(metric_id, []).append({"step": event["step"], "value": value, "status": event.get("status"),
                                                      "evaluation_id": event.get("evaluation_id")})
        for metric_id, status in event.get("measurement_status", {}).items():
            if metric_id not in event.get("metrics", {}):
                results.setdefault(metric_id, []).append({"step": event["step"], "value": None, "status": status,
                                                          "evaluation_id": event.get("evaluation_id")})
    return results


def evaluate(run, name):
    """Run a snapshot evaluation now (for ``every=None`` / manual evaluations)."""
    from ..metric_evaluation import evaluate as run_evaluation
    path = Path(getattr(run, "path", run))
    config = getattr(run, "config_path", None)
    return run_evaluation(path, name, config_path=config)


def previews(run):
    """Preview payloads published during training (``preview_every``), oldest first."""
    path = Path(getattr(run, "path", run)) / "previews" / "index.json"
    if not path.is_file():
        return []
    return [Path(item["path"]) for item in json.loads(path.read_text()).get("previews", [])]


def samples(run, name=None, *, count=None, seed=None):
    """Run the model's samplers on the run's EMA inference bundle and write views.

    Returns ``{sampler: {view: [paths]}}``. Files go to
    ``<run>/samplers/<sampler>/step-<N>/``. Samplers are read from the bundle's
    recorded config, i.e. the config the run trained with.
    """
    import torch
    from ..artifacts import load_inference
    from . import view
    from .adapters import resolve as import_object
    path = Path(getattr(run, "path", run))
    state, config, graph, prior, bundle_sha = load_inference(path)
    selected = config.get("samplers") or {}
    if name is not None:
        if name not in selected:
            raise KeyError(f"No sampler {name!r}; this run has {sorted(selected)}")
        selected = {name: selected[name]}
    written = {}
    for sampler_id, spec in selected.items():
        n = count or spec["count"]
        rng_seed = spec["seed"] if seed is None else seed
        batch = {key: value[torch.arange(n) % len(value)] for key, value in state["example_inputs"].items()}
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(rng_seed)
            z, _ = prior.sample(n, generator=torch.Generator().manual_seed(rng_seed))
            context = graph.generate(z, batch, prior=prior)
            inputs = {arg: graph.resolve(binding, context) for arg, binding in spec["inputs"].items()}
        function = import_object(spec["factory"])
        result = function(**spec["args"])(**inputs) if isinstance(function, type) else function(**inputs, **spec["args"])
        views = result if isinstance(result, dict) and "kind" not in result else {sampler_id: result}
        directory = path / "samplers" / sampler_id / f"step-{state.get('step', 0):08d}"
        metadata = {"sampler": sampler_id, "step": state.get("step"), "seed": rng_seed, "count": n,
                    "bundle_sha256": bundle_sha}
        written[sampler_id] = {key: view.write(directory, key, view.infer(value), metadata) for key, value in views.items()}
    return written

"""Graph nodes: references to values, networks, losses and observations.

Nothing here imports Torch or user code. A node records an import path and
JSON-like arguments; ``hypergan.graph.model`` lowers the nodes to the existing
configuration tables, which every process can rebuild from the file.
"""
import inspect
import json
import math
import re

_SNAKE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_2 = re.compile(r"([a-z0-9])([A-Z])")


def snake(name):
    return _SNAKE_2.sub(r"\1_\2", _SNAKE_1.sub(r"\1_\2", name)).lower()


def import_path(obj, what):
    """``module:qualname`` for a class or function that other processes can import."""
    if isinstance(obj, str):
        if obj.count(":") != 1 or not all(obj.split(":")):
            raise ValueError(f"{what} must be an importable object or a 'module:object' string, got {obj!r}")
        return obj
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not module or not qualname:
        raise TypeError(f"{what} must be a class or function, got {type(obj).__name__}")
    if module == "__main__":
        raise ValueError(
            f"{what} {qualname} is defined in the script being run (__main__). Every training process "
            f"rebuilds the model from the config file by import path, so move {qualname} into an "
            f"importable module (for example my_project/networks.py) and import it from there.")
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(f"{what} {qualname} is local to a function; define it at module level so it can be imported")
    return f"{module}:{qualname}"


def plain(value, location):
    """Check that an argument survives the config file unchanged (no None, NaN or objects)."""
    if value is None:
        raise ValueError(f"{location} is None; the config file has no null, so omit the argument instead")
    if isinstance(value, bool) or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v, f"{location}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError(f"{location} keys must be strings")
        return {k: plain(v, f"{location}.{k}") for k, v in value.items()}
    raise TypeError(f"{location} is a {type(value).__name__}; config arguments must be numbers, strings, "
                    "booleans, lists or dicts (pass objects by import path)")


class Ref:
    """A value in the model graph. Index it to select a nested output."""
    shape = None

    def __getitem__(self, key):
        if not isinstance(key, (str, int)) or isinstance(key, bool) or (isinstance(key, str) and ("." in key or not key)):
            raise KeyError("Select nested outputs with a string key (no dots) or an integer index")
        return Select(self, key)

    def __repr__(self):
        return f"<{type(self).__name__} {self.describe()}>"

    def describe(self):
        return ""


class Field(Ref):
    """A field of the training batch, such as ``hg.data("condition")``."""
    def __init__(self, field="real", shape=None):
        if not isinstance(field, str) or not field.isidentifier():
            raise ValueError("Data field names must be Python identifiers")
        self.field, self.shape = field, None if shape is None else plain(shape, f"data({field}).shape")

    def describe(self):
        return f"data.{self.field}"


class Latent(Ref):
    """The prior sample. The current engine has exactly one prior per model."""
    def __init__(self, dim, kind="particles", particles=None, **args):
        if type(dim) is not int or dim < 1:
            raise ValueError("latent dim must be a positive integer")
        if kind not in ("particles", "mog", "gaussian"):
            raise ValueError("latent kind must be particles, mog or gaussian")
        self.dim, self.kind = dim, kind
        self.args = {"z_dim": dim, **plain(args, "latent args")}
        if particles is not None:
            self.args["num_particles"] = plain(particles, "latent particles")
        self.shape = ["B", dim]

    def describe(self):
        return f"latent[{self.dim}, {self.kind}]"


class _Candidate(Ref):
    """The sample a critic scores: real on one side of the loss, fake on the other."""
    def describe(self):
        return "candidate"

    def __getitem__(self, key):
        raise TypeError("The critic candidate is a whole sample; select fields inside the critic")


candidate = _Candidate()


class Select(Ref):
    def __init__(self, parent, key):
        self.parent, self.key = parent, key
        shape = getattr(parent, "shape", None)
        self.shape = shape.get(key) if isinstance(shape, dict) and isinstance(key, str) else None

    def describe(self):
        return f"{self.parent.describe()}[{self.key!r}]"


class Net(Ref):
    """A network component. ``hg.net`` and ``hg.hndl`` construct these."""
    def __init__(self, factory, args, inputs, *, name=None, trainable=None, default_name=None,
                 shape=None, hndl=False, reuse=None, freeze=False):
        if name is not None and (not isinstance(name, str) or not name.isidentifier()):
            raise ValueError("Network names must be Python identifiers; they become checkpoint keys")
        if trainable is not None and type(trainable) is not bool:
            raise ValueError("trainable must be True, False or omitted")
        if not inputs:
            raise ValueError("A network needs at least one input reference, e.g. x=hg.latent(64)")
        self.factory, self.args, self.inputs = factory, args, inputs
        self.name, self.trainable, self.default_name = name, trainable, default_name
        self.shape, self.hndl, self.reuse_of, self.freeze = shape, hndl, reuse, freeze

    def reuse(self, *, name, freeze=False, **inputs):
        """Call the same weights on other inputs (``reuse`` in the config)."""
        if self.reuse_of is not None:
            return self.reuse_of.reuse(name=name, freeze=freeze, **inputs)
        bad = [k for k, v in inputs.items() if not isinstance(v, Ref)]
        if bad:
            raise TypeError(f"reuse() takes only input references; got non-references for {bad}")
        return Net("reuse", {}, dict(inputs), name=name, shape=self.shape, reuse=self, freeze=freeze)

    def describe(self):
        return self.name or self.default_name or self.factory


def split_kwargs(kwargs, location):
    """References are forward inputs; everything else is a constructor argument."""
    inputs = {k: v for k, v in kwargs.items() if isinstance(v, Ref)}
    args = {k: plain(v, f"{location}.{k}") for k, v in kwargs.items() if not isinstance(v, Ref)}
    return inputs, args


def net(module, *, name=None, trainable=None, **kwargs):
    """A PyTorch network: ``hg.net(Encoder, x=x, hidden=64)``.

    Keyword arguments that are graph references (``hg.data``, ``hg.latent``,
    ``hg.candidate`` or another network) become forward inputs; the rest are
    constructor arguments recorded in the config. ``module`` is a class or
    factory function in an importable module, or a ``"module:object"`` string.
    """
    factory = import_path(module, "hg.net module")
    inputs, args = split_kwargs(kwargs, f"net({factory})")
    default = snake(factory.split(":")[1].split(".")[-1])
    return Net(factory, args, inputs, name=name, trainable=trainable, default_name=default)


def hndl(source=None, *, file=None, shape, name=None, trainable=None, input_shape=None, **kwargs):
    """An HNDL network: ``hg.hndl("linear(64)\\nrelu()\\nlinear()", shape=["B", 2], x=z)``.

    ``shape`` is the output shape. Input shapes are inferred from the input
    references when they are known (latent, builtin data, other HNDL outputs)
    or given with ``input_shape``. ``file`` references a ``.hndl`` file, which
    the saved config refers to by relative path rather than copying it.
    """
    if (source is None) == (file is None):
        raise ValueError("hg.hndl takes exactly one of source or file")
    inputs, args = split_kwargs(kwargs, "hndl")
    if source is not None:
        args["source"] = source
    else:
        from pathlib import Path
        path = Path(file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"HNDL file not found: {file}")
        args["file"] = str(path)
    args["output_shape"] = plain(shape, "hndl shape")
    if input_shape is not None:
        args["input_shape"] = plain(input_shape, "hndl input_shape")
    return Net("hndl", args, inputs, name=name, trainable=trainable, shape=args["output_shape"], hndl=True)


def first_parameter(obj):
    """The first forward argument name of a module class (or of a callable)."""
    target = getattr(obj, "forward", obj)
    parameters = [p for p in inspect.signature(target).parameters.values()
                  if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if not parameters:
        raise ValueError(f"Cannot infer the input name of {obj!r}; build it with hg.net(..., name=ref)")
    return parameters[0].name


# Losses ---------------------------------------------------------------------

class Loss:
    pass


class Adversarial(Loss):
    def __init__(self, critic, real, fake, weight=1.0, penalty=None, name=None):
        if not isinstance(critic, Net):
            raise TypeError("hg.adversarial needs a critic network built with hg.net/hg.hndl on hg.candidate")
        if not isinstance(real, Ref) or not isinstance(fake, Ref):
            raise TypeError("real and fake must be graph references")
        self.critic, self.real, self.fake = critic, real, fake
        self.weight, self.penalty, self.name = plain(weight, "adversarial weight"), penalty, name


class Objective(Loss):
    def __init__(self, factory, args, inputs, weight, detach, name, default_name):
        self.factory, self.args, self.inputs = factory, args, inputs
        self.weight, self.detach, self.name, self.default_name = plain(weight, "loss weight"), list(detach), name, default_name


def adversarial(critic, real, fake, *, weight=1.0, penalty=None, name=None):
    """An RpGAN term: ``critic`` scores ``real`` against ``fake``.

    A network scored here is a critic (discriminator side); everything upstream
    of ``fake`` is generator side. The first adversarial loss is the model's
    main pair. ``penalty``: a coefficient for the K3P critic penalty (the main
    term always has one; further terms have none unless given).
    """
    return Adversarial(critic, real, fake, weight, penalty, name)


def _pair(kind, input, target, weight, name, detach):
    if not isinstance(input, Ref) or not isinstance(target, Ref):
        raise TypeError(f"hg.{kind} compares two graph references")
    return Objective(kind, {}, {"input": input, "target": target}, weight, detach, name, kind)


def l1(input, target, *, weight=1.0, name=None, detach=("target",)):
    """Mean absolute error between two outputs; the target is detached by default."""
    return _pair("l1", input, target, weight, name, detach)


def mse(input, target, *, weight=1.0, name=None, detach=("target",)):
    """Mean squared error between two outputs; the target is detached by default."""
    return _pair("mse", input, target, weight, name, detach)


def loss(function, *, weight=1.0, name=None, detach=(), **kwargs):
    """A custom generator-side loss.

    ``function`` is a plain function ``f(**inputs) -> scalar tensor`` or a class
    whose instances are called that way. References become its inputs; other
    keyword arguments are constructor (or extra function) arguments.
    """
    ref = import_path(function, "hg.loss function")
    inputs, args = split_kwargs(kwargs, f"loss({ref})")
    if not inputs:
        raise ValueError("hg.loss needs at least one input reference")
    default = snake(ref.split(":")[1].split(".")[-1])
    if inspect.isclass(function):
        return Objective(ref, args, inputs, weight, detach, name, default)
    return Objective("hypergan.graph.adapters:FunctionLoss", {"function": ref, "args": args},
                     inputs, weight, detach, name, default)


# Observations ---------------------------------------------------------------

SCALARS = ("d_loss", "g_loss", "d_adversarial", "g_adversarial", "d_adversarial_weighted",
           "g_adversarial_weighted", "gradient_penalty", "prior_loss", "lr_scale", "step", "step_seconds")


class Metric:
    def __init__(self, factory, args, inputs, every, name):
        self.factory, self.args, self.inputs, self.every, self.name = factory, args, inputs, every, name


def metric(function, *, name=None, every=1, label=None, unit=None, direction="none", **inputs):
    """A cheap scalar computed from values the update already produced.

    ``function``'s parameter names select update scalars (``g_loss``,
    ``d_loss``, ``gradient_penalty``, ...), e.g. ``def ratio(g_loss, d_loss)``.
    Pass ``param="d_loss"`` to bind a differently named parameter. A class
    with ``describe()``/``evaluate(*, context, **inputs)`` is used as is.
    """
    ref = import_path(function, "hg.metric function")
    if inspect.isclass(function):
        names = [n for n in inspect.signature(function.evaluate).parameters if n not in ("self", "context")]
        factory, args = ref, {}
    else:
        names = list(inspect.signature(function).parameters)
        factory = "hypergan.graph.adapters:FunctionMetric"
        args = {"function": ref, "direction": direction}
        args.update({k: v for k, v in (("label", label), ("unit", unit)) if v is not None})
    bound = {}
    for parameter in names:
        source = inputs.pop(parameter, parameter)
        source = source.removeprefix("update.")
        if source not in SCALARS:
            raise ValueError(f"metric parameter {parameter!r} does not name an update scalar; choose one of "
                             f"{', '.join(SCALARS)} or bind it, e.g. {parameter}='g_loss'")
        bound[parameter] = "update." + source
    if inputs:
        raise ValueError(f"metric bindings {sorted(inputs)} are not parameters of {ref}")
    return Metric(factory, args, bound, every, name or snake(ref.split(":")[1].split(".")[-1]))


class Evaluation:
    def __init__(self, factory, args, inputs, data, every, samples, batch_size, seed, device, output, name, on_error):
        self.factory, self.args, self.inputs, self.data = factory, args, inputs, data
        self.every, self.samples, self.batch_size, self.seed = every, samples, batch_size, seed
        self.device, self.output, self.name, self.on_error = device, output, name, on_error


def evaluation(function, data, *, name=None, every=None, samples=256, batch_size=16, seed=0,
               device="cpu", output=None, on_error="fail", **args):
    """An expensive measurement of an EMA snapshot, usually on holdout ``data``.

    ``function(generated, reference) -> float`` receives the concatenated
    samples (declare only the parameters you need); a class following the
    snapshot protocol (``evaluate(*, batches, context)``) is used as is.
    ``every=None`` means on request only (``hg.evaluate(run, name)``).
    ``output`` selects a model output other than the generator's.
    """
    ref = import_path(function, "hg.evaluation function")
    if inspect.isclass(function):
        factory, fargs, names = ref, plain(args, "evaluation args"), ["generated", "reference"]
    else:
        names = [n for n in inspect.signature(function).parameters if n in ("generated", "reference")]
        if not names:
            raise ValueError("An evaluation function takes generated and/or reference")
        factory, fargs = "hypergan.graph.adapters:FunctionEvaluation", {"function": ref, "args": plain(args, "evaluation args")}
    inputs = {n: "evaluation." + n for n in names}
    return Evaluation(factory, fargs, inputs, data, every, samples, batch_size, seed, device, output,
                      name or snake(ref.split(":")[1].split(".")[-1]), on_error)


class Sampler:
    def __init__(self, factory, args, inputs, count, seed, name):
        self.factory, self.args, self.inputs, self.count, self.seed, self.name = factory, args, inputs, count, seed, name


def sampler(function, *, name=None, count=16, seed=None, **kwargs):
    """What to generate for people to look at.

    ``function(**outputs)`` receives the referenced outputs (any tensors,
    nested dicts or lists) and returns something viewable: ``hg.view.image``,
    ``hg.view.points``, ``hg.view.audio``, ``hg.view.text``, a dict of those,
    or a tensor. References are its inputs; other keywords are arguments.
    """
    ref = import_path(function, "hg.sampler function")
    inputs, args = split_kwargs(kwargs, f"sampler({ref})")
    if not inputs:
        raise ValueError("hg.sampler needs at least one output reference, e.g. points=fake")
    return Sampler(ref, args, inputs, count, seed, name or snake(ref.split(":")[1].split(".")[-1]))


# Data sources ---------------------------------------------------------------

class DataSource:
    """Where training batches come from. HyperGAN owns batching, order and sharding."""
    def __init__(self, factory, args, kind, shapes=None):
        self.factory, self.args, self.kind, self.shapes = factory, args, kind, shapes or {}

    def __repr__(self):
        return f"<DataSource {self.kind} {self.factory}>"


def dataset(cls, *, real=None, shuffle=True, workers=0, **args):
    """An item-level dataset: the class implements ``__len__`` and ``__getitem__(i)``.

    Items are tensors (the real sample) or dicts of tensors/numbers. HyperGAN
    shuffles per epoch from the run seed, batches, records the position in
    checkpoints and will shard indices across processes. ``real`` names the
    item field the main adversarial term treats as real (inferred from it).
    """
    ref = import_path(cls, "hg.dataset class")
    config = {"dataset": ref, "args": plain(args, f"dataset({ref})"), "shuffle": shuffle, "workers": workers}
    if real is not None:
        config["real"] = real
    return DataSource("hypergan.graph.adapters:ItemDataset", config, "items")


def batches(factory, **args):
    """A batch-level source with the existing contract ``f(batch_size, *, generator) -> dict``."""
    return DataSource(import_path(factory, "hg.batches factory"), plain(args, "batches args"), "batches")


def gaussian_grid(side=10, noise=0.015):
    """The builtin two-dimensional Gaussian grid (field ``real``, shape [B, 2])."""
    return DataSource("gaussian_grid", {"side": side, "noise": noise}, "builtin", {"real": ["B", 2]})


def paired_linear(dimensions=2, scale=2.0, offset=0.0):
    """The builtin synthetic paired data: ``condition`` and ``real = condition*scale+offset``."""
    shape = ["B", dimensions]
    return DataSource("paired_linear", {"dimensions": dimensions, "scale": scale, "offset": offset}, "builtin",
                      {"real": shape, "condition": shape})


def as_source(data):
    if isinstance(data, DataSource):
        return data
    if inspect.isclass(data) and hasattr(data, "__getitem__") and hasattr(data, "__len__"):
        return dataset(data)
    if isinstance(data, str) or callable(data):
        return batches(data)
    raise TypeError("data must be hg.dataset(...), hg.batches(...), a builtin source, or a Dataset-like class")


def dumps_args(value):
    return json.dumps(value, sort_keys=True)

"""Declarations: plain frozen data objects built by plain functions.

Each network is declared with an explicit role, and every loss that trains it
is attached to that declaration. Nothing here imports Torch or runs user code;
user Python is only ever referenced by import path ("module:object").
"""
from dataclasses import dataclass, field
import importlib
import inspect

__all__ = [
    'Network', 'Optimizer', 'Penalty', 'Loss', 'Judge', 'Declaration', 'Prior', 'Data',
    'Metric', 'Evaluation', 'Sampler', 'Recipe', 'INHERIT', 'LATENT', 'CANDIDATE', 'GENERATED',
    'net', 'hndl', 'adam', 'k3p', 'mse', 'l1', 'loss', 'adversarial', 'spread',
    'generator', 'critic', 'encoder', 'frozen', 'shared', 'judge',
    'particles', 'mog', 'gaussian', 'data', 'items', 'training',
    'metric', 'evaluation', 'sampler', 'recipe', 'reference',
]

LATENT, CANDIDATE, GENERATED = 'latent', 'candidate', 'generated'
ENGINE_BUILTIN_NETWORKS = {'hndl', 'mlp', 'linear', 'identity'}
ENGINE_BUILTIN_DATA = {'gaussian_grid', 'paired_linear', 'image_folder'}


class _Inherit:
    def __repr__(self):
        return 'INHERIT'


INHERIT = _Inherit()  # a judge uses its critic's penalty


def reference(obj):
    """Import path of a user object. Code is referenced, never copied.

    Every process (and later every host) rebuilds the model by importing this
    path, so the object must live in an importable module, not in ``__main__``.
    """
    if isinstance(obj, str):
        if obj.count(':') != 1 or not all(obj.split(':')):
            raise ValueError(f'{obj!r}: use a module:object import path')
        return obj
    module = getattr(obj, '__module__', None)
    qualname = getattr(obj, '__qualname__', None)
    if not module or not qualname or '<locals>' in qualname:
        raise ValueError(f'{obj!r} has no importable module:object path; define it at module level')
    if module == '__main__':
        raise ValueError(
            f'{qualname} is defined in the script being run (__main__). Move it into an importable '
            f'module (for example my_project/nets.py): every worker process rebuilds the model from '
            f'the config by import path, and HyperGAN never copies your code into the run.')
    path = f'{module}:{qualname}'
    target = importlib.import_module(module)
    for part in qualname.split('.'):
        target = getattr(target, part, None)
    if target is not obj:
        raise ValueError(f'{path} does not import back to the same object')
    return path


# ---------------------------------------------------------------- what to build

@dataclass(frozen=True)
class Network:
    """A constructor: 'hndl' (HNDL source), an engine builtin, or 'module:object'."""
    factory: str
    args: dict = field(default_factory=dict)
    file: str | None = None  # HNDL file kept as a reference in saved configs


def net(constructor, **args):
    """Your own ``torch.nn.Module`` class (or 'module:Class'), with constructor args."""
    if isinstance(constructor, Network):
        if args:
            raise ValueError('net(Network, ...) takes no extra args')
        return constructor
    if isinstance(constructor, str) and constructor in ENGINE_BUILTIN_NETWORKS:
        return Network(constructor, dict(args))
    return Network(reference(constructor), dict(args))


def hndl(source=None, *, file=None, input_shape, output_shape, **options):
    """An HNDL network. ``file`` stays a path reference in the saved config."""
    if (source is None) == (file is None):
        raise ValueError('hndl() needs exactly one of source or file')
    args = {'input_shape': input_shape, 'output_shape': output_shape, **options}
    if source is not None:
        args['source'] = source
    return Network('hndl', args, None if file is None else str(file))


def _network(value):
    return value if isinstance(value, Network) else net(value)


# ---------------------------------------------------------------- how it trains

@dataclass(frozen=True)
class Optimizer:
    """Per-network optimizer settings. None means the engine default.

    Today's engine has one generator-side Adam and one critic Adam; the critic
    and prior learning rates are multiples of the generator's.
    """
    lr: float | None = None
    lr_mult: float | None = None
    betas: tuple | None = None
    options: dict = field(default_factory=dict)


def adam(lr=None, *, lr_mult=None, betas=None, **options):
    """Options: implementation, guard_ratio, guard_min_steps (critic), latent_damping (prior)."""
    return Optimizer(lr, lr_mult, None if betas is None else tuple(betas), dict(options))


@dataclass(frozen=True)
class Penalty:
    """ParticleGAN 0.8's K3P critic penalty. None fields use the engine default."""
    coeff: float | None = None
    kappa: float | None = None
    lazy_k: int | None = None
    anchor_weight: float | None = None
    anchor_decay: float | None = None


def k3p(coeff=None, *, kappa=None, lazy_k=None, anchor_weight=None, anchor_decay=None):
    return Penalty(coeff, kappa, lazy_k, anchor_weight, anchor_decay)


@dataclass(frozen=True)
class Loss:
    """A loss attached to the network it trains.

    kind: 'objective' (a weighted term), 'adversarial' (this network is judged
    by ``critic``), or 'spread' (the prior regularizer).
    """
    kind: str
    factory: str | None = None
    inputs: dict = field(default_factory=dict)
    weight: float | None = None
    detach: tuple | None = None
    args: dict = field(default_factory=dict)
    id: str | None = None
    critic: str | None = None
    verbatim_id: bool = False  # id is used as is (not prefixed by its network)


def _objective(factory, inputs, weight, id, detach, args):
    return Loss('objective', factory, dict(inputs), weight,
                None if detach is None else tuple(detach), dict(args), id)


def mse(input, target, *, weight=1.0, id=None, detach=None):
    """Mean squared error; ``target`` is detached unless ``detach`` says otherwise."""
    return _objective('mse', {'input': input, 'target': target}, weight, id, detach, {})


def l1(input, target, *, weight=1.0, id=None, detach=None):
    return _objective('l1', {'input': input, 'target': target}, weight, id, detach, {})


def loss(function, *, inputs, weight=1.0, id=None, detach=None, **args):
    """Your own loss: a plain function ``f(**inputs) -> scalar tensor``, or a
    'module:Class' whose instances are called with the bound inputs."""
    if isinstance(function, str) or inspect.isclass(function):
        return _objective(reference(function), inputs, weight, id, detach, args)
    return _objective('hypergan.api_per_network.adapters:function_loss', inputs, weight, id, detach,
                      {'function': reference(function), **args})


def adversarial(critic_name):
    """This network is trained to fool ``critic_name`` (the critic's judges hold weight/penalty)."""
    return Loss('adversarial', critic=critic_name)


def spread(*, weight=None, target_std=None, eps=None, rows=None):
    """The particle-prior spread regularizer; attach it to the prior."""
    return Loss('spread', args={k: v for k, v in dict(target_std=target_std, eps=eps, rows=rows).items()
                                if v is not None}, weight=weight)


@dataclass(frozen=True)
class Judge:
    """One real/fake comparison a critic makes. ``inputs`` overrides the critic's inputs."""
    real: str = 'batch.real'
    fake: str = GENERATED
    weight: float | None = None
    inputs: dict | None = None
    penalty: object = INHERIT  # INHERIT, None/False (no penalty) or Penalty (coeff only)
    id: str | None = None
    verbatim_id: bool = False


def judge(real='batch.real', fake=GENERATED, *, weight=None, inputs=None, penalty=INHERIT, id=None):
    return Judge(real, fake, weight, None if inputs is None else dict(inputs), penalty, id)


# ---------------------------------------------------------------- roles

@dataclass(frozen=True)
class Declaration:
    role: str  # generator | critic | encoder | frozen | shared
    network: Network | None
    inputs: dict
    losses: tuple = ()
    judges: tuple = ()
    penalty: Penalty | None = None
    optimizer: Optimizer | None = None
    reuse: str | None = None
    freeze_parameters: bool | None = None


def generator(network, *, inputs, losses=(), optimizer=None):
    """A network whose output is judged. Name the adversarial one 'generator'."""
    return Declaration('generator', _network(network), dict(inputs), tuple(losses), optimizer=optimizer)


def encoder(network, *, inputs, losses=(), optimizer=None):
    """A trainable auxiliary on the generator side (an encoder or any other module)."""
    return Declaration('encoder', _network(network), dict(inputs), tuple(losses), optimizer=optimizer)


def critic(network, *, inputs, judges=(judge(),), penalty=Penalty(), optimizer=None):
    """A critic trains only on its judges' RpGAN losses plus its penalty."""
    if isinstance(judges, Judge):
        judges = (judges,)
    if penalty is not None and not isinstance(penalty, Penalty):
        raise ValueError('critic penalty must be k3p(...) or None')
    return Declaration('critic', _network(network), dict(inputs), (), tuple(judges), penalty, optimizer)


def frozen(network, *, inputs):
    """A network with no trainable parameters (it still passes gradients to its inputs)."""
    return Declaration('frozen', _network(network), dict(inputs))


def shared(name, *, inputs, losses=(), freeze_parameters=False):
    """Call network ``name`` again with other inputs, sharing its parameters.

    Losses on this call train the shared parameters through this call's output.
    """
    return Declaration('shared', None, dict(inputs), tuple(losses), reuse=name,
                       freeze_parameters=freeze_parameters if freeze_parameters else None)


# ---------------------------------------------------------------- prior and data

@dataclass(frozen=True)
class Prior:
    kind: str
    args: dict
    losses: tuple = ()
    optimizer: Optimizer | None = None
    options: dict = field(default_factory=dict)  # initialization_device/seed, fixed_sigma


def particles(z_dim, count, *, losses=(spread(),), optimizer=None, **options):
    return Prior('particles', {'num_particles': count, 'z_dim': z_dim}, tuple(losses), optimizer, options)


def mog(z_dim, count, *, losses=(spread(),), optimizer=None, **options):
    return Prior('mog', {'num_particles': count, 'z_dim': z_dim}, tuple(losses), optimizer, options)


def gaussian(z_dim, **options):
    return Prior('gaussian', {'z_dim': z_dim}, (), None, options)


@dataclass(frozen=True)
class Data:
    factory: str
    args: dict = field(default_factory=dict)


def data(factory, **args):
    """A batch-level source: an engine builtin or ``f(batch_size, *, generator) -> dict``."""
    if isinstance(factory, str) and factory in ENGINE_BUILTIN_DATA:
        return Data(factory, args)
    return Data(reference(factory), args)


def items(dataset, *, shuffle=True, field='real', **args):
    """An item-level dataset (``__len__`` + ``__getitem__``). HyperGAN owns the rest:
    order, shuffling, seeding, resume position and sharding."""
    return Data('hypergan.api_per_network.adapters:ItemData',
                {'dataset': reference(dataset), 'args': args, 'shuffle': shuffle, 'field': field})


def training(**settings):
    """Engine [training] fields (steps, batch_size, seed, device, ema, lr_floor, ...)."""
    return dict(settings)


# ---------------------------------------------------------------- observation

@dataclass(frozen=True)
class Metric:
    """Cheap: a function of values the update already computed."""
    id: str
    factory: str
    inputs: dict
    args: dict = field(default_factory=dict)
    every: int | None = None
    timeout: float | None = None
    on_error: str | None = None


def metric(id, function, *, inputs, every=None, label=None, unit=None, direction=None, timeout=None,
           on_error=None, **args):
    """A plain function ``f(**inputs) -> float`` of update scalars (``update.g_loss`` ...),
    or a 'module:Class' with describe()/evaluate()."""
    if isinstance(function, str) or inspect.isclass(function):
        return Metric(id, reference(function), dict(inputs), args, every, timeout, on_error)
    present = {k: v for k, v in dict(label=label, unit=unit, direction=direction).items() if v is not None}
    return Metric(id, 'hypergan.api_per_network.adapters:FunctionMetric', dict(inputs),
                  {'function': reference(function), **present, **args}, every, timeout, on_error)


@dataclass(frozen=True)
class Evaluation:
    """Expensive: runs on a snapshot of the EMA model, usually on holdout data."""
    id: str
    factory: str
    inputs: dict
    data: Data
    samples: int
    batch_size: int
    seed: int
    args: dict = field(default_factory=dict)
    every: int | None = None  # None = manual only
    device: str | None = None
    generated: str | None = None
    timeout: float | None = None
    on_error: str | None = None


def evaluation(id, function, *, data, samples, batch_size, seed=0, every=None, device=None,
               inputs=None, generated=None, label=None, unit=None, direction=None, kind=None,
               timeout=None, on_error=None, **args):
    """A plain function ``f(batches) -> float`` over batches of ``{'generated', 'reference'}``,
    or a 'module:Class' with describe()/evaluate(batches=, context=)."""
    inputs = dict(inputs or {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'})
    if isinstance(function, str) or inspect.isclass(function):
        factory, args = reference(function), args
    else:
        present = {k: v for k, v in dict(label=label, unit=unit, direction=direction, kind=kind).items()
                   if v is not None}
        factory = 'hypergan.api_per_network.adapters:FunctionEvaluation'
        args = {'function': reference(function), **present, **args}
    return Evaluation(id, factory, inputs, data, samples, batch_size, seed, args, every, device,
                      generated, timeout, on_error)


@dataclass(frozen=True)
class Sampler:
    """What people look at: maps any generator output to something viewable."""
    name: str
    function: str
    output: str = GENERATED
    args: dict = field(default_factory=dict)


def sampler(name, function, *, output=GENERATED, **args):
    return Sampler(name, reference(function), output, args)


# ---------------------------------------------------------------- the whole model

@dataclass(frozen=True)
class Recipe:
    networks: dict
    data: Data
    prior: Prior
    training: dict = field(default_factory=dict)
    observe: tuple = ()
    sampling: dict = field(default_factory=dict)
    metric_settings: dict = field(default_factory=dict)
    name: str = 'custom/api'
    defaults: str | None = None


def recipe(networks, *, data, prior, training=None, observe=(), sampling=None, metric_settings=None,
           name='custom/api', defaults=None):
    """Everything needed to rebuild the model in any process. Lowers to one config file."""
    if not isinstance(networks, dict) or not networks:
        raise ValueError('networks must be a non-empty {name: declaration} dict')
    for name_, declaration in networks.items():
        if not isinstance(declaration, Declaration):
            raise ValueError(f'networks[{name_!r}] must be a role declaration (generator(), critic(), ...)')
    return Recipe(dict(networks), data, prior, dict(training or {}), tuple(observe), dict(sampling or {}),
                  dict(metric_settings or {}), name, defaults)

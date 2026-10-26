"""Torch-free execution profiles, separate from recipe and runtime checks."""
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


MAX_PROFILE_BYTES = 65536
_NAMES = ('cpu-single', 'cpu-replicated-gloo', 'cuda-replicated-nccl')
_KINDS = {'cpu-single': 'hypergan-training-checkpoint',
          'cpu-replicated-gloo': 'hypergan-distributed-training-checkpoint',
          'cuda-replicated-nccl': 'hypergan-distributed-training-checkpoint'}


_BUILTIN_STRATEGIES = {'gradient_allreduce', 'parameter_average', 'pseudo_gradient'}
_REGISTERED_STRATEGIES = set()
EXECUTION_FIELDS = ('name', 'world_size', 'accumulation_steps', 'nodes', 'node_rank',
                    'ranks_per_node', 'rendezvous', 'aggregation')


def profile_from_execution(execution):
    """Recover profile inputs without dropping saved aggregation or topology."""
    return {'schema_version': 1, 'execution': {
        key: execution[key] for key in EXECUTION_FIELDS if key in execution}}


def register_profile_strategy(name):
    """Register a custom aggregation strategy name for profile validation."""
    if not isinstance(name, str) or not name:
        raise ValueError('Strategy name must be a non-empty string')
    _REGISTERED_STRATEGIES.add(name)


def validate_replicated_recipe(config, *, accumulation_steps=1):
    """Reject native-only numerical policies before allocating ranks or a viewer."""
    training = config['training']
    if (training.get('phase_draws', 'shared') not in ('shared', 'independent')
            or training.get('data_rng_device', 'cpu') not in ('cpu', 'execution')
            or config.get('optimizer', {}).get('implementation', 'device_adam') not in ('device_adam', 'torch_fused_adam', 'recipe')):
        raise ValueError('Unsupported numerical policies currently require native execution; replicated/accumulated image execution is not qualified')
    if accumulation_steps > 1:
        unsupported = [name for name in ('input_noise_std', 'output_noise_std') if training.get(name, 0)]
        if training.get('phase_draws') == 'independent':
            unsupported.append('phase_draws=independent')
        if unsupported:
            raise ValueError('Replicated accumulation does not implement ' + ', '.join(unsupported)
                             + '; use accumulation_steps=1 to preserve these numerical policies')


def _table(value, name, allowed, required=()):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f'{name} must be a table with string keys')
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown:
        raise ValueError(f'Unknown {name} option(s): {", ".join(sorted(unknown))}')
    if missing:
        raise ValueError(f'Missing {name} option(s): {", ".join(sorted(missing))}')
    return value


def _positive_integer(value, name):
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _non_negative_integer(value, name):
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def _validate_rendezvous(uri):
    if not isinstance(uri, str) or not uri:
        raise ValueError('execution.rendezvous must be a non-empty string')
    if uri.startswith('file://'):
        return uri
    if uri.startswith('tcp://'):
        remainder = uri[len('tcp://'):]
        if ':' not in remainder:
            raise ValueError('execution.rendezvous tcp:// URI must be formatted as tcp://<host>:<port>')
        host, port_str = remainder.rsplit(':', 1)
        if not host:
            raise ValueError('execution.rendezvous tcp:// URI must have a non-empty host')
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f'execution.rendezvous tcp:// URI port must be an integer: {port_str}')
        if port < 1 or port > 65535:
            raise ValueError(f'execution.rendezvous tcp:// URI port must be between 1 and 65535: {port}')
        return uri
    raise ValueError('execution.rendezvous must be a file:// or tcp:// URI')


def _seconds(value, name):
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f'{name} must be finite positive seconds')
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} must be finite positive seconds') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'{name} must be finite positive seconds')
    return seconds


def _resolve_aggregation(raw_aggregation, name):
    if raw_aggregation is None:
        raw_aggregation = {}
    raw_aggregation = _table(raw_aggregation, 'aggregation',
                             ('strategy', 'sync_every_steps', 'outer_lr', 'outer_momentum'))
    strategy = raw_aggregation.get('strategy', 'gradient_allreduce')
    if not isinstance(strategy, str) or not strategy:
        raise ValueError('aggregation.strategy must be a non-empty string')
    if strategy not in _BUILTIN_STRATEGIES and strategy not in _REGISTERED_STRATEGIES:
        raise ValueError(f'Unknown aggregation.strategy: {strategy}')
    sync_every_steps = _positive_integer(raw_aggregation.get('sync_every_steps', 1), 'aggregation.sync_every_steps')
    if strategy == 'gradient_allreduce' and sync_every_steps != 1:
        raise ValueError('gradient_allreduce requires sync_every_steps = 1')

    outer_lr_val = raw_aggregation.get('outer_lr', 1.0)
    if type(outer_lr_val) not in (int, float) or isinstance(outer_lr_val, bool):
        raise ValueError('aggregation.outer_lr must be a finite positive float')
    try:
        outer_lr = float(outer_lr_val)
    except (OverflowError, ValueError) as exc:
        raise ValueError('aggregation.outer_lr must be a finite positive float') from exc
    if not math.isfinite(outer_lr) or outer_lr <= 0.0:
        raise ValueError('aggregation.outer_lr must be a finite positive float')

    outer_momentum_val = raw_aggregation.get('outer_momentum', 0.0)
    if type(outer_momentum_val) not in (int, float) or isinstance(outer_momentum_val, bool):
        raise ValueError('aggregation.outer_momentum must be a finite float in [0.0, 1.0)')
    try:
        outer_momentum = float(outer_momentum_val)
    except (OverflowError, ValueError) as exc:
        raise ValueError('aggregation.outer_momentum must be a finite float in [0.0, 1.0)') from exc
    if not math.isfinite(outer_momentum) or not (0.0 <= outer_momentum < 1.0):
        raise ValueError('aggregation.outer_momentum must be a finite float in [0.0, 1.0)')

    if name == 'cpu-single' and (strategy != 'gradient_allreduce' or sync_every_steps != 1):
        raise ValueError('cpu-single requires aggregation.strategy="gradient_allreduce" and sync_every_steps=1')

    return {'strategy': strategy, 'sync_every_steps': sync_every_steps,
            'outer_lr': outer_lr, 'outer_momentum': outer_momentum}


def resolve_execution_profile(values, config):
    """Resolve raw profile values using a recipe's global batch; import no factories.

    Derived fields are output-only. The execution dictionary is numerical
    identity; preflight timeouts are operational policy and not part of it.
    Structural success does not certify runtime compatibility or recovery.
    """
    values = _table(values, 'profile', ('schema_version', 'execution', 'preflight', 'aggregation'),
                    ('schema_version', 'execution'))
    if type(values['schema_version']) is not int or values['schema_version'] != 1:
        raise ValueError('Unsupported execution profile schema_version; expected integer 1')
    execution = _table(values['execution'], 'execution', EXECUTION_FIELDS, ('name',))
    name = execution['name']
    if not isinstance(name, str) or name not in _NAMES:
        raise ValueError('execution.name must be cpu-single, cpu-replicated-gloo or cuda-replicated-nccl')
    world = _positive_integer(execution.get('world_size', 1 if name == 'cpu-single' else 2), 'execution.world_size')
    accumulation = _positive_integer(execution.get('accumulation_steps', 1), 'execution.accumulation_steps')
    if world > 64:
        raise ValueError('execution.world_size must be at most 64')
    if name == 'cpu-single' and (world != 1 or accumulation != 1):
        raise ValueError('cpu-single requires world_size=1 and accumulation_steps=1')
    if name != 'cpu-single' and world < 2:
        raise ValueError(f'{name} requires world_size between 2 and 64')

    nodes = _positive_integer(execution.get('nodes', 1), 'execution.nodes')
    if name == 'cpu-single' and nodes != 1:
        raise ValueError('cpu-single requires nodes=1')
    if nodes > world:
        raise ValueError('execution.nodes cannot exceed execution.world_size')
    if world % nodes != 0:
        raise ValueError(f'Global execution.world_size ({world}) must divide evenly across execution.nodes ({nodes})')

    node_rank = _non_negative_integer(execution.get('node_rank', 0), 'execution.node_rank')
    if node_rank >= nodes:
        raise ValueError(f'execution.node_rank must be between 0 and nodes - 1 ({nodes - 1})')

    derived_ranks_per_node = world // nodes
    if 'ranks_per_node' in execution:
        rpn = _positive_integer(execution['ranks_per_node'], 'execution.ranks_per_node')
        if rpn != derived_ranks_per_node:
            raise ValueError(f'execution.ranks_per_node ({rpn}) must equal world_size // nodes ({derived_ranks_per_node})')
        ranks_per_node = rpn
    else:
        ranks_per_node = derived_ranks_per_node

    if 'rendezvous' in execution:
        rendezvous = _validate_rendezvous(execution['rendezvous'])
    elif nodes > 1:
        raise ValueError('Multi-node execution is experimental and requires an explicit execution.rendezvous = '
                         '"tcp://<node-0-host>:<port>" on every node; a loopback default cannot reach other hosts')
    else:
        rendezvous = 'file://'
    if nodes > 1 and not rendezvous.startswith('tcp://'):
        raise ValueError('Multi-node execution is experimental and requires a tcp://<node-0-host>:<port> rendezvous; '
                         'followers detect a failed node 0 through its TCP store')

    if name == 'cpu-single' and rendezvous.startswith('tcp://'):
        raise ValueError('cpu-single requires file:// rendezvous; tcp:// is unsupported for single-device execution')

    # Reconcile aggregation from profile root or execution table
    root_agg = values.get('aggregation')
    exec_agg = execution.get('aggregation')
    if root_agg is not None and exec_agg is not None and root_agg != exec_agg:
        raise ValueError('Conflicting aggregation tables specified in profile and execution')
    raw_aggregation = root_agg if root_agg is not None else exec_agg
    aggregation = _resolve_aggregation(raw_aggregation, name)

    if not isinstance(config, dict) or not isinstance(config.get('training'), dict):
        raise ValueError('Execution profile requires a recipe config with a training table')
    training = config['training']
    if name != 'cpu-single':
        validate_replicated_recipe(config, accumulation_steps=accumulation)
    if name == 'cuda-replicated-nccl':
        if training.get('device') != 'cuda':
            raise ValueError('cuda-replicated-nccl requires training.device=cuda; each rank owns its visible GPU index, so cuda:N is ambiguous')
    elif training.get('device') != 'cpu':
        raise ValueError('CPU execution profiles require training.device=cpu')
    batch = _positive_integer(training.get('batch_size'), 'training.batch_size')
    if batch % world:
        raise ValueError('Global training.batch_size must divide evenly across execution.world_size')
    local = batch // world
    if local % accumulation:
        raise ValueError('execution.accumulation_steps must divide local_batch_size evenly')
    limits = _table(values.get('preflight', {}), 'preflight', ('timeout', 'collective_timeout'))
    timeout = _seconds(limits.get('timeout', 60), 'preflight.timeout')
    collective = _seconds(limits.get('collective_timeout', 15), 'preflight.collective_timeout')
    if collective > timeout:
        raise ValueError('preflight.collective_timeout must not exceed preflight.timeout')
    return {'schema_version': 1,
            'execution': {'name': name, 'world_size': world, 'nodes': nodes, 'node_rank': node_rank,
                          'ranks_per_node': ranks_per_node, 'rendezvous': rendezvous,
                          'accumulation_steps': accumulation,
                          'global_batch_size': batch, 'local_batch_size': local,
                          'microbatch_size': local // accumulation,
                          'accumulation_algorithm': ('detached-logit-vjp-replay-v1' if accumulation > 1
                                                     else 'retained-local-graph-v1'),
                          'aggregation': aggregation},
            'aggregation': aggregation,
            'preflight': {'timeout': timeout, 'collective_timeout': collective}}


def experimental_features(profile):
    """Name the experimental parts of a resolved profile, for warnings and help."""
    execution = profile['execution']
    features = []
    if execution.get('nodes', 1) > 1:
        features.append(f"multi-node execution (nodes={execution['nodes']})")
    strategy = execution.get('aggregation', {}).get('strategy', 'gradient_allreduce')
    if strategy != 'gradient_allreduce':
        features.append(f'aggregation.strategy={strategy!r}')
    return features


def validate_checkpoint_cadence(profile, checkpoint_every):
    """Periodic checkpoints of local-SGD runs must land on synchronization steps.

    A checkpoint between synchronizations forces one (see
    ReplicatedTrainer.synchronize_replicas), which would silently change the
    configured sync schedule every checkpoint_every steps.
    """
    aggregation = profile['execution'].get('aggregation', {})
    sync = aggregation.get('sync_every_steps', 1)
    if (aggregation.get('strategy', 'gradient_allreduce') != 'gradient_allreduce'
            and type(checkpoint_every) is int and checkpoint_every % sync):
        raise ValueError(f"aggregation.strategy={aggregation['strategy']!r} is experimental: checkpoint_every "
                         f"({checkpoint_every}) must be a multiple of aggregation.sync_every_steps ({sync}), "
                         f"e.g. {max(sync, checkpoint_every - checkpoint_every % sync)}")


def load_execution_profile(path, config):
    """Read one bounded TOML profile, then perform structural validation only."""
    path = Path(path)
    with path.open('rb') as source:
        data = source.read(MAX_PROFILE_BYTES + 1)
    if len(data) > MAX_PROFILE_BYTES:
        raise ValueError(f'Execution profile exceeds {MAX_PROFILE_BYTES} bytes: {path}')
    try:
        values = tomllib.loads(data.decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'Invalid execution profile TOML in {path}: {exc}') from exc
    return resolve_execution_profile(values, config)


def validate_checkpoint_kind(kind, profile):
    """Check format routing only; this neither reads nor validates checkpoint state."""
    if not isinstance(kind, str) or kind not in (*_KINDS.values(), 'ema-inference'):
        raise ValueError('Unknown checkpoint kind; expected a native training checkpoint or ema-inference')
    if kind == 'ema-inference':
        raise ValueError('ema-inference bundles cannot resume training')
    execution = profile.get('execution') if isinstance(profile, dict) else None
    name = execution.get('name') if isinstance(execution, dict) else None
    if not isinstance(name, str) or name not in _NAMES:
        raise ValueError('Checkpoint kind validation requires a resolved execution profile')
    if kind != _KINDS[name]:
        raise ValueError(f'{name} requires checkpoint kind {_KINDS[name]}; no implicit format conversion')
    return kind

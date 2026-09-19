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
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _seconds(value, name):
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be finite positive seconds')
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} must be finite positive seconds') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'{name} must be finite positive seconds')
    return seconds


def resolve_execution_profile(values, config):
    """Resolve raw profile values using a recipe's global batch; import no factories.

    Derived fields are output-only. The execution dictionary is numerical
    identity; preflight timeouts are operational policy and not part of it.
    Structural success does not certify runtime compatibility or recovery.
    """
    values = _table(values, 'profile', ('schema_version', 'execution', 'preflight'),
                    ('schema_version', 'execution'))
    if type(values['schema_version']) is not int or values['schema_version'] != 1:
        raise ValueError('Unsupported execution profile schema_version; expected integer 1')
    execution = _table(values['execution'], 'execution', ('name', 'world_size', 'accumulation_steps'), ('name',))
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
    if not isinstance(config, dict) or not isinstance(config.get('training'), dict):
        raise ValueError('Execution profile requires a recipe config with a training table')
    training = config['training']
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
            'execution': {'name': name, 'world_size': world, 'accumulation_steps': accumulation,
                          'global_batch_size': batch, 'local_batch_size': local,
                          'microbatch_size': local // accumulation,
                          'accumulation_algorithm': ('detached-logit-vjp-replay-v1' if accumulation > 1
                                                     else 'retained-local-graph-v1')},
            'preflight': {'timeout': timeout, 'collective_timeout': collective}}


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

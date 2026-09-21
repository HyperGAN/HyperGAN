"""Public train/resume routing with torch-free validation before local viewing.

Recipes select the native device. Explicit profiles select fixed-topology
replication; deadlines are attempt policy, never numerical checkpoint identity.
Use a Python main guard when selecting a replicated profile.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from .config import config_values, fingerprint, load_config, resolve_config
from .previews import sample_name
from .execution_profiles import load_execution_profile, resolve_execution_profile

PROFILE_NAMES = ('cpu-single', 'cpu-replicated-gloo', 'cuda-replicated-nccl')
SERVICE_TIMEOUTS = ('startup_timeout', 'command_timeout', 'collective_timeout',
                    'total_timeout', 'observer_timeout', 'preview_timeout')
MAX_METADATA_BYTES = 16 * 1024 * 1024


def _read_metadata(path):
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_METADATA_BYTES + 1)
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError('Run/checkpoint metadata exceeds its byte bound')
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('Run/checkpoint metadata must be a JSON object')
    return value


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _differences(current, original):
    """Name the resolved top-level sections that differ, including defaulted fields."""
    keys = sorted(set(current) | set(original))
    return [key for key in keys if not _same(current.get(key), original.get(key))]


def _training_config(config_path, steps=None):
    config = load_config(config_path)
    if steps is not None:
        values = config_values(config)
        values['training']['steps'] = steps
        config = resolve_config(values)
    return config


def _profile(value, config):
    if value is None:
        return None
    if isinstance(value, str) and value in PROFILE_NAMES:
        return resolve_execution_profile({'schema_version': 1, 'execution': {'name': value}}, config)
    if isinstance(value, (str, Path)):
        return load_execution_profile(value, config)
    from .execution_preflight import _resolve_profile
    return _resolve_profile(value, config)


def _execution(profile):
    if profile is None or profile['execution']['name'] == 'cpu-single':
        return None
    return profile['execution']


def _policy(profile, service_policy):
    if service_policy is not None and not isinstance(service_policy, dict):
        raise ValueError('service_policy must be a dictionary of replicated deadlines')
    if _execution(profile) is None:
        if service_policy:
            raise ValueError('Service timeout options require a replicated execution profile')
        return None
    from .replicated_execution import _policy as resolve_policy
    return resolve_policy(profile, service_policy)


def _checkpoint(run_dir, checkpoint, manifest, execution, total_steps):
    """Pin a complete generation and reject routing/config conflicts without torch.

    The adapter validates all payload bytes, runtime, data and implementation
    during restore, under the run lock and before publishing a new attempt.
    """
    root = (run_dir / ('distributed-checkpoints' if execution else 'checkpoints')).resolve()
    pointer = None
    if checkpoint is None:
        if not (root / 'latest.json').exists():
            raise ValueError('No full training checkpoint exists; an inference model is not resumable')
        pointer = _read_metadata(root / 'latest.json')
        if type(pointer.get('schema_version')) is not int or pointer['schema_version'] != 1:
            raise ValueError('Unsupported training checkpoint pointer schema')
        if execution and pointer.get('kind') != 'hypergan-distributed-training-checkpoint':
            raise ValueError('Invalid distributed checkpoint pointer kind')
        name = pointer.get('checkpoint')
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Invalid training checkpoint pointer path')
        checkpoint = root / name
    target = Path(checkpoint)
    if not target.is_absolute():
        choices = (target, run_dir / target, root / target)
        target = next((choice for choice in choices if choice.is_dir()), root / target)
    if target.is_symlink():
        raise ValueError('Checkpoint must be an ordinary completed directory inside this run')
    target = target.resolve()
    if target.parent != root or target.name.startswith('.') or not target.is_dir():
        raise ValueError('Checkpoint must be a completed checkpoint directory inside this run')
    info = _read_metadata(target / 'manifest.json')
    if type(info.get('schema_version')) is not int or info['schema_version'] != 1:
        raise ValueError('Unsupported training checkpoint schema')
    if type(info.get('step')) is not int or not 0 <= info['step'] <= total_steps:
        raise ValueError('Checkpoint step is outside the original schedule')
    if pointer is not None and (type(pointer.get('step')) is not int or pointer['step'] != info['step']):
        raise ValueError('Checkpoint pointer step differs from generation')
    expected = 'hypergan-distributed-training-checkpoint' if execution else 'hypergan-training-checkpoint'
    if info.get('kind') != expected:
        raise ValueError(f'Selected execution requires checkpoint kind {expected}')
    if info.get('run_id') != manifest['run_id']:
        raise ValueError('Checkpoint belongs to a different run')
    identity = info.get('identity', {}) if execution else info
    if not isinstance(identity, dict) or identity.get('config_sha256') != manifest['config_sha256']:
        raise ValueError('Resume checkpoint configuration differs from the run manifest')
    from .checkpoint_compatibility import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(identity)
    if execution:
        topology = identity.get('topology', {})
        if not isinstance(topology, dict) or any(not _same(topology.get(key), value) for key, value in execution.items()):
            raise ValueError('Resume checkpoint numerical execution identity differs from the run manifest')
    if info.get('event_boundary') is None:
        raise ValueError('Controller checkpoint requires a durable event boundary')
    from .run_state import validate_event_boundary
    validate_event_boundary(run_dir, info)
    return target


@dataclass(frozen=True)
class PreparedExecution:
    """Validated routing; running rechecks config and checkpoint under controller ownership."""
    operation: str
    config: dict
    run_dir: Path
    config_path: object
    checkpoint: object
    steps: object
    profile: object
    service_policy: object
    controls: dict
    repeat_train: bool = False

    def run(self, *, on_event=None):
        # Revalidate files/options before dispatch: preparation is not a lock and
        # caller code may have changed the config while starting the viewer.
        if self.operation == 'train':
            current = prepare_train(self.config_path, self.run_dir, self.steps,
                profile=self.profile, service_policy=self.service_policy, **self.controls)
        else:
            current = prepare_resume(self.run_dir, self.checkpoint, self.config_path,
                steps=self.steps, _repeat_train=self.repeat_train,
                profile=self.profile, service_policy=self.service_policy, **self.controls)
        if current.operation != self.operation:
            raise FileExistsError(f'Run directory appeared after training was prepared: {self.run_dir}')
        if fingerprint(current.config) != fingerprint(self.config) or not _same(_execution(current.profile), _execution(self.profile)):
            raise ValueError('Prepared numerical configuration or execution identity changed')
        if self.repeat_train and not _same(config_values(current.config), config_values(self.config)):
            raise ValueError('Prepared training configuration changed')
        if _execution(self.profile) is None:
            from .training import train as run_train, resume as run_resume
            options = {}
        else:
            from .replicated_execution import run_train, run_resume
            options = {'profile': self.profile, 'service_policy': self.service_policy}
        if self.operation == 'train':
            return run_train(self.config_path, self.run_dir, self.steps,
                             on_event=on_event, **options, **self.controls)
        return run_resume(self.run_dir, self.checkpoint, self.config_path,
                          steps=self.steps, require_same_config=self.repeat_train,
                          on_event=on_event, **options, **self.controls)


def prepare_train(config_path, run_dir, steps=None, *, profile=None, service_policy=None,
                  checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                  preview_every=None, preview_keep=None, preview_keep_source=None,
                  preview_name=None):
    """Create a run or resume its latest full checkpoint with the same configuration."""
    from .run_controller import _controls, resolve_preview_keep
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        return prepare_resume(run_dir, config_path=config_path, steps=steps, _repeat_train=True,
            profile=profile, service_policy=service_policy, checkpoint_every=checkpoint_every,
            max_seconds=max_seconds, stop_after_steps=stop_after_steps,
            preview_every=preview_every, preview_keep=preview_keep,
            preview_keep_source=preview_keep_source, preview_name=preview_name)
    checkpoint_every = 100 if checkpoint_every is None else checkpoint_every
    preview_every = 0 if preview_every is None else preview_every
    preview_keep, preview_keep_source = resolve_preview_keep({}, preview_keep, preview_keep_source)
    preview_name = sample_name(preview_name)
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
    config = _training_config(config_path, steps)
    profile = _profile(profile, config)
    policy = _policy(profile, service_policy)
    if _execution(profile) is not None:
        from .replicated_execution import MAX_SAMPLE_COUNT
        if config['sampling']['count'] > MAX_SAMPLE_COUNT:
            raise ValueError(f'Replicated final inference sample count exceeds {MAX_SAMPLE_COUNT}')
    controls = dict(checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                    stop_after_steps=stop_after_steps, preview_every=preview_every,
                    preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                    preview_name=preview_name)
    return PreparedExecution('train', config, run_dir, config_path, None, steps, profile, policy, controls)


def prepare_resume(run_dir, checkpoint=None, config_path=None, *, steps=None, _repeat_train=False,
                   profile=None, service_policy=None,
                   checkpoint_every=None, max_seconds=None, stop_after_steps=None,
                   preview_every=None, preview_keep=None, preview_keep_source=None,
                   preview_name=None):
    """Infer persisted numerical routing and pin an earlier/latest complete snapshot."""
    from .run_controller import _controls, resolve_preview_keep
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f'Run directory does not exist or is not a directory: {run_dir}')
    manifest = _read_metadata(run_dir / 'manifest.json')
    required = {'schema_version', 'run_id', 'config', 'config_sha256', 'next_sample_sequence'}
    if not required.issubset(manifest) or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('Run manifest has no supported full recovery contract')
    config = _training_config(config_path, steps) if config_path is not None else resolve_config(manifest['config'])
    if config_path is None and steps is not None:
        raise ValueError('A training step override requires an explicit configuration')
    if _repeat_train:
        original = config_values(resolve_config(manifest['config']))
        changed = _differences(config_values(config), original)
        if changed:
            # Resolved defaults are part of the recorded configuration, so a default
            # that changed between releases surfaces here instead of silently applying.
            remedy = ('`hypergan resume RUN --config CONFIG` continues this run with changed '
                      'observation settings' if changed == ['metrics'] else
                      'use a new run directory for a different numerical configuration')
            raise ValueError('Training configuration differs from the original run in '
                             + ', '.join(changed) + '; ' + remedy)
    if fingerprint(config) != manifest['config_sha256']:
        raise ValueError('Resume configuration differs from the original run; total training schedule cannot change')
    saved = manifest.get('execution')
    if 'execution' in manifest and (not isinstance(saved, dict) or saved.get('name') not in PROFILE_NAMES[1:]):
        raise ValueError('Run has no supported numerical execution identity')
    if profile is None and saved is not None:
        profile = {'schema_version': 1, 'execution': {
            key: saved.get(key) for key in ('name', 'world_size', 'accumulation_steps')}}
    profile = _profile(profile, config)
    execution = _execution(profile)
    if not _same(execution, saved):
        raise ValueError('Resume numerical execution identity differs from the run manifest')
    policy = _policy(profile, service_policy)
    checkpoint_every = manifest.get('checkpoint_every', 100) if checkpoint_every is None else checkpoint_every
    preview_every = manifest.get('preview_every', 0) if preview_every is None else preview_every
    preview_keep, preview_keep_source = resolve_preview_keep(manifest, preview_keep, preview_keep_source)
    preview_name = sample_name(manifest.get('preview_name') if preview_name is None else preview_name)
    _controls(checkpoint_every, max_seconds, stop_after_steps, preview_every, preview_keep, preview_name)
    checkpoint = _checkpoint(run_dir, checkpoint, manifest, execution, config['training']['steps'])
    controls = dict(checkpoint_every=checkpoint_every, max_seconds=max_seconds,
                    stop_after_steps=stop_after_steps, preview_every=preview_every,
                    preview_keep=preview_keep, preview_keep_source=preview_keep_source,
                    preview_name=preview_name)
    return PreparedExecution('resume', config, run_dir, config_path, checkpoint, steps, profile, policy, controls, _repeat_train)


def train(config_path, run_dir, steps=None, *, on_event=None, **options):
    """Create or resume a run; existing runs require the same resolved configuration."""
    return prepare_train(config_path, run_dir, steps, **options).run(on_event=on_event)


def resume(run_dir, checkpoint=None, config_path=None, *, on_event=None, **options):
    """Resume using the saved profile; explicit profile changes must match exactly."""
    return prepare_resume(run_dir, checkpoint, config_path, **options).run(on_event=on_event)

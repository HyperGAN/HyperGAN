"""Explicit HyperGAN recovery contract, independent of release provenance.

Bump CURRENT_VERSION when a known change makes saved state or continuation
semantics incompatible. Schema-1 checkpoints written before this field existed
use version 1. Git revisions and package versions identify what ran; they do not
decide whether HyperGAN can restore it. External components and numerical runtime
settings retain their existing strict checks.
"""
import json
import warnings


CURRENT_VERSION = 2

# Audited 0.5.0 -> 0.6.0 constructor migration. Saved prior buffers/read settings
# still restore exactly; loss and penalty implementations must remain identical.
# These are exact source digests, not a blanket exception for library upgrades.
PARTICLEGAN_060_MIGRATION = {
    'particlegan.particle_prior': (
        'eabbc41c3c42376e0b748b06a87fae4f8d10314002dfd888a5bd30e8f311b69b',
        '17e39404cefca5963c82d9981f8582ea6c650c9aae66b51ce11ecf05896bb9a9'),
    'particlegan.recipes': (
        '33ea97c94f4dd79ee2c5451cdf9e9ce98272921d71a300918258e8df3e6a88bc',
        '79c4449d804f1b701dde8d4578c676ef31ceb4806f12705b8a237cec821d0cc9'),
}

class HyperGANHardwareWarning(RuntimeWarning):
    """Structured warning emitted when running or resuming on differing hardware or topology."""
    pass


# Keys identifying physical hardware topology or device enumeration.
# Differences in these keys emit HyperGANHardwareWarning and permit resume (Decision D9).
HARDWARE_TOLERANCE_KEYS = frozenset({
    'cuda.name',
    'cuda.capability',
    'cuda.rank_devices',
    'world_size',
    'device',
    'platform',
    'machine',
    'cuda.uuid',
    'cuda.visible_devices',
    'cuda.nccl',
    'cuda.gcnArchName',
})

DEVICE_IDENTITY_KEYS = HARDWARE_TOLERANCE_KEYS


def _is_hardware_tolerant(path: str) -> bool:
    return path in HARDWARE_TOLERANCE_KEYS or path.startswith('cuda.rank_devices.')


# Other keys in `runtime` decide behavior and stay hard failures: device
# type, dtype, world size, torch/cuda/cudnn versions, the deterministic, tf32 and
# matmul settings, the GPU model and capability, python/numpy and the platform.
# The explicit-sigma library migration below additionally requires audited
# source hashes. An accepted identity change still proves the model,
# capability and every numerical runtime setting match.

MAX_VALUE_CHARACTERS = 200

_ABSENT = object()


def validate_checkpoint_compatibility(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Invalid HyperGAN checkpoint compatibility metadata')
    version = metadata.get('hypergan_checkpoint_version', 1)
    if type(version) is not int or version != CURRENT_VERSION:
        raise ValueError(
            f'Unsupported HyperGAN checkpoint compatibility version: {version!r}; '
            f'this installation supports version {CURRENT_VERSION}')


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _flatten(value, prefix=''):
    """Dotted key paths for nested metadata, so a difference names one field."""
    flat = {}
    for key, item in value.items():
        path = f'{prefix}{key}'
        if isinstance(item, dict) and item:
            flat.update(_flatten(item, f'{path}.'))
        else:
            flat[path] = item
    return flat


def _render(value):
    if value is _ABSENT:
        return 'absent'
    text = json.dumps(value, sort_keys=True, allow_nan=False, default=repr)
    return text if len(text) <= MAX_VALUE_CHARACTERS else text[:MAX_VALUE_CHARACTERS] + '...'


def _differences(saved, current):
    """Every differing key path, as (path, saved value, current value)."""
    saved_flat, current_flat = _flatten(saved), _flatten(current)
    result = []
    for path in sorted(set(saved_flat) | set(current_flat)):
        left = saved_flat.get(path, _ABSENT)
        right = current_flat.get(path, _ABSENT)
        if left is _ABSENT or right is _ABSENT or not _same(left, right):
            result.append((path, left, right))
    return result


def _describe(differences):
    return '; '.join(f'{path}: saved {_render(left)}, current {_render(right)}'
                     for path, left, right in differences)


def _device_identity_warning(saved, differences):
    card = saved.get('cuda') if isinstance(saved.get('cuda'), dict) else {}
    model = card.get('name', 'the same device model')
    capability = _render(card['capability']) if 'capability' in card else 'unchanged'
    return ('Resuming on a different physical GPU of the same model '
            f'({model}, capability {capability}): {_describe(differences)}. '
            'The numerical runtime is unchanged, so the checkpoint restores as saved; '
            'export CUDA_DEVICE_ORDER=PCI_BUS_ID and select the card with '
            'CUDA_VISIBLE_DEVICES to pin one physical device across restarts.')


def validate_runtime(saved, current, *, warn=None, saved_implementation=None,
                     current_implementation=None):
    """Reject an incompatible runtime, naming every differing key path.

    A difference confined to DEVICE_IDENTITY_KEYS says which card ran, not what
    it computes, so it warns and resumes. Returns the warning messages; `warn`
    takes a sink that receives each message instead of `warnings.warn`, for a
    caller that also records them on the run. The ParticleGAN migration also
    requires both implementation inventories and their exact audited hashes.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise ValueError('Invalid resume runtime metadata')
    # A HyperGAN release can change without changing its recovery contract.
    saved = {key: value for key, value in saved.items() if key != 'hypergan'}
    current = {key: value for key, value in current.items() if key != 'hypergan'}
    upgraded = ((saved.get('particlegan'), current.get('particlegan')) == ('0.5.0', '0.6.0')
                and _particlegan_060_upgrade(saved_implementation, current_implementation))
    if upgraded:
        # Also check all other external implementations before qualifying the
        # version change. A different loss/penalty cannot slip through here.
        validate_implementation(saved_implementation, current_implementation)
        saved['particlegan'] = current['particlegan']
    differences = _differences(saved, current)
    if not differences and not upgraded:
        return []
    incompatible = [difference for difference in differences
                    if not _is_hardware_tolerant(difference[0])]
    if incompatible:
        raise ValueError(
            'Resume runtime/topology differs from checkpoint: ' + _describe(incompatible) +
            '. Resume with the runtime that wrote the checkpoint, or start a new run directory.')
    hardware_diffs = [difference for difference in differences
                      if _is_hardware_tolerant(difference[0])]
    messages = []
    if hardware_diffs:
        card = saved.get('cuda') if isinstance(saved.get('cuda'), dict) else {}
        is_only_uuid_visible = all(d[0] in {'cuda.uuid', 'cuda.visible_devices'} for d in hardware_diffs)
        if is_only_uuid_visible and card.get('name'):
            messages.append(_device_identity_warning(saved, hardware_diffs))
        else:
            messages.append(
                f'Resuming on differing hardware/topology: {_describe(hardware_diffs)}. '
                'The numerical runtime settings are unchanged, so the checkpoint restores as saved.'
            )
    if upgraded:
        messages.append('Resuming ParticleGAN 0.5.0 state with the audited 0.6.0 explicit-sigma '
                        'implementation; saved centers, sigma and read settings restore unchanged.')
    for message in messages:
        if warn is None:
            warnings.warn(message, HyperGANHardwareWarning, stacklevel=2)
        else:
            warn(message)
    return messages



def _particlegan_060_upgrade(saved, current):
    return (isinstance(saved, dict) and isinstance(current, dict)
            and all(saved.get(key) == old and current.get(key) == new
                    for key, (old, new) in PARTICLEGAN_060_MIGRATION.items()))


def validate_implementation(saved, current):
    for value in (saved, current):
        if not isinstance(value, dict) or any(
                not isinstance(key, str) or not isinstance(digest, str)
                for key, digest in value.items()):
            raise ValueError('Invalid resume implementation metadata')
    def external(value):
        return {key: digest for key, digest in value.items()
                if key != 'hypergan' and not key.startswith('hypergan.')}
    left, right = external(saved), external(current)
    if _particlegan_060_upgrade(left, right):
        left.update({key: new for key, (_, new) in PARTICLEGAN_060_MIGRATION.items()})
    if not _same(left, right):
        raise ValueError('Resume implementation differs from checkpoint: external component or dependency changed')

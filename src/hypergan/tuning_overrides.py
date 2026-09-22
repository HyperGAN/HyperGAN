"""Strict checkpoint contract for startup-selected adversarial learning rates."""
import hashlib
import json
import math

MIN_G_LR_FACTOR = 0.1
MAX_G_LR_FACTOR = 1.0
_FIELDS = {'schema_version', 'scope', 'factor', 'baseline_g_lr', 'effective_g_lr', 'optimizer_config_sha256'}
_FIELDS_V2 = _FIELDS | {'d_factor', 'baseline_d_lr', 'effective_d_lr'}


def optimizer_fingerprint(config):
    return hashlib.sha256(json.dumps(config['optimizer'], sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def generator_lr_override(config, factor):
    if not _positive(factor) or not MIN_G_LR_FACTOR <= factor <= MAX_G_LR_FACTOR:
        raise ValueError('Startup generator learning-rate factor must be between 0.1 and 1')
    base = config['optimizer']['lr']
    return {'schema_version': 1, 'scope': 'generator-main-group', 'factor': float(factor),
            'baseline_g_lr': base, 'effective_g_lr': base * factor,
            'optimizer_config_sha256': optimizer_fingerprint(config)}


def optimizer_lr_override(config, g_factor, d_factor):
    """Record G/D-main overrides while preserving the original prior rate."""
    override = generator_lr_override(config, g_factor)
    if not _positive(d_factor) or not MIN_G_LR_FACTOR <= d_factor <= MAX_G_LR_FACTOR:
        raise ValueError('Startup discriminator learning-rate factor must be between 0.1 and 1')
    baseline = config['optimizer']['lr'] * config['optimizer']['d_lr_mult']
    override.update(schema_version=2, scope='generator-and-discriminator-main-groups',
                    d_factor=float(d_factor), baseline_d_lr=baseline, effective_d_lr=baseline * d_factor)
    return override


def scheduled_generator_lr(base_lr, warmup, step):
    """Unannealed G-main rate: update one starts low, update N reaches target.

    ``step=0`` describes the saved initialization. The selected base rate stays
    constant in checkpoint identity even after this optional ramp completes.
    """
    if warmup is None:
        return base_lr
    if step <= 1:
        return warmup['start_g_lr']
    if step >= warmup['steps']:
        return warmup['target_g_lr']
    fraction = (step - 1) / (warmup['steps'] - 1)
    return warmup['start_g_lr'] + (warmup['target_g_lr'] - warmup['start_g_lr']) * fraction


def checkpoint_g_lr_warmup(trainer, metadata):
    """Validate the optional schedule against the original config and override."""
    expected, overridden = checkpoint_base_lrs(trainer, metadata)
    tuning = metadata.get('initialization_tuning') if metadata is not None else None
    if not isinstance(tuning, dict) or 'g_lr_warmup' not in tuning:
        return None
    warmup = tuning['g_lr_warmup']
    if (not overridden or not isinstance(warmup, dict)
            or set(warmup) != {'steps', 'start_g_lr', 'target_g_lr'}
            or type(warmup['steps']) is not int or warmup['steps'] < 2
            or not _positive(warmup['start_g_lr']) or not _positive(warmup['target_g_lr'])
            or warmup['start_g_lr'] != expected[0][0]
            or warmup['target_g_lr'] != trainer.config['optimizer']['lr']):
        raise ValueError('Invalid startup generator learning-rate warmup metadata')
    return dict(warmup)


def checkpoint_base_lrs(trainer, metadata):
    """Return expected rates, permitting only explicitly recorded G/D overrides.

    Missing override retains the historical exact-original-rate check. A factor
    never compounds: the recorded effective rate is derived from source config,
    applied only to a newly constructed trainer's authorized main groups. Old
    schema-one checkpoints permit only G; schema two additionally records D.
    """
    expected = [list(rates) for rates in trainer.base_lrs]
    if metadata is None:
        return expected, False
    tuning = metadata.get('initialization_tuning')
    if not isinstance(tuning, dict) or 'optimizer_override' not in tuning:
        if isinstance(tuning, dict) and 'g_lr_warmup' in tuning:
            raise ValueError('Startup generator learning-rate warmup requires a validated optimizer override')
        return expected, False
    override = tuning['optimizer_override']
    if not isinstance(override, dict):
        raise ValueError('Invalid completed startup optimizer override metadata')
    version = override.get('schema_version')
    scope = {1: 'generator-main-group', 2: 'generator-and-discriminator-main-groups'}
    if (tuning.get('status') != 'complete' or type(version) is not int or version not in scope
            or set(override) != (_FIELDS if version == 1 else _FIELDS_V2)
            or override['scope'] != scope[version]):
        raise ValueError('Invalid completed startup optimizer override metadata')
    factor = override['factor']
    if (not _positive(factor) or not MIN_G_LR_FACTOR <= factor <= MAX_G_LR_FACTOR
            or not _positive(override['baseline_g_lr']) or not _positive(override['effective_g_lr'])):
        raise ValueError('Invalid startup generator learning-rate override values')
    baseline = trainer.config['optimizer']['lr']
    outcome = tuning.get('dynamics_outcome')
    selected = tuning.get('selected_g_lr_factor')
    d_factor = override['d_factor'] if version == 2 else 1.
    d_selected = tuning.get('selected_d_lr_factor', 1. if version == 1 else None)
    if (not _positive(d_factor) or not MIN_G_LR_FACTOR <= d_factor <= MAX_G_LR_FACTOR
            or type(d_selected) not in (int, float) or d_selected != d_factor):
        raise ValueError('Invalid startup discriminator learning-rate override values or recorded decision')
    if (type(selected) not in (int, float) or selected != factor
            or outcome not in ('selected', 'kept_baseline', 'unresolved', 'skipped')
            or (factor != 1 or d_factor != 1) != (outcome == 'selected')
            or override['baseline_g_lr'] != baseline or override['effective_g_lr'] != baseline * factor
            or override['optimizer_config_sha256'] != optimizer_fingerprint(trainer.config)
            or expected[0][0] != baseline):
        raise ValueError('Startup generator learning-rate override differs from original configuration or recorded decision')
    expected[0][0] = override['effective_g_lr']
    if version == 2:
        d_baseline = baseline * trainer.config['optimizer']['d_lr_mult']
        if (not _positive(override['baseline_d_lr']) or not _positive(override['effective_d_lr'])
                or override['baseline_d_lr'] != d_baseline
                or override['effective_d_lr'] != d_baseline * d_factor
                or expected[1][0] != d_baseline):
            raise ValueError('Startup discriminator learning-rate override differs from original configuration')
        expected[1][0] = override['effective_d_lr']
    return expected, True

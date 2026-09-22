"""Strict checkpoint contract for startup-selected generator learning rates."""
import hashlib
import json
import math

MIN_G_LR_FACTOR = 0.1
MAX_G_LR_FACTOR = 1.0
_FIELDS = {'schema_version', 'scope', 'factor', 'baseline_g_lr', 'effective_g_lr', 'optimizer_config_sha256'}


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


def checkpoint_base_lrs(trainer, metadata):
    """Return expected rates, permitting only the explicitly recorded G override.

    Missing override retains the historical exact-original-rate check. A factor
    never compounds: the recorded effective rate is derived from source config,
    applied only to a newly constructed trainer's generator group zero.
    """
    expected = [list(rates) for rates in trainer.base_lrs]
    if metadata is None:
        return expected, False
    tuning = metadata.get('initialization_tuning')
    if not isinstance(tuning, dict) or 'optimizer_override' not in tuning:
        return expected, False
    override = tuning['optimizer_override']
    if (tuning.get('status') != 'complete' or not isinstance(override, dict) or set(override) != _FIELDS
            or type(override['schema_version']) is not int or override['schema_version'] != 1
            or override['scope'] != 'generator-main-group'):
        raise ValueError('Invalid completed startup optimizer override metadata')
    factor = override['factor']
    if (not _positive(factor) or not MIN_G_LR_FACTOR <= factor <= MAX_G_LR_FACTOR
            or not _positive(override['baseline_g_lr']) or not _positive(override['effective_g_lr'])):
        raise ValueError('Invalid startup generator learning-rate override values')
    baseline = trainer.config['optimizer']['lr']
    outcome = tuning.get('dynamics_outcome')
    selected = tuning.get('selected_g_lr_factor')
    if (type(selected) not in (int, float) or selected != factor
            or outcome not in ('selected', 'kept_baseline', 'unresolved', 'skipped')
            or (factor != 1) != (outcome == 'selected')
            or override['baseline_g_lr'] != baseline or override['effective_g_lr'] != baseline * factor
            or override['optimizer_config_sha256'] != optimizer_fingerprint(trainer.config)
            or expected[0][0] != baseline):
        raise ValueError('Startup generator learning-rate override differs from original configuration or recorded decision')
    expected[0][0] = override['effective_g_lr']
    return expected, True

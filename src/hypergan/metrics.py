"""Finite scalar publication and immutable, torch-free metric definitions.

Numerical adapters always validate their complete internal results. Publication
selects already computed values; it never computes objectives or executes plugins.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .run_state import atomic_json
from .metric_plugins import validate_custom, enabled_custom

PRESET_VERSION = 'standard/v1'
DEFAULT_METRICS = {'preset': 'standard', 'disable': [], 'every_steps': 1, 'overrides': {}, 'custom': {}}
MAX_CATALOG_BYTES = 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def objective_id(term):
    """An explicit recipe ID is preferable; content identity survives reordering."""
    return term.get('id') or 'objective-' + digest(term)[:16]


def _available(config):
    specs = {
        'loss/d_total': ('d_loss', 'Discriminator total', 'loss'),
        'loss/g_total': ('g_loss', 'Generator total', 'loss'),
        'loss/total': ('combined', 'Combined loss (D + G)', 'loss'),
        'loss/d_adversarial': ('d_adversarial_weighted', 'D adversarial contribution', 'loss'),
        'loss/g_adversarial': ('g_adversarial_weighted', 'G adversarial contribution', 'loss'),
        'loss/d_adversarial_raw': ('d_adversarial', 'D adversarial (raw)', 'loss'),
        'loss/g_adversarial_raw': ('g_adversarial', 'G adversarial (raw)', 'loss'),
        'loss/gradient_penalty': ('gradient_penalty', 'D gradient penalty contribution', 'loss'),
        'loss/prior_regularizer': ('prior_loss', 'G prior regularizer contribution', 'loss'),
        'optimizer/lr_scale': ('lr_scale', 'Learning rate multiplier', 'ratio'),
        'timing/step_seconds': ('step_seconds', 'Complete update duration', 'seconds'),
    }
    for term in config['objectives']:
        name = objective_id(term)
        specs['loss/objectives/' + name] = ('objective:' + name, name + ' contribution', 'loss')
    return specs


def validate_metrics(config):
    spec = config['metrics']
    if spec['preset'] not in ('standard', 'none'):
        raise ValueError('metrics.preset must be standard or none')
    if type(spec['every_steps']) is not int or spec['every_steps'] < 1:
        raise ValueError('metrics.every_steps must be a positive integer')
    validate_custom(spec['custom'])
    available = _available(config)
    if set(available) & set(spec['custom']):
        raise ValueError('Custom metric IDs cannot replace built-in definitions')
    available.update({name: None for name in spec['custom']})
    if not isinstance(spec['disable'], list) or any(not isinstance(name, str) or name not in available for name in spec['disable']):
        raise ValueError('metrics.disable must list known metric IDs')
    if len(set(spec['disable'])) != len(spec['disable']):
        raise ValueError('metrics.disable contains duplicate IDs')
    if not isinstance(spec['overrides'], dict):
        raise ValueError('metrics.overrides must be a table')
    for name, override in spec['overrides'].items():
        if name not in available:
            raise ValueError(f'Unknown metric ID: {name}')
        if not isinstance(override, dict) or set(override) != {'enabled'} or type(override['enabled']) is not bool:
            raise ValueError(f'metrics.overrides.{name} requires only enabled = true/false')
        if name in spec['disable'] and override['enabled']:
            raise ValueError(f'Conflicting metric enable and disable: {name}')


def metric_catalog(config):
    from .config import fingerprint
    spec = config['metrics']
    metrics = {}
    for name, (source, label, unit) in _available(config).items():
        enabled = spec['overrides'].get(name, {}).get('enabled', spec['preset'] == 'standard')
        if not enabled or name in spec['disable']:
            continue
        definition = {'kind': 'scalar', 'source': source, 'label': label, 'unit': unit,
                      'numerical_sha256': fingerprint(config), 'preset_version': PRESET_VERSION,
                      'direction': 'none', 'scope': 'complete_update',
                      'owner': 'discriminator' if name.startswith(('loss/d_', 'loss/gradient')) else
                               'generator' if name.startswith(('loss/g_', 'loss/prior', 'loss/objectives/')) else 'run',
                      'view': {'panel': 'Loss' if name.startswith('loss/') else 'Progress'}}
        if name == 'loss/total':
            definition['formula'] = 'loss/d_total + loss/g_total'
            definition['description'] = 'Diagnostic sum; not a joint optimization objective or quality score.'
        if 'adversarial' in name:
            definition['coefficient'] = config['adversarial']['weight']
        if name == 'loss/gradient_penalty':
            definition.update(coefficient=config['gradient_penalty']['coeff'], lazy_k=config['gradient_penalty']['lazy_k'],
                              raw_available=False, description='Applied weighted contribution; raw penalty is not exposed by the upstream callable.')
        if name == 'loss/prior_regularizer':
            definition.update(coefficient=config['prior_regularizer']['weight'], raw_available=False,
                              description='Applied weighted contribution; raw regularizer is not exposed by the upstream callable.')
        if source.startswith('objective:'):
            term = next(term for term in config['objectives'] if objective_id(term) == source.split(':', 1)[1])
            definition['coefficient'] = term['weight']
            definition['raw_available'] = False
        metrics[name] = dict(definition, definition_hash=digest(definition))
    for name, custom in enabled_custom(config).items():
        runtime = config.get('_metric_runtime', {}).get(name)
        if runtime is None:
            raise ValueError('Custom metric catalogs require bounded runtime preflight')
        definition = dict(runtime['descriptor'], source='custom:' + name, factory=custom['factory'],
                          factory_sources=runtime['factory_sources'], protocol=runtime['protocol'],
                          specification=deepcopy(custom), numerical_sha256=fingerprint(config),
                          scope='snapshot' if custom['mode'] == 'snapshot' else 'complete_update')
        metrics[name] = dict(definition, definition_hash=digest(definition))
    catalog = {'schema_version': 1, 'preset_version': PRESET_VERSION,
               'observation': deepcopy(spec), 'metrics': metrics}
    if len(json.dumps(catalog).encode()) > MAX_CATALOG_BYTES:
        raise ValueError('Resolved metrics catalog exceeds 1 MiB')
    return catalog


def publish_catalog(run_dir, config):
    """Called under the controller's run lock; an existing revision is immutable."""
    catalog = metric_catalog(config)
    revision = digest(catalog)
    path = Path(run_dir) / 'metrics' / f'catalog-{revision}.json'
    if path.exists():
        if read_catalog(run_dir, revision) != catalog:
            raise ValueError('Immutable metric catalog differs from its revision')
    else:
        atomic_json(path, catalog)
    return revision


def read_catalog(run_dir, revision=None, *, _open_file=None):
    root = Path(run_dir)
    if revision is None:
        revision = json.loads((root / 'manifest.json').read_text())['metrics_catalog']
    if not isinstance(revision, str) or re.fullmatch('[0-9a-f]{64}', revision) is None:
        raise ValueError('Invalid metric catalog revision')
    path = root / 'metrics' / f'catalog-{revision}.json'
    if path.is_symlink():
        raise ValueError('Metric catalog must not be a symlink')
    with (_open_file(path) if _open_file else path.open('rb')) as stream:
        value = stream.read(MAX_CATALOG_BYTES + 1)
    if len(value) > MAX_CATALOG_BYTES:
        raise ValueError('Metric catalog exceeds 1 MiB')
    catalog = json.loads(value)
    if not isinstance(catalog, dict) or catalog.get('schema_version') != 1 or digest(catalog) != revision:
        raise ValueError('Metric catalog content does not match its revision')
    if not isinstance(catalog.get('metrics'), dict):
        raise ValueError('Invalid metric catalog definitions')
    for name, descriptor in catalog['metrics'].items():
        if (not isinstance(name, str) or not name or not isinstance(descriptor, dict)
                or descriptor.get('kind') not in ('scalar', 'histogram') or not isinstance(descriptor.get('source'), str)
                or descriptor.get('definition_hash') != digest({k: v for k, v in descriptor.items() if k != 'definition_hash'})):
            raise ValueError('Invalid metric catalog definition or hash')
    return catalog


def select_metrics(config, catalog, row, step, step_seconds):
    """Sample the completed boundary, never average skipped steps or recompute."""
    if step % config['metrics']['every_steps']:
        return {}, {}, 'cadence'
    values = dict(row, step_seconds=step_seconds)
    if 'd_loss' in row and 'g_loss' in row:
        values['combined'] = row['d_loss'] + row['g_loss']
    for term, value in zip(config['objectives'], row.get('objectives', [])):
        values['objective:' + objective_id(term)] = value
    metrics, statuses = {}, {}
    for name, definition in catalog['metrics'].items():
        source = definition['source']
        if source.startswith('custom:'):
            continue
        if source not in values:
            statuses[name] = {'status': 'unavailable', 'reason': 'Execution did not provide this scalar'}
            continue
        value = values[source]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Nonfinite or invalid metric value: {name}')
        metrics[name] = value
        if name == 'loss/gradient_penalty':
            penalty = config['gradient_penalty']
            applied = penalty['arm'] != 'f_none' and step % penalty['lazy_k'] == 0
            statuses[name] = {'status': 'available', 'applied': applied,
                              'effective_coefficient': penalty['coeff'] * penalty['lazy_k'] if applied else 0.0}
    return metrics, statuses, 'sampled' if metrics or statuses else 'disabled'


def validate_update_scalars(row, objective_count):
    """Mandatory adapter completion validation, independent of publication settings."""
    for name in ('d_loss', 'g_loss', 'd_adversarial', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'lr_scale', 'd_adversarial_weighted', 'g_adversarial_weighted'):
        if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]):
            raise ValueError(f'Invalid finite global metric: {name}')
    objectives = row.get('objectives')
    if (type(objectives) is not list or len(objectives) != objective_count
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in objectives)):
        raise ValueError('Invalid global objective metrics')

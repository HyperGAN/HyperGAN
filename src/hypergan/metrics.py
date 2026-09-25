"""Finite scalar publication and immutable, torch-free metric definitions.

Numerical adapters always validate their complete internal results. Publication
selects already computed values; it never computes objectives or executes plugins.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .run_state import atomic_json
from .metric_plugins import DEFAULT_EVALUATION_EVERY_STEPS, validate_custom, enabled_custom

PRESET_VERSION = 'standard/v2'
PREVIEW_METRICS = {
    'generated_rms': ('Generated sample spread', 'data'),
    'reference_rms': ('Real sample spread', 'data'),
    'ratio': ('Generated/real diversity', 'ratio'),
    'pooled4_generated_rms': ('Generated coarse image spread (4x4)', 'data'),
    'pooled4_reference_rms': ('Real coarse image spread (4x4)', 'data'),
    'pooled4_ratio': ('Generated/real coarse diversity (4x4)', 'ratio'),
}
DEFAULT_METRICS = {'preset': 'standard', 'disable': [], 'every_steps': 1, 'overrides': {}, 'custom': {}}
MAX_CATALOG_BYTES = 1024 * 1024
# Completed updates averaged by the published throughput metric. A short
# trailing window charts cleanly without hiding a sustained slowdown.
THROUGHPUT_WINDOW = 20
SAMPLES_SEEN_DEFINITION = ('Real examples drawn from the data stream: one completed update consumes '
                           'exactly one global batch (training.batch_size), which gradient accumulation '
                           'and a world size larger than one split but never change.')


class Throughput:
    """Trailing-window steps per second over complete update boundaries.

    The window is attempt-local: a resumed attempt restarts it rather than
    averaging across the idle gap. A window whose measured durations sum to
    zero has no rate to report; it publishes nothing instead of dividing.
    """

    def __init__(self, window=THROUGHPUT_WINDOW):
        if type(window) is not int or window < 1:
            raise ValueError('Throughput window must be a positive integer')
        self.durations = deque(maxlen=window)

    def observe(self, step_seconds):
        """Record one update duration; return the smoothed rate, or None."""
        if type(step_seconds) not in (int, float) or not math.isfinite(step_seconds) or step_seconds < 0:
            raise ValueError('Update duration must be a finite, nonnegative number of seconds')
        self.durations.append(float(step_seconds))
        total = math.fsum(self.durations)
        if total <= 0:
            return None
        rate = len(self.durations) / total
        return rate if math.isfinite(rate) else None


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
        'timing/training_seconds': ('training_seconds', 'Time spent training', 'seconds'),
        'throughput/steps_per_second': ('steps_per_second', 'Steps per second', 'steps/second'),
        'progress/samples_seen': ('samples_seen', 'Samples seen', 'samples'),
    }
    for term in config['objectives']:
        name = objective_id(term)
        specs['loss/objectives/' + name] = ('objective:' + name, name + ' contribution', 'loss')
    for name, (label, unit) in PREVIEW_METRICS.items():
        specs['diversity/' + name] = ('preview:' + name, label, unit)
    return specs


def preview_metrics_enabled(config):
    spec = config['metrics']
    return any(spec['overrides'].get('diversity/' + name, {}).get('enabled',
                   spec['preset'] == 'standard') and 'diversity/' + name not in spec['disable']
               for name in PREVIEW_METRICS)


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


def snapshot_metrics(config, trigger=None):
    """Enabled snapshot metric IDs, optionally restricted to a single trigger."""
    return sorted(name for name, spec in enabled_custom(config).items()
                  if spec['mode'] == 'snapshot' and (trigger is None or spec['trigger'] == trigger))


def same_gpu(left, right):
    """Whether two requests may name the same GPU; bare 'cuda' matches any index."""
    if not left.startswith('cuda') or not right.startswith('cuda'):
        return False
    return left == right or 'cuda' in (left, right)


def evaluation_warnings(config):
    """Notice when declared snapshot metrics never run, or share the training GPU."""
    messages = []
    manual = snapshot_metrics(config, 'manual')
    interval = snapshot_metrics(config, 'interval')
    if manual and not interval:
        messages.append(
            'No automatic evaluation is scheduled: snapshot '
            + ('metrics ' if len(manual) > 1 else 'metric ') + ', '.join(manual)
            + ' set trigger = "manual", so this run records an empty evaluation schedule and '
            'publishes no result until `hypergan evaluate` is run explicitly. Omit trigger '
            '(or set trigger = "interval") and name an evaluation device to evaluate every '
            f'{DEFAULT_EVALUATION_EVERY_STEPS} steps by default.')
    custom = config['metrics']['custom']
    shared = [name for name in interval
              if same_gpu(custom[name]['evaluation']['device'], config['training']['device'])]
    if shared:
        devices = sorted({custom[name]['evaluation']['device'] for name in shared})
        messages.append(
            'Interval evaluation of ' + ', '.join(shared) + ' uses evaluation device '
            + ', '.join(devices) + f", which may be the training device {config['training']['device']}: "
            'the evaluator then shares that GPU with training and slows it down. Name a separate '
            'visible GPU to avoid that contention.')
    return messages


def manual_only_snapshot_metrics(config):
    """Every enabled snapshot metric, but only when none of them is scheduled."""
    manual = snapshot_metrics(config, 'manual')
    return manual if manual and not snapshot_metrics(config, 'interval') else []


def manual_evaluation_hint(config, run='RUN', config_path='CONFIG'):
    """One loud line naming the exact edit; printed once after the warnings block."""
    manual = manual_only_snapshot_metrics(config)
    if not manual:
        return None
    rest = f' (and {", ".join(manual[1:])} the same way)' if len(manual) > 1 else ''
    return (f'remove \'trigger = "manual"\' from [metrics.custom.{manual[0]}] to evaluate it every '
            f'{DEFAULT_EVALUATION_EVERY_STEPS} steps{rest}, then apply it with '
            f'`hypergan resume {run} --config {config_path}`')


def manual_evaluation_reminder(config, run='RUN', config_path='CONFIG'):
    """Compact periodic reminder while a run's snapshot metrics are all manual."""
    manual = manual_only_snapshot_metrics(config)
    if not manual:
        return None
    return ('no automatic evaluation is scheduled: ' + ', '.join(manual)
            + ' set trigger = "manual", so no snapshot result is published while this run trains; '
            f'edit the config and apply it with `hypergan resume {run} --config {config_path}`')


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
        if source.startswith('preview:'):
            definition.update(scope='preview', owner='generator', view={'panel': 'Diversity'},
                formula=('generated_rms / reference_rms' if name.endswith('ratio') else
                         'sqrt(2 * mean(unbiased variance across samples))'),
                description='Distinct-pair RMS spread across the bounded EMA preview batch, before image quantization. '
                            + ('NCHW images average-pooled to 4x4. ' if 'pooled4_' in name else '')
                            + 'Real samples come from the last completed local batch (rank zero in distributed runs) without cycling. '
                            'Zero generated spread means identical outputs; ratio one matches real spread. '
                            'Not image quality or semantic coverage: noise can score highly and varying '
                            'conditions can hide ignored latent inputs. Published at preview cadence, '
                            'independent of metrics.every_steps; no value when previews are disabled.')
        if name == 'throughput/steps_per_second':
            definition.update(window=THROUGHPUT_WINDOW, direction='maximize',
                              description=f'Trailing average over the last {THROUGHPUT_WINDOW} complete updates. '
                                          'The window is attempt-local and restarts on resume.')
        if name == 'timing/training_seconds':
            definition['description'] = ('Cumulative wall clock spent inside training attempts. Resume continues '
                                         'the total; time between attempts is not counted.')
        if name == 'progress/samples_seen':
            definition['description'] = SAMPLES_SEEN_DEFINITION
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


def select_metrics(config, catalog, row, step, step_seconds, progress=None):
    """Sample the completed boundary, never average skipped steps or recompute.

    ``progress`` carries controller-measured scalars for this same boundary
    (throughput, cumulative training time, samples seen). A scalar the
    controller cannot measure yet is omitted, never fabricated.
    """
    if step % config['metrics']['every_steps']:
        return {}, {}, 'cadence'
    if progress is not None and (not isinstance(progress, dict)
                                 or any(not isinstance(key, str) for key in progress)):
        raise ValueError('Controller progress scalars must be keyed by name')
    values = dict(row, step_seconds=step_seconds, **(progress or {}))
    if 'd_loss' in row and 'g_loss' in row:
        values['combined'] = row['d_loss'] + row['g_loss']
    for term, value in zip(config['objectives'], row.get('objectives', [])):
        values['objective:' + objective_id(term)] = value
    metrics, statuses = {}, {}
    for name, definition in catalog['metrics'].items():
        source = definition['source']
        if source.startswith(('custom:', 'preview:')):
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
            applied = step % penalty['lazy_k'] == 0
            statuses[name] = {'status': 'available', 'applied': applied,
                              'effective_coefficient': penalty['coeff'] * penalty['lazy_k'] if applied else 0.0}
    return metrics, statuses, 'sampled' if metrics or statuses else 'disabled'


def select_preview_metrics(catalog, record):
    """Publish already computed preview diagnostics at their original source step."""
    diversity = record.get('diversity', {})
    measured, unavailable = diversity.get('metrics', {}), diversity.get('unavailable', {})
    metrics, statuses = {}, {}
    for name, definition in catalog['metrics'].items():
        if not definition['source'].startswith('preview:'):
            continue
        key = definition['source'].split(':', 1)[1]
        if key in measured:
            value = measured[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid finite preview diversity metric: {name}')
            metrics[name] = value
        else:
            statuses[name] = {'status': 'unavailable',
                              'reason': unavailable.get(key, 'Preview did not provide this diagnostic')}
    return metrics, statuses


def validate_update_scalars(row, objective_count):
    """Mandatory adapter completion validation, independent of publication settings."""
    for name in ('d_loss', 'g_loss', 'd_adversarial', 'g_adversarial', 'prior_loss', 'gradient_penalty', 'lr_scale', 'd_adversarial_weighted', 'g_adversarial_weighted'):
        if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]):
            raise ValueError(f'Invalid finite global metric: {name}')
    objectives = row.get('objectives')
    if (type(objectives) is not list or len(objectives) != objective_count
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in objectives)):
        raise ValueError('Invalid global objective metrics')

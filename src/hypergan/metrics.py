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
import sys
import types

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
DEFAULT_METRICS = {'preset': 'standard', 'disable': [], 'every_steps': 10, 'overrides': {}, 'custom': {}}
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


def adversarial_terms_spec(config):
    """Ordered adversarial term descriptors (primary first, then extra adversarial_terms)."""
    primary = config.get('adversarial') or {}
    penalty = config.get('gradient_penalty') or {}
    lazy_k = penalty.get('lazy_k', 1)
    primary_coeff = penalty.get('coeff', 0.0)
    terms = [{
        'id': primary.get('id') or 'adversarial',
        'critic': 'discriminator',
        'weight': primary.get('weight', 1.0),
        'critic_weight': primary.get('critic_weight', primary.get('weight', 1.0)),
        'generator_weight': primary.get('generator_weight', primary.get('weight', 1.0)),
        'penalty': True,
        'penalty_coeff': primary_coeff,
        'lazy_k': lazy_k,
        'primary': True,
    }]
    for spec in config.get('adversarial_terms') or ():
        if not isinstance(spec, dict) or not isinstance(spec.get('id'), str) or not spec['id']:
            continue
        has_penalty = bool(spec.get('penalty', False))
        terms.append({
            'id': spec['id'],
            'critic': spec.get('component', 'discriminator'),
            'weight': spec.get('weight', 1.0),
            'critic_weight': spec.get('critic_weight', spec.get('weight', 1.0)),
            'generator_weight': spec.get('generator_weight', spec.get('weight', 1.0)),
            'penalty': has_penalty,
            'penalty_coeff': spec.get('penalty_coeff', primary_coeff) if has_penalty else 0.0,
            'lazy_k': lazy_k,
            'primary': False,
        })
    return terms


def _available(config):
    adv_terms = adversarial_terms_spec(config)
    primary_id = adv_terms[0]['id']
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
        'critic/real_mean': ('critic/real_mean', f'Critic real score mean ({primary_id})', 'score'),
        'critic/fake_mean': ('critic/fake_mean', f'Critic fake score mean ({primary_id})', 'score'),
        'critic/margin': ('critic/margin', f'Critic score margin ({primary_id}) E[D(x)] - E[D(G(z))]', 'score'),
        'critic/accuracy': ('critic/accuracy', f'Critic ranking accuracy ({primary_id})', 'ratio'),
        'critic/thermal_scale': ('critic/thermal_scale', 'Synchronized critic thermal scale', 'ratio'),
        'grad_norm/critic': ('grad_norm/critic', 'Critic L2 gradient norm', 'norm'),
        'grad_norm/generator': ('grad_norm/generator', 'Generator L2 gradient norm', 'norm'),
        'grad_norm/prior': ('grad_norm/prior', 'Prior L2 gradient norm', 'norm'),
        'optimizer/d_guard_ratio': ('optimizer/d_guard_ratio', 'Discriminator guard ratio', 'ratio'),
        'prior/particle_rms': ('prior/particle_rms', 'Particle table RMS', 'rms'),
    }
    for term in adv_terms:
        tid = term['id']
        specs[f'loss/critics/{tid}/d_adversarial'] = (
            f'loss/critics/{tid}/d_adversarial', f'D adversarial contribution ({tid})', 'loss')
        specs[f'loss/critics/{tid}/gradient_penalty'] = (
            f'loss/critics/{tid}/gradient_penalty', f'D gradient penalty contribution ({tid})', 'loss')
        specs[f'loss/critics/{tid}/d_total'] = (
            f'loss/critics/{tid}/d_total', f'Discriminator total ({tid})', 'loss')
        specs[f'critic/{tid}/real_mean'] = (
            f'critic/{tid}/real_mean', f'Critic real score mean ({tid})', 'score')
        specs[f'critic/{tid}/fake_mean'] = (
            f'critic/{tid}/fake_mean', f'Critic fake score mean ({tid})', 'score')
        specs[f'critic/{tid}/margin'] = (
            f'critic/{tid}/margin', f'Critic score margin ({tid}) E[D(x)] - E[D(G(z))]', 'score')
        specs[f'critic/{tid}/thermal_scale'] = (
            f'critic/{tid}/thermal_scale', f'Effective critic thermal scale ({tid})', 'ratio')
        specs[f'critic/{tid}/raw_thermal_scale'] = (
            f'critic/{tid}/raw_thermal_scale', f'Pre-coupling critic thermal scale ({tid})', 'ratio')
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


def normalize_device(value):
    """Canonical 'cpu', 'cuda' or 'cuda:N' for one device: N, gpuN, cuda[:N] or cpu."""
    if type(value) is int and value >= 0:
        return f'cuda:{value}'
    text = value.strip().lower() if isinstance(value, str) else ''
    match = re.fullmatch(r'(?:cuda:|gpu)?([0-9]+)', text)
    if match:
        return f'cuda:{int(match.group(1))}'
    if text in ('cpu', 'cuda'):
        return text
    raise ValueError(f'Expected one device (cpu, cuda, cuda:N, N or gpuN), got {value!r}')


def _device_set(value):
    items = value.split(',') if isinstance(value, str) else value if isinstance(value, (list, tuple)) else [value]
    return {normalize_device(item) for item in items if not isinstance(item, str) or item.strip()}


def same_gpu(left, right):
    """Whether two device requests may name the same GPU.

    The one collision rule for config warnings and ``hypergan evaluate``. Bare
    'cuda' matches any index: replicated training expands it to every visible
    GPU. Either side may list several devices ('0,1'); cpu never collides.
    """
    left, right = ({device for device in _device_set(side) if device != 'cpu'} for side in (left, right))
    return any(a == b or 'cuda' in (a, b) for a in left for b in right)


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
            'publishes no result until `hypergan evaluate` is run explicitly. Set `every = '
            f'{DEFAULT_EVALUATION_EVERY_STEPS}` in [evaluations.<id>] (or omit trigger / set '
            'trigger = "interval" in legacy [metrics.custom.<id>]) and name an evaluation device '
            f'to evaluate every {DEFAULT_EVALUATION_EVERY_STEPS} steps by default.')
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
    return (f'set \'every = {DEFAULT_EVALUATION_EVERY_STEPS}\' in [evaluations.{manual[0]}] '
            f'(or remove \'trigger = "manual"\' from [metrics.custom.{manual[0]}]) to evaluate it every '
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
    adv_terms = adversarial_terms_spec(config)
    adv_by_id = {term['id']: term for term in adv_terms}
    primary_term = adv_terms[0]
    for name, (source, label, unit) in _available(config).items():
        enabled = spec['overrides'].get(name, {}).get('enabled', spec['preset'] == 'standard')
        if not enabled or name in spec['disable']:
            continue
        definition = {'kind': 'scalar', 'source': source, 'label': label, 'unit': unit,
                      'numerical_sha256': fingerprint(config), 'preset_version': PRESET_VERSION,
                      'direction': 'none', 'scope': 'complete_update',
                      'owner': 'discriminator' if name.startswith(('loss/d_', 'loss/gradient', 'loss/critics/', 'critic/', 'optimizer/d_guard_ratio', 'grad_norm/critic')) else
                               'generator' if name.startswith(('loss/g_', 'loss/prior', 'loss/objectives/', 'grad_norm/generator', 'grad_norm/prior', 'prior/particle_rms')) else 'run',
                      'view': {'panel': 'Loss' if name.startswith('loss/') else 'Diagnostics' if name.startswith(('critic/', 'grad_norm/')) else 'Progress'}}
        if name == 'loss/total':
            definition['formula'] = 'loss/d_total + loss/g_total'
            definition['description'] = 'Diagnostic sum; not a joint optimization objective or quality score.'
        if name in ('critic/real_mean', 'critic/fake_mean', 'critic/margin', 'critic/accuracy'):
            definition.update(term_id=primary_term['id'], critic=primary_term['critic'])
            if name == 'critic/margin':
                definition['formula'] = 'critic/real_mean - critic/fake_mean'
        if name == 'critic/thermal_scale':
            definition['description'] = (
                'Effective ParticleGAN thermal scale applied across critics '
                '(under multi-critic training, conservatively synchronized as max_i s_i so relative weights do not drift).'
            )
        if name.startswith('loss/critics/'):
            parts = name.split('/')
            tid, metric_kind = parts[2], parts[3]
            term = adv_by_id[tid]
            definition.update(term_id=tid, critic=term['critic'])
            if metric_kind == 'd_adversarial':
                definition['coefficient'] = term['critic_weight']
                definition['description'] = f"Weighted discriminator adversarial contribution for term '{tid}' on critic '{term['critic']}'."
            elif metric_kind == 'gradient_penalty':
                definition.update(
                    coefficient=term['penalty_coeff'],
                    lazy_k=term['lazy_k'],
                    penalty_enabled=term['penalty'],
                    raw_available=False,
                    description=f"Applied discriminator gradient penalty contribution for term '{tid}' on critic '{term['critic']}'.",
                )
            elif metric_kind == 'd_total':
                definition['formula'] = f'loss/critics/{tid}/d_adversarial + loss/critics/{tid}/gradient_penalty'
                definition['description'] = f"Total discriminator loss contribution for term '{tid}' on critic '{term['critic']}'."
        elif name.startswith('critic/') and name.count('/') == 2:
            parts = name.split('/')
            tid, metric_kind = parts[1], parts[2]
            term = adv_by_id[tid]
            definition.update(term_id=tid, critic=term['critic'])
            if metric_kind == 'margin':
                definition['formula'] = f'critic/{tid}/real_mean - critic/{tid}/fake_mean'
                definition['description'] = f"Mean real minus fake critic score margin for term '{tid}' on critic '{term['critic']}'."
            elif metric_kind == 'real_mean':
                definition['description'] = f"Mean real critic score from the training forward pass for term '{tid}' on critic '{term['critic']}'."
            elif metric_kind == 'fake_mean':
                definition['description'] = f"Mean fake critic score from the training forward pass for term '{tid}' on critic '{term['critic']}'."
            elif metric_kind == 'thermal_scale':
                definition['description'] = (
                    f"Effective synchronized thermal scale applied to term '{tid}' on critic '{term['critic']}'."
                )
            elif metric_kind == 'raw_thermal_scale':
                definition['description'] = (
                    f"Pre-coupling per-critic candidate thermal scale observed for term '{tid}' on critic '{term['critic']}'."
                )
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
        if name in ('loss/d_adversarial', 'loss/g_adversarial', 'loss/d_adversarial_raw', 'loss/g_adversarial_raw'):
            phase_weight = 'critic_weight' if name.startswith('loss/d_') else 'generator_weight'
            terms = [config['adversarial'], *config.get('adversarial_terms', [])]
            weights = [term.get(phase_weight, term['weight']) for term in terms]
            if len(set(weights)) == 1:
                definition['coefficient'] = weights[0]
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
    reserved = _available(config).keys() | spec['custom'].keys()
    for name, custom in enabled_custom(config).items():
        runtime = config.get('_metric_runtime', {}).get(name)
        if runtime is None:
            raise ValueError('Custom metric catalogs require bounded runtime preflight')
        descriptor = runtime['descriptor']
        outputs = descriptor.get('outputs')
        for key, presentation in (outputs or {name: descriptor}).items():
            metric_id = f'{name}/{key}' if outputs else name
            if outputs and (len(metric_id) > 128 or metric_id in reserved or metric_id in metrics):
                raise ValueError(f'Duplicate or overlong snapshot output ID: {metric_id}')
            definition = dict(presentation, source='custom:' + name, factory=custom['factory'],
                              factory_sources=runtime['factory_sources'], protocol=runtime['protocol'],
                              specification=deepcopy(custom), numerical_sha256=fingerprint(config),
                              scope='snapshot' if custom['mode'] == 'snapshot' else 'complete_update')
            if outputs:
                definition.update(evaluator_id=name, output_id=key)
            metrics[metric_id] = dict(definition, definition_hash=digest(definition))
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
    has_train_metrics = any(
        not d['source'].startswith(('custom:', 'preview:'))
        for d in catalog['metrics'].values()
    )
    if not has_train_metrics:
        return {}, {}, 'disabled'
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
    for alt, primary in (
        ('critic_real_mean', 'critic/real_mean'),
        ('critic_fake_mean', 'critic/fake_mean'),
        ('critic_margin', 'critic/margin'),
        ('critic_accuracy', 'critic/accuracy'),
        ('grad_norm_critic', 'grad_norm/critic'),
        ('grad_norm_generator', 'grad_norm/generator'),
        ('grad_norm_prior', 'grad_norm/prior'),
        ('d_guard_ratio', 'optimizer/d_guard_ratio'),
        ('particle_rms', 'prior/particle_rms'),
        ('diversity_ratio', 'diversity/ratio'),
    ):
        if alt in values and primary not in values:
            values[primary] = values[alt]
        elif primary in values and alt not in values:
            values[alt] = values[primary]
    adv_by_id = {term['id']: term for term in adversarial_terms_spec(config)}
    metrics, statuses = {}, {}
    for name, definition in catalog['metrics'].items():
        source = definition['source']
        if source.startswith(('custom:', 'preview:')):
            if name == 'diversity/ratio' and 'diversity/ratio' in values:
                value = values['diversity/ratio']
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'Nonfinite or invalid metric value: {name}')
                metrics[name] = float(value)
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
        elif name.startswith('loss/critics/') and name.endswith('/gradient_penalty'):
            tid = name.split('/')[2]
            term = adv_by_id.get(tid)
            if term is not None:
                applied = bool(term['penalty']) and (step % term['lazy_k'] == 0)
                statuses[name] = {'status': 'available', 'applied': applied,
                                  'effective_coefficient': term['penalty_coeff'] * term['lazy_k'] if applied else 0.0}
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
    for name in (
        'critic_real_mean', 'critic_fake_mean', 'critic_margin', 'critic_accuracy',
        'grad_norm_critic', 'grad_norm_generator', 'grad_norm_prior',
        'd_guard_ratio', 'particle_rms', 'diversity_ratio',
        'critic/real_mean', 'critic/fake_mean', 'critic/margin', 'critic/accuracy',
        'grad_norm/critic', 'grad_norm/generator', 'grad_norm/prior',
        'optimizer/d_guard_ratio', 'prior/particle_rms', 'diversity/ratio',
    ):
        if name in row and (type(row[name]) not in (int, float) or not math.isfinite(row[name])):
            raise ValueError(f'Invalid finite global metric: {name}')
    for name, value in row.items():
        if isinstance(name, str) and name.startswith(('loss/critics/', 'critic/')):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'Invalid finite global metric: {name}')
    objectives = row.get('objectives')
    if (type(objectives) is not list or len(objectives) != objective_count
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in objectives)):
        raise ValueError('Invalid global objective metrics')


def compare(runs, metric="fid50k_train", at_steps=None):
    """Compare multiple runs for a given metric across step milestones."""
    from .inspection import compare_runs
    return compare_runs(runs, metric=metric, at_steps=at_steps)


def get_metric(run, name, *, latest=False):
    """Retrieve scalar time series or latest value for a specific metric from a run."""
    from .api import metrics as api_metrics
    series = api_metrics(run, names=[name]).get(name, [])
    if latest:
        return series[-1][1] if series else None
    return series


class _MetricsModule(types.ModuleType):
    """Callable module wrapper allowing hg.metrics(run) and module attribute access."""

    def __call__(self, *args, **kwargs):
        """Published scalar metrics: ``{name: [(step, value), ...]}`` in step order."""
        from .api import metrics as api_metrics
        return api_metrics(*args, **kwargs)

    def __repr__(self):
        return f"<module '{self.__name__}' (callable metrics)>"


# Replace module class in sys.modules so hypergan.metrics is callable as a function
_self = sys.modules[__name__]
_self.__class__ = _MetricsModule
sys.modules[__name__] = _self

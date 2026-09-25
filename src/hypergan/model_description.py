"""Describe a run's model for people: formulation, losses, optimizers and networks.

Two tiers, split on whether torch is needed:

* ``describe_run`` (torch-free) reads the resolved configuration recorded in
  ``manifest.json`` and returns the formulation, per-player loss terms with their
  metric series ids, optimizers and schedule, the prior, data/training settings,
  one entry per component with its role and HNDL source, and binding edges. The
  viewer calls it per request (a few milliseconds, cached by configuration).
* ``record_networks`` / ``build_networks`` (torch and hndl) produce the per-layer
  network detail: op, shapes, source line and parameter counts per HNDL node.
  Training records it as ``<run>/model.json`` right after constructing the
  component graph; ``hypergan model RUN --write`` backfills older runs. The
  viewer only reads that file, so it never imports torch.

Every absolute or home-relative local path is shown as ``…/basename``.
"""
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import time

SCHEMA_VERSION = 1
MODEL_FILE = 'model.json'
MAX_MODEL_BYTES = 4 * 1048576   # recorded network detail read by the viewer
MAX_SOURCE_CHARS = 65_536        # hndl's own parser limit; longer text is truncated for display
MAX_NODES = 512                  # per subgraph; the largest seen is 226
MAX_ARG_CHARS = 200              # per displayed argument value
MAX_SUBGRAPHS = 64

# ParticleGAN 0.8 has exactly one formulation (config.py refuses anything else);
# the text follows ParticleGAN docs/k3p.md.
K3P_EQUATIONS = {
    'd_adversarial': 'softplus(−(D(x_real) − D(x_fake)))',
    'g_adversarial': 'softplus(−(D(x_fake) − D(x_real)))',
    'A': 'mean(‖∇D(real)‖²/d) + mean(relu(‖∇D(fake)‖/√d − κ)²)',
    'B': 'mean(relu(‖∇D(real)‖ − κ)²) + mean(relu(‖∇D(fake)‖ − κ)²)',
    'P': 'mean(‖∇D(real) − ∇D̄(real)‖²/d),  D̄ = EMA critic (anchor_decay)',
    's': 'max(0, min(1, 2r) − 2f)/(1 − 2f),  r = critic LR / max critic LR, f = network LR floor',
    'penalty': 'c/2 · (s·A + (1 − s)·(B + P)), applied every lazy_k steps with coefficient lazy_k·c',
    'd_total': 'Σ terms weight·d_adversarial + Σ penalized critics penalty',
    'g_total': 'Σ terms weight·g_adversarial + prior regularizer + Σ objectives weight·objective',
}
LEGACY_PENALTY_KEYS = ('arm', 'norm', 'target_anneal', 'total_steps', 'method', 'fd_eps')
# ParticleGAN 0.8 Recipe defaults (docs/k3p.md): infer `defaults = "particlegan"` on
# manifests that do not record it. A tuned run reports "unknown".
PG08_DEFAULTS = {('optimizer', 'lr'): 0.00425, ('optimizer', 'd_lr_mult'): 1.0,
                 ('optimizer', 'prior_lr_mult'): 2.0, ('training', 'network_lr_horizon_cap'): 1600,
                 ('training', 'network_lr_floor'): 0.01, ('training', 'input_noise_std'): 0.5,
                 ('training', 'output_noise_std'): 0.029}


# ---------------------------------------------------------------- redaction

def _is_local_path(text):
    return isinstance(text, str) and (text.startswith('/') or text.startswith('~/')
                                      or re.match(r'^[A-Za-z]:[\\/]', text) is not None)


def redact(value):
    """Absolute local paths become '…/basename'; relative paths and digests are kept."""
    if isinstance(value, Mapping):
        return {str(k): redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if _is_local_path(value):
        return '…/' + re.split(r'[\\/]', value.rstrip('/\\'))[-1]
    return value


_QUOTED_PATH = re.compile(r"""(["'])((?:/|~/)[^"'\n]*)\1""")
_BARE_PATH = re.compile(r'(?<![\w.~/…])(?:~/|/)(?:[\w.\-]+/)+([\w.\-]+)')


def _redact_source(text):
    """Quoted absolute paths in HNDL literals."""
    return _QUOTED_PATH.sub(lambda m: m.group(1) + '…/' + os.path.basename(m.group(2).rstrip('/')) + m.group(1), text)


def redact_text(text):
    """Diagnostics: quoted and bare absolute paths."""
    return _BARE_PATH.sub(lambda m: '…/' + m.group(1), _redact_source(str(text)))


# ---------------------------------------------------------------- helpers

def _plain(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    return repr(value)


def _bounded(value, limit=MAX_ARG_CHARS):
    try:
        value = redact(json.loads(json.dumps(value, default=_plain)))
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return {'truncated': redact_text(repr(value))[:limit]}
    return {'truncated': text[:limit]} if len(text) > limit else value


def unavailable(reason):
    return {'status': 'unavailable', 'reason': reason}


def _section(config, name):
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _product(a, b):
    return a * b if _number(a) and _number(b) else None


def _sections(text):
    """Comment paragraphs: a '#' line at the top or after a blank line opens a section."""
    lines = text.split('\n')
    return [{'title': line.strip().lstrip('#').strip()[:160], 'line': i}
            for i, line in enumerate(lines, 1)
            if line.strip().startswith('#') and (i == 1 or not lines[i - 2].strip())
            and line.strip().lstrip('#').strip()]


def _normalized(text):
    return text.replace('\r\n', '\n').replace('\r', '\n').strip()


_KNOWN_SOURCES = None


def _known_sources():
    """Digest -> display name of packaged (and, in a checkout, example) network files."""
    global _KNOWN_SOURCES
    if _KNOWN_SOURCES is None:
        found = {}
        package = Path(__file__).with_name('networks')
        examples = Path(__file__).resolve().parents[2] / 'examples' / 'networks'
        for directory, label in ((examples, 'examples/networks/'), (package, 'hypergan/networks/')):
            try:
                paths = sorted(directory.glob('*.hndl'))
            except OSError:
                continue
            for path in paths:
                try:
                    if path.stat().st_size > 1048576:
                        continue
                    text = _normalized(path.read_text(encoding='utf-8'))
                except (OSError, UnicodeDecodeError):
                    continue
                found[hashlib.sha256(text.encode()).hexdigest()] = label + path.name
        _KNOWN_SOURCES = found
    return _KNOWN_SOURCES


def _source_view(text, parameters=None, kind='inline'):
    shown = _redact_source(text[:MAX_SOURCE_CHARS])
    view = {'kind': kind, 'text': shown, 'truncated': len(text) > MAX_SOURCE_CHARS,
            'lines': text.count('\n') + 1, 'sections': _sections(shown),
            'file': _known_sources().get(hashlib.sha256(_normalized(text).encode()).hexdigest())}
    if parameters:
        view['parameters'] = redact(parameters)
    return view


def _objective_id(term):
    from .metrics import objective_id
    return objective_id(term)


def _producer(path):
    """Which graph element a binding path reads from."""
    head, _, rest = str(path).partition('.')
    if head == 'components':
        return 'components.' + rest.split('.')[0]
    if head in ('latent', 'prior'):
        return 'prior'
    if head == 'batch':
        return 'data'
    if head == 'generated':
        return 'components.generator'
    return head


# ---------------------------------------------------------------- tier A

def roles(config):
    components = config['components']
    critics = {'discriminator'} | {t.get('component') for t in config.get('adversarial_terms') or ()}
    result = {}
    for name, spec in components.items():
        inputs = list((spec.get('inputs') or {}).values())
        if 'reuse' in spec:
            result[name] = 'alias'
        elif name == 'generator':
            result[name] = 'generator'
        elif name in critics:
            result[name] = 'critic'
        elif any(path in ('prior.means', 'prior.sigma') for path in inputs):
            result[name] = 'encoder'
        else:
            result[name] = 'auxiliary'
    return result


def _edges(config):
    edges = []
    for name, spec in config['components'].items():
        for port, path in (spec.get('inputs') or {}).items():
            edges.append({'from': _producer(path), 'to': 'components.' + name, 'port': port, 'path': path})
        if 'reuse' in spec:
            edges.append({'from': 'components.' + spec['reuse'], 'to': 'components.' + name,
                          'port': 'weights', 'path': 'reuse'})
    for term in config.get('objectives') or ():
        oid = _objective_id(term)
        for port, path in (term.get('inputs') or {}).items():
            edges.append({'from': _producer(path), 'to': 'objective.' + oid, 'port': port, 'path': path,
                          'detached': port in (term.get('detach') or ())})
    return edges


def _particlegan_version(manifest):
    return ((manifest.get('source') or {}).get('particlegan_distribution_version')
            or (manifest.get('runtime') or {}).get('particlegan'))


def _formulation(config, manifest):
    """K3P for ParticleGAN 0.8 configurations; recorded fields shown verbatim otherwise."""
    penalty, adversarial = _section(config, 'gradient_penalty'), _section(config, 'adversarial')
    training = _section(config, 'training')
    version = _particlegan_version(manifest)
    legacy = any(k in penalty for k in LEGACY_PENALTY_KEYS) or any(k in adversarial for k in ('loss_type', 'mode'))
    raw = {'adversarial': redact(adversarial), 'gradient_penalty': redact(penalty)}
    floor = training.get('network_lr_floor')
    if not legacy:
        return {'family': 'k3p', 'name': 'K3P', 'loss': 'RpGAN logistic (relativistic paired)',
                'particlegan': version or '≥0.8 (configuration schema)', 'equations': dict(K3P_EQUATIONS),
                'parameters': {'coeff': penalty.get('coeff'), 'kappa': penalty.get('kappa'),
                               'lazy_k': penalty.get('lazy_k'), 'anchor_weight': penalty.get('anchor_weight'),
                               'anchor_decay': penalty.get('anchor_decay'),
                               'blend_floor_f': floor if floor is not None else training.get('lr_floor'),
                               'adversarial_weight': adversarial.get('weight')},
                'raw': raw}
    loss, mode = adversarial.get('loss_type', 'logistic'), adversarial.get('mode', 'rp')
    equations = ({k: K3P_EQUATIONS[k] for k in ('d_adversarial', 'g_adversarial', 'd_total', 'g_total')}
                 if (loss, mode) == ('logistic', 'rp') else {})
    return {'family': 'legacy', 'name': f"pre-K3P penalty (arm={penalty.get('arm')}, norm={penalty.get('norm')})",
            'loss': f'RpGAN {loss} ({mode})', 'particlegan': version or '<0.8 (configuration schema)',
            'equations': equations,
            'parameters': {**{k: penalty.get(k) for k in ('coeff', 'kappa', 'lazy_k', *LEGACY_PENALTY_KEYS) if k in penalty},
                           'adversarial_weight': adversarial.get('weight')},
            'note': 'Recorded by a ParticleGAN release before 0.8. The penalty is shown by its recorded fields, '
                    'not re-derived; current HyperGAN refuses these fields, so this run cannot resume as-is.',
            'raw': raw}


def _adversarial_terms(config):
    penalty, adversarial = _section(config, 'gradient_penalty'), _section(config, 'adversarial')
    discriminator = config['components'].get('discriminator') or {}
    terms = [{'id': 'adversarial', 'critic': 'discriminator', 'weight': adversarial.get('weight', 1.0),
              'real': 'batch.real', 'fake': 'generated', 'inputs': discriminator.get('inputs') or {},
              'penalty_coeff': penalty.get('coeff'), 'implicit': True}]
    for term in config.get('adversarial_terms') or ():
        terms.append({'id': term.get('id'), 'critic': term.get('component'), 'weight': term.get('weight', 1.0),
                      'real': term.get('real'), 'fake': term.get('fake'), 'inputs': term.get('inputs') or {},
                      'penalty_coeff': term.get('penalty_coeff', penalty.get('coeff')) if term.get('penalty') else None,
                      'implicit': False})
    return terms


def _losses(config, family):
    terms = _adversarial_terms(config)
    several = len(terms) > 1
    penalty = _section(config, 'gradient_penalty')
    penalty_kind = 'k3p_penalty' if family == 'k3p' else 'penalty_' + str(penalty.get('arm', 'legacy'))
    penalty_fields = (('kappa', 'lazy_k', 'anchor_weight', 'anchor_decay') if family == 'k3p'
                      else ('kappa', 'lazy_k', 'arm', 'norm'))

    def series(metric):
        if several:
            return {'metric': None, 'metric_note': f'every term is summed in {metric}; no per-term series'}
        return {'metric': metric, 'metric_note': None}

    discriminator, generator = [], []
    for term in terms:
        discriminator.append({'id': term['id'], 'kind': 'rpgan_d', 'critic': term['critic'], 'weight': term['weight'],
                              'real': term['real'], 'fake': term['fake'], 'detach': ['fake sample'],
                              **series('loss/d_adversarial')})
        if term['penalty_coeff'] is not None:
            discriminator.append({'id': f"{term['id']}:penalty", 'kind': penalty_kind, 'critic': term['critic'],
                                  'coeff': term['penalty_coeff'], **{k: penalty.get(k) for k in penalty_fields},
                                  **series('loss/gradient_penalty')})
        generator.append({'id': term['id'], 'kind': 'rpgan_g', 'critic': term['critic'], 'weight': term['weight'],
                          'real': term['real'], 'fake': term['fake'], 'detach': ['real score'],
                          **series('loss/g_adversarial')})
    spread = _section(config, 'prior_regularizer')
    if spread and _section(config, 'prior').get('kind') != 'gaussian':
        generator.append({'id': 'prior_regularizer', 'kind': 'particle_spread', 'weight': spread.get('weight'),
                          'target_std': spread.get('target_std'), 'eps': spread.get('eps'), 'rows': spread.get('rows'),
                          'active': _number(spread.get('weight')) and spread['weight'] > 0,
                          'metric': 'loss/prior_regularizer', 'metric_note': None})
    for term in config.get('objectives') or ():
        oid = _objective_id(term)
        generator.append({'id': oid, 'kind': term.get('factory'), 'weight': term.get('weight'),
                          'inputs': term.get('inputs') or {}, 'detach': list(term.get('detach') or ()),
                          'args': redact(term.get('args') or {}), 'metric': 'loss/objectives/' + oid,
                          'metric_note': None})
    return {'discriminator': discriminator, 'generator': generator,
            'totals': {'discriminator': 'loss/d_total', 'generator': 'loss/g_total', 'combined': 'loss/total'},
            'raw': {'d_adversarial': 'loss/d_adversarial_raw', 'g_adversarial': 'loss/g_adversarial_raw'}}


def _prior(config):
    prior = _section(config, 'prior')
    args = dict(prior.get('args') or {})
    kind = prior.get('kind')
    result = {'kind': kind, 'z_dim': args.get('z_dim'), 'num_particles': args.get('num_particles'),
              'learnable': args.get('learnable', kind != 'gaussian'),
              'initialization': {'device': prior.get('initialization_device'), 'seed': prior.get('initialization_seed')},
              'args': redact(args)}
    if kind == 'mog':
        if prior.get('fixed_sigma') is not None:
            result['sigma'] = {'mode': 'fixed', 'value': prior['fixed_sigma']}
        elif args.get('sigma') is not None:
            result['sigma'] = {'mode': 'explicit', 'value': args['sigma']}
        else:
            result['sigma'] = {'mode': 'calibrated', 'value': None, 'sigma_rel': args.get('sigma_rel', 0.025),
                               'reason': 'calibrated at run start from sigma_rel; the value is not in the configuration'}
    return result


def _optimizers(config, family):
    optimizer, training = _section(config, 'optimizer'), _section(config, 'training')
    critics = list(dict.fromkeys(['discriminator'] + [t.get('component') for t in config.get('adversarial_terms') or ()]))
    generator_groups = [name for name, role in roles(config).items() if role not in ('critic', 'alias')]
    k3p = family == 'k3p'
    return {
        'generator': {'type': 'K3PGeneratorAdam' if k3p else 'Adam', 'implementation': optimizer.get('implementation'),
                      'lr': optimizer.get('lr'), 'betas': optimizer.get('betas'), 'components': generator_groups,
                      'latent_damping_max_rate': optimizer.get('latent_damping_max_rate')},
        'prior': {'lr': _product(optimizer.get('lr'), optimizer.get('prior_lr_mult')),
                  'lr_mult': optimizer.get('prior_lr_mult'), 'betas': optimizer.get('prior_betas')},
        'critic': {'type': 'K3PCriticAdam' if k3p else 'Adam', 'implementation': optimizer.get('implementation'),
                   'lr': _product(optimizer.get('lr'), optimizer.get('d_lr_mult')), 'lr_mult': optimizer.get('d_lr_mult'),
                   'betas': optimizer.get('betas'), 'components': critics,
                   'ema_critic': bool(_section(config, 'gradient_penalty').get('anchor_weight')),
                   'guard': {'ratio': optimizer.get('d_guard_ratio'), 'min_steps': optimizer.get('d_guard_min_steps')}},
        'schedule': {'lr_anneal_start': training.get('lr_anneal_start'), 'lr_floor': training.get('lr_floor'),
                     'network_lr_floor': training.get('network_lr_floor'),
                     'network_lr_horizon_cap': training.get('network_lr_horizon_cap'), 'metric': 'optimizer/lr_scale'},
        'ema': training.get('ema'),
        'noise': {'critic_input_std': training.get('input_noise_std'),
                  'critic_input_anneal_end': training.get('input_noise_anneal_end'),
                  'generator_output_std': training.get('output_noise_std'),
                  'generator_output_warmup': training.get('output_noise_warmup')},
        'raw': redact(optimizer),
    }


def _defaults_source(config, manifest):
    recorded = manifest.get('defaults')
    if recorded in ('hypergan', 'particlegan'):
        return recorded
    if all(_section(config, s).get(k) == v for (s, k), v in PG08_DEFAULTS.items()):
        return 'particlegan (inferred)'
    return 'unknown'


def _network(name, spec, role):
    group = ('critic' if role == 'critic' else 'shared with ' + str(spec['reuse']) if 'reuse' in spec
             else 'generator')
    entry = {'name': name, 'role': role, 'optimizer_group': group, 'factory': spec.get('factory', 'reuse'),
             'inputs': spec.get('inputs') or {}, 'trainable': spec.get('trainable', True),
             'freeze_parameters': spec.get('freeze_parameters', False)}
    args = spec.get('args') if isinstance(spec.get('args'), dict) else {}
    if 'reuse' in spec:
        entry['reuse_of'] = spec['reuse']
        entry['graph'] = unavailable(f"shares the weights of {spec['reuse']}")
        return entry
    if spec.get('factory') == 'hndl':
        entry['input_shape'] = args.get('input_shape')
        entry['output_shape'] = args.get('output_shape')
        entry['source'] = (_source_view(args['source'], args.get('parameters')) if isinstance(args.get('source'), str)
                           else unavailable('no inline HNDL source recorded'))
        if args.get('pretrained_providers'):
            entry['pretrained_providers'] = redact(args['pretrained_providers'])
    else:
        entry['args'] = _bounded({k: v for k, v in args.items() if k != 'networks'}, 4096)
        templates = args.get('networks') if isinstance(args.get('networks'), dict) else {}
        entry['templates'] = [dict(name=t, **_source_view(text, kind='template'))
                              for t, text in templates.items() if isinstance(text, str)]
        if not entry['templates']:
            entry['source'] = unavailable('Python constructor with no HNDL templates')
    entry['graph'] = unavailable('network detail not recorded for this run; '
                                 'run `hypergan model RUN --write` to record it')
    return entry


def describe_config(config, manifest=None):
    """Tier A. Pure dict work, defensive against every recorded configuration schema."""
    manifest = manifest or {}
    if not isinstance(config, dict) or not isinstance(config.get('components'), dict):
        raise FileNotFoundError('No model configuration recorded for this run')
    source = manifest.get('source') or {}
    runtime = manifest.get('runtime') or {}
    formulation = _formulation(config, manifest)
    component_roles = roles(config)
    data, metrics = _section(config, 'data'), _section(config, 'metrics')
    return {
        'schema_version': SCHEMA_VERSION,
        'run_id': manifest.get('run_id'),
        'name': config.get('name'),
        'config_sha256': manifest.get('config_sha256'),
        'provenance': {'hypergan_commit': source.get('hypergan_commit') or source.get('commit'),
                       'dirty': source.get('hypergan_dirty', source.get('dirty')),
                       'particlegan': formulation['particlegan'], 'hndl': runtime.get('hndl'),
                       'torch': runtime.get('torch'), 'defaults_source': _defaults_source(config, manifest)},
        'formulation': formulation,
        'prior': _prior(config),
        'losses': _losses(config, formulation['family']),
        'optimizers': _optimizers(config, formulation['family']),
        'networks': [_network(n, s, component_roles[n]) for n, s in config['components'].items()],
        'edges': _edges(config),
        'data': {'factory': data.get('factory'), 'args': _bounded(data.get('args') or {}, 4096)},
        'training': redact(_section(config, 'training')),
        'sampling': redact(_section(config, 'sampling')),
        'metrics': {'preset': metrics.get('preset'), 'custom': sorted(metrics.get('custom') or {})},
        'run': {'global_batch_size': manifest.get('global_batch_size'),
                'attempt_index': manifest.get('attempt_index'), 'total_steps': manifest.get('total_steps')},
        'warnings': [redact_text(w)[:500] for w in (manifest.get('warnings') or [])[:32]],
    }


def _merge_recorded(result, recorded, config_sha256):
    """Attach recorded network detail when it belongs to this exact configuration."""
    if not isinstance(recorded, dict):
        return result
    if recorded.get('config_sha256') != config_sha256:
        reason = 'recorded network detail belongs to a different configuration; run `hypergan model RUN --write`'
        for entry in result['networks']:
            if 'reuse_of' not in entry:
                entry['graph'] = unavailable(reason)
        return result
    components = recorded.get('components') if isinstance(recorded.get('components'), dict) else {}
    for entry in result['networks']:
        graph = components.get(entry['name'])
        if 'reuse_of' in entry or not isinstance(graph, dict):
            continue
        entry['graph'] = _sanitize_graph(graph, origin=recorded.get('origin', 'recorded'))
    result['networks_recorded'] = {'origin': recorded.get('origin', 'recorded'), 'hndl': recorded.get('hndl'),
                                   'seconds': recorded.get('seconds')}
    return result


def _sanitize_graph(graph, *, origin):
    """Recorded files are run-owned data: keep known fields, bounded and redacted."""
    allowed = ('status', 'reason', 'error', 'parameters', 'seconds', 'provider_warning')
    result = {k: redact(graph[k]) for k in allowed if k in graph}
    result['origin'] = origin
    result['status'] = str(result.get('status', 'unavailable'))[:32]
    subgraphs = []
    for sub in (graph.get('subgraphs') or [])[:MAX_SUBGRAPHS]:
        if not isinstance(sub, dict):
            continue
        nodes = [n for n in (sub.get('nodes') or [])[:MAX_NODES] if isinstance(n, dict)]
        subgraphs.append({**{k: redact(sub[k]) for k in ('module_path', 'node_count', 'nodes_truncated',
                                                          'semantic_digest', 'input_shape', 'output_shape',
                                                          'parameters') if k in sub},
                          'nodes': [{k: redact(v) for k, v in n.items()} for n in nodes]})
    result['subgraphs'] = subgraphs
    return result


def describe_run(manifest, *, recorded=None, catalog=None):
    """The viewer's /model document from a run manifest and optional recorded detail."""
    if not isinstance(manifest, dict):
        raise FileNotFoundError('No model configuration recorded for this run')
    started = time.perf_counter()
    result = describe_config(manifest.get('config'), manifest)
    _merge_recorded(result, recorded, manifest.get('config_sha256'))
    if isinstance(catalog, dict) and isinstance(catalog.get('metrics'), dict):
        available = catalog['metrics']
        for side in ('discriminator', 'generator'):
            for term in result['losses'][side]:
                if term.get('metric') and term['metric'] not in available:
                    term['metric_note'] = 'not in this run\'s metric catalog'
    result['seconds'] = round(time.perf_counter() - started, 4)
    return result


def load(source):
    """A resolved configuration, a manifest dict, a run directory, manifest.json or a .toml."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / 'manifest.json'
        if path.suffix == '.json':
            with path.open('rb') as stream:
                data = stream.read(8 * 1048576 + 1)
            if len(data) > 8 * 1048576:
                raise ValueError('manifest exceeds 8 MiB')
            manifest = json.loads(data)
            return manifest, manifest.get('config') if isinstance(manifest, dict) else None
        from .config import load_config
        return {}, load_config(path)
    if isinstance(source, dict) and 'config' in source and 'components' not in source:
        return source, source['config']
    return {}, source


# ---------------------------------------------------------------- tier B (torch + hndl)

def _op_meta(registry, op):
    try:
        spec = registry.by_identity(op)
        return spec.alias, spec.category
    except Exception:
        return str(op).split('@')[0], None


def _node_row(registry, node, params=None, trainable_params=None, shapes=True):
    alias, category = _op_meta(registry, node.op)
    source = dict(getattr(node, 'source', None) or {})
    origins = dict(source.get('argument_origins') or {})
    row = {'id': node.id, 'op': alias, 'category': category,
           'args': {str(k): _bounded(v) for k, v in dict(node.args).items()},
           'explicit': sorted(k for k, origin in origins.items() if origin == 'explicit'),
           'inputs': {str(k): str(v) for k, v in dict(node.inputs).items()}, 'line': source.get('line')}
    if shapes and getattr(node, 'output_shapes', None) is not None:
        row['in'] = {k: list(v) for k, v in dict(node.input_shapes).items()}
        row['out'] = {k: list(v) for k, v in dict(node.output_shapes).items()}
        row['trainable'] = dict(getattr(node, 'trainability', None) or {}).get('default')
    if params is not None:
        row['params'] = params
    if trainable_params is not None:
        row['trainable_params'] = trainable_params
    return row


def _plan_graph(plan, module=None, counts=None, registry=None):
    registry = registry or plan.registry
    nodes = list(plan.nodes)
    rows, total, trainable = [], 0, 0
    for index, node in enumerate(nodes):
        params = trainable_params = None
        if module is not None and f'n_{node.id}' in module.nodes:
            tensors = list(module.nodes[f'n_{node.id}'].parameters())
            params = sum(t.numel() for t in tensors)
            trainable_params = sum(t.numel() for t in tensors if t.requires_grad)
        elif counts is not None and node.id in counts:
            params = counts[node.id]
            trainable_params = params if dict(node.trainability).get('default') is not False else 0
        total += params or 0
        trainable += trainable_params or 0
        if index < MAX_NODES:
            rows.append(_node_row(registry, node, params, trainable_params))
    return {'node_count': len(nodes), 'nodes_truncated': len(nodes) > MAX_NODES,
            'semantic_digest': plan.semantic_digest, 'input_shape': _bounded(plan.inputs, 4096),
            'output_shape': _bounded(plan.outputs, 4096), 'nodes': rows,
            'parameters': {'total': total, 'trainable': trainable, 'frozen': total - trainable}}


def module_graph(module):
    """Every hndl GraphModule inside one constructed component."""
    from hndl.torch import GraphModule
    subgraphs = []
    for path, sub in module.named_modules():
        if isinstance(sub, GraphModule) and getattr(sub, 'plan', None) is not None:
            subgraphs.append(dict(module_path=path or '(root)', **_plan_graph(sub.plan, module=sub)))
    tensors = list(module.parameters())
    total = sum(t.numel() for t in tensors)
    trainable = sum(t.numel() for t in tensors if t.requires_grad)
    return {'status': 'built' if subgraphs else 'built-no-hndl', 'subgraphs': subgraphs[:MAX_SUBGRAPHS],
            'parameters': {'total': total, 'trainable': trainable, 'frozen': total - trainable,
                           'note': 'pretrained nodes count the whole backbone'}}


def _document(components, config_sha256, origin, started):
    try:
        from importlib.metadata import version
        hndl_version = version('hndl')
    except Exception:
        hndl_version = None
    document = {'schema_version': SCHEMA_VERSION, 'config_sha256': config_sha256, 'origin': origin,
                'hndl': hndl_version, 'components': components}
    if started is not None:  # the train-time record stays byte-stable across attempts
        document['seconds'] = round(time.perf_counter() - started, 3)
    return document


def record_networks(graph, config_sha256):
    """Train-time writer: a live ComponentGraph -> the model.json document."""
    components = {}
    for name, module in graph.models.items():
        try:
            components[name] = module_graph(module)
        except Exception as exc:  # detail is advisory; never fail training for it
            components[name] = unavailable(f'{type(exc).__name__}: {redact_text(exc)[:300]}')
    return _document(components, config_sha256, 'recorded', None)


def _captured(args, error):
    """Nodes without shapes when full resolution needs files this host does not have."""
    from hndl import HNDLError, Registry
    from hndl.config import capture_config
    from .hndl_augmentation import register_augmentation
    from .network_config import render_source
    from .pretrained_providers import register_providers
    registry = register_augmentation(Registry.builtins())
    try:
        registry = register_providers(registry, args.get('pretrained_providers'))
    except Exception:
        registry = register_providers(register_augmentation(Registry.builtins()))
    shape = lambda v: {k: tuple(s) for k, s in v.items()} if isinstance(v, dict) else tuple(v)
    try:
        graph = capture_config(render_source(args['source'], args.get('parameters')),
                               input_shape=shape(args['input_shape']), output_shape=shape(args['output_shape']),
                               registry=registry)
    except (HNDLError, ValueError, KeyError, TypeError) as exc:
        return unavailable(f'{error}; capture failed: {redact_text(exc)[:300]}')
    nodes = list(graph.nodes)
    return {'status': 'captured', 'reason': f'{error}; shapes and parameter counts unavailable',
            'subgraphs': [{'module_path': '(config)', 'node_count': len(nodes), 'nodes_truncated': len(nodes) > MAX_NODES,
                           'nodes': [_node_row(registry, n, shapes=False) for n in nodes[:MAX_NODES]]}]}


def build_networks(config, config_sha256=None):
    """Backfill: construct each component on the meta device (no allocation) and walk it."""
    import torch
    from .recipes import construct
    started = time.perf_counter()
    components = {}
    for name, spec in config['components'].items():
        if 'reuse' in spec:
            continue
        began = time.perf_counter()
        try:
            with torch.device('meta'):
                module = construct(spec)
            if not isinstance(module, torch.nn.Module):
                raise ValueError('constructor did not return a torch.nn.Module')
            if not spec.get('trainable', True):
                module.requires_grad_(False)
            components[name] = module_graph(module)
        except Exception as exc:
            error = f'{type(exc).__name__}: {redact_text(exc)[:300]}'
            components[name] = (_captured(spec.get('args') or {}, error) if spec.get('factory') == 'hndl'
                                else unavailable(f'meta build failed: {error}'))
        components[name]['seconds'] = round(time.perf_counter() - began, 3)
    return _document(components, config_sha256, 'backfill', started)

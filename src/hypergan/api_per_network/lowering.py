"""Lower per-network declarations to the existing HyperGAN configuration, and back.

``lower`` emits only what a declaration states; ``resolve_config`` fills the
rest, so ``defaults = "particlegan"`` and every existing default keep working.
``lift`` reads any existing configuration into declarations.
"""
from copy import deepcopy
import os
from pathlib import Path
import warnings

from ..config import DEFAULT
from .declare import (
    INHERIT, Data, Declaration, Evaluation, Judge, Loss, Metric, Network, Optimizer, Penalty, Prior,
    Recipe, adversarial)

ROOTS = {'latent', 'batch', 'generated', 'candidate', 'components', 'prior'}
PRIMARY_GENERATOR, PRIMARY_CRITIC = 'generator', 'discriminator'
_PENALTY_FIELDS = ('coeff', 'kappa', 'lazy_k', 'anchor_weight', 'anchor_decay')


def bind(path, names, where):
    """Engine binding for ``path``; a bare network name means that network's output."""
    if not isinstance(path, str) or not path or not all(path.split('.')):
        raise ValueError(f'{where}: bindings are non-empty dotted paths, got {path!r}')
    root = path.split('.')[0]
    if root in ROOTS:
        return path
    if root in names:
        return 'components.' + path
    raise ValueError(f'{where}: unknown binding {path!r}; use latent, batch.<field>, generated, '
                     f'candidate or a declared network name ({", ".join(names)})')


def producer(path):
    """The network whose output a binding reads directly, if any."""
    parts = path.split('.')
    if parts[0] == 'generated':
        return PRIMARY_GENERATOR
    if parts[0] == 'components':
        return parts[1]
    return None


def _qualified(owner, ident, verbatim):
    return ident if ident is None or verbatim else f'{owner}.{ident}'


def _judged(recipe):
    """{critic: [(index, judge, engine real, engine fake)]} and {producer: {critics}}."""
    names = recipe.networks
    judged, fooled = {}, {}
    for name, declaration in names.items():
        if declaration.role != 'critic':
            continue
        rows = judged.setdefault(name, [])
        for index, item in enumerate(declaration.judges):
            if not isinstance(item, Judge):
                raise ValueError(f'networks.{name}.judges[{index}] must be judge(...)')
            where = f'networks.{name}.judges[{index}]'
            real, fake = bind(item.real, names, where), bind(item.fake, names, where)
            rows.append((index, item, real, fake))
            source = producer(fake)
            if source is not None:
                fooled.setdefault(source, set()).add(name)
    return judged, fooled


def upstream(path, components, *, seen=None):
    """Networks a binding's value depends on (reuse follows the shared parameters)."""
    seen = set() if seen is None else seen
    source = producer(path)
    if source is None or source in seen or source not in components:
        return seen
    seen.add(source)
    spec = components[source]
    if 'reuse' in spec:
        seen.add(spec['reuse'])
    for value in spec['inputs'].values():
        upstream(value, components, seen=seen)
    return seen


def lower(recipe):
    """The raw configuration dict (what a TOML file parses to) for ``recipe``."""
    if not isinstance(recipe, Recipe):
        raise TypeError('lower() takes a Recipe from recipe(...)')
    names = recipe.networks
    for name in names:
        if not name.isidentifier():
            raise ValueError(f'Network names must be Python identifiers: {name!r}')
    if names.get(PRIMARY_GENERATOR, Declaration('', None, {})).role != 'generator':
        raise ValueError("Engine limit: declare the adversarial generator as networks['generator'] = "
                         "generator(...); its output is the 'generated' binding. Other generators may use any name.")
    if names.get(PRIMARY_CRITIC, Declaration('', None, {})).role != 'critic':
        raise ValueError("Engine limit: declare the first critic as networks['discriminator'] = critic(...). "
                         "Other critics may use any name.")
    config = {'schema_version': 1, 'name': recipe.name}
    if recipe.defaults is not None:
        config['defaults'] = recipe.defaults
    config['data'] = {'factory': recipe.data.factory, 'args': deepcopy(recipe.data.args)}
    config.update(_lower_prior(recipe.prior))
    components, objectives, terms = {}, [], []
    adversarial_section, penalty_section, optimizer = {}, {}, {}
    judged, fooled = _judged(recipe)
    primary_penalty = names[PRIMARY_CRITIC].penalty
    for name, declaration in names.items():
        where = f'networks.{name}'
        inputs = {key: bind(value, names, where + '.inputs') for key, value in declaration.inputs.items()}
        if declaration.role == 'shared':
            if declaration.reuse not in names:
                raise ValueError(f'{where}: shared() names an unknown network {declaration.reuse!r}')
            spec = {'reuse': declaration.reuse, 'inputs': inputs}
            if declaration.freeze_parameters is not None:
                spec['freeze_parameters'] = declaration.freeze_parameters
        else:
            network = declaration.network
            spec = {'factory': network.factory, 'args': deepcopy(network.args), 'inputs': inputs}
            if network.file is not None:
                spec['args']['file'] = str(Path(network.file).resolve())
            if declaration.role == 'frozen':
                spec['trainable'] = False
        components[name] = spec
        if declaration.role in ('critic', 'frozen') and declaration.losses:
            raise ValueError(f'{where}: a {declaration.role} network has no losses of its own; '
                             + ('its judges and penalty train it' if declaration.role == 'critic'
                                else 'attach the losses to the network they train'))
        for loss in declaration.losses:
            if not isinstance(loss, Loss):
                raise ValueError(f'{where}.losses must contain mse(), l1(), loss() or adversarial()')
            if loss.kind == 'spread':
                raise ValueError(f'{where}: spread() regularizes the prior; attach it to particles(losses=...)')
            if loss.kind == 'adversarial':
                continue
            term = {'factory': loss.factory,
                    'inputs': {key: bind(value, names, where + '.losses') for key, value in loss.inputs.items()}}
            if loss.weight is not None:
                term['weight'] = loss.weight
            if loss.detach is not None:
                term['detach'] = list(loss.detach)
            if loss.args:
                term['args'] = deepcopy(loss.args)
            ident = _qualified(name, loss.id, loss.verbatim_id)
            if ident is not None:
                term['id'] = ident
            objectives.append((name, term))
        if declaration.role == 'critic':
            if sum(value == 'candidate' for value in inputs.values()) != 1:
                raise ValueError(f'{where}: bind exactly one critic input to CANDIDATE')
            if not declaration.judges:
                raise ValueError(f'{where}: a critic needs at least one judge(real, fake)')
            if name != PRIMARY_CRITIC and declaration.penalty is not None:
                _shared_penalty_settings(where, declaration.penalty, primary_penalty)
            for index, item, real, fake in judged[name]:
                penalty = declaration.penalty if item.penalty is INHERIT else (item.penalty or None)
                if penalty is not None and item.penalty is not INHERIT:
                    _shared_penalty_settings(f'{where}.judges[{index}]', penalty, primary_penalty)
                if name == PRIMARY_CRITIC and index == 0:
                    if (real, fake) != ('batch.real', 'generated') or item.inputs is not None or item.id is not None:
                        raise ValueError("Engine limit: the first judge of 'discriminator' compares batch.real "
                                         "with generated through the critic's own inputs; add other "
                                         "comparisons as further judges")
                    if item.weight is not None:
                        adversarial_section['weight'] = item.weight
                    if penalty is None:
                        penalty_section['coeff'] = 0.0
                    continue
                term = {'id': _qualified(name, item.id or ('main' if index == 0 else f'judge{index}'),
                                         item.verbatim_id),
                        'component': name, 'real': real, 'fake': fake}
                if item.weight is not None:
                    term['weight'] = item.weight
                term['penalty'] = penalty is not None
                if penalty is not None and penalty.coeff is not None:
                    term['penalty_coeff'] = penalty.coeff
                if item.inputs is not None:
                    term['inputs'] = {key: bind(value, names, f'{where}.judges[{index}].inputs')
                                      for key, value in item.inputs.items()}
                terms.append(term)
            if name == PRIMARY_CRITIC and primary_penalty is not None:
                for key in _PENALTY_FIELDS:
                    value = getattr(primary_penalty, key)
                    if value is not None:
                        penalty_section.setdefault(key, value)
    _check_adversarial_claims(recipe, fooled)
    _check_losses_reach_owner(components, objectives)
    optimizer.update(_lower_optimizers(recipe))
    config['components'] = components
    if objectives:
        config['objectives'] = [term for _, term in objectives]
    if terms:
        config['adversarial_terms'] = terms
    if adversarial_section:
        config['adversarial'] = adversarial_section
    if penalty_section:
        config['gradient_penalty'] = penalty_section
    if optimizer:
        config['optimizer'] = optimizer
    if recipe.training:
        config['training'] = deepcopy(recipe.training)
    if recipe.sampling:
        config['sampling'] = deepcopy(recipe.sampling)
    metrics = _lower_observations(recipe)
    if metrics:
        config['metrics'] = metrics
    return config


def _shared_penalty_settings(where, penalty, primary):
    for key in _PENALTY_FIELDS[1:]:
        value = getattr(penalty, key)
        if value is not None and value != getattr(primary or Penalty(), key):
            raise ValueError(f'Engine limit ({where}): every critic shares one K3P configuration (one '
                             f'critic optimizer and EMA critic); only coeff may differ. Set {key} on '
                             f"'discriminator'.")


def _check_adversarial_claims(recipe, fooled):
    for name, declaration in recipe.networks.items():
        claimed = {loss.critic for loss in declaration.losses if isinstance(loss, Loss) and loss.kind == 'adversarial'}
        for critic_name in claimed:
            if recipe.networks.get(critic_name, Declaration('', None, {})).role != 'critic':
                raise ValueError(f'networks.{name}: adversarial({critic_name!r}) names no declared critic')
        judged_by = fooled.get(name, set())
        missing, extra = judged_by - claimed, claimed - judged_by
        if missing:
            raise ValueError(f'networks.{name}: its output is judged by {sorted(missing)}; add '
                             + ', '.join(f'adversarial({c!r})' for c in sorted(missing))
                             + ' to its losses so its declaration shows everything that trains it')
        if extra:
            raise ValueError(f'networks.{name}: declares adversarial({sorted(extra)[0]!r}) but no judge of '
                             f'that critic scores this network\'s output as fake')
    for source in fooled:
        if source not in recipe.networks:
            raise ValueError(f'A judge scores {source!r}, which is not a declared network')


def _check_losses_reach_owner(components, objectives):
    for owner, term in objectives:
        detached = set(term.get('detach', ['target'] if 'target' in term['inputs'] else []))
        reached = set()
        for key, value in term['inputs'].items():
            if key not in detached:
                upstream(value, components, seen=reached)
        if owner not in reached:
            raise ValueError(f'networks.{owner}: loss {term.get("id", term["factory"])!r} reads '
                             f'{sorted(term["inputs"].values())}, which do not depend on {owner}; attach it '
                             f'to a network it trains ({", ".join(sorted(reached)) or "none"})')


def _lower_prior(prior):
    if not isinstance(prior, Prior):
        raise ValueError('prior must be particles(...), mog(...) or gaussian(...)')
    section = {'kind': prior.kind, 'args': deepcopy(prior.args), **deepcopy(prior.options)}
    spreads = [loss for loss in prior.losses if loss.kind == 'spread']
    if len(spreads) != len(prior.losses) or len(spreads) > 1:
        raise ValueError('A prior takes at most one spread() loss')
    if not spreads:
        regularizer = {'weight': 0.0}
    else:
        regularizer = dict(spreads[0].args)
        if spreads[0].weight is not None:
            regularizer['weight'] = spreads[0].weight
    return {'prior': section, **({'prior_regularizer': regularizer} if regularizer else {})}


def _lower_optimizers(recipe):
    section = {}
    for name, declaration in recipe.networks.items():
        options = declaration.optimizer
        if options is None:
            continue
        if not isinstance(options, Optimizer):
            raise ValueError(f'networks.{name}.optimizer must be adam(...)')
        extra = dict(options.options)
        if name == PRIMARY_GENERATOR:
            if options.lr_mult is not None:
                raise ValueError("networks.generator: the generator sets the base rate; use adam(lr=...)")
            _put(section, 'lr', options.lr)
            _put(section, 'betas', None if options.betas is None else list(options.betas))
            _put(section, 'implementation', extra.pop('implementation', None))
        elif name == PRIMARY_CRITIC:
            if options.lr is not None:
                raise ValueError("networks.discriminator: the engine sets the critic rate relative to the "
                                 "generator's; use adam(lr_mult=...)")
            if options.betas is not None:
                raise ValueError('Engine limit: critics share the generator optimizer betas; set betas on generator')
            _put(section, 'd_lr_mult', options.lr_mult)
            _put(section, 'd_guard_ratio', extra.pop('guard_ratio', None))
            _put(section, 'd_guard_min_steps', extra.pop('guard_min_steps', None))
        else:
            group = ('the critic optimizer of discriminator' if declaration.role == 'critic'
                     else 'the generator-side optimizer of generator')
            raise ValueError(f'Engine limit: networks.{name} is trained by {group}; per-network optimizers '
                             f'are declared here but not yet executed. Remove optimizer= from {name}.')
        if extra:
            raise ValueError(f'networks.{name}.optimizer: unknown options {sorted(extra)}')
    prior = recipe.prior.optimizer
    if prior is not None:
        extra = dict(prior.options)
        if prior.lr is not None:
            raise ValueError('prior: the engine sets the prior rate relative to the generator; use adam(lr_mult=...)')
        _put(section, 'prior_lr_mult', prior.lr_mult)
        _put(section, 'prior_betas', None if prior.betas is None else list(prior.betas))
        _put(section, 'latent_damping_max_rate', extra.pop('latent_damping', None))
        if extra:
            raise ValueError(f'prior.optimizer: unknown options {sorted(extra)}')
    return section


def _put(section, key, value):
    if value is not None:
        section[key] = value


def _lower_observations(recipe):
    metrics = deepcopy(recipe.metric_settings)
    custom = deepcopy(metrics.pop('custom', {}))
    for item in recipe.observe:
        if isinstance(item, Metric):
            spec = {'factory': item.factory, 'args': deepcopy(item.args), 'inputs': dict(item.inputs), 'mode': 'scalar'}
            _put(spec, 'every_steps', item.every)
        elif isinstance(item, Evaluation):
            evaluation = {'data': {'factory': item.data.factory, 'args': deepcopy(item.data.args)},
                          'sample_count': item.samples, 'batch_size': item.batch_size, 'seed': item.seed}
            _put(evaluation, 'device', item.device)
            _put(evaluation, 'generated', item.generated)
            spec = {'factory': item.factory, 'args': deepcopy(item.args), 'inputs': dict(item.inputs),
                    'mode': 'snapshot', 'evaluation': evaluation}
            if item.every is None:
                spec['trigger'] = 'manual'
            else:
                spec.update(trigger='interval', every_steps=item.every)
        else:
            continue  # samplers are read-side (see samples()/previews())
        _put(spec, 'timeout', item.timeout)
        _put(spec, 'on_error', item.on_error)
        if item.id in custom:
            raise ValueError(f'Observation id {item.id!r} is repeated')
        custom[item.id] = spec
    if custom:
        metrics['custom'] = custom
    return metrics


# ------------------------------------------------------------------ lift

def _split_id(ident, owners):
    if isinstance(ident, str) and '.' in ident and ident.split('.', 1)[0] in owners:
        return ident.split('.', 1)[0], ident.split('.', 1)[1], False
    return None, ident, ident is not None


def lift(raw, *, base=None):
    """Declarations for an existing raw configuration (a parsed TOML file)."""
    raw = deepcopy(raw)
    base = Path.cwd() if base is None else Path(base)
    components = raw.get('components') or deepcopy(DEFAULT['components'])
    raw.setdefault('data', deepcopy(DEFAULT['data']))
    terms = raw.get('adversarial_terms') or []
    critics = list(dict.fromkeys([PRIMARY_CRITIC] + [term['component'] for term in terms]))
    judges = {name: [] for name in critics}
    fooled = {PRIMARY_GENERATOR: [PRIMARY_CRITIC]}
    adversarial_section = raw.get('adversarial') or {}
    judges[PRIMARY_CRITIC].append(Judge(weight=adversarial_section.get('weight')))
    for term in terms:
        name = term['component']
        _, ident, verbatim = _split_id(term['id'], {name})
        penalty = Penalty(coeff=term.get('penalty_coeff')) if term.get('penalty', False) else None
        judges[name].append(Judge(term['real'], term['fake'], term.get('weight'), term.get('inputs'),
                                  penalty, ident, verbatim))
        source = producer(term['fake'])
        if source is not None and name not in fooled.setdefault(source, []):
            fooled[source].append(name)
    gp = raw.get('gradient_penalty') or {}
    primary_penalty = Penalty(**{key: gp[key] for key in _PENALTY_FIELDS if key in gp})
    losses = {name: [adversarial(critic_name) for critic_name in fooled.get(name, [])] for name in components}
    owners = []
    for term in raw.get('objectives') or []:
        owner, ident, verbatim = _split_id(term.get('id'), set(components))
        if owner is None:
            detached = set(term.get('detach', ['target'] if 'target' in term['inputs'] else []))
            sources = [producer(value) for key, value in term['inputs'].items() if key not in detached]
            owner = next((source for source in sources if source and source not in critics), PRIMARY_GENERATOR)
        owners.append(owner)
        detach = term.get('detach')
        losses[owner].append(Loss('objective', term['factory'], dict(term['inputs']), term.get('weight'),
                                  None if detach is None else tuple(detach), dict(term.get('args') or {}),
                                  ident, verbatim_id=verbatim))
    # Lowering emits losses and judges network by network, so order the networks
    # the way the file orders its objectives and extra adversarial terms.
    order = _follow(_follow(list(components), owners, 'Objectives'),
                    [term['component'] for term in terms], 'Adversarial terms')
    optimizer = raw.get('optimizer') or {}
    networks = {}
    for name in order:
        spec = components[name]
        inputs = dict(spec['inputs'])
        if 'reuse' in spec:
            networks[name] = Declaration('shared', None, inputs, tuple(losses[name]), reuse=spec['reuse'],
                                         freeze_parameters=spec.get('freeze_parameters'))
            continue
        args = dict(spec.get('args') or {})
        file = args.pop('file', None)
        if isinstance(args.get('network_files'), dict):
            # Template files stay references: absolute here, relative again when saved.
            args['network_files'] = {key: str((base / value).resolve())
                                     for key, value in args['network_files'].items()}
        network = Network(spec['factory'], args, None if file is None else str((base / file).resolve()))
        if name in critics:
            options = {key: optimizer[engine] for key, engine in (('guard_ratio', 'd_guard_ratio'),
                       ('guard_min_steps', 'd_guard_min_steps')) if engine in optimizer}
            settings = (Optimizer(lr_mult=optimizer.get('d_lr_mult'), options=options)
                        if name == PRIMARY_CRITIC and (options or 'd_lr_mult' in optimizer) else None)
            penalty, rows = _critic_penalty(name, judges[name], primary_penalty)
            networks[name] = Declaration('critic', network, inputs, (), tuple(rows), penalty, settings)
        elif spec.get('trainable', True) is False:
            networks[name] = Declaration('frozen', network, inputs)
        else:
            role = 'generator' if name == PRIMARY_GENERATOR or name in fooled else 'encoder'
            settings = None
            if name == PRIMARY_GENERATOR and {'lr', 'betas', 'implementation'} & set(optimizer):
                betas = optimizer.get('betas')
                settings = Optimizer(lr=optimizer.get('lr'), betas=None if betas is None else tuple(betas),
                                     options={k: optimizer[k] for k in ('implementation',) if k in optimizer})
            networks[name] = Declaration(role, network, inputs, tuple(losses[name]), optimizer=settings)
    return Recipe(networks, Data(raw['data']['factory'], dict(raw['data'].get('args') or {})),
                  _lift_prior(raw, optimizer), dict(raw.get('training') or {}), _lift_observations(raw),
                  dict(raw.get('sampling') or {}),
                  {k: v for k, v in (raw.get('metrics') or {}).items() if k != 'custom'},
                  raw.get('name', 'custom/api'), raw.get('defaults'))


def _follow(order, sequence, what):
    """Reorder the networks named in ``sequence`` into its order, keeping everyone else's slot."""
    groups = list(dict.fromkeys(sequence))
    if [name for index, name in enumerate(sequence) if index == 0 or sequence[index - 1] != name] != groups:
        warnings.warn(f'{what} interleave networks; lowering groups them by network, which changes '
                      f'their order and the recipe fingerprint', RuntimeWarning, stacklevel=3)
    slots = iter(groups)
    return [next(slots) if name in groups else name for name in order]


def _critic_penalty(name, rows, primary):
    """Critic-level penalty; judges that match it inherit it."""
    if name == PRIMARY_CRITIC:
        first, rest = rows[0], rows[1:]
        rows = [first] + [_inherit(row, primary) for row in rest]
        return primary, rows
    specs = [row.penalty for row in rows]
    if all(spec == specs[0] for spec in specs):
        return specs[0], [Judge(row.real, row.fake, row.weight, row.inputs, INHERIT, row.id, row.verbatim_id)
                          for row in rows]
    return None, rows


def _inherit(row, penalty):
    if row.penalty is not None and row.penalty.coeff == penalty.coeff:
        return Judge(row.real, row.fake, row.weight, row.inputs, INHERIT, row.id, row.verbatim_id)
    return row


def _lift_prior(raw, optimizer):
    section = dict(raw.get('prior') or {'kind': DEFAULT['prior']['kind'], 'args': deepcopy(DEFAULT['prior']['args'])})
    kind, args = section.pop('kind', DEFAULT['prior']['kind']), dict(section.pop('args', DEFAULT['prior']['args']))
    regularizer = raw.get('prior_regularizer')
    losses = (Loss('spread', args={}),) if regularizer is None else (
        Loss('spread', weight=regularizer.get('weight'),
             args={k: v for k, v in regularizer.items() if k != 'weight'}),)
    settings = None
    if {'prior_lr_mult', 'prior_betas', 'latent_damping_max_rate'} & set(optimizer):
        betas = optimizer.get('prior_betas')
        settings = Optimizer(lr_mult=optimizer.get('prior_lr_mult'), betas=None if betas is None else tuple(betas),
                             options={'latent_damping': optimizer['latent_damping_max_rate']}
                             if 'latent_damping_max_rate' in optimizer else {})
    return Prior(kind, args, losses, settings, section)


def _lift_observations(raw):
    items = []
    for ident, spec in ((raw.get('metrics') or {}).get('custom') or {}).items():
        common = dict(timeout=spec.get('timeout'), on_error=spec.get('on_error'))
        if spec.get('mode', 'scalar') == 'scalar':
            items.append(Metric(ident, spec['factory'], dict(spec['inputs']), dict(spec.get('args') or {}),
                                spec.get('every_steps'), **common))
            continue
        evaluation = spec['evaluation']
        manual = spec.get('trigger') == 'manual'
        items.append(Evaluation(
            ident, spec['factory'], dict(spec['inputs']),
            Data(evaluation['data']['factory'], dict(evaluation['data']['args'])),
            evaluation['sample_count'], evaluation['batch_size'], evaluation['seed'],
            dict(spec.get('args') or {}), None if manual else spec.get('every_steps', 10000),
            evaluation.get('device'), evaluation.get('generated'), **common))
    return tuple(items)


def relative_file(path, directory):
    """A saved config references HNDL files relative to itself when they share a tree."""
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if os.path.commonpath([path, directory]) in (os.sep, str(Path.home())):
        return str(path)
    return os.path.relpath(path, directory)

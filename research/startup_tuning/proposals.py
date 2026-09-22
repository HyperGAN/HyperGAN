"""Reversible explicit research proposals; never infer or select multipliers."""
from contextlib import contextmanager
import copy
from fnmatch import fnmatchcase
import json
import math

import torch

from hypergan.signal_structure import _hash, _storage
from hypergan.startup_dynamics import _restore, _snapshot


def _positive(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _ownership(trainer):
    from hndl.operators.pretrained import Pretrained
    registered = {}
    for prefix, root in (('graph', trainer.graph), ('prior', trainer.prior)):
        for name, value in list(root.named_parameters(remove_duplicate=False)) + list(root.named_buffers(remove_duplicate=False)):
            registered[prefix + '.' + name] = value
    forbidden = set()
    for root in (trainer.graph, trainer.prior):
        for module in root.modules():
            if isinstance(module, Pretrained):
                forbidden.update(_storage(value) for value in list(module.parameters()) + list(module.buffers()))
    aliases = {}
    for name, value in registered.items():
        aliases.setdefault(_storage(value), []).append(name)
    declared_g = {id(p) for p in trainer.graph.models['generator'].parameters()}
    g = {id(p) for p in trainer.program.generator_parameters} & declared_g
    d = {id(p) for p in trainer.program.critic_parameters}
    prior = {id(p) for p in trainer.prior.parameters()}
    if g & d or (g | d) & prior:
        raise ValueError('Overlapping generator/discriminator/prior ownership')
    allowed = {}
    for name, value in registered.items():
        if not name.startswith('graph.models.') or not isinstance(value, torch.nn.Parameter):
            continue
        component = name.split('.')[2]
        if trainer.config['components'].get(component, {}).get('factory') != 'hndl':
            continue
        if (not value.requires_grad or id(value) not in g | d or _storage(value) in forbidden
                or len(aliases[_storage(value)]) != 1):
            continue
        allowed[name] = (value, 'generator' if id(value) in g else 'discriminator')
    return registered, allowed


def _rules(plan, kind, registered, allowed):
    rules = plan.get(kind, [])
    if not isinstance(rules, list):
        raise ValueError(f'{kind} must be a list')
    rows, assigned = [], {}
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {'pattern', 'multiplier'}:
            raise ValueError(f'{kind} rules require exactly pattern and multiplier')
        pattern = rule['pattern']
        if not isinstance(pattern, str) or not pattern:
            raise ValueError('Parameter pattern must be a nonempty string')
        multiplier = _positive(rule['multiplier'], 'multiplier')
        matched = [name for name in registered if fnmatchcase(name, pattern)]
        if not matched:
            raise ValueError(f'Parameter pattern matched zero registered tensors: {pattern}')
        if any(name not in allowed for name in matched):
            raise ValueError(f'Pattern matches protected, prior, aliased, or unknown ownership: {pattern}')
        for name in matched:
            if name in assigned:
                raise ValueError(f'Overlapping {kind} rules match {name}')
            assigned[name] = multiplier
        rows.append(dict(pattern=pattern, multiplier=multiplier, matched_names=matched))
    return rows, assigned


def _layout(trainer, allowed, assigned, plan):
    names = {id(value): name for name, (value, _) in allowed.items()}
    optimizers = (trainer.opt_g, trainer.opt_d)
    owners = (trainer.program.generator_parameters, trainer.program.critic_parameters)
    prior_ids = {id(p) for p in trainer.program.prior_parameters}
    layouts, rates, records = [], [], []
    for index, (optimizer, parameters) in enumerate(zip(optimizers, owners)):
        role = 'generator' if index == 0 else 'discriminator'
        if not optimizer.param_groups or len(trainer.base_lrs[index]) != len(optimizer.param_groups):
            raise ValueError('Optimizer/base-rate layout mismatch')
        ids = {id(p) for p in parameters}
        main = optimizer.param_groups[0]
        if {id(p) for p in main['params']} != ids:
            raise ValueError('Main optimizer group does not exactly match declared player ownership')
        if any(id(p) not in names for p in main['params']):
            raise ValueError('Player optimizer includes protected, aliased, or unknown ownership')
        if index == 1 and len(optimizer.param_groups) != 1:
            raise ValueError('Expected one original discriminator group')
        remaining = [p for group in optimizer.param_groups[1:] for p in group['params']]
        if len({id(p) for p in main['params'] + remaining}) != len(main['params']) + len(remaining):
            raise ValueError('Optimizer contains duplicate parameter references')
        if index == 0 and {id(p) for p in remaining} != prior_ids:
            raise ValueError('Only unchanged prior groups may follow the original generator group')
        base = _positive(plan.get('g_lr' if index == 0 else 'd_lr', trainer.base_lrs[index][0]), role + ' LR')
        buckets = {}
        for parameter in main['params']:
            name = names[id(parameter)]
            rate = _positive(base * assigned.get(name, 1.), name + ' effective LR')
            buckets.setdefault(rate, []).append(parameter)
        split = [{**main, 'params': members, 'lr': rate} for rate, members in buckets.items()]
        # Original prior groups keep their indices and dictionaries. Additional
        # G groups follow them, rather than being inserted in front of prior.
        groups = [split[0], *optimizer.param_groups[1:], *split[1:]]
        base_rates = [split[0]['lr'], *trainer.base_lrs[index][1:], *[group['lr'] for group in split[1:]]]
        layouts.append(groups)
        rates.append(base_rates)
        records.append({'player': role, 'base_lr': base,
                        'parameters': [{'path': names[id(p)], 'multiplier': assigned.get(names[id(p)], 1.),
                                        'effective_lr': base * assigned.get(names[id(p)], 1.)}
                                       for p in main['params']],
                        'group_base_lrs': base_rates})
    return layouts, rates, records


@contextmanager
def apply_proposal(trainer, plan):
    """Apply an explicit, validated step-zero plan, then restore all entry state.

    Patterns use registered ``graph.models...`` parameter paths. Scaling an
    explicitly named owned bias is supported as well as scaling a weight.
    Optimizer group splitting is temporary and is not a checkpoint schema.
    The caller may snapshot/restore within this context using its split layout;
    persistent checkpoints require a separately designed ownership contract.
    """
    if (trainer.step != 0 or trainer.opt_g.state or trainer.opt_d.state
            or getattr(trainer, 'g_lr_warmup', None) is not None):
        raise ValueError('Proposals require fresh step zero, empty optimizers, and no G warmup')
    allowed_fields = {'schema_version', 'g_lr', 'd_lr', 'init_scales', 'layer_lr_multipliers', 'evidence'}
    if (not isinstance(plan, dict) or set(plan) - allowed_fields
            or type(plan.get('schema_version')) is not int or plan['schema_version'] != 1):
        raise ValueError('Unsupported proposal schema')
    try:
        plan = json.loads(json.dumps(plan, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Proposal and evidence must be finite JSON') from error
    registered, allowed = _ownership(trainer)
    init_rows, scales = _rules(plan, 'init_scales', registered, allowed)
    lr_rows, multipliers = _rules(plan, 'layer_lr_multipliers', registered, allowed)
    layouts, rates, optimizer_rows = _layout(trainer, allowed, multipliers, plan)
    ema = dict(trainer.ema_graph.named_parameters(remove_duplicate=False))
    ema_aliases, ema_forbidden = {}, set()
    from hndl.operators.pretrained import Pretrained
    for name, tensor in list(trainer.ema_graph.named_parameters(remove_duplicate=False)) + list(trainer.ema_graph.named_buffers(remove_duplicate=False)):
        ema_aliases.setdefault(_storage(tensor), []).append(name)
    for module in trainer.ema_graph.modules():
        if isinstance(module, Pretrained):
            ema_forbidden.update(_storage(value) for value in list(module.parameters()) + list(module.buffers()))
    initialized, init_records = [], []
    for name, multiplier in scales.items():
        parameter, role = allowed[name]
        destination = ema.get(name.removeprefix('graph.'))
        if destination is None or destination.shape != parameter.shape or destination.dtype != parameter.dtype:
            raise ValueError('Initialization requires a matching EMA parameter: ' + name)
        if _storage(destination) in ema_forbidden or len(ema_aliases[_storage(destination)]) != 1:
            raise ValueError('Initialization EMA destination is pretrained or aliased: ' + name)
        before = parameter.detach().cpu().clone()
        scaled = before * multiplier
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError('Initialization produces nonfinite parameter: ' + name)
        initialized.append((parameter, destination, scaled))
        init_records.append(dict(path=name, player=role, multiplier=multiplier,
                                 before_sha256=_hash([(name, before)]), after_sha256=_hash([(name, scaled)])))
    metadata = dict(schema_version=1, plan=plan, initialization_rules=init_rows,
                    initialization_parameters=init_records, layer_lr_rules=lr_rows,
                    optimizers=optimizer_rows, effective_base_lrs=rates,
                    prior_rates_unchanged=True, pretrained_state_calibrated=False)
    json.dumps(metadata, allow_nan=False)
    # All validation and proposed tensor arithmetic precede the first mutation.
    entry = _snapshot(trainer)
    optimizers = (trainer.opt_g, trainer.opt_d)
    original_groups = [optimizer.param_groups for optimizer in optimizers]
    original_parameters = [[group['params'] for group in groups] for groups in original_groups]
    try:
        with torch.no_grad():
            for parameter, destination, value in initialized:
                parameter.copy_(value)
                destination.copy_(value)
        for optimizer, groups in zip(optimizers, layouts):
            optimizer.param_groups = groups
        trainer.base_lrs = copy.deepcopy(rates)
        yield copy.deepcopy(metadata)
    finally:
        for optimizer, groups in zip(optimizers, original_groups):
            optimizer.param_groups = groups
        _restore(trainer, entry)
        # load_state_dict reconstructs dictionaries; reinstate the caller's
        # original group/list objects as well as their restored values.
        for optimizer, groups, params in zip(optimizers, original_groups, original_parameters):
            restored = optimizer.param_groups
            for original, saved, members in zip(groups, restored, params):
                original.clear()
                original.update(saved)
                original['params'] = members
            optimizer.param_groups = groups

"""A small TOML writer for configuration dicts (no tomli_w is installed).

The output is an ordinary HyperGAN config file: ``hypergan train FILE`` reads
it. With a recipe, each network's table is followed by the losses that train
it, so the file reads per network like the Python declarations.
"""
import json
import math
import re

_BARE = re.compile(r'[A-Za-z0-9_-]+')
_INLINE_WIDTH = 96


def _key(key):
    return key if _BARE.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _string(text):
    if '\n' in text and "'''" not in text and not text.endswith("'") and not any(
            ord(c) < 32 and c not in '\n\t' or c == '\x7f' for c in text):
        return "'''\n" + text + "'''"
    return json.dumps(text, ensure_ascii=False)


def value(item):
    if isinstance(item, bool):
        return 'true' if item else 'false'
    if isinstance(item, int):
        return str(item)
    if isinstance(item, float):
        if math.isnan(item):
            return 'nan'
        if math.isinf(item):
            return 'inf' if item > 0 else '-inf'
        return repr(item)
    if isinstance(item, str):
        return _string(item)
    if isinstance(item, (list, tuple)):
        return '[' + ', '.join(value(entry) for entry in item) + ']'
    if isinstance(item, dict):
        return '{' + ', '.join(f'{_key(k)} = {value(v)}' for k, v in item.items() if v is not None) + '}' \
            if item else '{}'
    raise TypeError(f'Cannot write {type(item).__name__} to TOML')


def _multiline(item):
    if isinstance(item, str):
        return '\n' in item
    if isinstance(item, dict):
        return any(_multiline(v) for v in item.values())
    if isinstance(item, (list, tuple)):
        return any(_multiline(v) for v in item)
    return False


def _inline(item):
    return not _multiline(item) and len(value(item)) <= _INLINE_WIDTH


def table(path, body, lines, *, array=False):
    """Append ``[path]`` (or ``[[path]]``) and its nested tables to ``lines``."""
    scalars, tables, arrays = [], [], []
    for key, item in body.items():
        if item is None:
            continue  # TOML has no null; every None in a config is its default
        if isinstance(item, dict) and item and not _inline(item):
            tables.append((key, item))
        elif (isinstance(item, list) and item and all(isinstance(e, dict) for e in item)
              and not _inline(item)):
            arrays.append((key, item))
        else:
            scalars.append((key, item))
    if path is not None and (scalars or array or not (tables or arrays)):
        lines.append(f'[[{path}]]' if array else f'[{path}]')
    for key, item in scalars:
        lines.append(f'{_key(key)} = {value(item)}')
    for key, item in tables:
        lines.append('')
        table(_join(path, key), item, lines)
    for key, entries in arrays:
        for entry in entries:
            lines.append('')
            table(_join(path, key), entry, lines, array=True)


def _join(path, key):
    return _key(key) if path is None else f'{path}.{_key(key)}'


def dumps(config, recipe=None, *, header=None):
    """TOML text for a raw config dict; grouped per network when ``recipe`` is given."""
    lines = [f'# {line}' if line else '#' for line in (header or '').splitlines()]
    top = {k: v for k, v in config.items() if not isinstance(v, (dict, list))}
    for key, item in top.items():
        lines.append(f'{_key(key)} = {value(item)}')
    done = set(top)

    def section(key, comment=None):
        if key in config and key not in done:
            lines.append('')
            if comment:
                lines.append(f'# {comment}')
            if isinstance(config[key], list):
                for entry in config[key]:
                    table(_key(key), entry, lines, array=True)
                    lines.append('')
                lines.pop()
            else:
                table(_key(key), config[key], lines)
            done.add(key)

    section('data', 'data: what one item/batch is; HyperGAN owns order, seeding, resume and sharding')
    if recipe is None:
        for key in config:
            section(key)
        return '\n'.join(lines).strip() + '\n'
    section('prior', 'prior: the latent source; its losses and optimizer follow')
    section('prior_regularizer', 'loss that trains the prior')
    objectives = list(config.get('objectives') or [])
    terms = list(config.get('adversarial_terms') or [])
    for name, declaration in recipe.networks.items():
        lines.append('')
        lines.append(f'# ---- network {name}: {declaration.role} ----')
        table(f'components.{_key(name)}', config['components'][name], lines)
        own = [loss for loss in declaration.losses if loss.kind == 'objective']
        claimed = [loss.critic for loss in declaration.losses if loss.kind == 'adversarial']
        if claimed:
            lines.append(f'# trained to fool: {", ".join(claimed)} (weights live on those critics)')
        for _ in own:
            lines.append('')
            lines.append(f'# loss that trains {name}')
            table('objectives', objectives.pop(0), lines, array=True)
        if declaration.role == 'critic':
            if name == 'discriminator':
                for key in ('adversarial', 'gradient_penalty'):
                    if key in config:
                        lines.append('')
                        table(key, config[key], lines)
                        done.add(key)
            while terms and terms[0]['component'] == name:
                lines.append('')
                lines.append(f'# another judgment that trains {name}')
                table('adversarial_terms', terms.pop(0), lines, array=True)
    if objectives or terms:
        raise ValueError('Config terms are not grouped in network order')
    done.update({'components', 'objectives', 'adversarial_terms'})
    section('optimizer', 'optimizers: generator-side lr/betas, critic d_*, prior prior_* (engine groups)')
    for key in config:
        section(key)
    return '\n'.join(lines).strip() + '\n'

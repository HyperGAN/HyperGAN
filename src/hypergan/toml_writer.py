"""A small TOML writer for model files; no third-party dependency.

It writes the subset of TOML that model and recipe files use: tables, arrays of
tables, strings (multi-line for HNDL source), integers, finite floats, booleans
and arrays. ``None`` has no TOML spelling, so keys whose value is ``None`` are
omitted; a loader then applies its default. The output is read back by
``tomllib`` to the same value (tests check this), which is what a saved model's
fingerprint depends on.
"""
import math
import re

_BARE = re.compile(r'[A-Za-z0-9_-]+')
# Short scalar-only tables read better inline, e.g. inputs = { x = "latent" }.
_INLINE_WIDTH = 88


def _key(name):
    if not isinstance(name, str):
        raise TypeError(f'TOML keys must be strings, got {type(name).__name__}')
    return name if _BARE.fullmatch(name) else _string(name)


def _string(value):
    escaped = (value.replace('\\', '\\\\').replace('"', '\\"').replace('\b', '\\b')
               .replace('\t', '\\t').replace('\n', '\\n').replace('\f', '\\f').replace('\r', '\\r'))
    escaped = ''.join(c if ord(c) >= 0x20 and ord(c) != 0x7f else f'\\u{ord(c):04x}' for c in escaped)
    return f'"{escaped}"'


def _multiline(value):
    # Basic multi-line string: a newline right after the opening quotes is
    # trimmed by TOML, so the body starts on its own line. Escape backslashes
    # and any quote that would start a closing delimiter.
    body = value.replace('\\', '\\\\').replace('"""', '""\\"')
    body = ''.join(c if c in '\n\t' or (ord(c) >= 0x20 and ord(c) != 0x7f) else f'\\u{ord(c):04x}' for c in body)
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    return f'"""\n{body}"""'


def _scalar(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('TOML model files cannot store non-finite numbers')
        return repr(value)
    if isinstance(value, str):
        return _multiline(value) if '\n' in value else _string(value)
    raise TypeError(f'Unsupported TOML value {type(value).__name__}')


def _is_table(value):
    return isinstance(value, dict)


def _is_table_array(value):
    return isinstance(value, list) and value and all(isinstance(item, dict) for item in value)


def _inline(value):
    if isinstance(value, dict):
        items = [f'{_key(k)} = {_inline(v)}' for k, v in value.items() if v is not None]
        return '{ ' + ', '.join(items) + ' }' if items else '{}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_inline(item) for item in value) + ']'
    if isinstance(value, str) and '\n' in value:
        return _string(value)
    return _scalar(value)


def _inline_candidate(value):
    """Small tables of scalars (or scalar lists) stay on one line."""
    def simple(item):
        if isinstance(item, (list, tuple)):
            return all(simple(x) for x in item)
        if isinstance(item, dict):
            return all(simple(x) for x in item.values())
        return not (isinstance(item, str) and '\n' in item)
    return isinstance(value, dict) and simple(value) and len(_inline(value)) <= _INLINE_WIDTH


def _table(lines, path, table, header):
    scalars, tables, arrays = [], [], []
    for name, value in table.items():
        if value is None:
            continue
        if _is_table(value) and value and not _inline_candidate(value):
            tables.append((name, value))
        elif _is_table_array(value):
            arrays.append((name, value))
        else:
            scalars.append((name, value))
    if header is not None and (scalars or not (tables or arrays)):
        lines.append(header)
    for name, value in scalars:
        rendered = _inline(value) if isinstance(value, (dict, list, tuple)) else _scalar(value)
        lines.append(f'{_key(name)} = {rendered}')
    for name, value in tables:
        child = path + [name]
        lines.append('')
        _table(lines, child, value, '[' + '.'.join(_key(p) for p in child) + ']')
    for name, value in arrays:
        child = path + [name]
        for item in value:
            lines.append('')
            lines.append('[[' + '.'.join(_key(p) for p in child) + ']]')
            _table(lines, child, item, None)


def dumps(document, *, comment=None):
    """Serialize a mapping as TOML text."""
    if not isinstance(document, dict):
        raise TypeError('A TOML document must be a mapping')
    lines = [f'# {line}'.rstrip() for line in comment.splitlines()] if comment else []
    _table(lines, [], document, None)
    text = '\n'.join(lines).lstrip('\n')
    return re.sub(r'\n{3,}', '\n\n', text) + '\n'

"""A small TOML writer for configuration tables.

The standard library reads TOML (``tomllib``) but does not write it, and the
training environment has no ``tomli_w``. Configuration values are plain JSON-like
data (tables, arrays, strings, integers, finite floats and booleans), so the
writer only needs that subset. ``tomllib.loads(dumps(value)) == value`` holds for
every value it accepts; anything else (``None``, NaN, sets, objects) is refused
rather than silently changed.
"""
import json
import math
import re

_BARE = re.compile(r"[A-Za-z0-9_-]+")
_INLINE_WIDTH = 88


def _key(key):
    if not isinstance(key, str):
        raise TypeError(f"TOML keys must be strings, got {type(key).__name__}")
    return key if _BARE.fullmatch(key) else json.dumps(key)


def _string(value):
    if "\n" in value and "'''" not in value and all(c in "\n\t" or ord(c) >= 32 and c != "\x7f" for c in value):
        # A literal multi-line string keeps HNDL/source text readable; the
        # newline right after the opening quotes is trimmed by TOML readers.
        return "'''\n" + value + "'''"
    return _basic(value)


def _basic(value):
    # TOML forbids a raw DEL; JSON escapes are otherwise valid TOML escapes.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _scalar(value, location):
    if value is None:
        raise ValueError(f"TOML has no null; omit {location} instead")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    raise TypeError(f"{location}: cannot write {type(value).__name__} to TOML")


def _inline(value, location):
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_inline(v, f'{location}.{k}')}" for k, v in value.items()) + " }" if value else "{}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_inline(v, f"{location}[{i}]") for i, v in enumerate(value)) + "]"
    if isinstance(value, str) and "\n" in value:
        return _basic(value)
    return _scalar(value, location)


def _is_table_array(value):
    return isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value)


def _prefers_inline(value, location):
    """Small flat tables (such as ``inputs``) read best on one line."""
    if not isinstance(value, dict):
        return False
    if any(isinstance(v, dict) and v or _is_table_array(v) or isinstance(v, str) and "\n" in v for v in value.values()):
        return False
    return len(_inline(value, location)) <= _INLINE_WIDTH


def _table(lines, path, table, array=False):
    # Top-level sections always get headers; nested small tables stay inline.
    scalars = [(k, v) for k, v in table.items()
               if not (isinstance(v, dict) and (not path or not _prefers_inline(v, f"{path}.{k}")))
               and not _is_table_array(v)]
    children = [(k, v) for k, v in table.items() if (k, v) not in scalars]
    header = ".".join(_key(part) for part in path)
    if path and (scalars or not children or array):
        lines.append("")
        lines.append(f"[[{header}]]" if array else f"[{header}]")
    for key, value in scalars:
        location = ".".join(path + [key])
        text = _inline(value, location) if isinstance(value, (dict, list, tuple)) else _scalar(value, location)
        lines.append(f"{_key(key)} = {text}")
    for key, value in children:
        if isinstance(value, dict):
            _table(lines, path + [key], value)
        else:
            for item in value:
                _table(lines, path + [key], item, array=True)


def dumps(value, header=None):
    """Serialize a top-level table. ``header`` lines become leading comments."""
    if not isinstance(value, dict):
        raise TypeError("A TOML document is a table")
    lines = [f"# {line}".rstrip() for line in (header or "").splitlines()]
    _table(lines, [], value)
    return "\n".join(lines).lstrip("\n") + "\n"

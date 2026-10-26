"""Standalone workspace discovery, configuration, and interpolation for HyperGAN.

A workspace is defined by a `hypergan.toml` file found by walking up directory
trees from an experiment file or current working directory. It defines common paths,
runs roots, GPU device aliases, and named execution environments.
"""

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_VAR_REGEX = re.compile(r"\$\{([a-zA-Z0-9_.-]+)\}")
_BARE_VAR_REGEX = re.compile(r"(?<!\$)\{([a-zA-Z0-9_.-]+)\}")


def find_workspace(start_path: Path | str | None = None) -> Path | None:
    """Walk up parent directories from start_path to discover hypergan.toml.

    Returns the absolute path to hypergan.toml if found, or None.
    """
    if start_path is None:
        curr = Path.cwd().resolve()
    else:
        curr = Path(start_path).resolve()
        if curr.is_file():
            curr = curr.parent

    while True:
        candidate = curr / "hypergan.toml"
        if candidate.is_file():
            return candidate
        parent = curr.parent
        if parent == curr:
            break
        curr = parent
    return None


def load_workspace(workspace_file: Path | str) -> dict:
    """Load a hypergan.toml workspace file.

    Returns a dictionary containing [workspace], [paths], [gpus], [envs],
    and optional default [run] and [checkpoints] tables.
    """
    path = Path(workspace_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Workspace file not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid workspace configuration in {path}")

    ws = data.setdefault("workspace", {})
    root_dir = str(path.parent)
    ws.setdefault("root", root_dir)

    data.setdefault("paths", {})
    data.setdefault("gpus", {})
    data.setdefault("envs", {})

    if "runs_root" in ws:
        if "{" in str(ws["runs_root"]):
            ws["runs_root"] = interpolate_variables(ws["runs_root"], data)
        runs_root = Path(ws["runs_root"])
        if not runs_root.is_absolute():
            ws["runs_root"] = str((path.parent / runs_root).resolve())
        else:
            ws["runs_root"] = str(runs_root)

    # Self-interpolate paths and workspace tables
    if "paths" in data:
        data["paths"] = interpolate_variables(data["paths"], data)
    if "runs_root" in ws and "{" in str(ws["runs_root"]):
        ws["runs_root"] = interpolate_variables(ws["runs_root"], data)
    if "run_dir" in ws and "{" in str(ws["run_dir"]):
        try:
            ws["run_dir"] = interpolate_variables(ws["run_dir"], data)
        except KeyError:
            pass

    return data


def _build_context(workspace_dict: dict, env_vars: dict | None = None) -> dict[str, str]:
    context: dict[str, str] = {}
    ws = workspace_dict.get("workspace", {})
    if "root" in ws:
        context["workspace.root"] = str(ws["root"])
        context["root"] = str(ws["root"])
    if "runs_root" in ws:
        context["workspace.runs_root"] = str(ws["runs_root"])
        context["runs_root"] = str(ws["runs_root"])

    paths = workspace_dict.get("paths", {})
    if isinstance(paths, dict):
        for k, v in paths.items():
            context[f"paths.{k}"] = str(v)
            context[k] = str(v)

    if isinstance(ws, dict):
        for k, v in ws.items():
            if k not in ("root", "runs_root"):
                context[f"workspace.{k}"] = str(v)

    if env_vars:
        for k, v in env_vars.items():
            if v is not None:
                context[str(k)] = str(v)
        if "stem" in context and "name" not in context:
            context["name"] = context["stem"]
        if "name" in context and "stem" not in context:
            context["stem"] = context["name"]

    return context


def interpolate_model_paths(spec: dict, workspace_dict: dict) -> dict:
    """Substitute workspace ``${var}`` references in a model file's path fields only.

    Path fields are the keys ``hndl``, ``file``, ``root``, ``manifest``,
    ``cache_dir``, ``weights_path``, ``source_path``, ``path`` and the entries of
    ``network_files``. Variables are ``${paths.<key>}`` (or ``${<key>}``),
    ``${workspace.root}``/``${root}`` and ``${workspace.runs_root}``/``${runs_root}``.
    Other strings are left alone: HNDL sources use ``${param}`` for their own
    template parameters, and labels may contain braces. The process environment
    is never read, so a model resolves the same way in every shell.
    """
    from .model_file import _PATH_KEYS

    context = _build_context(workspace_dict)

    def substitute(value: str, location: str) -> str:
        def lookup(match: re.Match) -> str:
            key = match.group(1)
            for candidate in (key, key[6:] if key.startswith("paths.") else f"paths.{key}"):
                if candidate in context:
                    return context[candidate]
            raise ValueError(
                f"{location}: undefined workspace variable '${{{key}}}'. "
                f"Defined paths: {sorted(workspace_dict.get('paths', {}))}; workspace fields: root, runs_root"
            )
        return _VAR_REGEX.sub(lookup, value)

    def walk(item: Any, location: str, is_path: bool) -> Any:
        if isinstance(item, str):
            return substitute(item, location) if is_path and "${" in item else item
        if isinstance(item, dict):
            return {k: walk(v, f"{location}.{k}" if location else str(k),
                            is_path or k in _PATH_KEYS) for k, v in item.items()}
        if isinstance(item, list):
            return [walk(v, f"{location}[{i}]", is_path) for i, v in enumerate(item)]
        return deepcopy(item)

    return walk(spec, "", False)


def interpolate_variables(data: Any, workspace_dict: dict, env_vars: dict | None = None) -> Any:
    """Recursively replace ${var} and known {var} in strings, dictionaries, and lists.

    Scope includes:
    - ${workspace.root}, ${workspace.runs_root}, {root}, {runs_root}
    - ${paths.<key>}, ${<key>}, {paths.<key>}, {<key>}
    - Explicit env_vars
    - Fallback to os.environ[key] (for ${var} syntax)

    Raises KeyError if an undefined ${var} variable is encountered.
    """
    context = _build_context(workspace_dict, env_vars)

    def _lookup_known(k: str) -> Any:
        if k in context:
            return context[k]
        if k.startswith("paths.") and k[6:] in context:
            return context[k[6:]]
        if f"paths.{k}" in context:
            return context[f"paths.{k}"]
        if env_vars and k in env_vars:
            return env_vars[k]
        return None

    def _interpolate_string(val: str) -> Any:
        # Check if the entire string is exactly a single ${var} variable
        single_match = _VAR_REGEX.fullmatch(val.strip())
        if single_match:
            k = single_match.group(1)
            known = _lookup_known(k)
            if known is not None:
                return known
            if k in os.environ:
                return os.environ[k]
            raise KeyError(
                f"Undefined variable in template: '${k}'. "
                f"Defined paths: {sorted(workspace_dict.get('paths', {}).keys())}, "
                f"workspace fields: ['root', 'runs_root']"
            )

        single_bare = _BARE_VAR_REGEX.fullmatch(val.strip())
        if single_bare:
            k = single_bare.group(1)
            known = _lookup_known(k)
            if known is not None:
                return known

        def _sub_dollar(m: re.Match) -> str:
            k = m.group(1)
            known = _lookup_known(k)
            if known is not None:
                return str(known)
            if k in os.environ:
                return str(os.environ[k])
            raise KeyError(
                f"Undefined variable in template: '${k}'. "
                f"Defined paths: {sorted(workspace_dict.get('paths', {}).keys())}, "
                f"workspace fields: ['root', 'runs_root']"
            )

        def _sub_bare(m: re.Match) -> str:
            k = m.group(1)
            known = _lookup_known(k)
            if known is not None:
                return str(known)
            return m.group(0)

        out = _VAR_REGEX.sub(_sub_dollar, val) if "${" in val else val
        if "{" in out:
            out = _BARE_VAR_REGEX.sub(_sub_bare, out)
        return out

    def _walk(item: Any, key: Any = None) -> Any:
        if isinstance(item, str):
            if "{" in item and key != "source":
                return _interpolate_string(item)
            return item
        elif isinstance(item, dict):
            return {k: _walk(v, k) for k, v in item.items()}
        elif isinstance(item, list):
            return [_walk(elem, key) for elem in item]
        return deepcopy(item)

    return _walk(data)


def resolve_device_aliases(
    devices: str | list[str] | None,
    workspace_dict: dict | None = None
) -> str | list[str]:
    """Map friendly GPU aliases ('gpu0', 'gpu1') through workspace [gpus] table.

    Returns the mapped devices in the same shape (str or list) as input.
    """
    if devices is None:
        return []

    gpu_map = workspace_dict.get("gpus", {}) if workspace_dict else {}

    is_str_input = isinstance(devices, str)
    if is_str_input:
        raw_items = [d.strip() for d in devices.split(",") if d.strip()]
    elif isinstance(devices, (list, tuple)):
        raw_items = [str(d).strip() for d in devices if str(d).strip()]
    elif isinstance(devices, int):
        raw_items = [str(devices)]
    else:
        raise ValueError(f"Invalid devices specification: {devices!r}")

    if len(raw_items) == 1 and raw_items[0].lower() == "auto":
        import sys
        num_cuda = 0
        if "torch" in sys.modules:
            try:
                import torch
                num_cuda = torch.cuda.device_count()
            except Exception:
                num_cuda = 0
        else:
            try:
                from .gpu_inventory import inspect_gpu_hardware
                hw = inspect_gpu_hardware()
                num_cuda = len(hw)
            except Exception:
                num_cuda = 0
        if num_cuda > 1:
            raw_items = [str(i) for i in range(num_cuda)]
        elif num_cuda == 1:
            raw_items = ["0"]
        else:
            raw_items = ["cpu"]

    resolved = []
    for item in raw_items:
        if item in gpu_map:
            resolved.append(str(gpu_map[item]))
        elif item.startswith("gpu") and item[3:].isdigit():
            if gpu_map:
                raise ValueError(
                    f"Unknown GPU alias '{item}'; defined aliases in hypergan.toml: {sorted(gpu_map.keys())}"
                )
            resolved.append(item[3:])
        elif item.startswith("gpu") and gpu_map and item not in gpu_map:
            raise ValueError(
                f"Unknown GPU alias '{item}'; defined aliases in hypergan.toml: {sorted(gpu_map.keys())}"
            )
        else:
            resolved.append(item)

    if is_str_input:
        return ",".join(resolved)
    return resolved


def default_run_dir(stem: str, *, name: str | None = None, start: Path | str | None = None) -> Path:
    """The directory for a new run when none is given; the CLI and the API both use this.

    Each call makes a fresh ``<stem>-<adj>-<noun>`` directory unless ``name`` is
    given. Inside a workspace (found from ``start``, else the current directory)
    its ``run_dir`` template decides, with ``${name}`` the directory name,
    ``${slug}`` the ``<adj>-<noun>`` part and ``${stem}`` the name or stem; without
    a template the run goes under ``runs_root``. Otherwise it goes in the current
    directory.
    """
    from .registry import generate_run_slug

    if name is not None:
        name = re.sub(r"[\\/]+", "-", str(name)).strip("-")
        if not name:
            raise ValueError("Run name must contain a character other than '/'")
        if name in (".", ".."):
            raise ValueError(f"run name {name!r} is not a directory name")
    ws_file = find_workspace(start)
    ws_data = load_workspace(ws_file) if ws_file else None
    ws_sec = ws_data.get("workspace", {}) if ws_data else {}
    runs_root = Path(ws_sec["runs_root"]).resolve() if "runs_root" in ws_sec else Path.cwd().resolve()
    dir_name, slug = (name, name) if name else generate_run_slug(stem, parent_dir=runs_root)
    if "run_dir" in ws_sec:
        return Path(interpolate_variables(
            ws_sec["run_dir"], ws_data, {"stem": name or stem, "name": dir_name, "slug": slug})).resolve()
    return (runs_root / dir_name).resolve()

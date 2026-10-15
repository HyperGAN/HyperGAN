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

    if "runs_root" in ws:
        runs_root = Path(ws["runs_root"])
        if not runs_root.is_absolute():
            ws["runs_root"] = str((path.parent / runs_root).resolve())
        else:
            ws["runs_root"] = str(runs_root)

    data.setdefault("paths", {})
    data.setdefault("gpus", {})
    data.setdefault("envs", {})

    # Self-interpolate paths and workspace tables
    context = _build_context(data)
    if "paths" in data:
        data["paths"] = interpolate_variables(data["paths"], data)
    if "runs_root" in ws and "${" in str(ws["runs_root"]):
        ws["runs_root"] = interpolate_variables(ws["runs_root"], data)
    if "run_dir" in ws and "${" in str(ws["run_dir"]):
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
            context[str(k)] = str(v)

    return context


def interpolate_variables(data: Any, workspace_dict: dict, env_vars: dict | None = None) -> Any:
    """Recursively replace ${var} in strings, dictionaries, and lists.

    Scope includes:
    - ${workspace.root}, ${workspace.runs_root}
    - ${paths.<key>} or ${<key>}
    - Explicit env_vars
    - Fallback to os.environ[key]

    Raises KeyError if an undefined variable is encountered.
    """
    context = _build_context(workspace_dict, env_vars)

    def _interpolate_string(val: str) -> Any:
        # Check if the entire string is exactly a single variable
        single_match = _VAR_REGEX.fullmatch(val.strip())
        if single_match:
            k = single_match.group(1)
            if k in context:
                return context[k]
            if k.startswith("paths.") and k[6:] in context:
                return context[k[6:]]
            if f"paths.{k}" in context:
                return context[f"paths.{k}"]
            if env_vars and k in env_vars:
                return env_vars[k]
            if k in os.environ:
                return os.environ[k]
            raise KeyError(
                f"Undefined variable in template: '${k}'. "
                f"Defined paths: {sorted(workspace_dict.get('paths', {}).keys())}, "
                f"workspace fields: ['root', 'runs_root']"
            )

        def _sub(m: re.Match) -> str:
            k = m.group(1)
            if k in context:
                return str(context[k])
            if k.startswith("paths.") and k[6:] in context:
                return str(context[k[6:]])
            if f"paths.{k}" in context:
                return str(context[f"paths.{k}"])
            if env_vars and k in env_vars:
                return str(env_vars[k])
            if k in os.environ:
                return str(os.environ[k])
            raise KeyError(
                f"Undefined variable in template: '${k}'. "
                f"Defined paths: {sorted(workspace_dict.get('paths', {}).keys())}, "
                f"workspace fields: ['root', 'runs_root']"
            )

        return _VAR_REGEX.sub(_sub, val)

    def _walk(item: Any, key: Any = None) -> Any:
        if isinstance(item, str):
            if "${" in item and key != "source":
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

    resolved = []
    for item in raw_items:
        if item in gpu_map:
            resolved.append(str(gpu_map[item]))
        elif item.startswith("gpu") and gpu_map and item not in gpu_map:
            raise ValueError(
                f"Unknown GPU alias '{item}'; defined aliases in hypergan.toml: {sorted(gpu_map.keys())}"
            )
        else:
            resolved.append(item)

    if is_str_input:
        return ",".join(resolved)
    return resolved

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
                import subprocess
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
                num_cuda = len(lines)
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


def resolve_run_dir(
    config_path: Path | str,
    workspace: dict | Path | str | None = None,
    cli_run_dir: Path | str | None = None,
    name: str | None = None,
) -> Path:
    """Resolve the target run directory for a training configuration.

    Uses cli_run_dir if explicitly provided; otherwise interpolates the workspace
    run_dir template (with ${name} resolving to `name or stem` and ${stem} to the
    config filename stem) or falls back to <runs_root>/<name or stem>.
    """
    if cli_run_dir is not None:
        return Path(cli_run_dir).resolve()

    cfg_path = Path(config_path).resolve()
    stem = cfg_path.stem
    eff_name = name or stem

    if workspace is None:
        ws_file = find_workspace(cfg_path)
        ws_data = load_workspace(ws_file) if ws_file else None
    elif isinstance(workspace, (str, Path)):
        ws_data = load_workspace(workspace)
    elif isinstance(workspace, dict):
        ws_data = workspace
    else:
        ws_data = None

    if ws_data:
        ws_sec = ws_data.get("workspace", {})
        ws_root = Path(ws_sec.get("root", Path.cwd())).resolve()
        runs_root = Path(ws_sec["runs_root"]).resolve() if "runs_root" in ws_sec else ws_root
        if "run_dir" in ws_sec:
            run_dir_str = interpolate_variables(
                ws_sec["run_dir"],
                ws_data,
                {"stem": stem, "name": eff_name, "slug": eff_name},
            )
            res_path = Path(run_dir_str)
            if not res_path.is_absolute():
                return (ws_root / res_path).resolve()
            return res_path.resolve()
        return (runs_root / eff_name).resolve()

    return (Path.cwd() / eff_name).resolve()


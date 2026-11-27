from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys


def hypergan_source(package_dir=None):
    """Read the live checkout identity or the revision stamped into a distribution.

    Missing Git/build information is explicitly unknown. A commit identifies the
    checkout base; ``hypergan_dirty`` records whether its files were modified.
    """
    package_dir = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    result = {'hypergan_commit': None, 'hypergan_dirty': None, 'hypergan_provenance': 'unknown'}
    root = package_dir.parent.parent
    if (root / '.git').exists():
        try:
            commit = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root,
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
            if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit):
                return result
            result.update(hypergan_commit=commit, hypergan_provenance='git')
            result['hypergan_dirty'] = bool(subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=root,
                text=True, stderr=subprocess.DEVNULL, timeout=5))
        except (OSError, subprocess.SubprocessError):
            pass
        return result
    try:
        value = json.loads((package_dir / '_build_provenance.json').read_text())
        if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
                or value['schema_version'] != 1):
            return result
        commit, dirty = value.get('commit'), value.get('dirty')
        if commit is not None and (not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit)):
            return result
        if dirty is not None and type(dirty) is not bool:
            return result
        return dict(hypergan_commit=commit, hypergan_dirty=dirty,
                    hypergan_provenance='build' if commit is not None else 'unknown')
    except (OSError, ValueError):
        return result


def inspect_git_repository(path):
    """Query git commit, dirty status, and provenance for a given directory."""
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    result = {'commit': None, 'dirty': None, 'provenance': 'unknown'}
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--verify', 'HEAD'], cwd=path,
            text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit):
            result['commit'] = commit
            result['provenance'] = 'git'
            result['dirty'] = bool(subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=path,
                text=True, stderr=subprocess.DEVNULL, timeout=5
            ))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def write_launch_provenance(run_dir, attempt=1, metadata=None):
    """Write RUN/launch/<attempt:04d>.json capturing launch environment and git state."""
    run_dir = Path(run_dir).resolve()
    launch_dir = run_dir / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)

    target_file = launch_dir / f"{attempt:04d}.json"
    temp_file = launch_dir / f".{attempt:04d}.json.tmp"

    data = {
        "schema_version": 1,
        "attempt": attempt,
        "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "hostname": socket.gethostname(),
        "user": getpass.getuser() if hasattr(getpass, "getuser") else os.environ.get("USER", ""),
        "git": {
            "hypergan": hypergan_source(),
        },
        "environment": {
            k: os.environ[k] for k in (
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "_HYPERGAN_REEXEC"
            ) if k in os.environ
        },
    }

    if metadata:
        for k, v in metadata.items():
            if k == "git" and isinstance(v, dict):
                data["git"].update(v)
            elif k == "environment" and isinstance(v, dict):
                data["environment"].update(v)
            else:
                data[k] = v

    if "worktree_git" not in data and metadata:
        wt = metadata.get("worktree")
        if not wt and "src_paths" in metadata:
            s_paths = metadata["src_paths"]
            if isinstance(s_paths, (list, tuple)) and s_paths:
                wt = s_paths[0]
            elif isinstance(s_paths, (str, Path)):
                wt = s_paths

        if wt:
            if isinstance(wt, (str, Path)):
                wt_path = Path(wt).resolve()
                repo_root = wt_path.parent if wt_path.name == "src" else wt_path
                git_info = inspect_git_repository(repo_root)
                data["worktree_git"] = git_info
                repo_name = repo_root.name or "worktree"
                data["git"][repo_name] = git_info
            elif isinstance(wt, (list, tuple)):
                git_dict = {}
                for item in wt:
                    item_path = Path(item).resolve()
                    repo_root = item_path.parent if item_path.name == "src" else item_path
                    repo_name = repo_root.name or "worktree"
                    git_dict[repo_name] = inspect_git_repository(repo_root)
                data["worktree_git"] = git_dict
                data["git"].update(git_dict)

    temp_file.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp_file.replace(target_file)
    return target_file


def _extract_hndl_refs(toml_path: Path) -> list[str]:
    """Collect relative or absolute .hndl paths referenced by a TOML model/recipe file."""
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    refs: list[str] = []
    networks = data.get("networks")
    if isinstance(networks, dict):
        for spec in networks.values():
            if not isinstance(spec, dict):
                continue
            for key in ("hndl", "file"):
                val = spec.get(key)
                if isinstance(val, str) and val:
                    refs.append(val)
            nf = spec.get("network_files")
            if isinstance(nf, dict):
                for val in nf.values():
                    if isinstance(val, str) and val:
                        refs.append(val)
    components = data.get("components")
    if isinstance(components, dict):
        for spec in components.values():
            if not isinstance(spec, dict):
                continue
            args = spec.get("args")
            if not isinstance(args, dict):
                continue
            val = args.get("file")
            if isinstance(val, str) and val:
                refs.append(val)
            nf = args.get("network_files")
            if isinstance(nf, dict):
                for sub_val in nf.values():
                    if isinstance(sub_val, str) and sub_val:
                        refs.append(sub_val)
    return refs


def snapshot_config_sources(run_dir, source_files, *, entry_file=None):
    """Copy raw source TOML and referenced .hndl files into RUN/config-source/ with manifest.json."""
    run_dir = Path(run_dir).resolve()
    target_dir = run_dir / "config-source"
    target_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[Path, Path] = {}
    records: list[tuple[Path, str, str]] = []  # (src_path, rel_snapshot, kind)

    def _copy_unique(src_path: Path, preferred_rel: Path | None = None) -> Path:
        if preferred_rel is not None:
            cand = target_dir / preferred_rel
            try:
                cand.resolve().relative_to(target_dir.resolve())
            except ValueError:
                cand = target_dir / src_path.name
        else:
            cand = target_dir / src_path.name
        if cand.exists() and copied.get(cand) != src_path:
            dest_name = src_path.name
            parent_prefix = src_path.parent.name
            candidate = f"{parent_prefix}_{dest_name}" if parent_prefix else f"sub_{dest_name}"
            idx = 1
            while (target_dir / candidate).exists() and copied.get(target_dir / candidate) != src_path:
                candidate = f"{parent_prefix}_{idx}_{dest_name}" if parent_prefix else f"sub_{idx}_{dest_name}"
                idx += 1
            cand = target_dir / candidate
        cand.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, cand)
        copied[cand] = src_path
        return cand

    resolved_sources: list[Path] = []
    for src in source_files or []:
        p = Path(src).resolve()
        if p.is_file():
            resolved_sources.append(p)
            target_p = _copy_unique(p)
            rel_snap = target_p.relative_to(target_dir).as_posix()
            kind = "workspace" if p.name == "hypergan.toml" else "toml"
            records.append((p, rel_snap, kind))

    # Determine the entry TOML (explicitly provided, else last non-workspace TOML from collect_source_files).
    entry_src = Path(entry_file).resolve() if entry_file is not None else None
    if entry_src is None:
        non_ws_tomls = [p for p in resolved_sources if p.suffix.lower() == ".toml" and p.name != "hypergan.toml"]
        if non_ws_tomls:
            entry_src = non_ws_tomls[-1]

    # Snapshot referenced .hndl files alongside the TOML files.
    seen_hndl: set[Path] = set()
    for toml_src in resolved_sources:
        if toml_src.suffix.lower() != ".toml" or toml_src.name == "hypergan.toml":
            continue
        for ref in _extract_hndl_refs(toml_src):
            if ref.startswith(("${", "~")):
                continue
            ref_path = Path(ref)
            src_hndl = ref_path.resolve() if ref_path.is_absolute() else (toml_src.parent / ref_path).resolve()
            if not src_hndl.is_file() or src_hndl in seen_hndl:
                continue
            seen_hndl.add(src_hndl)
            pref_rel = ref_path if (not ref_path.is_absolute() and ".." not in ref_path.parts) else None
            target_h = _copy_unique(src_hndl, preferred_rel=pref_rel)
            rel_snap = target_h.relative_to(target_dir).as_posix()
            records.append((src_hndl, rel_snap, "hndl"))

    entry_snapshot = None
    files_manifest: list[dict] = []
    for src_p, rel_snap, kind in records:
        role = "entry" if (entry_src is not None and src_p == entry_src) else ("parent" if kind == "toml" else kind)
        item = {"source": str(src_p), "snapshot": rel_snap, "role": role}
        if role == "entry":
            entry_snapshot = rel_snap
            files_manifest.insert(0, item)
        else:
            files_manifest.append(item)

    manifest_payload = {
        "schema_version": 1,
        "entry": entry_snapshot,
        "files": files_manifest,
    }
    manifest_tmp = target_dir / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(target_dir / "manifest.json")

    return target_dir


def _numbered(directory, pattern):
    """The largest number that ``pattern``'s first group captures from an entry name of directory, or 0."""
    try:
        names = [p.name for p in Path(directory).iterdir()]
    except OSError:
        return 0
    return max((int(m.group(1)) for m in map(re.compile(pattern).fullmatch, names) if m), default=0)


def determine_attempt(run_dir: Path | str) -> int:
    """The 1-based number of the run's next attempt.

    One number names both ``attempts/NNNN-<id>/`` and ``launch/NNNN.json``: it is
    one past every number either has used. A process re-executed for a run
    inherits the number its parent reserved, and already recorded under
    ``launch/``, through ``_HYPERGAN_REEXEC_ATTEMPT``, unless an attempt
    directory has used it since.
    """
    run_dir = Path(run_dir).resolve()
    attempts = _numbered(run_dir / "attempts", r"(\d+)-.+")
    reexec_att = os.environ.get("_HYPERGAN_REEXEC_ATTEMPT")
    if reexec_att:
        try:
            reserved = int(reexec_att)
        except ValueError:
            reserved = 0
        if reserved > attempts:
            return reserved
    return max(attempts, _numbered(run_dir / "launch", r"(\d{4,})\.json")) + 1

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


def snapshot_config_sources(run_dir, source_files):
    """Copy raw source TOML files into RUN/config-source/ preserving comments and formatting."""
    run_dir = Path(run_dir).resolve()
    target_dir = run_dir / "config-source"
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for src in source_files or []:
        p = Path(src).resolve()
        if p.is_file():
            dest_name = p.name
            target_p = target_dir / dest_name
            if target_p.exists() and copied.get(target_p) != p:
                parent_prefix = p.parent.name
                candidate = f"{parent_prefix}_{dest_name}" if parent_prefix else f"sub_{dest_name}"
                idx = 1
                while (target_dir / candidate).exists() and copied.get(target_dir / candidate) != p:
                    candidate = f"{parent_prefix}_{idx}_{dest_name}" if parent_prefix else f"sub_{idx}_{dest_name}"
                    idx += 1
                target_p = target_dir / candidate
            shutil.copy2(p, target_p)
            copied[target_p] = p

    return target_dir


def determine_attempt(run_dir: Path | str) -> int:
    """Determine the next 1-based attempt sequence number for the run."""
    run_dir = Path(run_dir).resolve()
    reexec_att = os.environ.get("_HYPERGAN_REEXEC_ATTEMPT")
    if reexec_att:
        try:
            return int(reexec_att)
        except ValueError:
            pass
    launch_dir = run_dir / "launch"
    if launch_dir.is_dir():
        existing = [p for p in launch_dir.glob("*.json") if re.fullmatch(r"\d{4}\.json", p.name)]
        if existing:
            return max(int(p.stem) for p in existing) + 1
    return 1


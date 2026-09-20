"""HyperGAN release provenance, independent of checkpoint compatibility."""

import json
from pathlib import Path
import re
import subprocess


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

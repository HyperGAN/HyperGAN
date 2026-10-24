"""Bundled offline browser resources; no training dependencies."""
import hashlib
from importlib.resources import files
from pathlib import Path

CORE_ASSETS = ('app.js', 'index.html', 'style.css', 'view-worker.js', 'runs.js', 'runs.css')


def compute_asset_hash(asset_dir: Path | None = None) -> str:
    """Deterministic SHA-256 digest of core web assets."""
    hasher = hashlib.sha256()
    for name in sorted(CORE_ASSETS):
        if asset_dir is not None:
            content = (Path(asset_dir) / name).read_bytes()
        else:
            try:
                content = files('hypergan.web_assets').joinpath(name).read_bytes()
            except Exception:
                dev_file = Path(__file__).resolve().parent / name
                content = dev_file.read_bytes() if dev_file.is_file() else b''
        hasher.update(name.encode('utf-8') + b'\0' + content + b'\0')
    return hasher.hexdigest()


__all__ = ['compute_asset_hash', 'CORE_ASSETS']

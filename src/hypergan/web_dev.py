"""Viewer development mode: browser assets served fresh from a working checkout.

Training starts its viewer in a detached supervised subprocess, so an inherited
environment variable is the one switch that survives every launch path. CLI
flags set it for their own process rather than adding a second mechanism.

Normal mode keeps the packaged resources and the existing response headers.
"""
import hashlib
import json
import os
from pathlib import Path

ENABLED_VARIABLE = 'HYPERGAN_VIEWER_DEV'
ASSETS_VARIABLE = 'HYPERGAN_VIEWER_ASSETS'

_ENABLED = {'1', 'true', 'yes', 'on'}
_DISABLED = {'', '0', 'false', 'no', 'off'}

RELOAD_ASSET = '/dev/reload.js'
VERSION_ROUTE = '/dev/version'

# Same-origin module; the viewer's script-src/connect-src stay 'self'.
RELOAD_SCRIPT = """// HyperGAN viewer development reload; absent unless """ + ENABLED_VARIABLE + """ is set.
let current = null;
async function poll() {
  try {
    const response = await fetch('""" + VERSION_ROUTE + """', { cache: 'no-store' });
    if (response.ok) {
      const { version } = await response.json();
      if (current === null) current = version;
      else if (version !== current) return location.reload();
    }
  } catch (error) {
    // A stopped or restarting server is expected; keep polling.
  }
  setTimeout(poll, 1000);
}
poll();
"""


def enabled(environ=None):
    """Explicit on/off; an unrecognised value is an actionable error, not 'off'."""
    raw = (os.environ if environ is None else environ).get(ENABLED_VARIABLE, '')
    value = raw.strip().lower()
    if value in _ENABLED:
        return True
    if value in _DISABLED:
        return False
    raise ValueError(f'{ENABLED_VARIABLE} must be one of 1/0/true/false/yes/no/on/off, not {raw!r}')


def enable(environ=None):
    """Turn dev mode on for this process and every child that inherits it."""
    (os.environ if environ is None else environ)[ENABLED_VARIABLE] = '1'


def checkout(start=None):
    """The repository checkout containing `start`, identified by the frontend build."""
    directory = Path.cwd() if start is None else Path(start)
    directory = directory.resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / 'frontend' / 'build.mjs').is_file() and (candidate / 'src' / 'hypergan').is_dir():
            return candidate
    return None


class DevAssets:
    """Resolved asset directories plus the response policy for one server."""

    def __init__(self, *, active=False, web_assets=None, reducer_assets=None, origin='package'):
        self.active = active
        self._web_assets = web_assets
        self._reducer_assets = reducer_assets
        self.origin = origin

    def web_asset(self, name):
        if self._web_assets is None:
            from importlib.resources import files
            return files('hypergan').joinpath('web_assets', name)
        return self._web_assets / name

    def reducer_asset(self, name):
        if self._reducer_assets is None:
            from .metrics_reducer import assets
            return assets().joinpath(name)
        return self._reducer_assets / name

    @property
    def headers(self):
        # Normal mode keeps the server-wide policy; dev mode states it per
        # response so no cache validator can turn a refresh into a 304.
        return {'Cache-Control': 'no-store'} if self.active else {}

    def version(self, names):
        """Identity of the files actually served, from their size and mtime."""
        state = []
        for name in names:
            resource = self.web_asset(name)
            path = Path(str(resource))
            try:
                status = path.stat()
                state.append([name, status.st_size, status.st_mtime_ns])
            except OSError:
                state.append([name, None, None])
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self):
        return (f'web_assets={self._web_assets or "packaged hypergan.web_assets"} '
                f'reducers={self._reducer_assets or "packaged hypergan.metrics_reducer.assets"} '
                f'({self.origin})')


def dev_assets(environ=None, cwd=None):
    """Packaged resources unless dev mode is on; then a checkout when one is found."""
    if not enabled(environ):
        return DevAssets()
    values = os.environ if environ is None else environ
    override = (values.get(ASSETS_VARIABLE) or '').strip()
    if override:
        directory = Path(override).expanduser()
        if not directory.is_dir():
            raise ValueError(f'{ASSETS_VARIABLE} must name an existing directory, not {override!r}')
        return DevAssets(active=True, web_assets=directory.resolve(), origin=ASSETS_VARIABLE)
    root = checkout(cwd)
    if root is None:
        return DevAssets(active=True, origin='package (no checkout found)')
    package = root / 'src' / 'hypergan'
    reducer = package / 'metrics_reducer' / 'assets'
    return DevAssets(active=True, web_assets=package / 'web_assets',
                     reducer_assets=reducer if reducer.is_dir() else None, origin=str(root))

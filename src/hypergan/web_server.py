"""Optional loopback ASGI server; browser and external clients share one API."""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import quote

from . import inspection as _insp_mod
from . import registry as _reg_mod
from .web_dev import RELOAD_ASSET, RELOAD_SCRIPT, VERSION_ROUTE, dev_assets
from .web_files import read_artifact, read_bytes, read_json, _open
from .web_service import (
    MAX_GROUPS,
    MAX_QUEUE_BYTES,
    MAX_STREAMS,
    MAX_SUBSCRIBERS,
    PREVIEW_INDEX_BYTES,
    MapSpec,
    ObservationService,
    ViewSpec,
    _finite_nonneg_float,
    _positive_int,
    compute_throughput_and_eta,
    cursor_offset,
    extract_fid_and_health,
    extract_topology_and_batch,
    sse,
)
from .web_session import LocalSession

STATIC_ASSETS = {'index.html', 'app.js', 'view-worker.js', 'style.css', 'runs.js', 'runs.css', 'logo.png'}


# Seconds an event stream waits for its client to accept one frame.
STREAM_SEND_DEADLINE = 5.0


class StreamSendDeadline(OSError):
    """An event-stream client stopped reading; its connection is dropped."""


def _is_send_deadline(error, depth=0):
    if error is None or depth > 8:
        return False
    if isinstance(error, StreamSendDeadline):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_is_send_deadline(item, depth + 1) for item in error.exceptions)
    return _is_send_deadline(error.__cause__ or error.__context__, depth + 1)


def _quiet_send_deadline(record):
    """Logging filter: a stalled browser tab is routine, not a server error.

    Raising out of the ASGI app is the only way to make the server close a
    connection whose client no longer drains it, and uvicorn logs that as
    'Exception in ASGI application'. Record it at debug level instead.
    """
    if record.exc_info and _is_send_deadline(record.exc_info[1]):
        logging.getLogger(__name__).debug('Closed an event stream whose client stopped reading')
        return False
    return True

_disk_cache: dict[Path, tuple[float, int]] = {}


def get_run_disk_bytes(run_dir: Path, ttl_seconds: float = 20.0) -> int:
    now = time.monotonic()
    if run_dir in _disk_cache:
        cached_time, cached_val = _disk_cache[run_dir]
        if now - cached_time < ttl_seconds:
            return cached_val
    total = 0
    try:
        for entry in run_dir.rglob('*'):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    _disk_cache[run_dir] = (now, total)
    return total


def format_disk_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _stat_watermark(path: Path) -> tuple[int, int, int]:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size, getattr(st, "st_ino", 0))
    except OSError:
        return (0, 0, 0)


def _summary_watermark(run_dir: Path) -> tuple[tuple[int, int, int], ...]:
    return (
        _stat_watermark(run_dir / "summary.json"),
        _stat_watermark(run_dir / "manifest.json"),
        _stat_watermark(run_dir / ".run.lock"),
        _stat_watermark(run_dir / "launch" / "0001.json"),
        _stat_watermark(run_dir / "previews" / "index.json"),
    )


def _disk_run_id(run_dir: Path) -> str | None:
    for relative in ("manifest.json", "summary.json"):
        try:
            loaded = read_json(run_dir, relative)
            if isinstance(loaded, dict) and loaded.get("run_id"):
                return str(loaded["run_id"])
        except (OSError, ValueError):
            pass
    return None


def _created_at(value) -> str | None:
    """Normalize recorded creation times to UTC; never infer them from mtime."""
    try:
        if type(value) in (int, float):
            date = datetime.fromtimestamp(value, timezone.utc)
        elif isinstance(value, str):
            date = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
        else:
            return None
        return date.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _run_created_at(run_dir: Path, manifest: dict, summary: dict | None) -> str | None:
    for value in (manifest.get('created_at'), (summary or {}).get('created_at')):
        if date := _created_at(value):
            return date
    # The first launch survives resumes; later launch timestamps describe attempts.
    try:
        launch = read_json(run_dir, 'launch/0001.json')
        return _created_at(launch.get('timestamp')) if isinstance(launch, dict) else None
    except (OSError, ValueError):
        return None


def _is_stale_unmodified_run(run_dir: Path, max_idle_seconds: float = 3600.0) -> bool:
    latest_mtime = 0.0
    for candidate in (run_dir / ".run.lock", run_dir / "summary.json", run_dir / "manifest.json", run_dir / "events.jsonl"):
        try:
            if candidate.exists():
                latest_mtime = max(latest_mtime, candidate.stat().st_mtime)
        except OSError:
            pass
    if latest_mtime <= 0.0:
        return False
    return (time.time() - latest_mtime) > max_idle_seconds


def read_run_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()

    summary = None
    try:
        loaded_summary = read_json(run_dir, "summary.json")
        if isinstance(loaded_summary, dict):
            summary = loaded_summary
    except (OSError, ValueError):
        summary = None

    manifest = {}
    try:
        loaded_manifest = read_json(run_dir, "manifest.json")
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest
    except (OSError, ValueError):
        manifest = {}

    host_info = _insp_mod.inspect_run_host(run_dir)
    try:
        is_active, pid = _insp_mod.check_run_active(run_dir)
    except OSError:
        is_active, pid = False, None
    if pid is None and not is_active:
        if summary and isinstance(summary.get("pid"), int):
            pid = summary["pid"]
        elif isinstance(manifest.get("pid"), int):
            pid = manifest["pid"]

    run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or run_dir.name
    name = (
        (manifest.get("config", {}).get("name") if isinstance(manifest.get("config"), dict) else None)
        or (summary.get("name") if summary else None)
        or manifest.get("name")
        or run_dir.name
    )
    slug = (
        (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or (manifest.get("config", {}).get("slug") if isinstance(manifest.get("config"), dict) else None)
        or _reg_mod.extract_or_derive_slug(str(run_id), name=str(name), run_dir_name=run_dir.name)
    )
    short_id = (
        (summary.get("short_id") if summary else None)
        or manifest.get("short_id")
        or str(run_id)[:8]
    )
    raw_status = (summary.get("status") if summary else None) or manifest.get("status") or "unknown"
    if is_active:
        status = "running"
        active = True
    elif (
        str(raw_status).strip().lower() in ("running", "initializing")
        and ((run_dir / ".run.lock").exists() or _is_stale_unmodified_run(run_dir))
    ):
        status = "stopped"
        active = False
    else:
        status = raw_status
        active = str(status).strip().lower() in ("running", "training", "initializing", "starting")
    running = bool(is_active)

    step = (summary.get("step") if summary and "step" in summary else None)
    if step is None:
        step = manifest.get("steps", 0)

    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    total_steps = (
        (summary.get("total_steps") if summary else None)
        or manifest.get("total_steps")
        or (summary.get("max_steps") if summary else None)
        or manifest.get("max_steps")
        or (summary.get("target_steps") if summary else None)
        or manifest.get("target_steps")
        or _positive_int(training_cfg.get("steps"))
    )
    max_steps = (
        (summary.get("max_steps") if summary else None)
        or manifest.get("max_steps")
        or total_steps
        or (summary.get("target_steps") if summary else None)
        or manifest.get("target_steps")
    )
    steps_per_second = _finite_nonneg_float(
        summary.get("steps_per_second") if summary else None,
        summary.get("speed_its") if summary else None,
        summary.get("steps_per_sec") if summary else None,
        manifest.get("steps_per_second"),
        manifest.get("speed_its"),
    )

    topo_batch = extract_topology_and_batch(manifest, summary)
    samples_per_second, eta_seconds = compute_throughput_and_eta(
        active=active,
        step=step,
        total_steps=total_steps,
        steps_per_second=steps_per_second,
        global_batch_size=topo_batch["global_batch_size"],
        summary=summary,
    )

    fid_health = extract_fid_and_health(manifest, summary)
    latest_fid = fid_health["latest_fid"]
    best_fid = fid_health["best_fid"]
    best_step = fid_health["best_step"]
    best_metric = fid_health["best_metric"]
    health_alerts = fid_health["health_alerts"]
    warnings = fid_health["warnings"]

    updated_at = (summary.get("updated_at") if summary else None) or manifest.get("updated_at")

    disk_bytes = get_run_disk_bytes(run_dir)
    disk_formatted = format_disk_bytes(disk_bytes)

    has_preview = False
    thumbnail_revision: str | None = None
    try:
        pdata = read_json(run_dir, "previews/index.json", max_bytes=PREVIEW_INDEX_BYTES)
        previews_list = pdata.get("previews") if isinstance(pdata, dict) else None
        if isinstance(previews_list, list) and previews_list:
            has_preview = True
            chosen = None
            for entry in reversed(previews_list):
                if isinstance(entry, dict) and isinstance(entry.get("image_grid"), dict):
                    chosen = entry
                    break
            if chosen is None:
                for entry in reversed(previews_list):
                    if isinstance(entry, dict):
                        chosen = entry
                        break
            if chosen is not None:
                grid = chosen.get("image_grid") if isinstance(chosen.get("image_grid"), dict) else {}
                raw_rev = None
                for candidate_hash in (grid.get("sha256"), chosen.get("sha256")):
                    if isinstance(candidate_hash, str):
                        cleaned_rev = re.sub(r"[^A-Za-z0-9]", "", candidate_hash)[:12]
                        if cleaned_rev:
                            raw_rev = cleaned_rev
                            break
                if not raw_rev:
                    fallback_token = json.dumps(
                        [
                            chosen.get("sample_sequence"),
                            chosen.get("step"),
                            grid.get("path") or chosen.get("path"),
                            len(previews_list),
                        ],
                        sort_keys=True,
                        default=str,
                    )
                    raw_rev = hashlib.sha256(fallback_token.encode("utf-8")).hexdigest()[:12]
                thumbnail_revision = raw_rev
    except (OSError, ValueError):
        pass
    if not has_preview and (run_dir / "previews").is_dir():
        try:
            newest_sig: tuple[int, int, str] | None = None
            for png_path in (run_dir / "previews").rglob("*.png"):
                try:
                    if png_path.is_file() and not png_path.is_symlink():
                        st = png_path.stat()
                        sig = (st.st_mtime_ns, st.st_size, png_path.name)
                        if newest_sig is None or sig > newest_sig:
                            newest_sig = sig
                except OSError:
                    continue
            if newest_sig is not None:
                has_preview = True
                thumbnail_revision = hashlib.sha256(
                    f"{newest_sig[0]}:{newest_sig[1]}".encode("utf-8")
                ).hexdigest()[:12]
        except Exception:
            pass

    if has_preview:
        base_thumb = f"/api/v1/runs/{quote(str(run_id), safe='')}/thumbnail"
        thumbnail_url = f"{base_thumb}?rev={thumbnail_revision}" if thumbnail_revision else base_thumb
    else:
        thumbnail_url = None

    return {
        "run_id": run_id,
        "short_id": short_id,
        "slug": slug,
        "name": name,
        "status": status,
        "active": active,
        "running": running,
        "pid": pid,
        "hostname": host_info.get("hostname"),
        "remote": host_info.get("remote", False),
        "topology": topo_batch["topology"],
        "global_batch_size": topo_batch["global_batch_size"],
        "local_batch_size": topo_batch["local_batch_size"],
        "microbatch_size": topo_batch["microbatch_size"],
        "accumulation_steps": topo_batch["accumulation_steps"],
        "batch_label": topo_batch["batch_label"],
        "batch_detail": topo_batch["batch_detail"],
        "step": step,
        "total_steps": total_steps,
        "max_steps": max_steps,
        "steps_per_second": steps_per_second,
        "speed_its": steps_per_second,
        "samples_per_second": samples_per_second,
        "eta_seconds": eta_seconds,
        "latest_fid": latest_fid,
        "fid_latest": latest_fid,
        "best_fid": best_fid,
        "fid_best": best_fid,
        "best_step": best_step,
        "best_metric": best_metric,
        "health_alerts": health_alerts,
        "warnings": warnings,
        "created_at": _run_created_at(run_dir, manifest, summary),
        "updated_at": updated_at,
        "disk_bytes": disk_bytes,
        "disk_formatted": disk_formatted,
        "thumbnail_revision": thumbnail_revision,
        "thumbnail_url": thumbnail_url,
        "run_dir": str(run_dir),
        "url": f"/runs/{quote(str(run_id), safe='')}",
    }


def discover_runs(root: Path) -> list[dict]:
    return ServiceRegistry(root)._discover_runs()


def _render_run_row(r: dict) -> str:
    def escape(value):
        return html.escape(str(value))

    run_id = str(r.get('run_id', 'unknown'))
    name = str(r.get('name') or run_id)
    slug = r.get('slug') or _reg_mod.extract_or_derive_slug(run_id, name=name)
    short_id = r.get('short_id') or run_id[:8]
    status = str(r.get('status') or 'unknown').strip().lower()
    active = bool(r.get('active') or r.get('running') or status == 'running')
    step = r.get('step') or 0
    total = r.get('total_steps') or r.get('max_steps') or 0
    percent = min(100, max(0, int(step / total * 100))) if total > 0 else 0
    progress = f'{step:,} / {total:,}' if total > 0 else f'{step:,}'
    speed = r.get('steps_per_second')
    speed_text = f'{speed:.1f} it/s' if isinstance(speed, (int, float)) else '—'
    created = _created_at(r.get('created_at'))
    date_text = datetime.fromisoformat(created).strftime('%b %d, %Y · %H:%M UTC') if created else 'Unknown'
    url = escape('/runs/' + quote(run_id, safe=''))

    thumb_url = r.get("thumbnail_url")
    if thumb_url:
        preview_cell = f'<td data-label="Preview" class="thumbnail-cell"><a href="{url}"><img class="run-thumbnail" src="{escape(thumb_url)}" alt="Preview of {escape(name)}" loading="lazy" width="48" height="48" /></a></td>'
    else:
        preview_cell = '<td data-label="Preview" class="thumbnail-cell"><span class="no-thumbnail" title="No previews yet">—</span></td>'

    best_fid = r.get("best_fid") or r.get("fid_best")
    latest_fid = r.get("latest_fid") or r.get("fid_latest")
    if isinstance(best_fid, (int, float)):
        if isinstance(latest_fid, (int, float)) and abs(latest_fid - best_fid) > 0.01:
            fid_text = f'<span class="best-fid" title="Best FID">{best_fid:.2f}</span> <small class="latest-fid" title="Latest FID">({latest_fid:.2f})</small>'
        else:
            fid_text = f'<span class="best-fid" title="Best FID">{best_fid:.2f}</span>'
    elif isinstance(latest_fid, (int, float)):
        fid_text = f'<span class="latest-fid" title="Latest FID">{latest_fid:.2f}</span>'
    else:
        fid_text = '<span class="no-fid">—</span>'
    fid_cell = f'<td data-label="FID" data-field="fid">{fid_text}</td>'

    return f'''<tr data-id="{escape(run_id)}" data-name="{escape(name)}" data-slug="{escape(slug)}"
      data-status="{escape(status)}" data-active="{str(active).lower()}" data-created-at="{escape(created or '')}">
      <td data-label="Run"><a class="run-link" href="{url}">{escape(name)}</a>
        <div class="run-id-sub"><span class="slug-badge">{escape(slug)}</span>
          <span class="id-badge" title="{escape(run_id)}">{escape(short_id)}</span></div></td>
      {preview_cell}
      <td data-label="Status"><span class="badge" data-status="{escape(status)}">{escape(status)}</span></td>
      {fid_cell}
      <td data-label="Created at"><time data-field="created" datetime="{escape(created or '')}"
        title="{escape(created or 'Creation time was not recorded')}">{date_text}</time></td>
      <td data-label="Progress"><div class="progress-bar-container"><span data-field="steps">{progress}</span>
        <div class="progress-bar" aria-hidden="true"{'' if total > 0 else ' hidden'}>
          <div class="progress-fill" style="width: {percent}%"></div></div></div></td>
      <td data-label="Speed" data-field="speed">{speed_text}</td>
      <td data-label="Disk" data-field="disk">{escape(r.get('disk_formatted') or '—')}</td>
      <td data-label="Actions"><div class="actions-cell"><a class="btn" href="{url}">Open run →</a>
        <button type="button" class="btn btn-checkpoint" data-action="checkpoint"{'' if active else ' hidden'}>Checkpoint</button>
        <button type="button" class="btn btn-sample" data-action="sample"{'' if active else ' hidden'}>Sample</button>
        <button type="button" class="btn btn-stop" data-action="stop"{'' if active else ' hidden'}>Stop</button>
      </div></td>
    </tr>'''


def render_run_index_html(runs: list[dict], runs_root: Path, *, login: bool = False) -> str:
    """Dashboard page; with `login` it carries no run data, only the sign-in form."""
    runs = [] if login else sorted(runs, key=lambda r: (_created_at(r.get('created_at')) or '', str(r.get('run_id', ''))), reverse=True)
    rows = "\n".join(_render_run_row(r) for r in runs)
    root = '' if login else html.escape(str(runs_root))
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HyperGAN Runs — Dashboard</title>
  <link rel="icon" type="image/png" href="/assets/logo.png">
  <link rel="stylesheet" href="/assets/runs.css">
  <script src="/assets/runs.js" defer></script>
</head>
<body>
  <main class="container">
    <header class="header">
      <a class="brand" href="/" aria-label="HyperGAN home"><img class="brand-mark" src="/assets/logo.png" width="28" height="28" alt=""> HYPERGAN</a>
      <span class="root-path" title="{root}">{root}</span>
    </header>
    <div class="page-heading"><div><h1>Training runs</h1>
      <p>Find an experiment and follow its progress.</p></div>
      <div class="refresh-controls"><span id="refresh-status" role="status">Updates every 3 seconds</span>
        <button id="refresh" type="button" class="btn">Refresh</button></div>
    </div>
    <div class="controls" role="search" aria-label="Find training runs">
      <label class="search-control" for="search">Search runs
        <input type="search" id="search" placeholder="Name, slug, or ID" autocomplete="off"></label>
      <div class="control"><label for="status-filter">Status</label><select id="status-filter">
        <option value="">All statuses</option>
        <option value="running">Running</option>
        <option value="complete">Complete</option>
        <option value="stopped">Stopped</option>
        <option value="failed">Failed</option>
        <option value="unknown">Unknown</option>
      </select></div>
      <div class="control"><label for="sort-order">Sort by</label><select id="sort-order">
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="name">Name A–Z</option>
      </select></div>
      <button id="clear-filters" type="button" class="btn" hidden>Clear filters</button>
    </div>
    <form id="login" class="login-panel"{'' if login else ' hidden'}>
      <h2>Sign in to this dashboard</h2>
      <p>Paste the session token printed in your terminal or saved in the viewer's private session file.</p>
      <label for="token">Session token</label>
      <input id="token" type="password" autocomplete="off" spellcheck="false" required>
      <button type="submit" class="btn">Sign in</button>
      <p id="login-error" role="alert"></p>
    </form>
    <p id="run-count" role="status">{len(runs)} run{'s' if len(runs) != 1 else ''}</p>
    <p id="action-status" role="status" hidden></p>
    <noscript><p>Enable JavaScript to search, filter, and receive live updates.</p></noscript>
    <div class="table-container">
      <table class="run-table" id="runs-table">
        <caption class="sr-only">Training runs with creation time, status, and progress</caption>
        <thead><tr><th scope="col">Run / Name</th><th scope="col">Preview</th><th scope="col">Status</th><th scope="col">FID</th>
          <th scope="col">Created at <span id="time-zone">(UTC)</span></th><th scope="col">Progress</th>
          <th scope="col">Speed</th><th scope="col">Disk</th><th scope="col">Actions</th></tr></thead>
        <tbody id="runs-body">{rows}</tbody>
      </table>
      <div id="empty-state" class="empty-state"{ ' hidden' if runs else ''}>
        <h2 id="empty-title">No training runs yet</h2>
        <p id="empty-description">Start a training run in this workspace. It will appear here automatically.</p>
      </div>
    </div>
    <template id="run-row-template">{_render_run_row({})}</template>
  </main>
</body>
</html>'''


class ServiceRegistry:
    def __init__(self, root: Path, poll_seconds: float = 0.25, history_timeout: float = 180.0, max_services: int = 16):
        self.root = Path(root).resolve()
        self.runs_root = self.root
        self.poll_seconds = poll_seconds
        self.history_timeout = history_timeout
        self.max_services = max_services
        self._services: OrderedDict[str, ObservationService] = OrderedDict()
        self.services = self._services
        self._summary_cache: dict[Path, tuple[tuple, dict]] = {}
        self._lock = asyncio.Lock()

    def is_single(self) -> bool:
        if (self.root / "manifest.json").is_file() or (self.root / "summary.json").is_file():
            return True
        if (self.root / "hypergan.toml").is_file() or (self.root / "runs").is_dir():
            return False
        try:
            if self.root.is_dir() and any(
                (d / "manifest.json").is_file() or (d / "summary.json").is_file()
                for d in self.root.iterdir() if d.is_dir()
            ):
                return False
        except OSError:
            pass
        from .web_autostart import find_runs_root
        if find_runs_root(self.root) != self.root:
            return True
        if (self.root / "events.jsonl").is_file() or (self.root / "launch").is_dir() or (self.root / "checkpoints").is_dir():
            return True
        try:
            if not self.root.exists():
                return True
            subdirs = [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if not subdirs:
                return True
        except OSError:
            return True
        return False

    def _iter_registered_run_dirs(self) -> list[Path]:
        """Registered runs nested under this root; a dashboard never serves another root's runs."""
        if self.is_single():
            return []
        from .web_autostart import registered_run_dirs
        return registered_run_dirs(self.root)

    def _read_cached_summary(self, run_dir: Path) -> dict:
        run_dir = Path(run_dir).resolve()
        wm = _summary_watermark(run_dir)
        cached = self._summary_cache.get(run_dir)
        is_active = False
        if (run_dir / ".run.lock").exists():
            try:
                is_active, _ = _insp_mod.check_run_active(run_dir)
            except OSError:
                is_active = False

        if cached is not None:
            cached_wm, cached_row = cached
            if not is_active and not cached_row.get("running", False) and cached_wm == wm:
                return dict(cached_row)

        row = read_run_summary(run_dir)
        self._summary_cache[run_dir] = (_summary_watermark(run_dir), dict(row))
        return row

    def _discover_runs(self) -> list[dict]:
        root = self.root
        if (root / "manifest.json").is_file():
            return [self._read_cached_summary(root)]

        candidate_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        search_roots = [root]
        if (root / "runs").is_dir():
            search_roots.append(root / "runs")
        for sroot in search_roots:
            try:
                for entry in sorted(sroot.iterdir()):
                    if entry.is_dir() and ((entry / "manifest.json").is_file() or (entry / "summary.json").is_file()):
                        resolved = entry.resolve()
                        if resolved not in seen_dirs:
                            seen_dirs.add(resolved)
                            candidate_dirs.append(resolved)
            except OSError:
                pass

        for reg_dir in self._iter_registered_run_dirs():
            if reg_dir not in seen_dirs:
                seen_dirs.add(reg_dir)
                candidate_dirs.append(reg_dir)

        results: list[dict] = []
        seen_run_ids: set[str] = set()
        for rdir in candidate_dirs:
            try:
                if not rdir.is_dir():
                    continue
                summary = self._read_cached_summary(rdir)
                rid = str(summary.get("run_id") or rdir.name)
                if rid not in seen_run_ids:
                    seen_run_ids.add(rid)
                    results.append(summary)
            except OSError:
                continue
        results.sort(
            key=lambda r: (
                bool(r.get("running") or r.get("active")),
                _created_at(r.get("created_at")) or _created_at(r.get("updated_at")) or "",
                str(r.get("run_id", "")),
            ),
            reverse=True,
        )
        return results

    def discover_runs(self) -> list[dict]:
        return self._discover_runs()

    def _matches_run_entry(self, entry: Path, run_id: str) -> bool:
        if entry.name == run_id:
            return True
        try:
            summary = self._read_cached_summary(entry)
            if run_id in (
                summary.get("run_id"),
                summary.get("slug"),
                summary.get("short_id"),
                summary.get("name"),
            ):
                return True
        except Exception:
            pass
        try:
            manifest = read_json(entry, "manifest.json")
            cfg_name = manifest.get("config", {}).get("name") if isinstance(manifest.get("config"), dict) else None
            if run_id in (
                manifest.get("run_id"),
                manifest.get("slug"),
                manifest.get("short_id"),
                manifest.get("name"),
                cfg_name,
            ):
                return True
        except Exception:
            pass
        return False

    def resolve_run_dir(self, run_id: str) -> Path | None:
        """The run directory under this root that `run_id` names, or None."""
        if not run_id:
            return None
        root_resolved = self.root
        candidate = (self.root / run_id).resolve()
        if not candidate.is_relative_to(root_resolved):
            raise FileNotFoundError(f"Unknown run identity: {run_id}")

        single = self.is_single()
        if single:
            if run_id == "run" or self._matches_run_entry(self.root, run_id):
                return self.root
        else:
            found = self._find_run_under_root(candidate, run_id)
            if found is not None:
                return found

        # Tags and other registry aliases, still only for runs under this root.
        try:
            resolved = _reg_mod.resolve_run_target(run_id, cwd=self.root, allow_latest=False)
        except (OSError, ValueError, LookupError):
            return None
        if resolved and resolved.is_dir() and ((resolved / "manifest.json").is_file() or (resolved / "summary.json").is_file()):
            res_p = resolved.resolve()
            if res_p == self.root if single else res_p.is_relative_to(root_resolved):
                return res_p
        return None

    def _find_run_under_root(self, candidate: Path, run_id: str) -> Path | None:
        # Check direct subdirectory (and runs/ subdirectory if present)
        for direct in (candidate, (self.root / "runs" / run_id).resolve()):
            if not direct.is_relative_to(self.root):
                continue
            if direct.is_dir() and not direct.is_symlink() and ((direct / "manifest.json").is_file() or (direct / "summary.json").is_file()):
                return direct

        # Scan subdirectories and registered runs
        search_dirs: list[Path] = []
        search_roots = [self.root]
        if (self.root / "runs").is_dir():
            search_roots.append(self.root / "runs")
        for sroot in search_roots:
            try:
                if sroot.is_dir():
                    for entry in sroot.iterdir():
                        if entry.is_dir() and not entry.is_symlink() and ((entry / "manifest.json").is_file() or (entry / "summary.json").is_file()):
                            resolved = entry.resolve()
                            if resolved not in search_dirs:
                                search_dirs.append(resolved)
            except OSError:
                pass
        for reg_dir in self._iter_registered_run_dirs():
            if reg_dir not in search_dirs:
                search_dirs.append(reg_dir)

        for entry in search_dirs:
            if self._matches_run_entry(entry, run_id):
                return entry
        return None

    def _retire_service(self, svc: ObservationService, reason: str) -> None:
        if svc not in self._services.values():
            svc.retire(reason)

    def _evict_stale_for_dir(self, run_dir: Path, disk_rid: str | None) -> None:
        has_files = run_dir.is_dir() and ((run_dir / "manifest.json").is_file() or (run_dir / "summary.json").is_file())
        evicted: list[ObservationService] = []
        for existing_key, existing_svc in list(self._services.items()):
            if existing_svc.root == run_dir:
                known_rid = existing_svc.run_id or getattr(existing_svc, "_expected_run_id", None)
                if known_rid is not None and (not has_files or (disk_rid is not None and known_rid != disk_rid)):
                    del self._services[existing_key]
                    evicted.append(existing_svc)
        if evicted:
            self._summary_cache.pop(run_dir, None)
            _disk_cache.pop(run_dir, None)
            for svc in evicted:
                self._retire_service(svc, "The run in this directory was replaced; reload the view")

    def _evict_lru_if_needed(self) -> list[ObservationService]:
        """Retire least recently used services beyond `max_services`.

        A service a browser is still watching is skipped while idle ones remain,
        so open tabs do not evict each other in turn. A hard cap of four times the
        limit still bounds memory; its subscribers are told to reconnect.
        """
        evicted_services: list[ObservationService] = []
        if self.max_services <= 0:
            return evicted_services
        while len(self._services) > self.max_services:
            # The most recent entry is the service being requested right now.
            older = list(self._services.items())[:-1]
            idle = next((key for key, svc in older if not svc.subscribers), None)
            if idle is None and len(self._services) <= 4 * self.max_services:
                break
            key = idle if idle is not None else older[0][0]
            evicted = self._services.pop(key)
            if evicted not in self._services.values():
                evicted_services.append(evicted)
                self._retire_service(evicted, "The viewer released this run to serve others; reconnect to resume")
        return evicted_services

    def get_service(self, run_id: str, run_dir: Path | None = None) -> ObservationService:
        """Cached service for `run_id`; projection is owned by the viewer's projector, never a request."""
        if run_id in self._services:
            svc = self._services[run_id]
            disk_rid = _disk_run_id(svc.root)
            self._evict_stale_for_dir(svc.root, disk_rid)
            if run_id in self._services:
                self._services.move_to_end(run_id)
                return self._services[run_id]

        if run_dir is None:
            run_dir = self.resolve_run_dir(run_id)
        if run_dir is None:
            raise FileNotFoundError(f"Unknown run identity: {run_id}")

        disk_rid = _disk_run_id(run_dir)
        self._evict_stale_for_dir(run_dir, disk_rid)

        svc = None
        for existing_key, existing_svc in list(self._services.items()):
            if existing_svc.root == run_dir:
                svc = existing_svc
                del self._services[existing_key]
                break

        if svc is None:
            svc = ObservationService(run_dir, poll_seconds=self.poll_seconds, history_timeout=self.history_timeout)
        if getattr(svc, "_expected_run_id", None) is None and disk_rid is not None:
            svc._expected_run_id = disk_rid

        self._services[run_id] = svc
        self._services.move_to_end(run_id)
        self._evict_lru_if_needed()
        return svc

    async def get(self, run_id: str) -> ObservationService:
        async with self._lock:
            was_cached = run_id in self._services
            # Resolution scans the root and reads summaries: keep it off the loop.
            run_dir = None if was_cached else await asyncio.to_thread(self.resolve_run_dir, run_id)
            if not was_cached and run_dir is None:
                raise FileNotFoundError(f"Unknown run identity: {run_id}")
            svc = self.get_service(run_id, run_dir)
            if svc._discovery is None:
                await svc.start()
            elif svc.run_id is None and (svc.root / "manifest.json").is_file():
                await svc._activate()
            elif not was_cached or not any(k.startswith("projection:") for k in svc.streams):
                await svc.discover()
            return svc

    async def close_all(self):
        async with self._lock:
            unique = set(self._services.values())
            self._services.clear()
            for svc in unique:
                try:
                    await svc.close()
                except Exception:
                    pass


CONTROL_ACTIONS = {'checkpoint': 'request_checkpoint', 'stop': 'request_stop', 'sample': 'request_sample'}

# Every per-run route, mounted under both `/api/runs` and `/api/v1/runs`. The
# auth layer reads the same table, so a route cannot be served without it.
RUN_ROUTES = (
    ('', 'run', None),
    ('/summary', 'run_summary', None),
    ('/status', 'run', None),
    ('/stop', 'control:stop', ['POST']),
    ('/checkpoint', 'control:checkpoint', ['POST']),
    ('/sample', 'control:sample', ['POST']),
    ('/requests/{request_id}', 'request_status', None),
    ('/metrics', 'run_metrics', None),
    ('/metrics/catalog', 'catalog', None),
    ('/contributions', 'contributions', None),
    ('/model', 'model', None),
    ('/architecture', 'model', None),
    ('/events', 'events', None),
    ('/views', 'views', None),
    ('/views/{map_revision}/bootstrap', 'bootstrap', None),
    ('/stream', 'stream', None),
    ('/artifacts', 'artifacts', None),
    ('/artifacts/{artifact_id}', 'artifacts', None),
    ('/thumbnail', 'thumbnail', None),
)
RUN_PREFIXES = ('/api/runs', '/api/v1/runs')


def create_app(run_dir, session, *, poll_seconds=.25, history_timeout=180, dev=None):
    """Construct without importing torch, running maps or computing measurements."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response, StreamingResponse
        from starlette.routing import Match, Route
    except ImportError as exc:
        raise RuntimeError('Local serving requires hypergan[web]') from exc
    shutdown_event = asyncio.Event()

    class ClosingStream(StreamingResponse):
        """SSE that also ends on viewer shutdown or a client that stops reading.

        Only the ASGI contract is used: Starlette ends the response when
        `receive` yields `http.disconnect`, and this wrapper yields one as well
        on shutdown or after a missed send deadline, then completes the response.
        Starlette only listens on `receive` while the server advertises ASGI
        spec_version < 2.4 (uvicorn 0.54 reports 2.3); a server advertising 2.4
        would skip this shutdown wakeup and rely on send failing instead.
        """
        async def __call__(self, scope, receive, send):
            stop = asyncio.Event()
            progress = {'started': False, 'complete': False}

            async def stream_receive():
                waiters = [asyncio.ensure_future(receive()), asyncio.ensure_future(stop.wait()),
                           asyncio.ensure_future(shutdown_event.wait())]
                try:
                    done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    return waiters[0].result() if waiters[0] in done else {'type': 'http.disconnect'}
                finally:
                    for waiter in waiters:
                        waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)

            async def deadline_send(message):
                if message['type'] == 'http.response.start':
                    progress['started'] = True
                    return await send(message)
                progress['complete'] = not message.get('more_body', False)
                try:
                    await asyncio.wait_for(send(message), timeout=STREAM_SEND_DEADLINE)
                except asyncio.TimeoutError as exc:
                    stop.set()
                    raise StreamSendDeadline('Stream client exceeded the send deadline') from exc
                # Yield once per frame so a lost connection is observed before
                # a replay burst writes the next one.
                await asyncio.sleep(0)

            try:
                await super().__call__(scope, stream_receive, deadline_send)
                if progress['started'] and not progress['complete']:
                    await deadline_send({'type': 'http.response.body', 'body': b'', 'more_body': False})
            finally:
                await self.body_iterator.aclose()

    root = Path(run_dir).resolve()
    registry = ServiceRegistry(root, poll_seconds=poll_seconds, history_timeout=history_timeout)
    single_service = None
    if registry.is_single():
        single_service = ObservationService(root, poll_seconds=poll_seconds, history_timeout=history_timeout)
        registry.services[root.name] = single_service

    dev = dev_assets() if dev is None else dev
    if dev.active:
        import sys
        print(f'Viewer development mode: assets read per request from {dev.describe()}',
              file=sys.stderr, flush=True)

    @asynccontextmanager
    async def lifespan(app):
        if single_service is not None:
            await single_service.start()
        try:
            yield
        finally:
            shutdown_event.set()
            await registry.close_all()
            if single_service is not None:
                await single_service.close()

    def authenticated(request):
        return request.scope.get('state', {}).get('authenticated', False)

    async def session_exchange(request):
        if request.headers.get('content-type', '').split(';')[0] != 'application/json':
            return JSONResponse({'error': 'Expected application/json'}, status_code=415)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                return JSONResponse({'error': 'Session body exceeds 4096 bytes'}, status_code=413)
        try:
            value = json.loads(raw)
            if type(value) is not dict or set(value) != {'token'}:
                raise ValueError('Expected token object')
            cookie = session.exchange(value['token'])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({'error': 'Invalid or expired viewer credential'}, status_code=401)
        result = JSONResponse({'authenticated': True})
        secure = session.cookie_secure and request.scope.get('state', {}).get('viewer_channel') == 'public'
        result.set_cookie(session.cookie_name, cookie, httponly=True, samesite='strict',
                          secure=secure, max_age=86400, path='/')
        return result

    def capabilities_payload():
        from . import __version__
        from .provenance import hypergan_source
        from .web_assets import compute_asset_hash
        from .metrics_reducer import descriptor
        from .web_service import MAX_BOOTSTRAP_BYTES

        source = hypergan_source()
        runs = registry._discover_runs()
        if registry.is_single() and single_service is not None:
            # Read only: the service registers its own streams when it activates.
            run_id = single_service.run_id
            status = single_service.manifest.get('status', 'waiting')
        else:
            run_id = None
            status = 'ready' if runs else 'waiting'
        return {
            'schema_version': 1,
            'api_version': 'v1',
            'run_id': run_id,
            'status': status,
            'server_instance_id': session.instance_id,
            'version': __version__,
            'git_commit': source.get('hypergan_commit'),
            'git_dirty': source.get('hypergan_dirty'),
            'asset_hash': compute_asset_hash(dev._web_assets if dev.active else None),
            'runs_root': str(root),
            'run_count': len(runs),
            'auth_mode': session.auth_mode,
            'public_origin': session.public_origin,
            'transports': ['sse'],
            'controls': list(CONTROL_ACTIONS),
            'reducer': descriptor(),
            'limits': {
                'subscribers': MAX_SUBSCRIBERS, 'queue_bytes': MAX_QUEUE_BYTES, 'groups': MAX_GROUPS,
                'bootstrap_bytes': MAX_BOOTSTRAP_BYTES, 'stream_count': MAX_STREAMS,
                'history_seconds': history_timeout
            }
        }

    async def capabilities(request):
        return JSONResponse(await asyncio.to_thread(capabilities_payload))

    async def root_index(request):
        if await asyncio.to_thread(registry.is_single):
            return await static(request)
        login = not authenticated(request)
        runs = [] if login else await asyncio.to_thread(registry._discover_runs)
        html_content = render_run_index_html(runs, root, login=login)
        return Response(html_content, media_type='text/html', headers={'Cache-Control': 'no-store'})

    async def run_detail_page(request):
        # The shell is public; only a signed-in viewer learns whether a run exists.
        if authenticated(request):
            run_id = request.path_params.get('run_id')
            if await asyncio.to_thread(registry.resolve_run_dir, run_id) is None:
                raise FileNotFoundError(f'Unknown run identity: {run_id}')
        return await static(request, asset_override='index.html')

    async def runs_list(request):
        return JSONResponse(await asyncio.to_thread(registry._discover_runs))

    async def _read_control_body(request) -> dict:
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                raise ValueError('Control body exceeds 4096 bytes')
        if not raw or not bytes(raw).strip():
            return {}
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Invalid JSON control body') from exc
        if not isinstance(body, dict) or not set(body).issubset({'request_id', 'attempt_id'}):
            raise ValueError('Control body must be a JSON object with optional request_id and attempt_id')
        return body

    def _run_active(target_dir: Path) -> bool:
        try:
            return bool(_insp_mod.check_run_active(target_dir)[0])
        except OSError:
            return False

    def control_endpoint(action):
        """One handler for every durable control request; stop also signals the trainer."""
        async def endpoint(request):
            from . import api as _api_mod
            from .run_requests import request_status
            run_id = request.path_params['run_id']

            def failure(message):
                return JSONResponse({'ok': False, 'action': action, 'run_id': run_id, 'error': message}, status_code=400)

            target_dir = await asyncio.to_thread(registry.resolve_run_dir, run_id)
            if target_dir is None:
                raise FileNotFoundError(f'Unknown run identity: {run_id}')
            try:
                body = await _read_control_body(request)
            except ValueError as exc:
                return failure(str(exc))
            request_id, attempt_id = body.get('request_id'), body.get('attempt_id')

            def reply(receipt, result):
                receipt_request = receipt.get('request') if isinstance(receipt, dict) else None
                payload = {
                    'ok': True, 'action': action, 'run_id': run_id, 'run_dir': str(target_dir),
                    'request_id': ((receipt_request or {}).get('request_id')
                                   or (result.get('request_id') if isinstance(result, dict) else None)
                                   or request_id),
                    'status': (receipt if isinstance(receipt, dict) else result or {}).get('status', 'pending'),
                    'receipt': receipt, 'result': result,
                }
                if action == 'stop':
                    payload.update(remote=bool(result.get('remote')), signaled=bool(result.get('signaled')))
                return JSONResponse(payload)

            if not await asyncio.to_thread(_run_active, target_dir):
                # A retry of a request the run already acknowledged keeps its
                # receipt; anything else is refused rather than queued for no one.
                receipt = None
                if request_id is not None:
                    try:
                        receipt = await asyncio.to_thread(request_status, target_dir, request_id)
                    except (OSError, ValueError, KeyError):
                        receipt = None
                if not isinstance(receipt, dict) or receipt.get('request', {}).get('action') != action:
                    return failure(f"Run '{run_id}' is not active")
                return reply(receipt, receipt)

            submit = getattr(_api_mod, CONTROL_ACTIONS[action])
            receipt = None
            try:
                receipt = await asyncio.to_thread(submit, target_dir, attempt_id=attempt_id, request_id=request_id)
            except (OSError, ValueError, RuntimeError) as exc:
                message = str(exc)
                initializing = 'initializing' in message.lower() or not (target_dir / 'manifest.json').is_file()
                # A trainer still in step 0 has no request inbox yet, but stop can signal it.
                if action != 'stop' or not initializing:
                    if initializing:
                        message = f'Run is still initializing step 0; {action} will be available once step 1 starts'
                    return failure(message)
            if action != 'stop':
                registry._summary_cache.pop(target_dir, None)
                return reply(receipt, receipt)
            stop_id = request_id or (receipt['request']['request_id'] if isinstance(receipt, dict) else None)
            try:
                result = await asyncio.to_thread(_insp_mod.stop_run, target_dir, request_id=stop_id, attempt_id=attempt_id)
            except (OSError, ValueError, RuntimeError) as exc:
                if receipt is None:
                    return failure(str(exc))
                result = {'status': 'stopping', 'run_id': run_id, 'run_dir': str(target_dir), 'receipt': receipt}
            if receipt is None and isinstance(result.get('receipt'), dict):
                receipt = result['receipt']
            registry._summary_cache.pop(target_dir, None)
            return reply(receipt, result)
        return endpoint

    async def request_status_endpoint(request):
        from . import api as _api_mod
        run_id = request.path_params['run_id']
        request_id = request.path_params['request_id']
        target_dir = await asyncio.to_thread(registry.resolve_run_dir, run_id)
        if target_dir is None:
            raise FileNotFoundError(f'Unknown run identity: {run_id}')
        receipt = await asyncio.to_thread(_api_mod.request_status, target_dir, request_id)
        return JSONResponse(receipt)

    async def run_summary_endpoint(request):
        svc = await registry.get(request.path_params['run_id'])
        summary = await asyncio.to_thread(registry._read_cached_summary, svc.root)
        pub = svc.public_manifest()
        return JSONResponse({**pub, **summary})

    async def run_metrics_summary(request):
        svc = await registry.get(request.path_params['run_id'])
        summary = await asyncio.to_thread(registry._read_cached_summary, svc.root)
        catalog_data = {}
        try:
            from .metrics import read_catalog
            revision = request.query_params.get('revision') or svc.manifest.get('metrics_catalog')
            catalog_data = await asyncio.to_thread(
                read_catalog, svc.root, revision,
                _open_file=lambda path: _open(svc.root, path.relative_to(svc.root).as_posix())
            )
        except Exception:
            catalog_data = {}
        return JSONResponse({
            'run_id': svc.run_id or request.path_params['run_id'],
            'summary': summary,
            'catalog': catalog_data,
            'metrics': catalog_data.get('metrics', {}) if isinstance(catalog_data, dict) else {},
        })

    async def run_contributions(request):
        """One bounded page of projected contributions; follow `cursor` while `has_more`."""
        svc = await registry.get(request.path_params['run_id'])
        cursor = request.query_params.get('cursor')
        limit = int(request.query_params.get('limit', '100'))
        if not 1 <= limit <= 1000:
            raise ValueError('limit must be in 1..1000')
        frames, _, page = await asyncio.to_thread(svc.page, 'projection:' + MapSpec().revision, cursor, limit=limit)
        return JSONResponse({'run_id': svc.run_id or request.path_params['run_id'], 'contributions': frames,
                             'cursor': page['cursor'], 'has_more': page['has_more']})

    async def run(request):
        svc = await registry.get(request.path_params['run_id'])
        return JSONResponse(svc.public_manifest())

    async def model(request):
        svc = await registry.get(request.path_params['run_id'])
        return JSONResponse(await asyncio.to_thread(svc.model_summary))

    async def catalog(request):
        svc = await registry.get(request.path_params['run_id'])
        from .metrics import read_catalog
        revision = request.query_params.get('revision') or svc.manifest.get('metrics_catalog')
        result = await asyncio.to_thread(read_catalog, svc.root, revision,
            _open_file=lambda path: _open(svc.root, path.relative_to(svc.root).as_posix()))
        return JSONResponse(result)

    async def events(request):
        svc = await registry.get(request.path_params['run_id'])
        cursor = request.query_params.get('cursor')
        stream_id = request.query_params.get('stream_id', 'training')
        limit = int(request.query_params.get('limit', '100'))
        if not 1 <= limit <= 1000:
            raise ValueError('limit must be in 1..1000')
        _, _, result = await asyncio.to_thread(svc.page, stream_id, cursor, limit=limit)
        return JSONResponse(result)

    async def views(request):
        svc = await registry.get(request.path_params['run_id'])
        mapping = MapSpec()
        view = ViewSpec(mapping.revision)
        available = ['projection:' + mapping.revision in svc.streams]
        return JSONResponse({'schema_version': 1, 'map_revision': mapping.revision,
                             'view_revision': view.revision, 'status': 'available' if available[0] else 'missing',
                             'descriptor': view.descriptor(), 'discovery_error': svc.discovery_error,
                             'streams': [{'stream_id': s.stream_id, 'cursor': s.cursor,
                                          'caught_up': s.caught_up, 'error': s.error} for s in svc.streams.values()]})

    async def bootstrap(request):
        svc = await registry.get(request.path_params['run_id'])
        params = request.query_params
        series = params.get('series', '').split(',')
        kwargs = dict(series=series, bucket_steps=int(params.get('bucket_steps', '1')),
                      step_from=int(params.get('step_from', '0')))
        if 'step_to' in params:
            kwargs['step_to'] = int(params['step_to'])
        try:
            job = svc.bootstrap(request.path_params['map_revision'], **kwargs)
        except LookupError as exc:
            return JSONResponse({'status': 'indexing', 'reason': str(exc)}, status_code=202)
        if job.error:
            return JSONResponse({'status': 'error', 'error': job.error, 'job_id': job.key}, status_code=422)
        if job.result is None:
            return JSONResponse({'status': 'reducing', 'job_id': job.key}, status_code=202)
        return JSONResponse(job.result)

    async def stream(request):
        run_id = request.path_params.get('run_id')
        if run_id:
            svc = await registry.get(run_id)
        elif single_service is not None and await asyncio.to_thread(registry.is_single):
            # A root that was empty at startup becomes a dashboard once runs appear.
            svc = single_service
        else:
            async def root_events():
                yield sse('ready', {'status': 'ready'})
                while True:
                    await asyncio.sleep(15.0)
                    yield sse('ping', {})
            return ClosingStream(root_events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

        stream_id = request.query_params.get('stream_id', '*')
        cursor = request.query_params.get('cursor')
        # Refuse a bad selector, a full viewer or a malformed cursor as HTTP 400,
        # before the event stream starts.
        svc.check_subscription(stream_id)
        cursor_offset(cursor)
        return ClosingStream(svc.events(stream_id, cursor), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    async def artifacts(request):
        svc = await registry.get(request.path_params['run_id'])
        await svc.refresh_artifacts()
        artifact_id = request.path_params.get('artifact_id')
        if artifact_id is None:
            return JSONResponse(svc.artifact_index())
        record = svc.artifacts.get(artifact_id)
        if record is None:
            raise FileNotFoundError('Artifact is not indexed')
        if record.get('status') == 'unavailable':
            raise ValueError(record['reason'])
        data = await asyncio.to_thread(read_artifact, svc.root, record)
        if record.get('media_type') == 'image/png' and record.get('modality') == 'image':
            from .image_grids import inspect_png
            await asyncio.to_thread(inspect_png, data)
            return Response(data, media_type='image/png',
                            headers={'Content-Disposition': 'inline; filename="grid.png"'})
        if record.get('media_type') == 'application/json':
            return Response(data, media_type='application/json',
                            headers={'Content-Disposition': 'attachment; filename="samples.json"'})
        return Response(data, media_type='application/octet-stream',
                        headers={'Content-Disposition': 'attachment; filename="artifact.bin"'})

    async def thumbnail_endpoint(request):
        run_id = request.path_params['run_id']
        svc = await registry.get(run_id)
        target_dir = svc.root
        if target_dir is None or not target_dir.is_dir():
            raise FileNotFoundError(f"Unknown run directory for run: {run_id}")

        target_root = target_dir.resolve()

        def find_png_bytes() -> bytes | None:
            from .image_grids import inspect_png
            try:
                data = read_json(target_root, "previews/index.json", max_bytes=PREVIEW_INDEX_BYTES)
                for prev in reversed(data.get("previews", [])):
                    grid_info = prev.get("image_grid")
                    if isinstance(grid_info, dict) and isinstance(grid_info.get("path"), str):
                        try:
                            raw_bytes = read_bytes(target_root, grid_info["path"], max_bytes=16 * 1048576)
                            inspect_png(raw_bytes)
                            return raw_bytes
                        except Exception:
                            continue
            except (OSError, ValueError):
                pass
            previews_dir = (target_root / "previews").resolve()
            if previews_dir.is_dir() and previews_dir.is_relative_to(target_root):
                try:
                    pngs = sorted(previews_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    for p in pngs:
                        try:
                            rel = p.relative_to(target_root).as_posix()
                            raw_bytes = read_bytes(target_root, rel, max_bytes=16 * 1048576)
                            inspect_png(raw_bytes)
                            return raw_bytes
                        except Exception:
                            continue
                except Exception:
                    pass
            return None

        png_data = await asyncio.to_thread(find_png_bytes)
        if png_data is None:
            raise FileNotFoundError("No preview thumbnail available for this run")
        rev = (request.query_params.get("rev") or "").strip()
        cache_control = "private, max-age=3600" if rev else "no-cache"
        return Response(png_data, media_type="image/png", headers={"Cache-Control": cache_control})

    async def static(request, asset_override=None):
        asset = asset_override or request.path_params.get('asset', 'index.html')
        if asset not in STATIC_ASSETS:
            raise FileNotFoundError('Unknown static asset')
        resource = dev.web_asset(asset)
        if not resource.is_file():
            if asset == 'index.html':
                return Response('HyperGAN viewer assets are not installed.', media_type='text/plain', status_code=503)
            raise FileNotFoundError('Unknown static asset')
        media = (
            'text/html' if asset.endswith('.html')
            else 'text/css' if asset.endswith('.css')
            else 'image/png' if asset.endswith('.png')
            else 'text/javascript'
        )
        body = resource.read_bytes()
        if dev.active and asset == 'index.html':
            body = body.replace(b'</body>', f'<script type="module" src="{RELOAD_ASSET}"></script>\n  </body>'.encode(), 1)
        return Response(body, media_type=media, headers=dev.headers)

    async def reducer_asset(request):
        asset = request.path_params['asset']
        if asset not in {'host.js', 'client.js', 'worker.js', 'reducer.wasm', 'reducer.json'}:
            raise FileNotFoundError('Unknown reducer asset')
        media = 'application/wasm' if asset.endswith('.wasm') else 'application/json' if asset.endswith('.json') else 'text/javascript'
        return Response(dev.reducer_asset(asset).read_bytes(), media_type=media, headers=dev.headers)

    async def dev_version(request):
        """Identity of the served browser assets, so a dev page can reload itself."""
        return JSONResponse({'version': dev.version(sorted(STATIC_ASSETS))}, headers=dev.headers)

    async def dev_reload(request):
        return Response(RELOAD_SCRIPT, media_type='text/javascript', headers=dev.headers)

    async def openapi(request):
        return JSONResponse(openapi_schema(cookie_name=session.cookie_name, auth_mode=session.auth_mode))

    async def error_handler(request, exc):
        code = 404 if isinstance(exc, FileNotFoundError) else 400
        return JSONResponse({'error': str(exc)}, status_code=code, headers={'Cache-Control': 'no-store'})

    run_handlers = {'run': run, 'run_summary': run_summary_endpoint, 'request_status': request_status_endpoint,
                    'run_metrics': run_metrics_summary, 'catalog': catalog, 'contributions': run_contributions,
                    'model': model, 'events': events, 'views': views, 'bootstrap': bootstrap,
                    'stream': stream, 'artifacts': artifacts, 'thumbnail': thumbnail_endpoint,
                    **{'control:' + action: control_endpoint(action) for action in CONTROL_ACTIONS}}
    # (path, endpoint, methods, public). Public routes are the HTML shell, its
    # static assets and the credential exchange; in token mode everything else
    # under /api requires the bearer or the session cookie.
    table = [
        ('/', root_index, None, True),
        ('/runs/{run_id}', run_detail_page, None, True),
        ('/runs/{run_id}/{path:path}', run_detail_page, None, True),
        ('/assets/{asset}', static, None, True),
        ('/reducers/{asset}', reducer_asset, None, True),
        ('/api/v1/session', session_exchange, ['POST'], True),
        ('/api/v1/capabilities', capabilities, None, False),
        ('/api/v1/openapi.json', openapi, None, False),
        ('/api/v1/stream', stream, None, False),
    ]
    for prefix in RUN_PREFIXES:
        table.append((prefix, runs_list, None, False))
        table.extend((prefix + '/{run_id}' + suffix, run_handlers[name], methods, False)
                     for suffix, name, methods in RUN_ROUTES)
    if dev.active:
        table.extend([(VERSION_ROUTE, dev_version, None, True), (RELOAD_ASSET, dev_reload, None, True)])
    routes = [Route(path, endpoint, methods=methods) for path, endpoint, methods, _ in table]
    public_routes = {id(route) for route, entry in zip(routes, table) if entry[3]}

    def is_public(scope):
        for route in routes:
            if route.matches(scope)[0] is Match.FULL:
                return id(route) in public_routes
        return False

    app = Starlette(routes=routes, lifespan=lifespan,
                    exception_handlers={ValueError: error_handler, FileNotFoundError: error_handler})
    app.state.observations = single_service
    app.state.registry = registry
    app.state.shutdown_event = shutdown_event

    class AuthenticatedApp:
        state = app.state

        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http':
                return await app(scope, receive, send)
            request = Request(scope, receive=receive)
            channel = session.match_request(request.headers.get('host'), request.headers.get('origin'),
                                            forwarded_proto=request.headers.get('x-forwarded-proto'),
                                            forwarded_host=request.headers.get('x-forwarded-host'))
            if channel is None:
                return await JSONResponse({'error': 'Invalid local Host or Origin; list another name for this machine '
                                                     'in HYPERGAN_VIEWER_HOSTS or name a proxy with --public-origin'},
                                           status_code=403)(scope, receive, send)
            signed_in = session.authenticated(authorization=request.headers.get('authorization'),
                                              cookie=request.cookies.get(session.cookie_name))
            state = scope.setdefault('state', {})
            state['viewer_channel'] = channel
            state['authenticated'] = signed_in
            if not signed_in and not is_public(scope):
                return await JSONResponse({'error': 'Viewer credential required'}, status_code=401)(scope, receive, send)

            async def secure_send(message):
                if message['type'] == 'http.response.start':
                    headers = list(message.get('headers', []))
                    present = {name.lower() for name, _ in headers}
                    headers.extend([(b'x-content-type-options', b'nosniff'),
                                    (b'referrer-policy', b'no-referrer'),
                                    (b'content-security-policy', b"default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self'; worker-src 'self'; object-src 'none'; form-action 'self'; frame-ancestors 'none'")])
                    if b'cache-control' not in present:
                        headers.append((b'cache-control', b'no-store'))
                    message = dict(message, headers=headers)
                await send(message)
            return await app(scope, receive, secure_send)
    return AuthenticatedApp()


def openapi_schema(*, auth_mode='token', cookie_name=LocalSession.cookie_name):
    """Versioned machine-readable route contract; rich event schemas are separate."""
    paths = {}
    descriptions = {
        '/capabilities': 'Authenticated server identity and resource limits',
        '/runs': 'Discovered runs under the root directory with their summaries',
        '/runs/{run_id}': 'Current run manifest',
        '/runs/{run_id}/summary': 'Current run summary and status metadata',
        '/runs/{run_id}/status': 'Current run manifest and live status',
        '/runs/{run_id}/metrics': 'Run metrics summary and definition catalog',
        '/runs/{run_id}/metrics/catalog': 'Immutable metric definition catalog',
        '/runs/{run_id}/contributions': 'Projected view contributions for the run',
        '/runs/{run_id}/model': 'Run model: GAN formulation, per-player losses with metric ids, optimizers, prior, '
                                'data/training settings and networks (HNDL source; per-layer detail when model.json '
                                'was recorded); 404 when the run records no configuration',
        '/runs/{run_id}/architecture': 'Run model architecture summary (alias of /model)',
        '/runs/{run_id}/events': 'Bounded raw source or projection frame page',
        '/runs/{run_id}/views': 'Available view descriptors and stream watermarks',
        '/runs/{run_id}/views/{map_revision}/bootstrap': 'Historical shared-reducer state through a fixed projection cursor; 202 while indexing/reducing',
        '/runs/{run_id}/stream': 'SSE frame/control stream; reconnect only from last applied cursor',
        '/stream': 'SSE run discovery and future stream registration',
        '/runs/{run_id}/artifacts': 'Explicit artifact index',
        '/runs/{run_id}/artifacts/{artifact_id}': 'Bounded digest-verified indexed artifact download',
        '/runs/{run_id}/requests/{request_id}': 'Durable control request acknowledgement receipt',
    }
    for route, description in descriptions.items():
        params = [{'name': name, 'in': 'path', 'required': True, 'schema': {'type': 'string'}}
                  for name in ('run_id', 'map_revision', 'artifact_id', 'request_id') if '{' + name + '}' in route]
        paths['/api/v1' + route] = {'get': {'description': description, 'parameters': params,
            'security': [{'bearerAuth': []}, {'cookieAuth': []}],
            'responses': {'200': {'description': 'Success'}, '202': {'description': 'Background indexing or reduction'},
                          '400': {'description': 'Invalid query or cursor'}, '401': {'description': 'Credential required'},
                          '404': {'description': 'Run, projection or artifact not available'}}}}
    paths['/api/runs'] = {'get': {'description': 'Discovered runs under the root directory with their summaries', 'parameters': [],
        'security': [{'bearerAuth': []}, {'cookieAuth': []}],
        'responses': {'200': {'description': 'Success'}, '202': {'description': 'Background indexing or reduction'},
                      '400': {'description': 'Invalid query or cursor'}, '401': {'description': 'Credential required'},
                      '404': {'description': 'Run, projection or artifact not available'}}}}
    run_id_param = [{'name': 'run_id', 'in': 'path', 'required': True, 'schema': {'type': 'string'}}]
    for action_route, action_desc in (
        ('/api/v1/runs/{run_id}/checkpoint', 'Request an immediate checkpoint on an active run'),
        ('/api/v1/runs/{run_id}/stop', 'Request a graceful stop on an active run'),
        ('/api/v1/runs/{run_id}/sample', 'Request an immediate preview/sample on an active run'),
    ):
        paths[action_route] = {'post': {'description': action_desc, 'parameters': run_id_param,
            'security': [{'bearerAuth': []}, {'cookieAuth': []}],
            'responses': {'200': {'description': 'Action requested'}, '400': {'description': 'Run is not active or still initializing'},
                          '401': {'description': 'Credential required'}, '404': {'description': 'Run not found'}}}}
    paths['/api/v1/session'] = {'post': {'description': 'Exchange local bearer for HttpOnly SameSite session cookie',
        'requestBody': {'required': True, 'content': {'application/json': {'schema': {'type': 'object', 'required': ['token'],
                        'additionalProperties': False, 'properties': {'token': {'type': 'string'}}}}}},
        'responses': {'200': {'description': 'Authenticated'}, '401': {'description': 'Invalid credential'}}}}
    if auth_mode == 'none':
        for path in paths.values():
            for operation in path.values():
                operation.pop('security', None)
    from .web_schema import enrich
    return enrich({'openapi': '3.1.0', 'info': {'title': 'HyperGAN local observation API', 'version': '1.0.0'},
            'paths': paths, 'components': {'securitySchemes': {
                'bearerAuth': {'type': 'http', 'scheme': 'bearer'},
                'cookieAuth': {'type': 'apiKey', 'in': 'cookie', 'name': cookie_name}}}})


def run_socket(run_dir, bound_socket, session, *, history_seconds=180):
    """Serve a prebound loopback socket; the caller owns credentials/lifecycle."""
    import uvicorn
    app = create_app(run_dir, session, history_timeout=history_seconds)
    config = uvicorn.Config(app, host=session.bind_host, port=bound_socket.getsockname()[1],
                            access_log=False, log_level='warning', timeout_graceful_shutdown=1)
    logging.getLogger('uvicorn.error').addFilter(_quiet_send_deadline)

    class _Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            super().handle_exit(sig, frame)
            shutdown_event = getattr(app.state, 'shutdown_event', None)
            if shutdown_event is not None:
                try:
                    asyncio.get_running_loop().call_soon_threadsafe(shutdown_event.set)
                except RuntimeError:
                    pass

        async def shutdown(self, sockets=None):
            shutdown_event = getattr(app.state, 'shutdown_event', None)
            if shutdown_event is not None:
                shutdown_event.set()
            await super().shutdown(sockets=sockets)

    server = _Server(config)
    server.run(sockets=[bound_socket])

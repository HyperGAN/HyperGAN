"""Optional loopback ASGI server; browser and external clients share one API."""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import html
import inspect
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

from . import inspection as _insp_mod
from . import registry as _reg_mod
from .web_autostart import ensure_run_projected, is_projection_stale_or_missing, project_run
from .web_dev import RELOAD_ASSET, RELOAD_SCRIPT, VERSION_ROUTE, dev_assets
from .web_files import read_artifact, read_json, _open
from .web_service import MAX_STREAMS, ObservationService, RunObservationService, MapSpec, ViewSpec
from .web_session import LocalSession

STATIC_ASSETS = {'index.html', 'app.js', 'view-worker.js', 'style.css', 'styles.css', 'uplot.js', 'uplot.css', 'runs.js', 'runs.css', 'logo.png'}

_disk_cache: dict[Path, tuple[float, int]] = {}
_ORIG_LIST_REGISTERED_RUNS = getattr(_reg_mod, "list_registered_runs", None)
_ORIG_CHECK_RUN_ACTIVE = getattr(_insp_mod, "check_run_active", None)
_ORIG_STOP_RUN = getattr(_insp_mod, "stop_run", None)


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
    )


def _disk_run_id(run_dir: Path) -> str | None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("run_id"):
                return str(loaded["run_id"])
        except (OSError, json.JSONDecodeError, UnicodeError):
            pass
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("run_id"):
                return str(loaded["run_id"])
        except (OSError, json.JSONDecodeError, UnicodeError):
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
        launch = json.loads((run_dir / 'launch' / '0001.json').read_text(encoding='utf-8'))
        return _created_at(launch.get('timestamp')) if isinstance(launch, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def read_run_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    summary = None
    if summary_path.is_file():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, dict):
                summary = loaded_summary
        except (OSError, json.JSONDecodeError, UnicodeError):
            summary = None

    manifest = {}
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        except (OSError, json.JSONDecodeError, UnicodeError):
            manifest = {}

    is_active = False
    pid = None
    if (
        (run_dir / ".run.lock").exists()
        or (run_dir / "launch").is_dir()
        or getattr(_insp_mod, "check_run_active", None) is not _ORIG_CHECK_RUN_ACTIVE
    ):
        try:
            is_active, pid = _insp_mod.check_run_active(run_dir)
        except Exception:
            is_active, pid = False, None
    else:
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
    status = "running" if is_active else raw_status
    active = bool(is_active or status == "running" or (summary and (summary.get("active") or summary.get("running"))))
    running = bool(is_active)

    step = (summary.get("step") if summary and "step" in summary else None)
    if step is None:
        step = manifest.get("steps", 0)

    total_steps = (
        (summary.get("total_steps") if summary else None)
        or manifest.get("total_steps")
        or (summary.get("max_steps") if summary else None)
        or manifest.get("max_steps")
        or (summary.get("target_steps") if summary else None)
        or manifest.get("target_steps")
    )
    max_steps = (
        (summary.get("max_steps") if summary else None)
        or manifest.get("max_steps")
        or total_steps
        or (summary.get("target_steps") if summary else None)
        or manifest.get("target_steps")
    )
    steps_per_second = (
        ((summary.get("steps_per_second") or summary.get("speed_its") or summary.get("steps_per_sec")) if summary else None)
        or manifest.get("steps_per_second")
        or manifest.get("speed_its")
    )

    latest_fid = None
    if summary:
        latest_fid = summary.get("latest_fid") or summary.get("fid_latest")
        if latest_fid is None and isinstance(summary.get("metrics"), dict):
            latest_fid = summary["metrics"].get("fid50k_train") or summary["metrics"].get("fid")

    best = summary.get("best") if summary and isinstance(summary.get("best"), dict) else None
    best_fid = (best.get("value") if best else None) or (summary.get("best_fid") if summary else None) or (summary.get("fid_best") if summary else None)
    best_step = (best.get("step") if best else None) or (summary.get("best_step") if summary else None)

    updated_at = (summary.get("updated_at") if summary else None) or manifest.get("updated_at")

    disk_bytes = get_run_disk_bytes(run_dir)
    disk_formatted = format_disk_bytes(disk_bytes)

    return {
        "run_id": run_id,
        "short_id": short_id,
        "slug": slug,
        "name": name,
        "status": status,
        "active": active,
        "running": running,
        "pid": pid,
        "step": step,
        "total_steps": total_steps,
        "max_steps": max_steps,
        "steps_per_second": steps_per_second,
        "speed_its": steps_per_second,
        "latest_fid": latest_fid,
        "fid_latest": latest_fid,
        "best_fid": best_fid,
        "fid_best": best_fid,
        "best_step": best_step,
        "created_at": _run_created_at(run_dir, manifest, summary),
        "updated_at": updated_at,
        "disk_bytes": disk_bytes,
        "disk_formatted": disk_formatted,
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
    return f'''<tr data-id="{escape(run_id)}" data-name="{escape(name)}" data-slug="{escape(slug)}"
      data-status="{escape(status)}" data-active="{str(active).lower()}" data-created-at="{escape(created or '')}">
      <td data-label="Run"><a class="run-link" href="{url}">{escape(name)}</a>
        <div class="run-id-sub"><span class="slug-badge">{escape(slug)}</span>
          <span class="id-badge" title="{escape(run_id)}">{escape(short_id)}</span></div></td>
      <td data-label="Status"><span class="badge" data-status="{escape(status)}">{escape(status)}</span></td>
      <td data-label="Created at"><time data-field="created" datetime="{escape(created or '')}"
        title="{escape(created or 'Creation time was not recorded')}">{date_text}</time></td>
      <td data-label="Progress"><div class="progress-bar-container"><span data-field="steps">{progress}</span>
        <div class="progress-bar" aria-hidden="true"{'' if total > 0 else ' hidden'}>
          <div class="progress-fill" style="width: {percent}%"></div></div></div></td>
      <td data-label="Speed" data-field="speed">{speed_text}</td>
      <td data-label="Disk" data-field="disk">{escape(r.get('disk_formatted') or '—')}</td>
      <td data-label="Actions"><div class="actions-cell"><a class="btn" href="{url}">Open run →</a>
        <button type="button" class="btn btn-checkpoint" data-action="checkpoint"{'' if active else ' hidden'}>Checkpoint</button>
        <button type="button" class="btn btn-stop" data-action="stop"{'' if active else ' hidden'}>Stop</button>
      </div></td>
    </tr>'''


def render_run_index_html(runs: list[dict], runs_root: Path) -> str:
    runs = sorted(runs, key=lambda r: (_created_at(r.get('created_at')) or '', str(r.get('run_id', ''))), reverse=True)
    rows = "\n".join(_render_run_row(r) for r in runs)
    root = html.escape(str(runs_root))
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
    <p id="run-count" role="status">{len(runs)} run{'s' if len(runs) != 1 else ''}</p>
    <p id="action-status" role="status" hidden></p>
    <noscript><p>Enable JavaScript to search, filter, and receive live updates.</p></noscript>
    <div class="table-container">
      <table class="run-table" id="runs-table">
        <caption class="sr-only">Training runs with creation time, status, and progress</caption>
        <thead><tr><th scope="col">Run / Name</th><th scope="col">Status</th>
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
        if (self.root / "manifest.json").is_file():
            return True
        try:
            if self.root.is_dir() and any((d / "manifest.json").is_file() for d in self.root.iterdir() if d.is_dir()):
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
        entries = []
        list_fn = getattr(_reg_mod, "list_registered_runs", None)
        is_patched = list_fn is not None and list_fn is not _ORIG_LIST_REGISTERED_RUNS
        has_backfill_kw = False
        if list_fn is not None and not is_patched:
            try:
                has_backfill_kw = "backfill_cwd" in inspect.signature(list_fn).parameters
            except Exception:
                has_backfill_kw = False

        if list_fn is not None and (is_patched or has_backfill_kw):
            try:
                entries = list_fn(backfill_cwd=False)
            except TypeError:
                try:
                    entries = list_fn()
                except Exception:
                    entries = []
            except Exception:
                entries = []
        else:
            try:
                reg_dir = _reg_mod.get_registry_dir(self.root)
                data = _reg_mod._read_registry_unlocked(reg_dir)
                entries = list(data.get("runs", {}).values())
            except Exception:
                try:
                    data = _reg_mod._read_registry_unlocked()
                    entries = list(data.get("runs", {}).values())
                except Exception:
                    entries = []


        dirs: list[Path] = []
        seen: set[Path] = set()
        for item in entries or []:
            rdir_val = item.get("run_dir") if isinstance(item, dict) else (str(item) if isinstance(item, (str, Path)) else None)
            if not rdir_val:
                continue
            try:
                p = Path(rdir_val).resolve()
                if p not in seen and p.is_dir() and ((p / "manifest.json").is_file() or (p / "summary.json").is_file()):
                    seen.add(p)
                    dirs.append(p)
            except OSError:
                continue
        return dirs

    def _read_cached_summary(self, run_dir: Path) -> dict:
        run_dir = Path(run_dir).resolve()
        wm = _summary_watermark(run_dir)
        cached = self._summary_cache.get(run_dir)
        is_active = False
        if (
            (run_dir / ".run.lock").exists()
            or getattr(_insp_mod, "check_run_active", None) is not _ORIG_CHECK_RUN_ACTIVE
        ):
            try:
                is_active, _ = _insp_mod.check_run_active(run_dir)
            except Exception:
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
        try:
            for entry in sorted(root.iterdir()):
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
        if not run_id:
            return None
        root_resolved = self.root.resolve()
        candidate = (self.root / run_id).resolve()
        try:
            if not candidate.is_relative_to(root_resolved):
                raise FileNotFoundError(f"Unknown run identity: {run_id}")
        except ValueError:
            raise FileNotFoundError(f"Unknown run identity: {run_id}")

        if self.is_single() and not self._iter_registered_run_dirs():
            if run_id == "run" or self._matches_run_entry(self.root, run_id):
                return self.root
            return None

        if (self.root / "manifest.json").is_file():
            if run_id == "run" or self._matches_run_entry(self.root, run_id):
                return self.root

        # Check direct subdirectory
        if candidate.is_dir() and not candidate.is_symlink() and ((candidate / "manifest.json").is_file() or (candidate / "summary.json").is_file()):
            return candidate

        # Scan subdirectories and registered runs
        search_dirs: list[Path] = []
        try:
            if self.root.is_dir():
                for entry in self.root.iterdir():
                    if entry.is_dir() and not entry.is_symlink() and ((entry / "manifest.json").is_file() or (entry / "summary.json").is_file()):
                        search_dirs.append(entry.resolve())
        except OSError:
            pass
        for reg_dir in self._iter_registered_run_dirs():
            if reg_dir not in search_dirs:
                search_dirs.append(reg_dir)

        for entry in search_dirs:
            if self._matches_run_entry(entry, run_id):
                return entry

        if not self.is_single():
            try:
                resolved = _reg_mod.resolve_run_target(run_id, cwd=self.root, allow_latest=False)
                if resolved and resolved.is_dir() and ((resolved / "manifest.json").is_file() or (resolved / "summary.json").is_file()):
                    res_p = resolved.resolve()
                    if res_p.is_relative_to(root_resolved) or bool(os.environ.get("HYPERGAN_REGISTRY_DIR")):
                        return res_p
            except Exception:
                pass
        return None

    def _retire_service(self, svc: ObservationService) -> None:
        if svc in self._services.values():
            return
        svc._closed = True
        tasks = [s.task for s in svc.streams.values()] + [job.task for job in svc.jobs.values()]
        if svc._discovery:
            tasks.append(svc._discovery)
        for task in tasks:
            if task:
                try:
                    task.cancel()
                except Exception:
                    pass
        if hasattr(svc, "_executor"):
            try:
                svc._executor.shutdown(wait=False)
            except Exception:
                pass

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
                self._retire_service(svc)

    def _evict_lru_if_needed(self) -> list[ObservationService]:
        evicted_services: list[ObservationService] = []
        if self.max_services <= 0:
            return evicted_services
        while len(self._services) > self.max_services:
            _, evicted = self._services.popitem(last=False)
            if evicted not in self._services.values():
                evicted_services.append(evicted)
                self._retire_service(evicted)
        return evicted_services

    def get_service(self, run_id: str) -> ObservationService:
        if run_id in self._services:
            svc = self._services[run_id]
            disk_rid = _disk_run_id(svc.root)
            self._evict_stale_for_dir(svc.root, disk_rid)
            if run_id in self._services:
                self._services.move_to_end(run_id)
                svc = self._services[run_id]
                ensure_run_projected(svc.root)
                return svc

        run_dir = self.resolve_run_dir(run_id)
        if run_dir is None:
            raise FileNotFoundError(f"Unknown run identity: {run_id}")

        disk_rid = _disk_run_id(run_dir)
        self._evict_stale_for_dir(run_dir, disk_rid)

        ensure_run_projected(run_dir)

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
            svc = self.get_service(run_id)
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


def create_app(run_dir, session, *, poll_seconds=.25, history_timeout=180, dev=None):
    """Construct without importing torch, running maps or computing measurements."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response, StreamingResponse
        from starlette.routing import Route
    except ImportError as exc:
        raise RuntimeError('Local serving requires hypergan[web]') from exc
    shutdown_event = asyncio.Event()

    def _is_transport_closing(cycle):
        if cycle is None:
            return False
        if getattr(cycle, 'disconnected', False):
            return True
        transport = getattr(cycle, 'transport', None)
        if transport is None:
            return False
        if getattr(transport, '_conn_lost', 0) or getattr(transport, '_closing', False):
            return True
        is_closing = getattr(transport, 'is_closing', None)
        return bool(is_closing and is_closing())

    def _mark_cycle_disconnected(cycle, stream_stop=None):
        if stream_stop is not None:
            stream_stop.set()
        if cycle is not None and not getattr(cycle, 'disconnected', False):
            cycle.disconnected = True
            msg_event = getattr(cycle, 'message_event', None)
            if msg_event is not None:
                msg_event.set()

    class ClosingStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            state = scope.setdefault('state', {})
            cycle = state.get('_uvicorn_cycle')
            stream_stop = state.setdefault('_stream_stop', asyncio.Event())
            was_keep_alive = bool(getattr(cycle, 'keep_alive', False))

            def _stopping():
                if shutdown_event.is_set() or stream_stop.is_set():
                    return True
                if _is_transport_closing(cycle):
                    return True
                if was_keep_alive and not getattr(cycle, 'keep_alive', True):
                    return True
                return False

            async def stream_receive():
                receive_task = asyncio.create_task(receive())
                stop_task = asyncio.create_task(stream_stop.wait())
                shutdown_task = asyncio.create_task(shutdown_event.wait())
                try:
                    while not _stopping():
                        done, _ = await asyncio.wait(
                            {receive_task, stop_task, shutdown_task},
                            timeout=0.05,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if receive_task in done:
                            return receive_task.result()
                    _mark_cycle_disconnected(cycle, stream_stop)
                    transport = getattr(cycle, 'transport', None)
                    if transport is not None and hasattr(transport, 'close'):
                        transport.close()
                    return {'type': 'http.disconnect'}
                finally:
                    for task in (receive_task, stop_task, shutdown_task):
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

            try:
                return await super().__call__(scope, stream_receive, send)
            finally:
                await self.body_iterator.aclose()

    root = Path(run_dir).resolve()
    registry = ServiceRegistry(root, poll_seconds=poll_seconds, history_timeout=history_timeout)
    single_service = ObservationService(root, poll_seconds=poll_seconds, history_timeout=history_timeout)
    if registry.is_single():
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

    async def capabilities(request):
        from . import __version__
        from .provenance import hypergan_source
        from .web_assets import compute_asset_hash
        from .metrics_reducer import descriptor

        source = hypergan_source()
        manifest_file = root / "manifest.json"
        if manifest_file.is_file():
            if single_service.run_id is None:
                try:
                    m = read_json(root, "manifest.json")
                    if m.get("run_id"):
                        single_service.run_id = m["run_id"]
                        single_service.manifest = m
                        registry.services[m["run_id"]] = single_service
                except Exception:
                    pass

        runs = registry._discover_runs()
        if registry.is_single():
            run_id = single_service.run_id
            status = single_service.manifest.get('status', 'waiting')
        else:
            run_id = None
            status = 'ready' if runs else 'waiting'

        return JSONResponse({
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
            'controls': [],
            'reducer': descriptor(),
            'limits': {
                'subscribers': 32, 'queue_bytes': 1048576, 'groups': 2048,
                'bootstrap_bytes': 1048576, 'stream_count': MAX_STREAMS,
                'history_seconds': history_timeout
            }
        })

    async def root_index(request):
        if registry.is_single() and not registry._iter_registered_run_dirs():
            ensure_run_projected(root)
            return await static(request)
        runs = registry._discover_runs()
        html_content = render_run_index_html(runs, root)
        return Response(html_content, media_type='text/html', headers={'Cache-Control': 'no-store'})

    async def run_detail_page(request):
        run_id = request.path_params.get('run_id')
        target_dir = registry.resolve_run_dir(run_id)
        if target_dir is None:
            raise FileNotFoundError(f'Unknown run identity: {run_id}')
        ensure_run_projected(target_dir)
        return await static(request, asset_override='index.html')

    async def runs_list(request):
        return JSONResponse(registry._discover_runs())

    def _is_run_considered_active(target_dir: Path, *, allow_status_when_patched: bool = False) -> tuple[bool, int | None]:
        try:
            is_active, pid = _insp_mod.check_run_active(target_dir)
        except Exception:
            is_active, pid = False, None
        if is_active:
            return True, pid
        if allow_status_when_patched and getattr(_insp_mod, "check_run_active", None) is _ORIG_CHECK_RUN_ACTIVE:
            summary = read_run_summary(target_dir)
            if summary.get("status") == "running" and not (target_dir / ".run.lock").exists():
                return True, summary.get("pid")
        return False, pid

    async def stop_run_endpoint(request):
        run_id = request.path_params['run_id']
        target_dir = registry.resolve_run_dir(run_id)
        if target_dir is None:
            raise FileNotFoundError(f'Unknown run identity: {run_id}')
        stop_patched = getattr(_insp_mod, "stop_run", None) is not _ORIG_STOP_RUN
        is_active, _ = _is_run_considered_active(target_dir, allow_status_when_patched=stop_patched)
        if not is_active:
            return JSONResponse({'ok': False, 'action': 'stop', 'run_id': run_id, 'error': f"Run '{run_id}' is not active"}, status_code=400)
        res = await asyncio.to_thread(_insp_mod.stop_run, target_dir)
        registry._summary_cache.pop(target_dir, None)
        return JSONResponse({'ok': True, 'action': 'stop', 'run_id': run_id, 'run_dir': str(target_dir), 'result': res})

    async def checkpoint_run_endpoint(request):
        from . import api as _api_mod
        from . import checkpoints as _ckpt_mod
        run_id = request.path_params['run_id']
        target_dir = registry.resolve_run_dir(run_id)
        if target_dir is None:
            raise FileNotFoundError(f'Unknown run identity: {run_id}')
        ckpt_fn = getattr(_ckpt_mod, 'request_checkpoint', None) or getattr(_api_mod, 'request_checkpoint', None)
        is_active, _ = _is_run_considered_active(target_dir, allow_status_when_patched=True)
        if not is_active:
            return JSONResponse({'ok': False, 'action': 'checkpoint', 'run_id': run_id, 'error': f"Run '{run_id}' is not active"}, status_code=400)
        try:
            res = await asyncio.to_thread(ckpt_fn, target_dir)
        except Exception as exc:
            msg = str(exc)
            if 'initializing' in msg.lower() or not (target_dir / 'manifest.json').is_file():
                msg = 'Run is still initializing step 0; checkpoint will be available once step 1 starts'
            return JSONResponse({'ok': False, 'action': 'checkpoint', 'run_id': run_id, 'error': msg}, status_code=400)
        registry._summary_cache.pop(target_dir, None)
        return JSONResponse({'ok': True, 'action': 'checkpoint', 'run_id': run_id, 'run_dir': str(target_dir), 'result': res})

    async def run_summary_endpoint(request):
        svc = await registry.get(request.path_params['run_id'])
        summary = registry._read_cached_summary(svc.root)
        pub = svc.public_manifest()
        return JSONResponse({**pub, **summary})

    async def run_metrics_summary(request):
        svc = await registry.get(request.path_params['run_id'])
        summary = registry._read_cached_summary(svc.root)
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
        svc = await registry.get(request.path_params['run_id'])
        mapping = MapSpec()
        contrib_path = svc.root / 'views' / mapping.revision / 'contributions.jsonl'
        if not contrib_path.is_file():
            contrib_path = svc.root / 'views' / 'default' / 'contributions.jsonl'
        items = []
        if contrib_path.is_file():
            for line in contrib_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
        return JSONResponse({'run_id': svc.run_id or request.path_params['run_id'], 'contributions': items})

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
        elif registry.is_single():
            svc = registry.services.get(root.name) or next(iter(registry.services.values()), None)
            if not svc:
                svc = await registry.get(root.name)
        else:
            async def root_events():
                yield 'event: ready\ndata: {"status": "ready"}\n\n'.encode('utf-8')
                while True:
                    await asyncio.sleep(15.0)
                    yield 'event: ping\ndata: {}\n\n'.encode('utf-8')
            return ClosingStream(root_events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

        stream_id = request.query_params.get('stream_id', '*')
        cursor = request.query_params.get('cursor')
        temporary, _ = svc.subscribe(stream_id)
        svc.subscribers.remove(temporary)
        if cursor is not None:
            from .web_service import cursor_offset
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

    routes = [
        Route('/', root_index),
        Route('/runs/{run_id}', run_detail_page),
        Route('/runs/{run_id}/{path:path}', run_detail_page),
        Route('/assets/{asset}', static),
        Route('/reducers/{asset}', reducer_asset),
        Route('/api/runs', runs_list),
        Route('/api/runs/{run_id}', run),
        Route('/api/runs/{run_id}/summary', run_summary_endpoint),
        Route('/api/runs/{run_id}/status', run),
        Route('/api/runs/{run_id}/stop', stop_run_endpoint, methods=['POST']),
        Route('/api/runs/{run_id}/checkpoint', checkpoint_run_endpoint, methods=['POST']),
        Route('/api/runs/{run_id}/metrics', run_metrics_summary),
        Route('/api/runs/{run_id}/metrics/catalog', catalog),
        Route('/api/runs/{run_id}/contributions', run_contributions),
        Route('/api/runs/{run_id}/model', model),
        Route('/api/runs/{run_id}/architecture', model),
        Route('/api/runs/{run_id}/events', events),
        Route('/api/runs/{run_id}/views', views),
        Route('/api/runs/{run_id}/views/{map_revision}/bootstrap', bootstrap),
        Route('/api/runs/{run_id}/stream', stream),
        Route('/api/runs/{run_id}/artifacts', artifacts),
        Route('/api/runs/{run_id}/artifacts/{artifact_id}', artifacts),
        Route('/api/v1/session', session_exchange, methods=['POST']),
        Route('/api/v1/capabilities', capabilities),
        Route('/api/v1/openapi.json', openapi),
        Route('/api/v1/runs', runs_list),
        Route('/api/v1/stream', stream),
        Route('/api/v1/runs/{run_id}', run),
        Route('/api/v1/runs/{run_id}/summary', run_summary_endpoint),
        Route('/api/v1/runs/{run_id}/status', run),
        Route('/api/v1/runs/{run_id}/stop', stop_run_endpoint, methods=['POST']),
        Route('/api/v1/runs/{run_id}/checkpoint', checkpoint_run_endpoint, methods=['POST']),
        Route('/api/v1/runs/{run_id}/metrics', run_metrics_summary),
        Route('/api/v1/runs/{run_id}/metrics/catalog', catalog),
        Route('/api/v1/runs/{run_id}/contributions', run_contributions),
        Route('/api/v1/runs/{run_id}/model', model),
        Route('/api/v1/runs/{run_id}/architecture', model),
        Route('/api/v1/runs/{run_id}/events', events),
        Route('/api/v1/runs/{run_id}/views', views),
        Route('/api/v1/runs/{run_id}/views/{map_revision}/bootstrap', bootstrap),
        Route('/api/v1/runs/{run_id}/stream', stream),
        Route('/api/v1/runs/{run_id}/artifacts', artifacts),
        Route('/api/v1/runs/{run_id}/artifacts/{artifact_id}', artifacts),
    ]
    if dev.active:
        routes.extend([Route(VERSION_ROUTE, dev_version), Route(RELOAD_ASSET, dev_reload)])
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
                return await JSONResponse({'error': 'Invalid local Host or Origin'}, status_code=403)(scope, receive, send)
            cycle = getattr(send, '__self__', None)
            state = scope.setdefault('state', {})
            state['viewer_channel'] = channel
            state['_uvicorn_cycle'] = cycle
            stream_stop = state.setdefault('_stream_stop', asyncio.Event())
            path = scope['path']
            public = (path == '/' or path.startswith('/assets/') or path.startswith('/reducers/')
                      or path == '/api/v1/session' or path == '/api/v1/runs'
                      or path == '/api/runs' or path.startswith('/api/runs/')
                      or path.startswith('/runs/')
                      or (dev.active and path.startswith('/dev/')))
            if not public and not session.authenticated(authorization=request.headers.get('authorization'),
                                                       cookie=request.cookies.get(session.cookie_name)):
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
                if _is_transport_closing(cycle):
                    _mark_cycle_disconnected(cycle, stream_stop)
                    if message['type'] == 'http.response.body' and scope['path'].endswith('/stream'):
                        await asyncio.sleep(0)
                    return
                if message['type'] == 'http.response.body' and scope['path'].endswith('/stream'):
                    try:
                        await asyncio.wait_for(send(message), timeout=5)
                    except asyncio.TimeoutError as exc:
                        raise OSError('Stream client exceeded the send deadline') from exc
                    if _is_transport_closing(cycle):
                        _mark_cycle_disconnected(cycle, stream_stop)
                    await asyncio.sleep(0)
                else:
                    await send(message)
                    if _is_transport_closing(cycle):
                        _mark_cycle_disconnected(cycle, stream_stop)
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
    }
    for route, description in descriptions.items():
        params = [{'name': name, 'in': 'path', 'required': True, 'schema': {'type': 'string'}}
                  for name in ('run_id', 'map_revision', 'artifact_id') if '{' + name + '}' in route]
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

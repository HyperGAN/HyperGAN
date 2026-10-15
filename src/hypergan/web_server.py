"""Optional loopback ASGI server; browser and external clients share one API."""
import asyncio
from contextlib import asynccontextmanager
import html
import json
from pathlib import Path
import time

from .web_dev import RELOAD_ASSET, RELOAD_SCRIPT, VERSION_ROUTE, dev_assets
from .web_files import read_artifact, read_json, _open
from .web_service import ObservationService, MapSpec, ViewSpec
from .web_session import LocalSession

STATIC_ASSETS = {'index.html', 'app.js', 'view-worker.js', 'style.css', 'styles.css', 'uplot.js', 'uplot.css'}

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


def read_run_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    summary = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            summary = None

    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            manifest = {}

    run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or run_dir.name
    name = (manifest.get("config", {}).get("name") if isinstance(manifest.get("config"), dict) else None) or (summary.get("name") if summary else None) or run_dir.name
    status = (summary.get("status") if summary else None) or manifest.get("status") or "unknown"

    step = (summary.get("step") if summary and "step" in summary else None)
    if step is None:
        step = manifest.get("steps", 0)

    total_steps = (summary.get("total_steps") if summary else None) or manifest.get("total_steps")
    steps_per_second = ((summary.get("steps_per_second") or summary.get("speed_its")) if summary else None) or manifest.get("steps_per_second") or manifest.get("speed_its")

    latest_fid = None
    if summary:
        latest_fid = summary.get("latest_fid") or summary.get("fid_latest")
        if latest_fid is None and isinstance(summary.get("metrics"), dict):
            latest_fid = summary["metrics"].get("fid50k_train") or summary["metrics"].get("fid")

    best = summary.get("best") if summary and isinstance(summary.get("best"), dict) else None
    best_fid = (best.get("value") if best else None) or (summary.get("best_fid") if summary else None) or (summary.get("fid_best") if summary else None)
    best_step = (best.get("step") if best else None) or (summary.get("best_step") if summary else None)

    disk_bytes = get_run_disk_bytes(run_dir)
    disk_formatted = format_disk_bytes(disk_bytes)

    return {
        "run_id": run_id,
        "name": name,
        "status": status,
        "step": step,
        "total_steps": total_steps,
        "steps_per_second": steps_per_second,
        "speed_its": steps_per_second,
        "latest_fid": latest_fid,
        "fid_latest": latest_fid,
        "best_fid": best_fid,
        "fid_best": best_fid,
        "best_step": best_step,
        "disk_bytes": disk_bytes,
        "disk_formatted": disk_formatted,
        "url": f"/runs/{run_id}",
    }


def discover_runs(root: Path) -> list[dict]:
    root = Path(root).resolve()
    if (root / "manifest.json").is_file():
        return [read_run_summary(root)]
    results = []
    try:
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and (entry / "manifest.json").is_file():
                results.append(read_run_summary(entry))
    except OSError:
        pass
    return results


def render_run_index_html(runs: list[dict], runs_root: Path) -> str:
    rows = []
    for r in runs:
        run_id = html.escape(str(r.get("run_id", "unknown")))
        name = html.escape(str(r.get("name", run_id)))
        status = html.escape(str(r.get("status", "unknown")))
        step = r.get("step", 0)
        total_steps = r.get("total_steps")
        sps = r.get("steps_per_second")
        speed_str = f"{sps:.1f} it/s" if sps is not None else "—"
        latest_fid_val = r.get("latest_fid")
        latest_fid_str = f"{latest_fid_val:.2f}" if latest_fid_val is not None else "—"
        best_fid_val = r.get("best_fid")
        best_step_val = r.get("best_step")
        if best_fid_val is not None:
            if best_step_val is not None:
                step_suffix = f" @ {best_step_val / 1000:.1f}k" if best_step_val >= 1000 else f" @ step {best_step_val:,}"
            else:
                step_suffix = ""
            best_fid_str = f"{best_fid_val:.2f}{step_suffix}"
        else:
            best_fid_str = "—"
        disk_str = html.escape(str(r.get("disk_formatted", "—")))
        url = html.escape(str(r.get("url", f"/runs/{run_id}")))

        if total_steps and total_steps > 0:
            pct = min(100, max(0, int((step / total_steps) * 100)))
            progress_html = (f'<div class="progress-bar-container">'
                             f'<span>{step:,} / {total_steps:,}</span>'
                             f'<div class="progress-bar"><div class="progress-fill" style="width: {pct}%"></div></div>'
                             f'</div>')
        else:
            progress_html = f'<span>{step:,}</span>'

        rows.append(
            f'<tr data-status="{status}" data-name="{name}" data-id="{run_id}">'
            f'  <td><a class="run-link" href="{url}">{name}</a><div class="run-id-sub">{run_id}</div></td>'
            f'  <td><span class="badge" data-status="{status}">{status}</span></td>'
            f'  <td>{progress_html}</td>'
            f'  <td>{speed_str}</td>'
            f'  <td>{latest_fid_str}</td>'
            f'  <td>{best_fid_str}</td>'
            f'  <td>{disk_str}</td>'
            f'  <td><a class="btn" href="{url}">Observatory →</a></td>'
            f'</tr>'
        )

    rows_html = "\n".join(rows) if rows else '<tr id="empty-row"><td colspan="8" class="empty-state">No training runs found in this root directory.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HyperGAN Runs — Dashboard</title>
  <link rel="stylesheet" href="/assets/style.css">
  <style>
    body {{ margin: 0; padding: 0; background: #0b0d13; color: #e2e8f0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .header {{ display: flex; align-items: center; justify-content: space-between; padding-bottom: 20px; border-bottom: 1px solid #1e2433; margin-bottom: 24px; }}
    .brand {{ display: flex; align-items: center; gap: 8px; text-decoration: none; color: #fff; font-weight: 700; font-size: 1.1rem; letter-spacing: 0.05em; }}
    .brand-mark {{ background: #38bdf8; color: #0b0d13; padding: 2px 8px; border-radius: 4px; font-weight: 900; }}
    .brand-separator {{ color: #475569; }}
    .brand-detail {{ color: #94a3b8; font-weight: 400; }}
    .root-path {{ font-family: ui-monospace, monospace; font-size: 0.85rem; color: #64748b; background: #131722; padding: 4px 10px; border-radius: 6px; border: 1px solid #1e2433; }}
    .controls {{ display: flex; gap: 16px; margin-bottom: 20px; align-items: center; }}
    .search-input {{ flex: 1; max-width: 400px; background: #131722; border: 1px solid #1e2433; color: #e2e8f0; padding: 8px 14px; border-radius: 6px; font-size: 0.9rem; }}
    .search-input:focus {{ outline: none; border-color: #38bdf8; }}
    .filter-select {{ background: #131722; border: 1px solid #1e2433; color: #e2e8f0; padding: 8px 14px; border-radius: 6px; font-size: 0.9rem; }}
    .run-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: #131722; border: 1px solid #1e2433; border-radius: 8px; overflow: hidden; }}
    .run-table th {{ background: #181d2a; padding: 12px 16px; text-align: left; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; border-bottom: 1px solid #1e2433; }}
    .run-table td {{ padding: 14px 16px; border-bottom: 1px solid #1e2433; font-size: 0.9rem; }}
    .run-table tr:last-child td {{ border-bottom: none; }}
    .run-table tr:hover td {{ background: #181d2a; }}
    .run-link {{ color: #38bdf8; text-decoration: none; font-weight: 600; display: block; }}
    .run-link:hover {{ text-decoration: underline; }}
    .run-id-sub {{ font-family: ui-monospace, monospace; font-size: 0.75rem; color: #64748b; margin-top: 2px; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
    .badge[data-status="running"] {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
    .badge[data-status="complete"] {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
    .badge[data-status="failed"] {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .badge[data-status="stopped"], .badge[data-status="interrupted"] {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .progress-bar-container {{ display: flex; align-items: center; gap: 8px; }}
    .progress-bar {{ width: 80px; height: 6px; background: #1e2433; border-radius: 3px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: #38bdf8; border-radius: 3px; }}
    .btn {{ display: inline-block; padding: 6px 12px; background: #1e293b; color: #e2e8f0; text-decoration: none; border-radius: 6px; font-size: 0.8rem; font-weight: 500; border: 1px solid #334155; }}
    .btn:hover {{ background: #334155; color: #fff; }}
    .empty-state {{ text-align: center; padding: 48px; color: #64748b; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <a class="brand" href="/">
        <span class="brand-mark">H</span> HYPERGAN <span class="brand-separator">/</span> <span class="brand-detail">RUNS</span>
      </a>
      <span class="root-path">{html.escape(str(runs_root))}</span>
    </div>
    <div class="controls">
      <input type="text" id="search" class="search-input" placeholder="Filter runs by name or ID…" oninput="filterRows()">
      <select id="status-filter" class="filter-select" onchange="filterRows()">
        <option value="">All Statuses</option>
        <option value="running">Running</option>
        <option value="complete">Complete</option>
        <option value="stopped">Stopped</option>
        <option value="failed">Failed</option>
      </select>
    </div>
    <table class="run-table" id="runs-table">
      <thead>
        <tr>
          <th>Run / Name</th>
          <th>Status</th>
          <th>Progress</th>
          <th>Speed</th>
          <th>Latest FID</th>
          <th>Best FID</th>
          <th>Disk</th>
          <th>Observatory</th>
        </tr>
      </thead>
      <tbody id="runs-body">
        {rows_html}
      </tbody>
    </table>
  </div>
  <script>
    function filterRows() {{
      const q = document.getElementById('search').value.toLowerCase();
      const status = document.getElementById('status-filter').value;
      const rows = document.querySelectorAll('#runs-body tr');
      rows.forEach(row => {{
        const text = (row.getAttribute('data-name') || '').toLowerCase() + ' ' + (row.getAttribute('data-id') || '').toLowerCase();
        const rowStatus = row.getAttribute('data-status') || '';
        const matchesQ = text.includes(q);
        const matchesStatus = !status || rowStatus === status;
        row.style.display = (matchesQ && matchesStatus) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>"""


class ServiceRegistry:
    def __init__(self, root: Path, poll_seconds: float = 0.25, history_timeout: float = 180.0):
        self.root = Path(root).resolve()
        self.poll_seconds = poll_seconds
        self.history_timeout = history_timeout
        self.services: dict[str, ObservationService] = {}
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

        if self.is_single():
            try:
                manifest = read_json(self.root, "manifest.json")
                canonical_id = manifest.get("run_id")
            except Exception:
                canonical_id = None
            if run_id in (canonical_id, self.root.name, "run"):
                return self.root
            return None

        # Check direct subdirectory
        if candidate.is_dir() and not candidate.is_symlink() and (candidate / "manifest.json").is_file():
            return candidate

        # Scan subdirectories
        try:
            for entry in self.root.iterdir():
                if entry.is_dir() and not entry.is_symlink() and (entry / "manifest.json").is_file():
                    if entry.name == run_id:
                        return entry
                    try:
                        manifest = read_json(entry, "manifest.json")
                        if manifest.get("run_id") == run_id:
                            return entry
                    except Exception:
                        pass
        except OSError:
            pass
        return None

    async def get(self, run_id: str) -> ObservationService:
        async with self._lock:
            if run_id in self.services:
                return self.services[run_id]
            run_dir = self.resolve_run_dir(run_id)
            if run_dir is None:
                raise FileNotFoundError(f"Unknown run identity: {run_id}")
            for svc in self.services.values():
                if svc.root == run_dir:
                    self.services[run_id] = svc
                    return svc
            svc = ObservationService(run_dir, poll_seconds=self.poll_seconds, history_timeout=self.history_timeout)
            await svc.start()
            self.services[run_id] = svc
            if svc.run_id:
                self.services[svc.run_id] = svc
            return svc

    async def close_all(self):
        async with self._lock:
            unique = set(self.services.values())
            self.services.clear()
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
    class ClosingStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                return await super().__call__(scope, receive, send)
            finally:
                await self.body_iterator.aclose()

    root = Path(run_dir).resolve()
    registry = ServiceRegistry(root, poll_seconds=poll_seconds, history_timeout=history_timeout)
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

        runs = discover_runs(root)
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
                'bootstrap_bytes': 1048576, 'stream_count': 64,
                'history_seconds': history_timeout
            }
        })

    async def root_index(request):
        if registry.is_single():
            return await static(request)
        runs = discover_runs(root)
        html_content = render_run_index_html(runs, root)
        return Response(html_content, media_type='text/html', headers={'Cache-Control': 'no-store'})

    async def run_detail_page(request):
        run_id = request.path_params.get('run_id')
        if registry.resolve_run_dir(run_id) is None:
            raise FileNotFoundError(f'Unknown run identity: {run_id}')
        return await static(request, asset_override='index.html')

    async def runs_list(request):
        return JSONResponse(discover_runs(root))

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
        media = 'text/html' if asset.endswith('.html') else 'text/css' if asset.endswith('.css') else 'text/javascript'
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
        Route('/api/v1/session', session_exchange, methods=['POST']),
        Route('/api/v1/capabilities', capabilities),
        Route('/api/v1/openapi.json', openapi),
        Route('/api/v1/runs', runs_list),
        Route('/api/v1/stream', stream),
        Route('/api/v1/runs/{run_id}', run),
        Route('/api/v1/runs/{run_id}/status', run),
        Route('/api/v1/runs/{run_id}/metrics/catalog', catalog),
        Route('/api/v1/runs/{run_id}/model', model),
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
            scope.setdefault('state', {})['viewer_channel'] = channel
            path = scope['path']
            public = (path == '/' or path.startswith('/assets/') or path.startswith('/reducers/')
                      or path == '/api/v1/session' or path == '/api/v1/runs' or path.startswith('/runs/')
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
                if message['type'] == 'http.response.body' and scope['path'].endswith('/stream'):
                    try:
                        await asyncio.wait_for(send(message), timeout=5)
                    except asyncio.TimeoutError as exc:
                        raise OSError('Stream client exceeded the send deadline') from exc
                else:
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
        '/runs/{run_id}/metrics/catalog': 'Immutable metric definition catalog',
        '/runs/{run_id}/model': 'Run model: GAN formulation, per-player losses with metric ids, optimizers, prior, '
                                'data/training settings and networks (HNDL source; per-layer detail when model.json '
                                'was recorded); 404 when the run records no configuration',
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
    server = uvicorn.Server(uvicorn.Config(app, host=session.bind_host, port=bound_socket.getsockname()[1],
                                        access_log=False, log_level='warning'))
    server.run(sockets=[bound_socket])

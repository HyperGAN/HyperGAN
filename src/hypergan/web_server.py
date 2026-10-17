"""Optional loopback ASGI server; browser and external clients share one API."""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import html
import inspect
import json
import os
from pathlib import Path
import time

from . import inspection as _insp_mod
from . import registry as _reg_mod
from .web_autostart import ensure_run_projected, is_projection_stale_or_missing, project_run
from .web_dev import RELOAD_ASSET, RELOAD_SCRIPT, VERSION_ROUTE, dev_assets
from .web_files import read_artifact, read_json, _open
from .web_service import ObservationService, RunObservationService, MapSpec, ViewSpec
from .web_session import LocalSession

STATIC_ASSETS = {'index.html', 'app.js', 'view-worker.js', 'style.css', 'styles.css', 'uplot.js', 'uplot.css'}

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


def _stat_watermark(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def _summary_watermark(run_dir: Path) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (
        _stat_watermark(run_dir / "summary.json"),
        _stat_watermark(run_dir / "manifest.json"),
        _stat_watermark(run_dir / ".run.lock"),
    )


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
        "updated_at": updated_at,
        "disk_bytes": disk_bytes,
        "disk_formatted": disk_formatted,
        "run_dir": str(run_dir),
        "url": f"/runs/{run_id}",
    }


def discover_runs(root: Path) -> list[dict]:
    return ServiceRegistry(root)._discover_runs()


def render_run_index_html(runs: list[dict], runs_root: Path) -> str:
    rows = []
    for r in runs:
        run_id = html.escape(str(r.get("run_id", "unknown")))
        name = html.escape(str(r.get("name", run_id)))
        slug_raw = r.get("slug") or _reg_mod.extract_or_derive_slug(str(r.get("run_id", "unknown")), name=str(r.get("name", "")))
        slug = html.escape(str(slug_raw))
        short_id = html.escape(str(r.get("short_id") or str(r.get("run_id", "unknown"))[:8]))
        status = html.escape(str(r.get("status", "unknown")))
        is_running = bool(r.get("active") or r.get("running") or str(r.get("status", "")).lower() == "running")
        step = r.get("step", 0)
        total_steps = r.get("total_steps")
        sps = r.get("steps_per_second")
        speed_str = f"{sps:.1f} it/s" if isinstance(sps, (int, float)) else "—"
        latest_fid_val = r.get("latest_fid")
        latest_fid_str = f"{latest_fid_val:.2f}" if isinstance(latest_fid_val, (int, float)) else "—"
        best_fid_val = r.get("best_fid")
        best_step_val = r.get("best_step")
        if isinstance(best_fid_val, (int, float)):
            if isinstance(best_step_val, (int, float)):
                step_suffix = f" @ {best_step_val / 1000:.1f}k" if best_step_val >= 1000 else f" @ step {int(best_step_val):,}"
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

        controls_html = f'<a class="btn" href="{url}">Observatory →</a>'
        if is_running:
            controls_html += (
                f' <button type="button" class="btn btn-checkpoint" data-run-id="{run_id}" data-action="checkpoint" '
                f'onclick="fetch(\'/api/runs/{run_id}/checkpoint\', {{method: \'POST\'}}).then(() => refreshRuns())">Checkpoint</button>'
                f' <button type="button" class="btn btn-stop" data-run-id="{run_id}" data-action="stop" '
                f'onclick="fetch(\'/api/runs/{run_id}/stop\', {{method: \'POST\'}}).then(() => refreshRuns())">Stop</button>'
            )

        rows.append(
            f'<tr data-status="{status}" data-name="{name}" data-id="{run_id}" data-slug="{slug}">'
            f'  <td><a class="run-link" href="{url}">{name}</a>'
            f'<div class="run-id-sub"><span class="slug-badge">{slug}</span> <span class="id-badge">{short_id}</span> <span class="full-id">{run_id}</span></div></td>'
            f'  <td><span class="badge" data-status="{status}">{status}</span></td>'
            f'  <td>{progress_html}</td>'
            f'  <td>{speed_str}</td>'
            f'  <td>{latest_fid_str}</td>'
            f'  <td>{best_fid_str}</td>'
            f'  <td>{disk_str}</td>'
            f'  <td class="actions-cell">{controls_html}</td>'
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
    .run-id-sub {{ font-family: ui-monospace, monospace; font-size: 0.75rem; color: #64748b; margin-top: 4px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
    .slug-badge {{ background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); padding: 1px 6px; border-radius: 4px; font-weight: 600; }}
    .id-badge {{ background: #181d2a; color: #94a3b8; border: 1px solid #1e2433; padding: 1px 6px; border-radius: 4px; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
    .badge[data-status="running"] {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
    .badge[data-status="complete"], .badge[data-status="completed"] {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
    .badge[data-status="failed"] {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .badge[data-status="stopped"], .badge[data-status="interrupted"] {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .progress-bar-container {{ display: flex; align-items: center; gap: 8px; }}
    .progress-bar {{ width: 80px; height: 6px; background: #1e2433; border-radius: 3px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: #38bdf8; border-radius: 3px; }}
    .actions-cell {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
    .btn {{ display: inline-block; padding: 6px 12px; background: #1e293b; color: #e2e8f0; text-decoration: none; border-radius: 6px; font-size: 0.8rem; font-weight: 500; border: 1px solid #334155; cursor: pointer; }}
    .btn:hover {{ background: #334155; color: #fff; }}
    .btn-checkpoint {{ border-color: rgba(56, 189, 248, 0.4); color: #38bdf8; }}
    .btn-stop {{ border-color: rgba(239, 68, 68, 0.4); color: #f87171; }}
    .empty-state {{ text-align: center; padding: 48px; color: #64748b; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <a class="brand" href="/">
        <span class="brand-mark">H</span> HYPERGAN <span class="brand-separator">/</span> <span class="brand-detail">RUNS</span>
      </a>
      <span class="root-path">…/{html.escape(runs_root.name or str(runs_root))}</span>
    </div>
    <div class="controls">
      <input type="text" id="search" class="search-input" placeholder="Filter runs by name, slug, or ID…" oninput="filterRows()">
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
        const text = ((row.getAttribute('data-name') || '') + ' ' + (row.getAttribute('data-id') || '') + ' ' + (row.getAttribute('data-slug') || '')).toLowerCase();
        const rowStatus = row.getAttribute('data-status') || '';
        const matchesQ = text.includes(q);
        const matchesStatus = !status || rowStatus === status;
        row.style.display = (matchesQ && matchesStatus) ? '' : 'none';
      }});
    }}
    async function triggerRunAction(runId, action) {{
      await fetch('/api/runs/' + encodeURIComponent(runId) + '/' + action, {{method: 'POST'}});
      await refreshRuns();
    }}
    async function refreshRuns() {{
      try {{
        const res = await fetch('/api/runs', {{cache: 'no-store'}});
        if (!res.ok) return;
        const runs = await res.json();
        if (!Array.isArray(runs)) return;
        const tbody = document.getElementById('runs-body');
        if (!tbody || !runs.length) return;
        runs.forEach(r => {{
          const row = tbody.querySelector(`tr[data-id="${{r.run_id}}"]`);
          if (row) {{
            const st = r.status || 'unknown';
            row.setAttribute('data-status', st);
            const badge = row.querySelector('.badge');
            if (badge) {{
              badge.setAttribute('data-status', st);
              badge.textContent = st;
            }}
            const cells = row.querySelectorAll('td');
            const step = Number(r.step || 0);
            const total = Number(r.total_steps || 0);
            if (cells[2]) {{
              if (total > 0) {{
                const pct = Math.min(100, Math.max(0, Math.floor((step / total) * 100)));
                cells[2].innerHTML = `<div class="progress-bar-container"><span>${{step.toLocaleString()}} / ${{total.toLocaleString()}}</span><div class="progress-bar"><div class="progress-fill" style="width: ${{pct}}%"></div></div></div>`;
              }} else {{
                cells[2].innerHTML = `<span>${{step.toLocaleString()}}</span>`;
              }}
            }}
            if (cells[3]) {{
              cells[3].textContent = (r.steps_per_second != null) ? `${{Number(r.steps_per_second).toFixed(1)}} it/s` : '—';
            }}
            if (cells[4]) {{
              cells[4].textContent = (r.latest_fid != null) ? Number(r.latest_fid).toFixed(2) : '—';
            }}
            if (cells[5]) {{
              if (r.best_fid != null) {{
                const bs = r.best_step != null ? (r.best_step >= 1000 ? ` @ ${{(r.best_step / 1000).toFixed(1)}}k` : ` @ step ${{Number(r.best_step).toLocaleString()}}`) : '';
                cells[5].textContent = `${{Number(r.best_fid).toFixed(2)}}${{bs}}`;
              }} else {{
                cells[5].textContent = '—';
              }}
            }}
            if (cells[6] && r.disk_formatted) {{
              cells[6].textContent = r.disk_formatted;
            }}
            const actionsCell = row.querySelector('.actions-cell') || cells[7];
            const isRunning = Boolean(r.active || r.running || String(st).toLowerCase() === 'running');
            if (actionsCell && !isRunning) {{
              actionsCell.querySelectorAll('.btn-checkpoint, .btn-stop').forEach(btn => btn.remove());
            }}
          }}
        }});
        filterRows();
      }} catch (_err) {{}}
    }}
    setInterval(() => {{
      if (document.querySelector('#runs-body tr[data-status="running"]')) {{
        refreshRuns();
      }}
    }}, 3000);
  </script>
</body>
</html>"""


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

    def _evict_lru_if_needed(self) -> list[ObservationService]:
        evicted_services: list[ObservationService] = []
        if self.max_services <= 0:
            return evicted_services
        while len(self._services) > self.max_services:
            _, evicted = self._services.popitem(last=False)
            if evicted not in self._services.values():
                evicted_services.append(evicted)
                try:
                    evicted._executor.shutdown(wait=False)
                except Exception:
                    pass
        return evicted_services

    def get_service(self, run_id: str) -> ObservationService:
        if run_id in self._services:
            self._services.move_to_end(run_id)
            svc = self._services[run_id]
            ensure_run_projected(svc.root)
            return svc

        run_dir = self.resolve_run_dir(run_id)
        if run_dir is None:
            raise FileNotFoundError(f"Unknown run identity: {run_id}")

        ensure_run_projected(run_dir)

        svc = None
        for existing_key, existing_svc in list(self._services.items()):
            if existing_svc.root == run_dir:
                svc = existing_svc
                del self._services[existing_key]
                break

        if svc is None:
            svc = ObservationService(run_dir, poll_seconds=self.poll_seconds, history_timeout=self.history_timeout)

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
    class ClosingStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                return await super().__call__(scope, receive, send)
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
                'bootstrap_bytes': 1048576, 'stream_count': 64,
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
    server = uvicorn.Server(uvicorn.Config(app, host=session.bind_host, port=bound_socket.getsockname()[1],
                                        access_log=False, log_level='warning'))
    server.run(sockets=[bound_socket])

"""Bounded observation services shared by HTTP, browser and agent consumers.

One tail per file fans unchanged frames out. Only explicit historical bootstrap
jobs reduce; connected viewers never create live reduced state on the server.
"""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from itertools import islice
import re
import stat as _stat
import time

from . import inspection as _insp_mod
from .event_views import MapSpec, ViewSpec, read_projection_page
from .run_events import read_event_page
from .summary import is_fid_metric
from .web_files import read_json, read_bytes, _open

MAX_QUEUE_BYTES = 1048576
MAX_SUBSCRIBERS = 32
MAX_BOOTSTRAP_BYTES = 1048576
MAX_GROUPS = 2048
MAX_STREAMS = 4096
MAX_ATTEMPTS = 4096
MAX_BOOTSTRAPS = 8
# Preview retention spans the whole run (thinned to --preview-keep, 128 by
# default), and --preview-keep all keeps every generation: a 100k-step run at
# --preview-every 100 then publishes a thousand of them. The index stays bounded
# and validated, with room for several times that.
MAX_PREVIEWS = 4096
PREVIEW_INDEX_BYTES = 16 * 1048576
# Historical bootstrap caches kept per run and map revision, and runs kept in
# the cache root; the least recently used beyond either bound are deleted.
MAX_VIEW_CACHES = 16
MAX_VIEW_CACHE_RUNS = 64
VIEW_CACHE_ENV = 'HYPERGAN_VIEWER_CACHE'
_HEX = re.compile('[0-9a-f]{64}\\Z')
_EVAL_DIR_RE = re.compile(r'^(?:step-\d+-[A-Za-z0-9_.-]+|[0-9a-f]{32})\Z')


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


_SAMPLE_NAME = re.compile('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}')


def view_cache_root():
    """Private per-user directory for derived viewer state, never a run directory.

    Viewing must not write into runs: a shared or read-only run still charts,
    and reloading a page does not leave files behind in the experiment.
    """
    configured = os.environ.get(VIEW_CACHE_ENV)
    if configured:
        return Path(configured)
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'hypergan' / 'viewer'


def _sample_name(*candidates):
    """Short bounded group name for the viewer; never a path or artifact identity."""
    for value in candidates:
        if isinstance(value, str) and _SAMPLE_NAME.fullmatch(value):
            return value
    return 'sample'


def _positive_int(value) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def _nonneg_int(value) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _finite_float(*candidates) -> float | None:
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fval = float(value)
            if math.isfinite(fval):
                return fval
    return None


def _finite_nonneg_float(*candidates) -> float | None:
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fval = float(value)
            if math.isfinite(fval) and fval >= 0.0:
                return fval
    return None


def _clean_gpu_name(raw: str) -> str:
    """Strip leading vendor prefix from GPU device names (e.g. 'NVIDIA RTX A6000' -> 'RTX A6000')."""
    if not isinstance(raw, str):
        return ""
    cleaned = raw.strip()
    if cleaned.upper().startswith("NVIDIA "):
        cleaned = cleaned[7:].strip()
    return cleaned


def _classify_device_token(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    token = raw.strip().lower()
    if not token or token in ("unknown", "auto", "none"):
        return None
    base = token.split(":", 1)[0].split("-", 1)[0]
    if base in ("rocm", "hip") or "rocm" in token:
        return "rocm"
    if base == "xpu" or "xpu" in token:
        return "xpu"
    if base == "mps" or "mps" in token:
        return "mps"
    if base in ("cuda", "gpu", "nccl") or "cuda" in token or "nccl" in token:
        return "cuda"
    if base == "cpu" or "cpu" in token:
        return "cpu"
    return None


def extract_topology_and_batch(manifest: dict, summary: dict | None = None) -> dict:
    """Extract canonical hardware topology and batch hierarchy from run metadata."""
    manifest = manifest if isinstance(manifest, dict) else {}
    summary = summary if isinstance(summary, dict) else None
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    rt_cuda = runtime.get("cuda") if isinstance(runtime.get("cuda"), dict) else None

    has_cuda_dict = bool(rt_cuda)
    device_kind: str | None = None
    if has_cuda_dict and rt_cuda.get("hip"):
        device_kind = "rocm"
    else:
        for clue in (runtime.get("device"), execution.get("name"), training_cfg.get("device")):
            classified = _classify_device_token(clue)
            if classified is not None:
                device_kind = classified
                break
        if device_kind is None and has_cuda_dict:
            device_kind = "cuda"

    raw_world_size = _positive_int(execution.get("world_size")) or _positive_int(runtime.get("world_size"))
    world_size = raw_world_size if raw_world_size is not None else (1 if device_kind is not None else None)
    raw_nodes = _positive_int(execution.get("nodes"))
    nodes = raw_nodes if raw_nodes is not None else (1 if world_size is not None else None)
    raw_rpn = _positive_int(execution.get("ranks_per_node"))
    if raw_rpn is not None:
        ranks_per_node = raw_rpn
    elif world_size is not None and nodes is not None:
        ranks_per_node = max(1, world_size // nodes)
    else:
        ranks_per_node = None

    is_accelerator = device_kind in ("cuda", "mps", "rocm", "xpu")
    gpu_count = (world_size or 1) if is_accelerator else 0

    gpu_names: list[str] = []
    if rt_cuda:
        rank_devices = rt_cuda.get("rank_devices")
        if isinstance(rank_devices, list) and rank_devices:
            for dev in rank_devices:
                if isinstance(dev, dict) and isinstance(dev.get("name"), str):
                    cleaned = _clean_gpu_name(dev["name"])
                    if cleaned:
                        gpu_names.append(cleaned)
        if not gpu_names and isinstance(rt_cuda.get("name"), str):
            cleaned = _clean_gpu_name(rt_cuda["name"])
            if cleaned:
                gpu_names.append(cleaned)

    gpu_name: str | None = None
    topo_detail: str | None = None
    if gpu_names and is_accelerator:
        counts: OrderedDict[str, int] = OrderedDict()
        for n in gpu_names:
            counts[n] = counts.get(n, 0) + 1
        if len(counts) == 1:
            gpu_name = next(iter(counts))
            base_detail = gpu_name if gpu_count == 1 else f"{gpu_count}× {gpu_name}"
        else:
            gpu_name = " + ".join(counts.keys())
            parts = [f"{cnt}× {n}" if cnt > 1 else n for n, cnt in counts.items()]
            base_detail = " + ".join(parts)
        if (nodes or 1) > 1 and ranks_per_node:
            topo_detail = f"{base_detail} · {ranks_per_node}/node"
        else:
            topo_detail = base_detail
    elif device_kind is not None and (nodes or 1) > 1 and ranks_per_node:
        topo_detail = f"{ranks_per_node}/node"

    topo_label: str | None = None
    if device_kind == "cpu":
        ws = world_size or 1
        nd = nodes or 1
        if ws > 1 and nd > 1:
            topo_label = f"{ws} CPU ranks · {nd} nodes"
        elif ws > 1:
            topo_label = f"{ws} CPU ranks"
        else:
            topo_label = "CPU"
    elif device_kind == "mps":
        topo_label = "MPS"
    elif device_kind in ("cuda", "rocm", "xpu"):
        nd = nodes or 1
        if gpu_count == 1:
            topo_label = "1 GPU"
        elif nd > 1:
            topo_label = f"{gpu_count} GPUs · {nd} nodes"
        else:
            topo_label = f"{gpu_count} GPUs"

    topology = {
        "device_kind": device_kind,
        "world_size": world_size,
        "nodes": nodes,
        "ranks_per_node": ranks_per_node,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "label": topo_label,
        "detail": topo_detail,
    }

    global_batch_size = (
        _positive_int(manifest.get("global_batch_size"))
        or _positive_int(execution.get("global_batch_size"))
        or _positive_int(summary.get("global_batch_size") if summary else None)
        or _positive_int(training_cfg.get("batch_size"))
    )
    ws_eff = world_size or 1
    exec_local = _positive_int(execution.get("local_batch_size"))
    if exec_local is not None:
        local_batch_size = exec_local
    elif global_batch_size is not None and ws_eff > 1 and global_batch_size % ws_eff == 0:
        local_batch_size = global_batch_size // ws_eff
    else:
        local_batch_size = global_batch_size

    accumulation_steps = _positive_int(execution.get("accumulation_steps")) or 1
    exec_micro = _positive_int(execution.get("microbatch_size"))
    if exec_micro is not None:
        microbatch_size = exec_micro
    elif local_batch_size is not None and accumulation_steps > 1 and local_batch_size % accumulation_steps == 0:
        microbatch_size = local_batch_size // accumulation_steps
    else:
        microbatch_size = local_batch_size

    batch_label = f"Batch {global_batch_size}" if global_batch_size is not None else None
    batch_detail: str | None = None
    if ws_eff > 1 and local_batch_size is not None:
        unit = "GPU" if gpu_count > 1 else "rank"
        batch_detail = f"{local_batch_size} / {unit}"
        if accumulation_steps > 1:
            batch_detail = f"{batch_detail} · ×{accumulation_steps} accum"
    elif accumulation_steps > 1:
        batch_detail = f"×{accumulation_steps} accum"

    return {
        "topology": topology,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "microbatch_size": microbatch_size,
        "accumulation_steps": accumulation_steps,
        "batch_label": batch_label,
        "batch_detail": batch_detail,
    }


def _is_stale_unmodified_dir(run_dir: Path, max_idle_seconds: float = 3600.0) -> bool:
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


def compute_throughput_and_eta(
    *,
    active: bool,
    step: object,
    total_steps: object,
    steps_per_second: object,
    global_batch_size: int | None,
    summary: dict | None = None,
) -> tuple[float | None, float | None]:
    """Compute samples_per_second and active-only eta_seconds consistently."""
    sps = _finite_nonneg_float(steps_per_second)
    if sps is not None and isinstance(global_batch_size, int) and not isinstance(global_batch_size, bool) and global_batch_size > 0:
        samples_per_second: float | None = float(sps) * global_batch_size
    else:
        samples_per_second = None

    eta_seconds: float | None = None
    if active:
        step_int = _nonneg_int(step)
        total_int = _positive_int(total_steps)
        if sps is not None and sps > 0.0 and step_int is not None and total_int is not None and total_int > step_int:
            eta_seconds = round((total_int - step_int) / float(sps), 2)
        elif isinstance(summary, dict):
            eta_seconds = _finite_nonneg_float(summary.get("eta_seconds"))
    return samples_per_second, eta_seconds


def extract_fid_and_health(manifest: dict, summary: dict | None = None) -> dict:
    """Extract canonical FID, best metric/step, health_alerts, and warnings."""
    manifest = manifest if isinstance(manifest, dict) else {}
    summary = summary if isinstance(summary, dict) else None

    best = summary.get("best") if (summary and isinstance(summary.get("best"), dict)) else None
    if best is None and isinstance(manifest.get("best_checkpoint"), dict):
        best = manifest["best_checkpoint"]

    best_metric: str | None = None
    if best and isinstance(best.get("metric"), str) and best["metric"].strip():
        best_metric = best["metric"].strip()
    elif summary and isinstance(summary.get("best_metric"), str) and summary["best_metric"].strip():
        best_metric = summary["best_metric"].strip()

    best_is_fid = False
    if best is not None:
        raw_unit = best.get("unit")
        if isinstance(raw_unit, str) and raw_unit.strip():
            best_is_fid = raw_unit.strip().upper() == "FID"
        elif best.get("metric"):
            best_is_fid = is_fid_metric({"metric": best.get("metric")})
        else:
            best_is_fid = True

    best_fid = _finite_float(
        summary.get("best_fid") if summary else None,
        summary.get("fid_best") if summary else None,
        best.get("value") if (best and best_is_fid) else None,
        manifest.get("best_fid"),
        manifest.get("fid_best"),
    )

    best_step = _nonneg_int(best.get("step") if best else None)
    if best_step is None and summary:
        best_step = _nonneg_int(summary.get("best_step"))
    if best_step is None:
        best_step = _nonneg_int(manifest.get("best_step"))

    metrics_dict = summary.get("metrics") if (summary and isinstance(summary.get("metrics"), dict)) else {}
    latest_dict = summary.get("latest") if (summary and isinstance(summary.get("latest"), dict)) else {}
    latest_fid = _finite_float(
        summary.get("latest_fid") if summary else None,
        summary.get("fid_latest") if summary else None,
        metrics_dict.get("fid50k_train"),
        metrics_dict.get("fid"),
        latest_dict.get("fid50k_train"),
        latest_dict.get("fid"),
        manifest.get("latest_fid"),
        manifest.get("fid_latest"),
    )

    raw_alerts = summary.get("health_alerts") if (summary and isinstance(summary.get("health_alerts"), list)) else None
    if raw_alerts is None and isinstance(manifest.get("health_alerts"), list):
        raw_alerts = manifest.get("health_alerts")
    health_alerts = [str(a) for a in (raw_alerts or []) if isinstance(a, str) and a.strip()]

    raw_warnings = manifest.get("warnings")
    warnings = [str(w) for w in raw_warnings if isinstance(w, str)] if isinstance(raw_warnings, list) else []

    return {
        "best_fid": best_fid,
        "latest_fid": latest_fid,
        "best_step": best_step,
        "best_metric": best_metric,
        "health_alerts": health_alerts,
        "warnings": warnings,
    }


from foldlitedb.streaming import Subscriber, Stream, cursor_offset, sse, stream_events


@dataclass
class BootstrapJob:
    key: str
    query: dict
    task: object = None
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.monotonic)
    completed: float | None = None
    delivered: bool = False


class ObservationService:
    def __init__(self, run_dir, *, poll_seconds=.25, history_timeout=180, cache_dir=None):
        self.root = Path(run_dir).resolve()
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if type(history_timeout) not in (int, float) or not math.isfinite(history_timeout) or not 1 <= history_timeout <= 3600:
            raise ValueError('history timeout must be finite seconds in 1..3600')
        if type(poll_seconds) not in (int, float) or not math.isfinite(poll_seconds) or not .005 <= poll_seconds <= 60:
            raise ValueError('poll interval must be finite seconds in .005..60')
        self.poll_seconds = poll_seconds
        self.history_timeout = history_timeout
        self.streams = {}
        self._failed = {}
        self.source_paths = {'training': 'events.jsonl'}
        self.subscribers = set()
        self.jobs = OrderedDict()
        self.attempts = {}
        self.artifacts = {}
        self._artifact_signature = None
        self.artifact_error = None
        self._model_cache = None
        self._summary_cache = None
        self._artifact_lock = asyncio.Lock()
        self.run_id = None
        self.manifest = {'status': 'waiting'}
        self.metadata_revision = None
        self._discovery = None
        self._closed = False
        self._history_slots = asyncio.Semaphore(1)
        self._source_revision = 0
        self.discovery_error = None

    async def start(self):
        await self._activate()
        self._discovery = asyncio.create_task(self._watch())
        return self

    async def _activate(self):
        if not (self.root / 'manifest.json').exists():
            return
        self.manifest = await asyncio.to_thread(read_json, self.root, 'manifest.json')
        await asyncio.to_thread(self._read_cached_summary)
        self.run_id = self.manifest['run_id']
        self.metadata_revision = self._metadata_key(self.manifest)
        self._register('training')
        await self.discover()
        await self.refresh_artifacts()
        self.notify('metadata', {'run': self.public_manifest(), 'revision': self.metadata_revision})

    @staticmethod
    def _metadata_key(manifest):
        return _digest({k: manifest.get(k) for k in ('run_id', 'attempt_id', 'metrics_catalog', 'status', 'recovery_parent')})

    def _read_cached_summary(self) -> dict | None:
        try:
            info = (self.root / 'summary.json').lstat()
            signature = (info.st_mtime_ns, info.st_size) if _stat.S_ISREG(info.st_mode) else None
        except OSError:
            signature = None
        if self._summary_cache is not None and self._summary_cache[0] == signature:
            return self._summary_cache[1]
        loaded = None
        if signature is not None:
            try:
                candidate = read_json(self.root, 'summary.json')
                if isinstance(candidate, dict):
                    loaded = candidate
            except (OSError, ValueError):
                loaded = None
        self._summary_cache = (signature, loaded)
        return loaded

    def public_manifest(self):
        summary = self._read_cached_summary()
        fields = ('schema_version', 'run_id', 'attempt_id', 'attempt_index', 'status', 'steps',
                  'total_steps', 'last_durable_step', 'seconds', 'metrics_catalog',
                  'observation_sha256', 'checkpoint_every', 'stop_reason', 'possible_lost_steps',
                  'evaluation_schedule', 'training_seconds', 'steps_per_second', 'samples_seen',
                  'global_batch_size', 'hostname', 'remote')
        result = {key: self.manifest[key] for key in fields if key in self.manifest}
        cfg = self.manifest.get('config') if isinstance(self.manifest.get('config'), dict) else {}
        result['name'] = (
            cfg.get('name')
            or self.manifest.get('name')
            or (summary.get('name') if summary else None)
            or self.root.name
        )

        is_active = False
        try:
            is_active = bool(_insp_mod.check_run_active(self.root)[0])
        except Exception:
            is_active = False

        raw_status = self.manifest.get('status') or (summary.get('status') if summary else None) or 'waiting'
        if (
            not is_active
            and str(raw_status).strip().lower() in ('running', 'initializing')
            and summary
            and str(summary.get('status') or '').strip().lower() in (
                'stopped', 'complete', 'completed', 'failed', 'interrupted', 'cancelled', 'canceled'
            )
        ):
            raw_status = summary['status']

        if is_active:
            status = 'running'
            active = True
        elif (
            str(raw_status).strip().lower() in ('running', 'initializing')
            and ((self.root / '.run.lock').exists() or _is_stale_unmodified_dir(self.root))
        ):
            status = 'stopped'
            active = False
        else:
            status = raw_status
            active = str(status).strip().lower() in ('running', 'training', 'initializing', 'starting')
        running = bool(is_active)

        result['status'] = status
        result['active'] = active
        result['running'] = running

        if 'hostname' not in result or 'remote' not in result:
            try:
                host_info = _insp_mod.inspect_run_host(self.root)
                if host_info.get('hostname') and 'hostname' not in result:
                    result['hostname'] = host_info['hostname']
                if host_info.get('remote') and 'remote' not in result:
                    result['remote'] = True
            except Exception:
                pass

        topo_batch = extract_topology_and_batch(self.manifest, summary)
        result['topology'] = topo_batch['topology']
        result['global_batch_size'] = topo_batch['global_batch_size']
        result['local_batch_size'] = topo_batch['local_batch_size']
        result['microbatch_size'] = topo_batch['microbatch_size']
        result['accumulation_steps'] = topo_batch['accumulation_steps']
        result['batch_label'] = topo_batch['batch_label']
        result['batch_detail'] = topo_batch['batch_detail']

        sps = _finite_nonneg_float(
            self.manifest.get('steps_per_second'),
            self.manifest.get('speed_its'),
            summary.get('steps_per_second') if summary else None,
            summary.get('speed_its') if summary else None,
            summary.get('steps_per_sec') if summary else None,
        )
        if sps is not None and 'steps_per_second' not in result:
            result['steps_per_second'] = sps

        step_val = self.manifest.get('steps')
        if step_val is None and summary:
            step_val = summary.get('step')
        training_cfg = cfg.get('training') if isinstance(cfg.get('training'), dict) else {}
        total_val = (
            self.manifest.get('total_steps')
            or (summary.get('total_steps') if summary else None)
            or self.manifest.get('max_steps')
            or (summary.get('max_steps') if summary else None)
            or self.manifest.get('target_steps')
            or (summary.get('target_steps') if summary else None)
            or _positive_int(training_cfg.get('steps'))
        )
        if total_val is not None and 'total_steps' not in result:
            result['total_steps'] = total_val

        samples_per_second, eta_seconds = compute_throughput_and_eta(
            active=active,
            step=step_val,
            total_steps=total_val,
            steps_per_second=sps,
            global_batch_size=topo_batch['global_batch_size'],
            summary=summary,
        )
        result['samples_per_second'] = samples_per_second
        result['eta_seconds'] = eta_seconds

        fid_health = extract_fid_and_health(self.manifest, summary)
        result['best_fid'] = fid_health['best_fid']
        result['latest_fid'] = fid_health['latest_fid']
        result['best_step'] = fid_health['best_step']
        result['best_metric'] = fid_health['best_metric']
        result['health_alerts'] = fid_health['health_alerts']
        result['warnings'] = fid_health['warnings']

        boundary = self.manifest.get('durable_event_boundary')
        if isinstance(boundary, dict):
            result['durable_event_boundary'] = boundary
        result['metric_consistency'] = self.metric_consistency()
        return result

    def model_summary(self):
        """The run's formulation, losses, optimizers and networks (torch-free).

        Built from the recorded configuration plus, when training or a backfill
        wrote one for this exact configuration, the bounded model.json detail.
        Cached until the configuration, the catalog, that file or the attempt changes.
        """
        from .model_description import MAX_MODEL_BYTES, MODEL_FILE, describe_run
        from .metrics import read_catalog
        manifest = self.manifest
        if not isinstance(manifest.get('config'), dict):
            raise FileNotFoundError('No model configuration recorded for this run')
        try:
            info = (self.root / MODEL_FILE).lstat()
            signature = (info.st_mtime_ns, info.st_size)
        except OSError:
            signature = None
        # The document also carries per-attempt manifest fields (provenance,
        # warnings, attempt and step counts), so they are part of the key.
        attempt = json.dumps([manifest.get(k) for k in ('attempt_id', 'attempt_index', 'total_steps',
                                                         'global_batch_size', 'source', 'runtime', 'warnings')],
                             sort_keys=True, default=str)
        key = (manifest.get('config_sha256'), manifest.get('metrics_catalog'), signature, attempt)
        if self._model_cache is not None and self._model_cache[0] == key:
            return self._model_cache[1]
        recorded = None
        if signature is not None:
            try:
                recorded = read_json(self.root, MODEL_FILE, max_bytes=MAX_MODEL_BYTES)
            except ValueError:
                recorded = None
        catalog = None
        if manifest.get('metrics_catalog'):
            try:
                catalog = read_catalog(self.root, manifest['metrics_catalog'],
                    _open_file=lambda path: _open(self.root, path.relative_to(self.root).as_posix()))
            except (OSError, ValueError, KeyError):
                catalog = None
        result = describe_run(manifest, recorded=recorded, catalog=catalog)
        result['run_id'] = self.run_id
        if signature is not None and recorded is None:
            for entry in result['networks']:
                if 'reuse_of' not in entry:
                    entry['graph'] = {'status': 'unavailable', 'reason': 'model.json is unreadable or exceeds its byte budget'}
        self._model_cache = (key, result)
        return result

    def metric_consistency(self):
        """Report projection progress separately from the durable training prefix.

        This is server-observed projection progress, not an acknowledgement that
        a particular browser has rendered those frames or a new durability check.
        A caught-up training tail alone says nothing about the metric projector.
        """
        boundary = self.manifest.get('durable_event_boundary')
        result = {'status': 'unavailable', 'committed_step': None,
                  'committed_offset': None, 'projected_offset': None}
        if (not isinstance(boundary, dict) or boundary.get('run_id') != self.run_id
                or type(boundary.get('offset')) is not int or boundary['offset'] < 1
                or type(boundary.get('step')) is not int or boundary['step'] < 0):
            return result
        result.update(committed_step=boundary['step'], committed_offset=boundary['offset'])
        stream = self.streams.get('projection:' + MapSpec().revision)
        if stream is not None and stream.error:
            return result
        projected = stream.projected_offset if stream is not None else None
        result.update(projected_offset=projected,
                      status='caught_up' if projected is not None and projected >= boundary['offset'] else 'pending')
        return result

    def artifact_index(self):
        result = {'schema_version': 1, 'artifacts': {
            key: {name: value for name, value in record.items() if name != 'path'}
            for key, record in self.artifacts.items()}}
        if self.artifact_error:
            result['error'] = self.artifact_error
        return result

    def _relocator(self):
        """Map an indexed path to one relative to this run directory.

        Producers record paths relative to the run; older runs hold absolute
        paths, which name the old location once a run is moved, renamed or copied. The old run directory is the longest prefix
        whose remainder exists here; it is found once and reused for the index.
        """
        stale_prefixes = []

        def relative_path(value):
            path = Path(value)
            # A rooted path without a drive (`/x` on Windows) is not relative.
            if not path.root:
                return path.as_posix()
            try:
                return path.relative_to(self.root).as_posix()
            except ValueError:
                pass
            parts = path.parts
            for prefix in stale_prefixes:
                if parts[:len(prefix)] == prefix:
                    return Path(*parts[len(prefix):]).as_posix()
            for split in range(1, len(parts)):
                remainder = Path(*parts[split:])
                if (self.root / remainder).exists():
                    stale_prefixes.append(parts[:split])
                    return remainder.as_posix()
            raise ValueError(f'Indexed file {path.name!r} is not inside this run')
        return relative_path

    async def refresh_artifacts(self):
        async with self._artifact_lock:
            def inventory():
                signature = [self.manifest.get('sample_path')]
                for relative in ('previews/index.json', 'artifacts/index.json'):
                    path = self.root / relative
                    try:
                        stat = path.lstat()
                        signature.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
                    except FileNotFoundError:
                        signature.append(None)
                if signature == self._artifact_signature:
                    return None
                try:
                    return signature, collect(signature), None
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    # An optional bad index must not fail the run: keep what was
                    # served, report it once, and retry only when the files change.
                    return signature, self.artifacts, f'Artifact index unavailable: {exc}'

            def collect(signature):
                records = {}
                relative_path = self._relocator()

                def placed(record, value):
                    # A file the index names but this run no longer holds is
                    # unavailable on its own; the rest of the index still serves.
                    try:
                        record['path'] = relative_path(value)
                    except ValueError as exc:
                        record.update(path=None, status='unavailable', reason=str(exc))
                    return record
                if signature[1] is not None:
                    index = read_json(self.root, 'previews/index.json',
                                      max_bytes=PREVIEW_INDEX_BYTES)
                    previews = index.get('previews')
                    if (index.get('schema_version') != 1 or index.get('run_id') != self.run_id
                            or not isinstance(previews, list) or len(previews) > MAX_PREVIEWS):
                        raise ValueError('Invalid bounded preview index')
                    for preview in previews:
                        identity = preview['identity']
                        # Digest keys stay the artifact identity; `name` indexes the
                        # source across steps so a viewer can group its history.
                        key = 'preview-' + _digest(identity)[:32]
                        name = _sample_name(preview.get('name'), identity.get('name'))
                        if preview.get('representation') != 'png':
                            records[key] = placed(dict(bytes=preview['bytes'],
                                sha256=preview.get('sha256'), role='sample', modality='tensor', name=name,
                                media_type='application/json', provenance=dict(identity, step=preview['step'], name=name),
                                shape=preview['shape']), preview['path'])
                            if not isinstance(records[key]['sha256'], str) and 'status' not in records[key]:
                                records[key].update(status='unavailable', reason='Preview predates indexed content digests')
                        from .previews import MAX_INPUT_GRIDS, MAX_EXTRA_GRIDS
                        grid_fields = (('image_grid', '-grid'), ('real_image_grid', '-real-grid')) + tuple(
                            (f'input_image_grid_{n}', f'-input-{n}') for n in range(MAX_INPUT_GRIDS)) + tuple(
                            (f'extra_image_grid_{n}', f'-extra-{n}') for n in range(MAX_EXTRA_GRIDS)) + (
                            ('comparison_image_grid', '-comparison'),)
                        for field, suffix in grid_fields:
                            grid = preview.get(field)
                            if grid is None:
                                continue
                            grid_name = _sample_name(grid.get('name'), name if field == 'image_grid' else 'x')
                            record = dict(bytes=grid['bytes'],
                                sha256=grid['sha256'], role='sample', modality='image', media_type='image/png',
                                name=grid_name,
                                provenance=dict(identity, step=preview['step'], name=grid_name),
                                width=grid['width'], height=grid['height'])
                            if field == 'image_grid':
                                record['shape'] = preview['shape']
                            records[key + suffix] = placed(record, grid['path'])
                if signature[2] is not None:
                    index = read_json(self.root, 'artifacts/index.json')
                    indexed = index.get('artifacts')
                    if index.get('schema_version') != 1 or not isinstance(indexed, dict) or len(indexed) > 256:
                        raise ValueError('Invalid bounded artifact index')
                    for key, record in indexed.items():
                        if key in records or not isinstance(key, str) or not 1 <= len(key) <= 128:
                            raise ValueError('Invalid/duplicate artifact ID')
                        # Explicit indexes may name their source; otherwise the stable
                        # artifact ID is its own group name.
                        records[key] = placed(dict(record, name=_sample_name(record.get('name'), key)),
                                              record['path'])
                if signature[0]:
                    try:
                        path = relative_path(signature[0])
                    except ValueError:
                        path = Path(signature[0]).name
                    key = 'final-sample-' + _digest(path)[:32]
                    try:
                        data = read_bytes(self.root, path, max_bytes=16 * 1048576)
                        payload = json.loads(data)
                        saved = payload.get('identity') if isinstance(payload.get('identity'), dict) else {}
                        name = _sample_name(payload.get('name'), saved.get('name'))
                        records[key] = dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            role='sample', modality='tensor', media_type='application/json', name=name,
                            # `final` marks the run's finished sample so a viewer can
                            # name it rather than showing another anonymous tensor.
                            provenance=dict(saved, step=payload.get('step'), name=name, final=True),
                            shape=payload.get('shape'))
                    except (ValueError, OSError) as exc:
                        records[key] = dict(status='unavailable', role='sample', reason=str(exc), name='final')
                return records
            result = await asyncio.to_thread(inventory)
            if result is not None:
                previous = self.artifact_error
                self._artifact_signature, self.artifacts, self.artifact_error = result
                if self.artifact_error != previous:
                    self.notify('discovery_error', {'reason': self.artifact_error or '',
                                                    'status': 'error' if self.artifact_error else 'available'})
                self.notify('artifacts', {'revision': _digest(self.artifact_index())})

    def retire(self, reason):
        """Stop serving this run: every subscriber is told to reconnect.

        A live view must never keep its heartbeat while its tails are gone, or
        the page would show a frozen run as "Live stream".
        """
        for subscriber in list(self.subscribers):
            # A closed subscriber delivers one queued frame, so drop pending
            # frames and leave only the reason the page must reconnect.
            while not subscriber.queue.empty():
                subscriber.queue.get_nowait()
            subscriber.size = 0
            subscriber.offer('reset_required', {'reason': reason})
            subscriber.closed = True
        self._closed = True
        tasks = [s.task for s in self.streams.values()] + [job.task for job in self.jobs.values()]
        if self._discovery:
            tasks.append(self._discovery)
        for task in tasks:
            if task:
                task.cancel()

    async def close(self):
        self._closed = True
        tasks = [s.task for s in self.streams.values()] + [job.task for job in self.jobs.values()]
        if self._discovery:
            tasks.append(self._discovery)
        for task in tasks:
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in tasks if t), return_exceptions=True)

    def notify(self, event, value):
        for subscriber in list(self.subscribers):
            subscriber.offer(event, value)

    def _register(self, stream_id):
        if stream_id in self.streams:
            return
        if len(self.streams) >= MAX_STREAMS:
            raise ValueError(f'Run exceeds {MAX_STREAMS} registered observation streams')
        stream = Stream(stream_id)
        self.streams[stream_id] = stream
        self.notify('stream_added', {'stream_id': stream_id})
        stream.task = asyncio.create_task(self._tail(stream))

    def view_cache_dir(self, map_revision):
        """Where this run's bootstrap caches for one map revision live."""
        owner = _digest({'root': str(self.root), 'run_id': self.run_id})[:32]
        return (self.cache_dir or view_cache_root()) / owner / map_revision

    def _source_file(self, stream_id):
        if stream_id in self.source_paths:
            return self.root / self.source_paths[stream_id]
        return self.root / 'views' / stream_id[11:] / 'contributions.jsonl'

    def _file_identity(self, stream_id):
        try:
            stat = self._source_file(stream_id).stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size

    def _recover_failed_streams(self):
        """Re-register a failed stream once its file was replaced or truncated.

        A stream whose history became unreadable stays failed (and visible) while
        its file is unchanged; a rebuilt projection or a cleaned run gets a new
        tail from its start, and connected viewers are told to reload.
        """
        for stream_id, failed in list(self._failed.items()):
            current = self._file_identity(stream_id)
            if current is None or (current[:2] == failed[:2] and current[2] >= failed[2]):
                continue
            del self._failed[stream_id]
            self.streams.pop(stream_id, None)
            if stream_id == 'training':
                self.attempts.clear()
            else:
                for job in self.jobs.values():
                    if job.task:
                        job.task.cancel()
                self.jobs.clear()
            self._register(stream_id)
            self.notify('reset_required', {'stream_id': stream_id, 'reason': 'Observation stream was rebuilt'})

    async def discover(self):
        # Bad/over-budget optional inventories must not freeze run metadata or
        # trigger endless bootstrap resets for already registered consumers.
        self._recover_failed_streams()
        previous = self.discovery_error
        try:
            await self._discover()
            self.discovery_error = None
        except (OSError, ValueError, KeyError) as exc:
            self.discovery_error = str(exc)
        if self.discovery_error != previous:
            self.notify('discovery_error', {'reason': self.discovery_error or '', 'status': 'error' if self.discovery_error else 'available'})

    async def _discover(self):
        for relative, kind in (('views', 'projection'), ('metrics/evaluations', 'evaluation')):
            directory = self.root / relative
            if not directory.exists():
                continue
            found = await asyncio.to_thread(lambda: list(islice(directory.iterdir(), MAX_STREAMS + 1)))
            if len(found) > MAX_STREAMS:
                raise ValueError('Run exceeds bounded observation directory count')
            for path in sorted(found):
                if kind == 'projection':
                    if not _HEX.fullmatch(path.name):
                        continue
                    stream_id = 'projection:' + path.name
                    if stream_id in self.streams:
                        continue
                    if path.is_symlink() or not path.is_dir() or not (path / 'projection.json').is_file():
                        continue
                    self._register(stream_id)
                elif kind == 'evaluation':
                    if not _EVAL_DIR_RE.fullmatch(path.name):
                        continue
                    stream_id = 'evaluation:' + path.name
                    if stream_id in self.streams:
                        continue
                    if path.is_symlink() or not path.is_dir() or not (path / 'stream.json').is_file():
                        continue
                    record = await asyncio.to_thread(read_json, self.root, relative + '/' + path.name + '/stream.json')
                    expected = relative + '/' + path.name + '/events.jsonl'
                    if (record.get('schema_version') != 1 or record.get('run_id') != self.run_id
                            or record.get('stream_id') != stream_id or record.get('stream_generation') != path.name
                            or record.get('path') != expected):
                        raise ValueError('Invalid evaluation stream registration')
                    self.source_paths[stream_id] = expected
                    self._register(stream_id)

    async def _watch(self):
        while True:
            try:
                if self.run_id is None:
                    await self._activate()
                    await asyncio.sleep(self.poll_seconds)
                    continue
                await self.discover()
                manifest = await asyncio.to_thread(read_json, self.root, 'manifest.json')
                if manifest.get('run_id') != self.run_id:
                    raise ValueError('Observed run identity changed')
                prev_summary_sig = self._summary_cache[0] if self._summary_cache is not None else None
                await asyncio.to_thread(self._read_cached_summary)
                new_summary_sig = self._summary_cache[0] if self._summary_cache is not None else None
                revision = self._metadata_key(manifest)
                changed = manifest != self.manifest or new_summary_sig != prev_summary_sig
                self.manifest = manifest
                await self.refresh_artifacts()
                if changed:
                    self.notify('heartbeat', {'run': self.public_manifest()})
                if revision != self.metadata_revision:
                    self.manifest, self.metadata_revision = manifest, revision
                    self.notify('metadata', {'run': self.public_manifest(), 'revision': revision})
            except (OSError, ValueError, KeyError) as exc:
                self.notify('reset_required', {'reason': str(exc)})
            await asyncio.sleep(self.poll_seconds)

    def page(self, stream_id, cursor, *, limit=256):
        def safe_open(path):
            return _open(self.root, Path(path).relative_to(self.root).as_posix())
        if stream_id in self.source_paths:
            source = self.root / self.source_paths[stream_id]
            if not source.exists():
                raise FileNotFoundError('Training event stream has not been published yet')
            page = read_event_page(source.parent, cursor, limit=limit, max_bytes=1048576, include_cursors=True, _open_file=safe_open,
                _stream_identity={'run_id': self.run_id, 'stream_id': stream_id,
                                  'stream_generation': self.run_id if stream_id == 'training' else stream_id[11:]})
            for event in page['events']:
                if event.get('run_id') != self.run_id or event.get('stream_id', 'training') != stream_id:
                    raise ValueError('Source document run/stream identity differs from its registration')
            return page['events'], page['event_cursors'], page
        if not stream_id.startswith('projection:') or not _HEX.fullmatch(stream_id[11:]):
            raise ValueError('Unknown stream selector')
        if not (self.root / 'views' / stream_id[11:] / 'contributions.jsonl').exists():
            raise FileNotFoundError('Projection log has not been published yet')
        page = read_projection_page(self.root, stream_id[11:], cursor, limit=limit, max_bytes=1048576, _open_file=safe_open)
        if any(frame['source']['run_id'] != self.run_id for frame in page['frames']):
            raise ValueError('Projection document run identity differs from the selected run')
        return page['frames'], page['frame_cursors'], page

    def _index_source(self, event):
        if event.get('event') in ('start', 'resume'):
            attempt = event['attempt_id']
            if attempt in self.attempts or event['sequence'] != 1:
                raise ValueError('Duplicate or misplaced attempt lifecycle event')
            if event.get('parent_attempt_id') is not None and not isinstance(event['parent_attempt_id'], str):
                raise ValueError('Invalid recovery parent attempt identity')
            if attempt not in self.attempts and len(self.attempts) >= MAX_ATTEMPTS:
                raise ValueError('Run exceeds bounded lineage index of 4096 attempts')
            self.attempts[attempt] = {'parent': event.get('parent_attempt_id'),
                                     'restored_step': event.get('restored_step', 0)}
            self.notify('bootstrap_ready', {'status': 'lineage_ready'})

    async def _tail(self, stream):
        while True:
            try:
                frames, cursors, page = await asyncio.to_thread(self.page, stream.stream_id, stream.cursor, limit=2048)
                for frame, cursor in zip(frames, cursors):
                    if stream.stream_id in self.source_paths:
                        generation = frame.get('stream_generation', frame['run_id'])
                        if stream.generation is not None and stream.generation != generation:
                            raise ValueError('Source stream generation changed')
                        stream.generation = generation
                    if stream.stream_id == 'training':
                        self._index_source(frame)
                    elif stream.stream_id.startswith('projection:'):
                        if frame['source']['stream_id'] != 'training' or frame['source']['stream_generation'] != self.run_id:
                            raise ValueError('Metric projection is not from the training stream')
                        offset = cursor_offset(frame['source_cursor'])
                        if offset <= (stream.projected_offset or 0):
                            raise ValueError('Projection source cursor did not advance')
                        stream.projected_offset = offset
                    stream.cursor = cursor
                    stream.sequence = frame.get('projection_sequence', stream.sequence + 1)
                    envelope = {'stream_id': stream.stream_id, 'cursor': cursor, 'frame': frame}
                    for subscriber in list(self.subscribers):
                        if subscriber.accepts(stream.stream_id):
                            subscriber.offer('frame', envelope)
                stream.cursor = page['cursor']
                was_caught_up = stream.caught_up
                stream.caught_up = not page['has_more']
                if stream.caught_up and not was_caught_up:
                    self.notify('bootstrap_ready', {'stream_id': stream.stream_id, 'status': 'index_ready'})
                stream.error = None
                if frames and stream.stream_id == 'projection:' + MapSpec().revision:
                    self.notify('heartbeat', {'run': self.public_manifest()})
                if not page['has_more']:
                    if stream.stream_id.startswith('evaluation:') and stream.caught_up and stream.sequence >= 1:
                        return
                    await asyncio.sleep(self.poll_seconds)
                else:
                    await asyncio.sleep(0)
            except FileNotFoundError as exc:
                if stream.projected_offset is not None:
                    changed = stream.error != str(exc)
                    stream.error = str(exc)
                    if changed and stream.stream_id == 'projection:' + MapSpec().revision:
                        self.notify('heartbeat', {'run': self.public_manifest()})
                await asyncio.sleep(self.poll_seconds)
            except (OSError, ValueError, KeyError) as exc:
                stream.error = str(exc)
                # Corrupt/replaced complete history must not be silently skipped:
                # this tail ends, and discovery starts a new one only when the
                # file itself is replaced.
                self._failed[stream.stream_id] = self._file_identity(stream.stream_id) or (None, None, 0)
                if stream.stream_id == 'projection:' + MapSpec().revision:
                    self.notify('heartbeat', {'run': self.public_manifest()})
                self.notify('reset_required', {'stream_id': stream.stream_id, 'reason': str(exc)})
                return

    def check_subscription(self, stream_id):
        """Raise ValueError unless `subscribe(stream_id)` would succeed now."""
        if len(self.subscribers) >= MAX_SUBSCRIBERS:
            raise ValueError(f'Viewer has reached its {MAX_SUBSCRIBERS}-consumer limit')
        if stream_id != '*' and stream_id != 'training' and not (
                (stream_id.startswith('projection:') and _HEX.fullmatch(stream_id[11:])) or
                (stream_id.startswith('evaluation:') and _EVAL_DIR_RE.fullmatch(stream_id[11:]))):
            raise ValueError('Invalid stream selector')

    def subscribe(self, stream_id):
        self.check_subscription(stream_id)
        subscriber = Subscriber(stream_id, max_bytes=MAX_QUEUE_BYTES)
        self.subscribers.add(subscriber)
        # No await between registration and snapshot: queued live suffix starts
        # after these watermarks. Historical replay through them cannot lose events.
        watermarks = {key: value.cursor for key, value in self.streams.items() if subscriber.accepts(key)}
        return subscriber, watermarks

    async def events(self, stream_id, cursor=None):
        subscriber, watermarks = self.subscribe(stream_id)
        try:
            async for encoded in stream_events(self.page, subscriber, watermarks,
                    stream_id=stream_id, cursor=cursor, metadata=self.public_manifest, metadata_key="run"):
                yield encoded
        finally:
            self.subscribers.discard(subscriber)

    def lineage(self):
        source = self.streams.get('training')
        if source is None:
            raise LookupError('Run is waiting for its first events')
        if not source.caught_up:
            raise LookupError('Source lineage index is catching up')
        if source.error:
            raise ValueError(source.error)
        current = self.manifest.get('attempt_id')
        result, seen, through = [], set(), None
        while current is not None:
            if current in seen or len(seen) >= MAX_ATTEMPTS:
                raise ValueError('Invalid recovery lineage cycle/length')
            seen.add(current)
            attempt = self.attempts.get(current)
            if attempt is None:
                raise LookupError('Selected attempt is not yet indexed')
            result.append({'attempt_id': current, 'through_step': through})
            restored = attempt['restored_step']
            if type(restored) is not int or restored < 0:
                raise ValueError('Invalid restored step in source lineage')
            through = restored if through is None else min(through, restored)
            current = attempt['parent']
        return list(reversed(result))

    def bootstrap(self, map_revision, *, series, bucket_steps, step_from=0, step_to=None):
        if not _HEX.fullmatch(map_revision):
            raise ValueError('Invalid map revision')
        if type(bucket_steps) is not int or not 1 <= bucket_steps <= 2 ** 53 - 1:
            raise ValueError('bucket_steps must be a positive safe integer')
        if type(step_from) is not int or not 0 <= step_from <= 2 ** 53 - 1:
            raise ValueError('step_from must be a nonnegative safe integer')
        if step_to is not None and (type(step_to) is not int or not step_from <= step_to <= 2 ** 53 - 1):
            raise ValueError('step_to must be a safe integer at least step_from')
        if not isinstance(series, list) or not 1 <= len(series) <= 32 or len(set(series)) != len(series):
            raise ValueError('Select 1..32 distinct metric IDs')
        if any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in series):
            raise ValueError('Invalid metric IDs')
        query = dict(map_revision=map_revision, series=sorted(series), bucket_steps=bucket_steps,
                     step_from=step_from, step_to=step_to)
        lineage = self.lineage()
        query['lineage_revision'] = _digest(lineage)
        key = _digest(query)
        stream = self.streams.get('projection:' + map_revision)
        if stream is None:
            if map_revision == MapSpec().revision and (self.root / self.source_paths['training']).is_file():
                from .web_autostart import projection_failure
                failure = projection_failure(self.root)
                if failure is not None:
                    raise FileNotFoundError(f'Projection could not be built: {failure}')
                # The viewer's projector builds it; discovery then announces it.
                raise LookupError('Projection is being built')
            raise FileNotFoundError('Projection missing; run the explicit projection service')
        if stream.error:
            raise ValueError(stream.error)
        if key in self.jobs:
            cached = self.jobs[key]
            age = time.monotonic() - cached.completed if cached.completed is not None else 0
            lag = stream.sequence - cached.result['projection_sequence'] if cached.result is not None else 0
            # One fixed-H job while pending, then a completion lease for its
            # notification/refetch. Only a later explicit query may refresh it.
            if (cached.completed is None or not cached.delivered or age < 5
                    or (cached.result is not None and (lag == 0 or (lag <= 4096 and age <= 30)))):
                if cached.completed is not None:
                    cached.delivered = True
                self.jobs.move_to_end(key)
                return cached
            self.jobs.pop(key)
        if not stream.caught_up:
            raise LookupError('Projection index is catching up')
        if len(self.jobs) >= MAX_BOOTSTRAPS:
            completed = next((k for k, job in self.jobs.items() if job.task.done()), None)
            if completed is None:
                raise ValueError('Historical bootstrap queue is full')
            self.jobs.pop(completed)
        job = BootstrapJob(key, query)
        self.jobs[key] = job
        job.task = asyncio.create_task(self._reduce_history(job, stream.cursor, stream.sequence, lineage))
        return job

    async def _reduce_history(self, job, cursor, sequence, lineage):
        try:
            async with self._history_slots:
                result = await self._history(job.query, cursor, sequence, lineage)
            job.result = result
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            job.error = str(exc)
        job.completed = time.monotonic()
        self.notify('bootstrap_ready', {'query': job.query, 'job_id': job.key,
                                       'status': 'error' if job.error else 'ready'})

    async def _history(self, query, cursor, sequence, lineage):
        from foldlitedb.database import View, RustReducer
        from foldlitedb.projection import materialize_jsonl_view
        from .metrics_reducer import Reducer
        reducer = await asyncio.to_thread(Reducer)
        allowed = {x['attempt_id']: x['through_step'] for x in lineage}
        selected = set(query['series'])

        def metric_points(document):
            for emission in document.payload['document']['emissions']:
                metric, attempt, step = emission['key']
                if (metric not in selected or attempt not in allowed or
                        (allowed[attempt] is not None and step > allowed[attempt]) or
                        step < query['step_from'] or
                        (query['step_to'] is not None and step > query['step_to'])):
                    continue
                yield ([metric, emission['definition_hash'], attempt,
                        step // query['bucket_steps'] * query['bucket_steps']],
                       {'value': emission['value'], 'position': [step, emission['id']]})

        # This directory is disposable derived state. Keep all source-format and
        # lineage semantics in HyperGAN; Foldlite owns document replay and state.
        directory = self.view_cache_dir(query['map_revision'])
        for _ in range(2):  # before creating anything through a link, and after
            if directory.is_symlink() or directory.parent.is_symlink():
                raise ValueError('View cache directory must not be a symlink')
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / (_digest(query) + '.fold')
        if any(Path(str(path) + suffix).is_symlink() for suffix in ('', '.cache', '.lock')):
            raise ValueError('View cache file must not be a symlink')
        from .event_views import _load
        import base64
        marker = _load(base64.urlsafe_b64decode(cursor))
        view = View('metrics', version='1', map=metric_points,
                    reducer=RustReducer('envelope/v1', runtime=reducer, positioned=True),
                    references={'query': query, 'lineage': lineage,
                                'map_generation': marker['generation']})
        def read_page(saved, *, limit=2048):
            return self.page('projection:' + query['map_revision'], saved, limit=limit)
        try:
            rows = await materialize_jsonl_view(path, generation=marker['generation'], view=view,
                read_page=read_page, target_cursor=cursor, target_position=sequence,
                document_position=lambda document: document['projection_sequence'],
                cursor_position=cursor_offset, max_groups=MAX_GROUPS, timeout=self.history_timeout)
        except Exception as exc:
            from foldlitedb.types import DatabaseError
            if isinstance(exc, (DatabaseError, RuntimeError)) or 'Native log' in str(exc):
                raise RuntimeError(f"Cannot read or update derived view cache; verify cache access or remove {path} while observers are stopped, then retry") from exc
            raise
        await asyncio.to_thread(_prune_view_caches, directory, path)
        view = ViewSpec(query['map_revision'], bucket_steps=query['bucket_steps'])
        result = dict(schema_version=1, run_id=self.run_id, map_revision=query['map_revision'],
                      view_revision=view.revision, module_sha256=reducer.module_sha256,
                      bucket_steps=query['bucket_steps'], lineage_revision=query['lineage_revision'],
                      lineage=lineage, groups=sorted([{'key': row.key, 'state': row.state} for row in rows],
                                                    key=lambda group: group['key']),
                      cursor=cursor, projection_sequence=sequence, coverage={'complete': True},
                      step_from=query['step_from'], step_to=query['step_to'])
        if len(_json(result).encode()) > MAX_BOOTSTRAP_BYTES:
            raise ValueError('Bootstrap response exceeds 1 MiB; narrow the view')
        return result


def _prune_view_caches(directory, keep):
    """Bound the derived cache: `keep` (in `directory`) was just used.

    Within the run, the most recently used MAX_VIEW_CACHES caches remain; in the
    cache root, the MAX_VIEW_CACHE_RUNS most recently used runs remain.
    """
    import shutil

    def newest_first(paths):
        stamped = []
        for path in paths:
            try:
                stamped.append((path.stat().st_mtime_ns, path))
            except OSError:
                pass
        return [path for _, path in sorted(stamped, reverse=True)]
    owner = directory.parent
    try:
        os.utime(keep)
        os.utime(owner)
    except OSError:
        return
    for stale in newest_first(directory.glob('*.fold'))[MAX_VIEW_CACHES:]:
        if stale == keep:
            continue
        for suffix in ('', '.cache', '.lock'):
            try:
                Path(str(stale) + suffix).unlink(missing_ok=True)
            except OSError:
                pass
    try:
        # Dot directories hold other viewer state (projection failure records).
        runs = [path for path in owner.parent.iterdir()
                if path.is_dir() and not path.is_symlink() and not path.name.startswith('.')]
    except OSError:
        return
    for stale in newest_first(runs)[MAX_VIEW_CACHE_RUNS:]:
        if stale != owner:
            shutil.rmtree(stale, ignore_errors=True)


RunObservationService = ObservationService

"""Daemonless global run registry, Docker-style memorable slugs, and target resolution.

Operates strictly on lightweight JSON metadata and OS advisory locks (.run.lock).
Never imports torch, particlegan, or heavy training dependencies.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from ._file_lock import lock_file, unlock_file
from .summary import atomic_write_json, read_summary_json
from .workspace import find_workspace, load_workspace

_FID_RE = re.compile(r'(?:^|[^a-z0-9])fid(?:$|[^a-z]|50k)', re.IGNORECASE)


def _is_fid_metric(metric_name: Any) -> bool:
    if not metric_name:
        return False
    return bool(_FID_RE.search(str(metric_name)))



@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock_file(fd, blocking=True)
        try:
            yield
        finally:
            unlock_file(fd)
    finally:
        os.close(fd)

ADJECTIVES: Tuple[str, ...] = (
    "amber", "arctic", "azure", "bold", "bright", "calm", "clever", "cobalt",
    "coral", "cosmic", "crisp", "crimson", "crystal", "cyber", "daring", "dawn",
    "deep", "delta", "eager", "echo", "ember", "emerald", "epic", "fair",
    "fast", "fierce", "fleet", "fluent", "focal", "frost", "gentle", "gilded",
    "glacial", "golden", "grand", "granite", "harmonic", "hyper", "iconic", "indigo",
    "iron", "ivory", "jade", "keen", "kinetic", "lively", "lucid", "lunar",
    "magic", "marble", "mellow", "metric", "misty", "modern", "mystic", "neon",
    "nimble", "noble", "nova", "obsidian", "onyx", "opal", "orbit", "polar",
)

NOUNS: Tuple[str, ...] = (
    "anchor", "apex", "atlas", "aurora", "beacon", "blade", "breeze", "canyon",
    "cipher", "comet", "condor", "cortex", "cosmos", "crane", "crest", "crown",
    "delta", "drake", "dune", "eagle", "eclipse", "ember", "falcon", "fjord",
    "flame", "flux", "forge", "fox", "galaxy", "glacier", "griffin", "harbor",
    "hawk", "helix", "heron", "horizon", "husky", "kestrel", "lagoon", "lantern",
    "lynx", "matrix", "meadow", "meteor", "mirage", "monolith", "nebula", "nexus",
    "oasis", "orbit", "orion", "osprey", "otter", "panther", "peak", "phoenix",
    "photon", "pioneer", "prism", "pulsar", "quartz", "quasar", "raven", "reef",
)

_PYTEST_PATH_RE = re.compile(r"^(.*[/\\]pytest-of-[^/\\]+[/\\]pytest-\d+[/\\][^/\\]+)")


def slug_from_id(run_id: str) -> str:
    """Deterministically derive a memorable '<adj>-<noun>' slug from a run_id."""
    digest = hashlib.sha256(str(run_id).encode("utf-8")).digest()
    adj = ADJECTIVES[digest[0] % len(ADJECTIVES)]
    noun = NOUNS[digest[1] % len(NOUNS)]
    return f"{adj}-{noun}"


def extract_or_derive_slug(run_id: str, name: str = "", run_dir_name: str = "") -> str:
    """Return existing '<adj>-<noun>' suffix if present on name/dir, else derive from run_id."""
    for candidate in (run_dir_name, name):
        if not candidate:
            continue
        parts = candidate.lower().split("-")
        if len(parts) >= 2 and parts[-2] in ADJECTIVES and parts[-1] in NOUNS:
            return f"{parts[-2]}-{parts[-1]}"
    return slug_from_id(run_id or name or run_dir_name or "default")


def _ensure_unique_slug(
    candidate_slug: str,
    run_id: str,
    existing_runs: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> str:
    """Ensure candidate_slug does not collide with another run_id in existing_runs."""
    entries = existing_runs.values() if isinstance(existing_runs, dict) else existing_runs
    used_slugs = set()
    rid_str = str(run_id or "")
    for item in entries:
        if not isinstance(item, dict):
            continue
        other_id = str(item.get("run_id") or "")
        other_slug = str(item.get("slug") or "")
        if other_slug and other_id != rid_str:
            used_slugs.add(other_slug)

    if candidate_slug and candidate_slug not in used_slugs:
        return candidate_slug

    for salt in range(1, 256):
        rehashed = slug_from_id(f"{rid_str}:{salt}")
        if rehashed not in used_slugs:
            return rehashed

    base = candidate_slug or slug_from_id(rid_str or "default")
    idx = 2
    while f"{base}-{idx}" in used_slugs:
        idx += 1
    return f"{base}-{idx}"


def generate_run_slug(stem: str, parent_dir: Optional[Union[Path, str]] = None) -> Tuple[str, str]:
    """Generate a unique Docker-style '<stem>-<adj>-<noun>' run name and its '<adj>-<noun>' short slug."""
    clean_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(stem).strip()).strip("-") or "run"
    if clean_stem.startswith("train-"):
        clean_stem = clean_stem[len("train-"):]
    parent = Path(parent_dir).resolve() if parent_dir else None

    for _ in range(128):
        rand_bytes = os.urandom(2)
        adj = ADJECTIVES[rand_bytes[0] % len(ADJECTIVES)]
        noun = NOUNS[rand_bytes[1] % len(NOUNS)]
        short_slug = f"{adj}-{noun}"
        full_name = f"{clean_stem}-{short_slug}"
        if parent is None or not (parent / full_name).exists():
            return full_name, short_slug

    suffix = os.urandom(3).hex()
    short_slug = slug_from_id(suffix)
    return f"{clean_stem}-{short_slug}-{suffix[:4]}", short_slug


def is_run_dir(path: Union[Path, str]) -> bool:
    """Return True if path looks like a valid HyperGAN run directory."""
    try:
        p = Path(path)
        if not p.is_dir():
            return False
        return (
            (p / "manifest.json").is_file()
            or (p / "summary.json").is_file()
            or (p / "recipe.toml").is_file()
            or (p / "checkpoints").is_dir()
            or (p / "distributed-checkpoints").is_dir()
            or (p / "events.jsonl").is_file()
        )
    except OSError:
        return False


def find_workspace_runs_root(start_path: Optional[Union[Path, str]] = None) -> Optional[Path]:
    """Discover hypergan.toml from start_path and return its resolved runs_root if configured."""
    ws_file = find_workspace(start_path)
    if ws_file is None:
        return None
    try:
        ws_data = load_workspace(ws_file)
        runs_root = ws_data.get("workspace", {}).get("runs_root")
        if runs_root:
            return Path(runs_root).resolve()
    except Exception:
        pass
    return None


_LAST_PYTEST_ROOT: Dict[str, Path] = {}
_RECENTLY_PRUNED: Dict[str, Dict[str, Any]] = {}


def _is_isolated_test_env(cwd: Optional[Union[Path, str]] = None) -> Optional[Path]:
    """Detect if running inside a pytest temporary directory and return its test root."""
    if cwd is not None:
        try:
            m = _PYTEST_PATH_RE.match(str(Path(cwd).resolve()))
            if m:
                root = Path(m.group(1))
                cur_test = os.environ.get("PYTEST_CURRENT_TEST", "")
                if cur_test:
                    _LAST_PYTEST_ROOT[cur_test] = root
                return root
        except OSError:
            pass

    try:
        m = _PYTEST_PATH_RE.match(str(Path.cwd().resolve()))
        if m:
            root = Path(m.group(1))
            cur_test = os.environ.get("PYTEST_CURRENT_TEST", "")
            if cur_test:
                _LAST_PYTEST_ROOT[cur_test] = root
            return root
    except OSError:
        pass

    cur_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    if cur_test:
        cached = _LAST_PYTEST_ROOT.get(cur_test)
        if cached is not None and cached.exists():
            return cached
        # Inspect call stack frames for pytest tmp_path
        for depth in range(1, 16):
            try:
                frame = sys._getframe(depth)
            except ValueError:
                break
            for val in frame.f_locals.values():
                if isinstance(val, Path):
                    m = _PYTEST_PATH_RE.match(str(val))
                    if m:
                        root = Path(m.group(1))
                        _LAST_PYTEST_ROOT[cur_test] = root
                        return root
    return None


def get_registry_dir(cwd: Optional[Union[Path, str]] = None) -> Path:
    """Return the directory storing the run registry (runs.json and registry.lock).

    Honors HYPERGAN_HOME or HYPERGAN_STATE_DIR if set. Automatically isolates pytest
    temporary directories so unit tests never pollute or read the host user's
    ~/.local/share/hypergan registry.
    """
    env_home = os.environ.get("HYPERGAN_HOME") or os.environ.get("HYPERGAN_STATE_DIR")
    if env_home:
        return Path(env_home).resolve()

    cwd_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    pytest_root = _is_isolated_test_env(cwd_path if cwd is not None else None)
    if pytest_root is not None:
        return pytest_root / ".hypergan-registry"

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return cwd_path / ".hypergan-registry"

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "hypergan"

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "hypergan"
    return Path.home() / ".local" / "share" / "hypergan"


_get_state_dir = get_registry_dir


def _read_registry_unlocked(reg_dir: Path) -> Dict[str, Any]:
    reg_file = reg_dir / "runs.json"
    if reg_file.is_file():
        try:
            data = json.loads(reg_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("runs"), dict):
                data.setdefault("roots", [])
                seen_entries: List[Dict[str, Any]] = []
                for k, entry in data["runs"].items():
                    if isinstance(entry, dict):
                        rid = str(entry.get("run_id") or k)
                        raw_slug = str(entry.get("slug") or "") or extract_or_derive_slug(
                            rid,
                            name=str(entry.get("name") or ""),
                            run_dir_name=Path(str(entry.get("run_dir") or k)).name,
                        )
                        entry["slug"] = _ensure_unique_slug(raw_slug, rid, seen_entries)
                        seen_entries.append(entry)
                return data
        except Exception:
            pass
    return {"schema_version": 1, "runs": {}, "roots": []}


_load_registry_unlocked = _read_registry_unlocked


def _write_registry_unlocked(reg_dir: Path, data: Dict[str, Any]) -> None:
    reg_dir.mkdir(parents=True, exist_ok=True)
    if "roots" in data and isinstance(data["roots"], list):
        data["roots"] = [
            r for r in data["roots"]
            if str(Path(r).resolve()) != "/tmp" and str(r).strip() != "/tmp"
        ]
    atomic_write_json(reg_dir / "runs.json", data)


def find_live_pid_for_run(run_dir: Path) -> Optional[int]:
    """Scan /proc on Linux to find a live training process operating on run_dir."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    run_dir_str = str(run_dir.resolve())
    run_name = run_dir.name
    uid = os.getuid() if hasattr(os, "getuid") else None

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if uid is not None and entry.stat().st_uid != uid:
                continue
            cmdline_bytes = (entry / "cmdline").read_bytes()
            if not cmdline_bytes:
                continue
            parts = [p.decode("utf-8", errors="ignore") for p in cmdline_bytes.split(b"\x00") if p]
            if not parts:
                continue
            joined = " ".join(parts)
            if "hypergan" not in joined:
                continue
            # Exclude helper subprocesses
            if "hypergan.bounded_cli_output" in joined or "hypergan.web_autostart" in joined:
                continue
            if not any(tok in ("train", "resume") for tok in parts):
                continue
            if run_dir_str in parts or any(run_dir_str in p for p in parts) or any(p.endswith("/" + run_name) or p == run_name for p in parts):
                return int(entry.name)
        except OSError:
            continue
    return None


def _discover_system_roots_and_runs() -> Tuple[List[Path], List[Path]]:
    """Discover active runs_roots and run_dirs from viewer state files and /proc on host."""
    if os.environ.get("HYPERGAN_HOME") or os.environ.get("HYPERGAN_STATE_DIR"):
        return [], []
    roots: List[Path] = []
    run_dirs: List[Path] = []

    # 1. Check /tmp/hypergan-viewers-<uid>/*/state.json
    uid = os.getuid() if hasattr(os, "getuid") else None
    if uid is not None:
        viewer_tmp = Path(f"/tmp/hypergan-viewers-{uid}")
        if viewer_tmp.is_dir():
            for state_file in viewer_tmp.glob("*/state.json"):
                try:
                    sdata = json.loads(state_file.read_text(encoding="utf-8"))
                    root_str = sdata.get("root")
                    if root_str:
                        rp = Path(root_str).resolve()
                        if rp.is_dir():
                            if is_run_dir(rp):
                                run_dirs.append(rp)
                            elif str(rp) != "/tmp":
                                roots.append(rp)
                except Exception:
                    continue

    # 2. Check sibling/common training-runs workspace relative to HyperGAN checkout
    try:
        pkg_root = Path(__file__).resolve().parents[2]
        sibling_ws = pkg_root.parent / "training-runs" / "hypergan.toml"
        if sibling_ws.is_file():
            ws_data = load_workspace(sibling_ws)
            rr = ws_data.get("workspace", {}).get("runs_root")
            if rr and Path(rr).is_dir() and str(Path(rr).resolve()) != "/tmp":
                roots.append(Path(rr).resolve())
    except Exception:
        pass

    # 3. Scan /proc for active hypergan train / resume processes
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if uid is not None and entry.stat().st_uid != uid:
                    continue
                cmdline_bytes = (entry / "cmdline").read_bytes()
                if not cmdline_bytes or b"hypergan" not in cmdline_bytes:
                    continue
                parts = [p.decode("utf-8", errors="ignore") for p in cmdline_bytes.split(b"\x00") if p]
                if any("bounded_cli_output" in p or "web_autostart" in p for p in parts):
                    continue
                if not any(tok in ("train", "resume") for tok in parts):
                    continue
                for token in parts:
                    if token.startswith("-"):
                        continue
                    if "/" in token:
                        cand = Path(token)
                        if cand.is_dir() and is_run_dir(cand):
                            resolved = cand.resolve()
                            run_dirs.append(resolved)
                            if resolved.parent.is_dir() and str(resolved.parent) != "/tmp":
                                roots.append(resolved.parent)
            except OSError:
                continue

    return roots, run_dirs


discover_runs_on_host = _discover_system_roots_and_runs


def _read_run_id_fast(run_dir: Path) -> Optional[str]:
    """Read run_id from summary.json or manifest.json without crashing on corrupted files."""
    summary = read_summary_json(run_dir)
    if isinstance(summary, dict) and summary.get("run_id"):
        return str(summary["run_id"])
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("run_id"):
                return str(loaded["run_id"])
        except Exception:
            pass
    return None


def _relocate_missing_run_unlocked(
    entry: Dict[str, Any],
    *,
    cwd: Optional[Union[Path, str]] = None,
    extra_roots: Optional[List[Union[Path, str]]] = None,
) -> Optional[Path]:
    """Search for a renamed or moved run directory with the same run_id and heal entry in-place."""
    if not isinstance(entry, dict):
        return None
    target_run_id = str(entry.get("run_id") or "").strip()
    if not target_run_id:
        return None

    search_roots: List[Path] = []
    old_run_dir = entry.get("run_dir")
    if old_run_dir:
        old_p = Path(old_run_dir)
        try:
            if str(old_p.parent.resolve()) != "/tmp" and str(old_p.parent).strip() != "/tmp":
                search_roots.append(old_p.parent)
            if old_p.parent.parent != old_p.parent and str(old_p.parent.parent.resolve()) != "/tmp" and str(old_p.parent.parent).strip() != "/tmp":
                search_roots.append(old_p.parent.parent)
        except Exception:
            pass

    if entry.get("runs_root"):
        try:
            rr = Path(entry["runs_root"])
            if str(rr.resolve()) != "/tmp" and str(rr).strip() != "/tmp":
                search_roots.append(rr)
        except Exception:
            pass

    ws_ref = entry.get("workspace") or entry.get("workspace_root")
    if ws_ref:
        ws_runs = find_workspace_runs_root(ws_ref)
        if ws_runs:
            search_roots.append(ws_runs)
        ws_p = Path(ws_ref)
        search_roots.append(ws_p if ws_p.is_dir() else ws_p.parent)

    cwd_path = Path(cwd).resolve() if cwd is not None else (_is_isolated_test_env() or Path.cwd().resolve())
    has_ws = find_workspace(cwd_path) is not None
    ws_cwd_runs = find_workspace_runs_root(cwd_path)
    if ws_cwd_runs and str(ws_cwd_runs) != "/tmp":
        search_roots.append(ws_cwd_runs)
    if has_ws:
        if str(cwd_path) != "/tmp":
            search_roots.append(cwd_path)
    else:
        cwd_runs = cwd_path / "runs"
        if cwd_runs.is_dir() and str(cwd_runs) != "/tmp":
            search_roots.append(cwd_runs)

    pytest_root = _is_isolated_test_env(cwd_path)
    if pytest_root is not None:
        search_roots.append(pytest_root)

    if extra_roots:
        for r in extra_roots:
            try:
                search_roots.append(Path(r))
            except Exception:
                continue

    seen_roots = set()
    unique_roots: List[Path] = []
    for r in search_roots:
        try:
            rp = r.resolve()
            if rp.is_dir() and str(rp) != "/tmp" and str(rp) not in seen_roots:
                seen_roots.add(str(rp))
                unique_roots.append(rp)
        except OSError:
            continue

    def _apply_heal(found_dir: Path) -> Path:
        resolved_found = found_dir.resolve()
        old_dir_str = str(entry.get("run_dir") or "")
        old_basename = Path(old_dir_str).name if old_dir_str else ""
        if old_dir_str and old_dir_str != str(resolved_found):
            entry["_old_run_dir"] = old_dir_str
        if old_basename and old_basename != resolved_found.name:
            entry["_old_dir_name"] = old_basename
            if entry.get("name") == old_basename:
                entry["name"] = resolved_found.name
        entry["run_dir"] = str(resolved_found)
        entry["runs_root"] = str(resolved_found.parent)
        return resolved_found

    # Pass 1: Check immediate child directories (1 level deep) of each search root
    sub_containers: List[Path] = []
    for root in unique_roots:
        try:
            for child in root.iterdir():
                if child.name.startswith(".") or not child.is_dir():
                    continue
                if is_run_dir(child):
                    if _read_run_id_fast(child) == target_run_id:
                        return _apply_heal(child)
                else:
                    sub_containers.append(child)
        except OSError:
            continue

    # Pass 2: Check 1 level inside non-run subdirectories (handles mv run_a subdir/run_b)
    for sub in sub_containers[:32]:
        try:
            for grandchild in sub.iterdir():
                if grandchild.name.startswith(".") or not grandchild.is_dir():
                    continue
                if is_run_dir(grandchild) and _read_run_id_fast(grandchild) == target_run_id:
                    return _apply_heal(grandchild)
        except OSError:
            continue

    return None


def relocate_missing_run(
    entry: Dict[str, Any],
    *,
    cwd: Optional[Union[Path, str]] = None,
    extra_roots: Optional[List[Union[Path, str]]] = None,
) -> Optional[Path]:
    """Public helper to locate a moved/renamed run directory and persist healed entry to runs.json."""
    old_key = str(entry.get("run_dir") or "")
    found = _relocate_missing_run_unlocked(entry, cwd=cwd, extra_roots=extra_roots)
    if found is not None:
        reg_dir = get_registry_dir(cwd or found)
        if (reg_dir / "runs.json").is_file():
            with file_lock(reg_dir / "registry.lock"):
                data = _read_registry_unlocked(reg_dir)
                if old_key and old_key in data["runs"] and old_key != str(found):
                    del data["runs"][old_key]
                data["runs"][str(found)] = entry
                _write_registry_unlocked(reg_dir, data)
    return found


def _extract_run_entry(run_dir: Path, existing: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Build or refresh a registry metadata entry from run_dir on disk with Tier 1 summary_mtime_ns caching."""
    from .inspection import _read_events_tail_fallback, check_run_active

    try:
        run_dir = run_dir.resolve()
    except OSError:
        return None
    if not is_run_dir(run_dir):
        return None

    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    recipe_path = run_dir / "recipe.toml"
    events_path = run_dir / "events.jsonl"

    summary_mtime_ns: Optional[int] = None
    try:
        if summary_path.is_file():
            summary_mtime_ns = summary_path.stat().st_mtime_ns
    except OSError:
        summary_mtime_ns = None

    is_active, pid = check_run_active(run_dir)
    if is_active and pid is None:
        pid = find_live_pid_for_run(run_dir)
    elif not is_active:
        pid = None

    # Tier 1 O(1) cache hit: inactive run with unchanged summary.json st_mtime_ns
    if (
        existing is not None
        and not is_active
        and existing.get("status") not in ("running", "initializing")
        and existing.get("run_dir") == str(run_dir)
        and existing.get("run_id")
    ):
        if summary_mtime_ns is not None and existing.get("summary_mtime_ns") == summary_mtime_ns:
            if existing.get("pid") is not None or existing.get("active"):
                existing = dict(existing)
                existing["pid"] = None
                existing["active"] = False
            return existing

    mtime = 0.0
    for p in (summary_path, manifest_path, recipe_path, events_path):
        try:
            if p.is_file():
                mtime = max(mtime, p.stat().st_mtime)
        except OSError:
            pass

    if (
        existing is not None
        and not is_active
        and existing.get("status") not in ("running", "initializing")
        and summary_mtime_ns is None
        and existing.get("summary_mtime_ns") is None
        and existing.get("_mtime") == mtime
        and mtime > 0
        and existing.get("run_dir") == str(run_dir)
        and existing.get("run_id")
    ):
        if existing.get("pid") is not None or existing.get("active"):
            existing = dict(existing)
            existing["pid"] = None
            existing["active"] = False
        return existing

    summary = read_summary_json(run_dir)
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except Exception:
            manifest = {}

    events_fallback: Dict[str, Any] = {}
    if not summary and not manifest:
        events_fallback = _read_events_tail_fallback(run_dir)

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = (
        (summary.get("run_id") if summary else None)
        or manifest.get("run_id")
        or events_fallback.get("run_id")
        or (existing.get("run_id") if existing else None)
        or run_dir.name
    )
    raw_name = (
        (summary.get("name") if summary else None)
        or manifest.get("name")
        or config.get("name")
        or (existing.get("name") if existing else None)
        or run_dir.name
    )
    # If the user renamed the folder on disk and raw_name matched the previous folder name, reflect the new folder name
    if existing:
        old_dir_names = set()
        if existing.get("run_dir") and Path(existing["run_dir"]).name != run_dir.name:
            old_dir_names.add(Path(existing["run_dir"]).name)
        if existing.get("_old_dir_name"):
            old_dir_names.add(str(existing["_old_dir_name"]))
        if str(raw_name) in old_dir_names:
            raw_name = run_dir.name

    name = str(raw_name)
    slug = (
        (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or (existing.get("slug") if existing else None)
        or extract_or_derive_slug(str(run_id), name=name, run_dir_name=run_dir.name)
    )

    raw_status = (
        (summary.get("status") if (summary and summary.get("status") not in (None, "", "unknown")) else None)
        or (manifest.get("status") if manifest.get("status") not in (None, "", "unknown") else None)
    )
    if not raw_status:
        if not events_fallback:
            events_fallback = _read_events_tail_fallback(run_dir)
        raw_status = events_fallback.get("status") or "incomplete"

    recorded_status = str(raw_status).lower()
    if is_active:
        status = "running"
    elif recorded_status in ("running", "initializing"):
        status = "stopped"
    elif recorded_status == "complete":
        status = "completed"
    else:
        status = recorded_status

    step = summary.get("step") if summary else None
    if step is None:
        step = manifest.get("steps")
    if step is None:
        if not events_fallback:
            events_fallback = _read_events_tail_fallback(run_dir)
        step = events_fallback.get("step", 0)

    total_steps = (
        (summary.get("total_steps") if summary else None)
        or (summary.get("max_steps") if summary else None)
        or config.get("training", {}).get("steps")
        or config.get("train", {}).get("steps")
        or manifest.get("total_steps")
        or manifest.get("max_steps")
    )
    sps = (
        (summary.get("steps_per_sec") or summary.get("steps_per_second") or summary.get("speed_its") or summary.get("speed"))
        if summary
        else None
    )
    if sps is None:
        sps = manifest.get("steps_per_second") or manifest.get("speed")
    if sps is None and events_fallback:
        sps = events_fallback.get("steps_per_sec")

    best = summary.get("best") if summary else None
    best_fid = summary.get("best_fid") if summary else None
    if best and isinstance(best, dict):
        b_metric = str(best.get("metric") or "")
        if _is_fid_metric(b_metric) and isinstance(best.get("value"), (int, float)):
            best_fid = best.get("value")
        elif not _is_fid_metric(b_metric):
            best_fid = None
    elif best_fid is not None and isinstance(best, dict) and not _is_fid_metric(best.get("metric", "")):
        best_fid = None
    if best_fid is None and events_fallback:
        best_fid = events_fallback.get("best_fid")

    device = (
        (summary.get("device") if summary else None)
        or config.get("training", {}).get("device")
        or config.get("train", {}).get("device")
        or manifest.get("device")
        or (existing.get("device") if existing else None)
        or "cuda"
    )

    latest = (summary.get("latest", {}) if (summary and isinstance(summary.get("latest"), dict)) else {}) or (
        events_fallback.get("latest_metrics", {}) if events_fallback else {}
    )
    latest_metric_str = "--"
    for k in ("loss/g_total", "loss_g", "loss/total"):
        if k in latest and isinstance(latest[k], (int, float)):
            latest_metric_str = f"loss_g={latest[k]:.2f}"
            break

    updated_at = (summary.get("updated_at") if summary else None) or manifest.get("updated_at")
    if not updated_at and mtime > 0:
        updated_at = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    elif not updated_at:
        updated_at = ""

    max_steps_int = int(total_steps) if total_steps is not None else None
    speed_float = float(sps) if sps is not None else None

    result: Dict[str, Any] = {
        "run_id": str(run_id),
        "short_id": str(run_id)[:8],
        "slug": str(slug),
        "name": name,
        "run_dir": str(run_dir),
        "runs_root": str(run_dir.parent),
        "status": status,
        "active": bool(is_active),
        "pid": pid if is_active else None,
        "summary_mtime_ns": summary_mtime_ns if summary is not None else None,
        "step": int(step or 0),
        "max_steps": max_steps_int,
        "total_steps": max_steps_int,
        "speed": speed_float,
        "steps_per_sec": speed_float,
        "best_fid": float(best_fid) if best_fid is not None else None,
        "device": str(device),
        "latest_metric": latest_metric_str,
        "updated_at": str(updated_at),
        "_mtime": mtime,
    }
    if existing:
        for preserve_key in ("workspace", "workspace_root", "_old_dir_name", "_old_run_dir"):
            if preserve_key in existing:
                result[preserve_key] = existing[preserve_key]
    return result


_inspect_run_entry = _extract_run_entry


def register_run(
    run_dir: Union[Path, str],
    *,
    run_id: Optional[str] = None,
    name: Optional[str] = None,
    slug: Optional[str] = None,
    status: Optional[str] = None,
    pid: Optional[int] = None,
    workspace: Optional[Union[Path, str]] = None,
    workspace_root: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Register or update a run in the global/scoped registry with Tier 1 summary header cache."""
    run_path = Path(run_dir).resolve()
    reg_dir = get_registry_dir(run_path)
    reg_dir.mkdir(parents=True, exist_ok=True)
    lock_path = reg_dir / "registry.lock"

    with file_lock(lock_path):
        data = _read_registry_unlocked(reg_dir)
        key = str(run_path)
        prev = data["runs"].get(key)
        entry = _extract_run_entry(run_path, existing=None) or {}
        if prev:
            for k in ("workspace", "workspace_root", "_old_dir_name", "_old_run_dir"):
                if k in prev and k not in entry:
                    entry[k] = prev[k]
        if run_id:
            entry["run_id"] = str(run_id)
            entry["short_id"] = str(run_id)[:8]
        elif not entry.get("run_id"):
            entry["run_id"] = run_path.name
            entry["short_id"] = run_path.name[:8]
        if name:
            entry["name"] = str(name)
        elif not entry.get("name"):
            entry["name"] = run_path.name
        if slug:
            entry["slug"] = str(slug)
        elif not entry.get("slug"):
            entry["slug"] = extract_or_derive_slug(
                entry.get("run_id", run_path.name),
                name=entry.get("name", ""),
                run_dir_name=run_path.name,
            )
        entry["slug"] = _ensure_unique_slug(
            str(entry["slug"]),
            str(entry["run_id"]),
            {k: v for k, v in data["runs"].items() if k != key},
        )
        if status is not None:
            entry["status"] = "completed" if status == "complete" else status.lower()
            entry["active"] = entry["status"] == "running"
        elif not entry.get("status"):
            entry["status"] = "running"
            entry["active"] = True
        if pid is not None:
            entry["pid"] = int(pid)
        elif not entry.get("active"):
            entry["pid"] = None
        ws_target = workspace or workspace_root
        if ws_target is not None:
            ws_str = str(Path(ws_target).resolve())
            entry["workspace"] = ws_str
            entry["workspace_root"] = ws_str
        elif "workspace" not in entry:
            ws_discovered = find_workspace(run_path)
            if ws_discovered is not None:
                ws_str = str(ws_discovered.parent.resolve())
                entry["workspace"] = ws_str
                entry["workspace_root"] = ws_str

        summary_path = run_path / "summary.json"
        try:
            entry["summary_mtime_ns"] = summary_path.stat().st_mtime_ns if summary_path.is_file() else None
        except OSError:
            entry["summary_mtime_ns"] = None

        entry.setdefault("step", 0)
        entry.setdefault("max_steps", entry.get("total_steps"))
        entry.setdefault("total_steps", entry.get("max_steps"))
        entry.setdefault("speed", entry.get("steps_per_sec"))
        entry.setdefault("steps_per_sec", entry.get("speed"))
        entry.setdefault("best_fid", None)
        entry.setdefault("device", "cuda")
        entry["run_dir"] = key
        entry["runs_root"] = str(run_path.parent)
        if not entry.get("updated_at"):
            entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        data["runs"][key] = entry
        _RECENTLY_PRUNED.pop(str(entry.get("run_id", "")), None)

        parent_str = str(run_path.parent)
        try:
            is_tmp = str(Path(parent_str).resolve()) == "/tmp" or parent_str.strip() == "/tmp"
        except Exception:
            is_tmp = parent_str.strip() == "/tmp"
        if not is_tmp and parent_str not in data["roots"]:
            data["roots"].append(parent_str)
        _write_registry_unlocked(reg_dir, data)
        return entry


def unregister_run(
    run_dir_or_id: Union[Path, str],
    *,
    cwd: Optional[Union[Path, str]] = None,
) -> bool:
    """Cleanly remove a run from runs.json by directory path, run_id, short_id, slug, or name."""
    target_str = str(run_dir_or_id).strip()
    if not target_str:
        return False

    try:
        cand_path = Path(target_str)
        resolved_str = str(cand_path.resolve()) if (cand_path.is_absolute() or cwd is None) else str((Path(cwd).resolve() / cand_path).resolve())
    except OSError:
        resolved_str = target_str

    if cwd is not None:
        reg_dir = get_registry_dir(cwd)
    elif Path(target_str).is_absolute():
        reg_dir = get_registry_dir(target_str)
    else:
        reg_dir = get_registry_dir(None)

    if not (reg_dir / "runs.json").is_file():
        return False

    removed = False
    with file_lock(reg_dir / "registry.lock"):
        data = _read_registry_unlocked(reg_dir)
        keys_to_delete = []
        for k, entry in list(data["runs"].items()):
            if k in (resolved_str, target_str):
                keys_to_delete.append(k)
                continue
            if not isinstance(entry, dict):
                continue
            rdir = str(entry.get("run_dir") or "")
            rid = str(entry.get("run_id") or "")
            sid = str(entry.get("short_id") or rid[:8])
            slug = str(entry.get("slug") or "")
            name = str(entry.get("name") or "")
            if (
                rdir in (resolved_str, target_str)
                or rid == target_str
                or (len(target_str) >= 4 and rid.startswith(target_str))
                or sid == target_str
                or slug == target_str
                or name == target_str
                or (rdir and Path(rdir).name == target_str)
            ):
                keys_to_delete.append(k)

        for k in keys_to_delete:
            popped = data["runs"].pop(k, None)
            if isinstance(popped, dict) and popped.get("run_id"):
                _RECENTLY_PRUNED.pop(str(popped["run_id"]), None)
            removed = True

        if removed:
            _write_registry_unlocked(reg_dir, data)
    return removed


def list_registered_runs(
    cwd: Optional[Union[Path, str]] = None,
    extra_roots: Optional[List[Union[Path, str]]] = None,
) -> List[Dict[str, Any]]:
    """Synchronize and return registered runs, auto-healing moved folders and using Tier 1 mtime cache."""
    cwd_path = Path(cwd).resolve() if cwd is not None else (_is_isolated_test_env() or Path.cwd().resolve())
    reg_dir = get_registry_dir(cwd_path)
    reg_dir.mkdir(parents=True, exist_ok=True)

    is_test = (
        _is_isolated_test_env(cwd_path) is not None
        or bool(os.environ.get("PYTEST_CURRENT_TEST"))
        or bool(os.environ.get("HYPERGAN_HOME"))
        or bool(os.environ.get("HYPERGAN_STATE_DIR"))
    )

    with file_lock(reg_dir / "registry.lock"):
        data = _read_registry_unlocked(reg_dir)
        changed = False
        roots_to_scan: List[Path] = []
        direct_run_dirs: List[Path] = []

        if extra_roots:
            for r in extra_roots:
                try:
                    rp = Path(r).resolve()
                    if is_run_dir(rp):
                        direct_run_dirs.append(rp)
                    elif rp.is_dir():
                        roots_to_scan.append(rp)
                except OSError:
                    continue

        # Workspace runs_root from cwd
        has_ws = find_workspace(cwd_path) is not None
        ws_runs_root = find_workspace_runs_root(cwd_path)
        if ws_runs_root and ws_runs_root.is_dir() and str(ws_runs_root) != "/tmp":
            roots_to_scan.append(ws_runs_root)

        # CWD itself
        if is_run_dir(cwd_path):
            direct_run_dirs.append(cwd_path)
        elif has_ws:
            if cwd_path.is_dir() and str(cwd_path) != "/tmp":
                roots_to_scan.append(cwd_path)
        else:
            # Outside a workspace (no hypergan.toml):
            # Only scan <cwd>/runs if present; NEVER scan raw cwd (e.g. /tmp)
            cwd_runs = cwd_path / "runs"
            if cwd_runs.is_dir() and str(cwd_runs) != "/tmp":
                roots_to_scan.append(cwd_runs)

        # Persisted roots in registry (prune deleted roots gracefully)
        valid_persisted_roots: List[str] = []
        for r_str in list(data.get("roots", [])):
            try:
                rp = Path(r_str)
                is_tmp = str(rp.resolve()) == "/tmp" or str(r_str).strip() == "/tmp"
            except Exception:
                is_tmp = str(r_str).strip() == "/tmp"
            if is_tmp:
                changed = True
                continue
            if rp.is_dir():
                valid_persisted_roots.append(r_str)
                if not is_test or str(rp).startswith(str(reg_dir.parent)):
                    roots_to_scan.append(rp)
            else:
                changed = True
        if valid_persisted_roots != data.get("roots", []):
            data["roots"] = valid_persisted_roots
            changed = True

        # Host system auto-discovery (only outside isolated tests)
        if not is_test:
            sys_roots, sys_runs = _discover_system_roots_and_runs()
            roots_to_scan.extend(sys_roots)
            direct_run_dirs.extend(sys_runs)

        # Verify existing registered runs BEFORE scanning roots; attempt relocate_missing_run before pruning
        for existing_key in list(data["runs"].keys()):
            entry_obj = data["runs"].get(existing_key)
            rdir_str = (entry_obj.get("run_dir") if isinstance(entry_obj, dict) else None) or existing_key
            ep = Path(rdir_str)
            if ep.is_dir() and is_run_dir(ep):
                resolved_ep = ep.resolve()
                if existing_key != str(resolved_ep):
                    data["runs"].pop(existing_key, None)
                    if isinstance(entry_obj, dict):
                        entry_obj["run_dir"] = str(resolved_ep)
                        data["runs"][str(resolved_ep)] = entry_obj
                    changed = True
                direct_run_dirs.append(resolved_ep)
            else:
                old_entry = data["runs"].pop(existing_key, None)
                changed = True
                if isinstance(old_entry, dict):
                    old_entry_snapshot = dict(old_entry)
                    found_dir = _relocate_missing_run_unlocked(
                        old_entry,
                        cwd=cwd_path,
                        extra_roots=roots_to_scan,
                    )
                    if found_dir is not None:
                        healed_key = str(found_dir.resolve())
                        data["runs"][healed_key] = old_entry
                        direct_run_dirs.append(found_dir.resolve())
                    else:
                        rid_key = str(old_entry_snapshot.get("run_id") or existing_key)
                        _RECENTLY_PRUNED[rid_key] = old_entry_snapshot

        seen_roots = set()
        for root in roots_to_scan:
            try:
                r_str = str(root.resolve())
            except OSError:
                continue
            if r_str in seen_roots:
                continue
            seen_roots.add(r_str)
            if r_str not in data["roots"] and root != cwd_path and r_str != "/tmp":
                data["roots"].append(r_str)
                changed = True
            try:
                for child in root.iterdir():
                    if child.name.startswith(".") or not child.is_dir():
                        continue
                    if is_run_dir(child):
                        direct_run_dirs.append(child.resolve())
            except OSError:
                continue

        seen_runs = set()
        results: List[Dict[str, Any]] = []
        for rdir in direct_run_dirs:
            try:
                key = str(rdir.resolve())
            except OSError:
                continue
            if key in seen_runs:
                continue
            seen_runs.add(key)
            prev = data["runs"].get(key)
            entry = _extract_run_entry(rdir, existing=prev)
            if entry is not None:
                entry["slug"] = _ensure_unique_slug(
                    str(entry.get("slug") or ""),
                    str(entry.get("run_id") or key),
                    results,
                )
                if data["runs"].get(key) != entry:
                    data["runs"][key] = entry
                    changed = True
                _RECENTLY_PRUNED.pop(str(entry.get("run_id", "")), None)
                results.append(entry)

        if changed:
            _write_registry_unlocked(reg_dir, data)

    results.sort(key=lambda r: (bool(r.get("active")), str(r.get("updated_at", "") or "")), reverse=True)
    return results


def sync_registry(
    cwd: Optional[Union[Path, str]] = None,
    extra_roots: Optional[List[Union[Path, str]]] = None,
) -> List[Dict[str, Any]]:
    """Synchronize the registry with known roots, workspace runs_root, and live host runs."""
    return list_registered_runs(cwd=cwd, extra_roots=extra_roots)


def _format_disambiguation_list(entries: List[Dict[str, Any]]) -> str:
    lines = []
    for e in entries[:10]:
        short_id = e.get("short_id") or e.get("run_id", "")[:8]
        slug = e.get("slug", "")
        name = e.get("name", "")
        status = "RUNNING" if e.get("active") else e.get("status", "unknown").upper()
        step = e.get("step", 0)
        lines.append(f"  {short_id} ({slug})  {name}  [{status}, step {step:,}]  ({e.get('run_dir')})")
    return "\n".join(lines)


def _entry_matches_target(entry: Dict[str, Any], q_lower: str, cand_resolved_str: str) -> bool:
    """Return True if a registry entry matches target query by run_id prefix, slug, name, or path."""
    rdir = str(entry.get("run_dir") or "")
    old_rdir = str(entry.get("_old_run_dir") or "")
    rid = str(entry.get("run_id") or "").lower()
    sid = str(entry.get("short_id") or rid[:8]).lower()
    slug = str(entry.get("slug") or "").lower()
    name = str(entry.get("name") or "").lower()
    dir_name = Path(rdir).name.lower() if rdir else ""
    old_dir_name = str(entry.get("_old_dir_name") or "").lower()

    if cand_resolved_str and cand_resolved_str in (rdir, old_rdir):
        return True
    if q_lower in (rid, sid, slug, name, dir_name, old_dir_name) and q_lower:
        return True
    if dir_name and f"train-{q_lower}" == dir_name:
        return True
    if q_lower and (
        (rid and rid.startswith(q_lower))
        or (slug and slug.startswith(q_lower))
        or (name and name.startswith(q_lower))
        or (dir_name and dir_name.startswith(q_lower))
    ):
        return True
    return False


def resolve_run_target(
    target: Optional[Union[Path, str]] = None,
    *,
    cwd: Optional[Union[Path, str]] = None,
    allow_latest: bool = True,
    require_active: bool = False,
) -> Path:
    """Universal run resolver supporting zero-arg active/latest default, short IDs, slugs, names, and paths."""
    cwd_path = Path(cwd).resolve() if cwd is not None else (_is_isolated_test_env() or Path.cwd().resolve())

    if target is not None and str(target).strip() != "":
        target_str = str(target).strip()

        # 1. Direct filesystem path check (fast path)
        cand_path = Path(target_str)
        if not cand_path.is_absolute():
            cand_path = cwd_path / cand_path
        if cand_path.is_file() and cand_path.name == "manifest.json":
            return cand_path.parent.resolve()
        if is_run_dir(cand_path):
            return cand_path.resolve()

        # 2. Check workspace runs_root relative path
        ws_runs_root = find_workspace_runs_root(cwd_path)
        stem_str = target_str[:-5] if target_str.endswith(".toml") else target_str
        if ws_runs_root and ws_runs_root.is_dir():
            for variant in (target_str, stem_str, f"train-{stem_str}"):
                ws_cand = ws_runs_root / variant
                if is_run_dir(ws_cand):
                    return ws_cand.resolve()

        # Snapshot raw runs.json before sync_registry prunes deleted entries
        reg_dir = get_registry_dir(cwd_path)
        raw_before_sync: List[Dict[str, Any]] = []
        if (reg_dir / "runs.json").is_file():
            with file_lock(reg_dir / "registry.lock"):
                raw_data = _read_registry_unlocked(reg_dir)
                for k, v in raw_data.get("runs", {}).items():
                    if isinstance(v, dict):
                        snap = dict(v)
                        snap.setdefault("run_dir", k)
                        raw_before_sync.append(snap)

        # 3. Query synchronized registry (auto-heals relocated runs and prunes deleted ones)
        entries = list_registered_runs(cwd=cwd_path)

        # Special @ aliases
        if target_str in ("@active", "active"):
            active_entries = [e for e in entries if e.get("active")]
            if len(active_entries) == 1:
                return Path(active_entries[0]["run_dir"])
            if not active_entries:
                raise ValueError("No active training runs found.")
            raise ValueError(
                "Multiple training runs are active. Specify a run ID, slug, or name:\n"
                + _format_disambiguation_list(active_entries)
            )

        if target_str in ("@latest", "latest"):
            if not entries:
                raise FileNotFoundError("No runs found.")
            return Path(entries[0]["run_dir"])

        if target_str in ("@best", "best"):
            with_fid = [e for e in entries if e.get("best_fid") is not None]
            if not with_fid:
                raise ValueError("No runs with evaluated FID found.")
            with_fid.sort(key=lambda e: float(e["best_fid"]))
            return Path(with_fid[0]["run_dir"])

        # 4. Exact match on run_id, short_id, slug, name, or directory name (including healed old name)
        q_lower = stem_str.lower()
        cand_resolved_str = str(cand_path.resolve())
        exact_matches = [
            e for e in entries
            if q_lower in (
                e.get("run_id", "").lower(),
                e.get("short_id", "").lower(),
                e.get("slug", "").lower(),
                e.get("name", "").lower(),
                Path(e["run_dir"]).name.lower(),
                str(e.get("_old_dir_name") or "").lower(),
            )
            or f"train-{q_lower}" == Path(e["run_dir"]).name.lower()
            or cand_resolved_str in (str(e.get("run_dir") or ""), str(e.get("_old_run_dir") or ""))
        ]
        if len(exact_matches) == 1:
            return Path(exact_matches[0]["run_dir"])
        if len(exact_matches) > 1:
            active_exact = [e for e in exact_matches if e.get("active")]
            if len(active_exact) == 1:
                return Path(active_exact[0]["run_dir"])
            return Path(exact_matches[0]["run_dir"])

        # 5. Prefix match on run_id, slug, name, or directory name
        prefix_matches = [
            e for e in entries
            if e.get("run_id", "").lower().startswith(q_lower)
            or e.get("slug", "").lower().startswith(q_lower)
            or e.get("name", "").lower().startswith(q_lower)
            or Path(e["run_dir"]).name.lower().startswith(q_lower)
        ]
        if len(prefix_matches) == 1:
            return Path(prefix_matches[0]["run_dir"])
        if len(prefix_matches) > 1:
            active_pref = [e for e in prefix_matches if e.get("active")]
            if len(active_pref) == 1:
                return Path(active_pref[0]["run_dir"])
            raise ValueError(
                f"Ambiguous run reference {target_str!r} matches {len(prefix_matches)} runs:\n"
                + _format_disambiguation_list(prefix_matches)
            )

        # 6. Substring match on name, slug, or directory name
        sub_matches = [
            e for e in entries
            if q_lower in e.get("name", "").lower()
            or q_lower in e.get("slug", "").lower()
            or q_lower in Path(e["run_dir"]).name.lower()
        ]
        if len(sub_matches) == 1:
            return Path(sub_matches[0]["run_dir"])
        if len(sub_matches) > 1:
            active_sub = [e for e in sub_matches if e.get("active")]
            if len(active_sub) == 1:
                return Path(active_sub[0]["run_dir"])
            raise ValueError(
                f"Ambiguous run query {target_str!r} matches {len(sub_matches)} runs:\n"
                + _format_disambiguation_list(sub_matches)
            )

        # 7. Check if target matched a deleted entry in raw runs.json (or recently pruned)
        live_ids = {str(e.get("run_id") or "") for e in entries}
        stale_candidates = [
            e for e in [*raw_before_sync, *_RECENTLY_PRUNED.values()]
            if str(e.get("run_id") or "") not in live_ids
        ]
        for stale in stale_candidates:
            if _entry_matches_target(stale, q_lower, cand_resolved_str):
                raise FileNotFoundError(
                    f"Run '{stale.get('slug') or stale.get('name')}' ({str(stale.get('run_id', ''))[:8]}) "
                    f"was at {stale.get('run_dir')}, which no longer exists (removed stale entry from registry)."
                )

        # Fallback to resolved candidate path so callers raise their expected FileNotFoundError/ValueError
        return cand_path.resolve()

    # Zero-argument resolution (target is None)
    # 1. If CWD (or parent) is itself a run directory
    for p in (cwd_path, cwd_path.parent):
        if is_run_dir(p):
            return p.resolve()

    entries = list_registered_runs(cwd=cwd_path)
    active_entries = [e for e in entries if e.get("active")]

    if len(active_entries) == 1:
        return Path(active_entries[0]["run_dir"])
    if len(active_entries) > 1:
        raise ValueError(
            "Multiple training runs are currently active. Specify a run ID, slug, or name:\n"
            + _format_disambiguation_list(active_entries)
        )

    if require_active:
        raise ValueError("No active training runs found.")

    if allow_latest and entries:
        return Path(entries[0]["run_dir"])

    raise ValueError("No runs found. Start a training run with `hypergan train <model.toml>`.")

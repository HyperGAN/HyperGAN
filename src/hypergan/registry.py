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

_PYTEST_PATH_RE = re.compile(r"^(.*/pytest-of-[^/\\]+[/\\]pytest-\d+[/\\][^/\\]+)")


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


def _is_isolated_test_env(cwd: Optional[Path] = None) -> Optional[Path]:
    """Detect if running inside a pytest temporary directory and return its test root."""
    check_path = str((cwd or Path.cwd()).resolve())
    m = _PYTEST_PATH_RE.match(check_path)
    if m:
        return Path(m.group(1))
    return None


def get_registry_dir(cwd: Optional[Union[Path, str]] = None) -> Path:
    """Return the directory storing the run registry (runs.json and registry.lock).

    Honors HYPERGAN_HOME if set. Automatically isolates pytest temporary directories
    so unit tests never pollute or read the host user's ~/.local/share/hypergan registry.
    """
    env_home = os.environ.get("HYPERGAN_HOME")
    if env_home:
        return Path(env_home).resolve()

    cwd_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    pytest_root = _is_isolated_test_env(cwd_path)
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


def _read_registry_unlocked(reg_dir: Path) -> Dict[str, Any]:
    reg_file = reg_dir / "runs.json"
    if reg_file.is_file():
        try:
            data = json.loads(reg_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("runs"), dict):
                data.setdefault("roots", [])
                return data
        except Exception:
            pass
    return {"schema_version": 1, "runs": {}, "roots": []}


def _write_registry_unlocked(reg_dir: Path, data: Dict[str, Any]) -> None:
    reg_dir.mkdir(parents=True, exist_ok=True)
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
                            else:
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
            if rr and Path(rr).is_dir():
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
                            if resolved.parent.is_dir():
                                roots.append(resolved.parent)
            except OSError:
                continue

    return roots, run_dirs


def _extract_run_entry(run_dir: Path, existing: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Build or refresh a registry metadata entry from run_dir on disk."""
    from .inspection import check_run_active

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
    mtime = 0.0
    for p in (summary_path, manifest_path, recipe_path, events_path):
        try:
            if p.is_file():
                mtime = max(mtime, p.stat().st_mtime)
        except OSError:
            pass

    is_active, pid = check_run_active(run_dir)
    if is_active and pid is None:
        pid = find_live_pid_for_run(run_dir)

    # Fast cache hit if run is not active, directory path & mtime have not changed
    if (
        existing is not None
        and not is_active
        and existing.get("status") not in ("running", "initializing")
        and existing.get("_mtime") == mtime
        and existing.get("run_dir") == str(run_dir)
        and existing.get("run_id")
    ):
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

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    run_id = (summary.get("run_id") if summary else None) or manifest.get("run_id") or run_dir.name
    raw_name = (
        (summary.get("name") if summary else None)
        or manifest.get("name")
        or config.get("name")
        or run_dir.name
    )
    # If the user renamed the folder on disk and raw_name matched the previous folder name, reflect the new folder name
    if existing and existing.get("run_dir") and Path(existing["run_dir"]).name != run_dir.name:
        old_dir_name = Path(existing["run_dir"]).name
        if str(raw_name) == old_dir_name:
            raw_name = run_dir.name

    name = str(raw_name)
    slug = (
        (summary.get("slug") if summary else None)
        or manifest.get("slug")
        or (existing.get("slug") if existing else None)
        or extract_or_derive_slug(str(run_id), name=name, run_dir_name=run_dir.name)
    )

    recorded_status = ((summary.get("status") if summary else None) or manifest.get("status") or "unknown").lower()
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
        step = manifest.get("steps", 0)
    total_steps = (
        (summary.get("total_steps") if summary else None)
        or config.get("training", {}).get("steps")
        or config.get("train", {}).get("steps")
        or manifest.get("total_steps")
    )
    best = summary.get("best") if summary else None
    best_fid = (best.get("value") if isinstance(best, dict) else None) or (summary.get("best_fid") if summary else None)

    updated_at = (summary.get("updated_at") if summary else None) or manifest.get("updated_at")
    if not updated_at and mtime > 0:
        updated_at = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    elif not updated_at:
        updated_at = ""

    return {
        "run_id": str(run_id),
        "short_id": str(run_id)[:8],
        "slug": str(slug),
        "name": name,
        "run_dir": str(run_dir),
        "runs_root": str(run_dir.parent),
        "status": status,
        "active": bool(is_active),
        "pid": pid,
        "step": int(step or 0),
        "total_steps": int(total_steps) if total_steps is not None else None,
        "best_fid": float(best_fid) if best_fid is not None else None,
        "updated_at": str(updated_at),
        "_mtime": mtime,
    }


def register_run(
    run_dir: Union[Path, str],
    *,
    run_id: Optional[str] = None,
    name: Optional[str] = None,
    slug: Optional[str] = None,
    status: str = "running",
    pid: Optional[int] = None,
    workspace_root: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Register or update a run in the global/scoped registry."""
    run_path = Path(run_dir).resolve()
    reg_dir = get_registry_dir(run_path)
    reg_dir.mkdir(parents=True, exist_ok=True)
    lock_path = reg_dir / "registry.lock"

    with file_lock(lock_path):
        data = _read_registry_unlocked(reg_dir)
        key = str(run_path)
        entry = _extract_run_entry(run_path, existing=None) or {}
        if run_id:
            entry["run_id"] = str(run_id)
            entry["short_id"] = str(run_id)[:8]
        if name:
            entry["name"] = str(name)
        if slug:
            entry["slug"] = str(slug)
        elif not entry.get("slug"):
            entry["slug"] = extract_or_derive_slug(
                entry.get("run_id", run_path.name),
                name=entry.get("name", ""),
                run_dir_name=run_path.name,
            )
        if status:
            entry["status"] = "completed" if status == "complete" else status.lower()
            entry["active"] = entry["status"] == "running"
        if pid is not None:
            entry["pid"] = int(pid)
        if workspace_root is not None:
            entry["workspace_root"] = str(Path(workspace_root).resolve())
        entry["run_dir"] = key
        entry["runs_root"] = str(run_path.parent)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        data["runs"][key] = entry

        parent_str = str(run_path.parent)
        if parent_str not in data["roots"]:
            data["roots"].append(parent_str)
        _write_registry_unlocked(reg_dir, data)
        return entry


def unregister_run(run_dir: Union[Path, str]) -> None:
    """Remove a deleted run directory from the registry."""
    run_path = Path(run_dir).resolve()
    reg_dir = get_registry_dir(run_path)
    if not (reg_dir / "runs.json").is_file():
        return
    with file_lock(reg_dir / "registry.lock"):
        data = _read_registry_unlocked(reg_dir)
        if str(run_path) in data["runs"]:
            del data["runs"][str(run_path)]
            _write_registry_unlocked(reg_dir, data)


def sync_registry(
    cwd: Optional[Union[Path, str]] = None,
    extra_roots: Optional[List[Union[Path, str]]] = None,
) -> List[Dict[str, Any]]:
    """Synchronize the registry with known roots, workspace runs_root, and live host runs."""
    cwd_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    reg_dir = get_registry_dir(cwd_path)
    reg_dir.mkdir(parents=True, exist_ok=True)

    is_test = (
        _is_isolated_test_env(cwd_path) is not None
        or bool(os.environ.get("PYTEST_CURRENT_TEST"))
        or bool(os.environ.get("HYPERGAN_HOME"))
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
        ws_runs_root = find_workspace_runs_root(cwd_path)
        if ws_runs_root and ws_runs_root.is_dir():
            roots_to_scan.append(ws_runs_root)

        # CWD itself
        if is_run_dir(cwd_path):
            direct_run_dirs.append(cwd_path)
        elif cwd_path.is_dir():
            roots_to_scan.append(cwd_path)

        # Persisted roots in registry (prune deleted roots gracefully)
        valid_persisted_roots: List[str] = []
        for r_str in list(data.get("roots", [])):
            rp = Path(r_str)
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

        seen_roots = set()
        for root in roots_to_scan:
            try:
                r_str = str(root.resolve())
            except OSError:
                continue
            if r_str in seen_roots:
                continue
            seen_roots.add(r_str)
            if r_str not in data["roots"] and root != cwd_path:
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

        # Verify existing registered runs; prune deleted ones and track by run_id to detect renames
        removed_by_run_id: Dict[str, Dict[str, Any]] = {}
        for existing_key in list(data["runs"].keys()):
            ep = Path(existing_key)
            if ep.is_dir() and is_run_dir(ep):
                direct_run_dirs.append(ep)
            else:
                old_entry = data["runs"].pop(existing_key, None)
                if isinstance(old_entry, dict) and old_entry.get("run_id"):
                    removed_by_run_id[str(old_entry["run_id"])] = old_entry
                changed = True

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
            if prev is None and removed_by_run_id:
                # Check if this directory is a renamed/moved run matching a removed run_id
                s_tmp = read_summary_json(rdir)
                r_id = s_tmp.get("run_id") if s_tmp else None
                if not r_id and (rdir / "manifest.json").is_file():
                    try:
                        r_id = json.loads((rdir / "manifest.json").read_text(encoding="utf-8")).get("run_id")
                    except Exception:
                        pass
                if r_id and str(r_id) in removed_by_run_id:
                    prev = removed_by_run_id[str(r_id)]

            entry = _extract_run_entry(rdir, existing=prev)
            if entry is not None:
                if data["runs"].get(key) != entry:
                    data["runs"][key] = entry
                    changed = True
                results.append(entry)

        if changed:
            _write_registry_unlocked(reg_dir, data)

    results.sort(key=lambda r: (bool(r.get("active")), str(r.get("updated_at", "") or "")), reverse=True)
    return results


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


def resolve_run_target(
    target: Optional[Union[Path, str]] = None,
    *,
    cwd: Optional[Union[Path, str]] = None,
    allow_latest: bool = True,
    require_active: bool = False,
) -> Path:
    """Universal run resolver supporting zero-arg active/latest default, short IDs, slugs, names, and paths."""
    cwd_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()

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

        # 3. Query synchronized registry
        entries = sync_registry(cwd=cwd_path)

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

        # 4. Exact match on run_id, short_id, slug, name, or directory name
        q_lower = stem_str.lower()
        exact_matches = [
            e for e in entries
            if q_lower in (
                e.get("run_id", "").lower(),
                e.get("short_id", "").lower(),
                e.get("slug", "").lower(),
                e.get("name", "").lower(),
                Path(e["run_dir"]).name.lower(),
            )
            or f"train-{q_lower}" == Path(e["run_dir"]).name.lower()
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

        # Fallback to resolved candidate path so callers raise their expected FileNotFoundError/ValueError
        return cand_path.resolve()

    # Zero-argument resolution (target is None)
    # 1. If CWD (or parent) is itself a run directory
    for p in (cwd_path, cwd_path.parent):
        if is_run_dir(p):
            return p.resolve()

    entries = sync_registry(cwd=cwd_path)
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

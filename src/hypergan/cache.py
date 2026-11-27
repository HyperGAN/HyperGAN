"""On-demand external URL cache (~/.hypergan/cache/) with multi-rank file locking.

Downloads external HTTPS resources on demand, verifies SHA-256 digests and
file sizes, uses cross-process locking (<path>.lock) to prevent race conditions
across multi-GPU ranks, and enforces strict offline and local-path invariants.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
import urllib.error
import urllib.parse
import urllib.request

from ._file_lock import lock_file, unlock_file

RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

INCEPTION_V3_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)
INCEPTION_V3_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"

DINOV2_VITL14_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
)
DINOV2_VITL14_SHA256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
DINOV2_VITL14_SIZE = 1217586395

_CHUNK_BYTES = 64 * 1024
_DEFAULT_TIMEOUT = 60.0
_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


class _CacheValidationError(ValueError):
    """Internal marker for checksum/size mismatches where .part must be discarded."""


def _thread_lock_for(lock_path: Path) -> threading.RLock:
    key = lock_path.resolve(strict=False)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _validate_cache_filename(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"Cache filename must be a string, got {type(name).__name__}")
    cleaned = name.strip()
    if (
        not cleaned
        or cleaned in (".", "..")
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
        or os.path.splitdrive(cleaned)[0]
        or Path(cleaned).is_absolute()
        or Path(cleaned).name != cleaned
    ):
        raise ValueError(f"Invalid cache filename (path traversal or separator rejected): {name!r}")
    return cleaned


class OfflineError(FileNotFoundError):
    """Raised when an external resource is not cached and HYPERGAN_OFFLINE=1 is set."""


def is_offline() -> bool:
    """Check whether HyperGAN is running in strict offline mode."""
    return os.environ.get("HYPERGAN_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def get_cache_dir() -> Path:
    """Return the root cache directory (~/.hypergan/cache/ or $HYPERGAN_CACHE_DIR)."""
    env_dir = os.environ.get("HYPERGAN_CACHE_DIR")
    if env_dir and env_dir.strip():
        cache_dir = Path(env_dir.strip()).expanduser().resolve()
    else:
        cache_dir = (Path.home() / ".hypergan" / "cache").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _is_url(path_or_url: str | Path) -> bool:
    s = str(path_or_url).strip()
    return s.startswith("http://") or s.startswith("https://")


def _url_to_filename(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    raw = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    name = raw.strip()
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or os.path.splitdrive(name)[0]
        or Path(name).name != name
    ):
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"blob-{digest}"
    return name


@contextmanager
def cache_file_lock(lock_path: Path):
    """Context manager acquiring an exclusive per-path multi-process and multi-thread lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink() or lock_path.is_dir():
        raise ValueError(f"Unsafe cache lock path: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    with _thread_lock_for(lock_path):
        if lock_path.is_symlink() or lock_path.is_dir():
            raise ValueError(f"Unsafe cache lock path: {lock_path}")
        fd = os.open(str(lock_path), flags, 0o666)
        try:
            lock_file(fd, blocking=True)
            try:
                yield
            finally:
                unlock_file(fd)
        finally:
            os.close(fd)


def _validate_local_file(path: Path, expected_sha256: str | None = None, expected_size: int | None = None) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required local artifact is missing: {path}; no automatic downloads")
    if expected_sha256 is not None or expected_size is not None:
        actual_size, actual_sha = _hash_file(path)
        if expected_size is not None and actual_size != expected_size:
            raise ValueError(
                f"Local artifact size mismatch for {path}: expected {expected_size} bytes, got {actual_size} bytes"
            )
        if expected_sha256 is not None:
            expected_lower = expected_sha256.lower().strip()
            if actual_sha != expected_lower:
                raise ValueError(
                    f"Local artifact SHA256 mismatch for {path}: expected {expected_lower}, got {actual_sha}"
                )
    return path


def ensure_cached_url(
    url: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    filename: str | None = None,
    cache_dir: str | Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download url to cache directory with multi-rank locking, verification, and resume."""
    if not _is_url(url):
        raise ValueError(f"ensure_cached_url requires an http:// or https:// URL, got {url!r}")

    target_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else get_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        dest_name = _validate_cache_filename(filename)
    else:
        base_name = _validate_cache_filename(_url_to_filename(url))
        if expected_sha256 is None and not base_name.startswith("blob-"):
            url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            dest_name = f"{url_digest}-{base_name}"
        else:
            dest_name = base_name
    dest_path = target_dir / dest_name
    if dest_path.parent.resolve() != target_dir:
        raise ValueError(f"Cache destination escapes target directory: {dest_name!r}")
    lock_path = dest_path.with_name(dest_path.name + ".lock")
    part_path = dest_path.with_name(dest_path.name + ".part")

    def _reject_unsafe_paths():
        for label, path_obj in (("destination", dest_path), ("partial download", part_path), ("lock", lock_path)):
            if path_obj.is_symlink() or path_obj.is_dir():
                raise ValueError(f"Unsafe cache {label} path: {path_obj}")

    _reject_unsafe_paths()

    with cache_file_lock(lock_path):
        _reject_unsafe_paths()

        # 1. Fast-path check: already cached and valid
        if dest_path.is_file():
            try:
                _validate_local_file(dest_path, expected_sha256, expected_size)
                return dest_path
            except ValueError:
                dest_path.unlink(missing_ok=True)

        # 2. Offline guard
        if is_offline():
            raise OfflineError(
                f"HyperGAN is running in offline mode (HYPERGAN_OFFLINE=1) and artifact is not cached: "
                f"{url} -> {dest_path}"
            )

        # 3. Handle partial download resume
        existing_bytes = part_path.stat().st_size if part_path.is_file() else 0
        if expected_size is not None and existing_bytes > expected_size:
            part_path.unlink(missing_ok=True)
            existing_bytes = 0
        if existing_bytes > 0 and (
            (expected_size is not None and existing_bytes == expected_size)
            or (expected_size is None and expected_sha256 is not None)
        ):
            actual_size, actual_sha = _hash_file(part_path)
            size_ok = expected_size is None or actual_size == expected_size
            sha_ok = expected_sha256 is None or actual_sha == expected_sha256.lower().strip()
            if size_ok and sha_ok:
                _reject_unsafe_paths()
                os.replace(part_path, dest_path)
                return dest_path
            if expected_size is not None and existing_bytes == expected_size:
                part_path.unlink(missing_ok=True)
                existing_bytes = 0

        headers = {"User-Agent": "HyperGAN/2.0 cache-downloader"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if existing_bytes > 0 and exc.code == 416:
                part_path.unlink(missing_ok=True)
                existing_bytes = 0
                req = urllib.request.Request(url, headers={"User-Agent": "HyperGAN/2.0 cache-downloader"})
                response = urllib.request.urlopen(req, timeout=timeout)
            else:
                raise

        with response:
            status = getattr(response, "status", None) or response.getcode()
            if existing_bytes > 0 and status == 206:
                mode = "ab"
                initial_bytes = existing_bytes
            else:
                mode = "wb"
                initial_bytes = 0

            content_length_raw = response.headers.get("Content-Length") if response.headers else None
            expected_response_bytes = (
                int(content_length_raw)
                if content_length_raw is not None and content_length_raw.strip().isdigit()
                else None
            )

            written_session = 0
            try:
                _reject_unsafe_paths()
                open_flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                open_flags |= os.O_APPEND if mode == "ab" else os.O_TRUNC
                part_fd = os.open(str(part_path), open_flags, 0o666)
                with os.fdopen(part_fd, mode) as handle:
                    while True:
                        chunk = response.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        written_session += len(chunk)
                        total_bytes = initial_bytes + written_session
                        if expected_size is not None and total_bytes > expected_size:
                            raise _CacheValidationError(
                                f"Download from {url} exceeded declared size of {expected_size} bytes "
                                f"(received {total_bytes} bytes)"
                            )
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())

                if expected_response_bytes is not None and written_session < expected_response_bytes:
                    raise OSError(
                        f"Truncated download from {url}: expected {expected_response_bytes} bytes "
                        f"in response body, got {written_session} bytes"
                    )

                actual_size, actual_sha = _hash_file(part_path)
                if expected_size is not None and actual_size != expected_size:
                    raise _CacheValidationError(
                        f"Downloaded artifact size mismatch for {dest_name}: "
                        f"expected {expected_size} bytes, got {actual_size} bytes"
                    )
                if expected_sha256 is not None:
                    expected_lower = expected_sha256.lower().strip()
                    if actual_sha != expected_lower:
                        raise _CacheValidationError(
                            f"Downloaded artifact SHA256 mismatch for {dest_name}: "
                            f"expected {expected_lower}, got {actual_sha}"
                        )

                _reject_unsafe_paths()
                os.replace(part_path, dest_path)
                return dest_path
            except _CacheValidationError:
                # Discard partial file only on size/hash validation mismatch;
                # preserve .part across transient network/timeout errors so HTTP Range resume works.
                part_path.unlink(missing_ok=True)
                raise


def resolve_or_cache(
    path_or_url: str | Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    filename: str | None = None,
    cache_dir: str | Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Resolve local path without copying, or download external URL into cache."""
    s = str(path_or_url).strip()

    # Local file:// URI pass-through
    if s.startswith("file://"):
        parsed = urllib.parse.urlsplit(s)
        local_path = Path(urllib.request.url2pathname(parsed.path)).expanduser().resolve()
        return _validate_local_file(local_path, expected_sha256, expected_size)

    # Remote HTTP / HTTPS URL caching
    if _is_url(s):
        return ensure_cached_url(
            s,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            filename=filename,
            cache_dir=cache_dir,
            timeout=timeout,
        )

    # Local filesystem path pass-through
    local_path = Path(s).expanduser().resolve()
    return _validate_local_file(local_path, expected_sha256, expected_size)

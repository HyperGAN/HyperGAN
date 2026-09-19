"""Bounded local viewer file access; paths come from indexes, never HTTP input."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


@contextmanager
def _open(root, relative):
    root = Path(root).resolve(strict=True)
    if not isinstance(relative, str):
        raise ValueError("Indexed path must be a relative path inside the run")
    parts = PurePosixPath(relative).parts
    if (not isinstance(relative, str) or not parts or relative != "/".join(parts)
            or any(p in (".", "..") or "\\" in p or ":" in p for p in parts)
            or relative.startswith("/")):
        raise ValueError("Indexed path must be a relative path inside the run")
    descriptors = []
    try:
        if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            descriptors.append(parent)
            for part in parts[:-1]:
                parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                descriptors.append(parent)
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        else:
            # Windows has no dir_fd traversal. Refuse every symlink/reparse path
            # before opening; the selected run remains owner-controlled local data.
            path = root
            for part in parts:
                path /= part
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError("Indexed paths must not traverse links")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        descriptors.append(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Indexed artifact must be a regular file")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            yield stream
    except OSError as error:
        raise ValueError(f"Indexed file is unavailable: {error.strerror}") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_bytes(root, relative, *, max_bytes=1048576):
    if type(max_bytes) is not int or not 1 <= max_bytes <= 16 * 1048576:
        raise ValueError("File byte budget must be in 1..16777216")
    with _open(root, relative) as stream:
        if os.fstat(stream.fileno()).st_size > max_bytes:
            raise ValueError("Indexed file exceeds byte budget")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Indexed file exceeds byte budget")
    return data


def read_json(root, relative, *, max_bytes=1048576):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    value = json.loads(read_bytes(root, relative, max_bytes=max_bytes), parse_constant=reject)
    json.dumps(value, allow_nan=False)  # Exponents such as 1e999 also fail.
    return value


def read_artifact(root, record):
    """Read a selected immutable index entry, checking its size and digest."""
    size = record.get("bytes")
    digest = record.get("sha256")
    if type(size) is not int or not 0 <= size <= 16 * 1048576:
        raise ValueError("Invalid indexed artifact size")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Invalid indexed artifact digest")
    data = read_bytes(root, record["path"], max_bytes=max(1, size))
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Indexed artifact content changed")
    return data

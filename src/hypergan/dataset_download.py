"""Opt-in dataset acquisition, checksum verification, and manifest preparation.

This module is intentionally free of heavy runtime dependencies (such as torch
or numpy) so that ``hypergan download-data`` and ``hypergan.dataset_info`` work
in lightweight environments. Training never invokes network downloads implicitly.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request

CIFAR10_SHA256 = {
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}

QUALIFIED_DATASETS = {
    "cifar10": {
        "name": "cifar10",
        "title": "CIFAR-10 (Python binary batches)",
        "version": "python-1.0",
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "archive_filename": "cifar-10-python.tar.gz",
        "expected_bytes": 170_498_071,
        "archive_sha256": "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        "extracted_dir": "cifar-10-batches-py",
        "files_sha256": dict(CIFAR10_SHA256),
        "max_extract_files": 32,
        "max_extract_bytes": 256 * 1024 * 1024,
        "factory": "hypergan.datasets:CIFAR10Data",
        "citation": (
            "Alex Krizhevsky (2009), 'Learning Multiple Layers of Features from Tiny Images', "
            "University of Toronto technical report."
        ),
        "usage_notice": (
            "CIFAR-10 is distributed by Alex Krizhevsky (University of Toronto) for research "
            "and educational use. Cite 'Learning Multiple Layers of Features from Tiny Images' (2009) "
            "when publishing results. HyperGAN does not redistribute or relicense dataset bytes."
        ),
    },
}

_DATASET_ALIASES = {
    "cifar-10": "cifar10",
}

_CHUNK_BYTES = 64 * 1024


def _normalize_dataset_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name must be a non-empty string")
    cleaned = name.strip().lower()
    return _DATASET_ALIASES.get(cleaned, cleaned)


def _resolve_dataset_spec(name: str) -> dict:
    canonical = _normalize_dataset_name(name)
    if canonical not in QUALIFIED_DATASETS:
        qualified = ", ".join(sorted(QUALIFIED_DATASETS))
        raise ValueError(
            f"Unsupported or unqualified dataset {name!r}; qualified datasets: {qualified}. "
            "Datasets such as MNIST, CelebA, ImageNet, and COCO are not advertised or bundled "
            "until each source, license, and checksum specification is individually qualified; "
            "prepare local images with data.builtin = 'image_folder' instead."
        )
    return deepcopy(QUALIFIED_DATASETS[canonical])


def dataset_info(name: str = "cifar10") -> dict:
    """Return source, version, expected size, checksums, and usage information for a qualified dataset."""
    spec = _resolve_dataset_spec(name)
    return {
        "name": spec["name"],
        "title": spec["title"],
        "version": spec["version"],
        "url": spec["url"],
        "archive_filename": spec["archive_filename"],
        "expected_bytes": spec["expected_bytes"],
        "archive_sha256": spec["archive_sha256"],
        "extracted_dir": spec["extracted_dir"],
        "files_sha256": deepcopy(spec["files_sha256"]),
        "factory": spec["factory"],
        "citation": spec["citation"],
        "usage_notice": spec["usage_notice"],
    }


def list_datasets() -> list[dict]:
    """List all qualified datasets available via explicit opt-in download."""
    return [dataset_info(name) for name in sorted(QUALIFIED_DATASETS)]


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _verify_extracted_dir(extracted_root: Path, files_sha256: dict[str, str]) -> list[dict]:
    if not extracted_root.exists() or not extracted_root.is_dir() or extracted_root.is_symlink():
        raise FileNotFoundError(f"Extracted dataset directory does not exist: {extracted_root}")
    verified = []
    for rel_name, expected_sha in files_sha256.items():
        file_path = extracted_root / rel_name
        if file_path.is_symlink():
            raise ValueError(f"Dataset file must not be a symlink: {file_path}")
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset file is missing: {file_path}")
        size, actual_sha = _hash_file(file_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Dataset file SHA256 mismatch for {rel_name} in {extracted_root}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        verified.append({"path": rel_name, "bytes": size, "sha256": actual_sha})
    return verified


def _build_config_snippet(factory: str, root: Path) -> str:
    escaped = str(root.resolve()).replace("\\", "/")
    return f'[data]\nfactory = "{factory}"\n\n[data.args]\nroot = "{escaped}"\n'


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _validate_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Download URL must be a non-empty http:// or https:// URL")
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Download URL must use http:// or https:// scheme, got {url!r}")
    return url.strip()


def _download_archive(
    url: str,
    archive_path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    timeout: float,
) -> tuple[int, str, bool]:
    """Download ``url`` into ``archive_path.with_suffix('.tar.gz.part')`` with HTTP Range resume."""
    validated_url = _validate_url(url)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = archive_path.with_name(archive_path.name + ".part")

    if part_path.is_symlink() or part_path.is_dir():
        raise ValueError(f"Unsafe partial download path: {part_path}")

    existing_bytes = part_path.stat().st_size if part_path.is_file() else 0
    if expected_bytes is not None and existing_bytes > expected_bytes:
        part_path.unlink(missing_ok=True)
        existing_bytes = 0

    if existing_bytes > 0 and expected_bytes is not None and existing_bytes == expected_bytes:
        size, sha = _hash_file(part_path)
        if expected_sha256 is None or sha == expected_sha256:
            os.replace(part_path, archive_path)
            return size, sha, True
        part_path.unlink(missing_ok=True)
        existing_bytes = 0

    headers = {"User-Agent": "HyperGAN/2.0 dataset-downloader"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(validated_url, headers=headers)
    resumed_partial = False
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if existing_bytes > 0 and exc.code == 416:
            part_path.unlink(missing_ok=True)
            existing_bytes = 0
            req = urllib.request.Request(validated_url, headers={"User-Agent": "HyperGAN/2.0 dataset-downloader"})
            response = urllib.request.urlopen(req, timeout=timeout)
        else:
            raise

    with response:
        status = getattr(response, "status", None) or response.getcode()
        if existing_bytes > 0 and status == 206:
            mode = "ab"
            initial_bytes = existing_bytes
            resumed_partial = True
        else:
            mode = "wb"
            initial_bytes = 0
            resumed_partial = False

        content_length_raw = response.headers.get("Content-Length") if response.headers else None
        expected_response_bytes = (
            int(content_length_raw)
            if content_length_raw is not None and content_length_raw.strip().isdigit()
            else None
        )

        written_session = 0
        with part_path.open(mode) as handle:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                written_session += len(chunk)
                total_so_far = initial_bytes + written_session
                if expected_bytes is not None and total_so_far > expected_bytes:
                    raise ValueError(
                        f"Download from {validated_url} exceeded expected size of "
                        f"{expected_bytes} bytes (received {total_so_far} bytes)"
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

    if expected_response_bytes is not None and written_session != expected_response_bytes:
        raise IOError(
            f"Truncated download from {validated_url}: expected {expected_response_bytes} bytes "
            f"in response body, got {written_session} bytes"
        )

    actual_size, actual_sha = _hash_file(part_path)
    if expected_bytes is not None and actual_size != expected_bytes:
        raise ValueError(
            f"Downloaded archive size mismatch for {archive_path.name}: "
            f"expected {expected_bytes} bytes, got {actual_size} bytes"
        )
    if expected_sha256 is not None and actual_sha != expected_sha256:
        part_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded archive SHA256 mismatch for {archive_path.name}: "
            f"expected {expected_sha256}, got {actual_sha}"
        )

    os.replace(part_path, archive_path)
    return actual_size, actual_sha, resumed_partial


def _validate_and_extract_tar(archive_path: Path, staging_dir: Path, spec: dict) -> Path:
    expected_top = spec["extracted_dir"]
    max_files = int(spec.get("max_extract_files", 32))
    max_bytes = int(spec.get("max_extract_bytes", 256 * 1024 * 1024))
    staging_root = staging_dir.resolve()

    with tarfile.open(archive_path, mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError(f"Archive {archive_path} is empty")

        file_count = 0
        declared_bytes = 0
        normalized_members: list[tuple[tarfile.TarInfo, PurePosixPath]] = []

        for member in members:
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"Unsafe tar member type in {archive_path.name}: {member.name!r} "
                    "(only ordinary files and directories are permitted)"
                )
            raw_name = member.name.replace("\\", "/")
            if not raw_name or raw_name.startswith("/"):
                raise ValueError(f"Unsafe absolute tar member path in {archive_path.name}: {member.name!r}")
            rel_posix = PurePosixPath(raw_name)
            parts = rel_posix.parts
            if not parts or ".." in parts or ":" in parts[0]:
                raise ValueError(f"Unsafe path traversal in {archive_path.name}: {member.name!r}")
            if parts[0] != expected_top:
                raise ValueError(
                    f"Unexpected top-level entry {member.name!r} in {archive_path.name}; "
                    f"expected {expected_top!r}"
                )
            target_path = (staging_root / Path(*parts)).resolve()
            if target_path != staging_root and staging_root not in target_path.parents:
                raise ValueError(f"Tar member escapes staging directory: {member.name!r}")

            if member.isfile():
                if member.size < 0:
                    raise ValueError(f"Invalid negative size for tar member {member.name!r}")
                file_count += 1
                declared_bytes += member.size
                if file_count > max_files:
                    raise ValueError(
                        f"Archive {archive_path.name} exceeds maximum file count ({max_files})"
                    )
                if declared_bytes > max_bytes:
                    raise ValueError(
                        f"Archive {archive_path.name} exceeds maximum extracted size ({max_bytes} bytes)"
                    )
            normalized_members.append((member, rel_posix))

        actual_extracted_bytes = 0
        for member, rel_posix in normalized_members:
            out_path = staging_root / Path(*rel_posix.parts)
            if member.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read file stream for tar member {member.name!r}")
            member_written = 0
            with stream, out_path.open("xb") as dst:
                for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                    member_written += len(chunk)
                    actual_extracted_bytes += len(chunk)
                    if member_written > member.size or actual_extracted_bytes > max_bytes:
                        raise ValueError(
                            f"Tar member {member.name!r} exceeded declared or maximum byte bounds"
                        )
                    dst.write(chunk)
                if member_written != member.size:
                    raise ValueError(
                        f"Tar member {member.name!r} size mismatch: header declared {member.size} bytes, "
                        f"extracted {member_written} bytes"
                    )
                dst.flush()
                os.fsync(dst.fileno())

    return staging_root / expected_top


def download_dataset(
    name: str = "cifar10",
    destination: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    url: str | None = None,
    source_archive: str | Path | None = None,
    verify_only: bool = False,
    timeout: float = 60.0,
) -> dict:
    """Acquire, verify, and extract a qualified dataset into ``destination``.

    Writes ``<destination>/dataset-manifest.json`` and returns the manifest
    dictionary with acquisition status fields.
    """
    spec = _resolve_dataset_spec(name)
    if destination is None or (isinstance(destination, str) and not destination.strip()):
        raise ValueError("An explicit destination directory is required to prepare a dataset")
    if timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")

    dest_path = Path(destination).expanduser().resolve()
    if dest_path.exists() and (not dest_path.is_dir() or dest_path.is_symlink()):
        raise ValueError(f"Destination path must be a directory (not a file or symlink): {dest_path}")

    cache_path = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else (dest_path / ".cache").resolve()
    )
    extracted_target = dest_path / spec["extracted_dir"]
    manifest_path = dest_path / "dataset-manifest.json"
    effective_url = _validate_url(url) if url is not None else spec["url"]

    # 1. Check existing extracted directory for idempotency or verify_only.
    if extracted_target.exists():
        try:
            verified_files = _verify_extracted_dir(extracted_target, spec["files_sha256"])
        except (FileNotFoundError, ValueError):
            if verify_only:
                raise
            verified_files = None
        else:
            archive_candidate = (
                Path(source_archive).expanduser().resolve()
                if source_archive is not None
                else (cache_path / spec["archive_filename"])
            )
            archive_size = spec["expected_bytes"]
            archive_sha = spec["archive_sha256"]
            reused_archive = False
            if archive_candidate.is_file():
                cand_size, cand_sha = _hash_file(archive_candidate)
                if (
                    (spec["expected_bytes"] is None or cand_size == spec["expected_bytes"])
                    and (spec["archive_sha256"] is None or cand_sha == spec["archive_sha256"])
                ):
                    archive_size = cand_size
                    archive_sha = cand_sha
                    reused_archive = True
                elif verify_only:
                    raise ValueError(
                        f"Archive verification failed for {archive_candidate}: "
                        f"expected sha256={spec['archive_sha256']} ({spec['expected_bytes']} bytes), "
                        f"got sha256={cand_sha} ({cand_size} bytes)"
                    )

            manifest = {
                "schema_version": 1,
                "dataset": spec["name"],
                "title": spec["title"],
                "version": spec["version"],
                "source_url": effective_url,
                "archive_filename": spec["archive_filename"],
                "archive_sha256": archive_sha,
                "archive_bytes": archive_size,
                "root": str(extracted_target.resolve()),
                "files": verified_files,
                "citation": spec["citation"],
                "usage_notice": spec["usage_notice"],
                "data_config": {
                    "factory": spec["factory"],
                    "args": {"root": str(extracted_target.resolve())},
                },
                "config_snippet": _build_config_snippet(spec["factory"], extracted_target),
            }
            if not manifest_path.is_file() and not verify_only:
                _write_json_atomic(manifest_path, manifest)
            return {
                **manifest,
                "manifest_path": str(manifest_path),
                "cache_dir": str(cache_path),
                "reused_extracted": True,
                "reused_archive": reused_archive,
                "resumed_partial": False,
                "verified": True,
            }

    # 2. Acquire and verify the archive (from source_archive, cache, or network).
    reused_archive = False
    resumed_partial = False
    if source_archive is not None:
        archive_path = Path(source_archive).expanduser().resolve()
        if archive_path.is_symlink() or not archive_path.is_file():
            raise FileNotFoundError(f"Source archive does not exist or is a symlink: {archive_path}")
        archive_size, archive_sha = _hash_file(archive_path)
        if spec["expected_bytes"] is not None and archive_size != spec["expected_bytes"]:
            raise ValueError(
                f"Source archive size mismatch for {archive_path}: "
                f"expected {spec['expected_bytes']} bytes, got {archive_size} bytes"
            )
        if spec["archive_sha256"] is not None and archive_sha != spec["archive_sha256"]:
            raise ValueError(
                f"Source archive SHA256 mismatch for {archive_path}: "
                f"expected {spec['archive_sha256']}, got {archive_sha}"
            )
        reused_archive = True
    else:
        archive_path = cache_path / spec["archive_filename"]
        if archive_path.is_symlink():
            raise ValueError(f"Cached archive must not be a symlink: {archive_path}")
        if archive_path.is_file():
            cand_size, cand_sha = _hash_file(archive_path)
            if (
                (spec["expected_bytes"] is None or cand_size == spec["expected_bytes"])
                and (spec["archive_sha256"] is None or cand_sha == spec["archive_sha256"])
            ):
                archive_size, archive_sha = cand_size, cand_sha
                reused_archive = True
            elif verify_only:
                raise ValueError(
                    f"Cached archive verification failed for {archive_path}: "
                    f"expected sha256={spec['archive_sha256']} ({spec['expected_bytes']} bytes), "
                    f"got sha256={cand_sha} ({cand_size} bytes)"
                )
            else:
                archive_path.unlink(missing_ok=True)

        if not reused_archive:
            if verify_only:
                raise FileNotFoundError(
                    f"Dataset {spec['name']!r} is not extracted at {extracted_target} "
                    f"and no cached archive exists at {archive_path}"
                )
            archive_size, archive_sha, resumed_partial = _download_archive(
                effective_url,
                archive_path,
                expected_bytes=spec["expected_bytes"],
                expected_sha256=spec["archive_sha256"],
                timeout=timeout,
            )

    if verify_only:
        raise FileNotFoundError(
            f"Archive {archive_path} is verified ({archive_size} bytes, sha256={archive_sha}), "
            f"but extracted dataset directory {extracted_target} is missing; "
            "run without --verify-only to extract and write dataset-manifest.json"
        )

    # 3. Validate and extract archive into a staging directory, then verify batch files.
    dest_path.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".extract-{spec['name']}-", dir=dest_path))
    backup_dir: Path | None = None
    try:
        staged_extracted = _validate_and_extract_tar(archive_path, staging_dir, spec)
        verified_files = _verify_extracted_dir(staged_extracted, spec["files_sha256"])

        if extracted_target.exists():
            backup_dir = dest_path / f".backup-{spec['extracted_dir']}-{os.getpid()}"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            os.replace(extracted_target, backup_dir)
        os.replace(staged_extracted, extracted_target)
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not extracted_target.exists():
            os.replace(backup_dir, extracted_target)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    manifest = {
        "schema_version": 1,
        "dataset": spec["name"],
        "title": spec["title"],
        "version": spec["version"],
        "source_url": effective_url,
        "archive_filename": spec["archive_filename"],
        "archive_sha256": archive_sha,
        "archive_bytes": archive_size,
        "root": str(extracted_target.resolve()),
        "files": verified_files,
        "citation": spec["citation"],
        "usage_notice": spec["usage_notice"],
        "data_config": {
            "factory": spec["factory"],
            "args": {"root": str(extracted_target.resolve())},
        },
        "config_snippet": _build_config_snippet(spec["factory"], extracted_target),
    }
    _write_json_atomic(manifest_path, manifest)

    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_path),
        "reused_extracted": False,
        "reused_archive": reused_archive,
        "resumed_partial": resumed_partial,
        "verified": True,
    }

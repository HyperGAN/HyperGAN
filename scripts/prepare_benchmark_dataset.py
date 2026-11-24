#!/usr/bin/env python3
"""Deterministic, SHA-256-verified 64x64 benchmark dataset fetcher & preparer.

Prepares clean, zero-auth-wall public visual benchmarks in HyperGAN's
``image_folder`` layout (64x64 RGB PNGs sharded 1,000 per subdirectory with an
atomic ``dataset-manifest.json`` provenance receipt). Uses only the Python
standard library and Pillow (zero extra dependencies, zero Kaggle/GDrive login).

Supported datasets:
  - ``ffhq64`` (default): Flickr-Faces-HQ 64x64 (70,000 aligned RGB faces from
    HuggingFace ``Dmini/FFHQ-64x64``, 877.7 MB direct CDN zip, supports full
    unbiased FID-50k and FD_DINOv2-50k without wrapping).
  - ``flowers102``: Oxford Flowers-102 (8,189 botanical RGB photos from Oxford
    VGG ``thor.robots.ox.ac.uk``, 344.9 MB direct ``.tgz``, center-cropped and
    Lanczos-resized to 64x64 RGB PNGs).

Examples:
    python scripts/prepare_benchmark_dataset.py --dataset ffhq64
    python scripts/prepare_benchmark_dataset.py --dataset flowers102 --output ~/.hypergan/cache/datasets/flowers102
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming IO
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
_MANIFEST_NAME = "dataset-manifest.json"
_SHARD_SIZE = 1000


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    name: str
    title: str
    url: str
    archive_filename: str
    archive_format: str  # "zip" or "tgz"
    expected_bytes: int
    archive_sha256: str
    expected_count: int
    native_resolution: int | None
    citation: str
    license_note: str


BENCHMARK_DATASETS: dict[str, BenchmarkDatasetSpec] = {
    "ffhq64": BenchmarkDatasetSpec(
        name="ffhq64",
        title="FFHQ-64 (Flickr-Faces-HQ 64x64)",
        url="https://huggingface.co/datasets/Dmini/FFHQ-64x64/resolve/main/ffhq-64x64.zip",
        archive_filename="ffhq-64x64.zip",
        archive_format="zip",
        expected_bytes=877_660_214,
        archive_sha256="76af01fc2b10b6631e08713365c7ae2a9d8ee99a7a62ac834ade684a913094ff",
        expected_count=70_000,
        native_resolution=64,
        citation=(
            "Tero Karras, Samuli Laine, and Timo Aila (CVPR 2019), "
            "'A Style-Based Generator Architecture for Generative Adversarial Networks'."
        ),
        license_note=(
            "FFHQ images are distributed under Creative Commons BY-NC-SA 4.0 by NVIDIA "
            "(individual Flickr source licenses noted in original NVLabs metadata)."
        ),
    ),
    "flowers102": BenchmarkDatasetSpec(
        name="flowers102",
        title="Oxford Flowers-102",
        url="https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz",
        archive_filename="102flowers.tgz",
        archive_format="tgz",
        expected_bytes=344_862_509,
        archive_sha256="2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246",
        expected_count=8_189,
        native_resolution=None,
        citation=(
            "Maria-Elena Nilsback and Andrew Zisserman (ICVGIP 2008), "
            "'Automated Flower Classification over a Large Number of Classes'."
        ),
        license_note=(
            "Oxford Flowers-102 is distributed by the Visual Geometry Group (University of Oxford) "
            "for research and educational use."
        ),
    ),
}


def default_cache_root() -> Path:
    env = os.environ.get("HYPERGAN_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hypergan" / "cache"


def default_output_dir(dataset: str) -> Path:
    return default_cache_root() / "datasets" / dataset


def default_archive_cache_dir() -> Path:
    return default_cache_root() / "archives"


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _is_safe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name:
        return False
    parts = PurePosixPath(name).parts
    if not parts or any(part in ("..", "") for part in parts):
        return False
    return True


def _is_image_member(name: str) -> bool:
    if not _is_safe_member_name(name) or name.endswith("/"):
        return False
    base = PurePosixPath(name).name
    if base.startswith(".") or base.startswith("._"):
        return False
    return PurePosixPath(name).suffix.lower() in _IMAGE_SUFFIXES


def download_and_verify_archive(
    spec: BenchmarkDatasetSpec,
    archive_path: Path,
    *,
    timeout: float = 60.0,
    retries: int = 3,
) -> Path:
    """Download ``spec.url`` atomically with HTTP Range resume and verify size + SHA-256."""
    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.is_file():
        size, actual_sha = sha256_file(archive_path)
        if size == spec.expected_bytes and actual_sha == spec.archive_sha256:
            print(f"[cache hit] Verified archive {archive_path} ({size:,} bytes, sha256={actual_sha[:12]}...)", file=sys.stderr)
            return archive_path
        print(
            f"[cache mismatch] Re-downloading {archive_path} "
            f"(got {size:,} bytes sha256={actual_sha[:12]}..., expected {spec.expected_bytes:,} bytes sha256={spec.archive_sha256[:12]}...)",
            file=sys.stderr,
        )
        archive_path.unlink()

    part_path = archive_path.with_name(archive_path.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            existing = part_path.stat().st_size if part_path.is_file() else 0
            if existing > spec.expected_bytes:
                part_path.unlink()
                existing = 0

            headers = {"User-Agent": "HyperGAN-Benchmark-Fetcher/1.0"}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

            req = urllib.request.Request(spec.url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status != 206:
                    # Server ignored Range; restart cleanly.
                    existing = 0
                    mode = "wb"
                written = existing
                started = time.monotonic()
                last_log = started
                with part_path.open(mode) as out:
                    while True:
                        chunk = resp.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last_log >= 5.0:
                            pct = 100.0 * written / max(1, spec.expected_bytes)
                            rate = (written - existing) / max(1e-6, now - started) / (1024 * 1024)
                            print(
                                f"  downloading {spec.archive_filename}: "
                                f"{written / 1e6:.1f}/{spec.expected_bytes / 1e6:.1f} MB ({pct:.1f}%, {rate:.1f} MB/s)",
                                file=sys.stderr,
                            )
                            last_log = now

            size, actual_sha = sha256_file(part_path)
            if size != spec.expected_bytes:
                raise ValueError(
                    f"Archive size mismatch for {spec.name}: expected {spec.expected_bytes:,} bytes, got {size:,}"
                )
            if actual_sha != spec.archive_sha256:
                raise ValueError(
                    f"Archive SHA-256 mismatch for {spec.name}: "
                    f"expected {spec.archive_sha256}, got {actual_sha}"
                )
            os.replace(part_path, archive_path)
            print(f"[verified] Downloaded {archive_path} ({size:,} bytes, sha256={actual_sha[:12]}...)", file=sys.stderr)
            return archive_path
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if isinstance(exc, ValueError) and part_path.is_file():
                # Checksum/size mismatch: drop corrupted partial file before retry.
                part_path.unlink(missing_ok=True)
            if attempt < retries:
                backoff = min(10.0, 2.0 ** attempt)
                print(f"[retry {attempt}/{retries}] Download failed ({exc}); retrying in {backoff:.0f}s...", file=sys.stderr)
                time.sleep(backoff)
    raise RuntimeError(f"Failed to download and verify {spec.name} after {retries} attempts: {last_error}") from last_error


def _process_image_bytes(
    raw_bytes: bytes,
    dest_path: Path,
    *,
    resolution: int,
    allow_fast_copy: bool,
) -> tuple[bool, tuple[int, int]]:
    """Decode, verify, center-crop + Lanczos resize (if needed), and write RGB PNG."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(raw_bytes)) as img:
        img.load()
        orig_size = img.size
        is_native_png = (
            allow_fast_copy
            and img.format == "PNG"
            and img.mode == "RGB"
            and img.size == (resolution, resolution)
            and "exif" not in img.info
        )
        if is_native_png:
            dest_path.write_bytes(raw_bytes)
            return False, orig_size

        transposed = ImageOps.exif_transpose(img)
        if transposed.mode in ("RGBA", "LA", "PA") or (
            transposed.mode == "P" and "transparency" in getattr(transposed, "info", {})
        ):
            rgba = transposed.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            rgb = transposed.convert("RGB")

        if rgb.size != (resolution, resolution):
            rgb = ImageOps.fit(
                rgb,
                (resolution, resolution),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        rgb.save(dest_path, format="PNG", compress_level=1)
        return True, orig_size


def _target_relpath(index: int) -> Path:
    shard = (index // _SHARD_SIZE) * _SHARD_SIZE
    return Path(f"{shard:05d}") / f"img{index:08d}.png"


def _extract_from_zip(
    archive_path: Path,
    stage_dir: Path,
    *,
    resolution: int,
    workers: int,
    max_images: int | None,
    allow_fast_copy: bool,
) -> tuple[int, int]:
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(name for name in zf.namelist() if _is_image_member(name))
        if max_images is not None:
            members = members[:max_images]

        resized_count = 0
        total = len(members)
        if workers <= 1:
            for idx, name in enumerate(members):
                raw = zf.read(name)
                was_resized, _ = _process_image_bytes(
                    raw,
                    stage_dir / _target_relpath(idx),
                    resolution=resolution,
                    allow_fast_copy=allow_fast_copy,
                )
                resized_count += int(was_resized)
        else:
            # Read in bounded chunks so RAM stays small even for 70k images.
            chunk_size = 512
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, total, chunk_size):
                    batch = members[start:start + chunk_size]
                    payloads = [(start + offset, zf.read(name)) for offset, name in enumerate(batch)]
                    futures = [
                        pool.submit(
                            _process_image_bytes,
                            raw,
                            stage_dir / _target_relpath(idx),
                            resolution=resolution,
                            allow_fast_copy=allow_fast_copy,
                        )
                        for idx, raw in payloads
                    ]
                    for fut in futures:
                        was_resized, _ = fut.result()
                        resized_count += int(was_resized)
        return total, resized_count


def _extract_from_tgz(
    archive_path: Path,
    stage_dir: Path,
    *,
    resolution: int,
    workers: int,
    max_images: int | None,
    allow_fast_copy: bool,
) -> tuple[int, int]:
    with tarfile.open(archive_path, "r:gz") as tf:
        lexical_members = sorted(
            (m for m in tf.getmembers() if m.isfile() and _is_image_member(m.name)),
            key=lambda m: m.name,
        )
        if max_images is not None:
            lexical_members = lexical_members[:max_images]

        rank_by_name = {m.name: idx for idx, m in enumerate(lexical_members)}
        # Iterate in forward archive byte order so gzip never rewinds on extractfile().
        forward_members = sorted(lexical_members, key=lambda m: m.offset_data)

        resized_count = 0
        total = len(forward_members)
        if workers <= 1:
            for member in forward_members:
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Failed to read tar member: {member.name}")
                raw = extracted.read()
                idx = rank_by_name[member.name]
                was_resized, _ = _process_image_bytes(
                    raw,
                    stage_dir / _target_relpath(idx),
                    resolution=resolution,
                    allow_fast_copy=allow_fast_copy,
                )
                resized_count += int(was_resized)
        else:
            chunk_size = 256
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, total, chunk_size):
                    batch = forward_members[start:start + chunk_size]
                    payloads = []
                    for member in batch:
                        extracted = tf.extractfile(member)
                        if extracted is None:
                            raise ValueError(f"Failed to read tar member: {member.name}")
                        payloads.append((rank_by_name[member.name], extracted.read()))
                    futures = [
                        pool.submit(
                            _process_image_bytes,
                            raw,
                            stage_dir / _target_relpath(idx),
                            resolution=resolution,
                            allow_fast_copy=allow_fast_copy,
                        )
                        for idx, raw in payloads
                    ]
                    for fut in futures:
                        was_resized, _ = fut.result()
                        resized_count += int(was_resized)
        return total, resized_count


def verify_prepared_dataset(
    output_dir: Path,
    *,
    dataset: str | None = None,
    resolution: int | None = None,
    expected_count: int | None = None,
    sample_check: int = 32,
) -> dict:
    """Verify an existing prepared dataset directory and its manifest receipt."""
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset is not None and manifest.get("dataset") != dataset:
        raise ValueError(f"Manifest dataset mismatch: expected {dataset!r}, got {manifest.get('dataset')!r}")
    if resolution is not None and manifest.get("resolution") != resolution:
        raise ValueError(f"Manifest resolution mismatch: expected {resolution}, got {manifest.get('resolution')}")

    pngs = sorted(output_dir.rglob("*.png"))
    recorded_count = manifest.get("image_count")
    if len(pngs) != recorded_count:
        raise ValueError(f"Image count mismatch in {output_dir}: found {len(pngs)}, manifest says {recorded_count}")
    if expected_count is not None and len(pngs) != expected_count:
        raise ValueError(f"Image count mismatch in {output_dir}: found {len(pngs)}, expected {expected_count}")

    target_res = resolution or manifest["resolution"]
    stride = max(1, len(pngs) // max(1, sample_check))
    checked = pngs[::stride][:sample_check]
    for path in checked:
        with Image.open(path) as img:
            if img.mode != "RGB" or img.size != (target_res, target_res):
                raise ValueError(
                    f"Image {path} has mode={img.mode} size={img.size}, expected RGB ({target_res}, {target_res})"
                )
    return manifest


def prepare_dataset(
    dataset: str = "ffhq64",
    output_dir: Path | str | None = None,
    *,
    cache_dir: Path | str | None = None,
    archive_path: Path | str | None = None,
    resolution: int = 64,
    workers: int | None = None,
    max_images: int | None = None,
    force: bool = False,
) -> dict:
    """Download, verify, resize (Lanczos), and atomically publish a 64x64 benchmark dataset."""
    if dataset not in BENCHMARK_DATASETS:
        choices = ", ".join(sorted(BENCHMARK_DATASETS))
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of: {choices}")
    if resolution < 8 or resolution > 1024:
        raise ValueError(f"Unsupported resolution {resolution}; expected 8..1024")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be a positive integer")

    spec = BENCHMARK_DATASETS[dataset]
    out_path = Path(output_dir).expanduser().resolve() if output_dir is not None else default_output_dir(dataset).resolve()
    expected_count = max_images if max_images is not None else spec.expected_count

    if not force and (out_path / _MANIFEST_NAME).is_file():
        try:
            existing = verify_prepared_dataset(
                out_path,
                dataset=dataset,
                resolution=resolution,
                expected_count=expected_count,
            )
            if existing.get("archive_sha256") == spec.archive_sha256 or archive_path is not None:
                print(
                    f"[ready] {out_path} already prepared "
                    f"({existing['image_count']:,} images @ {resolution}x{resolution} RGB)",
                    file=sys.stderr,
                )
                return existing
        except (OSError, ValueError) as exc:
            print(f"[rebuild] Existing directory {out_path} failed check ({exc}); rebuilding...", file=sys.stderr)

    if archive_path is not None:
        resolved_archive = Path(archive_path).expanduser().resolve()
        if not resolved_archive.is_file():
            raise FileNotFoundError(f"Provided archive does not exist: {resolved_archive}")
        archive_bytes, archive_sha = sha256_file(resolved_archive)
    else:
        cache_root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else default_archive_cache_dir().resolve()
        resolved_archive = download_and_verify_archive(spec, cache_root / spec.archive_filename)
        archive_bytes, archive_sha = spec.expected_bytes, spec.archive_sha256

    num_workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = out_path.with_name(f".{out_path.name}.staging-{os.getpid()}")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    try:
        allow_fast_copy = spec.native_resolution == resolution
        if spec.archive_format == "zip":
            total, resized_count = _extract_from_zip(
                resolved_archive,
                stage_dir,
                resolution=resolution,
                workers=num_workers,
                max_images=max_images,
                allow_fast_copy=allow_fast_copy,
            )
        elif spec.archive_format == "tgz":
            total, resized_count = _extract_from_tgz(
                resolved_archive,
                stage_dir,
                resolution=resolution,
                workers=num_workers,
                max_images=max_images,
                allow_fast_copy=allow_fast_copy,
            )
        else:
            raise ValueError(f"Unsupported archive format: {spec.archive_format!r}")

        if max_images is None and archive_path is None and total != spec.expected_count:
            raise ValueError(
                f"Extracted image count mismatch for {spec.name}: expected {spec.expected_count:,}, got {total:,}"
            )
        if total == 0:
            raise ValueError(f"No images extracted from {resolved_archive}")

        manifest = {
            "schema_version": 1,
            "dataset": spec.name,
            "title": spec.title,
            "url": spec.url,
            "archive_filename": spec.archive_filename,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "resolution": resolution,
            "mode": "RGB",
            "format": "PNG",
            "resize_policy": "center_crop",
            "resize_filter": "lanczos",
            "image_count": total,
            "resized_count": resized_count,
            "pillow_version": PILLOW_VERSION,
            "citation": spec.citation,
            "license_note": spec.license_note,
        }
        (stage_dir / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_prepared_dataset(
            stage_dir,
            dataset=spec.name,
            resolution=resolution,
            expected_count=total,
        )

        if out_path.exists():
            backup = out_path.with_name(f".{out_path.name}.old-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(out_path, backup)
            try:
                os.replace(stage_dir, out_path)
            except BaseException:
                os.replace(backup, out_path)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(stage_dir, out_path)
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    elapsed = time.monotonic() - started
    print(
        f"[done] Prepared {spec.name}: {total:,} RGB PNGs ({resolution}x{resolution}, "
        f"{resized_count:,} resized via Lanczos) in {out_path} ({elapsed:.1f}s)",
        file=sys.stderr,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, verify, and prepare deterministic 64x64 benchmark image_folder datasets.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(BENCHMARK_DATASETS),
        default="ffhq64",
        help="Dataset identifier to prepare (default: ffhq64).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination image_folder directory (default: ~/.hypergan/cache/datasets/<dataset>).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Archive cache directory (default: ~/.hypergan/cache/archives).",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional local archive path (skips network download).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Target square PNG resolution (default: 64).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker threads for parallel verification/Lanczos resizing.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on extracted images (for smoke tests only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild output directory even if a valid manifest already exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an already-prepared output directory and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output if args.output is not None else default_output_dir(args.dataset)
    if args.verify_only:
        manifest = verify_prepared_dataset(
            out_dir,
            dataset=args.dataset,
            resolution=args.resolution,
            expected_count=args.max_images or BENCHMARK_DATASETS[args.dataset].expected_count,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    manifest = prepare_dataset(
        dataset=args.dataset,
        output_dir=out_dir,
        cache_dir=args.cache_dir,
        archive_path=args.archive,
        resolution=args.resolution,
        workers=args.workers,
        max_images=args.max_images,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

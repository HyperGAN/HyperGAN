"""Deterministic, explicitly preprocessed image folders; optional imports stay lazy."""
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import stat
import time
import weakref


_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_FORMATS = ("JPEG", "PNG", "BMP", "WEBP", "TIFF")
# A file whose mtime/ctime is this close to the scan that inspected it may have
# been rewritten within one filesystem timestamp tick ("racy" entries, as in
# git's index). Such cache records are inspected again instead of trusted.
_RACY_NS = 3_000_000_000


class ImageDataError(ValueError):
    """A source read/decode failure, distinct from a loader programming error."""


class EvaluationDataExhausted(ValueError):
    """A finite, non-wrapping evaluation pass has no samples left.

    Deliberately not ``StopIteration``: raised inside a generator, PEP 479
    turns that into an unexplained ``RuntimeError``.
    """


def _digest(value):
    """SHA-256 of canonical JSON; the one identity digest for data loaders."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"image_folder {name} must be a positive integer")


def shard_slice(batch_size, rank, world_size):
    """This rank's ``slice`` of a global batch, or None when unsharded."""
    if world_size is None or world_size == 1:
        return None
    if (type(world_size) is not int or world_size < 1 or type(rank) is not int
            or not 0 <= rank < world_size or batch_size % world_size):
        raise ValueError("Sharded data needs integer 0 <= rank < world_size and a batch_size divisible by world_size")
    local = batch_size // world_size
    return slice(rank * local, (rank + 1) * local)


def default_manifest_cache():
    """Inspection cache root: $HYPERGAN_CACHE_DIR, else $XDG_CACHE_HOME/hypergan."""
    base = os.environ.get("HYPERGAN_CACHE_DIR")
    if not base:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "hypergan"
    return Path(base).expanduser() / "image-folder"


class ImageFolder:
    """Threaded image batches with caller-owned torch RNG and resumable order.

    No files are downloaded or silently skipped. Labels use top-level directory
    names, sorted lexically; flat/recursive unlabelled datasets need no class folders.
    """

    alpha: str = "error"
    follow_symlinks: bool = False

    def __init__(self, root, *, height, width, mode="RGB", resize="none",
                 interpolation="bilinear", fill=0, recursive=True, labels=False,
                 shuffle=True, alpha="error", follow_symlinks=False,
                 max_pixels=16_777_216, max_file_bytes=67_108_864,
                 workers=4, prefetch_batches=1, manifest_cache=None):
        # manifest_cache: None uses default_manifest_cache(), a path overrides
        # it, and False inspects every file on every construction.
        self._configure_workers(workers, prefetch_batches)
        if manifest_cache is not None and manifest_cache is not False and (
                not isinstance(manifest_cache, (str, Path)) or not str(manifest_cache).strip()):
            raise ValueError("image_folder manifest_cache must be a nonempty path, false, or omitted")
        for name, value in (("height", height), ("width", width), ("max_pixels", max_pixels), ("max_file_bytes", max_file_bytes)):
            _positive_int(value, name)
        if mode not in {"RGB", "L"}:
            raise ValueError("image_folder mode must be RGB or L (one-channel grayscale)")
        if resize not in {"none", "stretch", "center_crop", "pad"}:
            raise ValueError("image_folder resize must be none, stretch, center_crop, or pad")
        if interpolation not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise ValueError("image_folder interpolation must be nearest, bilinear, bicubic, or lanczos")
        if type(fill) is not int or not 0 <= fill <= 255:
            raise ValueError("image_folder fill must be an integer in [0, 255]")
        if fill != 0 and resize != "pad":
            raise ValueError("image_folder fill only applies to resize='pad'")
        for name, value in (("recursive", recursive), ("labels", labels), ("shuffle", shuffle)):
            if type(value) is not bool:
                raise ValueError(f"image_folder {name} must be boolean")
        if alpha not in {"composite_white", "composite_black", "error"}:
            raise ValueError("image_folder alpha must be composite_white, composite_black, or error")
        if type(follow_symlinks) is not bool:
            raise ValueError("image_folder follow_symlinks must be boolean")
        self.alpha = alpha
        self.follow_symlinks = follow_symlinks
        if labels and not recursive:
            raise ValueError("image_folder labels=true requires recursive=true and named class subdirectories")
        if height * width > max_pixels:
            raise ValueError("image_folder output height*width exceeds max_pixels")
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Image folder does not exist or is not a directory: {self.root}")
        try:
            from PIL import Image, ImageOps, __version__
        except ImportError as exc:
            raise ValueError("Image loading requires Pillow; install 'hypergan[train,image]' in this Python environment") from exc
        self._image, self._ops = Image, ImageOps
        self.height, self.width, self.mode = height, width, mode
        self.resize, self.interpolation, self.fill = resize, interpolation, fill
        self.labels, self.shuffle = labels, shuffle
        self.max_pixels, self.max_file_bytes = max_pixels, max_file_bytes
        preprocessing = {"version": 1, "height": height, "width": width, "mode": mode,
                         "resize": resize, "interpolation": interpolation, "fill": fill,
                         "exif_orientation": "transpose", "alpha": alpha, "frames": "single",
                         "dtype": "float32", "layout": "NCHW", "range": [-1, 1],
                         "pillow_version": __version__, "max_pixels": max_pixels,
                         "max_file_bytes": max_file_bytes}
        scan_ns = time.time_ns()
        files, ignored = [], []
        if recursive:
            if follow_symlinks:
                candidate_paths = []
                for dirpath, _, filenames in os.walk(self.root, followlinks=True):
                    for fname in filenames:
                        candidate_paths.append(Path(dirpath) / fname)
            else:
                candidate_paths = list(self.root.rglob("*"))
        else:
            candidate_paths = list(self.root.iterdir())
        for path in sorted(candidate_paths, key=lambda p: p.relative_to(self.root).as_posix()):
            relative = path.relative_to(self.root).as_posix()
            info = path.stat() if follow_symlinks else path.lstat()
            if not follow_symlinks and stat.S_ISLNK(info.st_mode):
                raise ValueError(f"Image folder contains symlink {relative}; use ordinary files/directories for a stable inventory")
            if not stat.S_ISREG(info.st_mode):
                continue
            if path.suffix.lower() not in _EXTENSIONS:
                ignored.append(relative)
                continue
            if labels and len(path.relative_to(self.root).parts) < 2:
                raise ValueError(f"Labelled image {relative} needs a named top-level class directory, e.g. cats/example.png")
            files.append((relative, [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino]))
        if not files:
            raise ValueError(f"No usable images in {self.root}: accepted=0, ignored_extensions={len(ignored)}, recursive={recursive}. Supported extensions: {', '.join(sorted(_EXTENSIONS))}; check the path and folder layout")
        self.entries = self._inventory(files, scan_ns, preprocessing, recursive, manifest_cache)
        if labels:
            for entry in self.entries:
                entry["class"] = entry["path"].split("/", 1)[0]
        self.class_map = {name: index for index, name in enumerate(sorted({entry["class"] for entry in self.entries}))} if labels else {}
        self.inventory = {"accepted": len(self.entries), "ignored_extensions": len(ignored), "ignored_paths": ignored, "supported_extensions": sorted(_EXTENSIONS)}
        # References, not copies: resume_identity() copies on demand, so a
        # million-image inventory is held once per process.
        self._identity = {
            "schema_version": 1, "kind": "image_folder", "entries": self.entries,
            "class_map": self.class_map, "labels": labels, "shuffle": shuffle,
            "recursive": recursive, "preprocessing": preprocessing,
        }
        self._identity_sha256 = _digest(self._identity)
        self._permutation, self._cursor, self._epoch = [], 0, 0

    def _inspect(self, relative):
        """Read, hash, decode and preprocess one file, so bad files fail before training."""
        entry = {"path": relative}
        content = self._read_bytes(entry)
        image, source = self._decode(content, relative)
        self._preprocess(image, relative)  # Fail incompatible sizes before a batch starts.
        entry.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), **source)
        return entry

    def _inventory(self, files, scan_ns, preprocessing, recursive, manifest_cache):
        """Inventory entries, reusing cached inspections of unchanged files.

        Cache files are keyed by root, listing mode, preprocessing and Pillow
        version. A record is trusted only when size, mtime, ctime and inode all
        match and it is not racy. Every training read still checks the source
        SHA-256, so the cache only skips the up-front decode.
        """
        cache_path = None
        if manifest_cache is not False:
            try:
                directory = default_manifest_cache() if manifest_cache is None else Path(manifest_cache).expanduser()
            except RuntimeError:  # No resolvable home directory: inspect everything.
                directory = None
            if directory is not None:
                key = _digest({"kind": "image_folder_inspection", "version": 1, "root": str(self.root),
                               "recursive": recursive, "preprocessing": preprocessing})
                cache_path = directory / f"{key}.json"
        cached = self._read_inspection_cache(cache_path)
        records, entries, pending = {}, [None] * len(files), []
        for position, (relative, signature) in enumerate(files):
            record = cached.get(relative)
            if (record is not None and record["signature"] == signature
                    and max(signature[1], signature[2]) < record["scanned_ns"] - _RACY_NS):
                entries[position] = {"path": relative, **record["entry"]}
                records[relative] = record
            else:
                pending.append(position)
        names = [files[position][0] for position in pending]
        if self.workers and len(names) > 1:
            results = []
            # Bound queued work and memory even on million-image folders.
            for start in range(0, len(names), 1024):
                results.extend(self._executor().map(self._inspect, names[start:start + 1024]))
        else:
            results = [self._inspect(name) for name in names]
        for position, entry in zip(pending, results):
            relative, signature = files[position]
            entries[position] = entry
            records[relative] = {"signature": signature, "scanned_ns": scan_ns,
                                 "entry": {k: v for k, v in entry.items() if k != "path"}}
        if cache_path is not None and (pending or records.keys() != cached.keys()):
            try:
                from .run_state import atomic_json
                atomic_json(cache_path, {"schema_version": 1, "kind": "image_folder_inspection",
                                         "root": str(self.root), "preprocessing": preprocessing,
                                         "files": records})
            except OSError as exc:
                logging.getLogger(__name__).warning("Image inspection cache not written to %s: %s", cache_path, exc)
        return entries

    @staticmethod
    def _read_inspection_cache(path):
        """Well-formed cache records by relative path; anything unreadable is ignored."""
        if path is None:
            return {}
        try:
            with path.open("rb") as stream:
                cache = json.load(stream)
        except (OSError, ValueError):
            return {}
        if not isinstance(cache, dict) or cache.get("schema_version") != 1 or not isinstance(cache.get("files"), dict):
            return {}
        valid = {}
        for relative, record in cache["files"].items():
            try:
                entry, signature = record["entry"], record["signature"]
                sha = entry["sha256"]
                if (set(record) != {"signature", "scanned_ns", "entry"} or type(record["scanned_ns"]) is not int
                        or not isinstance(signature, list) or len(signature) != 4 or any(type(v) is not int for v in signature)
                        or set(entry) != {"sha256", "bytes", "source_width", "source_height", "source_mode"}
                        or not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                        or entry["bytes"] != signature[0] or entry["source_mode"] not in {"RGB", "RGBA", "L", "LA", "P", "PA", "1", "CMYK"}
                        or any(type(entry[k]) is not int or entry[k] < 1 for k in ("source_width", "source_height"))):
                    continue
            except (KeyError, TypeError):
                continue
            valid[relative] = record
        return valid

    def _configure_workers(self, workers, prefetch_batches):
        for name, value in (("workers", workers), ("prefetch_batches", prefetch_batches)):
            if type(value) is not int or value < 0:
                raise ValueError(f"image_folder {name} must be a nonnegative integer")
        self.workers, self.prefetch_batches = workers, prefetch_batches
        self._pool, self._pool_finalizer, self._pending = None, None, {}
        self._pixel_cache = None
        self.bad_image_policy = 'error'
        self.max_bad_images, self.max_consecutive_bad_images = 100, 8
        self._excluded, self._bad_image_streak = {}, 0

    def _executor(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="image-loader")
            # Also reclaim idle threads when a standalone loader is discarded.
            self._pool_finalizer = weakref.finalize(self, self._pool.shutdown,
                                                    wait=False, cancel_futures=True)
        return self._pool

    def _discard_prefetch(self):
        pending, self._pending = self._pending, {}
        for future in pending.values():
            future.cancel()
        # Workers only read files and produce immutable pixels, never sampler/RNG
        # state. Drain running reads before recovery or closing the loader.
        wait(pending.values())

    def close(self):
        """Drain outstanding reads and release the pool; a later call reopens it."""
        self._discard_prefetch()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool_finalizer.detach()
            self._pool = self._pool_finalizer = None

    def __getstate__(self):
        # Checkpoint validation deep-copies trainers. Worker queues are disposable
        # derived state, and locks/futures must never be copied into a candidate.
        return dict(self.__dict__, _pool=None, _pool_finalizer=None, _pending={})

    def _pixels(self, entry):
        # Always verify the source, even on cache hits. A missing/changed file
        # must still fail or be excluded by the configured bad-image policy.
        content = self._read_bytes(entry)
        if self._pixel_cache is not None:
            cached = self._pixel_cache.get(entry['sha256'])
            if cached is not None:
                return cached
        image, _ = self._decode(content, entry["path"])
        pixels = self._preprocess(image, entry["path"]).tobytes()
        if self._pixel_cache is not None:
            self._pixel_cache.put(entry['sha256'], pixels)
        return pixels

    def _verified_prefetch(self, entry, future):
        pixels = future.result()
        # Preserve read-time content and path checks even if a file was replaced
        # after its pixels were prefetched. This avoids stale-cache acceptance.
        self._read_bytes(entry)
        return pixels

    def _prefetch_pixels(self, entry):
        # Cached pixels can be read speculatively without opening the source
        # twice: _verified_prefetch always checks it before accepting the image.
        if self._pixel_cache is not None:
            cached = self._pixel_cache.get(entry['sha256'])
            if cached is not None:
                return cached
        return self._pixels(entry)

    def _prefetch(self, batch_size, local=None):
        if not self.workers or not self.prefetch_batches:
            return
        # Only look ahead in the existing permutation: never draw an epoch or
        # advance the caller-owned RNG just to keep workers busy. A sharded
        # rank reads only its own slice of each upcoming global batch.
        local = slice(0, batch_size) if local is None else local
        window = []
        for start in range(self._cursor, self._cursor + batch_size * self.prefetch_batches, batch_size):
            window.extend(self._permutation[start + local.start:start + local.stop])
        indices = [i for i in window if i not in self._excluded]
        wanted = set(indices)
        stale = [self._pending.pop(index) for index in list(self._pending) if index not in wanted]
        for future in stale:
            future.cancel()
        wait(stale)
        for index in indices:
            if index not in self._pending:
                self._pending[index] = self._executor().submit(self._prefetch_pixels, self.entries[index])

    def _read_bytes(self, entry):
        path = self.root / entry["path"]
        try:
            if not getattr(self, "follow_symlinks", False):
                # Recheck containment: files/directories may be replaced after preflight.
                if not path.is_relative_to(self.root) or not path.resolve().is_relative_to(self.root):
                    raise ValueError("symlink or path escape since inventory")
                # Walk only the relative path. Rebuilding every ancestor's parents
                # for membership tests adds substantial Python work per image.
                current = path
                while current != self.root:
                    if current.is_symlink():
                        raise ValueError("symlink or path escape since inventory")
                    current = current.parent
            else:
                if not path.is_file():
                    raise ValueError("file does not exist or is not a regular file")
            with path.open("rb") as stream:
                content = stream.read(self.max_file_bytes + 1)
            if not content:
                raise ValueError("zero-byte image")
            if len(content) > self.max_file_bytes:
                raise ValueError(f"file exceeds max_file_bytes={self.max_file_bytes}")
            if "sha256" in entry and hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("content changed since inventory; rebuild the dataset and start a new run, or restore the original file")
            return content
        except (OSError, ValueError) as exc:
            raise ImageDataError(f"Cannot read image '{entry['path']}': {exc}") from exc

    def _decode(self, content, name):
        try:
            with self._image.open(BytesIO(content), formats=_FORMATS) as image:
                # Do not mutate process-global warning filters from decoder
                # threads. Enforce Pillow's warning limit explicitly as an error.
                limit = self._image.MAX_IMAGE_PIXELS
                if limit is not None and image.width * image.height > limit:
                    raise self._image.DecompressionBombWarning("decoded size exceeds Pillow limit")
                if image.width * image.height > self.max_pixels:
                    raise ValueError(f"decoded size {image.size} exceeds max_pixels={self.max_pixels}")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("animated/multi-frame images are unsupported; extract one frame explicitly")

                alpha_policy = getattr(self, "alpha", "error")
                has_alpha = image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in getattr(image, "info", {}))
                if has_alpha and alpha_policy == "error":
                    raise ValueError(f"source mode {image.mode!r} contains alpha channel; explicitly convert to RGB or L (composite alpha before loading)")

                image.load()  # Detect truncation during preflight, not deep in training.
                source = {"source_width": image.width, "source_height": image.height, "source_mode": image.mode}

                transposed = self._ops.exif_transpose(image)
                if has_alpha or transposed.mode in {"RGBA", "LA", "PA"}:
                    rgba = transposed.convert("RGBA")
                    bg_color = (255, 255, 255, 255) if alpha_policy == "composite_white" else (0, 0, 0, 255)
                    background = self._image.new("RGBA", rgba.size, bg_color)
                    result = self._image.alpha_composite(background, rgba).convert(self.mode)
                elif transposed.mode in {"RGB", "L", "1", "CMYK", "P"}:
                    result = transposed.convert(self.mode)
                else:
                    raise ValueError(f"source mode {image.mode!r} is unsupported; explicitly convert to RGB or L")
            return result, source
        except (OSError, ValueError, self._image.DecompressionBombError, self._image.DecompressionBombWarning) as exc:
            raise ImageDataError(f"Cannot decode image '{name}': {exc}. Repair/remove this file before training; no images were skipped") from exc

    def _preprocess(self, image, name):
        size = (self.width, self.height)
        method = getattr(self._image.Resampling, self.interpolation.upper())
        if self.resize == "none":
            if image.size != size:
                raise ValueError(f"Image '{name}' has width,height={image.size}, expected {size}; choose an explicit resize policy (stretch, center_crop, pad) or correct height/width")
            return image
        if self.resize == "stretch":
            return image.resize(size, method)
        if self.resize == "center_crop":
            return self._ops.fit(image, size, method=method, centering=(0.5, 0.5))
        color = self.fill if self.mode == "L" else (self.fill,) * 3
        return self._ops.pad(image, size, method=method, color=color, centering=(0.5, 0.5))

    def resume_identity(self):
        """JSON-compatible identity; relocation is allowed, data/decoding changes aren't."""
        return deepcopy(self._identity)

    def state_dict(self):
        """Sampler state only: caller must also save the supplied generator state."""
        state = {"schema_version": 1, "identity_sha256": self._identity_sha256,
                "permutation": list(self._permutation), "cursor": self._cursor, "epoch": self._epoch}
        if self.bad_image_policy == 'skip':
            state.update(schema_version=2, bad_image_streak=self._bad_image_streak,
                         excluded=[{'index': i, 'path': self.entries[i]['path'], 'reason': reason}
                                   for i, reason in sorted(self._excluded.items())])
        return state

    def load_state_dict(self, state):
        fields = {"schema_version", "identity_sha256", "permutation", "cursor", "epoch"}
        if isinstance(state, dict) and state.get('schema_version') == 2:
            fields |= {'excluded', 'bad_image_streak'}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("Invalid image_folder sampler state fields")
        if type(state["schema_version"]) is not int or state["schema_version"] not in (1, 2) or state["identity_sha256"] != self._identity_sha256:
            raise ValueError("Image folder sampler schema or data/preprocessing/class-map identity mismatch")
        order, cursor, epoch = state["permutation"], state["cursor"], state["epoch"]
        n = len(self.entries)
        if not isinstance(order, list) or any(type(i) is not int for i in order) or (order and sorted(order) != list(range(n))):
            raise ValueError("Invalid image_folder sampler permutation")
        if type(cursor) is not int or not 0 <= cursor <= len(order) or type(epoch) is not int or epoch < 0 or (not order and (cursor != 0 or epoch != 0)) or (order and epoch < 1):
            raise ValueError("Invalid image_folder sampler cursor/epoch")
        if not self.shuffle and order and order != list(range(n)):
            raise ValueError("Unshuffled image_folder state must use inventory order")
        excluded, streak = {}, 0
        if state['schema_version'] == 2:
            if self.bad_image_policy != 'skip' or not isinstance(state['excluded'], list):
                raise ValueError('Excluded images require bad_image_policy=skip')
            for item in state['excluded']:
                if not isinstance(item, dict) or set(item) != {'index', 'path', 'reason'}:
                    raise ValueError('Invalid excluded image record')
                index, reason = item['index'], item['reason']
                if (type(index) is not int or not 0 <= index < n or index in excluded
                        or item['path'] != self.entries[index]['path']
                        or not isinstance(reason, str) or not reason):
                    raise ValueError('Invalid excluded image index/path/reason')
                excluded[index] = reason
            streak = state['bad_image_streak']
            if (len(excluded) > self.max_bad_images or len(excluded) >= n
                    or type(streak) is not int or not 0 <= streak < self.max_consecutive_bad_images
                    or streak > len(excluded) or (excluded and not order)):
                raise ValueError('Invalid excluded image count or failure streak')
        self._discard_prefetch()
        self._permutation, self._cursor, self._epoch = list(order), cursor, epoch
        self._excluded, self._bad_image_streak = excluded, streak

    def _exclude_image(self, index, error):
        if index in self._excluded:
            return
        self._excluded[index] = str(error)
        self._bad_image_streak += 1
        logging.getLogger(__name__).warning('Skipping image %r (%d excluded): %s',
                                           self.entries[index]['path'], len(self._excluded), error)
        if (len(self._excluded) > self.max_bad_images
                or self._bad_image_streak >= self.max_consecutive_bad_images
                or len(self._excluded) >= len(self.entries)):
            raise ValueError('Image failure limit reached '
                             f'({len(self._excluded)} excluded, {self._bad_image_streak} consecutive; '
                             f'max_bad_images={self.max_bad_images}, '
                             f'max_consecutive_bad_images={self.max_consecutive_bad_images}); '
                             'check the dataset/storage before continuing')

    def _draw(self, count, generator):
        """Advance the sampler by ``count`` non-excluded indices, drawing epochs as needed."""
        import torch
        candidates = []
        while len(candidates) < count:
            if self._cursor == len(self._permutation):
                self._permutation = torch.randperm(len(self.entries), generator=generator).tolist() if self.shuffle else list(range(len(self.entries)))
                self._cursor = 0
                self._epoch += 1
            index = self._permutation[self._cursor]
            self._cursor += 1
            if index not in self._excluded:
                candidates.append(index)
        return candidates

    def _read_candidates(self, candidates, futures):
        """Accepted ``(index, pixels)`` in order; ``futures`` is refilled for rollback."""
        futures.clear()
        if self.workers:
            pool = self._executor()
            for index in candidates:
                entry = self.entries[index]
                prefetched = self._pending.pop(index, None)
                futures.append(pool.submit(self._pixels, entry) if prefetched is None
                               else pool.submit(self._verified_prefetch, entry, prefetched))
        accepted = []
        for position, index in enumerate(candidates):
            if index in self._excluded:
                continue
            try:
                value = futures[position].result() if self.workers else self._pixels(self.entries[index])
            except ImageDataError as exc:
                if self.bad_image_policy != 'skip':
                    raise
                self._exclude_image(index, exc)
            else:
                accepted.append((index, value))
                self._bad_image_streak = 0
        wait(futures)
        return accepted

    def __call__(self, batch_size, *, generator, rank=0, world_size=1):
        import torch
        _positive_int(batch_size, "batch_size")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("image_folder requires a caller-owned CPU torch.Generator")
        local = shard_slice(batch_size, rank, world_size)
        # Strict loading reads only this rank's slice. Skipping must make the
        # same exclusions on every rank, so each rank reads the whole global
        # batch (refilling skipped slots) and keeps its slice afterwards.
        local_only = local is not None and self.bad_image_policy != 'skip'
        # Permutations are replaced at epoch boundaries, never mutated. Keep
        # their reference for rollback instead of copying the entire inventory
        # (hundreds of thousands of indices) twice per training update.
        old_sampler = self._permutation, self._cursor, self._epoch
        old_excluded, old_streak = self._excluded.copy(), self._bad_image_streak
        old_rng = generator.get_state()
        futures = []
        try:
            if local_only:
                accepted = self._read_candidates(self._draw(batch_size, generator)[local], futures)
            else:
                accepted = []
                while len(accepted) < batch_size:
                    accepted += self._read_candidates(self._draw(batch_size - len(accepted), generator), futures)
                if local is not None:
                    accepted = accepted[local]
            # Start the next reads before CPU normalization/assembly, too.
            self._prefetch(batch_size, local if local_only else None)
            channels = 3 if self.mode == "RGB" else 1
            tensor = torch.frombuffer(bytearray(b"".join(pixels for _, pixels in accepted)), dtype=torch.uint8)
            tensor = tensor.reshape(len(accepted), self.height, self.width, channels).permute(0, 3, 1, 2)
            batch = {"real": tensor.to(torch.float32).div_(127.5).sub_(1).contiguous()}
            if self.labels:
                batch["labels"] = torch.tensor([self.class_map[self.entries[index]["class"]]
                                                for index, _ in accepted], dtype=torch.int64)
            return batch
        except BaseException:
            for future in futures:
                future.cancel()
            wait(futures)
            self._discard_prefetch()
            self._permutation, self._cursor, self._epoch = old_sampler
            self._excluded, self._bad_image_streak = old_excluded, old_streak
            generator.set_state(old_rng)
            raise

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("image_folder index must be an integer")
        if index < 0:
            index += len(self.entries)
        if not 0 <= index < len(self.entries):
            raise IndexError("image_folder index out of range")
        import torch
        pixels = self._pixels(self.entries[index])
        channels = 3 if self.mode == "RGB" else 1
        tensor = torch.frombuffer(bytearray(pixels), dtype=torch.uint8)
        tensor = tensor.reshape(self.height, self.width, channels).permute(2, 0, 1)
        real = tensor.to(torch.float32).div_(127.5).sub_(1).contiguous()
        item = {"real": real}
        if self.labels:
            item["labels"] = torch.tensor(self.class_map[self.entries[index]["class"]], dtype=torch.int64)
        return item


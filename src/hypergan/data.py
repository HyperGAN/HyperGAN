"""Deterministic, explicitly preprocessed image folders; optional imports stay lazy."""
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import weakref


_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"image_folder {name} must be a positive integer")


class ImageFolder:
    """Threaded image batches with caller-owned torch RNG and resumable order.

    No files are downloaded or silently skipped. Labels use top-level directory
    names, sorted lexically; flat/recursive unlabelled datasets need no class folders.
    """

    def __init__(self, root, *, height, width, mode="RGB", resize="none",
                 interpolation="bilinear", fill=0, recursive=True, labels=False,
                 shuffle=True, max_pixels=16_777_216, max_file_bytes=67_108_864,
                 workers=4, prefetch_batches=1):
        self._configure_workers(workers, prefetch_batches)
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
        self.entries = []
        ignored = []
        paths = self.root.rglob("*") if recursive else self.root.iterdir()
        for path in sorted(paths, key=lambda p: p.relative_to(self.root).as_posix()):
            relative = path.relative_to(self.root).as_posix()
            if path.is_symlink():
                raise ValueError(f"Image folder contains symlink {relative}; use ordinary files/directories for a stable inventory")
            if not path.is_file():
                continue
            if path.suffix.lower() not in _EXTENSIONS:
                ignored.append(relative)
                continue
            if labels and len(path.relative_to(self.root).parts) < 2:
                raise ValueError(f"Labelled image {relative} needs a named top-level class directory, e.g. cats/example.png")
            entry = {"path": relative}
            content = self._read_bytes(entry)
            image, source = self._decode(content, relative)
            self._preprocess(image, relative)  # Fail incompatible sizes before a batch starts.
            entry.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), **source)
            if labels:
                entry["class"] = path.relative_to(self.root).parts[0]
            self.entries.append(entry)
        if not self.entries:
            raise ValueError(f"No usable images in {self.root}: accepted=0, ignored_extensions={len(ignored)}, recursive={recursive}. Supported extensions: {', '.join(sorted(_EXTENSIONS))}; check the path and folder layout")
        self.class_map = {name: index for index, name in enumerate(sorted({entry["class"] for entry in self.entries}))} if labels else {}
        self.inventory = {"accepted": len(self.entries), "ignored_extensions": len(ignored), "ignored_paths": ignored, "supported_extensions": sorted(_EXTENSIONS)}
        self._identity = {
            "schema_version": 1, "kind": "image_folder", "entries": deepcopy(self.entries),
            "class_map": deepcopy(self.class_map), "labels": labels, "shuffle": shuffle,
            "recursive": recursive,
            "preprocessing": {"version": 1, "height": height, "width": width, "mode": mode,
                              "resize": resize, "interpolation": interpolation, "fill": fill,
                              "exif_orientation": "transpose", "alpha": "reject", "frames": "single",
                              "dtype": "float32", "layout": "NCHW", "range": [-1, 1],
                              "pillow_version": __version__, "max_pixels": max_pixels,
                              "max_file_bytes": max_file_bytes},
        }
        self._identity_sha256 = _digest(self._identity)
        self._permutation, self._cursor, self._epoch = [], 0, 0

    def _configure_workers(self, workers, prefetch_batches):
        for name, value in (("workers", workers), ("prefetch_batches", prefetch_batches)):
            if type(value) is not int or value < 0:
                raise ValueError(f"image_folder {name} must be a nonnegative integer")
        self.workers, self.prefetch_batches = workers, prefetch_batches
        self._pool, self._pool_finalizer, self._pending = None, None, {}

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
        image, _ = self._decode(self._read_bytes(entry), entry["path"])
        return self._preprocess(image, entry["path"]).tobytes()

    def _verified_prefetch(self, entry, future):
        pixels = future.result()
        # Preserve read-time content and path checks even if a file was replaced
        # after its pixels were prefetched. This avoids stale-cache acceptance.
        self._read_bytes(entry)
        return pixels

    def _prefetch(self, batch_size):
        if not self.workers or not self.prefetch_batches:
            return
        # Only look ahead in the existing permutation: never draw an epoch or
        # advance the caller-owned RNG just to keep workers busy.
        stop = self._cursor + batch_size * self.prefetch_batches
        indices = self._permutation[self._cursor:stop]
        wanted = set(indices)
        stale = [self._pending.pop(index) for index in list(self._pending) if index not in wanted]
        for future in stale:
            future.cancel()
        wait(stale)
        for index in indices:
            if index not in self._pending:
                self._pending[index] = self._executor().submit(self._pixels, self.entries[index])

    def _read_bytes(self, entry):
        path = self.root / entry["path"]
        try:
            # Recheck containment: files/directories may be replaced after preflight.
            if not path.resolve().is_relative_to(self.root) or any(p.is_symlink() for p in (path, *path.parents) if p != self.root and self.root in p.parents):
                raise ValueError("symlink or path escape since inventory")
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
            raise ValueError(f"Cannot read image '{entry['path']}': {exc}") from exc

    def _decode(self, content, name):
        try:
            with self._image.open(BytesIO(content)) as image:
                # Do not mutate process-global warning filters from decoder
                # threads. Enforce Pillow's warning limit explicitly as an error.
                limit = self._image.MAX_IMAGE_PIXELS
                if limit is not None and image.width * image.height > limit:
                    raise self._image.DecompressionBombWarning("decoded size exceeds Pillow limit")
                if image.width * image.height > self.max_pixels:
                    raise ValueError(f"decoded size {image.size} exceeds max_pixels={self.max_pixels}")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("animated/multi-frame images are unsupported; extract one frame explicitly")
                if image.mode not in {"RGB", "L"}:
                    raise ValueError(f"source mode {image.mode!r} is unsupported; explicitly convert to RGB or L (composite alpha before loading)")
                image.load()  # Detect truncation during preflight, not deep in training.
                source = {"source_width": image.width, "source_height": image.height, "source_mode": image.mode}
                result = self._ops.exif_transpose(image).convert(self.mode)
            return result, source
        except (OSError, ValueError, self._image.DecompressionBombError, self._image.DecompressionBombWarning) as exc:
            raise ValueError(f"Cannot decode image '{name}': {exc}. Repair/remove this file before training; no images were skipped") from exc

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
        return {"schema_version": 1, "identity_sha256": self._identity_sha256,
                "permutation": list(self._permutation), "cursor": self._cursor, "epoch": self._epoch}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"schema_version", "identity_sha256", "permutation", "cursor", "epoch"}:
            raise ValueError("Invalid image_folder sampler state fields")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1 or state["identity_sha256"] != self._identity_sha256:
            raise ValueError("Image folder sampler schema or data/preprocessing/class-map identity mismatch")
        order, cursor, epoch = state["permutation"], state["cursor"], state["epoch"]
        n = len(self.entries)
        if not isinstance(order, list) or any(type(i) is not int for i in order) or (order and sorted(order) != list(range(n))):
            raise ValueError("Invalid image_folder sampler permutation")
        if type(cursor) is not int or not 0 <= cursor <= len(order) or type(epoch) is not int or epoch < 0 or (not order and (cursor != 0 or epoch != 0)) or (order and epoch < 1):
            raise ValueError("Invalid image_folder sampler cursor/epoch")
        if not self.shuffle and order and order != list(range(n)):
            raise ValueError("Unshuffled image_folder state must use inventory order")
        self._discard_prefetch()
        self._permutation, self._cursor, self._epoch = list(order), cursor, epoch

    def __call__(self, batch_size, *, generator):
        import torch
        _positive_int(batch_size, "batch_size")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("image_folder requires a caller-owned CPU torch.Generator")
        old_state, old_rng = self.state_dict(), generator.get_state()
        indices, futures = [], []
        try:
            for _ in range(batch_size):
                if self._cursor == len(self._permutation):
                    self._permutation = torch.randperm(len(self.entries), generator=generator).tolist() if self.shuffle else list(range(len(self.entries)))
                    self._cursor = 0
                    self._epoch += 1
                indices.append(self._permutation[self._cursor])
                self._cursor += 1
            if self.workers:
                pool = self._executor()
                for index in indices:
                    entry = self.entries[index]
                    prefetched = self._pending.pop(index, None)
                    futures.append(pool.submit(self._pixels, entry) if prefetched is None
                                   else pool.submit(self._verified_prefetch, entry, prefetched))
                pixels = [future.result() for future in futures]
            else:
                pixels = [self._pixels(self.entries[index]) for index in indices]
            channels = 3 if self.mode == "RGB" else 1
            tensor = torch.frombuffer(bytearray(b"".join(pixels)), dtype=torch.uint8)
            tensor = tensor.reshape(batch_size, self.height, self.width, channels).permute(0, 3, 1, 2)
            batch = {"real": tensor.to(torch.float32).div_(127.5).sub_(1).contiguous()}
            if self.labels:
                batch["labels"] = torch.tensor([self.class_map[self.entries[index]["class"]]
                                                for index in indices], dtype=torch.int64)
            self._prefetch(batch_size)
            return batch
        except BaseException:
            for future in futures:
                future.cancel()
            wait(futures)
            self.load_state_dict(old_state)
            generator.set_state(old_rng)
            raise

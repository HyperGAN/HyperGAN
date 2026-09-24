"""Content-pinned logo pairs with an explicit, reproducible held-out split.

Preparation validates every image once; training verifies each accessed file
against that inventory. New/unlisted files never enter an existing run.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from .data import ImageDataError, ImageFolder, _EXTENSIONS, _digest, _positive_int


def _heldout(sha256):
    # Exact integer threshold: identical source bytes always stay together.
    return int(sha256, 16) * 20 < 2 ** 256


def _decoder(root, height, width):
    from PIL import Image, ImageOps
    data = object.__new__(ColorizationData)
    data.root = Path(root).expanduser().resolve()
    if not data.root.is_dir():
        raise ValueError(f'Colorization image root is not a directory: {data.root}')
    data._image, data._ops = Image, ImageOps
    data.height, data.width, data.mode = height, width, 'RGB'
    data.resize, data.interpolation, data.fill = 'pad', 'lanczos', 255
    data.max_pixels, data.max_file_bytes = 16_777_216, 67_108_864
    return data


def _write_exclusive(path, content):
    """Publish complete bytes atomically, without replacing an existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def approve_prepared_manifest(report, report_sha256, output):
    """Publish an explicitly reviewed rejection inventory without rescanning.

    The caller must inspect the report before passing its exact content hash.
    Rejected paths, source hashes and reasons remain pinned in the manifest.
    """
    raw = Path(report).read_bytes()
    if hashlib.sha256(raw).hexdigest() != report_sha256:
        raise ValueError('Preparation rejection report SHA256 mismatch')
    review = json.loads(raw)
    candidate = Path(report).parent / review['candidate_name']
    if candidate.parent.resolve() != Path(report).parent.resolve():
        raise ValueError('Unsafe preparation candidate path')
    content = candidate.read_bytes()
    if hashlib.sha256(content).hexdigest() != review['candidate_sha256']:
        raise ValueError('Preparation candidate SHA256 mismatch')
    _write_exclusive(Path(output), content)
    return {'path': str(output), 'sha256': review['candidate_sha256'],
            'train': review['train'], 'heldout': review['heldout'],
            'excluded_count': len(review['rejections'])}


def prepare_manifest(root, output, *, height=256, width=256, workers=8):
    """Validate source bytes and atomically write an inventory; never skip errors.

    Non-image extensions are listed explicitly (archives are not unpacked).
    Existing output is refused so a pinned dataset cannot be silently replaced.
    """
    from PIL import __version__
    for name, value in [('height', height), ('width', width), ('workers', workers)]:
        _positive_int(value, name)
    if height * width > 16_777_216:
        raise ValueError('Colorization output exceeds max_pixels=16777216')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'Manifest already exists: {output}; choose a new path')
    decoder = _decoder(root, height, width)
    paths, ignored = [], []
    for path in sorted(decoder.root.rglob('*')):
        name = path.relative_to(decoder.root).as_posix()
        if path.is_symlink():
            raise ValueError(f'Colorization dataset contains symlink: {name}')
        if path.is_file():
            (paths if path.suffix.lower() in _EXTENSIONS else ignored).append(name)
    if not paths:
        raise ValueError('Colorization dataset has no supported image files')

    def inspect(name):
        sha256, size = None, None
        try:
            content = decoder._read_bytes({'path': name})
            sha256, size = hashlib.sha256(content).hexdigest(), len(content)
            image, metadata = decoder._decode(content, name)
            decoder._preprocess(image, name)
            return {'path': name, 'sha256': sha256, 'bytes': size,
                    'split': 'heldout' if _heldout(sha256) else 'train', **metadata}
        except ValueError as exc:
            return {'path': name, 'sha256': sha256, 'bytes': size, 'reason': str(exc)}

    entries, rejected = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bound queued work and memory even on million-image folders.
        for start in range(0, len(paths), 1024):
            for result in pool.map(inspect, paths[start:start + 1024]):
                (rejected if 'reason' in result else entries).append(result)
    manifest = {'schema_version': 1, 'kind': 'colorization_images', 'entries': entries,
                'ignored_paths': ignored, 'excluded_images': rejected,
                'split_policy': 'source_sha256_integer * 20 < 2**256 => heldout',
                'preprocessing': {'version': 1, 'height': height, 'width': width,
                    'exif_orientation': 'transpose', 'alpha': 'composite_white',
                    'resize': 'aspect_fit_pad_white', 'interpolation': 'lanczos',
                    'frames': 'single', 'rgb_range': [-1, 1],
                    'gray': '0.299*R + 0.587*G + 0.114*B',
                    'pillow_version': __version__}}
    content = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()
    train = sum(e['split'] == 'train' for e in entries)
    heldout = sum(e['split'] == 'heldout' for e in entries)
    if rejected:
        candidate = output.with_suffix('.candidate.json')
        report = output.with_suffix('.rejections.json')
        _write_exclusive(candidate, content)
        review = {'schema_version': 1, 'root': str(decoder.root),
                  'candidate_name': candidate.name,
                  'candidate_sha256': hashlib.sha256(content).hexdigest(),
                  'train': train, 'heldout': heldout, 'rejections': rejected,
                  'ignored_paths': ignored}
        _write_exclusive(report, (json.dumps(review, sort_keys=True, indent=2) + '\n').encode())
        raise ValueError(f'{len(rejected)} invalid images; no images were skipped or manifest published. '
                         f'Inspect {report} and explicitly approve the pinned exclusion inventory')
    _write_exclusive(output, content)
    return {'path': str(output), 'sha256': hashlib.sha256(content).hexdigest(),
            'train': train, 'heldout': heldout,
            'ignored_paths': ignored}


class ColorizationData(ImageFolder):
    """Paired RGB/gray batches; inherits rollback and complete sampler recovery.

    ``shuffle=False`` is a finite sequential evaluation pass with no wrapping.
    ``shuffle=True`` traverses fresh permutations, wrapping between epochs.
    """
    def __init__(self, root, manifest, manifest_sha256, split='train', shuffle=True, repeats=1,
                 workers=4, prefetch_batches=1, bad_image_policy='error',
                 max_bad_images=100, max_consecutive_bad_images=8, cache_dir=None):
        self._configure_workers(workers, prefetch_batches)
        if cache_dir is not None and (not isinstance(cache_dir, (str, Path)) or not str(cache_dir).strip()):
            raise ValueError('cache_dir must be a nonempty path or None')
        if bad_image_policy not in {'error', 'skip'}:
            raise ValueError('bad_image_policy must be error or skip')
        _positive_int(max_bad_images, 'max_bad_images')
        _positive_int(max_consecutive_bad_images, 'max_consecutive_bad_images')
        if bad_image_policy == 'skip' and (split != 'train' or shuffle is not True):
            raise ValueError('Skipping bad images requires shuffled training; evaluation remains strict')
        self.bad_image_policy = bad_image_policy
        self.max_bad_images, self.max_consecutive_bad_images = max_bad_images, max_consecutive_bad_images
        from PIL import __version__
        if split not in {'train', 'heldout'} or type(shuffle) is not bool:
            raise ValueError('Colorization split must be train/heldout and shuffle must be boolean')
        _positive_int(repeats, 'repeats')
        if repeats > 16 or (shuffle and repeats != 1):
            raise ValueError('Colorization repeats is at most 16 and requires shuffle=false')
        raw = Path(manifest).expanduser().read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('Colorization manifest SHA256 mismatch; restore the pinned manifest')
        inventory = json.loads(raw)
        if inventory.get('schema_version') != 1 or inventory.get('kind') != 'colorization_images':
            raise ValueError('Unsupported colorization manifest version or kind')
        policy = inventory['preprocessing']
        height, width = policy['height'], policy['width']
        for name, value in [('height', height), ('width', width)]:
            _positive_int(value, name)
        expected = {'version': 1, 'height': height, 'width': width,
                    'exif_orientation': 'transpose', 'alpha': 'composite_white',
                    'resize': 'aspect_fit_pad_white', 'interpolation': 'lanczos',
                    'frames': 'single', 'rgb_range': [-1, 1],
                    'gray': '0.299*R + 0.587*G + 0.114*B', 'pillow_version': __version__}
        if policy != expected or height * width > 16_777_216:
            raise ValueError('Colorization preprocessing/Pillow version differs from the pinned manifest')
        self.__dict__.update(_decoder(root, height, width).__dict__)
        seen = set()
        for entry in inventory['entries']:
            name, sha = entry['path'], entry['sha256']
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or name != path.as_posix() or name in seen:
                raise ValueError('Colorization manifest has unsafe/duplicate paths')
            if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('Colorization manifest has invalid content hashes')
            if entry['split'] != ('heldout' if _heldout(sha) else 'train'):
                raise ValueError('Colorization manifest split disagrees with source content hash')
            seen.add(name)
        self.entries = [entry for entry in inventory['entries'] if entry['split'] == split]
        if not self.entries:
            raise ValueError(f'Colorization manifest has no {split} images')
        # Hash order makes a bounded held-out prefix independent of directory,
        # filename, collection and encoding order while remaining reproducible.
        order = 'sha256,path' if split == 'heldout' and not shuffle else 'manifest_path'
        if order == 'sha256,path':
            self.entries.sort(key=lambda entry: (entry['sha256'], entry['path']))
        unique_count = len(self.entries)
        self.entries = [entry for entry in self.entries for _ in range(repeats)]
        self.labels, self.shuffle, self.class_map = False, shuffle, {}
        self._identity = {'schema_version': 1, 'kind': 'colorization_images',
                          'manifest_sha256': manifest_sha256, 'split': split,
                          'sample_count': len(self.entries), 'shuffle': shuffle,
                          'unique_count': unique_count, 'repeats': repeats,
                          'preprocessing': deepcopy(policy)}
        if order == 'sha256,path':
            self._identity['order'] = order
        self._identity_sha256 = _digest(self._identity)
        if cache_dir is not None:
            from .image_cache import PixelCache
            namespace = _digest({'kind': 'colorization_pixels', 'version': 1,
                                 'preprocessing': policy})
            self._pixel_cache = PixelCache(cache_dir, namespace, height * width * 3)
        self._permutation, self._cursor, self._epoch = [], 0, 0

    def _decode(self, content, name):
        try:
            with self._image.open(BytesIO(content)) as image:
                # Do not mutate process-global warning filters from decoder
                # threads. Enforce Pillow's warning limit explicitly as an error.
                limit = self._image.MAX_IMAGE_PIXELS
                if limit is not None and image.width * image.height > limit:
                    raise self._image.DecompressionBombWarning("decoded size exceeds Pillow limit")
                if image.width * image.height > self.max_pixels:
                    raise ValueError(f'decoded image exceeds max_pixels={self.max_pixels}')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('animated/multi-frame images need explicit frame extraction')
                if image.mode not in {'RGB', 'RGBA', 'L', 'LA', 'P', '1'}:
                    raise ValueError(f'unsupported source mode {image.mode!r}')
                image.load()
                source = {'source_width': image.width, 'source_height': image.height, 'source_mode': image.mode}
                rgba = self._ops.exif_transpose(image).convert('RGBA')
                background = self._image.new('RGBA', rgba.size, (255, 255, 255, 255))
                result = self._image.alpha_composite(background, rgba).convert('RGB')
            return result, source
        except (OSError, ValueError, self._image.DecompressionBombError, self._image.DecompressionBombWarning) as exc:
            suffix = '; no images were skipped' if getattr(self, 'bad_image_policy', 'error') == 'error' else ''
            raise ImageDataError(f"Cannot decode colorization image '{name}': {exc}{suffix}") from exc

    def __call__(self, batch_size, *, generator):
        _positive_int(batch_size, 'batch_size')
        if not self.shuffle and self._cursor + batch_size > len(self.entries):
            raise StopIteration('Colorization evaluation exhausted; it never wraps or repeats')
        batch = super().__call__(batch_size, generator=generator)
        rgb = batch['real']
        batch['gray'] = rgb[:, 0:1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114
        return batch

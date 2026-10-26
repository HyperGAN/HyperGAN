"""Explicit Inception FID for immutable image snapshots; never downloads weights.

The first CIFAR protocol uses torch-fidelity 0.3.0, uint8 RGB, float32
features with TF32 disabled and float64 sample statistics. The evaluation data
factory and sample count determine the reference split: a small smoke score is
not interchangeable with the 50,000-image training-reference protocol.

Reference statistics depend only on the reference images and the extractor, so
they are cached under $HYPERGAN_CACHE_DIR/fid-reference, keyed by the complete
reference protocol, and reused by later evaluations of any run.
"""
import hashlib
import importlib.metadata
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import uuid

from .diversity_metrics import merge_moments


INCEPTION_SHA256 = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
# The only torch-fidelity release this protocol accepts; the fid extra pins it.
TORCH_FIDELITY_VERSION = '0.3.0'
REFERENCE_CACHE_VERSION = 1


def _checked_weights(path, expected):
    path = Path(path).expanduser().resolve()
    if expected != INCEPTION_SHA256:
        raise ValueError('InceptionFID requires the pinned torch-fidelity Inception weight SHA256')
    if not path.is_file():
        raise ValueError(f'Inception weights are missing: {path}; supply the pinned local file explicitly')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Inception weight SHA256 does not match the pinned file')
    return path


class _Moments:
    """Merge centered float64 moments without retaining all image features."""
    def __init__(self):
        self.count = 0
        self.mean = self.centered = None

    def add(self, features):
        import numpy as np
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
            raise ValueError('FID features must be a finite nonempty matrix')
        mean = value.mean(axis=0)
        if self.count and mean.shape != self.mean.shape:
            raise ValueError('FID feature dimensions changed')
        residual = value - mean
        self.count, self.mean, self.centered = merge_moments(
            self.count, self.mean, self.centered, len(value), mean, residual.T @ residual,
            lambda delta: np.outer(delta, delta))

    def statistics(self):
        if self.count < 2:
            raise ValueError('FID needs at least two generated and reference samples')
        return {'mu': self.mean, 'sigma': self.centered / (self.count - 1)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def default_reference_cache():
    from .data import default_cache_root
    return default_cache_root() / 'fid-reference'


def read_reference_statistics(path, identity, count):
    """Cached float64 reference moments, or None unless digest, identity and count all match.

    The file is the SHA-256 of an uncompressed, pickle-free .npz followed by
    the .npz itself; any mismatch or read error is a cache miss.
    """
    import numpy as np
    try:
        content = Path(path).read_bytes()
    except OSError:
        return None
    header, _, payload = content.partition(b'\n')
    if header != hashlib.sha256(payload).hexdigest().encode():
        return None
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            stored = arrays['identity'].tobytes()
            mu, sigma, stored_count = arrays['mu'], arrays['sigma'], arrays['count']
    except (OSError, ValueError, KeyError):
        return None
    if (stored != _canonical(identity) or stored_count.shape != () or int(stored_count) != count
            or mu.dtype != np.float64 or sigma.dtype != np.float64 or mu.ndim != 1
            or sigma.shape != (len(mu), len(mu)) or not np.isfinite(mu).all() or not np.isfinite(sigma).all()):
        return None
    return {'mu': mu, 'sigma': sigma}


def write_reference_statistics(path, identity, statistics, count):
    """Atomically publish reference moments; a failed write only loses the cache."""
    import numpy as np
    buffer = BytesIO()
    np.savez(buffer, identity=np.frombuffer(_canonical(identity), dtype=np.uint8),
             mu=statistics['mu'], sigma=statistics['sigma'], count=np.int64(count))
    payload = buffer.getvalue()
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open('xb') as stream:
            stream.write(hashlib.sha256(payload).hexdigest().encode() + b'\n' + payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        logging.getLogger(__name__).warning('FID reference statistics not cached at %s: %s', path, exc)


def _uint8_rgb(value):
    import torch
    if (value.ndim != 4 or value.shape[1] != 3 or not len(value)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('FID expects finite floating-point NCHW RGB images')
    return ((value.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


class InceptionFID:
    """Manual snapshot metric with paired generated/reference input bindings."""
    def __init__(self, weights_path, weights_sha256=INCEPTION_SHA256, reference_cache=None):
        # reference_cache: None uses default_reference_cache(), a path overrides
        # it, and False recomputes the reference statistics on every evaluation.
        if reference_cache is not None and reference_cache is not False and (
                not isinstance(reference_cache, (str, Path)) or not str(reference_cache).strip()):
            raise ValueError('InceptionFID reference_cache must be a nonempty path, false, or omitted')
        self.weights_path = _checked_weights(weights_path, weights_sha256)
        self.weights_sha256 = weights_sha256
        self.reference_cache = reference_cache
        try:
            version = importlib.metadata.version('torch-fidelity')
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("FID requires the 'hypergan[fid]' optional dependencies") from exc
        if version != TORCH_FIDELITY_VERSION:
            raise ValueError(f'This FID protocol requires torch-fidelity=={TORCH_FIDELITY_VERSION}, found {version}')
        self.torch_fidelity_version = version

    def _reference_cache(self, context):
        """The cache file and identity for this evaluation's reference, or (None, None)."""
        protocol = context.get('reference_identity')
        if protocol is None or self.reference_cache is False:
            return None, None
        try:
            directory = (default_reference_cache() if self.reference_cache is None
                         else Path(self.reference_cache).expanduser())
        except RuntimeError:  # No resolvable home directory: recompute.
            return None, None
        identity = {'schema_version': REFERENCE_CACHE_VERSION, 'kind': 'inception_fid_reference',
                    'weights_sha256': self.weights_sha256,
                    'torch_fidelity': self.torch_fidelity_version,
                    'extractor': 'inception-v3-compat/2048', 'protocol': protocol}
        return directory / f'{hashlib.sha256(_canonical(identity)).hexdigest()}.fidref', identity

    def describe(self):
        return {'kind': 'scalar', 'label': 'Inception FID', 'unit': 'FID', 'direction': 'minimize',
                'description': 'torch-fidelity 0.3.0 Inception-v3 compat 2048; pinned weights; RGB uint8 clamp/round; FP32 extractor without TF32; float64 unbiased covariance. Compare only matching reference splits and sample counts.'}

    def _extractor(self):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        # Recheck immediately before loading; describe/preflight is a separate process.
        _checked_weights(self.weights_path, self.weights_sha256)
        return FeatureExtractorInceptionV3('inception-v3-compat', ['2048'],
            feature_extractor_weights_path=str(self.weights_path)).float().eval().requires_grad_(False)

    def evaluate(self, *, batches, context):
        import math
        import torch
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        model = self._extractor()
        generated, reference = _Moments(), _Moments()
        cache_path, cache_identity = self._reference_cache(context)
        cached = (None if cache_path is None else
                  read_reference_statistics(cache_path, cache_identity, context['sample_count']))
        cached_rows = 0
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        current_device = None
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.inference_mode():
                for batch in batches:
                    for name, accumulator in (('generated', generated), ('reference', reference)):
                        value = _uint8_rgb(batch[name])
                        if current_device is None:
                            current_device = value.device
                            model.to(current_device)
                        if value.device != current_device:
                            raise ValueError('FID batches must remain on one device')
                        if accumulator is reference and cached is not None:
                            cached_rows += len(value)
                            continue
                        features = model(value)[0]
                        accumulator.add(features.cpu().numpy())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
        reference_count = reference.count if cached is None else cached_rows
        if generated.count != reference_count or generated.count != context['sample_count']:
            raise ValueError('FID requires the complete declared generated/reference sample counts')
        reference_statistics = reference.statistics() if cached is None else cached
        if cached is None and cache_path is not None:
            write_reference_statistics(cache_path, cache_identity, reference_statistics, reference.count)
        value = float(fid_statistics_to_metric(generated.statistics(), reference_statistics,
                      verbose=False)['frechet_inception_distance'])
        if not math.isfinite(value):
            raise ValueError('FID computation returned a nonfinite value')
        return value

"""Explicit Inception FID for immutable image snapshots; never downloads weights.

The first CIFAR protocol uses torch-fidelity 0.3.0, uint8 RGB, float32
features with TF32 disabled and float64 sample statistics. The evaluation data
factory and sample count determine the reference split: a small smoke score is
not interchangeable with the 50,000-image training-reference protocol.
"""
import hashlib
import importlib.metadata
from pathlib import Path


INCEPTION_SHA256 = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'


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
        residual = value - mean
        centered = residual.T @ residual
        if self.count == 0:
            self.mean, self.centered = mean, centered
        else:
            if mean.shape != self.mean.shape:
                raise ValueError('FID feature dimensions changed')
            delta = mean - self.mean
            total = self.count + len(value)
            self.centered += centered + np.outer(delta, delta) * (self.count * len(value) / total)
            self.mean += delta * (len(value) / total)
        self.count += len(value)

    def statistics(self):
        if self.count < 2:
            raise ValueError('FID needs at least two generated and reference samples')
        return {'mu': self.mean, 'sigma': self.centered / (self.count - 1)}


def _uint8_rgb(value):
    import torch
    if (value.ndim != 4 or value.shape[1] != 3 or not len(value)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('FID expects finite floating-point NCHW RGB images')
    return ((value.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


class InceptionFID:
    """Manual snapshot metric with paired generated/reference input bindings."""
    def __init__(self, weights_path, weights_sha256=INCEPTION_SHA256):
        self.weights_path = _checked_weights(weights_path, weights_sha256)
        self.weights_sha256 = weights_sha256
        try:
            version = importlib.metadata.version('torch-fidelity')
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("FID requires the 'hypergan[fid]' optional dependencies") from exc
        if version != '0.3.0':
            raise ValueError('This FID protocol requires torch-fidelity==0.3.0')

    def describe(self):
        return {'kind': 'scalar', 'label': 'Inception FID', 'unit': 'FID', 'direction': 'minimize',
                'description': 'torch-fidelity 0.3.0 Inception-v3 compat 2048; pinned weights; RGB uint8 clamp/round; FP32 extractor without TF32; float64 unbiased covariance. Compare only matching reference splits and sample counts.'}

    def evaluate(self, *, batches, context):
        import math
        import torch
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        # Recheck immediately before loading; describe/preflight is a separate process.
        _checked_weights(self.weights_path, self.weights_sha256)
        model = FeatureExtractorInceptionV3('inception-v3-compat', ['2048'],
            feature_extractor_weights_path=str(self.weights_path)).float().eval().requires_grad_(False)
        generated, reference = _Moments(), _Moments()
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
                        features = model(value)[0]
                        accumulator.add(features.cpu().numpy())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
        if generated.count != reference.count or generated.count != context['sample_count']:
            raise ValueError('FID requires the complete declared generated/reference sample counts')
        value = float(fid_statistics_to_metric(generated.statistics(), reference.statistics(),
                      verbose=False)['frechet_inception_distance'])
        if not math.isfinite(value):
            raise ValueError('FID computation returned a nonfinite value')
        return value

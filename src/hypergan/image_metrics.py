"""Pinned image distribution evaluators for immutable EMA snapshots.

Protocols:
- ``InceptionFID``: torch-fidelity 0.4.0 Inception-v3 compat 2048, uint8 RGB,
  float32 features with TF32 disabled, float64 unbiased covariance.
- ``DINOv2FD``: Stein et al. (NeurIPS 2023) FD_DINOv2 over Meta's pinned
  DINOv2 ViT-L/14 (LVD-142M) CLS representation (1024-d), uint8 RGB ->
  antialiased bicubic 224x224 resize -> ImageNet normalization, float32
  without TF32, float64 unbiased covariance.
- ``InceptionSuite`` / ``DINOv2Suite``: single-pass multi-scalar snapshot
  evaluators returning Fréchet Distance plus Kynkäänniemi et al. (2019)
  Precision/Recall (default ``pr_k=3``) and Naeem et al. (2020)
  Density/Coverage (default ``dc_k=5``) on a deterministic manifold subset
  (default ``manifold_samples=10000``).

Reference statistics (and suite manifold features/radii) depend only on the
reference images and extractor protocol, so they are cached under
``$HYPERGAN_CACHE_DIR/fid-reference`` and reused across runs.

Top-level imports remain Torch-free and NumPy-free.
"""
import argparse
import hashlib
import importlib.metadata
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import time
import uuid

from .cache import (
    DINOV2_VITL14_SHA256,
    DINOV2_VITL14_SIZE,
    DINOV2_VITL14_URL,
    INCEPTION_V3_SHA256,
    INCEPTION_V3_URL,
    resolve_or_cache,
)
from .diversity_metrics import merge_moments


INCEPTION_SHA256 = INCEPTION_V3_SHA256
DINOV2_SHA256 = DINOV2_VITL14_SHA256
# Pinned torch-fidelity release; fid extra pins torch-fidelity>=0.4.0,<0.5.
TORCH_FIDELITY_VERSION = '0.4.0'
REFERENCE_CACHE_VERSION = 1

DEFAULT_MANIFOLD_SAMPLES = 10000
DEFAULT_PR_K = 3
DEFAULT_DC_K = 5
DEFAULT_KNN_CHUNK_SIZE = 512

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINOV2_IMAGE_SIZE = 224
_DINOV2_PREPROCESS_TAG = 'uint8-bicubic-224-aa-imagenet-norm'
_MANIFOLD_SELECTOR_TAG = 'rng0-sorted-v1'


def _checked_weights(path=None, expected=INCEPTION_SHA256):
    if expected != INCEPTION_SHA256:
        raise ValueError('InceptionFID requires the pinned torch-fidelity Inception weight SHA256')
    target = path if path is not None else INCEPTION_V3_URL
    try:
        return resolve_or_cache(target, expected_sha256=expected)
    except FileNotFoundError as exc:
        raise ValueError(f'Inception weights are missing: {target}; supply the pinned local file explicitly') from exc
    except ValueError as exc:
        raise ValueError(f'Inception weight SHA256 does not match the pinned file: {exc}') from exc


def _checked_dinov2_weights(path=None, expected=DINOV2_SHA256):
    if expected != DINOV2_SHA256:
        raise ValueError('DINOv2 evaluation requires the pinned Meta DINOv2 ViT-L/14 weight SHA256')
    target = path if path is not None else DINOV2_VITL14_URL
    expected_size = DINOV2_VITL14_SIZE if path is None and expected == DINOV2_VITL14_SHA256 else None
    try:
        return resolve_or_cache(target, expected_sha256=expected, expected_size=expected_size)
    except FileNotFoundError as exc:
        raise ValueError(f'DINOv2 weights are missing: {target}; supply the pinned local file explicitly') from exc
    except ValueError as exc:
        raise ValueError(f'DINOv2 weight SHA256 does not match the pinned file: {exc}') from exc


def _validate_reference_cache_arg(reference_cache, owner):
    if reference_cache is not None and reference_cache is not False and (
            not isinstance(reference_cache, (str, Path)) or not str(reference_cache).strip()):
        raise ValueError(f'{owner} reference_cache must be a nonempty path, false, or omitted')
    return reference_cache


def _validate_suite_knn_args(manifold_samples, pr_k, dc_k, knn_chunk_size):
    for name, value in (
        ('manifold_samples', manifold_samples),
        ('pr_k', pr_k),
        ('dc_k', dc_k),
        ('knn_chunk_size', knn_chunk_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if manifold_samples <= max(pr_k, dc_k):
        raise ValueError(
            f'manifold_samples ({manifold_samples}) must exceed max(pr_k={pr_k}, dc_k={dc_k})'
        )


def _check_torch_fidelity_version():
    try:
        version = importlib.metadata.version('torch-fidelity')
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("FID requires the 'hypergan[fid]' optional dependencies") from exc
    if version != TORCH_FIDELITY_VERSION:
        raise ValueError(f'This FID protocol requires torch-fidelity=={TORCH_FIDELITY_VERSION}, found {version}')
    return version


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


def frechet_distance(stat_1, stat_2):
    """Fréchet distance between float64 Gaussian moments without ``scipy.linalg.sqrtm``.

    Matches torch-fidelity 0.4.0's eigenvalue formulation so modern SciPy 1.18+
    is supported without ``sqrtm(..., disp=False)`` incompatibilities.
    """
    import numpy as np
    mu1 = np.asarray(stat_1['mu'], dtype=np.float64)
    sigma1 = np.asarray(stat_1['sigma'], dtype=np.float64)
    mu2 = np.asarray(stat_2['mu'], dtype=np.float64)
    sigma2 = np.asarray(stat_2['sigma'], dtype=np.float64)
    if (mu1.ndim != 1 or mu1.shape != mu2.shape or sigma1.ndim != 2
            or sigma1.shape != sigma2.shape or sigma1.shape != (len(mu1), len(mu1))):
        raise ValueError('Fréchet distance statistics have incompatible shapes')
    if not (np.isfinite(mu1).all() and np.isfinite(mu2).all()
            and np.isfinite(sigma1).all() and np.isfinite(sigma2).all()):
        raise ValueError('Fréchet distance statistics must be finite')
    diff = mu1 - mu2
    eigvals = np.linalg.eigvals(sigma1.dot(sigma2)).astype(np.complex128)
    tr_covmean = np.sum(np.sqrt(eigvals).real)
    value = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    if not math.isfinite(value):
        raise ValueError('Fréchet distance computation returned a nonfinite value')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def default_reference_cache():
    from .data import default_cache_root
    return default_cache_root() / 'fid-reference'


def _write_digested_npz(path, arrays, label):
    import numpy as np
    buffer = BytesIO()
    np.savez(buffer, **arrays)
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
        logging.getLogger(__name__).warning('%s reference statistics not cached at %s: %s', label, path, exc)


def _read_digested_npz(path):
    try:
        content = Path(path).read_bytes()
    except OSError:
        return None
    header, sep, payload = content.partition(b'\n')
    if not sep or header != hashlib.sha256(payload).hexdigest().encode():
        return None
    return payload


def read_reference_statistics(path, identity, count):
    """Cached float64 reference moments, or None unless digest, identity and count all match.

    The file is the SHA-256 of an uncompressed, pickle-free .npz followed by
    the .npz itself; any mismatch or read error is a cache miss.
    """
    import numpy as np
    payload = _read_digested_npz(path)
    if payload is None:
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
    _write_digested_npz(
        path,
        {
            'identity': np.frombuffer(_canonical(identity), dtype=np.uint8),
            'mu': statistics['mu'],
            'sigma': statistics['sigma'],
            'count': np.int64(count),
        },
        label='FID',
    )


def read_suite_reference(path, identity, count, manifold_count):
    """Cached float64 moments plus float32 k-NN manifold features and radii, or None on miss."""
    import numpy as np
    payload = _read_digested_npz(path)
    if payload is None:
        return None
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            stored = arrays['identity'].tobytes()
            mu = arrays['mu']
            sigma = arrays['sigma']
            stored_count = arrays['count']
            stored_manifold_count = arrays['manifold_count']
            manifold_features = arrays['manifold_features']
            radii_pr = arrays['manifold_radii_pr']
            radii_dc = arrays['manifold_radii_dc']
    except (OSError, ValueError, KeyError):
        return None
    if (stored != _canonical(identity)
            or stored_count.shape != () or int(stored_count) != count
            or stored_manifold_count.shape != () or int(stored_manifold_count) != manifold_count
            or mu.dtype != np.float64 or sigma.dtype != np.float64 or mu.ndim != 1
            or sigma.shape != (len(mu), len(mu))
            or not np.isfinite(mu).all() or not np.isfinite(sigma).all()
            or manifold_features.dtype != np.float32
            or manifold_features.shape != (manifold_count, len(mu))
            or not np.isfinite(manifold_features).all()
            or radii_pr.dtype != np.float32 or radii_pr.shape != (manifold_count,)
            or radii_dc.dtype != np.float32 or radii_dc.shape != (manifold_count,)
            or not np.isfinite(radii_pr).all() or not np.isfinite(radii_dc).all()
            or (radii_pr < 0).any() or (radii_dc < 0).any()):
        return None
    return {
        'mu': mu,
        'sigma': sigma,
        'manifold_features': manifold_features,
        'manifold_radii_pr': radii_pr,
        'manifold_radii_dc': radii_dc,
    }


def write_suite_reference(path, identity, reference_data, count):
    """Atomically publish suite reference moments, manifold subset features, and k-NN radii."""
    import numpy as np
    manifold_features = np.asarray(reference_data['manifold_features'], dtype=np.float32)
    radii_pr = np.asarray(reference_data['manifold_radii_pr'], dtype=np.float32)
    radii_dc = np.asarray(reference_data['manifold_radii_dc'], dtype=np.float32)
    _write_digested_npz(
        path,
        {
            'identity': np.frombuffer(_canonical(identity), dtype=np.uint8),
            'mu': np.asarray(reference_data['mu'], dtype=np.float64),
            'sigma': np.asarray(reference_data['sigma'], dtype=np.float64),
            'count': np.int64(count),
            'manifold_count': np.int64(len(manifold_features)),
            'manifold_features': manifold_features,
            'manifold_radii_pr': radii_pr,
            'manifold_radii_dc': radii_dc,
        },
        label='Suite',
    )


def _uint8_rgb(value):
    import torch
    if (value.ndim != 4 or value.shape[1] != 3 or not len(value)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('FID expects finite floating-point NCHW RGB images')
    return ((value.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def _dinov2_preprocess(uint8_images):
    """Stein et al. (2023) DINOv2 preprocessing from quantized uint8 RGB NCHW tensors.

    Converts uint8 [0, 255] -> float32 [0, 1], resizes to 224x224 via
    antialiased bicubic interpolation (matching PIL BICUBIC) with [0, 1] clamp,
    and applies ImageNet channel normalization.
    """
    import torch
    import torch.nn.functional as F
    if uint8_images.ndim != 4 or uint8_images.shape[1] != 3 or uint8_images.dtype != torch.uint8:
        raise ValueError('DINOv2 preprocessing expects uint8 NCHW RGB images')
    x = uint8_images.to(dtype=torch.float32).div_(255.0)
    if x.shape[-2:] != (_DINOV2_IMAGE_SIZE, _DINOV2_IMAGE_SIZE):
        x = F.interpolate(
            x,
            size=(_DINOV2_IMAGE_SIZE, _DINOV2_IMAGE_SIZE),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _build_dinov2_vitl14(
    *,
    img_size=518,
    patch_size=14,
    in_chans=3,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4.0,
    interpolate_offset=0.1,
    interpolate_antialias=False,
):
    """Self-contained PyTorch DINOv2 ViT-L/14 matching Meta's LVD-142M checkpoint keys.

    Matches ``facebookresearch/dinov2`` ``dinov2_vitl14`` state_dict keys with
    ``strict=True`` and produces identical CLS embeddings (``norm(tokens)[:, 0]``)
    without ``torch.hub`` git clones or third-party model registries.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    class _LayerScale(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * self.gamma

    class _Attention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim, bias=True)

        def forward(self, x):
            bsz, n_tokens, channels = x.shape
            qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            return self.proj(out.transpose(1, 2).reshape(bsz, n_tokens, channels))

    class _Mlp(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
            self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class _Block(nn.Module):
        def __init__(self, dim, heads, ratio):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = _Attention(dim, heads)
            self.ls1 = _LayerScale(dim)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = _Mlp(dim, int(dim * ratio))
            self.ls2 = _LayerScale(dim)

        def forward(self, x):
            x = x + self.ls1(self.attn(self.norm1(x)))
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x

    class _PatchEmbed(nn.Module):
        def __init__(self, p_size, chans, dim):
            super().__init__()
            self.proj = nn.Conv2d(chans, dim, kernel_size=p_size, stride=p_size)

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    class _DINOv2ViTL14(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = patch_size
            self.interpolate_offset = interpolate_offset
            self.interpolate_antialias = interpolate_antialias
            num_patches = (img_size // patch_size) ** 2
            self.patch_embed = _PatchEmbed(patch_size, in_chans, embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
            self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
            self.blocks = nn.ModuleList([_Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        def interpolate_pos_encoding(self, x, w, h):
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            n_pos = self.pos_embed.shape[1] - 1
            if npatch == n_pos and w == h:
                return self.pos_embed
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            grid = math.isqrt(n_pos)
            if n_pos != grid * grid:
                raise ValueError('DINOv2 pos_embed patch grid must be square')
            kwargs = {}
            if self.interpolate_offset:
                kwargs['scale_factor'] = (
                    float(w0 + self.interpolate_offset) / grid,
                    float(h0 + self.interpolate_offset) / grid,
                )
            else:
                kwargs['size'] = (w0, h0)
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, grid, grid, dim).permute(0, 3, 1, 2),
                mode='bicubic',
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            if (w0, h0) != patch_pos_embed.shape[-2:]:
                raise ValueError(
                    f'DINOv2 pos_embed interpolation shape mismatch: expected {(w0, h0)}, got {patch_pos_embed.shape[-2:]}'
                )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

        def forward(self, x):
            bsz, _, w, h = x.shape
            tokens = self.patch_embed(x)
            cls = self.cls_token.expand(bsz, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            tokens = tokens + self.interpolate_pos_encoding(tokens, w, h)
            for blk in self.blocks:
                tokens = blk(tokens)
            return self.norm(tokens)[:, 0]

    return _DINOv2ViTL14()


def load_dinov2_vitl14(weights_path, expected_sha256=DINOV2_SHA256):
    """Verify SHA-256 and load Meta's DINOv2 ViT-L/14 weights into a frozen eval module."""
    import torch
    resolved = _checked_dinov2_weights(weights_path, expected_sha256)
    model = _build_dinov2_vitl14()
    state_dict = torch.load(resolved, map_location='cpu', weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f'DINOv2 weights file does not contain a state_dict mapping: {resolved}')
    model.load_state_dict(state_dict, strict=True)
    return model.float().eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# k-NN Manifold Precision / Recall (Kynkäänniemi et al. 2019) and
# Density / Coverage (Naeem et al. 2020)
# ---------------------------------------------------------------------------

def _manifold_indices(sample_count, manifold_samples):
    import numpy as np
    target = min(sample_count, manifold_samples)
    if target == sample_count:
        return np.arange(sample_count, dtype=np.int64)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(sample_count, size=target, replace=False)).astype(np.int64)


class _ManifoldSubset:
    """Stream-collect pre-selected sorted row indices as float32 features."""
    def __init__(self, indices):
        import numpy as np
        self.indices = np.asarray(indices, dtype=np.int64)
        self._cursor = 0
        self._ptr = 0
        self._chunks = []

    def add(self, features):
        import numpy as np
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or not len(arr) or not np.isfinite(arr).all():
            raise ValueError('Manifold features must be a finite nonempty 2D matrix')
        start = self._cursor
        end = start + len(arr)
        self._cursor = end
        if self._ptr >= len(self.indices):
            return
        next_ptr = self._ptr + int(np.searchsorted(self.indices[self._ptr:], end, side='left'))
        if next_ptr > self._ptr:
            local = self.indices[self._ptr:next_ptr] - start
            self._chunks.append(np.ascontiguousarray(arr[local], dtype=np.float32))
            self._ptr = next_ptr

    def array(self):
        import numpy as np
        if self._ptr != len(self.indices):
            raise ValueError('Manifold subset did not receive all declared samples')
        if not self._chunks:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(self._chunks, axis=0)


def knn_radii(features, ks, *, chunk_size=DEFAULT_KNN_CHUNK_SIZE, device=None):
    """Exact chunked L2 k-th nearest-neighbor radii excluding self-distance (i == j).

    Computes all requested ``ks`` in a single chunked pass over ``features`` in
    O(chunk_size * N) memory. Centers features before squared-norm expansion to
    avoid catastrophic cancellation while staying fast on CPU and CUDA.
    """
    import numpy as np
    import torch

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('knn_chunk_size must be a positive integer')
    unique_ks = sorted(set(ks))
    if not unique_ks or any(type(k) is not int or k < 1 for k in unique_ks):
        raise ValueError('k-NN ks must be positive integers')

    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError('k-NN features must be a finite nonempty 2D matrix')
    n_samples = len(arr)
    k_max = unique_ks[-1]
    if n_samples <= k_max:
        raise ValueError(f'k-NN manifold requires at least {k_max + 1} samples, got {n_samples}')

    dev = torch.device('cpu') if device is None else torch.device(device)
    with torch.inference_mode():
        x = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        xc = x - x.mean(dim=0, keepdim=True)
        x2 = (xc * xc).sum(dim=1)
        out = {k: torch.empty(n_samples, dtype=torch.float32, device=dev) for k in unique_ks}
        for i in range(0, n_samples, chunk_size):
            j = min(i + chunk_size, n_samples)
            d2 = x2[i:j].unsqueeze(1) + x2.unsqueeze(0) - 2.0 * (xc[i:j] @ xc.T)
            d2.clamp_min_(0.0)
            row_idx = torch.arange(j - i, device=dev)
            d2[row_idx, row_idx + i] = float('inf')
            topk_d2, _ = torch.topk(d2, k=k_max, dim=1, largest=False, sorted=True)
            for k in unique_ks:
                out[k][i:j] = topk_d2[:, k - 1].sqrt()
        return {k: out[k].cpu().numpy() for k in unique_ks}


def compute_prdc(
    real_features,
    fake_features,
    *,
    pr_k=DEFAULT_PR_K,
    dc_k=DEFAULT_DC_K,
    real_radii_pr=None,
    real_radii_dc=None,
    chunk_size=DEFAULT_KNN_CHUNK_SIZE,
    device=None,
):
    """Compute Kynkäänniemi et al. (2019) Precision/Recall and Naeem et al. (2020) Density/Coverage.

    Uses double-chunked exact L2 distances so peak memory stays O(chunk_size^2)
    even for large manifolds. Precomputed ``real_radii_pr`` and ``real_radii_dc``
    skip the O(N_r^2) reference k-NN pass on cache hits.
    """
    import numpy as np
    import torch

    for name, value in (('pr_k', pr_k), ('dc_k', dc_k), ('chunk_size', chunk_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    real_arr = np.asarray(real_features, dtype=np.float32)
    fake_arr = np.asarray(fake_features, dtype=np.float32)
    if (real_arr.ndim != 2 or fake_arr.ndim != 2 or not len(real_arr) or not len(fake_arr)
            or real_arr.shape[1] != fake_arr.shape[1]):
        raise ValueError('PRDC requires nonempty 2D real and fake feature matrices with matching dimensions')
    if not (np.isfinite(real_arr).all() and np.isfinite(fake_arr).all()):
        raise ValueError('PRDC features must be finite')

    n_real, n_fake = len(real_arr), len(fake_arr)
    if n_real <= max(pr_k, dc_k):
        raise ValueError(f'PRDC requires at least {max(pr_k, dc_k) + 1} reference samples, got {n_real}')
    if n_fake <= pr_k:
        raise ValueError(f'PRDC requires at least {pr_k + 1} generated samples, got {n_fake}')

    needed_real_ks = []
    if real_radii_pr is None:
        needed_real_ks.append(pr_k)
    if real_radii_dc is None:
        needed_real_ks.append(dc_k)
    if needed_real_ks:
        computed = knn_radii(real_arr, needed_real_ks, chunk_size=chunk_size, device=device)
        if real_radii_pr is None:
            real_radii_pr = computed[pr_k]
        if real_radii_dc is None:
            real_radii_dc = computed[dc_k]

    real_radii_pr = np.asarray(real_radii_pr, dtype=np.float32)
    real_radii_dc = np.asarray(real_radii_dc, dtype=np.float32)
    if (real_radii_pr.shape != (n_real,) or real_radii_dc.shape != (n_real,)
            or not np.isfinite(real_radii_pr).all() or not np.isfinite(real_radii_dc).all()):
        raise ValueError('Precomputed reference radii must be finite 1D arrays matching real_features')

    fake_radii_pr = knn_radii(fake_arr, (pr_k,), chunk_size=chunk_size, device=device)[pr_k]

    dev = torch.device('cpu') if device is None else torch.device(device)
    with torch.inference_mode():
        r = torch.as_tensor(real_arr, dtype=torch.float32, device=dev)
        g = torch.as_tensor(fake_arr, dtype=torch.float32, device=dev)
        origin = r.mean(dim=0, keepdim=True)
        rc = r - origin
        gc = g - origin
        r2 = (rc * rc).sum(dim=1)
        g2 = (gc * gc).sum(dim=1)

        r_rad_pr = torch.as_tensor(real_radii_pr, dtype=torch.float32, device=dev)
        r_rad_dc = torch.as_tensor(real_radii_dc, dtype=torch.float32, device=dev)
        g_rad_pr = torch.as_tensor(fake_radii_pr, dtype=torch.float32, device=dev)

        precision_hits = torch.zeros(n_fake, dtype=torch.bool, device=dev)
        recall_hits = torch.zeros(n_real, dtype=torch.bool, device=dev)
        density_counts = torch.zeros(n_fake, dtype=torch.int64, device=dev)
        coverage_hits = torch.zeros(n_real, dtype=torch.bool, device=dev)

        for i in range(0, n_real, chunk_size):
            j = min(i + chunk_size, n_real)
            r_pr_blk = r_rad_pr[i:j].unsqueeze(1)
            r_dc_blk = r_rad_dc[i:j].unsqueeze(1)
            r2_blk = r2[i:j].unsqueeze(1)
            rc_blk = rc[i:j]
            for p in range(0, n_fake, chunk_size):
                q = min(p + chunk_size, n_fake)
                d = (r2_blk + g2[p:q].unsqueeze(0) - 2.0 * (rc_blk @ gc[p:q].T)).clamp_min_(0.0).sqrt_()
                precision_hits[p:q] |= (d < r_pr_blk).any(dim=0)
                recall_hits[i:j] |= (d < g_rad_pr[p:q].unsqueeze(0)).any(dim=1)
                in_dc = d < r_dc_blk
                density_counts[p:q] += in_dc.sum(dim=0)
                coverage_hits[i:j] |= in_dc.any(dim=1)

        scores = {
            'precision': float(precision_hits.double().mean().item()),
            'recall': float(recall_hits.double().mean().item()),
            'density': float(density_counts.double().mean().item() / float(dc_k)),
            'coverage': float(coverage_hits.double().mean().item()),
        }
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('PRDC computation returned a nonfinite value')
    return scores


# ---------------------------------------------------------------------------
# Snapshot Evaluators
# ---------------------------------------------------------------------------

def _resolve_cache_dir(reference_cache):
    if reference_cache is False:
        return None
    try:
        return (default_reference_cache() if reference_cache is None
                else Path(reference_cache).expanduser())
    except RuntimeError:  # No resolvable home directory: recompute.
        return None


class InceptionFID:
    """Manual snapshot metric with paired generated/reference input bindings."""
    def __init__(self, weights_path=None, weights_sha256=INCEPTION_SHA256, reference_cache=None):
        # reference_cache: None uses default_reference_cache(), a path overrides
        # it, and False recomputes the reference statistics on every evaluation.
        _validate_reference_cache_arg(reference_cache, 'InceptionFID')
        self.weights_path = _checked_weights(weights_path, weights_sha256)
        self.weights_sha256 = weights_sha256
        self.reference_cache = reference_cache
        self.torch_fidelity_version = _check_torch_fidelity_version()

    def _reference_cache(self, context):
        """The cache file and identity for this evaluation's reference, or (None, None)."""
        protocol = context.get('reference_identity')
        if protocol is None or self.reference_cache is False:
            return None, None
        directory = _resolve_cache_dir(self.reference_cache)
        if directory is None:
            return None, None
        identity = {'schema_version': REFERENCE_CACHE_VERSION, 'kind': 'inception_fid_reference',
                    'weights_sha256': self.weights_sha256,
                    'torch_fidelity': self.torch_fidelity_version,
                    'extractor': 'inception-v3-compat/2048', 'protocol': protocol}
        return directory / f'{hashlib.sha256(_canonical(identity)).hexdigest()}.fidref', identity

    def describe(self):
        return {'kind': 'scalar', 'label': 'Inception FID', 'unit': 'FID', 'direction': 'minimize',
                'description': 'torch-fidelity 0.4.0 Inception-v3 compat 2048; pinned weights; RGB uint8 clamp/round; FP32 extractor without TF32; float64 unbiased covariance. Compare only matching reference splits and sample counts.'}

    def _extractor(self):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        # Recheck immediately before loading; describe/preflight is a separate process.
        _checked_weights(self.weights_path, self.weights_sha256)
        return FeatureExtractorInceptionV3('inception-v3-compat', ['2048'],
            feature_extractor_weights_path=str(self.weights_path)).float().eval().requires_grad_(False)

    def evaluate(self, *, batches, context):
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


class DINOv2FD:
    """Fréchet Distance in Meta's pinned DINOv2 ViT-L/14 (Stein et al. NeurIPS 2023)."""
    def __init__(self, weights_path=None, weights_sha256=DINOV2_SHA256, reference_cache=None):
        _validate_reference_cache_arg(reference_cache, 'DINOv2FD')
        self.weights_path = _checked_dinov2_weights(weights_path, weights_sha256)
        self.weights_sha256 = weights_sha256
        self.reference_cache = reference_cache

    def _reference_cache(self, context):
        protocol = context.get('reference_identity')
        if protocol is None or self.reference_cache is False:
            return None, None
        directory = _resolve_cache_dir(self.reference_cache)
        if directory is None:
            return None, None
        identity = {
            'schema_version': REFERENCE_CACHE_VERSION,
            'kind': 'dinov2_fd_reference',
            'weights_sha256': self.weights_sha256,
            'extractor': 'dinov2-vitl14/cls-1024',
            'preprocess': _DINOV2_PREPROCESS_TAG,
            'protocol': protocol,
        }
        return directory / f'{hashlib.sha256(_canonical(identity)).hexdigest()}.fidref', identity

    def describe(self):
        return {
            'kind': 'scalar',
            'label': 'DINOv2 FD',
            'unit': 'FD',
            'direction': 'minimize',
            'description': (
                'DINOv2 ViT-L/14 Fréchet Distance (Stein et al. 2023 FD_DINOv2); '
                'pinned LVD-142M weights; RGB uint8 clamp/round; bicubic 224x224 resize; '
                'ImageNet normalization; FP32 extractor without TF32; float64 unbiased covariance. '
                'Compare only matching reference splits and sample counts.'
            ),
        }

    def _extractor(self):
        return load_dinov2_vitl14(self.weights_path, self.weights_sha256)

    def evaluate(self, *, batches, context):
        import torch
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
                        uint8 = _uint8_rgb(batch[name])
                        if current_device is None:
                            current_device = uint8.device
                            model.to(current_device)
                        if uint8.device != current_device:
                            raise ValueError('DINOv2FD batches must remain on one device')
                        if accumulator is reference and cached is not None:
                            cached_rows += len(uint8)
                            continue
                        features = model(_dinov2_preprocess(uint8))
                        accumulator.add(features.cpu().numpy())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
        reference_count = reference.count if cached is None else cached_rows
        if generated.count != reference_count or generated.count != context['sample_count']:
            raise ValueError('DINOv2FD requires the complete declared generated/reference sample counts')
        reference_statistics = reference.statistics() if cached is None else cached
        if cached is None and cache_path is not None:
            write_reference_statistics(cache_path, cache_identity, reference_statistics, reference.count)
        return frechet_distance(generated.statistics(), reference_statistics)


class _FeatureSuiteBase:
    """Shared single-pass Fréchet Distance + PRDC evaluator with cached reference manifolds."""
    _fd_key = None
    _owner_name = None

    def __init__(
        self,
        *,
        reference_cache=None,
        manifold_samples=DEFAULT_MANIFOLD_SAMPLES,
        pr_k=DEFAULT_PR_K,
        dc_k=DEFAULT_DC_K,
        knn_chunk_size=DEFAULT_KNN_CHUNK_SIZE,
    ):
        _validate_reference_cache_arg(reference_cache, self._owner_name)
        _validate_suite_knn_args(manifold_samples, pr_k, dc_k, knn_chunk_size)
        self.reference_cache = reference_cache
        self.manifold_samples = manifold_samples
        self.pr_k = pr_k
        self.dc_k = dc_k
        self.knn_chunk_size = knn_chunk_size

    def _extractor(self):
        raise NotImplementedError

    def _extract_features(self, model, uint8_images):
        raise NotImplementedError

    def _compute_fd(self, generated_stats, reference_stats):
        return frechet_distance(generated_stats, reference_stats)

    def _suite_cache_identity(self, protocol):
        raise NotImplementedError

    def _reference_cache(self, context):
        protocol = context.get('reference_identity')
        if protocol is None or self.reference_cache is False:
            return None, None
        directory = _resolve_cache_dir(self.reference_cache)
        if directory is None:
            return None, None
        identity = self._suite_cache_identity(protocol)
        return directory / f'{hashlib.sha256(_canonical(identity)).hexdigest()}.suiteref', identity

    def evaluate(self, *, batches, context):
        import torch
        sample_count = context['sample_count']
        if type(sample_count) is not int or sample_count < 2:
            raise ValueError(f'{self._owner_name} requires sample_count >= 2')
        manifold_count = min(sample_count, self.manifold_samples)
        min_manifold = max(self.pr_k, self.dc_k) + 1
        if manifold_count < min_manifold:
            raise ValueError(
                f'{self._owner_name} requires at least {min_manifold} samples for '
                f'pr_k={self.pr_k}, dc_k={self.dc_k}; got sample_count={sample_count}'
            )

        indices = _manifold_indices(sample_count, self.manifold_samples)
        model = self._extractor()
        generated, reference = _Moments(), _Moments()
        gen_manifold = _ManifoldSubset(indices)
        ref_manifold = _ManifoldSubset(indices)

        cache_path, cache_identity = self._reference_cache(context)
        cached = (None if cache_path is None else
                  read_suite_reference(cache_path, cache_identity, sample_count, manifold_count))
        cached_rows = 0
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        current_device = None
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.inference_mode():
                for batch in batches:
                    for name, accumulator, subset in (
                        ('generated', generated, gen_manifold),
                        ('reference', reference, ref_manifold),
                    ):
                        uint8 = _uint8_rgb(batch[name])
                        if current_device is None:
                            current_device = uint8.device
                            model.to(current_device)
                        if uint8.device != current_device:
                            raise ValueError(f'{self._owner_name} batches must remain on one device')
                        if accumulator is reference and cached is not None:
                            cached_rows += len(uint8)
                            continue
                        features = self._extract_features(model, uint8)
                        features_np = features.cpu().numpy()
                        accumulator.add(features_np)
                        subset.add(features_np)

            reference_count = reference.count if cached is None else cached_rows
            if generated.count != reference_count or generated.count != sample_count:
                raise ValueError(
                    f'{self._owner_name} requires the complete declared generated/reference sample counts'
                )

            gen_stats = generated.statistics()
            gen_manifold_arr = gen_manifold.array()

            if cached is None:
                ref_stats = reference.statistics()
                ref_manifold_arr = ref_manifold.array()
                ref_radii = knn_radii(
                    ref_manifold_arr,
                    (self.pr_k, self.dc_k),
                    chunk_size=self.knn_chunk_size,
                    device=current_device,
                )
                ref_radii_pr = ref_radii[self.pr_k]
                ref_radii_dc = ref_radii[self.dc_k]
                if cache_path is not None:
                    write_suite_reference(
                        cache_path,
                        cache_identity,
                        {
                            'mu': ref_stats['mu'],
                            'sigma': ref_stats['sigma'],
                            'manifold_features': ref_manifold_arr,
                            'manifold_radii_pr': ref_radii_pr,
                            'manifold_radii_dc': ref_radii_dc,
                        },
                        reference.count,
                    )
            else:
                ref_stats = {'mu': cached['mu'], 'sigma': cached['sigma']}
                ref_manifold_arr = cached['manifold_features']
                ref_radii_pr = cached['manifold_radii_pr']
                ref_radii_dc = cached['manifold_radii_dc']

            fd_value = self._compute_fd(gen_stats, ref_stats)
            prdc = compute_prdc(
                ref_manifold_arr,
                gen_manifold_arr,
                pr_k=self.pr_k,
                dc_k=self.dc_k,
                real_radii_pr=ref_radii_pr,
                real_radii_dc=ref_radii_dc,
                chunk_size=self.knn_chunk_size,
                device=current_device,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn

        result = {
            self._fd_key: fd_value,
            'precision': prdc['precision'],
            'recall': prdc['recall'],
            'density': prdc['density'],
            'coverage': prdc['coverage'],
        }
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError(f'{self._owner_name} computation returned a nonfinite value')
        return result


class InceptionSuite(_FeatureSuiteBase):
    """Single-pass Inception-v3 evaluator: FID + Precision/Recall (k=3) + Density/Coverage (k=5)."""
    _fd_key = 'fid'
    _owner_name = 'InceptionSuite'

    def __init__(
        self,
        weights_path=None,
        weights_sha256=INCEPTION_SHA256,
        reference_cache=None,
        manifold_samples=DEFAULT_MANIFOLD_SAMPLES,
        pr_k=DEFAULT_PR_K,
        dc_k=DEFAULT_DC_K,
        knn_chunk_size=DEFAULT_KNN_CHUNK_SIZE,
    ):
        super().__init__(
            reference_cache=reference_cache,
            manifold_samples=manifold_samples,
            pr_k=pr_k,
            dc_k=dc_k,
            knn_chunk_size=knn_chunk_size,
        )
        self.weights_path = _checked_weights(weights_path, weights_sha256)
        self.weights_sha256 = weights_sha256
        self.torch_fidelity_version = _check_torch_fidelity_version()

    def _suite_cache_identity(self, protocol):
        return {
            'schema_version': REFERENCE_CACHE_VERSION,
            'kind': 'inception_suite_reference',
            'weights_sha256': self.weights_sha256,
            'torch_fidelity': self.torch_fidelity_version,
            'extractor': 'inception-v3-compat/2048',
            'manifold_samples': self.manifold_samples,
            'manifold_selector': _MANIFOLD_SELECTOR_TAG,
            'pr_k': self.pr_k,
            'dc_k': self.dc_k,
            'protocol': protocol,
        }

    def describe(self):
        manifold = f'up to {self.manifold_samples} deterministic subset samples'
        return {
            'kind': 'scalars',
            'outputs': {
                'fid': {
                    'label': 'Inception FID',
                    'unit': 'FID',
                    'direction': 'minimize',
                    'description': (
                        'torch-fidelity 0.4.0 Inception-v3 compat 2048 FID; pinned weights; '
                        'RGB uint8 clamp/round; FP32 without TF32; float64 unbiased covariance.'
                    ),
                },
                'precision': {
                    'label': 'Inception Precision',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Kynkäänniemi et al. (2019) k-NN Precision (k={self.pr_k}) on Inception-v3 '
                        f'2048 features over {manifold}.'
                    ),
                },
                'recall': {
                    'label': 'Inception Recall',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Kynkäänniemi et al. (2019) k-NN Recall (k={self.pr_k}) on Inception-v3 '
                        f'2048 features over {manifold}.'
                    ),
                },
                'density': {
                    'label': 'Inception Density',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Naeem et al. (2020) k-NN Density (k={self.dc_k}) on Inception-v3 '
                        f'2048 features over {manifold}.'
                    ),
                },
                'coverage': {
                    'label': 'Inception Coverage',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Naeem et al. (2020) k-NN Coverage (k={self.dc_k}) on Inception-v3 '
                        f'2048 features over {manifold}.'
                    ),
                },
            },
        }

    def _extractor(self):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        _checked_weights(self.weights_path, self.weights_sha256)
        return FeatureExtractorInceptionV3(
            'inception-v3-compat',
            ['2048'],
            feature_extractor_weights_path=str(self.weights_path),
        ).float().eval().requires_grad_(False)

    def _extract_features(self, model, uint8_images):
        return model(uint8_images)[0]

    def _compute_fd(self, generated_stats, reference_stats):
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        value = float(
            fid_statistics_to_metric(generated_stats, reference_stats, verbose=False)[
                'frechet_inception_distance'
            ]
        )
        if not math.isfinite(value):
            raise ValueError('FID computation returned a nonfinite value')
        return value


class DINOv2Suite(_FeatureSuiteBase):
    """Single-pass DINOv2 ViT-L/14 evaluator: FD_DINOv2 + Precision/Recall (k=3) + Density/Coverage (k=5)."""
    _fd_key = 'fd_dinov2'
    _owner_name = 'DINOv2Suite'

    def __init__(
        self,
        weights_path=None,
        weights_sha256=DINOV2_SHA256,
        reference_cache=None,
        manifold_samples=DEFAULT_MANIFOLD_SAMPLES,
        pr_k=DEFAULT_PR_K,
        dc_k=DEFAULT_DC_K,
        knn_chunk_size=DEFAULT_KNN_CHUNK_SIZE,
    ):
        super().__init__(
            reference_cache=reference_cache,
            manifold_samples=manifold_samples,
            pr_k=pr_k,
            dc_k=dc_k,
            knn_chunk_size=knn_chunk_size,
        )
        self.weights_path = _checked_dinov2_weights(weights_path, weights_sha256)
        self.weights_sha256 = weights_sha256

    def _suite_cache_identity(self, protocol):
        return {
            'schema_version': REFERENCE_CACHE_VERSION,
            'kind': 'dinov2_suite_reference',
            'weights_sha256': self.weights_sha256,
            'extractor': 'dinov2-vitl14/cls-1024',
            'preprocess': _DINOV2_PREPROCESS_TAG,
            'manifold_samples': self.manifold_samples,
            'manifold_selector': _MANIFOLD_SELECTOR_TAG,
            'pr_k': self.pr_k,
            'dc_k': self.dc_k,
            'protocol': protocol,
        }

    def describe(self):
        manifold = f'up to {self.manifold_samples} deterministic subset samples'
        return {
            'kind': 'scalars',
            'outputs': {
                'fd_dinov2': {
                    'label': 'DINOv2 FD',
                    'unit': 'FD',
                    'direction': 'minimize',
                    'description': (
                        'DINOv2 ViT-L/14 Fréchet Distance (Stein et al. 2023 FD_DINOv2); '
                        'pinned LVD-142M weights; RGB uint8 clamp/round; bicubic 224x224 resize; '
                        'ImageNet normalization; FP32 without TF32; float64 unbiased covariance.'
                    ),
                },
                'precision': {
                    'label': 'DINOv2 Precision',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Kynkäänniemi et al. (2019) k-NN Precision (k={self.pr_k}) on DINOv2 ViT-L/14 '
                        f'1024-d CLS features over {manifold}.'
                    ),
                },
                'recall': {
                    'label': 'DINOv2 Recall',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Kynkäänniemi et al. (2019) k-NN Recall (k={self.pr_k}) on DINOv2 ViT-L/14 '
                        f'1024-d CLS features over {manifold}.'
                    ),
                },
                'density': {
                    'label': 'DINOv2 Density',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Naeem et al. (2020) k-NN Density (k={self.dc_k}) on DINOv2 ViT-L/14 '
                        f'1024-d CLS features over {manifold}.'
                    ),
                },
                'coverage': {
                    'label': 'DINOv2 Coverage',
                    'unit': 'ratio',
                    'direction': 'maximize',
                    'description': (
                        f'Naeem et al. (2020) k-NN Coverage (k={self.dc_k}) on DINOv2 ViT-L/14 '
                        f'1024-d CLS features over {manifold}.'
                    ),
                },
            },
        }

    def _extractor(self):
        return load_dinov2_vitl14(self.weights_path, self.weights_sha256)

    def _extract_features(self, model, uint8_images):
        return model(_dinov2_preprocess(uint8_images))


# ---------------------------------------------------------------------------
# Standalone EMA Bundle Evaluation Helper & CLI
# ---------------------------------------------------------------------------

def _normalize_eval_data_spec(data, bundle_config):
    """Normalize a standalone data argument into a deterministic engine data spec."""
    from copy import deepcopy
    if callable(data):
        return None
    if data == 'train':
        base = deepcopy(bundle_config['data'])
        if base.get('factory') == 'image_folder':
            base.setdefault('args', {})['shuffle'] = False
        elif base.get('factory') == 'hypergan.image_data:CIFAR10Data':
            base.setdefault('args', {})['sampling'] = 'sequential'
            base['args']['horizontal_flip'] = False
        return base
    if not isinstance(data, dict):
        raise ValueError("data must be 'train', a data spec dict, or a callable dataset")
    if set(data) == {'factory', 'args'}:
        return deepcopy(data)
    builtin = data.get('builtin')
    if builtin == 'image_folder':
        args = deepcopy(data.get('args', {}))
        for k, v in data.items():
            if k not in ('builtin', 'args'):
                args[k] = deepcopy(v)
        args.setdefault('shuffle', False)
        return {'factory': 'image_folder', 'args': args}
    if builtin == 'cifar10':
        args = deepcopy(data.get('args', {}))
        for k, v in data.items():
            if k not in ('builtin', 'args'):
                args[k] = deepcopy(v)
        args.setdefault('sampling', 'sequential')
        args.setdefault('horizontal_flip', False)
        return {'factory': 'hypergan.image_data:CIFAR10Data', 'args': args}
    raise ValueError('data spec dict must provide {factory, args} or builtin="image_folder"|"cifar10"')


def _instantiate_metric_spec(
    item,
    *,
    reference_cache,
    manifold_samples,
    pr_k,
    dc_k,
    knn_chunk_size,
    inception_weights_path,
    dinov2_weights_path,
):
    if not isinstance(item, str):
        if not callable(getattr(item, 'describe', None)) or not callable(getattr(item, 'evaluate', None)):
            raise ValueError('Custom metric instance must expose describe() and evaluate()')
        return item
    key = item.strip().lower()
    if key == 'inception':
        return InceptionSuite(
            weights_path=inception_weights_path,
            reference_cache=reference_cache,
            manifold_samples=manifold_samples,
            pr_k=pr_k,
            dc_k=dc_k,
            knn_chunk_size=knn_chunk_size,
        )
    if key == 'dinov2':
        return DINOv2Suite(
            weights_path=dinov2_weights_path,
            reference_cache=reference_cache,
            manifold_samples=manifold_samples,
            pr_k=pr_k,
            dc_k=dc_k,
            knn_chunk_size=knn_chunk_size,
        )
    if key == 'inception_fid':
        return InceptionFID(weights_path=inception_weights_path, reference_cache=reference_cache)
    if key == 'dinov2_fd':
        return DINOv2FD(weights_path=dinov2_weights_path, reference_cache=reference_cache)
    raise ValueError(f"Unknown metric name {item!r}; expected 'inception', 'dinov2', 'inception_fid', or 'dinov2_fd'")


def _resolve_metric_map(
    metrics,
    *,
    reference_cache,
    manifold_samples,
    pr_k,
    dc_k,
    knn_chunk_size,
    inception_weights_path,
    dinov2_weights_path,
):
    kwargs = dict(
        reference_cache=reference_cache,
        manifold_samples=manifold_samples,
        pr_k=pr_k,
        dc_k=dc_k,
        knn_chunk_size=knn_chunk_size,
        inception_weights_path=inception_weights_path,
        dinov2_weights_path=dinov2_weights_path,
    )
    if isinstance(metrics, dict):
        if not metrics:
            raise ValueError('metrics mapping must not be empty')
        return {str(name): _instantiate_metric_spec(spec, **kwargs) for name, spec in metrics.items()}
    if isinstance(metrics, str):
        metrics = (metrics,)
    resolved = {}
    for item in metrics:
        name = item if isinstance(item, str) else type(item).__name__.lower()
        resolved[name] = _instantiate_metric_spec(item, **kwargs)
    if not resolved:
        raise ValueError('metrics must specify at least one metric')
    return resolved


def evaluate_bundle(
    bundle_path,
    *,
    data='train',
    metrics=('inception', 'dinov2'),
    sample_count=50000,
    batch_size=128,
    seed=10000,
    prior_seed=None,
    data_seed=None,
    device='cpu',
    generated_binding=None,
    reference_cache=None,
    manifold_samples=DEFAULT_MANIFOLD_SAMPLES,
    pr_k=DEFAULT_PR_K,
    dc_k=DEFAULT_DC_K,
    knn_chunk_size=DEFAULT_KNN_CHUNK_SIZE,
    inception_weights_path=None,
    dinov2_weights_path=None,
    output_json=None,
    publish_run_dir=None,
    evaluator_prefix=None,
):
    """Evaluate an exported ``model.pt`` EMA bundle on one or more image metrics.

    Uses the exact same deterministic RNG seeding (``data_seed = seed + 1``,
    ``prior_seed = seed + 2``) and reference cache identity protocol as
    ``metric_evaluation_worker.evaluate_snapshot`` so cached reference moments
    and manifolds are shared seamlessly across live training and post-hoc
    benchmark runs. Optionally writes a JSON receipt to ``output_json`` and/or
    publishes scores into a HyperGAN run directory via ``evaluation_results.import_scores``.
    """
    from copy import deepcopy
    import random
    import numpy as np
    import torch

    from .artifacts import _restore_buffers
    from .config import resolve_config
    from .evaluation_results import import_scores
    from .metric_evaluation import _sha256
    from .metric_evaluation_worker import _bounded_identity, _draw, _sources
    from .metric_plugins import check_evaluation_data, finite_json, scalar_outputs, scalar_values
    from .metrics import normalize_device
    from .numerical_policy import apply_backend_policy, backend_info
    from .recipes import ComponentGraph, construct, execution_device, make_prior, move_tensors
    from .run_state import atomic_json

    if type(sample_count) is not int or sample_count < 2:
        raise ValueError('sample_count must be an integer >= 2')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32 - 1]')

    candidate = Path(bundle_path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / 'model.pt'
    if not candidate.is_file():
        raise FileNotFoundError(f'EMA bundle not found: {candidate}')

    bundle_sha256 = _sha256(candidate)
    saved = torch.load(candidate, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('schema_version') != 1 or saved.get('kind') != 'ema-inference':
        raise ValueError('evaluate_bundle requires a schema_version=1 kind="ema-inference" EMA bundle')

    config = resolve_config(saved['config'])
    norm_device = normalize_device(device)
    apply_backend_policy({'training': {**config['training'], 'device': norm_device}})
    exec_device = execution_device(norm_device)

    binding = generated_binding or config['sampling'].get('generated', 'generated')
    if binding.startswith('components.') and binding.split('.')[1] not in saved['components']:
        raise ValueError(f'Bundle does not contain generated component: {binding}')

    metric_map = _resolve_metric_map(
        metrics,
        reference_cache=reference_cache,
        manifold_samples=manifold_samples,
        pr_k=pr_k,
        dc_k=dc_k,
        knn_chunk_size=knn_chunk_size,
        inception_weights_path=inception_weights_path,
        dinov2_weights_path=dinov2_weights_path,
    )

    data_spec = _normalize_eval_data_spec(data, config)
    callable_initial_state = None
    if data_spec is None:
        if callable(getattr(data, 'state_dict', None)) and callable(getattr(data, 'load_state_dict', None)):
            callable_initial_state = deepcopy(data.state_dict())

    eff_data_seed = (seed + 1) if data_seed is None else data_seed
    eff_prior_seed = (seed + 2) if prior_seed is None else prior_seed
    if type(eff_data_seed) is not int or not 0 <= eff_data_seed < 2**32:
        raise ValueError('data_seed must be an integer in [0, 2**32 - 1]')
    if type(eff_prior_seed) is not int or not 0 <= eff_prior_seed < 2**32:
        raise ValueError('prior_seed must be an integer in [0, 2**32 - 1]')

    graph = ComponentGraph(saved['components']).float().eval().requires_grad_(False)
    for name, model in graph.models.items():
        model.load_state_dict(saved['model_states'][name])
        _restore_buffers(model, saved['model_buffers'][name])
    prior = make_prior(config['prior'], device='cpu', recipe=config.get('recipe')).float().eval().requires_grad_(False)
    prior.load_state_dict(saved['prior'])
    _restore_buffers(prior, saved['prior_buffers'])
    graph, prior = graph.to(exec_device), prior.to(exec_device)

    inputs_spec = {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'}
    eval_spec = {'sample_count': sample_count, 'batch_size': batch_size, 'seed': seed}
    owner = dict(saved.get('identity') or {})
    step = int(saved.get('step', 0))

    metric_results = {}
    published_receipts = {}
    bounded_data_identity = None

    for metric_name, instance in metric_map.items():
        descriptor = finite_json(instance.describe())
        if data_spec is not None:
            data_obj = construct(data_spec)
        else:
            data_obj = data
            if callable_initial_state is not None:
                data_obj.load_state_dict(deepcopy(callable_initial_state))
        check_evaluation_data(data_obj, eval_spec, inputs_spec)

        raw_data_identity = (
            data_obj.resume_identity() if callable(getattr(data_obj, 'resume_identity', None)) else None
        )
        if raw_data_identity is None and data_spec is not None:
            raw_data_identity = {'factory': data_spec['factory'], 'args': data_spec['args']}
        content_pinned = raw_data_identity is not None and not (
            isinstance(raw_data_identity, dict)
            and raw_data_identity.get('kind') == 'item_data'
            and raw_data_identity.get('dataset_identity') is None
        )
        reference_identity = None
        if content_pinned:
            data_id_sha = hashlib.sha256(_canonical(raw_data_identity)).hexdigest()
            bounded_data_identity = _bounded_identity(raw_data_identity)
            data_sources = {}
            try:
                data_sources = _sources([type(data_obj)])
            except ValueError:
                data_sources = {'class': f'{type(data_obj).__module__}:{type(data_obj).__qualname__}'}
            reference_identity = {
                'data_identity_sha256': data_id_sha,
                'data_sources': data_sources,
                'data_factory': data_spec['factory'] if data_spec is not None else type(data_obj).__qualname__,
                'sample_count': sample_count,
                'batch_size': batch_size,
                'seed': seed,
                'inputs': inputs_spec,
                'device': str(exec_device),
                'torch': torch.__version__,
                'numpy': np.__version__,
                'backend': backend_info(),
            }
            if data_seed is not None:
                reference_identity['data_seed'] = eff_data_seed
        elif raw_data_identity is not None:
            bounded_data_identity = _bounded_identity(raw_data_identity)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data_rng = torch.Generator().manual_seed(eff_data_seed)
        prior_rng = torch.Generator(device=exec_device).manual_seed(eff_prior_seed)

        count = 0

        def _batches():
            nonlocal count
            while count < sample_count:
                size = min(batch_size, sample_count - count)
                batch = _draw(data_obj, size, data_rng, count, sample_count)
                if not isinstance(batch, dict) or not isinstance(batch.get('real'), torch.Tensor) or len(batch['real']) != size:
                    raise ValueError('Evaluation data must return a batched real tensor')
                batch = move_tensors(batch, exec_device)
                z, _ = prior.sample(size, generator=prior_rng)
                graph_context = graph.generate(z, batch, prior=prior)
                gen = graph.resolve(binding, graph_context)
                if (not isinstance(gen, torch.Tensor) or len(gen) != size
                        or not torch.isfinite(gen).all() or not torch.isfinite(batch['real']).all()):
                    raise ValueError('Evaluation generated/reference tensors must be finite and matching batch size')
                count += size
                yield {'generated': gen, 'reference': batch['real']}

        context = {
            'sample_count': sample_count,
            'seed': seed,
            'step': step,
            'snapshot_sha256': bundle_sha256,
            'run_id': owner.get('run_id', 'standalone'),
            'attempt_id': owner.get('attempt_id', 'standalone'),
            'reference_identity': reference_identity,
        }
        t0 = time.monotonic()
        with torch.inference_mode():
            raw_result = instance.evaluate(batches=_batches(), context=context)
        elapsed = time.monotonic() - t0
        if count != sample_count:
            raise ValueError(f'Metric {metric_name!r} did not consume all {sample_count} samples (got {count})')

        if descriptor.get('kind') == 'scalars':
            outputs_meta = scalar_outputs(descriptor['outputs'])
            values = scalar_values(raw_result, outputs_meta)
            entry = {
                'kind': 'scalars',
                'values': values,
                'outputs': outputs_meta,
                'seconds': elapsed,
            }
        else:
            val = float(raw_result)
            if not math.isfinite(val):
                raise ValueError(f'Metric {metric_name!r} returned a nonfinite scalar')
            local_key = 'fd_dinov2' if isinstance(instance, DINOv2FD) else 'fid' if isinstance(instance, InceptionFID) else 'value'
            meta = {k: v for k, v in descriptor.items() if k in ('label', 'unit', 'direction', 'description')}
            entry = {
                'kind': 'scalar',
                'value': val,
                'values': {local_key: val},
                'outputs': scalar_outputs({local_key: meta}),
                'seconds': elapsed,
            }
        metric_results[metric_name] = entry

        if publish_run_dir is not None:
            eval_id = f'{evaluator_prefix}_{metric_name}' if evaluator_prefix else metric_name
            pub_outputs = {
                k: {mk: mv for mk, mv in meta.items() if mk in ('label', 'unit', 'direction', 'description')}
                for k, meta in entry['outputs'].items()
            }
            pub_protocol = {
                'schema_version': 1,
                'evaluator_class': f'{type(instance).__module__}:{type(instance).__qualname__}',
                'sample_count': sample_count,
                'batch_size': batch_size,
                'seed': seed,
                'data_seed': eff_data_seed,
                'prior_seed': eff_prior_seed,
                'device': str(exec_device),
                'generated_binding': binding,
                'data_identity': bounded_data_identity,
                'ema': True,
            }
            pub_snapshot_id = dict(owner)
            pub_snapshot_id['source_step'] = step
            published_receipts[eval_id] = import_scores(
                Path(publish_run_dir).expanduser().resolve(),
                eval_id,
                step=step,
                values=entry['values'],
                outputs=pub_outputs,
                attempt_id=owner.get('attempt_id'),
                snapshot_identity=pub_snapshot_id,
                snapshot_sha256=bundle_sha256,
                protocol=pub_protocol,
            )

    receipt = {
        'schema_version': 1,
        'bundle_path': str(candidate),
        'bundle_sha256': bundle_sha256,
        'step': step,
        'snapshot_identity': owner,
        'sample_count': sample_count,
        'batch_size': batch_size,
        'seed': seed,
        'data_seed': eff_data_seed,
        'prior_seed': eff_prior_seed,
        'device': str(exec_device),
        'generated_binding': binding,
        'data_identity': bounded_data_identity,
        'metrics': metric_results,
    }
    if published_receipts:
        receipt['published_receipts'] = published_receipts
    if output_json is not None:
        atomic_json(Path(output_json).expanduser().resolve(), receipt)
    return receipt


def main(argv=None):
    """CLI entry point: ``python -m hypergan.image_metrics <bundle.pt> ...``."""
    parser = argparse.ArgumentParser(
        prog='python -m hypergan.image_metrics',
        description='Evaluate an exported HyperGAN EMA model.pt bundle on InceptionSuite / DINOv2Suite.',
    )
    parser.add_argument('bundle', type=Path, help='Path to model.pt EMA bundle or directory containing model.pt')
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['inception', 'dinov2'],
        choices=['inception', 'dinov2', 'inception_fid', 'dinov2_fd'],
        help='Metrics to compute (default: inception dinov2)',
    )
    parser.add_argument('--samples', type=int, default=50000, help='Total generated/reference samples (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=128, help='Evaluation batch size (default: 128)')
    parser.add_argument('--seed', type=int, default=10000, help='Deterministic evaluation seed (default: 10000)')
    parser.add_argument('--prior-seed', type=int, default=None, help='Override prior RNG seed (default: seed + 2)')
    parser.add_argument('--data-seed', type=int, default=None, help='Override data RNG seed (default: seed + 1)')
    parser.add_argument('--device', default='cpu', help='Execution device: cpu, cuda, cuda:0, ... (default: cpu)')
    parser.add_argument('--generated', dest='generated_binding', default=None, help='Override generated output binding')
    parser.add_argument('--manifold-samples', type=int, default=DEFAULT_MANIFOLD_SAMPLES, help='k-NN manifold subset size')
    parser.add_argument('--pr-k', type=int, default=DEFAULT_PR_K, help='Precision/Recall k-NN neighbor count')
    parser.add_argument('--dc-k', type=int, default=DEFAULT_DC_K, help='Density/Coverage k-NN neighbor count')
    parser.add_argument('--knn-chunk-size', type=int, default=DEFAULT_KNN_CHUNK_SIZE, help='Chunk size for k-NN L2 distances')
    parser.add_argument('--reference-cache', default=None, help='Reference cache directory or "false" to disable')
    parser.add_argument('--inception-weights', type=Path, default=None, help='Optional local Inception-v3 weights path')
    parser.add_argument('--dinov2-weights', type=Path, default=None, help='Optional local DINOv2 ViT-L/14 weights path')

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--data-folder', type=Path, default=None, help='Reference image_folder directory')
    data_group.add_argument('--cifar10-root', type=Path, default=None, help='Reference CIFAR-10 root directory')
    data_group.add_argument('--data-json', default=None, help='Reference data spec as JSON string')
    parser.add_argument('--width', type=int, default=None, help='Image width for --data-folder')
    parser.add_argument('--height', type=int, default=None, help='Image height for --data-folder')
    parser.add_argument('--split', default='train', choices=['train', 'test'], help='Split for --cifar10-root')

    parser.add_argument('--output', type=Path, default=None, help='Optional path to write JSON receipt')
    parser.add_argument('--publish-run', type=Path, default=None, help='Optional run dir to publish scores into')
    parser.add_argument('--evaluator-prefix', default=None, help='Optional prefix for published evaluator IDs')

    args = parser.parse_args(argv)
    ref_cache = False if isinstance(args.reference_cache, str) and args.reference_cache.strip().lower() == 'false' else args.reference_cache

    if args.data_folder is not None:
        folder_args = {'root': str(args.data_folder), 'shuffle': False}
        if args.width is not None:
            folder_args['width'] = args.width
        if args.height is not None:
            folder_args['height'] = args.height
        data = {'factory': 'image_folder', 'args': folder_args}
    elif args.cifar10_root is not None:
        data = {
            'factory': 'hypergan.image_data:CIFAR10Data',
            'args': {
                'root': str(args.cifar10_root),
                'split': args.split,
                'sampling': 'sequential',
                'horizontal_flip': False,
            },
        }
    elif args.data_json is not None:
        data = json.loads(args.data_json)
    else:
        data = 'train'

    receipt = evaluate_bundle(
        args.bundle,
        data=data,
        metrics=args.metrics,
        sample_count=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        prior_seed=args.prior_seed,
        data_seed=args.data_seed,
        device=args.device,
        generated_binding=args.generated_binding,
        reference_cache=ref_cache,
        manifold_samples=args.manifold_samples,
        pr_k=args.pr_k,
        dc_k=args.dc_k,
        knn_chunk_size=args.knn_chunk_size,
        inception_weights_path=args.inception_weights,
        dinov2_weights_path=args.dinov2_weights,
        output_json=args.output,
        publish_run_dir=args.publish_run,
        evaluator_prefix=args.evaluator_prefix,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

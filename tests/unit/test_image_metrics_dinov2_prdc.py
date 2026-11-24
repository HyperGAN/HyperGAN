"""Unit and integration tests for DINOv2FD, InceptionSuite, DINOv2Suite, PRDC math, and evaluate_bundle.

Pure CPU-only tests with synthetic extractors and zero network downloads.
"""
import hashlib
import json
import math
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
import scipy.linalg
from scipy.spatial.distance import cdist
import torch
from torch import nn

import hypergan as hg
from hypergan.artifacts import bundle_state, save_bundle
from hypergan.cache import (
    DINOV2_VITL14_SHA256,
    DINOV2_VITL14_SIZE,
    DINOV2_VITL14_URL,
)
from hypergan.config import config_values, resolve_config
from hypergan.image_metrics import (
    DEFAULT_DC_K,
    DEFAULT_MANIFOLD_SAMPLES,
    DEFAULT_PR_K,
    DINOV2_SHA256,
    INCEPTION_SHA256,
    TORCH_FIDELITY_VERSION,
    DINOv2FD,
    DINOv2Suite,
    InceptionSuite,
    _ManifoldSubset,
    _build_dinov2_vitl14,
    _checked_dinov2_weights,
    _dinov2_preprocess,
    _manifold_indices,
    compute_prdc,
    evaluate_bundle,
    frechet_distance,
    knn_radii,
    load_dinov2_vitl14,
    main as image_metrics_cli_main,
    read_suite_reference,
    write_suite_reference,
)
from hypergan.metric_evaluation_worker import evaluate_snapshot
from hypergan.metric_plugins import _factory, scalar_outputs
from hypergan.metrics import metric_catalog
from hypergan.run_state import atomic_json
from hypergan.training import ReferenceTrainer


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# ---------------------------------------------------------------------------
# Synthetic Deterministic Extractors & Test Helpers
# ---------------------------------------------------------------------------

class _SyntheticInceptionExtractor:
    """Deterministic uint8 RGB -> 24-d feature projection recording call counts."""
    def __init__(self, log, dim=24):
        self.log = log
        self.dim = dim

    def to(self, device):
        return self

    def __call__(self, images):
        assert images.dtype == torch.uint8
        flat = images.reshape(len(images), -1).double() / 255.0
        rng = np.random.default_rng(17)
        proj = torch.from_numpy(rng.normal(size=(flat.shape[1], self.dim)))
        feats = (flat @ proj + 0.25 * flat.square() @ proj.abs()).float()
        self.log.append((len(images), feats.cpu().numpy().copy()))
        return (feats,)


class _SyntheticDINOv2Extractor:
    """Deterministic preprocessed 224x224 float32 -> 24-d CLS feature projection."""
    def __init__(self, log, dim=24):
        self.log = log
        self.dim = dim

    def to(self, device):
        return self

    def __call__(self, images):
        assert images.dtype == torch.float32
        assert images.shape[1:] == (3, 224, 224)
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (4, 4)).reshape(len(images), -1).double()
        rng = np.random.default_rng(29)
        proj = torch.from_numpy(rng.normal(size=(pooled.shape[1], self.dim)))
        feats = (pooled @ proj + 0.1 * pooled.square() @ proj.abs()).float()
        self.log.append((len(images), feats.cpu().numpy().copy()))
        return feats


class FakeDINOv2FD(DINOv2FD):
    log = []

    def __init__(self, reference_cache=None):
        self.weights_path = None
        self.weights_sha256 = DINOV2_SHA256
        self.reference_cache = reference_cache

    def _extractor(self):
        return _SyntheticDINOv2Extractor(FakeDINOv2FD.log)


class FakeInceptionSuite(InceptionSuite):
    log = []

    def __init__(self, reference_cache=None, manifold_samples=DEFAULT_MANIFOLD_SAMPLES, pr_k=DEFAULT_PR_K, dc_k=DEFAULT_DC_K, knn_chunk_size=16):
        self.weights_path = None
        self.weights_sha256 = INCEPTION_SHA256
        self.torch_fidelity_version = TORCH_FIDELITY_VERSION
        super(InceptionSuite, self).__init__(
            reference_cache=reference_cache,
            manifold_samples=manifold_samples,
            pr_k=pr_k,
            dc_k=dc_k,
            knn_chunk_size=knn_chunk_size,
        )

    def _extractor(self):
        return _SyntheticInceptionExtractor(FakeInceptionSuite.log)


class FakeDINOv2Suite(DINOv2Suite):
    log = []

    def __init__(self, reference_cache=None, manifold_samples=DEFAULT_MANIFOLD_SAMPLES, pr_k=DEFAULT_PR_K, dc_k=DEFAULT_DC_K, knn_chunk_size=16):
        self.weights_path = None
        self.weights_sha256 = DINOV2_SHA256
        super(DINOv2Suite, self).__init__(
            reference_cache=reference_cache,
            manifold_samples=manifold_samples,
            pr_k=pr_k,
            dc_k=dc_k,
            knn_chunk_size=knn_chunk_size,
        )

    def _extractor(self):
        return _SyntheticDINOv2Extractor(FakeDINOv2Suite.log)


class _ScaleView(nn.Module):
    def forward(self, x):
        return x * 0.6 + 0.05


def _textbook_prdc_oracle(real_features, fake_features, pr_k=3, dc_k=5):
    """Exact O(N^2) brute-force NumPy/SciPy reference implementation for testing."""
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)

    d_rr = cdist(real, real, metric='euclidean')
    np.fill_diagonal(d_rr, np.inf)
    r_pr = np.partition(d_rr, pr_k - 1, axis=1)[:, pr_k - 1]
    r_dc = np.partition(d_rr, dc_k - 1, axis=1)[:, dc_k - 1]

    d_ff = cdist(fake, fake, metric='euclidean')
    np.fill_diagonal(d_ff, np.inf)
    f_pr = np.partition(d_ff, pr_k - 1, axis=1)[:, pr_k - 1]

    d_rf = cdist(real, fake, metric='euclidean')
    precision = float((d_rf < r_pr[:, None]).any(axis=0).mean())
    recall = float((d_rf < f_pr[None, :]).any(axis=1).mean())
    density = float((d_rf < r_dc[:, None]).sum(axis=0).mean() / float(dc_k))
    coverage = float((d_rf.min(axis=1) < r_dc).mean())
    return {
        'precision': precision,
        'recall': recall,
        'density': density,
        'coverage': coverage,
        'r_pr': r_pr,
        'r_dc': r_dc,
    }


def _scipy_fd_oracle(gen_features, ref_features):
    mu1, mu2 = gen_features.mean(0), ref_features.mean(0)
    sigma1 = np.cov(gen_features, rowvar=False)
    sigma2 = np.cov(ref_features, rowvar=False)
    root = scipy.linalg.sqrtm(sigma1 @ sigma2).real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(sigma1 + sigma2 - 2.0 * root))


def _make_images(rng, count, h=8, w=8):
    return torch.from_numpy(rng.uniform(-1.0, 1.0, size=(count, 3, h, w))).float()


def _make_batches(generated, reference, sizes):
    batches, start = [], 0
    for size in sizes:
        batches.append({
            'generated': generated[start:start + size],
            'reference': reference[start:start + size],
        })
        start += size
    return batches


def _protocol(**changes):
    return {
        'data_identity_sha256': 'b' * 64,
        'sample_count': 24,
        'batch_size': 8,
        'seed': 11,
        'device': 'cpu',
        'backend': {'cudnn_allow_tf32': False},
        **changes,
    }


# ---------------------------------------------------------------------------
# 1. Torch-free Import & Pinned Weight Constants / Offline Guard
# ---------------------------------------------------------------------------

def test_image_metrics_import_is_torch_and_numpy_free():
    """Importing hypergan.image_metrics must not import torch or numpy."""
    code = (
        "import sys; "
        "import hypergan.image_metrics as im; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly'; "
        "assert 'numpy' not in sys.modules, 'numpy imported eagerly'; "
        "assert hasattr(im, 'DINOv2FD') and hasattr(im, 'InceptionSuite') and hasattr(im, 'DINOv2Suite')"
    )
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert res.returncode == 0, res.stderr


def test_dinov2_pinned_weights_validation_and_offline_cache(tmp_path, monkeypatch):
    assert DINOV2_SHA256 == DINOV2_VITL14_SHA256
    assert len(DINOV2_VITL14_SHA256) == 64
    assert DINOV2_VITL14_SIZE == 1217586395
    assert DINOV2_VITL14_URL.endswith('/dinov2_vitl14_pretrain.pth')

    missing = tmp_path / 'missing_dinov2.pth'
    with pytest.raises(ValueError, match='missing'):
        _checked_dinov2_weights(missing, DINOV2_SHA256)

    wrong = tmp_path / 'wrong_dinov2.pth'
    wrong.write_bytes(b'corrupted bytes')
    with pytest.raises(ValueError, match='does not match'):
        _checked_dinov2_weights(wrong, DINOV2_SHA256)

    with pytest.raises(ValueError, match='pinned'):
        _checked_dinov2_weights(wrong, '0' * 64)

    # Offline mode refuses uncached remote URL without attempting network access
    empty_cache = tmp_path / 'empty_cache'
    empty_cache.mkdir()
    monkeypatch.setenv('HYPERGAN_CACHE_DIR', str(empty_cache))
    monkeypatch.setenv('HYPERGAN_OFFLINE', '1')
    with pytest.raises(ValueError, match='missing'):
        _checked_dinov2_weights(None, DINOV2_SHA256)

    # Pre-cached file under HYPERGAN_CACHE_DIR succeeds completely offline
    payload = b'synthetic dinov2 weights payload'
    sha = hashlib.sha256(payload).hexdigest()
    cached = empty_cache / 'dinov2_vitl14_pretrain.pth'
    cached.write_bytes(payload)
    monkeypatch.setattr('hypergan.image_metrics.DINOV2_SHA256', sha)
    monkeypatch.setattr('hypergan.image_metrics.DINOV2_VITL14_SHA256', sha)
    monkeypatch.setattr('hypergan.image_metrics.DINOV2_VITL14_SIZE', len(payload))
    assert _checked_dinov2_weights(None, sha) == cached.resolve()


# ---------------------------------------------------------------------------
# 2. Self-Contained DINOv2 ViT-L/14 Architecture & Preprocessing
# ---------------------------------------------------------------------------

def test_dinov2_vitl14_state_dict_schema_matches_meta_checkpoint():
    """Verify full-scale ViT-L/14 state_dict has the exact 343 Meta checkpoint keys and shapes."""
    with torch.device('meta'):
        model = _build_dinov2_vitl14()
    sd = model.state_dict()
    assert len(sd) == 343
    assert tuple(sd['cls_token'].shape) == (1, 1, 1024)
    assert tuple(sd['pos_embed'].shape) == (1, 1370, 1024)
    assert tuple(sd['mask_token'].shape) == (1, 1024)
    assert tuple(sd['patch_embed.proj.weight'].shape) == (1024, 3, 14, 14)
    assert tuple(sd['patch_embed.proj.bias'].shape) == (1024,)
    assert tuple(sd['blocks.0.attn.qkv.weight'].shape) == (3072, 1024)
    assert tuple(sd['blocks.0.ls1.gamma'].shape) == (1024,)
    assert tuple(sd['blocks.23.mlp.fc1.weight'].shape) == (4096, 1024)
    assert tuple(sd['blocks.23.ls2.gamma'].shape) == (1024,)
    assert tuple(sd['norm.weight'].shape) == (1024,)
    assert tuple(sd['norm.bias'].shape) == (1024,)


def test_dinov2_small_vit_forward_pos_embed_interpolation_and_determinism(tmp_path, monkeypatch):
    """Exercise real ViT forward, pos_embed bicubic interpolation, strict load, and determinism on CPU."""
    torch.manual_seed(123)
    tiny = _build_dinov2_vitl14(img_size=70, patch_size=14, embed_dim=32, depth=2, num_heads=4, mlp_ratio=2.0)
    weights_path = tmp_path / 'tiny_dino.pth'
    torch.save(tiny.state_dict(), weights_path)
    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    monkeypatch.setattr('hypergan.image_metrics.DINOV2_SHA256', sha)
    monkeypatch.setattr(
        'hypergan.image_metrics._build_dinov2_vitl14',
        lambda: _build_dinov2_vitl14(img_size=70, patch_size=14, embed_dim=32, depth=2, num_heads=4, mlp_ratio=2.0),
    )
    loaded = load_dinov2_vitl14(weights_path, expected_sha256=sha)
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        x_224 = torch.randn(3, 3, 224, 224)
        x_native = torch.randn(2, 3, 70, 70)
        with torch.inference_mode():
            y1 = loaded(x_224)
            y2 = loaded(x_224)
            y_native = loaded(x_native)
        assert y1.shape == (3, 32)
        assert y_native.shape == (2, 32)
        assert torch.isfinite(y1).all() and torch.isfinite(y_native).all()
        assert torch.equal(y1, y2)
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def test_dinov2_preprocess_resize_and_imagenet_normalization():
    # Constant RGB image of (128, 64, 192) stays uniform after bicubic resize -> exact normalized channels
    img = torch.zeros((2, 3, 32, 32), dtype=torch.uint8)
    img[:, 0] = 128
    img[:, 1] = 64
    img[:, 2] = 192
    out = _dinov2_preprocess(img)
    assert out.shape == (2, 3, 224, 224)
    assert out.dtype == torch.float32
    expected = torch.tensor([
        (128.0 / 255.0 - 0.485) / 0.229,
        (64.0 / 255.0 - 0.456) / 0.224,
        (192.0 / 255.0 - 0.406) / 0.225,
    ], dtype=torch.float32)
    for c in range(3):
        assert torch.allclose(out[:, c], expected[c].expand(2, 224, 224), atol=1e-5)

    # 224x224 input skips resize and preserves exact per-pixel values
    exact = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
    out_exact = _dinov2_preprocess(exact)
    manual = (exact.float() / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    assert torch.allclose(out_exact, manual, atol=1e-6)

    with pytest.raises(ValueError, match='uint8'):
        _dinov2_preprocess(torch.zeros((1, 3, 32, 32), dtype=torch.float32))


# ---------------------------------------------------------------------------
# 3. Fréchet Distance (SciPy-sqrtm-free) & PRDC k-NN Math Verification
# ---------------------------------------------------------------------------

def test_frechet_distance_matches_scipy_oracle_without_calling_sqrtm(monkeypatch):
    rng = np.random.default_rng(7)
    gen = rng.normal(loc=0.3, scale=1.2, size=(64, 16))
    ref = rng.normal(loc=-0.2, scale=0.9, size=(64, 16))
    expected = _scipy_fd_oracle(gen, ref)

    mock_sqrtm = MagicMock(side_effect=AssertionError('scipy.linalg.sqrtm must never be called'))
    monkeypatch.setattr(scipy.linalg, 'sqrtm', mock_sqrtm)

    stat_gen = {'mu': gen.mean(0), 'sigma': np.cov(gen, rowvar=False)}
    stat_ref = {'mu': ref.mean(0), 'sigma': np.cov(ref, rowvar=False)}
    got = frechet_distance(stat_gen, stat_ref)
    assert got == pytest.approx(expected, rel=1e-9)
    assert frechet_distance(stat_ref, stat_ref) == pytest.approx(0.0, abs=1e-9)
    assert mock_sqrtm.call_count == 0

    with pytest.raises(ValueError, match='incompatible shapes'):
        frechet_distance({'mu': np.zeros(4), 'sigma': np.eye(4)}, {'mu': np.zeros(5), 'sigma': np.eye(5)})
    with pytest.raises(ValueError, match='finite'):
        frechet_distance({'mu': np.array([np.nan, 0.0]), 'sigma': np.eye(2)}, {'mu': np.zeros(2), 'sigma': np.eye(2)})


def test_knn_radii_excludes_self_distance_and_matches_oracle():
    # Explicit 1D points at [0, 1, 3, 6, 10]:
    # Non-self sorted distances:
    #   0 -> [1, 3, 6, 10], 1 -> [1, 2, 5, 9], 3 -> [2, 3, 3, 7], 6 -> [3, 4, 5, 6], 10 -> [4, 7, 9, 10]
    pts = np.array([[0.0], [1.0], [3.0], [6.0], [10.0]], dtype=np.float32)
    radii = knn_radii(pts, (1, 2, 3), chunk_size=2)
    np.testing.assert_allclose(radii[1], [1.0, 1.0, 2.0, 3.0, 4.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(radii[2], [3.0, 2.0, 3.0, 4.0, 7.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(radii[3], [6.0, 5.0, 3.0, 5.0, 9.0], rtol=1e-6, atol=1e-6)

    # Compare against SciPy brute-force oracle across small and large chunks and huge coordinate offset
    rng = np.random.default_rng(101)
    real = rng.normal(size=(73, 32)).astype(np.float32) + 1e5
    fake = rng.normal(loc=0.25, size=(67, 32)).astype(np.float32) + 1e5
    oracle = _textbook_prdc_oracle(real, fake, pr_k=3, dc_k=5)

    got_radii = knn_radii(real, (3, 5), chunk_size=11)
    np.testing.assert_allclose(got_radii[3], oracle['r_pr'], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_radii[5], oracle['r_dc'], rtol=1e-5, atol=1e-5)

    for chunk in (7, 19, 128):
        got = compute_prdc(real, fake, pr_k=3, dc_k=5, chunk_size=chunk)
        for key in ('precision', 'recall', 'density', 'coverage'):
            assert got[key] == pytest.approx(oracle[key], abs=1e-12)


def test_prdc_geometric_properties_identical_iid_mode_drop_and_shifted():
    rng = np.random.default_rng(2026)

    # 1. Identical distributions (exact same manifold) -> Precision=1, Recall=1, Density=1, Coverage=1
    base = rng.normal(size=(120, 16)).astype(np.float32)
    identical = compute_prdc(base, base, pr_k=3, dc_k=5, chunk_size=23)
    assert identical['precision'] == pytest.approx(1.0)
    assert identical['recall'] == pytest.approx(1.0)
    assert identical['density'] == pytest.approx(1.0)
    assert identical['coverage'] == pytest.approx(1.0)

    # 2. Two independent i.i.d. draws from the same low-d Gaussian -> high P/R, Density ~ 1, Coverage ~ 1 - 2^-5
    iid_r = rng.normal(size=(400, 4)).astype(np.float32)
    iid_g = rng.normal(size=(400, 4)).astype(np.float32)
    iid = compute_prdc(iid_r, iid_g, pr_k=3, dc_k=5, chunk_size=64)
    assert 0.80 <= iid['precision'] <= 1.0
    assert 0.80 <= iid['recall'] <= 1.0
    assert 0.85 <= iid['density'] <= 1.15
    assert 0.90 <= iid['coverage'] <= 1.0

    # 3. Mode-dropped subset: 4 well-separated modes in R, G samples only mode 0
    centers = np.array([[0, 0, 0, 0], [30, 0, 0, 0], [0, 30, 0, 0], [0, 0, 30, 0]], dtype=np.float32)
    real_modes = np.concatenate([rng.normal(loc=c, scale=1.0, size=(80, 4)) for c in centers], axis=0).astype(np.float32)
    fake_single_mode = rng.normal(loc=centers[0], scale=1.0, size=(320, 4)).astype(np.float32)
    dropped = compute_prdc(real_modes, fake_single_mode, pr_k=3, dc_k=5, chunk_size=37)
    assert dropped['precision'] > 0.85
    assert dropped['recall'] < 0.30
    assert dropped['coverage'] < 0.30
    assert dropped['density'] > 0.70

    # 4. Shifted / disjoint distribution -> Precision=0, Recall=0, Density=0, Coverage=0
    shifted = compute_prdc(base, base + 50.0, pr_k=3, dc_k=5, chunk_size=23)
    assert shifted == {'precision': 0.0, 'recall': 0.0, 'density': 0.0, 'coverage': 0.0}


def test_prdc_validation_boundaries():
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    with pytest.raises(ValueError, match='at least 6'):
        knn_radii(x, (5,))
    with pytest.raises(ValueError, match='at least 6'):
        compute_prdc(x, x, pr_k=3, dc_k=5)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        compute_prdc(np.zeros((10, 4), dtype=np.float32), np.vstack([bad[:, :4], np.zeros((6, 4), dtype=np.float32)]))
    with pytest.raises(ValueError, match='matching dimensions'):
        compute_prdc(np.zeros((10, 4), dtype=np.float32), np.zeros((10, 5), dtype=np.float32))


def test_manifold_subset_is_batch_size_invariant_and_uniform_across_ordered_data():
    full = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    idx = _manifold_indices(100, 25)
    assert len(idx) == 25
    assert np.all(idx[:-1] < idx[1:])
    # Spans across the entire [0, 100) range rather than taking a biased prefix
    assert idx[0] < 20 and idx[-1] > 80

    s_one = _ManifoldSubset(idx)
    s_one.add(full)
    s_uneven = _ManifoldSubset(idx)
    cursor = 0
    for b in (1, 7, 13, 2, 19, 31, 27):
        s_uneven.add(full[cursor:cursor + b])
        cursor += b
    assert cursor == 100
    np.testing.assert_array_equal(s_one.array(), full[idx])
    np.testing.assert_array_equal(s_uneven.array(), full[idx])


# ---------------------------------------------------------------------------
# 4. DINOv2FD, InceptionSuite, DINOv2Suite Evaluators & Caching
# ---------------------------------------------------------------------------

def test_evaluators_descriptors_conform_to_hypergan_plugin_contract():
    dinov2_fd = FakeDINOv2FD(reference_cache=False)
    d_fd = dinov2_fd.describe()
    assert d_fd['kind'] == 'scalar'
    assert d_fd['direction'] == 'minimize'
    assert d_fd['unit'] == 'FD'

    inc_suite = FakeInceptionSuite(reference_cache=False)
    d_inc = inc_suite.describe()
    assert d_inc['kind'] == 'scalars'
    assert set(d_inc['outputs']) == {'fid', 'precision', 'recall', 'density', 'coverage'}
    norm_inc = scalar_outputs(d_inc['outputs'])
    assert norm_inc['fid']['direction'] == 'minimize'
    for k in ('precision', 'recall', 'density', 'coverage'):
        assert norm_inc[k]['direction'] == 'maximize'

    dino_suite = FakeDINOv2Suite(reference_cache=False)
    d_dino = dino_suite.describe()
    assert d_dino['kind'] == 'scalars'
    assert set(d_dino['outputs']) == {'fd_dinov2', 'precision', 'recall', 'density', 'coverage'}
    norm_dino = scalar_outputs(d_dino['outputs'])
    assert norm_dino['fd_dinov2']['direction'] == 'minimize'
    for k in ('precision', 'recall', 'density', 'coverage'):
        assert norm_dino[k]['direction'] == 'maximize'


def test_dinov2_fd_matches_oracle_caches_reference_and_restores_tf32(tmp_path, monkeypatch):
    rng = np.random.default_rng(31)
    gen = _make_images(rng, 24) * 0.7 + 0.1
    ref = _make_images(rng, 24)

    metric = FakeDINOv2FD(reference_cache=str(tmp_path))
    FakeDINOv2FD.log.clear()
    first = metric.evaluate(
        batches=_make_batches(gen, ref, (10, 9, 5)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    assert len(FakeDINOv2FD.log) == 6  # 3 generated + 3 reference
    gen_feats = np.concatenate([f for _, f in FakeDINOv2FD.log[0::2]], axis=0).astype(np.float64)
    ref_feats = np.concatenate([f for _, f in FakeDINOv2FD.log[1::2]], axis=0).astype(np.float64)
    assert first == pytest.approx(_scipy_fd_oracle(gen_feats, ref_feats), rel=1e-9)

    [cache_file] = list(tmp_path.glob('*.fidref'))
    FakeDINOv2FD.log.clear()
    second = metric.evaluate(
        batches=_make_batches(gen, ref, (10, 9, 5)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    assert len(FakeDINOv2FD.log) == 3  # generated only; reference skipped!
    assert second == first

    # Corrupted cache file triggers miss and atomic rewrite
    damaged = bytearray(cache_file.read_bytes())
    damaged[-1] ^= 0xFF
    cache_file.write_bytes(bytes(damaged))
    FakeDINOv2FD.log.clear()
    repaired = metric.evaluate(
        batches=_make_batches(gen, ref, (10, 9, 5)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    assert len(FakeDINOv2FD.log) == 6
    assert repaired == first

    # TF32 restoration on error
    orig_mm, orig_cu = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        with pytest.raises(ValueError, match='complete declared'):
            metric.evaluate(batches=[], context={'sample_count': 24, 'reference_identity': None})
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_mm
        torch.backends.cudnn.allow_tf32 = orig_cu


@pytest.mark.parametrize('suite_cls,fd_key', [
    (FakeInceptionSuite, 'fid'),
    (FakeDINOv2Suite, 'fd_dinov2'),
])
def test_feature_suites_single_pass_oracle_and_cached_manifold(tmp_path, monkeypatch, suite_cls, fd_key):
    rng = np.random.default_rng(55)
    gen = _make_images(rng, 24) * 0.65 + 0.1
    ref = _make_images(rng, 24)

    suite = suite_cls(reference_cache=str(tmp_path), manifold_samples=16, pr_k=3, dc_k=5, knn_chunk_size=7)
    suite_cls.log.clear()
    first = suite.evaluate(
        batches=_make_batches(gen, ref, (9, 8, 7)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    # Single pass: exactly 3 generated + 3 reference extractor calls
    assert len(suite_cls.log) == 6
    gen_feats = np.concatenate([f for _, f in suite_cls.log[0::2]], axis=0)
    ref_feats = np.concatenate([f for _, f in suite_cls.log[1::2]], axis=0)

    # Verify Fréchet distance against full 24-sample SciPy oracle
    expected_fd = _scipy_fd_oracle(gen_feats.astype(np.float64), ref_feats.astype(np.float64))
    assert first[fd_key] == pytest.approx(expected_fd, rel=1e-7)

    # Verify PRDC against 16-sample deterministic subset oracle
    idx = _manifold_indices(24, 16)
    oracle = _textbook_prdc_oracle(ref_feats[idx], gen_feats[idx], pr_k=3, dc_k=5)
    for k in ('precision', 'recall', 'density', 'coverage'):
        assert first[k] == pytest.approx(oracle[k], abs=1e-12)

    [suite_cache] = list(tmp_path.glob('*.suiteref'))

    # Second run hits cache: skips reference extraction AND skips reference k-NN radii computation!
    suite_cls.log.clear()
    knn_calls = []
    real_knn = knn_radii

    def _spy_knn(features, ks, **kwargs):
        knn_calls.append(tuple(ks))
        return real_knn(features, ks, **kwargs)

    monkeypatch.setattr('hypergan.image_metrics.knn_radii', _spy_knn)
    second = suite.evaluate(
        batches=_make_batches(gen, ref, (9, 8, 7)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    assert len(suite_cls.log) == 3  # generated batches ONLY
    assert knn_calls == [(3,)]  # fake pr_k ONLY; reference (3, 5) skipped via cache!
    assert second == first

    # Different k or manifold_samples writes a separate cache entry (no stale reuse)
    suite_alt = suite_cls(reference_cache=str(tmp_path), manifold_samples=16, pr_k=2, dc_k=4, knn_chunk_size=7)
    suite_cls.log.clear()
    suite_alt.evaluate(
        batches=_make_batches(gen, ref, (9, 8, 7)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    assert len(suite_cls.log) == 6
    assert len(list(tmp_path.glob('*.suiteref'))) == 2

    # Corrupted .suiteref is rejected and cleanly recomputed
    bad = bytearray(suite_cache.read_bytes())
    bad[-1] ^= 0xAA
    suite_cache.write_bytes(bytes(bad))
    suite_cls.log.clear()
    third = suite.evaluate(
        batches=_make_batches(gen, ref, (9, 8, 7)),
        context={'sample_count': 24, 'reference_identity': _protocol()},
    )
    assert len(suite_cls.log) == 6
    assert third == first

    # Identical generated & reference gives FD ~ 0 and PRDC = 1.0
    ident = suite_cls(reference_cache=False, manifold_samples=16, pr_k=3, dc_k=5, knn_chunk_size=7).evaluate(
        batches=_make_batches(ref, ref, (12, 12)),
        context={'sample_count': 24, 'reference_identity': None},
    )
    assert abs(ident[fd_key]) < 1e-5
    assert ident['precision'] == pytest.approx(1.0)
    assert ident['recall'] == pytest.approx(1.0)
    assert ident['density'] == pytest.approx(1.0)
    assert ident['coverage'] == pytest.approx(1.0)


def test_suite_reference_cache_integrity_and_constructor_validation(tmp_path):
    path = tmp_path / 'entry.suiteref'
    rng = np.random.default_rng(19)
    feats = rng.normal(size=(10, 6)).astype(np.float32)
    ref_data = {
        'mu': feats.astype(np.float64).mean(0),
        'sigma': np.cov(feats.astype(np.float64), rowvar=False),
        'manifold_features': feats,
        'manifold_radii_pr': np.ones(10, dtype=np.float32),
        'manifold_radii_dc': np.full(10, 2.0, dtype=np.float32),
    }
    ident = {'kind': 'suite_test', 'pr_k': 3, 'dc_k': 5}
    write_suite_reference(path, ident, ref_data, count=20)
    loaded = read_suite_reference(path, ident, count=20, manifold_count=10)
    assert loaded is not None
    np.testing.assert_array_equal(loaded['manifold_features'], feats)
    assert read_suite_reference(path, ident, count=19, manifold_count=10) is None
    assert read_suite_reference(path, ident, count=20, manifold_count=9) is None
    assert read_suite_reference(path, dict(ident, pr_k=4), count=20, manifold_count=10) is None

    # Negative radii rejected even if file digest is valid
    write_suite_reference(path, ident, dict(ref_data, manifold_radii_pr=np.full(10, -1.0, dtype=np.float32)), count=20)
    assert read_suite_reference(path, ident, count=20, manifold_count=10) is None

    # Invalid constructor arguments on real classes
    with pytest.raises(ValueError, match='manifold_samples'):
        InceptionSuite('unused', manifold_samples=5, pr_k=3, dc_k=5)
    with pytest.raises(ValueError, match='positive integer'):
        DINOv2Suite('unused', pr_k=0)
    with pytest.raises(ValueError, match='reference_cache'):
        DINOv2FD('unused', reference_cache='   ')
    with pytest.raises(ValueError, match='reference_cache'):
        InceptionSuite('unused', reference_cache='')
    with pytest.raises(ValueError, match='reference_cache'):
        DINOv2Suite('unused', reference_cache='')


# ---------------------------------------------------------------------------
# 5. End-to-End Snapshot Worker & Standalone evaluate_bundle + Publication
# ---------------------------------------------------------------------------

def _make_folder(root, count):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new('RGB', (4, 4), ((i * 37) % 256, (i * 73) % 256, (i * 19) % 256)).save(root / f'{i:04d}.png')
    return root


def _make_snapshot_recipe(folder_root, sample_count, cache_dir, factory_ref, metric_id='suite'):
    config = resolve_config({})
    config['training']['device'] = 'cpu'
    config['prior']['args']['num_particles'] = 16
    config['components']['scaled'] = {
        'factory': f'{__name__}:_ScaleView',
        'inputs': {'x': 'batch.real'},
        'trainable': False,
    }
    config['objectives'] = [{'factory': 'mse', 'inputs': {'input': 'generated', 'target': 'components.scaled'}}]
    config['metrics']['custom'] = {
        metric_id: {
            'factory': factory_ref,
            'args': {'reference_cache': str(cache_dir), 'manifold_samples': 12, 'pr_k': 3, 'dc_k': 5},
            'mode': 'snapshot',
            'trigger': 'manual',
            'inputs': {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'},
            'evaluation': {
                'device': 'cpu',
                'sample_count': sample_count,
                'batch_size': 8,
                'seed': 91,
                'generated': 'components.scaled',
                'data': {
                    'factory': 'image_folder',
                    'args': {'root': str(folder_root), 'height': 4, 'width': 4, 'shuffle': False},
                },
            },
        }
    }
    return config_values(config)


def test_evaluate_snapshot_worker_and_catalog_for_suites(tmp_path):
    folder = _make_folder(tmp_path / 'imgs', 20)
    for metric_id, factory_cls, fd_key in (
        ('inc20', FakeInceptionSuite, 'fid'),
        ('dino20', FakeDINOv2Suite, 'fd_dinov2'),
    ):
        cfg = _make_snapshot_recipe(
            folder, 20, tmp_path / f'cache_{metric_id}', f'{__name__}:{factory_cls.__name__}', metric_id=metric_id
        )
        resolved = resolve_config(cfg)
        trainer = ReferenceTrainer(resolved)
        trainer.artifact_identity = {'run_id': 'run-suite', 'attempt_id': 'att-1'}
        snap = tmp_path / f'{metric_id}_snap.pt'
        torch.save(bundle_state(trainer, {'real': torch.zeros(2, 2)}), snap)

        spec = trainer.config['metrics']['custom'][metric_id]
        expected = _factory(spec['factory'], spec['args'])[1]
        resolved['_metric_runtime'] = {metric_id: expected}
        catalog = metric_catalog(resolved)
        expected_ids = {f'{metric_id}/{k}' for k in (fd_key, 'precision', 'recall', 'density', 'coverage')}
        assert expected_ids <= set(catalog['metrics'])

        factory_cls.log.clear()
        result = evaluate_snapshot(
            spec,
            expected,
            str(snap),
            hashlib.sha256(snap.read_bytes()).hexdigest(),
            trainer.artifact_identity,
        )
        assert set(result['values']) == {fd_key, 'precision', 'recall', 'density', 'coverage'}
        assert all(math.isfinite(v) for v in result['values'].values())

        # Second call hits cached .suiteref
        factory_cls.log.clear()
        again = evaluate_snapshot(
            spec,
            expected,
            str(snap),
            hashlib.sha256(snap.read_bytes()).hexdigest(),
            trainer.artifact_identity,
        )
        assert again['values'] == result['values']
        assert len(factory_cls.log) == 3  # ceil(20 / 8) generated batches only


def test_evaluate_bundle_json_receipt_and_run_publication(tmp_path, monkeypatch):
    folder = _make_folder(tmp_path / 'ref_imgs', 20)
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    atomic_json(run_dir / 'manifest.json', {'run_id': 'run-bundle', 'attempt_id': 'att-7', 'status': 'complete'})

    cfg = _make_snapshot_recipe(folder, 20, tmp_path / 'unused_cache', f'{__name__}:FakeInceptionSuite')
    trainer = ReferenceTrainer(resolve_config(cfg))
    trainer.step = 420
    trainer.artifact_identity = {'run_id': 'run-bundle', 'attempt_id': 'att-7'}
    save_bundle(run_dir, trainer, {'real': torch.zeros(2, 2)})

    receipt_path = tmp_path / 'receipt.json'
    cache_dir = tmp_path / 'ref_cache'
    FakeInceptionSuite.log.clear()
    FakeDINOv2Suite.log.clear()

    receipt = evaluate_bundle(
        run_dir / 'model.pt',
        data={'builtin': 'image_folder', 'root': str(folder), 'height': 4, 'width': 4},
        metrics={
            'inception': FakeInceptionSuite(reference_cache=str(cache_dir), manifold_samples=12, pr_k=3, dc_k=5),
            'dinov2': FakeDINOv2Suite(reference_cache=str(cache_dir), manifold_samples=12, pr_k=3, dc_k=5),
        },
        sample_count=20,
        batch_size=8,
        seed=100,
        device='cpu',
        generated_binding='components.scaled',
        output_json=receipt_path,
        publish_run_dir=run_dir,
    )

    assert receipt['step'] == 420
    assert receipt['sample_count'] == 20
    assert set(receipt['metrics']) == {'inception', 'dinov2'}
    assert set(receipt['metrics']['inception']['values']) == {'fid', 'precision', 'recall', 'density', 'coverage'}
    assert set(receipt['metrics']['dinov2']['values']) == {'fd_dinov2', 'precision', 'recall', 'density', 'coverage'}
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text(encoding='utf-8'))['bundle_sha256'] == receipt['bundle_sha256']

    # Verify all 10 scalar series were published atomically into the run history
    published = hg.evaluations(run_dir)
    expected_series = {
        'inception/fid', 'inception/precision', 'inception/recall', 'inception/density', 'inception/coverage',
        'dinov2/fd_dinov2', 'dinov2/precision', 'dinov2/recall', 'dinov2/density', 'dinov2/coverage',
    }
    assert expected_series <= set(published)
    for series in expected_series:
        assert published[series][0]['step'] == 420
        assert math.isfinite(published[series][0]['value'])

    # Re-evaluating and re-publishing is idempotent (and hits the reference cache!)
    FakeInceptionSuite.log.clear()
    FakeDINOv2Suite.log.clear()
    receipt2 = evaluate_bundle(
        run_dir,  # directory resolves to run_dir/model.pt automatically
        data={'builtin': 'image_folder', 'root': str(folder), 'height': 4, 'width': 4},
        metrics={
            'inception': FakeInceptionSuite(reference_cache=str(cache_dir), manifold_samples=12, pr_k=3, dc_k=5),
            'dinov2': FakeDINOv2Suite(reference_cache=str(cache_dir), manifold_samples=12, pr_k=3, dc_k=5),
        },
        sample_count=20,
        batch_size=8,
        seed=100,
        device='cpu',
        generated_binding='components.scaled',
        publish_run_dir=run_dir,
    )
    assert receipt2['metrics']['inception']['values'] == receipt['metrics']['inception']['values']
    assert receipt2['metrics']['dinov2']['values'] == receipt['metrics']['dinov2']['values']
    assert len(FakeInceptionSuite.log) == 3
    assert len(FakeDINOv2Suite.log) == 3

    # CLI argument parser delegates cleanly to evaluate_bundle
    captured = {}

    def _fake_eval(bundle, **kwargs):
        captured['bundle'] = bundle
        captured['kwargs'] = kwargs
        return {'ok': True}

    monkeypatch.setattr('hypergan.image_metrics.evaluate_bundle', _fake_eval)
    rc = image_metrics_cli_main([
        str(run_dir / 'model.pt'),
        '--metrics', 'inception', 'dinov2',
        '--samples', '16',
        '--batch-size', '4',
        '--seed', '7',
        '--device', 'cpu',
        '--data-folder', str(folder),
        '--width', '4',
        '--height', '4',
        '--reference-cache', 'false',
    ])
    assert rc == 0
    assert captured['bundle'] == run_dir / 'model.pt'
    assert captured['kwargs']['metrics'] == ['inception', 'dinov2']
    assert captured['kwargs']['sample_count'] == 16
    assert captured['kwargs']['reference_cache'] is False
    assert captured['kwargs']['data'] == {
        'factory': 'image_folder',
        'args': {'root': str(folder), 'shuffle': False, 'width': 4, 'height': 4},
    }


def test_toml_model_file_declares_dinov2_and_suites_without_args(tmp_path):
    """Verify hypergan-model/1 TOML configs declare DINOv2FD, InceptionSuite, and DINOv2Suite cleanly."""
    folder = _make_folder(tmp_path / 'data', 12)
    toml_text = f'''
format = "hypergan-model/1"
name = "metrics-contract-smoke"

[data]
builtin = "image_folder"
root = "{folder.as_posix()}"
width = 4
height = 4

[prior]
kind = "gaussian"
z_dim = 4

[networks.generator]
role = "generator"
inputs = {{ x = "latent" }}
input_shape = ["B", 4]
output_shape = ["B", 3, 4, 4]
source = """
linear(48)
tanh()
reshape(3, 4, 4)
"""

[networks.critic]
role = "critic"
inputs = {{ x = "candidate" }}
input_shape = ["B", 3, 4, 4]
output_shape = ["B", 1]
source = """
flatten()
linear(1)
"""

[[losses]]
type = "adversarial"
critic = "critic"
penalty = 1.0

[train]
steps = 2
batch_size = 4
device = "cpu"

[evaluations.inception50k]
factory = "hypergan.image_metrics:InceptionSuite"
device = "cpu"
samples = 12
batch_size = 4
seed = 100
data = "train"

[evaluations.dinov2_50k]
factory = "hypergan.image_metrics:DINOv2Suite"
device = "cpu"
samples = 12
batch_size = 4
seed = 100
data = "train"

[evaluations.dinov2_fd]
factory = "hypergan.image_metrics:DINOv2FD"
device = "cpu"
samples = 12
batch_size = 4
seed = 100
data = "train"
'''
    cfg_path = tmp_path / 'model.toml'
    cfg_path.write_text(toml_text, encoding='utf-8')
    lowered = hg.lower(cfg_path)
    resolved = resolve_config(lowered)
    custom = resolved['metrics']['custom']
    assert custom['inception50k']['factory'] == 'hypergan.image_metrics:InceptionSuite'
    assert custom['dinov2_50k']['factory'] == 'hypergan.image_metrics:DINOv2Suite'
    assert custom['dinov2_fd']['factory'] == 'hypergan.image_metrics:DINOv2FD'
    for name in ('inception50k', 'dinov2_50k', 'dinov2_fd'):
        assert custom[name]['mode'] == 'snapshot'
        assert custom[name]['args'] == {}
        assert custom[name]['inputs'] == {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'}
        assert custom[name]['evaluation']['data']['args']['shuffle'] is False


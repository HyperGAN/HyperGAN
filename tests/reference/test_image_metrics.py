"""Pinned evaluator preprocessing and streaming statistics versus direct references."""
import numpy as np
import pytest
import torch

from hypergan.image_metrics import INCEPTION_SHA256, _Moments, _checked_weights, _uint8_rgb


def test_streamed_sample_statistics_match_numpy_for_uneven_batches():
    values = np.random.default_rng(42).normal(size=(37, 9)) + 1e5
    moments = _Moments()
    for batch in (values[:1], values[1:12], values[12:]):
        moments.add(batch)
    result = moments.statistics()
    np.testing.assert_allclose(result['mu'], values.mean(0), rtol=0, atol=3e-11)
    np.testing.assert_allclose(result['sigma'], np.cov(values, rowvar=False), rtol=0, atol=2e-11)
    with pytest.raises(ValueError, match='at least two'):
        _Moments().statistics()
    with pytest.raises(ValueError, match='finite'):
        moments.add([[float('nan')]])


def test_uint8_rounding_and_clamping_match_source_protocol():
    values = torch.tensor([-2., -1., 0., 1., 2.]).reshape(1, 1, 1, 5).repeat(1, 3, 1, 1)
    result = _uint8_rgb(values)
    assert result.dtype == torch.uint8
    assert result[0, 0, 0].tolist() == [0, 0, 128, 255, 255]
    with pytest.raises(ValueError, match='RGB'):
        _uint8_rgb(values[:, :1])
    with pytest.raises(ValueError, match='finite'):
        _uint8_rgb(values * float('nan'))


def test_weights_fail_before_extractor_import_or_download(tmp_path):
    path = tmp_path / 'weights.pth'
    with pytest.raises(ValueError, match='missing'):
        _checked_weights(path, INCEPTION_SHA256)
    path.write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='does not match'):
        _checked_weights(path, INCEPTION_SHA256)
    with pytest.raises(ValueError, match='pinned'):
        _checked_weights(path, '0' * 64)


def test_inception_fid_evaluation_pipeline(tmp_path, monkeypatch):
    """Verify InceptionFID end-to-end evaluation pipeline and Scipy 1.18+ / torch-fidelity 0.4.0 compatibility."""
    import hashlib
    from pathlib import Path
    from torch import nn
    from hypergan.image_metrics import InceptionFID

    fake_weights = tmp_path / "fake_inception.pth"
    fake_weights.write_bytes(b"dummy inception weights")
    fake_sha = hashlib.sha256(fake_weights.read_bytes()).hexdigest()

    class DummyInception(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.eval()

        def forward(self, x):
            b = x.shape[0]
            v = x.float().view(b, -1)[:, :16]
            features = torch.zeros(b, 2048)
            features[:, :16] = v
            return (features,)

    monkeypatch.setattr('hypergan.image_metrics._checked_weights', lambda path, expected: Path(path))
    monkeypatch.setattr('torch_fidelity.feature_extractor_inceptionv3.FeatureExtractorInceptionV3', DummyInception)

    fid = InceptionFID(fake_weights, weights_sha256=fake_sha)
    assert fid.describe()['unit'] == 'FID'
    assert 'torch-fidelity 0.4.0' in fid.describe()['description']

    batches = [
        {'generated': torch.randn(4, 3, 32, 32), 'reference': torch.randn(4, 3, 32, 32) + 0.5},
        {'generated': torch.randn(4, 3, 32, 32), 'reference': torch.randn(4, 3, 32, 32) + 0.5},
    ]
    context = {'sample_count': 8}

    score = fid.evaluate(batches=batches, context=context)
    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score > 0.0


def test_inception_fid_real_weights_if_available():
    """Verify InceptionFID with real cached weights if present in environment."""
    from pathlib import Path
    from hypergan.image_metrics import InceptionFID

    weights = Path.home() / ".cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
    if not weights.is_file():
        pytest.skip("Inception weights not present in local cache")

    fid = InceptionFID(weights)
    batch = {
        'generated': torch.randn(4, 3, 32, 32),
        'reference': torch.randn(4, 3, 32, 32),
    }
    context = {'sample_count': 4}
    score = fid.evaluate(batches=[batch], context=context)
    assert isinstance(score, float)
    assert np.isfinite(score)


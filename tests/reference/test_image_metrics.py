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

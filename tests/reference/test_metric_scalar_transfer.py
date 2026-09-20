"""Scalar observation transport preserves values without scalar host reads."""
import pytest
import torch

from hypergan.training import _pack_metric_scalars


def test_metric_pack_preserves_mixed_dtypes_shapes_and_detaches():
    values = [
        torch.tensor(0.1, dtype=torch.float32, requires_grad=True),
        torch.tensor([[1.0000000001]], dtype=torch.float64, requires_grad=True),
        torch.tensor([16777217], dtype=torch.int64),
        torch.tensor(0.3, dtype=torch.float16),
        torch.tensor(0.7, dtype=torch.bfloat16),
    ]
    expected = [float(value.detach()) for value in values]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        packed = _pack_metric_scalars(values, torch.device('cpu'))
    assert packed.tolist() == expected
    assert packed.dtype == torch.float64
    assert packed.shape == (len(values),)
    assert not packed.requires_grad and packed.grad_fn is None
    assert all(value.grad is None for value in values)
    # A scalar read here synchronizes CUDA. Check the transport independently
    # of the necessary finite-loss/gradient checks in the numerical update.
    assert not any(event.key == 'aten::_local_scalar_dense' for event in profile.key_averages())


def test_metric_pack_rejects_non_scalar_values():
    with pytest.raises(RuntimeError, match='shape'):
        _pack_metric_scalars([torch.ones(2)], torch.device('cpu'))

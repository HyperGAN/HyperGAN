"""Scalar observation transport preserves values without scalar host reads."""
import pytest
import torch

from hypergan.training import _MetricScalarTransfer


def test_metric_transfer_preserves_mixed_dtypes_shapes_without_initializing_cuda(monkeypatch):
    values = [
        torch.tensor(0.1, dtype=torch.float32, requires_grad=True),
        torch.tensor([[1.0000000001]], dtype=torch.float64, requires_grad=True),
        torch.tensor([16777217], dtype=torch.int64),
        torch.tensor(0.3, dtype=torch.float16),
        torch.tensor(0.7, dtype=torch.bfloat16),
    ]
    expected = [float(value.detach()) for value in values]
    def no_cuda(*args, **kwargs):
        raise AssertionError('CPU scalar observations must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', no_cuda)
    transfer = _MetricScalarTransfer(torch.device('cpu'))
    assert transfer(values) == expected
    assert all(value.grad is None for value in values)
    assert transfer._groups == []


def test_metric_transfer_rejects_non_scalar_values():
    with pytest.raises(ValueError, match='one element'):
        _MetricScalarTransfer(torch.device('cpu'))([torch.ones(2)])

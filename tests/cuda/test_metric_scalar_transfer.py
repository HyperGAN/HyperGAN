"""Explicit single-GPU scalar staging qualification; CPU CI does not run this."""
import torch

from hypergan.training import _MetricScalarTransfer


def test_scalar_staging_exact_values_one_fence_and_no_gpu_kernels():
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    transfer = _MetricScalarTransfer(device)
    values = [torch.tensor(.1, device=device, requires_grad=True),
              torch.tensor([[1.0000000001]], device=device, dtype=torch.float64),
              torch.tensor([16777217], device=device, dtype=torch.int64),
              torch.tensor(.3, device=device, dtype=torch.float16),
              torch.tensor(.7, device=device, dtype=torch.bfloat16),
              torch.tensor(.4)]
    expected = [float(value.detach()) for value in values]
    assert transfer(values) == expected
    pointers = [buffer.data_ptr() for _, buffer, _ in transfer._groups]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        assert transfer(values) == expected
    assert pointers == [buffer.data_ptr() for _, buffer, _ in transfer._groups]
    events = {event.key: event.count for event in profile.key_averages()}
    assert events.get('cudaStreamSynchronize') == 1
    assert events.get('aten::_local_scalar_dense', 0) == 0
    assert events.get('cudaLaunchKernel', 0) == 0
    assert all(buffer.is_pinned() and not buffer.requires_grad for _, buffer, _ in transfer._groups)
    assert all(value.grad is None for value in values)
    # A changed custom-objective dtype rebuilds the bounded cache safely.
    assert transfer([values[0].double()]) == [expected[0]]
    assert len(transfer._groups) == 1 and transfer._groups[0][1].numel() == 1

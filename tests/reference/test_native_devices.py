"""Native device fallback warns; callers may explicitly require exact hardware."""
import pytest
import torch
from hypergan.checkpoint_compatibility import HyperGANHardwareWarning
from hypergan.checkpoints import read_checkpoint
from hypergan.config import write_default
from hypergan.recipes import execution_device
from hypergan.training import train


def test_unavailable_cuda_warns_and_trains_on_cpu(tmp_path, monkeypatch):
    config=write_default(tmp_path/'project',device='cuda', defaults='hypergan')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.warns(HyperGANHardwareWarning, match='CUDA is unavailable.*falling back to cpu'):
        result = train(config,tmp_path/'run')
    assert result['status'] == 'complete' and result['steps'] == 5
    _, metadata, _ = read_checkpoint(tmp_path/'run')
    assert metadata['runtime']['device'] == 'cpu'


def test_invalid_cuda_index_warns_and_resolves_to_visible_device(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('device resolution initialized CUDA'))
    with pytest.warns(HyperGANHardwareWarning, match='index 9 is unavailable; falling back to cuda:0'):
        assert execution_device('cuda:9') == torch.device('cuda:0')


@pytest.mark.parametrize('available,count,requested,message', [
    (False, 0, 'cuda', 'CUDA was requested but is unavailable'),
    (True, 2, 'cuda:9', 'index 9 is unavailable'),
])
def test_explicit_no_fallback_refuses_unavailable_cuda(monkeypatch, available, count, requested, message):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: count)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('device resolution initialized CUDA'))
    with pytest.raises(ValueError, match=message):
        execution_device(requested, fallback=False)

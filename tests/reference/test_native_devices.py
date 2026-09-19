"""Explicit unavailable native devices fail before creating a run."""
import pytest
import torch
from hypergan.config import write_default
from hypergan.training import train


def test_unavailable_cuda_does_not_fall_back_or_create_run(tmp_path, monkeypatch):
    config=write_default(tmp_path/'project',device='cuda')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CUDA was requested but is unavailable'):
        train(config,tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_invalid_cuda_index_does_not_create_run(tmp_path, monkeypatch):
    config=write_default(tmp_path/'project',device='cuda:9')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    with pytest.raises(ValueError,match='index 9 is unavailable'):
        train(config,tmp_path/'run')
    assert not (tmp_path/'run').exists()

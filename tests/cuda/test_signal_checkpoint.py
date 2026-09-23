"""Explicit GPU qualification of read-only checkpoint diagnostic device remapping."""
import hashlib

import torch

from hypergan.config import write_default
from hypergan.execution import train
from hypergan.signal_diagnostic import diagnose


def _files(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def test_cpu_checkpoint_can_be_diagnosed_on_cuda_without_touching_source(tmp_path):
    assert torch.cuda.is_available(), 'Checkpoint device remapping requires a CUDA device'
    config = write_default(tmp_path / 'project', device='cpu')
    root = tmp_path / 'run'
    train(config, root, steps=2, stop_after_steps=1)
    before = _files(root)
    result = diagnose(root, device='cuda')
    protocol = result['checkpoint']['diagnostic_protocol']
    assert protocol['source_runtime']['device'] == 'cpu'
    assert torch.device(protocol['evaluation_runtime']['device']).type == 'cuda'
    assert protocol['named_rng_streams']['prior'] == 'reset-from-config-backend-incompatible'
    assert protocol['global_cuda_rng'] == 'reset-from-config-backend-incompatible'
    assert result['state_verification']['optimizer_steps'] == 0
    assert _files(root) == before


def test_cuda_checkpoint_can_be_diagnosed_on_cpu_without_touching_source(tmp_path):
    assert torch.cuda.is_available(), 'Checkpoint device remapping requires a CUDA device'
    config = write_default(tmp_path / 'project', device='cuda')
    root = tmp_path / 'run'
    train(config, root, steps=2, stop_after_steps=1)
    before = _files(root)
    result = diagnose(root, device='cpu')
    protocol = result['checkpoint']['diagnostic_protocol']
    assert torch.device(protocol['source_runtime']['device']).type == 'cuda'
    assert protocol['evaluation_runtime']['device'] == 'cpu'
    assert protocol['named_rng_streams']['prior'] == 'reset-from-config-backend-incompatible'
    assert result['step'] == 1
    assert _files(root) == before

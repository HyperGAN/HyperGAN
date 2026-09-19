"""Explicit CUDA metric parity: native execution and complete two-GPU groups."""
import importlib.util
from pathlib import Path

import torch


def fixture(name):
    path = Path(__file__).parents[1] / 'reference' / name
    spec = importlib.util.spec_from_file_location('_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_replicated_metrics_on_off_and_revision_resume(tmp_path):
    assert torch.cuda.device_count() >= 2, 'This acceptance requires two local GPUs'
    fixture('test_metrics_replicated.py').verify_replicated_metrics(tmp_path, 'cuda')


def test_cuda_native_metrics_on_off_and_revision_resume(tmp_path):
    assert torch.cuda.is_available(), 'This acceptance requires CUDA'
    module = fixture('test_metrics_runtime.py')
    original = module.config
    def gpu_config(*args, **kwargs):
        path = original(*args, **kwargs)
        path.write_text(path.read_text().replace('device = "cpu"', 'device = "cuda:0"'))
        return path
    module.config = gpu_config
    module.test_metrics_on_off_cadence_and_changed_resume_are_numerically_identical(tmp_path)

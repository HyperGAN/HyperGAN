"""Explicit two-local-GPU qualification of the installed public CLI."""
import importlib.util
from pathlib import Path

import torch


_spec = importlib.util.spec_from_file_location(
    '_public_cpu_acceptance', Path(__file__).parents[1] / 'reference/test_public_execution_acceptance.py')
_public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_public)


def test_public_two_gpu_train_stop_resume_earlier_snapshot_and_observation(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Public NCCL acceptance requires two GPUs'
    _public.recover_public_job(tmp_path, device='cuda')

"""Explicit two-local-GPU qualification of the installed public CLI."""
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.multi_gpu


_spec = importlib.util.spec_from_file_location(
    '_public_cpu_acceptance', Path(__file__).parents[1] / 'reference/test_public_execution_acceptance.py')
_public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_public)


def test_public_two_gpu_train_stop_resume_earlier_snapshot_and_observation(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Public NCCL acceptance requires two GPUs'
    _public.recover_public_job(tmp_path, device='cuda')


def test_public_device_list_selects_two_gpus_and_resume_infers_profile(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Device-list acceptance requires two GPUs'
    config, _ = _public.setup(tmp_path, device='cuda')
    run = tmp_path / 'device-list-run'
    stopped = _public.result(_public.cli(
        tmp_path, tmp_path / 'stop-pids', 'train', config, '--run-dir', run,
        '--devices', '0,1', '--stop-after-steps', 1, '--checkpoint-every', 1,
        '--no-previews', '--no-server')[0])
    assert stopped['status'] == 'stopped' and stopped['last_durable_step'] == 1
    assert stopped['execution']['name'] == 'cuda-replicated-nccl'
    assert stopped['execution']['world_size'] == 2
    assert stopped['execution']['accumulation_steps'] == 1
    resumed = _public.result(_public.cli(
        tmp_path, tmp_path / 'resume-pids', 'resume', run, '--stop-after-steps', 1,
        '--no-previews', '--no-server')[0])
    assert resumed['status'] == 'stopped' and resumed['last_durable_step'] == 2
    assert resumed['attempt_index'] == 2
    assert resumed['execution'] == stopped['execution']

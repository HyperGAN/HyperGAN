"""Explicit local CUDA custom scalar and immutable snapshot acceptance."""
import importlib.util
from pathlib import Path

import torch


def module(name):
    path=Path(__file__).parents[1]/'reference'/name
    spec=importlib.util.spec_from_file_location('_cuda_'+path.stem,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cuda_custom_scalar_isolation(tmp_path):
    assert torch.cuda.is_available(), 'CUDA acceptance requires a local GPU'
    fixture=module('test_custom_metrics.py')
    fixture.PLUGIN=fixture.PLUGIN.replace('        torch.rand(9)', '        torch.rand(9)\n        torch.rand(9,device="cuda:0")')
    setup=fixture.setup
    def gpu_setup(*args,**kwargs):
        kwargs['timeout']=15
        driver=setup(*args,**kwargs)
        for name in ('base.toml','custom.toml'):
            path=tmp_path/name
            path.write_text(path.read_text().replace('device = "cpu"','device = "cuda:0"'))
        return driver
    fixture.setup=gpu_setup
    fixture.test_supervised_scalar_cadence_and_rng_do_not_change_complete_state(tmp_path)


def test_cuda_manual_snapshot_repeatability_and_training_isolation(tmp_path):
    assert torch.cuda.is_available(), 'CUDA acceptance requires a local GPU'
    fixture=module('test_metric_evaluation.py')
    setup=fixture.setup
    def gpu_setup(*args,**kwargs):
        driver=setup(*args,**kwargs)
        for name in ('config.toml','partial.toml'):
            path=tmp_path/name
            path.write_text(path.read_text().replace('device = "cpu"','device = "cuda:0"'))
        return driver
    fixture.setup=gpu_setup
    fixture.test_snapshot_is_pinned_repeatable_and_publishes_independent_late_stream(tmp_path)


def test_two_gpu_custom_scalar_complete_state_parity(tmp_path):
    assert torch.cuda.device_count() >= 2, 'This acceptance requires two local GPUs'
    fixture=module('test_custom_metrics.py')
    run=fixture.run
    def nccl_run(driver,mode):
        driver.write_text(driver.read_text().replace('cpu-replicated-gloo','cuda-replicated-nccl'))
        for name in ('base.toml','custom.toml'):
            path=tmp_path/name
            path.write_text(path.read_text().replace('device = "cpu"','device = "cuda"'))
        return run(driver,mode)
    fixture.run=nccl_run
    fixture.test_replicated_custom_scalar_keeps_parent_torch_free_and_state_exact(tmp_path)

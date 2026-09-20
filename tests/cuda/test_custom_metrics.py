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
    fixture.PLUGIN=fixture.PLUGIN.replace('        torch.rand(9)', '        torch.rand(9)\n        assert not torch.cuda.is_available(), "Scalar observers must not compete for the training GPU"')
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


def test_cuda_manual_snapshot_repeatability_and_training_isolation(tmp_path, monkeypatch):
    # Inherited by the fresh training/evaluation processes before CUDA initializes.
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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


def test_cuda_histogram_uses_cpu_aggregation_under_strict_determinism(monkeypatch):
    from hypergan.metric_examples import ColorHistogramDifference
    assert torch.cuda.is_available(), 'CUDA acceptance requires a local GPU'
    original = torch.histc
    calls = []

    def cpu_histogram(value, **kwargs):
        assert value.device.type == 'cpu'
        calls.append(value.numel())
        return original(value, **kwargs)

    monkeypatch.setattr(torch, 'histc', cpu_histogram)
    enabled, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        batches = [{'generated': torch.zeros(1,3,2,2,device='cuda'),
                    'reference': torch.ones(1,3,2,2,device='cuda')}]
        result = ColorHistogramDifference(bins=2,low=0,high=1).evaluate(batches=batches,context={})
        assert result == {'edges':[0.0,0.5,1.0], 'counts':[1.0,1.0]}
        assert calls == [12,12]
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


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

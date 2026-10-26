"""Runtime and source metadata record measured values, not constants."""
import json

import pytest
import torch

from tests.config_fixtures import write_default


@pytest.fixture
def threads():
    previous = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_runtime_info_reads_threads_dtype_and_world_size(threads, monkeypatch):
    from hypergan.training import runtime_info

    torch.set_num_threads(3)
    info = runtime_info('cpu')
    assert info['threads'] == 3
    assert (info['dtype'], info['default_dtype'], info['world_size']) == ('float32', 'torch.float32', 1)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        info = runtime_info('cpu')
    finally:
        torch.set_default_dtype(previous)
    assert (info['dtype'], info['default_dtype']) == ('float64', 'torch.float64')
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
    assert runtime_info('cpu')['world_size'] == 4


def test_environment_records_the_pinned_training_threads(tmp_path, threads):
    from hypergan.config import load_config
    from hypergan.single_execution import SingleProcessExecution

    torch.set_num_threads(3)
    execution = SingleProcessExecution(load_config(write_default(tmp_path / 'config.toml', device='cpu', defaults='hypergan')))
    try:
        assert execution.environment()['runtime']['threads'] == 1
    finally:
        execution.shutdown()
    assert torch.get_num_threads() == 3
    # A closed execution leaves the caller's thread count alone.
    execution.environment()
    assert torch.get_num_threads() == 3


def test_resume_compares_the_pinned_thread_count(tmp_path, threads):
    from hypergan.checkpoints import read_checkpoint
    from hypergan.training import resume, train

    config = write_default(tmp_path / 'config.toml', device='cpu', defaults='hypergan')
    run = tmp_path / 'run'
    torch.set_num_threads(3)
    train(config, run, steps=2, stop_after_steps=1)
    assert resume(run)['status'] == 'complete'
    _, checkpoint, _ = read_checkpoint(run)
    assert checkpoint['runtime']['threads'] == 1
    assert torch.get_num_threads() == 3


def test_source_info_names_the_installed_particlegan_and_recipe():
    from hypergan.training import _version, source_info

    plain = source_info({})
    assert plain['integration_reference'] == {
        'repository': 'https://github.com/255BITS/ParticleGAN', 'version': _version('particlegan'), 'preset': None}
    recipe = source_info({'recipe': {'name': 'cifar'}})
    assert recipe['integration_reference']['preset'] == 'cifar'
    assert recipe['particlegan_distribution_version'] == _version('particlegan')


@pytest.mark.parametrize('direct_url,commit', [
    ({'url': 'https://github.com/255BITS/ParticleGAN', 'vcs_info': {'vcs': 'git', 'commit_id': 'c' * 40}}, 'c' * 40),
    ({'url': 'file:///src/ParticleGAN', 'dir_info': {'editable': True}}, None),
    ({'vcs_info': 'not a table'}, None),
    (None, None),
])
def test_particlegan_commit_comes_from_installer_metadata(monkeypatch, direct_url, commit):
    import importlib.metadata
    from hypergan import training

    class Distribution:
        def read_text(self, name):
            assert name == 'direct_url.json'
            return None if direct_url is None else json.dumps(direct_url)

    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: Distribution())
    assert training._distribution_commit('particlegan') == commit

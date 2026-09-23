"""Native-loop parity and non-interference checks for the research adapter."""
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('depth_toy', Path(__file__).resolve().parents[2] / 'research/depth_toy/run.py')
source = Path(os.environ.get('PARTICLEGAN_ROOT', '/home/martyn/dev/ParticleGAN'))
pytestmark = pytest.mark.skipif(not (source / 'examples/100gaussians.py').exists(),
                                reason='requires local ParticleGAN research source')


@pytest.fixture(scope='module')
def api():
    torch.set_num_threads(1)
    return runner.load_source(source)


def assert_models_equal(left, right):
    for key in ('G', 'D', 'prior', 'ema_G', 'ema_prior'):
        assert runner.digest(left[key].state_dict()) == runner.digest(right[key].state_dict()), key


def test_native_training_parity(api, monkeypatch, tmp_path):
    # The native example imports plotting eagerly; all rendering is disabled.
    monkeypatch.setitem(sys.modules, 'matplotlib', types.ModuleType('matplotlib'))
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', types.ModuleType('matplotlib.pyplot'))
    native = load('native_100gaussians', source / 'examples/100gaussians.py')
    monkeypatch.setattr(native, 'save_fake_scatter', lambda *a, **kw: None)
    monkeypatch.setattr(native, 'mode_coverage', lambda *a, **kw: (0, 0.))
    expected = native.train(epochs=1, steps_per_epoch=4, device_str='cpu',
                            out_dir=str(tmp_path), return_details=True)
    _, actual = runner.train_case(*api, depth=3, steps=4, diagnostics=False, progress=False)
    assert_models_equal(expected, actual)


def test_diagnostics_do_not_change_training(api):
    _, plain = runner.train_case(*api, depth=3, steps=4, diagnostics=False, progress=False)
    measured, instrumented = runner.train_case(*api, depth=3, steps=4, diagnostics=True, progress=False)
    assert_models_equal(plain, instrumented)
    assert measured['records'][-1]['update']['layers']
    assert measured['records'][-1]['update']['g_only']['gain'] > 0


def test_depth_changes_preserve_shared_initialization(api):
    cases = [runner.initialize(*api, depth=depth, steps=7000) for depth in (3, 8, 16)]
    for key in ('critic', 'prior', 'shared_generator'):
        assert len({case[-1][key] for case in cases}) == 1
    for index in range(8):
        assert runner.digest(cases[1][1].net[2 * index].state_dict()) == runner.digest(cases[2][1].net[2 * index].state_dict())


def test_movement_distinguishes_shared_and_differentiated_motion():
    request = torch.tensor([[1., 0.], [-1., 0.]])
    actual = torch.tensor([[1., 0.], [1., 0.]])
    row = runner.movement(request, actual)
    assert row['requested']['shared_energy_fraction'] == 0
    assert row['actual']['shared_energy_fraction'] == 1
    assert row['cosine'] == 0
    assert runner.movement(request * 0, actual * 0)['gain'] is None


def test_distribution_coverage_is_not_just_spread():
    centers = torch.cartesian_prod(torch.arange(10) - 4.5, torch.arange(10) - 4.5)
    real = centers.repeat(10, 1)
    good = runner.distribution(real, real)
    collapsed = runner.distribution(real[:1].expand_as(real), real)
    assert good['modes'] == 100 and good['hq'] == 1 and good['sliced_w1'] == 0
    assert collapsed['modes'] == 1 and collapsed['hq'] == 1
    assert collapsed['between_sample_rms'] == 0 and collapsed['sliced_w1'] > 0

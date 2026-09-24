"""Explicit MoG scales avoid calibration while legacy recipes retain their meaning."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from particlegan import MoGParticlePrior, calibrate_mog_sigma

from hypergan.config import resolve_config
from hypergan.recipes import make_prior


@pytest.mark.parametrize('sigma', [0.0, 0.2])
@pytest.mark.parametrize('setting', ['fixed_sigma', 'sigma'])
def test_explicit_sigma_never_calibrates_and_preserves_rng(monkeypatch, sigma, setting):
    def forbidden(*args):
        pytest.fail('Explicit sigma triggered nearest-neighbor calibration')
    monkeypatch.setattr('particlegan.calibrate_mog_sigma', forbidden)
    spec = {'kind': 'mog', 'initialization_device': 'cpu', 'initialization_seed': 71,
            'args': {'num_particles': 12, 'z_dim': 4, 'sigma_rel': .025}}
    (spec if setting == 'fixed_sigma' else spec['args'])[setting] = sigma
    original = deepcopy(spec)
    before = torch.random.get_rng_state()
    actual = make_prior(spec, device='cpu')
    expected = MoGParticlePrior(num_particles=12, z_dim=4, sigma=sigma,
                                generator=torch.Generator().manual_seed(71))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert spec == original
    assert torch.equal(actual.z, expected.z)
    assert actual.sigma.item() == expected.sigma.item()
    rngs = [torch.Generator().manual_seed(73) for _ in range(2)]
    left, right = actual.sample(8, generator=rngs[0]), expected.sample(8, generator=rngs[1])
    assert all(torch.equal(a, b) for a, b in zip(left, right))
    assert torch.equal(rngs[0].get_state(), rngs[1].get_state())


@pytest.mark.parametrize('relative', [None, .03, 0.0])
def test_legacy_relative_recipe_calibrates_existing_centers(relative):
    spec = {'kind': 'mog', 'initialization_seed': 71,
            'args': {'num_particles': 12, 'z_dim': 4}}
    if relative is not None:
        spec['args']['sigma_rel'] = relative
    actual = make_prior(spec, device='cpu')
    original = MoGParticlePrior(num_particles=12, z_dim=4, sigma=0,
                                generator=torch.Generator().manual_seed(71))
    sigma, d0 = calibrate_mog_sigma(original.means(), .025 if relative is None else relative)
    assert torch.equal(actual.z, original.z)
    assert torch.equal(actual.sigma, sigma)
    assert torch.equal(actual.d0, d0)
    assert actual.sigma_rel == (.025 if relative is None else relative)
    assert actual._noise_enabled == bool(sigma > 0)


@pytest.mark.parametrize('args', [{'sigma': -1}, {'sigma': True}, {'sigma': float('inf')},
                                  {'sigma': None}, {'sigma_rel': -1}, {'sigma_rel': True}])
def test_invalid_noise_config_rejected_before_construction(args):
    with pytest.raises(ValueError, match='sigma|finite'):
        resolve_config({'prior': {'kind': 'mog', 'args': {'num_particles': 12, 'z_dim': 4, **args}}})


def test_ambiguous_explicit_scale_rejected():
    spec = {'kind': 'mog', 'fixed_sigma': .2,
            'args': {'num_particles': 12, 'z_dim': 4, 'sigma': .3}}
    with pytest.raises(ValueError, match='not both'):
        resolve_config({'prior': spec})
    with pytest.raises(ValueError, match='not both'):
        make_prior(spec, device='cpu')


def test_audited_050_checkpoint_metadata_uses_qualified_resume(tmp_path):
    from hypergan.checkpoint_compatibility import PARTICLEGAN_060_MIGRATION
    from hypergan.checkpoints import read_checkpoint
    from hypergan.config import write_default
    from hypergan.training import resume, train
    from tests.reference.test_recovery import equal

    config = write_default(tmp_path / 'config', device='cpu')
    config.write_text(config.read_text().replace('kind = "particles"', 'kind = "mog"\nfixed_sigma = 0.2')
                      .replace('num_particles = 20000', 'num_particles = 12'))
    train(config, tmp_path / 'full')
    stopped = train(config, tmp_path / 'split', stop_after_steps=2)
    path = Path(stopped['checkpoint_path']) / 'manifest.json'
    metadata = json.loads(path.read_text())
    metadata['runtime']['particlegan'] = '0.5.0'
    for key, (old, _) in PARTICLEGAN_060_MIGRATION.items():
        metadata['implementation'][key] = old
    path.write_text(json.dumps(metadata))
    with pytest.warns(RuntimeWarning, match='audited 0.6.0'):
        finished = resume(tmp_path / 'split')
    assert any('audited 0.6.0' in warning for warning in finished['resume_warnings'])
    equal(read_checkpoint(tmp_path / 'full')[2], read_checkpoint(tmp_path / 'split')[2])

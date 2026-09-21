"""Native local-checkpoint providers: integrity, ownership and derivatives."""
import hashlib

import pytest
import torch
from hndl import HNDLError, Registry

from hypergan.hndl_networks import build_network
from hypergan.pretrained_providers import register_providers


@pytest.fixture
def resnet_artifact(tmp_path):
    # This is the external checkpoint architecture, not a HyperGAN network.
    from torchvision.models import resnet18
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(183)
        external = resnet18(weights=None)
    path = tmp_path / 'resnet18.pth'
    torch.save(external.state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_resnet_provider_frozen_eval_and_two_input_derivatives(resnet_artifact):
    path, digest = resnet_artifact
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = (f'pretrained({str(path)!r}, provider="torchvision_resnet18", '
                  f'sha256="{digest}", layer="layer3", trainable=False, name="features")\n'
                  'global_avg_pool()\nlinear(1, name="head")')
        model = build_network(source, input_shape=('B', 3, 32, 32), output_shape=('B', 1))
        model.train()
        backbone = model['features'].model
        assert all(not module.training for module in backbone.modules())
        assert all(not parameter.requires_grad for parameter in backbone.parameters())
        assert all(not module.inplace for module in backbone.modules() if isinstance(module, torch.nn.ReLU))
        frozen = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
        candidate = torch.randn(2, 3, 32, 32, requires_grad=True)
        score = model(candidate)
        first, = torch.autograd.grad(score.square().sum(), candidate, create_graph=True)
        second, = torch.autograd.grad(first.square().sum(), candidate, retain_graph=True)
        assert first.abs().sum() > 0 and torch.isfinite(first).all()
        assert second.abs().sum() > 0 and torch.isfinite(second).all()
        (score.square().mean() + first.square().mean()).backward()
        assert model['head'].weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in backbone.parameters())
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        # HyperGAN's G phase freezes D temporarily and restores each flag,
        # rather than enabling every D parameter again afterward.
        flags = [parameter.requires_grad for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        generated = torch.randn(2, 3, 32, 32, requires_grad=True)
        model(generated).square().mean().backward()
        assert generated.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in model.parameters())
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        assert model['head'].weight.requires_grad
        assert all(not parameter.requires_grad for parameter in backbone.parameters())
        assert all(not module.training for module in backbone.modules())
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        assert model['features'].source.config['sha256'] == digest
    finally:
        torch.set_num_threads(previous_threads)


def test_bad_checksum_rejected_before_provider_construction(tmp_path, monkeypatch):
    import hypergan.pretrained_providers as providers
    path = tmp_path / 'bad.pth'
    path.write_bytes(b'not a checkpoint')

    def forbidden(*args, **kwargs):
        pytest.fail('Checksum validation must precede model construction and downloads')

    monkeypatch.setattr(providers, '_torchvision_resnet18', forbidden)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    source = (f'pretrained({str(path)!r}, provider="torchvision_resnet18", '
              f'sha256="{"0" * 64}", layer="layer3")')
    with pytest.raises(HNDLError, match='(?i)sha256|checksum'):
        build_network(source, input_shape=('B', 3, 32, 32), output_shape=('B', 256, 2, 2))


def test_provider_registration_is_local_and_preserves_explicit_builders():
    first, second = Registry.builtins(), Registry.builtins()
    register_providers(first)
    assert 'torchvision_resnet18' in first.pretrained_providers
    assert 'torchvision_resnet18' not in second.pretrained_providers
    builder = lambda: None
    second.pretrained_provider('torchvision_resnet18', builder)
    assert register_providers(second).pretrained_builder('torchvision_resnet18') is builder


def test_ordinary_network_does_not_import_torchvision(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == 'torchvision' or name.startswith('torchvision.'):
            pytest.fail('An ordinary HNDL network must not require torchvision')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked_import)
    model = build_network('linear(1)', input_shape=('B', 4), output_shape=('B', 1))
    assert model(torch.ones(2, 4)).shape == (2, 1)

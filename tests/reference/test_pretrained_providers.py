"""Native local-checkpoint providers: integrity, ownership and derivatives."""
import hashlib
from pathlib import Path

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


@pytest.fixture
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def multiscale_resnet_source(artifact):
    path, digest = artifact
    return (f'f1, f2, f3 = pretrained({str(path)!r}, provider="torchvision_resnet18", '
            f'sha256="{digest}", layers=("layer1", "layer2", "layer3"), '
            'trainable=False, name="features")')


@pytest.mark.parametrize('side', [32, 64])
def test_resnet_layers_native_shapes_match_one_shared_prefix(resnet_artifact, single_cpu_thread, side):
    from torchvision.models.resnet import ResNet
    shapes = {f'f{index}': ('B', channels, side // stride, side // stride)
              for index, channels, stride in ((1, 64, 4), (2, 128, 8), (3, 256, 16))}
    model = build_network(multiscale_resnet_source(resnet_artifact),
                          input_shape=('B', 3, side, side), output_shape=shapes)
    backbone = model['features'].model
    assert sum(isinstance(module, ResNet) for module in model.modules()) == 1
    model.eval()
    pixels = torch.linspace(-1, 1, 2 * 3 * side * side).reshape(2, 3, side, side)
    # Oracle traverses the same external model's stages, independent of HNDL's
    # intermediate capture; no second backbone or independently loaded weights.
    with torch.no_grad():
        value = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(pixels))))
        expected = {}
        for index in (1, 2, 3):
            value = getattr(backbone, f'layer{index}')(value)
            expected[f'f{index}'] = value.clone()
    names = ('', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3',
             'layer4', 'avgpool', 'fc')
    calls = dict.fromkeys(names, 0)
    def count(name):
        def record(module, inputs):
            calls[name] += 1
        return record
    handles = [backbone.get_submodule(name).register_forward_pre_hook(count(name)) for name in names]
    try:
        with torch.no_grad():
            outputs = model(pixels)
    finally:
        for handle in handles:
            handle.remove()
    assert set(outputs) == set(shapes)
    for name, output in outputs.items():
        assert output.shape == (2, *shapes[name][1:])
        torch.testing.assert_close(output, expected[name], rtol=0, atol=0)
    assert calls == {name: int(name not in ('layer4', 'avgpool', 'fc')) for name in names}


def test_resnet_layers_each_support_two_derivatives_and_phase_freezing(resnet_artifact, single_cpu_thread):
    model = build_network(multiscale_resnet_source(resnet_artifact),
                          input_shape=('B', 3, 32, 32),
                          output_shape={'f1': ('B', 64, 8, 8), 'f2': ('B', 128, 4, 4),
                                        'f3': ('B', 256, 2, 2)})
    backbone = model['features'].model
    frozen = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
    for training in (True, False, True):
        model.train(training)
        assert all(not module.training for module in backbone.modules())
        assert all(not parameter.requires_grad for parameter in backbone.parameters())
    candidate = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32).requires_grad_()
    outputs = model(candidate)
    for output in outputs.values():
        first, = torch.autograd.grad(output.square().mean(), candidate, create_graph=True)
        second, = torch.autograd.grad(first.square().sum(), candidate, retain_graph=True)
        assert torch.isfinite(first).all() and first.abs().sum() > 0
        assert torch.isfinite(second).all() and second.abs().sum() > 0
    # G-phase flag restoration must leave the pretrained parameters frozen.
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False)
    generated = candidate.detach().clone().requires_grad_()
    sum(output.square().mean() for output in model(generated).values()).backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    for parameter, flag in zip(model.parameters(), flags):
        parameter.requires_grad_(flag)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in backbone.parameters())
    assert all(not module.training for module in backbone.modules())
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


def test_resnet_single_layer_tuple_selects_explicit_hndl_input(resnet_artifact, single_cpu_thread):
    path, digest = resnet_artifact
    source = (f'only, = pretrained({str(path)!r}, provider="torchvision_resnet18", '
              f'sha256="{digest}", layers=("layer2",), trainable=False, name="features")\n'
              'global_avg_pool(only)')
    model = build_network(source, input_shape=('B', 3, 32, 32), output_shape=('B', 128))
    backbone = model['features'].model
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    with torch.no_grad():
        value = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(pixels))))
        expected = backbone.layer2(backbone.layer1(value)).mean(dim=(-2, -1))
        actual = model(pixels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 128)


@pytest.mark.parametrize('batch_size', [1, 3])
def test_fixed_context_example_shared_pass_normalization_and_training_gradients(
        resnet_artifact, single_cpu_thread, batch_size):
    """Exercise the editable production network, including both critic branches."""
    from torchvision.models.resnet import ResNet
    source = (Path(__file__).parents[2] / 'examples/networks/resnet18-multiscale-discriminator-128.hndl').read_text()
    path, digest = resnet_artifact
    model = build_network(source, input_shape=('B', 3, 128, 128), output_shape=('B', 1),
                          parameters={'weights_path': str(path), 'weights_sha256': digest})
    backbone = model['resnet'].model
    assert sum(isinstance(module, ResNet) for module in model.modules()) == 1
    frozen = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
    for training in (False, True):
        model.train(training)
        assert all(not module.training for module in backbone.modules())
    captured, calls = {}, {'prefix': 0, 'tail': 0}
    def backbone_input(module, inputs):
        calls['prefix'] += 1
        captured['normalized'] = inputs[0].detach().clone()
    def tail_input(module, inputs):
        calls['tail'] += 1
    def capture_split(name):
        def capture(module, inputs, output):
            captured[name] = tuple(value.detach().clone() for value in output)
        return capture
    handles = [backbone.register_forward_pre_hook(backbone_input),
               backbone.layer4.register_forward_pre_hook(tail_input)]
    handles.extend(model[f'feature{index}_split'].register_forward_hook(capture_split(index))
                   for index in (1, 2, 3))
    candidate = torch.linspace(-1, 1, batch_size * 3 * 128 * 128).reshape(batch_size, 3, 128, 128).requires_grad_()
    try:
        score = model(candidate)
    finally:
        for handle in handles:
            handle.remove()
    assert score.shape == (batch_size, 1) and torch.isfinite(score).all()
    assert calls == {'prefix': 1, 'tail': 0}
    mean = candidate.new_tensor((.485, .456, .406)).reshape(1, 3, 1, 1)
    std = candidate.new_tensor((.229, .224, .225)).reshape(1, 3, 1, 1)
    expected_candidate = (candidate.detach() * .5 + .5 - mean) / std
    # A zero RGB image in [-1,1] becomes .5 before ImageNet normalization;
    # appending zeros after normalization would mean a different reference.
    expected_context = ((.5 - mean) / std).expand_as(candidate)
    torch.testing.assert_close(captured['normalized'], torch.cat((expected_candidate, expected_context)),
                               rtol=2e-6, atol=2e-7)
    with torch.no_grad():
        oracle = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(captured['normalized']))))
        expected_features = {}
        for index in (1, 2, 3):
            oracle = getattr(backbone, f'layer{index}')(oracle)
            expected_features[index] = oracle.clone()
    for index, channels, side in ((1, 64, 32), (2, 128, 16), (3, 256, 8)):
        candidate_features, context_features = captured[index]
        assert candidate_features.shape == context_features.shape == (batch_size, channels, side, side)
        torch.testing.assert_close(candidate_features, expected_features[index][:batch_size], rtol=0, atol=0)
        torch.testing.assert_close(context_features, expected_features[index][batch_size:], rtol=0, atol=0)
        torch.testing.assert_close(context_features, context_features[:1].expand_as(context_features))
        assert not torch.equal(candidate_features, context_features)
    first, = torch.autograd.grad(score.square().sum(), candidate, create_graph=True)
    second, = torch.autograd.grad(first.square().sum(), candidate, retain_graph=True)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    assert torch.isfinite(second).all() and second.abs().sum() > 0
    # The D phase updates trainable heads through a second-order penalty.
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=.001)
    head_before = {name: model[name].weight.detach().clone()
                   for name in ('pixel_score', 'feature1_score', 'feature2_score', 'feature3_score')}
    (score.square().mean() + .1 * first.square().mean()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    optimizer.step()
    for name, before in head_before.items():
        assert not torch.equal(model[name].weight, before)
    model.zero_grad(set_to_none=True)
    # During G's phase, gradients still reach the candidate while all D
    # parameters are temporarily frozen and their individual flags restored.
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False)
    generated = candidate.detach().clone().requires_grad_()
    model(generated).square().mean().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    for parameter, flag in zip(model.parameters(), flags):
        parameter.requires_grad_(flag)
    assert all(model[name].weight.requires_grad for name in head_before)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(not module.training for module in backbone.modules())
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)

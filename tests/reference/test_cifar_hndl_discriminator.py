"""Single-file CIFAR discriminator parity, gradient ownership and recovery."""
import hashlib
from pathlib import Path

import pytest
import torch
from particlegan.grad_regularizers import GradientPenalty
from torch.nn import functional as F

from hypergan.config import load_config
from hypergan.hndl_networks import HNDLNetwork
from hypergan.image_components import CIFARDiscriminator, _legacy_feature_name


@pytest.fixture
def discriminators(tmp_path):
    from torchvision.models import resnet18
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(183)
            external = resnet18(weights=None)
            state = external.state_dict()
            path = tmp_path / 'resnet18.pth'
            torch.save(state, path)
            feature_digest = hashlib.sha256()
            for name, value in state.items():
                legacy_name = _legacy_feature_name(name)
                if legacy_name is not None:
                    feature_digest.update(legacy_name.encode())
                    feature_digest.update(value.contiguous().numpy().tobytes())
            parameters = {'weights_path': str(path),
                          'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            args = load_config(Path(__file__).parents[2] / 'examples/cifar-transgan.toml')[
                'components']['discriminator']['args']
            native = HNDLNetwork(**{**args, 'parameters': parameters})
            legacy = CIFARDiscriminator(**parameters, feature_state_sha256=feature_digest.hexdigest())
        yield native, legacy
    finally:
        torch.set_num_threads(previous)


def match_trainable_weights(native, legacy):
    """Copy by explicit layer identity; require coverage of every trainable tensor."""
    graph = native.network
    mapping = {'input': 'pixel_input', 'output': 'pixel_score'}
    mapping.update({f'attention_{part}': f'attention_{part}'
                    for part in ('query', 'key', 'value', 'project')})
    for index in range(3):
        for old, new in (('n1', 'norm1'), ('c1', 'conv1'), ('cond', 'condition'),
                         ('n2', 'norm2'), ('c2', 'conv2'), ('skip', 'shortcut')):
            if index == 2 and old == 'skip':
                continue
            mapping[f'block{index}_{old}'] = f'pixel_block{index}_{new}'
    pairs = [(legacy.critic.pixel.network[old], graph[new]) for old, new in mapping.items()]
    for index, head in enumerate(legacy.critic.project, 1):
        pairs.extend((head[old], graph[f'feature{index}_{new}'])
                     for old, new in (('project', 'project'), ('n2', 'norm'),
                                      ('n4', 'conv'), ('n8', 'score')))
    for old, new in pairs:
        new.load_state_dict(old.state_dict(), strict=True)
    for model, side in ((legacy, 0), (native, 1)):
        assert {id(p) for p in model.parameters() if p.requires_grad} == {
            id(p) for pair in pairs for p in pair[side].parameters()}
    return pairs


@pytest.mark.parametrize('batch_size', [1, 3])
def test_matched_weights_preserve_scores_and_two_input_derivatives(discriminators, batch_size):
    native, legacy = discriminators
    pairs = match_trainable_weights(native, legacy)
    candidate = torch.linspace(-.9, .9, batch_size * 3 * 32 * 32).reshape(batch_size, 3, 32, 32)
    actual_input = candidate.clone().requires_grad_()
    expected_input = candidate.clone().requires_grad_()
    actual, expected = native(actual_input).squeeze(1), legacy(expected_input)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_first, = torch.autograd.grad(actual.sum(), actual_input, create_graph=True)
    expected_first, = torch.autograd.grad(expected.sum(), expected_input, create_graph=True)
    torch.testing.assert_close(actual_first, expected_first, rtol=2e-4, atol=2e-6)
    actual_second, = torch.autograd.grad(actual_first.square().sum(), actual_input, retain_graph=True)
    expected_second, = torch.autograd.grad(expected_first.square().sum(), expected_input, retain_graph=True)
    torch.testing.assert_close(actual_second, expected_second, rtol=3e-4, atol=2e-6)
    assert actual_first.norm() > 0 and actual_second.norm() > 0
    # Also compare parameter derivatives, including the gradient-penalty path.
    (actual.square().mean() + actual_first.square().sum()).backward()
    (expected.square().mean() + expected_first.square().sum()).backward()
    for old, new in pairs:
        for old_parameter, new_parameter in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=3e-4, atol=3e-6)


def test_preprocessing_shared_backbone_and_fixed_context(discriminators):
    native, legacy = discriminators
    graph = native.network
    backbone = graph['resnet'].model
    captured, calls = {}, []

    def record_input(module, inputs):
        captured['normalized'] = inputs[0].detach()
        calls.append('backbone')

    def record_attention(module, inputs):
        captured['attention'] = inputs[0].shape

    def record_split(index):
        def record(module, inputs, output):
            captured[index] = tuple(value.detach() for value in output)
        return record

    handles = [backbone.register_forward_pre_hook(record_input),
               backbone.layer4.register_forward_pre_hook(lambda *args: calls.append('tail')),
               graph['attention_query'].register_forward_pre_hook(record_attention)]
    handles.extend(graph[f'feature{index}_split'].register_forward_hook(record_split(index))
                   for index in (1, 2, 3))
    candidate = torch.linspace(-1, 1, 3 * 3 * 32 * 32).reshape(3, 3, 32, 32)
    try:
        with torch.no_grad():
            score = native(candidate)
    finally:
        for handle in handles:
            handle.remove()
    assert score.shape == (3, 1)
    assert calls == ['backbone']
    assert captured['attention'] == (3, 64, 16, 16)
    mean, std = legacy.critic.mean, legacy.critic.std
    resized = F.interpolate(candidate, size=64, mode='bilinear', align_corners=False)
    expected = (resized * .5 + .5 - mean) / std
    context = ((.5 - mean) / std).expand_as(expected)
    torch.testing.assert_close(captured['normalized'], torch.cat((expected, context)), rtol=2e-6, atol=2e-7)
    with torch.no_grad():
        condition_features = legacy.critic.condition_features(legacy.context)
    for index, channels, size in ((1, 64, 16), (2, 128, 8), (3, 256, 4)):
        candidate_features, context_features = captured[index]
        assert candidate_features.shape == context_features.shape == (3, channels, size, size)
        torch.testing.assert_close(context_features, condition_features[index - 1].expand_as(context_features),
                                   rtol=2e-5, atol=2e-6)
        assert not torch.equal(candidate_features, context_features)


def test_bcap_training_freeze_and_strict_reload(discriminators):
    native, _ = discriminators
    backbone = native.network['resnet'].model
    frozen = {name: tensor.detach().clone() for name, tensor in backbone.state_dict().items()}
    for mode in (False, True):
        native.train(mode)
        assert all(not module.training for module in backbone.modules())
    real = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    fake = real.flip(-1)
    optimizer = torch.optim.Adam((p for p in native.parameters() if p.requires_grad), lr=.001)
    before = {name: p.detach().clone() for name, p in native.named_parameters() if p.requires_grad}
    penalty = GradientPenalty(kappa=0)(native, real, fake)
    loss = F.softplus(native(fake) - native(real)).mean() + penalty
    loss.backward()
    assert torch.isfinite(loss) and penalty > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in native.parameters() if p.requires_grad)
    optimizer.step()
    assert any(not torch.equal(p, before[name]) for name, p in native.named_parameters() if p.requires_grad)
    native.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in native.parameters()]
    native.requires_grad_(False)
    generated = fake.clone().requires_grad_()
    native(generated).sum().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.norm() > 0
    assert all(p.grad is None for p in native.parameters())
    for parameter, flag in zip(native.parameters(), flags, strict=True):
        parameter.requires_grad_(flag)
    assert all(not p.requires_grad for p in backbone.parameters())
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    state = {name: tensor.detach().clone() for name, tensor in native.state_dict().items()}
    with torch.no_grad():
        expected = native(real)
        native.network['pixel_score'].weight.add_(1)
        native.load_state_dict(state, strict=True)
        torch.testing.assert_close(native(real), expected, rtol=0, atol=0)

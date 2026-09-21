"""Real HNDL local-provider loading with a lightweight upstream DINO fixture."""
from pathlib import Path
from types import SimpleNamespace

from hndl import HNDLError
import pytest
import torch

from hypergan.hndl_networks import HNDLNetwork, build_network
from tests.dinov3_fixtures import dinov3_assets


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _UpstreamDINOFixture(torch.nn.Module):
    """Only the external model API is mocked; its small network is HNDL too."""
    def __init__(self, side):
        super().__init__()
        self.network = build_network('avg_pool(16)\nconv(384, kernel_size=1)\nbatch_norm()\ntanh()',
                                     input_shape=('B', 3, side, side),
                                     output_shape=('B', 384, side // 16, side // 16))
        self.calls = []

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        self.calls.append((tuple(x.shape), n, reshape, norm))
        features = self.network(x)
        # Distinct values expose incorrect readout ordering/channel splits.
        return tuple(features * (index + 1) / 4 for index in range(4))

    def forward_features(self, x):
        return {'x_norm_patchtokens': self.network(x).flatten(2).transpose(1, 2)}


def assets(monkeypatch, tmp_path, side=128):
    source, commit, weights, digest = dinov3_assets(monkeypatch, tmp_path, lambda: _UpstreamDINOFixture(side))
    return {'dinov3_vits16': {'source_path': source, 'source_commit': commit}}, {
        'weights_path': weights, 'weights_sha256': digest}


@pytest.mark.parametrize('batch_size', [1, 3])
def test_dino128_fixed_context_one_readout_all_heads_and_phase_gradients(monkeypatch, tmp_path, batch_size):
    providers, parameters = assets(monkeypatch, tmp_path)
    source = (Path(__file__).parents[2] / 'examples/networks/dinov3-multidepth-discriminator-128.hndl').read_text()
    model = HNDLNetwork(source, input_shape=('B', 3, 128, 128), output_shape=('B', 1),
                        parameters=parameters, pretrained_providers=providers)
    backbone = model.network['backbone'].model
    frozen = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
    for mode in (False, True):
        model.train(mode)
        assert all(not module.training for module in backbone.modules())
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = (inputs, output)
        return hook
    names = ['backbone', *(f'feature{index}_split' for index in range(1, 5)),
             *(f'feature{index}_project' for index in range(1, 5)),
             'pixel_score', *(f'feature{index}_score' for index in range(1, 5))]
    handles = [model.network[name].register_forward_hook(capture(name)) for name in names]
    x = torch.linspace(-1, 1, batch_size * 3 * 128 * 128).reshape(batch_size, 3, 128, 128).requires_grad_()
    rng = torch.get_rng_state().clone()
    try:
        score = model(x=x)
    finally:
        for handle in handles:
            handle.remove()
    assert score.shape == (batch_size, 1) and torch.isfinite(score).all()
    assert model.training and torch.equal(rng, torch.get_rng_state())
    feature_score = sum(captured[f'feature{index}_score'][1] for index in range(1, 5)) * .5
    torch.testing.assert_close(score, (captured['pixel_score'][1] + feature_score) * (2 ** -.5))
    assert backbone.calls == [((2 * batch_size, 3, 128, 128), (2, 5, 8, 11), True, True)]
    (normalized,), all_depths = captured['backbone']
    assert all_depths.shape == (2 * batch_size, 1536, 8, 8)
    mean = x.new_tensor((.485, .456, .406)).reshape(1, 3, 1, 1)
    std = x.new_tensor((.229, .224, .225)).reshape(1, 3, 1, 1)
    expected_context = ((.5 - mean) / std).expand_as(x)
    expected_candidate = (x.detach() * .5 + .5 - mean) / std
    torch.testing.assert_close(normalized, torch.cat((expected_candidate, expected_context)), rtol=2e-6, atol=2e-7)
    for index in range(1, 5):
        candidate, context = captured[f'feature{index}_split'][1]
        assert candidate.shape == context.shape == (batch_size, 384, 8, 8)
        joined = captured[f'feature{index}_project'][0][0]
        assert joined.shape == (batch_size, 768, 8, 8)
        torch.testing.assert_close(joined, torch.cat((candidate, context), dim=1), rtol=0, atol=0)
        expected = all_depths[:, 384 * (index - 1):384 * index]
        torch.testing.assert_close(candidate, expected[:batch_size], rtol=0, atol=0)
        torch.testing.assert_close(context, expected[batch_size:], rtol=0, atol=0)
        torch.testing.assert_close(context, context[:1].expand_as(context))
        assert not torch.equal(candidate, context)
    first, = torch.autograd.grad(score.square().mean(), x, create_graph=True)
    second, = torch.autograd.grad(first.square().sum(), x, retain_graph=True)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    assert torch.isfinite(second).all() and second.abs().sum() > 0
    heads = ['pixel_score', *(f'feature{index}_score' for index in range(1, 5))]
    before = {name: model.network[name].weight.detach().clone() for name in heads}
    (score.square().mean() + .1 * first.square().mean()).backward()
    for name in heads:
        gradient = model.network[name].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=.001).step()
    assert all(not torch.equal(model.network[name].weight, before[name]) for name in heads)
    model.zero_grad(set_to_none=True)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False)
    generated = x.detach().clone().requires_grad_()
    model(generated).square().mean().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    for parameter, flag in zip(model.parameters(), flags):
        parameter.requires_grad_(flag)
    assert all(model.network[name].weight.requires_grad for name in heads)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(not module.training for module in backbone.modules())
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


@pytest.mark.parametrize('side', [128, 256])
@pytest.mark.parametrize('readout', ['multidepth', 'patch_tokens'])
def test_native_dino_readouts_keep_input_patch_grid(monkeypatch, tmp_path, side, readout):
    providers, parameters = assets(monkeypatch, tmp_path, side)
    grid = side // 16
    shape = ('B', 1536, grid, grid) if readout == 'multidepth' else ('B', grid * grid, 384)
    source = ('pretrained(${weights_path}, provider="dinov3_vits16", sha256=${weights_sha256}, '
              f'readout="{readout}", trainable=False, name="backbone")')
    model = build_network(source, input_shape=('B', 3, side, side), output_shape=shape,
                          parameters=parameters, pretrained_providers=providers)
    x = torch.linspace(-1, 1, 3 * side * side).reshape(1, 3, side, side).requires_grad_()
    output = model(x)
    assert output.shape == (1, *shape[1:]) and torch.isfinite(output).all()
    first, = torch.autograd.grad(output.square().mean(), x, create_graph=True)
    second, = torch.autograd.grad(first.square().sum(), x)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    assert torch.isfinite(second).all() and second.abs().sum() > 0
    if readout == 'multidepth':
        for index, depth in enumerate(output.chunk(4, dim=1), 1):
            torch.testing.assert_close(depth, output[:, :384] * index)


def test_configured_dino_still_verifies_local_weight_digest(monkeypatch, tmp_path):
    providers, parameters = assets(monkeypatch, tmp_path)
    parameters['weights_sha256'] = '0' * 64
    source = ('pretrained(${weights_path}, provider="dinov3_vits16", sha256=${weights_sha256}, '
              'readout="multidepth")')
    with pytest.raises(HNDLError, match='(?i)sha256|checksum'):
        build_network(source, input_shape=('B', 3, 128, 128), output_shape=('B', 1536, 8, 8),
                      parameters=parameters, pretrained_providers=providers)


def test_configured_dino_rejects_unpinned_source_before_import(tmp_path):
    with pytest.raises(ValueError, match='source_commit'):
        build_network('linear(1)', input_shape=('B', 2), output_shape=('B', 1),
                      pretrained_providers={'dinov3_vits16': {'source_path': str(tmp_path),
                                                            'source_commit': 'main'}})


@pytest.mark.parametrize('count,channels,height,width', [(3, 384, 8, 8), (4, 383, 8, 8),
                                                        (4, 384, 16, 16), (4, 384, 8, 7)])
def test_multidepth_readout_rejects_malformed_upstream_maps(count, channels, height, width):
    from hypergan.pretrained_providers import _dinov3_multidepth
    maps = tuple(torch.zeros(1, channels, height, width) for _ in range(count))
    upstream = SimpleNamespace(get_intermediate_layers=lambda *args, **kwargs: maps)
    with pytest.raises(ValueError, match='four .* intermediate maps'):
        _dinov3_multidepth(upstream, torch.zeros(1, 3, 128, 128))


@pytest.mark.parametrize('shape', [(1, 63, 384), (1, 64, 383)])
def test_patch_readout_rejects_malformed_upstream_tokens(shape):
    from hypergan.pretrained_providers import _dinov3_patch_tokens
    upstream = SimpleNamespace(forward_features=lambda x: {'x_norm_patchtokens': torch.zeros(shape)})
    with pytest.raises(ValueError, match='64 patch tokens'):
        _dinov3_patch_tokens(upstream, torch.zeros(1, 3, 128, 128))

"""256px conditional colorizer and DINOv3 multidepth projected critic under ParticleGAN defaults."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

import hypergan.api as hg
from hypergan.colorization_data import luma
from hypergan.config import load_config
from hypergan.hndl_networks import HNDLNetwork, build_network
from hypergan.training import initialize_fresh_parameters
from tests.dinov3_fixtures import dinov3_assets


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _UpstreamDINOFixture256(nn.Module):
    """Four native 16x16 patch maps for 256x256 input."""
    def __init__(self, side=256):
        super().__init__()
        self.network = build_network(
            "avg_pool(16)\nconv(384, kernel_size=1)\nlayer_norm()\ntanh()",
            input_shape=("B", 3, side, side),
            output_shape=("B", 384, side // 16, side // 16),
        )
        self.calls = []

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        self.calls.append((tuple(x.shape), n, reshape, norm))
        features = self.network(x)
        return tuple(features * (index + 1) / 4 for index in range(4))


def _assets_256(monkeypatch, tmp_path):
    source, commit, weights, digest = dinov3_assets(
        monkeypatch, tmp_path, lambda: _UpstreamDINOFixture256(256)
    )
    return {"dinov3_vits16": {"source_path": source, "source_commit": commit}}, {
        "weights_path": weights,
        "weights_sha256": digest,
    }


def test_colorization_generator_256_polytope_luminance_lock_and_zero_chroma_origin():
    root = Path(__file__).parents[2]
    source = (root / "examples/networks/colorization-generator-256.hndl").read_text()
    gen = HNDLNetwork(
        source,
        input_shape={"z": ("B", 128), "gray": ("B", 1, 256, 256)},
        output_shape=("B", 3, 256, 256),
    )
    graph = nn.Module()
    graph.models = nn.ModuleDict({"colorizer": gen})
    initialize_fresh_parameters(graph, seed=0)

    assert not any(isinstance(m, nn.BatchNorm2d) for m in gen.modules())

    gray = torch.linspace(-1.0, 1.0, 2 * 1 * 256 * 256).reshape(2, 1, 256, 256)
    z_zero = torch.zeros(2, 128)
    out_zero = gen(z=z_zero, gray=gray)
    assert out_zero.shape == (2, 3, 256, 256)
    assert (out_zero - gray.expand_as(out_zero)).abs().max().item() < 1e-5

    torch.manual_seed(7)
    z = (torch.randn(2, 128) * 2.0).requires_grad_()
    out = gen(z=z, gray=gray)
    assert torch.isfinite(out).all()
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0
    assert (luma(out) - gray).abs().max().item() < 1e-5
    assert (out - gray.expand_as(out)).abs().mean().item() > 1e-3

    out.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_dinov3_projected_256_orthogonal_init_and_second_derivatives(monkeypatch, tmp_path):
    providers, parameters = _assets_256(monkeypatch, tmp_path)
    root = Path(__file__).parents[2]
    source = (root / "examples/networks/dinov3-projected-discriminator-256.hndl").read_text()
    model = HNDLNetwork(
        source,
        input_shape=("B", 3, 256, 256),
        output_shape=("B", 1),
        parameters=parameters,
        pretrained_providers=providers,
    )
    backbone_module = model.network["backbone"]
    backbone = backbone_module.model
    frozen_before = copy.deepcopy(backbone.state_dict())

    graph = nn.Module()
    graph.models = nn.ModuleDict({"critic": model})
    initialize_fresh_parameters(graph, seed=0)

    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, frozen_before[key], rtol=0, atol=0)
    assert all(not p.requires_grad for p in backbone.parameters())

    outside = [m for m in model.network.modules() if id(m) not in {id(b) for b in backbone_module.modules()}]
    convs = [m for m in outside if isinstance(m, nn.Conv2d)]
    group_norms = [m for m in outside if isinstance(m, nn.GroupNorm)]
    linears = [m for m in outside if isinstance(m, nn.Linear)]
    assert len(convs) == 18
    assert len(group_norms) == 13
    assert len(linears) == 5
    assert all(m.bias is None for m in linears)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in outside)

    x = torch.linspace(-1, 1, 2 * 3 * 256 * 256).reshape(2, 3, 256, 256).requires_grad_()
    score = model(x=x)
    assert score.shape == (2, 1) and torch.isfinite(score).all()
    assert backbone.calls == [((2, 3, 256, 256), (2, 5, 8, 11), True, True)]

    first, = torch.autograd.grad(score.mean(), x, create_graph=True, retain_graph=True)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    (score.mean() + first.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for module in (*convs, *linears):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0


def test_logos_colorization_dinov3_256_archive_recipe_loads_and_validates():
    root = Path(__file__).parents[2]
    recipe_path = root / "examples/archive/logos-colorization-dinov3-256.toml"
    model = hg.load(recipe_path)
    warnings = hg.validate(model)
    assert isinstance(warnings, list)

    resolved = load_config(recipe_path)
    assert resolved["training"]["initialization"] == "deterministic_orthogonal"
    assert resolved["prior_regularizer"]["rows"] == "full"
    assert resolved["components"]["generator"]["args"]["source"] == (
        root / "examples/networks/colorization-generator-256.hndl"
    ).read_text()
    assert resolved["components"]["discriminator"]["args"]["source"] == (
        root / "examples/networks/dinov3-projected-discriminator-256.hndl"
    ).read_text()

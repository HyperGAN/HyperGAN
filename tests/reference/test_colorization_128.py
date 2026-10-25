"""128px conditional colorization generator and hypergan-model/1 recipe contracts."""
from pathlib import Path

import pytest
import torch
from torch import nn

from hypergan.colorization_metrics import (
    GrayscaleStructureDistance,
    RepeatedConditionChromaDiversity,
    _luma,
)
from hypergan.config import load_config
from hypergan.hndl_networks import HNDLNetwork, build_network
from hypergan.previews import render_preview
from hypergan.training import ReferenceTrainer, initialize_fresh_parameters
from tests.dinov3_fixtures import dinov3_assets


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def isolated_backend_policy(monkeypatch):
    """Let the recipe's process-wide backend policy apply in any test order.

    Any earlier in-process ``optimizer.step()`` initialises CUDA on a GPU host,
    after which ``apply_backend_policy`` refuses a CUBLAS workspace that the
    environment does not already carry. Pin the recipe's value up front, as a
    fresh process would, and restore every flag the policy sets afterwards.
    """
    def pin(config):
        workspace = config["training"]["backend"].get("cublas_workspace_config")
        if workspace is not None:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)

    flags = [(torch.backends.cudnn, "deterministic"), (torch.backends.cudnn, "benchmark"),
             (torch.backends.cuda.matmul, "allow_tf32"), (torch.backends.cudnn, "allow_tf32")]
    saved = [getattr(owner, name) for owner, name in flags]
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield pin
    finally:
        for (owner, name), value in zip(flags, saved):
            setattr(owner, name, value)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class _UpstreamDINOFixture(nn.Module):
    """Four native 8x8 patch maps."""

    def __init__(self, side=128):
        super().__init__()
        self.network = build_network(
            "avg_pool(16)\nconv(384, kernel_size=1)\nlayer_norm()\ntanh()",
            input_shape=("B", 3, side, side),
            output_shape=("B", 384, side // 16, side // 16),
        )

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        features = self.network(x)
        return tuple(features * (index + 1) / 4 for index in range(4))


def _build_initialized_generator(seed=0):
    root = Path(__file__).parents[2]
    source = (root / "examples/networks/colorization-generator-128.hndl").read_text()
    model = HNDLNetwork(
        source,
        input_shape={"z": ("B", 128), "gray": ("B", 1, 128, 128)},
        output_shape=("B", 3, 128, 128),
    )
    graph = nn.Module()
    graph.models = nn.ModuleDict({"generator": model})
    initialize_fresh_parameters(graph, seed=seed)
    return model


def test_colorization_128_model_file_lowers_with_particlegan_defaults():
    root = Path(__file__).parents[2]
    resolved = load_config(root / "examples/logos-colorization-128.toml")
    assert resolved["format"] == "hypergan-model/1"
    assert resolved["name_map"] == {"colorizer": "generator", "critic": "discriminator"}
    assert resolved["training"]["initialization"] == "deterministic_orthogonal"
    assert resolved["training"]["batch_size"] == 64
    assert resolved["prior"]["kind"] == "particles"
    assert resolved["prior"]["args"] == {"num_particles": 4096, "z_dim": 128}
    assert resolved["prior_regularizer"]["rows"] == "sampled_unique"
    gen = resolved["components"]["generator"]
    assert gen["inputs"] == {"z": "latent", "gray": "batch.gray"}
    assert gen["args"]["input_shape"] == {"z": ["B", 128], "gray": ["B", 1, 128, 128]}
    assert gen["args"]["output_shape"] == ["B", 3, 128, 128]
    assert gen["args"]["source"] == (
        root / "examples/networks/colorization-generator-128.hndl"
    ).read_text()
    disc = resolved["components"]["discriminator"]
    assert disc["args"]["source"] == (
        root / "examples/networks/dinov3-projected-discriminator-128.hndl"
    ).read_text()
    assert resolved["sampling"]["comparison"] == [
        {"label": "X", "binding": "batch.real"},
        {"label": "B", "binding": "batch.gray"},
        {"label": "X_hat", "binding": "components.generator"},
    ]
    assert "diversity" in resolved["metrics"]["custom"]
    assert resolved["metrics"]["custom"]["diversity"]["args"]["repeats"] == 4


def test_exact_polytope_luminance_lock_and_background_invariance():
    model = _build_initialized_generator(seed=0)
    gen = torch.Generator().manual_seed(42)
    z = torch.randn(4, 128, generator=gen)
    real_rgb = torch.rand(4, 3, 128, 128, generator=gen) * 2.0 - 1.0
    # Include exact white (+1), exact black (-1), and extreme saturated RGB corners.
    real_rgb[0, :, :32, :32] = 1.0
    real_rgb[1, :, :32, :32] = -1.0
    real_rgb[2, 0, :32, :32] = 1.0
    real_rgb[2, 1:, :32, :32] = -1.0
    real_rgb[3, :2, :32, :32] = 1.0
    real_rgb[3, 2, :32, :32] = -1.0
    gray = _luma(real_rgb)

    with torch.no_grad():
        out = model(z=z, gray=gray)
    assert out.shape == (4, 3, 128, 128)
    assert torch.isfinite(out).all()
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0

    out_gray = _luma(out)
    torch.testing.assert_close(out_gray, gray, rtol=0.0, atol=1e-5)
    # White background (gray == +1) must map to exact white (1, 1, 1).
    torch.testing.assert_close(
        out[0, :, :32, :32], torch.ones_like(out[0, :, :32, :32]), rtol=0.0, atol=1e-6
    )
    # Black background (gray == -1) must map to exact black (-1, -1, -1).
    torch.testing.assert_close(
        out[1, :, :32, :32], -torch.ones_like(out[1, :, :32, :32]), rtol=0.0, atol=1e-6
    )
    # GrayscaleStructureDistance must be near zero at initialization by construction.
    structure = GrayscaleStructureDistance().evaluate(
        batches=[{"generated": out, "reference": real_rgb}], context={}
    )
    assert structure < 1e-5


def test_multimodal_particle_diversity_and_full_gradient_flow():
    model = _build_initialized_generator(seed=0)
    # Verify HNDL bidirectional shape inference solved the adaptive_norm style projections.
    assert model.network["style16"].out_features == 256
    assert model.network["style32"].out_features == 128
    assert model.network["style64"].out_features == 64

    gen = torch.Generator().manual_seed(7)
    base_gray = torch.rand(1, 1, 128, 128, generator=gen) * 1.4 - 0.7
    gray = base_gray.repeat(4, 1, 1, 1).requires_grad_(True)
    z = torch.randn(4, 128, generator=gen).requires_grad_(True)

    out = model(z=z, gray=gray)
    diversity = RepeatedConditionChromaDiversity(repeats=4).evaluate(
        batches=[
            {
                "generated": out.detach(),
                "reference": base_gray.expand(1, 3, 128, 128).repeat(4, 1, 1, 1),
            }
        ],
        context={},
    )
    assert diversity > 0.05

    # Bias-free palette head guarantees zero chroma when z == 0.
    with torch.no_grad():
        zero_out = model(z=torch.zeros_like(z[:1]), gray=base_gray)
        torch.testing.assert_close(
            zero_out, base_gray.expand(1, 3, 128, 128), rtol=0.0, atol=1e-5
        )

    # Differentiate a chroma-sensitive objective through the 7-stage polytope projection.
    weights = torch.tensor([1.3, -0.7, 0.4]).view(1, 3, 1, 1)
    loss = (out * weights).square().mean()
    loss.backward()

    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert (z.grad.norm(dim=1) > 0).all()
    assert gray.grad is not None and torch.isfinite(gray.grad).all()
    assert (gray.grad.flatten(1).norm(dim=1) > 0).all()

    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name

    for node_name in (
        "enc128",
        "enc64",
        "enc32",
        "enc16",
        "enc8",
        "palette_tokens",
        "bottleneck_block0",
        "palette_xattn0",
        "bottleneck_block1",
        "palette_xattn1",
        "up16_conv",
        "dec16",
        "style16",
        "up32_conv",
        "dec32",
        "style32",
        "up64_conv",
        "dec64",
        "style64",
        "up128_conv",
        "dec128",
        "palette_head",
    ):
        assert (
            sum(p.grad.abs().sum() for p in model.network[node_name].parameters()) > 0
        ), node_name


def test_colorization_128_backend_policy_applies_after_cuda_initialization(monkeypatch, isolated_backend_policy):
    from hypergan.numerical_policy import apply_backend_policy

    root = Path(__file__).parents[2]
    config = load_config(root / "examples/logos-colorization-128.toml")
    config["training"]["device"] = "cpu"
    config["training"]["backend"]["deterministic_algorithms"] = False
    isolated_backend_policy(config)
    # The state an earlier in-process optimizer step leaves on a GPU host.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    apply_backend_policy(config)


def test_colorization_128_trainer_step_and_comparison_preview(monkeypatch, tmp_path, isolated_backend_policy):
    source, commit, weights, digest = dinov3_assets(
        monkeypatch, tmp_path, lambda: _UpstreamDINOFixture(128)
    )
    root = Path(__file__).parents[2]
    config = load_config(root / "examples/logos-colorization-128.toml")
    config["training"]["device"] = "cpu"
    config["training"]["batch_size"] = 2
    config["training"]["steps"] = 1
    config["training"]["backend"]["deterministic_algorithms"] = False
    config["prior"]["args"]["num_particles"] = 8
    config["sampling"]["count"] = 2
    disc_args = config["components"]["discriminator"]["args"]
    disc_args["parameters"] = {"weights_path": weights, "weights_sha256": digest}
    disc_args["pretrained_providers"] = {
        "dinov3_vits16": {"source_path": source, "source_commit": commit}
    }

    class _SyntheticColorizationData:
        def __init__(self):
            gen = torch.Generator().manual_seed(11)
            self._real = torch.rand(4, 3, 128, 128, generator=gen) * 2.0 - 1.0
            self._gray = _luma(self._real)

        def __call__(self, batch_size, *, generator=None):
            idx = torch.arange(batch_size) % len(self._real)
            return {
                "real": self._real[idx],
                "gray": self._gray[idx],
            }

    monkeypatch.setattr(
        "tests.reference.test_colorization_128._SyntheticColorizationData",
        _SyntheticColorizationData,
        raising=False,
    )
    config["data"] = {
        "factory": "tests.reference.test_colorization_128:_SyntheticColorizationData",
        "args": {},
    }
    isolated_backend_policy(config)
    trainer = ReferenceTrainer(config)
    before_z = trainer.prior.z.detach().clone()
    row, batch = trainer.update()
    assert trainer.step == 1
    assert torch.isfinite(torch.tensor(row["d_loss"]))
    assert torch.isfinite(torch.tensor(row["g_loss"]))
    assert not torch.equal(trainer.prior.z, before_z)

    preview = render_preview(
        trainer, batch, {"run_id": "test", "lineage_id": "0", "step": 1}
    )
    assert preview["conditioning"] == "last-completed-batch-cycled"
    assert "comparison_image_grid" in preview
    assert [col["label"] for col in preview["comparison_image_grid"]["sources"]] == [
        "X",
        "B",
        "X_hat",
    ]

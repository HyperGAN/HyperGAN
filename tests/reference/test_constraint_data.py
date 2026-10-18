"""Unit and integration tests for data-less constraint GAN data, objectives, metrics, and previews."""
import base64
import copy
import io
import json
import math
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn

from hypergan.checkpoints import data_contract, read_checkpoint
from hypergan.config import DEFAULT, resolve_config
from hypergan.constraint_data import (
    ConstraintHiddenTargetData,
    ConstraintSatisfactionMetric,
    ConstraintTargetData,
    OptionProbability,
    TotalVariationSmoothness,
)
from hypergan.distributed_checkpoints import _digest
from hypergan.metric_plugins import _factory
from hypergan.previews import publish_preview_payload, render_preview
from hypergan.training import ReferenceTrainer, resume, train


class TinyImageGenerator(nn.Module):
    def __init__(self, z_dim=4, size=64):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(z_dim, 3 * 4 * 4)

    def forward(self, x):
        base = torch.tanh(self.fc(x)).view(len(x), 3, 4, 4)
        if self.size == 4:
            return base
        return nn.functional.interpolate(base, size=(self.size, self.size), mode="nearest")


class TinyConstraintHead(nn.Module):
    def __init__(self, num_options=5):
        super().__init__()
        self.proj = nn.Linear(3, num_options, bias=True)
        self.requires_grad_(False)

    def forward(self, x):
        pooled = x.mean(dim=(-2, -1))
        logits = self.proj(pooled)
        return logits - logits.mean(dim=-1, keepdim=True)


class TinyLogitDiscriminator(nn.Module):
    def __init__(self, num_options=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_options, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


def test_constraint_target_data_single_mode_logits_and_probs():
    probs = [0.92, 0.02, 0.02, 0.02, 0.02]
    unnormalized = [92, 2, 2, 2, 2]
    data_logits = ConstraintTargetData(probs, mode="logits", jitter_std=0.0)
    data_unnorm = ConstraintTargetData(unnormalized, mode="logits", jitter_std=0.0)
    assert data_logits.resume_identity() == data_unnorm.resume_identity()

    batch = data_logits(8)["real"]
    assert batch.shape == (8, 5)
    assert batch.dtype == torch.float32
    torch.testing.assert_close(batch.mean(dim=-1), torch.zeros(8), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        batch.softmax(dim=-1),
        torch.tensor(probs, dtype=torch.float32).expand(8, -1),
        atol=1e-6,
        rtol=1e-6,
    )

    data_probs = ConstraintTargetData(probs, mode="probs", jitter_std=0.0)
    batch_probs = data_probs(6)["real"]
    assert batch_probs.shape == (6, 5)
    torch.testing.assert_close(
        batch_probs,
        torch.tensor(probs, dtype=torch.float32).expand(6, -1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_constraint_target_data_multi_mode_and_rng_reproducibility():
    modes = [
        [0.90, 0.04, 0.02, 0.02, 0.02],
        [0.04, 0.90, 0.02, 0.02, 0.02],
    ]
    data_clean = ConstraintTargetData(modes, mode="probs", jitter_std=0.0)
    rng = torch.Generator().manual_seed(1234)
    samples = data_clean(64, generator=rng)["real"]
    assert samples.shape == (64, 5)
    top_indices = samples.argmax(dim=-1)
    assert set(top_indices.tolist()) == {0, 1}

    data_jitter = ConstraintTargetData(modes, mode="logits", jitter_std=0.15)
    g1 = torch.Generator().manual_seed(999)
    g2 = torch.Generator().manual_seed(999)
    b1 = data_jitter(16, generator=g1)["real"]
    b2 = data_jitter(16, generator=g2)["real"]
    assert torch.equal(b1, b2)
    torch.testing.assert_close(b1.mean(dim=-1), torch.zeros(16), atol=1e-6, rtol=0.0)

    spec = {
        "factory": "hypergan.constraint_data:ConstraintTargetData",
        "args": {"target_probs": modes, "mode": "logits", "jitter_std": 0.15},
    }
    contract = data_contract(data_jitter, spec)
    assert contract["supported"] is True
    assert contract["stateful"] is False
    assert contract["identity"] == data_jitter.resume_identity()


def test_constraint_target_data_validation():
    for invalid_probs in (
        [],
        [1.0],
        [0.5, 0.0],
        [0.5, -0.5],
        [0.5, float("nan")],
        [0.5, float("inf")],
        [True, False],
        [[0.5, 0.5], [0.5, 0.25, 0.25]],
        [[0.5], [0.5]],
        "invalid",
    ):
        with pytest.raises(ValueError):
            ConstraintTargetData(invalid_probs)
    with pytest.raises(ValueError, match="mode"):
        ConstraintTargetData([0.5, 0.5], mode="invalid")
    for invalid_jitter in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="jitter_std"):
            ConstraintTargetData([0.5, 0.5], jitter_std=invalid_jitter)
    data = ConstraintTargetData([0.5, 0.5])
    for invalid_batch_size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="batch_size"):
            data(invalid_batch_size)


def test_option_probability_and_tv_smoothness():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32, requires_grad=True)
    opt0 = OptionProbability(index=0, input_mode="logits")
    val0 = opt0(logits)
    expected = logits.softmax(dim=-1)[:, 0].mean()
    torch.testing.assert_close(val0, expected)

    probs = logits.detach().softmax(dim=-1)
    opt1_probs = OptionProbability(index=1, input_mode="probs")
    torch.testing.assert_close(opt1_probs(probs), probs[:, 1].mean())

    with pytest.raises(ValueError, match="out of bounds"):
        OptionProbability(index=3)(logits)
    with pytest.raises(ValueError, match="index"):
        OptionProbability(index=-1)
    with pytest.raises(ValueError, match="index"):
        OptionProbability(index=True)

    tv = TotalVariationSmoothness(delta=0.05)
    flat = torch.ones(2, 3, 8, 8, requires_grad=True)
    loss_flat = tv(flat)
    assert float(loss_flat.item()) == pytest.approx(0.0, abs=1e-7)
    loss_flat.backward()
    assert torch.all(flat.grad == 0.0)

    varying = torch.zeros(1, 1, 4, 4, requires_grad=True)
    with torch.no_grad():
        varying[:, :, ::2, ::2] = 1.0
    loss_var = tv(varying)
    expected_step = math.sqrt(1.0 + 0.05 ** 2) - 0.05
    assert float(loss_var.item()) == pytest.approx(expected_step, rel=1e-5)
    loss_var.backward()
    assert varying.grad is not None and torch.isfinite(varying.grad).all()

    for bad_delta in (0.0, -0.1, float("nan"), True):
        with pytest.raises(ValueError, match="delta"):
            TotalVariationSmoothness(delta=bad_delta)
    with pytest.raises(ValueError, match="4D"):
        tv(torch.zeros(2, 4))


def test_constraint_satisfaction_metric_statistics_and_batch_invariance():
    # Construct 4 samples with 5 options: 3 strongly favoring index 0, 1 favoring index 1
    row_a = torch.tensor([10.0, 0.0, 0.0, 0.0, -5.0], dtype=torch.float32)
    row_b = torch.tensor([0.0, 10.0, 0.0, 0.0, -5.0], dtype=torch.float32)
    logits = torch.stack([row_a, row_a, row_a, row_b], dim=0)
    probs = logits.double().softmax(dim=-1)

    for statistic in ("target_prob", "unknown_prob", "target_accuracy", "marginal_entropy"):
        instance, meta = _factory(
            "hypergan.constraint_data:ConstraintSatisfactionMetric",
            {"statistic": statistic, "target_index": 0, "input_mode": "logits"},
        )
        assert meta["descriptor"]["kind"] == "scalar"
        whole = instance.evaluate(batches=[{"generated": logits}], context={})
        split = instance.evaluate(
            batches=[{"generated": logits[i : i + 1]} for i in range(len(logits))],
            context={},
        )
        assert split == pytest.approx(whole, abs=1e-14)

        probs_metric = ConstraintSatisfactionMetric(
            statistic=statistic, target_index=0, input_mode="probs"
        )
        assert probs_metric.evaluate(batches=[{"generated": probs.float()}], context={}) == pytest.approx(
            whole, abs=1e-6
        )

    acc = ConstraintSatisfactionMetric(statistic="target_accuracy", target_index=0).evaluate(
        batches=[{"generated": logits}], context={}
    )
    assert acc == pytest.approx(0.75)

    target_p = ConstraintSatisfactionMetric(statistic="target_prob", target_index=0).evaluate(
        batches=[{"generated": logits}], context={}
    )
    assert target_p == pytest.approx(float(probs[:, 0].mean().item()), abs=1e-7)

    unknown_p = ConstraintSatisfactionMetric(statistic="unknown_prob").evaluate(
        batches=[{"generated": logits}], context={}
    )
    assert unknown_p == pytest.approx(float(probs[:, -1].mean().item()), abs=1e-9)

    # Balanced 2-mode probability batch has marginal entropy log(2)
    balanced_probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    ent = ConstraintSatisfactionMetric(statistic="marginal_entropy", input_mode="probs").evaluate(
        batches=[{"generated": balanced_probs}], context={}
    )
    assert ent == pytest.approx(math.log(2.0), abs=1e-7)

    with pytest.raises(ValueError, match="nonempty"):
        ConstraintSatisfactionMetric().evaluate(batches=[], context={})
    with pytest.raises(ValueError, match="finite"):
        ConstraintSatisfactionMetric().evaluate(
            batches=[{"generated": logits * float("nan")}], context={}
        )
    with pytest.raises(ValueError, match="out of bounds"):
        ConstraintSatisfactionMetric(target_index=5).evaluate(
            batches=[{"generated": logits}], context={}
        )


def test_constraint_gan_preview_renders_64x64_png_and_trains_with_detached_monitors(tmp_path):
    config = copy.deepcopy(DEFAULT)
    config["data"] = {
        "factory": "hypergan.constraint_data:ConstraintTargetData",
        "args": {
            "target_probs": [0.92, 0.02, 0.02, 0.02, 0.02],
            "mode": "logits",
            "jitter_std": 0.15,
        },
    }
    config["components"]["image_generator"] = {
        "factory": __name__ + ":TinyImageGenerator",
        "args": {"z_dim": 4, "size": 64},
        "inputs": {"x": "latent"},
        "trainable": True,
    }
    config["components"]["generator"] = {
        "factory": __name__ + ":TinyConstraintHead",
        "args": {"num_options": 5},
        "inputs": {"x": "components.image_generator"},
        "trainable": True,
    }
    config["components"]["discriminator"] = {
        "factory": __name__ + ":TinyLogitDiscriminator",
        "args": {"num_options": 5},
        "inputs": {"x": "candidate"},
        "trainable": True,
    }
    config["objectives"] = [
        {
            "id": "tv_smoothness",
            "factory": "hypergan.constraint_data:TotalVariationSmoothness",
            "args": {"delta": 0.05},
            "inputs": {"image": "components.image_generator"},
            "weight": 0.05,
        },
        {
            "id": "apple_prob",
            "factory": "hypergan.constraint_data:OptionProbability",
            "args": {"index": 0},
            "inputs": {"logits": "generated"},
            "detach": ["logits"],
            "weight": 1.0,
        },
    ]
    config["prior"]["args"].update(num_particles=32, z_dim=4)
    config["sampling"].update(generated="components.image_generator", count=16, seed=35002)
    config["training"].update(steps=2, batch_size=4, device="cpu", seed=25002)
    config["metrics"] = {"preset": "standard", "every_steps": 1}
    resolved = resolve_config(config)

    trainer = ReferenceTrainer(resolved)
    row, batch = trainer.update()
    assert batch["real"].shape == (4, 5)
    assert len(row["objectives"]) == 2
    assert 0.0 <= row["objectives"][1] <= 1.0

    identity = {"run_id": "run", "attempt_id": "0001-" + "a" * 32, "sample_sequence": 1}
    payload = render_preview(trainer, batch, identity)
    assert payload["count"] == 16
    assert payload["shape"] == [16, 3, 64, 64]
    assert payload["representation"] == "png"
    assert payload["samples"] is None
    assert "real_image_grid" not in payload
    with Image.open(io.BytesIO(base64.b64decode(payload["image_grid"]["png_base64"]))) as image:
        assert image.mode == "RGB"
        assert image.size == (256, 256)

    record, _, errors = publish_preview_payload(tmp_path, payload, identity, trainer.step)
    assert not errors
    assert record["representation"] == "png"
    assert record["bytes"] < 8192
    assert Path(record["image_grid"]["path"]).is_file()


def test_constraint_data_checkpoint_resume_determinism(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config_path = tmp_path / "constraint.toml"
        config_path.write_text(f"""[data]
factory = "hypergan.constraint_data:ConstraintTargetData"
[data.args]
target_probs = [[0.90, 0.04, 0.02, 0.02, 0.02], [0.04, 0.90, 0.02, 0.02, 0.02]]
mode = "logits"
jitter_std = 0.15

[components.image_generator]
factory = "{__name__}:TinyImageGenerator"
args = {{ z_dim = 4, size = 4 }}
inputs = {{ x = "latent" }}
trainable = true

[components.generator]
factory = "{__name__}:TinyConstraintHead"
args = {{ num_options = 5 }}
inputs = {{ x = "components.image_generator" }}

[components.discriminator]
factory = "{__name__}:TinyLogitDiscriminator"
args = {{ num_options = 5 }}
inputs = {{ x = "candidate" }}

[[objectives]]
id = "tv_smoothness"
factory = "hypergan.constraint_data:TotalVariationSmoothness"
inputs = {{ image = "components.image_generator" }}
weight = 0.05

[[objectives]]
id = "apple_prob"
factory = "hypergan.constraint_data:OptionProbability"
inputs = {{ logits = "generated" }}
detach = ["logits"]
weight = 1.0
args = {{ index = 0 }}

[prior.args]
num_particles = 16
z_dim = 4

[sampling]
generated = "components.image_generator"
count = 4
seed = 91

[training]
device = "cpu"
steps = 4
batch_size = 4
seed = 101

[metrics]
preset = "standard"
every_steps = 1
""")
        train(config_path, tmp_path / "continuous", checkpoint_every=1)
        train(config_path, tmp_path / "resumed", checkpoint_every=1, stop_after_steps=2)
        finished = resume(tmp_path / "resumed")
        assert finished["status"] == "complete"
        assert _digest(read_checkpoint(tmp_path / "continuous")[2]) == _digest(
            read_checkpoint(tmp_path / "resumed")[2]
        )
    finally:
        torch.set_num_threads(old_threads)


def test_hndl_contrastive_log_odds_operator_and_gradients():
    from hypergan.hndl_networks import build_network

    net = build_network(
        "contrastive_log_odds(target_index=1)\nlinear(8)\nleaky_relu(0.2)\nlinear(1)",
        input_shape=("B", 5),
        output_shape=("B", 1),
    )
    x = torch.tensor(
        [[0.5, 2.0, -1.0, 0.0, -0.5], [-1.0, 3.0, 0.5, -0.5, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    # Verify exact one-vs-rest log-odds calculation on the first node
    clo = net.nodes.n_n0
    margin = clo(x)
    expected = x[:, 1:2] - torch.logsumexp(torch.cat([x[:, :1], x[:, 2:]], dim=-1), dim=-1, keepdim=True)
    torch.testing.assert_close(margin, expected)

    margin.sum().backward()
    assert x.grad is not None
    # Target index 1 receives +1 gradient; distractors receive negative softmax weights summing to -1
    torch.testing.assert_close(x.grad[:, 1], torch.ones(2))
    distractor_grads = torch.cat([x.grad[:, :1], x.grad[:, 2:]], dim=-1)
    assert torch.all(distractor_grads < 0)
    torch.testing.assert_close(distractor_grads.sum(dim=-1), -torch.ones(2))

    with pytest.raises(ValueError, match="K >= 2"):
        clo(torch.zeros(2, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        build_network(
            "contrastive_log_odds(target_index=5)",
            input_shape=("B", 5),
            output_shape=("B", 1),
        )(torch.zeros(2, 5))


def test_constraint_hidden_target_data_and_option_probability_hidden_state():
    probs = [0.992, 0.002, 0.002, 0.002, 0.002]
    data_zero = ConstraintHiddenTargetData(
        probs, hidden_dim=32, jitter_std=0.0, null_std=0.0, target_index=0
    )
    batch_zero = data_zero(8)["real"]
    assert batch_zero.shape == (8, 32)
    assert batch_zero.dtype == torch.float32
    torch.testing.assert_close(batch_zero[:, 5:], torch.zeros(8, 27), atol=0.0, rtol=0.0)

    # Verify OptionProbability(input_mode="hidden_state") recovers the exact target probabilities
    for idx, expected_p in enumerate(probs):
        opt = OptionProbability(index=idx, input_mode="hidden_state", num_classes=5, target_index=0)
        torch.testing.assert_close(
            opt(batch_zero), torch.tensor(expected_p, dtype=torch.float32), atol=1e-6, rtol=1e-6
        )

    # Nonzero target_index=1
    probs_t1 = [0.05, 0.85, 0.05, 0.05]
    data_t1 = ConstraintHiddenTargetData(
        probs_t1, hidden_dim=16, jitter_std=0.0, null_std=0.05, target_index=1
    )
    rng = torch.Generator().manual_seed(42)
    batch_t1 = data_t1(12, generator=rng)["real"]
    assert batch_t1.shape == (12, 16)
    assert batch_t1[:, 4:].abs().sum() > 0.0
    for idx, expected_p in enumerate(probs_t1):
        opt = OptionProbability(index=idx, input_mode="hidden_state", num_classes=4, target_index=1)
        torch.testing.assert_close(
            opt(batch_t1), torch.tensor(expected_p, dtype=torch.float32), atol=1e-6, rtol=1e-6
        )

    spec = {
        "factory": "hypergan.constraint_data:ConstraintHiddenTargetData",
        "args": {
            "target_probs": probs_t1,
            "hidden_dim": 16,
            "jitter_std": 0.0,
            "null_std": 0.05,
            "target_index": 1,
        },
    }
    contract = data_contract(data_t1, spec)
    assert contract["supported"] is True
    assert contract["stateful"] is False
    assert contract["identity"] == data_t1.resume_identity()

    # Validation checks
    for bad_dim in (5, 3, 0, -1, True, 16.0):
        with pytest.raises(ValueError, match="hidden_dim"):
            ConstraintHiddenTargetData(probs, hidden_dim=bad_dim)
    for bad_null in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="null_std"):
            ConstraintHiddenTargetData(probs, null_std=bad_null)
    for bad_idx in (-1, 5, True, 0.0):
        with pytest.raises(ValueError, match="target_index"):
            ConstraintHiddenTargetData(probs, target_index=bad_idx)
    with pytest.raises(ValueError, match="num_classes"):
        OptionProbability(index=0, input_mode="hidden_state")
    with pytest.raises(ValueError, match="out of bounds"):
        OptionProbability(index=5, input_mode="hidden_state", num_classes=5)
    with pytest.raises(ValueError, match="target_index"):
        OptionProbability(index=0, input_mode="hidden_state", num_classes=5, target_index=5)
    with pytest.raises(ValueError, match="width"):
        OptionProbability(index=0, input_mode="hidden_state", num_classes=5)(torch.zeros(2, 4))



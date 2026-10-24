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
    JointMultiQueryConstraintData,
    MultiQueryConstraintData,
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


def test_multi_query_constraint_data_and_state_conditioned_option_probability():
    import hypergan.pretrained_providers as pp

    queries = [
        {
            "state": {"task": "apple", "aspect": "subject"},
            "question": "What is the main subject?",
            "options": ["red apple", "orange", "multiple", "noise"],
            "target_probs": [0.992, 0.002, 0.002, 0.002, 0.002],
        },
        {
            "state": {"task": "apple", "aspect": "variety"},
            "question": "Which natural red apple variety does this resemble?",
            "options": ["Red Delicious", "Honeycrisp", "unnatural purple", "monochrome"],
            "target_probs": [
                [0.992, 0.002, 0.002, 0.002, 0.002],
                [0.002, 0.992, 0.002, 0.002, 0.002],
            ],
            "weight": 2.0,
        },
    ]
    data = MultiQueryConstraintData(queries, state_embed_dim=16, jitter_std=0.0)
    pp._IMAJEV_ACTIVE_QUERY_QUEUE.pop(data._bank_signature, None)

    rng1 = torch.Generator().manual_seed(777)
    rng2 = torch.Generator().manual_seed(777)
    b1 = data(12, generator=rng1)["real"]
    q_idx_1 = pp.pop_active_query(data._bank_signature)
    b2 = data(12, generator=rng2)["real"]
    q_idx_2 = pp.pop_active_query(data._bank_signature)

    assert b1.shape == (12, 21)
    assert q_idx_1 == q_idx_2
    assert torch.equal(b1, b2)
    # Unit-norm state embedding prefix [B, 16]
    torch.testing.assert_close(b1[:, :16].norm(dim=-1), torch.ones(12), atol=1e-5, rtol=1e-5)

    # Check OptionProbability with input_mode="state_conditioned_log_odds" and multi-index support
    opt_target = OptionProbability(index=0, input_mode="state_conditioned_log_odds", num_classes=5)
    opt_alt = OptionProbability(index=1, input_mode="state_conditioned_log_odds", num_classes=5)
    opt_defects = OptionProbability(index=[2, 3], input_mode="state_conditioned_log_odds", num_classes=5)
    opt_unknown = OptionProbability(index=4, input_mode="state_conditioned_log_odds", num_classes=5)

    total_p = opt_target(b1) + opt_alt(b1) + opt_defects(b1) + opt_unknown(b1)
    torch.testing.assert_close(total_p, torch.tensor(1.0, dtype=torch.float32), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(opt_defects(b1), torch.tensor(0.004, dtype=torch.float32), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(opt_unknown(b1), torch.tensor(0.002, dtype=torch.float32), atol=1e-6, rtol=1e-6)

    spec = {
        "factory": "hypergan.constraint_data:MultiQueryConstraintData",
        "args": {"queries": queries, "state_embed_dim": 16, "jitter_std": 0.0},
    }
    contract = data_contract(data, spec)
    assert contract["supported"] is True
    assert contract["stateful"] is False
    assert contract["identity"] == data.resume_identity()

    # Validation checks
    with pytest.raises(ValueError, match="2..64"):
        MultiQueryConstraintData([queries[0]])
    with pytest.raises(ValueError, match="same number of options"):
        MultiQueryConstraintData([
            queries[0],
            {**queries[1], "options": ["a", "b"], "target_probs": [0.9, 0.05, 0.05]},
        ])
    with pytest.raises(ValueError, match="duplicates"):
        OptionProbability(index=[1, 1], input_mode="state_conditioned_log_odds", num_classes=5)


def test_joint_multi_query_constraint_data_batch_generation_and_shapes():
    """Verify JointMultiQueryConstraintData output shapes [B, Q*K] and [B, Q], bounds in (-tau, tau)."""
    queries = [
        {
            "aspect": "subject",
            "question": "What is the main subject?",
            "options": ["apple fruit", "orange or citrus", "banana or pear", "noise or static"],
            "target_probs": [0.92, 0.02, 0.02, 0.02, 0.02],
        },
        {
            "aspect": "color",
            "question": "What color is the apple?",
            "options": ["bright red", "crisp green", "golden yellow", "noise"],
            "target_probs": [
                [0.92, 0.02, 0.02, 0.02, 0.02],
                [0.02, 0.92, 0.02, 0.02, 0.02],
                [0.02, 0.02, 0.92, 0.02, 0.02],
            ],
        },
        {
            "aspect": "bites",
            "question": "How many bites are taken out?",
            "options": ["zero bites", "one bite", "multiple bites", "noise"],
            "target_probs": [
                [0.92, 0.02, 0.02, 0.02, 0.02],
                [0.02, 0.92, 0.02, 0.02, 0.02],
                [0.02, 0.02, 0.92, 0.02, 0.02],
            ],
        },
    ]
    # Q = 3 queries, K = 5 candidates each -> Q * K = 15
    data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)
    batch = data(16)

    assert "real" in batch
    assert "modes" in batch
    real = batch["real"]
    modes = batch["modes"]

    assert real.shape == (16, 15)
    assert modes.shape == (16, 3)
    assert real.dtype == torch.float32
    assert torch.isfinite(real).all()

    # Soft-capping bounds in (-tau, tau)
    assert torch.all(real > -3.0)
    assert torch.all(real < 3.0)

    # Query 0 has 1 mode -> modes[:, 0] == 0 identically
    assert torch.all(modes[:, 0] == 0)
    # Queries 1 and 2 have 3 modes -> modes in {0, 1, 2}
    assert torch.all(modes[:, 1] >= 0) and torch.all(modes[:, 1] < 3)
    assert torch.all(modes[:, 2] >= 0) and torch.all(modes[:, 2] < 3)


def test_joint_multi_query_mode_distribution_and_balance():
    """Verify that multi-mode queries sample uniformly across modes with entropy >= 0.85 ln(M)."""
    queries = [
        {
            "aspect": "color",
            "question": "Color?",
            "options": ["red", "green", "yellow", "noise"],
            "target_probs": [
                [0.92, 0.02, 0.02, 0.02, 0.02],
                [0.02, 0.92, 0.02, 0.02, 0.02],
                [0.02, 0.02, 0.92, 0.02, 0.02],
            ],
        },
    ]
    data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)
    rng = torch.Generator().manual_seed(9999)
    N = 1200
    batch = data(N, generator=rng)
    modes = batch["modes"][:, 0]

    counts = torch.bincount(modes, minlength=3).tolist()
    # Expectation for each mode is N/3 = 400. Assert within +/- 75
    for c in counts:
        assert 325 <= c <= 475, f"Mode count {c} deviated too far from expectation 400"

    probs = [c / N for c in counts]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(3.0)
    assert entropy >= 0.85 * max_entropy, f"Marginal entropy {entropy} < 0.85 * {max_entropy}"


def test_joint_multi_query_independence_across_batch_and_queries():
    """Verify statistical independence of sampled modes between samples and between queries."""
    queries = [
        {
            "aspect": "color",
            "question": "Color?",
            "options": ["red", "green", "yellow", "noise"],
            "target_probs": [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.03, 0.90, 0.03, 0.02, 0.02],
                [0.03, 0.03, 0.90, 0.02, 0.02],
            ],
        },
        {
            "aspect": "bites",
            "question": "Bites?",
            "options": ["zero", "one", "multiple", "noise"],
            "target_probs": [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.03, 0.90, 0.03, 0.02, 0.02],
                [0.03, 0.03, 0.90, 0.02, 0.02],
            ],
        },
    ]
    data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)

    # 1. Sample-to-sample independence: batch does not collapse to a single mode combination
    rng = torch.Generator().manual_seed(42)
    batch_small = data(32, generator=rng)
    modes_small = batch_small["modes"]
    unique_combinations = len(set(tuple(row.tolist()) for row in modes_small))
    assert unique_combinations >= 6, f"Expected diverse mode combinations, got only {unique_combinations}"

    # 2. Query-to-query independence within sample: contingency table test
    N = 1800
    batch_large = data(N, generator=rng)
    m1 = batch_large["modes"][:, 0]
    m2 = batch_large["modes"][:, 1]

    contingency = torch.zeros(3, 3, dtype=torch.float32)
    for i in range(N):
        contingency[m1[i], m2[i]] += 1

    joint_probs = contingency / N
    p1 = joint_probs.sum(dim=1, keepdim=True)
    p2 = joint_probs.sum(dim=0, keepdim=True)
    indep_probs = p1 @ p2

    max_dev = (joint_probs - indep_probs).abs().max().item()
    assert max_dev < 0.035, f"Query independence violation: max joint deviation {max_dev} >= 0.035"


def test_joint_multi_query_shape_compatibility_with_joint_log_odds():
    """Verify that JointMultiQueryConstraintData targets match joint_log_odds shape and pass through discriminator."""
    from hypergan.hndl_networks import build_network

    queries = [
        {
            "aspect": f"query_{i}",
            "question": f"Question {i}?",
            "options": ["opt0", "opt1", "opt2", "noise"],
            "target_probs": [0.92, 0.02, 0.02, 0.02, 0.02],
        }
        for i in range(6)
    ]
    data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.05)
    real_targets = data(8)["real"]
    assert real_targets.shape == (8, 30)

    critic_net = build_network(
        "linear(32)\nleaky_relu(0.2)\nlinear(1)",
        input_shape=("B", 30),
        output_shape=("B", 1),
    )
    score_real = critic_net(real_targets)
    assert score_real.shape == (8, 1)
    assert torch.isfinite(score_real).all()

    # Simulated joint_log_odds output of shape (8, 30)
    fake_joint_log_odds = torch.tanh(torch.randn(8, 30)) * 3.0
    score_fake = critic_net(fake_joint_log_odds)
    assert score_fake.shape == (8, 1)
    assert torch.isfinite(score_fake).all()

    loss = score_real.mean() - score_fake.mean()
    loss.backward()
    assert critic_net.nodes.n_n0.weight.grad is not None


def test_joint_multi_query_removal_of_global_queue_dependency():
    """Verify that JointMultiQueryConstraintData does not touch or require _IMAJEV_ACTIVE_QUERY_QUEUE."""
    import hypergan.pretrained_providers as pp

    queries = [
        {
            "aspect": "q1",
            "question": "Question 1?",
            "options": ["a", "b", "c", "noise"],
            "target_probs": [0.92, 0.02, 0.02, 0.02, 0.02],
        },
        {
            "aspect": "q2",
            "question": "Question 2?",
            "options": ["x", "y", "z", "noise"],
            "target_probs": [0.92, 0.02, 0.02, 0.02, 0.02],
        },
    ]
    data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)

    if hasattr(pp, "_IMAJEV_ACTIVE_QUERY_QUEUE"):
        pp._IMAJEV_ACTIVE_QUERY_QUEUE.clear()

    batch = data(16)
    assert batch["real"].shape == (16, 10)

    if hasattr(pp, "_IMAJEV_ACTIVE_QUERY_QUEUE"):
        assert len(pp._IMAJEV_ACTIVE_QUERY_QUEUE) == 0

    spec = {
        "factory": "hypergan.constraint_data:JointMultiQueryConstraintData",
        "args": {"queries": queries, "tau": 3.0, "jitter_std": 0.0},
    }
    contract = data_contract(data, spec)
    assert contract["supported"] is True
    assert contract["stateful"] is False
    assert contract["identity"] == data.resume_identity()


def test_joint_multi_query_rng_reproducibility_and_jitter():
    """Verify deterministic reproducibility with torch.Generator and proper jitter behavior."""
    queries = [
        {
            "aspect": "color",
            "question": "Color?",
            "options": ["red", "green", "yellow", "noise"],
            "target_probs": [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.03, 0.90, 0.03, 0.02, 0.02],
            ],
        },
    ]
    data_jitter = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.15)
    g1 = torch.Generator().manual_seed(777)
    g2 = torch.Generator().manual_seed(777)
    b1 = data_jitter(16, generator=g1)
    b2 = data_jitter(16, generator=g2)
    assert torch.equal(b1["real"], b2["real"])
    assert torch.equal(b1["modes"], b2["modes"])

    data_clean = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)
    b_clean = data_clean(32, generator=torch.Generator().manual_seed(123))
    mode_0_rows = [b_clean["real"][i] for i in range(32) if b_clean["modes"][i, 0] == 0]
    if len(mode_0_rows) > 1:
        for row in mode_0_rows[1:]:
            torch.testing.assert_close(row, mode_0_rows[0], atol=1e-6, rtol=0.0)


def test_joint_multi_query_validation():
    """Verify input validation for JointMultiQueryConstraintData."""
    valid_q = [
        {"question": "Q1?", "options": ["a", "b"], "target_probs": [0.9, 0.05, 0.05]},
        {"question": "Q2?", "options": ["c", "d"], "target_probs": [0.9, 0.05, 0.05]},
    ]
    for bad_queries in ([], "not_a_list", [valid_q[0]] * 65):
        with pytest.raises(ValueError):
            JointMultiQueryConstraintData(bad_queries)
    for bad_tau in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="tau"):
            JointMultiQueryConstraintData(valid_q, tau=bad_tau)
    for bad_jitter in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="jitter_std"):
            JointMultiQueryConstraintData(valid_q, jitter_std=bad_jitter)
    data = JointMultiQueryConstraintData(valid_q)
    for bad_bs in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="batch_size"):
            data(bad_bs)


def test_option_probability_joint_log_odds():
    """Verify OptionProbability with input_mode='joint_log_odds' across queries."""
    # 2 queries, 4 classes each -> total width 8
    # Q0: u = [5.0, -5.0, -5.0, -5.0], Q1: u = [-5.0, 5.0, -5.0, -5.0]
    u = torch.tensor([[5.0, -5.0, -5.0, -5.0, -5.0, 5.0, -5.0, -5.0]], dtype=torch.float32)

    opt_q0_c0 = OptionProbability(index=0, input_mode="joint_log_odds", num_classes=4, query_index=0)
    opt_q0_c1 = OptionProbability(index=1, input_mode="joint_log_odds", num_classes=4, query_index=0)
    opt_q1_c1 = OptionProbability(index=1, input_mode="joint_log_odds", num_classes=4, query_index=1)
    opt_q1_multi = OptionProbability(index=[1, 2], input_mode="joint_log_odds", num_classes=4, query_index=1)

    p0 = opt_q0_c0(u)
    p1 = opt_q0_c1(u)
    p2 = opt_q1_c1(u)
    p_multi = opt_q1_multi(u)

    assert p0.item() > 0.95
    assert p1.item() < 0.05
    assert p2.item() > 0.95
    assert p_multi.item() >= p2.item()

    # Slicing error when tensor width < (query_index + 1) * num_classes
    opt_q2 = OptionProbability(index=0, input_mode="joint_log_odds", num_classes=4, query_index=2)
    with pytest.raises(ValueError, match="tensor width"):
        opt_q2(u)

    # Validation errors
    with pytest.raises(ValueError, match="query_index"):
        OptionProbability(index=0, input_mode="joint_log_odds", num_classes=4, query_index=-1)
    with pytest.raises(ValueError, match="query_index"):
        OptionProbability(index=0, input_mode="joint_log_odds", num_classes=4, query_index=True)


def test_optical_wavefront_prior_flat_image_is_zero():
    """Verify OpticalWavefrontPrior gives zero loss and zero gradient on constant/flat images."""
    from hypergan.constraint_data import OpticalWavefrontPrior

    prior = OpticalWavefrontPrior()
    flat = (torch.ones(2, 3, 64, 64) * 0.7).requires_grad_(True)
    losses = prior(flat, return_dict=True)
    assert float(losses["loss_seam"].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(losses["loss_biharmonic"].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(losses["loss_psd"].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(losses["total_loss"].item()) == pytest.approx(0.0, abs=1e-6)

    losses["total_loss"].backward()
    assert flat.grad is not None
    assert torch.all(flat.grad.abs() < 1e-6)


def test_optical_wavefront_prior_linear_ramp_interior_curvature():
    """Verify that biharmonic curvature is zero on linear intensity ramps (preserving Lambertian shading)."""
    import torch.nn.functional as F

    ramp = torch.linspace(0, 1, 64).view(1, 1, 1, 64).repeat(1, 3, 64, 1)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    lap_interior = F.conv2d(ramp, k, padding=0, groups=3)
    assert float(lap_interior.abs().max().item()) == pytest.approx(0.0, abs=1e-6)


def test_optical_wavefront_prior_patch_seam_spike_detection():
    """Verify patch-seam penalty detects 8px/16px ViT boundary steps and produces restorative gradients."""
    from hypergan.constraint_data import OpticalWavefrontPrior

    prior = OpticalWavefrontPrior(seam_threshold=1.05)
    y = torch.linspace(-1, 1, 64).unsqueeze(1).repeat(1, 64)
    x = torch.linspace(-1, 1, 64).unsqueeze(0).repeat(64, 1)
    sphere = (1.0 - torch.clamp(x**2 + y**2, max=1.0)).sqrt().unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)

    loss_smooth = prior(sphere, return_dict=True)
    assert loss_smooth["R_seam"].item() <= 1.05
    assert loss_smooth["loss_seam"].item() == 0.0

    # Inject seam step along 16px seam
    spiked = sphere.clone().requires_grad_(True)
    with torch.no_grad():
        spiked[:, :, 16, :] += 0.5
        spiked[:, :, :, 16] += 0.5
    loss_spiked = prior(spiked, return_dict=True)
    assert loss_spiked["R_seam"].item() > 1.05
    assert loss_spiked["loss_seam"].item() > 0.0
    loss_spiked["loss_seam"].backward()
    assert spiked.grad is not None and torch.isfinite(spiked.grad).all()


def test_optical_wavefront_prior_radial_psd_falloff():
    """Verify PSD prior penalizes high-frequency white noise much more than smooth 1/f fields."""
    from hypergan.constraint_data import OpticalWavefrontPrior
    import torch.fft as fft

    prior = OpticalWavefrontPrior(weight_seam=0.0, weight_biharmonic=0.0, weight_psd=1.0)
    H, W = 64, 64
    fy = fft.fftfreq(H).unsqueeze(1) * H
    fx = fft.rfftfreq(W).unsqueeze(0) * W
    f_rad = torch.sqrt(fy ** 2 + fx ** 2)
    amp = 1.0 / (1.0 + f_rad)
    F_smooth = amp * torch.exp(1j * torch.rand(H, W // 2 + 1) * 6.28)
    smooth_img = fft.irfft2(F_smooth, s=(H, W), norm="ortho").unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)

    loss_smooth = prior(smooth_img)
    noise_img = torch.randn(2, 3, 64, 64)
    loss_noise = prior(noise_img)

    assert float(loss_noise.item()) > float(loss_smooth.item()) * 10.0


def test_optical_wavefront_prior_laplacian_kernels():
    """Verify both 5-point and 9-point Laplacian kernels compute curvature and valid backward gradients."""
    from hypergan.constraint_data import OpticalWavefrontPrior

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    p5 = OpticalWavefrontPrior(laplacian_kernel="5point", weight_seam=0.0, weight_psd=0.0)
    p9 = OpticalWavefrontPrior(laplacian_kernel="9point", weight_seam=0.0, weight_psd=0.0)

    l5 = p5(x)
    l9 = p9(x)
    assert float(l5.item()) > 0.0
    assert float(l9.item()) > 0.0
    l5.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_optical_wavefront_prior_scalar_vs_dict():
    """Verify OpticalWavefrontPrior returns scalar tensor or diagnostic dictionary."""
    from hypergan.constraint_data import OpticalWavefrontPrior

    prior = OpticalWavefrontPrior()
    x = torch.randn(2, 3, 64, 64)

    scalar = prior(x)
    assert isinstance(scalar, torch.Tensor)
    assert scalar.numel() == 1

    d = prior(x, return_dict=True)
    assert isinstance(d, dict)
    for key in ("total_loss", "loss_seam", "loss_biharmonic", "loss_psd", "R_seam"):
        assert key in d
        assert isinstance(d[key], torch.Tensor)
        assert d[key].numel() == 1
    assert float(d["total_loss"].item()) == pytest.approx(float(scalar.item()), rel=1e-5)


def test_optical_wavefront_prior_differentiability_and_gradcheck():
    """Verify analytical autograd gradients match numerical finite-difference Jacobian via gradcheck."""
    from hypergan.constraint_data import OpticalWavefrontPrior

    p_bi = OpticalWavefrontPrior(weight_seam=0.0, weight_psd=0.0, weight_biharmonic=1.0, delta=0.05)
    x = torch.randn(1, 3, 16, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda img: p_bi(img), (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    spiked = x.clone().detach()
    with torch.no_grad():
        spiked[:, :, 8, :] += 2.0
        spiked[:, :, :, 8] += 2.0
    spiked.requires_grad = True
    p_seam = OpticalWavefrontPrior(weight_seam=1.0, weight_biharmonic=0.0, weight_psd=0.0, seam_threshold=1.05)
    assert torch.autograd.gradcheck(lambda img: p_seam(img), (spiked,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_optical_wavefront_prior_validation():
    """Verify input argument and tensor dimension validations."""
    from hypergan.constraint_data import OpticalWavefrontPrior

    for bad_weight in (-0.1, float("nan"), True):
        with pytest.raises(ValueError, match="weight_seam"):
            OpticalWavefrontPrior(weight_seam=bad_weight)
    with pytest.raises(ValueError, match="patch_size"):
        OpticalWavefrontPrior(patch_size=1)
    with pytest.raises(ValueError, match="secondary_patch_size"):
        OpticalWavefrontPrior(secondary_patch_size=0)
    with pytest.raises(ValueError, match="seam_threshold"):
        OpticalWavefrontPrior(seam_threshold=0.0)
    with pytest.raises(ValueError, match="laplacian_kernel"):
        OpticalWavefrontPrior(laplacian_kernel="7point")
    with pytest.raises(ValueError, match="alpha"):
        OpticalWavefrontPrior(alpha=-1.0)
    with pytest.raises(ValueError, match="margin_db"):
        OpticalWavefrontPrior(margin_db=-0.5)
    with pytest.raises(ValueError, match="padding_mode"):
        OpticalWavefrontPrior(padding_mode="circular")
    with pytest.raises(ValueError, match="delta"):
        OpticalWavefrontPrior(delta=-0.01)

    p = OpticalWavefrontPrior()
    with pytest.raises(ValueError, match="requires an image"):
        p()
    with pytest.raises(ValueError, match="4D floating-point"):
        p(torch.zeros(3, 64, 64))
    with pytest.raises(ValueError, match="C=3"):
        p(torch.zeros(2, 1, 64, 64))
    with pytest.raises(ValueError, match="dimensions"):
        p(torch.zeros(2, 3, 8, 8))


def test_optical_wavefront_prior_hypergan_training_loop_integration():
    """Verify OpticalWavefrontPrior in a complete HyperGAN ReferenceTrainer step."""
    from hypergan.config import DEFAULT, resolve_config
    from hypergan.training import ReferenceTrainer

    config = copy.deepcopy(DEFAULT)
    config["data"] = {
        "factory": "hypergan.constraint_data:ConstraintTargetData",
        "args": {"target_probs": [0.92, 0.02, 0.02, 0.02, 0.02], "mode": "logits", "jitter_std": 0.0},
    }
    config["components"]["image_generator"] = {
        "factory": __name__ + ":TinyImageGenerator",
        "args": {"z_dim": 4, "size": 32},
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
            "id": "optical_wavefront",
            "factory": "hypergan.constraint_data:OpticalWavefrontPrior",
            "args": {"weight_seam": 1.0, "weight_biharmonic": 0.1, "weight_psd": 0.01},
            "inputs": {"image": "components.image_generator"},
            "weight": 0.05,
        }
    ]
    config["prior"]["args"].update(num_particles=16, z_dim=4)
    config["sampling"].update(generated="components.image_generator", count=4, seed=42)
    config["training"].update(device="cpu", steps=2, batch_size=4, seed=42)
    config["metrics"] = {"preset": "standard", "every_steps": 1}

    resolved = resolve_config(config)
    trainer = ReferenceTrainer(resolved)
    row, batch = trainer.update()
    assert len(row["objectives"]) == 1
    assert row["objectives"][0] > 0.0


def test_marginal_shannon_entropy_regularizer_collapsed_vs_diverse():
    """Verify that mode collapse triggers a positive penalty with non-zero gradients,
    while balanced diverse batches incur zero penalty."""
    from hypergan.constraint_data import MarginalShannonEntropyRegularizer

    reg = MarginalShannonEntropyRegularizer(
        query_indices=[1, 2], num_classes=5, num_modes=3, min_entropy_ratio=0.85, hinge=True
    )
    # Batch size 16, 2 queries * 5 classes = 10 classes (query 1: 5..9, query 2: 10..14)
    # Collapsed batch: all images have option 0 high
    u_col = torch.full((16, 15), -5.0, requires_grad=True)
    u_col.data[:, 5] = 5.0   # query 1 option 0
    u_col.data[:, 10] = 5.0  # query 2 option 0

    loss_col = reg(u_col)
    assert loss_col.item() > 0.5, "Collapsed batch failed to trigger entropy penalty"
    loss_col.backward()
    assert u_col.grad is not None and u_col.grad.norm().item() > 0.01

    # Diverse batch: perfectly balanced across 3 modes
    u_div = torch.full((15, 15), -5.0)
    for i in range(15):
        m = i % 3
        u_div[i, 5 + m] = 5.0
        u_div[i, 10 + m] = 5.0
    loss_div = reg(u_div)
    assert loss_div.item() == 0.0, "Diverse batch improperly penalized under hinge loss"


def test_marginal_shannon_entropy_regularizer_validation():
    """Verify validation of constructor arguments."""
    from hypergan.constraint_data import MarginalShannonEntropyRegularizer

    for bad_q in ([], "not_a_list", [-1]):
        with pytest.raises(ValueError):
            MarginalShannonEntropyRegularizer(query_indices=bad_q)
    for bad_classes in (0, 1, 1.5):
        with pytest.raises(ValueError):
            MarginalShannonEntropyRegularizer(num_classes=bad_classes)
    for bad_ratio in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            MarginalShannonEntropyRegularizer(min_entropy_ratio=bad_ratio)


def test_normalized_vision_cosine_repulsion_identical_vs_orthogonal():
    """Verify that identical vectors incur (1 - margin) penalty, orthogonal incur 0."""
    from hypergan.constraint_data import NormalizedVisionCosineRepulsion

    rep = NormalizedVisionCosineRepulsion(margin=0.88)

    # B=1 returns 0.0
    assert rep(torch.randn(1, 2560)).item() == 0.0

    # B=4 identical vectors: cos(v_i, v_j) = 1.0 -> penalty = 1.0 - 0.88 = 0.12
    v_col = torch.randn(1, 2560).expand(4, -1).clone().requires_grad_(True)
    loss_col = rep(v_col)
    assert abs(loss_col.item() - 0.12) < 1e-4

    # B=4 orthogonal vectors: cos(v_i, v_j) = 0.0 -> penalty = 0.0
    v_ortho = torch.eye(4, 2560)
    assert rep(v_ortho).item() == 0.0

    # Gradient flow
    v_rand = torch.randn(4, 2560, requires_grad=True)
    rep_loss = rep(v_rand, margin=0.0)  # force active penalty
    rep_loss.backward()
    assert v_rand.grad is not None and v_rand.grad.norm().item() > 0


def test_normalized_vision_cosine_repulsion_validation():
    """Verify input validation for NormalizedVisionCosineRepulsion."""
    from hypergan.constraint_data import NormalizedVisionCosineRepulsion

    for bad_margin in (-0.1, 1.0, 1.5, float('nan')):
        with pytest.raises(ValueError):
            NormalizedVisionCosineRepulsion(margin=bad_margin)
    rep = NormalizedVisionCosineRepulsion()
    for bad_input in (torch.randn(10), torch.randn(2, 2, 2)):
        with pytest.raises(ValueError):
            rep(bad_input)

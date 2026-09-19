"""Numerical contract and native-artifact checks, against ParticleGAN 0.5.0."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from particlegan import get_recipe

from hypergan.artifacts import sample, save_bundle
from hypergan.config import DEFAULT, load_config, resolve_config, write_default
from hypergan.training import ReferenceTrainer, train


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_modules_equal(left, right, gradients=False):
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if gradients:
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_one_update_matches_upstream_loop_gradients_weights_prior_and_ema():
    trainer = ReferenceTrainer(resolve_config({}))
    generator = copy.deepcopy(trainer.graph.models["generator"])
    critic = copy.deepcopy(trainer.graph.models["discriminator"])
    prior = copy.deepcopy(trainer.prior)
    ema_g, ema_prior = copy.deepcopy(generator), copy.deepcopy(prior)
    recipe = get_recipe(total_steps=5, batch_size=16)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    gan, penalty, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    rng = torch.Generator().manual_seed(9)
    real = torch.randn(16, 2, generator=rng)
    ids = torch.tensor([0, 1, 2, 2, 4, 6, 7, 7, 9, 11, 12, 12, 14, 15, 16, 17])
    # Direct transcription of the pinned upstream caller-owned update: fixed draws
    # isolate numerical parity from this runtime's separately named RNG streams.
    fake = generator(prior(ids))
    opt_d.zero_grad(set_to_none=True)
    d_loss = gan.d_loss(critic(real), critic(fake.detach()))
    d_loss = d_loss + penalty(critic, real, fake.detach(), step=1)
    d_loss.backward()
    opt_d.step()
    critic.requires_grad_(False)
    opt_g.zero_grad(set_to_none=True)
    g_loss = gan.g_loss(critic(fake), critic(real).detach())
    p_loss = spread(prior.z[ids.unique()])
    (g_loss + p_loss).backward()
    opt_g.step()
    critic.requires_grad_(True)
    with torch.no_grad():
        for target, current in ((ema_g, generator), (ema_prior, prior)):
            for average, online in zip(target.parameters(), current.parameters()):
                average.lerp_(online, 1 - recipe.ema_decay)
    row, _ = trainer.update({"real": real}, (trainer.prior(ids), ids))
    assert row["d_loss"] == float(d_loss.detach())
    assert row["g_loss"] == float((g_loss + p_loss).detach())
    assert_modules_equal(trainer.graph.models["generator"], generator, gradients=True)
    assert_modules_equal(trainer.graph.models["discriminator"], critic, gradients=True)
    assert_modules_equal(trainer.prior, prior, gradients=True)
    assert_modules_equal(trainer.ema_graph.models["generator"], ema_g)
    assert_modules_equal(trainer.ema_prior, ema_prior)


def test_run_and_fresh_process_inference(tmp_path):
    config = write_default(tmp_path / "project", device="cpu")
    run = tmp_path / "run"
    manifest = train(config, run)
    assert manifest["status"] == "complete" and manifest["steps"] == 5
    assert manifest["qualification"]["scope"] == "numerical-reference"
    assert manifest["resume_supported"]
    assert manifest["source"]["particlegan_distribution_commit"] is None
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "start" and events[-1]["event"] == "complete"
    original = json.loads(Path(manifest["sample_path"]).read_text())
    output = tmp_path / "reloaded.json"
    code = "from hypergan.artifacts import sample; import sys; sample(sys.argv[1],count=256,seed=123,output=sys.argv[2]); assert 'hypergan.training' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(run), str(output)], cwd=tmp_path, check=True)
    assert json.loads(output.read_text()) == original
    state = torch.load(manifest["bundle_path"], weights_only=True)
    assert "discriminator" not in state["model_states"]
    assert "optimizer" not in state
    with pytest.raises(FileExistsError):
        train(config, run)


def test_paired_encoder_custom_factory_and_objective_have_effect(tmp_path):
    example = Path(__file__).parents[2] / "examples" / "paired-linear.toml"
    config = load_config(example)
    config["components"]["encoder"]["factory"] = "torch.nn:Linear"
    config["objectives"][0]["factory"] = "torch.nn:MSELoss"
    trainer = ReferenceTrainer(config)
    old = [p.detach().clone() for p in trainer.graph.models["encoder"].parameters()]
    row, batch = trainer.update()
    assert row["objectives"][0] > 0
    assert any(not torch.equal(before, after) for before, after in zip(old, trainer.graph.models["encoder"].parameters()))
    z = torch.zeros(8, 4)
    a = trainer.graph.generate(z, {"condition": torch.zeros(8, 2)})["generated"]
    b = trainer.graph.generate(z, {"condition": torch.ones(8, 2)})["generated"]
    assert not torch.equal(a, b)
    manifest = train(example, tmp_path / "paired")
    assert manifest["qualification"]["status"] == "unqualified"
    first = sample(tmp_path / "paired", count=8, seed=25, output=tmp_path / "a.json", inputs={"condition": torch.ones(8, 2)})
    second = sample(tmp_path / "paired", count=8, seed=25, output=tmp_path / "b.json", inputs={"condition": torch.zeros(8, 2)})
    assert json.loads(first.read_text())["samples"] != json.loads(second.read_text())["samples"]


def test_invalid_custom_constructor_argument_fails_instead_of_being_ignored():
    raw = copy.deepcopy(DEFAULT)
    raw["components"]["generator"]["args"]["typo"] = 1
    with pytest.raises(ValueError, match="Invalid constructor"):
        ReferenceTrainer(resolve_config(raw))


def test_unavailable_custom_constructor_is_actionable():
    raw = copy.deepcopy(DEFAULT)
    raw["components"]["generator"]["factory"] = "torch.nn:DoesNotExist"
    with pytest.raises(ValueError, match="Unavailable custom constructor"):
        ReferenceTrainer(resolve_config(raw))


def test_tampered_bundle_rejected(tmp_path):
    config = write_default(tmp_path / "project", device="cpu")
    manifest = train(config, tmp_path / "run")
    with Path(manifest["bundle_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        sample(tmp_path / "run")


def test_integer_conditioning_survives_native_bundle(tmp_path):
    raw = copy.deepcopy(DEFAULT)
    raw["components"]["encoder"] = {"factory": "torch.nn:Embedding", "args": {"num_embeddings": 4, "embedding_dim": 2}, "inputs": {"input": "batch.condition"}}
    raw["components"]["generator"]["args"]["input_dim"] = 6
    raw["components"]["generator"]["inputs"]["condition"] = "components.encoder"
    trainer = ReferenceTrainer(resolve_config(raw))
    batch = {"real": torch.zeros(16, 2), "condition": torch.arange(16) % 4}
    trainer.update(batch)
    save_bundle(tmp_path, trainer, batch)
    path = sample(tmp_path, count=8)
    result = json.loads(path.read_text())
    assert result["shape"] == [8, 2]
    assert result["inputs"]["condition"] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_interrupted_manifest_is_terminal(tmp_path, monkeypatch):
    path = write_default(tmp_path / "project", device="cpu")
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(ReferenceTrainer, "update", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train(path, tmp_path / "run")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["resume_supported"]
    assert manifest["last_durable_step"] == 0

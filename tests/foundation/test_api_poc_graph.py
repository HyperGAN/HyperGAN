"""The hypergan.graph proof of concept: lowering, naming, roles, files, adapters, runs."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("particlegan")
pytest.importorskip("hndl")

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "api" / "graph"
sys.path.insert(0, str(EXAMPLES))

import hypergan.graph as hg  # noqa: E402
from hypergan.config import DEFAULT, config_values, fingerprint, load_config, resolve_config  # noqa: E402
from hypergan.graph import toml_writer  # noqa: E402
from hypergan.graph.adapters import ItemDataset  # noqa: E402
import my_project  # noqa: E402
from my_project import ConditionalGenerator, Decoder, Encoder, Generator, PairCritic, RingItems  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CRITIC = str(EXAMPLES / "critic.hndl")


def simple(**kwargs):
    x, z = hg.data(), hg.latent(4)
    fake = hg.net(Generator, z=z)
    d = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate)
    return hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake)], device="cpu", **kwargs)


def multi(steps=3, samplers=False):
    x, c, z = hg.data(), hg.data("condition"), hg.latent(4)
    code = hg.net(Encoder, x=c, code=4)
    fake = hg.net(ConditionalGenerator, z=z, code=code["mu"])
    recon = hg.net(Decoder, code=code["mu"])
    pair = hg.net(PairCritic, x=hg.candidate, condition=c)
    marginal = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate, name="marginal")
    return hg.model(hg.paired_linear(), [
        hg.adversarial(pair, real=x, fake=fake),
        hg.adversarial(marginal, real=x, fake=fake, weight=0.5, penalty=1.0),
        hg.l1(recon, c, name="reconstruction"),
        hg.mse(fake, x, weight=0.1, name="paired"),
    ], steps=steps, batch_size=8, device="cpu",
        samplers=[hg.sampler(my_project.points_and_codes, points=fake, code=code, count=6)] if samplers else ())


# TOML writer -------------------------------------------------------------------

def test_toml_writer_round_trips_the_default_config_and_awkward_strings():
    def drop_none(value):
        if isinstance(value, dict):
            return {k: drop_none(v) for k, v in value.items() if v is not None}
        return value
    value = drop_none(deepcopy(DEFAULT))
    value["objectives"] = [{"id": "a", "factory": "l1", "inputs": {"input": "generated", "target": "batch.real"},
                            "weight": 0.25, "detach": ["target"], "args": {"nested": {"deep": [1, 2.5, "x"]}}}]
    value["metrics"]["overrides"] = {"loss/total": {"enabled": False}}
    value["name"] = 'quote " back \\ tab \t unicode é del \x7f'
    value["components"]["generator"]["args"]["source"] = "linear(4)\n# it's '' fine\nlinear()\n"
    value["components"]["discriminator"]["args"]["source"] = "a\r\nb'''c"
    assert tomllib.loads(toml_writer.dumps(value)) == value
    with pytest.raises(ValueError, match="null"):
        toml_writer.dumps({"a": None})
    with pytest.raises(ValueError, match="finite"):
        toml_writer.dumps({"a": float("nan")})


# Lowering, naming and roles -------------------------------------------------------

def test_simple_model_lowers_to_named_components_and_matches_the_shorthand():
    model = simple(steps=5)
    components = model.config["components"]
    assert list(components) == ["generator", "discriminator"]
    assert components["generator"] == {"factory": "my_project:Generator", "inputs": {"z": "latent"}}
    assert components["discriminator"]["args"]["input_shape"] == ["B", 2]
    assert model.roles == {"generator": "generator", "discriminator": "critic"}
    shorthand = hg.model(hg.gaussian_grid(), generator=Generator,
                         discriminator=hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate),
                         latent=4, device="cpu", steps=5)
    assert hg.fingerprint(shorthand) == hg.fingerprint(model)


def test_shorthand_with_hndl_sources_reproduces_the_reference_recipe():
    source = DEFAULT["components"]["generator"]["args"]["source"]
    model = hg.model(hg.gaussian_grid(), generator=source, discriminator=source,
                     latent=hg.latent(4, particles=20000), name="reference/100gaussians")
    assert hg.fingerprint(model) == fingerprint(resolve_config({}))


def test_multi_network_roles_come_from_the_losses():
    model = multi()
    assert model.roles == {"generator": "generator", "discriminator": "critic", "encoder": "generator-side",
                           "marginal": "critic", "decoder": "generator-side"}
    config = model.config
    assert config["components"]["generator"]["inputs"] == {"z": "latent", "code": "components.encoder.mu"}
    assert config["adversarial_terms"] == [{"id": "marginal", "component": "marginal", "weight": 0.5, "real": "batch.real",
                                            "fake": "generated", "penalty": True, "penalty_coeff": 1.0}]
    assert [t["id"] for t in config["objectives"]] == ["reconstruction", "paired"]
    assert config["objectives"][0]["inputs"] == {"input": "components.decoder", "target": "batch.condition"}


def test_names_never_come_from_call_order():
    x, z = hg.data(), hg.latent(4)
    fake = hg.net(Generator, z=z)
    d = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate)
    one = hg.net(Decoder, code=z)
    two = hg.net(Decoder, code=z)
    losses = [hg.adversarial(d, real=x, fake=fake), hg.l1(one, x, name="a"), hg.l1(two, x, name="b")]
    with pytest.raises(ValueError, match="would be named 'decoder'.*pass name="):
        hg.model(hg.gaussian_grid(), losses)
    unnamed = hg.hndl("linear()", shape=["B", 2], x=z)
    with pytest.raises(ValueError, match="needs name="):
        hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake), hg.l1(unnamed, x)])
    renamed = hg.net(PairCritic, x=hg.candidate, condition=x, name="critic")
    with pytest.raises(ValueError, match="checkpoint key is 'discriminator'"):
        hg.model(hg.gaussian_grid(), [hg.adversarial(renamed, real=x, fake=fake)])
    with pytest.raises(ValueError, match="publish as loss/objectives/l1"):
        hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake), hg.l1(fake, x), hg.l1(fake, x)])


def test_conditioning_only_networks_are_frozen_and_reported():
    x, c, z = hg.data(), hg.data("condition"), hg.latent(4)
    fake = hg.net(ConditionalGenerator, z=z, code=hg.net(Encoder, x=c, code=4)["mu"])
    features = hg.net(Decoder, code=c, out=2, name="features")
    d = hg.net(PairCritic, x=hg.candidate, condition=features)
    model = hg.model(hg.paired_linear(), [hg.adversarial(d, real=x, fake=fake)], device="cpu")
    assert model.config["components"]["features"]["trainable"] is False
    assert model.roles["features"] == "frozen"
    assert any("features is frozen" in note for note in model.notes)


def test_user_code_must_be_importable_by_every_process():
    class Local(torch.nn.Module):
        def forward(self, z):
            return z
    with pytest.raises(ValueError, match="local to a function"):
        hg.net(Local, z=hg.latent(4))
    Main = type("Main", (), {"__module__": "__main__", "__qualname__": "Main"})
    with pytest.raises(ValueError, match="importable module"):
        hg.net(Main, z=hg.latent(4))
    with pytest.raises(ValueError, match="no null"):
        hg.net(Generator, z=hg.latent(4), hidden=None)


def test_one_prior_per_model_and_real_field_contracts():
    x = hg.data()
    fake = hg.net(ConditionalGenerator, z=hg.latent(4), code=hg.latent(4))
    d = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate)
    with pytest.raises(ValueError, match="one prior per model"):
        hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake)])
    with pytest.raises(ValueError, match="field 'real'"):
        hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=hg.data("x"), fake=hg.net(Generator, z=hg.latent(4)))])


# Files ------------------------------------------------------------------------

def test_config_round_trip_keeps_the_fingerprint_and_refuses_edited_files(tmp_path):
    model = multi()
    path = hg.save(model, tmp_path / "model.toml")
    text = path.read_text()
    assert 'file = "' in text and "linear(64)" not in text, "the .hndl file is referenced, not copied"
    loaded = hg.load(path)
    assert hg.fingerprint(loaded) == hg.fingerprint(model) == fingerprint(load_config(path))
    assert loaded.roles == model.roles
    hg.save(model, path)
    path.write_text(text.replace("weight = 0.1", "weight = 0.2"))
    with pytest.raises(FileExistsError, match="edited"):
        hg.save(model, path)
    assert hg.fingerprint(hg.load(path)) != hg.fingerprint(model)
    hg.save(model, path, overwrite=True)
    other = tmp_path / "hand.toml"
    other.write_text('schema_version = 1\n')
    with pytest.raises(FileExistsError, match="not written by hg.save"):
        hg.save(model, other)


def test_samplers_are_observation_not_numerical_identity():
    fake_ref = {}
    def build(samplers):
        x, z = hg.data(), hg.latent(4)
        fake = hg.net(Generator, z=z)
        fake_ref["fake"] = fake
        d = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate)
        return hg.model(hg.gaussian_grid(), [hg.adversarial(d, real=x, fake=fake)], samplers=samplers(fake))
    plain_model = build(lambda fake: ())
    sampled = build(lambda fake: [hg.sampler(my_project.scatter, points=fake)])
    assert hg.fingerprint(plain_model) == hg.fingerprint(sampled)
    resolved = hg.resolve(sampled)
    assert config_values(resolved)["samplers"]["scatter"] == {
        "factory": "my_project:scatter", "args": {}, "inputs": {"points": "generated"}, "count": 16, "seed": 123}
    raw = deepcopy(sampled.config)
    raw["samplers"]["scatter"]["inputs"] = {"points": "candidate"}
    with pytest.raises(ValueError, match="candidate"):
        resolve_config(raw)


# Item-level data ----------------------------------------------------------------

def test_item_dataset_owns_order_epochs_and_resume_position():
    data = ItemDataset("my_project:RingItems", {"count": 10}, real="x")
    generator = torch.Generator().manual_seed(3)
    batches = [data(4, generator=generator) for _ in range(5)]
    assert all(set(batch) == {"real"} and batch["real"].shape == (4, 2) for batch in batches)
    assert data.epoch == 1 and data.position == 10
    replay = ItemDataset("my_project:RingItems", {"count": 10}, real="x")
    replay_generator = torch.Generator().manual_seed(3)
    first = replay.plan(12, replay_generator)
    assert sorted(first[:10]) == list(range(10)), "each epoch visits every item once"
    state, stream = replay.state_dict(), replay_generator.get_state()
    expected = replay.plan(8, replay_generator)
    restored = ItemDataset("my_project:RingItems", {"count": 10}, real="x")
    restored.load_state_dict(state)
    assert restored.plan(8, torch.Generator().set_state(stream)) == expected
    identity = data.resume_identity()
    assert identity["length"] == 10 and identity["identity"] == {"kind": "ring", "count": 10}
    assert len(identity["source_sha256"]) == 64


# End to end (in-process, CPU, a few steps) ----------------------------------------

def test_train_item_dataset_model_and_read_it_back(tmp_path):
    x, z = hg.data("x", shape=["B", 2]), hg.latent(4)
    fake = hg.net(Generator, z=z, hidden=16)
    d = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate)
    run = hg.train(hg.dataset(RingItems, count=64),
                   [hg.adversarial(d, real=x, fake=fake), hg.loss(my_project.radius_loss, input=fake, weight=0.1)],
                   samplers=[hg.sampler(my_project.scatter, points=fake, count=8)],
                   run=tmp_path / "ring", steps=3, batch_size=8, device="cpu")
    assert run.config_path == (tmp_path / "ring.toml").resolve()
    assert run.step == 3 and run.manifest["resume_supported"] is True
    series = hg.metrics(run)
    assert [step for step, _ in series["loss/objectives/radius_loss"]] == [1, 2, 3]
    assert set(hg.last(run, ["loss/d_total"])) == {"loss/d_total"}
    written = hg.samples(run)
    paths = written["scatter"]["scatter"]
    assert [p.suffix for p in paths] == [".svg", ".json"] and all(p.is_file() for p in paths)
    again = hg.train(hg.load(run.config_path), run=tmp_path / "ring")
    assert again.step == 3 and again.manifest["run_id"] == run.manifest["run_id"], "re-running the same config is a no-op resume"


def test_train_multi_network_model_and_sample_nested_outputs(tmp_path):
    run = hg.train(multi(steps=3, samplers=True), run=tmp_path / "multi")
    last = hg.last(run)
    assert run.step == 3
    assert {"loss/objectives/reconstruction", "loss/objectives/paired", "loss/d_total"} <= set(last)
    views = hg.samples(run)["points_and_codes"]
    assert sorted(views) == ["code", "points"]
    import json
    code = json.loads(views["code"][0].read_text())
    assert torch.tensor(code["value"]).shape == (4, 4)


@pytest.mark.heavy
def test_custom_metric_and_evaluation_run_in_their_workers(tmp_path):
    x, z = hg.data("x", shape=["B", 2]), hg.latent(4)
    fake = hg.net(Generator, z=z, hidden=16)
    d = hg.hndl(file=CRITIC, shape=["B", 1], x=hg.candidate)
    run = hg.train(hg.dataset(RingItems, count=64), [hg.adversarial(d, real=x, fake=fake)],
                   metrics=[hg.metric(my_project.g_over_d, every=1)],
                   evaluations=[hg.evaluation(my_project.ring_error, hg.dataset(RingItems, count=32, seed=1),
                                              every=4, samples=16, batch_size=8, device="cpu")],
                   run=tmp_path / "observed", steps=4, batch_size=8, device="cpu")
    assert hg.metrics(run, ["g_over_d"])["g_over_d"], "the custom scalar metric published at least once"
    rows = hg.evaluations(run)["ring_error"]
    assert rows[0]["step"] == 4 and rows[0]["status"] == "complete" and rows[0]["value"] >= 0

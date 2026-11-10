"""Tests for 'extends' configuration inheritance and Decision D3 fingerprint invariance."""

import json
from pathlib import Path
import pytest

import hypergan.api as hg
from hypergan.config import fingerprint, numerical_values, resolve_config
from hypergan.model_file import resolve_inheritance


def test_single_level_inheritance_deep_merge(tmp_path):
    base_file = tmp_path / "base.toml"
    base_file.write_text("""
format = "hypergan-model/1"
name = "base_model"

[data]
builtin = "gaussian_grid"

[prior]
kind = "gaussian"
z_dim = 16

[train]
steps = 1000
batch_size = 64
lr = 0.0002

[penalty]
lazy_k = 1
coeff = 1.0

[run]
devices = ["gpu0"]
checkpoint_every = 1000
preview_every = 100
""")

    child_file = tmp_path / "child.toml"
    child_file.write_text("""
format = "hypergan-model/1"
extends = "base.toml"

[train]
lr = 0.0004

[penalty]
lazy_k = 16

[run]
devices = ["gpu1"]
""")

    with child_file.open("rb") as f:
        import tomllib
        raw_child = tomllib.load(f)

    merged = resolve_inheritance(raw_child, config_path=child_file)

    # Tables deep merged:
    assert merged["train"]["steps"] == 1000
    assert merged["train"]["batch_size"] == 64
    assert merged["train"]["lr"] == 0.0004

    assert merged["penalty"]["coeff"] == 1.0
    assert merged["penalty"]["lazy_k"] == 16

    assert merged["run"]["checkpoint_every"] == 1000
    assert merged["run"]["preview_every"] == 100
    assert merged["run"]["devices"] == ["gpu1"]

    # Name defaulted to file stem of child
    assert merged["name"] == "child"
    assert "extends" not in merged


def test_arrays_are_replaced_not_concatenated(tmp_path):
    base_file = tmp_path / "base.toml"
    base_file.write_text("""
format = "hypergan-model/1"

[[losses]]
type = "adversarial"
critic = "critic1"

[[losses]]
type = "prior"
""")

    child_file = tmp_path / "child.toml"
    child_file.write_text("""
format = "hypergan-model/1"
extends = "base.toml"

[[losses]]
type = "adversarial"
critic = "critic2"
penalty = 0.5
""")

    with child_file.open("rb") as f:
        import tomllib
        raw_child = tomllib.load(f)

    merged = resolve_inheritance(raw_child, config_path=child_file)
    assert len(merged["losses"]) == 1
    assert merged["losses"][0]["critic"] == "critic2"
    assert merged["losses"][0]["penalty"] == 0.5


def test_chained_inheritance(tmp_path):
    grandparent = tmp_path / "grandparent.toml"
    grandparent.write_text("""
format = "hypergan-model/1"
[train]
steps = 5000
batch_size = 32
lr = 0.0001
""")

    parent = tmp_path / "parent.toml"
    parent.write_text("""
format = "hypergan-model/1"
extends = "grandparent.toml"
[train]
lr = 0.0002
batch_size = 64
""")

    child = tmp_path / "child.toml"
    child.write_text("""
format = "hypergan-model/1"
extends = "parent.toml"
[train]
lr = 0.0005
""")

    with child.open("rb") as f:
        import tomllib
        raw_child = tomllib.load(f)

    merged = resolve_inheritance(raw_child, config_path=child)
    assert merged["train"]["steps"] == 5000
    assert merged["train"]["batch_size"] == 64
    assert merged["train"]["lr"] == 0.0005
    assert merged["name"] == "child"


def test_circular_inheritance_detection_raises_value_error(tmp_path):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"

    a.write_text('format = "hypergan-model/1"\nextends = "b.toml"\n')
    b.write_text('format = "hypergan-model/1"\nextends = "a.toml"\n')

    with a.open("rb") as f:
        import tomllib
        raw_a = tomllib.load(f)

    with pytest.raises(ValueError, match="Circular inheritance detected"):
        resolve_inheritance(raw_a, config_path=a)


def test_self_inheritance_detection_raises_value_error(tmp_path):
    a = tmp_path / "self.toml"
    a.write_text('format = "hypergan-model/1"\nextends = "self.toml"\n')

    with a.open("rb") as f:
        import tomllib
        raw_a = tomllib.load(f)

    with pytest.raises(ValueError, match="Circular inheritance detected"):
        resolve_inheritance(raw_a, config_path=a)


def test_missing_parent_raises_file_not_found(tmp_path):
    child = tmp_path / "child.toml"
    child.write_text('format = "hypergan-model/1"\nextends = "does_not_exist.toml"\n')

    with child.open("rb") as f:
        import tomllib
        raw_child = tomllib.load(f)

    with pytest.raises(FileNotFoundError, match="Inherited configuration 'does_not_exist.toml' not found"):
        resolve_inheritance(raw_child, config_path=child)


def test_relative_path_anchoring(tmp_path):
    base_dir = tmp_path / "base_dir"
    base_dir.mkdir()
    sub_dir = tmp_path / "experiments"
    sub_dir.mkdir()

    hndl_file = base_dir / "network.hndl"
    hndl_file.write_text("input x: [B, 16]\noutput y: [B, 32]\n")

    parent = base_dir / "parent.toml"
    parent.write_text("""
format = "hypergan-model/1"
[networks.generator]
role = "generator"
hndl = "network.hndl"
input_shape = ["B", 16]
output_shape = ["B", 32]
inputs = { x = "latent" }
""")

    child = sub_dir / "child.toml"
    child.write_text(f"""
format = "hypergan-model/1"
extends = "{parent.as_posix()}"
[train]
steps = 200
""")

    with child.open("rb") as f:
        import tomllib
        raw_child = tomllib.load(f)

    merged = resolve_inheritance(raw_child, config_path=child)
    # The relative 'network.hndl' in parent was anchored to base_dir
    assert merged["networks"]["generator"]["hndl"] == str(hndl_file.resolve())


def test_decision_d3_name_excluded_from_numerical_fingerprint():
    base_cfg = {
        "name": "experiment_1",
        "data": {"factory": "gaussian_grid", "args": {"side": 10, "noise": 0.015}},
        "training": {"steps": 100, "batch_size": 32, "seed": 42},
    }
    renamed_cfg = {
        "name": "experiment_renamed_arbitrary_string",
        "data": {"factory": "gaussian_grid", "args": {"side": 10, "noise": 0.015}},
        "training": {"steps": 100, "batch_size": 32, "seed": 42},
    }

    res1 = resolve_config(base_cfg)
    res2 = resolve_config(renamed_cfg)

    # Numerical values must be identical:
    assert numerical_values(res1) == numerical_values(res2)

    # Fingerprint must be bit-for-bit identical:
    assert fingerprint(res1) == fingerprint(res2)


def test_hg_load_resolves_extends(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text("""
format = "hypergan-model/1"
[data]
builtin = "gaussian_grid"
[train]
steps = 100
""")

    child = tmp_path / "my_exp.toml"
    child.write_text("""
format = "hypergan-model/1"
extends = "base.toml"
[train]
steps = 500
""")

    m = hg.load(child)
    assert m.spec["train"]["steps"] == 500
    assert m.spec["data"]["builtin"] == "gaussian_grid"
    assert m.spec["name"] == "my_exp"
    assert "extends" not in m.spec


def test_extends_keyed_loss_overrides_and_append(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text("""
format = "hypergan-model/1"
[data]
builtin = "gaussian_grid"
[networks.generator]
role = "generator"
source = "linear(2)\\n"
input_shape = ["B", 4]
output_shape = ["B", 2]
inputs = { x = "latent" }
[networks.discriminator]
role = "critic"
source = "linear(1)\\n"
input_shape = ["B", 2]
output_shape = ["B", 1]
inputs = { x = "candidate" }

[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

[[losses]]
type = "prior"
weight = 1.0
""")

    child = tmp_path / "child.toml"
    child.write_text("""
format = "hypergan-model/1"
extends = "base.toml"

[losses.discriminator]
penalty = 2.5

[losses.prior]
weight = 0.5

[losses.reconstruction]
fn = "l1"
weight = 0.25
""")

    m = hg.load(child)
    losses = m.spec["losses"]
    assert isinstance(losses, list) and len(losses) == 3
    assert losses[0] == {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 2.5}
    assert losses[1] == {"type": "prior", "weight": 0.5}
    assert losses[2] == {"type": "reconstruction", "id": "reconstruction", "fn": "l1", "weight": 0.25}
    lowered = hg.lower(m)
    assert lowered["gradient_penalty"]["coeff"] == 2.5
    assert lowered["prior_regularizer"]["weight"] == 0.5
    assert len(lowered["objectives"]) == 1
    assert lowered["objectives"][0]["factory"] == "l1"
    assert lowered["objectives"][0]["weight"] == 0.25


def test_relative_path_anchoring_uncreated_paths_and_urls(tmp_path):
    base_dir = tmp_path / "base_dir"
    base_dir.mkdir()
    sub_dir = tmp_path / "sub_dir"
    sub_dir.mkdir()

    parent = base_dir / "parent.toml"
    parent.write_text("""
format = "hypergan-model/1"
[data]
builtin = "image_folder"
root = "./uncreated_data_dir"
[evaluations.fid.args]
weights_path = "https://example.com/weights.pth"
cache_dir = "~/.cache/hypergan"
manifest = "${paths.data}/manifest.json"
""")

    child = sub_dir / "child.toml"
    child.write_text(f'format = "hypergan-model/1"\nextends = "{parent.as_posix()}"\n')

    with child.open("rb") as f:
        import tomllib
        raw_child = tomllib.load(f)

    merged = resolve_inheritance(raw_child, config_path=child)
    assert merged["data"]["root"] == str((base_dir / "uncreated_data_dir").resolve())
    assert merged["evaluations"]["fid"]["args"]["weights_path"] == "https://example.com/weights.pth"
    assert merged["evaluations"]["fid"]["args"]["cache_dir"] == "~/.cache/hypergan"
    assert merged["evaluations"]["fid"]["args"]["manifest"] == "${paths.data}/manifest.json"


def test_evaluations_data_train_shorthand_and_train_optimizer_subtable():
    from hypergan.model_file import ModelFileError, lower
    from hypergan.schema_info import get_schema

    spec = {
        "format": "hypergan-model/1",
        "name": "eval-shorthand-test",
        "data": {"builtin": "cifar10", "root": "./data/cifar10", "sampling": "random", "horizontal_flip": True},
        "networks": {
            "generator": {"role": "generator", "source": "linear(2)\n", "input_shape": ["B", 4], "output_shape": ["B", 2], "inputs": {"x": "latent"}},
            "discriminator": {"role": "critic", "source": "linear(1)\n", "input_shape": ["B", 2], "output_shape": ["B", 1], "inputs": {"x": "candidate"}},
        },
        "losses": [{"type": "adversarial", "critic": "discriminator"}],
        "train": {
            "steps": 100,
            "batch_size": 32,
            "optimizer": {"lr": 0.0004, "d_lr_mult": 2.0},
        },
        "evaluations": {
            "div": {
                "factory": "hypergan.diversity_metrics:SampleDiversity",
                "samples": 64,
                "batch_size": 16,
                "seed": 7,
                "data": "train",
            }
        },
    }
    lowered = lower(spec)
    assert lowered["optimizer"]["lr"] == 0.0004
    assert lowered["optimizer"]["d_lr_mult"] == 2.0
    eval_data = lowered["metrics"]["custom"]["div"]["evaluation"]["data"]
    assert eval_data["factory"] == "hypergan.image_data:CIFAR10Data"
    assert eval_data["args"]["sampling"] == "sequential"
    assert eval_data["args"]["horizontal_flip"] is False

    # Top-level [optimizer] gives actionable hint
    bad_spec = dict(spec, optimizer={"lr": 0.0001})
    with pytest.raises(ModelFileError, match=r"place optimizer settings in '\[train\]' or '\[train\.optimizer\]'"):
        lower(bad_spec)

    # Schema sync assertions
    schema = get_schema()
    assert "cifar10" in schema["sections"]["data"]["fields"]["builtin"]["choices"]
    assert schema["sections"]["prior"]["fields"]["fixed_sigma"]["type"] == "number"
    assert "optimizer" not in schema["sections"]["networks"]["properties"]
    assert set(schema["sections"]["losses"]["items"]["fields"]["type"]["choices"]) == {
        "adversarial", "prior", "reconstruction", "objective"
    }
    assert set(schema["sections"]["metrics"]["fields"]["preset"]["choices"]) == {"standard", "none"}
    for sec in ("penalty", "sampling", "samplers", "defaults", "aggregation"):
        assert get_schema(sec)["section"] == sec

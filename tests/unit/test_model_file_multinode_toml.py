import tomllib
from copy import deepcopy
import pytest

from hypergan import api
from hypergan.config import resolve_config, load_config
from hypergan.toml_writer import dumps


def test_model_file_aggregation_table_parsing(tmp_path):
    model_toml = tmp_path / "model.toml"
    model_toml.write_text("""
format = "hypergan-model/1"
extends = "gaussian-grid"

[aggregation]
strategy = "pseudo_gradient"
sync_every_steps = 20
outer_lr = 0.8
outer_momentum = 0.5
""")

    raw = tomllib.loads(model_toml.read_text())
    assert "aggregation" in raw
    agg = raw["aggregation"]
    assert agg["strategy"] == "pseudo_gradient"
    assert agg["sync_every_steps"] == 20
    assert agg["outer_lr"] == 0.8
    assert agg["outer_momentum"] == 0.5

    m = api.load(model_toml)
    lowered = api.lower(m)
    assert "aggregation" in lowered
    assert lowered["aggregation"]["strategy"] == "pseudo_gradient"
    assert lowered["aggregation"]["sync_every_steps"] == 20


def test_model_file_aggregation_sync_every_alias_normalization(tmp_path):
    model_toml = tmp_path / "model.toml"
    model_toml.write_text("""
format = "hypergan-model/1"
extends = "gaussian-grid"

[aggregation]
strategy = "parameter_average"
sync_every = 15
""")

    m = api.load(model_toml)
    recipe = api.lower(m)
    assert "aggregation" in recipe
    assert recipe["aggregation"]["strategy"] == "parameter_average"
    assert recipe["aggregation"]["sync_every_steps"] == 15


def test_model_file_run_multinode_fields(tmp_path):
    model_toml = tmp_path / "model.toml"
    model_toml.write_text("""
format = "hypergan-model/1"
extends = "gaussian-grid"

[run]
nodes = 4
node_rank = 1
rendezvous = "tcp://master:29500"
strategy = "pseudo_gradient"
sync_every = 25
""")

    m = api.load(model_toml)
    recipe = api.lower(m)
    assert "run" in recipe
    assert recipe["run"]["nodes"] == 4
    assert recipe["run"]["node_rank"] == 1
    assert recipe["run"]["rendezvous"] == "tcp://master:29500"
    # [run] strategy/sync_every normalized into [aggregation]
    assert "aggregation" in recipe
    assert recipe["aggregation"]["strategy"] == "pseudo_gradient"
    assert recipe["aggregation"]["sync_every_steps"] == 25


def test_config_preserves_aggregation_table(tmp_path):
    raw_config = {
        "format": {"name": "hypergan-model", "version": 1},
        "training": {"steps": 100, "batch_size": 16, "device": "cpu", "seed": 42},
        "aggregation": {
            "strategy": "pseudo_gradient",
            "sync_every_steps": 50,
            "outer_lr": 0.5,
            "outer_momentum": 0.9,
        },
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 4, "out_features": 2}, "inputs": {"input": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"input": "candidate"}},
        },
        "objectives": [],
    }

    resolved = resolve_config(raw_config)
    assert "aggregation" in resolved
    assert resolved["aggregation"]["strategy"] == "pseudo_gradient"
    assert resolved["aggregation"]["sync_every_steps"] == 50

    cfg_file = tmp_path / "recipe.toml"
    cfg_file.write_text(dumps(raw_config))

    loaded = load_config(cfg_file)
    assert "aggregation" in loaded
    assert loaded["aggregation"]["strategy"] == "pseudo_gradient"
    assert loaded["aggregation"]["sync_every_steps"] == 50

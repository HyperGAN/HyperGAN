import io
from pathlib import Path
import sys
import pytest
from unittest.mock import MagicMock

from hypergan import cli


def test_cli_parser_multinode_flags():
    parser = cli._build_parser()
    
    # Train parser
    args = parser.parse_args([
        "train", "recipe.toml",
        "--nodes", "4",
        "--node-rank", "1",
        "--rendezvous", "tcp://coordinator:29500",
        "--strategy", "pseudo_gradient",
        "--sync-every", "50",
        "--experimental",
    ])
    assert args.nodes == 4
    assert args.node_rank == 1
    assert args.rendezvous == "tcp://coordinator:29500"
    assert args.strategy == "pseudo_gradient"
    assert args.sync_every == 50
    assert args.experimental is True

    # Resume parser
    args_resume = parser.parse_args([
        "resume", "runs/test_run",
        "--nodes", "2",
        "--node-rank", "0",
        "--rendezvous", "tcp://coordinator:29500",
        "--strategy", "parameter_average",
        "--sync-every", "10",
    ])
    assert args_resume.nodes == 2
    assert args_resume.node_rank == 0
    assert args_resume.rendezvous == "tcp://coordinator:29500"
    assert args_resume.strategy == "parameter_average"
    assert args_resume.sync_every == 10


def test_cli_parser_multinode_negative_validation():
    parser = cli._build_parser()

    # --nodes cannot be 0 or negative
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--nodes", "0"])

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--nodes", "-2"])

    # --node-rank cannot be negative (0 is allowed)
    args = parser.parse_args(["train", "recipe.toml", "--node-rank", "0"])
    assert args.node_rank == 0

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--node-rank", "-1"])

    # --sync-every cannot be 0 or negative
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--sync-every", "0"])

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--sync-every", "-5"])

    # --strategy must be valid choice
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--strategy", "invalid_strat"])


def test_cli_dispatch_forwards_multinode_options(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.toml"
    recipe.write_text('extends = "gaussian-grid"\n')

    parser = cli._build_parser()
    args = parser.parse_args([
        "train", str(recipe),
        "--nodes", "4",
        "--node-rank", "2",
        "--rendezvous", "tcp://1.2.3.4:29500",
        "--strategy", "parameter_average",
        "--sync-every", "25",
    ])

    captured_options = {}

    def mock_prepare(model_or_config, run_dir, steps=None, **options):
        captured_options.update(options)
        prepared = MagicMock()
        prepared.config = {"training": {"steps": 10}}
        prepared.profile = None
        prepared.run.return_value = {"status": "ok"}
        return prepared

    from hypergan import api
    monkeypatch.setattr(api, "prepare", mock_prepare)
    from contextlib import contextmanager
    @contextmanager
    def mock_viewer(*args, **kwargs):
        yield
    monkeypatch.setattr(cli, "_training_viewer", mock_viewer)

    cli._dispatch(args, output=MagicMock())

    assert captured_options.get("nodes") == 4
    assert captured_options.get("node_rank") == 2
    assert captured_options.get("rendezvous") == "tcp://1.2.3.4:29500"
    assert captured_options.get("strategy") == "parameter_average"
    assert captured_options.get("sync_every") == 25


def test_cli_print_training_header_shows_cluster_and_strategy(tmp_path, capsys):
    execution = {
        "name": "cuda-replicated-nccl",
        "nodes": 4,
        "node_rank": 1,
        "world_size": 8,
        "ranks_per_node": 2,
        "aggregation": {
            "strategy": "pseudo_gradient",
            "sync_every_steps": 100,
        },
    }

    cli._print_training_header(tmp_path / "run", "test_exp", "cuda", execution=execution)
    err = capsys.readouterr().err

    assert "Cluster:        Node 1/4 (8 global ranks, 2 local)" in err
    assert "Strategy:       pseudo_gradient (sync every 100 steps)" in err


def test_cli_print_training_header_omits_single_node_defaults(tmp_path, capsys):
    execution = {
        "name": "cpu-replicated-gloo",
        "nodes": 1,
        "node_rank": 0,
        "world_size": 2,
        "ranks_per_node": 2,
        "aggregation": {
            "strategy": "gradient_allreduce",
            "sync_every_steps": 1,
        },
    }

    cli._print_training_header(tmp_path / "run", "test_exp", "cpu", execution=execution)
    err = capsys.readouterr().err

    assert "Cluster:" not in err
    assert "Strategy:" not in err

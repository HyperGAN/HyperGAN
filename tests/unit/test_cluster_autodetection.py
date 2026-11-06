import os
import pytest
from unittest.mock import patch

from hypergan.execution import detect_cluster_environment, _apply_execution_options


def test_detect_cluster_environment_slurm():
    env = {
        "SLURM_NNODES": "4",
        "SLURM_NODEID": "2",
    }
    with patch.dict(os.environ, env, clear=True):
        detected = detect_cluster_environment()
        assert detected.get("nodes") == 4
        assert detected.get("node_rank") == 2
        assert "rendezvous" not in detected


def test_detect_cluster_environment_torchrun():
    env = {
        "NNODES": "8",
        "NODE_RANK": "3",
        "MASTER_ADDR": "10.0.1.5",
        "MASTER_PORT": "29505",
    }
    with patch.dict(os.environ, env, clear=True):
        detected = detect_cluster_environment()
        assert detected.get("nodes") == 8
        assert detected.get("node_rank") == 3
        assert detected.get("rendezvous") == "tcp://10.0.1.5:29505"


def test_detect_cluster_environment_torchrun_group_rank_and_default_port():
    env = {
        "NNODES": "2",
        "GROUP_RANK": "1",
        "MASTER_ADDR": "master.cluster.local",
    }
    with patch.dict(os.environ, env, clear=True):
        detected = detect_cluster_environment()
        assert detected.get("nodes") == 2
        assert detected.get("node_rank") == 1
        assert detected.get("rendezvous") == "tcp://master.cluster.local:29500"


def test_detect_cluster_environment_invalid_values_shielded():
    env = {
        "SLURM_NNODES": "not_a_number",
        "SLURM_NODEID": "-5",
    }
    with patch.dict(os.environ, env, clear=True):
        detected = detect_cluster_environment()
        assert "nodes" not in detected
        assert "node_rank" not in detected


def test_four_tier_precedence_cli_overrides_env_and_toml():
    # Tier 1 (CLI) > Tier 2 (Env) > Tier 3 (TOML) > Tier 4 (Defaults)
    env = {
        "SLURM_NNODES": "4",
        "SLURM_NODEID": "1",
        "MASTER_ADDR": "coordinator",
    }
    config = {
        "run": {
            "nodes": 3,
            "node_rank": 0,
        },
        "training": {"device": "cpu"},
    }
    with patch.dict(os.environ, env, clear=True):
        # CLI explicitly passes nodes=2, overriding SLURM_NNODES=4 and config nodes=3
        profile, devices = _apply_execution_options(
            nodes=2,
            node_rank=None,  # Not passed on CLI, should fall back to Env (1)
            rendezvous=None,  # Falls back to Env (tcp://coordinator:29500)
            config=config,
            profile=None,
        )
        assert profile is not None
        exec_cfg = profile["execution"]
        assert exec_cfg["nodes"] == 2
        assert exec_cfg["node_rank"] == 1
        assert exec_cfg["rendezvous"] == "tcp://coordinator:29500"


def test_four_tier_precedence_env_overrides_toml():
    env = {
        "NNODES": "4",
        "NODE_RANK": "2",
    }
    config = {
        "run": {
            "nodes": 8,
            "node_rank": 0,
            "rendezvous": "tcp://custom:29500",
        },
        "training": {"device": "cpu"},
    }
    with patch.dict(os.environ, env, clear=True):
        profile, devices = _apply_execution_options(
            config=config,
            profile=None,
        )
        assert profile is not None
        exec_cfg = profile["execution"]
        assert exec_cfg["nodes"] == 4  # Env won over TOML
        assert exec_cfg["node_rank"] == 2  # Env won over TOML
        assert exec_cfg["rendezvous"] == "tcp://custom:29500"  # TOML used since Env had none


def test_dynamic_profile_synthesis_multi_gpu(monkeypatch):
    import torch
    if hasattr(torch, "cuda"):
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    config = {"training": {"device": "cuda"}}
    with patch.dict(os.environ, {}, clear=True):
        profile, devices = _apply_execution_options(
            devices="0,1",
            config=config,
            profile=None,
        )
        assert profile is not None
        exec_cfg = profile["execution"]
        assert exec_cfg["name"] == "cuda-replicated-nccl"
        assert exec_cfg["world_size"] == 2
        assert exec_cfg["nodes"] == 1
        assert exec_cfg["ranks_per_node"] == 2


def test_dynamic_profile_synthesis_diloco_aggregation():
    config = {
        "training": {"device": "cpu"},
        "aggregation": {
            "strategy": "pseudo_gradient",
            "sync_every_steps": 50,
            "outer_lr": 0.7,
            "outer_momentum": 0.9,
        },
    }
    with patch.dict(os.environ, {}, clear=True):
        profile, devices = _apply_execution_options(
            config=config,
            profile=None,
        )
        assert profile is not None
        agg = profile["aggregation"]
        assert agg["strategy"] == "pseudo_gradient"
        assert agg["sync_every_steps"] == 50
        assert agg["outer_lr"] == 0.7
        assert agg["outer_momentum"] == 0.9

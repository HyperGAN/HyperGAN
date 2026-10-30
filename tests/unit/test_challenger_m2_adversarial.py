import builtins
from contextlib import nullcontext
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import stat
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from hypergan import cli
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_checkpoints import (
    prepare_distributed_checkpoint,
    restore_distributed_checkpoint,
    save_distributed_checkpoint,
)
from hypergan.config import config_values, fingerprint, load_config
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.execution import (
    _apply_execution_options,
    _same_topology,
    detect_cluster_environment,
    prepare_resume,
    PreparedExecution,
)
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.run_state import EventJournal
from tests.config_fixtures import write_default
from tests.distributed_fixtures import make_toy_config


def _assert_optimizer_states_equal(opt1, opt2):
    sd1 = opt1.state_dict()
    sd2 = opt2.state_dict()
    assert len(sd1["param_groups"]) == len(sd2["param_groups"])
    for pg1, pg2 in zip(sd1["param_groups"], sd2["param_groups"]):
        for k in pg1:
            if k != "params":
                assert pg1[k] == pg2[k], f"Optimizer param group mismatch for key {k}: {pg1[k]} != {pg2[k]}"

    # State tensors
    assert set(sd1["state"].keys()) == set(sd2["state"].keys())
    for p_id in sd1["state"]:
        state1 = sd1["state"][p_id]
        state2 = sd2["state"][p_id]
        assert set(state1.keys()) == set(state2.keys())
        for k in state1:
            v1, v2 = state1[k], state2[k]
            if isinstance(v1, torch.Tensor):
                torch.testing.assert_close(v1, v2, rtol=0, atol=0)
            else:
                assert v1 == v2, f"Optimizer state mismatch for {k}: {v1} != {v2}"


def _worker_nfs_free_adversarial(rank, world_size, dir_rank0, dir_rank1):
    local_run_dir = Path(dir_rank0) if rank == 0 else Path(dir_rank1)

    cfg = make_toy_config(device="cpu", batch_size=8, steps=10)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)

    # Perform 2 training updates so optimizer states and step counts are active
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    local_ids = ids.chunk(world_size)[rank]
    local_batch = {"real": data.chunk(world_size)[rank]}

    trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
    trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))

    assert trainer.step == 2
    assert trainer.checkpoint_ready is True

    metadata = {
        "run_id": "adversarial_nfs_run",
        "attempt_id": "attempt_01",
        "next_sample_sequence": 3,
    }

    # Intercept filesystem writes on follower rank (rank 1) to assert strictly 0 bytes written
    write_tracker = {"bytes_written": 0, "write_calls": 0}
    orig_open = builtins.open
    orig_write = os.write

    if rank != 0:
        def strict_open(*args, **kwargs):
            mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
            if any(m in mode for m in ("w", "a", "+", "x")):
                raise PermissionError(f"Follower rank {rank} attempted file write: {args}")
            return orig_open(*args, **kwargs)

        def strict_write(fd, data):
            # Only allow stdout/stderr (fd 1, 2)
            if fd not in (1, 2):
                write_tracker["bytes_written"] += len(data)
                write_tracker["write_calls"] += 1
                raise PermissionError(f"Follower rank {rank} attempted os.write on fd {fd}")
            return orig_write(fd, data)

        builtins.open = strict_open
        os.write = strict_write

    try:
        checkpoint_path = save_distributed_checkpoint(local_run_dir, trainer, local_batch, metadata)
    finally:
        if rank != 0:
            builtins.open = orig_open
            os.write = strict_write

    if rank == 0:
        assert checkpoint_path.exists()
        ckpt_root = Path(dir_rank0) / "distributed-checkpoints"
        assert (ckpt_root / "latest.json").exists()
        assert (ckpt_root / checkpoint_path.name / "rank-00000.pt").exists()
        assert (ckpt_root / checkpoint_path.name / "rank-00001.pt").exists()
        assert (ckpt_root / checkpoint_path.name / "manifest.json").exists()
    else:
        # Follower must have written strictly 0 bytes to disk
        assert write_tracker["bytes_written"] == 0
        assert write_tracker["write_calls"] == 0
        # If dir_rank1 was non-existent, it must still not exist
        if not str(dir_rank1).endswith("readonly"):
            assert not Path(dir_rank1).exists()

    # Multi-rank checkpoint restore where rank 0 reads files and scatters state dictionaries
    fresh_trainer = ReplicatedCPUTrainer(cfg, world_size=world_size)
    restored_path, info, last_batch = restore_distributed_checkpoint(
        local_run_dir,
        fresh_trainer,
        {"run_id": "adversarial_nfs_run"},
        checkpoint=None,
    )

    # 1. Step count equality
    assert fresh_trainer.step == trainer.step == 2

    # 2. Generator & Discriminator parameter equality
    for model_key in ("generator", "discriminator"):
        p_orig = list(trainer.graph.models[model_key].parameters())
        p_rest = list(fresh_trainer.graph.models[model_key].parameters())
        assert len(p_orig) == len(p_rest)
        for p0, p1 in zip(p_orig, p_rest):
            torch.testing.assert_close(p0, p1, rtol=0, atol=0)

    # 3. Prior states equality
    for k, v in trainer.prior.state_dict().items():
        torch.testing.assert_close(v, fresh_trainer.prior.state_dict()[k], rtol=0, atol=0)

    # 4. Optimizer states equality (opt_g and opt_d)
    _assert_optimizer_states_equal(trainer.opt_g, fresh_trainer.opt_g)
    _assert_optimizer_states_equal(trainer.opt_d, fresh_trainer.opt_d)


def test_nfs_free_save_and_restore_adversarial_nonexistent_run_dir(tmp_path):
    dir_rank0 = tmp_path / "node_0_run"
    dir_rank0.mkdir()
    (dir_rank0 / "manifest.json").write_text(json.dumps({"run_id": "adversarial_nfs_run"}))

    # Follower directory intentionally does not exist on disk
    dir_rank1 = tmp_path / "deeply" / "nested" / "nonexistent" / "node_1_run"

    launch_cpu_workers(
        _worker_nfs_free_adversarial,
        args=(str(dir_rank0), str(dir_rank1)),
        world_size=2,
        timeout=30,
    )


def test_nfs_free_save_and_restore_adversarial_readonly_run_dir(tmp_path):
    dir_rank0 = tmp_path / "node_0_run"
    dir_rank0.mkdir()
    (dir_rank0 / "manifest.json").write_text(json.dumps({"run_id": "adversarial_nfs_run"}))

    # Follower directory exists but is strictly read-only (0o555)
    dir_rank1 = tmp_path / "node_1_readonly"
    dir_rank1.mkdir()
    os.chmod(dir_rank1, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    try:
        launch_cpu_workers(
            _worker_nfs_free_adversarial,
            args=(str(dir_rank0), str(dir_rank1)),
            world_size=2,
            timeout=30,
        )
    finally:
        os.chmod(dir_rank1, stat.S_IRWXU)


def test_same_topology_port_and_rendezvous_invariance():
    base_exec = {
        "name": "cuda-replicated-nccl",
        "world_size": 4,
        "nodes": 2,
        "ranks_per_node": 2,
        "node_rank": 0,
        "rendezvous": "tcp://coordinator:29500",
        "accumulation_steps": 1,
        "aggregation": {"strategy": "gradient_allreduce", "sync_every_steps": 1},
    }

    # 1. Ephemeral port change in rendezvous must NOT reject resume
    resumed_exec_diff_port = dict(base_exec, rendezvous="tcp://coordinator:48912")
    assert _same_topology(base_exec, resumed_exec_diff_port) is True

    # 2. node_rank difference must NOT reject resume
    resumed_exec_diff_node_rank = dict(base_exec, node_rank=1)
    assert _same_topology(base_exec, resumed_exec_diff_node_rank) is True

    # 3. Missing rendezvous or node_rank in one topology must NOT reject resume
    no_rendezvous = {k: v for k, v in base_exec.items() if k != "rendezvous"}
    assert _same_topology(base_exec, no_rendezvous) is True
    assert _same_topology(no_rendezvous, base_exec) is True

    # 4. Changes to actual numerical topology MUST be rejected
    # Changed world_size
    diff_world_size = dict(base_exec, world_size=8)
    assert _same_topology(base_exec, diff_world_size) is False

    # Changed nodes
    diff_nodes = dict(base_exec, nodes=4)
    assert _same_topology(base_exec, diff_nodes) is False

    # Changed aggregation strategy
    diff_strategy = dict(base_exec, aggregation={"strategy": "pseudo_gradient", "sync_every_steps": 50})
    assert _same_topology(base_exec, diff_strategy) is False

    # Comparison with None must be False
    assert _same_topology(base_exec, None) is False
    assert _same_topology(None, base_exec) is False
    assert _same_topology(None, None) is True


def test_follower_node_routing_to_run_follower():
    # Verify PreparedExecution.run() routes follower nodes directly to run_follower
    profile = {
        "schema_version": 1,
        "execution": {
            "name": "cpu-replicated-gloo",
            "world_size": 2,
            "nodes": 2,
            "node_rank": 1,  # Follower node
            "ranks_per_node": 1,
            "accumulation_steps": 1,
        },
    }
    config = {"training": {"steps": 10, "device": "cpu", "batch_size": 8}}
    prep = PreparedExecution(
        operation="train",
        config=config,
        run_dir=Path("/tmp/follower_run"),
        config_path="recipe.toml",
        checkpoint=None,
        steps=10,
        profile=profile,
        service_policy=None,
        controls={},
    )

    with patch("hypergan.replicated_execution.run_follower") as mock_follower:
        mock_follower.return_value = {"status": "completed", "node_rank": 1}
        result = prep.run()
        assert mock_follower.called
        assert result == {"status": "completed", "node_rank": 1}
        args, kwargs = mock_follower.call_args
        assert args[0] == config
        assert args[1] == Path("/tmp/follower_run")
        assert args[2] == 10
        assert kwargs["profile"] == profile


def test_follower_prepare_resume_without_local_manifest_succeeds(tmp_path):
    """Feature 22:
    When a follower node calls prepare_resume without a local manifest.json (NFS-free environment),
    prepare_resume succeeds without requiring a local manifest, returns a PreparedExecution instance,
    and prep.run() dispatches to run_follower.
    """
    follower_run_dir = tmp_path / "follower_nonexistent_run"
    config_file = tmp_path / "model.toml"
    config_file.write_text('extends = "gaussian-grid"\n')

    prep = prepare_resume(
        run_dir=follower_run_dir,
        config_path=config_file,
        nodes=2,
        node_rank=1,
        rendezvous="tcp://127.0.0.1:29500",
    )
    assert isinstance(prep, PreparedExecution)
    assert prep.operation == "resume"
    assert prep.checkpoint is None
    assert prep.profile["execution"]["node_rank"] == 1
    assert prep.profile["execution"]["nodes"] == 2

    with patch("hypergan.replicated_execution.run_follower") as mock_follower:
        mock_follower.return_value = {"status": "completed", "node_rank": 1}
        result = prep.run()
        assert mock_follower.called
        assert result == {"status": "completed", "node_rank": 1}


def test_follower_prepare_resume_with_conflicting_manifest_raises_topology_mismatch(tmp_path):
    """When a follower node has a local manifest with a conflicting execution topology,
    prepare_resume still validates saved against execution and raises ValueError.
    """
    config_file = tmp_path / "model.toml"
    config_file.write_text('extends = "gaussian-grid"\n')
    cfg = load_config(config_file)

    follower_run_dir = tmp_path / "follower_run_with_manifest"
    follower_run_dir.mkdir(parents=True)
    manifest_data = {
        "schema_version": 1,
        "run_id": "test_run_123",
        "config": config_values(cfg),
        "config_sha256": "abc",
        "next_sample_sequence": 0,
        "execution": {
            "name": "cpu-replicated-gloo",
            "world_size": 4,
            "nodes": 4,
            "node_rank": 0,
        },
    }
    (follower_run_dir / "manifest.json").write_text(json.dumps(manifest_data))

    # Follower attempts to resume with nodes=2 (conflicting with saved nodes=4)
    with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
        prepare_resume(
            run_dir=follower_run_dir,
            nodes=2,
            node_rank=1,
            rendezvous="tcp://127.0.0.1:29500",
        )


def test_four_tier_precedence_hierarchy_slurm_cli_toml_defaults():
    # Tier 1 (CLI) > Tier 2 (Slurm Env) > Tier 3 (TOML) > Tier 4 (Defaults)

    # 1. Full hierarchy: CLI present -> overrides Slurm, TOML, and Defaults
    env_slurm = {
        "SLURM_NNODES": "4",
        "SLURM_NODEID": "2",
        "MASTER_ADDR": "slurm.master.local",
        "MASTER_PORT": "29501",
    }
    config_toml = {
        "run": {
            "nodes": 3,
            "node_rank": 1,
            "rendezvous": "tcp://toml.master.local:29502",
        },
        "aggregation": {
            "strategy": "pseudo_gradient",
            "sync_every_steps": 25,
        },
        "training": {"device": "cpu"},
    }

    with patch.dict(os.environ, env_slurm, clear=True):
        profile, devices = _apply_execution_options(
            nodes=8,
            node_rank=3,
            rendezvous="tcp://cli.master.local:29500",
            strategy="parameter_average",
            sync_every=10,
            config=config_toml,
            profile=None,
        )
        assert profile is not None
        exec_cfg = profile["execution"]
        agg_cfg = profile["aggregation"]
        assert exec_cfg["nodes"] == 8  # CLI won
        assert exec_cfg["node_rank"] == 3  # CLI won
        assert exec_cfg["rendezvous"] == "tcp://cli.master.local:29500"  # CLI won
        assert agg_cfg["strategy"] == "parameter_average"  # CLI won
        assert agg_cfg["sync_every_steps"] == 10  # CLI won

    # 2. CLI omitted -> Slurm Env overrides TOML and Defaults
    with patch.dict(os.environ, env_slurm, clear=True):
        profile, devices = _apply_execution_options(
            nodes=None,
            node_rank=None,
            rendezvous=None,
            strategy=None,
            sync_every=None,
            config=config_toml,
            profile=None,
        )
        assert profile is not None
        exec_cfg = profile["execution"]
        agg_cfg = profile["aggregation"]
        assert exec_cfg["nodes"] == 4  # Slurm won over TOML (3)
        assert exec_cfg["node_rank"] == 2  # Slurm won over TOML (1)
        assert exec_cfg["rendezvous"] == "tcp://slurm.master.local:29501"  # Slurm won over TOML
        assert agg_cfg["strategy"] == "pseudo_gradient"  # TOML used (Slurm has no strategy)
        assert agg_cfg["sync_every_steps"] == 25  # TOML used (Slurm has no sync_every)

    # 3. CLI and Slurm omitted -> TOML overrides Defaults
    with patch.dict(os.environ, {}, clear=True):
        profile, devices = _apply_execution_options(
            nodes=None,
            node_rank=None,
            rendezvous=None,
            strategy=None,
            sync_every=None,
            config=config_toml,
            profile=None,
        )
        assert profile is not None
        exec_cfg = profile["execution"]
        agg_cfg = profile["aggregation"]
        assert exec_cfg["nodes"] == 3  # TOML won over Default (1)
        assert exec_cfg["node_rank"] == 1  # TOML won over Default (0)
        assert exec_cfg["rendezvous"] == "tcp://toml.master.local:29502"  # TOML won over Default
        assert agg_cfg["strategy"] == "pseudo_gradient"  # TOML won over Default
        assert agg_cfg["sync_every_steps"] == 25  # TOML won over Default

    # 4. All omitted -> Defaults apply
    with patch.dict(os.environ, {}, clear=True):
        profile, devices = _apply_execution_options(
            nodes=None,
            node_rank=None,
            rendezvous=None,
            strategy=None,
            sync_every=None,
            config={"training": {"device": "cpu"}},
            profile=None,
        )
        # Default single node/cpu does not synthesize replicated profile
        assert profile is None


def test_cli_parser_multinode_adversarial_boundaries():
    parser = cli._build_parser()

    # Valid boundaries
    args = parser.parse_args(["train", "recipe.toml", "--nodes", "1", "--node-rank", "0", "--sync-every", "1"])
    assert args.nodes == 1
    assert args.node_rank == 0
    assert args.sync_every == 1

    # Zero or negative nodes must exit
    for bad_nodes in ["0", "-1", "-100"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "recipe.toml", "--nodes", bad_nodes])

    # Negative node_rank must exit
    for bad_rank in ["-1", "-5"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "recipe.toml", "--node-rank", bad_rank])

    # Zero or negative sync_every must exit
    for bad_sync in ["0", "-1"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "recipe.toml", "--sync-every", bad_sync])

    # Invalid strategy choice must exit
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "recipe.toml", "--strategy", "unsupported_federated_avg"])


def test_cli_resume_follower_without_manifest_and_viewer_suppressed(tmp_path):
    """Verify Feature 22:
    CLI resume on follower node without local manifest succeeds, dispatches to run_follower,
    and suppresses web viewer autostart.
    """
    config_file = tmp_path / "model.toml"
    config_file.write_text('extends = "gaussian-grid"\n')
    follower_dir = tmp_path / "deeply" / "nested" / "nonexistent_run"

    parser = cli._build_parser()
    args = parser.parse_args([
        "resume", str(follower_dir),
        "--config", str(config_file),
        "--nodes", "2",
        "--node-rank", "1",
        "--rendezvous", "tcp://127.0.0.1:29500",
    ])

    # 1. Viewer autostart is suppressed
    viewer_ctx = cli._training_viewer(args)
    from contextlib import nullcontext
    assert isinstance(viewer_ctx, nullcontext)

    # 2. CLI dispatch succeeds and routes to run_follower
    with patch("hypergan.replicated_execution.run_follower") as mock_follower:
        mock_follower.return_value = {"status": "completed", "node_rank": 1}
        cli._dispatch(args, output=MagicMock())
        assert mock_follower.called


def test_follower_resume_hostile_paths_matrix(tmp_path):
    """Stress-test Feature 22 hostile follower path conditions:
    Deeply nested nonexistent path, nonexistent parent, read-only directory,
    broken symlink, and valid symlink.
    """
    config_file = tmp_path / "model.toml"
    config_file.write_text('extends = "gaussian-grid"\n')

    deep_path = tmp_path / "deep" / "nested" / "never" / "created" / "run"
    no_parent = Path("/nonexistent_dummy_parent_hypergan_12345/run")
    ro_dir = tmp_path / "readonly_dir"
    ro_dir.mkdir()
    ro_dir.chmod(0o444)
    broken_link = tmp_path / "broken_symlink"
    broken_link.symlink_to(tmp_path / "does_not_exist_xyz")
    valid_target = tmp_path / "valid_target"
    valid_target.mkdir()
    valid_symlink = tmp_path / "valid_symlink"
    valid_symlink.symlink_to(valid_target)

    for p in [deep_path, no_parent, ro_dir, broken_link, valid_symlink]:
        prep = prepare_resume(
            run_dir=p,
            config_path=config_file,
            nodes=2,
            node_rank=1,
            rendezvous="tcp://127.0.0.1:29500",
        )
        assert isinstance(prep, PreparedExecution)
        assert prep.operation == "resume"
        assert prep.checkpoint is None
        assert prep.profile["execution"]["node_rank"] == 1
        with patch("hypergan.replicated_execution.run_follower") as mock_follower:
            mock_follower.return_value = {"status": "completed", "node_rank": 1}
            res = prep.run()
            assert mock_follower.called
            assert res == {"status": "completed", "node_rank": 1}


def test_follower_resume_config_matrix(tmp_path):
    """Stress-test Feature 22 follower config conditions:
    Valid config succeeds, missing config file raises FileNotFoundError,
    corrupt config raises syntax error, and omitted config raises ValueError.
    """
    deep_path = tmp_path / "deep" / "run"
    valid_cfg = tmp_path / "valid.toml"
    valid_cfg.write_text('extends = "gaussian-grid"\n')

    # 1. Valid config
    prep = prepare_resume(run_dir=deep_path, config_path=valid_cfg, nodes=2, node_rank=1, rendezvous="tcp://127.0.0.1:29500")
    assert isinstance(prep, PreparedExecution)

    # 2. Missing config file
    with pytest.raises(FileNotFoundError):
        prepare_resume(run_dir=deep_path, config_path=tmp_path / "nonexistent.toml", nodes=2, node_rank=1, rendezvous="tcp://127.0.0.1:29500")

    # 3. Corrupt config syntax
    corrupt_cfg = tmp_path / "corrupt.toml"
    corrupt_cfg.write_text("extends = [invalid syntax\n")
    with pytest.raises(Exception):
        prepare_resume(run_dir=deep_path, config_path=corrupt_cfg, nodes=2, node_rank=1, rendezvous="tcp://127.0.0.1:29500")

    # 4. Omitted config without shared filesystem
    with pytest.raises(ValueError, match="requires an explicit configuration"):
        prepare_resume(run_dir=deep_path, config_path=None, nodes=2, node_rank=1, rendezvous="tcp://127.0.0.1:29500")


def test_follower_resume_node_rank_sources_matrix(tmp_path):
    """Stress-test Feature 19 & 22:
    Follower rank detection across CLI, SLURM_NODEID, NODE_RANK, GROUP_RANK, and TOML [run].
    Leader (rank 0) rejection when run_dir does not exist.
    """
    config_file = tmp_path / "model.toml"
    config_file.write_text('extends = "gaussian-grid"\n')
    deep_path = tmp_path / "deep" / "run"

    # 1. CLI node_rank=1
    prep_cli = prepare_resume(run_dir=deep_path, config_path=config_file, nodes=2, node_rank=1, rendezvous="tcp://127.0.0.1:29500")
    assert prep_cli.profile["execution"]["node_rank"] == 1

    # 2. SLURM_NODEID=1
    with patch.dict(os.environ, {"SLURM_NODEID": "1", "SLURM_NNODES": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}):
        prep_slurm = prepare_resume(run_dir=deep_path, config_path=config_file)
        assert prep_slurm.profile["execution"]["node_rank"] == 1

    # 3. NODE_RANK=1
    with patch.dict(os.environ, {"NODE_RANK": "1", "NNODES": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}):
        prep_node = prepare_resume(run_dir=deep_path, config_path=config_file)
        assert prep_node.profile["execution"]["node_rank"] == 1

    # 4. GROUP_RANK=1
    with patch.dict(os.environ, {"GROUP_RANK": "1", "NNODES": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}):
        prep_group = prepare_resume(run_dir=deep_path, config_path=config_file)
        assert prep_group.profile["execution"]["node_rank"] == 1

    # 5. TOML [run] node_rank = 1
    toml_rank = tmp_path / "rank_model.toml"
    toml_rank.write_text('extends = "gaussian-grid"\n[run]\nnodes = 2\nnode_rank = 1\nrendezvous = "tcp://127.0.0.1:29500"\n')
    prep_toml = prepare_resume(run_dir=deep_path, config_path=toml_rank)
    assert prep_toml.profile["execution"]["node_rank"] == 1

    # 6. Negative test: SLURM_NODEID=0 (leader) requires existing run_dir
    with patch.dict(os.environ, {"SLURM_NODEID": "0"}):
        with pytest.raises(ValueError):
            prepare_resume(run_dir=deep_path, config_path=config_file)


def test_follower_resume_viewer_suppression_across_all_rank_sources():
    """Verify viewer suppression across CLI, environment variables, and TOML config."""
    parser = cli._build_parser()
    args = parser.parse_args(["resume", "dummy_run", "--node-rank", "1"])
    assert isinstance(cli._training_viewer(args), nullcontext)

    args_no_rank = parser.parse_args(["resume", "dummy_run"])
    with patch.dict(os.environ, {"SLURM_NODEID": "1"}):
        assert isinstance(cli._training_viewer(args_no_rank), nullcontext)
    with patch.dict(os.environ, {"NODE_RANK": "1"}):
        assert isinstance(cli._training_viewer(args_no_rank), nullcontext)
    with patch.dict(os.environ, {"GROUP_RANK": "1"}):
        assert isinstance(cli._training_viewer(args_no_rank), nullcontext)
    assert isinstance(cli._training_viewer(args_no_rank, config={"run": {"node_rank": 1}}), nullcontext)


def test_leader_resume_single_device_cpu_rejected_with_replicated_gloo(tmp_path):
    """Leader resuming a single-device CPU run with replicated Gloo profile or multi-node flags must fail."""
    cpu_proj = write_default(tmp_path / "cpu_proj", device="cpu", defaults="hypergan")
    cpu_cfg = load_config(cpu_proj)
    cpu_run = tmp_path / "cpu_run"
    cpu_run.mkdir()
    (cpu_run / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "cpu_run", "config": config_values(cpu_cfg),
        "config_sha256": fingerprint(cpu_cfg), "next_sample_sequence": 0
    }))
    (cpu_run / "checkpoints").mkdir()
    (cpu_run / "checkpoints" / "latest.json").write_text(json.dumps({
        "schema_version": 1, "checkpoint": "snap", "step": 0, "kind": "hypergan-training-checkpoint"
    }))
    (cpu_run / "checkpoints" / "snap").mkdir()
    (cpu_run / "checkpoints" / "snap" / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "cpu_run", "step": 0, "kind": "hypergan-training-checkpoint",
        "config_sha256": fingerprint(cpu_cfg), "event_boundary": {"step": 0, "sequence": 0}
    }))

    with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
        prepare_resume(cpu_run, profile="cpu-replicated-gloo")

    with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
        prepare_resume(cpu_run, nodes=2, node_rank=0, rendezvous="tcp://127.0.0.1:29500")


def test_leader_resume_single_device_cuda_rejected_with_replicated_nccl(tmp_path):
    """Leader resuming a single-device CUDA run with replicated NCCL profile or multi-node flags must fail."""
    cuda_proj = write_default(tmp_path / "cuda_proj", device="cuda", defaults="hypergan")
    cuda_cfg = load_config(cuda_proj)
    cuda_run = tmp_path / "cuda_run"
    cuda_run.mkdir()
    (cuda_run / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "cuda_run", "config": config_values(cuda_cfg),
        "config_sha256": fingerprint(cuda_cfg), "next_sample_sequence": 0
    }))
    (cuda_run / "checkpoints").mkdir()
    (cuda_run / "checkpoints" / "latest.json").write_text(json.dumps({
        "schema_version": 1, "checkpoint": "snap", "step": 0, "kind": "hypergan-training-checkpoint"
    }))
    (cuda_run / "checkpoints" / "snap").mkdir()
    (cuda_run / "checkpoints" / "snap" / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "cuda_run", "step": 0, "kind": "hypergan-training-checkpoint",
        "config_sha256": fingerprint(cuda_cfg), "event_boundary": {"step": 0, "sequence": 0}
    }))

    with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
        prepare_resume(cuda_run, profile="cuda-replicated-nccl")

    with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
        prepare_resume(cuda_run, nodes=2, node_rank=0, rendezvous="tcp://127.0.0.1:29500")


def test_leader_resume_replicated_conflicting_topology_rejected(tmp_path):
    """Leader resuming a replicated run with conflicting nodes/world-size must fail."""
    rep_proj = write_default(tmp_path / "rep_proj", device="cpu", defaults="hypergan")
    rep_cfg = load_config(rep_proj)
    rep_run = tmp_path / "rep_run"
    rep_run.mkdir()
    profile_dict = {
        "schema_version": 1,
        "execution": {
            "name": "cpu-replicated-gloo",
            "accumulation_steps": 1,
            "world_size": 2,
            "nodes": 2,
            "node_rank": 0,
            "ranks_per_node": 1,
            "rendezvous": "tcp://10.0.0.1:29500",
            "aggregation": {"strategy": "gradient_allreduce", "sync_every_steps": 1, "outer_lr": 1.0, "outer_momentum": 0.0}
        }
    }
    resolved_profile = resolve_execution_profile(profile_dict, rep_cfg)
    exec_info = resolved_profile["execution"]
    (rep_run / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "rep_run_id", "config": config_values(rep_cfg),
        "config_sha256": fingerprint(rep_cfg), "next_sample_sequence": 0,
        "checkpoint_every": 100, "preview_every": 0, "preview_keep": 1, "execution": exec_info,
    }))
    droot = rep_run / "distributed-checkpoints"
    snap = droot / "snap"
    snap.mkdir(parents=True)
    journal = EventJournal(rep_run)
    journal.append({"run_id": "rep_run_id", "attempt_id": "att1", "step": 0, "sequence": 1, "event": "start"})
    (snap / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "rep_run_id", "attempt_id": "att1", "step": 0,
        "event_boundary": journal.commit_boundary(), "kind": "hypergan-distributed-training-checkpoint",
        "identity": {
            "config_sha256": fingerprint(rep_cfg), "hypergan_checkpoint_version": 2,
            "topology": {k: v for k, v in exec_info.items() if k not in ("node_rank", "rendezvous")}
        }
    }))
    (droot / "latest.json").write_text(json.dumps({
        "schema_version": 1, "checkpoint": "snap", "step": 0, "kind": "hypergan-distributed-training-checkpoint"
    }))

    bad_profile = deepcopy(resolved_profile)
    bad_profile["execution"]["world_size"] = 4
    bad_profile["execution"]["nodes"] = 4
    bad_profile["execution"]["ranks_per_node"] = 1
    bad_profile["execution"]["local_batch_size"] = 4
    bad_profile["execution"]["microbatch_size"] = 4
    with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
        prepare_resume(rep_run, profile=bad_profile)

    with pytest.raises(ValueError):
        prepare_resume(rep_run, nodes=4, node_rank=0)


def test_leader_resume_replicated_ephemeral_port_succeeds(tmp_path):
    """Leader resuming a replicated run with a changed rendezvous port must succeed."""
    rep_proj = write_default(tmp_path / "rep_proj", device="cpu", defaults="hypergan")
    rep_cfg = load_config(rep_proj)
    rep_run = tmp_path / "rep_run"
    rep_run.mkdir()
    profile_dict = {
        "schema_version": 1,
        "execution": {
            "name": "cpu-replicated-gloo",
            "accumulation_steps": 1,
            "world_size": 2,
            "nodes": 2,
            "node_rank": 0,
            "ranks_per_node": 1,
            "rendezvous": "tcp://10.0.0.1:29500",
            "aggregation": {"strategy": "gradient_allreduce", "sync_every_steps": 1, "outer_lr": 1.0, "outer_momentum": 0.0}
        }
    }
    resolved_profile = resolve_execution_profile(profile_dict, rep_cfg)
    exec_info = resolved_profile["execution"]
    (rep_run / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "rep_run_id", "config": config_values(rep_cfg),
        "config_sha256": fingerprint(rep_cfg), "next_sample_sequence": 0,
        "checkpoint_every": 100, "preview_every": 0, "preview_keep": 1, "execution": exec_info,
    }))
    droot = rep_run / "distributed-checkpoints"
    snap = droot / "snap"
    snap.mkdir(parents=True)
    journal = EventJournal(rep_run)
    journal.append({"run_id": "rep_run_id", "attempt_id": "att1", "step": 0, "sequence": 1, "event": "start"})
    (snap / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "rep_run_id", "attempt_id": "att1", "step": 0,
        "event_boundary": journal.commit_boundary(), "kind": "hypergan-distributed-training-checkpoint",
        "identity": {
            "config_sha256": fingerprint(rep_cfg), "hypergan_checkpoint_version": 2,
            "topology": {k: v for k, v in exec_info.items() if k not in ("node_rank", "rendezvous")}
        }
    }))
    (droot / "latest.json").write_text(json.dumps({
        "schema_version": 1, "checkpoint": "snap", "step": 0, "kind": "hypergan-distributed-training-checkpoint"
    }))

    prep_port = prepare_resume(rep_run, rendezvous="tcp://10.0.0.1:39999")
    assert prep_port.profile["execution"]["rendezvous"] == "tcp://10.0.0.1:39999"

    alt_port_profile = deepcopy(resolved_profile)
    alt_port_profile["execution"]["rendezvous"] = "tcp://10.0.0.1:45678"
    prep_port2 = prepare_resume(rep_run, profile=alt_port_profile)
    assert prep_port2.profile["execution"]["rendezvous"] == "tcp://10.0.0.1:45678"

    with patch("hypergan.replicated_execution.run_resume") as mock_resume:
        mock_resume.return_value = {"status": "completed"}
        rc_port = cli.main(["resume", str(rep_run), "--rendezvous", "tcp://10.0.0.1:39999", "--no-server"])
        assert rc_port == 0
        assert mock_resume.called




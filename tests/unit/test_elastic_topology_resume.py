"""End-to-end and unit tests for Elastic Cross-Topology Resume in HyperGAN 2.0.

All tests execute strictly on CPU / cpu-replicated-gloo without touching physical GPUs
or default viewer ports.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import hypergan.api as hg
from hypergan import cli
from hypergan.artifacts import export_generator, sample
from hypergan.checkpoints import (
    file_sha256,
    read_checkpoint,
    resolve_checkpoint,
    resolve_resume_checkpoint_dir,
    restore_trainer_cross_topology,
)
from hypergan.config import config_values, fingerprint, load_config, resolve_config
from hypergan.distributed_checkpoints import (
    _deserialize,
    _shared_digest,
    distributed_checkpoint_identity,
    prepare_distributed_checkpoint,
)
from hypergan.distributed_commit import CheckpointCommitAuthority
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.execution import prepare_resume, prepare_train
from hypergan.inspection import run_status
from hypergan.model_file import ConfigError
from hypergan.summary import collect_checkpoints
from hypergan.training import ReferenceTrainer
from tests.config_fixtures import write_default


def _write_cpu_project(tmp_path: Path, *, steps: int = 8, batch_size: int = 16, lr_floor: float = 1.0) -> Path:
    proj = write_default(tmp_path / "project", device="cpu", defaults="hypergan")
    text = proj.read_text(encoding="utf-8")
    text = text.replace("steps = 5", f"steps = {steps}")
    text = text.replace("batch_size = 8", f"batch_size = {batch_size}")
    if "lr_floor" not in text:
        text = text.replace(f"steps = {steps}", f"steps = {steps}\nlr_floor = {lr_floor}")
    proj.write_text(text, encoding="utf-8")
    return proj


def _assert_state_tensors_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert set(left.keys()) == set(right.keys())
        for k in left:
            _assert_state_tensors_equal(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_tensors_equal(a, b)
    else:
        assert left == right


def test_scale_up_1_to_2_ranks_and_per_rank_divergence(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=8, batch_size=16)
    run_dir = tmp_path / "run_1_to_2"

    # 1. Start on 1 CPU rank for 4 steps
    m1 = prepare_train(proj, run_dir, stop_after_steps=4, checkpoint_every=4).run()
    assert m1["steps"] == 4
    assert m1["status"] == "stopped"
    assert "execution" not in m1
    ckpt_step4, _, state_step4 = read_checkpoint(run_dir)
    assert ckpt_step4.parent.name == "checkpoints"
    assert state_step4["step"] == 4

    # Verify cross-topology restore directly on two simulated ranks draws disjoint data batches
    # and distinct prior/noise RNG streams while sharing identical model/EMA/prior/optimizer state.
    cfg = load_config(proj)
    strategy_r0 = {
        "name": "cpu-replicated-gloo",
        "world_size": 2,
        "local_batch_size": 8,
        "global_batch_size": 16,
        "accumulation_steps": 1,
        "microbatch_size": 8,
        "data_sampling": "replicated-global-draw-rank-slice",
    }
    t0 = ReferenceTrainer(cfg)
    t0.rank, t0.world_size, t0.local_batch_size, t0.strategy_info = 0, 2, 8, strategy_r0
    t1 = ReferenceTrainer(cfg)
    t1.rank, t1.world_size, t1.local_batch_size, t1.strategy_info = 1, 2, 8, strategy_r0
    b0_restored = restore_trainer_cross_topology(t0, state_step4, rank=0, world_size=2)
    b1_restored = restore_trainer_cross_topology(t1, state_step4, rank=1, world_size=2)

    assert t0.step == t1.step == 4
    _assert_state_tensors_equal(t0.graph.state_dict(), state_step4["graph"])
    _assert_state_tensors_equal(t0.graph.state_dict(), t1.graph.state_dict())
    _assert_state_tensors_equal(t0.ema_graph.state_dict(), t1.ema_graph.state_dict())
    # Adapted last_batch matches new local_batch_size=8
    batch_key = next(k for k, v in b0_restored.items() if isinstance(v, torch.Tensor))
    assert b0_restored[batch_key].shape[0] == 8 and b1_restored[batch_key].shape[0] == 8
    # Data stream is synchronized for replicated-global-draw-rank-slice so rank 0 and rank 1 slice disjoint halves!
    assert torch.equal(t0.streams["data"].get_state(), t1.streams["data"].get_state())
    global_b0 = t0.batch()
    global_b1 = t1.batch()
    _assert_state_tensors_equal(global_b0, global_b1)
    assert not torch.equal(global_b0[batch_key][:8], global_b1[batch_key][8:])
    # Prior, penalty, and noise streams MUST diverge across ranks!
    for stream_name in ("prior", "penalty", "noise"):
        assert not torch.equal(t0.streams[stream_name].get_state(), t1.streams[stream_name].get_state())

    # 2. Resume end-to-end on 2-rank cpu-replicated-gloo from step 4 -> 8
    m2 = prepare_resume(run_dir, profile="cpu-replicated-gloo", checkpoint_every=2).run()
    assert m2["steps"] == 8
    assert m2["status"] == "complete"
    assert m2["execution"]["name"] == "cpu-replicated-gloo"
    assert m2["execution"]["world_size"] == 2
    assert any("Execution topology changed on resume" in w for w in m2["resume_warnings"])
    assert any("Execution topology changed on resume" in w for w in m2["warnings"])
    assert len(m2["topology_history"]) == 1
    assert m2["topology_history"][0]["step"] == 4
    assert m2["topology_history"][0]["from_execution"] is None
    assert m2["topology_history"][0]["to_execution"]["world_size"] == 2

    # Inspect committed distributed checkpoint at step 8: shared weights match, per-rank streams diverge
    dist_latest = json.loads((run_dir / "distributed-checkpoints" / "latest.json").read_text(encoding="utf-8"))
    assert dist_latest["step"] == 8
    dist_ckpt = run_dir / "distributed-checkpoints" / dist_latest["checkpoint"]
    r0 = _deserialize((dist_ckpt / "rank-00000.pt").read_bytes())
    r1 = _deserialize((dist_ckpt / "rank-00001.pt").read_bytes())
    assert _shared_digest(r0) == _shared_digest(r1)
    assert not torch.equal(r0["streams"]["prior"], r1["streams"]["prior"])
    r_bkey = next(k for k, v in r0["last_batch"].items() if isinstance(v, torch.Tensor))
    assert not torch.equal(r0["last_batch"][r_bkey], r1["last_batch"][r_bkey])


def test_scale_down_2_to_1_rank_and_artifacts_status(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=8, batch_size=16)
    run_dir = tmp_path / "run_2_to_1"

    # 1. Start on 2-rank cpu-replicated-gloo and stop at step 4 (no terminal model.pt yet!)
    m1 = prepare_train(proj, run_dir, profile="cpu-replicated-gloo", stop_after_steps=4, checkpoint_every=4).run()
    assert m1["steps"] == 4
    assert m1["status"] == "stopped"
    assert m1["execution"]["world_size"] == 2
    assert not (run_dir / "checkpoints").exists()
    assert (run_dir / "distributed-checkpoints" / "latest.json").is_file()

    # Verify sample, export_generator, and run_status work seamlessly on distributed-only checkpoints!
    sample_path_dist = sample(run_dir, count=4, seed=7, output=tmp_path / "dist_samples.json")
    assert sample_path_dist.is_file()
    assert json.loads(sample_path_dist.read_text(encoding="utf-8"))["step"] == 4

    export_path_dist = export_generator(run_dir, tmp_path / "dist_exported.pt")
    assert export_path_dist.is_file()

    status_dist = run_status(run_dir)
    assert status_dist["step"] == 4
    assert any(c["step"] == 4 for c in status_dist["recent_checkpoints"])

    # 2. Scale down 2 -> 1 using devices="cpu" (SingleProcessExecution)
    m2 = prepare_resume(run_dir, devices="cpu", checkpoint_every=4).run()
    assert m2["steps"] == 8
    assert m2["status"] == "complete"
    assert "execution" not in m2
    assert "service_policy" not in m2
    assert any("Execution topology changed on resume" in w for w in m2["resume_warnings"])
    assert len(m2["topology_history"]) == 1
    assert m2["topology_history"][0]["from_execution"]["world_size"] == 2
    assert m2["topology_history"][0]["to_execution"] is None

    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "execution" not in saved_manifest
    assert "service_policy" not in saved_manifest
    assert (run_dir / "checkpoints" / "latest.json").is_file()
    assert json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))["step"] == 8


def test_rescale_replicated_2_to_4_to_2_and_batch_size_divisibility(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=9, batch_size=16)
    run_dir = tmp_path / "run_2_4_2"

    # Step 0 -> 3 on 2 ranks
    m1 = prepare_train(proj, run_dir, profile="cpu-replicated-gloo", stop_after_steps=3, checkpoint_every=3).run()
    assert m1["steps"] == 3
    assert m1["execution"]["world_size"] == 2

    # Non-divisible world_size=3 with batch_size=16 raises clear actionable error
    profile_3 = {"schema_version": 1, "execution": {"name": "cpu-replicated-gloo", "world_size": 3}}
    with pytest.raises(ValueError, match=r"Global training\.batch_size must divide evenly across execution\.world_size \(got batch_size=16, world_size=3\)"):
        prepare_resume(run_dir, profile=profile_3)

    # Step 3 -> 6 on 4 ranks
    profile_4 = {"schema_version": 1, "execution": {"name": "cpu-replicated-gloo", "world_size": 4}}
    m2 = prepare_resume(run_dir, profile=profile_4, stop_after_steps=3, checkpoint_every=3).run()
    assert m2["steps"] == 6
    assert m2["execution"]["world_size"] == 4
    assert m2["execution"]["local_batch_size"] == 4
    ckpt6 = Path(m2["checkpoint_path"])
    assert len(list(ckpt6.glob("rank-*.pt"))) == 4

    # Step 6 -> 9 scaling back to 2 ranks AND explicitly adjusting batch_size via sets!
    profile_3_scaled = {"schema_version": 1, "execution": {"name": "cpu-replicated-gloo", "world_size": 3}}
    m3 = hg.prepare_resume(
        run_dir,
        profile=profile_3_scaled,
        sets={"train.batch_size": 24},
        checkpoint_every=3,
    ).run()
    assert m3["steps"] == 9
    assert m3["status"] == "complete"
    assert m3["execution"]["world_size"] == 3
    assert m3["execution"]["global_batch_size"] == 24
    assert m3["execution"]["local_batch_size"] == 8
    assert len(m3["topology_history"]) == 2


def test_ping_pong_1_to_2_to_1_latest_and_explicit_checkpoint_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=9, batch_size=16)
    run_dir = tmp_path / "run_ping_pong"

    # Phase 1: 1 rank, steps 0 -> 3 (writes checkpoints/ at step 3)
    m1 = prepare_train(proj, run_dir, stop_after_steps=3, checkpoint_every=3).run()
    assert m1["steps"] == 3

    # Phase 2: 2 ranks, steps 3 -> 6 (writes distributed-checkpoints/ at step 6; checkpoints/latest.json remains step 3!)
    m2 = prepare_resume(run_dir, profile="cpu-replicated-gloo", stop_after_steps=3, checkpoint_every=3).run()
    assert m2["steps"] == 6
    assert json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))["step"] == 3
    assert json.loads((run_dir / "distributed-checkpoints" / "latest.json").read_text(encoding="utf-8"))["step"] == 6

    # Phase 3: 1 rank, checkpoint=None MUST pick step 6 from distributed-checkpoints/, NEVER stale step 3!
    prep3 = prepare_resume(run_dir, devices="cpu", checkpoint_every=3)
    assert prep3.checkpoint.parent.name == "distributed-checkpoints"
    assert json.loads((prep3.checkpoint / "manifest.json").read_text(encoding="utf-8"))["step"] == 6
    m3 = prep3.run()
    assert m3["steps"] == 9
    assert m3["status"] == "complete"
    assert Path(m3["resumed_from"]).parent.name == "distributed-checkpoints"

    # Verify latest & explicit step resolution across both checkpoint roots
    latest_dir, _ = resolve_resume_checkpoint_dir(run_dir, None)
    assert latest_dir.parent.name == "checkpoints"
    assert json.loads((latest_dir / "manifest.json").read_text(encoding="utf-8"))["step"] == 9

    ckpt_6 = resolve_checkpoint(run_dir, 6)
    assert ckpt_6.parent.name == "distributed-checkpoints"
    ckpt_3 = resolve_checkpoint(run_dir, 3)
    assert ckpt_3.parent.name == "checkpoints"

    all_ckpts = collect_checkpoints(run_dir)
    recorded_steps = sorted({c["step"] for c in all_ckpts})
    assert {3, 6, 9}.issubset(recorded_steps)


def test_same_topology_bitwise_determinism_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=6, batch_size=16)

    # 1. Single-process bitwise determinism (6 uninterrupted vs 3 + 3 same-topology resume)
    single_full = tmp_path / "single_full"
    single_split = tmp_path / "single_split"
    prepare_train(proj, single_full, checkpoint_every=6).run()
    prepare_train(proj, single_split, stop_after_steps=3, checkpoint_every=3).run()
    resumed_single = prepare_resume(single_split, checkpoint_every=3).run()
    assert not resumed_single.get("resume_warnings")
    assert "topology_history" not in resumed_single
    _, _, full_s_state = read_checkpoint(single_full)
    _, _, split_s_state = read_checkpoint(single_split)
    _assert_state_tensors_equal(full_s_state["graph"], split_s_state["graph"])
    _assert_state_tensors_equal(full_s_state["ema_graph"], split_s_state["ema_graph"])
    _assert_state_tensors_equal(full_s_state["prior"], split_s_state["prior"])
    _assert_state_tensors_equal(full_s_state["optimizers"], split_s_state["optimizers"])
    _assert_state_tensors_equal(full_s_state["streams"], split_s_state["streams"])

    # 2. Replicated 2-rank bitwise determinism (6 uninterrupted vs 3 + 3 same-topology resume)
    rep_full = tmp_path / "rep_full"
    rep_split = tmp_path / "rep_split"
    prepare_train(proj, rep_full, profile="cpu-replicated-gloo", checkpoint_every=6).run()
    prepare_train(proj, rep_split, profile="cpu-replicated-gloo", stop_after_steps=3, checkpoint_every=3).run()
    resumed_rep = prepare_resume(rep_split, checkpoint_every=3).run()
    assert not resumed_rep.get("resume_warnings")
    assert "topology_history" not in resumed_rep

    full_ckpt = Path(json.loads((rep_full / "manifest.json").read_text(encoding="utf-8"))["checkpoint_path"])
    split_ckpt = Path(json.loads((rep_split / "manifest.json").read_text(encoding="utf-8"))["checkpoint_path"])
    for rank_file in ("rank-00000.pt", "rank-00001.pt"):
        full_r = _deserialize((full_ckpt / rank_file).read_bytes())
        split_r = _deserialize((split_ckpt / rank_file).read_bytes())
        _assert_state_tensors_equal(full_r["graph"], split_r["graph"])
        _assert_state_tensors_equal(full_r["ema_graph"], split_r["ema_graph"])
        _assert_state_tensors_equal(full_r["prior"], split_r["prior"])
        _assert_state_tensors_equal(full_r["optimizers"], split_r["optimizers"])
        _assert_state_tensors_equal(full_r["streams"], split_r["streams"])


def test_device_migration_cuda_checkpoint_to_cpu_with_explicit_device_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=6, batch_size=8)
    run_dir = tmp_path / "cuda_to_cpu_run"

    # Train 3 steps on CPU, then rewrite saved manifest & checkpoint metadata to simulate a CUDA-origin run
    # (including CUDA runtime block and fused=True optimizer param_groups, as saved on a real GPU).
    prepare_train(proj, run_dir, stop_after_steps=3, checkpoint_every=3).run()

    cuda_cfg_raw = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    cuda_cfg_raw["training"]["device"] = "cuda"
    if isinstance(cuda_cfg_raw.get("prior", {}).get("args"), dict) and "device" in cuda_cfg_raw["prior"]["args"]:
        cuda_cfg_raw["prior"]["args"]["device"] = "cuda"
    cuda_resolved = resolve_config(cuda_cfg_raw)
    cuda_sha = fingerprint(cuda_resolved)

    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_manifest["config"] = config_values(cuda_resolved)
    run_manifest["config_sha256"] = cuda_sha
    run_manifest["runtime"] = {
        **run_manifest["runtime"],
        "device": "cuda",
        "cuda": {"name": "Simulated A6000", "capability": [8, 6], "visible_devices": "0"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    ckpt_dir = Path(run_manifest["checkpoint_path"])
    ckpt_manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    ckpt_manifest["config"] = config_values(cuda_resolved)
    ckpt_manifest["config_sha256"] = cuda_sha
    ckpt_manifest["runtime"] = copy.deepcopy(run_manifest["runtime"])
    (ckpt_dir / "manifest.json").write_text(json.dumps(ckpt_manifest), encoding="utf-8")

    state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
    for opt_key in ("opt_d", "opt_g"):
        if opt_key in state.get("optimizers", {}):
            for pg in state["optimizers"][opt_key].get("param_groups", []):
                pg["fused"] = True
    torch.save(state, ckpt_dir / "state.pt")
    ckpt_manifest["state_sha256"] = file_sha256(ckpt_dir / "state.pt")
    (ckpt_dir / "manifest.json").write_text(json.dumps(ckpt_manifest), encoding="utf-8")

    # Implicit config_path with device="cpu" WITHOUT explicit -d / devices= is still rejected
    with pytest.raises(ConfigError, match="training.device"):
        prepare_resume(run_dir, config_path=proj)

    # Explicit devices="cpu" (-d cpu) migrates cleanly from CUDA checkpoint to CPU!
    migrated = prepare_resume(run_dir, devices="cpu", checkpoint_every=3).run()
    assert migrated["steps"] == 6
    assert migrated["status"] == "complete"
    assert migrated["config"]["training"]["device"] == "cpu"
    assert any("Execution topology changed on resume" in w for w in migrated["resume_warnings"])
    assert migrated["topology_history"][-1]["from_device"] == "cuda"
    assert migrated["topology_history"][-1]["to_device"] == "cpu"


def _provenance_worker_entry(rank, world_size, proj_path, run_dir_str, controller_id, boundary):
    import hypergan.distributed_checkpoints as dist_ckpts
    from hypergan.execution_profiles import resolve_execution_profile

    cfg = load_config(proj_path)
    resolved_prof = resolve_execution_profile(
        {"schema_version": 1, "execution": {"name": "cpu-replicated-gloo", "world_size": world_size}},
        cfg,
    )
    strategy = resolved_prof["execution"]
    torch.set_num_threads(1)

    calls = {"count": 0}
    orig_source_info = dist_ckpts.source_info

    def mutating_source_info(c):
        calls["count"] += 1
        base = dict(orig_source_info(c))
        base["hypergan_commit"] = f"commit-{calls['count']:04d}"
        base["hypergan_dirty"] = bool(calls["count"] % 2)
        return base

    dist_ckpts.source_info = mutating_source_info
    try:
        trainer = ReplicatedTrainer(cfg, world_size=world_size, accumulation_steps=1, execution=strategy)
        startup_identity = distributed_checkpoint_identity(trainer)
        _, last_batch = trainer.update()
        # Simulate git commit / dirty worktree change on disk before periodic checkpoint
        receipt = prepare_distributed_checkpoint(
            Path(run_dir_str),
            trainer,
            last_batch,
            {
                "run_id": "r1",
                "attempt_id": "att1",
                "next_sample_sequence": 1,
                "event_boundary": boundary,
            },
            controller_id=controller_id,
            command_sequence=1,
        )
        payload = {"startup_identity": startup_identity, "receipt": receipt, "source_info_calls": calls["count"]}
        (Path(run_dir_str) / f"worker-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
    finally:
        dist_ckpts.source_info = orig_source_info


def test_mid_run_git_commit_or_source_mutation_does_not_crash_distributed_checkpoint(tmp_path, monkeypatch):
    """Regression test: a git commit or file edit on disk mid-run must not break CheckpointCommitAuthority."""
    from hypergan.cpu_workers import launch_cpu_workers
    from hypergan.run_state import EventJournal

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=4, batch_size=8)
    run_dir = tmp_path / "run_provenance_cache"
    run_dir.mkdir(parents=True)
    journal = EventJournal(run_dir)
    journal.append({"run_id": "r1", "attempt_id": "att1", "step": 1, "sequence": 1, "event": "train"})
    boundary = journal.commit_boundary()

    import uuid
    controller_id = uuid.uuid4().hex

    launch_cpu_workers(
        _provenance_worker_entry,
        world_size=2,
        args=(str(proj), str(run_dir), controller_id, boundary),
    )
    worker_results = [
        json.loads((run_dir / f"worker-{r}.json").read_text(encoding="utf-8"))
        for r in range(2)
    ]
    assert worker_results[0]["source_info_calls"] == 1
    assert worker_results[1]["source_info_calls"] == 1

    commit_authority = CheckpointCommitAuthority(
        run_dir,
        run_id="r1",
        attempt_id="att1",
        identity=worker_results[0]["startup_identity"],
    )
    commit_authority._controller_id = controller_id
    try:
        committed = commit_authority.commit(
            worker_results[0]["receipt"],
            expected_command_sequence=1,
            expected_event_boundary=boundary,
        )
        assert committed.is_dir()
    finally:
        commit_authority.close()


def test_cli_e2e_elastic_resume_sample_export_status(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=9, batch_size=16)
    run_dir = tmp_path / "cli_elastic_run"

    # 1. Train 3 steps on single CPU (--no-server avoids port 8765)
    rc1 = cli.main([
        "train", str(proj), "--run-dir", str(run_dir),
        "-d", "cpu", "--stop-after-steps", "3", "--checkpoint-every", "3", "--no-server", "--quiet",
    ])
    assert rc1 == 0
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["steps"] == 3

    # 2. Scale up 1 -> 2 ranks via CLI `hypergan resume --profile cpu-replicated-gloo`
    rc2 = cli.main([
        "resume", str(run_dir), "--profile", "cpu-replicated-gloo",
        "--stop-after-steps", "3", "--checkpoint-every", "3", "--no-server", "--quiet",
    ])
    assert rc2 == 0
    m2 = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m2["steps"] == 6
    assert m2["execution"]["world_size"] == 2

    # 3. Scale down 2 -> 1 rank via CLI `hypergan resume -d cpu`
    rc3 = cli.main([
        "resume", str(run_dir), "-d", "cpu", "--checkpoint-every", "3", "--no-server", "--quiet",
    ])
    assert rc3 == 0
    m3 = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m3["steps"] == 9
    assert m3["status"] == "complete"
    assert "execution" not in m3

    # 4. CLI status, sample, and export
    assert cli.main(["status", str(run_dir), "--json"]) == 0
    out_sample = tmp_path / "cli_sample.json"
    assert cli.main(["sample", str(run_dir), "--count", "4", "--output", str(out_sample)]) == 0
    assert out_sample.is_file()
    out_export = tmp_path / "cli_export.pt"
    assert cli.main(["export", str(run_dir), "--output", str(out_export)]) == 0
    assert out_export.is_file()

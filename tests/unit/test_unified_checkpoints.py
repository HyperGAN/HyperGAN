"""Unit and integration tests for Unified Checkpoints (schema_version=2) and
legacy distributed-checkpoints/ (schema_version=1) backward compatibility.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

from hypergan.artifacts import export_generator, sample
from hypergan.checkpoints import (
    prune_checkpoints,
    resolve_checkpoint,
    resolve_resume_checkpoint_dir,
    trainer_state,
)
from hypergan.config import load_config
from hypergan.distributed_checkpoints import (
    KIND as DIST_KIND,
    _combine_trainer_state,
    _deserialize,
    _deserialize_shard,
    _rank_state,
    _serialize,
    _shared_digest,
    _split_trainer_state,
    distributed_checkpoint_identity,
    read_distributed_checkpoint_for_elastic,
    restore_distributed_checkpoint,
    save_distributed_checkpoint,
)
from hypergan.execution import prepare_resume, prepare_train
from hypergan.metric_evaluation import evaluate
from hypergan.summary import collect_checkpoints
from hypergan.training import ReferenceTrainer
from tests.config_fixtures import write_default


class VectorSpreadProbe:
    """Tiny vector snapshot metric for testing evaluate() on 2D GaussianGrid fixtures."""

    def describe(self):
        return {
            "kind": "scalar",
            "label": "Vector spread probe",
            "unit": "l2",
            "direction": "minimize",
        }

    def evaluate(self, *, batches, context):
        gen_vals = []
        ref_vals = []
        for batch in batches:
            gen_vals.append(batch["generated"].detach().double())
            ref_vals.append(batch["reference"].detach().double())
        g = torch.cat(gen_vals, dim=0)
        r = torch.cat(ref_vals, dim=0)
        return float((g.mean(dim=0) - r.mean(dim=0)).norm().item())


def _write_cpu_project(tmp_path: Path, *, steps: int = 6, batch_size: int = 16) -> Path:
    proj = write_default(tmp_path / "project", device="cpu", defaults="hypergan")
    text = proj.read_text(encoding="utf-8")
    text = text.replace("steps = 5", f"steps = {steps}")
    text = text.replace("batch_size = 8", f"batch_size = {batch_size}")
    if "lr_floor" not in text:
        text = text.replace(f"steps = {steps}", f"steps = {steps}\nlr_floor = 1.0")
    proj.write_text(text, encoding="utf-8")
    return proj


def test_unified_checkpoint_deduplicates_weights_and_writes_model_json(tmp_path, monkeypatch):
    """Verify 4-rank unified checkpoint stores 1x shared state.pt + tiny per-rank shards (< 20 KB)
    and writes model.json at startup on Rank 0.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=2, batch_size=16)

    single_run = tmp_path / "single_run"
    prepare_train(proj, single_run, stop_after_steps=2, checkpoint_every=2).run()
    single_ckpt = resolve_checkpoint(single_run, None)
    single_size = sum(f.stat().st_size for f in single_ckpt.rglob("*") if f.is_file())

    rep4_run = tmp_path / "rep4_run"
    profile_4 = {"schema_version": 1, "execution": {"name": "cpu-replicated-gloo", "world_size": 4}}
    prepare_train(proj, rep4_run, profile=profile_4, stop_after_steps=2, checkpoint_every=2).run()

    # Rank 0 writes model.json for the Web Observatory
    assert (rep4_run / "model.json").is_file()
    model_doc = json.loads((rep4_run / "model.json").read_text(encoding="utf-8"))
    assert model_doc.get("origin") == "recorded" and "components" in model_doc

    # Unified checkpoints/ used; no distributed-checkpoints/ created
    assert (rep4_run / "checkpoints" / "latest.json").is_file()
    assert not (rep4_run / "distributed-checkpoints").exists()

    rep4_ckpt = resolve_checkpoint(rep4_run, None)
    manifest = json.loads((rep4_ckpt / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == DIST_KIND
    assert manifest["state_file"] == "state.pt"
    assert (rep4_ckpt / "state.pt").is_file()
    assert len(manifest["ranks"]) == 4

    for idx, rec in enumerate(manifest["ranks"]):
        assert rec["rank"] == idx
        assert rec["file"] == f"ranks/rank-{idx:05d}.pt"
        shard_path = rep4_ckpt / rec["file"]
        assert shard_path.is_file()
        assert shard_path.stat().st_size == rec["bytes"] < 64 * 1024
        shard = _deserialize_shard(shard_path.read_bytes())
        assert set(shard.keys()) == {"step", "streams", "rng", "data", "last_batch"}
        assert "graph" not in shard and "optimizers" not in shard and "ema_graph" not in shard

    # Total 4-rank shard overhead is tiny (~140 KB for 4 ranks of RNG/stream states), with 0 duplicated model weights
    total_shard_bytes = sum((rep4_ckpt / r["file"]).stat().st_size for r in manifest["ranks"])
    assert total_shard_bytes < 256 * 1024
    state_bytes = (rep4_ckpt / "state.pt").stat().st_size
    assert abs(state_bytes - (single_ckpt / "state.pt").stat().st_size) < 4096


def test_evaluate_supports_unified_and_legacy_distributed_checkpoints(tmp_path, monkeypatch):
    """Verify hypergan evaluate (evaluate()) directly evaluates both v2 unified distributed
    checkpoints and legacy v1 distributed-checkpoints/.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proj = _write_cpu_project(tmp_path, steps=4, batch_size=8)
    # Add a snapshot custom metric to project config
    cfg_file = proj
    cfg_file.write_text(
        cfg_file.read_text(encoding="utf-8")
        + '\n[metrics.custom.probe_eval]\n'
        + 'factory = "tests.unit.test_unified_checkpoints:VectorSpreadProbe"\n'
        + 'mode = "snapshot"\n'
        + 'trigger = "manual"\n'
        + 'inputs = { generated = "evaluation.generated", reference = "evaluation.reference" }\n'
        + '[metrics.custom.probe_eval.evaluation]\n'
        + 'device = "cpu"\n'
        + 'sample_count = 8\n'
        + 'batch_size = 4\n'
        + 'seed = 42\n'
        + 'data = { factory = "gaussian_grid", args = {} }\n'
        + 'generated = "generated"\n',
        encoding="utf-8",
    )

    run_dir = tmp_path / "eval_dist_run"
    prepare_train(proj, run_dir, profile="cpu-replicated-gloo", stop_after_steps=2, checkpoint_every=2).run()

    ckpt_v2 = resolve_checkpoint(run_dir, None)
    res_v2 = evaluate(
        run_dir,
        metric_id="probe_eval",
        checkpoint=ckpt_v2.name,
        devices="cpu",
    )
    assert res_v2["status"] == "complete"
    assert res_v2["source_step"] == 2
    assert isinstance(res_v2["result"]["value"], float)

    # Construct a legacy v1 distributed-checkpoints/ directory from the v2 checkpoint
    legacy_run = tmp_path / "legacy_run"
    legacy_run.mkdir()
    for fname in ("manifest.json", "events.jsonl", "config.toml"):
        if (run_dir / fname).is_file():
            (legacy_run / fname).write_bytes((run_dir / fname).read_bytes())
    legacy_root = legacy_run / "distributed-checkpoints"
    legacy_ckpt = legacy_root / ckpt_v2.name
    legacy_ckpt.mkdir(parents=True)

    v2_info = json.loads((ckpt_v2 / "manifest.json").read_text(encoding="utf-8"))
    shared_state = _deserialize((ckpt_v2 / "state.pt").read_bytes())
    v1_ranks = []
    for idx in range(2):
        shard = _deserialize_shard((ckpt_v2 / "ranks" / f"rank-{idx:05d}.pt").read_bytes())
        full_rank_state = _combine_trainer_state(shared_state, shard)
        raw_bytes = _serialize(full_rank_state)
        r_file = f"rank-{idx:05d}.pt"
        (legacy_ckpt / r_file).write_bytes(raw_bytes)
        v1_ranks.append({
            "rank": idx,
            "file": r_file,
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        })

    v1_info = {
        k: v
        for k, v in v2_info.items()
        if k not in ("state_file", "state_sha256", "state_bytes", "diverged_ranks")
    }
    v1_info["schema_version"] = 1
    v1_info["ranks"] = v1_ranks
    (legacy_ckpt / "manifest.json").write_text(json.dumps(v1_info, indent=2), encoding="utf-8")
    (legacy_root / "latest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "kind": DIST_KIND,
            "checkpoint": legacy_ckpt.name,
            "step": 2,
        }),
        encoding="utf-8",
    )

    # Verify legacy v1 checkpoint works with resolve_checkpoint, sample, export_generator, evaluate, and resume!
    resolved_legacy = resolve_checkpoint(legacy_run, None)
    assert resolved_legacy == legacy_ckpt.resolve()

    sample_out = sample(legacy_run, count=4, seed=7, output=tmp_path / "legacy_samples.json")
    assert sample_out.is_file()

    export_out = export_generator(legacy_run, tmp_path / "legacy_export.pt")
    assert export_out.is_file()

    res_v1 = evaluate(
        legacy_run,
        metric_id="probe_eval",
        checkpoint=legacy_ckpt.name,
        devices="cpu",
    )
    assert res_v1["status"] == "complete"
    assert res_v1["source_step"] == 2
    assert res_v1["result"]["value"] == pytest.approx(res_v2["result"]["value"])

    # Resume legacy v1 run -> writes new v2 checkpoint into unified checkpoints/!
    resumed = prepare_resume(legacy_run, profile="cpu-replicated-gloo", checkpoint_every=2).run()
    assert resumed["steps"] == 4
    assert (legacy_run / "checkpoints" / "latest.json").is_file()
    latest_after_resume, _ = resolve_resume_checkpoint_dir(legacy_run, None)
    assert latest_after_resume.parent.name == "checkpoints"
    assert json.loads((latest_after_resume / "manifest.json").read_text(encoding="utf-8"))["step"] == 4

    # Prune cleans up across both checkpoints/ and legacy distributed-checkpoints/
    prune_checkpoints(legacy_run, policy={"keep_last": 0, "keep_every": 0, "keep_best": 0})
    remaining = collect_checkpoints(legacy_run)
    assert [c["step"] for c in remaining] == [4]


def test_checkpoint_provenance_caches_runtime_package_versions(monkeypatch):
    """Verify mid-run editable package metadata updates do not drift checkpoint identity."""
    import importlib.metadata
    from hypergan.config import resolve_config
    from hypergan.distributed_checkpoints import distributed_checkpoint_identity
    from hypergan.training import ReferenceTrainer

    monkeypatch.setattr("hypergan.distributed_checkpoints._group", lambda: (0, 2))
    monkeypatch.setattr("torch.distributed.get_backend", lambda: "gloo")

    cfg = resolve_config({"training": {"device": "cpu", "batch_size": 8, "steps": 4}})
    trainer = ReferenceTrainer(cfg)
    trainer.rank = 0
    trainer.world_size = 2
    trainer.strategy_info = {
        "name": "cpu-replicated-gloo",
        "world_size": 2,
        "global_batch_size": 8,
        "local_batch_size": 4,
    }

    first = distributed_checkpoint_identity(trainer)
    orig_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "99.99.99" if name == "hypergan" else orig_version(name),
    )
    second = distributed_checkpoint_identity(trainer)
    assert second == first
    assert second["runtime"]["hypergan"] == first["runtime"]["hypergan"]


def test_rank_shard_serialization_supports_high_resolution_batches():
    """256x256+ rank-local batches (e.g. [32, 4, 256, 256] > 32 MiB) must serialize and deserialize cleanly."""
    import torch
    from hypergan.distributed_checkpoints import _deserialize_shard, _serialize_shard

    shard = {
        "step": 5,
        "streams": {"data": b"rng"},
        "rng": {"cpu": b"cpu"},
        "data": {"schema_version": 1, "permutation": list(range(1000)), "cursor": 64, "epoch": 1},
        "last_batch": {
            "real": torch.zeros(32, 3, 256, 256, dtype=torch.float32),
            "gray": torch.zeros(32, 1, 256, 256, dtype=torch.float32),
        },
    }
    payload = _serialize_shard(shard)
    assert len(payload) > 16 * 1024 * 1024
    restored = _deserialize_shard(payload)
    assert restored["step"] == 5
    assert tuple(restored["last_batch"]["real"].shape) == (32, 3, 256, 256)
    assert tuple(restored["last_batch"]["gray"].shape) == (32, 1, 256, 256)

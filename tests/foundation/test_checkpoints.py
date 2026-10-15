"""Unit and integration tests for checkpoint retention, safety invariants S1-S5,
resolution, EMA extraction, and bundle export."""
import hashlib
import json
from pathlib import Path
import pytest
import torch

from hypergan.checkpoints import (
    DEFAULT_CHECKPOINT_POLICY,
    normalize_checkpoint_policy,
    resolve_checkpoint,
    extract_ema_bundle,
    export_checkpoint_bundle,
    prune_checkpoints,
)


def _make_dummy_checkpoint(ckpt_dir: Path, step: int, run_id: str = "test-run", attempt_id: str = "att-1"):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "ema_graph": {"models.generator.weight": torch.randn(2, 2)},
        "ema_prior": {"prior_param": torch.randn(2)},
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    manifest = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "step": step,
        "config": {
            "components": {
                "generator": {
                    "factory": "custom_gen",
                    "inputs": {"z": "prior.z"},
                }
            },
            "sampling": {},
        },
    }
    (ckpt_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_normalize_checkpoint_policy():
    # Default policy
    norm_def = normalize_checkpoint_policy(None)
    assert norm_def["keep_last"] == 3
    assert norm_def["keep_every"] == 50000
    assert norm_def["export_bundle"] == "on_eval"
    assert norm_def["keep_best"]["metric"] == "fid50k_train"
    assert norm_def["keep_best"]["mode"] == "min"
    assert norm_def["keep_best"]["count"] == 2

    # Shorthand keep_best = 5
    norm_int = normalize_checkpoint_policy({"keep_best": 5})
    assert norm_int["keep_best"]["count"] == 5
    assert norm_int["keep_best"]["metric"] == "fid50k_train"

    # Custom dict
    custom = {
        "keep_last": 10,
        "keep_best": {"metric": "loss/d_total", "mode": "min", "count": 1},
        "keep_every": 10000,
        "export_bundle": "never",
    }
    norm_custom = normalize_checkpoint_policy(custom)
    assert norm_custom["keep_last"] == 10
    assert norm_custom["keep_best"]["metric"] == "loss/d_total"
    assert norm_custom["keep_best"]["count"] == 1
    assert norm_custom["keep_every"] == 10000
    assert norm_custom["export_bundle"] == "never"

    # Validation errors
    with pytest.raises(ValueError, match="Unknown field"):
        normalize_checkpoint_policy({"unknown_key": 123})
    with pytest.raises(ValueError, match="keep_last must be a non-negative integer"):
        normalize_checkpoint_policy({"keep_last": -1})
    with pytest.raises(ValueError, match="export_bundle"):
        normalize_checkpoint_policy({"export_bundle": "invalid_choice"})
    with pytest.raises(ValueError, match="mode"):
        normalize_checkpoint_policy({"keep_best": {"metric": "m", "mode": "invalid", "count": 1}})


def test_resolve_checkpoint_targets(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True)

    # 1. Empty checkpoints directory raises ValueError
    with pytest.raises(ValueError, match="No training checkpoints found"):
        resolve_checkpoint(run_dir, "latest")

    # Create synthetic checkpoints: step 100, step 200, step 300
    c100 = ckpt_root / "att-1-step-00000100-uuid1"
    c200 = ckpt_root / "att-1-step-00000200-uuid2"
    c300 = ckpt_root / "att-1-step-00000300-uuid3"
    _make_dummy_checkpoint(c100, 100)
    _make_dummy_checkpoint(c200, 200)
    _make_dummy_checkpoint(c300, 300)

    # 2. Resolve "latest" without latest.json resolves highest step (300)
    resolved_latest = resolve_checkpoint(run_dir, "latest")
    assert resolved_latest == c300

    # 3. Resolve "latest" with latest.json pointing to c200
    (ckpt_root / "latest.json").write_text(json.dumps({"checkpoint": c200.name}), encoding="utf-8")
    assert resolve_checkpoint(run_dir, "latest") == c200

    # 4. Resolve integer / numeric string step
    assert resolve_checkpoint(run_dir, 100) == c100
    assert resolve_checkpoint(run_dir, "200") == c200
    with pytest.raises(ValueError, match="Checkpoint for step 999 not found"):
        resolve_checkpoint(run_dir, 999)

    # 5. Resolve "best" when no evaluations exist raises ValueError
    with pytest.raises(ValueError, match="No evaluation records available"):
        resolve_checkpoint(run_dir, "best")

    # Add evaluation records:
    # step 100: fid = 15.0
    # step 200: fid = 10.0
    # step 300: fid = 10.0 (tie with step 200; tie-breaker: recency = step 300)
    eval_dir = run_dir / "metrics" / "evaluations"
    e1 = eval_dir / "step-00000100-e1"
    e2 = eval_dir / "step-00000200-e2"
    e3 = eval_dir / "step-00000300-e3"
    for ed, s, val in [(e1, 100, 15.0), (e2, 200, 10.0), (e3, 300, 10.0)]:
        ed.mkdir(parents=True)
        receipt = {
            "status": "complete",
            "metric_id": "fid50k_train",
            "source_step": s,
            "result": {"value": val, "step": s},
        }
        (ed / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    best_resolved = resolve_checkpoint(run_dir, "best")
    assert best_resolved == c300  # Tie broke in favor of higher step (300)

    # 6. Resolve path
    assert resolve_checkpoint(run_dir, str(c100)) == c100
    assert resolve_checkpoint(run_dir, str(c100 / "state.pt")) == c100
    assert resolve_checkpoint(run_dir, c200.name) == c200


def test_extract_ema_bundle_and_export(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    _make_dummy_checkpoint(ckpt_dir, 500)

    # Extract bundle
    bundle = extract_ema_bundle(ckpt_dir)
    assert bundle["schema_version"] == 1
    assert bundle["kind"] == "ema-inference"
    assert bundle["resume_supported"] is False
    assert bundle["step"] == 500
    assert "generator" in bundle["components"]
    assert "models.generator.weight" in bundle["model_states"]["generator"] or "weight" in bundle["model_states"]["generator"]
    assert "prior_param" in bundle["prior"]

    # Export bundle
    out_dir = tmp_path / "exported"
    pt_path, sha_path = export_checkpoint_bundle(ckpt_dir, out_dir)
    assert pt_path.is_file()
    assert sha_path.is_file()
    assert pt_path.name == "model.pt"
    assert sha_path.name == "model.sha256"

    # Verify sha256
    expected_digest = hashlib.sha256(pt_path.read_bytes()).hexdigest()
    assert sha_path.read_text(encoding="utf-8").strip() == expected_digest

    # Extract from exported bundle
    exported_bundle = extract_ema_bundle(pt_path)
    assert exported_bundle["kind"] == "ema-inference"
    assert exported_bundle["step"] == 500

    # Corrupt state without ema weights
    bad_ckpt = tmp_path / "bad_ckpt"
    bad_ckpt.mkdir()
    torch.save({"step": 10}, bad_ckpt / "state.pt")
    (bad_ckpt / "manifest.json").write_text(json.dumps({"config": {"components": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing EMA weights"):
        extract_ema_bundle(bad_ckpt)


def test_prune_checkpoints_safety_invariants_s1_to_s5(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True)

    # Create 6 checkpoints: steps 0, 1000, 2000, 3000, 4000, 5000
    ckpts = {}
    for step in [0, 1000, 2000, 3000, 4000, 5000]:
        c_dir = ckpt_root / f"att-1-step-{step:08d}-uuid{step}"
        _make_dummy_checkpoint(c_dir, step)
        ckpts[step] = c_dir

    # Mark 5000 as latest
    (ckpt_root / "latest.json").write_text(json.dumps({"checkpoint": ckpts[5000].name}), encoding="utf-8")

    # Invariant S2: Pin step 2000 with active running evaluation
    eval_dir = run_dir / "metrics" / "evaluations" / "running_eval"
    eval_dir.mkdir(parents=True)
    running_receipt = {
        "status": "running",
        "metric_id": "fid50k_train",
        "source_step": 2000,
    }
    (eval_dir / "receipt.json").write_text(json.dumps(running_receipt), encoding="utf-8")

    # Invariant S5: Evaluations for step 1000 and 3000
    # Step 1000 has best FID (5.0), step 3000 has FID (8.0), step 4000 has FID (25.0)
    for s, fid in [(1000, 5.0), (3000, 8.0), (4000, 25.0)]:
        ed = run_dir / "metrics" / "evaluations" / f"eval_{s}"
        ed.mkdir(parents=True)
        r = {
            "status": "complete",
            "metric_id": "fid50k_train",
            "source_step": s,
            "result": {"value": fid, "step": s},
        }
        (ed / "receipt.json").write_text(json.dumps(r), encoding="utf-8")

    # Policy:
    # keep_last = 1 (retains 5000)
    # keep_every = 5000 (retains 0 and 5000) - Invariant S4
    # keep_best = 1 (retains 1000) - Invariant S5
    # Running eval on 2000 - Invariant S2
    # Latest 5000 - Invariant S1
    # Checkpoint 4000 has newer durable 5000, is not in keep_last(1)=5000, not milestone, not best -> candidate for deletion
    # Checkpoint 3000 is not in keep_best(1), not milestone, not running -> candidate for deletion
    policy = {
        "keep_last": 1,
        "keep_every": 5000,
        "keep_best": {"metric": "fid50k_train", "mode": "min", "count": 1},
        "export_bundle": "never",
    }

    # First test dry_run
    dry_report = prune_checkpoints(run_dir, policy=policy, dry_run=True)
    assert dry_report["dry_run"] is True
    assert dry_report["retained_count"] == 4
    assert dry_report["deleted_count"] == 2

    retained_steps = {r["step"] for r in dry_report["retained"]}
    deleted_steps = {d["step"] for d in dry_report["deleted"]}
    assert retained_steps == {0, 1000, 2000, 5000}
    assert deleted_steps == {3000, 4000}

    # Verify no files were deleted in dry-run
    assert ckpts[3000].is_dir()
    assert ckpts[4000].is_dir()

    # Now execute actual prune
    real_report = prune_checkpoints(run_dir, policy=policy, dry_run=False)
    assert real_report["dry_run"] is False
    assert real_report["deleted_count"] == 2

    # Verify deleted checkpoints are gone
    assert not ckpts[3000].exists()
    assert not ckpts[4000].exists()

    # Verify retained checkpoints still exist
    assert ckpts[0].is_dir()
    assert ckpts[1000].is_dir()
    assert ckpts[2000].is_dir()
    assert ckpts[5000].is_dir()

    # Verify S1: Latest (5000) retained
    # Verify S2: Active evaluation (2000) retained
    # Verify S4: Milestone (0) retained
    # Verify S5: Best (1000) retained


def test_prune_checkpoints_invariant_s3_and_empty_run(tmp_path):
    run_dir = tmp_path / "run"
    assert prune_checkpoints(run_dir)["retained_count"] == 0

    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True)
    assert prune_checkpoints(run_dir)["retained_count"] == 0

    # Single checkpoint: never deleted (S3: no newer durable)
    c10 = ckpt_root / "att-1-step-00000010-uuid10"
    _make_dummy_checkpoint(c10, 10)
    rep = prune_checkpoints(run_dir, policy={"keep_last": 0, "keep_every": 0})
    assert rep["retained_count"] == 1
    assert rep["deleted_count"] == 0
    assert c10.is_dir()

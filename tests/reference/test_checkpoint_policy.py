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

    norm_false = normalize_checkpoint_policy({"keep_best": False})
    assert norm_false["keep_best"] is None

    best_resolved = resolve_checkpoint(run_dir, "best")
    assert best_resolved == c300  # Tie broke in favor of higher step (300)

    # 6. Resolve path
    assert resolve_checkpoint(run_dir, str(c100)) == c100
    assert resolve_checkpoint(run_dir, str(c100 / "state.pt")) == c100
    assert resolve_checkpoint(run_dir, c200.name) == c200

    # 7. A metric with no declared direction and no keep_best entry defines no best
    eval_dir_loss = tmp_path / "run_loss" / "metrics" / "evaluations"
    ckpt_dir_loss = tmp_path / "run_loss" / "checkpoints"
    ckpt_dir_loss.mkdir(parents=True)
    c1_l = ckpt_dir_loss / "att-1-step-00000100-u1"
    c2_l = ckpt_dir_loss / "att-1-step-00000200-u2"
    _make_dummy_checkpoint(c1_l, 100)
    _make_dummy_checkpoint(c2_l, 200)
    el1 = eval_dir_loss / "step-00000100-e1"
    el2 = eval_dir_loss / "step-00000200-e2"
    el1.mkdir(parents=True)
    el2.mkdir(parents=True)
    (el1 / "receipt.json").write_text(json.dumps({"status": "complete", "metric_id": "discriminator_loss", "source_step": 100, "result": {"value": 0.3, "step": 100}}), encoding="utf-8")
    (el2 / "receipt.json").write_text(json.dumps({"status": "complete", "metric_id": "discriminator_loss", "source_step": 200, "result": {"value": 0.8, "step": 200}}), encoding="utf-8")
    with pytest.raises(ValueError, match="No primary metric"):
        resolve_checkpoint(tmp_path / "run_loss", "best")

    # 8. Test manifest.json keep_best config override
    manifest = {
        "run_id": "test-run",
        "config": {
            "checkpoints": {
                "keep_best": {"metric": "discriminator_loss", "mode": "min", "count": 1}
            }
        }
    }
    (tmp_path / "run_loss" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert resolve_checkpoint(tmp_path / "run_loss", "best") == c1_l


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


class _UnsafeCustomPayload:
    pass


def test_extract_ema_bundle_rejects_unrestricted_pickle_payload(tmp_path):
    import pickle

    ckpt_dir = tmp_path / "unsafe_ckpt"
    ckpt_dir.mkdir(parents=True)
    manifest = {
        "run_id": "test-run",
        "attempt_id": "att-1",
        "step": 10,
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

    # Save a payload containing a custom class not allowed by weights_only=True
    torch.save({"payload": _UnsafeCustomPayload()}, ckpt_dir / "state.pt")
    with pytest.raises(pickle.UnpicklingError):
        extract_ema_bundle(ckpt_dir)

    # Also test model.pt path directly and via directory fallback
    model_dir = tmp_path / "unsafe_model_dir"
    model_dir.mkdir(parents=True)
    torch.save({"kind": "ema-inference", "payload": _UnsafeCustomPayload()}, model_dir / "model.pt")
    with pytest.raises(pickle.UnpicklingError):
        extract_ema_bundle(model_dir / "model.pt")
    with pytest.raises(pickle.UnpicklingError):
        extract_ema_bundle(model_dir)


_EVAL_FIXTURE_MODULE = """
class ToyDistance:
    def describe(self):
        return {"kind": "scalar", "label": "Toy distance"}

    def evaluate(self, *, batches, context):
        vals = [(b["generated"] - b["reference"]).square().mean().item() for b in batches]
        return sum(vals) / len(vals)
"""


def _write_tiny_train_config(
    path: Path,
    *,
    steps: int = 3,
    checkpoints_toml: str = "",
    eval_interval: int | None = None,
) -> Path:
    from tests.config_fixtures import write_default

    write_default(path, device="cpu", defaults="hypergan")
    text = (
        path.read_text(encoding="utf-8")
        .replace("steps = 5", f"steps = {steps}")
        .replace("batch_size = 16", "batch_size = 8")
        .replace("num_particles = 20000", "num_particles = 32")
        .replace("count = 256", "count = 4")
    )
    if eval_interval is not None:
        text += f"""
[metrics.custom.distance]
factory = "ckpt_eval_fixture:ToyDistance"
mode = "snapshot"
trigger = "interval"
every_steps = {eval_interval}
on_busy = "skip"
on_error = "fail"
timeout = 30
inputs = {{generated="evaluation.generated", reference="evaluation.reference"}}

[metrics.custom.distance.evaluation]
device = "cpu"
sample_count = 4
batch_size = 4
seed = 17
[metrics.custom.distance.evaluation.data]
factory = "gaussian_grid"
args = {{side=10, noise=0.015}}
"""
    if checkpoints_toml:
        text += "\n" + checkpoints_toml.strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_export_checkpoint_bundle_atomic_and_cleans_tmp_on_failure(tmp_path, monkeypatch):
    from hypergan.checkpoints import atomic_copy_bundle

    c100 = tmp_path / "c100"
    c200 = tmp_path / "c200"
    _make_dummy_checkpoint(c100, 100)
    _make_dummy_checkpoint(c200, 200)

    out_dir = tmp_path / "out"
    pt_path, sha_path = export_checkpoint_bundle(c100, out_dir)
    assert pt_path == (out_dir / "model.pt").resolve()
    assert sha_path == (out_dir / "model.sha256").resolve()
    first_digest = hashlib.sha256(pt_path.read_bytes()).hexdigest()
    assert sha_path.read_text(encoding="utf-8") == first_digest + "\n"
    assert not list(out_dir.glob("*.tmp"))

    real_save = torch.save

    def failing_save(obj, f, *args, **kwargs):
        if hasattr(f, "write"):
            f.write(b"partial-corrupt-bundle")
        raise OSError("simulated disk full during bundle export")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="simulated disk full"):
        export_checkpoint_bundle(c200, out_dir)
    monkeypatch.setattr(torch, "save", real_save)

    # Previous valid bundle and checksum remain untouched, and temporary files are removed.
    assert hashlib.sha256(pt_path.read_bytes()).hexdigest() == first_digest
    assert sha_path.read_text(encoding="utf-8") == first_digest + "\n"
    loaded_raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    assert loaded_raw["step"] == 100
    assert not list(out_dir.glob("*.tmp"))

    # Also verify torch-free atomic_copy_bundle and cleanup on failure.
    copy_dir = tmp_path / "copy_out"
    copied_pt, copied_sha = atomic_copy_bundle(pt_path, first_digest, copy_dir)
    assert copied_pt.read_bytes() == pt_path.read_bytes()
    assert copied_sha.read_text(encoding="utf-8") == first_digest + "\n"
    assert not list(copy_dir.glob("*.tmp"))


def test_train_export_bundle_on_checkpoint_and_failure_isolation(tmp_path, monkeypatch):
    import hypergan as hg
    import hypergan.checkpoints as ckpt_mod

    # 1. Minimal config without [checkpoints] does NOT write <run_dir>/model.pt
    cfg_minimal = _write_tiny_train_config(tmp_path / "minimal.toml", steps=2)
    run_min = hg.train(cfg_minimal, tmp_path / "run_minimal", checkpoint_every=1, device="cpu")
    assert not (run_min.path / "model.pt").exists()
    assert not (run_min.path / "model.sha256").exists()
    assert "exported_bundle_path" not in run_min.manifest

    # 2. export_bundle = "on_checkpoint" atomically writes model.pt + model.sha256
    cfg_ckpt = _write_tiny_train_config(
        tmp_path / "on_ckpt.toml",
        steps=3,
        checkpoints_toml='[checkpoints]\nexport_bundle = "on_checkpoint"',
    )
    run_ckpt = hg.train(cfg_ckpt, tmp_path / "run_on_ckpt", checkpoint_every=2, device="cpu")
    model_pt = run_ckpt.path / "model.pt"
    model_sha = run_ckpt.path / "model.sha256"
    assert model_pt.is_file()
    assert model_sha.is_file()
    assert hashlib.sha256(model_pt.read_bytes()).hexdigest() == model_sha.read_text(encoding="utf-8").strip()
    assert run_ckpt.manifest["exported_bundle_path"] == str(model_pt)

    bundle = hg.load_bundle(model_pt, device="cpu")
    assert bundle["step"] == 3
    samples = bundle.sample(count=2, seed=7)
    assert samples.shape == (2, 2)
    assert torch.isfinite(samples).all()

    # 3. Export hiccup is isolated as an observer_error without invalidating training checkpoints
    def broken_export(*args, **kwargs):
        raise OSError("transient export hiccup")

    monkeypatch.setattr(ckpt_mod, "export_checkpoint_bundle", broken_export)
    run_err = hg.train(cfg_ckpt, tmp_path / "run_ckpt_err", checkpoint_every=2, device="cpu")
    assert run_err.manifest["status"] == "complete"
    assert Path(run_err.manifest["checkpoint_path"]).is_dir()
    assert any(err.get("source") == "export_bundle" for err in run_err.manifest.get("observation_errors", []))


def test_train_export_bundle_on_eval_and_never(tmp_path, monkeypatch):
    import hypergan as hg

    (tmp_path / "ckpt_eval_fixture.py").write_text(_EVAL_FIXTURE_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    # 1. export_bundle = "on_eval" with eval every 2 steps and checkpoint_every = 3 (total steps = 3):
    # Step 2 has NO training checkpoint in checkpoints/, yet model.pt is exported at step 2!
    cfg_eval = _write_tiny_train_config(
        tmp_path / "on_eval.toml",
        steps=3,
        eval_interval=2,
        checkpoints_toml='[checkpoints]\nexport_bundle = "on_eval"',
    )
    run_eval = hg.train(cfg_eval, tmp_path / "run_on_eval", checkpoint_every=3, device="cpu")
    ckpt_names = [d.name for d in (run_eval.path / "checkpoints").iterdir() if d.is_dir() and not d.name.startswith(".")]
    assert not any("-step-00000002-" in name for name in ckpt_names)

    model_pt = run_eval.path / "model.pt"
    model_sha = run_eval.path / "model.sha256"
    assert model_pt.is_file()
    assert model_sha.is_file()
    assert hashlib.sha256(model_pt.read_bytes()).hexdigest() == model_sha.read_text(encoding="utf-8").strip()
    assert run_eval.manifest["exported_bundle_path"] == str(model_pt)

    bundle = hg.load_bundle(model_pt, device="cpu")
    assert bundle["step"] == 2
    samples = bundle.sample(count=2, seed=11)
    assert samples.shape == (2, 2)
    assert torch.isfinite(samples).all()

    # 2. export_bundle = "never" writes no <run_dir>/model.pt during checkpoints or evaluations
    cfg_never = _write_tiny_train_config(
        tmp_path / "never.toml",
        steps=2,
        eval_interval=2,
        checkpoints_toml='[checkpoints]\nexport_bundle = "never"',
    )
    run_never = hg.train(cfg_never, tmp_path / "run_never", checkpoint_every=1, device="cpu")
    assert not (run_never.path / "model.pt").exists()
    assert not (run_never.path / "model.sha256").exists()
    assert "exported_bundle_path" not in run_never.manifest


def test_config_warns_on_keep_best_and_checkpoint_every_mismatch(tmp_path):
    import hypergan as hg
    from hypergan.config import load_config

    # Indivisible eval every_steps=2 vs checkpoint_every=3 with keep_best enabled -> warns
    cfg_mismatch = _write_tiny_train_config(
        tmp_path / "mismatch.toml",
        steps=6,
        eval_interval=2,
        checkpoints_toml=(
            '[run]\ncheckpoint_every = 3\n\n'
            '[checkpoints]\nkeep_best = { metric = "distance", mode = "min", count = 2 }'
        ),
    )
    resolved = load_config(cfg_mismatch)
    assert any(
        "not divisible by checkpoint_every (3)" in w and "keep_best" in w
        for w in resolved["warnings"]
    )

    # Divisible eval every_steps=6 vs checkpoint_every=3 -> no cadence warning
    cfg_aligned = _write_tiny_train_config(
        tmp_path / "aligned.toml",
        steps=6,
        eval_interval=6,
        checkpoints_toml=(
            '[run]\ncheckpoint_every = 3\n\n'
            '[checkpoints]\nkeep_best = { metric = "distance", mode = "min", count = 2 }'
        ),
    )
    resolved_aligned = load_config(cfg_aligned)
    assert not any("not divisible by checkpoint_every" in w for w in resolved_aligned["warnings"])

    # Disabled keep_best (count = 0) -> no cadence warning
    cfg_disabled = _write_tiny_train_config(
        tmp_path / "disabled_best.toml",
        steps=6,
        eval_interval=2,
        checkpoints_toml='[run]\ncheckpoint_every = 3\n\n[checkpoints]\nkeep_best = 0',
    )
    resolved_disabled = load_config(cfg_disabled)
    assert not any("not divisible by checkpoint_every" in w for w in resolved_disabled["warnings"])

    # Also verify model-file [train] checkpoint_every + shorthand keep_best + [evaluations.<name>]
    base_model = Path(hg.load("gaussian-grid").path).read_text(encoding="utf-8")
    model_toml = tmp_path / "model.toml"
    model_toml.write_text(
        base_model.replace('device = "cpu"', 'device = "cpu"\ncheckpoint_every = 3')
        + """
[evaluations.distance]
fn = "ckpt_eval_fixture:ToyDistance"
device = "cpu"
every = 2
samples = 4
batch_size = 4
seed = 17
inputs = { generated = "evaluation.generated", reference = "evaluation.reference" }
data = { builtin = "gaussian_grid", args = { side = 10, noise = 0.015 } }

[checkpoints]
keep_best = 2
""",
        encoding="utf-8",
    )
    from hypergan.config import resolve_config

    lowered = resolve_config(hg.lower(hg.load(model_toml)))
    assert any(
        "not divisible by checkpoint_every (3)" in w and "keep_best" in w
        for w in lowered["warnings"]
    )

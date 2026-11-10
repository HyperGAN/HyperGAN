"""Unit and integration tests for lock-free read-only metric evaluation,
concurrent execution under active run lock, S2 step pinning, and device collision detection."""
import json
from pathlib import Path
import pytest
import torch

from hypergan.config import fingerprint, resolve_config
from hypergan.metric_evaluation import evaluate
from hypergan.run_state import atomic_json, run_lock
from hypergan.checkpoints import prune_checkpoints


def _setup_run(run_dir: Path, step: int = 1000, status: str = "running", train_device: str = "cuda:0"):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config({
        "metrics": {
            "custom": {
                "fid50k_train": {
                    "factory": "math:sin",
                    "mode": "snapshot",
                    "inputs": {"generated": "evaluation.generated"},
                    "evaluation": {
                        "device": "cuda:0",
                        "sample_count": 64,
                        "batch_size": 32,
                        "seed": 42,
                        "data": {"factory": "gaussian_grid", "args": {}},
                        "generated": "components.generator.output",
                    },
                }
            }
        },
        "components": {
            "generator": {
                "factory": "linear",
                "inputs": {"z": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "inputs": {"x": "candidate"},
            },
        },
        "sampling": {},
        "training": {
            "device": "cuda:0" if "," in str(train_device) else train_device,
        },
    })
    cfg_hash = fingerprint(cfg)
    manifest = {
        "run_id": "test-run-m2",
        "attempt_id": "att-1",
        "status": status,
        "config": cfg,
        "config_sha256": cfg_hash,
        "runtime": {
            "device": train_device,
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)

    # Make checkpoint
    ckpt_dir = run_dir / "checkpoints" / f"att-1-step-{step:08d}-uuid1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "ema_graph": {"models.generator.weight": torch.randn(2, 2)},
        "ema_prior": {"prior_param": torch.randn(2)},
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    ckpt_manifest = {
        "run_id": "test-run-m2",
        "attempt_id": "att-1",
        "step": step,
        "config": cfg,
    }
    atomic_json(ckpt_dir / "manifest.json", ckpt_manifest)
    atomic_json(run_dir / "checkpoints" / "latest.json", {"checkpoint": ckpt_dir.name})
    return run_dir, cfg


def test_lock_free_evaluation_under_active_run_lock(tmp_path, monkeypatch):
    run_dir, cfg = _setup_run(tmp_path / "run", step=1000, status="running", train_device="cuda:0")

    # Mock _evaluate_pinned to verify directory, snapshot, pre-execution receipt, and S2 pinning
    recorded_pre_receipt = {}
    recorded_directory = {}

    def mock_evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity, warnings=None):
        recorded_directory["path"] = directory
        recorded_directory["snapshot_exists_during_eval"] = snapshot.is_file()
        receipt_path = directory / "receipt.json"
        recorded_pre_receipt.update(json.loads(receipt_path.read_text(encoding="utf-8")))

        # Invariant S2 verification: while this eval is running, prune_checkpoints must retain step 1000!
        prune_rep = prune_checkpoints(root, policy={"keep_last": 0, "keep_every": 0})
        retained_steps = {r["step"] for r in prune_rep["retained"]}
        assert 1000 in retained_steps, "Active evaluation step was not pinned by Invariant S2!"

        # Return mock complete receipt
        result_receipt = {
            "status": "complete",
            "metric_id": metric_id,
            "source_step": identity["source_step"],
            "evaluation_id": identity["evaluation_id"],
            "warnings": warnings or [],
        }
        atomic_json(receipt_path, result_receipt)
        return result_receipt

    runtime_mock = {
        "fid50k_train": {
            "descriptor": {"kind": "scalar"},
            "factory_sources": {"module": "a" * 64},
            "protocol": "hypergan-metric/v1",
        }
    }
    monkeypatch.setattr("hypergan.metric_evaluation.prepare_custom", lambda c: c.setdefault("_metric_runtime", runtime_mock))
    monkeypatch.setattr("hypergan.metric_evaluation._evaluate_pinned", mock_evaluate_pinned)

    # 1. Acquire run_lock as training would
    with run_lock(run_dir):
        # 2. Call evaluate() concurrently on another device without error
        receipt = evaluate(run_dir, metric_id="fid50k_train", devices="cuda:1")

    # 3. Verify execution and S2 pinning receipt
    assert receipt["status"] == "complete"
    assert receipt["source_step"] == 1000
    assert recorded_pre_receipt["status"] == "running"
    assert recorded_pre_receipt["source_step"] == 1000
    assert recorded_directory["snapshot_exists_during_eval"] is True

    # 4. Verify step-named directory formatting
    eval_dir = recorded_directory["path"]
    assert eval_dir.name.startswith("step-00001000-")

    # 5. Verify snapshot was unlinked in finally:
    assert not (eval_dir / "snapshot.pt").exists()


def test_device_collision_warning_detection(tmp_path, monkeypatch):
    run_dir, cfg = _setup_run(tmp_path / "run", step=2000, status="running", train_device="cuda:0")

    recorded_warnings = []

    def mock_evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity, warnings=None):
        recorded_warnings.extend(warnings or [])
        return {"status": "complete", "warnings": warnings or []}

    runtime_mock = {
        "fid50k_train": {
            "descriptor": {"kind": "scalar"},
            "factory_sources": {"module": "a" * 64},
            "protocol": "hypergan-metric/v1",
        }
    }
    monkeypatch.setattr("hypergan.metric_evaluation.prepare_custom", lambda c: c.setdefault("_metric_runtime", runtime_mock))
    monkeypatch.setattr("hypergan.metric_evaluation._evaluate_pinned", mock_evaluate_pinned)

    # Running training is on cuda:0; evaluating on cuda:0 (or 0) must record DEVICE_COLLISION
    evaluate(run_dir, metric_id="fid50k_train", devices="cuda:0")
    assert any(w.get("code") == "DEVICE_COLLISION" for w in recorded_warnings)

    recorded_warnings.clear()
    # Evaluating on cuda:1 when training is on cuda:0 has no collision
    evaluate(run_dir, metric_id="fid50k_train", devices="cuda:1")
    assert not any(w.get("code") == "DEVICE_COLLISION" for w in recorded_warnings)

    recorded_warnings.clear()
    # Multi-GPU training on "0,1": evaluating on "0" collides, "2" does not
    run_multi, _ = _setup_run(tmp_path / "run_multi", step=2000, status="running", train_device="0,1")
    evaluate(run_multi, metric_id="fid50k_train", devices="0")
    assert any(w.get("code") == "DEVICE_COLLISION" for w in recorded_warnings)

    recorded_warnings.clear()
    evaluate(run_multi, metric_id="fid50k_train", devices="2")
    assert not any(w.get("code") == "DEVICE_COLLISION" for w in recorded_warnings)


def test_evaluate_step_resolution(tmp_path, monkeypatch):
    run_dir, cfg = _setup_run(tmp_path / "run", step=1000, status="complete", train_device="cuda:0")

    # Add a second checkpoint for step 500
    ckpt_500 = run_dir / "checkpoints" / "att-1-step-00000500-uuid500"
    ckpt_500.mkdir(parents=True, exist_ok=True)
    state = {
        "step": 500,
        "ema_graph": {"models.generator.weight": torch.randn(2, 2)},
        "ema_prior": {"prior_param": torch.randn(2)},
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_500 / "state.pt")
    atomic_json(ckpt_500 / "manifest.json", {"run_id": "test-run-m2", "step": 500, "config": cfg})

    evaluated_step = {}

    def mock_evaluate_pinned(root, selected, metric_id, directory, snapshot, expected_hash, identity, warnings=None):
        evaluated_step["step"] = identity["source_step"]
        return {"status": "complete", "step": identity["source_step"]}

    runtime_mock = {
        "fid50k_train": {
            "descriptor": {"kind": "scalar"},
            "factory_sources": {"module": "a" * 64},
            "protocol": "hypergan-metric/v1",
        }
    }
    monkeypatch.setattr("hypergan.metric_evaluation.prepare_custom", lambda c: c.setdefault("_metric_runtime", runtime_mock))
    monkeypatch.setattr("hypergan.metric_evaluation._evaluate_pinned", mock_evaluate_pinned)

    # Evaluate targeting step 500
    evaluate(run_dir, metric_id="fid50k_train", checkpoint=500, devices="cpu")
    assert evaluated_step["step"] == 500


def test_step_named_stream_discovery_and_recovery(tmp_path):
    import asyncio
    from hypergan.web_service import ObservationService
    from hypergan.metric_evaluation import _recover_abandoned

    run_dir, cfg = _setup_run(tmp_path / "run_disc", step=1000, status="running")
    # Simulate a real completed evaluation event and stream.json on disk
    eval_id = "step-00001000-" + "f" * 32
    eval_dir = run_dir / "metrics" / "evaluations" / eval_id
    eval_dir.mkdir(parents=True)
    stream_id = "evaluation:" + eval_id
    event = {
        "schema_version": 2, "event": "evaluation", "run_id": "test-run-m2",
        "attempt_id": eval_id, "sequence": 1, "stream_id": stream_id,
        "stream_generation": eval_id, "step": 1000, "seconds": 0.5,
        "catalog": "c" * 64, "metrics": {"fid50k_train": 12.3}, "status": "complete",
        "source_position_known": True, "evaluation_id": eval_id,
    }
    encoded = json.dumps(event).encode() + b"\n"
    (eval_dir / "events.jsonl").write_bytes(encoded)
    stream_record = {
        "schema_version": 1,
        "stream_id": stream_id,
        "stream_generation": eval_id,
        "run_id": "test-run-m2",
        "path": f"metrics/evaluations/{eval_id}/events.jsonl",
        "role": "measurement",
        "modality": "scalar",
    }
    atomic_json(eval_dir / "stream.json", stream_record)

    # ObservationService should discover it without ValueError
    obs = ObservationService(run_dir)
    obs.run_id = "test-run-m2"
    asyncio.run(obs._discover())
    assert stream_id in obs.streams

    # Test _recover_abandoned on an interrupted step-named evaluation
    abandoned_id = "step-00001000-" + "e" * 32
    abandoned_dir = run_dir / "metrics" / "evaluations" / abandoned_id
    abandoned_dir.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "status": "running",
        "run_id": "test-run-m2",
        "evaluation_id": abandoned_id,
        "source_step": 1000,
        "snapshot_sha256": "s" * 64,
        "metric_id": "fid50k_train",
        "catalog": "c" * 64,
    }
    atomic_json(abandoned_dir / "receipt.json", receipt)
    _recover_abandoned(run_dir, "test-run-m2")
    assert (abandoned_dir / "stream.json").is_file()
    rec = json.loads((abandoned_dir / "receipt.json").read_text())
    assert rec["status"] == "failed"


@pytest.mark.parametrize("eval_device,train_device,collides", [
    ("cuda:0", "cuda:0", True), ("cuda:1", "cuda:0", False), ("cpu", "cuda", False),
    # Bare "cuda" may be any GPU (replicated training spans every visible one).
    ("cuda", "cuda:1", True), ("cuda:1", "cuda", True), ("cuda", "cuda", True),
])
def test_config_and_manual_evaluation_device_warnings_agree(tmp_path, monkeypatch, eval_device, train_device, collides):
    import copy
    from hypergan.metrics import evaluation_warnings
    run_dir, cfg = _setup_run(tmp_path / "run", step=2000, status="running", train_device=train_device)
    config = copy.deepcopy(cfg)
    config["metrics"]["custom"]["fid50k_train"]["evaluation"]["device"] = eval_device
    config_warns = any("may be the training device" in message for message in evaluation_warnings(config))

    recorded = []
    monkeypatch.setattr("hypergan.metric_evaluation.prepare_custom", lambda c: c.setdefault("_metric_runtime", {
        "fid50k_train": {"descriptor": {"kind": "scalar"}, "factory_sources": {"module": "a" * 64},
                         "protocol": "hypergan-metric/v1"}}))
    monkeypatch.setattr("hypergan.metric_evaluation._evaluate_pinned",
                        lambda *args, warnings=None: recorded.extend(warnings or []) or {"status": "complete"})
    evaluate(run_dir, metric_id="fid50k_train", devices=eval_device)
    manual_warns = any(w.get("code") == "DEVICE_COLLISION" for w in recorded)
    assert config_warns == manual_warns == collides


@pytest.mark.parametrize("devices", ["1,2", "gpu", "cuda:x", "tpu", ""])
def test_manual_evaluation_rejects_anything_but_one_device(tmp_path, devices):
    run_dir, _ = _setup_run(tmp_path / "run", step=2000, status="running")
    with pytest.raises(ValueError, match="Evaluation runs on one device"):
        evaluate(run_dir, metric_id="fid50k_train", devices=devices)

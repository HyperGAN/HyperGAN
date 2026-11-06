"""Adversarial Stress Test Suite for Milestone 1 (Challenger 2).

Scope:
1. InceptionFID Scipy 1.18+ & Numerical Integrity:
   - Stress test InceptionFID across varying batch sizes, shapes, and verify that scipy.linalg.sqrtm is never called.
   - Verify numerical determinism and stability under torch-fidelity 0.4.0.
2. Operational Bug Sweep Verification:
   - Metric direction dispatch in inspection.py: stress test compare_runs across 'minimize' (loss, fid) and 'maximize' (accuracy, psnr) metrics.
   - Test suite state isolation: verify that running tests concurrently does NOT pollute ~/.local/share/hypergan/runs.json.
   - CUDA initialization order: test device override when CUDA is already initialized in-process.
"""
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from hypergan.config import fingerprint
from hypergan.execution import prepare_resume, prepare_train
from hypergan.image_metrics import InceptionFID, _Moments, _uint8_rgb
from hypergan.inspection import compare_runs
from hypergan.registry import get_registry_dir, register_run
from hypergan.summary import atomic_write_json
from tests.config_fixtures import write_default


# ==============================================================================
# 1. InceptionFID Scipy 1.18+ & Numerical Integrity
# ==============================================================================

class DummyFeatureExtractor(nn.Module):
    """Synthetic feature extractor returning deterministic 2048-d feature vectors."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.eval()

    def forward(self, x):
        b = x.shape[0]
        # Linear projection of flattened image to 2048 features
        flat = x.float().view(b, -1)
        features = torch.zeros(b, 2048, dtype=torch.float32)
        n_feat = min(flat.shape[1], 32)
        features[:, :n_feat] = flat[:, :n_feat]
        return (features,)


def _setup_mock_inception(tmp_path, monkeypatch):
    """Set up mock weights and dummy feature extractor."""
    weights = tmp_path / "mock_inception.pth"
    weights.write_bytes(b"mock inception model weights")
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()

    monkeypatch.setattr("hypergan.image_metrics._checked_weights", lambda path, expected: Path(path))
    monkeypatch.setattr(
        "torch_fidelity.feature_extractor_inceptionv3.FeatureExtractorInceptionV3",
        DummyFeatureExtractor
    )
    return weights, sha


def test_inception_fid_scipy_sqrtm_is_never_called(tmp_path, monkeypatch):
    """Verify that scipy.linalg.sqrtm is NEVER called during InceptionFID evaluation across batch sizes and shapes."""
    import scipy.linalg

    mock_sqrtm = MagicMock(side_effect=AssertionError("CRITICAL: scipy.linalg.sqrtm was called!"))
    monkeypatch.setattr(scipy.linalg, "sqrtm", mock_sqrtm)

    weights, sha = _setup_mock_inception(tmp_path, monkeypatch)
    fid = InceptionFID(weights, weights_sha256=sha)

    # Test varying batch sizes: 1, 2, 3, 5, 8, 16, 32
    batch_sizes = [1, 2, 3, 5, 8, 16, 32]
    for bsize in batch_sizes:
        num_batches = 2 if bsize >= 8 else 4
        total_samples = bsize * num_batches
        batches = [
            {
                "generated": torch.randn(bsize, 3, 32, 32),
                "reference": torch.randn(bsize, 3, 32, 32) + 0.1,
            }
            for _ in range(num_batches)
        ]
        context = {"sample_count": total_samples}
        score = fid.evaluate(batches=batches, context=context)
        assert isinstance(score, float)
        assert math.isfinite(score)

    # Test mixed / irregular batch sizes: e.g. [3, 7, 2, 4] -> total 16
    mixed_sizes = [3, 7, 2, 4]
    mixed_batches = [
        {
            "generated": torch.randn(sz, 3, 32, 32),
            "reference": torch.randn(sz, 3, 32, 32) + 0.2,
        }
        for sz in mixed_sizes
    ]
    score = fid.evaluate(batches=mixed_batches, context={"sample_count": sum(mixed_sizes)})
    assert isinstance(score, float)
    assert math.isfinite(score)

    # Test varying image shapes: 32x32, 64x64, 128x128, 299x299, 48x64, 33x47, 224x224
    shapes = [
        (32, 32),
        (64, 64),
        (128, 128),
        (299, 299),
        (48, 64),
        (33, 47),
        (224, 224),
    ]
    for h, w in shapes:
        batches = [
            {
                "generated": torch.randn(4, 3, h, w),
                "reference": torch.randn(4, 3, h, w) + 0.1,
            },
            {
                "generated": torch.randn(4, 3, h, w),
                "reference": torch.randn(4, 3, h, w) + 0.1,
            },
        ]
        score = fid.evaluate(batches=batches, context={"sample_count": 8})
        assert isinstance(score, float)
        assert math.isfinite(score)

    # Absolutely verify mock_sqrtm was NEVER called
    assert mock_sqrtm.call_count == 0


def test_inception_fid_numerical_determinism(tmp_path, monkeypatch):
    """Verify bit-exact numerical determinism across multiple evaluation passes with torch-fidelity 0.4.0."""
    weights, sha = _setup_mock_inception(tmp_path, monkeypatch)
    fid = InceptionFID(weights, weights_sha256=sha)

    torch.manual_seed(12345)
    g1 = torch.randn(8, 3, 32, 32)
    r1 = torch.randn(8, 3, 32, 32)

    batches = [{"generated": g1, "reference": r1}]
    context = {"sample_count": 8}

    score1 = fid.evaluate(batches=batches, context=context)
    score2 = fid.evaluate(batches=batches, context=context)
    score3 = fid.evaluate(batches=batches, context=context)

    # Must be bit-exact identical across repeated calls
    assert score1 == score2 == score3


def test_inception_fid_numerical_stability_edge_cases(tmp_path, monkeypatch):
    """Stress test numerical stability under edge conditions: identical inputs, degenerate covariance, bounds."""
    weights, sha = _setup_mock_inception(tmp_path, monkeypatch)
    fid = InceptionFID(weights, weights_sha256=sha)

    # 1. Identical generated and reference distributions -> FID should be ~0.0
    torch.manual_seed(999)
    img1 = torch.randn(6, 3, 32, 32)
    img2 = torch.randn(6, 3, 32, 32)
    batches = [
        {"generated": img1, "reference": img1},
        {"generated": img2, "reference": img2},
    ]
    score_identical = fid.evaluate(batches=batches, context={"sample_count": 12})
    # Mathematically 0.0; within numerical roundoff for 2048-dim rank-11 covariance eigenvalue sum
    assert abs(score_identical) < 0.02

    # 2. Degenerate covariance: all zero/constant generated images
    gen_const = torch.zeros(6, 3, 32, 32)
    ref_norm = torch.randn(6, 3, 32, 32)
    batches_degen = [{"generated": gen_const, "reference": ref_norm}]
    score_degen = fid.evaluate(batches=batches_degen, context={"sample_count": 6})
    assert math.isfinite(score_degen)

    # 3. Minimum allowable sample count (sample_count = 2)
    batches_min2 = [
        {"generated": torch.randn(2, 3, 32, 32), "reference": torch.randn(2, 3, 32, 32)}
    ]
    score_min2 = fid.evaluate(batches=batches_min2, context={"sample_count": 2})
    assert math.isfinite(score_min2)

    # 4. Under minimum sample count (sample_count = 1) -> must raise ValueError
    batches_under1 = [
        {"generated": torch.randn(1, 3, 32, 32), "reference": torch.randn(1, 3, 32, 32)}
    ]
    with pytest.raises(ValueError, match="at least two generated and reference"):
        fid.evaluate(batches=batches_under1, context={"sample_count": 1})

    # 5. Mismatched sample counts
    with pytest.raises(ValueError, match="complete declared generated/reference"):
        fid.evaluate(batches=batches_min2, context={"sample_count": 999})

    # 6. Non-finite input images (NaN / Inf)
    nan_batch = [{"generated": torch.full((2, 3, 32, 32), float("nan")), "reference": torch.randn(2, 3, 32, 32)}]
    with pytest.raises(ValueError, match="finite floating-point"):
        fid.evaluate(batches=nan_batch, context={"sample_count": 2})


def test_inception_fid_tf32_flag_restoration(tmp_path, monkeypatch):
    """Verify torch.backends.cuda.matmul.allow_tf32 and cudnn.allow_tf32 are always restored."""
    weights, sha = _setup_mock_inception(tmp_path, monkeypatch)
    fid = InceptionFID(weights, weights_sha256=sha)

    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32

    # Set to true to verify restoration
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    batches = [{"generated": torch.randn(4, 3, 32, 32), "reference": torch.randn(4, 3, 32, 32)}]
    fid.evaluate(batches=batches, context={"sample_count": 4})

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True

    # Also verify restoration even if evaluation raises an exception
    with pytest.raises(ValueError):
        fid.evaluate(batches=[], context={"sample_count": 10})

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True

    # Restore originals
    torch.backends.cuda.matmul.allow_tf32 = orig_matmul
    torch.backends.cudnn.allow_tf32 = orig_cudnn


# ==============================================================================
# 2. Operational Bug Sweep: Metric Direction Dispatch in inspection.py
# ==============================================================================

def _create_mock_run(run_dir: Path, name: str, evals: list, catalog: dict = None):
    """Helper to populate a mock run directory with summary.json and optional metrics catalog."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "config": {"name": name, "training": {"device": "cpu"}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "name": name,
        "evaluations": evals,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    if catalog:
        cat_dir = run_dir / "metrics"
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_file = cat_dir / "catalog-001.json"
        cat_file.write_text(json.dumps({"metrics": catalog}), encoding="utf-8")


def test_compare_runs_metric_direction_dispatch(tmp_path):
    """Stress test compare_runs step selection under minimize vs maximize metrics and tie-breaking."""
    # Run 1: Minimize metric (FID50k_train)
    # Evals: step 1000 -> 35.0, step 2000 -> 22.0, step 3000 -> 28.0. Best should be 2000 (22.0).
    r1_dir = tmp_path / "run_min_1"
    _create_mock_run(
        r1_dir,
        "run_min_1",
        [
            {"step": 1000, "metric": "fid50k_train", "value": 35.0, "direction": "minimize"},
            {"step": 2000, "metric": "fid50k_train", "value": 22.0, "direction": "minimize"},
            {"step": 3000, "metric": "fid50k_train", "value": 28.0, "direction": "minimize"},
        ]
    )

    # Run 2: Minimize metric with ties
    # Evals: step 1000 -> 15.0, step 2000 -> 15.0. Tie should pick later step 2000.
    r2_dir = tmp_path / "run_min_2"
    _create_mock_run(
        r2_dir,
        "run_min_2",
        [
            {"step": 1000, "metric": "fid50k_train", "value": 15.0, "direction": "minimize"},
            {"step": 2000, "metric": "fid50k_train", "value": 15.0, "direction": "minimize"},
            {"step": 3000, "metric": "fid50k_train", "value": 25.0, "direction": "minimize"},
        ]
    )

    cmp_min = compare_runs([r1_dir, r2_dir], metric="fid50k_train", at_steps=["best", "latest", 1000, 2000])
    mat_min = cmp_min["matrix"]

    # Best for Run 1 is step 2000 (22.0)
    assert mat_min["best"]["run_min_1"]["step"] == 2000
    assert mat_min["best"]["run_min_1"]["value"] == 22.0

    # Best for Run 2 is step 2000 (15.0, tie broken to later step)
    assert mat_min["best"]["run_min_2"]["step"] == 2000
    assert mat_min["best"]["run_min_2"]["value"] == 15.0

    # Latest for Run 1 is step 3000 (28.0)
    assert mat_min["latest"]["run_min_1"]["step"] == 3000
    assert mat_min["latest"]["run_min_1"]["value"] == 28.0

    # Explicit step 1000
    assert mat_min["1000"]["run_min_1"]["value"] == 35.0
    assert mat_min["1000"]["run_min_2"]["value"] == 15.0


def test_compare_runs_maximize_metric(tmp_path):
    """Stress test compare_runs when metric is declared 'maximize' (accuracy, psnr, modes_covered)."""
    # Run 1: Maximize metric (accuracy)
    # Evals: step 1000 -> 0.75, step 2000 -> 0.94, step 3000 -> 0.88. Best should be 2000 (0.94).
    r1_dir = tmp_path / "run_max_1"
    _create_mock_run(
        r1_dir,
        "run_max_1",
        [
            {"step": 1000, "metric": "accuracy", "value": 0.75, "direction": "maximize"},
            {"step": 2000, "metric": "accuracy", "value": 0.94, "direction": "maximize"},
            {"step": 3000, "metric": "accuracy", "value": 0.88, "direction": "maximize"},
        ]
    )

    # Run 2: Maximize metric with ties: step 1000 -> 0.90, step 2000 -> 0.90. Tie should pick later step 2000.
    r2_dir = tmp_path / "run_max_2"
    _create_mock_run(
        r2_dir,
        "run_max_2",
        [
            {"step": 1000, "metric": "accuracy", "value": 0.90, "direction": "maximize"},
            {"step": 2000, "metric": "accuracy", "value": 0.90, "direction": "maximize"},
            {"step": 3000, "metric": "accuracy", "value": 0.82, "direction": "maximize"},
        ]
    )

    cmp_max = compare_runs([r1_dir, r2_dir], metric="accuracy", at_steps=["best", "latest"])
    mat_max = cmp_max["matrix"]

    # Best for Run 1 must be highest value (step 2000, value 0.94), NOT minimum (0.75)
    assert mat_max["best"]["run_max_1"]["step"] == 2000
    assert mat_max["best"]["run_max_1"]["value"] == 0.94

    # Best for Run 2 must be tie-broken to later step 2000
    assert mat_max["best"]["run_max_2"]["step"] == 2000
    assert mat_max["best"]["run_max_2"]["value"] == 0.90


def test_compare_runs_catalog_direction_dispatch(tmp_path):
    """Verify direction is extracted from metrics/catalog-*.json when omitted from summary evaluations."""
    r_dir = tmp_path / "run_catalog"
    catalog = {
        "psnr": {"direction": "maximize", "unit": "dB"}
    }
    # Evaluations omit direction
    evals = [
        {"step": 500, "metric": "psnr", "value": 24.5},
        {"step": 1000, "metric": "psnr", "value": 31.2},
        {"step": 1500, "metric": "psnr", "value": 29.0},
    ]
    _create_mock_run(r_dir, "run_catalog", evals, catalog=catalog)

    cmp_res = compare_runs([r_dir], metric="psnr", at_steps=["best"])
    # Should maximize psnr based on catalog specification
    assert cmp_res["matrix"]["best"]["run_catalog"]["step"] == 1000
    assert cmp_res["matrix"]["best"]["run_catalog"]["value"] == 31.2


# ==============================================================================
# 3. Test Suite State Isolation: Host ~/.local/share/hypergan/runs.json
# ==============================================================================

def _isolate_worker_proc(worker_state_str, worker_runs_dir_str, count):
    """Subprocess task to exercise isolated state directory and registry."""
    import os
    from pathlib import Path
    from hypergan.registry import get_registry_dir, register_run

    worker_state = Path(worker_state_str)
    worker_runs_dir = Path(worker_runs_dir_str)
    os.environ["HYPERGAN_STATE_DIR"] = str(worker_state)

    for i in range(count):
        r_dir = worker_runs_dir / f"run_{i}"
        r_dir.mkdir(parents=True, exist_ok=True)
        register_run(r_dir, name=f"run_{i}")

    reg_dir = get_registry_dir()
    assert reg_dir == worker_state.resolve()
    assert (worker_state / "runs.json").is_file()
    return True


def test_host_runs_json_isolation_under_concurrency(tmp_path):
    """Empirically verify that concurrent test executions do NOT touch host ~/.local/share/hypergan/runs.json."""
    host_runs_json = Path.home() / ".local" / "share" / "hypergan" / "runs.json"

    # Record initial state of host runs.json if it exists
    host_existed = host_runs_json.is_file()
    init_mtime = host_runs_json.stat().st_mtime if host_existed else None
    init_size = host_runs_json.stat().st_size if host_existed else None
    init_hash = hashlib.sha256(host_runs_json.read_bytes()).hexdigest() if host_existed else None

    # Launch 4 concurrent worker processes performing registry operations
    tasks = []
    for i in range(4):
        worker_state = tmp_path / f"worker_{i}" / "state"
        worker_state.mkdir(parents=True, exist_ok=True)
        worker_runs = tmp_path / f"worker_{i}" / "runs"
        worker_runs.mkdir(parents=True, exist_ok=True)
        tasks.append((str(worker_state), str(worker_runs), 5))

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_isolate_worker_proc, s, r, c) for s, r, c in tasks]
        for f in concurrent.futures.as_completed(futures):
            assert f.result() is True

    # Verify host runs.json was NEVER modified or polluted
    if host_existed:
        assert host_runs_json.is_file(), "Host runs.json was deleted!"
        assert host_runs_json.stat().st_size == init_size, "Host runs.json size changed!"
        assert host_runs_json.stat().st_mtime == init_mtime, "Host runs.json mtime was modified!"
        assert hashlib.sha256(host_runs_json.read_bytes()).hexdigest() == init_hash, "Host runs.json content changed!"
    else:
        assert not host_runs_json.exists(), "Host runs.json was created unexpectedly!"


# ==============================================================================
# 4. CUDA Initialization Order & Device Override
# ==============================================================================

def test_cuda_initialization_order_device_override_behavior(tmp_path, monkeypatch):
    """Stress test device selection when CUDA is already initialized in-process."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)

    # 1. When CUDA is initialized in-process, attempting to select GPU devices ('0', '1', 'cuda:0')
    # must be refused with ValueError because CUDA_VISIBLE_DEVICES cannot apply after init.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    path = write_default(tmp_path / "proj1", device="cuda", defaults="hypergan")

    with pytest.raises(ValueError, match="CUDA is already initialized in this process"):
        prepare_train(path, tmp_path / "run1", devices="0")

    with pytest.raises(ValueError, match="CUDA is already initialized in this process"):
        prepare_train(path, tmp_path / "run2", devices="1")

    with pytest.raises(ValueError, match="CUDA is already initialized in this process"):
        prepare_train(path, tmp_path / "run3", devices="cuda:0")

    # 2. Selecting 'cpu' when CUDA is initialized does NOT change GPU mask and must NOT raise ValueError
    prep_cpu = prepare_train(path, tmp_path / "run_cpu", devices="cpu")
    assert prep_cpu.config["training"]["device"] == "cpu"

    # 3. When CUDA_VISIBLE_DEVICES already matches requested value, it should NOT refuse
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    prep_same = prepare_train(path, tmp_path / "run_same", devices="0")
    assert prep_same.config["training"]["device"] == "cuda"

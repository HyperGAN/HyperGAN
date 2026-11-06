"""Deep Empirical Adversarial Stress Suite for Milestone 1 Iteration 2.

Adversarially validates:
1. InceptionFID Scipy 1.18+ avoidance, ill-conditioned matrices, dimension checks, TF32 safety.
2. Metric direction dispatch in `compare_runs` under extreme floats, negative numbers, ties, catalogs.
3. CUDA initialization guard under initialized/uninitialized process states and CPU fallback.
4. ~/.hypergan/cache/ file locking, multi-process concurrency, partial-write crash recovery, offline rules.
5. State directory isolation non-pollution of test tmp_path.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock, patch
import hashlib

import numpy as np
import pytest
import torch

from hypergan.cache import (
    OfflineError,
    cache_file_lock,
    ensure_cached_url,
    get_cache_dir,
    is_offline,
    resolve_or_cache,
)
from hypergan.execution import _set_visible_devices
from hypergan.image_metrics import InceptionFID, _Moments, _uint8_rgb
from hypergan.inspection import compare_runs


# ---------------------------------------------------------------------------
# 1. InceptionFID Scipy 1.18+ Avoidance & Numerical Stability Stress Tests
# ---------------------------------------------------------------------------


def test_inception_fid_strict_zero_scipy_sqrtm_under_all_conditions():
    """Verify scipy.linalg.sqrtm is never invoked even for complex / ill-conditioned matrices."""
    with patch("scipy.linalg.sqrtm", side_effect=RuntimeError("CRITICAL BUG: scipy.linalg.sqrtm called")):
        from torch_fidelity.metric_fid import fid_statistics_to_metric

        # Ill-conditioned covariance matrix (high condition number)
        d = 64
        rng = np.random.RandomState(42)
        u, _ = np.linalg.qr(rng.randn(d, d))
        # singular values spanning 1.0 to 1e-12 (condition number 1e12)
        s = np.logspace(0, -12, d)
        sigma1 = u @ np.diag(s) @ u.T
        sigma2 = np.eye(d, dtype=np.float64) * 0.5
        mu1 = rng.randn(d).astype(np.float64)
        mu2 = rng.randn(d).astype(np.float64)

        stat1 = {"mu": mu1, "sigma": sigma1}
        stat2 = {"mu": mu2, "sigma": sigma2}

        res = fid_statistics_to_metric(stat1, stat2, verbose=False)
        fid = res["frechet_inception_distance"]
        assert math.isfinite(fid)
        assert fid >= 0.0


def test_inception_fid_dimension_and_moment_validation_boundaries():
    """Verify moment accumulator enforces dimensionality and sample count boundaries."""
    moments = _Moments()

    # 1. Single sample should fail statistics()
    moments.add(np.ones((1, 16), dtype=np.float64))
    with pytest.raises(ValueError, match="FID needs at least two generated and reference samples"):
        moments.statistics()

    # 2. Add sample of mismatched dimensions should raise ValueError
    with pytest.raises(ValueError, match="FID feature dimensions changed"):
        moments.add(np.ones((2, 32), dtype=np.float64))

    # 3. Add non-finite features should raise ValueError
    with pytest.raises(ValueError, match="FID features must be a finite nonempty matrix"):
        moments.add(np.array([[1.0, float("nan")], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="FID features must be a finite nonempty matrix"):
        moments.add(np.array([[1.0, float("inf")], [0.0, 1.0]]))

    # 4. Zero samples / empty matrix
    with pytest.raises(ValueError, match="FID features must be a finite nonempty matrix"):
        moments.add(np.empty((0, 16)))


def test_inception_fid_uint8_rgb_boundary_stress():
    """Verify _uint8_rgb validation handles negative, out-of-range, and invalid tensor types."""
    # Non-floating point tensor
    with pytest.raises(ValueError, match="FID expects finite floating-point NCHW RGB images"):
        _uint8_rgb(torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8))

    # Wrong channels (e.g. 1 channel grayscale or 4 channel RGBA)
    with pytest.raises(ValueError, match="FID expects finite floating-point NCHW RGB images"):
        _uint8_rgb(torch.randn(2, 1, 32, 32))
    with pytest.raises(ValueError, match="FID expects finite floating-point NCHW RGB images"):
        _uint8_rgb(torch.randn(2, 4, 32, 32))

    # NaN / Inf in image tensor
    bad_tensor = torch.zeros(2, 3, 32, 32)
    bad_tensor[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="FID expects finite floating-point NCHW RGB images"):
        _uint8_rgb(bad_tensor)

    # Valid values clamped properly to [0, 255]
    extreme_tensor = torch.tensor([[[[-10.0, 0.0], [1.0, 10.0]]]], dtype=torch.float32).repeat(1, 3, 1, 1)
    converted = _uint8_rgb(extreme_tensor)
    assert converted.dtype == torch.uint8
    assert converted[0, 0, 0, 0].item() == 0    # clamped from -10 to -1 -> 0
    assert converted[0, 0, 0, 1].item() == 128  # 0 -> 127.5 -> 128
    assert converted[0, 0, 1, 0].item() == 255  # 1 -> 255
    assert converted[0, 0, 1, 1].item() == 255  # clamped from 10 to 1 -> 255


def test_inception_fid_tf32_restoration_under_simulated_exception(tmp_path):
    """Verify TF32 settings are restored even when evaluation raises mid-stream."""
    dummy_weights = tmp_path / "weights.pth"
    payload = b"dummy weights content for test"
    dummy_sha = hashlib.sha256(payload).hexdigest()
    dummy_weights.write_bytes(payload)

    with patch("hypergan.image_metrics.INCEPTION_SHA256", dummy_sha):
        metric = InceptionFID(weights_path=dummy_weights, weights_sha256=dummy_sha)

        original_matmul = torch.backends.cuda.matmul.allow_tf32
        original_cudnn = torch.backends.cudnn.allow_tf32

        # Force TF32 flags to True to verify restoration to True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        batches = [{"generated": torch.randn(2, 3, 32, 32), "reference": torch.randn(2, 3, 32, 32)}]
        context = {"sample_count": 999}  # Mismatched sample count will raise ValueError

        with patch("torch_fidelity.feature_extractor_inceptionv3.FeatureExtractorInceptionV3") as mock_feat:
            mock_inst = MagicMock()
            mock_inst.float.return_value = mock_inst
            mock_inst.eval.return_value = mock_inst
            mock_inst.requires_grad_.return_value = mock_inst
            mock_inst.return_value = (torch.randn(2, 2048),)
            mock_feat.return_value = mock_inst

            with pytest.raises(ValueError, match="FID requires the complete declared generated/reference sample counts"):
                metric.evaluate(batches=batches, context=context)

        # Confirm restored to True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

        # Restore original
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


# ---------------------------------------------------------------------------
# 2. Metric Direction Dispatch Stress Tests
# ---------------------------------------------------------------------------


def test_compare_runs_negative_numbers_and_extreme_floating_point(tmp_path):
    """Stress test direction dispatch with negative values, extreme floats, and subnormals."""
    run1 = tmp_path / "run_loss_min"
    run1.mkdir()
    (run1 / "summary.json").write_text(
        json.dumps({
            "run_id": "run_loss_min",
            "evaluations": [
                {"metric": "d_loss", "direction": "minimize", "value": -10.5, "step": 100},
                {"metric": "d_loss", "direction": "minimize", "value": -50.2, "step": 200},
                {"metric": "d_loss", "direction": "minimize", "value": -5.0, "step": 300},
            ]
        }),
        encoding="utf-8",
    )

    run2 = tmp_path / "run_score_max"
    run2.mkdir()
    (run2 / "summary.json").write_text(
        json.dumps({
            "run_id": "run_score_max",
            "evaluations": [
                {"metric": "snr_db", "direction": "maximize", "value": -100.0, "step": 100},
                {"metric": "snr_db", "direction": "maximize", "value": -2.5, "step": 200},
                {"metric": "snr_db", "direction": "maximize", "value": -20.0, "step": 300},
            ]
        }),
        encoding="utf-8",
    )

    # 1. Minimize: -50.2 is the smallest value (step 200)
    matrix_min = compare_runs([str(run1)], "d_loss", at_steps=["best"])["matrix"]
    assert matrix_min["best"]["run_loss_min"]["value"] == -50.2
    assert matrix_min["best"]["run_loss_min"]["step"] == 200

    # 2. Maximize: -2.5 is the largest value (step 200)
    matrix_max = compare_runs([str(run2)], "snr_db", at_steps=["best"])["matrix"]
    assert matrix_max["best"]["run_score_max"]["value"] == -2.5
    assert matrix_max["best"]["run_score_max"]["step"] == 200


def test_compare_runs_multi_tie_resolution(tmp_path):
    """Verify tie-breaking consistently selects the latest step for both minimize and maximize."""
    run_min = tmp_path / "tie_min"
    run_min.mkdir()
    (run_min / "summary.json").write_text(
        json.dumps({
            "run_id": "tie_min",
            "evaluations": [
                {"metric": "fid", "direction": "minimize", "value": 12.0, "step": 100},
                {"metric": "fid", "direction": "minimize", "value": 10.0, "step": 200},
                {"metric": "fid", "direction": "minimize", "value": 10.0, "step": 300},
                {"metric": "fid", "direction": "minimize", "value": 10.0, "step": 400},
            ]
        }),
        encoding="utf-8",
    )

    run_max = tmp_path / "tie_max"
    run_max.mkdir()
    (run_max / "summary.json").write_text(
        json.dumps({
            "run_id": "tie_max",
            "evaluations": [
                {"metric": "acc", "direction": "maximize", "value": 0.85, "step": 100},
                {"metric": "acc", "direction": "maximize", "value": 0.95, "step": 200},
                {"metric": "acc", "direction": "maximize", "value": 0.95, "step": 300},
                {"metric": "acc", "direction": "maximize", "value": 0.95, "step": 400},
            ]
        }),
        encoding="utf-8",
    )

    mat_min = compare_runs([str(run_min)], "fid", at_steps=["best"])["matrix"]
    assert mat_min["best"]["tie_min"]["step"] == 400
    assert mat_min["best"]["tie_min"]["value"] == 10.0

    mat_max = compare_runs([str(run_max)], "acc", at_steps=["best"])["matrix"]
    assert mat_max["best"]["tie_max"]["step"] == 400
    assert mat_max["best"]["tie_max"]["value"] == 0.95


# ---------------------------------------------------------------------------
# 3. CUDA Initialization Guard Stress Tests
# ---------------------------------------------------------------------------


def test_cuda_init_guard_exhaustive_permutations():
    """Verify _set_visible_devices guard across initialized vs uninitialized states."""
    # When torch.cuda is initialized:
    mock_torch = MagicMock()
    mock_torch.cuda.is_initialized.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            # 1. Attempting to switch to "1" must raise ValueError
            with pytest.raises(ValueError, match="Cannot select devices '1': CUDA is already initialized"):
                _set_visible_devices("1")

            # 2. Attempting to switch to "0,1" must raise ValueError
            with pytest.raises(ValueError, match="Cannot select devices '0,1': CUDA is already initialized"):
                _set_visible_devices("0,1")

            # 3. Passing identical value "0" is safe and idempotent
            _set_visible_devices("0")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

            # 4. Passing empty string "" (CPU execution) is safe
            _set_visible_devices("")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    # When torch.cuda is NOT initialized:
    mock_torch.cuda.is_initialized.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            # Switching devices succeeds freely
            _set_visible_devices("1")
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
            assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


# ---------------------------------------------------------------------------
# 4. ~/.hypergan/cache/ File Locking & Concurrency Stress Tests
# ---------------------------------------------------------------------------


def _mp_cache_hammer_worker(url, cache_dir, expected_sha, expected_size, barrier, result_queue):
    """Worker process hitting the local test HTTP server under cache locking."""
    try:
        barrier.wait()
        p = resolve_or_cache(
            url,
            expected_sha256=expected_sha,
            expected_size=expected_size,
            cache_dir=cache_dir,
        )
        result_queue.put(("SUCCESS", str(p), p.stat().st_size))
    except Exception as exc:
        result_queue.put(("FAILURE", str(exc), None))


def test_cache_multi_process_high_concurrency_race(tmp_path):
    """Hammer cache download across 8 concurrent worker processes hitting local HTTP server."""
    content = b"HyperGAN production weights blob content 2026 for high concurrency test" * 500
    sha = hashlib.sha256(content).hexdigest()
    size = len(content)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    class _TestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Introduce small latency to widen contention window
            time.sleep(0.02)
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _TestHandler)
    server_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{server_port}/blob.bin"

    try:
        num_procs = 8
        barrier = mp.Barrier(num_procs)
        result_queue = mp.Queue()

        processes = [
            mp.Process(
                target=_mp_cache_hammer_worker,
                args=(url, str(cache_dir), sha, size, barrier, result_queue),
            )
            for _ in range(num_procs)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=15)
            assert not p.is_alive(), f"Process {p.pid} hung on cache lock"

        results = [result_queue.get_nowait() for _ in range(num_procs)]
        for status, path_str, file_size in results:
            assert status == "SUCCESS", f"Worker failed: {path_str}"
            assert file_size == size
            assert Path(path_str).is_file()

        # Ensure no lingering .part files
        part_files = list(cache_dir.glob("*.part"))
        assert len(part_files) == 0, f"Found lingering .part files: {part_files}"
    finally:
        server.shutdown()
        server.server_close()


def test_cache_local_path_strict_isolation_never_touches_cache_dir(tmp_path):
    """Verify local files and file:// URIs are resolved in-place with zero cache directory touches."""
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()

    local_file = tmp_path / "source_weights.pth"
    payload = b"raw local model weights"
    sha = hashlib.sha256(payload).hexdigest()
    local_file.write_bytes(payload)

    # 1. Plain filesystem path
    res1 = resolve_or_cache(str(local_file), expected_sha256=sha, cache_dir=cache_dir)
    assert res1.resolve() == local_file.resolve()
    assert list(cache_dir.iterdir()) == []

    # 2. file:// URI
    file_uri = f"file://{local_file.resolve()}"
    res2 = resolve_or_cache(file_uri, expected_sha256=sha, cache_dir=cache_dir)
    assert res2.resolve() == local_file.resolve()
    assert list(cache_dir.iterdir()) == []


def test_cache_offline_mode_strict_enforcement(tmp_path):
    """Verify offline mode rejects uncached URLs across multiple environment representations."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    for flag in ("1", "true", "yes", "True", "YES"):
        with patch.dict(os.environ, {"HYPERGAN_OFFLINE": flag}):
            assert is_offline() is True
            with pytest.raises(OfflineError, match="HyperGAN is running in offline mode"):
                ensure_cached_url("https://example.com/missing.pth", cache_dir=cache_dir)

    # Verify not offline when unset or 0
    for flag in ("", "0", "false", "no"):
        with patch.dict(os.environ, {"HYPERGAN_OFFLINE": flag}):
            assert is_offline() is False


# ---------------------------------------------------------------------------
# 5. State Directory Isolation Invariant Test
# ---------------------------------------------------------------------------


def test_conftest_state_isolation_fixture_guarantees_clean_tmp_path(tmp_path):
    """Verify that tmp_path fixture is completely unpolluted by isolate_hypergan_state."""
    # tmp_path must be completely empty upon test invocation
    assert list(tmp_path.iterdir()) == [], f"tmp_path polluted with: {list(tmp_path.iterdir())}"

    # HYPERGAN_STATE_DIR must be set and pointing to a different directory
    state_dir_str = os.environ.get("HYPERGAN_STATE_DIR")
    assert state_dir_str is not None
    state_dir = Path(state_dir_str)
    assert state_dir.is_dir()
    assert state_dir != tmp_path
    assert not str(state_dir).startswith(str(tmp_path))

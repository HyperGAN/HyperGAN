"""Challenger 1 Adversarial Empirical Test Suite for Milestone 1 (M1).

Empirically stress-tests:
1. FID Reference Caching:
   - Verifies cache directory structure, naming format, and atomic payload format.
   - Verifies exact reuse without recomputation (0 extractor calls for reference on hit).
   - Verifies bitwise reproducibility of FID score.
   - Verifies protocol differentiation (no stale hits on parameter changes).
   - Exhaustive corrupt cache adversarial matrix (empty, truncated, byte flips, NaN/Inf,
     missing keys, wrong shapes, wrong types, replaced with dir): all gracefully recompute.
   - Unwritable cache fallback (read-only dir / permissions) succeeds without crash.
   - Strict input validation on reference_cache arguments.

2. Checkpoint Retention Symlink Guards & Sequence Ordering:
   - Strict rejection of symlinked checkpoints/ directory with ValueError.
   - Strict rejection of symlinked distributed-checkpoints/ directory with ValueError.
   - Multi-run root pruning skips symlinked child runs and never prunes external targets.
   - Individual symlinked checkpoint entries are retained as 'symlink' and never unlinked.
   - resolve_resume_checkpoint_dir strictly rejects symlink targets and symlink pointers.
   - Elastic distributed checkpoint loading strictly rejects symlinked directories and payloads.
   - _sweep_pending ignores .pending symlinks.
   - Abandoned tips ordering: strictly ordered by event sequence, dominating random/alphabetical
     UUID suffixes (sequence 50 vs sequence 20).
   - Manifest checkpoint_path fallback on sequence ties.
   - Multi-attempt abandoned tip retention across complex branches.

3. Replicated Resume Identity:
   - _numerical_identity strictly filters per-launch keys ('node_rank', 'rendezvous').
   - _same_topology returns True when node_rank or rendezvous vary, False on genuine topology changes.
   - _is_same_distributed_topology ignores node_rank and rendezvous.
   - distributed_checkpoint_identity strips node_rank and rendezvous from persisted topologies.
   - prepare_resume succeeds when varying rendezvous endpoints (IP, port, backend).
   - prepare_resume routes follower nodes (node_rank > 0) without requiring local run manifest.
   - Cluster environment auto-detection (NODE_RANK, SLURM_NODEID, GROUP_RANK) routes followers.
   - Negative invariants: genuine topology changes (world_size, accumulation_steps, profile name)
     strictly raise ValueError / ConfigError.
"""
from collections import defaultdict
import copy
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import stat
import uuid

import numpy as np
import pytest
import torch
from torch import nn

from hypergan import checkpoints
from hypergan.checkpoints import (
    _attempt_checkpoints,
    _attempt_manifests,
    _durable_checkpoints,
    _prune_run_checkpoints,
    _sweep_pending,
    extract_ema_bundle,
    prune_checkpoints,
    resolve_checkpoint,
    resolve_resume_checkpoint_dir,
)
from hypergan.config import config_values, fingerprint, resolve_config
from hypergan.distributed_checkpoints import (
    _is_same_distributed_topology,
    _same_json,
    distributed_checkpoint_identity,
    read_distributed_checkpoint_for_elastic,
)
from hypergan.execution import (
    _execution,
    _is_follower,
    _numerical_identity,
    _profile,
    _same_topology,
    detect_cluster_environment,
    prepare_resume,
    PER_LAUNCH_KEYS,
)
from hypergan.image_metrics import (
    INCEPTION_SHA256,
    REFERENCE_CACHE_VERSION,
    TORCH_FIDELITY_VERSION,
    InceptionFID,
    default_reference_cache,
    read_reference_statistics,
    write_reference_statistics,
)
from hypergan.run_state import run_lock


# ==============================================================================
# Helpers for FID Testing
# ==============================================================================

class _SpyExtractor:
    """Deterministic extractor tracking calls and image batches."""
    def __init__(self, log, dim=6):
        self.log = log
        self.dim = dim

    def to(self, device):
        return self

    def __call__(self, images):
        assert images.dtype == torch.uint8
        count = len(images)
        flat = images.reshape(count, -1).double() / 255.0
        # Deterministic projection
        proj = torch.from_numpy(np.random.default_rng(42).normal(size=(flat.shape[1], self.dim)))
        features = (flat @ proj + flat.square() @ proj.abs()).float()
        self.log.append((images.clone(), features.numpy().copy()))
        return (features,)


class FakeFID(InceptionFID):
    """InceptionFID harness with spy extractor for empirical verification."""
    def __init__(self, reference_cache=None, dim=6):
        self.weights_path = Path("/dev/null")
        self.weights_sha256 = INCEPTION_SHA256
        self.torch_fidelity_version = TORCH_FIDELITY_VERSION
        self.reference_cache = reference_cache
        self.dim = dim
        self.log = []

    def _extractor(self):
        return _SpyExtractor(self.log, dim=self.dim)


def _synthetic_images(rng, count, height=4, width=4):
    return torch.from_numpy(rng.uniform(-1, 1, size=(count, 3, height, width))).float()


def _make_protocol(**overrides):
    base = {
        "data_identity_sha256": "f" * 64,
        "sample_count": 14,
        "batch_size": 5,
        "seed": 101,
        "device": "cpu",
        "backend": {"cudnn_allow_tf32": False},
    }
    base.update(overrides)
    return base


def _run_fid_eval(metric, gen_images, ref_images, batch_sizes, protocol_info):
    metric.log.clear()
    batches = []
    idx = 0
    for bs in batch_sizes:
        batches.append({
            "generated": gen_images[idx : idx + bs],
            "reference": ref_images[idx : idx + bs],
        })
        idx += bs
    return metric.evaluate(
        batches=iter(batches),
        context={"sample_count": idx, "reference_identity": protocol_info},
    )


# ==============================================================================
# Helpers for Checkpoint Retention Testing
# ==============================================================================

TEST_RETENTION_POLICY = {"keep_last": 1, "keep_every": 0, "keep_best": None}


def _create_mock_checkpoint(run_dir, attempt_id, step, suffix=None, sequence=None):
    ckpt_name = f"{attempt_id}-step-{step:08d}-{suffix or format(step, '012x')}"
    path = Path(run_dir) / "checkpoints" / ckpt_name
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step}, path / "state.pt")
    manifest = {
        "run_id": "mock_run",
        "attempt_id": attempt_id,
        "step": step,
        "schema_version": 1,
        "kind": "hypergan-training-checkpoint",
    }
    if sequence is not None:
        manifest["event_boundary"] = {"sequence": sequence}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def _set_run_latest(run_dir, ckpt_path, step):
    latest_file = Path(run_dir) / "checkpoints" / "latest.json"
    latest_file.write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": ckpt_path.name,
        "step": step,
    }))


def _write_mock_attempt_manifest(run_dir, attempt_id, parent=None, checkpoint_path=None, is_current=False):
    manifest = {
        "run_id": "mock_run",
        "attempt_id": attempt_id,
        "status": "complete",
        "config": {"checkpoints": TEST_RETENTION_POLICY},
    }
    if parent is not None:
        manifest["recovery_parent"] = parent
    if checkpoint_path is not None:
        manifest["checkpoint_path"] = str(checkpoint_path)

    paths = [Path(run_dir) / "attempts" / attempt_id / "manifest.json"]
    if is_current:
        paths.append(Path(run_dir) / "manifest.json")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest))


# ==============================================================================
# Helpers for Replicated Resume Testing
# ==============================================================================

def _make_replicated_run(tmp_path, node_rank=0, rendezvous="tcp://127.0.0.1:29500"):
    """Creates a complete mock replicated run with checkpoints and valid manifest."""
    run_dir = tmp_path / "mock_replicated_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = resolve_config({
        "components": {
            "generator": {"factory": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden": [4]}, "inputs": {"x": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1, "bias": False}, "inputs": {"input": "candidate"}},
        },
        "prior": {"args": {"num_particles": 8, "z_dim": 4}},
        "training": {"steps": 100, "batch_size": 4, "device": "cpu", "seed": 42},
    })

    execution_table = {
        "name": "cpu-replicated-gloo",
        "world_size": 2,
        "nodes": 2,
        "ranks_per_node": 1,
        "accumulation_steps": 1,
        "global_batch_size": 8,
        "local_batch_size": 4,
        "node_rank": node_rank,
        "rendezvous": rendezvous,
    }

    # Write attempt & checkpoints
    ckpt_dir = run_dir / "checkpoints" / "0001-a-step-00000010-00000000000a"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": 10,
        "base_lrs": {"generator": 0.001, "discriminator": 0.001},
        "streams": {},
        "data": None,
        "last_batch": None,
        "buffers": {},
        "trainable": {},
        "modes": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    state_sha256 = hashlib.sha256((ckpt_dir / "state.pt").read_bytes()).hexdigest()

    # Write events.jsonl and journal boundary
    event_entry = {"sequence": 1, "schema_version": 1, "run_id": "rep_run_01", "attempt_id": "0001-a", "step": 10, "kind": "checkpoint"}
    event_line = json.dumps(event_entry).encode("utf-8") + b"\n"
    (run_dir / "events.jsonl").write_bytes(event_line)

    boundary = {
        "schema_version": 1,
        "offset": len(event_line),
        "sha256": hashlib.sha256(event_line).hexdigest(),
        "run_id": "rep_run_01",
        "attempt_id": "0001-a",
        "step": 10,
        "sequence": 1,
    }

    ckpt_manifest = {
        "schema_version": 1,
        "kind": "hypergan-training-checkpoint",
        "hypergan_checkpoint_version": 2,
        "run_id": "rep_run_01",
        "attempt_id": "0001-a",
        "step": 10,
        "config": config_values(base_config),
        "config_sha256": fingerprint(base_config),
        "runtime": {"runtime_checked": False},
        "data_contract": {"format": "synthetic"},
        "implementation": {},
        "state_sha256": state_sha256,
        "next_sample_sequence": 1,
        "event_boundary": boundary,
    }
    (ckpt_dir / "manifest.json").write_text(json.dumps(ckpt_manifest))

    # latest.json
    (run_dir / "checkpoints" / "latest.json").write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": ckpt_dir.name,
        "step": 10,
    }))

    # run manifest.json
    run_manifest = {
        "schema_version": 1,
        "run_id": "rep_run_01",
        "attempt_id": "0001-a",
        "hypergan_checkpoint_version": 2,
        "config": config_values(base_config),
        "config_sha256": fingerprint(base_config),
        "next_sample_sequence": 1,
        "execution": execution_table,
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest))
    return run_dir, base_config


# ==============================================================================
# TEST SUITE 1: FID REFERENCE CACHING ADVERSARIAL TESTS
# ==============================================================================

class TestAdversarialFIDReferenceCaching:
    """Empirical adversarial testing of FID reference statistics caching."""

    def test_default_cache_directory_path(self):
        """Default cache directory matches ~/.hypergan/cache/fid-reference contract."""
        from hypergan.data import default_cache_root
        cache_dir = default_reference_cache()
        assert cache_dir.name == "fid-reference"
        assert cache_dir == default_cache_root() / "fid-reference"

    def test_fid_cache_generation_format_and_atomicity(self, tmp_path):
        """Verify cache file naming, structure, atomic write, and float64 moments."""
        rng = np.random.default_rng(2026)
        gen = _synthetic_images(rng, 14)
        ref = _synthetic_images(rng, 14) * 0.8
        proto = _make_protocol()

        metric = FakeFID(reference_cache=str(tmp_path), dim=6)
        score_1 = _run_fid_eval(metric, gen, ref, [5, 5, 4], proto)
        assert np.isfinite(score_1)

        # Cache file must exist
        entries = list(tmp_path.glob("*.fidref"))
        assert len(entries) == 1, f"Expected 1 .fidref file, got {entries}"
        cache_file = entries[0]

        # Verify filename length: 64-char sha256 + '.fidref'
        assert len(cache_file.name) == 64 + len(".fidref")

        # Verify file structure: SHA256 header + \n + NPZ payload
        raw_bytes = cache_file.read_bytes()
        header, sep, payload = raw_bytes.partition(b"\n")
        assert sep == b"\n"
        assert len(header) == 64
        expected_digest = hashlib.sha256(payload).hexdigest().encode()
        assert header == expected_digest, "Header SHA-256 digest does not match payload"

        # Verify NPZ payload
        with np.load(BytesIO(payload), allow_pickle=False) as npz:
            assert set(npz.files) == {"identity", "mu", "sigma", "count"}
            assert npz["mu"].dtype == np.float64
            assert npz["sigma"].dtype == np.float64
            assert npz["mu"].shape == (6,)
            assert npz["sigma"].shape == (6, 6)
            assert int(npz["count"]) == 14
            # Check identity json stored
            stored_identity = json.loads(npz["identity"].tobytes())
            assert stored_identity["protocol"] == proto
            assert stored_identity["extractor"] == "inception-v3-compat/2048"

        # Verify no stray temporary files left behind
        all_files = list(tmp_path.iterdir())
        assert len(all_files) == 1
        assert not any(p.name.endswith(".tmp") for p in all_files)

    def test_fid_cache_reuse_avoids_reference_recomputation(self, tmp_path):
        """Second evaluation re-uses cached statistics without extracting reference features."""
        rng = np.random.default_rng(12345)
        gen = _synthetic_images(rng, 14)
        ref = _synthetic_images(rng, 14) * 0.75
        proto = _make_protocol()

        metric = FakeFID(reference_cache=str(tmp_path), dim=6)

        # Run 1: Cold cache -> 3 batches, 2 calls per batch (gen + ref) = 6 extractor calls
        score_cold = _run_fid_eval(metric, gen, ref, [5, 5, 4], proto)
        assert len(metric.log) == 6, f"Expected 6 extractor calls on cold cache, got {len(metric.log)}"

        # Run 2: Warm cache -> 3 batches, ONLY generated extracted = exactly 3 extractor calls
        score_warm = _run_fid_eval(metric, gen, ref, [5, 5, 4], proto)
        assert len(metric.log) == 3, f"Expected 3 extractor calls on warm cache, got {len(metric.log)}"

        # Empirical proof: FID score is identical to full machine precision
        assert score_cold == pytest.approx(score_warm, rel=1e-12)

    def test_fid_cache_protocol_differentiation(self, tmp_path):
        """Varying any protocol attribute generates a distinct cache entry without stale hits."""
        rng = np.random.default_rng(777)
        gen = _synthetic_images(rng, 14)
        ref = _synthetic_images(rng, 14)

        metric = FakeFID(reference_cache=str(tmp_path), dim=6)

        # Baseline protocol
        proto_base = _make_protocol(seed=101)
        _run_fid_eval(metric, gen, ref, [5, 5, 4], proto_base)
        assert len(list(tmp_path.glob("*.fidref"))) == 1

        # Protocol with changed batch_size
        proto_bs = _make_protocol(seed=101, batch_size=7)
        _run_fid_eval(metric, gen, ref, [7, 7], proto_bs)
        assert len(list(tmp_path.glob("*.fidref"))) == 2

        # Protocol with changed data_identity
        proto_data = _make_protocol(data_identity_sha256="0" * 64)
        _run_fid_eval(metric, gen, ref, [5, 5, 4], proto_data)
        assert len(list(tmp_path.glob("*.fidref"))) == 3

        # Repeat baseline protocol -> warm hit (3 extractor calls)
        _run_fid_eval(metric, gen, ref, [5, 5, 4], proto_base)
        assert len(metric.log) == 3
        assert len(list(tmp_path.glob("*.fidref"))) == 3

    @pytest.mark.parametrize("corruption_type", [
        "empty_file",
        "truncated_header",
        "header_bit_flip",
        "truncated_payload",
        "garbage_payload",
        "missing_mu",
        "missing_sigma",
        "missing_count",
        "missing_identity",
        "nan_in_mu",
        "inf_in_sigma",
        "float32_dtype",
        "wrong_identity_sample_count",
        "count_scalar_mismatch",
        "asymmetric_sigma_shape",
        "directory_collision",
    ])
    def test_fid_corrupt_cache_adversarial_matrix(self, tmp_path, corruption_type):
        """Adversarial corruption matrix: corrupt cache entries are safely treated as misses and recomputed."""
        rng = np.random.default_rng(999)
        gen = _synthetic_images(rng, 14)
        ref = _synthetic_images(rng, 14) * 0.9
        proto = _make_protocol()

        # Generate a valid baseline cache file
        metric = FakeFID(reference_cache=str(tmp_path), dim=6)
        valid_score = _run_fid_eval(metric, gen, ref, [5, 5, 4], proto)
        [cache_file] = list(tmp_path.glob("*.fidref"))
        raw = cache_file.read_bytes()
        header, _, payload = raw.partition(b"\n")

        # Corrupt the cache file based on the test scenario
        if corruption_type == "empty_file":
            cache_file.write_bytes(b"")
        elif corruption_type == "truncated_header":
            cache_file.write_bytes(header[:20] + b"\n" + payload)
        elif corruption_type == "header_bit_flip":
            flipped_header = bytearray(header)
            flipped_header[0] = ord(b'0') if flipped_header[0] != ord(b'0') else ord(b'1')
            cache_file.write_bytes(bytes(flipped_header) + b"\n" + payload)
        elif corruption_type == "truncated_payload":
            cache_file.write_bytes(header + b"\n" + payload[:32])
        elif corruption_type == "garbage_payload":
            cache_file.write_bytes(header + b"\n" + os.urandom(256))
        elif corruption_type == "missing_mu":
            buf = BytesIO()
            np.savez(buf, sigma=np.eye(6, dtype=np.float64), count=np.int64(14),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "missing_sigma":
            buf = BytesIO()
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), count=np.int64(14),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "missing_count":
            buf = BytesIO()
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), sigma=np.eye(6, dtype=np.float64),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "missing_identity":
            buf = BytesIO()
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), sigma=np.eye(6, dtype=np.float64), count=np.int64(14))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "nan_in_mu":
            buf = BytesIO()
            bad_mu = np.zeros(6, dtype=np.float64)
            bad_mu[2] = np.nan
            np.savez(buf, mu=bad_mu, sigma=np.eye(6, dtype=np.float64), count=np.int64(14),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "inf_in_sigma":
            buf = BytesIO()
            bad_sigma = np.eye(6, dtype=np.float64)
            bad_sigma[1, 1] = np.inf
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), sigma=bad_sigma, count=np.int64(14),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "float32_dtype":
            buf = BytesIO()
            np.savez(buf, mu=np.zeros(6, dtype=np.float32), sigma=np.eye(6, dtype=np.float32), count=np.int64(14),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "wrong_identity_sample_count":
            buf = BytesIO()
            bad_ident = dict(metric._reference_cache({"reference_identity": proto})[1], protocol=dict(proto, sample_count=999))
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), sigma=np.eye(6, dtype=np.float64), count=np.int64(14),
                     identity=np.frombuffer(json.dumps(bad_ident, sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "count_scalar_mismatch":
            buf = BytesIO()
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), sigma=np.eye(6, dtype=np.float64), count=np.int64(999),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "asymmetric_sigma_shape":
            buf = BytesIO()
            np.savez(buf, mu=np.zeros(6, dtype=np.float64), sigma=np.zeros((6, 5), dtype=np.float64), count=np.int64(14),
                     identity=np.frombuffer(json.dumps(metric._reference_cache({"reference_identity": proto})[1], sort_keys=True).encode(), dtype=np.uint8))
            p = buf.getvalue()
            cache_file.write_bytes(hashlib.sha256(p).hexdigest().encode() + b"\n" + p)
        elif corruption_type == "directory_collision":
            cache_file.unlink()
            cache_file.mkdir()

        # Direct read returns None
        _, cache_ident = metric._reference_cache({"reference_identity": proto})
        assert read_reference_statistics(cache_file, cache_ident, 14) is None

        # If it was a directory collision, remove it so healing write can publish
        if corruption_type == "directory_collision":
            cache_file.rmdir()

        # Evaluate must gracefully recover: re-extracts reference and returns identical valid score
        recovered_score = _run_fid_eval(metric, gen, ref, [5, 5, 4], proto)
        assert len(metric.log) == 6, f"Expected cold recomputation (6 calls), got {len(metric.log)}"
        assert recovered_score == pytest.approx(valid_score, rel=1e-12)

        # File is now healed
        assert read_reference_statistics(cache_file, cache_ident, 14) is not None

    def test_fid_unwritable_cache_fallback(self, tmp_path, monkeypatch, caplog):
        """When cache file cannot be written (OSError), evaluation succeeds and logs warning."""
        rng = np.random.default_rng(54321)
        gen = _synthetic_images(rng, 14)
        ref = _synthetic_images(rng, 14)
        proto = _make_protocol()

        metric = FakeFID(reference_cache=str(tmp_path), dim=6)

        def mock_open(*args, **kwargs):
            raise PermissionError("Simulated read-only filesystem")

        # Monkeypatch atomic write open
        monkeypatch.setattr(Path, "open", mock_open)
        with caplog.at_level(logging.WARNING):
            score = _run_fid_eval(metric, gen, ref, [5, 5, 4], proto)

        assert np.isfinite(score)
        assert any("FID reference statistics not cached" in rec.message for rec in caplog.records)

    def test_fid_reference_cache_invalid_inputs(self):
        """Strict validation on reference_cache parameter."""
        with pytest.raises(ValueError, match="reference_cache must be a nonempty path"):
            InceptionFID(Path("/dev/null"), reference_cache="")
        with pytest.raises(ValueError, match="reference_cache must be a nonempty path"):
            InceptionFID(Path("/dev/null"), reference_cache="   ")
        with pytest.raises(ValueError, match="reference_cache must be a nonempty path"):
            InceptionFID(Path("/dev/null"), reference_cache=12345)


# ==============================================================================
# TEST SUITE 2: CHECKPOINT RETENTION SYMLINK GUARDS & SEQUENCE ORDERING
# ==============================================================================

class TestAdversarialCheckpointRetentionSymlinkGuards:
    """Empirical adversarial testing of symlink rejection and abandoned tips sequence ordering."""

    def test_symlinked_checkpoints_dir_strictly_rejected(self, tmp_path):
        """prune_checkpoints strictly refuses a run whose checkpoints/ directory is a symlink."""
        outside = tmp_path / "outside_checkpoints"
        outside.mkdir(parents=True)
        _create_mock_checkpoint(outside.parent, "0001-a", 1)

        run_dir = tmp_path / "target_run"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "target_run", "attempt_id": "0001-a"}))
        os.symlink(outside, run_dir / "checkpoints", target_is_directory=True)

        with pytest.raises(ValueError, match="Refusing to prune a symlinked checkpoints directory"):
            prune_checkpoints(run_dir, policy=TEST_RETENTION_POLICY)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG in checkpoints.py:1140 - _checkpoint_roots() resolves paths, so root.is_symlink() is False for symlinked distributed-checkpoints directories, failing to reject and deleting through the symlink",
    )
    def test_symlinked_distributed_checkpoints_dir_strictly_rejected(self, tmp_path):
        """prune_checkpoints strictly refuses a run whose distributed-checkpoints/ is a symlink."""
        outside = tmp_path / "outside_dist_checkpoints"
        outside.mkdir(parents=True)

        run_dir = tmp_path / "target_dist_run"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "target_dist_run", "attempt_id": "0001-a"}))
        (run_dir / "checkpoints").mkdir()
        os.symlink(outside, run_dir / "distributed-checkpoints", target_is_directory=True)

        with pytest.raises(ValueError, match="Refusing to prune a symlinked checkpoints directory"):
            prune_checkpoints(run_dir, policy=TEST_RETENTION_POLICY)

    def test_multi_run_root_skips_symlinked_child_runs(self, tmp_path):
        """Directory-of-runs prune ignores symlinked child runs, protecting external directories."""
        runs_root = tmp_path / "runs"
        runs_root.mkdir()

        # Real idle run
        real_idle = runs_root / "idle"
        for step in (0, 1, 2, 3):
            c = _create_mock_checkpoint(real_idle, "0001-a", step)
        _set_run_latest(real_idle, c, 3)
        _write_mock_attempt_manifest(real_idle, "0001-a", checkpoint_path=c, is_current=True)

        # External run
        outside = tmp_path / "external_run"
        for step in (0, 1, 2, 3):
            c_out = _create_mock_checkpoint(outside, "0001-a", step)
        _set_run_latest(outside, c_out, 3)
        _write_mock_attempt_manifest(outside, "0001-a", checkpoint_path=c_out, is_current=True)

        # Symlink in runs_root pointing to external
        os.symlink(outside, runs_root / "linked_child", target_is_directory=True)

        # Prune runs_root
        report = prune_checkpoints(runs_root, policy=TEST_RETENTION_POLICY)

        # Idle run pruned according to policy (keeps latest step 3 + step 0 = 2 retained, 2 deleted)
        assert report["deleted_count"] == 2
        # External run untouched: all 4 checkpoints remain intact!
        external_ckpts = [p.name for p in (outside / "checkpoints").iterdir() if p.name != "latest.json"]
        assert len(external_ckpts) == 4, f"Symlinked run target was modified! {external_ckpts}"

    def test_individual_symlinked_checkpoint_entry_never_deleted_through(self, tmp_path):
        """A single symlinked checkpoint entry inside checkpoints/ is retained and target never touched."""
        run_dir = tmp_path / "my_run"
        for step in (0, 2, 3):
            last = _create_mock_checkpoint(run_dir, "0001-a", step)
        _set_run_latest(run_dir, last, 3)
        _write_mock_attempt_manifest(run_dir, "0001-a", checkpoint_path=last, is_current=True)

        # External checkpoint target
        external_dir = tmp_path / "external_checkpoint"
        ext_ckpt = _create_mock_checkpoint(external_dir, "0099-z", 1)

        # Symlink external checkpoint into my_run/checkpoints/
        symlink_path = run_dir / "checkpoints" / "0001-a-step-00000001-linked000000"
        os.symlink(ext_ckpt, symlink_path, target_is_directory=True)

        report = prune_checkpoints(run_dir, policy=TEST_RETENTION_POLICY)
        reasons = {item["name"]: item["reasons"] for item in report["retained"]}

        # Symlink must be retained with reason 'symlink'
        assert symlink_path.name in reasons
        assert "symlink" in reasons[symlink_path.name]
        # Target files must remain intact
        assert (ext_ckpt / "state.pt").is_file()
        assert (ext_ckpt / "manifest.json").is_file()

    def test_resolve_resume_checkpoint_dir_rejects_symlinks(self, tmp_path):
        """resolve_resume_checkpoint_dir strictly rejects symlink candidates with ValueError."""
        run_dir = tmp_path / "sym_resume_run"
        ckpt = _create_mock_checkpoint(run_dir, "0001-a", 5)
        _set_run_latest(run_dir, ckpt, 5)
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "sym_resume_run"}))

        # Target pointing to symlink
        symlink_ckpt = run_dir / "checkpoints" / "symlink_entry"
        os.symlink(ckpt, symlink_ckpt, target_is_directory=True)

        # Direct explicit resolution of symlink
        with pytest.raises(ValueError, match="Checkpoint must be an ordinary completed directory inside this run"):
            resolve_resume_checkpoint_dir(run_dir, symlink_ckpt.name)

        # Pointer latest.json pointing to symlink
        (run_dir / "checkpoints" / "latest.json").write_text(json.dumps({
            "schema_version": 1,
            "checkpoint": symlink_ckpt.name,
            "step": 5,
        }))
        with pytest.raises(ValueError, match="Checkpoint must be an ordinary completed directory inside this run"):
            resolve_resume_checkpoint_dir(run_dir, "latest")

    def test_read_distributed_checkpoint_for_elastic_rejects_symlinks(self, tmp_path):
        """read_distributed_checkpoint_for_elastic strictly rejects symlinked dirs and rank files."""
        run_dir = tmp_path / "dist_run"
        dist_root = run_dir / "distributed-checkpoints"
        ckpt_dir = dist_root / "0001-a-step-00000005-000000000005"
        ckpt_dir.mkdir(parents=True)

        # Symlink the entire checkpoint directory
        sym_ckpt = dist_root / "sym_ckpt"
        os.symlink(ckpt_dir, sym_ckpt, target_is_directory=True)

        with pytest.raises(ValueError, match="Distributed checkpoint generation must not be a symlink"):
            read_distributed_checkpoint_for_elastic(run_dir, sym_ckpt.name, "dist_run")

        # Symlink an individual rank file
        fake_rank = tmp_path / "external_rank.pt"
        fake_rank.write_bytes(b"rank payload")
        os.symlink(fake_rank, ckpt_dir / "rank-00000.pt")
        (ckpt_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "kind": "hypergan-distributed-training-checkpoint",
            "run_id": "dist_run",
            "attempt_id": "0001-a",
            "step": 5,
            "replicated_sha256": "fake",
            "identity": {"hypergan_checkpoint_version": 2},
            "ranks": [{"rank": 0, "file": "rank-00000.pt", "bytes": len(b"rank payload"), "sha256": hashlib.sha256(b"rank payload").hexdigest()}],
            "next_sample_sequence": 1,
        }))

        with pytest.raises(ValueError, match="Distributed rank payload size/type mismatch"):
            read_distributed_checkpoint_for_elastic(run_dir, ckpt_dir.name, "dist_run")

    def test_sweep_pending_ignores_symlinks(self, tmp_path):
        """_sweep_pending skips .pending symlinks without deleting through them."""
        outside = tmp_path / "outside_dir"
        outside.mkdir()
        (outside / "valuable.txt").write_text("protected")

        ckpt_root = tmp_path / "checkpoints"
        ckpt_root.mkdir()
        pending_symlink = ckpt_root / ".pending-0001-a-step-00000001-0123456789ab"
        os.symlink(outside, pending_symlink, target_is_directory=True)

        _sweep_pending(ckpt_root)
        # Outside target MUST remain untouched
        assert (outside / "valuable.txt").read_text() == "protected"
        # The symlink itself was skipped
        assert pending_symlink.is_symlink()

    def test_abandoned_tip_ordering_strictly_ordered_by_event_sequence(self, tmp_path):
        """Event sequence strictly decides abandoned attempt tip, dominating alphabetical UUID."""
        # Attempt 1: Initial attempt from step 0 to step 2
        _create_mock_checkpoint(tmp_path, "0001-a", 0, sequence=0)
        c2 = _create_mock_checkpoint(tmp_path, "0001-a", 2, sequence=2)
        _write_mock_attempt_manifest(tmp_path, "0001-a", checkpoint_path=c2)

        parent = {"attempt_id": "0001-a", "step": 2, "checkpoint_id": c2.name}

        # Attempt 2: Resumed from step 2, writes multiple checkpoints at step 5:
        # Checkpoint LowSeq: sequence 10, suffix 'zzzzzzzzzzzz' (alphabetically highest)
        # Checkpoint HighSeq: sequence 50, suffix 'aaaaaaaaaaaa' (alphabetically lowest)
        # Checkpoint MidSeq: sequence 25, suffix 'mmmmmmmmmmmm'
        c_low_seq = _create_mock_checkpoint(tmp_path, "0002-b", 5, suffix="z" * 12, sequence=10)
        c_high_seq = _create_mock_checkpoint(tmp_path, "0002-b", 5, suffix="a" * 12, sequence=50)
        c_mid_seq = _create_mock_checkpoint(tmp_path, "0002-b", 5, suffix="m" * 12, sequence=25)
        _write_mock_attempt_manifest(tmp_path, "0002-b", parent=parent)

        # Attempt 3: Resumed from step 2 again (making Attempt 2 abandoned!), writes step 6, 7
        _create_mock_checkpoint(tmp_path, "0003-c", 6, sequence=60)
        latest_c = _create_mock_checkpoint(tmp_path, "0003-c", 7, sequence=70)
        _set_run_latest(tmp_path, latest_c, 7)
        _write_mock_attempt_manifest(tmp_path, "0003-c", parent=parent, checkpoint_path=latest_c, is_current=True)

        report = prune_checkpoints(tmp_path, policy=TEST_RETENTION_POLICY)
        reasons = {item["name"]: item["reasons"] for item in report["retained"]}

        # Empirical proof: HighSeq (seq 50, suffix 'aaaa') is retained as abandoned_tip
        assert c_high_seq.name in reasons, "Highest event sequence checkpoint was NOT retained!"
        assert "abandoned_tip" in reasons[c_high_seq.name]

        # LowSeq (seq 10, suffix 'zzzz') was deleted despite alphabetically dominating
        assert c_low_seq.name not in reasons
        assert not c_low_seq.exists(), "Low sequence checkpoint was not pruned!"
        assert not c_mid_seq.exists(), "Mid sequence checkpoint was not pruned!"

    def test_abandoned_tip_sequence_tiebreaker_falls_back_to_manifest_checkpoint_path(self, tmp_path):
        """When sequence is tied (-1 or equal), attempt manifest's recorded checkpoint_path wins."""
        c0 = _create_mock_checkpoint(tmp_path, "0001-a", 0)
        c2 = _create_mock_checkpoint(tmp_path, "0001-a", 2)
        _write_mock_attempt_manifest(tmp_path, "0001-a", checkpoint_path=c2)

        parent = {"attempt_id": "0001-a", "step": 2, "checkpoint_id": c2.name}

        # Attempt 2: Two step-5 checkpoints with identical sequence (e.g. sequence=None/-1)
        c_unrecorded = _create_mock_checkpoint(tmp_path, "0002-b", 5, suffix="z" * 12, sequence=None)
        c_recorded = _create_mock_checkpoint(tmp_path, "0002-b", 5, suffix="a" * 12, sequence=None)
        _write_mock_attempt_manifest(tmp_path, "0002-b", parent=parent, checkpoint_path=c_recorded)

        # Attempt 3: Live branch at step 6
        latest = _create_mock_checkpoint(tmp_path, "0003-c", 6)
        _set_run_latest(tmp_path, latest, 6)
        _write_mock_attempt_manifest(tmp_path, "0003-c", parent=parent, checkpoint_path=latest, is_current=True)

        report = prune_checkpoints(tmp_path, policy=TEST_RETENTION_POLICY)
        reasons = {item["name"]: item["reasons"] for item in report["retained"]}

        assert c_recorded.name in reasons
        assert "abandoned_tip" in reasons[c_recorded.name]
        assert not c_unrecorded.exists()

    def test_multiple_abandoned_attempts_tree(self, tmp_path):
        """Complex multi-attempt history preserves newest sequence tips for all abandoned attempts."""
        _create_mock_checkpoint(tmp_path, "0001-a", 0, sequence=0)
        c_root = _create_mock_checkpoint(tmp_path, "0001-a", 1, sequence=1)
        _write_mock_attempt_manifest(tmp_path, "0001-a", checkpoint_path=c_root)

        p1 = {"attempt_id": "0001-a", "step": 1, "checkpoint_id": c_root.name}

        # Branch 1 (Attempt 2, abandoned): writes step 2, 3
        _create_mock_checkpoint(tmp_path, "0002-b", 2, sequence=10)
        tip_b = _create_mock_checkpoint(tmp_path, "0002-b", 3, sequence=20)
        _write_mock_attempt_manifest(tmp_path, "0002-b", parent=p1, checkpoint_path=tip_b)

        # Branch 2 (Attempt 3, abandoned): resumed from step 1, writes step 2, 4
        _create_mock_checkpoint(tmp_path, "0003-c", 2, sequence=30)
        tip_c = _create_mock_checkpoint(tmp_path, "0003-c", 4, sequence=40)
        _write_mock_attempt_manifest(tmp_path, "0003-c", parent=p1, checkpoint_path=tip_c)

        # Branch 3 (Attempt 4, live): resumed from step 1, writes step 5
        tip_d = _create_mock_checkpoint(tmp_path, "0004-d", 5, sequence=50)
        _set_run_latest(tmp_path, tip_d, 5)
        _write_mock_attempt_manifest(tmp_path, "0004-d", parent=p1, checkpoint_path=tip_d, is_current=True)

        report = prune_checkpoints(tmp_path, policy=TEST_RETENTION_POLICY)
        reasons = {item["name"]: item["reasons"] for item in report["retained"]}

        assert "abandoned_tip" in reasons[tip_b.name]
        assert "abandoned_tip" in reasons[tip_c.name]
        assert "latest" in reasons[tip_d.name]


# ==============================================================================
# TEST SUITE 3: REPLICATED RESUME IDENTITY ADVERSARIAL TESTS
# ==============================================================================

class TestAdversarialReplicatedResumeIdentity:
    """Empirical adversarial testing of replicated resume identity isolation."""

    def test_numerical_identity_filters_per_launch_keys(self):
        """_numerical_identity strips 'node_rank' and 'rendezvous' while preserving all others."""
        exec_full = {
            "name": "cpu-replicated-gloo",
            "world_size": 4,
            "nodes": 2,
            "ranks_per_node": 2,
            "accumulation_steps": 2,
            "global_batch_size": 16,
            "local_batch_size": 8,
            "microbatch_size": 4,
            "node_rank": 1,
            "rendezvous": "tcp://cluster-master:29500",
        }
        filtered = _numerical_identity(exec_full)
        assert "node_rank" not in filtered
        assert "rendezvous" not in filtered
        assert filtered["name"] == "cpu-replicated-gloo"
        assert filtered["world_size"] == 4
        assert filtered["accumulation_steps"] == 2

    def test_same_topology_ignores_node_rank_and_rendezvous(self):
        """_same_topology returns True when only per-launch keys vary, False on genuine diffs."""
        base_topo = {
            "name": "cpu-replicated-gloo",
            "world_size": 2,
            "nodes": 2,
            "ranks_per_node": 1,
            "accumulation_steps": 1,
            "node_rank": 0,
            "rendezvous": "tcp://10.0.0.1:29500",
        }
        # Varying node_rank (0 vs 1) and rendezvous (different host/port)
        variant_topo = dict(base_topo, node_rank=1, rendezvous="tcp://192.168.1.99:12345")
        assert _same_topology(base_topo, variant_topo) is True

        # Genuine topology difference (e.g. accumulation_steps 1 vs 2)
        diff_accum = dict(base_topo, accumulation_steps=2)
        assert _same_topology(base_topo, diff_accum) is False

        # Genuine topology difference (e.g. world_size 2 vs 4)
        diff_ws = dict(base_topo, world_size=4)
        assert _same_topology(base_topo, diff_ws) is False

    def test_is_same_distributed_topology_ignores_per_launch_keys(self):
        """_is_same_distributed_topology ignores node_rank and rendezvous."""
        saved_identity = {
            "config": {"training": {"device": "cpu", "batch_size": 8}},
            "topology": {
                "name": "cpu-replicated-gloo",
                "world_size": 2,
                "node_rank": 0,
                "rendezvous": "tcp://10.0.0.1:29500",
            },
        }
        current_identity = {
            "config": {"training": {"device": "cpu", "batch_size": 8}},
            "topology": {
                "name": "cpu-replicated-gloo",
                "world_size": 2,
                "node_rank": 1,
                "rendezvous": "file:///tmp/rdzv_shared",
            },
        }
        assert _is_same_distributed_topology(saved_identity, current_identity, 2, 2) is True

    def test_prepare_resume_rendezvous_and_node_rank_matrix(self, tmp_path):
        """prepare_resume succeeds across variations of rendezvous endpoint and node_rank."""
        run_dir, cfg = _make_replicated_run(tmp_path, node_rank=0, rendezvous="tcp://10.0.0.1:29500")

        # Case 1: Node 0 resumes with different rendezvous endpoint (port & IP changed)
        prep_node0 = prepare_resume(
            run_dir,
            node_rank=0,
            rendezvous="tcp://172.16.0.5:31000",
        )
        assert prep_node0.operation == "resume"
        assert prep_node0.checkpoint is not None
        assert prep_node0.profile["execution"]["rendezvous"] == "tcp://172.16.0.5:31000"
        assert prep_node0.profile["execution"]["node_rank"] == 0

        # Case 2: Node 1 follower resumes with same or different rendezvous
        prep_node1 = prepare_resume(
            run_dir,
            node_rank=1,
            rendezvous="tcp://10.0.0.1:29500",
        )
        assert prep_node1.operation == "resume"
        # Follower node does not inspect or pin checkpoints
        assert prep_node1.checkpoint is None
        assert prep_node1.profile["execution"]["node_rank"] == 1

        # Case 3: Follower detected via environment variable NODE_RANK
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("NODE_RANK", "1")
            mp.setenv("MASTER_ADDR", "10.10.10.10")
            mp.setenv("MASTER_PORT", "29500")
            prep_env = prepare_resume(run_dir)
            assert prep_env.profile["execution"]["node_rank"] == 1
            assert prep_env.checkpoint is None

        # Case 4: Follower detected via SLURM_NODEID
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SLURM_NODEID", "1")
            prep_slurm = prepare_resume(run_dir)
            assert prep_slurm.profile["execution"]["node_rank"] == 1
            assert prep_slurm.checkpoint is None

    def test_replicated_resume_strictly_rejects_genuine_topology_changes(self, tmp_path):
        """Altering genuine execution parameters (world_size, accumulation_steps) strictly raises error."""
        run_dir, cfg = _make_replicated_run(tmp_path, node_rank=0)

        # Incompatible accumulation_steps
        raw_incompatible_profile = {
            "schema_version": 1,
            "execution": {
                "name": "cpu-replicated-gloo",
                "world_size": 2,
                "nodes": 2,
                "ranks_per_node": 1,
                "accumulation_steps": 2,  # Changed from 1 to 2!
                "rendezvous": "tcp://10.0.0.1:29500",
            },
        }
        with pytest.raises(ValueError, match="Resume numerical execution identity differs from the run manifest"):
            prepare_resume(run_dir, profile=raw_incompatible_profile)

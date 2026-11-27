"""Regression tests for forensic audit core training, metrics, data, cache, and benchmark fixes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import subprocess
import tarfile
import threading
import time
import zipfile

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.distance import cdist
import torch
from torch import nn

import hypergan as hg
from hypergan.artifacts import bundle_state, save_bundle
from hypergan.cache import ensure_cached_url
from hypergan.colorization_metrics import ChromaDistributionDistance, _rgb
from hypergan.config import config_values, load_config, resolve_config
from hypergan.data import EvaluationDataExhausted, ImageFolder
from hypergan import dataset_download
from hypergan.image_data import CIFAR10Data
from hypergan.image_metrics import (
    _normalize_eval_data_spec,
    compute_prdc,
    evaluate_bundle,
    knn_radii,
)
from hypergan.item_data import ItemData
from hypergan.metric_evaluation_worker import evaluate_snapshot
from hypergan.metric_plugins import _factory, check_evaluation_data
from hypergan.metrics import metric_catalog, select_metrics
from hypergan.objective_program import _select_pullback_kind, batch_diversity_ratio
from hypergan.pretrained_providers import _VERIFIED, _checkout_stamp, _resnet18_shape_fn, _verify_checkout
from hypergan.recipes import ComponentGraph
from hypergan.run_state import atomic_json
from hypergan.summary import write_summary_json
from hypergan.training import ReferenceTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
PREP_SCRIPT = REPO_ROOT / "scripts" / "prepare_benchmark_dataset.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ScalePost(nn.Module):
    def __init__(self, gain: float = 0.5):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


class _FrozenPost(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.75 + 0.1


class _PointMass2D:
    resume_stateless = True

    def __init__(self, value: float = 0.5):
        self.value = float(value)

    def __call__(self, batch_size: int, *, generator):
        del generator
        return {"real": torch.full((batch_size, 2), self.value)}


class _TinyItems:
    def __init__(self, count: int = 8):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index: int):
        return torch.tensor([float(index), float(index) + 0.5])


def _load_prep_module():
    spec = importlib.util.spec_from_file_location("prepare_benchmark_dataset_audit", PREP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["prepare_benchmark_dataset_audit"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Core training & ParticleGAN parity
# ---------------------------------------------------------------------------

def test_generator_probe_post_component_referencing_generated():
    """output_subgraph and _GeneratorProbeModule must recurse into generator for 'generated' bindings."""
    cfg = resolve_config({
        "defaults": "particlegan",
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {"source": "linear(8)\nleaky_relu(0.2)\nlinear(2)", "input_shape": ["B", 4], "output_shape": ["B", 2]},
                "inputs": {"x": "latent"},
            },
            "post": {
                "factory": f"{__name__}:_FrozenPost",
                "inputs": {"x": "generated"},
                "trainable": False,
            },
            "discriminator": {
                "factory": "hndl",
                "args": {"source": "linear(8)\nleaky_relu(0.2)\nlinear(1)", "input_shape": ["B", 2], "output_shape": ["B", 1]},
                "inputs": {"x": "candidate"},
            },
        },
        "adversarial": {"fake": "components.post"},
        "sampling": {"generated": "components.post"},
        "training": {"steps": 2, "batch_size": 8, "seed": 3, "device": "cpu", "precision": "float32"},
    })
    graph = ComponentGraph(cfg["components"])
    models, trainable_names, batch_keys, prior_keys = graph.output_subgraph("components.post")
    assert "generator" in models
    assert "post" in models
    assert "generator" in trainable_names
    assert "post" not in trainable_names
    assert batch_keys == ()
    assert prior_keys == ()

    probe = graph.build_generator_probe("components.post")
    z = torch.randn(4, 4)
    out = probe(z)
    assert out.shape == (4, 2)
    expected = graph.generate(z, {})["generated"] * 0.75 + 0.1
    torch.testing.assert_close(out, expected)

    # Full ReferenceTrainer step with sampling.generated = "components.post" must consume probe without crashing.
    trainer = ReferenceTrainer(cfg)
    assert any(id(p) in {id(x) for x in trainer.graph.models["generator"].parameters()} for p in trainer._generator_probe_params)
    row, _ = trainer.update()
    assert torch.isfinite(torch.tensor(row["d_loss"]))
    assert torch.isfinite(torch.tensor(row["g_loss"]))
    assert trainer.opt_g.probe_consumptions >= 1


@pytest.mark.parametrize("defaults_spec,expected_kind", [
    ("particlegan", "axiom"),
    ("particlegan:unified_v15", "unified_v15"),
    ("particlegan:unified_v11", "unified_v11"),
])
def test_pullback_step_runs_on_unconditional_2d_mog(defaults_spec, expected_kind):
    """run_native_program invokes ParticleGAN *_pullback_step on unconditional 2D MoG matching GANTrainer."""
    cfg = load_config({
        "format": "hypergan-model/1",
        "name": f"test/pullback-{expected_kind}",
        "defaults": defaults_spec,
        "data": {"builtin": "gaussian_grid"},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "num_particles": 32, "z_dim": 4, "initialization_seed": 2},
        "networks": {
            "generator": {
                "role": "generator",
                "inputs": {"x": "latent"},
                "input_shape": ["B", 4],
                "output_shape": ["B", 2],
                "source": "linear(16)\nleaky_relu(0.2)\nlinear(2)",
            },
            "discriminator": {
                "role": "critic",
                "inputs": {"x": "candidate"},
                "input_shape": ["B", 2],
                "output_shape": ["B", 1],
                "source": "linear(16)\nleaky_relu(0.2)\nlinear(1)",
            },
        },
        "losses": [
            {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
            {"type": "prior"},
        ],
        "train": {"steps": 2, "batch_size": 16, "seed": 7, "device": "cpu", "precision": "float32"},
    })
    trainer = ReferenceTrainer(cfg)
    sample_real = torch.randn(16, 2)
    z, ids = trainer.prior.sample(16, generator=torch.Generator().manual_seed(1))
    assert _select_pullback_kind(trainer, trainer.program, sample_real, z, ids) == expected_kind
    assert trainer.spread.quantizer_initialized is False

    row, _ = trainer.update()
    assert torch.isfinite(torch.tensor(row["d_loss"]))
    assert torch.isfinite(torch.tensor(row["g_loss"]))
    # Pullback step initializes the quantizer and row Adam buffers in the prior regularizer.
    assert trainer.spread.quantizer_initialized is True
    assert getattr(trainer.opt_d.record, "_g_probe_cache", None) is None


def test_axiom_pullback_runs_on_point_mass_and_skips_when_objectives_present():
    """Axiom pullback triggers on point-mass 2D data and is skipped when auxiliary objectives exist."""
    base_cfg = resolve_config({
        "defaults": "particlegan",
        "data": {"factory": f"{__name__}:_PointMass2D", "args": {"value": 0.5}},
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}, "initialization_seed": 2},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {"source": "linear(16)\nleaky_relu(0.2)\nlinear(2)", "input_shape": ["B", 4], "output_shape": ["B", 2]},
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": "hndl",
                "args": {"source": "linear(16)\nleaky_relu(0.2)\nlinear(1)", "input_shape": ["B", 2], "output_shape": ["B", 1]},
                "inputs": {"x": "candidate"},
            },
        },
        "training": {"steps": 2, "batch_size": 8, "seed": 5, "device": "cpu", "precision": "float32"},
    })
    trainer = ReferenceTrainer(base_cfg)
    real = torch.full((8, 2), 0.5)
    z, ids = trainer.prior.sample(8, generator=torch.Generator().manual_seed(2))
    assert _select_pullback_kind(trainer, trainer.program, real, z, ids) == "axiom"
    row, _ = trainer.update()
    assert torch.isfinite(torch.tensor(row["g_loss"]))

    # Adding an auxiliary objective must keep the standard objective program path active.
    obj_cfg = resolve_config({
        **config_values(base_cfg),
        "objectives": [{"factory": "mse", "inputs": {"input": "generated", "target": "batch.real"}, "weight": 0.5}],
    })
    obj_trainer = ReferenceTrainer(obj_cfg)
    assert _select_pullback_kind(obj_trainer, obj_trainer.program, real, z, ids) is None


# ---------------------------------------------------------------------------
# 2. Metrics & collapse detection
# ---------------------------------------------------------------------------

def test_batch_diversity_ratio_detects_mode_collapse_and_triggers_alert(tmp_path):
    """Collapsed generator repeating one non-flat image must report 0.0 and trigger summary alert."""
    rng = torch.Generator().manual_seed(11)
    real = torch.randn(16, 3, 8, 8, generator=rng)
    single_nonflat_image = torch.randn(1, 3, 8, 8, generator=rng)
    collapsed_fake = single_nonflat_image.expand(16, -1, -1, -1).clone()

    # Old global pixel std ratio was ~1.0 here; true inter-sample ratio must be 0.0.
    ratio = float(batch_diversity_ratio(collapsed_fake, real).item())
    assert ratio == 0.0

    # Diverse fake matching real distribution spread reports ~1.0.
    healthy_fake = torch.randn(16, 3, 8, 8, generator=rng)
    healthy_ratio = float(batch_diversity_ratio(healthy_fake, real).item())
    assert 0.5 < healthy_ratio < 1.5

    # Edge cases: B < 2 or zero real spread return 1.0 safely.
    assert float(batch_diversity_ratio(collapsed_fake[:1], real[:1]).item()) == 1.0
    constant_real = torch.ones(8, 3, 4, 4)
    assert float(batch_diversity_ratio(collapsed_fake[:8, :, :4, :4], constant_real).item()) == 1.0

    # Negative diversity/ratio rejected in select_metrics; 0.0 fires health alert in summary.json.
    cfg = resolve_config({"metrics": {"every_steps": 1}})
    catalog = metric_catalog(cfg)
    with pytest.raises(ValueError, match="Nonfinite or invalid metric value: diversity/ratio"):
        select_metrics(cfg, catalog, {"d_loss": 0.5, "g_loss": 0.5, "diversity/ratio": -0.1}, step=1, step_seconds=0.01)

    run_dir = tmp_path / "collapsed_run"
    run_dir.mkdir()
    atomic_json(run_dir / "manifest.json", {"run_id": "run-col", "status": "running", "steps": 10, "total_steps": 100})
    write_summary_json(run_dir, metrics={"diversity/ratio": ratio})
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert any("Mode collapse risk" in alert for alert in summary["health_alerts"])


def test_knn_radii_and_prdc_no_float32_catastrophic_cancellation_and_disables_tf32():
    """Tight clusters at large norm (~48) must not cancel to 0.0; TF32 must be disabled and restored."""
    rng = np.random.default_rng(42)
    # Two distant high-norm (~48) clusters so centering does NOT collapse ||x - mean||.
    u = rng.normal(size=(1, 1024))
    u = (u / np.linalg.norm(u)) * 48.0
    cluster_a = (u + rng.normal(scale=0.003 / np.sqrt(1024), size=(24, 1024))).astype(np.float32)
    cluster_b = (-u + rng.normal(scale=0.003 / np.sqrt(1024), size=(24, 1024))).astype(np.float32)
    real = np.vstack([cluster_a, cluster_b])
    fake = np.vstack([
        (u + rng.normal(scale=0.003 / np.sqrt(1024), size=(24, 1024))).astype(np.float32),
        (-u + rng.normal(scale=0.003 / np.sqrt(1024), size=(24, 1024))).astype(np.float32),
    ])

    d_rr_oracle = cdist(real.astype(np.float64), real.astype(np.float64), metric="euclidean")
    np.fill_diagonal(d_rr_oracle, np.inf)
    r3_oracle = np.partition(d_rr_oracle, 2, axis=1)[:, 2]

    orig_mm = torch.backends.cuda.matmul.allow_tf32
    orig_cu = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        got_radii = knn_radii(real, (3,), chunk_size=13)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert np.all(got_radii[3] > 0.0)
        np.testing.assert_allclose(got_radii[3], r3_oracle, rtol=1e-4, atol=1e-6)

        prdc = compute_prdc(real, fake, pr_k=3, dc_k=5, chunk_size=13)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

        d_ff_oracle = cdist(fake.astype(np.float64), fake.astype(np.float64), metric="euclidean")
        np.fill_diagonal(d_ff_oracle, np.inf)
        f3_oracle = np.partition(d_ff_oracle, 2, axis=1)[:, 2]
        r5_oracle = np.partition(d_rr_oracle, 4, axis=1)[:, 4]
        d_rf_oracle = cdist(real.astype(np.float64), fake.astype(np.float64), metric="euclidean")
        assert prdc["precision"] == pytest.approx(float((d_rf_oracle < r3_oracle[:, None]).any(axis=0).mean()), abs=1e-12)
        assert prdc["recall"] == pytest.approx(float((d_rf_oracle < f3_oracle[None, :]).any(axis=1).mean()), abs=1e-12)
        assert prdc["density"] == pytest.approx(float((d_rf_oracle < r5_oracle[:, None]).sum(axis=0).mean() / 5.0), abs=1e-12)
        assert prdc["coverage"] == pytest.approx(float((d_rf_oracle.min(axis=1) < r5_oracle).mean()), abs=1e-12)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_mm
        torch.backends.cudnn.allow_tf32 = orig_cu


def test_normalize_eval_data_spec_recognizes_all_dataset_factories():
    """_normalize_eval_data_spec must handle hypergan.datasets:CIFAR10Data, ColorizationData, ItemData."""
    for factory in ("hypergan.image_data:CIFAR10Data", "hypergan.datasets:CIFAR10Data"):
        spec = _normalize_eval_data_spec(
            "train",
            {"data": {"factory": factory, "args": {"root": "/tmp/c", "sampling": "random", "horizontal_flip": True}}},
        )
        assert spec["args"]["sampling"] == "sequential"
        assert spec["args"]["horizontal_flip"] is False

    for factory in (
        "image_folder",
        "hypergan.data:ImageFolder",
        "hypergan.datasets:ImageFolder",
        "hypergan.item_data:ItemData",
        "hypergan.datasets:ItemData",
    ):
        spec = _normalize_eval_data_spec(
            "train",
            {"data": {"factory": factory, "args": {"shuffle": True}}},
        )
        assert spec["args"]["shuffle"] is False

    color_spec = _normalize_eval_data_spec(
        "train",
        {"data": {"factory": "hypergan.colorization_data:ColorizationData", "args": {"shuffle": True, "bad_image_policy": "skip"}}},
    )
    assert color_spec["args"]["shuffle"] is False
    assert color_spec["args"]["bad_image_policy"] == "error"


class _CaptureRefIdentityMetric:
    captured: list = []

    def describe(self):
        return {"kind": "scalar", "label": "capture", "unit": "value", "direction": "none"}

    def evaluate(self, *, batches, context):
        self.captured.append(context.get("reference_identity"))
        for _ in batches:
            pass
        return 0.0


def test_reference_identity_matches_between_worker_and_evaluate_bundle_and_ignores_training_tf32(tmp_path, monkeypatch):
    """Worker and evaluate_bundle must match data_sources (including factory module) and normalize TF32=False."""
    images = np.arange(8 * 3072, dtype=np.int64).astype(np.uint8).reshape(8, 3072)
    labels = list(range(8))
    batch_file = tmp_path / "test_batch"
    payload = pickle.dumps({"data": images, "labels": labels}, protocol=2)
    batch_file.write_bytes(payload)
    monkeypatch.setitem(dataset_download.CIFAR10_SHA256, "test_batch", hashlib.sha256(payload).hexdigest())

    cfg = resolve_config({
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "backend": {"matmul_allow_tf32": True, "cudnn_allow_tf32": True},
        },
        "prior": {"kind": "particles", "args": {"num_particles": 8, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {"source": "linear(3072)\nreshape(3,32,32)\ntanh()", "input_shape": ["B", 4], "output_shape": ["B", 3, 32, 32]},
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": "hndl",
                "args": {"source": "flatten()\nlinear(1)", "input_shape": ["B", 3, 32, 32], "output_shape": ["B", 1]},
                "inputs": {"x": "candidate"},
            },
        },
        "data": {
            "factory": "hypergan.datasets:CIFAR10Data",
            "args": {"root": str(tmp_path), "split": "test", "sampling": "sequential", "horizontal_flip": False},
        },
    })
    trainer = ReferenceTrainer(cfg)
    trainer.artifact_identity = {"run_id": "run-ref", "attempt_id": "att-1"}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_bundle(run_dir, trainer, {"real": torch.zeros(2, 3, 32, 32)})
    snap = run_dir / "model.pt"

    spec = {
        "factory": f"{__name__}:_CaptureRefIdentityMetric",
        "args": {},
        "mode": "snapshot",
        "trigger": "manual",
        "inputs": {"generated": "evaluation.generated", "reference": "evaluation.reference"},
        "evaluation": {
            "device": "cpu",
            "sample_count": 4,
            "batch_size": 4,
            "seed": 19,
            "data": {
                "factory": "hypergan.datasets:CIFAR10Data",
                "args": {"root": str(tmp_path), "split": "test", "sampling": "sequential", "horizontal_flip": False},
            },
        },
    }
    expected = _factory(spec["factory"], spec["args"])[1]
    prev_mm, prev_cu = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    prev_threads = torch.get_num_threads()
    _CaptureRefIdentityMetric.captured.clear()
    try:
        evaluate_snapshot(
            spec,
            expected,
            str(snap),
            hashlib.sha256(snap.read_bytes()).hexdigest(),
            trainer.artifact_identity,
        )
        evaluate_bundle(
            snap,
            data="train",
            metrics={"cap": _CaptureRefIdentityMetric()},
            sample_count=4,
            batch_size=4,
            seed=19,
            device="cpu",
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cu
        torch.set_num_threads(prev_threads)

    assert len(_CaptureRefIdentityMetric.captured) == 2
    worker_ref, bundle_ref = _CaptureRefIdentityMetric.captured
    assert worker_ref == bundle_ref
    assert "hypergan.datasets" in bundle_ref["data_sources"]
    assert "hypergan.image_data" in bundle_ref["data_sources"]
    assert worker_ref["backend"]["matmul_allow_tf32"] is False
    assert worker_ref["backend"]["cudnn_allow_tf32"] is False


def test_colorization_rgb_requires_floating_point_tensor():
    """colorization_metrics._rgb must reject integer, bool, and complex tensors."""
    for bad_dtype in (torch.int64, torch.uint8, torch.bool, torch.complex64):
        bad = torch.zeros((2, 3, 4, 4), dtype=bad_dtype)
        with pytest.raises(ValueError, match="floating-point"):
            _rgb(bad)
        with pytest.raises(ValueError, match="floating-point"):
            ChromaDistributionDistance().evaluate(batches=[{"generated": bad, "reference": bad}], context={})
    good = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    assert _rgb(good).dtype == torch.float64


# ---------------------------------------------------------------------------
# 3. Data pipelines & cache security
# ---------------------------------------------------------------------------

class _FlakyHTTPServer:
    """HTTP server supporting multiple routes, 206 Range, 416, and intentional mid-stream truncation."""

    def __init__(self, routes: dict[str, bytes], *, fail_first_at: dict[str, int] | None = None, delay: float = 0.0):
        self.routes = routes
        self.fail_first_at = dict(fail_first_at or {})
        self.delay = delay
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self.active_handlers = 0
        self.peak_concurrency = 0
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with parent._lock:
                    parent.active_handlers += 1
                    parent.peak_concurrency = max(parent.peak_concurrency, parent.active_handlers)
                    range_hdr = self.headers.get("Range")
                    parent.requests.append({"path": self.path, "range": range_hdr})
                    cut = parent.fail_first_at.pop(self.path, None)
                try:
                    if parent.delay > 0:
                        time.sleep(parent.delay)
                    if self.path not in parent.routes:
                        self.send_response(404)
                        self.end_headers()
                        return
                    data = parent.routes[self.path]
                    if range_hdr and range_hdr.startswith("bytes="):
                        start = int(range_hdr[len("bytes="):].split("-", 1)[0])
                        if start >= len(data):
                            self.send_response(416)
                            self.end_headers()
                            return
                        body = data[start:]
                        self.send_response(206)
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                        self.end_headers()
                        self.wfile.write(body[:cut] if cut is not None else body)
                        return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data[:cut] if cut is not None else data)
                finally:
                    with parent._lock:
                        parent.active_handlers -= 1

            def log_message(self, format, *args):
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.allow_reuse_address = True
        # Use ThreadingHTTPServer-like behavior via threading.Thread per connection for concurrency test
        from http.server import ThreadingHTTPServer
        self._httpd.server_close()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_port
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_cache_hardening_traversal_symlinks_collision_resume_and_parallel_locks(tmp_path):
    """cache.py rejects traversal & symlinks, avoids URL collisions, preserves .part on transient error, uses per-path locks."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # 1. Path traversal & empty filename rejected.
    for bad_name in ("../escape.pth", "/etc/passwd", "a/b.pth", "a\\b.pth", "..", ".", "", "   "):
        with pytest.raises(ValueError, match="Invalid cache filename"):
            ensure_cached_url("https://example.com/model.pth", filename=bad_name, cache_dir=cache_dir)

    # 2. Symlink on dest_path / part_path rejected.
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (cache_dir / "sym.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe cache destination path"):
        ensure_cached_url("https://example.com/sym.bin", expected_sha256="0" * 64, cache_dir=cache_dir)
    (cache_dir / "sym.bin").unlink()

    (cache_dir / "sym2.bin.part").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe cache partial download path"):
        ensure_cached_url("https://example.com/sym2.bin", expected_sha256="0" * 64, cache_dir=cache_dir)

    payload_a = b"ALPHA_PAYLOAD_" * 500
    payload_b = b"BETA_PAYLOAD__" * 500
    sha_a = hashlib.sha256(payload_a).hexdigest()

    # 3. Distinct URLs sharing basename when expected_sha256 is None must not collide.
    with _FlakyHTTPServer(
        {"/v1/model.pth": payload_a, "/v2/model.pth": payload_b},
        fail_first_at={"/v1/model.pth": len(payload_a) // 2},
    ) as server:
        # First call truncates mid-stream -> OSError, .part MUST be preserved for Range resume!
        with pytest.raises(OSError, match="Truncated download"):
            ensure_cached_url(server.url("/v1/model.pth"), expected_size=len(payload_a), cache_dir=cache_dir)
        parts = list(cache_dir.glob("*-model.pth.part"))
        assert len(parts) == 1
        assert parts[0].stat().st_size == len(payload_a) // 2

        # Retry resumes from preserved .part using Range header!
        path_a = ensure_cached_url(server.url("/v1/model.pth"), expected_size=len(payload_a), cache_dir=cache_dir)
        assert path_a.read_bytes() == payload_a
        assert any(r["range"] == f"bytes={len(payload_a) // 2}-" for r in server.requests)

        # Second distinct URL with same basename does NOT collide with path_a!
        path_b = ensure_cached_url(server.url("/v2/model.pth"), cache_dir=cache_dir)
        assert path_b != path_a
        assert path_b.read_bytes() == payload_b
        assert path_a.read_bytes() == payload_a

        # 4. existing_bytes == expected_size promotes valid .part without HTTP request.
        pre_part = cache_dir / "full.bin.part"
        pre_part.write_bytes(payload_a)
        req_count_before = len(server.requests)
        promoted = ensure_cached_url(
            server.url("/v1/model.pth"),
            expected_sha256=sha_a,
            expected_size=len(payload_a),
            filename="full.bin",
            cache_dir=cache_dir,
        )
        assert promoted.read_bytes() == payload_a
        assert not pre_part.exists()
        assert len(server.requests) == req_count_before

    # 5. Per-path thread locking allows concurrent downloads to distinct files in parallel.
    with _FlakyHTTPServer({"/p1.bin": payload_a, "/p2.bin": payload_b}, delay=0.15) as server:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(ensure_cached_url, server.url("/p1.bin"), expected_sha256=sha_a, cache_dir=cache_dir)
            f2 = pool.submit(ensure_cached_url, server.url("/p2.bin"), expected_sha256=hashlib.sha256(payload_b).hexdigest(), cache_dir=cache_dir)
            assert f1.result().read_bytes() == payload_a
            assert f2.result().read_bytes() == payload_b
        assert server.peak_concurrency == 2


def test_prepare_benchmark_dataset_security_checks(tmp_path, monkeypatch):
    """prepare_benchmark_dataset enforces SHA-256/size/count on --archive, handles 416/full .part, bombs, and PNG tRNS/APNG."""
    prep = _load_prep_module()

    # Build a tiny valid zip containing an RGB PNG with tRNS transparency chunk.
    zip_path = tmp_path / "mini.zip"
    trns_buf = BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(trns_buf, format="PNG", transparency=(10, 20, 30))
    trns_bytes = trns_buf.getvalue()
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("00000/img00000000.png", trns_bytes)
    zip_size, zip_sha = prep.sha256_file(zip_path)

    # 1. Passing --archive with mismatched SHA-256/size MUST fail (no bypass!).
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        prep.prepare_dataset("ffhq64", output_dir=tmp_path / "bad_out", archive_path=zip_path, max_images=1)

    # Patch spec for valid archive -> tRNS RGB PNG must NOT fast-copy (resized_count == 1) and must composite cleanly.
    spec = replace(
        prep.BENCHMARK_DATASETS["ffhq64"],
        expected_bytes=zip_size,
        archive_sha256=zip_sha,
        expected_count=1,
    )
    monkeypatch.setitem(prep.BENCHMARK_DATASETS, "ffhq64", spec)
    out_dir = tmp_path / "trns_out"
    manifest = prep.prepare_dataset("ffhq64", output_dir=out_dir, archive_path=zip_path, workers=1)
    assert manifest["resized_count"] == 1
    out_img = next(out_dir.rglob("*.png"))
    with Image.open(out_img) as img:
        assert img.mode == "RGB"
        assert "transparency" not in img.info
        # Transparent (10, 20, 30) pixel composited onto white background -> (255, 255, 255).
        assert img.getpixel((0, 0)) == (255, 255, 255)

    # 2. verify_prepared_dataset must reject mismatched archive_sha256 in manifest.
    manifest_file = out_dir / "dataset-manifest.json"
    bad_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    bad_manifest["archive_sha256"] = "0" * 64
    manifest_file.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="archive_sha256 mismatch"):
        prep.verify_prepared_dataset(out_dir, dataset="ffhq64", resolution=64, expected_count=1)

    # 3. Member count bomb guard in _extract_from_zip.
    with pytest.raises(ValueError, match="exceeding limit"):
        prep._extract_from_zip(zip_path, tmp_path / "bomb_stage", resolution=64, workers=1, max_images=None, allow_fast_copy=True, max_members=0)

    # 4. Member byte size bomb guard in _extract_from_zip.
    with pytest.raises(ValueError, match="exceeds limit"):
        prep._extract_from_zip(zip_path, tmp_path / "bomb_stage2", resolution=64, workers=1, max_images=None, allow_fast_copy=True, max_member_bytes=4)

    # 5. HTTP 416 / already-full .part resume must not wedge.
    archive_target = tmp_path / "dl" / "mini.zip"
    archive_target.parent.mkdir(parents=True)
    part_file = archive_target.with_name(archive_target.name + ".part")
    part_file.write_bytes(zip_path.read_bytes())
    with _FlakyHTTPServer({"/mini.zip": zip_path.read_bytes()}) as server:
        dl_spec = replace(spec, url=server.url("/mini.zip"))
        got = prep.download_and_verify_archive(dl_spec, archive_target, retries=1)
        assert got == archive_target.resolve()
        assert len(server.requests) == 0  # Promoted full valid .part without sending 416!


def test_item_data_len_exhaustion_sharding_and_close():
    """ItemData implements __len__, raises EvaluationDataExhausted when shuffle=False exhausts, validates sharding, closes pool."""
    data = ItemData(f"{__name__}:_TinyItems", args={"count": 6}, shuffle=False, workers=2)
    try:
        assert len(data) == 6
        rng = torch.Generator().manual_seed(7)
        b1 = data(4, generator=rng)
        assert b1["real"].shape == (4, 2)
        b2 = data(2, generator=rng)
        assert b2["real"].shape == (2, 2)

        saved_state = data.state_dict()
        saved_rng = rng.get_state()
        with pytest.raises(EvaluationDataExhausted, match="never wraps"):
            data(1, generator=rng)
        assert data.state_dict() == saved_state
        assert torch.equal(rng.get_state(), saved_rng)

        # Invalid rank/world_size/batch_size rejected.
        for rank, world_size, bs in ((0, 2, 3), (2, 2, 4), (-1, 2, 4), (1, 1, 2), (0, 0, 2)):
            with pytest.raises(ValueError, match="rank < world_size"):
                data(bs, generator=rng, rank=rank, world_size=world_size)

        # check_evaluation_data rejects shuffle=True ItemData for reference evaluation.
        shuffled = ItemData(f"{__name__}:_TinyItems", args={"count": 6}, shuffle=True, workers=0)
        with pytest.raises(ValueError, match="must not shuffle"):
            check_evaluation_data(shuffled, {"sample_count": 4}, {"reference": "evaluation.reference"})
        with pytest.raises(EvaluationDataExhausted):
            check_evaluation_data(data, {"sample_count": 7}, {"reference": "evaluation.reference"})
    finally:
        data.close()
        assert data._pool is None


def test_cifar10_data_len_no_toctou_sharding_and_eval_checks(tmp_path, monkeypatch):
    """CIFAR10Data exposes __len__, uses single-read pickle.loads, validates sharding, and is checked by check_evaluation_data."""
    images = np.arange(8 * 3072, dtype=np.int64).astype(np.uint8).reshape(8, 3072)
    labels = list(range(8))
    payload = pickle.dumps({"data": images, "labels": labels}, protocol=2)
    (tmp_path / "test_batch").write_bytes(payload)
    monkeypatch.setitem(dataset_download.CIFAR10_SHA256, "test_batch", hashlib.sha256(payload).hexdigest())

    # Ensure pickle.load (file-handle TOCTOU) is never called; only verified in-memory bytes via pickle.loads.
    monkeypatch.setattr(pickle, "load", lambda *a, **kw: pytest.fail("pickle.load must not be called (TOCTOU)"))

    seq = CIFAR10Data(tmp_path, split="test", sampling="sequential", horizontal_flip=False)
    assert len(seq) == 8
    check_evaluation_data(seq, {"sample_count": 8}, {"reference": "evaluation.reference"})
    with pytest.raises(EvaluationDataExhausted):
        check_evaluation_data(seq, {"sample_count": 9}, {"reference": "evaluation.reference"})

    rand = CIFAR10Data(tmp_path, split="test", sampling="random", horizontal_flip=False)
    with pytest.raises(ValueError, match='sampling = "sequential"'):
        check_evaluation_data(rand, {"sample_count": 4}, {"reference": "evaluation.reference"})

    flipped = CIFAR10Data(tmp_path, split="test", sampling="sequential", horizontal_flip=True)
    with pytest.raises(ValueError, match="horizontal_flip = false"):
        check_evaluation_data(flipped, {"sample_count": 4}, {"reference": "evaluation.reference"})

    rng = torch.Generator().manual_seed(1)
    for rank, world_size, bs in ((0, 2, 3), (2, 2, 4), (-1, 2, 4), (1, 1, 2)):
        with pytest.raises(ValueError, match="rank < world_size"):
            seq(bs, generator=rng, rank=rank, world_size=world_size)


def test_image_folder_symlink_cycle_cache_key_and_png_trns(tmp_path):
    """ImageFolder breaks symlink cycles, includes follow_symlinks in identity/cache key, and detects PNG tRNS."""
    root = tmp_path / "cycle_ds"
    sub = root / "sub"
    sub.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(sub / "a.png")
    (sub / "loop").symlink_to(root, target_is_directory=True)

    cache_dir = tmp_path / "inv_cache"
    ds = ImageFolder(root, height=4, width=4, follow_symlinks=True, recursive=True, workers=0, manifest_cache=cache_dir)
    assert len(ds) == 1
    assert ds.resume_identity()["follow_symlinks"] is True

    # Separate cache key for follow_symlinks=False vs True on a symlink-free folder.
    clean = tmp_path / "clean_ds"
    clean.mkdir()
    Image.new("RGB", (4, 4), (40, 50, 60)).save(clean / "b.png")
    c2 = tmp_path / "inv_cache2"
    d_false = ImageFolder(clean, height=4, width=4, follow_symlinks=False, workers=0, manifest_cache=c2)
    d_true = ImageFolder(clean, height=4, width=4, follow_symlinks=True, workers=0, manifest_cache=c2)
    assert d_false.resume_identity()["follow_symlinks"] is False
    assert d_true.resume_identity()["follow_symlinks"] is True
    assert d_false.resume_identity() != d_true.resume_identity()
    assert len(list(c2.glob("*.json"))) == 2

    # RGB PNG with tRNS chunk must be rejected under alpha="error" and composited under alpha="composite_white".
    trns_dir = tmp_path / "trns_ds"
    trns_dir.mkdir()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(trns_dir / "trns.png", format="PNG", transparency=(10, 20, 30))
    with pytest.raises(ValueError, match="alpha channel"):
        ImageFolder(trns_dir, height=4, width=4, alpha="error", workers=0, manifest_cache=False)
    comp = ImageFolder(trns_dir, height=4, width=4, alpha="composite_white", workers=0, manifest_cache=False)
    batch = comp(1, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(batch["real"], torch.ones(1, 3, 4, 4))


def test_dataset_download_dangling_symlink_at_extracted_target_rejected(tmp_path, monkeypatch):
    """Dangling symlink at extracted_target must fail cleanly with ValueError."""
    archive = tmp_path / "synthetic.tar.gz"
    files_sha: dict[str, str] = {}
    with tarfile.open(archive, "w:gz") as tf:
        d = tarfile.TarInfo(name="cifar-10-batches-py")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        for name in (*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"):
            payload = f"payload-{name}".encode()
            files_sha[name] = hashlib.sha256(payload).hexdigest()
            info = tarfile.TarInfo(name=f"cifar-10-batches-py/{name}")
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))
    raw = archive.read_bytes()
    patched = dict(dataset_download.QUALIFIED_DATASETS["cifar10"])
    patched.update(expected_bytes=len(raw), archive_sha256=hashlib.sha256(raw).hexdigest(), files_sha256=files_sha)
    monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "cifar-10-batches-py").symlink_to(tmp_path / "nonexistent_target")
    with pytest.raises(ValueError, match="must not be a symlink"):
        dataset_download.download_dataset("cifar10", dest, source_archive=archive)


def test_pretrained_providers_resnet18_ceiling_shapes_and_checkout_stamp(tmp_path):
    """_resnet18_shape_fn uses ceiling division and _checkout_stamp detects unstaged/untracked working-tree edits."""
    assert _resnet18_shape_fn([2, 3, 33, 35], layer="layer1") == (2, 64, 9, 9)
    assert _resnet18_shape_fn([2, 3, 33, 35], layer="layer2") == (2, 128, 5, 5)
    assert _resnet18_shape_fn([2, 3, 33, 35], layer="layer3") == (2, 256, 3, 3)
    assert _resnet18_shape_fn([2, 3, 33, 35], layer="layer4") == (2, 512, 2, 2)
    assert _resnet18_shape_fn([2, 3, 24, 24], layer="layer4") == (2, 512, 1, 1)

    # Create a real temporary git repo to verify _checkout_stamp invalidates on unstaged edits & untracked files.
    repo = tmp_path / "dinov3_repo"
    (repo / "dinov3").mkdir(parents=True)
    tracked = repo / "dinov3" / "mod.py"
    tracked.write_text("X = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    try:
        _verify_checkout(repo, commit)
        stamp_clean = _checkout_stamp(repo)

        # Unstaged modification to tracked file must change stamp and fail _verify_checkout!
        tracked.write_text("X = 2 # modified unstaged\n", encoding="utf-8")
        assert _checkout_stamp(repo) != stamp_clean
        with pytest.raises(ValueError, match="dirty=True"):
            _verify_checkout(repo, commit)

        # Restore tracked file -> passes again; then add an untracked file -> must invalidate!
        tracked.write_text("X = 1\n", encoding="utf-8")
        _verify_checkout(repo, commit)
        stamp_clean2 = _checkout_stamp(repo)
        (repo / "dinov3" / "untracked.py").write_text("Y = 3\n", encoding="utf-8")
        assert _checkout_stamp(repo) != stamp_clean2
        with pytest.raises(ValueError, match="dirty=True"):
            _verify_checkout(repo, commit)
    finally:
        _VERIFIED.pop((repo, commit), None)


# ---------------------------------------------------------------------------
# 4. Examples parity
# ---------------------------------------------------------------------------

def test_examples_cifar10_transgan_uses_particlegan_defaults():
    """examples/cifar10-transgan.toml must use defaults = 'particlegan' (no legacy ka2 fallback)."""
    example_path = REPO_ROOT / "examples" / "cifar10-transgan.toml"
    model = hg.load(example_path)
    assert model.spec.get("defaults") == "particlegan"
    resolved = load_config(example_path)
    assert resolved.get("recipe", {}).get("name") == "axiom"
    assert resolved["recipe"]["critic_formulation"] == "axiom"

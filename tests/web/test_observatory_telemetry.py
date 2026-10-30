import asyncio
import json
import os
from pathlib import Path
import re
import time

import pytest
from starlette.testclient import TestClient

from hypergan import inspection
from hypergan.image_grids import encode_png
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_server import (
    ServiceRegistry,
    _summary_watermark,
    create_app,
    read_run_summary,
)
from hypergan.web_service import (
    ObservationService,
    extract_topology_and_batch,
)
from hypergan.web_session import LocalSession


def _write_minimal_catalog(run_dir: Path) -> str:
    definition = {"kind": "scalar", "source": "g_loss", "label": "Generator loss"}
    definition["definition_hash"] = digest(definition)
    catalog = {"schema_version": 1, "metrics": {"loss/g_total": definition}}
    revision = digest(catalog)
    atomic_json(run_dir / "metrics" / f"catalog-{revision}.json", catalog)
    return revision


def test_topology_and_batch_2gpu_single_node():
    manifest = {
        "schema_version": 1,
        "run_id": "cifar10-transgan-2gpu",
        "global_batch_size": 256,
        "steps_per_second": 1.46,
        "execution": {
            "name": "cuda-replicated-nccl",
            "world_size": 2,
            "nodes": 1,
            "ranks_per_node": 2,
            "global_batch_size": 256,
            "local_batch_size": 128,
            "microbatch_size": 128,
            "accumulation_steps": 1,
        },
        "runtime": {
            "device": "cuda:0",
            "world_size": 2,
            "cuda": {
                "rank_devices": [
                    {"rank": 0, "index": 0, "name": "NVIDIA RTX A6000"},
                    {"rank": 1, "index": 1, "name": "NVIDIA RTX A6000"},
                ]
            },
        },
        "config": {"training": {"device": "cuda", "batch_size": 256, "steps": 100000}},
    }
    res = extract_topology_and_batch(manifest)
    topo = res["topology"]
    assert topo == {
        "device_kind": "cuda",
        "world_size": 2,
        "nodes": 1,
        "ranks_per_node": 2,
        "gpu_count": 2,
        "gpu_name": "RTX A6000",
        "label": "2 GPUs",
        "detail": "2× RTX A6000",
    }
    assert res["global_batch_size"] == 256
    assert res["local_batch_size"] == 128
    assert res["microbatch_size"] == 128
    assert res["accumulation_steps"] == 1
    assert res["batch_label"] == "Batch 256"
    assert res["batch_detail"] == "128 / GPU"


def test_topology_and_batch_8gpu_2nodes_and_accumulation():
    manifest = {
        "run_id": "cluster-8gpu",
        "execution": {
            "name": "cuda-replicated-nccl",
            "world_size": 8,
            "nodes": 2,
            "global_batch_size": 512,
            "local_batch_size": 64,
            "accumulation_steps": 2,
        },
        "runtime": {
            "device": "cuda:0",
            "cuda": {"name": "NVIDIA H100"},
        },
    }
    res = extract_topology_and_batch(manifest)
    topo = res["topology"]
    assert topo["device_kind"] == "cuda"
    assert topo["world_size"] == 8
    assert topo["nodes"] == 2
    assert topo["ranks_per_node"] == 4
    assert topo["gpu_count"] == 8
    assert topo["gpu_name"] == "H100"
    assert topo["label"] == "8 GPUs · 2 nodes"
    assert topo["detail"] == "8× H100 · 4/node"
    assert res["global_batch_size"] == 512
    assert res["local_batch_size"] == 64
    assert res["microbatch_size"] == 32
    assert res["accumulation_steps"] == 2
    assert res["batch_label"] == "Batch 512"
    assert res["batch_detail"] == "64 / GPU · ×2 accum"


def test_topology_and_batch_1gpu_and_heterogeneous_gpus():
    manifest_1gpu = {
        "run_id": "single-gpu",
        "runtime": {
            "device": "cuda:0",
            "world_size": 1,
            "cuda": {"name": "NVIDIA RTX A6000"},
        },
        "config": {"training": {"device": "cuda", "batch_size": 64}},
    }
    res = extract_topology_and_batch(manifest_1gpu)
    assert res["topology"]["label"] == "1 GPU"
    assert res["topology"]["detail"] == "RTX A6000"
    assert res["topology"]["gpu_count"] == 1
    assert res["batch_label"] == "Batch 64"
    assert res["batch_detail"] is None
    assert res["local_batch_size"] == 64
    assert res["microbatch_size"] == 64

    # 1-GPU with gradient accumulation
    manifest_1gpu_accum = {
        **manifest_1gpu,
        "execution": {"accumulation_steps": 4},
    }
    res_accum = extract_topology_and_batch(manifest_1gpu_accum)
    assert res_accum["batch_label"] == "Batch 64"
    assert res_accum["batch_detail"] == "×4 accum"
    assert res_accum["microbatch_size"] == 16

    # Heterogeneous 2-GPU workstation
    manifest_hetero = {
        "run_id": "hetero-2gpu",
        "execution": {"name": "cuda-replicated-nccl", "world_size": 2, "nodes": 1},
        "runtime": {
            "cuda": {
                "rank_devices": [
                    {"rank": 0, "name": "NVIDIA RTX A6000"},
                    {"rank": 1, "name": "NVIDIA RTX 4090"},
                ]
            }
        },
    }
    res_hetero = extract_topology_and_batch(manifest_hetero)
    assert res_hetero["topology"]["label"] == "2 GPUs"
    assert res_hetero["topology"]["gpu_name"] == "RTX A6000 + RTX 4090"
    assert res_hetero["topology"]["detail"] == "RTX A6000 + RTX 4090"


def test_topology_and_batch_cpu_mps_and_minimal_manifest():
    # Single-process CPU
    res_cpu = extract_topology_and_batch(
        {"run_id": "cpu-run", "config": {"training": {"device": "cpu", "batch_size": 16}}}
    )
    assert res_cpu["topology"]["device_kind"] == "cpu"
    assert res_cpu["topology"]["gpu_count"] == 0
    assert res_cpu["topology"]["label"] == "CPU"
    assert res_cpu["topology"]["detail"] is None
    assert res_cpu["batch_label"] == "Batch 16"
    assert res_cpu["batch_detail"] is None

    # Replicated CPU ranks
    res_cpu_rep = extract_topology_and_batch(
        {
            "run_id": "cpu-ddp",
            "execution": {"name": "cpu-replicated-gloo", "world_size": 4, "nodes": 1, "global_batch_size": 64},
        }
    )
    assert res_cpu_rep["topology"]["label"] == "4 CPU ranks"
    assert res_cpu_rep["batch_detail"] == "16 / rank"

    # Apple Silicon MPS
    res_mps = extract_topology_and_batch(
        {"run_id": "mps-run", "runtime": {"device": "mps"}, "config": {"training": {"batch_size": 32}}}
    )
    assert res_mps["topology"]["device_kind"] == "mps"
    assert res_mps["topology"]["gpu_count"] == 1
    assert res_mps["topology"]["label"] == "MPS"
    assert res_mps["batch_label"] == "Batch 32"

    # Minimal unrecorded manifest must NOT falsely claim CPU
    res_bare = extract_topology_and_batch({"run_id": "bare", "steps": 5})
    assert res_bare["topology"] == {
        "device_kind": None,
        "world_size": None,
        "nodes": None,
        "ranks_per_node": None,
        "gpu_count": 0,
        "gpu_name": None,
        "label": None,
        "detail": None,
    }
    assert res_bare["global_batch_size"] is None
    assert res_bare["batch_label"] is None
    assert res_bare["batch_detail"] is None


def test_inactive_active_crashed_throughput_and_eta(tmp_path, monkeypatch):
    # 1. Stopped run with recorded steps_per_second and stale eta_seconds in summary.json
    stopped_dir = tmp_path / "stopped_2gpu"
    stopped_dir.mkdir()
    rev = _write_minimal_catalog(stopped_dir)
    atomic_json(
        stopped_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "stopped_2gpu",
            "status": "stopped",
            "steps": 400,
            "total_steps": 1000,
            "global_batch_size": 256,
            "steps_per_second": 1.46,
            "metrics_catalog": rev,
            "execution": {
                "name": "cuda-replicated-nccl",
                "world_size": 2,
                "nodes": 1,
                "global_batch_size": 256,
                "local_batch_size": 128,
            },
            "runtime": {
                "device": "cuda:0",
                "cuda": {
                    "rank_devices": [
                        {"rank": 0, "name": "NVIDIA RTX A6000"},
                        {"rank": 1, "name": "NVIDIA RTX A6000"},
                    ]
                },
            },
        },
    )
    atomic_json(
        stopped_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": "stopped_2gpu",
            "status": "stopped",
            "step": 400,
            "total_steps": 1000,
            "steps_per_second": 1.46,
            "eta_seconds": 410.96,  # stale persisted ETA must be suppressed when inactive
        },
    )

    s_stopped = read_run_summary(stopped_dir)
    assert s_stopped["active"] is False
    assert s_stopped["eta_seconds"] is None
    assert s_stopped["samples_per_second"] == pytest.approx(1.46 * 256)
    assert s_stopped["topology"]["label"] == "2 GPUs"
    assert s_stopped["topology"]["detail"] == "2× RTX A6000"
    assert s_stopped["batch_label"] == "Batch 256"
    assert s_stopped["batch_detail"] == "128 / GPU"

    # 2. Active run computes accurate eta_seconds and samples_per_second
    active_dir = tmp_path / "active_2gpu"
    active_dir.mkdir()
    rev2 = _write_minimal_catalog(active_dir)
    atomic_json(
        active_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "active_2gpu",
            "status": "running",
            "steps": 250,
            "total_steps": 1000,
            "global_batch_size": 256,
            "steps_per_second": 1.5,
            "metrics_catalog": rev2,
            "execution": {"name": "cuda-replicated-nccl", "world_size": 2, "nodes": 1},
        },
    )
    monkeypatch.setattr(inspection, "check_run_active", lambda p: (Path(p).resolve() == active_dir.resolve(), 4242))
    s_active = read_run_summary(active_dir)
    assert s_active["status"] == "running"
    assert s_active["active"] is True
    assert s_active["running"] is True
    assert s_active["samples_per_second"] == pytest.approx(384.0)
    assert s_active["eta_seconds"] == pytest.approx(500.0)

    # 3. Crashed run (manifest says "running", unlocked .run.lock exists, check_run_active is False)
    crashed_dir = tmp_path / "crashed_run"
    crashed_dir.mkdir()
    rev3 = _write_minimal_catalog(crashed_dir)
    atomic_json(
        crashed_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "crashed_run",
            "status": "running",
            "steps": 100,
            "total_steps": 1000,
            "global_batch_size": 128,
            "steps_per_second": 2.0,
            "metrics_catalog": rev3,
        },
    )
    (crashed_dir / ".run.lock").write_text(json.dumps({"pid": 999999}), encoding="utf-8")

    s_crashed = read_run_summary(crashed_dir)
    assert s_crashed["status"] == "stopped"
    assert s_crashed["active"] is False
    assert s_crashed["running"] is False
    assert s_crashed["eta_seconds"] is None

    async def check_svc():
        svc = await ObservationService(crashed_dir, poll_seconds=1.0).start()
        try:
            pub = svc.public_manifest()
            assert pub["status"] == "stopped"
            assert pub["active"] is False
            assert pub["running"] is False
            assert pub["eta_seconds"] is None
        finally:
            await svc.close()

    asyncio.run(check_svc())


def test_thumbnail_revision_watermark_and_cache_control(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_dir = runs_root / "thumb_rev_run"
    run_dir.mkdir()
    rev_cat = _write_minimal_catalog(run_dir)
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "thumb_rev_run",
            "status": "running",
            "steps": 10,
            "total_steps": 100,
            "metrics_catalog": rev_cat,
        },
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "thumb_rev_run",
                "attempt_id": "att-1",
                "stream_id": "training",
                "stream_generation": "thumb_rev_run",
                "sequence": 1,
                "step": 0,
                "event": "start",
                "catalog": rev_cat,
                "parent_attempt_id": None,
                "restored_step": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    svc_reg = ServiceRegistry(runs_root)
    initial_row = svc_reg._discover_runs()[0]
    assert initial_row["thumbnail_revision"] is None
    assert initial_row["thumbnail_url"] is None
    wm0 = _summary_watermark(run_dir)

    # Publish first preview in previews/index.json (without touching summary.json or manifest.json)
    previews_dir = run_dir / "previews"
    previews_dir.mkdir()
    png1 = encode_png(b"\xff\x00\x00", 1, 1, 3)
    png2 = encode_png(b"\x00\xff\x00", 1, 1, 3)
    (previews_dir / "grid_1.png").write_bytes(png1)
    (previews_dir / "grid_2.png").write_bytes(png2)

    sha1 = "a1b2c3d4e5f60718293a4b5c6d7e8f90" * 2
    sha2 = "f9e8d7c6b5a4938271605f4e3d2c1b0a" * 2
    idx_path = previews_dir / "index.json"
    idx_data = {
        "schema_version": 1,
        "previews": [
            {
                "step": 10,
                "sample_sequence": 1,
                "path": "previews/0001.json",
                "bytes": 64,
                "sha256": sha1,
                "shape": [1, 3, 1, 1],
                "identity": {"name": "fixed_latents"},
                "image_grid": {
                    "path": "previews/grid_1.png",
                    "bytes": len(png1),
                    "sha256": sha1,
                    "width": 1,
                    "height": 1,
                },
            }
        ],
    }
    atomic_json(idx_path, idx_data)
    wm1 = _summary_watermark(run_dir)
    assert wm1 != wm0

    row1 = svc_reg._discover_runs()[0]
    assert row1["thumbnail_revision"] == "a1b2c3d4e5f6"
    assert row1["thumbnail_url"] == "/api/v1/runs/thumb_rev_run/thumbnail?rev=a1b2c3d4e5f6"

    # Append second preview and ensure watermark + revision + URL advance
    idx_data["previews"].append(
        {
            "step": 20,
            "sample_sequence": 2,
            "path": "previews/0002.json",
            "bytes": 64,
            "sha256": sha2,
            "shape": [1, 3, 1, 1],
            "identity": {"name": "fixed_latents"},
            "image_grid": {
                "path": "previews/grid_2.png",
                "bytes": len(png2),
                "sha256": sha2,
                "width": 1,
                "height": 1,
            },
        }
    )
    atomic_json(idx_path, idx_data)
    st = idx_path.stat()
    os.utime(idx_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    wm2 = _summary_watermark(run_dir)
    assert wm2 != wm1

    row2 = svc_reg._discover_runs()[0]
    assert row2["thumbnail_revision"] == "f9e8d7c6b5a4"
    assert row2["thumbnail_url"] == "/api/v1/runs/thumb_rev_run/thumbnail?rev=f9e8d7c6b5a4"

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        # Versioned request gets private max-age caching
        res_ver = client.get(row2["thumbnail_url"])
        assert res_ver.status_code == 200
        assert res_ver.headers["content-type"] == "image/png"
        assert res_ver.headers["cache-control"] == "private, max-age=3600"
        assert res_ver.content == png2

        # Unversioned request preserves no-cache
        res_unver = client.get("/api/v1/runs/thumb_rev_run/thumbnail")
        assert res_unver.status_code == 200
        assert res_unver.headers["cache-control"] == "no-cache"
        assert res_unver.content == png2

    # Fallback standalone PNG without previews/index.json also derives a 12-char hex revision
    fallback_dir = runs_root / "fallback_png_run"
    (fallback_dir / "previews").mkdir(parents=True)
    atomic_json(fallback_dir / "manifest.json", {"run_id": "fallback_png_run", "status": "stopped", "steps": 5})
    (fallback_dir / "previews" / "latest.png").write_bytes(png1)
    fallback_summary = read_run_summary(fallback_dir)
    assert fallback_summary["thumbnail_revision"] is not None
    assert re.fullmatch(r"[0-9a-f]{12}", fallback_summary["thumbnail_revision"])
    assert fallback_summary["thumbnail_url"] == (
        f"/api/v1/runs/fallback_png_run/thumbnail?rev={fallback_summary['thumbnail_revision']}"
    )


def test_observation_service_public_manifest_and_endpoints_enriched(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_dir = runs_root / "obs_enriched"
    run_dir.mkdir()
    rev_cat = _write_minimal_catalog(run_dir)

    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "obs_enriched",
            "attempt_id": "att-1",
            "status": "running",
            "steps": 500,
            "total_steps": 2500,
            "global_batch_size": 256,
            "steps_per_second": 2.0,
            "metrics_catalog": rev_cat,
            "warnings": ["TF32 enabled on Ampere"],
            "execution": {
                "name": "cuda-replicated-nccl",
                "world_size": 2,
                "nodes": 1,
                "ranks_per_node": 2,
                "global_batch_size": 256,
                "local_batch_size": 128,
                "microbatch_size": 128,
                "accumulation_steps": 1,
            },
            "runtime": {
                "device": "cuda:0",
                "world_size": 2,
                "cuda": {
                    "rank_devices": [
                        {"rank": 0, "name": "NVIDIA RTX A6000"},
                        {"rank": 1, "name": "NVIDIA RTX A6000"},
                    ]
                },
            },
            "config": {
                "name": "CIFAR-10 TransGAN 2-GPU",
                "training": {"device": "cuda", "batch_size": 256, "steps": 2500},
            },
        },
    )
    summary_path = run_dir / "summary.json"
    atomic_json(
        summary_path,
        {
            "schema_version": 1,
            "run_id": "obs_enriched",
            "status": "running",
            "step": 500,
            "total_steps": 2500,
            "steps_per_second": 2.0,
            "latest_fid": 24.5,
            "best": {"metric": "fid50k_train", "value": 19.25, "step": 400, "unit": "fid"},
            "health_alerts": ["Discriminator gradient norm spike at step 480"],
            "updated_at": time.time(),
        },
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "obs_enriched",
                "attempt_id": "att-1",
                "stream_id": "training",
                "stream_generation": "obs_enriched",
                "sequence": 1,
                "step": 0,
                "event": "start",
                "catalog": rev_cat,
                "parent_attempt_id": None,
                "restored_step": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(inspection, "check_run_active", lambda p: (True, 12345))

    session = LocalSession(8765)
    app = create_app(runs_root, session=session)
    with TestClient(app, base_url=session.origin) as client:
        for endpoint in ("/api/v1/runs/obs_enriched", "/api/v1/runs/obs_enriched/summary"):
            payload = client.get(endpoint).json()
            assert payload["run_id"] == "obs_enriched"
            assert payload["name"] == "CIFAR-10 TransGAN 2-GPU"
            assert payload["status"] == "running"
            assert payload["active"] is True
            assert payload["running"] is True
            assert payload["topology"]["label"] == "2 GPUs"
            assert payload["topology"]["detail"] == "2× RTX A6000"
            assert payload["global_batch_size"] == 256
            assert payload["local_batch_size"] == 128
            assert payload["microbatch_size"] == 128
            assert payload["accumulation_steps"] == 1
            assert payload["batch_label"] == "Batch 256"
            assert payload["batch_detail"] == "128 / GPU"
            assert payload["samples_per_second"] == pytest.approx(512.0)
            assert payload["eta_seconds"] == pytest.approx(1000.0)
            assert payload["best_fid"] == pytest.approx(19.25)
            assert payload["latest_fid"] == pytest.approx(24.5)
            assert payload["best_step"] == 400
            assert payload["best_metric"] == "fid50k_train"
            assert payload["health_alerts"] == ["Discriminator gradient norm spike at step 480"]
            assert payload["warnings"] == ["TF32 enabled on Ampere"]

        # Update summary.json on disk and verify ObservationService summary cache invalidates
        updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        updated_summary["latest_fid"] = 18.0
        updated_summary["best"] = {"metric": "fid50k_train", "value": 18.0, "step": 500, "unit": "fid"}
        updated_summary["health_alerts"] = []
        atomic_json(summary_path, updated_summary)
        st = summary_path.stat()
        os.utime(summary_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        refreshed = client.get("/api/v1/runs/obs_enriched").json()
        assert refreshed["best_fid"] == pytest.approx(18.0)
        assert refreshed["latest_fid"] == pytest.approx(18.0)
        assert refreshed["best_step"] == 500
        assert refreshed["health_alerts"] == []

        # /api/v1/runs list endpoint carries identical topology/batch/throughput telemetry
        runs_list = client.get("/api/v1/runs").json()
        assert isinstance(runs_list, list)
        listed = next(r for r in runs_list if r["run_id"] == "obs_enriched")
        assert listed["topology"]["label"] == "2 GPUs"
        assert listed["topology"]["detail"] == "2× RTX A6000"
        assert listed["batch_label"] == "Batch 256"
        assert listed["batch_detail"] == "128 / GPU"
        assert listed["samples_per_second"] == pytest.approx(512.0)
        assert listed["eta_seconds"] == pytest.approx(1000.0)
        assert listed["best_fid"] == pytest.approx(18.0)
        assert listed["best_metric"] == "fid50k_train"


def test_non_fid_primary_metric_does_not_masquerade_as_fid(tmp_path):
    run_dir = tmp_path / "is_run"
    run_dir.mkdir()
    atomic_json(run_dir / "manifest.json", {"run_id": "is_run", "status": "stopped", "steps": 300})
    atomic_json(
        run_dir / "summary.json",
        {
            "run_id": "is_run",
            "status": "stopped",
            "step": 300,
            "best": {"metric": "eval/inception_score", "value": 9.42, "step": 250, "unit": "IS"},
        },
    )
    summary = read_run_summary(run_dir)
    assert summary["best_fid"] is None
    assert summary["fid_best"] is None
    assert summary["best_step"] == 250
    assert summary["best_metric"] == "eval/inception_score"


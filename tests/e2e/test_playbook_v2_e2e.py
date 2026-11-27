"""HyperGAN 2.0.0 GA Playbook v2 End-to-End Test Suite.

Comprehensive, requirement-driven opaque-box test suite across Tiers 1-4:
- Tier 1: Feature Coverage (Category-Partition Happy Paths across R1-R4)
- Tier 2: Boundary & Corner Cases (Boundary Value Analysis, Limits, Stress)
- Tier 3: Cross-Feature Combinations (Pairwise Feature Interactions)
- Tier 4: Real-World Application Scenarios (Realistic End-to-End Workflows)

All expected outputs are derived from authoritative specifications in PROJECT.md,
ORIGINAL_REQUEST.md, and mathematical oracles defined in tests/e2e/harness_v2.py.
Every feature covered here has shipped: zero capability-probing skips are used.
"""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import safetensors.torch
import torch

import hypergan
import hypergan.api as hg
from hypergan.artifacts import LoadedGenerator
import hypergan.cache as hg_cache
import hypergan.cli as cli
from hypergan.data import ImageDataError, ImageFolder
from hypergan.execution import (
    _is_follower,
    detect_cluster_environment,
    prepare_resume,
    prepare_train,
)
from hypergan.image_metrics import InceptionFID, _Moments, frechet_distance
from hypergan.network_config import validate_network_args
from hypergan.recipe_generator import generate_recipe
from hypergan.registry import register_run
from hypergan.run_info import RunInfo
from hypergan.summary import write_summary_json
from tests.e2e.harness_v2 import (
    create_synthetic_image_folder,
    oracle_lerp,
    oracle_sha256,
    oracle_slerp,
    run_cli,
)

_CLUSTER_ENV_KEYS = (
    "SLURM_NNODES",
    "SLURM_NODEID",
    "NNODES",
    "NODE_RANK",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _clear_cluster_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove cluster env vars so tests control them deterministically."""
    for key in _CLUSTER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Suite Isolation Fixture (Clean State & Environment)
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide complete filesystem and environment variable isolation."""
    state_dir = tmp_path / "hypergan_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "hypergan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _clear_cluster_env(monkeypatch)
    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HYPERGAN_QUEUE_FILE", str(state_dir / "queue.json"))
    monkeypatch.setenv("HYPERGAN_NO_SERVER", "1")
    monkeypatch.setenv("CI", "1")
    monkeypatch.chdir(tmp_path)
    return state_dir


# ===========================================================================
# Tier 1: Feature Coverage (Category-Partition Happy Paths across R1-R4)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature Coverage verifying primary behavioral contracts."""

    def test_tier1_cli_version_and_help(self, isolated_env):
        """Verify CLI entrypoint reports current __version__ and primary subcommands."""
        rc_ver, out_ver, err_ver = run_cli("--version")
        assert rc_ver == 0, err_ver
        assert hypergan.__version__ in out_ver

        rc_help, out_help, err_help = run_cli("--help")
        assert rc_help == 0, err_help
        for subcmd in ("new", "validate", "train", "resume", "export", "sample", "ps", "status", "inspect"):
            assert subcmd in out_help

    def test_tier1_cli_new_list_templates(self, isolated_env):
        """Verify `hypergan new --list` lists all starter architecture templates."""
        rc, out, err = run_cli("new", "--list")
        assert rc == 0, err
        for tmpl in ("dcgan", "tiny-transformer", "resnet", "gaussian-grid"):
            assert tmpl in out

    def test_tier1_cli_recipes_list(self, isolated_env):
        """Verify `hypergan recipes` returns valid JSON list of packaged models."""
        rc, out, err = run_cli("recipes")
        assert rc == 0, err
        data = json.loads(out)
        assert isinstance(data, list) and len(data) >= 1
        names = [item["name"] if isinstance(item, dict) else str(item) for item in data]
        assert "gaussian-grid" in names

    def test_tier1_cli_project_scaffold_and_validate(self, isolated_env, tmp_path):
        """Verify scaffolding gaussian-grid project and validating both file and directory."""
        proj_dir = tmp_path / "proj_gaussian"
        rc_new, out_new, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        assert cfg_file.is_file()
        assert (proj_dir / "hypergan.toml").is_file()

        rc_val, out_val, err_val = run_cli("validate", str(cfg_file))
        assert rc_val == 0, err_val
        assert "Valid:" in out_val

        rc_dir, out_dir, err_dir = run_cli("validate", str(proj_dir))
        assert rc_dir == 0, err_dir
        assert "Valid:" in out_dir

        rc_json, out_json, err_json = run_cli("validate", str(cfg_file), "--json")
        assert rc_json == 0, err_json
        payload = json.loads(out_json)
        assert isinstance(payload, dict)
        assert payload["training"]["device"] == "cpu"
        assert "generator" in payload["components"]
        assert "discriminator" in payload["components"]

    def test_tier1_cache_dir_resolution(self, isolated_env, monkeypatch, tmp_path):
        """Verify get_cache_dir resolves HYPERGAN_CACHE_DIR or ~/.hypergan/cache/."""
        expected_cache = (tmp_path / "custom_cache").resolve()
        monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(expected_cache))
        resolved = hg_cache.get_cache_dir()
        assert Path(resolved) == expected_cache
        assert expected_cache.is_dir()

        monkeypatch.delenv("HYPERGAN_CACHE_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        default_resolved = hg_cache.get_cache_dir()
        assert default_resolved == (tmp_path / "fake_home" / ".hypergan" / "cache").resolve()
        assert default_resolved.is_dir()

    def test_tier1_cache_local_path_bypass(self, isolated_env, tmp_path):
        """Verify resolve_or_cache returns local files directly without copying into cache."""
        cache_dir = Path(os.environ["HYPERGAN_CACHE_DIR"])
        local_file = tmp_path / "local_weights.pth"
        payload = b"dummy_weights_tensor_1234"
        local_file.write_bytes(payload)

        result = hg_cache.resolve_or_cache(local_file)
        assert Path(result) == local_file.resolve()
        assert Path(result).read_bytes() == payload
        assert list(cache_dir.iterdir()) == []

    def test_tier1_cache_offline_mode_enforcement(self, isolated_env, monkeypatch):
        """Verify HYPERGAN_OFFLINE=1 rejects remote downloading of uncached URLs."""
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
        with pytest.raises(hg_cache.OfflineError, match="(?i)offline"):
            hg_cache.ensure_cached_url("https://example.com/weights.pth")

    def test_tier1_cache_sha256_verification(self, isolated_env, tmp_path):
        """Verify SHA-256 verification and file locking in hypergan.cache."""
        test_bytes = b"authoritative_model_weights_sha256_verification_payload"
        expected_digest = oracle_sha256(test_bytes)
        assert len(expected_digest) == 64

        artifact = tmp_path / "verified_weights.bin"
        artifact.write_bytes(test_bytes)

        validated = hg_cache._validate_local_file(
            artifact, expected_sha256=expected_digest, expected_size=len(test_bytes)
        )
        assert validated == artifact

        # Uppercase hex digest normalization and public resolve_or_cache pass-through
        resolved = hg_cache.resolve_or_cache(
            artifact, expected_sha256=expected_digest.upper(), expected_size=len(test_bytes)
        )
        assert resolved == artifact.resolve()
        assert resolved.read_bytes() == test_bytes

        # File locking context manager creates and cleanly releases the lock file
        lock_path = tmp_path / "verified_weights.bin.lock"
        with hg_cache.cache_file_lock(lock_path):
            assert lock_path.exists()

    def test_tier1_runpod_stub_removed(self, isolated_env):
        """Verify runpod_runner.py stub is removed from the codebase."""
        runpod_path = Path(hypergan.__file__).parent / "runpod_runner.py"
        assert not runpod_path.exists(), f"Deprecated runpod_runner.py stub still exists: {runpod_path}"
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("hypergan.runpod_runner")

    def test_tier1_queue_cli_operations(self, isolated_env, tmp_path):
        """Verify queue CLI operations (ls, add, stop --all) mutate queue state."""
        rc_ls, out_ls, err_ls = run_cli("queue", "ls", "--json")
        assert rc_ls == 0, err_ls
        jobs = json.loads(out_ls)
        assert jobs == []

        proj_dir = tmp_path / "proj_queue_ops"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_add, _, err_add = run_cli("queue", "add", str(cfg_file), "-n", "queued-smoke", "--steps", "2")
        assert rc_add == 0, err_add

        rc_ls2, out_ls2, err_ls2 = run_cli("queue", "ls", "--json")
        assert rc_ls2 == 0, err_ls2
        jobs2 = json.loads(out_ls2)
        assert len(jobs2) == 1
        assert jobs2[0]["name"] == "queued-smoke"
        assert jobs2[0]["status"] == "queued"

        rc_stop, _, err_stop = run_cli("queue", "stop", "--all")
        assert rc_stop == 0, err_stop

        rc_ls3, out_ls3, err_ls3 = run_cli("queue", "ls", "--status", "queued", "--json")
        assert rc_ls3 == 0, err_ls3
        assert json.loads(out_ls3) == []

    def test_tier1_checkpoint_pruning_accounting(self, isolated_env, tmp_path):
        """Verify checkpoints prune executes under lock and accounts reclaimed bytes."""
        proj_dir = tmp_path / "proj_prune"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        # 4 steps with checkpoint_every=1 produces steps 0,1,2,3,4; keep_last=3 retains 0,2,3,4 and prunes step 1
        rc_train, _, err_train = run_cli(
            "train", str(cfg_file), "--steps", "4", "-d", "cpu", "--checkpoint-every", "1", "--no-server"
        )
        assert rc_train == 0, err_train

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st_data = json.loads(out_st)
        run_dir = Path(st_data["run_dir"])

        rc_dry, out_dry, err_dry = run_cli("checkpoints", "prune", str(run_dir), "--dry-run", "--json")
        assert rc_dry == 0, err_dry
        prune_report = json.loads(out_dry)
        assert prune_report["dry_run"] is True
        assert prune_report["retained_count"] == 4
        assert prune_report["bytes_reclaimed"] == prune_report["reclaimed_bytes"]

        # Applying a tighter retention policy accounts exact reclaimed bytes in dry-run, then deletes them
        ckpt_root = run_dir / "checkpoints"
        before_dirs = sorted(p.name for p in ckpt_root.iterdir() if p.is_dir())
        assert len(before_dirs) == 4

        tight_dry = hg.prune_checkpoints(run_dir, policy={"keep_last": 1, "keep_every": 0}, dry_run=True)
        assert tight_dry["dry_run"] is True
        assert tight_dry["pruned_count"] == 2
        assert tight_dry["reclaimed_bytes"] > 0
        # Dry-run must not remove any checkpoint directories on disk
        assert sorted(p.name for p in ckpt_root.iterdir() if p.is_dir()) == before_dirs

        tight_exec = hg.prune_checkpoints(run_dir, policy={"keep_last": 1, "keep_every": 0}, dry_run=False)
        assert tight_exec["dry_run"] is False
        assert tight_exec["deleted_count"] == tight_dry["pruned_count"]
        assert tight_exec["reclaimed_bytes"] == tight_dry["reclaimed_bytes"]
        assert tight_exec["retained_count"] == 2  # step 0 and latest step 4
        assert len([p for p in ckpt_root.iterdir() if p.is_dir()]) == 2

    def test_tier1_inception_fid_scipy_compatibility(self, isolated_env):
        """Verify InceptionFID and frechet_distance work accurately on SciPy 1.18+ without sqrtm disp crash."""
        assert hasattr(InceptionFID, "evaluate")

        # 1. Identical distributions -> FID == 0
        mu = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
        sigma = np.diag([2.0, 0.5, 4.0, 1.5]).astype(np.float64)
        fid_zero = frechet_distance({"mu": mu, "sigma": sigma}, {"mu": mu, "sigma": sigma})
        assert math.isfinite(fid_zero)
        assert abs(fid_zero) < 1e-6

        # 2. Pure mean shift with identical covariances -> exact FID == ||mu1 - mu2||^2
        delta = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64)
        fid_shift = frechet_distance({"mu": mu, "sigma": sigma}, {"mu": mu + delta, "sigma": sigma})
        assert math.isfinite(fid_shift)
        assert math.isclose(fid_shift, 25.0, rel_tol=1e-6, abs_tol=1e-6)

        # 3. Distinct diagonal covariances with equal means -> exact analytic trace formula:
        #    tr(diag(4,9)) + tr(diag(1,16)) - 2 * tr(sqrt(diag(4,144))) = 13 + 17 - 2*(2 + 12) = 2.0
        s1 = {"mu": np.zeros(2, dtype=np.float64), "sigma": np.diag([4.0, 9.0])}
        s2 = {"mu": np.zeros(2, dtype=np.float64), "sigma": np.diag([1.0, 16.0])}
        fid_cov = frechet_distance(s1, s2)
        assert math.isfinite(fid_cov)
        assert math.isclose(fid_cov, 2.0, rel_tol=1e-6, abs_tol=1e-6)

        # 4. Rank-deficient covariances via _Moments (stress case for eigval/sqrtm stability)
        rng = np.random.default_rng(42)
        m1, m2 = _Moments(), _Moments()
        m1.add(rng.normal(loc=0.0, scale=1.0, size=(4, 16)))
        m2.add(rng.normal(loc=0.5, scale=1.2, size=(4, 16)))
        fid_rank_def = frechet_distance(m1.statistics(), m2.statistics())
        assert math.isfinite(fid_rank_def)
        assert fid_rank_def >= -1e-6

    def test_tier1_multinode_cli_flags_and_strategy(self, isolated_env):
        """Verify multi-node CLI flags (--nodes, --strategy, --sync-every) are accepted and parsed."""
        rc, out, err = run_cli("train", "--help")
        assert rc == 0, err
        for flag in ("--nodes", "--node-rank", "--rendezvous", "--strategy", "--sync-every"):
            assert flag in out

        parsed = cli._parser().parse_args([
            "train", "gaussian-grid",
            "--nodes", "4",
            "--node-rank", "2",
            "--rendezvous", "tcp://10.0.0.1:29500",
            "--strategy", "pseudo_gradient",
            "--sync-every", "8",
        ])
        assert parsed.nodes == 4
        assert parsed.node_rank == 2
        assert parsed.rendezvous == "tcp://10.0.0.1:29500"
        assert parsed.strategy == "pseudo_gradient"
        assert parsed.sync_every == 8

    def test_tier1_cluster_environment_auto_detection(self, isolated_env, monkeypatch, tmp_path):
        """Verify cluster environment variables (SLURM_NNODES, SLURM_NODEID, MASTER_ADDR) auto-detection."""
        # 1. Slurm auto-detection with default port 29500
        monkeypatch.setenv("SLURM_NNODES", "4")
        monkeypatch.setenv("SLURM_NODEID", "1")
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
        detected = detect_cluster_environment()
        assert detected == {
            "nodes": 4,
            "node_rank": 1,
            "rendezvous": "tcp://10.0.0.1:29500",
        }

        # Verify prepare_train consumes auto-detected cluster env into resolved execution profile
        proj_dir = tmp_path / "proj_slurm_detect"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        prepared = prepare_train(cfg_file, tmp_path / "run_slurm", steps=2, device="cpu")
        assert prepared.profile is not None
        exec_cfg = prepared.profile["execution"]
        assert exec_cfg["name"] == "cpu-replicated-gloo"
        assert exec_cfg["nodes"] == 4
        assert exec_cfg["node_rank"] == 1
        assert exec_cfg["world_size"] == 4
        assert exec_cfg["ranks_per_node"] == 1
        assert exec_cfg["rendezvous"] == "tcp://10.0.0.1:29500"

        # 2. Torchrun auto-detection with explicit port
        _clear_cluster_env(monkeypatch)
        monkeypatch.setenv("NNODES", "2")
        monkeypatch.setenv("NODE_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.2")
        monkeypatch.setenv("MASTER_PORT", "29601")
        assert detect_cluster_environment() == {
            "nodes": 2,
            "node_rank": 0,
            "rendezvous": "tcp://10.0.0.2:29601",
        }

        # 3. Malformed values are safely ignored
        _clear_cluster_env(monkeypatch)
        monkeypatch.setenv("SLURM_NNODES", "not_an_int")
        monkeypatch.setenv("SLURM_NODEID", "-3")
        assert detect_cluster_environment() == {}

    def test_tier1_static_hndl_ast_validation(self, isolated_env):
        """Verify static HNDL AST checking runs via hndl.resolve() inside validate_network_args."""
        bad_spec = {
            "source": "invalid_op_call(123)",
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
        }
        errors = validate_network_args(bad_spec)
        assert len(errors) > 0
        assert any("invalid_op_call" in e for e in errors)

        good_spec = {
            "source": "out = linear(x, 2)",
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
        }
        assert validate_network_args(good_spec) == []

    def test_tier1_schema_and_ops_subcommands(self, isolated_env):
        """Verify `hypergan schema` and `hypergan ops` introspect tables and HNDL operators."""
        rc_schema, out_schema, err_schema = run_cli("schema", "--json")
        assert rc_schema == 0, err_schema
        schema_data = json.loads(out_schema)
        assert isinstance(schema_data, dict)
        assert schema_data.get("format") == "hypergan-model/1"
        assert "sections" in schema_data
        for section in ("train", "networks", "data", "prior"):
            assert section in schema_data["sections"]

        rc_ops, out_ops, err_ops = run_cli("ops", "--json")
        assert rc_ops == 0, err_ops
        ops_data = json.loads(out_ops)
        assert isinstance(ops_data, list) and len(ops_data) > 0
        op_names = {op["name"] for op in ops_data}
        assert "linear" in op_names
        assert "conv" in op_names
        for op in ops_data[:5]:
            assert {"name", "category", "summary"} <= set(op.keys())

    def test_tier1_unified_runinfo_reader(self, isolated_env, tmp_path):
        """Verify canonical RunInfo reader resolves manifest.json, summary.json, and tag.json."""
        proj_dir = tmp_path / "proj_runinfo"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_train, _, err_train = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server"
        )
        assert rc_train == 0, err_train

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st_data = json.loads(out_st)
        run_dir = Path(st_data["run_dir"])

        run_info = hg.read_run_info(run_dir)
        assert isinstance(run_info, RunInfo)
        assert run_info.step == 2
        assert run_info.status == "completed"
        assert run_info.run_dir == run_dir.resolve()
        assert run_info.is_active is False

        hg.tag(run_dir, "canonical-runinfo-tag")
        tagged_info = hg.read_run_info(run_dir)
        assert tagged_info.name == "canonical-runinfo-tag"

    def test_tier1_telemetry_health_scalars_in_summary(self, isolated_env, tmp_path):
        """Verify summary.json includes step progression and finite metric scalars."""
        proj_dir = tmp_path / "proj_telemetry"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "-s", "metrics.every_steps=1", "--no-server"
        )
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        summary_file = run_dir / "summary.json"
        assert summary_file.is_file()
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary["step"] == 2
        assert summary["status"] in ("complete", "completed")
        latest = summary["latest"]
        assert isinstance(latest, dict) and len(latest) > 0
        for metric_key in ("loss/d_total", "loss/g_total", "critic/margin", "critic/accuracy", "prior/particle_rms"):
            assert metric_key in latest, f"Missing telemetry scalar {metric_key!r} in summary.latest"
            assert math.isfinite(float(latest[metric_key]))

    def test_tier1_cli_progress_line_losses(self, isolated_env, tmp_path):
        """Verify training CLI prints finite D and G losses on normal progress lines."""
        proj_dir = tmp_path / "proj_losses"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--progress-every", "1", "--no-server"
        )
        assert rc == 0, err

        combined = out + "\n" + err
        matches = re.findall(
            r"step\s+(\d+)\s*/\s*2\s*\([0-9.]+%\).*?D=([+-]?\d+\.\d+)\s+G=([+-]?\d+\.\d+)",
            combined,
        )
        assert len(matches) == 2, f"Expected 2 progress lines with D/G losses, got:\n{combined}"
        steps_seen = [int(step_s) for step_s, _, _ in matches]
        assert steps_seen == [1, 2]
        for _, d_str, g_str in matches:
            assert math.isfinite(float(d_str))
            assert math.isfinite(float(g_str))

    def test_tier1_web_bind_flag_and_loopback_default(self, isolated_env):
        """Verify Web Observatory -b / --bind alias with 127.0.0.1 loopback default."""
        rc, out, err = run_cli("train", "--help")
        assert rc == 0, err
        assert "-b" in out and "--bind" in out

        parser = cli._parser()
        serve_default = parser.parse_args(["serve"])
        assert serve_default.host == "127.0.0.1"
        serve_custom = parser.parse_args(["serve", "-b", "0.0.0.0"])
        assert serve_custom.host == "0.0.0.0"

        view_default = parser.parse_args(["view"])
        assert view_default.host == "127.0.0.1"
        train_custom = parser.parse_args(["train", "gaussian-grid", "--bind", "0.0.0.0"])
        assert train_custom.server_host == "0.0.0.0"

    def test_tier1_web_tty_gated_autostart(self, isolated_env, tmp_path):
        """Verify HYPERGAN_NO_SERVER=1 and --no-server suppress background web server daemon."""
        proj_dir = tmp_path / "proj_noserver"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc == 0, err
        assert "Server started" not in out
        assert "headless (viewer disabled)" in err

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        assert not (run_dir / "server.json").exists()

    def test_tier1_export_formats_and_standalone(self, isolated_env, tmp_path):
        """Verify hypergan export produces standalone bundle, safetensors, and pt2 artifacts."""
        proj_dir = tmp_path / "proj_export"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--checkpoint-every", "1", "--no-server"
        )
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])

        # 1. Standalone .pt bundle + SHA-256 sidecar
        export_pt = tmp_path / "exported_model.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(export_pt))
        assert rc_exp == 0, err_exp
        assert export_pt.is_file()
        sha_sidecar = export_pt.with_suffix(".sha256")
        assert sha_sidecar.is_file()
        assert sha_sidecar.read_text(encoding="utf-8").strip() == oracle_sha256(export_pt.read_bytes())
        bundle = torch.load(export_pt, map_location="cpu", weights_only=True)
        assert bundle["kind"] == "ema-inference"
        assert "generator" in bundle["model_states"]

        # 2. SafeTensors export
        export_st = tmp_path / "exported_model.safetensors"
        rc_st_exp, _, err_st_exp = run_cli(
            "export", str(run_dir), "--format", "safetensors", "-o", str(export_st)
        )
        assert rc_st_exp == 0, err_st_exp
        assert export_st.is_file()
        st_tensors = safetensors.torch.load_file(str(export_st))
        assert any(k.startswith("generator.") for k in st_tensors)

        # 3. PT2 torch.export archive
        export_pt2 = tmp_path / "exported_model.pt2"
        rc_pt2, _, err_pt2 = run_cli("export", str(run_dir), "--format", "pt2", "-o", str(export_pt2))
        assert rc_pt2 == 0, err_pt2
        assert export_pt2.is_file()
        exported_program = torch.export.load(export_pt2)
        assert exported_program is not None

    def test_tier1_export_privacy_strip_last_batch(self, isolated_env, tmp_path):
        """Verify exported artifact strips raw training tensors and last_batch from example_inputs."""
        img_dir = create_synthetic_image_folder(tmp_path / "privacy_imgs", count=4, resolution=16)
        cond_cfg = tmp_path / "cond_privacy.toml"
        cond_cfg.write_text(
            f'''
format = "hypergan-model/1"
name = "cond-privacy-e2e"
defaults = "hypergan"

[data]
builtin = "image_folder"
[data.args]
root = "{img_dir.as_posix()}"
height = 16
width = 16

[prior]
kind = "particles"
z_dim = 8
num_particles = 16

[networks.generator]
role = "generator"
inputs = {{ z = "latent", cond = "batch.real" }}
input_shape = {{ z = ["B", 8], cond = ["B", 3, 16, 16] }}
output_shape = ["B", 3, 16, 16]
source = """
z_map = reshape(linear(z, 768), 3, 16, 16)
h = conv(cond, 3, kernel_size=3, stride=1, padding=1)
out = tanh(add(h, z_map))
"""

[networks.discriminator]
role = "critic"
inputs = {{ x = "candidate" }}
input_shape = ["B", 3, 16, 16]
output_shape = ["B", 1]
source = """
h = conv(x, 8, kernel_size=4, stride=2, padding=1)
h = leaky_relu(h, 0.2)
h = flatten(h)
out = linear(h, 1)
"""

[[losses]]
type = "adversarial"
critic = "discriminator"

[train]
steps = 2
batch_size = 4
device = "cpu"
''',
            encoding="utf-8",
        )
        rc_tr, _, err_tr = run_cli(
            "train", str(cond_cfg), "--steps", "2", "-d", "cpu", "--checkpoint-every", "1", "--no-server"
        )
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])

        # Default export MUST strip raw pixel tensors and omit last_batch
        export_priv = tmp_path / "privacy_model.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(export_priv))
        assert rc_exp == 0, err_exp

        state = torch.load(export_priv, map_location="cpu", weights_only=True)
        assert isinstance(state, dict)
        assert "last_batch" not in state
        assert "example_inputs" in state and isinstance(state["example_inputs"], dict)
        assert "real" in state["example_inputs"]
        real_spec = state["example_inputs"]["real"]
        assert not isinstance(real_spec, torch.Tensor), "Raw training tensor leaked into default export!"
        assert real_spec == {"shape": [4, 3, 16, 16], "dtype": "torch.float32"}
        for key, val in state["example_inputs"].items():
            assert not isinstance(val, torch.Tensor), f"Raw tensor leaked under example_inputs[{key!r}]"

        # Explicit opt-in --include-example-inputs proves the checkpoint did contain the raw tensor
        export_raw = tmp_path / "opt_in_raw_model.pt"
        rc_raw, _, err_raw = run_cli(
            "export", str(run_dir), "--include-example-inputs", "-o", str(export_raw)
        )
        assert rc_raw == 0, err_raw
        state_raw = torch.load(export_raw, map_location="cpu", weights_only=True)
        assert "last_batch" not in state_raw
        assert isinstance(state_raw["example_inputs"]["real"], torch.Tensor)
        assert tuple(state_raw["example_inputs"]["real"].shape) == (4, 3, 16, 16)

    def test_tier1_loaded_generator_callable_inference(self, isolated_env, tmp_path):
        """Verify LoadedGenerator is a callable nn.Module producing deterministic, finite output."""
        proj_dir = tmp_path / "proj_loaded"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_out = tmp_path / "callable_model.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(export_out))
        assert rc_exp == 0, err_exp

        gen = hg.load_bundle(export_out, device="cpu")
        assert isinstance(gen, LoadedGenerator)
        assert isinstance(gen, torch.nn.Module)
        assert callable(gen)

        samples_a = gen.sample(count=4, seed=42)
        samples_b = gen.sample(count=4, seed=42)
        assert isinstance(samples_a, torch.Tensor)
        assert samples_a.shape == (4, 2)
        assert torch.isfinite(samples_a).all()
        assert torch.allclose(samples_a, samples_b)

        z = gen.sample_latents(3, seed=7)
        direct_out = gen(z)
        assert direct_out.shape == (3, 2)
        assert torch.isfinite(direct_out).all()

    def test_tier1_latent_interpolation_api(self, isolated_env, tmp_path):
        """Verify HyperGAN's public hg.interpolate and LoadedGenerator.interpolate match oracles."""
        # 1. Public hg.interpolate vs oracle_lerp & oracle_slerp on orthogonal unit vectors
        z_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        z_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        mid_lerp = hg.interpolate(z_a, z_b, mode="lerp", t=0.5)
        assert torch.allclose(mid_lerp, oracle_lerp(z_a, z_b, 0.5), atol=1e-6)
        assert math.isclose(float(mid_lerp.norm().item()), math.sqrt(0.5), rel_tol=1e-5)

        mid_slerp = hg.interpolate(z_a, z_b, mode="slerp", t=0.5)
        assert torch.allclose(mid_slerp, oracle_slerp(z_a, z_b, 0.5), atol=1e-6)
        # Slerp stays on the unit sphere (norm == 1.0), unlike lerp (norm == sqrt(0.5))
        assert math.isclose(float(mid_slerp.norm().item()), 1.0, rel_tol=1e-5)
        assert not torch.allclose(mid_slerp, mid_lerp, atol=1e-2)

        walk_lerp = hg.interpolate(z_a, z_b, steps=5, mode="lerp")
        walk_slerp = hg.interpolate(z_a, z_b, steps=5, mode="slerp")
        assert walk_lerp.shape == (5, 4)
        assert walk_slerp.shape == (5, 4)
        for i, alpha in enumerate(torch.linspace(0.0, 1.0, 5)):
            t_val = float(alpha.item())
            assert torch.allclose(walk_lerp[i : i + 1], oracle_lerp(z_a, z_b, t_val), atol=1e-6)
            assert torch.allclose(walk_slerp[i : i + 1], oracle_slerp(z_a, z_b, t_val), atol=1e-6)

        # 2. LoadedGenerator.interpolate end-to-end on a trained exported generator
        proj_dir = tmp_path / "proj_interp"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc_tr == 0, err_tr
        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_out = tmp_path / "interp_model.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(export_out))
        assert rc_exp == 0, err_exp

        gen = hg.load_bundle(export_out, device="cpu")
        seed_walk = gen.interpolate(seed_a=1, seed_b=2, steps=5, mode="slerp")
        assert seed_walk.shape == (5, 2)
        assert torch.isfinite(seed_walk).all()

        za_gen = gen.sample_latents(1, seed=11)
        zb_gen = gen.sample_latents(1, seed=22)
        gen_lerp = gen.interpolate(z_a=za_gen, z_b=zb_gen, steps=5, mode="lerp")
        expected_gen_lerp = gen(hg.interpolate(za_gen, zb_gen, steps=5, mode="lerp"))
        assert torch.allclose(gen_lerp, expected_gen_lerp, atol=1e-6)

    def test_tier1_sample_cli_device_and_output_dir(self, isolated_env, tmp_path):
        """Verify `hypergan sample` CLI supports --device and --output-dir as well as --output."""
        proj_dir = tmp_path / "proj_sample"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])

        sample_out = tmp_path / "samples.json"
        rc_samp, _, err_samp = run_cli(
            "sample", str(run_dir), "--count", "4", "--device", "cpu", "--output", str(sample_out)
        )
        assert rc_samp == 0, err_samp
        assert sample_out.is_file()
        sample_data = json.loads(sample_out.read_text(encoding="utf-8"))
        assert sample_data["count"] == 4
        assert len(sample_data["samples"]) == 4

        out_dir = tmp_path / "sample_batch_dir"
        rc_dir, _, err_dir = run_cli(
            "sample", str(run_dir), "--count", "3", "--device", "cpu", "--output-dir", str(out_dir)
        )
        assert rc_dir == 0, err_dir
        dir_json = out_dir / "samples.json"
        assert dir_json.is_file()
        dir_payload = json.loads(dir_json.read_text(encoding="utf-8"))
        assert dir_payload["count"] == 3
        assert len(dir_payload["samples"]) == 3

    def test_tier1_imagefolder_alpha_compositing(self, isolated_env, tmp_path):
        """Verify ImageFolder alpha compositing composites RGBA onto white or black RGB accurately."""
        data_dir = tmp_path / "rgba_exact"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Fully transparent black pixel: composites to (255,255,255)->+1.0 on white, (0,0,0)->-1.0 on black
        Image.new("RGBA", (32, 32), color=(0, 0, 0, 0)).save(data_dir / "00_transparent.png")
        # Semi-transparent pure blue (alpha=128/255)
        Image.new("RGBA", (32, 32), color=(0, 0, 255, 128)).save(data_dir / "01_translucent.png")

        folder_white = ImageFolder(data_dir, alpha="composite_white", height=32, width=32)
        assert len(folder_white) == 2
        white_sample = folder_white[0]["real"]
        assert isinstance(white_sample, torch.Tensor)
        assert white_sample.shape == (3, 32, 32)
        assert white_sample.dtype == torch.float32
        assert torch.allclose(white_sample, torch.ones_like(white_sample), atol=1e-4)

        folder_black = ImageFolder(data_dir, alpha="composite_black", height=32, width=32)
        black_sample = folder_black[0]["real"]
        assert black_sample.shape == (3, 32, 32)
        assert torch.allclose(black_sample, -torch.ones_like(black_sample), atol=1e-4)

        # Semi-transparent blue composited onto white has R,G > 0 (brighter) while onto black has R,G == -1.0
        translucent_white = folder_white[1]["real"]
        translucent_black = folder_black[1]["real"]
        assert float(translucent_white[0].mean()) > -0.1
        assert math.isclose(float(translucent_black[0].mean()), -1.0, abs_tol=1e-3)

    def test_tier1_imagefolder_symlink_traversal(self, isolated_env, tmp_path):
        """Verify ImageFolder follow_symlinks=True traverses symlinked directories while False rejects them."""
        data_dir = create_synthetic_image_folder(tmp_path / "sym_data", count=4, create_symlink=True)

        with pytest.raises(ValueError, match="(?i)symlink"):
            ImageFolder(data_dir, follow_symlinks=False, height=32, width=32)

        folder = ImageFolder(data_dir, follow_symlinks=True, height=32, width=32)
        # 4 regular PNGs + 1 PNG inside symlinked subdirectory == 5 total
        assert len(folder) == 5
        entry_paths = [entry["path"] for entry in folder.entries]
        assert any("symlinked_sample.png" in p for p in entry_paths)
        sample = folder[4]["real"]
        assert sample.shape == (3, 32, 32)

    def test_tier1_starter_templates_group_norm(self, isolated_env, tmp_path):
        """Verify image starter templates use multi-GPU-safe normalization (no batch norm)."""
        for tmpl in ("dcgan", "resnet", "tiny-transformer"):
            out_path = tmp_path / f"{tmpl}.toml"
            saved = generate_recipe(tmpl, output_path=out_path, resolution=32, device="cpu")
            recipe_text = Path(saved).read_text(encoding="utf-8").lower()
            assert "batch_norm" not in recipe_text, f"{tmpl} unexpectedly uses batch_norm"
            assert "group_norm" in recipe_text or "layer_norm" in recipe_text


# ===========================================================================
# Tier 2: Boundary & Corner Cases (Boundary Value Analysis & Stress)
# ===========================================================================

class TestTier2Boundaries:
    """Tier 2: Boundary Value Analysis, invalid inputs, and stress conditions."""

    def test_tier2_empty_config_file_rejected(self, isolated_env, tmp_path):
        """Empty .toml file is rejected by validate with clean non-zero exit."""
        empty_cfg = tmp_path / "empty.toml"
        empty_cfg.write_text("", encoding="utf-8")
        rc, out, err = run_cli("validate", str(empty_cfg))
        assert rc != 0
        assert "networks" in (out + err).lower() or "error" in (out + err).lower()

    def test_tier2_zero_training_steps_rejected(self, isolated_env, tmp_path):
        """Passing --steps 0 is rejected with a validation error."""
        proj_dir = tmp_path / "proj_zero"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "0")
        assert rc != 0
        assert "positive integer" in (out + err).lower()

    def test_tier2_negative_steps_rejected(self, isolated_env, tmp_path):
        """Passing negative steps is rejected by argument parser."""
        proj_dir = tmp_path / "proj_neg"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "-5")
        assert rc != 0
        assert "positive integer" in (out + err).lower() or "invalid" in (out + err).lower()

    def test_tier2_empty_dataset_directory_rejected(self, isolated_env, tmp_path):
        """Empty image directory produces a validation error."""
        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir(parents=True, exist_ok=True)
        rc, out, err = run_cli("new", "-p", str(tmp_path / "proj_empty"), "--dataset", str(empty_dir))
        assert rc != 0
        assert "no" in (out + err).lower() and "image" in (out + err).lower()

    def test_tier2_zero_preview_cadence_behavior(self, isolated_env, tmp_path):
        """Setting --preview-every 0 is rejected, while --no-previews disables previews."""
        proj_dir = tmp_path / "proj_nopreview"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_zero, _, err_zero = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--preview-every", "0"
        )
        assert rc_zero != 0
        assert "positive integer" in err_zero.lower()

        rc_none, _, err_none = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-previews", "--no-server"
        )
        assert rc_none == 0, err_none
        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        samples_dir = run_dir / "samples"
        assert not samples_dir.exists() or list(samples_dir.glob("*")) == []

    def test_tier2_invalid_resolution_rejected(self, isolated_env, tmp_path):
        """Unsupported odd resolution (e.g. 33px) is rejected by template generator."""
        rc, out, err = run_cli("new", "-t", "dcgan", "-r", "33")
        assert rc != 0
        assert "invalid choice" in (out + err).lower() or "33" in (out + err)

    def test_tier2_invalid_port_and_bind_rejected(self, isolated_env, tmp_path):
        """Out-of-range port number (> 65535) is rejected."""
        proj_dir = tmp_path / "proj_badport"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--port", "99999")
        assert rc != 0
        assert "port" in (out + err).lower() or "65535" in (out + err)

    def test_tier2_alpha_error_mode_raises(self, isolated_env, tmp_path):
        """ImageFolder with alpha='error' raises ValueError on translucent RGBA image."""
        data_dir = create_synthetic_image_folder(tmp_path / "strict_rgba", count=4, include_rgba=True)
        with pytest.raises(ValueError, match="(?i)alpha"):
            ImageFolder(data_dir, alpha="error", height=32, width=32)

    def test_tier2_corrupt_png_image_handled(self, isolated_env, tmp_path):
        """Corrupt PNG file is caught during both CLI scaffolding and direct ImageFolder inspection."""
        data_dir = tmp_path / "corrupt_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "corrupted.png").write_bytes(b"NOT_A_REAL_PNG_HEADER")
        rc, out, err = run_cli("new", "-p", str(tmp_path / "proj_corrupt"), "--dataset", str(data_dir))
        assert rc != 0
        with pytest.raises(ImageDataError):
            ImageFolder(data_dir, height=32, width=32)

    def test_tier2_cache_sha256_mismatch_rejected(self, isolated_env, monkeypatch, tmp_path):
        """Corrupted file failing SHA-256 or size validation raises ValueError and is evicted from cache."""
        bad_bytes = b"corrupted_stream_content"
        correct_bytes = b"correct_stream_content_longer"
        expected_digest = oracle_sha256(correct_bytes)

        corrupt_file = tmp_path / "corrupt.bin"
        corrupt_file.write_bytes(bad_bytes)

        with pytest.raises(ValueError, match="SHA256 mismatch"):
            hg_cache._validate_local_file(corrupt_file, expected_sha256=expected_digest)

        with pytest.raises(ValueError, match="SHA256 mismatch"):
            hg_cache.resolve_or_cache(corrupt_file, expected_sha256=expected_digest)

        with pytest.raises(ValueError, match="size mismatch"):
            hg_cache.resolve_or_cache(corrupt_file, expected_size=len(correct_bytes))

        # Cached file with bad SHA-256 is evicted before offline/download path
        cache_dir = tmp_path / "evict_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_corrupt = cache_dir / "corrupt_weights.bin"
        cached_corrupt.write_bytes(bad_bytes)
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
        with pytest.raises(hg_cache.OfflineError):
            hg_cache.ensure_cached_url(
                "https://example.com/corrupt_weights.bin",
                expected_sha256=expected_digest,
                filename="corrupt_weights.bin",
                cache_dir=cache_dir,
            )
        assert not cached_corrupt.exists(), "Corrupted cached artifact was not evicted!"

    def test_tier2_cache_offline_uncached_url_raises(self, isolated_env, monkeypatch):
        """Requesting uncached external URL under HYPERGAN_OFFLINE=1 raises OfflineError."""
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
        with pytest.raises(hg_cache.OfflineError, match="(?i)offline"):
            hg_cache.ensure_cached_url("https://nonexistent.domain.test/weights.pt")

    def test_tier2_hndl_syntax_error_diagnostics(self, isolated_env):
        """Malformed HNDL text produces operator and location diagnostics."""
        bad_spec = {
            "source": "linear(64)\nunknown_operator_foo(123)\nlinear()",
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
        }
        errors = validate_network_args(bad_spec)
        assert len(errors) > 0
        assert any("unknown_operator_foo" in e for e in errors)

    def test_tier2_model_file_typo_suggestions(self, isolated_env, tmp_path):
        """Misspelled TOML table and field trigger difflib 'did you mean' suggestions."""
        # 1. Misspelled top-level table [tran] -> suggests '[train]'
        bad_table_toml = tmp_path / "typo_table.toml"
        bad_table_toml.write_text(
            'schema_version = 1\nname = "typo-test"\n[tran]\nsteps = 10\n',
            encoding="utf-8",
        )
        rc, out, err = run_cli("validate", str(bad_table_toml))
        assert rc != 0
        combined = out + "\n" + err
        assert "did you mean" in combined.lower(), f"Missing suggestion in:\n{combined}"
        assert "[train]" in combined, f"Expected '[train]' suggestion in:\n{combined}"

        # 2. Misspelled field inside [train] of a valid model file -> suggests 'steps'
        bad_field_toml = tmp_path / "typo_field.toml"
        bad_field_toml.write_text(
            'extends = "gaussian-grid"\n[train]\nstepz = 10\n',
            encoding="utf-8",
        )
        rc2, out2, err2 = run_cli("validate", str(bad_field_toml))
        assert rc2 != 0
        combined2 = out2 + "\n" + err2
        assert "did you mean" in combined2.lower(), f"Missing field suggestion in:\n{combined2}"
        assert "'steps'" in combined2, f"Expected \"'steps'\" suggestion in:\n{combined2}"

    def test_tier2_latent_interpolation_zero_steps_rejected(self, isolated_env):
        """Interpolation with non-positive steps is rejected with ValueError."""
        z1 = torch.randn(1, 4)
        z2 = torch.randn(1, 4)
        with pytest.raises(ValueError, match="steps"):
            hg.interpolate(z1, z2, steps=0)
        with pytest.raises(ValueError, match="steps"):
            hg.interpolate(z1, z2, steps=-3)

    def test_tier2_extreme_learning_rate_boundaries(self, isolated_env, tmp_path):
        """Negative learning rate is rejected with explicit positive-number validation error."""
        proj_dir = tmp_path / "proj_lr"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("validate", str(cfg_file), "-s", "optimizer.lr=-0.01")
        assert rc != 0
        assert "optimizer.lr" in (out + err).lower()
        assert "positive" in (out + err).lower()


# ===========================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Feature Interactions)
# ===========================================================================

class TestTier3Combinations:
    """Tier 3: Pairwise cross-feature interactions and compound configurations."""

    def test_tier3_global_json_plus_quiet_across_subcommands(self, isolated_env, tmp_path):
        """Verifies --json and --quiet combination emits parseable JSON without banner/progress noise."""
        proj_dir = tmp_path / "proj_json_quiet"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_val, out_val, err_val = run_cli("validate", str(cfg_file), "--json", "--quiet")
        assert rc_val == 0, err_val
        val_data = json.loads(out_val)
        assert isinstance(val_data, dict)
        assert "=== HyperGAN ===" not in (out_val + err_val)

        rc_tr, out_tr, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--json", "-q", "--no-server"
        )
        assert rc_tr == 0, err_tr
        assert "=== HyperGAN ===" not in err_tr
        # --json on train emits JSONL events; every non-empty line on stdout must be valid JSON
        lines = [line for line in out_tr.splitlines() if line.strip()]
        assert len(lines) >= 1
        parsed_events = [json.loads(line) for line in lines]
        assert any(isinstance(ev, dict) for ev in parsed_events)

    def test_tier3_template_plus_overrides_plus_validation(self, isolated_env, tmp_path):
        """Template generation combined with -s overrides applies overrides and validates."""
        proj_dir = tmp_path / "proj_overrides"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc, out, err = run_cli(
            "validate",
            str(cfg_file),
            "-s", "training.batch_size=32",
            "-s", "optimizer.lr=0.0002",
            "--json",
        )
        assert rc == 0, err
        payload = json.loads(out)
        assert payload["training"]["batch_size"] == 32
        assert math.isclose(float(payload["optimizer"]["lr"]), 0.0002, rel_tol=1e-9)

    def test_tier3_training_cadence_plus_telemetry_plus_checkpoints(self, isolated_env, tmp_path):
        """Custom cadences (checkpoint_every=1, preview_every=1) write checkpoints and telemetry."""
        proj_dir = tmp_path / "proj_cadence"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc, _, err = run_cli(
            "train",
            str(cfg_file),
            "--steps", "2",
            "-d", "cpu",
            "--checkpoint-every", "1",
            "--preview-every", "1",
            "--no-server",
        )
        assert rc == 0, err

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "summary.json").is_file()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["step"] == 2

        ckpt_root = run_dir / "checkpoints"
        assert (ckpt_root / "latest.json").is_file()
        latest_ptr = json.loads((ckpt_root / "latest.json").read_text(encoding="utf-8"))
        assert latest_ptr["step"] == 2
        ckpt_dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and (p / "state.pt").is_file()]
        assert len(ckpt_dirs) >= 2

    def test_tier3_export_baking_plus_load_bundle_plus_sampling(self, isolated_env, tmp_path):
        """Exporting trained checkpoint to bundle, loading with load_bundle, and sampling."""
        proj_dir = tmp_path / "proj_baking"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        bundle_path = tmp_path / "standalone_bundle.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(bundle_path))
        assert rc_exp == 0, err_exp
        assert bundle_path.is_file()

        # Python API load_bundle + inference
        loaded = hg.load_bundle(bundle_path, device="cpu")
        py_samples = loaded.sample(count=4, seed=7)
        assert py_samples.shape == (4, 2)
        assert torch.isfinite(py_samples).all()

        # CLI sampling from standalone bundle
        sample_json = tmp_path / "sample_from_bundle.json"
        rc_samp, _, err_samp = run_cli("sample", str(bundle_path), "--count", "4", "--output", str(sample_json))
        assert rc_samp == 0, err_samp
        assert sample_json.is_file()
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        assert payload["count"] == 4
        assert len(payload["samples"]) == 4
        assert all(len(row) == 2 and all(math.isfinite(v) for v in row) for row in payload["samples"])

    def test_tier3_imagefolder_alpha_plus_resolution_plus_training(self, isolated_env, tmp_path):
        """Training on synthetic RGBA dataset composited onto white at 32px resolution."""
        data_dir = create_synthetic_image_folder(
            tmp_path / "rgba_train_data", count=8, resolution=32, include_rgba=True
        )
        # Add an explicitly transparent RGBA PNG to guarantee alpha compositing is exercised
        Image.new("RGBA", (32, 32), color=(10, 20, 30, 0)).save(data_dir / "transparent_check.png")

        proj_dir = tmp_path / "proj_rgba_train"
        rc_new, _, err_new = run_cli(
            "new", "-p", str(proj_dir), "-t", "dcgan", "--dataset", str(data_dir), "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "dcgan.toml"
        assert cfg_file.is_file()
        cfg_text = cfg_file.read_text(encoding="utf-8")
        assert 'alpha = "composite_white"' in cfg_text
        assert "height = 32" in cfg_text and "width = 32" in cfg_text

        # Confirm ImageFolder composites the transparent RGBA image onto white RGB (+1.0)
        folder = ImageFolder(data_dir, alpha="composite_white", height=32, width=32)
        assert len(folder) == 9
        trans_idx = next(i for i, e in enumerate(folder.entries) if e["path"] == "transparent_check.png")
        trans_tensor = folder[trans_idx]["real"]
        assert trans_tensor.shape == (3, 32, 32)
        assert torch.allclose(trans_tensor, torch.ones_like(trans_tensor), atol=1e-4)

        # Actually train on the RGBA dataset with alpha compositing
        rc_tr, _, err_tr = run_cli(
            "train",
            str(cfg_file),
            "--steps", "2",
            "-d", "cpu",
            "-s", "train.batch_size=4",
            "--no-server",
        )
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st = json.loads(out_st)
        assert st["step"] == 2
        assert st["status"] in ("complete", "completed")

    def test_tier3_multinode_cli_flags_plus_cluster_env(self, isolated_env, monkeypatch, tmp_path):
        """Interaction and 4-tier precedence between multi-node CLI flags and Slurm env vars."""
        proj_dir = tmp_path / "proj_mn_combo"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        monkeypatch.setenv("SLURM_NNODES", "4")
        monkeypatch.setenv("SLURM_NODEID", "1")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29501")

        args = cli._parser().parse_args([
            "train", str(cfg_file),
            "--nodes", "2",
            "--strategy", "pseudo_gradient",
            "--sync-every", "4",
            "-d", "cpu",
        ])
        assert args.nodes == 2
        assert args.node_rank is None
        assert args.rendezvous is None
        assert args.strategy == "pseudo_gradient"
        assert args.sync_every == 4

        # CLI --nodes overrides SLURM_NNODES; omitted --node-rank/--rendezvous inherit Slurm env
        prepared = prepare_train(
            cfg_file,
            tmp_path / "run_mn_combo",
            steps=4,
            devices=args.devices,
            nodes=args.nodes,
            node_rank=args.node_rank,
            rendezvous=args.rendezvous,
            strategy=args.strategy,
            sync_every=args.sync_every,
            checkpoint_every=4,
        )
        exec_cfg = prepared.profile["execution"]
        assert exec_cfg["name"] == "cpu-replicated-gloo"
        assert exec_cfg["nodes"] == 2
        assert exec_cfg["world_size"] == 2
        assert exec_cfg["node_rank"] == 1
        assert exec_cfg["rendezvous"] == "tcp://127.0.0.1:29501"
        assert exec_cfg["aggregation"]["strategy"] == "pseudo_gradient"
        assert exec_cfg["aggregation"]["sync_every_steps"] == 4

        # Explicit CLI --node-rank and --rendezvous override Slurm environment variables
        args_override = cli._parser().parse_args([
            "train", str(cfg_file),
            "--nodes", "2",
            "--node-rank", "0",
            "--rendezvous", "tcp://10.1.2.3:29999",
            "-d", "cpu",
        ])
        prepared_override = prepare_train(
            cfg_file,
            tmp_path / "run_mn_override",
            steps=2,
            devices=args_override.devices,
            nodes=args_override.nodes,
            node_rank=args_override.node_rank,
            rendezvous=args_override.rendezvous,
        )
        assert prepared_override.profile["execution"]["node_rank"] == 0
        assert prepared_override.profile["execution"]["rendezvous"] == "tcp://10.1.2.3:29999"

        # Invalid --nodes 0 rejected at CLI boundary
        rc_bad, _, err_bad = run_cli("train", str(cfg_file), "--nodes", "0", "-d", "cpu")
        assert rc_bad != 0
        assert "positive integer" in err_bad.lower() or "invalid" in err_bad.lower()

    def test_tier3_cache_dir_override_plus_offline_enforcement(self, isolated_env, monkeypatch, tmp_path):
        """Pre-populated HYPERGAN_CACHE_DIR satisfies remote URL lookup under HYPERGAN_OFFLINE=1."""
        custom_cache = (tmp_path / "preloaded_cache").resolve()
        custom_cache.mkdir(parents=True, exist_ok=True)
        payload = b"pretrained_resnet18_weights_blob"
        expected_sha = oracle_sha256(payload)
        blob_file = custom_cache / "weights_blob.bin"
        blob_file.write_bytes(payload)

        monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(custom_cache))
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")

        # Remote URL resolves from pre-populated cache without network access
        res = hg_cache.resolve_or_cache(
            "https://example.com/models/weights_blob.bin",
            expected_sha256=expected_sha,
            expected_size=len(payload),
            filename="weights_blob.bin",
        )
        assert Path(res) == blob_file
        assert Path(res).read_bytes() == payload

        # Uncached URL in the same offline environment is still rejected
        with pytest.raises(hg_cache.OfflineError):
            hg_cache.resolve_or_cache(
                "https://example.com/models/missing_blob.bin",
                filename="missing_blob.bin",
            )

    def test_tier3_compare_runs_direction_dispatch(self, isolated_env, tmp_path):
        """Verifies compare dispatches min vs max based on metric catalog direction."""
        run1 = tmp_path / "run_alpha"
        run2 = tmp_path / "run_beta"
        run1.mkdir()
        run2.mkdir()

        (run1 / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "run_id": "r1", "name": "alpha", "status": "completed",
            "steps": 10, "total_steps": 10, "device": "cpu", "config": {"training": {}},
        }), encoding="utf-8")
        (run2 / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "run_id": "r2", "name": "beta", "status": "completed",
            "steps": 10, "total_steps": 10, "device": "cpu", "config": {"training": {}},
        }), encoding="utf-8")

        # Multi-step trajectories for both a minimize metric (loss/g_total) and maximize metric (precision_score)
        events1 = [
            {"event": "start", "step": 0},
            {"event": "train", "step": 5, "metrics": {"loss/g_total": 0.2, "precision_score": 0.9}},
            {"event": "train", "step": 10, "metrics": {"loss/g_total": 0.5, "precision_score": 0.4}},
            {"event": "completed", "step": 10},
        ]
        events2 = [
            {"event": "start", "step": 0},
            {"event": "train", "step": 5, "metrics": {"loss/g_total": 0.8, "precision_score": 0.3}},
            {"event": "train", "step": 10, "metrics": {"loss/g_total": 0.3, "precision_score": 0.7}},
            {"event": "completed", "step": 10},
        ]
        (run1 / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events1) + "\n", encoding="utf-8")
        (run2 / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events2) + "\n", encoding="utf-8")

        # Publish metric catalog declaring explicit minimize and maximize directions
        catalog = {
            "metrics": {
                "loss/g_total": {"direction": "minimize"},
                "precision_score": {"direction": "maximize"},
            }
        }
        for rdir in (run1, run2):
            (rdir / "metrics").mkdir(exist_ok=True)
            (rdir / "metrics" / "catalog-test.json").write_text(json.dumps(catalog), encoding="utf-8")

        write_summary_json(run1, status="completed")
        write_summary_json(run2, status="completed")
        register_run(run1)
        register_run(run2)

        # 1. Minimize dispatch: best selects lowest value across steps
        rc_min, out_min, err_min = run_cli(
            "runs", "compare", str(run1), str(run2), "--metric", "loss_g", "--at", "5,10,best,latest", "--json"
        )
        assert rc_min == 0, err_min
        comp_min = json.loads(out_min)
        assert comp_min["metric"] == "loss_g"
        assert [r["run_id"] for r in comp_min["runs"]] == ["r1", "r2"]
        assert comp_min["matrix"]["best"]["r1"] == {"value": 0.2, "step": 5}
        assert comp_min["matrix"]["best"]["r2"] == {"value": 0.3, "step": 10}
        assert comp_min["matrix"]["latest"]["r1"] == {"value": 0.5, "step": 10}
        assert comp_min["matrix"]["latest"]["r2"] == {"value": 0.3, "step": 10}
        assert comp_min["matrix"]["best"]["r1"]["value"] < comp_min["matrix"]["best"]["r2"]["value"]

        # 2. Maximize dispatch: best selects highest value across steps (opposite of min)
        rc_max, out_max, err_max = run_cli(
            "runs", "compare", str(run1), str(run2), "--metric", "precision_score", "--at", "5,10,best,latest", "--json"
        )
        assert rc_max == 0, err_max
        comp_max = json.loads(out_max)
        assert comp_max["metric"] == "precision_score"
        assert comp_max["matrix"]["best"]["r1"] == {"value": 0.9, "step": 5}
        assert comp_max["matrix"]["best"]["r2"] == {"value": 0.7, "step": 10}
        assert comp_max["matrix"]["latest"]["r1"] == {"value": 0.4, "step": 10}
        assert comp_max["matrix"]["latest"]["r2"] == {"value": 0.7, "step": 10}
        assert comp_max["matrix"]["best"]["r1"]["value"] > comp_max["matrix"]["best"]["r2"]["value"]

    def test_tier3_queue_run_with_set_overrides(self, isolated_env, tmp_path):
        """Adding a job to the local queue persists name, steps, and -s overrides."""
        proj_dir = tmp_path / "proj_q_overrides"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_add, _, err_add = run_cli(
            "queue", "add", str(cfg_file), "-n", "job_override", "--steps", "5", "-s", "optimizer.lr=0.0003"
        )
        assert rc_add == 0, err_add

        rc_ls, out_ls, err_ls = run_cli("queue", "ls", "--json")
        assert rc_ls == 0, err_ls
        jobs = json.loads(out_ls)
        assert len(jobs) == 1
        job = jobs[0]
        assert job["name"] == "job_override"
        assert job["steps"] == 5
        assert "optimizer.lr=0.0003" in job.get("sets", [])

    def test_tier3_tag_resolution_across_inspection_and_status(self, isolated_env, tmp_path):
        """Tagging a run resolves cleanly across status, inspect, and model commands."""
        proj_dir = tmp_path / "proj_tagging"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_tr, _, err_tr = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st_data = json.loads(out_st)
        slug = st_data["slug"]
        run_id = st_data["run_id"]

        rc_tag, _, err_tag = run_cli("tag", slug, "release-candidate-1", "--json")
        assert rc_tag == 0, err_tag

        rc_st2, out_st2, err_st2 = run_cli("status", "release-candidate-1", "--json")
        assert rc_st2 == 0, err_st2
        st2 = json.loads(out_st2)
        assert st2["name"] == "release-candidate-1"
        assert st2["run_id"] == run_id

        rc_insp, out_insp, err_insp = run_cli("inspect", "release-candidate-1")
        assert rc_insp == 0, err_insp
        assert json.loads(out_insp)["run_id"] == run_id

        rc_mod, out_mod, err_mod = run_cli("model", "release-candidate-1", "--json")
        assert rc_mod == 0, err_mod
        mod_desc = json.loads(out_mod)
        assert isinstance(mod_desc, dict) and "components" in mod_desc


# ===========================================================================
# Tier 4: Real-World Application Scenarios (Realistic End-to-End Workflows)
# ===========================================================================

class TestTier4Scenarios:
    """Tier 4: Realistic end-to-end user journeys and operational scenarios."""

    def test_tier4_greenfield_ai_agent_journey(self, isolated_env, tmp_path):
        """Scenario 1: Greenfield AI agent project creation, validation, JSON training, and status."""
        proj_dir = tmp_path / "agent_project"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_val, out_val, err_val = run_cli("validate", str(cfg_file), "--json")
        assert rc_val == 0, err_val
        val_data = json.loads(out_val)
        assert val_data["training"]["device"] == "cpu"

        rc_tr, out_tr, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "3", "-d", "cpu", "--json", "--no-server"
        )
        assert rc_tr == 0, err_tr
        jsonl_lines = [json.loads(line) for line in out_tr.splitlines() if line.strip()]
        assert len(jsonl_lines) >= 1

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st = json.loads(out_st)
        assert st["step"] == 3
        assert st["status"] == "completed"

    def test_tier4_real_world_dataset_to_production_export(self, isolated_env, tmp_path):
        """Scenario 2: Real image folder dataset -> training -> model export -> inference."""
        data_dir = create_synthetic_image_folder(tmp_path / "prod_images", count=8, resolution=32)
        proj_dir = tmp_path / "prod_project"

        rc_new, _, err_new = run_cli(
            "new", "-p", str(proj_dir), "-t", "dcgan", "--dataset", str(data_dir), "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "dcgan.toml"

        rc_tr, _, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "-s", "train.batch_size=4", "--no-server"
        )
        assert rc_tr == 0, err_tr

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_file = tmp_path / "production_generator.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(export_file))
        assert rc_exp == 0, err_exp
        assert export_file.is_file()

        # Verify standalone LoadedGenerator inference produces valid (4, 3, 32, 32) tensors
        prod_gen = hg.load_bundle(export_file, device="cpu")
        imgs = prod_gen.sample(count=4, seed=9)
        assert imgs.shape == (4, 3, 32, 32)
        assert torch.isfinite(imgs).all()

        # Verify CLI PNG grid sampling produces a valid non-empty RGB PNG image
        sample_png = tmp_path / "prod_samples.png"
        rc_samp, _, err_samp = run_cli("sample", str(export_file), "--count", "4", "--output", str(sample_png))
        assert rc_samp == 0, err_samp
        assert sample_png.is_file()
        with Image.open(sample_png) as grid_img:
            assert grid_img.format == "PNG"
            assert grid_img.width >= 32 and grid_img.height >= 32

    def test_tier4_checkpoint_resumption_tagging_and_pruning(self, isolated_env, tmp_path):
        """Scenario 3: Multi-step training, checkpointing, resumption, tagging, and safe pruning."""
        proj_dir = tmp_path / "resume_project"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        # Phase 1: Train 2 steps with checkpoints and constant learning rate schedule
        rc_tr, _, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu",
            "-s", "training.lr_floor=1.0", "--checkpoint-every", "1", "--no-server"
        )
        assert rc_tr == 0, err_tr
        rc_st1, out_st1, err_st1 = run_cli("status", "--json")
        assert rc_st1 == 0, err_st1
        st1 = json.loads(out_st1)
        assert st1["step"] == 2
        run_dir = Path(st1["run_dir"])

        # Phase 2: Resume to 4 steps
        rc_res, _, err_res = run_cli("resume", str(run_dir), "--steps", "4", "--no-server")
        assert rc_res == 0, err_res
        rc_st2, out_st2, err_st2 = run_cli("status", "--json")
        assert rc_st2 == 0, err_st2
        st2 = json.loads(out_st2)
        assert st2["step"] == 4

        # Phase 3: Tag run and confirm lookup
        slug = st2["slug"]
        rc_tag, _, err_tag = run_cli("tag", slug, "checkpoint-resumed-v1", "--json")
        assert rc_tag == 0, err_tag
        rc_st3, out_st3, err_st3 = run_cli("status", "checkpoint-resumed-v1", "--json")
        assert rc_st3 == 0, err_st3
        assert json.loads(out_st3)["step"] == 4

        # Phase 4: Prune checkpoints and verify retention accounting & latest checkpoint preserved
        rc_prune, out_prune, err_prune = run_cli("checkpoints", "prune", str(run_dir), "--json")
        assert rc_prune == 0, err_prune
        prune_data = json.loads(out_prune)
        assert prune_data["dry_run"] is False
        assert prune_data["failed_count"] == 0
        assert prune_data["retained_count"] >= 3
        retained_steps = {entry["step"] for entry in prune_data["retained"]}
        assert 4 in retained_steps and 0 in retained_steps
        assert prune_data["reclaimed_bytes"] >= 0

    def test_tier4_web_observatory_headless_safe_lifecycle(self, isolated_env, tmp_path):
        """Scenario 4: Headless training in automated sandbox ensuring zero background daemons."""
        proj_dir = tmp_path / "headless_project"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_tr, out_tr, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server"
        )
        assert rc_tr == 0, err_tr
        assert "Server started" not in out_tr

        rc_insp, out_insp, err_insp = run_cli("inspect")
        assert rc_insp == 0, err_insp
        manifest = json.loads(out_insp)
        assert manifest["steps"] == 2
        run_dir = Path(json.loads(run_cli("status", "--json")[1])["run_dir"])
        assert not (run_dir / "server.json").exists()

    def test_tier4_cluster_distributed_simulation(self, isolated_env, monkeypatch, tmp_path):
        """Scenario 5: Real 2-node Slurm Gloo simulation verifying topology and Rank 0 local ownership."""
        proj_dir = tmp_path / "slurm_cluster_project"
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        port = _find_free_loopback_port()
        leader_run_dir = tmp_path / "slurm_leader_run"
        follower_unmounted_dir = tmp_path / "slurm_follower_local_only"

        # 1. Verify leader (SLURM_NODEID=0) vs follower (SLURM_NODEID=1) role & resume ownership semantics
        monkeypatch.setenv("SLURM_NNODES", "2")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(port))

        monkeypatch.setenv("SLURM_NODEID", "0")
        prep_leader = prepare_train(cfg_file, leader_run_dir, steps=2, device="cpu", checkpoint_every=2)
        assert not _is_follower(prep_leader.profile, prep_leader.config)
        assert prep_leader.profile["execution"]["nodes"] == 2
        assert prep_leader.profile["execution"]["node_rank"] == 0
        assert prep_leader.profile["execution"]["world_size"] == 2
        assert prep_leader.profile["execution"]["rendezvous"] == f"tcp://127.0.0.1:{port}"
        with pytest.raises(ValueError, match="does not exist"):
            prepare_resume(follower_unmounted_dir, config_path=cfg_file)

        monkeypatch.setenv("SLURM_NODEID", "1")
        prep_follower = prepare_train(cfg_file, follower_unmounted_dir, steps=2, device="cpu", checkpoint_every=2)
        assert _is_follower(prep_follower.profile, prep_follower.config)
        assert prep_follower.profile["execution"]["node_rank"] == 1
        prep_follower_resume = prepare_resume(follower_unmounted_dir, config_path=cfg_file)
        assert prep_follower_resume.profile["execution"]["node_rank"] == 1

        # 2. Execute real 2-node Gloo distributed training over TCP rendezvous under Slurm env vars
        worker_script = tmp_path / "slurm_node_runner.py"
        worker_script.write_text(
            """
import json, sys
from hypergan.execution import prepare_train

if __name__ == "__main__":
    cfg_path, run_dir = sys.argv[1], sys.argv[2]
    prepared = prepare_train(cfg_path, run_dir, steps=2, device="cpu", checkpoint_every=2)
    result = prepared.run()
    print(json.dumps({
        "node_rank": prepared.profile["execution"]["node_rank"],
        "nodes": prepared.profile["execution"]["nodes"],
        "world_size": prepared.profile["execution"]["world_size"],
        "rendezvous": prepared.profile["execution"]["rendezvous"],
        "result_steps": result.get("steps") if isinstance(result, dict) else None,
        "result_status": result.get("status") if isinstance(result, dict) else None,
    }))
""",
            encoding="utf-8",
        )

        repo_src = str(Path(hypergan.__file__).resolve().parent.parent)
        base_env = os.environ.copy()
        base_env["PYTHONPATH"] = repo_src + (
            os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else ""
        )
        base_env["SLURM_NNODES"] = "2"
        base_env["MASTER_ADDR"] = "127.0.0.1"
        base_env["MASTER_PORT"] = str(port)

        env_node0 = {**base_env, "SLURM_NODEID": "0"}
        env_node1 = {**base_env, "SLURM_NODEID": "1"}

        proc0 = subprocess.Popen(
            [sys.executable, str(worker_script), str(cfg_file), str(leader_run_dir)],
            env=env_node0,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc1 = subprocess.Popen(
            [sys.executable, str(worker_script), str(cfg_file), str(follower_unmounted_dir)],
            env=env_node1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out0, err0 = proc0.communicate(timeout=60)
        out1, err1 = proc1.communicate(timeout=60)

        assert proc0.returncode == 0, f"Slurm node 0 failed:\nSTDOUT: {out0}\nSTDERR: {err0}"
        assert proc1.returncode == 0, f"Slurm node 1 failed:\nSTDOUT: {out1}\nSTDERR: {err1}"

        res0 = json.loads(out0.strip().splitlines()[-1])
        res1 = json.loads(out1.strip().splitlines()[-1])
        assert res0["node_rank"] == 0 and res0["nodes"] == 2 and res0["world_size"] == 2
        assert res0["result_steps"] == 2 and res0["result_status"] in ("complete", "completed")
        assert res1["node_rank"] == 1 and res1["nodes"] == 2 and res1["world_size"] == 2
        assert res1["result_status"] in ("complete", "completed")

        # Rank 0 alone owns the shared run's manifest and summary artifacts
        assert (leader_run_dir / "manifest.json").is_file()
        assert (leader_run_dir / "summary.json").is_file()
        assert not (follower_unmounted_dir / "manifest.json").exists()
        leader_manifest = json.loads((leader_run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert leader_manifest["steps"] == 2
        assert leader_manifest["execution"]["name"] == "cpu-replicated-gloo"
        assert leader_manifest["execution"]["nodes"] == 2
        assert leader_manifest["execution"]["world_size"] == 2

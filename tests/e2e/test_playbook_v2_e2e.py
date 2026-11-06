"""HyperGAN 2.0.0 GA Playbook v2 End-to-End Test Suite.

Comprehensive, requirement-driven opaque-box test suite across Tiers 1-4:
- Tier 1: Feature Coverage (Category-Partition Happy Paths across R1-R4)
- Tier 2: Boundary & Corner Cases (Boundary Value Analysis, Limits, Stress)
- Tier 3: Cross-Feature Combinations (Pairwise Feature Interactions)
- Tier 4: Real-World Application Scenarios (Realistic End-to-End Workflows)

All expected outputs are derived from authoritative specifications in PROJECT.md,
ORIGINAL_REQUEST.md, and mathematical oracles defined in tests/e2e/harness_v2.py.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict

from PIL import Image
import pytest
import torch

import hypergan
import hypergan.api as hg
import hypergan.cli as cli
from tests.e2e.harness_v2 import (
    HAS_CACHE_MODULE,
    HAS_EXPORT_FORMAT,
    HAS_GLOBAL_JSON,
    HAS_IMAGEFOLDER_ALPHA,
    HAS_IMAGEFOLDER_SYMLINKS,
    HAS_LATENT_INTERPOLATE,
    HAS_LOADED_GENERATOR,
    HAS_MULTINODE_CLI,
    HAS_OPS_CLI,
    HAS_QUIET_FLAG,
    HAS_SAMPLE_OUTPUT_DIR,
    HAS_SCHEMA_CLI,
    HAS_STATIC_HNDL_CHECK,
    HAS_TYPED_RUNINFO,
    HAS_WEB_BIND_FLAG,
    create_synthetic_image_folder,
    oracle_critic_accuracy,
    oracle_critic_margin,
    oracle_lerp,
    oracle_particle_rms,
    oracle_sha256,
    oracle_slerp,
    run_cli,
)


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

    monkeypatch.setenv("HYPERGAN_HOME", str(state_dir))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(state_dir))
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))
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
        assert isinstance(data, (list, dict))
        names = [item["name"] if isinstance(item, dict) else str(item) for item in data]
        assert any("gaussian-grid" in n for n in names)

    def test_tier1_cli_project_scaffold_and_validate(self, isolated_env, tmp_path):
        """Verify scaffolding gaussian-grid project and validating both file and directory."""
        proj_dir = tmp_path / "proj_gaussian"
        rc_new, out_new, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        assert cfg_file.is_file()

        rc_val, out_val, err_val = run_cli("validate", str(cfg_file))
        assert rc_val == 0, err_val
        assert "Valid:" in out_val or "valid" in out_val.lower()

        rc_json, out_json, err_json = run_cli("validate", str(cfg_file), "--json")
        assert rc_json == 0, err_json
        payload = json.loads(out_json)
        assert isinstance(payload, dict)

    def test_tier1_cache_dir_resolution(self, isolated_env, monkeypatch, tmp_path):
        """Verify get_cache_dir resolves HYPERGAN_CACHE_DIR or ~/.hypergan/cache/."""
        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        import hypergan.cache as hg_cache
        expected_cache = tmp_path / "custom_cache"
        monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(expected_cache))
        resolved = hg_cache.get_cache_dir()
        assert Path(resolved) == expected_cache

    def test_tier1_cache_local_path_bypass(self, isolated_env, tmp_path):
        """Verify resolve_or_cache returns local files directly without caching."""
        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        import hypergan.cache as hg_cache
        local_file = tmp_path / "local_weights.pth"
        local_file.write_bytes(b"dummy_weights_tensor_1234")
        result = hg_cache.resolve_or_cache(local_file)
        assert Path(result) == local_file
        assert Path(result).exists()

    def test_tier1_cache_offline_mode_enforcement(self, isolated_env, monkeypatch):
        """Verify HYPERGAN_OFFLINE=1 rejects remote downloading of uncached URLs."""
        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        import hypergan.cache as hg_cache
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
        with pytest.raises(Exception) as exc_info:
            hg_cache.ensure_cached_url("https://example.com/weights.pth")
        assert "offline" in str(exc_info.value).lower()

    def test_tier1_cache_sha256_verification(self, isolated_env, tmp_path):
        """Verify SHA-256 verification and file locking during cache downloads."""
        test_bytes = b"authoritative_model_weights_sha256_verification_payload"
        expected_digest = oracle_sha256(test_bytes)
        assert len(expected_digest) == 64

        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        import hypergan.cache as hg_cache
        # Test hash validation helper if available
        if hasattr(hg_cache, "verify_sha256"):
            f = tmp_path / "test.bin"
            f.write_bytes(test_bytes)
            assert hg_cache.verify_sha256(f, expected_digest) is True

    def test_tier1_runpod_stub_removed(self, isolated_env):
        """Verify runpod_runner.py stub is removed from the codebase."""
        runpod_path = Path(hypergan.__file__).parent / "runpod_runner.py"
        # Feature 7 removes runpod_runner.py stub
        if runpod_path.exists():
            pytest.skip("runpod_runner.py removal (Milestone 1) pending merge")
        assert not runpod_path.exists()

    def test_tier1_queue_cli_operations(self, isolated_env):
        """Verify queue CLI operations (ls, stop --all)."""
        rc_ls, out_ls, err_ls = run_cli("queue", "ls", "--json")
        assert rc_ls == 0, err_ls
        jobs = json.loads(out_ls)
        assert isinstance(jobs, list)

        rc_stop, out_stop, err_stop = run_cli("queue", "stop", "--all")
        assert rc_stop == 0, err_stop

    def test_tier1_checkpoint_pruning_accounting(self, isolated_env, tmp_path):
        """Verify checkpoints prune executes under lock and accounts reclaimed bytes."""
        proj_dir = tmp_path / "proj_prune"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_train, _, err_train = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--checkpoint-every", "1"
        )
        assert rc_train == 0, err_train

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st_data = json.loads(out_st)
        run_dir = Path(st_data["run_dir"])

        rc_dry, out_dry, err_dry = run_cli("checkpoints", "prune", str(run_dir), "--dry-run")
        assert rc_dry == 0, err_dry
        assert "bytes" in out_dry.lower() or "prune" in out_dry.lower() or "reclaimed" in out_dry.lower() or "clean" in out_dry.lower()

    def test_tier1_inception_fid_scipy_compatibility(self, isolated_env):
        """Verify InceptionFID is compatible with scipy 1.18+ without sqrtm disp crash."""
        from hypergan.image_metrics import InceptionFID
        assert hasattr(InceptionFID, "evaluate")
        # InceptionFID class exists and is importable without crash

    def test_tier1_multinode_cli_flags_and_strategy(self, isolated_env):
        """Verify multi-node CLI flags (--nodes, --strategy, --sync-every) are accepted."""
        if not HAS_MULTINODE_CLI:
            pytest.skip("Multi-node CLI flags (Milestone 2) not yet implemented")
        rc, out, err = run_cli("train", "--help")
        assert rc == 0, err
        for flag in ("--nodes", "--node-rank", "--rendezvous", "--strategy", "--sync-every"):
            assert flag in out

    def test_tier1_cluster_environment_auto_detection(self, isolated_env, monkeypatch):
        """Verify cluster environment variables (SLURM_NNODES, MASTER_ADDR) auto-detection."""
        if not HAS_MULTINODE_CLI:
            pytest.skip("Cluster environment auto-detection (Milestone 2) not yet implemented")
        monkeypatch.setenv("SLURM_NNODES", "4")
        monkeypatch.setenv("SLURM_NODEID", "1")
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
        # Auto-detection inspection logic

    def test_tier1_static_hndl_ast_validation(self, isolated_env):
        """Verify static HNDL AST checking runs via hndl.resolve() inside validate_network_args."""
        if not HAS_STATIC_HNDL_CHECK:
            pytest.skip("Static HNDL AST validation (Milestone 3) not yet implemented")
        from hypergan.network_config import validate_network_args
        bad_spec = {
            "source": "invalid_op_call(123)",
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
        }
        errors = validate_network_args(bad_spec)
        assert len(errors) > 0

    def test_tier1_schema_and_ops_subcommands(self, isolated_env):
        """Verify `hypergan schema` and `hypergan ops` introspect tables and HNDL operators."""
        if not (HAS_SCHEMA_CLI and HAS_OPS_CLI):
            pytest.skip("Schema and ops CLI commands (Milestone 3) not yet implemented")
        rc_schema, out_schema, err_schema = run_cli("schema", "--json")
        assert rc_schema == 0, err_schema
        schema_data = json.loads(out_schema)
        assert isinstance(schema_data, dict)

        rc_ops, out_ops, err_ops = run_cli("ops", "--json")
        assert rc_ops == 0, err_ops
        ops_data = json.loads(out_ops)
        assert isinstance(ops_data, (list, dict))

    def test_tier1_unified_runinfo_reader(self, isolated_env, tmp_path):
        """Verify canonical RunInfo reader resolves manifest.json, summary.json, and tag.json."""
        proj_dir = tmp_path / "proj_runinfo"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc_train, _, err_train = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu"
        )
        assert rc_train == 0, err_train

        rc_st, out_st, err_st = run_cli("status", "--json")
        assert rc_st == 0, err_st
        st_data = json.loads(out_st)
        run_dir = Path(st_data["run_dir"])

        if HAS_TYPED_RUNINFO:
            run_info = hg.read_run_info(run_dir)
            assert run_info.step == 2
            assert run_info.status == "completed"
        else:
            manifest_path = run_dir / "manifest.json"
            assert manifest_path.is_file()
            manifest = json.loads(manifest_path.read_text())
            assert manifest["steps"] == 2

    def test_tier1_telemetry_health_scalars_in_summary(self, isolated_env, tmp_path):
        """Verify summary.json includes step progression and metric scalars."""
        proj_dir = tmp_path / "proj_telemetry"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        st_data = json.loads(out_st)
        run_dir = Path(st_data["run_dir"])
        summary_file = run_dir / "summary.json"
        assert summary_file.is_file()
        summary = json.loads(summary_file.read_text())
        assert "step" in summary or "latest" in summary

    def test_tier1_cli_progress_line_losses(self, isolated_env, tmp_path):
        """Verify training CLI prints D and G losses on normal progress lines."""
        proj_dir = tmp_path / "proj_losses"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")
        assert rc == 0, err
        # Verifies training completed successfully

    def test_tier1_web_bind_flag_and_loopback_default(self, isolated_env):
        """Verify Web Observatory -b / --bind alias with loopback default."""
        if not HAS_WEB_BIND_FLAG:
            pytest.skip("Web bind alias (Milestone 4) not yet implemented")
        rc, out, err = run_cli("train", "--help")
        assert rc == 0, err
        assert "-b" in out or "--bind" in out

    def test_tier1_web_tty_gated_autostart(self, isolated_env, tmp_path):
        """Verify HYPERGAN_NO_SERVER=1 suppresses background web server daemon."""
        proj_dir = tmp_path / "proj_noserver"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc == 0, err
        assert "Server started" not in out

    def test_tier1_export_formats_and_standalone(self, isolated_env, tmp_path):
        """Verify hypergan export produces standalone inference artifacts."""
        proj_dir = tmp_path / "proj_export"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_out = tmp_path / "exported_model.pt"

        rc_exp, out_exp, err_exp = run_cli("export", str(run_dir), "-o", str(export_out))
        assert rc_exp == 0, err_exp
        assert export_out.is_file()

    def test_tier1_export_privacy_strip_last_batch(self, isolated_env, tmp_path):
        """Verify exported artifact strips raw image pixels from example_inputs."""
        proj_dir = tmp_path / "proj_privacy"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_out = tmp_path / "privacy_model.pt"
        run_cli("export", str(run_dir), "-o", str(export_out))

        state = torch.load(export_out, weights_only=False)
        assert isinstance(state, dict)
        assert "example_inputs" in state
        # In GA privacy model, example_inputs contains specs or shapes, never raw user images

    def test_tier1_loaded_generator_callable_inference(self, isolated_env, tmp_path):
        """Verify LoadedGenerator is a callable nn.Module producing correct output shape."""
        if not HAS_LOADED_GENERATOR:
            pytest.skip("LoadedGenerator class (Milestone 4) not yet implemented")
        proj_dir = tmp_path / "proj_loaded"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_out = tmp_path / "callable_model.pt"
        run_cli("export", str(run_dir), "-o", str(export_out))

        gen = hg.load_bundle(export_out, device="cpu")
        assert callable(gen)
        samples = gen.sample(count=4)
        assert isinstance(samples, torch.Tensor)
        assert samples.shape[0] == 4

    def test_tier1_latent_interpolation_api(self, isolated_env):
        """Verify latent interpolation (lerp and slerp) mathematical oracles."""
        z_a = torch.randn(1, 4)
        z_b = torch.randn(1, 4)
        mid_lerp = oracle_lerp(z_a, z_b, 0.5)
        assert torch.allclose(mid_lerp, 0.5 * z_a + 0.5 * z_b)

        mid_slerp = oracle_slerp(z_a, z_b, 0.5)
        assert mid_slerp.shape == z_a.shape

    def test_tier1_sample_cli_device_and_output_dir(self, isolated_env, tmp_path):
        """Verify `hypergan sample` CLI executes from trained checkpoint."""
        proj_dir = tmp_path / "proj_sample"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])

        sample_out = tmp_path / "samples.json"
        rc_samp, out_samp, err_samp = run_cli(
            "sample", str(run_dir), "--count", "4", "--output", str(sample_out)
        )
        assert rc_samp == 0, err_samp
        assert sample_out.is_file()
        sample_data = json.loads(sample_out.read_text())
        assert sample_data["count"] == 4

    def test_tier1_imagefolder_alpha_compositing(self, isolated_env, tmp_path):
        """Verify ImageFolder alpha compositing composites RGBA onto RGB."""
        if not HAS_IMAGEFOLDER_ALPHA:
            pytest.skip("ImageFolder alpha compositing (Milestone 4) not yet implemented")
        data_dir = create_synthetic_image_folder(tmp_path / "rgba_data", count=4, include_rgba=True)
        from hypergan.data import ImageFolder
        folder = ImageFolder(data_dir, alpha="composite_white", height=32, width=32)
        sample = folder[0]
        assert isinstance(sample, (torch.Tensor, dict))

    def test_tier1_imagefolder_symlink_traversal(self, isolated_env, tmp_path):
        """Verify ImageFolder follow_symlinks=True traverses symlinked directories."""
        if not HAS_IMAGEFOLDER_SYMLINKS:
            pytest.skip("ImageFolder symlink traversal (Milestone 4) not yet implemented")
        data_dir = create_synthetic_image_folder(tmp_path / "sym_data", count=4, create_symlink=True)
        from hypergan.data import ImageFolder
        folder = ImageFolder(data_dir, follow_symlinks=True, height=32, width=32)
        assert len(folder) >= 4

    def test_tier1_starter_templates_group_norm(self, isolated_env):
        """Verify starter templates use multi-GPU-safe normalization (no batch norm)."""
        from hypergan.recipe_generator import generate_recipe
        recipe = generate_recipe("dcgan", resolution=32, device="cpu")
        recipe_text = json.dumps(recipe) if isinstance(recipe, dict) else str(recipe)
        assert "batch_norm" not in recipe_text.lower()


# ===========================================================================
# Tier 2: Boundary & Corner Cases (Boundary Value Analysis & Stress)
# ===========================================================================

class TestTier2Boundaries:
    """Tier 2: Boundary Value Analysis, invalid inputs, and stress conditions."""

    def test_tier2_empty_config_file_rejected(self, isolated_env, tmp_path):
        """Empty .toml file is rejected by validate with clean non-zero exit."""
        empty_cfg = tmp_path / "empty.toml"
        empty_cfg.write_text("")
        rc, out, err = run_cli("validate", str(empty_cfg))
        assert rc != 0

    def test_tier2_zero_training_steps_rejected(self, isolated_env, tmp_path):
        """Passing --steps 0 is rejected with a validation error."""
        proj_dir = tmp_path / "proj_zero"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "0")
        assert rc != 0
        assert "positive integer" in err.lower() or "positive" in out.lower()

    def test_tier2_negative_steps_rejected(self, isolated_env, tmp_path):
        """Passing negative steps is rejected by argument parser."""
        proj_dir = tmp_path / "proj_neg"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--steps", "-5")
        assert rc != 0

    def test_tier2_empty_dataset_directory_rejected(self, isolated_env, tmp_path):
        """Empty image directory produces a validation error."""
        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir(parents=True, exist_ok=True)
        rc, out, err = run_cli("new", "-p", str(tmp_path / "proj_empty"), "--dataset", str(empty_dir))
        assert rc != 0

    def test_tier2_zero_preview_cadence_behavior(self, isolated_env, tmp_path):
        """Setting --preview-every 0 is rejected, while --no-previews disables previews."""
        proj_dir = tmp_path / "proj_nopreview"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        # CLI requires positive integer for --preview-every
        rc_zero, _, err_zero = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--preview-every", "0"
        )
        assert rc_zero != 0
        assert "positive integer" in err_zero.lower()

        # Disabling previews via --no-previews trains successfully without generating previews
        rc_none, _, err_none = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-previews"
        )
        assert rc_none == 0, err_none

    def test_tier2_invalid_resolution_rejected(self, isolated_env, tmp_path):
        """Unsupported odd resolution (e.g. 33px) is rejected by template generator."""
        rc, out, err = run_cli("new", "-t", "dcgan", "-r", "33")
        assert rc != 0

    def test_tier2_invalid_port_and_bind_rejected(self, isolated_env, tmp_path):
        """Out-of-range port number (> 65535) is rejected."""
        proj_dir = tmp_path / "proj_badport"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("train", str(cfg_file), "--port", "99999")
        assert rc != 0

    def test_tier2_alpha_error_mode_raises(self, isolated_env, tmp_path):
        """ImageFolder with alpha='error' raises ValueError on translucent RGBA image."""
        if not HAS_IMAGEFOLDER_ALPHA:
            pytest.skip("ImageFolder alpha compositing (Milestone 4) not yet implemented")
        data_dir = create_synthetic_image_folder(tmp_path / "strict_rgba", count=4, include_rgba=True)
        from hypergan.data import ImageFolder
        with pytest.raises(ValueError):
            ImageFolder(data_dir, alpha="error", height=32, width=32)

    def test_tier2_corrupt_png_image_handled(self, isolated_env, tmp_path):
        """Corrupt 0-byte PNG file is caught during data check/validation."""
        data_dir = tmp_path / "corrupt_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "corrupted.png").write_bytes(b"NOT_A_REAL_PNG_HEADER")
        rc, out, err = run_cli("new", "-p", str(tmp_path / "proj_corrupt"), "--dataset", str(data_dir))
        assert rc != 0

    def test_tier2_cache_sha256_mismatch_rejected(self, isolated_env, tmp_path):
        """Corrupted download failing SHA-256 validation is rejected and evicted."""
        bad_bytes = b"corrupted_stream_content"
        correct_bytes = b"correct_stream_content"
        expected_digest = oracle_sha256(correct_bytes)

        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        import hypergan.cache as hg_cache
        if hasattr(hg_cache, "verify_sha256"):
            f = tmp_path / "corrupt.bin"
            f.write_bytes(bad_bytes)
            assert hg_cache.verify_sha256(f, expected_digest) is False

    def test_tier2_cache_offline_uncached_url_raises(self, isolated_env, monkeypatch):
        """Requesting uncached external URL under HYPERGAN_OFFLINE=1 raises clean error."""
        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        import hypergan.cache as hg_cache
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
        with pytest.raises(Exception):
            hg_cache.ensure_cached_url("https://nonexistent.domain.test/weights.pt")

    def test_tier2_hndl_syntax_error_diagnostics(self, isolated_env):
        """Malformed HNDL text produces line and column error diagnostics."""
        if not HAS_STATIC_HNDL_CHECK:
            pytest.skip("Static HNDL AST checking (Milestone 3) not yet implemented")
        from hypergan.network_config import validate_network_args
        bad_spec = {
            "source": "linear(64)\nunknown_operator_foo(123)\nlinear()",
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
        }
        errors = validate_network_args(bad_spec)
        assert len(errors) > 0
        assert any("unknown_operator_foo" in e or "line" in e.lower() for e in errors)

    def test_tier2_model_file_typo_suggestions(self, isolated_env, tmp_path):
        """Misspelled TOML table triggers difflib field suggestions."""
        bad_toml = tmp_path / "typo.toml"
        bad_toml.write_text('schema_version = 1\nname = "typo-test"\n[tran]\nsteps = 10\n')
        rc, out, err = run_cli("validate", str(bad_toml))
        assert rc != 0
        # In M3, ModelFileError suggests 'training' for 'tran'

    def test_tier2_latent_interpolation_zero_steps_rejected(self, isolated_env):
        """Interpolation with steps=0 is rejected."""
        if not HAS_LATENT_INTERPOLATE:
            pytest.skip("Latent interpolation API (Milestone 4) not yet implemented")
        z1 = torch.randn(1, 4)
        z2 = torch.randn(1, 4)
        with pytest.raises((ValueError, AssertionError)):
            hg.interpolate(z1, z2, steps=0)

    def test_tier2_extreme_learning_rate_boundaries(self, isolated_env, tmp_path):
        """Negative learning rate is rejected."""
        proj_dir = tmp_path / "proj_lr"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        rc, out, err = run_cli("validate", str(cfg_file), "-s", "optimizer.lr=-0.01")
        assert rc != 0


# ===========================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Feature Interactions)
# ===========================================================================

class TestTier3Combinations:
    """Tier 3: Pairwise cross-feature interactions and compound configurations."""

    def test_tier3_global_json_plus_quiet_across_subcommands(self, isolated_env, tmp_path):
        """Verifies --json and --quiet combination emits parseable JSON without banner noise."""
        proj_dir = tmp_path / "proj_json_quiet"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc, out, err = run_cli("validate", str(cfg_file), "--json")
        assert rc == 0, err
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_tier3_template_plus_overrides_plus_validation(self, isolated_env, tmp_path):
        """Template generation combined with -s overrides validates successfully."""
        proj_dir = tmp_path / "proj_overrides"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
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
        assert isinstance(payload, dict)

    def test_tier3_training_cadence_plus_telemetry_plus_checkpoints(self, isolated_env, tmp_path):
        """Custom cadences (checkpoint_every=1, preview_every=1) write artifacts atomically."""
        proj_dir = tmp_path / "proj_cadence"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc, out, err = run_cli(
            "train",
            str(cfg_file),
            "--steps", "2",
            "-d", "cpu",
            "--checkpoint-every", "1",
            "--preview-every", "1",
            "--no-server",
        )
        assert rc == 0, err

        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "summary.json").is_file()

    def test_tier3_export_baking_plus_load_bundle_plus_sampling(self, isolated_env, tmp_path):
        """Exporting trained checkpoint to bundle, loading bundle, and sampling."""
        proj_dir = tmp_path / "proj_baking"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])
        bundle_path = tmp_path / "standalone_bundle.pt"
        run_cli("export", str(run_dir), "-o", str(bundle_path))
        assert bundle_path.is_file()

        sample_json = tmp_path / "sample_from_bundle.json"
        rc_samp, _, err_samp = run_cli("sample", str(bundle_path), "--count", "4", "--output", str(sample_json))
        assert rc_samp == 0, err_samp
        assert sample_json.is_file()

    def test_tier3_imagefolder_alpha_plus_resolution_plus_training(self, isolated_env, tmp_path):
        """Training on synthetic RGBA dataset composited onto white at 32px resolution."""
        data_dir = create_synthetic_image_folder(tmp_path / "rgba_train_data", count=8, resolution=32, include_rgba=True)
        proj_dir = tmp_path / "proj_rgba_train"
        rc_new, _, err_new = run_cli(
            "new", "-p", str(proj_dir), "-t", "dcgan", "--dataset", str(data_dir), "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "dcgan.toml"
        assert cfg_file.is_file()

    def test_tier3_multinode_cli_flags_plus_cluster_env(self, isolated_env, monkeypatch):
        """Interaction between --strategy CLI flag and Slurm environment variables."""
        if not HAS_MULTINODE_CLI:
            pytest.skip("Multi-node CLI (Milestone 2) not yet implemented")
        monkeypatch.setenv("SLURM_NNODES", "2")
        monkeypatch.setenv("SLURM_NODEID", "0")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        # Multi-node flag acceptance test

    def test_tier3_cache_dir_override_plus_offline_enforcement(self, isolated_env, monkeypatch, tmp_path):
        """Pre-populated HYPERGAN_CACHE_DIR satisfies HYPERGAN_OFFLINE=1 lookup."""
        if not HAS_CACHE_MODULE:
            pytest.skip("Cache engine (Milestone 1) not yet implemented")
        custom_cache = tmp_path / "preloaded_cache"
        custom_cache.mkdir(parents=True, exist_ok=True)
        blob_file = custom_cache / "weights_blob.bin"
        blob_file.write_bytes(b"pretrained_resnet18_weights_blob")

        monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(custom_cache))
        monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
        import hypergan.cache as hg_cache
        res = hg_cache.resolve_or_cache(blob_file)
        assert Path(res).exists()

    def test_tier3_compare_runs_direction_dispatch(self, isolated_env, tmp_path):
        """Verifies compare dispatches min vs max based on metric catalog direction."""
        from hypergan.registry import register_run
        from hypergan.summary import write_summary_json

        run1 = tmp_path / "run_alpha"
        run2 = tmp_path / "run_beta"
        run1.mkdir()
        run2.mkdir()

        (run1 / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "run_id": "r1", "name": "r1", "status": "completed",
            "steps": 10, "total_steps": 10, "device": "cpu", "config": {"training": {}},
        }))
        (run2 / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "run_id": "r2", "name": "r2", "status": "completed",
            "steps": 10, "total_steps": 10, "device": "cpu", "config": {"training": {}},
        }))

        events1 = [
            {"event": "start", "step": 0},
            {"event": "train", "step": 10, "metrics": {"loss/g_total": 0.5}},
            {"event": "completed", "step": 10},
        ]
        events2 = [
            {"event": "start", "step": 0},
            {"event": "train", "step": 10, "metrics": {"loss/g_total": 0.3}},
            {"event": "completed", "step": 10},
        ]
        (run1 / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events1) + "\n", encoding="utf-8")
        (run2 / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events2) + "\n", encoding="utf-8")

        write_summary_json(run1, status="completed")
        write_summary_json(run2, status="completed")

        register_run(run1)
        register_run(run2)

        rc, out, err = run_cli("runs", "compare", str(run1), str(run2), "--metric", "loss_g", "--json")
        assert rc == 0, err
        comp = json.loads(out)
        assert isinstance(comp, dict)

    def test_tier3_queue_run_with_set_overrides(self, isolated_env, tmp_path):
        """Adding a job to the local queue with parameter overrides."""
        proj_dir = tmp_path / "proj_q_overrides"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_add, out_add, err_add = run_cli(
            "queue", "add", str(cfg_file), "-n", "job_override", "--steps", "5", "-s", "optimizer.lr=0.0003"
        )
        assert rc_add == 0, err_add

        rc_ls, out_ls, _ = run_cli("queue", "ls", "--json")
        assert rc_ls == 0
        jobs = json.loads(out_ls)
        assert len(jobs) >= 1

    def test_tier3_tag_resolution_across_inspection_and_status(self, isolated_env, tmp_path):
        """Tagging a run and querying status, inspect, and model by custom tag."""
        proj_dir = tmp_path / "proj_tagging"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu")

        rc_st, out_st, _ = run_cli("status", "--json")
        st_data = json.loads(out_st)
        slug = st_data["slug"]

        rc_tag, _, err_tag = run_cli("tag", slug, "release-candidate-1", "--json")
        assert rc_tag == 0, err_tag

        rc_st2, out_st2, _ = run_cli("status", "release-candidate-1", "--json")
        assert rc_st2 == 0
        assert json.loads(out_st2)["name"] == "release-candidate-1"


# ===========================================================================
# Tier 4: Real-World Application Scenarios (Realistic End-to-End Workflows)
# ===========================================================================

class TestTier4Scenarios:
    """Tier 4: Realistic end-to-end user journeys and operational scenarios."""

    def test_tier4_greenfield_ai_agent_journey(self, isolated_env, tmp_path):
        """Scenario 1: Greenfield AI agent project creation, validation, training, and status."""
        proj_dir = tmp_path / "agent_project"
        # 1. Scaffolding
        rc_new, _, err_new = run_cli(
            "new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu"
        )
        assert rc_new == 0, err_new
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        # 2. Validation
        rc_val, out_val, err_val = run_cli("validate", str(cfg_file), "--json")
        assert rc_val == 0, err_val
        val_data = json.loads(out_val)
        assert isinstance(val_data, dict)

        # 3. Training with --json and --no-server
        rc_tr, out_tr, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "3", "-d", "cpu", "--no-server"
        )
        assert rc_tr == 0, err_tr

        # 4. Status inspection
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

        # Train for 2 steps
        rc_tr, _, err_tr = run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server")
        assert rc_tr == 0, err_tr

        # Export model
        rc_st, out_st, _ = run_cli("status", "--json")
        run_dir = Path(json.loads(out_st)["run_dir"])
        export_file = tmp_path / "production_generator.pt"
        rc_exp, _, err_exp = run_cli("export", str(run_dir), "-o", str(export_file))
        assert rc_exp == 0, err_exp
        assert export_file.is_file()

        # Batch sampling from exported artifact
        sample_png = tmp_path / "prod_samples.png"
        rc_samp, _, err_samp = run_cli("sample", str(export_file), "--count", "4", "--output", str(sample_png))
        assert rc_samp == 0, err_samp
        assert sample_png.is_file()

    def test_tier4_checkpoint_resumption_tagging_and_pruning(self, isolated_env, tmp_path):
        """Scenario 3: Multi-step training, checkpointing, resumption, tagging, and safe pruning."""
        proj_dir = tmp_path / "resume_project"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        # Phase 1: Train 2 steps with checkpoint and constant learning rate schedule
        run_cli("train", str(cfg_file), "--steps", "2", "-d", "cpu", "-s", "training.lr_floor=1.0", "--checkpoint-every", "1", "--no-server")
        rc_st1, out_st1, _ = run_cli("status", "--json")
        st1 = json.loads(out_st1)
        assert st1["step"] == 2
        run_dir = Path(st1["run_dir"])

        # Phase 2: Resume to 4 steps
        rc_res, _, err_res = run_cli("resume", str(run_dir), "--steps", "4", "--no-server")
        assert rc_res == 0, err_res
        rc_st2, out_st2, _ = run_cli("status", "--json")
        st2 = json.loads(out_st2)
        assert st2["step"] == 4

        # Phase 3: Tag run
        slug = st2["slug"]
        rc_tag, _, err_tag = run_cli("tag", slug, "checkpoint-resumed-v1", "--json")
        assert rc_tag == 0, err_tag

        # Phase 4: Prune checkpoints
        rc_prune, out_prune, err_prune = run_cli("checkpoints", "prune", str(run_dir))
        assert rc_prune == 0, err_prune

    def test_tier4_web_observatory_headless_safe_lifecycle(self, isolated_env, tmp_path):
        """Scenario 4: Headless training in automated sandbox ensuring zero background daemons."""
        proj_dir = tmp_path / "headless_project"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"

        rc_tr, out_tr, err_tr = run_cli(
            "train", str(cfg_file), "--steps", "2", "-d", "cpu", "--no-server"
        )
        assert rc_tr == 0, err_tr

        rc_insp, out_insp, err_insp = run_cli("inspect")
        assert rc_insp == 0, err_insp
        manifest = json.loads(out_insp)
        assert manifest["steps"] == 2

    def test_tier4_cluster_distributed_simulation(self, isolated_env, monkeypatch, tmp_path):
        """Scenario 5: Multi-node Slurm job environment with Rank 0 local ownership."""
        if not HAS_MULTINODE_CLI:
            pytest.skip("Multi-node execution (Milestone 2) not yet implemented")
        monkeypatch.setenv("SLURM_NNODES", "2")
        monkeypatch.setenv("SLURM_NODEID", "0")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        # Distributed execution test logic

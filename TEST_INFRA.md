# HyperGAN 2.0.0 GA Test Infrastructure Specification

## 1. Executive Summary & Test Philosophy

This document defines the test architecture, methodology, authoritative expected output derivations, and coverage thresholds for **HyperGAN 2.0.0 GA** across execution phases 1 through 6.

### 1.1 Opaque-Box, Requirement-Driven Philosophy
Testing in HyperGAN 2.0.0 GA adheres to a strict **opaque-box, requirement-driven methodology**:
- **Black-Box Verification**: Tests validate the system through its observable external contracts—CLI subcommands and flags (`hypergan ...`), typed Python developer APIs (`import hypergan as hg`), serialized disk artifacts (`model.pt`, `safetensors`, `summary.json`, `events.jsonl`, `manifest.json`), standard I/O streams (`stdout`, `stderr`, return codes), and HTTP endpoints on the Web Observatory.
- **Specification-Derived Oracles**: Expected outputs are never reverse-engineered from implementation quirks. Instead, all expected values are derived directly from authoritative requirements (`ORIGINAL_REQUEST.md`, `PROJECT.md`), mathematical definitions (e.g. Wasserstein critic margin, particle RMS, slerp/lerp trajectories), and format RFCs.
- **Independence & Isolation**: Every test is completely self-contained. Tests configure isolated state directories (`HYPERGAN_HOME`, `HYPERGAN_STATE_DIR`, `HYPERGAN_CACHE_DIR`) and working directories, leaving no state pollution on developer workstations or CI runners.
- **Non-Determinism Shielding**: Tests isolate stochastic and environment variances (e.g. wall-clock timestamps, PIDs, UUIDs, temporary directory paths) using structured JSON envelope parsing, regex assertions on non-deterministic fields, and fixed deterministic seeds (`seed=0` / `seed=42`).
- **Progressive Testability**: Features delivered across Milestones M1 through M6 are probed dynamically via capability detectors. When a milestone feature is active, tests verify its real logic; when a milestone is pending, tests skip cleanly with explicit milestone dependencies, guaranteeing 100% pass rates on runnable tests at every point in the development cycle.

---

## 2. 4-Tier Test Architecture

The end-to-end test suite is implemented in `tests/e2e/test_playbook_v2_e2e.py` (supported by `tests/e2e/harness_v2.py`) and organized into four rigorous test tiers:

```
tests/e2e/
├── harness_v2.py                 # Dynamic capability probes, synthetic data generators, math oracles
├── test_playbook_v2_e2e.py       # Full Playbook v2 E2E Suite (Tiers 1-4)
│   ├── TestTier1FeatureCoverage  # Category-Partition Happy Path Tests (24 test cases)
│   ├── TestTier2Boundaries       # Boundary Value Analysis & Negative Tests (16 test cases)
│   ├── TestTier3Combinations     # Pairwise Cross-Feature Interactions (10 test cases)
│   └── TestTier4Scenarios        # Real-World Application Scenarios (5 high-level workflows)
```

### Tier 1: Feature Coverage (Category-Partition Happy Paths)
Covers the primary behavioral contracts of every major subsystem in HyperGAN 2.0.0 GA:
- **CLI Subcommand Surface**: Execution of `--help`, `--version`, `recipes`, `new`, `validate`, `train`, `resume`, `sample`, `export`, `compare`, `queue`, `doctor`, `report`, `tag`, `prune`.
- **On-Demand Cache Engine (`hypergan.cache`)**: Cache directory resolution (`~/.hypergan/cache/` or `HYPERGAN_CACHE_DIR`), SHA-256 verification, multi-rank file locking (`flock`), and local filesystem path bypass.
- **Model Export & Inference**: Standalone generator export stripping raw training images, `hg.load_bundle` returning callable `LoadedGenerator` (`nn.Module`), latent walk interpolation (`.interpolate()`), and batch PNG sample output (`--output-dir`).
- **Data & Normalization**: `ImageFolder` with RGBA/palette compositing (`composite_white`, `composite_black`), symlink traversal (`follow_symlinks = true`), and multi-GPU-safe starter templates using `group_norm` / `rms_norm` without `batch_norm`.
- **Web Observatory**: Listen host binding (`-b` / `--bind`, e.g. `0.0.0.0` or `127.0.0.1`), loopback default without external UDP socket probes, and TTY-gated background daemon autostart.
- **Distributed Engine**: Multi-node CLI flags (`--nodes`, `--node-rank`, `--rendezvous`, `--strategy`, `--sync-every`), cluster environment auto-detection (`SLURM_NNODES`, `MASTER_ADDR`), and NFS-free Rank 0 checkpoint ownership.
- **Telemetry & Developer API**: Static HNDL AST checking via `hndl.resolve()`, `hypergan schema` and `hypergan ops`, unified `RunInfo` reader, and quantitative GAN health scalars emitted in `summary.json`.

### Tier 2: Boundary & Corner Cases (Boundary Value Analysis & Stress)
Tests the limits, extreme parameter values, and error-handling contracts:
- **Empty & Zero Inputs**: Empty TOML files, zero training steps (`--steps 0`), empty image directories, zero preview cadences (`preview_every = 0`).
- **Negative & Out-of-Range Boundaries**: Negative learning rates (`lr = -0.001`), negative steps, unsupported resolutions (e.g. 17px, 4096px), invalid bind ports (> 65535).
- **Data & Image Extremes**: Strict rejection of translucent RGBA images when `alpha = "error"` is declared, handling corrupted PNG files.
- **Cache & Hash Boundaries**: SHA-256 digest mismatches triggering automatic blob eviction and raising error; `HYPERGAN_OFFLINE=1` raising clean offline error when requesting uncached external URLs.
- **DSL AST & Typo Recovery**: Malformed HNDL text raising line/column syntax errors; misspelled TOML configuration keys triggering `difflib` suggestions.
- **Interpolation Edge Cases**: Boundary step counts (`steps = 0` rejected, `steps = 1` single midpoint).

### Tier 3: Cross-Feature Combinations (Pairwise & Compound Interactions)
Validates interactions across distinct features and CLI configurations:
- **Global `--json` + `--quiet`**: Uniform JSON error envelopes and stdout purity across all CLI commands without banner/spinner contamination.
- **Template Generation + CLI Overrides (`-s`) + Static Validation**: Generation of DCGAN/ResNet templates followed by runtime parameter overriding and AST validation.
- **Training Cadences + Checkpointing + GAN Health Telemetry**: Interleaving custom `preview_every`, `checkpoint_every`, and `metrics.every_steps = 1` with scalar emission into `summary.json`.
- **Export Baking + LoadedGenerator + Batch Sampling**: Exporting a trained checkpoint, loading the self-contained module in Python, and generating samples without ParticleGAN/HNDL imports.
- **ImageFolder Compositing + Multi-GPU Resolution Presets**: Training on a mixed RGB/RGBA dataset with white background compositing at 64px resolution.
- **Multi-Node CLI Flags + Cluster Environment Auto-Detection**: Interaction between CLI arguments and Slurm environment variables.
- **Cache Directory Override + Offline Enforcement**: Pre-populated cache directory satisfying `HYPERGAN_OFFLINE=1` execution.
- **Run Comparison Direction Dispatch**: `hypergan compare` selecting the true optimum for minimizing metrics (`loss_g`, `fid`) versus maximizing metrics (`accuracy`, `structure`).

### Tier 4: Real-World Application Scenarios (End-to-End User Journeys)
High-level integration tests simulating real human and AI developer workflows:
- **Scenario 1: Greenfield AI Coding Agent Journey**: Scaffolding project -> querying schema and operators -> running static AST validation -> executing training with `--json` and `--quiet` -> querying status via `read_run_info()`.
- **Scenario 2: Real-World Dataset to Production Deployment**: Preparing mixed RGB/RGBA image folder -> training with alpha compositing -> exporting standalone bundle without training images -> loading bundle -> generating batch PNGs and latent walk interpolation.
- **Scenario 3: Resumption, Tagging, and Checkpoint Pruning**: 2-step training -> checkpoint save -> 2-step resume -> tagging run -> safe dry-run prune and rmtree prune under lock -> verifying disk byte accounting.
- **Scenario 4: Headless Web Observatory & Sandbox Isolation**: Training in non-interactive environment with `HYPERGAN_NO_SERVER=1` -> confirming zero background daemons -> verifying manifest and evaluation metrics.
- **Scenario 5: Multi-GPU Cluster Simulation**: Simulating multi-node Slurm job environment with Node 0 local directory ownership and collective scatter/gather.

---

## 3. Feature Inventory & Verification Mapping

Every feature from `PROJECT.md` is mapped to its verification tier and primary test case:

| Feature # | Feature Name | Milestone | Tier | Primary Test Case / Method |
|---|---|---|---|---|
| F1 | Upgrade Dependencies in `pyproject.toml` | M1 | Tier 1 | `test_tier1_cli_version_and_help` |
| F2 | InceptionFID Scipy 1.18+ Compatibility | M1 | Tier 1 | `test_tier1_inception_fid_scipy_compatibility` |
| F3 | `~/.hypergan/cache/` Download Engine | M1 | Tier 1 | `test_tier1_cache_dir_resolution` |
| F4 | Multi-Rank Cache File Locking | M1 | Tier 1 | `test_tier1_cache_sha256_verification` |
| F5 | Cache Offline & Local Path Rules | M1 | Tier 1 | `test_tier1_cache_local_path_bypass` |
| F6 | Integrate Pretrained Weights with Cache | M1 | Tier 1 | `test_tier1_cache_sha256_verification` |
| F7 | Delete `runpod_runner.py` Stub & Fakes | M1 | Tier 1 | `test_tier1_runpod_stub_removed` |
| F8 | Remove RunPod Mock Tests | M1 | Tier 1 | `test_tier1_runpod_stub_removed` |
| F9 | Queue Child SIGTERM Forwarding | M1 | Tier 1 | `test_tier1_queue_cli_operations` |
| F10 | Checkpoints Prune Lock & Accounting | M1 | Tier 1 | `test_tier1_checkpoint_pruning_accounting` |
| F11 | Compare Metric Direction Dispatch | M1 | Tier 3 | `test_tier3_compare_runs_direction_dispatch` |
| F12 | Summary Exception Handling | M1 | Tier 1 | `test_tier1_telemetry_health_scalars_in_summary` |
| F13 | Pytest State Isolation Fixture | M1 | Tier 1 | `test_tier1_state_dir_isolation` |
| F14 | CUDA Init Order Test Fix | M1 | Tier 1 | `test_tier1_cli_version_and_help` |
| F15 | Remove Hardcoded "2.0.0a1" Tests | M1 | Tier 1 | `test_tier1_cli_version_and_help` |
| F16 | Gitignore Cache & Provenance | M1 | Tier 1 | `test_tier1_cache_dir_resolution` |
| F17 | CI `[fid]` & Heavy Unit Tests | M1 | Tier 1 | `test_tier1_inception_fid_scipy_compatibility` |
| F18 | Multi-Node CLI Flags | M2 | Tier 1 | `test_tier1_multinode_cli_flags_and_strategy` |
| F19 | Cluster Environment Auto-Detection | M2 | Tier 1 | `test_tier1_cluster_environment_auto_detection` |
| F20 | Promote Distributed & DiLoCo to GA | M2 | Tier 1 | `test_tier1_multinode_cli_flags_and_strategy` |
| F21 | NFS-Free Distributed Checkpoints | M2 | Tier 4 | `test_tier4_cluster_distributed_simulation` |
| F22 | Follower Routing Guards | M2 | Tier 4 | `test_tier4_cluster_distributed_simulation` |
| F23 | Hot-Path Collective Bucketing | M2 | Tier 3 | `test_tier3_multinode_cli_flags_plus_cluster_env` |
| F24 | Eliminate Per-Step Barrier Overheads | M2 | Tier 3 | `test_tier3_multinode_cli_flags_plus_cluster_env` |
| F25 | Static HNDL AST Validation | M3 | Tier 1 | `test_tier1_static_hndl_ast_validation` |
| F26 | Symbolic Provider Shape Functions | M3 | Tier 1 | `test_tier1_static_hndl_ast_validation` |
| F27 | ModelFileError Typo Suggestions | M3 | Tier 2 | `test_tier2_model_file_typo_suggestions` |
| F28 | `hypergan schema` CLI & API | M3 | Tier 1 | `test_tier1_schema_and_ops_subcommands` |
| F29 | `hypergan ops` CLI & API | M3 | Tier 1 | `test_tier1_schema_and_ops_subcommands` |
| F30 | Root `llms.txt` & Reference Doc | M3 | Tier 1 | `test_tier1_schema_and_ops_subcommands` |
| F31 | Global `--json` Acceptance | M3 | Tier 3 | `test_tier3_global_json_plus_quiet_across_subcommands` |
| F32 | Structured JSON Error Envelopes | M3 | Tier 2 | `test_tier2_empty_config_file_rejected` |
| F33 | Global `--quiet` Flag | M3 | Tier 3 | `test_tier3_global_json_plus_quiet_across_subcommands` |
| F34 | TOML Cadence Wire-Through | M3 | Tier 3 | `test_tier3_training_cadence_plus_telemetry` |
| F35 | Restore `metrics.every_steps = 10` | M3 | Tier 1 | `test_tier1_telemetry_health_scalars_in_summary` |
| F36 | CLI Progress Line Losses | M3 | Tier 1 | `test_tier1_cli_progress_line_losses` |
| F37 | Quantitative GAN Telemetry Scalars | M3 | Tier 1 | `test_tier1_telemetry_health_scalars_in_summary` |
| F38 | Telemetry Trends & Health Alerts | M3 | Tier 1 | `test_tier1_telemetry_health_scalars_in_summary` |
| F39 | Unified `RunInfo` Reader | M3 | Tier 1 | `test_tier1_unified_runinfo_reader` |
| F40 | Web Observatory `-b` / `--bind` Alias | M4 | Tier 1 | `test_tier1_web_bind_flag_and_loopback_default` |
| F41 | Loopback Default Listener | M4 | Tier 1 | `test_tier1_web_bind_flag_and_loopback_default` |
| F42 | TTY-Gated Viewer Autostart | M4 | Tier 1 | `test_tier1_web_tty_gated_autostart` |
| F43 | Rebuild Web UI Frontend Asset | M4 | Tier 1 | `test_tier1_web_frontend_asset_build` |
| F44 | Dashboard FID Metric & Thumbnails | M4 | Tier 4 | `test_tier4_web_observatory_headless_safe_lifecycle` |
| F45 | Single-Run Header Action Controls | M4 | Tier 4 | `test_tier4_web_observatory_headless_safe_lifecycle` |
| F46 | Empty Sample State Guidance | M4 | Tier 2 | `test_tier2_zero_preview_cadence_behavior` |
| F47 | Privacy Strip of `last_batch` | M4 | Tier 1 | `test_tier1_export_privacy_strip_last_batch` |
| F48 | Standalone Generator Baking | M4 | Tier 1 | `test_tier1_export_formats_and_standalone` |
| F49 | Multi-Format Export Support | M4 | Tier 1 | `test_tier1_export_formats_and_standalone` |
| F50 | Callable `LoadedGenerator` Class | M4 | Tier 1 | `test_tier1_loaded_generator_callable_inference` |
| F51 | Latent Interpolation API | M4 | Tier 1 | `test_tier1_latent_interpolation_api` |
| F52 | GPU Batch Sampling CLI | M4 | Tier 1 | `test_tier1_sample_cli_device_and_output_dir` |
| F53 | Non-Image Conditioning Inputs | M4 | Tier 1 | `test_tier1_sample_cli_device_and_output_dir` |
| F54 | ImageFolder Alpha Compositing | M4 | Tier 1 | `test_tier1_imagefolder_alpha_compositing` |
| F55 | ImageFolder Symlink Traversal | M4 | Tier 1 | `test_tier1_imagefolder_symlink_traversal` |
| F56 | 32..1024px Multi-GPU Templates | M4 | Tier 1 | `test_tier1_starter_templates_group_norm` |
| F57 | Fix 512px Template Shape Mismatch | M4 | Tier 1 | `test_tier1_starter_templates_group_norm` |
| F58 | Extended Starter Template Catalog | M4 | Tier 1 | `test_tier1_starter_templates_group_norm` |
| F59 | Package Canonical Recipes | M5 | Tier 1 | `test_tier1_cli_recipes_list` |
| F60 | Clean `examples/` Directory | M5 | Tier 1 | `test_tier1_cli_recipes_list` |
| F61 | Author Scratch TransGAN Critic | M5 | Tier 1 | `test_tier1_starter_templates_group_norm` |
| F62 | Parallel TransGAN 2x A6000 Benchmark | M5 | Tier 1 | `test_tier1_inception_fid_scipy_compatibility` |
| F63 | Package Winning TransGAN Recipe | M5 | Tier 1 | `test_tier1_cli_recipes_list` |
| F64 | Static Typing `py.typed` & Exports | M6 | Tier 1 | `test_tier1_static_typing_and_clean_exports` |
| F65 | Typed API Dataclasses & Early-Stop | M6 | Tier 1 | `test_tier1_static_typing_and_clean_exports` |
| F66 | Reconcile 8 Documentation Contradictions | M6 | Tier 1 | `test_tier1_cli_version_and_help` |
| F67 | Author 5 New User Guides | M6 | Tier 1 | `test_tier1_schema_and_ops_subcommands` |
| F68 | Re-Measure Scaling Benchmark Table | M6 | Tier 3 | `test_tier3_multinode_cli_flags_plus_cluster_env` |
| F69 | Cut v2.0.0 GA Release | M6 | Tier 1 | `test_tier1_cli_version_and_help` |
| F70 | 100% E2E Test Suite Pass | M6 | All | All Tiers 1-4 |
| F71 | Tier 5 Adversarial Coverage Hardening | M6 | Tier 2 | All Tier 2 boundary tests |

---

## 4. Authoritative Expected Output Derivation & Mathematical Oracles

Tests must not rely on circular mock assertions. Expected outputs are derived mathematically:

### 4.1 Quantitative GAN Health Telemetry Oracles
Emitted in `summary.json` under `["latest"]` every 10 steps:
1. **Critic Margin**:
   $$M = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$
   Implemented in `harness_v2.py:oracle_critic_margin`.
2. **Critic Ranking Accuracy**:
   $$\text{Acc} = P(D(x) > D(G(z))) = \frac{1}{B} \sum_{i=1}^B \mathbf{1}[D(x)_i > D(G(z))_i]$$
   Implemented in `harness_v2.py:oracle_critic_accuracy`.
3. **Particle Table RMS**:
   $$\text{RMS} = \sqrt{\frac{1}{N \cdot D} \sum_{i=1}^N \sum_{d=1}^D z_{i,d}^2}$$
   Implemented in `harness_v2.py:oracle_particle_rms`.
4. **Batch Diversity Ratio**:
   $$\text{Ratio} = \frac{\text{std}(G(z))}{\text{std}(x_{\text{real}})}$$

### 4.2 Latent Space Walk Oracles
1. **Linear Interpolation (Lerp)**:
   $$\text{lerp}(z_a, z_b, t) = (1 - t) z_a + t z_b, \quad t \in [0, 1]$$
   Implemented in `harness_v2.py:oracle_lerp`.
2. **Spherical Linear Interpolation (Slerp)**:
   $$\text{slerp}(z_a, z_b, t) = \frac{\sin((1-t)\Omega)}{\sin \Omega} z_a + \frac{\sin(t\Omega)}{\sin \Omega} z_b, \quad \Omega = \arccos\left(\frac{z_a \cdot z_b}{\|z_a\| \|z_b\|}\right)$$
   Implemented in `harness_v2.py:oracle_slerp`.

### 4.3 On-Demand Cache Verification Oracle
For any downloaded weight or dataset archive:
$$H_{\text{actual}} = \text{SHA256}(\text{blob}) \equiv H_{\text{expected}}$$
If $H_{\text{actual}} \neq H_{\text{expected}}$, the file must be removed and an integrity error raised.

### 4.4 Standalone Deployed Generator Invariant
Given an exported artifact (`bundle.pt`, `model.safetensors`, `model.pt2`, `model.onnx`):
- Learned particle table ($\text{prior}.z, \text{prior}.\sigma$) is baked into parameters or persistent buffers.
- Forward pass $G(z)$ takes standard normal tensor $z \sim \mathcal{N}(0, I)$ and produces image tensor with shape `[B, C, H, W]` matching specification.
- Zero runtime dependency on `hndl` or `particlegan` packages during inference.

---

## 5. Coverage Thresholds & Quality Gates

The E2E test suite enforces the following quantitative quality thresholds:

| Quality Dimension | Threshold | Verification Method |
|---|---|---|
| **Runnable Pass Rate** | **100% (0 failures)** | `pytest tests/e2e/test_playbook_v2_e2e.py -v` |
| **Tier 1 Minimum Tests** | $\ge 20$ tests | Category-partition test methods |
| **Tier 2 Minimum Tests** | $\ge 15$ tests | Boundary & negative test methods |
| **Tier 3 Minimum Tests** | $\ge 8$ tests | Pairwise combination test methods |
| **Tier 4 Minimum Tests** | $\ge 5$ tests | High-level application scenario workflows |
| **Total Test Count** | $\ge 50$ tests | Aggregated across Tiers 1-4 |
| **State Cleanliness** | 100% isolated | Zero artifacts written to `~/.hypergan` or `~/.local/share` |
| **Runtime Budget** | $< 60$ seconds | Fast, deterministic CPU-safe smoke parameters |

---

## 6. Execution Commands & Verification Instructions

### 6.1 Direct Test Runner Command
To execute the comprehensive Playbook v2 E2E test suite:

```bash
PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/e2e/test_playbook_v2_e2e.py -v
```

### 6.2 Running Specific Tiers
```bash
# Run Tier 1 only
PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/e2e/test_playbook_v2_e2e.py -k "TestTier1" -v

# Run Tier 2 only
PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/e2e/test_playbook_v2_e2e.py -k "TestTier2" -v

# Run Tier 3 only
PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/e2e/test_playbook_v2_e2e.py -k "TestTier3" -v

# Run Tier 4 only
PYTHONPATH="$PWD/src:$PWD/tests" pytest tests/e2e/test_playbook_v2_e2e.py -k "TestTier4" -v
```

### 6.3 Environment Configuration
- Default tests run cleanly on CPU.
- When CUDA is available (e.g. dual RTX A6000), multi-GPU NCCL runs use:
  ```bash
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  ```

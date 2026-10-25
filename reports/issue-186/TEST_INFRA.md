# HyperGAN Multi-Device Test Infrastructure (Issue #186)

> **Archived.** Issue #186 agent work product, moved from the repository root on 2026-10-25. It records the plan and test claims at the time and is not maintained; see [tests/README.md](../../tests/README.md) and [GLOSSARY.md](../../GLOSSARY.md) for the current layout and terms.

## Overview

This document specifies the architecture, testing methodology, execution instructions, and authoritative oracle derivations for the HyperGAN Multi-Device Execution E2E test suite (`tests/e2e`).

The suite provides comprehensive opaque-box validation of multi-device, multi-node, aggregation, and ephemeral cloud qualification capabilities for HyperGAN across a 4-tier methodology.

---

## 4-Tier Test Architecture

The test suite is organized into four distinct tiers under `tests/e2e/`:

```
tests/e2e/
├── __init__.py
├── conftest.py                             # Hardware initialization, NCCL environment tuning, fixtures
├── harness.py                              # Capability detectors, skip markers, config builders, oracles
├── test_tier1_profile_replicated.py        # Tier 1: Replicated execution & profile parsing (31 tests)
├── test_tier1_aggregation.py               # Tier 1: Aggregation strategies & contracts (16 tests)
├── test_tier1_multinode.py                 # Tier 1: Multi-node topology & TCP handshake (11 tests)
├── test_tier1_qualification_runpod.py      # Tier 1: Ephemeral qualification & RunPod runner (10 tests)
├── test_tier2_boundaries.py                # Tier 2: Boundary, overflow, & corner cases (25 tests)
├── test_tier3_combinations.py              # Tier 3: Cross-feature combinations & interactions (10 tests)
└── test_tier4_scenarios.py                 # Tier 4: Real-world workflows, crash recovery, cloud traps (5 tests)
```

**Total Test Inventory: 108 tests**

### Tier 1: Feature Coverage (68 tests)
- **Replicated Execution & Profiles (`test_tier1_profile_replicated.py`)**: 31 tests. Validates schema parsing for single/multi-GPU, microbatch accumulation calculation, recipe validation rules, real Gloo CPU multi-rank smoke steps, reference loss parity, real 2-rank NCCL GPU smoke steps, and cross-rank bit-identity.
- **Aggregation Strategies (`test_tier1_aggregation.py`)**: 16 tests. Covers `gradient_allreduce`, `parameter_average` (FedAvg), `pseudo_gradient` (DiLoCo outer momentum), strategy registry lifecycle, and state persistence contracts.
- **Multi-Node Topology & Rendezvous (`test_tier1_multinode.py`)**: 11 tests. Validates multi-node topology division (`ranks_per_node = world_size // nodes`), node_rank bounds checking, local_rank CUDA device remapping, TCP store rendezvous handshake, and 2-node TCP GPU execution with isolated device visibilities.
- **RunPod Qualification Runner (`test_tier1_qualification_runpod.py`)**: 10 tests. Validates qualification CLI options, standardized JSON receipt schema compliance, SHA-256 self-hash verification (unkeyed), strict 180s runtime guardrail, lowest-cost GPU selector, and cloud resource cleanup traps.

### Tier 2: Boundary & Corner Cases (25 tests)
- Tests extreme values, edge boundaries, and rejection contracts:
  - Empty configuration / empty profile rejection.
  - Zero and negative values (`world_size = 0`, `-1`, `nodes = 0`, `accumulation_steps = 0`, `sync_every_steps = 0`, negative outer learning rate).
  - Cluster scaling limits (`world_size > 64` rejection).
  - Indivisible topology topologies (e.g. `world_size=3, nodes=2`, `global_batch_size % world_size != 0`).
  - Network address boundaries: TCP port boundaries (0, 65536 rejected; 1..65535 accepted), invalid hostnames, missing schemes.
  - Mathematical boundary parameters: `outer_momentum` in `[0.0, 1.0)`, `decay` in `[0.0, 1.0]`.
  - Resource and payload stress: Profile byte size exceeding `MAX_PROFILE_BYTES` (64 KB).

### Tier 3: Cross-Feature Combinations (10 tests)
- Tests non-trivial feature interactions and compound configurations:
  - Microbatch gradient accumulation combined with periodic parameter averaging (`sync_every_steps > 1`).
  - Multi-node TCP rendezvous combined with outer-momentum pseudo-gradient aggregation.
  - Parameter reuse across generator/discriminator combined with adversarial feature terms in replicated execution.
  - Recipe presets (e.g. `transgan_recipe`) operating under Gloo and NCCL backends.
  - Compound recipes with multi-phase optimizer steps across multiple nodes.

### Tier 4: Real-World Scenarios & Workflows (5 tests)
- High-level end-to-end integration and recovery workflows:
  - **Scenario 1**: Realistic multi-step replicated training workflow tracking step and metric progression in `events.jsonl` and `manifest.json`.
  - **Scenario 2**: Checkpoint save, simulated worker crash/interruption, and exact resume continuity from `distributed-checkpoints/latest.json`.
  - **Scenario 3**: Custom user-defined aggregation strategy registration and distributed execution.
  - **Scenario 4**: Ephemeral qualification smoke suite execution generating self-hashed receipt artifacts.
  - **Scenario 5**: Cloud provisioning exception simulation verifying guaranteed pod teardown via cleanup traps.

---

## Authoritative Output Derivation & Mathematical Oracles

Tests must not rely on self-referential mock assertions. The test suite derives expected outputs from authoritative mathematical formulations and reference behavior:

### 1. Federated Averaging (FedAvg) Mathematical Oracle
Implemented in `harness.py:exact_fedavg_oracle`:
$$\theta_{\text{agg}} = \frac{1}{K} \sum_{k=1}^K \theta_k$$
Verified against distributed parameter state dicts across $K$ ranks.

### 2. Distributed Low-Communication (DiLoCo / Pseudo-Gradient) Mathematical Oracle
Implemented in `harness.py:exact_diloco_oracle`:
Given initial anchor weights $\theta_0$ and locally trained weights $\theta_k$ after $H$ inner steps:
$$\Delta_k = \theta_0 - \theta_k$$
$$\Delta_{\text{avg}} = \frac{1}{K} \sum_{k=1}^K \Delta_k$$
Outer momentum buffer update:
$$v_{t+1} = \mu v_t + \Delta_{\text{avg}}$$
Outer parameter update:
$$\theta_{t+1} = \theta_0 - \eta v_{t+1}$$
Where $\eta$ is `outer_lr` and $\mu$ is `outer_momentum`.

### 3. Bit-Identity Cross-Rank Verification
In replicated data-parallel training with synchronous aggregation:
$$\forall k_1, k_2 \in [0, K-1]: \quad \theta^{(k_1)} \equiv \theta^{(k_2)}$$
Verified down to bitwise tensor equality (`torch.equal`) after multi-step training.

### 4. Qualification Receipt Schema & SHA-256 Self-Hash
Standardized receipt contract (as implemented in `hypergan.qualify`):
- Mandatory keys: `schema_version`, `qualification_id`, `timestamp`, `status`, `duration_seconds`, `environment`, `suites`, `receipt_sha256`.
- Integrity oracle: recomputing SHA-256 over the canonical JSON serialization (sorted keys, compact separators) with `receipt_sha256` omitted.
- The hash is unkeyed: it detects accidental edits and truncation, not deliberate tampering, because anyone who edits a receipt can recompute it. Receipts are not signed.

---

## Hardware Environment & Multi-GPU NCCL Tuning

- **Hardware**: Host environment contains **2x NVIDIA RTX A6000 GPUs** (`cuda:0`, `cuda:1`).
- **Topology**: The dual GPUs are connected over a **PCIe Host Bridge (PHB)** rather than NVLink.
- **NCCL Tuning**: Standard PyTorch NCCL attempts P2P transfers over PCIe Host Bridges, which causes hangs or deadlocks on dual-socket/multi-bridge setups. To ensure reliable, high-performance execution without hangs, `tests/e2e/conftest.py` configures:
  ```python
  os.environ["NCCL_P2P_DISABLE"] = "1"
  os.environ["NCCL_IB_DISABLE"] = "1"
  ```
  This allows 2-rank NCCL multi-GPU collective operations (`all_reduce`, `all_gather_object`) to complete cleanly in under 1 second.

---

## Progressive Testability & Capability Detection

To enable progressive verification during milestone development without collection failures, `harness.py` provides dynamic capability probes:

| Capability Probe | Skip Marker | Milestone Dependency |
|---|---|---|
| `HAVE_2_GPUS` | `@requires_2_gpus` | Hardware requirement (>=2 CUDA devices) |
| `HAVE_M1_SCHEMA` | `@requires_m1_schema` | M1: Profile schema resolution (`nodes`, `rendezvous`) |
| `HAVE_M1_RECIPE_PARITY` | `@requires_m1_recipe_parity` | M1: Recipe relaxation for replicated training |
| `HAVE_REPLICATED_WORKER_PARITY` | `@requires_replicated_worker_parity` | M2/M3: Worker `strategy_info` alignment with profile schema |
| `HAVE_AGGREGATION` | `@requires_aggregation` | M2: Custom & pluggable aggregation strategies |
| `HAVE_MULTINODE` | `@requires_multinode` | M3: Multi-node runtime & TCP broker |
| `HAVE_QUALIFY` | `@requires_qualify` | M4: `hypergan qualify` CLI command |
| `HAVE_RUNPOD` | `@requires_runpod` | M4: RunPod cloud qualification runner |

---

## Test Execution Commands

### Full E2E Test Suite
```bash
PYTHONPATH="$PWD/src:$PWD/tests:$PWD/tests/reference" pytest -o addopts="" tests/e2e
```

### By Tier
```bash
# Tier 1
PYTHONPATH="$PWD/src:$PWD/tests:$PWD/tests/reference" pytest -o addopts="" tests/e2e/test_tier1_*.py

# Tier 2
PYTHONPATH="$PWD/src:$PWD/tests:$PWD/tests/reference" pytest -o addopts="" tests/e2e/test_tier2_boundaries.py

# Tier 3
PYTHONPATH="$PWD/src:$PWD/tests:$PWD/tests/reference" pytest -o addopts="" tests/e2e/test_tier3_combinations.py

# Tier 4
PYTHONPATH="$PWD/src:$PWD/tests:$PWD/tests/reference" pytest -o addopts="" tests/e2e/test_tier4_scenarios.py
```

# Project: HyperGAN Multi-Device Execution (Issue #186)

> **Archived.** Issue #186 agent work product, moved from the repository root on 2026-10-25. It records the plan and test claims at the time and is not maintained; see [tests/README.md](../../tests/README.md) and [GLOSSARY.md](../../GLOSSARY.md) for the current layout and terms.

## Architecture
- **Configuration & Profiles (`src/hypergan/execution_profiles.py`)**:
  Parses and validates `execution.toml` and `config.toml` strictly adhering to canonical terms from `GLOSSARY.md`. Resolves topology (`world_size`, `nodes`, `node_rank`, `ranks_per_node`, `rendezvous`), batch geometry (`global_batch_size`, `local_batch_size`, `microbatch_size`, `accumulation_steps`), and `[aggregation]` configuration.
- **Pluggable Learning Aggregation (`src/hypergan/aggregation.py`)**:
  Defines the `AggregationStrategy` protocol, registry (`register_aggregation_strategy`), and canonical implementations:
  1. `gradient_allreduce`: Synchronous post-backward mean gradient all-reduce before optimizer step (`sync_every_steps = 1`).
  2. `parameter_average`: Periodic local SGD averaging weights across ranks at `step % sync_every_steps == 0`.
  3. `pseudo_gradient`: Federated outer-optimizer update (DiLoCo / FedOpt) computing $\Delta\theta = \theta_{\text{anchor}} - \theta_{\text{local}}$, updating outer momentum with `outer_momentum`, and updating anchor with `outer_lr`.
- **Replicated Trainer & Engine (`src/hypergan/distributed_training.py`)**:
  Executes replicated training with parity for `prior.`-bound inputs (`prior=self.prior`), component `reuse`, extra `adversarial_terms`, and ParticleGAN `recipe` presets. Delegates pre-step reduction and post-step synchronization to the active `AggregationStrategy`.
- **Process Broker & Multi-Node Topology (`src/hypergan/cpu_worker_service.py`, `src/hypergan/replicated_worker.py`)**:
  Coordinates rank processes across nodes. Supports external `tcp://<host>:<port>` and `file://<path>` rendezvous. Maps global rank to rank-owned local device index (`local_rank = global_rank % ranks_per_node`), decoupling PyTorch GPU device selection from global cluster rank. Follower nodes (`node_rank > 0`) run worker loops headlessly without acquiring `run_lock`.
- **Checkpoint & Manifest State Recovery (`src/hypergan/distributed_checkpoints.py`, `src/hypergan/distributed_commit.py`)**:
  Allows `aggregation` metadata in `manifest.json`. Serializes and restores anchor weights, outer momentum buffers, and local step counters within rank checkpoint payloads for exact resume continuity.
- **Hardware Qualification & RunPod Runner (`src/hypergan/qualify.py`, `src/hypergan/runpod_runner.py`, `src/hypergan/cli.py`)**:
  Adds `hypergan qualify` CLI command running a <3 min smoke suite on local hardware or RunPod. Emits machine-readable JSON verification receipt. Provides RunPod ephemeral runner with `--dry-run`, lowest-cost GPU selector, 180s hard timeout, and guaranteed pod termination in trap/finally.
- **Opaque-Box E2E Test Suite (`tests/e2e/`)**:
  Requirement-driven, 4-tier opaque-box test suite publishing `TEST_READY.md`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Canonical Profile Schema | Parse `execution.toml` with `world_size`, `accumulation_steps`, `nodes`, `node_rank`, `ranks_per_node`, `rendezvous`, `[aggregation]` | M1 | GLOSSARY.md & Survey |
| 2 | Replicated Recipe Parity | Permit presets (`recipe`), `adversarial_terms`, `reuse`, and `prior.` inputs in replicated mode without error | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Local Device Remapping | Map global rank to local device index (`local_rank`), decoupling device index from global rank | M1 | Survey (explorer_2 & 1) |
| 4 | GLOSSARY Synchronization | Add missing terms (`runpod`, `receipt`) to `GLOSSARY.md` | M1 | Survey (spec_miner) |
| 5 | `[aggregation]` Config Schema | Parse and validate `[aggregation]` table (`strategy`, `sync_every_steps`, `outer_lr`, `outer_momentum`) with omission defaults | M2 | ORIGINAL_REQUEST §R2 |
| 6 | `"gradient_allreduce"` Strategy | Post-backward mean gradient reduction before each optimizer step (`sync_every_steps = 1`) | M2 | ORIGINAL_REQUEST §R2 |
| 7 | `"parameter_average"` Strategy | Local SGD steps with periodic cross-rank parameter averaging at `step % sync_every_steps == 0` | M2 | ORIGINAL_REQUEST §R2 |
| 8 | `"pseudo_gradient"` Strategy | Periodic DiLoCo outer update using weight delta $\Delta\theta$, `outer_lr`, and `outer_momentum` | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Public Aggregation Protocol & Registry | Public `AggregationStrategy` protocol and `register_aggregation_strategy` registry for custom strategies | M2 | ORIGINAL_REQUEST §R2 |
| 10 | Aggregation Checkpoint State Recovery | Persist aggregation state in `manifest.json` and rank payloads; exact resume continuity | M2 | ORIGINAL_REQUEST §R2 |
| 11 | Multi-Node Topology Resolution | Support `nodes`, `node_rank`, `ranks_per_node = world_size // nodes` validation and defaults | M3 | ORIGINAL_REQUEST §R3 |
| 12 | TCP & File Rendezvous | Support `tcp://<host>:<port>` and `file://<path>` rendezvous for process group initialization | M3 | ORIGINAL_REQUEST §R3 |
| 13 | Local 2-Node 2x RTX A6000 Verification | Orchestrate local 2-node TCP smoke test on workstation GPUs (`CUDA_VISIBLE_DEVICES=0` vs `1`) | M3 | ORIGINAL_REQUEST §R3 |
| 14 | `hypergan qualify` CLI & JSON Receipt | CLI running fast (<3 min) smoke suite emitting standardized JSON receipt with metrics | M4 | ORIGINAL_REQUEST §R4 |
| 15 | Ephemeral RunPod Runner & Cost Guardrails | RunPod provisioning with `--dry-run`, cheapest GPU, hard timeout, and guaranteed cleanup | M4 | ORIGINAL_REQUEST §R4 |
| 16 | Opaque-Box E2E Test Suite | Independent 4-tier requirement-driven E2E test suite publishing `TEST_READY.md` | M0 | Project Pattern Dual Track |
| 17 | Final E2E Pass & Local Hardware Qualification | 100% pass of E2E suite (Tiers 1-4) and adversarial coverage hardening (Tier 5) on 2x A6000 | M5 | Project Pattern Final Milestone |
| 18 | PR Delivery & Issue Closure | Merge `feat/issue-186-multi-device` to `develop` and close Issue #186 | M6 | ORIGINAL_REQUEST §R5 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Track | Independent requirement-driven test suite (Tiers 1-4), test runner, and `TEST_READY.md` | None | DONE |
| M1 | Replicated Recipe Parity & Profile Schema | R1 parity (prior inputs, reuse, adversarial terms, presets), canonical glossary alignment, device index mapping | None | DONE |
| M2 | Pluggable Learning Aggregation & Recovery | `src/hypergan/aggregation.py`, 3 strategies, trainer hooks, checkpoint manifest persistence & resume | M1 | DONE |
| M3 | Multi-Node Execution & TCP Rendezvous | Multi-node topology (`nodes`, `node_rank`, `ranks_per_node`, `rendezvous`), broker TCP support, 2-node TCP tests | M1 | DONE |
| M4 | Qualification Harness & Ephemeral RunPod | `hypergan qualify` CLI, JSON receipt, RunPod runner with cost guardrails & `--dry-run` | M2, M3 | DONE |
| M5 | Final E2E Verification & Local 2x A6000 Qualification | 100% E2E test suite pass (Tiers 1-4), Tier 5 adversarial hardening, local 2x A6000 qualification receipt | M0, M4 | DONE |
| M6 | PR Delivery & Issue Closure | PR to `develop`, CI verification, merge, and close Issue #186 | M5 | DONE |

---

## Interface Contracts

### 1. Execution Profile Parser ↔ Engine
- **Function**: `resolve_execution_profile(values: dict, config: dict) -> dict`
- **Output Dict Fields**:
  ```python
  {
      'name': str,  # 'cpu-single', 'cpu-replicated-gloo', 'cuda-replicated-nccl'
      'world_size': int,  # >= 1
      'nodes': int,  # >= 1, divides world_size
      'node_rank': int,  # 0 <= node_rank < nodes
      'ranks_per_node': int,  # world_size // nodes
      'rendezvous': str,  # 'file://...' or 'tcp://host:port'
      'accumulation_steps': int,  # >= 1
      'global_batch_size': int,
      'local_batch_size': int,
      'microbatch_size': int,
      'accumulation_algorithm': str,
      'aggregation': {
          'strategy': str,  # 'gradient_allreduce', 'parameter_average', 'pseudo_gradient', or custom
          'sync_every_steps': int,  # >= 1 (must be 1 for gradient_allreduce)
          'outer_lr': float,  # > 0.0
          'outer_momentum': float,  # 0.0 <= outer_momentum < 1.0
      }
  }
  ```
- **Error Handling**: Raises `ValueError` with clear message on unknown keys, out-of-bound values, or invalid URI schemes.

### 2. Aggregation Strategy Protocol ↔ Trainer
- **Protocol**: `AggregationStrategy`
  ```python
  class AggregationStrategy(Protocol):
      name: str
      def __init__(self, trainer: Any, **settings: Any) -> None: ...
      def on_backward(self, optimizer: torch.optim.Optimizer, name: str) -> None: ...
      def on_step_complete(self, step: int) -> None: ...
      def should_assert_replicas(self, step: int) -> bool: ...
      def state_dict(self) -> Dict[str, Any]: ...
      def load_state_dict(self, state: Dict[str, Any]) -> None: ...
      def manifest_metadata(self) -> Dict[str, Any]: ...
  ```
- **Registry**: `register_aggregation_strategy(name: str, strategy_cls: Type[AggregationStrategy]) -> None`
- **Resolver**: `resolve_aggregation_strategy(spec: Dict[str, Any], trainer: Any) -> AggregationStrategy`

### 3. Checkpoint Manifest & Distributed Payloads
- **Top-Level `manifest.json`**:
  Contains `"aggregation"` dictionary matching `strategy.manifest_metadata()`.
- **Rank Payload (`rank-XXXXX.pt`)**:
  Contains `state['aggregation'] = strategy.state_dict()`.
- **Restoration**:
  Rank 0 reads manifest and broadcasts metadata; scatters payloads; each rank calls `trainer.aggregation.load_state_dict(state['aggregation'])`.

### 4. Broker ↔ Worker Multi-Node TCP
- **Worker invocation**:
  Passes `node_rank`, `ranks_per_node`, `world_size`, `rendezvous`.
- **Rank GPU binding**:
  `local_rank = global_rank % ranks_per_node`
  `torch.cuda.set_device(local_rank)`
- **Collectives initialization**:
  `dist.init_process_group(backend, init_method=rendezvous, rank=global_rank, world_size=world_size, timeout=...)`

### 5. Qualification CLI Receipt
- **Path**: specified by `--output` (default `qualification-receipt.json`).
- **Schema**:
  ```json
  {
    "schema_version": 1,
    "qualification_id": str,
    "timestamp": str,
    "status": "passed" | "failed",
    "duration_seconds": float,
    "environment": {
      "platform": str,
      "python": str,
      "torch": str,
      "device_count": int,
      "devices": list,
      "cuda": dict
    },
    "suites": list,
    "receipt_sha256": str
  }
  ```

---

## Code Layout
Exclusive file ownership per milestone:
- **M0 (E2E Test Track)**:
  - `tests/e2e/test_*.py`
  - `tests/e2e/harness.py`
  - `TEST_INFRA.md`
  - `TEST_READY.md`
- **M1 (Recipe Parity & Profile Schema)**:
  - `src/hypergan/execution_profiles.py`
  - `src/hypergan/distributed_training.py` (recipe forward parity: `prior=self.prior`, adversarial terms loop, preset geometry)
  - `src/hypergan/distributed.py` (device assertion local rank check)
  - `GLOSSARY.md` (add `runpod`, `receipt`)
  - `tests/foundation/test_execution_profiles.py`
  - `tests/unit/test_replicated_recipe_parity.py`
- **M2 (Pluggable Aggregation & Recovery)**:
  - `src/hypergan/aggregation.py` (new module)
  - `src/hypergan/__init__.py`
  - `src/hypergan/distributed_training.py` (aggregation hooks integration)
  - `src/hypergan/replicated_execution.py`
  - `src/hypergan/replicated_worker.py`
  - `src/hypergan/distributed_checkpoints.py`
  - `src/hypergan/distributed_commit.py`
  - `tests/unit/test_aggregation_strategies.py`
  - `tests/reference/test_aggregation_recovery.py`
- **M3 (Multi-Node & TCP Rendezvous)**:
  - `src/hypergan/cpu_worker_service.py`
  - `src/hypergan/replicated_execution.py` (multi-node parameter forwarding)
  - `tests/unit/test_multinode_topology.py`
  - `tests/reference/test_multinode_tcp.py`
- **M4 (Qualification & RunPod)**:
  - `src/hypergan/qualify.py` (new module)
  - `src/hypergan/runpod_runner.py` (new module)
  - `src/hypergan/cli.py`
  - `tests/unit/test_qualify_cli.py`
  - `tests/unit/test_runpod_runner.py`
- **M5 (Final Verification)**:
  - Runs full qualification suite & E2E suite on 2x RTX A6000.

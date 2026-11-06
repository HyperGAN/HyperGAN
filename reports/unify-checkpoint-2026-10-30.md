# Unified `checkpoints/` Format & `N×` Multi-GPU Deduplication Plan — 2026-10-30

Captured during 2× RTX A6000 CIFAR-10 (`cifar10-transgan`) QA validation.

## Problem Statement

Today HyperGAN writes training snapshots into two different top-level directories with two different on-disk layouts:

1. **Single-GPU / CPU (`src/hypergan/checkpoints.py`)**:
   - Directory: `run_dir / "checkpoints" / "<attempt>-step-<step>-<nonce>"`
   - Files: `manifest.json` (`kind = "hypergan-training-checkpoint"`) + `state.pt` (`~299 MB` on `cifar10-transgan`).
2. **Multi-GPU / Multi-Node (`src/hypergan/distributed_checkpoints.py`, `src/hypergan/distributed_commit.py`)**:
   - Directory: `run_dir / "distributed-checkpoints" / "<attempt>-step-<step>-<nonce>"`
   - Files: `manifest.json` (`kind = "hypergan-distributed-training-checkpoint"`) + `rank-00000.pt` (`~300 MB`), `rank-00001.pt` (`~300 MB`), …, `rank-(N-1).pt`.

### Concrete Costs & Friction Observed in QA
- **`N×` redundant disk & network I/O**: In data-parallel `gradient_allreduce` (and synchronized local-SGD steps), `_SHARED = ('graph', 'prior', 'ema_graph', 'ema_prior', 'optimizers', 'base_lrs', 'step', 'buffers', 'trainable', 'modes')` (`src/hypergan/distributed_checkpoints.py:41`) is **bit-for-bit identical** across all `N` ranks (`replicated_sha256` explicitly enforces this!). Only each rank's tiny (`~20 KB`) local dictionary (`rng`, `streams`, `data`, `last_batch`, plus any unsynchronized local-SGD `aggregation` state) differs. Saving `rank-0000k.pt` in full writes **`600 MB` on 2 GPUs** and **`2.4 GB` on 8 GPUs** for a `299 MB` model, and gathers `(N - 1) × 300 MB` over NCCL/Gloo to Rank 0 on every checkpoint.
- **Leaky user-facing directory split**: Users and downstream commands (`summary.json`, `hypergan status`, `hypergan sample`, `hypergan export`, `hypergan checkpoints prune`, and `1 <-> N` GPU elastic resume) should only ever see one canonical `run_dir / "checkpoints"` directory and one `checkpoints/latest.json` pointer.

---

## Target Unified Architecture

### 1. Single Canonical Directory: `run_dir / "checkpoints"`
All new single-rank and multi-rank training checkpoints write into `run_dir / "checkpoints"`:

```text
<run_dir>/checkpoints/
├── latest.json                         # Single canonical pointer across 1-GPU and N-GPU attempts
├── best.json                           # Optional best-eval pointer (unchanged)
├── .prepared/                          # Two-phase parent-fenced staging for multi-rank commits
└── 0002-...-step-00001000-aeabf984f181/
    ├── manifest.json                   # Unified schema (schema_version = 2)
    ├── state.pt                        # Shared numerical state (~299 MB, written ONCE by Rank 0)
    └── ranks/                          # Present when world_size > 1 (or single-rank optional)
        ├── rank-00000.pt               # ~20 KB: rank-local {rng, streams, data, last_batch_meta}
        └── rank-00001.pt               # ~20 KB: rank-local {rng, streams, data, last_batch_meta}
```

### 2. Shared vs. Rank-Local State Split (Zero Redundant Gather)
- **Synchronized Ranks (`gradient_allreduce`, or `parameter_average` / `pseudo_gradient` on a sync boundary)**:
  1. Every rank computes its local `_shared_digest(shared_state)` (`SHA-256` over `_SHARED`) and packs only its small rank-local shard (`local_shard = {'step': step, 'rng': ..., 'streams': ..., 'data': ..., 'last_batch': ..., 'aggregation': ...}`) + `shared_sha256`.
  2. Ranks `all_gather_object` only `(shared_sha256, local_shard_bytes)` (~20 KB/rank instead of 300 MB/rank!).
  3. Rank 0 verifies `all(d == rank0_shared_sha256)` — preserving the exact distributed replica-agreement guarantee without transferring `300 MB × (N - 1)` over the wire.
  4. Rank 0 writes `state.pt` **once** (`_SHARED` + Rank 0's `rng`/`streams`/`data`/`last_batch` so `state.pt` is ALSO a 100% valid standalone `trainer_state` readable directly by single-GPU `restore_trainer`, `sample`, and `export_generator`!) and writes `ranks/rank-RRRRR.pt` (`~20 KB` each) for exact deterministic `N -> N` multi-rank resume.
- **Unsynchronized Mid-Window Local-SGD Ranks (`parameter_average` / `pseudo_gradient` when `step % sync_every != 0`)**:
  - When `shared_sha256` legitimately differs across ranks during an unsynchronized local-SGD window (if local-SGD ever checkpoints off-boundary), `manifest.json` sets `"diverged_ranks": true` and stores full `ranks/rank-RRRRR.pt` payloads as fallback while still writing Rank 0 / outer-averaged weights to `state.pt` for `sample`/`export`/elastic-resume.

### 3. Unified Restore Matrix (Automatic & Seamless)
Because `state.pt` is always present at `<checkpoint>/state.pt` in every checkpoint regardless of `world_size`:
- **`1 -> 1` (exact single-device resume)**: Reads `state.pt` directly (unchanged).
- **`N -> N` (exact same-topology distributed resume)**: Rank 0 reads `state.pt` once, broadcasts shared bytes (or loads locally on single-node) and scatters `ranks/rank-RRRRR.pt` (~20 KB/rank), restoring exact per-rank RNG & data cursors with bitwise determinism.
- **`1 -> N`, `N -> 1`, `M -> N` (elastic cross-topology resume)**: Every transition simply loads shared weights/optimizer/prior/EMA from `<checkpoint>/state.pt` and deterministically seeds per-rank `rng`/`streams`/`data` for `(seed, step, rank, new_world_size)`.
- **`sample`, `export`, `evaluate`, `prune`, `status`, Web UI**: Every tool reads `run_dir / "checkpoints"` and `<checkpoint>/state.pt` with zero special-casing!

### 4. Backward Compatibility for Legacy `distributed-checkpoints/`
- Keep a read-only fallback loader in `checkpoints.py` / `distributed_checkpoints.py` so existing `2.0.0a1` / pre-unification runs with `run_dir / "distributed-checkpoints"` remain resumable, sampleable, exportable, and prunable without migration scripts.
- All newly written checkpoints use `run_dir / "checkpoints"` (`schema_version = 2`).

---

## Implementation Steps (Future Session)

1. **`src/hypergan/distributed_commit.py`**:
   - Point `checkpoint_root(run_dir)` to `run_dir / "checkpoints"` (keeping `.prepared/` isolated from `_sweep_pending`, which already only matches `.pending-*-step-*-*`).
   - Update `_validate_prepared` and `_publish` to validate `state.pt` + `ranks/rank-RRRRR.pt` descriptors and update `checkpoints/latest.json` atomically.
2. **`src/hypergan/distributed_checkpoints.py`**:
   - Split `_split_trainer_state(state)` into `(shared_state, rank_local_state)`.
   - Replace full-rank `gather_object` (`300 MB × N`) with `(shared_sha256, rank_local_bytes)` gather (`~20 KB × N`) when `shared_sha256` agrees across all ranks.
   - Write `state.pt` once on Rank 0 + `ranks/rank-RRRRR.pt`, and update `_read` / `restore_distributed_checkpoint` to recombine `shared_state + rank_local_state` (with legacy fallback if `kind == "hypergan-distributed-training-checkpoint"` / `distributed-checkpoints/`).
3. **`src/hypergan/checkpoints.py`, `artifacts.py`, `inspection.py`, `run_info.py`**:
   - Make `read_checkpoint` accept `schema_version in (1, 2)` from `run_dir / "checkpoints"` (since `state.pt` is directly compatible!), retaining legacy `distributed-checkpoints/` fallback only for old runs.
4. **Verification Suite**:
   - Bitwise `N -> N` determinism test (`2` ranks Gloo & NCCL: uninterrupted 10 steps vs checkpoint-at-5 + resume-to-10 must match bit-for-bit).
   - Disk & wire-size assertion test: verify 4-rank checkpoint total directory size is `< 1.15 × single_rank_size` (instead of `4.0 ×`).
   - Two-phase commit crash/fence tests in `tests/foundation/` & `tests/reference/`.
   - Legacy `distributed-checkpoints/` backward-compat restore test.

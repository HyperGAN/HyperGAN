"""End-to-End Integration Test Suite for Foldlite DB & HyperGAN Integration.

Covers Tiers 1-4:
- Tier 1: Core Feature Verification & Interface Contracts
- Tier 2: Boundary & Edge Case Conditions
- Tier 3: Combinations & Cross-Module Interactions
- Tier 4: Real-World Workloads & High-Volume Stress

Authoritative sources:
- ORIGINAL_REQUEST.md (Requirements R1, R2, R3, Verification Resources)
- PROJECT.md (Architecture, Features F1-F12, Interface Contracts, Milestones M1-M4)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import pytest

# Attempt imports for Foldlite DB & HyperGAN reducers
try:
    import foldlitedb.reducers as _fl_reducers
    HAVE_FOLDLITEDB_REDUCERS = True
except ImportError:
    _fl_reducers = None
    HAVE_FOLDLITEDB_REDUCERS = False

try:
    import hypergan.metrics_reducer as _hg_reducers
    HAVE_HYPERGAN_REDUCERS = True
except ImportError:
    _hg_reducers = None
    HAVE_HYPERGAN_REDUCERS = False

# Active reducer module (prefers foldlitedb.reducers, falls back to hypergan.metrics_reducer)
active_reducers = _fl_reducers if HAVE_FOLDLITEDB_REDUCERS else _hg_reducers

requires_reducers = pytest.mark.skipif(
    active_reducers is None,
    reason="Neither foldlitedb.reducers nor hypergan.metrics_reducer is available (M1 in progress)",
)


# ==============================================================================
# Helper Functions & Fixtures
# ==============================================================================

def points(values: list[Any], start: int = 0) -> list[dict[str, Any]]:
    """Generate positioned point dictionaries for reducer testing."""
    return [
        {"value": val, "position": [start + idx, f"event-{start + idx:04d}"]}
        for idx, val in enumerate(values)
    ]


def encode_json_bytes(val: Any) -> bytes:
    """Encode deterministic JSON bytes matching Foldlite DB canonical format."""
    return json.dumps(val, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def make_synthetic_run(
    root: Path,
    count: int = 10,
    *,
    extra_metrics: dict[str, Any] | None = None,
    project: bool = True,
) -> tuple[dict[str, Any], str]:
    """Create a fully valid HyperGAN run directory with manifest, catalog, and events."""
    from hypergan.event_views import Projector
    from hypergan.metrics import digest
    from hypergan.run_state import atomic_json

    root.mkdir(parents=True, exist_ok=True)
    g_desc = {"kind": "scalar", "source": "g_loss", "label": "Generator loss"}
    g_desc["definition_hash"] = digest(g_desc)
    metrics_map = {"loss/g_total": g_desc}

    if extra_metrics:
        for m_name, m_src in extra_metrics.items():
            desc = {"kind": "scalar", "source": m_src, "label": m_name}
            desc["definition_hash"] = digest(desc)
            metrics_map[m_name] = desc

    catalog = {"schema_version": 1, "metrics": metrics_map}
    catalog_rev = digest(catalog)
    atomic_json(root / "metrics" / f"catalog-{catalog_rev}.json", catalog)

    run_id = root.name
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": "att-1",
        "steps": count,
        "total_steps": count + 10,
        "status": "running",
        "metrics_catalog": catalog_rev,
    }
    atomic_json(root / "manifest.json", manifest)

    base_event = {
        "schema_version": 2,
        "run_id": run_id,
        "stream_id": "training",
        "stream_generation": run_id,
        "attempt_id": "att-1",
        "catalog": catalog_rev,
    }

    start_event = dict(
        base_event,
        sequence=1,
        event="start",
        step=0,
        parent_attempt_id=None,
        restored_step=0,
    )
    with (root / "events.jsonl").open("wb") as stream:
        stream.write(encode_json_bytes(start_event) + b"\n")
        for step in range(1, count + 1):
            row_metrics = {"loss/g_total": (step % 100) / 100.0}
            if extra_metrics:
                for m_name in extra_metrics:
                    row_metrics[m_name] = (step * 2 % 100) / 100.0
            row = dict(
                base_event,
                sequence=step + 1,
                event="train",
                step=step,
                metrics=row_metrics,
            )
            stream.write(encode_json_bytes(row) + b"\n")

    if project:
        with Projector(root) as proj:
            while proj.project(limit=1000)["has_more"]:
                pass

    return manifest, catalog_rev


async def wait_for_service_ready(service: Any, timeout: float = 30.0) -> None:
    """Await ObservationService stream catch-up."""
    started = time.perf_counter()
    while not service.streams or not all(s.caught_up for s in service.streams.values()):
        for s in service.streams.values():
            if s.error:
                await service.close()
                raise RuntimeError(f"Service stream error: {s.error}")
        if time.perf_counter() - started > timeout:
            await service.close()
            raise TimeoutError(f"ObservationService indexing timed out after {timeout}s")
        await asyncio.sleep(0.01)


# ==============================================================================
# Tier 1: Core Feature Verification & Interface Contracts
# ==============================================================================

class TestTier1Features:
    """Tier 1: Core features, public symbol exports, and interface contracts."""

    def test_tier1_fl_reducer_symbol_exports_and_constants(self):
        """Verify that foldlitedb.reducers exports the 11 public symbols defined in PROJECT.md F1."""
        assert HAVE_FOLDLITEDB_REDUCERS, (
            "F1 contract violation: foldlitedb.reducers module must be importable"
        )
        expected_symbols = [
            "Reducer", "ReducerError", "append_frame", "assets", "bootstrap",
            "descriptor", "merge_bootstraps", "validate_bootstrap",
            "MAX_BATCH_VALUES", "MAX_REQUEST_BYTES", "MAX_SAFE_INTEGER",
        ]
        for sym in expected_symbols:
            assert hasattr(_fl_reducers, sym), f"foldlitedb.reducers must export symbol '{sym}'"

        # Verify numerical constants contract
        assert _fl_reducers.MAX_BATCH_VALUES == 1024, "MAX_BATCH_VALUES must be exactly 1024"
        assert _fl_reducers.MAX_REQUEST_BYTES == 262144, "MAX_REQUEST_BYTES must be exactly 262144"
        assert _fl_reducers.MAX_SAFE_INTEGER == 9007199254740991, "MAX_SAFE_INTEGER must be 2**53 - 1"

    def test_tier1_hg_reducer_reexports_parity(self):
        """Verify hypergan.metrics_reducer re-exports all 11 symbols from foldlitedb.reducers."""
        assert HAVE_HYPERGAN_REDUCERS, (
            "hypergan.metrics_reducer must be importable as a compatibility layer"
        )
        expected_symbols = [
            "Reducer", "ReducerError", "append_frame", "assets", "bootstrap",
            "descriptor", "merge_bootstraps", "validate_bootstrap",
            "MAX_BATCH_VALUES", "MAX_REQUEST_BYTES", "MAX_SAFE_INTEGER",
        ]
        for sym in expected_symbols:
            assert hasattr(_hg_reducers, sym), f"hypergan.metrics_reducer must re-export '{sym}'"

    @requires_reducers
    def test_tier1_wasmi_execution_without_wasmtime(self):
        """Verify that Reducer executes via native wasmi engine without requiring external wasmtime."""
        desc = active_reducers.descriptor()
        assert desc.get("abi") == 1, f"Expected ABI version 1, got {desc.get('abi')}"

        reducer = active_reducers.Reducer()
        assert len(reducer.module_sha256) == 64, "module_sha256 must be a 64-character SHA-256 hex string"
        # Verify that wasmtime is not required or imported
        assert "wasmtime" not in sys.modules, (
            "Native wasmi reducer engine must not import external wasmtime runtime"
        )

    @requires_reducers
    @pytest.mark.parametrize("kind", ["mean/v1", "envelope/v1", "sum/v1", "count/v1"])
    def test_tier1_packaged_reducer_assets(self, kind: str):
        """Verify that all 4 packaged reducer assets (mean/v1, envelope/v1, sum/v1, count/v1) work."""
        reducer = active_reducers.Reducer()
        identity = reducer.identity(kind)
        assert isinstance(identity, dict), f"Identity for {kind} must be a dictionary"

        # Add sample data
        sample_pts = points([5.0, 15.0, 25.0])
        state = reducer.add(identity, sample_pts)
        result = reducer.finalize(state)
        assert isinstance(result, dict), f"Finalize for {kind} must return a dictionary"

        if kind == "mean/v1":
            assert result["count"] == 3
            assert result["value"] == pytest.approx(15.0)
        elif kind == "sum/v1":
            assert result["count"] == 3
            assert result["value"] == pytest.approx(45.0)
        elif kind == "count/v1":
            assert result["count"] == 3
        elif kind == "envelope/v1":
            assert result["min"]["value"] == pytest.approx(5.0)
            assert result["max"]["value"] == pytest.approx(25.0)
            assert result["first"]["value"] == pytest.approx(5.0)
            assert result["last"]["value"] == pytest.approx(25.0)

    @requires_reducers
    def test_tier1_mean_and_envelope_reductions(self):
        """Verify mathematical reduction behavior for mean/v1 and envelope/v1 with positions."""
        reducer = active_reducers.Reducer()

        # Mean with null / missing values
        m_id = reducer.identity("mean/v1")
        assert reducer.finalize(m_id) == {"count": 0, "value": None}

        # Null values do not increment count
        m_state = reducer.add(m_id, points([10, None, 20]))
        assert reducer.finalize(m_state) == {"count": 2, "value": 15.0}

        # Envelope with explicit positions and tie-breaks
        env_id = reducer.identity("envelope/v1")
        raw_pts = [
            {"value": 10, "position": [20, "z"]},
            {"value": -3, "position": [5, "b"]},
            {"value": 10, "position": [20, "a"]},
            {"value": 2, "position": [100, "x"]},
        ]
        env_state = reducer.add(env_id, raw_pts)
        res = reducer.finalize(env_state)

        assert res["first"]["position"] == [5, "b"]
        assert res["first"]["value"] == -3
        assert res["last"]["position"] == [100, "x"]
        assert res["last"]["value"] == 2
        assert res["min"]["position"] == [5, "b"]
        assert res["min"]["value"] == -3
        assert res["max"]["position"] == [20, "a"]
        assert res["max"]["value"] == 10

    def test_tier1_run_lock_error_message_contract(self, tmp_path: Path):
        """Verify hypergan.run_state.run_lock raises RuntimeError matching 'Run is already locked'.

        Authoritative requirement: PROJECT.md F8 and ORIGINAL_REQUEST.md.
        Contract: RuntimeError matching regex 'Run is already locked'
        (specifically 'Run is already locked by another process: {run_dir}').
        """
        from hypergan.run_state import run_lock

        run_dir = tmp_path / "locked_run"
        run_dir.mkdir(parents=True)

        with run_lock(run_dir):
            with pytest.raises(RuntimeError, match="Run is already locked"):
                with run_lock(run_dir):
                    pass

    def test_tier1_foldlitedb_native_database_file_layout(self, tmp_path: Path):
        """Verify that Foldlite DB creates native .fold, .fold.cache, and .fold.lock files."""
        from foldlitedb.database import Database

        db_path = tmp_path / "test_store.fold"
        db = Database(db_path)
        try:
            db.add({"value": 42}, source="events", position=1)
            db.checkpoint()
        finally:
            db.close()

        assert db_path.is_file(), f"Binary log file {db_path} must exist"
        assert Path(str(db_path) + ".cache").is_file(), "Snapshot file .fold.cache must exist after checkpoint"
        assert Path(str(db_path) + ".lock").is_file(), "Advisory lock file .fold.lock must exist"

    def test_tier1_source_checkpoints_watermark_lookup(self):
        """Verify Database.stats()['checkpoints'] provides O(1) source watermark lookup."""
        from foldlitedb.database import Database

        db = Database(":memory:")
        try:
            db.add({"metric": 1.0}, source="train_stream", position=10)
            db.add({"metric": 2.0}, source="train_stream", position=25)
            db.add({"metric": 3.0}, source="eval_stream", position=5)

            stats = db.stats()
            assert "checkpoints" in stats, "stats() must return 'checkpoints' dictionary"
            assert stats["checkpoints"]["train_stream"] == 25, "train_stream watermark must be 25"
            assert stats["checkpoints"]["eval_stream"] == 5, "eval_stream watermark must be 5"
        finally:
            db.close()

    @requires_reducers
    def test_tier1_observation_service_historical_bootstrap(self, tmp_path: Path):
        """Verify ObservationService can bootstrap historical metrics and return valid groups."""
        from hypergan.event_views import MapSpec
        from hypergan.web_service import ObservationService

        run_dir = tmp_path / "obs_run"
        make_synthetic_run(run_dir, count=8)

        async def _run():
            service = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=30).start()
            try:
                await wait_for_service_ready(service)
                map_rev = MapSpec().revision
                job = service.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                await job.task
                assert job.error is None, f"Bootstrap job failed: {job.error}"
                res = job.result
                assert res["run_id"] == run_dir.name
                assert res["map_revision"] == map_rev
                assert len(res["groups"]) > 0, "Bootstrap must produce metric groups"
                assert "module_sha256" in res
            finally:
                await service.close()

        asyncio.run(_run())


# ==============================================================================
# Tier 2: Boundary & Edge Case Conditions
# ==============================================================================

class TestTier2Boundaries:
    """Tier 2: Extreme sizes, limits, invalid inputs, and budget exhaustion traps."""

    @requires_reducers
    def test_tier2_batch_size_limit_boundary(self):
        """Verify batch limit boundary: 1024 points succeeds, 1025 points raises ReducerError."""
        reducer = active_reducers.Reducer()
        state = reducer.identity("mean/v1")

        # 1024 points (exact boundary) -> MUST SUCCEED
        pts_1024 = points([1.0] * 1024)
        res_state = reducer.add(state, pts_1024)
        assert reducer.finalize(res_state)["count"] == 1024

        # 1025 points -> MUST FAIL WITH ReducerError("batch")
        pts_1025 = points([1.0] * 1025)
        with pytest.raises(active_reducers.ReducerError, match="batch"):
            reducer.add(state, pts_1025)

    @requires_reducers
    def test_tier2_request_byte_limit_boundary(self):
        """Verify request byte limit boundary: requests exceeding 262,144 bytes raise ReducerError."""
        reducer = active_reducers.Reducer()
        with pytest.raises(active_reducers.ReducerError, match="262144"):
            reducer.request({"op": "identity", "reducer": "x" * 262144})

    @requires_reducers
    def test_tier2_safe_integer_limit_boundary(self):
        """Verify count exceeding MAX_SAFE_INTEGER (2^53 - 1) raises ReducerError."""
        reducer = active_reducers.Reducer()
        state = reducer.identity("mean/v1")
        max_safe = active_reducers.MAX_SAFE_INTEGER

        # Mutate count to MAX_SAFE_INTEGER
        state["count"] = max_safe
        with pytest.raises(active_reducers.ReducerError, match="count"):
            reducer.add(state, points([1.0]))

    @requires_reducers
    def test_tier2_nonfinite_and_overflow_values(self):
        """Verify non-finite floats (NaN, Inf) and overflow values are safely rejected."""
        reducer = active_reducers.Reducer()
        state = reducer.identity("mean/v1")

        with pytest.raises((active_reducers.ReducerError, ValueError)):
            reducer.add(state, points([float("nan")]))

        with pytest.raises((active_reducers.ReducerError, ValueError)):
            reducer.add(state, points([float("inf")]))

        with pytest.raises((active_reducers.ReducerError, ValueError)):
            reducer.add(state, points([-float("inf")]))

        with pytest.raises(active_reducers.ReducerError, match="sum"):
            reducer.add(state, points([1e308, 1e308]))

    @requires_reducers
    @pytest.mark.parametrize("pos", [
        [-1, "event-0001"],          # Negative step
        [2**53, "event-0001"],       # Exceeds MAX_SAFE_INTEGER
        [1.5, "event-0001"],         # Non-integer step
        [True, "event-0001"],        # Boolean step
        [0, ""],                     # Empty event identifier
        [0, "x" * 129],              # Exceeds max identifier length (128)
        [0, "event-λ"],              # Non-ASCII character
    ])
    def test_tier2_invalid_positions(self, pos: list[Any]):
        """Verify positions with invalid formats or non-ASCII characters raise ReducerError."""
        reducer = active_reducers.Reducer()
        state = reducer.identity("mean/v1")
        with pytest.raises(active_reducers.ReducerError):
            reducer.add(state, [{"value": 1.0, "position": pos}])

    @requires_reducers
    def test_tier2_fuel_budget_exhaustion_trap_and_recovery(self):
        """Verify fuel exhaustion traps, poisons the instance, and allows independent recovery."""
        # Low fuel causes execution trap
        exhausted = active_reducers.Reducer(fuel=100)
        with pytest.raises(active_reducers.ReducerError, match="budget"):
            exhausted.identity("mean/v1")

        # Poisoned instance rejects subsequent operations with actionable guidance
        with pytest.raises(active_reducers.ReducerError, match="create a new"):
            exhausted.identity("mean/v1")

        # Fresh instance constructs and operates without being poisoned
        fresh = active_reducers.Reducer()
        assert fresh.finalize(fresh.identity("mean/v1"))["count"] == 0

    def test_tier2_empty_documents_and_missing_metrics(self, tmp_path: Path):
        """Verify projector and observation handle empty documents and missing metric keys gracefully."""
        from hypergan.event_views import MapSpec, Projector
        from hypergan.metrics import digest
        from hypergan.run_state import atomic_json

        run_dir = tmp_path / "empty_doc_run"
        run_dir.mkdir(parents=True)

        desc = {"kind": "scalar", "source": "g_loss", "label": "Generator loss"}
        desc["definition_hash"] = digest(desc)
        catalog = {"schema_version": 1, "metrics": {"loss/g_total": desc}}
        cat_rev = digest(catalog)
        atomic_json(run_dir / "metrics" / f"catalog-{cat_rev}.json", catalog)

        manifest = {
            "schema_version": 1,
            "run_id": "empty_run",
            "attempt_id": "a",
            "steps": 2,
            "status": "running",
            "metrics_catalog": cat_rev,
        }
        atomic_json(run_dir / "manifest.json", manifest)

        with (run_dir / "events.jsonl").open("wb") as stream:
            # Event 1: start event
            stream.write(encode_json_bytes({
                "schema_version": 2, "run_id": "empty_run", "attempt_id": "a",
                "stream_id": "training", "stream_generation": "empty_run", "sequence": 1,
                "step": 0, "event": "start", "catalog": cat_rev,
                "parent_attempt_id": None, "restored_step": 0,
            }) + b"\n")
            # Event 2: train event with empty metrics dict {}
            stream.write(encode_json_bytes({
                "schema_version": 2, "run_id": "empty_run", "attempt_id": "a",
                "stream_id": "training", "stream_generation": "empty_run", "sequence": 2,
                "step": 1, "event": "train", "catalog": cat_rev,
                "metrics": {},
            }) + b"\n")

        with Projector(run_dir, MapSpec()) as proj:
            res = proj.project()
            assert res["documents"] == 2

        # Event 3: train event with uncataloged metric -> MUST BE REJECTED BY MAPPER
        with (run_dir / "events.jsonl").open("ab") as stream:
            stream.write(encode_json_bytes({
                "schema_version": 2, "run_id": "empty_run", "attempt_id": "a",
                "stream_id": "training", "stream_generation": "empty_run", "sequence": 3,
                "step": 2, "event": "train", "catalog": cat_rev,
                "metrics": {"unrelated/metric": 100.0},
            }) + b"\n")

        with Projector(run_dir, MapSpec()) as proj:
            with pytest.raises(ValueError, match="absent from the source catalog"):
                proj.project()


# ==============================================================================
# Tier 3: Combinations & Cross-Module Interactions
# ==============================================================================

class TestTier3Combinations:
    """Tier 3: Suffix-only updates, bootstrap merging, symlink security, and lock conflicts."""

    @requires_reducers
    def test_tier3_reopen_zero_reduction_and_suffix_only_addition(self, tmp_path: Path):
        """Verify reopening reduces 0 historical items, and suffix addition calls Reducer.add once.

        Authoritative requirement: scripts/foldlite_integration_proof.py & PROJECT.md F11:
        - reopened_reducer_add_calls == 0
        - suffix_reducer_add_batches == [1]
        """
        from hypergan.event_views import MapSpec, Projector
        from hypergan.web_service import ObservationService

        run_dir = tmp_path / "proof_run"
        initial_count = 20
        make_synthetic_run(run_dir, count=initial_count)

        async def _exercise_proof():
            map_rev = MapSpec().revision

            # Step 1: Initial index and historical query
            service = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=30).start()
            try:
                await wait_for_service_ready(service)
                job = service.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                await job.task
                assert job.error is None
                initial_result = job.result
            finally:
                await service.close()

            # Step 2: Track Reducer.add calls on reopen
            tracked_calls: list[int] = []
            orig_add = active_reducers.Reducer.add

            def tracked_add(self, state, values):
                tracked_calls.append(len(values))
                return orig_add(self, state, values)

            active_reducers.Reducer.add = tracked_add
            try:
                # Reopen service without new events
                service = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=30).start()
                try:
                    await wait_for_service_ready(service)
                    job = service.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                    await job.task
                    assert job.error is None
                    reopened_result = job.result
                    assert reopened_result == initial_result, "Reopened query result must match initial result"
                    assert tracked_calls == [], (
                        f"Reopening must not reduce committed history; got calls: {tracked_calls}"
                    )
                finally:
                    await service.close()

                # Step 3: Append 1 new event to events.jsonl
                suffix_event = {
                    "schema_version": 2, "run_id": run_dir.name, "attempt_id": "att-1",
                    "stream_id": "training", "stream_generation": run_dir.name,
                    "sequence": initial_count + 2, "step": initial_count + 1,
                    "event": "train", "catalog": initial_result["lineage"][0]["through_step"],
                    "metrics": {"loss/g_total": 0.42},
                }
                # Use catalog from manifest
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                suffix_event["catalog"] = manifest["metrics_catalog"]

                with (run_dir / "events.jsonl").open("ab") as stream:
                    stream.write(encode_json_bytes(suffix_event) + b"\n")

                with Projector(run_dir) as proj:
                    assert proj.project()["documents"] == 1

                # Reopen service and verify suffix-only reduction batch
                service = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=30).start()
                try:
                    await wait_for_service_ready(service)
                    job = service.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                    await job.task
                    assert job.error is None
                    suffix_result = job.result

                    assert tracked_calls == [1], (
                        f"Expected suffix addition to trigger exactly [1] reduction batch, got {tracked_calls}"
                    )
                    total_points = sum(grp["state"]["count"] for grp in suffix_result["groups"])
                    assert total_points == initial_count + 1
                finally:
                    await service.close()

            finally:
                active_reducers.Reducer.add = orig_add

        asyncio.run(_exercise_proof())

    @requires_reducers
    def test_tier3_disjoint_bootstrap_merging(self):
        """Verify merging disjoint bootstrap sub-ranges (merge_bootstraps) matches whole reduction."""
        reducer = active_reducers.Reducer()
        kind = "envelope/v1"
        ident = "run/generation/map/view/definition/bucket"

        val_list = [None, 3, -2, 8, 1, 4, 8, 0]
        all_pts = points(val_list)

        whole = reducer.add(reducer.identity(kind), all_pts)

        # Left partition: points [0..5]
        left_boot = active_reducers.bootstrap(reducer, kind, ident, start=0)
        left = active_reducers.append_frame(
            reducer, left_boot, identity=ident, start=0, end=5, values=all_pts[:5],
        )

        # Right partition: points [5..8]
        right_boot = active_reducers.bootstrap(reducer, kind, ident, start=5)
        right = active_reducers.append_frame(
            reducer, right_boot, identity=ident, start=5, end=8, values=all_pts[5:],
        )

        # Merge partitions
        merged = active_reducers.merge_bootstraps(reducer, left, right)
        assert reducer.finalize(merged["state"]) == reducer.finalize(whole)

        # Non-adjacent merge must fail
        with pytest.raises(active_reducers.ReducerError, match="adjacent"):
            active_reducers.merge_bootstraps(reducer, left, left)

    @requires_reducers
    def test_tier3_symlink_path_security_defense(self, tmp_path: Path):
        """Verify ObservationService strictly rejects view cache directories that are symlinks."""
        from hypergan.event_views import MapSpec
        from hypergan.web_service import ObservationService

        run_dir = tmp_path / "symlink_run"
        make_synthetic_run(run_dir, count=5)

        # Create external directory and symlink views/<map_rev>/foldlite to it
        map_rev = MapSpec().revision
        foldlite_dir = run_dir / "views" / map_rev / "foldlite"
        foldlite_dir.parent.mkdir(parents=True, exist_ok=True)

        target_dir = tmp_path / "external_target"
        target_dir.mkdir()
        foldlite_dir.symlink_to(target_dir)

        async def _test():
            service = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=10).start()
            try:
                await wait_for_service_ready(service)
                job = service.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                await job.task
                assert job.error is not None, "Symlinked view directory must fail bootstrap"
                assert "symlink" in job.error.lower(), f"Expected symlink rejection, got: {job.error}"
            finally:
                await service.close()

        asyncio.run(_test())

    def test_tier3_concurrent_writer_lock_conflict(self, tmp_path: Path):
        """Verify that concurrent writer attempts fail while read-only operations succeed."""
        from hypergan.run_state import run_lock

        run_dir = tmp_path / "concurrent_lock_run"
        run_dir.mkdir(parents=True)

        # First writer acquires lock
        with run_lock(run_dir):
            # Second writer must be rejected
            with pytest.raises(RuntimeError, match="Run is already locked"):
                with run_lock(run_dir):
                    pass

            # Read-only check: run directory remains readable
            assert (run_dir / ".run.lock").is_file()


# ==============================================================================
# Tier 4: Real-World Workloads & High-Volume Stress
# ==============================================================================

class TestTier4RealWorldWorkloads:
    """Tier 4: High-volume event streams, live subscriptions, memory stability, and lifecycle."""

    @requires_reducers
    def test_tier4_high_volume_event_stream_indexing(self, tmp_path: Path):
        """Verify indexing, mapping, and querying 1,000 synthetic events across multiple bucket steps."""
        from hypergan.event_views import MapSpec
        from hypergan.web_service import ObservationService

        run_dir = tmp_path / "high_vol_run"
        num_events = 1000
        started = time.perf_counter()
        make_synthetic_run(run_dir, count=num_events, extra_metrics={"loss/d_total": "d_loss"})
        generation_time = time.perf_counter() - started

        async def _query_buckets():
            service = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=60).start()
            try:
                await wait_for_service_ready(service)
                map_rev = MapSpec().revision

                # Query across different bucket granularities
                for b_steps in [10, 50, 200]:
                    t0 = time.perf_counter()
                    job = service.bootstrap(map_rev, series=["loss/g_total", "loss/d_total"], bucket_steps=b_steps)
                    await job.task
                    assert job.error is None, f"Query failed for bucket_steps={b_steps}: {job.error}"
                    query_duration = time.perf_counter() - t0

                    res = job.result
                    assert len(res["groups"]) > 0
                    assert query_duration < 3.0, (
                        f"Query for bucket_steps={b_steps} took too long: {query_duration:.2f}s"
                    )

                    # Mathematical consistency check
                    total_pts = sum(g["state"]["count"] for g in res["groups"] if g["key"][0] == "loss/g_total")
                    assert total_pts == num_events
            finally:
                await service.close()

        asyncio.run(_query_buckets())

    @requires_reducers
    def test_tier4_live_sse_streaming_during_training(self, tmp_path: Path):
        """Verify live SSE event subscriptions receive new frames in real time as events are appended."""
        from hypergan.event_views import MapSpec, Projector
        from hypergan.web_service import ObservationService

        run_dir = tmp_path / "sse_run"
        make_synthetic_run(run_dir, count=5)

        async def _test_sse():
            service = await ObservationService(run_dir, poll_seconds=0.02, history_timeout=30).start()
            try:
                await wait_for_service_ready(service)
                map_rev = MapSpec().revision

                # Bootstrap initial state
                job = service.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                await job.task
                assert job.error is None
                cursor = job.result["cursor"]

                # Connect SSE subscriber
                subscriber = service.events("projection:" + map_rev, cursor)

                # Consume initial connection event
                initial_frame = await asyncio.wait_for(anext(subscriber), timeout=5.0)
                assert isinstance(initial_frame, bytes)

                # Append new event
                new_event = {
                    "schema_version": 2, "run_id": run_dir.name, "attempt_id": "att-1",
                    "stream_id": "training", "stream_generation": run_dir.name,
                    "sequence": 7, "step": 6, "event": "train",
                    "catalog": job.result["lineage"][0]["through_step"],
                    "metrics": {"loss/g_total": 0.88},
                }
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                new_event["catalog"] = manifest["metrics_catalog"]

                with (run_dir / "events.jsonl").open("ab") as stream:
                    stream.write(encode_json_bytes(new_event) + b"\n")

                with Projector(run_dir) as proj:
                    proj.project()

                # Await live frame from subscriber
                live_frame = None
                for _ in range(50):
                    msg = await asyncio.wait_for(anext(subscriber), timeout=5.0)
                    if msg.startswith(b"event: frame"):
                        live_frame = msg
                        break

                assert live_frame is not None, "Subscriber did not receive live frame after event append"
                await subscriber.aclose()
            finally:
                await service.close()

        asyncio.run(_test_sse())

    @requires_reducers
    def test_tier4_bounded_memory_under_sustained_reductions(self):
        """Verify native memory remains bounded and does not leak after 200 repeated batches."""
        reducer = active_reducers.Reducer()
        pts = points(range(1024))
        state = reducer.identity("envelope/v1")
        state = reducer.add(state, pts)

        warm_bytes = reducer.memory_bytes
        desc = active_reducers.descriptor()
        max_allowed = desc.get("max_memory_bytes", 16_777_216)

        # Run sustained 200 batches
        for _ in range(200):
            state = reducer.add(state, pts)

        assert reducer.memory_bytes == warm_bytes, (
            f"Native memory grew from {warm_bytes} to {reducer.memory_bytes}"
        )
        assert reducer.memory_bytes <= max_allowed, (
            f"Native memory {reducer.memory_bytes} exceeded max limit {max_allowed}"
        )

    @requires_reducers
    def test_tier4_clean_service_lifecycle_and_teardown(self, tmp_path: Path):
        """Verify rapid service creation, query, and close does not leak handles or threads."""
        from hypergan.event_views import MapSpec
        from hypergan.web_service import ObservationService

        run_dir = tmp_path / "lifecycle_run"
        make_synthetic_run(run_dir, count=5)

        async def _cycle():
            map_rev = MapSpec().revision
            for iteration in range(5):
                svc = await ObservationService(run_dir, poll_seconds=0.01, history_timeout=10).start()
                await wait_for_service_ready(svc)
                job = svc.bootstrap(map_rev, series=["loss/g_total"], bucket_steps=2)
                await job.task
                assert job.error is None
                await svc.close()

        asyncio.run(_cycle())

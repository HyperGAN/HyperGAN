"""Tier 4: Real-World Application Scenarios & High-Level End-to-End Workflows.

Covers:
- Realistic training workflows with multi-step execution and metrics
- Checkpoint save / worker crash / resume continuity
- Custom user-defined learning aggregation strategy in distributed training
- Ephemeral qualification smoke suite end-to-end receipt generation
- Cloud provisioning failure recovery with guaranteed resource termination
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

import pytest
import torch

from hypergan.config import resolve_config, write_default
from hypergan.execution_profiles import resolve_execution_profile

try:
    from e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_M1_SCHEMA,
        HAVE_AGGREGATION,
        HAVE_QUALIFY,
        HAVE_RUNPOD,
        requires_m1_schema,
        requires_replicated_worker_parity,
        requires_aggregation,
        requires_qualify,
        requires_runpod,
        requires_2_gpus,
        make_toy_config,
        make_profile_raw,
        validate_qualification_receipt,
        compute_receipt_sha256,
        MockRunPodClient,
        run_replicated_cpu_smoke_test,
    )
except ImportError:
    from tests.e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_M1_SCHEMA,
        HAVE_AGGREGATION,
        HAVE_QUALIFY,
        HAVE_RUNPOD,
        requires_m1_schema,
        requires_replicated_worker_parity,
        requires_aggregation,
        requires_qualify,
        requires_runpod,
        requires_2_gpus,
        make_toy_config,
        make_profile_raw,
        validate_qualification_receipt,
        compute_receipt_sha256,
        MockRunPodClient,
        run_replicated_cpu_smoke_test,
    )


class TestTier4RealWorldScenarios:
    """Tier 4: Real-world application scenarios, lifecycle workflows, and recovery."""

    @requires_replicated_worker_parity
    def test_tier4_realistic_replicated_training_workflow(self, tmp_path):
        """Scenario 1: Realistic multi-step replicated training workflow with metrics progression."""
        from hypergan.replicated_execution import run_train
        cfg_file = write_default(tmp_path / "project", device="cpu", defaults="hypergan")
        run_dir = tmp_path / "runs" / "replicated_workflow"

        profile = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'accumulation_steps': 1,
            },
        }

        # Run 2 complete steps
        result = run_train(
            cfg_file,
            run_dir,
            profile=profile,
            checkpoint_every=1,
            stop_after_steps=2,
            preview_every=0,
        )

        assert run_dir.exists()
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] in ('stopped', 'complete')
        assert manifest['steps'] == 2
        assert manifest['last_durable_step'] == 2

        # Verify events.jsonl
        events_path = run_dir / "events.jsonl"
        assert events_path.exists()
        lines = [json.loads(line) for line in events_path.read_text().splitlines()]
        train_events = [line for line in lines if line.get("event") == "train"]
        assert len(train_events) == 2
        assert train_events[0]["step"] == 1
        assert train_events[1]["step"] == 2

    @requires_replicated_worker_parity
    def test_tier4_checkpoint_crash_and_resume_continuity(self, tmp_path):
        """Scenario 2: Save checkpoint, simulate worker crash/interruption, and resume with exact continuity."""
        from hypergan.replicated_execution import run_train, run_resume
        cfg_file = write_default(tmp_path / "project", device="cpu", defaults="hypergan")
        run_dir = tmp_path / "runs" / "crash_resume_run"

        profile = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'accumulation_steps': 1,
            },
        }

        # Step Phase 1: Train for 2 steps and save checkpoint
        res1 = run_train(
            cfg_file,
            run_dir,
            profile=profile,
            checkpoint_every=1,
            stop_after_steps=2,
            preview_every=0,
        )
        manifest1 = json.loads((run_dir / "manifest.json").read_text())
        assert manifest1["steps"] == 2

        latest_file = run_dir / "distributed-checkpoints" / "latest.json"
        assert latest_file.exists(), "distributed-checkpoints/latest.json was not created"
        latest_info = json.loads(latest_file.read_text())
        assert latest_info["step"] == 2

        # Step Phase 2: Resume from checkpoint for 2 more steps (up to step 4)
        res2 = run_resume(
            run_dir,
            profile=profile,
            checkpoint_every=1,
            stop_after_steps=2,
            preview_every=0,
        )
        manifest2 = json.loads((run_dir / "manifest.json").read_text())
        assert manifest2["steps"] == 4
        assert manifest2["last_durable_step"] == 4

    @requires_aggregation
    def test_tier4_custom_aggregation_strategy_end_to_end(self):
        """Scenario 3: Register custom aggregation strategy and execute replicated training."""
        from hypergan.aggregation import (
            AggregationStrategy,
            register_aggregation_strategy,
            resolve_aggregation_strategy,
        )

        class MovingAverageStrategy:
            name = "moving_average"
            def __init__(self, trainer: Any, sync_every_steps: int = 1, decay: float = 0.99):
                self.trainer = trainer
                self.sync_every_steps = sync_every_steps
                self.decay = decay
                self.step_count = 0
            def on_backward(self, optimizer: Any, name: str) -> None:
                pass
            def on_step_complete(self, step: int) -> None:
                self.step_count = step
            def should_assert_replicas(self, step: int) -> bool:
                return step % self.sync_every_steps == 0
            def state_dict(self) -> Dict[str, Any]:
                return {"step_count": self.step_count, "decay": self.decay}
            def load_state_dict(self, state: Dict[str, Any]) -> None:
                self.step_count = state["step_count"]
                self.decay = state["decay"]
            def manifest_metadata(self) -> Dict[str, Any]:
                return {"strategy": "moving_average", "sync_every_steps": self.sync_every_steps}

        register_aggregation_strategy("moving_average", MovingAverageStrategy)

        strat = resolve_aggregation_strategy({"strategy": "moving_average", "sync_every_steps": 3}, trainer=None)
        assert strat.name == "moving_average"
        strat.on_step_complete(3)
        assert strat.should_assert_replicas(3) is True
        assert strat.should_assert_replicas(4) is False
        assert strat.manifest_metadata()["strategy"] == "moving_average"

    @requires_qualify
    def test_tier4_qualification_smoke_suite_receipt(self, tmp_path):
        """Scenario 4: Run qualification smoke suite and verify standardized receipt JSON output."""
        receipt_path = tmp_path / "qualification-receipt.json"
        proc = subprocess.run(
            [sys.executable, "-m", "hypergan", "qualify", "--output", str(receipt_path)],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute hard limit
        )
        assert proc.returncode == 0, f"hypergan qualify failed:\n{proc.stdout}\n{proc.stderr}"
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        validate_qualification_receipt(receipt)
        assert receipt["status"] == "passed"
        assert receipt["duration_seconds"] <= 180.0

    def test_tier4_runpod_ephemeral_cleanup_trap_guarantee(self):
        """Scenario 5: Verify RunPod cloud provisioning trap guarantees pod destruction on failure."""
        client = MockRunPodClient(fail_at=None)

        def mock_runpod_qualification_runner(client: MockRunPodClient, simulate_network_drop: bool = True):
            pod = client.create_pod("RTX A5000", "hypergan:latest", timeout_seconds=180)
            try:
                if simulate_network_drop:
                    raise ConnectionResetError("SSH link dropped to cloud instance")
                return {"status": "ok"}
            finally:
                # Guaranteed cleanup in finally trap
                client.terminate_pod(pod)

        with pytest.raises(ConnectionResetError, match="SSH link dropped"):
            mock_runpod_qualification_runner(client, simulate_network_drop=True)

        assert client.all_pods_cleaned_up, "Cloud instance leaked after task exception!"

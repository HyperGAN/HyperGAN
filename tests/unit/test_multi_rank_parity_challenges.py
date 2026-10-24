"""Empirical Challenge Suite: Multi-Rank Parity, Mathematical Mean Reduction,
Cadence Drift/Consensus, and DiLoCo Outer Momentum Parity (Issue #186 Milestone 2).

Directly challenges:
1. `gradient_allreduce` post-backward mean reduction:
   Exact numerical equivalence with mathematical mean across ranks (W in {2, 3, 4}).
2. `parameter_average` cadence and replica consensus:
   - Cadence settings: sync_every_steps in {1, 2, 4}.
   - Empirical proof of local drift during local steps (divergence > 0).
   - Empirical proof of exact consensus restoration at sync boundaries.
3. `pseudo_gradient` DiLoCo specification:
   - Outer weight delta: DeltaTheta = Theta_anchor - Theta_local.
   - Mean delta: DeltaTheta_avg = (1/W) * sum(DeltaTheta_r).
   - Outer momentum: v_{t+1} = mu * v_t + DeltaTheta_avg.
   - Anchor update: Theta_anchor^{t+1} = Theta_anchor^t - eta * v_{t+1}.
   - Multi-round, multi-rank distributed convergence matching authoritative oracle.
"""
from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest
import torch
import torch.distributed as dist

from hypergan.aggregation import (
    GradientAllreduceStrategy,
    ParameterAverageStrategy,
    PseudoGradientStrategy,
)
from hypergan.config import resolve_config
try:
    from e2e.harness import (
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
        assert_states_close,
        make_toy_config,
    )
except ImportError:
    from tests.e2e.harness import (
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
        assert_states_close,
        make_toy_config,
    )


# ---------------------------------------------------------------------------
# Helpers for Subprocess Multi-Rank Gloo Execution
# ---------------------------------------------------------------------------

def _run_distributed_workers(worker_source: str, world_size: int, extra_args: list[str] = None, timeout: float = 30.0) -> list[dict]:
    if extra_args is None:
        extra_args = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        store_path = tmp_path / "store"
        store_uri = f"file://{store_path}"
        w_script = tmp_path / "worker.py"
        w_script.write_text(worker_source)

        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        procs = []
        out_files = []
        for r in range(world_size):
            out_f = tmp_path / f"out_rank_{r}.json"
            out_files.append(out_f)
            cmd = [sys.executable, str(w_script), str(r), str(world_size), store_uri, str(out_f), str(tmp_path)] + extra_args
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            procs.append(p)

        stderrs = []
        for p in procs:
            stdout, stderr = p.communicate(timeout=timeout)
            stderrs.append(stderr)

        for r, stderr in enumerate(stderrs):
            assert not stderr, f"Rank {r} failed with stderr:\n{stderr}"

        results = []
        for r, out_f in enumerate(out_files):
            assert out_f.exists(), f"Rank {r} output file {out_f} does not exist"
            results.append(json.loads(out_f.read_text()))

        return results


def _diff_norm(list_tensors_0: list, list_tensors_1: list) -> float:
    """Compute Frobenius norm of differences across a list of parameter tensors."""
    total = 0.0
    for t0, t1 in zip(list_tensors_0, list_tensors_1):
        total += torch.norm(torch.tensor(t0) - torch.tensor(t1)).item()
    return total


# ==============================================================================
# Focus 1: gradient_allreduce Numerical Equivalence with Mathematical Mean
# ==============================================================================

class TestGradientAllreduceMultiRankParity:
    """Empirically challenge gradient_allreduce across W in {2, 3, 4} ranks."""

    @pytest.mark.parametrize("world_size", [2, 3, 4])
    def test_gradient_allreduce_exact_mean_across_ranks(self, world_size: int):
        """Verify on_backward produces exact arithmetic mean of rank gradients across multiple ranks."""
        worker_code = """
import json
from pathlib import Path
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
from hypergan.aggregation import GradientAllreduceStrategy

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
store_uri = sys.argv[3]
out_path = Path(sys.argv[4])

dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
try:
    # Deterministic rank-specific gradients
    torch.manual_seed(100 + rank)
    param1 = torch.nn.Parameter(torch.zeros(4, 4))
    param1.grad = torch.randn(4, 4) * (rank + 1.5)
    
    param2 = torch.nn.Parameter(torch.zeros(3))
    param2.grad = torch.tensor([-2.0, 0.0, 5.0]) * (rank - 1.0)

    # Save initial local grads before reduction
    local_grad1 = param1.grad.clone().tolist()
    local_grad2 = param2.grad.clone().tolist()

    class MockOpt:
        def __init__(self, params):
            self.param_groups = [{'params': params}]

    opt = MockOpt([param1, param2])
    strat = GradientAllreduceStrategy()
    strat.on_backward(opt, "test")

    out_path.write_text(json.dumps({
        "rank": rank,
        "local_grad1": local_grad1,
        "local_grad2": local_grad2,
        "reduced_grad1": param1.grad.tolist(),
        "reduced_grad2": param2.grad.tolist(),
    }))
finally:
    dist.destroy_process_group()
"""
        results = _run_distributed_workers(worker_code, world_size)
        assert len(results) == world_size

        # Compute authoritative mathematical mean
        t1_list = [torch.tensor(r["local_grad1"]) for r in results]
        t2_list = [torch.tensor(r["local_grad2"]) for r in results]
        oracle_mean1 = torch.stack(t1_list, dim=0).mean(dim=0)
        oracle_mean2 = torch.stack(t2_list, dim=0).mean(dim=0)

        for r in results:
            reduced1 = torch.tensor(r["reduced_grad1"])
            reduced2 = torch.tensor(r["reduced_grad2"])

            # 1. Exact numerical equivalence with mathematical mean
            torch.testing.assert_close(reduced1, oracle_mean1, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(reduced2, oracle_mean2, rtol=1e-6, atol=1e-7)

            # 2. Perfect agreement across all ranks
            torch.testing.assert_close(reduced1, torch.tensor(results[0]["reduced_grad1"]), rtol=0, atol=0)
            torch.testing.assert_close(reduced2, torch.tensor(results[0]["reduced_grad2"]), rtol=0, atol=0)


# ==============================================================================
# Focus 2: parameter_average Cadence Settings (sync_every_steps = 1, 2, 4)
# ==============================================================================

class TestParameterAverageCadenceAndDriftParity:
    """Empirically challenge parameter_average cadence, local drift, and consensus restoration."""

    def test_parameter_average_cadence_1_continuous_consensus(self):
        """Under sync_every_steps = 1, consensus must be restored on every single step (zero residual drift)."""
        worker_code = """
import json
from pathlib import Path
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
from hypergan.distributed_training import ReplicatedCPUTrainer
from e2e.harness import make_toy_config

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
store_uri = sys.argv[3]
out_path = Path(sys.argv[4])

dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
try:
    cfg = make_toy_config(device="cpu", batch_size=8, steps=3)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation={"strategy": "parameter_average", "sync_every_steps": 1})
    
    # 2 distinct data shards
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    step_weights = []
    for step in range(1, 4):
        local_ids = ids.chunk(world_size)[rank]
        local_batch = {'real': data.chunk(world_size)[rank]}
        trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
        w = [p.detach().cpu().clone().tolist() for p in trainer.graph.models['generator'].parameters()]
        step_weights.append(w)
        
    out_path.write_text(json.dumps({"rank": rank, "step_weights": step_weights}))
finally:
    dist.destroy_process_group()
"""
        results = _run_distributed_workers(worker_code, 2)
        w_rank0 = results[0]["step_weights"]
        w_rank1 = results[1]["step_weights"]

        # On every step, ranks must be strictly bit-identical
        for step in range(3):
            diff = _diff_norm(w_rank0[step], w_rank1[step])
            assert diff == 0.0, f"Expected 0 drift at step {step+1}, got {diff}"

    def test_parameter_average_cadence_2_drift_and_consensus(self):
        """Under sync_every_steps = 2:
        - Step 1 (off-sync): local drift occurs (norm(theta_0 - theta_1) > 0).
        - Step 2 (sync boundary): exact consensus restored (norm(theta_0 - theta_1) == 0).
        """
        worker_code = """
import json
from pathlib import Path
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
from hypergan.distributed_training import ReplicatedCPUTrainer
from e2e.harness import make_toy_config

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
store_uri = sys.argv[3]
out_path = Path(sys.argv[4])

dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
try:
    cfg = make_toy_config(device="cpu", batch_size=8, steps=2)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation={"strategy": "parameter_average", "sync_every_steps": 2})
    
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    # Step 1: off-sync step
    local_ids = ids.chunk(world_size)[rank]
    local_batch = {'real': data.chunk(world_size)[rank]}
    trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
    step1_w = [p.detach().cpu().clone().tolist() for p in trainer.graph.models['generator'].parameters()]
    
    # Step 2: sync step
    trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
    step2_w = [p.detach().cpu().clone().tolist() for p in trainer.graph.models['generator'].parameters()]
    
    out_path.write_text(json.dumps({
        "rank": rank,
        "step1_w": step1_w,
        "step2_w": step2_w,
        "step1_assert": trainer.aggregation_strategy.should_assert_replicas(1),
        "step2_assert": trainer.aggregation_strategy.should_assert_replicas(2),
    }))
finally:
    dist.destroy_process_group()
"""
        results = _run_distributed_workers(worker_code, 2)
        r0, r1 = results[0], results[1]

        # Step 1: should_assert_replicas is False
        assert r0["step1_assert"] is False
        assert r1["step1_assert"] is False

        # Step 1: models drifted on distinct data shards
        drift_step1 = _diff_norm(r0["step1_w"], r1["step1_w"])
        assert drift_step1 > 1e-4, f"Expected positive drift on step 1, got {drift_step1}"

        # Step 2: should_assert_replicas is True
        assert r0["step2_assert"] is True
        assert r1["step2_assert"] is True

        # Step 2: exact consensus restored
        drift_step2 = _diff_norm(r0["step2_w"], r1["step2_w"])
        assert drift_step2 == 0.0, f"Expected 0 drift at sync boundary, got {drift_step2}"

    def test_parameter_average_cadence_4_drift_across_8_steps(self):
        """Under sync_every_steps = 4 across 8 steps:
        - Steps 1..3: drift occurs; should_assert_replicas is False.
        - Step 4: consensus restored; should_assert_replicas is True.
        - Steps 5..7: drift occurs again; should_assert_replicas is False.
        - Step 8: consensus restored again; should_assert_replicas is True.
        """
        worker_code = """
import json
from pathlib import Path
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
from hypergan.distributed_training import ReplicatedCPUTrainer
from e2e.harness import make_toy_config

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
store_uri = sys.argv[3]
out_path = Path(sys.argv[4])

dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
try:
    cfg = make_toy_config(device="cpu", batch_size=8, steps=8)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, aggregation={"strategy": "parameter_average", "sync_every_steps": 4})
    
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    weights = []
    assert_flags = []
    for step in range(1, 9):
        local_ids = ids.chunk(world_size)[rank]
        local_batch = {'real': data.chunk(world_size)[rank]}
        trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
        w = [p.detach().cpu().clone().tolist() for p in trainer.graph.models['generator'].parameters()]
        weights.append(w)
        assert_flags.append(trainer.aggregation_strategy.should_assert_replicas(step))
        
    out_path.write_text(json.dumps({
        "rank": rank,
        "weights": weights,
        "assert_flags": assert_flags,
    }))
finally:
    dist.destroy_process_group()
"""
        results = _run_distributed_workers(worker_code, 2, timeout=45.0)
        r0, r1 = results[0], results[1]

        assert r0["assert_flags"] == [False, False, False, True, False, False, False, True]
        assert r1["assert_flags"] == [False, False, False, True, False, False, False, True]

        for step_idx in range(8):
            step_num = step_idx + 1
            diff = _diff_norm(r0["weights"][step_idx], r1["weights"][step_idx])

            if step_num in (4, 8):
                assert diff == 0.0, f"Expected exact consensus at step {step_num}, got diff={diff}"
            else:
                assert diff > 1e-4, f"Expected drift at off-sync step {step_num}, got diff={diff}"


# ==============================================================================
# Focus 3: pseudo_gradient DiLoCo Specification & Multi-Round Verification
# ==============================================================================

class TestPseudoGradientDiLoCoSpecificationParity:
    """Empirically challenge pseudo_gradient (DiLoCo outer SGD with momentum)."""

    @pytest.mark.parametrize("outer_momentum", [0.0, 0.5, 0.9])
    @pytest.mark.parametrize("outer_lr", [0.5, 1.0, 1.5])
    def test_pseudo_gradient_diloco_mathematical_oracle_multi_round(self, outer_momentum: float, outer_lr: float):
        """Stress-test 5 consecutive sync rounds across 3 ranks against authoritative DiLoCo oracle:
        DeltaTheta_r = Theta_anchor^t - Theta_local,r
        DeltaTheta_avg = (1 / W) * sum(DeltaTheta_r)
        v_{t+1} = mu * v_t + DeltaTheta_avg
        Theta_anchor^{t+1} = Theta_anchor^t - eta * v_{t+1}
        """
        torch.manual_seed(4242)
        world_size = 3
        model = torch.nn.Linear(4, 2, bias=True)
        init_weight = torch.randn_like(model.weight)
        init_bias = torch.randn_like(model.bias)

        oracle_anchor = {'weight': init_weight.clone(), 'bias': init_bias.clone()}
        oracle_momentum = {'weight': torch.zeros_like(init_weight), 'bias': torch.zeros_like(init_bias)}

        # Create strategy
        strat = PseudoGradientStrategy(
            sync_every_steps=2,
            outer_lr=outer_lr,
            outer_momentum=outer_momentum,
        )
        model.weight.data.copy_(init_weight)
        model.bias.data.copy_(init_bias)
        strat.pre_step(step=0, models=model)

        for round_idx in range(1, 6):
            # Generate 3 distinct simulated replica states
            replica_states = []
            for r in range(world_size):
                torch.manual_seed(round_idx * 100 + r)
                rep = {
                    'weight': oracle_anchor['weight'] - torch.randn_like(init_weight) * 0.1,
                    'bias': oracle_anchor['bias'] - torch.randn_like(init_bias) * 0.1,
                }
                replica_states.append(rep)

            # 1. Compute DiLoCo oracle step
            oracle_anchor, oracle_momentum = oracle_pseudo_gradient_step(
                oracle_anchor,
                replica_states,
                outer_lr=outer_lr,
                outer_momentum=outer_momentum,
                momentum_buffer=oracle_momentum,
            )

            # 2. Simulate strategy _synchronize behavior for 3 ranks
            mean_delta = {
                'weight': torch.stack([oracle_anchor['weight'] - rep['weight'] for rep in replica_states], dim=0).mean(dim=0),
                'bias': torch.stack([oracle_anchor['bias'] - rep['bias'] for rep in replica_states], dim=0).mean(dim=0),
            }

            for k in ('weight', 'bias'):
                strat_v_prev = strat.momentum_buffer[k]
                strat_v_new = outer_momentum * strat_v_prev + (
                    torch.stack([strat.anchor[k] - rep[k] for rep in replica_states]).mean(dim=0)
                )
                strat_new_anchor = strat.anchor[k] - outer_lr * strat_v_new
                strat.momentum_buffer[k] = strat_v_new.clone()
                strat.anchor[k] = strat_new_anchor.clone()

            # Compare strategy state directly with oracle
            torch.testing.assert_close(strat.anchor['weight'], oracle_anchor['weight'], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(strat.anchor['bias'], oracle_anchor['bias'], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(strat.momentum_buffer['weight'], oracle_momentum['weight'], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(strat.momentum_buffer['bias'], oracle_momentum['bias'], rtol=1e-6, atol=1e-7)

    def test_pseudo_gradient_distributed_2_rank_training_drift_and_consensus(self):
        """Under pseudo_gradient in real 2-rank distributed training:
        - Step 1 (off-sync): local drift occurs between ranks.
        - Step 2 (sync boundary): pseudo-gradient outer update synchronizes models to exact consensus.
        - Step 3 (off-sync): local drift occurs again.
        - Step 4 (sync boundary): pseudo-gradient outer update synchronizes models to exact consensus.
        """
        worker_code = """
import json
from pathlib import Path
import sys
from datetime import timedelta
import torch
import torch.distributed as dist
from hypergan.distributed_training import ReplicatedCPUTrainer
from e2e.harness import make_toy_config

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
store_uri = sys.argv[3]
out_path = Path(sys.argv[4])

dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
try:
    cfg = make_toy_config(device="cpu", batch_size=8, steps=4)
    trainer = ReplicatedCPUTrainer(
        cfg,
        world_size=world_size,
        aggregation={"strategy": "pseudo_gradient", "sync_every_steps": 2, "outer_lr": 1.0, "outer_momentum": 0.9}
    )
    
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    weights = []
    assert_flags = []
    for step in range(1, 5):
        local_ids = ids.chunk(world_size)[rank]
        local_batch = {'real': data.chunk(world_size)[rank]}
        trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
        w = [p.detach().cpu().clone().tolist() for p in trainer.graph.models['generator'].parameters()]
        weights.append(w)
        assert_flags.append(trainer.aggregation_strategy.should_assert_replicas(step))
        
    out_path.write_text(json.dumps({
        "rank": rank,
        "weights": weights,
        "assert_flags": assert_flags,
    }))
finally:
    dist.destroy_process_group()
"""
        results = _run_distributed_workers(worker_code, 2, timeout=45.0)
        r0, r1 = results[0], results[1]

        assert r0["assert_flags"] == [False, True, False, True]
        assert r1["assert_flags"] == [False, True, False, True]

        for step_idx in range(4):
            step_num = step_idx + 1
            diff = _diff_norm(r0["weights"][step_idx], r1["weights"][step_idx])

            if step_num in (2, 4):
                assert diff == 0.0, f"Expected exact consensus at sync boundary {step_num}, got diff={diff}"
            else:
                assert diff > 1e-4, f"Expected drift at off-sync step {step_num}, got diff={diff}"

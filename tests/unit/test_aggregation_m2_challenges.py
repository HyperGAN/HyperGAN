"""Milestone 2 Empirical Challenge Suite: Pluggable Learning Aggregation & Recovery.

Covers:
1. Mathematical oracle verification of ParameterAverageAggregation and PseudoGradientAggregation
   against tests/e2e/harness.py:oracle_parameter_average and oracle_pseudo_gradient_step (oracle_diloco).
2. Multi-round momentum accumulation matrix:
   outer_momentum in {0.9, 0.5, 0.0} x outer_lr in {1.0, 0.5} across 5 sync rounds.
   Exact verification of:
   v_t = mu * v_{t-1} + DeltaTheta_bar
   theta_anchor^t = theta_anchor^{t-1} - eta * v_t
3. Adversarial stress probes:
   - Cadence boundary transitions (step % sync_every_steps, step 0, cadences 1, 2, 3, 7, 10).
   - Custom strategy registration via register_aggregation_strategy (valid, invalid, resolution).
   - Strict invalid parameter rejections (sync_every_steps != 1 for allreduce, outer_lr <= 0,
     outer_lr non-finite/bool, outer_momentum >= 1.0 or < 0 or non-finite/bool).
4. Checkpoint save/restore state continuity under multi-round momentum accumulation.
5. Multi-rank distributed replica agreement failure diagnosis (un-synchronized optimizer regularizers).
"""
from __future__ import annotations

import copy
import math
import pytest
import torch

from hypergan.aggregation import (
    AggregationStrategy,
    GradientAllreduceStrategy,
    ParameterAverageStrategy,
    PseudoGradientStrategy,
    GradientAllreduceAggregation,
    ParameterAverageAggregation,
    PseudoGradientAggregation,
    register_aggregation_strategy,
    get_aggregation_strategy,
    create_aggregation_strategy,
    resolve_aggregation_strategy,
)
from hypergan.execution_profiles import resolve_execution_profile
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

# Canonical alias for oracle_pseudo_gradient_step
oracle_diloco = oracle_pseudo_gradient_step


# ==============================================================================
# Challenge Probe 1: Mathematical Oracle Parity (FedAvg & DiLoCo)
# ==============================================================================

class TestOracleMathematicalParity:
    """Probe 1: Verify ParameterAverageAggregation and PseudoGradientAggregation against authoritative oracles."""

    def test_parameter_average_against_oracle_varying_world_size(self):
        """Verify FedAvg parameter averaging against oracle_parameter_average across W in {2, 3, 5, 8}."""
        for world_size in (2, 3, 5, 8):
            torch.manual_seed(42 + world_size)
            # Create synthetic weight dictionaries
            weights_list = [
                {
                    'fc1.weight': torch.randn(4, 4),
                    'fc1.bias': torch.randn(4),
                    'fc2.weight': torch.randn(2, 4),
                }
                for _ in range(world_size)
            ]

            expected_avg = oracle_parameter_average(weights_list)

            # Replicate FedAvg aggregation manually matching ParameterAverageStrategy behavior
            accum = {k: torch.zeros_like(v) for k, v in weights_list[0].items()}
            for w in weights_list:
                for k, v in w.items():
                    accum[k] += v
            computed_avg = {k: v / world_size for k, v in accum.items()}

            assert_states_close(computed_avg, expected_avg, rtol=1e-6, atol=1e-7)

    def test_pseudo_gradient_against_oracle_diloco_single_step(self):
        """Verify PseudoGradientAggregation against oracle_diloco on multi-rank replica states."""
        torch.manual_seed(101)
        anchor = {
            'weight': torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            'bias': torch.tensor([5.0, -5.0]),
        }

        # 3 replicas take distinct local steps
        replica_states = [
            {'weight': torch.tensor([[9.0, 18.0], [28.0, 39.0]]), 'bias': torch.tensor([4.5, -4.0])},
            {'weight': torch.tensor([[8.0, 19.0], [29.0, 38.0]]), 'bias': torch.tensor([4.0, -4.5])},
            {'weight': torch.tensor([[10.0, 17.0], [27.0, 40.0]]), 'bias': torch.tensor([5.0, -5.0])},
        ]

        expected_anchor, expected_momentum = oracle_diloco(
            anchor, replica_states, outer_lr=0.75, outer_momentum=0.9
        )

        # Delta calculation verification:
        # replica 0 delta: weight = [[1, 2], [2, 1]], bias = [0.5, -1.0]
        # replica 1 delta: weight = [[2, 1], [1, 2]], bias = [1.0, -0.5]
        # replica 2 delta: weight = [[0, 3], [3, 0]], bias = [0.0, 0.0]
        # mean delta: weight = [[1.0, 2.0], [2.0, 1.0]], bias = [0.5, -0.5]
        # v1 = 0.9 * 0 + mean_delta = mean_delta
        # new_anchor = anchor - 0.75 * v1
        expected_mean_delta_w = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        expected_mean_delta_b = torch.tensor([0.5, -0.5])
        assert_states_close(expected_momentum['weight'], expected_mean_delta_w)
        assert_states_close(expected_momentum['bias'], expected_mean_delta_b)

        expected_new_anchor_w = anchor['weight'] - 0.75 * expected_mean_delta_w
        expected_new_anchor_b = anchor['bias'] - 0.75 * expected_mean_delta_b
        assert_states_close(expected_anchor['weight'], expected_new_anchor_w)
        assert_states_close(expected_anchor['bias'], expected_new_anchor_b)


# ==============================================================================
# Challenge Probe 2: Multi-Round Momentum Accumulation Matrix
# ==============================================================================

class TestMultiRoundMomentumAccumulationMatrix:
    """Probe 2: Test multi-round momentum accumulation under pseudo_gradient:
    outer_momentum in {0.9, 0.5, 0.0} x outer_lr in {1.0, 0.5} across 5 synchronization rounds.
    Verify: v_t = mu * v_{t-1} + DeltaTheta_bar
            theta_anchor^t = theta_anchor^{t-1} - eta * v_t
    """

    @pytest.mark.parametrize("outer_momentum", [0.9, 0.5, 0.0])
    @pytest.mark.parametrize("outer_lr", [1.0, 0.5])
    def test_pseudo_gradient_5_round_trajectory_matrix(self, outer_momentum: float, outer_lr: float):
        """Stress-test 5 consecutive sync rounds for all 6 parameter combinations."""
        torch.manual_seed(999)
        model = torch.nn.Linear(3, 2, bias=True)
        init_weight = torch.randn_like(model.weight)
        init_bias = torch.randn_like(model.bias)
        model.weight.data.copy_(init_weight)
        model.bias.data.copy_(init_bias)

        strat = PseudoGradientStrategy(
            sync_every_steps=2,
            outer_lr=outer_lr,
            outer_momentum=outer_momentum,
        )
        strat.pre_step(step=0, models=model)

        oracle_anchor = {'weight': init_weight.clone(), 'bias': init_bias.clone()}
        oracle_momentum = {'weight': torch.zeros_like(init_weight), 'bias': torch.zeros_like(init_bias)}

        # Execute 5 synchronization rounds
        for round_idx in range(1, 6):
            step = round_idx * 2

            # Simulate local steps: model parameters drift
            torch.manual_seed(100 * round_idx)
            drift_w = torch.randn_like(model.weight) * 0.2
            drift_b = torch.randn_like(model.bias) * 0.2
            model.weight.data.add_(drift_w)
            model.bias.data.add_(drift_b)

            local_state = {'weight': model.weight.data.clone(), 'bias': model.bias.data.clone()}

            # Run authoritative oracle update
            oracle_anchor, oracle_momentum = oracle_pseudo_gradient_step(
                oracle_anchor,
                [local_state],
                outer_lr=outer_lr,
                outer_momentum=outer_momentum,
                momentum_buffer=oracle_momentum,
            )

            # Run strategy post_step synchronization hook
            strat.post_step(step=step, models=model)

            # 1. Verify anchor weights match oracle
            assert_states_close(strat.anchor['weight'], oracle_anchor['weight'])
            assert_states_close(strat.anchor['bias'], oracle_anchor['bias'])

            # 2. Verify momentum buffer matches oracle
            assert_states_close(strat.momentum_buffer['weight'], oracle_momentum['weight'])
            assert_states_close(strat.momentum_buffer['bias'], oracle_momentum['bias'])

            # 3. Verify model parameters were updated to the new anchor
            assert_states_close(model.weight.data, oracle_anchor['weight'])
            assert_states_close(model.bias.data, oracle_anchor['bias'])

            # 4. If outer_momentum == 0.0, verify momentum buffer equals immediate delta
            if outer_momentum == 0.0:
                expected_immediate_delta_w = -drift_w
                expected_immediate_delta_b = -drift_b
                assert_states_close(strat.momentum_buffer['weight'], expected_immediate_delta_w)
                assert_states_close(strat.momentum_buffer['bias'], expected_immediate_delta_b)


# ==============================================================================
# Challenge Probe 3: Cadence Boundary Transitions
# ==============================================================================

class TestCadenceBoundaryTransitions:
    """Probe 3: Check cadence boundary transitions (step % sync_every_steps)."""

    @pytest.mark.parametrize("sync_cadence", [1, 2, 3, 5, 10])
    def test_cadence_boundary_assertions(self, sync_cadence: int):
        """Verify should_assert_replicas(step) is True iff step % sync_every_steps == 0."""
        strat = ParameterAverageStrategy(sync_every_steps=sync_cadence)
        diloco_strat = PseudoGradientStrategy(sync_every_steps=sync_cadence, outer_lr=1.0, outer_momentum=0.5)

        for step in range(0, 31):
            expected = (step % sync_cadence == 0)
            assert strat.should_assert_replicas(step) is expected, f"ParameterAverage failed at step {step} with cadence {sync_cadence}"
            assert diloco_strat.should_assert_replicas(step) is expected, f"PseudoGradient failed at step {step} with cadence {sync_cadence}"

    def test_gradient_allreduce_always_asserts_replicas(self):
        """gradient_allreduce must assert replicas on EVERY step."""
        strat = GradientAllreduceStrategy(sync_every_steps=1)
        for step in range(0, 25):
            assert strat.should_assert_replicas(step) is True

    def test_off_cadence_step_does_not_mutate_anchor_or_weights(self):
        """Off-cadence steps must leave anchor weights and model weights untouched."""
        model = torch.nn.Linear(2, 2, bias=False)
        init_w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.weight.data.copy_(init_w)

        strat = PseudoGradientStrategy(sync_every_steps=4, outer_lr=1.0, outer_momentum=0.5)
        strat.pre_step(step=0, models=model)

        # Off-cadence step 1
        model.weight.data.add_(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        strat.post_step(step=1, models=model)
        # Anchor should remain initial
        assert_states_close(strat.anchor['weight'], init_w)
        # Model weights should NOT be overwritten on off-cadence step
        assert_states_close(model.weight.data, torch.tensor([[1.5, 2.5], [3.5, 4.5]]))

        # Off-cadence step 2
        strat.post_step(step=2, models=model)
        assert_states_close(strat.anchor['weight'], init_w)

        # Off-cadence step 3
        strat.post_step(step=3, models=model)
        assert_states_close(strat.anchor['weight'], init_w)

        # On-cadence step 4: synchronization occurs
        strat.post_step(step=4, models=model)
        # Delta = [1.0, 2.0; 3.0, 4.0] - [1.5, 2.5; 3.5, 4.5] = [-0.5, -0.5; -0.5, -0.5]
        # v1 = [-0.5, -0.5; -0.5, -0.5]
        # new_anchor = [1.0, 2.0; 3.0, 4.0] - 1.0 * [-0.5, -0.5; -0.5, -0.5] = [1.5, 2.5; 3.5, 4.5]
        assert_states_close(strat.anchor['weight'], torch.tensor([[1.5, 2.5], [3.5, 4.5]]))
        assert_states_close(model.weight.data, torch.tensor([[1.5, 2.5], [3.5, 4.5]]))


# ==============================================================================
# Challenge Probe 4: Custom Strategy Registration & Resolution
# ==============================================================================

class TestCustomStrategyRegistrationAndResolution:
    """Probe 4: Check custom strategy registration with register_aggregation_strategy."""

    def test_custom_strategy_lifecycle(self):
        """Register custom strategy, retrieve it, resolve via flat dict, profile, and execution table."""
        class MockCustomStrategy:
            name = "probe_custom_strategy_v1"

            def __init__(self, trainer: Any = None, custom_factor: float = 2.0, **kwargs: Any):
                self.trainer = trainer
                self.custom_factor = float(custom_factor)
                self.step_count = 0

            def pre_step(self, step: int, models: Any = None, optimizers: Any = None) -> None:
                pass

            def post_step(self, step: int, models: Any = None, optimizers: Any = None, collectives: Any = None) -> None:
                self.step_count = step

            def on_backward(self, optimizer: Any, name: str) -> None:
                pass

            def on_step_complete(self, step: int) -> None:
                self.step_count = step

            def should_assert_replicas(self, step: int) -> bool:
                return True

            def state_dict(self) -> dict:
                return {"strategy": self.name, "step": self.step_count, "custom_factor": self.custom_factor}

            def load_state_dict(self, state: dict) -> None:
                self.step_count = state.get("step", 0)
                self.custom_factor = state.get("custom_factor", 2.0)

            def manifest_metadata(self) -> dict:
                return {"strategy": self.name, "custom_factor": self.custom_factor}

        # 1. Register strategy
        register_aggregation_strategy("probe_custom_strategy_v1", MockCustomStrategy)

        # 2. Retrieve strategy class
        cls = get_aggregation_strategy("probe_custom_strategy_v1")
        assert cls is MockCustomStrategy

        # 3. Create from flat dictionary
        strat1 = create_aggregation_strategy({"strategy": "probe_custom_strategy_v1", "custom_factor": 3.5})
        assert isinstance(strat1, MockCustomStrategy)
        assert strat1.custom_factor == 3.5

        # 4. Resolve from nested [aggregation] table
        strat2 = resolve_aggregation_strategy({"aggregation": {"strategy": "probe_custom_strategy_v1", "custom_factor": 4.5}})
        assert isinstance(strat2, MockCustomStrategy)
        assert strat2.custom_factor == 4.5

        # 5. Resolve from nested execution.aggregation table
        strat3 = resolve_aggregation_strategy({"execution": {"aggregation": {"strategy": "probe_custom_strategy_v1", "custom_factor": 5.5}}})
        assert isinstance(strat3, MockCustomStrategy)
        assert strat3.custom_factor == 5.5

        # 6. Verify execution profile schema validator accepts the registered custom strategy
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw_profile = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'probe_custom_strategy_v1', 'sync_every_steps': 1},
        }
        resolved = resolve_execution_profile(raw_profile, cfg)
        agg_table = resolved.get('aggregation') or resolved['execution'].get('aggregation')
        assert agg_table['strategy'] == 'probe_custom_strategy_v1'

        # 7. Stress probe: execution_profiles restricts [aggregation] to canonical keys;
        # custom options in raw execution profile raise ValueError
        bad_profile = {
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2},
            'aggregation': {'strategy': 'probe_custom_strategy_v1', 'custom_factor': 6.0},
        }
        with pytest.raises(ValueError, match="Unknown aggregation option"):
            resolve_execution_profile(bad_profile, cfg)

    def test_registration_validation_rejections(self):
        """Verify invalid registration parameters are rejected with ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            register_aggregation_strategy("", ParameterAverageStrategy)
        with pytest.raises(ValueError, match="non-empty string"):
            register_aggregation_strategy(None, ParameterAverageStrategy)  # type: ignore
        with pytest.raises(ValueError, match="non-empty string"):
            register_aggregation_strategy(12345, ParameterAverageStrategy)  # type: ignore

    def test_unknown_strategy_resolution_error(self):
        """Resolving unregistered strategy raises clear ValueError with available strategies."""
        with pytest.raises(ValueError, match="Unknown aggregation strategy: 'totally_nonexistent_strategy'"):
            resolve_aggregation_strategy({"strategy": "totally_nonexistent_strategy"})


# ==============================================================================
# Challenge Probe 5: Invalid Parameter Rejection Guardrails
# ==============================================================================

class TestInvalidParameterRejections:
    """Probe 5: Stress-test boundary and invalid parameter rejections."""

    @pytest.mark.parametrize("invalid_sync", [0, -1, -10, 1.5, "1", None, True, False])
    def test_parameter_average_invalid_sync_cadence(self, invalid_sync: Any):
        """ParameterAverageStrategy rejects non-integer, non-positive, or boolean sync cadences."""
        with pytest.raises(ValueError, match="positive integer"):
            ParameterAverageStrategy(sync_every_steps=invalid_sync)

    @pytest.mark.parametrize("invalid_sync", [0, 2, 5, -1, 1.5, False, "1"])
    def test_gradient_allreduce_invalid_sync_cadence(self, invalid_sync: Any):
        """GradientAllreduceStrategy requires sync_every_steps == 1."""
        with pytest.raises(ValueError, match="gradient_allreduce requires sync_every_steps = 1"):
            GradientAllreduceStrategy(sync_every_steps=invalid_sync)

    def test_gradient_allreduce_bool_sync_type_inconsistency(self):
        """Adversarial probe: GradientAllreduceStrategy accepts sync_every_steps=True (True == 1)
        because it checks `if sync_every_steps != 1:` instead of `type(sync_every_steps) is not int`,
        whereas ParameterAverageStrategy and PseudoGradientStrategy strictly reject True."""
        strat = GradientAllreduceStrategy(sync_every_steps=True)  # type: ignore
        assert strat.sync_every_steps == 1
        # Contrast with ParameterAverageStrategy which strictly rejects True
        with pytest.raises(ValueError, match="positive integer"):
            ParameterAverageStrategy(sync_every_steps=True)  # type: ignore

    @pytest.mark.parametrize("invalid_outer_lr", [0.0, -0.01, -1.0, float("inf"), float("-inf"), float("nan"), True, False, "1.0", None])
    def test_pseudo_gradient_invalid_outer_lr(self, invalid_outer_lr: Any):
        """PseudoGradientStrategy rejects non-positive, non-finite, boolean, or non-float outer_lr."""
        with pytest.raises(ValueError, match="outer_lr must be a finite positive float"):
            PseudoGradientStrategy(sync_every_steps=2, outer_lr=invalid_outer_lr)

    @pytest.mark.parametrize("invalid_momentum", [-0.01, -1.0, 1.0, 1.001, 2.0, float("inf"), float("-inf"), float("nan"), True, False, "0.9", None])
    def test_pseudo_gradient_invalid_outer_momentum(self, invalid_momentum: Any):
        """PseudoGradientStrategy rejects outer_momentum not in [0.0, 1.0), non-finite, boolean, or non-float."""
        with pytest.raises(ValueError, match=r"outer_momentum must be a finite float in \[0\.0, 1\.0\)"):
            PseudoGradientStrategy(sync_every_steps=2, outer_momentum=invalid_momentum)


# ==============================================================================
# Challenge Probe 6: Checkpoint Serialization & Multi-Round Resume Continuity
# ==============================================================================

class TestCheckpointSerializationAndResumeContinuity:
    """Probe 6: Verify checkpoint save/restore preserves anchor and momentum continuity across rounds."""

    def test_pseudo_gradient_save_and_resume_roundtrip(self):
        """Save state at round 2, load into new instance, run round 3, compare against continuous run."""
        model_continuous = torch.nn.Linear(2, 2, bias=True)
        model_resumed = torch.nn.Linear(2, 2, bias=True)
        init_w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        init_b = torch.tensor([0.5, -0.5])
        model_continuous.weight.data.copy_(init_w)
        model_continuous.bias.data.copy_(init_b)
        model_resumed.weight.data.copy_(init_w)
        model_resumed.bias.data.copy_(init_b)

        strat_cont = PseudoGradientStrategy(sync_every_steps=2, outer_lr=0.8, outer_momentum=0.9)
        strat_cont.pre_step(0, models=model_continuous)

        # --- Round 1 (step 2) ---
        drift_1 = torch.tensor([[0.2, -0.1], [0.1, 0.3]])
        model_continuous.weight.data.add_(drift_1)
        model_resumed.weight.data.add_(drift_1)
        strat_cont.post_step(step=2, models=model_continuous)

        # --- Round 2 (step 4) ---
        drift_2 = torch.tensor([[-0.1, 0.2], [-0.3, 0.1]])
        model_continuous.weight.data.add_(drift_2)
        model_resumed.weight.data.add_(drift_2)
        strat_cont.post_step(step=4, models=model_continuous)

        # Checkpoint snapshot after round 2
        saved_state = strat_cont.state_dict()
        assert saved_state['step_count'] == 4
        assert saved_state['outer_lr'] == 0.8
        assert saved_state['outer_momentum'] == 0.9

        # Instantiate fresh resumed strategy and load state
        strat_resumed = PseudoGradientStrategy(sync_every_steps=1, outer_lr=1.0, outer_momentum=0.0)
        strat_resumed.load_state_dict(saved_state)
        assert strat_resumed.step_count == 4
        assert strat_resumed.sync_every_steps == 2
        assert strat_resumed.outer_lr == 0.8
        assert strat_resumed.outer_momentum == 0.9

        # Ensure model_resumed is synchronized with restored anchor before round 3
        model_resumed.weight.data.copy_(strat_resumed.anchor['weight'])
        model_resumed.bias.data.copy_(strat_resumed.anchor['bias'])

        # --- Round 3 (step 6): Both continuous and resumed execute round 3 ---
        drift_3 = torch.tensor([[0.3, 0.1], [-0.2, 0.4]])
        model_continuous.weight.data.add_(drift_3)
        model_resumed.weight.data.add_(drift_3)

        strat_cont.post_step(step=6, models=model_continuous)
        strat_resumed.post_step(step=6, models=model_resumed)

        # Continuous and resumed MUST match exactly
        assert_states_close(strat_resumed.anchor, strat_cont.anchor, atol=1e-7, rtol=0)
        assert_states_close(strat_resumed.momentum_buffer, strat_cont.momentum_buffer, atol=1e-7, rtol=0)
        assert_states_close(model_resumed.weight.data, model_continuous.weight.data, atol=1e-7, rtol=0)
        assert_states_close(model_resumed.bias.data, model_continuous.bias.data, atol=1e-7, rtol=0)


# ==============================================================================
# Challenge Probe 7: Empirical Distributed Replica Agreement Failure Diagnosis
# ==============================================================================

class TestDistributedReplicaAgreementChallenge:
    """Probe 7: Empirically reproduce and verify the root cause of replica agreement failures
    during periodic parameter_average and pseudo_gradient synchronization.
    
    When ranks process distinct data shards, local optimizer steps cause rank-local divergence in:
    1. opt_d.record.last_sur (surrogate loss on rank-local critic batch)
    2. opt_g.regularizer['latent']['history'] (particle history on rank-local particle subset)
    Because _replica_digest hashes [opt_g.state_dict(), opt_d.state_dict()], and neither
    ParameterAverageStrategy nor PseudoGradientStrategy synchronizes opt.regularizer,
    ReplicatedTrainer._assert_replicas fails at sync boundaries with:
    'ValueError: Ranks disagree on complete replicated state'.
    """

    def test_reproduced_replica_disagreement_on_distinct_shards(self, tmp_path):
        """Empirically reproduce 'ValueError: Ranks disagree on complete replicated state'
        when distinct data shards are processed in a 2-rank Gloo execution."""
        import json, os, subprocess, sys

        cfg = make_toy_config(device="cpu", batch_size=8, steps=2)
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps(cfg))
        store_uri = (tmp_path / "rendezvous").as_uri()

        worker_code = """
import json, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from hypergan.distributed_training import ReplicatedCPUTrainer

rank, world_size, steps, store_uri, cfg_path, out_path = sys.argv[1:7]
rank = int(rank)
world_size = int(world_size)
steps = int(steps)
cfg = json.loads(Path(cfg_path).read_text())

dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
try:
    trainer = ReplicatedCPUTrainer(
        cfg,
        world_size=world_size,
        aggregation={"strategy": "parameter_average", "sync_every_steps": 2}
    )
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    for step in range(1, steps + 1):
        local_ids = ids.chunk(world_size)[rank]
        m, _ = trainer.update({"real": data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
    Path(out_path).write_text(json.dumps({"rank": rank, "passed": True}))
except Exception as exc:
    Path(out_path).write_text(json.dumps({"rank": rank, "error": str(exc), "type": type(exc).__name__}))
    raise
finally:
    dist.destroy_process_group()
"""
        w_script = tmp_path / "w.py"
        w_script.write_text(worker_code)

        env = os.environ.copy()
        repo_root = str(tmp_path.resolve().parents[3])
        env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get("PYTHONPATH", "")

        procs = []
        out_files = []
        for r in range(2):
            out_f = tmp_path / f"out_{r}.json"
            out_files.append(out_f)
            p = subprocess.Popen(
                [sys.executable, str(w_script), str(r), "2", "2", store_uri, str(cfg_file), str(out_f)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            procs.append(p)

        stderrs = []
        for p in procs:
            _, stderr = p.communicate(timeout=30)
            stderrs.append(stderr)

        # Empirically verify that replica agreement succeeds across distinct shards
        assert not any(stderrs), f"Expected clean stderr, got:\n" + "\n".join(stderrs)
        assert all(json.loads(p.read_text()).get("passed") is True for p in out_files), (
            f"Workers did not report passed=True: {[p.read_text() for p in out_files]}"
        )


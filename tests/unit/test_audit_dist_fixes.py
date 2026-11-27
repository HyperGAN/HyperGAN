"""Comprehensive unit and 2-rank Gloo tests for distributed, aggregation, execution, and checkpoint audit fixes."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from hypergan import checkpoints as ckpt_mod
from hypergan.aggregation import (
    PseudoGradientStrategy,
    _synchronize_optimizer_moments,
    _synchronize_optimizer_regularizers,
    _synchronize_optimizers,
    sync_axiom_record,
    sync_empf_record,
    sync_unified_v11_record,
)
from hypergan.api import StopTraining
from hypergan.checkpoints import (
    MAX_METADATA_BYTES,
    _prune_lock,
    _prune_run_checkpoints,
    extract_ema_bundle,
    read_checkpoint,
    resolve_checkpoint,
    resolve_resume_checkpoint_dir,
    restore_trainer_cross_topology,
    write_checkpoint,
)
from hypergan.config import config_values, resolve_config
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_checkpoints import (
    KIND as DIST_KIND,
    MAX_RANK_BYTES,
    _read as read_dist_internal,
    _shared_digest,
    read_distributed_checkpoint_for_elastic,
    save_distributed_checkpoint,
)
from hypergan.distributed_training import (
    ReplicatedCPUTrainer,
    ReplicatedTrainer,
    _BroadcastProbeProxy,
)
from hypergan.execution import (
    _apply_execution_options,
    _set_visible_devices,
    prepare_train,
)
from hypergan.execution_profiles import resolve_execution_profile
from hypergan.particlegan_runtime import _set_thermal_scale
from hypergan.replicated_execution import ReplicatedExecution
from hypergan.replicated_worker import handle_command
from hypergan.run_controller import AttemptContext
from hypergan.training import ReferenceTrainer
from tests.config_fixtures import write_default


# ---------------------------------------------------------------------------
# Helpers & top-level spawn workers
# ---------------------------------------------------------------------------


def _assert_nested_close(left, right, path="", *, rtol=3e-5, atol=3e-6):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor):
            left = torch.tensor(left)
        if not isinstance(right, torch.Tensor):
            right = torch.tensor(right)
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=lambda m: f"{path}: {m}")
    elif isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, rel=rtol, abs=atol), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"{path}: {left.keys()} != {right.keys()}"
        for k in left:
            _assert_nested_close(left[k], right[k], f"{path}.{k}", rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{path}: len {len(left)} != {len(right)}"
        for i, (a, b) in enumerate(zip(left, right)):
            _assert_nested_close(a, b, f"{path}[{i}]", rtol=rtol, atol=atol)
    else:
        assert left == right, path


class _CriticWithBN(torch.nn.Module):
    observed_track_flags: list[bool] = []

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(in_features)
        # Start eval so _check_modules() passes at init; worker toggles train() to exercise G-step BN freeze
        self.bn.eval()
        self.head = torch.nn.Linear(in_features, 1, bias=False)

    def forward(self, input):
        _CriticWithBN.observed_track_flags.append(bool(self.bn.track_running_stats))
        return self.head(self.bn(input))


class _CollapsedGenerator(torch.nn.Module):
    """Generator producing an identical non-flat vector for every sample in the batch."""

    def __init__(self, in_features: int = 4, out_features: int = 4):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.tensor([-1.0, 0.5, 1.5, -0.5][:out_features]))

    def forward(self, input):
        return (input * 0.0) @ self.w.t() + self.bias.unsqueeze(0)


class _GradSpyGenerator(torch.nn.Module):
    """Records whether torch.is_grad_enabled() during each forward call."""

    grad_flags: list[bool] = []

    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, input):
        _GradSpyGenerator.grad_flags.append(torch.is_grad_enabled())
        return self.linear(input)


def _cb_return_false(event):
    if event.get("event") == "train" and event.get("step", 0) >= 1:
        return False
    return None


def _cb_raise_stop_training(event):
    if event.get("event") == "train" and event.get("step", 0) >= 1:
        raise StopTraining("budget reached")
    return None


def _worker_parity_generic(
    rank,
    world_size,
    config_dict,
    full_batch,
    shared_dir,
    accumulation_steps=1,
    steps=1,
    enable_critic_bn_train=False,
    deterministic_noise_offset=None,
):
    import hypergan.distributed_training as dt_mod
    import hypergan.training as tr_mod

    orig_dt_noise = dt_mod.with_noise
    orig_tr_noise = tr_mod.with_noise
    noise_calls = []
    if deterministic_noise_offset is not None:
        def det_noise(tensor, std, stream):
            if std > 0:
                noise_calls.append(float(std))
                return tensor + float(std) * float(deterministic_noise_offset)
            return tensor
        dt_mod.with_noise = det_noise
        tr_mod.with_noise = det_noise

    _CriticWithBN.observed_track_flags = []
    try:
        cfg = resolve_config(config_dict)
        trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, accumulation_steps=accumulation_steps)
        disc = trainer.graph.models["discriminator"] if "discriminator" in trainer.graph.models else None
        if enable_critic_bn_train and disc is not None and hasattr(disc, "bn"):
            trainer._check_modules = lambda: None
            if accumulation_steps == 1:
                disc.bn.train(True)
            else:
                # For accumulated_update, discovery checks buffers don't mutate during D discovery;
                # switch BN to train() right before G step via opt_d.step wrapper to prove G-step freezes BN!
                orig_d_step = trainer.opt_d.step

                def _d_step_and_enable_bn(*a, **kw):
                    res = orig_d_step(*a, **kw)
                    disc.bn.train(True)
                    return res

                trainer.opt_d.step = _d_step_and_enable_bn
        metrics_list = []
        for _ in range(steps):
            local_batch = {k: v.chunk(world_size)[rank] for k, v in full_batch.items() if not k.startswith("_")}
            ids = full_batch.get("_ids")
            if ids is not None and hasattr(trainer.prior, "z"):
                local_ids = ids.chunk(world_size)[rank]
                prior_arg = (trainer.prior.z[local_ids], local_ids)
            elif "_z" in full_batch:
                prior_arg = (full_batch["_z"].chunk(world_size)[rank], None)
            else:
                prior_arg = None
            if prior_arg is not None:
                m, _ = trainer.update(local_batch, prior_arg)
            else:
                m, _ = trainer.update(local_batch)
            metrics_list.append(m)
        if not enable_critic_bn_train:
            trainer._assert_replicas("parity check")
        bn_tracked = None
        bn_track_flag = None
        if disc is not None and hasattr(disc, "bn"):
            bn_tracked = int(disc.bn.num_batches_tracked.item())
            bn_track_flag = bool(disc.bn.track_running_stats)
        torch.save(
            {
                "metrics": metrics_list,
                "graph": copy.deepcopy(trainer.graph.state_dict()),
                "prior": copy.deepcopy(trainer.prior.state_dict()),
                "digest": trainer._replica_digest(),
                "bn_tracked": bn_tracked,
                "bn_track_flag": bn_track_flag,
                "bn_observed_flags": list(_CriticWithBN.observed_track_flags),
                "noise_calls": list(noise_calls),
            },
            Path(shared_dir) / f"rank-{rank}.pt",
        )
    finally:
        dt_mod.with_noise = orig_dt_noise
        tr_mod.with_noise = orig_tr_noise


def _worker_unified_v15_highd(rank, world_size, config_dict, full_batches, shared_dir):
    cfg = resolve_config(config_dict)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, accumulation_steps=1)
    assert trainer.recipe.name == "unified_v15"
    for fb in full_batches:
        local_batch = {"real": fb["real"].chunk(world_size)[rank]}
        local_ids = fb["_ids"].chunk(world_size)[rank]
        trainer.update(local_batch, (trainer.prior.z[local_ids], local_ids))
    trainer._assert_replicas("unified_v15 replica agreement")
    rec = trainer.opt_d.record
    torch.save(
        {
            "digest": trainer._replica_digest(),
            "highd_packet": rec.highd_basin_bank.to_packet(),
            "sparse_packet": rec.sparse_obs_bank.to_packet(),
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def _worker_unified_v15_sync_states(rank, world_size, shared_dir):
    from particlegan._unified_v15.critic import UnifiedV15StepRecord
    from particlegan._unified_v15.optimizer import UnifiedV15Optimizer

    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = UnifiedV15Optimizer([p], lr=0.01)
    rec = UnifiedV15StepRecord()
    opt.record = rec

    # Diverge per-param peak_rms_val, _g_peak_ref/_g_peak_ref_t, _v15_g_peak_ref, atomic_ledger, banks
    opt.state[p]["step"] = 1
    opt.state[p]["exp_avg"] = torch.full_like(p, 1.0 if rank == 0 else 3.0)
    opt.state[p]["exp_avg_sq"] = torch.full_like(p, 2.0 if rank == 0 else 4.0)
    opt.state[p]["peak_rms_val"] = 0.25 if rank == 0 else 0.75
    opt._g_peak_ref = 0.2 if rank == 0 else 0.8
    opt._g_peak_ref_t = torch.tensor(0.4 if rank == 0 else 0.8)
    rec._v15_g_peak_ref = 0.5 if rank == 0 else 1.5
    rec.t1c_ell_initialized = True

    if rank == 1:
        # Rank 1 initialized atomic_ledger, highd_basin_bank, sparse_obs_bank while rank 0 is uninitialized!
        rec.atomic_ledger_initialized = True
        rec.atomic_row_chart = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        rec.atomic_row_dist = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        rec.atomic_row_target = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        rec.highd_basin_bank.sigma_z_init = True
        rec.highd_basin_bank.sigma_z_ema = 1.75
        rec.highd_basin_bank.dim = 32
        rec.sparse_obs_bank.step = 5
        rec.sparse_obs_bank.dim = 4
        rec.sparse_obs_bank.dz = 0
        rec.sparse_obs_bank.buf_n = 2
        rec.sparse_obs_bank.buf_ptr = 2
        rec.sparse_obs_bank.p_y = 0.5
        rec.sparse_obs_bank.p_x = 0.5
        rec.sparse_obs_bank.sparse_push_res = 0.42
        rec.sparse_obs_bank.mu_y = torch.zeros(4, dtype=torch.float32)
        rec.sparse_obs_bank.mu_x = torch.zeros(4, dtype=torch.float32)
        rec.sparse_obs_bank.S_y = torch.eye(4, dtype=torch.float32)
        rec.sparse_obs_bank.S_x = torch.eye(4, dtype=torch.float32)
        rec.sparse_obs_bank.buf_y = torch.ones((2, 4), dtype=torch.float32)
        rec.sparse_obs_bank.buf_m = torch.ones((2, 4), dtype=torch.bool)

    _synchronize_optimizers(None, [opt], world_size)
    # Validate state_dict cleanly
    sd = opt.state_dict()
    opt.validate_state_dict(sd)

    torch.save(
        {
            "peak_rms_val": opt.state[p]["peak_rms_val"],
            "g_peak_ref": opt._g_peak_ref,
            "g_peak_ref_t": opt._g_peak_ref_t,
            "v15_g_peak_ref": rec._v15_g_peak_ref,
            "atomic_init": rec.atomic_ledger_initialized,
            "atomic_row_chart": None if rec.atomic_row_chart is None else rec.atomic_row_chart.tolist(),
            "highd_sigma_z": rec.highd_basin_bank.sigma_z_ema,
            "sparse_step": rec.sparse_obs_bank.step,
            "sparse_push_res": rec.sparse_obs_bank.sparse_push_res,
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def _worker_hot_path_agree(rank, world_size, config_dict, full_batch, shared_dir):
    cfg = resolve_config(config_dict)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, accumulation_steps=1)
    local_batch = {k: v.chunk(world_size)[rank] for k, v in full_batch.items() if not k.startswith("_")}
    local_z = full_batch["_z"].chunk(world_size)[rank]

    # Step 1 (cold path)
    trainer.update(local_batch, (local_z, None))

    # Step 2 (hot path): spy on dist.all_gather_object to verify 0 calls (including matching _agree!)
    orig_gather = dist.all_gather_object
    gather_calls = []

    def spy_gather(*args, **kwargs):
        gather_calls.append(args[1] if len(args) > 1 else kwargs)
        return orig_gather(*args, **kwargs)

    dist.all_gather_object = spy_gather
    try:
        trainer.update(local_batch, (local_z, None))
        assert trainer._in_hot_path is True
        trainer._agree("hot_matching", ("ok", 42))
    finally:
        dist.all_gather_object = orig_gather
    step2_gathers = list(gather_calls)

    # Step 3 (still hot path!): standalone _agree() mismatch MUST raise ValueError, NOT silently skip!
    caught_agree = None
    assert trainer._in_hot_path is True
    try:
        trainer._agree("hot_mismatch", f"rank-{rank}")
    except ValueError as exc:
        caught_agree = str(exc)

    # And real-shape disagreement during hot-path update MUST also raise (never silently skip!)
    bad_batch = dict(local_batch)
    if rank == 1:
        bad_batch["real"] = torch.zeros(local_batch["real"].shape[0], local_batch["real"].shape[1] + 1)
    caught_update = None
    try:
        trainer.update(bad_batch, (local_z, None))
    except (ValueError, RuntimeError) as exc:
        caught_update = str(exc)

    torch.save(
        {"step2_gathers": step2_gathers, "caught_agree": caught_agree, "caught_update": caught_update},
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def _worker_asymmetric_moments(rank, world_size, shared_dir):
    p = torch.nn.Parameter(torch.randn(3, 3))
    opt = torch.optim.Adam([p], lr=0.01)
    opt.state[p]["step"] = torch.tensor(float(rank + 1))
    opt.state[p]["exp_avg"] = torch.full_like(p, 1.0 if rank == 0 else 5.0)
    opt.state[p]["exp_avg_sq"] = torch.full_like(p, 2.0 if rank == 0 else 6.0)
    if rank == 1:
        # Lazy Axiom / v15 keys present ONLY on rank 1
        opt.state[p]["v_pow"] = torch.full_like(p, 3.5)
        opt.state[p]["log2_ref"] = -2.25
        opt.state[p]["log2_init"] = True
        opt.state[p]["peak_rms_val"] = 0.65

    _synchronize_optimizer_moments([opt], world_size, torch.device("cpu"))
    torch.save(
        {
            "step": float(opt.state[p]["step"].item()),
            "exp_avg": opt.state[p]["exp_avg"].tolist(),
            "exp_avg_sq": opt.state[p]["exp_avg_sq"].tolist(),
            "v_pow": opt.state[p]["v_pow"].tolist(),
            "log2_ref": opt.state[p]["log2_ref"],
            "log2_init": opt.state[p]["log2_init"],
            "peak_rms_val": opt.state[p]["peak_rms_val"],
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def _worker_rank0_uninitialized_records(rank, world_size, shared_dir):
    from particlegan._axiom.critic import AxiomStepRecord
    from particlegan._empf import EMPFStepRecord
    from particlegan._unified_v11.critic import UnifiedV11StepRecord

    dev = torch.device("cpu")

    # 1. EMPF: rank 0 uninitialized, rank 1 initialized
    empf = EMPFStepRecord()
    if rank == 1:
        empf.initialized = True
        empf.proto_initialized = True
        empf.proto_sum = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        empf.proto_count = torch.tensor([2.0, 5.0])
    sync_empf_record(empf, dev, world_size)

    # 2. UnifiedV11: rank 0 uninitialized chart + cond_fiber_bank, rank 1 initialized
    v11 = UnifiedV11StepRecord()
    if rank == 1:
        v11.t1c_ell_initialized = True
        v11.chart_initialized = True
        v11.chart_sum = torch.tensor([[1.0, 2.0, 3.0]])
        v11.chart_count = torch.tensor([4.0])
        v11.chart_sq_dev = torch.tensor([0.5])
        v11.chart_fake_mass = torch.tensor([0.25])
        v11.cond_fiber_bank.update_stratum(
            (0,),
            torch.randn(4, 1, dtype=torch.float64),
            torch.randn(4, 2, dtype=torch.float64),
            torch.randn(4, 2, dtype=torch.float64),
        )
    sync_unified_v11_record(v11, dev, world_size)

    # 3a. Axiom: rank 0 uninitialized, rank 1 initialized
    ax1 = AxiomStepRecord()
    if rank == 1:
        ax1.chart_state = {"centers": torch.tensor([[1.0, -1.0]])}
        ax1.atomic_ledger = {"row_chart": torch.tensor([0, 1])}
        ax1.cond_fiber_bank = {"strata": {"k": 1}}
    sync_axiom_record(ax1, dev, world_size)

    # 3b. Axiom: stale _cond_bank_obj cleared when synced cond_fiber_bank is None
    ax2 = AxiomStepRecord()
    if rank == 1:
        ax2._cond_bank_obj = SimpleNamespace(to_packet=lambda: None, load_packet=lambda pkt: None)
        ax2.cond_fiber_bank = None
    sync_axiom_record(ax2, dev, world_size)

    torch.save(
        {
            "empf_init": empf.proto_initialized,
            "empf_sum": empf.proto_sum.tolist(),
            "empf_cnt": empf.proto_count.tolist(),
            "v11_chart_init": v11.chart_initialized,
            "v11_chart_sum": v11.chart_sum.tolist(),
            "v11_strata_keys": sorted(v11.cond_fiber_bank.strata.keys()),
            "v11_strata_neff": v11.cond_fiber_bank.strata[(0,)]["n_eff"] if (0,) in v11.cond_fiber_bank.strata else None,
            "ax1_chart": ax1.chart_state["centers"].tolist(),
            "ax1_ledger": ax1.atomic_ledger["row_chart"].tolist(),
            "ax1_bank": ax1.cond_fiber_bank,
            "ax2_bank": ax2.cond_fiber_bank,
            "ax2_bank_obj": ax2._cond_bank_obj,
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def _worker_local_sgd_extra_states(rank, world_size, shared_dir):
    from particlegan.k3p import CriticSpikeGuard, DirectParticleResponse, LatentRowDamping
    from particlegan.ka2 import KA2StepRecord
    from particlegan.optim.direction_blend import DirectionBlendOptimizer

    # DirectionBlendOptimizer (includes constraint_geometry_stats & direction_blend_stats)
    p_g = torch.nn.Parameter(torch.randn(4, 4))
    opt_g = DirectionBlendOptimizer([p_g], lr=0.01)
    opt_g.constraint_geometry_stats = {
        "steps": 3 if rank == 0 else 7,
        "projected_steps": 1 if rank == 0 else 4,
        "max_derivative_before": 0.2 if rank == 0 else 0.9,
        "max_derivative_after": 0.05 if rank == 0 else 0.15,
    }
    opt_g.direction_blend_stats = {
        "conflict_steps": 2 if rank == 0 else 5,
        "blended_steps": 1 if rank == 0 else 3,
        "pareto_stalls": 0 if rank == 0 else 2,
        "retained_norm_ratio_sum": 1.0 if rank == 0 else 3.0,
        "max_retained_norm_ratio": 0.4 if rank == 0 else 0.8,
    }

    # K3P LatentRowDamping & DirectParticleResponse
    table = torch.nn.Parameter(torch.randn(4, 2))
    l_hist = torch.full((4, 2), 1.0 if rank == 0 else 3.0)
    ld = LatentRowDamping(table, l_hist)
    ld.observed = 2 if rank == 0 else 4
    ld.total = 4 if rank == 0 else 6
    ld.started = (rank == 1)

    d_param = torch.nn.Parameter(torch.randn(3))
    d_hist = torch.full((3,), 2.0 if rank == 0 else 6.0)
    dr = DirectParticleResponse([d_param], d_hist)
    dr.started = (rank == 1)
    dr.step_count = 3 if rank == 0 else 5
    dr.last_gain = 0.5 if rank == 0 else 1.5
    opt_k3p = SimpleNamespace(
        latent_history=l_hist,
        direct_history=d_hist,
        latent_damping=ld,
        direct_response=dr,
    )

    # KA2StepRecord & CriticSpikeGuard
    ka2 = KA2StepRecord(trust_mode="scale")
    ka2.calls = 10 if rank == 0 else 12
    ka2.observed_steps = 8 if rank == 0 else 9
    ka2.lr_max = 0.02 if rank == 0 else 0.03
    ka2.lr_last = 0.01 if rank == 0 else 0.015
    ka2.last_sur = 1.0 if rank == 0 else 3.0
    ka2.sur_hist = [0.5, 0.6] if rank == 0 else [0.9]
    ka2.sur_base = 0.55 if rank == 0 else 0.95
    ka2.w = 1.0 if rank == 0 else 0.0
    ka2.low_streak = 2 if rank == 0 else 5
    ka2.alpha = 0.2 if rank == 0 else 0.6
    ka2.alpha_raw = 0.4 if rank == 0 else 0.8
    ka2.last_ratio = 0.7 if rank == 0 else 1.1
    ka2.ema_updates = 4 if rank == 0 else 7
    ka2.ema_skips = 1 if rank == 0 else 2
    ka2.ema_reseeds = 0 if rank == 0 else 1

    guard = CriticSpikeGuard()
    guard._clipped = 1 if rank == 0 else 4
    opt_ka2 = SimpleNamespace(record=ka2, guard=guard)

    _synchronize_optimizer_regularizers(None, [opt_g, opt_k3p, opt_ka2])
    opt_g.validate_state_dict(opt_g.state_dict())

    torch.save(
        {
            "cg": dict(opt_g.constraint_geometry_stats),
            "db": dict(opt_g.direction_blend_stats),
            "ld": ld.state_dict(),
            "dr": {**dr.state_dict(), "last_gain": dr.last_gain},
            "ka2": ka2.state_dict(),
            "guard": guard.state_dict(),
        },
        Path(shared_dir) / f"rank-{rank}.pt",
    )


def _worker_independent_no_grad(rank, world_size, config_dict, full_batch, shared_dir):
    _GradSpyGenerator.grad_flags = []
    cfg = resolve_config(config_dict)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, accumulation_steps=1)
    local_batch = {k: v.chunk(world_size)[rank] for k, v in full_batch.items() if not k.startswith("_")}
    trainer.update(local_batch)
    torch.save({"grad_flags": list(_GradSpyGenerator.grad_flags)}, Path(shared_dir) / f"rank-{rank}.pt")


def _worker_collapsed_diversity(rank, world_size, config_dict, full_batch, shared_dir):
    cfg = resolve_config(config_dict)
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, accumulation_steps=1)
    local_batch = {k: v.chunk(world_size)[rank] for k, v in full_batch.items() if not k.startswith("_")}
    metrics, _ = trainer.update(local_batch)
    torch.save({"diversity_ratio": metrics["diversity/ratio"]}, Path(shared_dir) / f"rank-{rank}.pt")


# ---------------------------------------------------------------------------
# Tests: Item 1 (Double output_noise) & Item 2 (Critic BN freeze during G)
# ---------------------------------------------------------------------------


def test_output_noise_not_doubled_and_matches_reference(tmp_path, monkeypatch):
    import hypergan.training as tr_mod

    cfg = {
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": True},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "objectives": [
            {"factory": "mse", "inputs": {"input": "generated", "target": "batch.real"}, "weight": 0.5}
        ],
        "training": {
            "steps": 2,
            "batch_size": 8,
            "device": "cpu",
            "seed": 123,
            "output_noise_std": 0.25,
            "output_noise_warmup": 0.0,
        },
    }
    full_batch = {
        "real": torch.linspace(-1.0, 1.0, 16).reshape(8, 2),
        "_ids": torch.arange(8, dtype=torch.long),
    }

    # Use deterministic offset noise so rank-local RNG streams don't mask single-vs-double noise injection!
    ref_noise_calls = []

    def det_noise(tensor, std, stream):
        if std > 0:
            ref_noise_calls.append(float(std))
            return tensor + float(std) * 1.0
        return tensor

    monkeypatch.setattr(tr_mod, "with_noise", det_noise)
    ref = ReferenceTrainer(resolve_config(copy.deepcopy(cfg)))
    ref_m, _ = ref.update({"real": full_batch["real"]}, (ref.prior.z[full_batch["_ids"]], full_batch["_ids"]))

    launch_cpu_workers(
        _worker_parity_generic,
        args=(cfg, full_batch, str(tmp_path), 1, 1, False, 1.0),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["digest"] == r1["digest"]
    # Exactly the same number of output_noise applications as ReferenceTrainer (not double!)
    assert r0["noise_calls"] == ref_noise_calls
    _assert_nested_close(r0["graph"], ref.graph.state_dict(), "graph")
    assert r0["metrics"][0]["d_loss"] == pytest.approx(ref_m["d_loss"], rel=3e-5, abs=3e-6)
    assert r0["metrics"][0]["g_loss"] == pytest.approx(ref_m["g_loss"], rel=3e-5, abs=3e-6)
    assert r0["metrics"][0]["objectives"] == pytest.approx(ref_m["objectives"], rel=3e-5, abs=3e-6)


@pytest.mark.parametrize("acc_steps", [1, 2])
def test_critic_batchnorm_frozen_during_generator_step(tmp_path, acc_steps):
    cfg = {
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "gradient_penalty": {"coeff": 0.0},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 4, "bias": True},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "tests.unit.test_audit_dist_fixes:_CriticWithBN",
                "args": {"in_features": 4},
                "inputs": {"input": "candidate"},
            },
        },
        "training": {"steps": 2, "batch_size": 8, "device": "cpu", "seed": 77},
    }
    gen = torch.Generator().manual_seed(99)
    half_real = torch.randn(4, 4, generator=gen)
    half_z = torch.randn(4, 4, generator=gen)
    full_batch = {
        "real": torch.cat([half_real, half_real], dim=0),
        "_z": torch.cat([half_z, half_z], dim=0),
    }

    launch_cpu_workers(
        _worker_parity_generic,
        args=(cfg, full_batch, str(tmp_path), acc_steps, 1, True),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["digest"] == r1["digest"]
    # G-step MUST observe track_running_stats == False and restore it to True afterwards!
    flags = r0["bn_observed_flags"]
    assert flags and flags[-1] is False
    assert r0["bn_track_flag"] is True
    if acc_steps == 1:
        # D-step forwards track BN stats (True); G-step forwards freeze track_running_stats (False)!
        d_forwards = sum(1 for flag in flags if flag)
        g_forwards = sum(1 for flag in flags if not flag)
        assert d_forwards >= 2 and g_forwards >= 1
        assert r0["bn_tracked"] == d_forwards
        assert r0["bn_tracked"] < len(flags)
    else:
        # BN was enabled right before G step; G step froze track_running_stats so tracked stayed 0!
        assert r0["bn_tracked"] == 0


# ---------------------------------------------------------------------------
# Tests: Item 3 (unified_v15 multi-GPU parity & state sync & probes)
# ---------------------------------------------------------------------------


def test_unified_v15_spatial_cutoff_and_probe_budget():
    from particlegan._unified_v11.critic import UnifiedV11StepRecord
    from particlegan._unified_v15.critic import UnifiedV15StepRecord

    v15_rec = UnifiedV15StepRecord()
    v11_rec = UnifiedV11StepRecord()
    v15_term = SimpleNamespace(gan=SimpleNamespace(record=v15_rec), routes=(object(),))
    v11_term = SimpleNamespace(gan=SimpleNamespace(record=v11_rec), routes=(object(),))

    img_32 = torch.zeros(4, 3, 32, 32)  # dim = 3072 <= 4096 (v15) but > 1024 (v11)
    img_40 = torch.zeros(4, 3, 40, 40)  # dim = 4800 > 4096
    assert ReplicatedTrainer._is_highres_image_term(v15_term, img_32) is False
    assert ReplicatedTrainer._is_highres_image_term(v15_term, img_40) is True
    assert ReplicatedTrainer._is_highres_image_term(v11_term, img_32) is True

    dummy = object.__new__(ReplicatedTrainer)
    # 64 samples, dim=3072 -> v15 budget min(32, 65536//3072)=21 vs v11 min(16, 8192//3072)=2
    assert dummy._probe_budget_for_record(v15_rec, 64, 3072) == 21
    assert dummy._probe_budget_for_record(v15_rec, 64, 64) == 32
    assert dummy._probe_budget_for_record(v11_rec, 64, 3072) == 2
    assert dummy._probe_budget_for_record(v11_rec, 64, 64) == 16

    # _BroadcastProbeProxy handles multiple reference shapes (multi-input D probes) & forwards setattr
    class _Mod(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=-1, keepdim=True)

    mod = _Mod()
    proxy = _BroadcastProbeProxy(mod, [torch.zeros(2, 1), torch.zeros(3, 1)])
    assert proxy._ref_shapes == [(2, 1), (3, 1)]
    proxy._unified_v15_needs_eval_pre = True
    assert mod._unified_v15_needs_eval_pre is True
    with patch.object(dist, "get_rank", return_value=0), patch.object(dist, "broadcast", lambda t, src=0: None):
        out1 = proxy(torch.ones(2, 4))
        out2 = proxy(torch.ones(3, 4))
    assert out1.shape == (2, 1) and out2.shape == (3, 1)


def test_unified_v15_highd_and_sparse_latent_gather_replica_agreement(tmp_path):
    # 32D vector triggers v15's highd_basin_bank (17..256D) which requires global _g_last_latent!
    cfg = {
        "defaults": "particlegan:unified_v15",
        "gradient_penalty": {"coeff": 0.0},
        "recipe": {"t1_pullback_weight": 0.0},
        "training": {"steps": 2, "batch_size": 8, "device": "cpu", "seed": 42},
        "prior": {"kind": "mog", "fixed_sigma": 0.1, "args": {"num_particles": 16, "z_dim": 8}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 8, "out_features": 32, "bias": True},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 32, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
    }
    gen = torch.Generator().manual_seed(11)
    batches = [
        {"real": torch.randn(8, 32, generator=gen), "_ids": torch.arange(8, dtype=torch.long)}
        for _ in range(2)
    ]
    launch_cpu_workers(
        _worker_unified_v15_highd,
        args=(cfg, batches, str(tmp_path)),
        world_size=2,
        timeout=60,
        collective_timeout=20,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["digest"] == r1["digest"]
    _assert_nested_close(r0["highd_packet"], r1["highd_packet"], "highd_basin_bank")


def test_unified_v15_local_sgd_state_sync(tmp_path):
    launch_cpu_workers(
        _worker_unified_v15_sync_states,
        args=(str(tmp_path),),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0 == r1
    assert r0["peak_rms_val"] == pytest.approx(0.5)
    assert r0["g_peak_ref"] == pytest.approx(0.6)
    assert r0["g_peak_ref_t"] is None
    assert r0["v15_g_peak_ref"] == pytest.approx(1.0)
    assert r0["atomic_init"] is True
    assert r0["atomic_row_chart"] == [0, 1, 2, 3]
    assert r0["highd_sigma_z"] == pytest.approx(1.75)
    assert r0["sparse_step"] == 5
    assert r0["sparse_push_res"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Tests: Item 4 (_in_hot_path agreement active with 0 all_gather_object)
# ---------------------------------------------------------------------------


def test_hot_path_keeps_agree_checks_active_without_all_gather_object(tmp_path):
    cfg = {
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "training": {"steps": 4, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    full_batch = {
        "real": torch.linspace(-1.0, 1.0, 16).reshape(8, 2),
        "_z": torch.linspace(-0.5, 0.5, 32).reshape(8, 4),
    }
    launch_cpu_workers(
        _worker_hot_path_agree,
        args=(cfg, full_batch, str(tmp_path)),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["step2_gathers"] == [] and r1["step2_gathers"] == [], (r0["step2_gathers"], r1["step2_gathers"])
    assert r0["caught_agree"] is not None and r1["caught_agree"] is not None
    assert "Ranks disagree" in r0["caught_agree"] and "Ranks disagree" in r1["caught_agree"]
    assert r0["caught_update"] is not None and r1["caught_update"] is not None


# ---------------------------------------------------------------------------
# Tests: Item 5 (Asymmetric lazy optimizer keys do not deadlock)
# ---------------------------------------------------------------------------


def test_synchronize_optimizer_moments_asymmetric_lazy_keys(tmp_path):
    launch_cpu_workers(
        _worker_asymmetric_moments,
        args=(str(tmp_path),),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["step"] == 1.0 and r1["step"] == 2.0  # step untouched
    assert r0["exp_avg"] == r1["exp_avg"]
    assert r0["v_pow"] == r1["v_pow"]
    assert torch.allclose(torch.tensor(r0["v_pow"]), torch.full((3, 3), 3.5))
    assert r0["log2_ref"] == pytest.approx(-2.25) and r1["log2_ref"] == pytest.approx(-2.25)
    assert r0["log2_init"] is True and r1["log2_init"] is True
    assert r0["peak_rms_val"] == pytest.approx(0.65) and r1["peak_rms_val"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Tests: Item 6 (EMPF coupled gathering parity)
# ---------------------------------------------------------------------------


def test_empf_coupled_gathering_matches_reference(tmp_path):
    cfg = {
        "defaults": "particlegan:empf",
        "gradient_penalty": {"coeff": 0.0},
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
        "prior": {"kind": "mog", "fixed_sigma": 0.1, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
    }
    full_batch = {
        "real": torch.linspace(-1.0, 1.0, 16).reshape(8, 2),
        "_ids": torch.arange(8, dtype=torch.long),
    }
    ref = ReferenceTrainer(resolve_config(copy.deepcopy(cfg)))
    ref_m, _ = ref.update({"real": full_batch["real"]}, (ref.prior.z[full_batch["_ids"]], full_batch["_ids"]))

    launch_cpu_workers(
        _worker_parity_generic,
        args=(cfg, full_batch, str(tmp_path), 1, 1),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["digest"] == r1["digest"]
    _assert_nested_close(r0["graph"], ref.graph.state_dict(), "empf_graph")
    assert r0["metrics"][0]["d_loss"] == pytest.approx(ref_m["d_loss"], rel=3e-5, abs=3e-6)
    assert r0["metrics"][0]["g_loss"] == pytest.approx(ref_m["g_loss"], rel=3e-5, abs=3e-6)


# ---------------------------------------------------------------------------
# Tests: Item 7 (Rank-0 uninitialized else branches) & Item 8 (Local-SGD states + thermal_locked)
# ---------------------------------------------------------------------------


def test_rank0_uninitialized_record_sync_and_cond_bank_clear(tmp_path):
    launch_cpu_workers(
        _worker_rank0_uninitialized_records,
        args=(str(tmp_path),),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0 == r1
    assert r0["empf_init"] is True
    assert r0["empf_sum"] == [[1.0, 2.0], [3.0, 4.0]]
    assert r0["v11_chart_init"] is True
    assert r0["v11_chart_sum"] == [[1.0, 2.0, 3.0]]
    assert r0["v11_strata_keys"] == [(0,)]
    assert r0["v11_strata_neff"] > 0.0
    assert r0["ax1_chart"] == [[1.0, -1.0]]
    assert r0["ax1_ledger"] == [0, 1]
    assert r0["ax1_bank"] == {"strata": {"k": 1}}
    assert r0["ax2_bank"] is None
    assert r0["ax2_bank_obj"] is None


def test_local_sgd_extra_optimizer_states_and_axiom_thermal_locked(tmp_path):
    from particlegan._axiom.critic import AxiomStepRecord

    ax = AxiomStepRecord()
    assert ax.thermal_locked is False
    _set_thermal_scale(ax, 1e-5)
    assert ax.thermal_scale == pytest.approx(1e-5)
    assert ax.thermal_locked is True
    _set_thermal_scale(ax, 0.5)
    assert ax.thermal_locked is False

    launch_cpu_workers(
        _worker_local_sgd_extra_states,
        args=(str(tmp_path),),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0 == r1
    assert r0["cg"]["steps"] == 7 and r0["cg"]["projected_steps"] == 4
    assert r0["db"]["conflict_steps"] == 5 and r0["db"]["retained_norm_ratio_sum"] == pytest.approx(2.0)
    assert r0["ld"] == {"observed": 4, "total": 6, "started": True}
    assert r0["dr"]["started"] is True and r0["dr"]["step_count"] == 5 and r0["dr"]["last_gain"] == pytest.approx(1.0)
    assert r0["ka2"]["alpha"] == pytest.approx(0.4) and r0["ka2"]["last_sur"] == pytest.approx(2.0)
    assert r0["guard"]["clipped_tensors"] == 4


# ---------------------------------------------------------------------------
# Tests: Item 9 (Checkpoint security & hardening)
# ---------------------------------------------------------------------------


def _make_simple_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = resolve_config({"training": {"steps": 4, "batch_size": 8, "device": "cpu", "seed": 1}})
    trainer = ReferenceTrainer(cfg)
    _, batch = trainer.update()
    from hypergan.training import _implementation, _recovery_contract, runtime_info
    from hypergan.checkpoint_compatibility import CURRENT_VERSION
    from hypergan.config import fingerprint

    contract, _ = _recovery_contract(trainer)
    meta = {
        "run_id": "run-sec",
        "attempt_id": "att-1",
        "next_sample_sequence": 1,
        "config": config_values(cfg),
        "config_sha256": fingerprint(cfg),
        "runtime": runtime_info(trainer.device),
        "data_contract": contract,
        "implementation": _implementation(trainer),
        "hypergan_checkpoint_version": CURRENT_VERSION,
    }
    ckpt_dir = write_checkpoint(run_dir, trainer, batch, meta)
    return run_dir, ckpt_dir


def test_checkpoint_security_symlinks_traversal_bounds_sha256_and_prune(tmp_path, monkeypatch):
    run_dir, ckpt_dir = _make_simple_checkpoint(tmp_path)
    ckpt_root = run_dir / "checkpoints"

    # 1. Path traversal in latest.json rejected by read_checkpoint, resolve_resume_checkpoint_dir, resolve_checkpoint
    outside = tmp_path / "outside_ckpt"
    outside.mkdir()
    latest_path = ckpt_root / "latest.json"
    orig_latest = latest_path.read_text(encoding="utf-8")
    latest_path.write_text(json.dumps({"schema_version": 1, "checkpoint": "../outside_ckpt", "step": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="pointer path|Invalid"):
        read_checkpoint(run_dir)
    with pytest.raises(ValueError, match="pointer path|Invalid"):
        resolve_resume_checkpoint_dir(run_dir)
    with pytest.raises(ValueError, match="pointer path|Invalid"):
        resolve_checkpoint(run_dir, "latest")
    latest_path.write_text(orig_latest, encoding="utf-8")

    # 2. Symlinked latest.json rejected
    real_latest = tmp_path / "real_latest.json"
    real_latest.write_text(orig_latest, encoding="utf-8")
    latest_path.unlink()
    latest_path.symlink_to(real_latest)
    with pytest.raises(ValueError, match="symlink"):
        read_checkpoint(run_dir)
    with pytest.raises(ValueError, match="symlink"):
        resolve_checkpoint(run_dir, "latest")
    latest_path.unlink()
    latest_path.write_text(orig_latest, encoding="utf-8")

    # 3. Symlinked checkpoint dir rejected
    sym_ckpt = ckpt_root / "att-1-step-00000002-symlink123456"
    sym_ckpt.symlink_to(ckpt_dir)
    with pytest.raises(ValueError, match="symlink"):
        read_checkpoint(run_dir, sym_ckpt)
    with pytest.raises(ValueError, match="symlink"):
        extract_ema_bundle(sym_ckpt)
    sym_ckpt.unlink()

    # 4. Tampered state.pt rejected by extract_ema_bundle via state_sha256 check
    orig_state = (ckpt_dir / "state.pt").read_bytes()
    (ckpt_dir / "state.pt").write_bytes(orig_state + b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        extract_ema_bundle(ckpt_dir)
    (ckpt_dir / "state.pt").write_bytes(orig_state)

    # 5. Oversized JSON metadata rejected
    monkeypatch.setattr(ckpt_mod, "MAX_METADATA_BYTES", 32)
    with pytest.raises(ValueError, match="exceeds"):
        ckpt_mod._read_bounded_json(ckpt_dir / "manifest.json", max_bytes=32)
    monkeypatch.setattr(ckpt_mod, "MAX_METADATA_BYTES", MAX_METADATA_BYTES)

    # 6. _prune_lock does not unlink lockfile and keeps checkpoints/ clean
    lock_file = run_dir / ".prune.lock"
    with _prune_lock(ckpt_root):
        assert lock_file.is_file()
        ino_during = lock_file.stat().st_ino
    assert lock_file.is_file() and lock_file.stat().st_ino == ino_during
    assert not (ckpt_root / ".prune.lock").exists()

    # 7. _prune_run_checkpoints does not delete state.pt/manifest.json before rmtree fails
    ckpt_2 = ckpt_root / "att-1-step-00000002-aaaaaaaaaaaa"
    ckpt_2.mkdir()
    (ckpt_2 / "manifest.json").write_text(json.dumps({"step": 2, "attempt_id": "att-1"}), encoding="utf-8")
    (ckpt_2 / "state.pt").write_bytes(b"payload")
    latest_path.write_text(json.dumps({"schema_version": 1, "checkpoint": ckpt_2.name, "step": 2}), encoding="utf-8")

    orig_rmtree = ckpt_mod.shutil.rmtree

    def fail_first(path, *a, **kw):
        if Path(path) == ckpt_dir:
            raise PermissionError("simulated failure")
        return orig_rmtree(path, *a, **kw)

    monkeypatch.setattr(ckpt_mod.shutil, "rmtree", fail_first)
    rep = _prune_run_checkpoints(run_dir, policy={"keep_last": 1, "keep_every": 0})
    assert rep["failed_count"] == 1
    assert (ckpt_dir / "state.pt").is_file()
    assert (ckpt_dir / "manifest.json").is_file()


def test_distributed_checkpoint_security_state_file_bounds_and_ranks_symlink(tmp_path):
    proj = write_default(tmp_path / "project", device="cpu", defaults="hypergan")
    run_dir = tmp_path / "dist_run"
    prepare_train(proj, run_dir, profile="cpu-replicated-gloo", stop_after_steps=1, checkpoint_every=1).run()
    ckpt_dir = resolve_checkpoint(run_dir, "latest")
    mf_path = ckpt_dir / "manifest.json"
    orig_mf = json.loads(mf_path.read_text(encoding="utf-8"))
    expected_run = orig_mf["run_id"]

    # 1. Tamper state_file to path traversal -> rejected
    bad_mf = copy.deepcopy(orig_mf)
    bad_mf["state_file"] = "../escape.pt"
    mf_path.write_text(json.dumps(bad_mf), encoding="utf-8")
    with pytest.raises(ValueError, match="state file name"):
        read_distributed_checkpoint_for_elastic(run_dir, ckpt_dir, expected_run)

    # 2. Tamper state_bytes to bool / oversized / zero -> rejected
    for bad_bytes in (True, 0, MAX_RANK_BYTES + 1):
        bad_mf = copy.deepcopy(orig_mf)
        bad_mf["state_bytes"] = bad_bytes
        mf_path.write_text(json.dumps(bad_mf), encoding="utf-8")
        with pytest.raises(ValueError, match="state payload size"):
            read_distributed_checkpoint_for_elastic(run_dir, ckpt_dir, expected_run)

    mf_path.write_text(json.dumps(orig_mf), encoding="utf-8")

    # 3. Symlinked ranks/ directory rejected
    real_ranks = tmp_path / "moved_ranks"
    (ckpt_dir / "ranks").rename(real_ranks)
    (ckpt_dir / "ranks").symlink_to(real_ranks)
    with pytest.raises(ValueError, match="ranks directory"):
        read_distributed_checkpoint_for_elastic(run_dir, ckpt_dir, expected_run)
    (ckpt_dir / "ranks").unlink()
    real_ranks.rename(ckpt_dir / "ranks")


# ---------------------------------------------------------------------------
# Tests: Item 10 (synced_step, aggregation digest, _accumulated_update,
#        diversity/ratio, preflight routing, execution.py, preview-snapshot,
#        replicated on_event early stop)
# ---------------------------------------------------------------------------


def test_cross_topology_sets_synced_step_and_digests_include_aggregation():
    # 10a: restore_trainer_cross_topology updates synced_step (not last_synced_step)
    cfg = resolve_config({"training": {"steps": 4, "batch_size": 8, "device": "cpu", "seed": 7}})
    trainer = ReferenceTrainer(cfg)
    _, batch = trainer.update()
    from hypergan.checkpoints import trainer_state

    state = trainer_state(trainer, batch)
    fresh = ReferenceTrainer(cfg)
    strat = PseudoGradientStrategy(fresh, sync_every_steps=4)
    fresh.aggregation_strategy = strat
    assert strat.synced_step == 0
    restore_trainer_cross_topology(fresh, state, rank=0, world_size=1)
    assert strat.synced_step == 1
    assert strat.step_count == 1

    # 10b: _shared_digest changes when aggregation state differs
    base_state = {
        "graph": {"w": torch.ones(2)},
        "prior": {},
        "ema_graph": {},
        "ema_prior": {},
        "optimizers": [],
        "base_lrs": [],
        "buffers": {},
        "trainable": {},
        "modes": {},
        "step": 1,
        "aggregation": {"strategy": "pseudo_gradient", "anchor": {"w": torch.ones(2)}},
    }
    diff_state = copy.deepcopy(base_state)
    diff_state["aggregation"]["anchor"]["w"] = torch.full((2,), 2.0)
    assert _shared_digest(base_state) != _shared_digest(diff_state)


def test_accumulated_update_plain_adam_dualnorm_and_independent_no_grad(tmp_path):
    # Plain Adam critic with accumulation_steps=2 (no opt.record attribute)
    cfg_adam = {
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "optimizer": {"implementation": "torch_fused_adam"},
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    full_batch = {"real": torch.linspace(-1.0, 1.0, 16).reshape(8, 2), "_ids": torch.arange(8, dtype=torch.long)}
    d1 = tmp_path / "adam_acc"
    d1.mkdir()
    launch_cpu_workers(
        _worker_parity_generic,
        args=(cfg_adam, full_batch, str(d1), 2, 1),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    assert torch.load(d1 / "rank-0.pt", weights_only=False)["digest"] == torch.load(d1 / "rank-1.pt", weights_only=False)["digest"]

    # DualNorm (bcap recipe uses optimizer_family='dualnorm') with accumulation_steps=2
    cfg_dn = {
        "defaults": "particlegan:bcap",
        "prior": {"kind": "particles", "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": False},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    d2 = tmp_path / "dn_acc"
    d2.mkdir()
    launch_cpu_workers(
        _worker_parity_generic,
        args=(cfg_dn, full_batch, str(d2), 2, 1),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    assert torch.load(d2 / "rank-0.pt", weights_only=False)["digest"] == torch.load(d2 / "rank-1.pt", weights_only=False)["digest"]

    # Independent D-phase prepare() runs under torch.no_grad()
    cfg_ind = {
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {
                "factory": "tests.unit.test_audit_dist_fixes:_GradSpyGenerator",
                "args": {"in_features": 4, "out_features": 2},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42, "phase_draws": "independent"},
    }
    d3 = tmp_path / "ind_nograd"
    d3.mkdir()
    launch_cpu_workers(
        _worker_independent_no_grad,
        args=(cfg_ind, full_batch, str(d3)),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    flags = torch.load(d3 / "rank-0.pt", weights_only=False)["grad_flags"]
    assert flags[0] is False  # D phase generator prepare ran under torch.no_grad()
    assert flags[1] is True   # G phase generator prepare ran with gradients enabled


def test_global_intersample_diversity_ratio_detects_collapse(tmp_path):
    cfg = {
        "prior": {"kind": "gaussian", "args": {"z_dim": 4}},
        "prior_regularizer": {"weight": 0.0},
        "components": {
            "generator": {
                "factory": "tests.unit.test_audit_dist_fixes:_CollapsedGenerator",
                "args": {"in_features": 4, "out_features": 4},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 1, "bias": False},
                "inputs": {"input": "candidate"},
            },
        },
        "training": {"steps": 1, "batch_size": 8, "device": "cpu", "seed": 42},
    }
    gen = torch.Generator().manual_seed(5)
    full_batch = {"real": torch.randn(8, 4, generator=gen)}
    launch_cpu_workers(
        _worker_collapsed_diversity,
        args=(cfg, full_batch, str(tmp_path)),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )
    r0 = torch.load(tmp_path / "rank-0.pt", weights_only=False)
    r1 = torch.load(tmp_path / "rank-1.pt", weights_only=False)
    assert r0["diversity_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert r1["diversity_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_multinode_cpu_preflight_routes_through_worker_service(monkeypatch):
    import hypergan.execution_preflight as ep

    calls = []

    class FakeService:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            calls.append(("init", kwargs))
            self.directory = kwargs["args"][2]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def command(self, op):
            calls.append(("command", op))
            nr = self.kwargs["node_rank"]
            rpn = self.kwargs["ranks_per_node"]
            for r in range(nr * rpn, (nr + 1) * rpn):
                Path(self.directory, f"rank-{r}.json").write_text(
                    json.dumps({"rank": r, "status": "passed", "step": 0, "identity": {"ok": True}, "warnings": []})
                )

        def follow(self):
            calls.append(("follow", None))
            nr = self.kwargs["node_rank"]
            rpn = self.kwargs["ranks_per_node"]
            for r in range(nr * rpn, (nr + 1) * rpn):
                Path(self.directory, f"rank-{r}.json").write_text(
                    json.dumps({"rank": r, "status": "passed", "step": 0, "identity": {"ok": True}, "warnings": []})
                )

    monkeypatch.setattr(ep, "CPUWorkerService", FakeService)
    monkeypatch.setattr(ep, "launch_cpu_workers", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not use local launcher")))

    for node_rank, expected_call in ((0, ("command", "construct")), (1, ("follow", None))):
        calls.clear()
        report = ep.preflight(
            {"training": {"device": "cpu", "batch_size": 8}},
            {
                "schema_version": 1,
                "execution": {
                    "name": "cpu-replicated-gloo",
                    "world_size": 4,
                    "nodes": 2,
                    "node_rank": node_rank,
                    "rendezvous": "tcp://127.0.0.1:29500",
                },
            },
        )
        assert calls[0][0] == "init" and calls[0][1]["backend"] == "gloo"
        assert calls[1] == expected_call
        assert "hypergan.cpu_worker_service" in report["checker"]
        expected_ranks = [0, 1] if node_rank == 0 else [2, 3]
        assert [r["rank"] for r in report["ranks"]] == expected_ranks


def test_execution_options_world_size_single_device_visible_devices_and_reexec(tmp_path, monkeypatch):
    # 1. world_size=1 replicated synthesis fixed -> synthesizes world_size=2 and resolves cleanly
    cfg = resolve_config({"training": {"device": "cpu", "batch_size": 8}})
    cfg_raw = {**config_values(cfg), "aggregation": {"strategy": "pseudo_gradient", "sync_every_steps": 5}}
    with patch.dict(os.environ, {}, clear=True):
        prof, _ = _apply_execution_options(config=cfg_raw, profile=None)
    assert prof is not None
    assert prof["execution"]["world_size"] == 2
    assert prof["execution"]["ranks_per_node"] == 2
    resolved = resolve_execution_profile(prof, cfg)
    assert resolved["execution"]["world_size"] == 2

    # 2. Single-device override clears multi-GPU profile
    with patch.dict(os.environ, {}, clear=True):
        prof_cleared, _ = _apply_execution_options(
            devices="cpu",
            config=copy.deepcopy(cfg_raw),
            profile="cuda-replicated-nccl",
        )
    assert prof_cleared is None

    with patch.dict(os.environ, {}, clear=True):
        prof_cleared2, _ = _apply_execution_options(
            device="cpu",
            config=copy.deepcopy(cfg_raw),
            profile={"schema_version": 1, "execution": {"name": "cuda-replicated-nccl", "world_size": 2}},
        )
    assert prof_cleared2 is None

    # 3. _set_visible_devices('') does not touch CUDA_DEVICE_ORDER
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
        _set_visible_devices("")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert "CUDA_DEVICE_ORDER" not in os.environ

    # 4. prepare_train with _HYPERGAN_REEXEC=1 still resumes an existing run with manifest.json
    proj = write_default(tmp_path / "project", device="cpu", defaults="hypergan")
    run_dir = tmp_path / "reexec_run"
    prepare_train(proj, run_dir, stop_after_steps=1, checkpoint_every=1).run()
    monkeypatch.setenv("_HYPERGAN_REEXEC", "1")
    prep = prepare_train(proj, run_dir)
    assert prep.operation == "resume" and prep.repeat_train is True

    # Pre-manifest launch dir (only launch/) still starts fresh train under _HYPERGAN_REEXEC=1
    fresh_dir = tmp_path / "fresh_reexec"
    (fresh_dir / "launch").mkdir(parents=True)
    prep_fresh = prepare_train(proj, fresh_dir)
    assert prep_fresh.operation == "train"


def test_replicated_worker_preview_snapshot_validation_and_poisoning(tmp_path):
    attempt_dir = tmp_path / "attempt"
    preview_dir = attempt_dir / ".preview-abc"
    preview_dir.mkdir(parents=True)
    snap_path = preview_dir / "snapshot.pt"

    class FakeTrainer:
        step = 1
        checkpoint_ready = True
        _poisoned = False
        device = torch.device("cpu")

        def _phase(self, name, fn, force=False):
            assert force is True
            return fn()

    def make_state():
        return {
            "trainer": FakeTrainer(),
            "rank": 0,
            "batch": {"real": torch.zeros(2, 2)},
            "context": {
                "run_id": "run-1",
                "attempt_id": "att-1",
                "attempt_index": 1,
                "attempt_dir": str(attempt_dir),
                "run_dir": str(tmp_path),
            },
        }

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Wrong attempt_index type (bool instead of int) -> rejected and trainer poisoned
        st1 = make_state()
        with pytest.raises(ValueError, match="Preview snapshot destination or identity"):
            handle_command(
                st1,
                "preview-snapshot",
                {"identity": {"run_id": "run-1", "attempt_id": "att-1", "attempt_index": True}, "path": str(snap_path)},
            )
        assert st1["trainer"]._poisoned is True and st1["trainer"].checkpoint_ready is False

        # Existing snapshot path -> rejected and trainer poisoned
        snap_path.write_bytes(b"already-there")
        st2 = make_state()
        with pytest.raises(ValueError, match="Preview snapshot destination or identity"):
            handle_command(
                st2,
                "preview-snapshot",
                {"identity": {"run_id": "run-1", "attempt_id": "att-1", "attempt_index": 1}, "path": str(snap_path)},
            )
        assert st2["trainer"]._poisoned is True
    finally:
        torch.set_num_threads(prev_threads)


def test_replicated_execution_observe_propagates_early_stop(tmp_path):
    cfg = resolve_config({"training": {"device": "cpu", "batch_size": 8, "steps": 5}})
    profile = {"schema_version": 1, "execution": {"name": "cpu-replicated-gloo", "world_size": 2}}

    # Case 1: callback returning False -> observe() returns False
    ex1 = ReplicatedExecution(cfg, profile)
    ex1.configure_attempt(AttemptContext("run-1", "att-1", 1, tmp_path, tmp_path / "att"), preview_every=0, on_event=_cb_return_false)
    try:
        res = ex1.observe(None, {"event": "train", "step": 1})
        if res is None:
            # Wait for async observer worker to finish step 1, then deliver step 2
            ex1.observer._wait()
            res = ex1.observe(None, {"event": "train", "step": 2})
        assert res is False
    finally:
        ex1.close_observers()

    # Case 2: callback raising StopTraining -> observe() raises StopTraining with message
    ex2 = ReplicatedExecution(cfg, profile)
    ex2.configure_attempt(AttemptContext("run-2", "att-1", 1, tmp_path, tmp_path / "att2"), preview_every=0, on_event=_cb_raise_stop_training)
    try:
        with pytest.raises(StopTraining, match="budget reached"):
            ex2.observe(None, {"event": "train", "step": 1})
            ex2.observer._wait()
            ex2.observe(None, {"event": "train", "step": 2})
    finally:
        ex2.close_observers()

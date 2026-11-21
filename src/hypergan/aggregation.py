"""Pluggable Learning Aggregation Strategy System for Multi-Device Training (Issue #186).

Provides:
- AggregationStrategy protocol & base implementation
- Public strategy registry:
  - register_aggregation_strategy(name, strategy_cls)
  - get_aggregation_strategy(name)
  - create_aggregation_strategy(spec, trainer)
  - resolve_aggregation_strategy(spec, trainer) [alias]
- 3 Canonical strategies:
  1. GradientAllreduceStrategy ('gradient_allreduce'): synchronous gradient all-reduction after backward (sync_every_steps=1).
  2. ParameterAverageStrategy ('parameter_average'): periodic cross-rank parameter averaging (FedAvg / Local SGD).
  3. PseudoGradientStrategy ('pseudo_gradient'): DiLoCo outer-optimizer with momentum.

parameter_average and pseudo_gradient are experimental local-SGD strategies: replicas
differ between synchronizations, so a checkpoint at any other step first forces one
synchronization (see ``synchronize``).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, runtime_checkable

import torch
import torch.distributed as dist

from .execution_profiles import register_profile_strategy


@runtime_checkable
class AggregationStrategy(Protocol):
    """Protocol defining the learning aggregation contract with ReplicatedTrainer.

    ``reduces_gradients`` asks the trainer to mean-reduce every gradient across ranks
    before each optimizer step. A strategy without it keeps rank-local gradients and
    must make replicas agree itself, at least at the steps where
    ``should_assert_replicas`` is true. Optional ``is_synchronized(step)`` and
    ``synchronize()`` let checkpoints bring diverged replicas into agreement.
    """
    name: str
    reduces_gradients: bool

    def on_backward(self, optimizer: Any, name: str) -> None:
        """Post-backward gradient hook."""
        ...

    def on_step_complete(self, step: int) -> None:
        """Completed update step synchronization hook."""
        ...

    def should_assert_replicas(self, step: int) -> bool:
        """Check whether rank states must be bit-identical at this step."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpoint manifest / rank payloads."""
        ...

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint payload."""
        ...

    def manifest_metadata(self) -> Dict[str, Any]:
        """Return summary dictionary for top-level manifest.json."""
        ...


# ==============================================================================
# Canonical Strategy 1: "gradient_allreduce"
# ==============================================================================

REPLICA_CHECK_EVERY = 100


def _replica_check_every(value=None):
    """Steps between full replica digests for gradient_allreduce.

    The digest copies every parameter, buffer and optimizer tensor to the host
    and hashes it, so it is a debug check, not a per-step cost. Set
    HYPERGAN_REPLICA_CHECK_EVERY=1 to check after every update. Checkpoints
    always verify complete replica agreement regardless of this interval.
    """
    if value is None:
        raw = os.environ.get('HYPERGAN_REPLICA_CHECK_EVERY')
        if raw is None:
            return REPLICA_CHECK_EVERY
        try:
            value = int(raw)
        except ValueError:
            raise ValueError('HYPERGAN_REPLICA_CHECK_EVERY must be a positive integer') from None
    if type(value) is not int or value < 1:
        raise ValueError('replica_check_every must be a positive integer')
    return value


class GradientAllreduceStrategy:
    """Synchronous post-backward mean gradient all-reduce before optimizer step."""
    name: str = "gradient_allreduce"
    reduces_gradients: bool = True

    def __init__(self, trainer: Any = None, sync_every_steps: int = 1, replica_check_every: Optional[int] = None,
                 **kwargs: Any) -> None:
        if sync_every_steps != 1:
            raise ValueError("gradient_allreduce requires sync_every_steps = 1")
        self.trainer = trainer
        self.sync_every_steps = 1
        self.replica_check_every = _replica_check_every(replica_check_every)
        self.step_count = 0
        self.settings = kwargs

    def on_backward(self, optimizer: Any, name: str) -> None:
        # If running inside ReplicatedTrainer, _reduce_gradients handles the reduction.
        # Otherwise, in standalone mock environments, perform all-reduction here if dist is active.
        if self.trainer is not None and hasattr(self.trainer, "_reduce_gradients"):
            return

        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        parameters = [p for group in getattr(optimizer, 'param_groups', []) for p in group.get('params', [])]
        for p in parameters:
            if p.grad is not None:
                grad = p.grad.detach().contiguous()
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                p.grad = grad / world_size

    def on_step_complete(self, step: int, models: Any = None, optimizers: Any = None) -> None:
        self.step_count = step

    def should_assert_replicas(self, step: int) -> bool:
        # Reduced gradients keep replicas equal by construction; the digest only
        # catches nondeterministic kernels or rank-local state, so sample it.
        return step <= 1 or step % self.replica_check_every == 0

    def is_synchronized(self, step: int) -> bool:
        return True

    def synchronize(self, models: Any = None, optimizers: Any = None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "step": self.step_count,
            "step_count": self.step_count,
            "sync_every_steps": self.sync_every_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = state.get("step_count", state.get("step", self.step_count))
        self.sync_every_steps = state.get("sync_every_steps", 1)

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "sync_every_steps": self.sync_every_steps,
        }


def _resolve_comm_device(trainer: Any = None, optimizers: Any = None) -> torch.device:
    """Resolve communication device for collective operations."""
    comm_device = getattr(trainer, "device", None)
    if comm_device is not None:
        comm_device = torch.device(comm_device)
        if comm_device.type == "cuda" and comm_device.index is None:
            current = torch.cuda.current_device() if torch.cuda.is_available() else 0
            comm_device = torch.device("cuda", current)
        return comm_device
    if optimizers:
        opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        for opt in opts:
            for group in getattr(opt, "param_groups", []):
                for p in group.get("params", []):
                    if hasattr(p, "device"):
                        return p.device
    try:
        backend = dist.get_backend()
        return torch.device(
            f"cuda:{torch.cuda.current_device()}" if backend == "nccl" else "cpu"
        )
    except Exception:
        return torch.device("cpu")


def _synchronize_optimizer_moments(
    opts: Any, world_size: int, comm_device: torch.device
) -> None:
    """Synchronize optimizer floating moment buffers across ranks."""
    for opt in opts:
        for param_state in getattr(opt, "state", {}).values():
            for key, val in param_state.items():
                if (
                    key != "step"
                    and isinstance(val, torch.Tensor)
                    and val.is_floating_point()
                    and val.device == comm_device
                ):
                    data = val.data.contiguous()
                    dist.all_reduce(data, op=dist.ReduceOp.SUM)
                    val.data.copy_(data / world_size)


def _synchronize_optimizer_regularizers(
    trainer: Any = None, optimizers: Any = None
) -> None:
    """Synchronize all optimizer regularizer states across ranks (Issue #186 M2).

    ParticleGAN optimizers (DeviceK3PGeneratorAdam and DeviceKA2CriticAdam) maintain
    internal regularizer states (opt_g: latent_history, opt_d: record with last_sur,
    ema_residual, ema_data_scale, and ema_critic modules). During multi-rank local steps
    on distinct shards, these states drift. At synchronization boundaries, this function
    synchronizes both tensor buffers and scalar controller records across all ranks
    so that ReplicatedTrainer._replica_digest achieves bitwise replica agreement.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    opts = []
    if optimizers is not None:
        opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
    elif trainer is not None:
        opts = _strategy_optimizers(trainer)

    comm_device = _resolve_comm_device(trainer, opts)

    for opt in opts:
        # 1. Synchronize latent and direct history tensor buffers (opt_g)
        for hist_attr in ("latent_history", "direct_history"):
            hist = getattr(opt, hist_attr, None)
            if (
                hist is not None
                and isinstance(hist, torch.Tensor)
                and hist.is_floating_point()
                and hist.device == comm_device
            ):
                data = hist.data.contiguous()
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
                hist.data.copy_(data / world_size)

        # 1b. Synchronize Unified-v11 function-space conditioning scalars on optimizers
        if hasattr(opt, "_ema_func_gain"):
            fg = torch.tensor([float(opt._ema_func_gain)], dtype=torch.float64, device=comm_device)
            dist.all_reduce(fg, op=dist.ReduceOp.SUM)
            opt._ema_func_gain = max(1e-6, min(1.0, float((fg / world_size).item())))
        if hasattr(opt, "last_step_gain"):
            lg = torch.tensor([float(opt.last_step_gain)], dtype=torch.float64, device=comm_device)
            dist.all_reduce(lg, op=dist.ReduceOp.SUM)
            opt.last_step_gain = max(1e-6, float((lg / world_size).item()))

        # 2. Synchronize critic step record scalar records (opt_d.record)
        rec = getattr(opt, "record", None)
        if rec is not None:
            # Synchronize last_sur (scalar surrogate loss)
            if getattr(rec, "last_sur", None) is not None:
                sur = torch.tensor([float(rec.last_sur)], dtype=torch.float64, device=comm_device)
                dist.all_reduce(sur, op=dist.ReduceOp.SUM)
                rec.last_sur = float((sur / world_size).item())

            if hasattr(rec, "proto_initialized"):
                sync_empf_record(rec, comm_device, world_size)
            elif hasattr(rec, "t1c_ell_initialized"):
                sync_unified_v11_record(rec, comm_device, world_size)
            else:
                # Synchronize ema_residual and ema_data_scale
                if hasattr(rec, "ema_residual") and hasattr(rec, "ema_data_scale"):
                    stats = torch.tensor(
                        [float(rec.ema_residual), float(rec.ema_data_scale)],
                        dtype=torch.float64,
                        device=comm_device,
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    rec.ema_residual = float((stats[0] / world_size).item())
                    rec.ema_data_scale = float((stats[1] / world_size).item())

                if hasattr(rec, "_thermal_scale") and not hasattr(rec, "rhbi_ell_initialized"):
                    therm = torch.tensor([float(rec._thermal_scale)], dtype=torch.float64, device=comm_device)
                    dist.all_reduce(therm, op=dist.ReduceOp.SUM)
                    rec._thermal_scale = float((therm[0] / world_size).item())

            # Synchronize RHBI transport horizon, Weyl chord phase, and thermal settling state
            if hasattr(rec, "rhbi_ell_initialized") and hasattr(rec, "rhbi_thermal_initialized"):
                rhbi_floats = torch.tensor(
                    [
                        float(rec.rhbi_ell_ema),
                        float(rec.rhbi_chord_phase),
                        float(rec.rhbi_ema_ed),
                        float(rec.rhbi_ema_mu_sq),
                        float(rec.rhbi_ema_spread),
                        float(rec.rhbi_ema_q90),
                        float(rec.rhbi_ema_hmax),
                        float(rec.rhbi_ema_tail),
                        float(rec.rhbi_cond_ema),
                        float(rec.rhbi_het_ema),
                        float(rec._thermal_scale),
                    ],
                    dtype=torch.float64,
                    device=comm_device,
                )
                dist.all_reduce(rhbi_floats, op=dist.ReduceOp.SUM)
                rhbi_floats /= world_size
                vals = rhbi_floats.tolist()
                rec.rhbi_ell_ema = float(vals[0])
                rec.rhbi_chord_phase = float(vals[1]) % 1.0
                rec.rhbi_ema_ed = float(vals[2])
                rec.rhbi_ema_mu_sq = float(vals[3])
                rec.rhbi_ema_spread = float(vals[4])
                rec.rhbi_ema_q90 = float(vals[5])
                rec.rhbi_ema_hmax = float(vals[6])
                rec.rhbi_ema_tail = float(vals[7])
                rec.rhbi_cond_ema = float(vals[8])
                rec.rhbi_het_ema = float(vals[9])
                rec._thermal_scale = max(1e-6, min(1.0, float(vals[10])))

                rhbi_flags = torch.tensor(
                    [
                        int(bool(rec.rhbi_ell_initialized)),
                        int(bool(rec.rhbi_thermal_initialized)),
                        int(bool(rec.rhbi_cond_initialized)),
                        int(bool(rec.rhbi_het_initialized)),
                    ],
                    dtype=torch.int32,
                    device=comm_device,
                )
                dist.all_reduce(rhbi_flags, op=dist.ReduceOp.MAX)
                flags = rhbi_flags.tolist()
                rec.rhbi_ell_initialized = bool(flags[0])
                rec.rhbi_thermal_initialized = bool(flags[1])
                rec.rhbi_cond_initialized = bool(flags[2])
                rec.rhbi_het_initialized = bool(flags[3])

        # 3. Synchronize EMA critic network if attached to optimizer
        ema_critic = getattr(opt, "ema_critic", None)
        if ema_critic is not None and isinstance(ema_critic, torch.nn.Module):
            _average_modules([ema_critic], world_size)


def sync_empf_record(rec: Any, comm_device: torch.device, world_size: int) -> None:
    """Synchronize EMPF e-means prototype bank, phases, and thermal/step scales across ranks."""
    if rec is None or world_size <= 1 or not hasattr(rec, "proto_initialized"):
        return
    flags = torch.tensor(
        [int(bool(rec.initialized)), int(bool(rec.proto_initialized))],
        dtype=torch.int32,
        device=comm_device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    init_flag, proto_flag = flags.tolist()
    rec.initialized = bool(init_flag)

    scalars = torch.tensor(
        [
            float(rec.ema_residual),
            float(rec.ema_data_scale),
            float(rec.ema_cond_energy),
            float(rec._thermal_scale),
            float(rec._step_scale),
        ],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
    scalars /= world_size
    s_vals = scalars.tolist()
    rec.ema_residual = max(0.0, float(s_vals[0]))
    rec.ema_data_scale = max(0.0, float(s_vals[1]))
    rec.ema_cond_energy = float(s_vals[2])
    rec._thermal_scale = max(0.0, min(1.0, float(s_vals[3])))
    rec._step_scale = max(1e-6, min(1.0, float(s_vals[4])))

    phase_meta = torch.tensor(
        [float(rec.chord_phase), float(rec.revival_phase), float(int(rec.revivals))],
        dtype=torch.float64,
        device=comm_device,
    )
    rev_bounds = torch.tensor(
        [int(rec.revivals), -int(rec.revivals)],
        dtype=torch.int64,
        device=comm_device,
    )
    dist.all_reduce(rev_bounds, op=dist.ReduceOp.MAX)
    revivals_match = int(rev_bounds[0].item()) == -int(rev_bounds[1].item())
    dist.broadcast(phase_meta, src=0)
    p_vals = phase_meta.tolist()
    rec.chord_phase = float(p_vals[0]) % 1.0
    rec.revival_phase = float(p_vals[1]) % 1.0
    rec.revivals = max(0, int(round(p_vals[2])))

    if bool(proto_flag):
        # Verify all ranks hold matching prototype bank shapes; otherwise broadcast rank 0's bank.
        has_local = bool(rec.proto_initialized and isinstance(rec.proto_sum, torch.Tensor) and isinstance(rec.proto_count, torch.Tensor))
        shape_info = torch.tensor(
            [
                int(has_local),
                int(rec.proto_sum.shape[0]) if has_local else 0,
                int(rec.proto_sum.shape[1]) if has_local else 0,
            ],
            dtype=torch.int64,
            device=comm_device,
        )
        all_shapes = [torch.zeros_like(shape_info) for _ in range(world_size)]
        dist.all_gather(all_shapes, shape_info)
        shapes_list = [t.tolist() for t in all_shapes]
        uniform_bank = all(row[0] == 1 and row == shapes_list[0] for row in shapes_list)
        if uniform_bank and revivals_match:
            orig_dev = rec.proto_sum.device
            p_sum = rec.proto_sum.detach().to(device=comm_device, dtype=torch.float32).contiguous()
            p_cnt = rec.proto_count.detach().to(device=comm_device, dtype=torch.float32).contiguous()
            dist.all_reduce(p_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(p_cnt, op=dist.ReduceOp.SUM)
            p_sum /= world_size
            p_cnt = (p_cnt / world_size).clamp_min(1e-8)
            rec.proto_sum = p_sum.to(device=orig_dev)
            rec.proto_count = p_cnt.to(device=orig_dev)
            rec.proto_initialized = True
        elif shapes_list[0][0] == 1:
            k_n, d_n = int(shapes_list[0][1]), int(shapes_list[0][2])
            orig_dev = rec.proto_sum.device if has_local else comm_device
            p_sum = (
                rec.proto_sum.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                if dist.get_rank() == 0
                else torch.empty((k_n, d_n), dtype=torch.float32, device=comm_device)
            )
            p_cnt = (
                rec.proto_count.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                if dist.get_rank() == 0
                else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
            )
            dist.broadcast(p_sum, src=0)
            dist.broadcast(p_cnt, src=0)
            rec.proto_sum = p_sum.to(device=orig_dev)
            rec.proto_count = p_cnt.clamp_min(1e-8).to(device=orig_dev)
            rec.proto_initialized = True
    else:
        rec.proto_initialized = False
        rec.proto_sum = None
        rec.proto_count = None


def sync_unified_v11_record(rec: Any, comm_device: torch.device, world_size: int) -> None:
    """Synchronize Unified-v11 / t1_coupling critic step record scalars, charts, and fiber bank across ranks."""
    if rec is None or world_size <= 1 or not hasattr(rec, "t1c_ell_initialized"):
        return
    flags = torch.tensor(
        [
            int(bool(rec.t1c_ell_initialized)),
            int(bool(rec.t1c_thermal_initialized)),
            int(bool(rec.t1c_cond_initialized)),
            int(bool(rec.t1c_het_initialized)),
            int(bool(rec.chart_initialized)),
            int(bool(getattr(rec, "margin_initialized", False))),
        ],
        dtype=torch.int32,
        device=comm_device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    f_vals = flags.tolist()
    rec.t1c_ell_initialized = bool(f_vals[0])
    rec.t1c_thermal_initialized = bool(f_vals[1])
    rec.t1c_cond_initialized = bool(f_vals[2])
    rec.t1c_het_initialized = bool(f_vals[3])
    chart_flag = bool(f_vals[4])
    if hasattr(rec, "margin_initialized"):
        rec.margin_initialized = bool(f_vals[5])

    scalars = torch.tensor(
        [
            float(rec.t1c_ell_ema),
            float(rec.t1c_ema_ed),
            float(rec.t1c_ema_mu_sq),
            float(rec.t1c_ema_spread),
            float(rec.t1c_ema_q90),
            float(rec.t1c_ema_hmax),
            float(rec.t1c_ema_tail),
            float(rec.t1c_cond_ema),
            float(rec.t1c_het_ema),
            float(rec._thermal_scale),
            float(rec.ema_data_scale),
            float(getattr(rec, "_step_scale", 1.0)),
            float(getattr(rec, "ema_margin", 0.0)),
            float(getattr(rec, "ema_push_res", 1.0)),
        ],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
    scalars /= world_size
    s_vals = scalars.tolist()
    rec.t1c_ell_ema = max(0.0, float(s_vals[0]))
    rec.t1c_ema_ed = float(s_vals[1])
    rec.t1c_ema_mu_sq = float(s_vals[2])
    rec.t1c_ema_spread = float(s_vals[3])
    rec.t1c_ema_q90 = float(s_vals[4])
    rec.t1c_ema_hmax = float(s_vals[5])
    rec.t1c_ema_tail = float(s_vals[6])
    rec.t1c_cond_ema = float(s_vals[7])
    rec.t1c_het_ema = float(s_vals[8])
    rec._thermal_scale = max(1e-6, min(1.0, float(s_vals[9])))
    rec.ema_data_scale = max(1e-6, float(s_vals[10]))
    if hasattr(rec, "_step_scale"):
        rec._step_scale = max(1e-6, min(1.0, float(s_vals[11])))
    if hasattr(rec, "ema_margin"):
        rec.ema_margin = float(s_vals[12])
    if hasattr(rec, "ema_push_res"):
        rec.ema_push_res = max(0.0, float(s_vals[13]))

    phase_meta = torch.tensor(
        [float(rec.t1c_chord_phase), float(rec.revival_phase), float(int(rec.revivals))],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.broadcast(phase_meta, src=0)
    p_vals = phase_meta.tolist()
    rec.t1c_chord_phase = float(p_vals[0]) % 1.0
    rec.revival_phase = float(p_vals[1]) % 1.0
    rec.revivals = max(0, int(round(p_vals[2])))

    if chart_flag:
        has_local = bool(
            rec.chart_initialized
            and isinstance(rec.chart_sum, torch.Tensor)
            and isinstance(rec.chart_count, torch.Tensor)
            and isinstance(rec.chart_sq_dev, torch.Tensor)
            and isinstance(rec.chart_fake_mass, torch.Tensor)
        )
        shape_info = torch.tensor(
            [
                int(has_local),
                int(rec.chart_sum.shape[0]) if has_local else 0,
                int(rec.chart_sum.shape[1]) if has_local else 0,
            ],
            dtype=torch.int64,
            device=comm_device,
        )
        all_shapes = [torch.zeros_like(shape_info) for _ in range(world_size)]
        dist.all_gather(all_shapes, shape_info)
        shapes_list = [t.tolist() for t in all_shapes]
        if shapes_list[0][0] == 1:
            k_n, d_n = int(shapes_list[0][1]), int(shapes_list[0][2])
            orig_dev = rec.chart_sum.device if has_local else comm_device
            if all(row == shapes_list[0] for row in shapes_list):
                c_sum = rec.chart_sum.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                c_cnt = rec.chart_count.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                c_sq = rec.chart_sq_dev.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                c_fm = rec.chart_fake_mass.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                dist.all_reduce(c_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(c_cnt, op=dist.ReduceOp.SUM)
                dist.all_reduce(c_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(c_fm, op=dist.ReduceOp.SUM)
                c_sum /= world_size
                c_cnt = (c_cnt / world_size).clamp_min(1e-8)
                c_sq = (c_sq / world_size).clamp_min(1e-8)
                c_fm = (c_fm / world_size).clamp_min(0.0)
            else:
                is_root = dist.get_rank() == 0
                c_sum = rec.chart_sum.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_root else torch.empty((k_n, d_n), dtype=torch.float32, device=comm_device)
                c_cnt = rec.chart_count.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_root else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
                c_sq = rec.chart_sq_dev.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_root else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
                c_fm = rec.chart_fake_mass.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_root else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
                dist.broadcast(c_sum, src=0)
                dist.broadcast(c_cnt, src=0)
                dist.broadcast(c_sq, src=0)
                dist.broadcast(c_fm, src=0)
            rec.chart_sum = c_sum.to(device=orig_dev)
            rec.chart_count = c_cnt.to(device=orig_dev)
            rec.chart_sq_dev = c_sq.to(device=orig_dev)
            rec.chart_fake_mass = c_fm.to(device=orig_dev)
            rec.chart_initialized = True
    else:
        rec.chart_initialized = False
        rec.chart_sum = None
        rec.chart_count = None
        rec.chart_sq_dev = None
        rec.chart_fake_mass = None

    bank = getattr(rec, "cond_fiber_bank", None)
    if bank is not None and hasattr(bank, "to_packet") and hasattr(bank, "load_packet"):
        has_strata = torch.tensor([int(bool(getattr(bank, "strata", None)))], dtype=torch.int32, device=comm_device)
        dist.all_reduce(has_strata, op=dist.ReduceOp.MAX)
        if int(has_strata.item()) > 0:
            pkt = [bank.to_packet() if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(pkt, src=0)
            if pkt[0] is not None and dist.get_rank() != 0:
                bank.load_packet(pkt[0])


def _average_modules(modules: List[torch.nn.Module], world_size: int, *, parameters: bool = True) -> None:
    """Mean floating parameters and buffers across ranks; rank zero owns integer buffers."""
    for module in modules:
        if parameters:
            for p in module.parameters():
                data = p.data.contiguous()
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
                p.data.copy_(data / world_size)
        for buf in module.buffers():
            if buf.is_floating_point():
                data = buf.data.contiguous()
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
                buf.data.copy_(data / world_size)
            else:
                dist.broadcast(buf.data, src=0)


def _strategy_optimizers(trainer: Any = None, optimizers: Any = None) -> List[Any]:
    if optimizers is not None:
        return list(optimizers) if isinstance(optimizers, (list, tuple)) else [optimizers]
    if trainer is None:
        return []
    defaults = (getattr(trainer, "opt_g", None), getattr(trainer, "opt_d", None))
    return [opt for opt in getattr(trainer, "optimizers", defaults)
            if opt is not None and hasattr(opt, "state")]


def _synchronize_optimizers(trainer: Any, optimizers: Any, world_size: int) -> None:
    """Average optimizer moments and ParticleGAN regularizer state across ranks."""
    opts = _strategy_optimizers(trainer, optimizers)
    _synchronize_optimizer_moments(opts, world_size, _resolve_comm_device(trainer, opts))
    _synchronize_optimizer_regularizers(trainer, opts)


class _LocalSGDStrategy:
    """Shared cadence and checkpoint synchronization for experimental local-SGD strategies.

    Replicas train independently between synchronizations. ``synced_step`` records
    the last step whose state was synchronized, so a checkpoint at any step can force
    exactly one synchronization and a resumed run does not repeat it.
    """
    reduces_gradients: bool = False

    def _init_cadence(self, trainer: Any, sync_every_steps: int, settings: Dict[str, Any]) -> None:
        if type(sync_every_steps) is not int or sync_every_steps < 1:
            raise ValueError("aggregation.sync_every_steps must be a positive integer >= 1")
        self.trainer = trainer
        self.sync_every_steps = sync_every_steps
        self.step_count = 0
        self.synced_step = 0
        self.settings = settings

    def on_backward(self, optimizer: Any, name: str) -> None:
        pass

    def should_assert_replicas(self, step: int) -> bool:
        return step % self.sync_every_steps == 0

    def is_synchronized(self, step: int) -> bool:
        return self.should_assert_replicas(step) or self.synced_step == step

    def on_step_complete(self, step: int, models: Any = None, optimizers: Any = None) -> None:
        self.step_count = step
        if self.should_assert_replicas(step):
            self.synchronize(models=models, optimizers=optimizers)

    def synchronize(self, models: Any = None, optimizers: Any = None) -> None:
        self._synchronize(models=models, optimizers=optimizers)
        self.synced_step = self.step_count

    def _cadence_state(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "step": self.step_count,
            "step_count": self.step_count,
            "synced_step": self.synced_step,
            "sync_every_steps": self.sync_every_steps,
        }

    def _load_cadence(self, state: Dict[str, Any]) -> None:
        self.step_count = state.get("step_count", state.get("step", self.step_count))
        self.sync_every_steps = state.get("sync_every_steps", self.sync_every_steps)
        # Older checkpoints were only valid at sync boundaries.
        self.synced_step = state.get("synced_step", self.step_count)


# ==============================================================================
# Canonical Strategy 2: "parameter_average"
# ==============================================================================


class ParameterAverageStrategy(_LocalSGDStrategy):
    """Experimental periodic cross-rank parameter averaging (FedAvg / Local SGD)."""
    name: str = "parameter_average"

    def __init__(self, trainer: Any = None, sync_every_steps: int = 1, **kwargs: Any) -> None:
        self._init_cadence(trainer, sync_every_steps, kwargs)

    def _get_models(self, models: Any = None) -> List[torch.nn.Module]:
        if models is not None:
            if isinstance(models, torch.nn.Module):
                return [models]
            if isinstance(models, (list, tuple)):
                return [m for m in models if isinstance(m, torch.nn.Module)]
            if isinstance(models, dict):
                return [m for m in models.values() if isinstance(m, torch.nn.Module)]
        if self.trainer is None:
            return []
        found: List[torch.nn.Module] = []
        for attr in ("graph", "prior", "ema_graph", "ema_prior"):
            mod = getattr(self.trainer, attr, None)
            if isinstance(mod, torch.nn.Module) and mod not in found:
                found.append(mod)
        if hasattr(self.trainer, "models") and isinstance(self.trainer.models, dict):
            for m in self.trainer.models.values():
                if isinstance(m, torch.nn.Module) and m not in found:
                    found.append(m)
        return found

    def _synchronize(self, models: Any = None, optimizers: Any = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        _average_modules(self._get_models(models), world_size)
        _synchronize_optimizers(self.trainer, optimizers, world_size)

    def state_dict(self) -> Dict[str, Any]:
        return self._cadence_state()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_cadence(state)

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "sync_every_steps": self.sync_every_steps,
        }


# ==============================================================================
# Canonical Strategy 3: "pseudo_gradient" (DiLoCo / FedOpt)
# ==============================================================================

class PseudoGradientStrategy(_LocalSGDStrategy):
    """Experimental federated outer-optimizer update (DiLoCo / FedOpt) using pseudo-gradients.

    Unlike the usual DiLoCo, inner Adam moments and ParticleGAN regularizer state are
    also averaged at every synchronization, so replicas stay bitwise identical there.
    """
    name: str = "pseudo_gradient"

    def __init__(
        self,
        trainer: Any = None,
        sync_every_steps: int = 1,
        outer_lr: float = 1.0,
        outer_momentum: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._init_cadence(trainer, sync_every_steps, kwargs)
        if not isinstance(outer_lr, (int, float)) or isinstance(outer_lr, bool) or outer_lr <= 0.0 or not math.isfinite(outer_lr):
            raise ValueError("aggregation.outer_lr must be a finite positive float")
        if not isinstance(outer_momentum, (int, float)) or isinstance(outer_momentum, bool) or not (0.0 <= outer_momentum < 1.0) or not math.isfinite(outer_momentum):
            raise ValueError("aggregation.outer_momentum must be a finite float in [0.0, 1.0)")

        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)

        self.anchor: Dict[str, torch.Tensor] = {}
        self.momentum_buffer: Dict[str, torch.Tensor] = {}
        self._init_anchor_and_momentum()

    def _get_named_parameters(self, models: Any = None) -> Dict[str, torch.nn.Parameter]:
        named_params: Dict[str, torch.nn.Parameter] = {}
        if models is not None:
            if isinstance(models, torch.nn.Module):
                for k, v in models.named_parameters():
                    named_params[k] = v
            elif isinstance(models, dict):
                for mod_name, mod in models.items():
                    if isinstance(mod, torch.nn.Module):
                        for k, v in mod.named_parameters():
                            named_params[f"{mod_name}.{k}"] = v
            elif isinstance(models, (list, tuple)):
                for idx, mod in enumerate(models):
                    if isinstance(mod, torch.nn.Module):
                        for k, v in mod.named_parameters():
                            named_params[f"m{idx}.{k}"] = v
            return named_params

        if self.trainer is None:
            return named_params

        has_prior = hasattr(self.trainer, "prior") and getattr(self.trainer, "prior", None) is not None and isinstance(self.trainer.prior, torch.nn.Module)
        for attr in ("graph", "prior"):
            mod = getattr(self.trainer, attr, None)
            if isinstance(mod, torch.nn.Module):
                for k, v in mod.named_parameters():
                    key = f"{attr}.{k}" if has_prior else k
                    named_params[key] = v

        if hasattr(self.trainer, "models") and isinstance(self.trainer.models, dict):
            for mod_name, mod in self.trainer.models.items():
                if isinstance(mod, torch.nn.Module):
                    for k, v in mod.named_parameters():
                        named_params[f"{mod_name}.{k}"] = v

        return named_params

    def _get_modules(self, models: Any = None) -> List[torch.nn.Module]:
        """Modules whose parameters take the outer step; their buffers are averaged."""
        if models is not None:
            if isinstance(models, torch.nn.Module):
                return [models]
            values = models.values() if isinstance(models, dict) else models if isinstance(models, (list, tuple)) else ()
            return [m for m in values if isinstance(m, torch.nn.Module)]
        found: List[torch.nn.Module] = []
        if self.trainer is None:
            return found
        candidates = [getattr(self.trainer, attr, None) for attr in ("graph", "prior")]
        if hasattr(self.trainer, "models") and isinstance(self.trainer.models, dict):
            candidates.extend(self.trainer.models.values())
        for mod in candidates:
            if isinstance(mod, torch.nn.Module) and all(mod is not seen for seen in found):
                found.append(mod)
        return found

    def initialize(self, models: Any = None) -> None:
        """Capture the outer anchor from the current replicated parameters."""
        self.anchor, self.momentum_buffer = {}, {}
        self._init_anchor_and_momentum(models)

    def _init_anchor_and_momentum(self, models: Any = None) -> None:
        # The outer state lives beside its parameter (the trainer device), so a
        # synchronization makes no host round trip per parameter.
        params = self._get_named_parameters(models)
        for k, p in params.items():
            if k not in self.anchor:
                self.anchor[k] = p.detach().clone()
            if k not in self.momentum_buffer:
                self.momentum_buffer[k] = torch.zeros_like(self.anchor[k])

    def _synchronize(self, models: Any = None, optimizers: Any = None) -> None:
        if not self.anchor:
            self._init_anchor_and_momentum(models)

        params = self._get_named_parameters(models)
        if not params:
            return

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1

        for k, p in params.items():
            if k not in self.anchor:
                self.anchor[k] = p.detach().clone()
                self.momentum_buffer[k] = torch.zeros_like(self.anchor[k])

            # A no-op unless restored state was loaded without the trainer's parameters.
            anchor_t = self.anchor[k].to(device=p.device, dtype=p.dtype)
            delta = anchor_t - p.data

            if is_dist and world_size > 1:
                delta_contig = delta.contiguous()
                dist.all_reduce(delta_contig, op=dist.ReduceOp.SUM)
                delta = delta_contig / world_size

            v_prev = self.momentum_buffer[k].to(device=p.device, dtype=p.dtype)
            v_new = self.outer_momentum * v_prev + delta
            new_anchor = anchor_t - self.outer_lr * v_new
            self.momentum_buffer[k], self.anchor[k] = v_new, new_anchor
            p.data.copy_(new_anchor)

        if is_dist and world_size > 1:
            # Buffers (e.g. spectral-norm weight_u) evolve locally and take no outer step.
            _average_modules(self._get_modules(models), world_size, parameters=False)
            if self.trainer is not None:
                _average_modules([mod for mod in (getattr(self.trainer, "ema_graph", None),
                                                  getattr(self.trainer, "ema_prior", None))
                                  if isinstance(mod, torch.nn.Module)], world_size)
            _synchronize_optimizers(self.trainer, optimizers, world_size)

    def state_dict(self) -> Dict[str, Any]:
        if not self.anchor:
            self._init_anchor_and_momentum()
        return dict(self._cadence_state(),
                    outer_lr=self.outer_lr,
                    outer_momentum=self.outer_momentum,
                    anchor={k: v.clone() for k, v in self.anchor.items()},
                    momentum_buffer={k: v.clone() for k, v in self.momentum_buffer.items()})

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._load_cadence(state)
        self.outer_lr = state.get("outer_lr", self.outer_lr)
        self.outer_momentum = state.get("outer_momentum", self.outer_momentum)
        # Older checkpoints also carried aliases of the same tensors. Checkpoints
        # store the state on the CPU; it returns to its parameter's device here.
        devices = {k: p.device for k, p in self._get_named_parameters().items()}
        anchor_data = state.get("anchor") or state.get("anchor_weights") or state.get("weights") or {}
        self.anchor = {k: v.detach().to(devices.get(k, v.device), copy=True) for k, v in anchor_data.items()}
        momentum_data = state.get("momentum_buffer") or state.get("momentum") or {}
        self.momentum_buffer = {k: v.detach().to(devices.get(k, v.device), copy=True) for k, v in momentum_data.items()}

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "sync_every_steps": self.sync_every_steps,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
        }


# ==============================================================================
# Public Registry
# ==============================================================================

_REGISTRY: Dict[str, Type[AggregationStrategy]] = {
    "gradient_allreduce": GradientAllreduceStrategy,
    "parameter_average": ParameterAverageStrategy,
    "pseudo_gradient": PseudoGradientStrategy,
}


def register_aggregation_strategy(name: str, strategy_cls: Type[AggregationStrategy]) -> None:
    """Register custom aggregation strategy in public registry and execution profile validator."""
    if not isinstance(name, str) or not name:
        raise ValueError("Strategy name must be a non-empty string")
    _REGISTRY[name] = strategy_cls
    register_profile_strategy(name)


def get_aggregation_strategy(name: str) -> Type[AggregationStrategy]:
    """Retrieve strategy class by canonical strategy name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown aggregation strategy: {name!r}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def create_aggregation_strategy(
    spec: Any,
    trainer: Any = None,
) -> AggregationStrategy:
    """Instantiate strategy from profile or aggregation dictionary."""
    raw = spec
    if isinstance(raw, dict):
        if "aggregation" in raw:
            raw = raw["aggregation"]
        elif "execution" in raw and "aggregation" in raw["execution"]:
            raw = raw["execution"]["aggregation"]
    if not isinstance(raw, dict):
        raise ValueError("Aggregation specification must be a dictionary")

    cfg = dict(raw)
    strategy_name = cfg.pop("strategy", "gradient_allreduce")
    strategy_cls = get_aggregation_strategy(strategy_name)
    return strategy_cls(trainer=trainer, **cfg)


resolve_aggregation_strategy = create_aggregation_strategy

__all__ = [
    "AggregationStrategy",
    "GradientAllreduceStrategy",
    "ParameterAverageStrategy",
    "PseudoGradientStrategy",
    "register_aggregation_strategy",
    "get_aggregation_strategy",
    "create_aggregation_strategy",
    "resolve_aggregation_strategy",
]

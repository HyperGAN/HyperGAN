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


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.split(".")[-1]
    return getattr(torch, name)


def _synchronize_optimizer_moments(
    opts: Any, world_size: int, comm_device: torch.device
) -> None:
    """Synchronize optimizer floating moment buffers and per-param float/bool scalars across ranks.

    Agrees on the union of per-parameter state keys across ranks before issuing
    collectives so asymmetric lazy keys (e.g. AxiomOptimizer's ``v_pow``,
    ``log2_ref``, ``log2_init`` or UnifiedV15Optimizer's ``peak_rms``/``peak_rms_val``)
    never deadlock or diverge.
    """
    group_world = dist.get_world_size()
    rank = dist.get_rank()
    float_scalar_keys = ("log2_ref", "peak_rms_val")
    bool_scalar_keys = ("log2_init",)

    for opt in opts:
        flush_fast = getattr(opt, "_flush_fast_cache", None)
        if callable(flush_fast):
            flush_fast()
        params = [
            p
            for group in getattr(opt, "param_groups", [])
            for p in group.get("params", [])
        ]
        if not params:
            if hasattr(opt, "_fast_cache") and getattr(opt, "_fast_cache", None) is not None:
                opt._fast_cache = None
            continue

        opt_state = getattr(opt, "state", {})
        local_meta: List[Dict[str, Tuple[Any, ...]]] = []
        for p in params:
            param_state = opt_state.get(p)
            p_meta: Dict[str, Tuple[Any, ...]] = {}
            if isinstance(param_state, dict):
                for key, val in param_state.items():
                    if key == "step":
                        continue
                    if (
                        isinstance(val, torch.Tensor)
                        and val.is_floating_point()
                        and val.device == comm_device
                    ):
                        p_meta[key] = ("tensor", tuple(val.shape), str(val.dtype))
                    elif key in float_scalar_keys and isinstance(val, (int, float)) and not isinstance(val, bool):
                        p_meta[key] = ("float",)
                    elif key in bool_scalar_keys and isinstance(val, bool):
                        p_meta[key] = ("bool",)
            local_meta.append(p_meta)

        if group_world > 1:
            gathered_meta: List[Any] = [None] * group_world
            dist.all_gather_object(gathered_meta, local_meta)
        else:
            gathered_meta = [local_meta]

        for idx, p in enumerate(params):
            all_keys = sorted({k for peer in gathered_meta for k in peer[idx]})
            if not all_keys:
                continue
            param_state = opt_state.get(p)
            if not isinstance(param_state, dict):
                param_state = opt.state[p]

            for key in all_keys:
                present_ranks = [r for r in range(group_world) if key in gathered_meta[r][idx]]
                src = present_ranks[0]
                src_desc = gathered_meta[src][idx][key]
                kind = src_desc[0]
                has_local = key in local_meta[idx]
                denom = world_size if len(present_ranks) == group_world else len(present_ranks)

                if kind == "tensor":
                    shape, dtype_str = src_desc[1], src_desc[2]
                    dtype = _resolve_torch_dtype(dtype_str)
                    uniform_spec = all(
                        gathered_meta[r][idx][key] == src_desc for r in present_ranks
                    )
                    if uniform_spec:
                        if has_local:
                            data = param_state[key].data.contiguous().clone()
                        else:
                            data = torch.zeros(shape, dtype=dtype, device=comm_device)
                        dist.all_reduce(data, op=dist.ReduceOp.SUM)
                        synced = data / denom
                    else:
                        if rank == src:
                            buf = param_state[key].data.contiguous().clone()
                        else:
                            buf = torch.empty(shape, dtype=dtype, device=comm_device)
                        dist.broadcast(buf, src=src)
                        synced = buf
                    if has_local and isinstance(param_state.get(key), torch.Tensor) and param_state[key].shape == synced.shape and param_state[key].dtype == synced.dtype:
                        param_state[key].data.copy_(synced)
                    else:
                        param_state[key] = synced
                elif kind == "float":
                    local_val = float(param_state[key]) if has_local else 0.0
                    buf = torch.tensor([local_val], dtype=torch.float64, device=comm_device)
                    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                    synced_val = float((buf / denom).item())
                    if key == "peak_rms_val":
                        synced_val = max(0.0, synced_val)
                    param_state[key] = synced_val
                elif kind == "bool":
                    local_val = int(bool(param_state[key])) if has_local else 0
                    buf = torch.tensor([local_val], dtype=torch.int32, device=comm_device)
                    dist.all_reduce(buf, op=dist.ReduceOp.MAX)
                    param_state[key] = bool(buf.item())

        if hasattr(opt, "_fast_cache") and getattr(opt, "_fast_cache", None) is not None:
            opt._fast_cache = None


def _sync_unit_sketch(opt: Any, attr: str, comm_device: torch.device, world_size: int) -> None:
    """Synchronize a 1D unit-normalized sketch tensor attribute across ranks."""
    sk = getattr(opt, attr, None)
    has_sk = isinstance(sk, torch.Tensor) and sk.ndim == 1 and sk.numel() > 0
    sk_meta = torch.tensor([int(has_sk), int(sk.numel()) if has_sk else 0], dtype=torch.int64, device=comm_device)
    all_sk_meta = [torch.zeros_like(sk_meta) for _ in range(world_size)]
    dist.all_gather(all_sk_meta, sk_meta)
    sk_rows = [t.tolist() for t in all_sk_meta]
    init_ranks = [r for r, row in enumerate(sk_rows) if row[0] == 1]
    if len(init_ranks) == world_size and all(row == sk_rows[0] for row in sk_rows):
        orig_dev = sk.device
        sk_buf = sk.detach().to(device=comm_device, dtype=torch.float32).contiguous()
        dist.all_reduce(sk_buf, op=dist.ReduceOp.SUM)
        sk_nrm = sk_buf.norm()
        if bool(torch.isfinite(sk_nrm)) and float(sk_nrm.item()) > 1e-30:
            setattr(opt, attr, (sk_buf / sk_nrm).to(device=orig_dev))
        else:
            setattr(opt, attr, None)
    elif init_ranks:
        src = init_ranks[0]
        dim_sk = int(sk_rows[src][1])
        orig_dev = sk.device if has_sk else comm_device
        sk_buf = (
            sk.detach().to(device=comm_device, dtype=torch.float32).contiguous()
            if dist.get_rank() == src
            else torch.empty((dim_sk,), dtype=torch.float32, device=comm_device)
        )
        dist.broadcast(sk_buf, src=src)
        setattr(opt, attr, sk_buf.to(device=orig_dev))
    else:
        setattr(opt, attr, None)


def _sync_critic_step_record_base(rec: Any, comm_device: torch.device) -> None:
    """Synchronize CriticStepRecord base counters and LR telemetry from rank 0."""
    if not all(hasattr(rec, a) for a in ("lr_max", "lr_last", "anchor_started", "calls", "observed_steps")):
        return
    base_meta = torch.tensor(
        [
            float(rec.lr_max),
            -1.0 if rec.lr_last is None else float(rec.lr_last),
            float(int(bool(rec.anchor_started))),
            float(int(rec.calls)),
            float(int(rec.observed_steps)),
        ],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.broadcast(base_meta, src=0)
    b_vals = base_meta.tolist()
    rec.lr_max = max(0.0, float(b_vals[0]))
    rec.lr_last = None if b_vals[1] < 0.0 else float(b_vals[1])
    rec.anchor_started = bool(int(round(b_vals[2])))
    rec.calls = max(0, int(round(b_vals[3])))
    rec.observed_steps = max(0, int(round(b_vals[4])))


def _sync_ka2_record(rec: Any, comm_device: torch.device, world_size: int) -> None:
    """Synchronize KA2StepRecord controller fields across ranks."""
    has_sur = rec.last_sur is not None
    sur_buf = torch.tensor(
        [float(int(has_sur)), float(rec.last_sur) if has_sur else 0.0],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.all_reduce(sur_buf, op=dist.ReduceOp.SUM)
    sur_cnt = int(round(sur_buf[0].item()))
    rec.last_sur = None if sur_cnt <= 0 else float((sur_buf[1] / sur_cnt).item())

    alpha_buf = torch.tensor([float(rec.alpha)], dtype=torch.float64, device=comm_device)
    dist.all_reduce(alpha_buf, op=dist.ReduceOp.SUM)
    rec.alpha = max(0.0, float((alpha_buf[0] / world_size).item()))

    if hasattr(rec, "alpha_raw"):
        has_raw = rec.alpha_raw is not None
        raw_buf = torch.tensor(
            [float(int(has_raw)), float(rec.alpha_raw) if has_raw else 0.0],
            dtype=torch.float64,
            device=comm_device,
        )
        dist.all_reduce(raw_buf, op=dist.ReduceOp.SUM)
        raw_cnt = int(round(raw_buf[0].item()))
        rec.alpha_raw = None if raw_cnt <= 0 else max(0.0, float((raw_buf[1] / raw_cnt).item()))

    if dist.get_rank() == 0:
        payload = [
            {
                "sur_hist": [float(x) for x in rec.sur_hist],
                "sur_base": None if rec.sur_base is None else float(rec.sur_base),
                "w": 1.0 if float(rec.w) >= 0.5 else 0.0,
                "low_streak": max(0, int(rec.low_streak)),
                "last_ratio": None if rec.last_ratio is None else float(rec.last_ratio),
                "ema_updates": max(0, int(rec.ema_updates)),
                "ema_skips": max(0, int(rec.ema_skips)),
                "ema_reseeds": max(0, int(rec.ema_reseeds)),
            }
        ]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    synced = payload[0]
    if isinstance(synced, dict):
        rec.sur_hist = list(synced["sur_hist"])
        rec.sur_base = synced["sur_base"]
        rec.w = synced["w"]
        rec.low_streak = synced["low_streak"]
        rec.last_ratio = synced["last_ratio"]
        rec.ema_updates = synced["ema_updates"]
        rec.ema_skips = synced["ema_skips"]
        rec.ema_reseeds = synced["ema_reseeds"]
    _sync_critic_step_record_base(rec, comm_device)


def _synchronize_optimizer_regularizers(
    trainer: Any = None, optimizers: Any = None
) -> None:
    """Synchronize all optimizer regularizer states across ranks (Issue #186 M2).

    ParticleGAN optimizers (K3P/KA2 Adam, ConstraintGeometry/DirectionBlend,
    UnifiedV11/V12/V15 optimizers, and AxiomOptimizer) maintain internal regularizer
    states. During multi-rank local steps on distinct shards, these states drift.
    At synchronization boundaries, this function synchronizes both tensor buffers
    and scalar controller records across all ranks so that
    ReplicatedTrainer._replica_digest achieves bitwise replica agreement.
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

        # 1a. Synchronize K3P / KA2 regularizer objects (LatentRowDamping, DirectParticleResponse, CriticSpikeGuard)
        ld = getattr(opt, "latent_damping", None)
        if ld is not None and all(hasattr(ld, a) for a in ("observed", "total", "started")):
            ld_t = torch.tensor(
                [int(ld.observed), int(ld.total), int(bool(ld.started))],
                dtype=torch.int64,
                device=comm_device,
            )
            dist.all_reduce(ld_t, op=dist.ReduceOp.MAX)
            ld.observed = max(0, int(ld_t[0].item()))
            ld.total = max(ld.observed, int(ld_t[1].item()))
            ld.started = bool(ld_t[2].item())

        dr = getattr(opt, "direct_response", None)
        if dr is not None and all(hasattr(dr, a) for a in ("started", "step_count")):
            dr_t = torch.tensor(
                [int(bool(dr.started)), int(dr.step_count)],
                dtype=torch.int64,
                device=comm_device,
            )
            dist.all_reduce(dr_t, op=dist.ReduceOp.MAX)
            dr.started = bool(dr_t[0].item())
            dr.step_count = max(0, int(dr_t[1].item()))
            if hasattr(dr, "last_gain"):
                lg_dr = torch.tensor([float(dr.last_gain)], dtype=torch.float64, device=comm_device)
                dist.all_reduce(lg_dr, op=dist.ReduceOp.SUM)
                dr.last_gain = float((lg_dr[0] / world_size).item())

        guard = getattr(opt, "guard", None)
        if guard is not None and hasattr(guard, "_clipped"):
            cl = torch.tensor([int(guard._clipped)], dtype=torch.int64, device=comm_device)
            dist.all_reduce(cl, op=dist.ReduceOp.MAX)
            guard._clipped = max(0, int(cl[0].item()))

        # 1b. Synchronize Unified-v11 / v12 / v15 / Axiom / DualNorm optimizer scalars & stats
        flush_scalars = getattr(opt, "_flush_scalars", None)
        if callable(flush_scalars):
            flush_scalars()
        if hasattr(opt, "_ema_func_gain"):
            fg = torch.tensor([float(opt._ema_func_gain)], dtype=torch.float64, device=comm_device)
            dist.all_reduce(fg, op=dist.ReduceOp.SUM)
            opt._ema_func_gain = max(1e-6, min(1.0, float((fg / world_size).item())))
        if hasattr(opt, "last_step_gain"):
            lg = torch.tensor([float(opt.last_step_gain)], dtype=torch.float64, device=comm_device)
            dist.all_reduce(lg, op=dist.ReduceOp.SUM)
            opt.last_step_gain = max(1e-6, float((lg / world_size).item()))
        if hasattr(opt, "_ema_flip_cos"):
            fc = torch.tensor([float(opt._ema_flip_cos)], dtype=torch.float64, device=comm_device)
            dist.all_reduce(fc, op=dist.ReduceOp.SUM)
            opt._ema_flip_cos = max(-1.0, min(1.0, float((fc / world_size).item())))
        if hasattr(opt, "_g_peak_ref") or getattr(opt, "_g_peak_ref_t", None) is not None:
            gpr_t = getattr(opt, "_g_peak_ref_t", None)
            gpr_val = float(gpr_t.item()) if isinstance(gpr_t, torch.Tensor) else float(getattr(opt, "_g_peak_ref", 0.0))
            gpr_buf = torch.tensor([gpr_val], dtype=torch.float64, device=comm_device)
            dist.all_reduce(gpr_buf, op=dist.ReduceOp.SUM)
            opt._g_peak_ref = max(0.0, float((gpr_buf[0] / world_size).item()))
            if hasattr(opt, "_g_peak_ref_t"):
                opt._g_peak_ref_t = None
        if hasattr(opt, "_prev_flip_sketch"):
            _sync_unit_sketch(opt, "_prev_flip_sketch", comm_device, world_size)
        if hasattr(opt, "_ema_step_cos_val"):
            ax_sc = torch.tensor(
                [
                    float(opt._ema_step_cos_val),
                    float(getattr(opt, "_log2_damp_val", 0.0)),
                    float(getattr(opt, "_crit_skew_ema", 0.0)),
                ],
                dtype=torch.float64,
                device=comm_device,
            )
            dist.all_reduce(ax_sc, op=dist.ReduceOp.SUM)
            ax_sc /= world_size
            ax_vals = ax_sc.tolist()
            opt._ema_step_cos_val = max(-1.0, min(1.0, float(ax_vals[0])))
            if hasattr(opt, "_ema_step_cos_t"):
                opt._ema_step_cos_t = None
            if hasattr(opt, "_log2_damp_val"):
                opt._log2_damp_val = max(-10.0, min(0.0, float(ax_vals[1])))
            if hasattr(opt, "_log2_damp_t"):
                opt._log2_damp_t = None
            if hasattr(opt, "_crit_skew_ema"):
                opt._crit_skew_ema = max(0.0, float(ax_vals[2]))
        if hasattr(opt, "_prev_step_sketch"):
            _sync_unit_sketch(opt, "_prev_step_sketch", comm_device, world_size)

        cg_stats = getattr(opt, "constraint_geometry_stats", None)
        if isinstance(cg_stats, dict) and {"steps", "projected_steps", "max_derivative_before", "max_derivative_after"} <= set(cg_stats):
            cg_ints = torch.tensor(
                [int(cg_stats["steps"]), int(cg_stats["projected_steps"])],
                dtype=torch.int64,
                device=comm_device,
            )
            dist.all_reduce(cg_ints, op=dist.ReduceOp.MAX)
            cg_floats = torch.tensor(
                [float(cg_stats["max_derivative_before"]), float(cg_stats["max_derivative_after"])],
                dtype=torch.float64,
                device=comm_device,
            )
            dist.all_reduce(cg_floats, op=dist.ReduceOp.MAX)
            cg_stats["steps"] = max(0, int(cg_ints[0].item()))
            cg_stats["projected_steps"] = max(0, int(cg_ints[1].item()))
            cg_stats["max_derivative_before"] = max(0.0, float(cg_floats[0].item()))
            cg_stats["max_derivative_after"] = max(0.0, float(cg_floats[1].item()))

        db_stats = getattr(opt, "direction_blend_stats", None)
        if isinstance(db_stats, dict) and {
            "conflict_steps",
            "blended_steps",
            "pareto_stalls",
            "retained_norm_ratio_sum",
            "max_retained_norm_ratio",
        } <= set(db_stats):
            db_ints = torch.tensor(
                [int(db_stats["conflict_steps"]), int(db_stats["blended_steps"]), int(db_stats["pareto_stalls"])],
                dtype=torch.int64,
                device=comm_device,
            )
            dist.all_reduce(db_ints, op=dist.ReduceOp.MAX)
            db_sum = torch.tensor([float(db_stats["retained_norm_ratio_sum"])], dtype=torch.float64, device=comm_device)
            dist.all_reduce(db_sum, op=dist.ReduceOp.SUM)
            db_max = torch.tensor([float(db_stats["max_retained_norm_ratio"])], dtype=torch.float64, device=comm_device)
            dist.all_reduce(db_max, op=dist.ReduceOp.MAX)
            db_stats["conflict_steps"] = max(0, int(db_ints[0].item()))
            db_stats["blended_steps"] = max(0, int(db_ints[1].item()))
            db_stats["pareto_stalls"] = max(0, int(db_ints[2].item()))
            db_stats["retained_norm_ratio_sum"] = max(0.0, float((db_sum[0] / world_size).item()))
            db_stats["max_retained_norm_ratio"] = max(0.0, float(db_max[0].item()))

        # 2. Synchronize critic step record scalar records (opt_d.record)
        rec = getattr(opt, "record", None)
        if rec is not None:
            if hasattr(rec, "proto_initialized"):
                sync_empf_record(rec, comm_device, world_size)
                _sync_critic_step_record_base(rec, comm_device)
            elif hasattr(rec, "t1c_ell_initialized"):
                sync_unified_v11_record(rec, comm_device, world_size)
                _sync_critic_step_record_base(rec, comm_device)
            elif hasattr(rec, "chart_state") and hasattr(rec, "thermal_scale"):
                sync_axiom_record(rec, comm_device, world_size)
            elif hasattr(rec, "sur_hist") and hasattr(rec, "alpha") and hasattr(rec, "w"):
                _sync_ka2_record(rec, comm_device, world_size)
            else:
                # Synchronize last_sur (scalar surrogate loss)
                if getattr(rec, "last_sur", None) is not None:
                    sur = torch.tensor([float(rec.last_sur)], dtype=torch.float64, device=comm_device)
                    dist.all_reduce(sur, op=dist.ReduceOp.SUM)
                    rec.last_sur = float((sur / world_size).item())

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
                _sync_critic_step_record_base(rec, comm_device)

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

    if trainer is not None:
        thermal_coordinator = getattr(trainer, "thermal_coordinator", None)
        if thermal_coordinator is not None:
            thermal_coordinator.synchronize()


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
        # Verify all ranks hold matching prototype bank shapes; otherwise broadcast from lowest initialized rank.
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
        init_ranks = [r for r, row in enumerate(shapes_list) if row[0] == 1]
        uniform_bank = len(init_ranks) == world_size and all(row == shapes_list[0] for row in shapes_list)
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
        elif init_ranks:
            src = init_ranks[0]
            k_n, d_n = int(shapes_list[src][1]), int(shapes_list[src][2])
            orig_dev = rec.proto_sum.device if has_local else comm_device
            p_sum = (
                rec.proto_sum.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                if dist.get_rank() == src
                else torch.empty((k_n, d_n), dtype=torch.float32, device=comm_device)
            )
            p_cnt = (
                rec.proto_count.detach().to(device=comm_device, dtype=torch.float32).contiguous()
                if dist.get_rank() == src
                else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
            )
            dist.broadcast(p_sum, src=src)
            dist.broadcast(p_cnt, src=src)
            rec.proto_sum = p_sum.to(device=orig_dev)
            rec.proto_count = p_cnt.clamp_min(1e-8).to(device=orig_dev)
            rec.proto_initialized = True
        else:
            rec.proto_initialized = False
            rec.proto_sum = None
            rec.proto_count = None
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
            float(getattr(rec, "_g_func_gain", 1.0)),
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
    if hasattr(rec, "_g_func_gain"):
        rec._g_func_gain = max(1e-6, min(1.0, float(s_vals[14])))

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
        init_ranks = [r for r, row in enumerate(shapes_list) if row[0] == 1]
        if init_ranks:
            src = init_ranks[0]
            k_n, d_n = int(shapes_list[src][1]), int(shapes_list[src][2])
            orig_dev = rec.chart_sum.device if has_local else comm_device
            if len(init_ranks) == world_size and all(row == shapes_list[0] for row in shapes_list):
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
                is_src = dist.get_rank() == src
                c_sum = rec.chart_sum.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_src else torch.empty((k_n, d_n), dtype=torch.float32, device=comm_device)
                c_cnt = rec.chart_count.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_src else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
                c_sq = rec.chart_sq_dev.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_src else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
                c_fm = rec.chart_fake_mass.detach().to(device=comm_device, dtype=torch.float32).contiguous() if is_src else torch.empty((k_n,), dtype=torch.float32, device=comm_device)
                dist.broadcast(c_sum, src=src)
                dist.broadcast(c_cnt, src=src)
                dist.broadcast(c_sq, src=src)
                dist.broadcast(c_fm, src=src)
                c_cnt = c_cnt.clamp_min(1e-8)
                c_sq = c_sq.clamp_min(1e-8)
                c_fm = c_fm.clamp_min(0.0)
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
    else:
        rec.chart_initialized = False
        rec.chart_sum = None
        rec.chart_count = None
        rec.chart_sq_dev = None
        rec.chart_fake_mass = None

    if hasattr(rec, "atomic_ledger_initialized"):
        has_ledger = bool(
            rec.atomic_ledger_initialized
            and isinstance(getattr(rec, "atomic_row_chart", None), torch.Tensor)
            and isinstance(getattr(rec, "atomic_row_dist", None), torch.Tensor)
            and isinstance(getattr(rec, "atomic_row_target", None), torch.Tensor)
        )
        all_ledgers = [torch.zeros(1, dtype=torch.int32, device=comm_device) for _ in range(world_size)]
        dist.all_gather(all_ledgers, torch.tensor([int(has_ledger)], dtype=torch.int32, device=comm_device))
        ledger_ranks = [r for r, t in enumerate(all_ledgers) if int(t.item()) > 0]
        if ledger_ranks:
            src = ledger_ranks[0]
            pkt = [
                {
                    "row_chart": rec.atomic_row_chart.detach().cpu().clone(),
                    "row_dist": rec.atomic_row_dist.detach().cpu().clone(),
                    "row_target": rec.atomic_row_target.detach().cpu().clone(),
                }
                if dist.get_rank() == src
                else None
            ]
            dist.broadcast_object_list(pkt, src=src)
            synced_ledger = pkt[0]
            if isinstance(synced_ledger, dict):
                orig_dev = rec.atomic_row_chart.device if has_ledger else comm_device
                rec.atomic_row_chart = synced_ledger["row_chart"].to(device=orig_dev)
                rec.atomic_row_dist = synced_ledger["row_dist"].to(device=orig_dev)
                rec.atomic_row_target = synced_ledger["row_target"].to(device=orig_dev)
                rec.atomic_ledger_initialized = True
        else:
            rec.atomic_ledger_initialized = False
            rec.atomic_row_chart = None
            rec.atomic_row_dist = None
            rec.atomic_row_target = None

    bank = getattr(rec, "cond_fiber_bank", None)
    if bank is not None and hasattr(bank, "to_packet") and hasattr(bank, "load_packet"):
        has_strata = torch.tensor([int(bool(getattr(bank, "strata", None)))], dtype=torch.int32, device=comm_device)
        all_strata = [torch.zeros_like(has_strata) for _ in range(world_size)]
        dist.all_gather(all_strata, has_strata)
        strata_ranks = [r for r, t in enumerate(all_strata) if int(t.item()) > 0]
        if strata_ranks:
            src = strata_ranks[0]
            pkt = [_payload_to_cpu(bank.to_packet()) if dist.get_rank() == src else None]
            dist.broadcast_object_list(pkt, src=src)
            if pkt[0] is not None and dist.get_rank() != src:
                bank.load_packet(pkt[0])
                if hasattr(bank, "restore_device"):
                    bank.restore_device(comm_device)

    hb = getattr(rec, "highd_basin_bank", None)
    if hb is not None and hasattr(hb, "to_packet") and hasattr(hb, "load_packet"):
        hb_init = bool(getattr(hb, "k", 0) > 0 or getattr(hb, "sigma_z_init", False) or getattr(hb, "dim", 0) > 0)
        all_hb = [torch.zeros(1, dtype=torch.int32, device=comm_device) for _ in range(world_size)]
        dist.all_gather(all_hb, torch.tensor([int(hb_init)], dtype=torch.int32, device=comm_device))
        hb_ranks = [r for r, t in enumerate(all_hb) if int(t.item()) > 0]
        if hb_ranks:
            src = hb_ranks[0]
            pkt = [_payload_to_cpu(hb.to_packet()) if dist.get_rank() == src else None]
            dist.broadcast_object_list(pkt, src=src)
            if pkt[0] is not None and dist.get_rank() != src:
                hb.load_packet(pkt[0])
                if hasattr(hb, "restore_device"):
                    hb.restore_device(comm_device)

    sb = getattr(rec, "sparse_obs_bank", None)
    if sb is not None and hasattr(sb, "to_packet") and hasattr(sb, "load_packet"):
        sb_init = bool(
            getattr(sb, "step", 0) > 0
            or getattr(sb, "sparse_push_res", None) is not None
            or getattr(sb, "initialized", False)
            or getattr(sb, "k", 0) > 0
        )
        all_sb = [torch.zeros(1, dtype=torch.int32, device=comm_device) for _ in range(world_size)]
        dist.all_gather(all_sb, torch.tensor([int(sb_init)], dtype=torch.int32, device=comm_device))
        sb_ranks = [r for r, t in enumerate(all_sb) if int(t.item()) > 0]
        if sb_ranks:
            src = sb_ranks[0]
            pkt = [_payload_to_cpu(sb.to_packet()) if dist.get_rank() == src else None]
            dist.broadcast_object_list(pkt, src=src)
            if pkt[0] is not None and dist.get_rank() != src:
                sb.load_packet(pkt[0])
                if hasattr(sb, "restore_device"):
                    sb.restore_device(comm_device)

    if hasattr(rec, "highd_basin_bank"):
        rv_t = getattr(rec, "_v15_g_peak_ref_t", None)
        has_v15_ref = hasattr(rec, "_v15_g_peak_ref") or isinstance(rv_t, torch.Tensor)
        v15_val = float(rv_t.item()) if isinstance(rv_t, torch.Tensor) else float(getattr(rec, "_v15_g_peak_ref", 0.0))
        v15_buf = torch.tensor(
            [float(int(has_v15_ref)), v15_val if has_v15_ref else 0.0],
            dtype=torch.float64,
            device=comm_device,
        )
        dist.all_reduce(v15_buf, op=dist.ReduceOp.SUM)
        v15_cnt = int(round(v15_buf[0].item()))
        if v15_cnt > 0:
            rec._v15_g_peak_ref = max(0.0, float((v15_buf[1] / v15_cnt).item()))
            if hasattr(rec, "_v15_g_peak_ref_t"):
                rec._v15_g_peak_ref_t = None

    if hasattr(rec, "restore_device_state"):
        rec.restore_device_state(comm_device)


def _payload_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _payload_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_payload_to_cpu(v) for v in obj)
    return obj


def sync_axiom_record(rec: Any, comm_device: torch.device, world_size: int) -> None:
    """Synchronize AxiomStepRecord scalars, flags, charts, atomic ledger, and fiber bank across ranks."""
    if rec is None or world_size <= 1 or not (hasattr(rec, "chart_state") and hasattr(rec, "thermal_scale")):
        return
    if hasattr(rec, "_flush_ema"):
        rec._flush_ema()

    flags = torch.tensor(
        [
            int(bool(rec.grad_sq_initialized)),
            int(bool(rec.margin_initialized)),
            int(bool(rec.ell_initialized)),
            int(bool(rec.thermal_initialized)),
            int(bool(rec.cond_initialized)),
            int(bool(rec.het_initialized)),
            int(rec.chart_state is not None),
            int(rec.atomic_ledger is not None),
            int(rec._cond_bank_obj is not None or rec.cond_fiber_bank is not None),
        ],
        dtype=torch.int32,
        device=comm_device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    f_vals = flags.tolist()
    rec.grad_sq_initialized = bool(f_vals[0])
    rec.margin_initialized = bool(f_vals[1])
    rec.ell_initialized = bool(f_vals[2])
    rec.thermal_initialized = bool(f_vals[3])
    rec.cond_initialized = bool(f_vals[4])
    rec.het_initialized = bool(f_vals[5])
    has_chart, has_ledger, has_bank = bool(f_vals[6]), bool(f_vals[7]), bool(f_vals[8])

    scalars = torch.tensor(
        [
            float(rec.ema_grad_sq),
            float(rec.ema_margin),
            float(rec.ema_data_scale),
            float(rec.last_data_scale),
            float(rec.ema_push_res),
            float(rec.ell_ema),
            float(rec.thermal_scale),
            float(rec.step_scale),
            float(rec.cond_ema),
            float(rec.het_ema),
            float(rec.g_func_gain),
            float(rec.ema_ed),
            float(rec.ema_mu_sq),
            float(rec.ema_spread),
            float(rec.ema_q90),
            float(rec.ema_hmax),
            float(rec.ema_tail),
        ],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
    scalars /= world_size
    s_vals = scalars.tolist()
    rec.ema_grad_sq = max(0.0, float(s_vals[0]))
    rec.ema_margin = max(-60.0, min(60.0, float(s_vals[1])))
    rec.ema_data_scale = max(1e-30, float(s_vals[2]))
    rec.last_data_scale = max(1e-30, float(s_vals[3]))
    rec.ema_push_res = max(0.0, float(s_vals[4]))
    rec.ell_ema = max(0.0, float(s_vals[5]))
    rec.thermal_scale = max(1e-5, min(1.0, float(s_vals[6])))
    rec.thermal_locked = bool(rec.thermal_scale <= 1e-4)
    rec.step_scale = max(1e-6, min(1.0, float(s_vals[7])))
    rec.cond_ema = max(-0.5, min(2.0, float(s_vals[8])))
    rec.het_ema = max(0.0, float(s_vals[9]))
    rec.g_func_gain = max(1e-6, min(1.0, float(s_vals[10])))
    rec.ema_ed = float(s_vals[11])
    rec.ema_mu_sq = float(s_vals[12])
    rec.ema_spread = float(s_vals[13])
    rec.ema_q90 = float(s_vals[14])
    rec.ema_hmax = float(s_vals[15])
    rec.ema_tail = float(s_vals[16])

    phase_meta = torch.tensor(
        [
            float(rec.chord_phase),
            float(int(rec.calls)),
            float(int(rec.observed_steps)),
            -1.0 if rec.last_ratio is None else float(rec.last_ratio),
        ],
        dtype=torch.float64,
        device=comm_device,
    )
    dist.broadcast(phase_meta, src=0)
    p_vals = phase_meta.tolist()
    rec.chord_phase = float(p_vals[0]) % 1.0
    rec.calls = max(0, int(round(p_vals[1])))
    rec.observed_steps = max(0, int(round(p_vals[2])))
    rec.last_ratio = None if p_vals[3] < 0.0 else float(p_vals[3])

    _sync_critic_step_record_base(rec, comm_device)

    if has_chart or has_ledger or has_bank:
        local_pres = torch.tensor(
            [
                int(rec.chart_state is not None),
                int(rec.atomic_ledger is not None),
                int(getattr(rec, "_cond_bank_obj", None) is not None or rec.cond_fiber_bank is not None),
            ],
            dtype=torch.int32,
            device=comm_device,
        )
        all_pres = [torch.zeros_like(local_pres) for _ in range(world_size)]
        dist.all_gather(all_pres, local_pres)
        pres_rows = [t.tolist() for t in all_pres]
        chart_ranks = [r for r, row in enumerate(pres_rows) if row[0] == 1]
        ledger_ranks = [r for r, row in enumerate(pres_rows) if row[1] == 1]
        bank_ranks = [r for r, row in enumerate(pres_rows) if row[2] == 1]

        if chart_ranks:
            src = chart_ranks[0]
            pkt = [_payload_to_cpu(rec.chart_state) if dist.get_rank() == src else None]
            dist.broadcast_object_list(pkt, src=src)
            rec.chart_state = pkt[0]
        else:
            rec.chart_state = None

        if ledger_ranks:
            src = ledger_ranks[0]
            pkt = [_payload_to_cpu(rec.atomic_ledger) if dist.get_rank() == src else None]
            dist.broadcast_object_list(pkt, src=src)
            rec.atomic_ledger = pkt[0]
        else:
            rec.atomic_ledger = None

        if bank_ranks:
            src = bank_ranks[0]
            if dist.get_rank() == src:
                bank_obj = getattr(rec, "_cond_bank_obj", None)
                bank_pkt = (
                    _payload_to_cpu(bank_obj.to_packet())
                    if (bank_obj is not None and hasattr(bank_obj, "to_packet"))
                    else _payload_to_cpu(rec.cond_fiber_bank)
                )
                pkt = [bank_pkt]
            else:
                pkt = [None]
            dist.broadcast_object_list(pkt, src=src)
            synced_bank = pkt[0]
            rec.cond_fiber_bank = synced_bank
            bank_obj = getattr(rec, "_cond_bank_obj", None)
            if synced_bank is None:
                if hasattr(rec, "_cond_bank_obj"):
                    rec._cond_bank_obj = None
            elif bank_obj is not None and hasattr(bank_obj, "load_packet"):
                bank_obj.load_packet(synced_bank)
        else:
            rec.cond_fiber_bank = None
            if hasattr(rec, "_cond_bank_obj"):
                rec._cond_bank_obj = None

        if hasattr(rec, "restore_device_state"):
            rec.restore_device_state(comm_device)
    else:
        rec.chart_state = None
        rec.atomic_ledger = None
        rec.cond_fiber_bank = None
        if hasattr(rec, "_cond_bank_obj"):
            rec._cond_bank_obj = None


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

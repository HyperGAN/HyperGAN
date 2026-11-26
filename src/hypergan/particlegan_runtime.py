"""Runtime adapters for ParticleGAN critic penalties and multi-critic thermal coordination.

Two runtime guarantees live here:

1. **Multi-critic coupled thermal policy (`MultiCriticThermalCoordinator`)**:
   When a model trains multiple distinct ParticleGAN critics sharing a generator
   and prior (for example MisGAN ``dx`` + ``dm``), each critic maintains its own
   ``UnifiedV11StepRecord`` and domain-specific residual EMAs, while the applied
   ``_thermal_scale`` is synchronized conservatively across all critics to
   ``s_shared = max_i s_i`` at each step boundary and held constant throughout
   the step. This prevents one faster-cooling critic from silently drifting its
   relative gradient weight into the shared generator/prior (or cooling the
   shared generator optimizer / prior regularizer while another critic is still
   hot), while keeping checkpoints and replicated ranks bit-exact with zero
   uncheckpointed controller state.

2. **Dual-safe ``fwAD`` softmax & attention JVP (`DualSafeSoftmaxMode`)**:
   PyTorch's built-in forward-mode AD JVP for ``softmax`` / ``log_softmax``
   (including ``SDPBackend.MATH`` scaled dot-product attention) mutates a saved
   ``Exp`` tensor in place and fails during reverse-mode ``backward()`` with a
   version-counter ``RuntimeError`` (``pytorch#162350``, reproduced on PyTorch
   2.8 through 2.15). ``DualSafeSoftmaxMode`` is a thread-local
   ``TorchFunctionMode`` active only during critic penalty evaluation that
   intercepts softmax / log-softmax / SDPA strictly on forward-AD dual tensors
   and computes the exact analytic JVP in reverse-mode-safe operations with zero
   ``torch.autograd.grad`` calls and zero extra forward passes.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import math
import types
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

logger = logging.getLogger(__name__)

MULTI_CRITIC_THERMAL_POLICY = "coupled_max"


def _publish_record_thermal(record: Any, scale: float) -> None:
    """Mirror a synchronized thermal scale into ParticleGAN's legacy channel if active."""
    if not getattr(record, "legacy_thermal_channel", False):
        return
    try:
        from particlegan._bispf.thermal import _LEGACY_THERMAL_CHANNEL, _publish_thermal
    except Exception:
        return
    if not _LEGACY_THERMAL_CHANNEL:
        return
    for key in list(_LEGACY_THERMAL_CHANNEL):
        if isinstance(key, tuple) and len(key) == 2 and key[0] == id(record):
            _publish_thermal(record, key[1], float(scale))


def _thermal_scale_attr(record: Any) -> Optional[str]:
    if record is None:
        return None
    if hasattr(record, "_thermal_scale"):
        return "_thermal_scale"
    if hasattr(record, "thermal_scale"):
        return "thermal_scale"
    return None


def _get_thermal_scale(record: Any, default: float = 1.0) -> float:
    attr = _thermal_scale_attr(record)
    if attr is None:
        return float(default)
    return float(getattr(record, attr, default))


def _set_thermal_scale(record: Any, scale: float) -> None:
    attr = _thermal_scale_attr(record)
    if attr is not None:
        setattr(record, attr, float(scale))


def _thermal_observer_attr(record: Any) -> Optional[str]:
    if record is None:
        return None
    for attr in ("observe_t1c_thermal", "observe_thermal"):
        if callable(getattr(record, attr, None)):
            return attr
    return None


class MultiCriticThermalCoordinator:
    """Coordinate ``_thermal_scale`` / ``thermal_scale`` across multiple ParticleGAN critic records.

    Policy (``coupled_max``):
    - Single-critic models (<= 1 thermal critic record) are left untouched.
    - Multi-critic models keep independent per-critic EMAs (``t1c_ema_*`` / ``ema_*``,
      ``ema_margin``, ``ema_data_scale``, ``ema_push_res``) on each critic's
      step record, while applying a single synchronized effective thermal scale
      ``s_shared = max_i s_i`` to every critic record (and therefore to the
      shared ``opt_g``, ``prior``, and ``spread`` bound to the primary record).
    - During a step's generator phase, ``step_scope()`` holds every critic at
      the pre-step ``s_step`` so all critic forwards, OT coupling losses, prior
      regularization, and optimizer steps see the exact same thermal scale (zero
      intra-step skew). Each critic's thermal observer still updates its
      own EMAs and records its raw candidate scale; on normal step completion,
      all critic records advance together to ``s_next = max_i s_cand_i``.
    - Because every critic record stores the synchronized thermal scale at
      every step boundary, checkpoint/resume and replicated rank digests remain
      bit-exact with zero extra checkpoint fields.
    """

    def __init__(self, critic_optimizers: Dict[str, Any]) -> None:
        self.critic_records: Dict[str, Any] = {}
        self._record_to_name: Dict[int, str] = {}
        for name, optimizer in critic_optimizers.items():
            record = getattr(optimizer, "record", None)
            if _thermal_scale_attr(record) is not None:
                self.critic_records[name] = record
                self._record_to_name[id(record)] = name
        self.last_raw_scales: Dict[str, float] = {
            name: _get_thermal_scale(rec, 1.0)
            for name, rec in self.critic_records.items()
        }
        self._logged_floor_event: bool = False
        if self.active:
            self.synchronize()
            logger.info(
                "Multi-critic thermal policy=%s active across critics=%s (shared scale=%g)",
                self.policy,
                list(self.critic_records),
                self.effective_scale(),
            )

    @property
    def active(self) -> bool:
        return len(self.critic_records) > 1

    @property
    def policy(self) -> str:
        return MULTI_CRITIC_THERMAL_POLICY if self.active else "single_critic"

    def effective_scale(self) -> float:
        if not self.critic_records:
            return 1.0
        scales = [
            _get_thermal_scale(rec, 1.0)
            for rec in self.critic_records.values()
        ]
        finite = [s for s in scales if math.isfinite(s) and s > 0.0]
        if not finite:
            return 1.0
        return min(1.0, max(finite))

    def synchronize(self, scale: Optional[float] = None) -> float:
        if not self.critic_records:
            return 1.0
        target = self.effective_scale() if scale is None else float(scale)
        if not math.isfinite(target) or target <= 0.0:
            target = 1.0
        target = min(1.0, target)
        if self.active:
            for rec in self.critic_records.values():
                _set_thermal_scale(rec, target)
                _publish_record_thermal(rec, target)
        return target

    def term_scales(self, term: Any) -> Tuple[float, float]:
        """Return ``(effective_thermal_scale, raw_thermal_scale)`` for an adversarial term."""
        rec = getattr(getattr(term, "gan", None), "record", None)
        if rec is None:
            return 1.0, 1.0
        eff = _get_thermal_scale(rec, 1.0)
        if not math.isfinite(eff) or eff <= 0.0:
            eff = 1.0
        name = self._record_to_name.get(id(rec))
        raw = self.last_raw_scales.get(name, eff) if name is not None else eff
        if not math.isfinite(raw) or raw <= 0.0:
            raw = eff
        return eff, raw

    @contextlib.contextmanager
    def step_scope(self):
        """Hold synchronized thermal scale during the generator phase and commit at step end."""
        if not self.active:
            yield self.effective_scale()
            for name, rec in self.critic_records.items():
                self.last_raw_scales[name] = _get_thermal_scale(rec, 1.0)
            return

        s_step = self.synchronize()
        candidates: Dict[str, float] = {}
        saved_overrides: Dict[str, Tuple[Any, str, bool, Any]] = {}

        for name, rec in self.critic_records.items():
            obs_attr = _thermal_observer_attr(rec)
            if obs_attr is None:
                continue
            orig = getattr(rec, obs_attr)
            had_instance_attr = obs_attr in rec.__dict__
            prev_instance_val = rec.__dict__.get(obs_attr)
            saved_overrides[name] = (rec, obs_attr, had_instance_attr, prev_instance_val)

            def _intercept(
                *args: Any,
                _name: str = name,
                _rec: Any = rec,
                _orig: Callable[..., Any] = orig,
                **kwargs: Any,
            ) -> float:
                s_raw = float(_orig(*args, **kwargs))
                if not math.isfinite(s_raw) or s_raw <= 0.0:
                    s_raw = s_step
                prev = candidates.get(_name)
                candidates[_name] = s_raw if prev is None else max(prev, s_raw)
                # Hold all critics at s_step throughout the current generator step
                # so no term's OT loss or critic hook sees an asymmetric mid-step scale.
                _set_thermal_scale(_rec, s_step)
                return s_step

            setattr(rec, obs_attr, _intercept)

        succeeded = False
        try:
            yield s_step
            succeeded = True
        finally:
            for rec, obs_attr, had_attr, prev_val in saved_overrides.values():
                if had_attr:
                    rec.__dict__[obs_attr] = prev_val
                else:
                    rec.__dict__.pop(obs_attr, None)

        if succeeded:
            per_critic_next = {
                name: candidates.get(name, _get_thermal_scale(rec, s_step))
                for name, rec in self.critic_records.items()
            }
            s_next = min(1.0, max(per_critic_next.values()))
            self.last_raw_scales = per_critic_next
            self.synchronize(s_next)
            min_raw = min(per_critic_next.values())
            if not self._logged_floor_event and min_raw > 0.0 and (s_next / min_raw) >= 2.0:
                self._logged_floor_event = True
                logger.info(
                    "Multi-critic thermal policy=%s prevented cross-critic weight drift: "
                    "raw=%s -> synchronized=%g",
                    self.policy,
                    {k: round(v, 6) for k, v in per_critic_next.items()},
                    s_next,
                )


# ---------------------------------------------------------------------------
# Dual-safe fwAD softmax / log_softmax / SDPA JVP (workaround for pytorch#162350)
# ---------------------------------------------------------------------------

def _no_torch_function(*_args: Any, **_kwargs: Any) -> bool:
    return False


@functools.lru_cache(maxsize=16)
def _unwrap_composite(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a non-mutating function view that skips only the outer torch_function redirect."""
    g = dict(fn.__globals__)
    g["has_torch_function"] = _no_torch_function
    g["has_torch_function_unary"] = _no_torch_function
    g["has_torch_function_variadic"] = _no_torch_function
    return types.FunctionType(
        fn.__code__,
        g,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def _is_dual_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and fwAD.unpack_dual(value).tangent is not None


def _contains_dual(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    for item in args:
        if _is_dual_tensor(item):
            return True
    for item in kwargs.values():
        if _is_dual_tensor(item):
            return True
    return False


def _extract_softmax_args(
    func: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Optional[int], Optional[torch.dtype]]:
    x = args[0] if args else kwargs.get("input")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
    if func in (F.softmax, F.log_softmax):
        # Signature: (input, dim=None, _stacklevel=3, dtype=None)
        dtype = args[3] if len(args) > 3 else kwargs.get("dtype", None)
    elif func in (torch._softmax, torch._log_softmax):
        # Signature: (input, dim, half_to_float)
        half_to_float = bool(args[2]) if len(args) > 2 else bool(kwargs.get("half_to_float", False))
        dtype = torch.float32 if half_to_float else None
    else:
        # torch.softmax / Tensor.softmax / torch.special.softmax / torch._C._nn.softmax
        dtype = args[2] if len(args) > 2 else kwargs.get("dtype", None)
    return x, dim, dtype


def _default_softmax_dim(x: torch.Tensor, dim: Optional[int]) -> int:
    if dim is not None:
        return int(dim)
    # Match PyTorch's _get_softmax_dim legacy fallback when dim is omitted.
    return 0 if x.ndim in (0, 1, 3) else 1


def _dual_safe_softmax(
    x: torch.Tensor,
    dim: Optional[int],
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    primal, tangent = fwAD.unpack_dual(x)
    if tangent is None:
        return None
    eff_dim = _default_softmax_dim(primal, dim)
    out_dtype = dtype if dtype is not None else primal.dtype
    work_dtype = (
        torch.float32
        if out_dtype in (torch.float16, torch.bfloat16)
        else out_dtype
    )
    p_w = primal.to(work_dtype)
    t_w = tangent.to(work_dtype)
    # Plain reverse-mode softmax (no dual tensors inside -> avoids C++ softmax_jvp Exp inplace bug).
    y_w = torch.softmax(p_w, dim=eff_dim)
    # Analytic JVP: d(softmax(p))[t] = y * (t - sum(y * t, dim, keepdim=True)).
    # Guard y == 0 entries so -inf attention masks never produce 0 * inf = NaN.
    pos = y_w > 0
    yt = torch.where(pos, y_w * t_w, torch.zeros_like(y_w))
    dy_w = y_w * (t_w - yt.sum(dim=eff_dim, keepdim=True))
    dy_w = torch.where(pos, dy_w, torch.zeros_like(dy_w))
    return fwAD.make_dual(y_w.to(out_dtype), dy_w.to(out_dtype))


def _dual_safe_log_softmax(
    x: torch.Tensor,
    dim: Optional[int],
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    primal, tangent = fwAD.unpack_dual(x)
    if tangent is None:
        return None
    eff_dim = _default_softmax_dim(primal, dim)
    out_dtype = dtype if dtype is not None else primal.dtype
    work_dtype = (
        torch.float32
        if out_dtype in (torch.float16, torch.bfloat16)
        else out_dtype
    )
    p_w = primal.to(work_dtype)
    t_w = tangent.to(work_dtype)
    lsm_w = F.log_softmax(p_w, dim=eff_dim)
    y_w = torch.softmax(p_w, dim=eff_dim)
    pos = y_w > 0
    yt = torch.where(pos, y_w * t_w, torch.zeros_like(y_w))
    dlsm_w = t_w - yt.sum(dim=eff_dim, keepdim=True)
    return fwAD.make_dual(lsm_w.to(out_dtype), dlsm_w.to(out_dtype))


def _dual_safe_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if is_causal and attn_mask is not None:
        raise RuntimeError("Explicit attn_mask should not be set when is_causal=True")
    seq_q, seq_k = query.size(-2), key.size(-2)
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else float(scale)
    if enable_gqa and query.ndim >= 3 and key.ndim >= 3 and query.size(-3) != key.size(-3):
        rep = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(rep, dim=-3)
        value = value.repeat_interleave(rep, dim=-3)
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        causal_mask = torch.ones((seq_q, seq_k), dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias = torch.zeros((seq_q, seq_k), dtype=attn_weight.dtype, device=query.device)
        attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))
        attn_weight = attn_weight + attn_bias
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask.to(dtype=attn_weight.dtype)
    dual_attn = _dual_safe_softmax(attn_weight, dim=-1)
    attn_weight = dual_attn if dual_attn is not None else torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)
    return torch.matmul(attn_weight, value)


_SOFTMAX_FUNCS = frozenset(
    f
    for f in (
        torch.softmax,
        F.softmax,
        torch.Tensor.softmax,
        getattr(torch, "_softmax", None),
        getattr(torch.special, "softmax", None),
        getattr(torch._C._nn, "softmax", None),
    )
    if f is not None
)
_LOG_SOFTMAX_FUNCS = frozenset(
    f
    for f in (
        torch.log_softmax,
        F.log_softmax,
        torch.Tensor.log_softmax,
        getattr(torch, "_log_softmax", None),
        getattr(torch.special, "log_softmax", None),
        getattr(torch._C._nn, "log_softmax", None),
    )
    if f is not None
)
_SDPA_FUNCS = frozenset(
    f
    for f in (
        F.scaled_dot_product_attention,
        getattr(torch._C._nn, "scaled_dot_product_attention", None),
    )
    if f is not None
)
_PASSTHROUGH_COMPOSITES = frozenset(
    f
    for f in (
        getattr(F, "multi_head_attention_forward", None),
        getattr(F, "softmin", None),
        getattr(F, "gumbel_softmax", None),
    )
    if f is not None
)


class DualSafeSoftmaxMode(TorchFunctionMode):
    """Thread-local ``TorchFunctionMode`` computing exact reverse-safe ``fwAD`` JVPs for softmax ops."""

    def __torch_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kw = kwargs if kwargs is not None else {}
        if func in _SOFTMAX_FUNCS:
            x, dim, dtype = _extract_softmax_args(func, args, kw)
            if _is_dual_tensor(x):
                out = _dual_safe_softmax(x, dim, dtype=dtype)
                if out is not None:
                    return out
        elif func in _LOG_SOFTMAX_FUNCS:
            x, dim, dtype = _extract_softmax_args(func, args, kw)
            if _is_dual_tensor(x):
                out = _dual_safe_log_softmax(x, dim, dtype=dtype)
                if out is not None:
                    return out
        elif func in _SDPA_FUNCS:
            q = args[0] if len(args) > 0 else kw.get("query")
            k = args[1] if len(args) > 1 else kw.get("key")
            v = args[2] if len(args) > 2 else kw.get("value")
            if _is_dual_tensor(q) or _is_dual_tensor(k) or _is_dual_tensor(v):
                return _dual_safe_sdpa(*args, **kw)
        elif func in _PASSTHROUGH_COMPOSITES and _contains_dual(args, kw):
            unwrapped = _unwrap_composite(func)
            with self:
                return unwrapped(*args, **kw)
        return func(*args, **kw)


@contextlib.contextmanager
def critic_penalty_context():
    """Enter math SDPA backend (when available) and ``DualSafeSoftmaxMode`` for critic penalties."""
    with contextlib.ExitStack() as stack:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except Exception:
            sdpa_kernel = None
            SDPBackend = None
        if sdpa_kernel is not None and SDPBackend is not None:
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        stack.enter_context(DualSafeSoftmaxMode())
        yield


def invoke_critic_penalty(
    penalty_fn: Callable[..., torch.Tensor],
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Invoke a ParticleGAN critic penalty under ``critic_penalty_context()``."""
    with critic_penalty_context():
        return penalty_fn(critic, real, fake)

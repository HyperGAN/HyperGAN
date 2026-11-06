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
        for opt_name in ("opt_g", "opt_d"):
            opt = getattr(trainer, opt_name, None)
            if opt is not None:
                opts.append(opt)

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

        # 2. Synchronize critic step record scalar records (opt_d.record)
        rec = getattr(opt, "record", None)
        if rec is not None:
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

        # 3. Synchronize EMA critic network if attached to optimizer
        ema_critic = getattr(opt, "ema_critic", None)
        if ema_critic is not None and isinstance(ema_critic, torch.nn.Module):
            _average_modules([ema_critic], world_size)


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
    return [opt for opt in (getattr(trainer, "opt_g", None), getattr(trainer, "opt_d", None))
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
        params = self._get_named_parameters(models)
        for k, p in params.items():
            if k not in self.anchor:
                self.anchor[k] = p.detach().clone().cpu()
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
                self.anchor[k] = p.detach().clone().cpu()
                self.momentum_buffer[k] = torch.zeros_like(self.anchor[k])

            anchor_t = self.anchor[k].to(device=p.device, dtype=p.dtype)
            delta = anchor_t - p.data

            if is_dist and world_size > 1:
                delta_contig = delta.contiguous()
                dist.all_reduce(delta_contig, op=dist.ReduceOp.SUM)
                delta = delta_contig / world_size

            v_prev = self.momentum_buffer[k].to(device=p.device, dtype=p.dtype)
            v_new = self.outer_momentum * v_prev + delta
            self.momentum_buffer[k] = v_new.detach().clone().cpu()

            new_anchor = anchor_t - self.outer_lr * v_new
            self.anchor[k] = new_anchor.detach().clone().cpu()

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
        # Older checkpoints also carried aliases of the same tensors.
        anchor_data = state.get("anchor") or state.get("anchor_weights") or state.get("weights") or {}
        self.anchor = {k: v.clone() for k, v in anchor_data.items()}
        momentum_data = state.get("momentum_buffer") or state.get("momentum") or {}
        self.momentum_buffer = {k: v.clone() for k, v in momentum_data.items()}

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "sync_every_steps": self.sync_every_steps,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
        }


# ==============================================================================
# Aliases & Public Registry
# ==============================================================================

GradientAllreduceAggregation = GradientAllreduceStrategy
ParameterAverageAggregation = ParameterAverageStrategy
PseudoGradientAggregation = PseudoGradientStrategy

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
    "GradientAllreduceAggregation",
    "ParameterAverageAggregation",
    "PseudoGradientAggregation",
    "register_aggregation_strategy",
    "get_aggregation_strategy",
    "create_aggregation_strategy",
    "resolve_aggregation_strategy",
]

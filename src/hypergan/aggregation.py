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
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, runtime_checkable

import torch
import torch.distributed as dist

from .execution_profiles import register_profile_strategy


@runtime_checkable
class AggregationStrategy(Protocol):
    """Protocol defining the learning aggregation contract with ReplicatedTrainer."""
    name: str

    def pre_step(self, step: int, models: Any = None, optimizers: Any = None) -> None:
        """Pre-optimizer step hook."""
        ...

    def post_step(self, step: int, models: Any = None, optimizers: Any = None, collectives: Any = None) -> None:
        """Post-optimizer step hook."""
        ...

    def on_backward(self, optimizer: Any, name: str) -> None:
        """Post-backward gradient reduction hook."""
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

class GradientAllreduceStrategy:
    """Synchronous post-backward mean gradient all-reduce before optimizer step."""
    name: str = "gradient_allreduce"

    def __init__(self, trainer: Any = None, sync_every_steps: int = 1, **kwargs: Any) -> None:
        if sync_every_steps != 1:
            raise ValueError("gradient_allreduce requires sync_every_steps = 1")
        self.trainer = trainer
        self.sync_every_steps = 1
        self.step_count = 0
        self.settings = kwargs

    def pre_step(self, step: int, models: Any = None, optimizers: Any = None) -> None:
        pass

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

    def on_step_complete(self, step: int) -> None:
        self.step_count = step

    def post_step(self, step: int, models: Any = None, optimizers: Any = None, collectives: Any = None) -> None:
        self.on_step_complete(step)

    def should_assert_replicas(self, step: int) -> bool:
        return True

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

    comm_device = getattr(trainer, "device", None)
    if comm_device is None:
        try:
            backend = dist.get_backend()
            comm_device = torch.device(
                f"cuda:{torch.cuda.current_device()}" if backend == "nccl" else "cpu"
            )
        except Exception:
            comm_device = torch.device("cpu")

    for opt in opts:
        # 1. Synchronize latent and direct history tensor buffers (opt_g)
        for hist_attr in ("latent_history", "direct_history"):
            hist = getattr(opt, hist_attr, None)
            if hist is not None and isinstance(hist, torch.Tensor) and hist.is_floating_point():
                data = hist.data.contiguous()
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
                hist.data.copy_(data)

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
            for p in ema_critic.parameters():
                data = p.data.contiguous()
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
                p.data.copy_(data / world_size)
            for name, buf in ema_critic.named_buffers():
                if buf.is_floating_point():
                    data = buf.data.contiguous()
                    dist.all_reduce(data, op=dist.ReduceOp.SUM)
                    buf.data.copy_(data / world_size)
                else:
                    dist.broadcast(buf.data, src=0)


# ==============================================================================
# Canonical Strategy 2: "parameter_average"
# ==============================================================================


class ParameterAverageStrategy:
    """Periodic cross-rank parameter averaging (FedAvg / Local SGD)."""
    name: str = "parameter_average"

    def __init__(self, trainer: Any = None, sync_every_steps: int = 1, **kwargs: Any) -> None:
        if type(sync_every_steps) is not int or sync_every_steps < 1:
            raise ValueError("aggregation.sync_every_steps must be a positive integer >= 1")
        self.trainer = trainer
        self.sync_every_steps = sync_every_steps
        self.step_count = 0
        self.settings = kwargs

    def pre_step(self, step: int, models: Any = None, optimizers: Any = None) -> None:
        pass

    def on_backward(self, optimizer: Any, name: str) -> None:
        pass

    def should_assert_replicas(self, step: int) -> bool:
        return step % self.sync_every_steps == 0

    def on_step_complete(self, step: int) -> None:
        self.step_count = step
        if self.should_assert_replicas(step):
            self._synchronize()

    def post_step(self, step: int, models: Any = None, optimizers: Any = None, collectives: Any = None) -> None:
        self.step_count = step
        if self.should_assert_replicas(step):
            self._synchronize(models=models, optimizers=optimizers)

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

        model_list = self._get_models(models)
        for model in model_list:
            for p in model.parameters():
                data = p.data.contiguous()
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
                p.data.copy_(data / world_size)

            for name, buf in model.named_buffers():
                if buf.is_floating_point():
                    data = buf.data.contiguous()
                    dist.all_reduce(data, op=dist.ReduceOp.SUM)
                    buf.data.copy_(data / world_size)
                else:
                    dist.broadcast(buf.data, src=0)

        # Synchronize optimizer momentum states across ranks
        opts = []
        if optimizers is not None:
            opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        elif self.trainer is not None:
            for opt_name in ("opt_g", "opt_d"):
                opt = getattr(self.trainer, opt_name, None)
                if opt is not None and hasattr(opt, "state"):
                    opts.append(opt)

        for opt in opts:
            for param_state in getattr(opt, "state", {}).values():
                for key, val in param_state.items():
                    if isinstance(val, torch.Tensor) and val.is_floating_point():
                        data = val.data.contiguous()
                        dist.all_reduce(data, op=dist.ReduceOp.SUM)
                        val.data.copy_(data / world_size)

        _synchronize_optimizer_regularizers(self.trainer, opts)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "step": self.step_count,
            "step_count": self.step_count,
            "sync_every_steps": self.sync_every_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = state.get("step_count", state.get("step", self.step_count))
        self.sync_every_steps = state.get("sync_every_steps", self.sync_every_steps)

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "sync_every_steps": self.sync_every_steps,
        }


# ==============================================================================
# Canonical Strategy 3: "pseudo_gradient" (DiLoCo / FedOpt)
# ==============================================================================

class PseudoGradientStrategy:
    """Federated outer-optimizer update (DiLoCo / FedOpt) using pseudo-gradients."""
    name: str = "pseudo_gradient"

    def __init__(
        self,
        trainer: Any = None,
        sync_every_steps: int = 1,
        outer_lr: float = 1.0,
        outer_momentum: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if type(sync_every_steps) is not int or sync_every_steps < 1:
            raise ValueError("aggregation.sync_every_steps must be a positive integer >= 1")
        if not isinstance(outer_lr, (int, float)) or isinstance(outer_lr, bool) or outer_lr <= 0.0 or not math.isfinite(outer_lr):
            raise ValueError("aggregation.outer_lr must be a finite positive float")
        if not isinstance(outer_momentum, (int, float)) or isinstance(outer_momentum, bool) or not (0.0 <= outer_momentum < 1.0) or not math.isfinite(outer_momentum):
            raise ValueError("aggregation.outer_momentum must be a finite float in [0.0, 1.0)")

        self.trainer = trainer
        self.sync_every_steps = sync_every_steps
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.step_count = 0
        self.settings = kwargs

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

    def _init_anchor_and_momentum(self, models: Any = None) -> None:
        params = self._get_named_parameters(models)
        for k, p in params.items():
            if k not in self.anchor:
                self.anchor[k] = p.detach().clone().cpu()
            if k not in self.momentum_buffer:
                self.momentum_buffer[k] = torch.zeros_like(self.anchor[k])

    def pre_step(self, step: int, models: Any = None, optimizers: Any = None) -> None:
        if not self.anchor:
            self._init_anchor_and_momentum(models)

    def on_backward(self, optimizer: Any, name: str) -> None:
        pass

    def should_assert_replicas(self, step: int) -> bool:
        return step % self.sync_every_steps == 0

    def on_step_complete(self, step: int) -> None:
        self.step_count = step
        if self.should_assert_replicas(step):
            self._synchronize()

    def post_step(self, step: int, models: Any = None, optimizers: Any = None, collectives: Any = None) -> None:
        self.step_count = step
        if self.should_assert_replicas(step):
            self._synchronize(models=models, optimizers=optimizers)

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

        # Synchronize EMA models across ranks to maintain bitwise identical state
        if is_dist and world_size > 1 and self.trainer is not None:
            for ema_attr in ("ema_graph", "ema_prior"):
                ema_mod = getattr(self.trainer, ema_attr, None)
                if isinstance(ema_mod, torch.nn.Module):
                    for p in ema_mod.parameters():
                        data = p.data.contiguous()
                        dist.all_reduce(data, op=dist.ReduceOp.SUM)
                        p.data.copy_(data / world_size)
                    for name, buf in ema_mod.named_buffers():
                        if buf.is_floating_point():
                            data = buf.data.contiguous()
                            dist.all_reduce(data, op=dist.ReduceOp.SUM)
                            buf.data.copy_(data / world_size)
                        else:
                            dist.broadcast(buf.data, src=0)

            # Synchronize optimizer floating moments and regularizers across ranks
            opts = []
            if optimizers is not None:
                opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
            elif self.trainer is not None:
                for opt_name in ("opt_g", "opt_d"):
                    opt = getattr(self.trainer, opt_name, None)
                    if opt is not None and hasattr(opt, "state"):
                        opts.append(opt)

            for opt in opts:
                for param_state in getattr(opt, "state", {}).values():
                    for key, val in param_state.items():
                        if isinstance(val, torch.Tensor) and val.is_floating_point():
                            data = val.data.contiguous()
                            dist.all_reduce(data, op=dist.ReduceOp.SUM)
                            val.data.copy_(data / world_size)

            _synchronize_optimizer_regularizers(self.trainer, opts)

    def state_dict(self) -> Dict[str, Any]:
        if not self.anchor:
            self._init_anchor_and_momentum()
        return {
            "strategy": self.name,
            "step": self.step_count,
            "step_count": self.step_count,
            "sync_every_steps": self.sync_every_steps,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "anchor": {k: v.clone() for k, v in self.anchor.items()},
            "anchor_weights": {k: v.clone() for k, v in self.anchor.items()},
            "momentum": {k: v.clone() for k, v in self.momentum_buffer.items()},
            "momentum_buffer": {k: v.clone() for k, v in self.momentum_buffer.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = state.get("step_count", state.get("step", self.step_count))
        self.sync_every_steps = state.get("sync_every_steps", self.sync_every_steps)
        self.outer_lr = state.get("outer_lr", self.outer_lr)
        self.outer_momentum = state.get("outer_momentum", self.outer_momentum)
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

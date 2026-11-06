"""Shared distributed-test fixtures: a toy recipe and the aggregation oracles.

Unit, reference and e2e tests import these from here, so no shipped test tier
depends on the e2e harness. Tests import it as ``tests.distributed_fixtures``,
or as ``distributed_fixtures`` from worker scripts that put ``tests/`` on
``PYTHONPATH``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from hypergan.config import resolve_config


def make_toy_config(
    device: str = "cpu",
    batch_size: int = 8,
    steps: int = 3,
    extra_training: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal fast deterministic training configuration."""
    base: Dict[str, Any] = {
        'components': {
            'generator': {
                'factory': 'mlp',
                'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]},
                'inputs': {'x': 'latent'},
            },
            'discriminator': {
                'factory': 'linear',
                'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                'inputs': {'input': 'candidate'},
            },
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {
            'steps': steps,
            'batch_size': batch_size,
            'device': device,
            'seed': 42,
        },
        'sampling': {'count': 4},
    }
    if extra_training:
        base['training'].update(extra_training)
    return resolve_config(base)


def oracle_parameter_average(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Authoritative FedAvg mathematical oracle:
    Theta_mean = (1 / W) * sum_{i=1}^W Theta_i
    """
    assert len(weights_list) > 0, "weights_list cannot be empty"
    avg_state: Dict[str, torch.Tensor] = {}
    for key in weights_list[0].keys():
        stacked = torch.stack([w[key] for w in weights_list], dim=0)
        avg_state[key] = torch.mean(stacked, dim=0)
    return avg_state


def oracle_pseudo_gradient_step(
    anchor_state: Dict[str, torch.Tensor],
    replica_states: List[Dict[str, torch.Tensor]],
    outer_lr: float = 1.0,
    outer_momentum: float = 0.0,
    momentum_buffer: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Authoritative DiLoCo / FedOpt mathematical oracle (heavy-ball outer momentum):
    DeltaTheta_i = Theta_anchor - Theta_i
    DeltaTheta = (1 / W) * sum_{i=1}^W DeltaTheta_i
    v_{t} = outer_momentum * v_{t-1} + DeltaTheta
    Theta_anchor^{new} = Theta_anchor - outer_lr * v_{t}
    """
    new_anchor: Dict[str, torch.Tensor] = {}
    new_momentum: Dict[str, torch.Tensor] = {}

    if momentum_buffer is None:
        momentum_buffer = {k: torch.zeros_like(v) for k, v in anchor_state.items()}

    for key, anchor_val in anchor_state.items():
        deltas = [anchor_val - rep[key] for rep in replica_states]
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
        v_prev = momentum_buffer.get(key, torch.zeros_like(anchor_val))
        v_new = outer_momentum * v_prev + mean_delta
        new_momentum[key] = v_new
        new_anchor[key] = anchor_val - outer_lr * v_new

    return new_anchor, new_momentum


def assert_states_close(
    left: Any,
    right: Any,
    rtol: float = 3e-5,
    atol: float = 3e-6,
    path: str = "root",
) -> None:
    """Recursively compare state dictionaries, tensors, floats, or collections."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"{path}: right is not a torch.Tensor ({type(right)})"
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=lambda m: f"{path}: {m}")
    elif isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, rel=rtol, abs=atol), f"{path}: {left} != {right}"
    elif isinstance(left, dict):
        assert isinstance(right, dict), f"{path}: right is not a dict ({type(right)})"
        assert set(left.keys()) == set(right.keys()), f"{path}: keys differ ({set(left.keys()) ^ set(right.keys())})"
        for k in left:
            assert_states_close(left[k], right[k], rtol=rtol, atol=atol, path=f"{path}.{k}")
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)), f"{path}: right is not a sequence ({type(right)})"
        assert len(left) == len(right), f"{path}: sequence length mismatch ({len(left)} vs {len(right)})"
        for idx, (a, b) in enumerate(zip(left, right)):
            assert_states_close(a, b, rtol=rtol, atol=atol, path=f"{path}[{idx}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"

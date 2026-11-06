"""Pre-credit checks for policies that must not silently change across GPUs."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.execution_profiles import resolve_execution_profile


@pytest.mark.parametrize('policy', [
    {'input_noise_std': .05},
    {'output_noise_std': .05},
    {'phase_draws': 'independent'},
])
def test_accumulation_refuses_unimplemented_numerical_policies(policy):
    config = resolve_config({'training': {'device': 'cpu', 'batch_size': 8, **policy}})
    profile = {'schema_version': 1, 'execution': {
        'name': 'cpu-replicated-gloo', 'accumulation_steps': 2}}
    with pytest.raises(ValueError, match='accumulation_steps=1'):
        resolve_execution_profile(profile, config)
    # These settings are implemented by the retained-graph replicated path.
    profile['execution']['accumulation_steps'] = 1
    resolve_execution_profile(profile, config)


def test_reduced_gradient_overflow_is_refused_before_optimizer_step(monkeypatch):
    trainer = object.__new__(ReplicatedTrainer)
    trainer.device, trainer.world_size = torch.device('cpu'), 2
    trainer.aggregation_strategy = SimpleNamespace(
        reduces_gradients=True, on_backward=lambda optimizer, name: None)
    parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    optimizer = torch.optim.Adam(parameters)
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, torch.finfo(torch.float32).max * .75)

    def reduce(value, **kwargs):
        value.mul_(2)

    monkeypatch.setattr(dist, 'all_reduce', reduce)
    with pytest.raises(ValueError, match='Nonfinite reduced gradient'):
        trainer._reduce_gradients(optimizer, 'overflow')
    assert all(parameter.grad is None for parameter in parameters)
    assert not optimizer.state
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for parameter in parameters)

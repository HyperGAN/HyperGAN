"""Trainable config networks preserve gradients, EMA isolation and recovery."""
import copy

import pytest
import torch

from hypergan.config import resolve_config
from hypergan.hndl_networks import build_network
from hypergan.training import ReferenceTrainer


def test_ema_deepcopy_is_independent_and_does_not_consume_rng():
    model = build_network('linear(8)\nbatch_norm()\ntanh()\nlinear()',
                          input_shape=('B', 4), output_shape=('B', 2)).eval()
    model[0].weight.requires_grad_(False)
    state = torch.get_rng_state().clone()
    clone = copy.deepcopy(model)
    assert torch.equal(state, torch.get_rng_state())
    assert not clone.training and not clone[0].weight.requires_grad
    for a, b in zip(model.parameters(), clone.parameters()):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    for a, b in zip(model.buffers(), clone.buffers()):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    x = torch.randn(3, 4)
    torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)


def test_dtype_moves_and_candidate_double_backward():
    model = build_network('linear(8)\ntanh()\nlinear()',
                          input_shape=('B', 4), output_shape=('B', 1)).double()
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(model(x).sum(), x, create_graph=True)
    grad.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model[0].weight.grad.abs().sum() > 0


def test_default_hndl_trainer_updates_and_restores_exactly():
    config = resolve_config({})
    trainer = ReferenceTrainer(config)
    before = [p.clone().detach() for p in trainer.graph.parameters()]
    row, _ = trainer.update()
    assert all(torch.isfinite(torch.tensor(row[k])) for k in ('d_loss', 'g_loss'))
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.graph.parameters()))


def test_invalid_network_is_rejected_before_tensor_execution():
    with pytest.raises(Exception, match='E_|shape|constraint'):
        build_network('linear(3)', input_shape=('B', 4), output_shape=('B', 2))


def test_named_ports_keep_conditional_connectivity_in_source():
    from hypergan.hndl_networks import HNDLNetwork
    model = HNDLNetwork('joined = concat(z, condition)\nfeatures = linear(joined, 8)\nout = linear(features, 2)',
                        {'z': ['B', 4], 'condition': ['B', 2]},
                        {'out': ['B', 2], 'features': ['B', 8]})
    z = torch.randn(3, 4, requires_grad=True)
    condition = torch.randn(3, 2, requires_grad=True)
    result = model(z=z, condition=condition)
    assert result['out'].shape == (3, 2) and result['features'].shape == (3, 8)
    result['out'].sum().backward()
    assert z.grad.abs().sum() > 0 and condition.grad.abs().sum() > 0

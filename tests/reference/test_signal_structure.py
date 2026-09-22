"""Ownership checks for read-only generator transmission probes."""
import torch
from torch import nn
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.signal_structure import _inventory


def trainer():
    return ReferenceTrainer(resolve_config({}))


def test_pretrained_trainable_descendants_and_storage_aliases_excluded():
    from hndl.operators.pretrained import Pretrained
    t = trainer()
    g = t.graph.models['generator']
    # Construction-free native marker, avoiding downloads/provider loading.
    frozen = Pretrained.__new__(Pretrained)
    nn.Module.__init__(frozen)
    frozen.inner = nn.Linear(4, 4)
    g.add_module('pretrained_fixture', frozen)
    from types import SimpleNamespace
    t.program = SimpleNamespace(generator_parameters=tuple(t.program.generator_parameters) + tuple(frozen.parameters()))
    layers, _ = _inventory(t)
    assert all('pretrained_fixture' not in name for name, _ in layers)
    first_name, first = layers[0]
    # A different Parameter object sharing pretrained storage is protected too.
    frozen.register_parameter('tied', nn.Parameter(first.weight.detach()))
    layers, _ = _inventory(t)
    assert first_name not in dict(layers)


def test_uncertain_custom_subtree_is_not_calibratable():
    from types import SimpleNamespace
    t = trainer()
    class UnknownOwner(nn.Module):
        def __init__(self):
            super().__init__()
            self.affine = nn.Linear(4, 4)
    unknown = UnknownOwner()
    t.graph.models['generator'].add_module('unknown', unknown)
    t.program = SimpleNamespace(generator_parameters=tuple(t.program.generator_parameters) + tuple(unknown.parameters()))
    layers, _ = _inventory(t)
    assert all('unknown' not in name for name, _ in layers)



def test_unknown_generator_ownership_is_reported():
    t = trainer()
    t.config["components"]["generator"]["factory"] = "custom:Generator"
    layers, reason = _inventory(t)
    assert layers == [] and reason

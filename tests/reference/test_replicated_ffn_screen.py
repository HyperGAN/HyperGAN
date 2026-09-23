"""Replication separates initial function from Adam's width-dependent update."""
from collections import OrderedDict
import importlib.util
from pathlib import Path

import torch
from torch import nn


def test_replication_and_compensation_preserve_functional_adam_updates(monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'research/startup_tuning'
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('replicated_ffn_screen', root / 'replicated_ffn_screen.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def model(width):
        return nn.Sequential(OrderedDict(ffn=nn.Sequential(OrderedDict(
            up=nn.Linear(3, width), act=nn.GELU(), down=nn.Linear(width, 2))))).double()

    source, wide, compensated = model(4), model(16), model(16)
    with torch.no_grad():
        for i, parameter in enumerate(source.parameters()):
            parameter.copy_(torch.sin(torch.arange(parameter.numel()).reshape(parameter.shape) + i) * .2)
    for target in (wide, compensated):
        alignment = runner.replicate_state(target.state_dict(), source.state_dict(), {'ffn': 4})
        assert alignment['exact_source_sha256'] == alignment['exact_target_sha256']
    x = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 10
    target_y = torch.cos(x[:, :2])
    for target in (wide, compensated):
        torch.testing.assert_close(source(x), target(x), atol=1e-14, rtol=1e-14)

    opts = [torch.optim.Adam(source.parameters(), lr=.003, betas=(0., .999)),
            torch.optim.Adam(wide.parameters(), lr=.003, betas=(0., .999)),
            torch.optim.Adam([
                {'params': compensated.ffn.up.parameters(), 'eps': 1e-8 / 4},
                {'params': [compensated.ffn.down.weight], 'lr': .003 / 4},
                {'params': [compensated.ffn.down.bias]},
            ], lr=.003, betas=(0., .999))]
    for step in range(8):
        for model_, opt in zip((source, wide, compensated), opts):
            opt.zero_grad()
            (model_(x) - target_y).square().mean().backward()
            opt.step()
        torch.testing.assert_close(source(x), compensated(x), atol=1e-12, rtol=1e-12)
        if step == 0:
            assert float((source(x) - wide(x)).abs().max().detach()) > .001

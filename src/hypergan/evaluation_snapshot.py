"""Capture complete-boundary EMA inference state without training aliases or I/O."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import torch

from .artifacts import bundle_state
from .checkpoints import capture_rng, restore_rng
from .preview_snapshot import _freeze_cpu_state


def capture_evaluation_state(trainer, identity):
    rng, threads = capture_rng(), torch.get_num_threads()
    streams = {name: stream.get_state() for name, stream in trainer.streams.items()}
    try:
        memo = {id(module): module for module in list(sys.modules.values()) if isinstance(module, ModuleType)}
        snapshot = SimpleNamespace(config=copy.deepcopy(trainer.config), step=trainer.step,
            ema_graph=copy.deepcopy(trainer.ema_graph, memo), ema_prior=copy.deepcopy(trainer.ema_prior, memo),
            artifact_identity=copy.deepcopy(identity))
        # Evaluation supplies its own explicit conditioning/data. No current
        # training batch or live model crosses the supervised worker boundary.
        return _freeze_cpu_state(bundle_state(snapshot, {}))
    finally:
        restore_rng(rng)
        for name, value in streams.items():
            trainer.streams[name].set_state(value)
        torch.set_num_threads(threads)

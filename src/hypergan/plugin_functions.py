"""Plain-function adapters for the existing metric plugin protocol.

Users write ordinary functions; these classes give them the engine's
``describe()`` / ``evaluate()`` contract, so they run in the same bounded workers
with the same scheduling, receipts and catalog as class-based metrics.

A metric (cheap, every few steps) is a function of update scalars::

    def d_over_g(d, g):            # inputs = { d = "d_loss", g = "g_loss" }
        return d / g

An evaluation (expensive, from an EMA snapshot) is a function of the whole
generated and reference sets, concatenated along the batch axis::

    def modes_covered(generated, reference, side=10):
        ...
        return 0.93

``reduce = "stream"`` passes the bounded batch iterator instead, for sets too
large to hold at once. Returning ``{"edges": [...], "counts": [...]}`` with
``kind = "histogram"`` publishes a distribution.

The function's source file hash is part of the metric description, so editing
the function gives the metric a new definition hash on the next attempt.
"""
import hashlib
import inspect
from pathlib import Path

from .item_data import import_object


def _described(fn_path, fn, kind, label, unit, direction, description):
    source = inspect.getsourcefile(fn)
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else 'unavailable'
    text = f'{description + " " if description else ""}fn={fn_path} source_sha256={digest}'
    return {'kind': kind, 'label': label or fn_path.split(':')[1], 'unit': unit,
            'direction': direction, 'description': text[:1000]}


class ScalarFunction:
    """A metric: ``fn(**bound_update_scalars, **args) -> number``."""

    def __init__(self, fn, args=None, label=None, unit='value', direction='none', description=None):
        self.fn_path, self.fn, self.args = fn, import_object(fn), dict(args or {})
        self._description = _described(fn, self.fn, 'scalar', label, unit, direction, description)

    def describe(self):
        return dict(self._description)

    def evaluate(self, *, context, **inputs):
        return float(self.fn(**inputs, **self.args))


class SnapshotFunction:
    """An evaluation: ``fn(generated=..., reference=..., **args)`` over the full set."""

    def __init__(self, fn, args=None, reduce='concat', kind='scalar', label=None, unit='value',
                 direction='none', description=None):
        if reduce not in ('concat', 'stream'):
            raise ValueError('reduce must be concat or stream')
        if kind not in ('scalar', 'histogram'):
            raise ValueError('kind must be scalar or histogram')
        self.fn_path, self.fn, self.args, self.reduce, self.kind = fn, import_object(fn), dict(args or {}), reduce, kind
        self._description = _described(fn, self.fn, kind, label, unit, direction, description)

    def describe(self):
        return dict(self._description)

    def evaluate(self, *, batches, context):
        if self.reduce == 'stream':
            value = self.fn(batches, **self.args)
        else:
            import torch
            collected = {}
            for batch in batches:
                for key, tensor in batch.items():
                    collected.setdefault(key, []).append(tensor.detach().cpu())
            value = self.fn(**{key: torch.cat(parts) for key, parts in collected.items()}, **self.args)
        if self.kind == 'histogram':
            return {'edges': [float(x) for x in value['edges']], 'counts': [float(x) for x in value['counts']]}
        return float(value)

"""Import-path adapters that connect user code to the existing engine contracts.

Configs reference these as ``hypergan.graph.adapters:<Name>`` with the user's
own class or function as an argument, so every process imports the same code
and nothing is copied. Torch is imported lazily.
"""
import hashlib
import importlib
import inspect
from pathlib import Path


def resolve(reference):
    module, qualname = reference.split(":")
    value = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def source_sha256(reference):
    """Hash of the module file that defines ``reference`` (provenance, not identity)."""
    module = inspect.getmodule(resolve(reference))
    path = Path(getattr(module, "__file__", "") or "")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _collate(items, location="item"):
    import numpy as np
    import torch
    first = items[0]
    if isinstance(first, dict):
        keys = list(first)
        if any(not isinstance(item, dict) or list(item) != keys for item in items):
            raise ValueError(f"{location}: every item must have the same fields in the same order")
        return {key: _collate([item[key] for item in items], f"{location}.{key}") for key in keys}
    if isinstance(first, (list, tuple)) and not isinstance(first, str):
        if any(len(item) != len(first) for item in items):
            raise ValueError(f"{location}: sequence items must have equal length")
        return [_collate([item[i] for item in items], f"{location}[{i}]") for i in range(len(first))]
    if isinstance(first, np.ndarray):
        items = [torch.from_numpy(np.asarray(item)) for item in items]
        first = items[0]
    if isinstance(first, torch.Tensor):
        value = torch.stack(items)
    elif isinstance(first, (bool, int, float)):
        value = torch.tensor(items)
    else:
        raise TypeError(f"{location}: items must be tensors, arrays, numbers, or dicts/lists of them "
                        f"(got {type(first).__name__})")
    return value.float() if value.dtype == torch.float64 else value


class ItemDataset:
    """Batches an item-level dataset: the user writes only ``__len__``/``__getitem__``.

    Order: a fresh permutation per epoch drawn from HyperGAN's data stream
    (seeded by the run), so every process computes the same index plan. The
    position is saved in checkpoints. ``plan`` and ``load`` are separate so a
    sharded executor can give each rank only its slice of the plan.
    """

    def __init__(self, dataset, args=None, real="real", shuffle=True, workers=0):
        self.reference, self.args, self.real = dataset, dict(args or {}), real
        self.shuffle, self.workers = bool(shuffle), int(workers)
        self.dataset = resolve(dataset)(**self.args)
        self.length = len(self.dataset)
        if self.length < 1:
            raise ValueError(f"{dataset} is empty")
        self.order, self.position, self.epoch = None, 0, 0
        self._pool = None

    def plan(self, batch_size, generator):
        """The next ``batch_size`` indices, crossing epoch boundaries as needed."""
        import torch
        indices = []
        while len(indices) < batch_size:
            if self.order is None or self.position >= self.length:
                if self.order is not None:
                    self.epoch += 1
                if self.shuffle:
                    self.order = torch.randperm(self.length, generator=generator, device=generator.device).cpu()
                else:
                    self.order = torch.arange(self.length)
                self.position = 0
            take = min(batch_size - len(indices), self.length - self.position)
            indices.extend(self.order[self.position:self.position + take].tolist())
            self.position += take
        return indices

    def load(self, indices):
        """Load and collate the given items; the real field is renamed to ``real``."""
        get = getattr(self.dataset, "__getitem__", None) or self.dataset.load
        if self.workers > 0:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="hypergan-items")
            items = list(self._pool.map(get, indices))
        else:
            items = [get(index) for index in indices]
        if not isinstance(items[0], dict):
            items = [{"real": item} for item in items]
        elif self.real != "real":
            if self.real not in items[0]:
                raise ValueError(f"Items have no field {self.real!r} (fields: {sorted(items[0])})")
            if "real" in items[0]:
                raise ValueError(f"Items have both {self.real!r} and 'real'; rename one")
            items = [{("real" if k == self.real else k): v for k, v in item.items()} for item in items]
        return _collate(items)

    def __call__(self, batch_size, *, generator):
        return self.load(self.plan(batch_size, generator))

    def state_dict(self):
        return {"order": self.order, "position": self.position, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.order, self.position, self.epoch = state["order"], int(state["position"]), int(state["epoch"])

    def resume_identity(self):
        identity = {"dataset": self.reference, "args": self.args, "real": self.real, "shuffle": self.shuffle,
                    "length": self.length, "source_sha256": source_sha256(self.reference)}
        custom = getattr(self.dataset, "identity", None)
        if callable(custom):
            identity["identity"] = custom()
        return identity

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_pool"] = None
        return state


class FunctionLoss:
    """Calls ``function(**inputs, **args)``; the function holds no state."""
    resume_stateless = True

    def __init__(self, function, args=None):
        self.function, self.args = resolve(function), dict(args or {})

    def __call__(self, **inputs):
        return self.function(**inputs, **self.args)


class FunctionMetric:
    """A scalar metric from a plain function of update scalars."""

    def __init__(self, function, label=None, unit=None, direction="none"):
        self.reference, self.function = function, resolve(function)
        self.label, self.unit, self.direction = label, unit, direction

    def describe(self):
        # The source hash is part of the description, so editing the function
        # changes the metric's definition hash on the next attempt.
        described = {"kind": "scalar", "label": self.label or self.reference.split(":")[1], "direction": self.direction,
                     "description": f"{self.reference} (source sha256 {source_sha256(self.reference)})"}
        if self.unit:
            described["unit"] = self.unit
        return described

    def evaluate(self, *, context, **inputs):
        return float(self.function(**inputs))


class FunctionEvaluation:
    """A snapshot evaluation from ``function(generated, reference) -> float``.

    The declared sample count is consumed and concatenated first, so the
    function sees whole tensors. Return a dict ``{"edges", "counts"}`` and
    set ``kind="histogram"`` for a distribution.
    """

    def __init__(self, function, args=None, kind="scalar"):
        self.reference, self.function, self.args, self.kind = function, resolve(function), dict(args or {}), kind

    def describe(self):
        return {"kind": self.kind, "label": self.reference.split(":")[1],
                "description": f"{self.reference} (source sha256 {source_sha256(self.reference)})"}

    def evaluate(self, *, batches, context):
        import torch
        collected = {}
        for batch in batches:
            for key, value in batch.items():
                collected.setdefault(key, []).append(value.detach().cpu())
        value = self.function(**{k: torch.cat(v) for k, v in collected.items()}, **self.args)
        return float(value) if self.kind == "scalar" else value

"""Engine-side adapters for plain user functions and item-level datasets.

The config names these by import path, like any factory, and each names the
user's object by import path in turn. Workers import both; nothing is copied.
"""
import hashlib
import importlib
import inspect
from pathlib import Path


def load_reference(path):
    module, qualname = path.split(':')
    target = importlib.import_module(module)
    for part in qualname.split('.'):
        target = getattr(target, part)
    return target


def source_sha256(path):
    """Content hash of the module that defines ``path``: user edits change provenance."""
    module = importlib.import_module(path.split(':')[0])
    file = Path(getattr(module, '__file__', '') or '')
    return hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None


# ------------------------------------------------------------------ losses

def function_loss(function):
    """Return the user's loss function itself, so the engine hashes its module for
    provenance. A plain function holds no state, which is what resume requires."""
    target = load_reference(function)
    if not inspect.isfunction(target):
        raise ValueError(f'{function} must be a plain function f(**inputs) -> scalar tensor')
    target.resume_stateless = True
    return target


# ------------------------------------------------------------------ metrics

def _call(function, values, context):
    parameters = inspect.signature(function).parameters
    if 'context' in parameters:
        return function(**values, context=context)
    return function(**values)


class FunctionMetric:
    """A scalar metric from a plain function of update scalars."""

    def __init__(self, function, label=None, unit='value', direction='none'):
        self.reference = function
        self.function = load_reference(function)
        self.label, self.unit, self.direction = label or function.split(':')[1], unit, direction

    def describe(self):
        # The source hash is in the description so an edit changes the definition hash.
        return {'kind': 'scalar', 'label': self.label, 'unit': self.unit, 'direction': self.direction,
                'description': f'{self.reference} (source sha256 {source_sha256(self.reference)})'}

    def evaluate(self, *, context, **inputs):
        return float(_call(self.function, inputs, context))


class FunctionEvaluation:
    """A snapshot evaluation from a plain function of evaluation batches."""

    def __init__(self, function, label=None, unit='value', direction='none', kind='scalar'):
        if kind not in ('scalar', 'histogram'):
            raise ValueError('evaluation kind must be scalar or histogram')
        self.reference = function
        self.function = load_reference(function)
        self.label, self.unit, self.direction, self.kind = label or function.split(':')[1], unit, direction, kind

    def describe(self):
        return {'kind': self.kind, 'label': self.label, 'unit': self.unit, 'direction': self.direction,
                'description': f'{self.reference} (source sha256 {source_sha256(self.reference)})'}

    def evaluate(self, *, batches, context):
        result = _call(self.function, {'batches': batches}, context)
        return result if self.kind == 'histogram' else float(result)


# ------------------------------------------------------------------ data

def _collate(items):
    import torch
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    if isinstance(first, dict):
        return {key: _collate([item[key] for item in items]) for key in first}
    if isinstance(first, (list, tuple)):
        return type(first)(_collate(list(group)) for group in zip(*items))
    if isinstance(first, (bool, int, float)):
        return torch.tensor(items)
    raise ValueError(f'Dataset items must be tensors, numbers or dicts/lists of them, got {type(first).__name__}')


class ItemData:
    """Batches from an item-level dataset (``__len__`` + ``__getitem__`` or ``load(i)``).

    HyperGAN owns everything outside the user's item code: the epoch order
    (a permutation drawn from the run's data RNG), the resume position
    (``state_dict``), provenance (``resume_identity``) and sharding: replicated
    execution draws one global batch and slices it per rank; ``indices`` and
    ``load`` are separate so a multi-host runner can draw global indices
    everywhere and load only its own slice.
    """

    def __init__(self, dataset, args=None, shuffle=True, field='real'):
        self.reference, self.args = dataset, dict(args or {})
        self.dataset = load_reference(dataset)(**self.args)
        self.shuffle, self.field = shuffle, field
        self.length = len(self.dataset)
        if type(self.length) is not int or self.length < 1:
            raise ValueError(f'{dataset} must have a positive integer length')
        getter = getattr(self.dataset, 'load', None)
        self._item = getter if callable(getter) else self.dataset.__getitem__
        self._permutation, self._cursor, self._epoch = None, 0, 0

    def __len__(self):
        return self.length

    def indices(self, batch_size, generator):
        import torch
        chosen = []
        while len(chosen) < batch_size:
            if self._permutation is None or self._cursor >= self.length:
                if self._permutation is not None:
                    self._epoch += 1
                self._permutation = (torch.randperm(self.length, generator=generator, device=generator.device).cpu()
                                     if self.shuffle else torch.arange(self.length))
                self._cursor = 0
            take = min(batch_size - len(chosen), self.length - self._cursor)
            chosen.extend(self._permutation[self._cursor:self._cursor + take].tolist())
            self._cursor += take
        return chosen

    def load(self, indices):
        batch = _collate([self._item(index) for index in indices])
        return batch if isinstance(batch, dict) else {self.field: batch}

    def __call__(self, batch_size, *, generator):
        return self.load(self.indices(batch_size, generator))

    def state_dict(self):
        import torch
        return {'permutation': torch.empty(0, dtype=torch.int64) if self._permutation is None else self._permutation,
                'started': self._permutation is not None, 'cursor': self._cursor, 'epoch': self._epoch}

    def load_state_dict(self, state):
        self._permutation = state['permutation'].clone() if state['started'] else None
        self._cursor, self._epoch = int(state['cursor']), int(state['epoch'])

    def resume_identity(self):
        return {'kind': 'hypergan-item-dataset', 'dataset': self.reference, 'args': self.args,
                'length': self.length, 'shuffle': self.shuffle, 'field': self.field,
                'source_sha256': source_sha256(self.reference)}

"""Item-level datasets: the user says how to load item i; HyperGAN owns the rest.

A model file names a dataset class by import path::

    [data]
    dataset = "my_project.data:Points"   # __len__ and __getitem__(i) (or load(i))
    args = { path = "points.npy" }

The dataset returns one item: a tensor (the real sample) or a mapping/list of
tensors, numbers or arrays, nested as deeply as needed. A mapping item must
contain ``real``, the sample the primary critic compares against; every other
field is available to bindings as ``batch.<field>``.

``ItemData`` is the batch-level factory the engine already calls,
``__call__(batch_size, *, generator)``. It owns order, shuffling, epoch
boundaries, collation and resume position. Its randomness comes only from the
engine's data generator, and its checkpoint state is three integers (epoch,
cursor, epoch seed), independent of dataset size.

Sharding: today the engine draws the global batch in every replicated rank and
keeps that rank's slice (``replicated-global-draw-rank-slice``). ``indices`` is
the global index order for a draw, so a later engine can ask each rank or host
to load only its slice of it without changing a model file, the order, or the
resume state.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json


def import_object(reference):
    """Resolve ``module:object`` (object may be dotted)."""
    if not isinstance(reference, str) or reference.count(':') != 1 or not all(reference.split(':')):
        raise ValueError(f'Expected a module:object import path, got {reference!r}')
    module_name, attribute = reference.split(':')
    value = importlib.import_module(module_name)
    for part in attribute.split('.'):
        value = getattr(value, part)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def collate(items):
    """Stack a list of items (tensors, numbers, arrays, or nested dicts/lists of them)."""
    import torch
    first = items[0]
    if isinstance(first, dict):
        if any(not isinstance(item, dict) or item.keys() != first.keys() for item in items):
            raise ValueError('Every dataset item must have the same fields')
        return {key: collate([item[key] for item in items]) for key in first}
    if isinstance(first, (list, tuple)):
        if any(not isinstance(item, (list, tuple)) or len(item) != len(first) for item in items):
            raise ValueError('Every dataset item list must have the same length')
        return [collate([item[i] for item in items]) for i in range(len(first))]
    tensors = [torch.as_tensor(item) for item in items]
    if any(t.shape != tensors[0].shape for t in tensors):
        raise ValueError('Dataset items must have one shape per field to form a batch')
    batch = torch.stack(tensors)
    return batch.float() if batch.dtype == torch.float64 else batch


class ItemData:
    """Batch factory over an item-level dataset (resumable, seeded by the engine)."""

    def __init__(self, dataset, args=None, shuffle=True, workers=0):
        if type(shuffle) is not bool:
            raise ValueError('data.shuffle must be boolean')
        if type(workers) is not int or workers < 0:
            raise ValueError('data.workers must be a nonnegative integer')
        self.reference, self.args, self.shuffle, self.workers = dataset, dict(args or {}), shuffle, workers
        self.dataset = import_object(dataset)(**self.args)
        self.length = len(self.dataset)
        if type(self.length) is not int or self.length < 1:
            raise ValueError(f'{dataset} must have a positive integer length')
        load = getattr(self.dataset, 'load', None)
        self._load = load if callable(load) else self.dataset.__getitem__
        custom = getattr(self.dataset, 'identity', None)
        self._identity = {'schema_version': 1, 'kind': 'item_data', 'dataset': dataset, 'args': self.args,
                          'shuffle': shuffle, 'length': self.length,
                          'dataset_identity': custom() if callable(custom) else None}
        self._identity = json.loads(json.dumps(self._identity, sort_keys=True, allow_nan=False))
        self._identity_sha256 = _digest(self._identity)
        self._epoch, self._cursor, self._epoch_seed = 0, self.length, 0
        self._pool = None

    def _order(self):
        import torch
        if not self.shuffle:
            return None
        return torch.randperm(self.length, generator=torch.Generator().manual_seed(self._epoch_seed))

    def indices(self, batch_size, generator):
        """Global dataset indices for the next batch; advances the sampler."""
        result = []
        order = self._order() if self._cursor < self.length else None
        while len(result) < batch_size:
            if self._cursor == self.length:
                import torch
                # One draw from the engine's data stream per epoch; the epoch's
                # permutation is a pure function of it.
                self._epoch_seed = int(torch.randint(2 ** 62, (1,), generator=generator).item())
                self._epoch, self._cursor = self._epoch + 1, 0
                order = self._order()
            take = min(batch_size - len(result), self.length - self._cursor)
            span = range(self._cursor, self._cursor + take)
            result.extend(span if order is None else order[self._cursor:self._cursor + take].tolist())
            self._cursor += take
        return result

    def __call__(self, batch_size, *, generator):
        import torch
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        saved = (self._epoch, self._cursor, self._epoch_seed, generator.get_state())
        try:
            indices = self.indices(batch_size, generator)
            if self.workers:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='item-data')
                items = list(self._pool.map(self._load, indices))
            else:
                items = [self._load(index) for index in indices]
            items = [item if isinstance(item, (dict, list, tuple)) else {'real': item} for item in items]
            batch = collate(items)
            if not isinstance(batch, dict) or not isinstance(batch.get('real'), torch.Tensor):
                raise ValueError(f'{self.reference} items must be a tensor or a mapping with a real field')
            return batch
        except BaseException:
            # A failed draw leaves no partial position, so retry and resume agree.
            self._epoch, self._cursor, self._epoch_seed = saved[:3]
            generator.set_state(saved[3])
            raise

    def resume_identity(self):
        return json.loads(json.dumps(self._identity))

    def state_dict(self):
        return {'schema_version': 1, 'identity_sha256': self._identity_sha256,
                'epoch': self._epoch, 'cursor': self._cursor, 'epoch_seed': self._epoch_seed}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get('schema_version') != 1
                or set(state) != {'schema_version', 'identity_sha256', 'epoch', 'cursor', 'epoch_seed'}):
            raise ValueError('Invalid item_data sampler state')
        if state['identity_sha256'] != self._identity_sha256:
            raise ValueError('Item dataset identity (import path, args, length or dataset identity()) changed since the checkpoint')
        if any(type(state[k]) is not int or state[k] < 0 for k in ('epoch', 'cursor', 'epoch_seed')) or state['cursor'] > self.length:
            raise ValueError('Invalid item_data sampler position')
        self._epoch, self._cursor, self._epoch_seed = state['epoch'], state['cursor'], state['epoch_seed']

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_pool'] = None
        return state

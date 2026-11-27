"""Pinned local CIFAR-10 bytes with explicit training/evaluation sampling."""
import hashlib
from pathlib import Path
import pickle

import numpy as np
import torch

from .data import EvaluationDataExhausted
from .dataset_download import CIFAR10_SHA256


class CIFAR10Data:
    """Source train draws use replacement then per-image horizontal flips.

    Sequential mode starts at zero and refuses to wrap; use it without flips
    for the complete unaugmented training reference distribution in FID.
    All randomness comes from the caller's checkpointed torch.Generator.
    """
    def __init__(self, root, split='train', sampling='random', horizontal_flip=True):
        if split not in {'train', 'test'} or sampling not in {'random', 'sequential'}:
            raise ValueError('CIFAR split must be train/test and sampling random/sequential')
        if type(horizontal_flip) is not bool:
            raise ValueError('horizontal_flip must be a boolean')
        self.root = Path(root).expanduser()
        if (self.root / 'cifar-10-batches-py').is_dir():
            self.root = self.root / 'cifar-10-batches-py'
        self.split, self.sampling, self.horizontal_flip = split, sampling, horizontal_flip
        self.files = [f'data_batch_{i}' for i in range(1, 6)] if split == 'train' else ['test_batch']
        arrays, labels = [], []
        for name in self.files:
            path = (self.root / name).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'Required local artifact is missing: {path}; no automatic downloads')
            # Read bytes once and verify SHA-256 on those exact bytes before deserializing (no TOCTOU).
            raw_bytes = path.read_bytes()
            expected_sha = CIFAR10_SHA256[name]
            if len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha):
                raise ValueError('Expected a lowercase SHA256 digest')
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f'Local artifact SHA256 mismatch for {path}: expected {expected_sha}, got {actual_sha}'
                )
            batch = pickle.loads(raw_bytes, encoding='latin1')
            data = batch['data']
            targets = np.asarray(batch['labels'], dtype=np.int64)
            if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] != 3072 or targets.shape != (len(data),):
                raise ValueError(f'Invalid CIFAR batch shape or dtype: {name}')
            arrays.append(data.reshape(-1, 3, 32, 32))
            labels.append(targets)
        self.images = torch.from_numpy(np.concatenate(arrays)).contiguous()
        self.labels = torch.from_numpy(np.concatenate(labels))
        self.cursor = 0

    def __len__(self):
        return len(self.images)

    def __call__(self, batch_size, *, generator, rank=0, world_size=1):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('CIFAR batch_size must be a positive integer')
        if world_size is None:
            world_size = 1
        if (
            type(world_size) is not int
            or world_size < 1
            or type(rank) is not int
            or not 0 <= rank < world_size
            or batch_size % world_size != 0
        ):
            raise ValueError(
                'Sharded data needs integer 0 <= rank < world_size and a batch_size divisible by world_size'
            )
        sharded = world_size > 1
        local_batch_size = (batch_size // world_size) if sharded else batch_size
        device = generator.device
        if self.images.device != device:
            self.images = self.images.to(device)
            self.labels = self.labels.to(device)
        if self.sampling == 'random':
            ids = torch.randint(len(self.images), (batch_size,), device=device, generator=generator)
        else:
            if self.cursor + batch_size > len(self.images):
                raise EvaluationDataExhausted(
                    f'CIFAR {self.split} sequential evaluation exhausted: {self.cursor} of {len(self.images)} '
                    f'images used, {batch_size} requested; it never wraps or repeats')
            ids = torch.arange(self.cursor, self.cursor + batch_size, device=device)
        self.cursor += batch_size
        if sharded:
            start = rank * local_batch_size
            local_ids = ids[start:start + local_batch_size]
        else:
            local_ids = ids
        x = self.images[local_ids].float() / 127.5 - 1
        if self.horizontal_flip:
            flip = torch.rand((batch_size, 1, 1, 1), device=device, generator=generator) < .5
            local_flip = flip[start:start + local_batch_size] if sharded else flip
            x = torch.where(local_flip, x.flip(-1), x)
        return {'real': x, 'labels': self.labels[local_ids]}

    def state_dict(self):
        return {'cursor': self.cursor}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {'cursor'}
                or type(state['cursor']) is not int or state['cursor'] < 0
                or (self.sampling == 'sequential' and state['cursor'] > len(self.images))):
            raise ValueError('Invalid CIFAR data recovery cursor')
        self.cursor = state['cursor']

    def resume_identity(self):
        return {'dataset': 'CIFAR-10 python batches', 'split': self.split,
                'files': [{'name': name, 'sha256': CIFAR10_SHA256[name]} for name in self.files],
                'sample_count': len(self.images), 'sampling': self.sampling,
                'horizontal_flip': self.horizontal_flip, 'normalization': 'uint8.float()/127.5-1',
                'layout': 'NCHW', 'order': 'published batch order', 'rng_device': 'caller generator'}

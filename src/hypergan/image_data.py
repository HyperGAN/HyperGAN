"""Pinned local CIFAR-10 bytes with explicit training/evaluation sampling."""
from pathlib import Path
import pickle

import numpy as np
import torch

from .image_components import _verified_file

CIFAR10_SHA256 = {
    'data_batch_1': '54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd',
    'data_batch_2': '766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8',
    'data_batch_3': '0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9',
    'data_batch_4': '3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b',
    'data_batch_5': 'd91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154',
    'test_batch': 'f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831',
}


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
            path = _verified_file(self.root / name, CIFAR10_SHA256[name])
            # Deserialize only the exact published dataset bytes verified above.
            with path.open('rb') as handle:
                batch = pickle.load(handle, encoding='latin1')
            data = batch['data']
            targets = np.asarray(batch['labels'], dtype=np.int64)
            if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] != 3072 or targets.shape != (len(data),):
                raise ValueError(f'Invalid CIFAR batch shape or dtype: {name}')
            arrays.append(data.reshape(-1, 3, 32, 32))
            labels.append(targets)
        self.images = torch.from_numpy(np.concatenate(arrays)).contiguous()
        self.labels = torch.from_numpy(np.concatenate(labels))
        self.cursor = 0

    def __call__(self, batch_size, *, generator):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('CIFAR batch_size must be a positive integer')
        device = generator.device
        if self.images.device != device:
            self.images = self.images.to(device)
            self.labels = self.labels.to(device)
        if self.sampling == 'random':
            ids = torch.randint(len(self.images), (batch_size,), device=device, generator=generator)
        else:
            if self.cursor + batch_size > len(self.images):
                raise StopIteration('CIFAR sequential evaluation exhausted; it never wraps or repeats')
            ids = torch.arange(self.cursor, self.cursor + batch_size, device=device)
        x = self.images[ids].float() / 127.5 - 1
        if self.horizontal_flip:
            flip = torch.rand((len(x), 1, 1, 1), device=device, generator=generator) < .5
            x = torch.where(flip, x.flip(-1), x)
        self.cursor += batch_size
        return {'real': x, 'labels': self.labels[ids]}

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

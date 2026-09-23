"""Research-only logo reader with CIFAR's sampling and augmentation protocol."""
from copy import deepcopy
import torch

from hypergan.colorization_data import ColorizationData
from hypergan.data import _digest, _positive_int


class Logos32Data:
    """Pinned logo bytes -> 32px white padding; otherwise CIFAR's draw protocol.

    Verify the original manifest unchanged, then explicitly override only output
    geometry. Draw random IDs and flip bits on the caller generator's device,
    like CIFAR10Data. Decoding and file verification happen on CPU. Sequential,
    non-wrapping draws support the recipe's existing evaluation declarations.
    """
    def __init__(self, root, manifest, manifest_sha256, split='train',
                 sampling='random', horizontal_flip=True):
        if sampling not in ('random', 'sequential') or type(horizontal_flip) is not bool:
            raise ValueError('Expected random/sequential sampling and boolean horizontal_flip')
        self.decoder = ColorizationData(root, manifest, manifest_sha256, split=split, shuffle=True)
        inventory = self.decoder.resume_identity()
        self.decoder.height = self.decoder.width = 32
        self.sampling, self.horizontal_flip, self.cursor = sampling, horizontal_flip, 0
        self._identity = {'kind': 'logos32_cifar_draw_protocol_v1',
                          'source_inventory': inventory,
                          'effective_preprocessing': {**inventory['preprocessing'], 'height': 32, 'width': 32},
                          'sampling': sampling, 'horizontal_flip': horizontal_flip,
                          'normalization': 'uint8.float()/127.5-1', 'layout': 'NCHW',
                          'rng_device': 'caller generator', 'order': 'manifest_path'}
        self._identity_sha256 = _digest(self._identity)

    def __call__(self, batch_size, *, generator):
        _positive_int(batch_size, 'batch_size')
        device = generator.device
        count = len(self.decoder.entries)
        if self.sampling == 'sequential' and self.cursor + batch_size > count:
            raise StopIteration('Logo sequential evaluation exhausted; it never wraps or repeats')
        old_rng = generator.get_state()
        try:
            if self.sampling == 'random':
                ids = torch.randint(count, (batch_size,), device=device, generator=generator)
            else:
                ids = torch.arange(self.cursor, self.cursor + batch_size, device=device)
            images = []
            for index in ids.cpu().tolist():
                entry = self.decoder.entries[index]
                image, _ = self.decoder._decode(self.decoder._read_bytes(entry), entry['path'])
                image = self.decoder._preprocess(image, entry['path'])
                value = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                images.append(value.reshape(32, 32, 3).permute(2, 0, 1))
            x = torch.stack(images).to(device).float() / 127.5 - 1
            if self.horizontal_flip:
                flip = torch.rand((len(x), 1, 1, 1), device=device, generator=generator) < .5
                x = torch.where(flip, x.flip(-1), x)
            self.cursor += batch_size
            return {'real': x}
        except BaseException:
            generator.set_state(old_rng)
            raise

    def resume_identity(self):
        return deepcopy(self._identity)

    def state_dict(self):
        return {'schema_version': 1, 'identity_sha256': self._identity_sha256, 'cursor': self.cursor}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {'schema_version', 'identity_sha256', 'cursor'}
                or type(state['schema_version']) is not int or state['schema_version'] != 1
                or state['identity_sha256'] != self._identity_sha256
                or type(state['cursor']) is not int or state['cursor'] < 0
                or (self.sampling == 'sequential' and state['cursor'] > len(self.decoder.entries))):
            raise ValueError('Logo data recovery identity/cursor mismatch')
        self.cursor = state['cursor']

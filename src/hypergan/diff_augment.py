"""Differentiable image augmentation used by TransGAN section 3.4.

The policy and sampling distributions follow the official DiffAugment reference:
https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py
The official TransGAN basic policy uses translation,cutout,color with translation
ratio 0.2 and batch cutout probability 0.3:
https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/diff_aug.py
All randomness uses the input device's PyTorch RNG so training RNG replay works.
"""

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_POLICY = 'color,translation,cutout'


def _parse_policy(policy: str) -> tuple[str, ...]:
    if not isinstance(policy, str):
        raise TypeError('DiffAugment policy must be a comma-separated string')
    names = tuple(name.strip() for name in policy.split(',') if name.strip())
    unknown = set(names) - {'color', 'translation', 'cutout'}
    if unknown:
        raise ValueError(f'Unknown DiffAugment policy: {", ".join(sorted(unknown))}')
    return names


def _unit_interval(value: float, name: str) -> float:
    if not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'DiffAugment {name} must be a finite number in [0, 1]')
    return float(value)


def _color(x: torch.Tensor) -> torch.Tensor:
    shape = (x.shape[0], 1, 1, 1)
    brightness = torch.rand(shape, device=x.device, dtype=x.dtype) - 0.5
    x = x + brightness
    channel_mean = x.mean(dim=1, keepdim=True)
    saturation = 2 * torch.rand(shape, device=x.device, dtype=x.dtype)
    x = channel_mean + saturation * (x - channel_mean)
    image_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 0.5 + torch.rand(shape, device=x.device, dtype=x.dtype)
    return image_mean + contrast * (x - image_mean)


def _translation(x: torch.Tensor, ratio: float) -> torch.Tensor:
    batch, channels, height, width = x.shape
    max_y, max_x = int(height * ratio + 0.5), int(width * ratio + 0.5)
    dy = torch.randint(-max_y, max_y + 1, (batch, 1, 1), device=x.device)
    dx = torch.randint(-max_x, max_x + 1, (batch, 1, 1), device=x.device)
    rows = (torch.arange(height, device=x.device)[None, :, None] + dy + 1).clamp(0, height + 1)
    cols = (torch.arange(width, device=x.device)[None, None, :] + dx + 1).clamp(0, width + 1)
    indices = (rows * (width + 2) + cols).reshape(batch, 1, height * width)
    padded = F.pad(x, (1, 1, 1, 1)).flatten(2)
    return padded.gather(2, indices.expand(batch, channels, height * width)).reshape_as(x)


def _cutout(x: torch.Tensor, probability: float) -> torch.Tensor:
    if probability == 0:
        return x
    # One decision for the whole batch, as in TransGAN. Keep it on-device rather
    # than using a Python RNG or synchronizing CUDA to branch on its value.
    gate = torch.rand((), device=x.device) < probability if probability < 1 else None
    batch, _, height, width = x.shape
    cut_height, cut_width = int(height * 0.5 + 0.5), int(width * 0.5 + 0.5)
    center_y = torch.randint(0, height + (1 - cut_height % 2), (batch, 1, 1), device=x.device)
    center_x = torch.randint(0, width + (1 - cut_width % 2), (batch, 1, 1), device=x.device)
    top, left = center_y - cut_height // 2, center_x - cut_width // 2
    rows = torch.arange(height, device=x.device)[None, :, None]
    cols = torch.arange(width, device=x.device)[None, None, :]
    removed = (rows >= top) & (rows < top + cut_height) & (cols >= left) & (cols < left + cut_width)
    if gate is not None:
        removed = removed & gate
    return x * (~removed).unsqueeze(1).to(dtype=x.dtype)


def _apply(x: torch.Tensor, policy: tuple[str, ...], translation_ratio: float,
           cutout_probability: float) -> torch.Tensor:
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError('DiffAugment expects a floating-point NCHW tensor')
    if x.shape[2] < 1 or x.shape[3] < 1:
        raise ValueError('DiffAugment expects nonempty spatial dimensions')
    original = x
    for name in policy:
        if name == 'color':
            x = _color(x)
        elif name == 'translation':
            x = _translation(x, translation_ratio)
        else:
            x = _cutout(x, cutout_probability)
    return x if x is original else x.contiguous()


def diff_augment(x: torch.Tensor, policy: str = DEFAULT_POLICY, *,
                 translation_ratio: float = 0.125,
                 cutout_probability: float = 1.0) -> torch.Tensor:
    """Apply independent per-image augmentations to floating-point NCHW images.

    Color changes are not clamped. Translation pads with zero, and cutout zeros
    the selected region. Cutout activation is sampled once per batch, with
    independent per-image locations. Input tensors are never mutated or detached.
    """
    names = _parse_policy(policy)
    translation_ratio = _unit_interval(translation_ratio, 'translation_ratio')
    cutout_probability = _unit_interval(cutout_probability, 'cutout_probability')
    return _apply(x, names, translation_ratio, cutout_probability) if names else x


class DiffAugment(nn.Module):
    """Apply DiffAugment during training; evaluation is identity with no RNG use."""

    def __init__(self, policy: str = DEFAULT_POLICY, *, translation_ratio: float = 0.125,
                 cutout_probability: float = 1.0):
        super().__init__()
        self._policy = _parse_policy(policy)
        self.policy = ','.join(self._policy)
        self.translation_ratio = _unit_interval(translation_ratio, 'translation_ratio')
        self.cutout_probability = _unit_interval(cutout_probability, 'cutout_probability')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._policy:
            return x
        return _apply(x, self._policy, self.translation_ratio, self.cutout_probability)

    def extra_repr(self) -> str:
        return (f'policy={self.policy!r}, translation_ratio={self.translation_ratio}, '
                f'cutout_probability={self.cutout_probability}')

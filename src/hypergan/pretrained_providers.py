"""Trusted external checkpoint architectures exposed through native HNDL.

Providers construct published library models without downloading weights.
HNDL verifies and loads the configured local artifact and selects its output.
"""


def _torchvision_resnet18():
    # Keep torchvision optional for every network that does not request it.
    try:
        from torchvision.models import resnet18
        from torch import nn
    except ImportError as exc:
        raise ImportError('The torchvision_resnet18 provider requires hypergan[cifar]') from exc
    model = resnet18(weights=None)
    # Intermediate features participate in gradient penalties. Avoid mutation
    # of tensors retained by autograd or selected by the pretrained operator.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model


def register_providers(registry, providers=None):
    """Register host-approved builders on this registry, without global state."""
    from .network_config import validate_pretrained_providers
    options = {} if providers is None else providers
    validate_pretrained_providers(options)
    if 'torchvision_resnet18' not in registry.pretrained_providers:
        registry.pretrained_provider('torchvision_resnet18', _torchvision_resnet18)
    for name, config in options.items():
        registry.pretrained_provider(name, _dinov3_builder(config['source_path'], config['source_commit']),
                                     readouts={'patch_tokens': _dinov3_patch_tokens,
                                               'multidepth': _dinov3_multidepth})
    return registry


def _dinov3_builder(source_path, source_commit):
    """Verify the trusted local source before exposing its upstream constructor."""
    import importlib
    from pathlib import Path
    import subprocess
    import sys

    source = Path(source_path).expanduser().resolve()
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('DINOv3 source_commit must be a full lowercase Git SHA')
    if not (source / 'dinov3' / 'hub' / 'backbones.py').is_file():
        raise FileNotFoundError(f'DINOv3 source checkout missing: {source}')
    try:
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'Cannot verify local DINOv3 Git checkout: {source}') from exc
    if actual != source_commit or dirty:
        raise ValueError(f'DINOv3 source must be clean at {source_commit}; got {actual}, dirty={bool(dirty)}')
    sys.path.insert(0, str(source))
    try:
        package = importlib.import_module('dinov3')
        if Path(package.__file__).resolve().parent != source / 'dinov3':
            raise ValueError('A different DINOv3 package is imported; use the configured pinned checkout')
        backbones = importlib.import_module('dinov3.hub.backbones')
    finally:
        sys.path.remove(str(source))
    from functools import partial
    return partial(backbones.dinov3_vits16, pretrained=False)


def _dinov3_patch_tokens(model, x):
    """Native provider readout; all preprocessing and reshaping lives in HNDL."""
    import torch
    height, width = _dinov3_grid(x)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        tokens = model.forward_features(x)['x_norm_patchtokens']
    if tokens.shape != (len(x), height * width, 384):
        raise ValueError(f'DINOv3 ViT-S/16 must return {height * width} patch tokens of width 384')
    return tokens


def _dinov3_multidepth(model, x):
    """Expose four depths from one backbone call as one native readout tensor."""
    import torch
    height, width = _dinov3_grid(x)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        maps = model.get_intermediate_layers(x, n=(2, 5, 8, 11), reshape=True, norm=True)
    if len(maps) != 4 or any(feature.shape != (len(x), 384, height, width) for feature in maps):
        raise ValueError('DINOv3 ViT-S/16 must return four [batch,384,height/16,width/16] intermediate maps')
    return torch.cat(maps, dim=1)


def _dinov3_grid(x):
    if x.ndim != 4 or x.shape[1] != 3 or any(size < 16 or size % 16 for size in x.shape[-2:]):
        raise ValueError('DINOv3 expects RGB images with positive dimensions divisible by 16')
    return x.shape[-2] // 16, x.shape[-1] // 16


def dinov3_registry(source_path, source_commit):
    """Bind a verified external implementation to HNDL's native local provider."""
    from hndl import Registry
    registry = register_providers(Registry.builtins())
    registry.pretrained_provider('dinov3_vits16', _dinov3_builder(source_path, source_commit),
                                 readouts={'patch_tokens': _dinov3_patch_tokens,
                                           'multidepth': _dinov3_multidepth})
    return registry

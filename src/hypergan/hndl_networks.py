"""HNDL construction and binding adapters; architectures live in configuration."""
import torch
from torch import nn

from .network_config import packaged_source, render_source


def _resolve_runtime_parameters(parameters):
    """Resolve HTTP(S) weight URLs in HNDL template parameters into the local verified cache."""
    if not parameters:
        return parameters
    from .cache import OfflineError, _is_url, _url_to_filename, _validate_local_file, get_cache_dir, resolve_or_cache
    resolved = dict(parameters)
    weights_path = resolved.get('weights_path')
    if isinstance(weights_path, str) and _is_url(weights_path):
        sha = resolved.get('weights_sha256')
        expected_sha = sha.strip() if isinstance(sha, str) and sha.strip() else None
        if torch.empty(0).device.type == 'meta':
            cached_path = get_cache_dir() / _url_to_filename(weights_path.strip())
            if not cached_path.is_file():
                raise OfflineError(f'Artifact not cached for meta-device inspection: {weights_path}')
            resolved['weights_path'] = str(_validate_local_file(cached_path, expected_sha256=expected_sha))
        else:
            resolved['weights_path'] = str(resolve_or_cache(weights_path, expected_sha256=expected_sha))
    return resolved


def build_network(source=None, *, file=None, input_shape, output_shape,
                  parameters=None, registry=None, input_dtype=None, pretrained_providers=None):
    """Construct an unmodified native HNDL tensor/named-port network."""
    from hndl import Registry
    from hndl.torch import network
    from .hndl_augmentation import register_augmentation
    from .pretrained_providers import register_providers
    runtime_parameters = _resolve_runtime_parameters(parameters)
    source = render_source(source if source is not None else packaged_source(file), runtime_parameters)
    registry = register_augmentation(Registry.builtins() if registry is None else registry)
    registry = register_providers(registry, pretrained_providers)
    return network(source, input_shape=input_shape, output_shape=output_shape,
                   device='cpu', registry=registry, input_dtype=input_dtype)


class HNDLNetwork(nn.Module):
    """Bind component inputs to native HNDL tensor or named ports."""
    def __init__(self, source, input_shape, output_shape, parameters=None,
                 input_dtype=None, pretrained_providers=None):
        super().__init__()
        self.network = build_network(source, input_shape=input_shape,
                                     output_shape=output_shape, parameters=parameters,
                                     input_dtype=input_dtype, pretrained_providers=pretrained_providers)
        self.named_inputs = isinstance(input_shape, dict)

    def forward(self, *args, **inputs):
        if self.named_inputs:
            return self.network(*args, **inputs)
        if args:
            if len(args) != 1 or inputs:
                raise ValueError('HNDL component expects one positional tensor or named bindings')
            return self.network(args[0])
        if len(inputs) == 1:
            value = next(iter(inputs.values()))
        else:
            raise ValueError('Multiple HNDL inputs require named input_shape contracts')
        return self.network(value)

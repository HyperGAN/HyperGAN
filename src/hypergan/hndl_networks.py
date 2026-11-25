"""HNDL construction and binding adapters; architectures live in configuration."""
import inspect
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


class _AutocastAwareCheck:
    """Allow PyTorch autocast floating-point dtype promotions during HNDL validation.

    ``hndl 0.13.0`` validates every node port against ``self._runtime_dtype``
    (FP32 master weights) on the first call for an autocast signature, which
    rejects BF16/FP16 activations legally produced under ``torch.autocast``.
    When autocast is disabled, behavior is delegated unchanged; when autocast is
    enabled, shape, device, non-floating port dtypes, and registered-state
    integrity remain strictly enforced while floating-point ports may carry the
    active autocast dtype.
    """

    def __init__(self, module):
        self._module = module

    def __call__(self, value, expected, dtype, port, batch, *args, **kwargs):
        if (
            dtype is not None
            and isinstance(value, torch.Tensor)
            and value.dtype != dtype
            and dtype.is_floating_point
            and value.is_floating_point()
        ):
            device_type = value.device.type
            if (
                torch.is_autocast_enabled(device_type)
                and value.dtype == torch.get_autocast_dtype(device_type)
            ):
                dtype = value.dtype
        return type(self._module)._check(self._module, value, expected, dtype, port, batch, *args, **kwargs)


def _install_autocast_check(module):
    orig = getattr(type(module), '_check', None)
    if not callable(orig):
        return module
    try:
        params = list(inspect.signature(orig).parameters.values())
    except (TypeError, ValueError):
        return module
    # Only wrap when HNDL exposes the expected (self, value, expected, dtype, port, batch, ...)
    # contract; future HNDL versions with native autocast validation or refactored internals
    # degrade gracefully without error.
    positional = [
        p.name for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if positional[:6] != ['self', 'value', 'expected', 'dtype', 'port', 'batch']:
        return module
    module._check = _AutocastAwareCheck(module)
    return module


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
    return _install_autocast_check(
        network(source, input_shape=input_shape, output_shape=output_shape,
                device='cpu', registry=registry, input_dtype=input_dtype)
    )


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

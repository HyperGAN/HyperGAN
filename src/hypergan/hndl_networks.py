"""HNDL construction and binding adapters; architectures live in configuration."""
from torch import nn

from .network_config import packaged_source, render_source


def build_network(source=None, *, file=None, input_shape, output_shape,
                  parameters=None, registry=None, input_dtype=None, pretrained_providers=None):
    """Construct an unmodified native HNDL tensor/named-port network."""
    from hndl import Registry
    from hndl.torch import network
    from .pretrained_providers import register_providers
    source = render_source(source if source is not None else packaged_source(file), parameters)
    registry = register_providers(Registry.builtins() if registry is None else registry,
                                  pretrained_providers)
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

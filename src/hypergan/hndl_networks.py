"""HNDL construction and binding adapters; architectures live in configuration."""
import torch
from torch import nn

from .network_config import packaged_source, render_source


def build_network(source=None, *, file=None, input_shape, output_shape,
                  parameters=None, registry=None):
    """Construct an unmodified native HNDL tensor/named-port network."""
    from hndl.torch import network
    source = render_source(source if source is not None else packaged_source(file), parameters)
    return network(source, input_shape=input_shape, output_shape=output_shape,
                   device='cpu', registry=registry)


class HNDLNetwork(nn.Module):
    """Bind one tensor or explicitly concatenate a list of component inputs."""
    def __init__(self, source, input_shape, output_shape, parameters=None,
                 concat_inputs=None, concat_dim=-1):
        super().__init__()
        self.network = build_network(source, input_shape=input_shape,
                                     output_shape=output_shape, parameters=parameters)
        self.named_inputs = isinstance(input_shape, dict)
        self.concat_inputs = tuple(concat_inputs) if concat_inputs else None
        self.concat_dim = concat_dim

    def forward(self, *args, **inputs):
        if self.named_inputs:
            return self.network(*args, **inputs)
        if args:
            if len(args) != 1 or inputs:
                raise ValueError('HNDL component expects one positional tensor or named bindings')
            return self.network(args[0])
        if self.concat_inputs:
            if set(inputs) != set(self.concat_inputs):
                raise ValueError('HNDL concat_inputs must match component bindings')
            value = torch.cat([inputs[key] for key in self.concat_inputs], dim=self.concat_dim)
        elif len(inputs) == 1:
            value = next(iter(inputs.values()))
        else:
            raise ValueError('Multiple HNDL inputs require explicit concat_inputs')
        return self.network(value)

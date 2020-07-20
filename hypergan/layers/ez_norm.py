import torch.nn as nn
import hypergan as hg
from hypergan.layer_size import LayerSize

from hypergan.modules.modulated_conv2d import EqualLinear

class EzNorm(hg.Layer):
    def __init__(self, component, args, options):
        super(EzNorm, self).__init__(component, args, options)
        self.dim = options.dim or 1
        self.size = LayerSize(*component.current_size.dims)

        style_size = component.layer_output_sizes['w'].size()
        channels = component.current_size.channels
        dims = len(component.current_size.dims)
        equal_linear = options.equal_linear

        if equal_linear:
            self.beta = EqualLinear(style_size, channels, lr_mul=0.01)
        else:
            self.beta = nn.Linear(style_size, channels)
        if dims == 2:
            self.conv = nn.Conv1d(channels, 1, 1, 1, padding = 0)
        else:
            self.conv = nn.Conv2d(channels, 1, 1, 1, padding = 0)

        component.nn_init(self.beta, options.initializer)
        component.nn_init(self.conv, options.initializer)

    def forward(self, input, context):
        style = context['w']
        N = input.shape[0]
        D = input.shape[self.dim]
        view = [1 for x in input.shape]
        view[0] = N
        view[self.dim] = D

        return self.beta(style).view(*view) * self.conv(input)

    def output_size(self):
        return self.size


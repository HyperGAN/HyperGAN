import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

class EzNorm(hg.Layer):
    """
        ---
        description: 'layer ez_norm for configurable component'
        ---

        # ez_norm layer

        `ez_norm` is a custom normalization technique that uses a conv of the input by a linear projection of a style vector.

        ## Optional arguments

            `style` - The name of the style vector to use. Defaults to "w"

        ## input size

        Any 4-d tensor

        ## output size

        Same as input size

        ## syntax

        ```json
          "ez_norm style=[style vector name]"
        ```

        ## examples

        ```json
          "latent name=w",
          ...
          "cat self (ez_norm style=w)"
        ```
    """
    def __init__(self, component, args, options):
        super(EzNorm, self).__init__(component, args, options)
        self.dim = options.dim or 1
        self.size = LayerShape(*component.current_size.dims)

        style_size = component.layer_output_sizes[options.style or 'w'].size()
        #conv_size = component.layer_output_sizes[options.conv_psm or 'r'].size()
        channels = component.current_size.channels
        dims = len(component.current_size.dims)

        self.beta = nn.Linear(style_size, channels, bias=False)

        #if dims == 2:
        #    self.conv = nn.Conv1d(channels, 1, 1, 1, padding = 0)
        #else:
        #    self.conv = nn.Conv2d(channels, 1, 1, 1, padding = 0)

        component.nn_init(self.beta, options.initializer)
        #component.nn_init(self.conv, options.initializer)
        self.activation = nn.SELU()

    def forward(self, input, context):
        style = context[self.options.style or 'w']
        N = input.shape[0]
        D = input.shape[self.dim]
        view = [1 for x in input.shape]
        view[0] = N
        view[self.dim] = D

        if self.options.conv is None:
            return self.activation(self.beta(style).view(*view)) * input + input
            return self.beta(style).view(*view) + input
        else:
            conv = context[self.options.conv or 'r']
            return self.activation(self.beta(style).view(*view)) * input +self.activation(conv)*input

    def output_size(self):
        return self.size


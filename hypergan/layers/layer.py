import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

class Layer(hg.Layer):
    """
        ---
        description: 'layer layer for configurable component'
        ---

        # layer layer

        `layer` allows you to reference any layer defined in the rest of the network.

        ## arguments

            `layer_name` - The name of the layer to use

        ## Optional arguments

            `upsample` - If true, upsample the layer to the current size

        ## input size

        Any 4-d tensor

        ## output size

        if upsample true, the current input size
        otherwise the layer size

        ## syntax

        ```json
          "layer z"
        ```

        ## examples

        ```json

          "identity name=encoding",
          ...
          "add self (layer encoding upsample=true)"
        ```
    """
    def __init__(self, component, args, options):
        super(Layer, self).__init__(component, args, options)

        self.name = args[0]
        self.size = component.layer_output_sizes[args[0]]
        if options.upsample:
            self.size = LayerShape(self.size.channels, *component.current_size.dims[1:])
            self.upsample = nn.Upsample(self.size.dims[1:], mode="bilinear")
        if self.name in component.named_is_latent:
            component.is_latent = component.named_is_latent[self.name]
        else:
            component.is_latent = False

    def forward(self, input, context):
        if hasattr(self, 'upsample'):
            return self.upsample(context[self.name])
        return context[self.name]

    def output_size(self):
        return self.size



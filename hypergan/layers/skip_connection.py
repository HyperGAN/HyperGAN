import torch
import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

class SkipConnection(hg.Layer):
    """
        ---
        description: 'layer skip_connection for adding different layers together with varying sizes'
        ---

        ## syntax

        ```json
          "skip_connection other_layer"
        ```

    """
    def __init__(self, component, args, options):
        super(SkipConnection, self).__init__(component, args, options)
        dims = list(component.current_size.dims)
        self.name = args[0]
        ch_in = component.layer_output_sizes[self.name].channels
        ch_out = dims[0]
        self.size = LayerShape(*dims)
        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4),
                                nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False), nn.ReLU(),
                                nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.ReLU() )

    def forward(self, input, context):
        feat_small = context[self.name]
        return input * self.main(feat_small)

    def output_size(self):
        return self.size



class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


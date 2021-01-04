import torch.nn as nn
import hypergan as hg
import torch
from hypergan.layer_shape import LayerShape

class Noise(hg.Layer):
    """
        ---
        description: 'layer noise for configurable component'
        ---

        # noise layer

        Same as input size

        ## syntax

        ```json
          "noise"
        ```
    """
    def __init__(self, component, args, options):
        super(Noise, self).__init__(component, args, options)

    def forward(self, input, context):
        return torch.randn_like(input, device=input.device)


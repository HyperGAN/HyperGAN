import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

from hypergan.modules.modulated_conv2d import EqualLinear

class Upsample(hg.Layer):

    """
        ---
        description: 'layer upsample for configurable component'
        ---

        # upsample layer

        `upsample` resizes the input tensor to the specified size.

        ## Optional arguments

            * `h` - requested height. defaults to input height * 2
            * `w` - requested width. defaults to input width * 2

        ## input size

        Any 4-d tensor

        ## output size

        [B, input channels, h, w]

        ## syntax

        ```json
          "upsample"
        ```

        ## examples

        ```json
          "upsample w=96 h=96",
          "conv 4",
          "hardtanh"
        ```
    """
    def __init__(self, component, args, options):
        super(Upsample, self).__init__(component, args, options)
        w = options.w or component.current_size.width * 2
        h = options.h or component.current_size.height * 2
        self.layer = nn.Upsample((h, w), mode="bilinear")
        self.size = LayerShape(component.current_size.channels, h, w)

    def forward(self, input, context):
        return self.layer(input)

    def output_size(self):
        return self.size

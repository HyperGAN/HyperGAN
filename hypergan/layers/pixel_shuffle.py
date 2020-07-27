import torch.nn as nn
from hypergan.layer_size import LayerSize
import hypergan as hg

class PixelShuffle(hg.Layer):
    """
        ---
        description: 'layer pixel_shuffle for configurable component'
        ---

        # pixel_shuffle layer

        Implements PixelShuffle https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html

        ## input size

        Any 4-d tensor of the form `[B, C, H, W]`

        ## output size

        A 4d-tensor of the form `[B, C//4, H*2, W*2]`

        ## syntax

        ```json
          "pixel_shuffle"
        ```
    """

    def __init__(self, component, args, options):
        super(PixelShuffle, self).__init__(component, args, options)
        self.shuffle = nn.PixelShuffle(2)
        self.dims = list(component.current_size.dims).copy()

    def output_size(self):
        return LayerSize(self.dims[0]//4, self.dims[1]*2, self.dims[2]*2)

    def forward(self, input, context):
        return self.shuffle(input)

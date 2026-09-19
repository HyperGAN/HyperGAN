import torch.nn as nn
import torch
from hypergan.layer_shape import LayerShape
import hypergan as hg

class SegmentSoftmax(hg.Layer):
    """
        ---
        description: 'layer segment_softmax for configurable component'
        ---

        # segment_softmax layer

        `segment_softmax` is a custom layer that allows for masking multiple output channels.

        Suppose you have 30 channels and `segment_softmax 3`. First, the 30 channels split into 15/15.
        The first 15 will be used for softmax and multiplied against the second.
        Then each channel is softmaxed, multiplied, and summed.

        So 30 input channels with 3 output channels equate to 5 input channels for each output channel.

        ## input size

        Any 4-d tensor of the shape `[B, C, H, W]`

        ## output size

        [B, OUTPUT_CHANNELS, H, W]

        ## syntax

        ```json
          "segment_softmax OUTPUT_CHANNELS"
        ```

        ## examples

        At the end of the generator for RGB images:

        ```json
          "conv 30",
          "segment_softmax 3",
          "hardtanh"
        ```
    """


    def __init__(self, component, args, options):
        super(SegmentSoftmax, self).__init__(component, args, options)
        self.channels = args[0]
        self.dims = list(component.current_size.dims).copy()
        self.softmax = nn.Softmax(dim=2)

    def output_size(self):
        return LayerShape(*([self.channels]+self.dims[1:]))

    def forward(self, input, context):
        content, segment = torch.split(input, input.shape[1]//2, 1)
        content_shape = list(content.shape)
        new_shape = [content_shape[0], content_shape[1]//self.channels, self.channels, *content.shape[2:]]
        net_in = content.view(*new_shape)
        segment = segment.view(*new_shape)
        selection = self.softmax(segment)
        rendered = (selection * net_in).sum(dim=1).view([input.shape[0]]+list(self.output_size().dims))
        return rendered

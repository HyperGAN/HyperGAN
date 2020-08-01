import torch.nn as nn
import hyperchamber as hc
import hypergan as hg
from hypergan.layer_shape import LayerShape

class ResizableStack(hg.Layer):
    """
        ---
        description: 'layer resizable_stack for configurable component'
        ---

        # resizable_stack layer

        `resizable_stack` allows for variable size outputs on the generator. A conv stack is repeated until the output size is reached.

        If you specify "segment_softmax" this repeats the pattern:
          upsample
          normalize(expects style vector named 'w')
          conv ...
          activation(before last layer)

        and ends in:
          segment_softmax output_channels

        ## arguments
            * layer type. Defaults to "segment_softmax"
        ## optional arguments
            * segment_channels - The number of channels before segment_softmax. Defaults to 5
            * max_channels - The most channels for any conv. Default 256
            * style - the style vector to use. Default "w"
        ## input size

        Any 4-d tensor

        ## output size

        [B, output channels, output height, output width]

        ## syntax

        ```json
          "resizable_stack segment_softmax"
        ```

        ## examples

        ```json
          "identity name=w",
          "linear 8*8*256",
          "relu",
          "resizable_stack segment_softmax",
          "hardtanh"
        ```
    """
    def __init__(self, component, args, options):
        super(ResizableStack, self).__init__(component, args, options)
        self.size = LayerShape(component.gan.channels(), component.gan.height(), component.gan.width())
        self.max_channels = options.max_channels or 256
        self.segment_channels = options.segment_channels or 5
        self.style = options.style or "w"

        layers = []

        sizes = self.sizes(component.current_size.height, component.current_size.width, component.gan.height(), component.gan.width(), self.segment_channels * 2 * component.gan.channels())
        print("SIZES", sizes)
        for i, size in enumerate(sizes[1:]):
            c = min(size.channels, self.max_channels)
            upsample = hg.layers.Upsample(component, [], hc.Config({"w": size.width, "h": size.height}))
            component.current_size = upsample.output_size() #TODO abstract input_size
            _, add = component.parse_layer("add self (ez_norm initializer=(xavier_normal) style=" + self.style + ")")
            _, conv = component.parse_layer("conv2d " + str(size.channels) + " padding=0 initializer=(xavier_normal)")
            layers += [upsample, add, conv]
            if i < len(sizes) - 2:
                layers += [nn.ReLU()]

        layers += [hg.layers.SegmentSoftmax(component, [component.gan.channels()], {})]
        self.layers = nn.ModuleList(layers)

    def sizes(self, initial_height, initial_width, target_height, target_width, final_channels):
        channels = []
        hs = []
        ws = []
        sizes = []

        w = initial_width
        h = initial_height
        i = 0

        channels.append(final_channels)
        while w < target_width or h < target_height:
            if i > 0:
                h-=2 #padding
                w-=2 #padding
            h*=2 #upscale
            w*=2 #upscale
            channels.append(final_channels * 2**i)
            hs.append(min(h, target_height+2))
            ws.append(min(w, target_width+2))
            i+=1

        w = initial_width
        h = initial_height

        channels.reverse()
        for c,h,w in zip(channels, hs, ws):
            sizes.append(LayerShape(c, h, w))
        return sizes

    def forward(self, input, context):
        for module in self.layers:
            if isinstance(module, hg.Layer):
                input = module(input, context)
            else:
                input = module(input)
        return input

    def output_size(self):
        return self.size


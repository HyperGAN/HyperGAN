import torch.nn as nn
import hyperchamber as hc
import hypergan as hg
from hypergan.layer_shape import LayerShape
from hypergan.modules.adaptive_instance_norm import AdaptiveInstanceNorm

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
            * normalize - type of layer normalization to use
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
        self.options = options
        self.size = LayerShape(component.gan.channels(), component.gan.height(), component.gan.width())
        self.max_channels = options.max_channels or 512
        self.segment_channels = options.segment_channels or 5
        self.style = options.style or "w"
        self.output_channels = options.output_channels or component.gan.channels()
        segment_softmax = (options.segment_softmax is not False)
        skip_connection = (options.skip_connection is not False)
        mode = (options.mode or "deconv")
        attention_layer_index = options.attention

        layers = []

        if segment_softmax:
            sizes = self.sizes(component.current_size.height, component.current_size.width, component.gan.height(), component.gan.width(), self.segment_channels * 2 * self.output_channels, mode)
        else:
            sizes = self.sizes(component.current_size.height, component.current_size.width, component.gan.height(), component.gan.width(), self.output_channels, mode)
        layer_names = []
        for i, size in enumerate(sizes):
            c = min(size.channels, self.max_channels)
            name = "_g_conv"+str(i)
            if mode == "deconv":
                _, conv = component.parse_layer("deconv " + str(c) + "padding=0 filter=3 name="+name)
                component.current_size.channels = c

                layers += [conv]#, noise]
                layer_names += [name]
            elif mode == "resize_conv":
                upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
                component.current_size = upsample.output_size() #TODO abstract input_size
                _, conv = component.parse_layer("conv " + str(c) + "padding=0 filter=3 name="+name)
                component.current_size.channels = c
                layers += [upsample, conv]#, noise]
                layer_names += [None, name]
            elif mode == "residual":
                layer = "residual " + str(self.options.repeat or 1) + " block=up output_channels="+str(c)+" style="+self.style+" name="+name
                residual = component.parse_layer(layer)[1]
                layers += [ residual ]
                layer_names += [ name ]
            elif mode == "deconv_residual":
                layers += [
                    component.parse_layer("deconv " + str(c) + " padding=0 filter=3 name="+name)[1]
                ]
                norm_layers = self.norm_layers(component)
                layers += norm_layers
                layer_names += [None for x in norm_layers]
                layers += [nn.SELU()]
                layer = "residual " + str(self.options.repeat or 1) + " output_channels="+str(c)+" style="+self.style+" name="+name
                residual = component.parse_layer(layer)[1]
                layers += [ residual ]
                layer_names += [ None, name, None]

            if i == attention_layer_index:
                attention = component.parse_layer("add self (attention)")[1]
                layers += [ attention ]
                layer_names += [ None ]

            norm_layers = self.norm_layers(component)
            layers += norm_layers
            layer_names += [None for x in norm_layers]

            if i < (len(sizes) - 1):
                layers += [nn.SELU()]
                layer_names += [None]

            if (component.current_size.width > size.width or component.current_size.height > size.height):
                print("Unexpected size, slicing", component.current_size)
                _, slice = component.parse_layer("slice w="+str(size.width)+" h="+str(size.height))
                layers += [slice]
                layer_names += [None]

            if(i > 1 and skip_connection):
                #pass
                _, shortcut = component.parse_layer("skip_connection _g_conv"+str(i-2))
                layers += [shortcut]
                layer_names += [None]

        if segment_softmax:
            layers += [hg.layers.SegmentSoftmax(component, [self.output_channels], {})]
            layer_names += [None]
        self.layers = nn.ModuleList(layers)
        print("LAYERS", self.layers)
        self.layer_names = layer_names

    def sizes(self, initial_height, initial_width, target_height, target_width, final_channels, mode):
        channels = []
        hs = []
        ws = []
        sizes = []

        w = initial_width
        h = initial_height
        i = 0

        channels.append(final_channels)
        channel_exp = 6
        while w < target_width or h < target_height:
            if i > 0:
                if mode == "resize_conv":
                    h-=2
                    w-=2
                elif mode[:6] == "deconv":
                    w-=1
                    h-=1

            h*=2 #upscale
            w*=2 #upscale

            if mode[:6] == "deconv":
                w+=3
                h+=3

            channels.append(min(self.max_channels, 2**channel_exp))
            channel_exp += 1
            if mode == "resize_conv":
                hs.append(min(h, target_height+2))
                ws.append(min(w, target_width+2))
            else:
                hs.append(min(h, target_height))
                ws.append(min(w, target_width))
            i+=1

        channels.reverse()
        channels = channels[1:]
        print("___>>", channels)
        print("___>>", ws)
        print("___>>", hs)
        w = initial_width
        h = initial_height

        for c,h,w in zip(channels, hs, ws):
            sizes.append(LayerShape(c, h, w))
        print("SIZES", sizes)
        return sizes

    def forward(self, input, context):
        return self.forward_module_list(input, self.layer_names, self.layers, context)

    def output_size(self):
        return self.size


import torch.nn as nn
import torch
import hyperchamber as hc
import hypergan as hg
from hypergan.layer_shape import LayerShape
from torch.nn.parameter import Parameter
from hypergan.modules.no_op import NoOp

class Residual(hg.Layer):
    """
        ---
        description: 'layer residual for configurable component'
        ---

        # residual layer

        `residual` adds one or more residual blocks https://paperswithcode.com/method/residual-block

        ## optional arguments

        The number of residual blocks to add

        block - the type of residual layer block. up, down, or default

        ## input size

        Any 4-d tensor

        ## output size

        Same as input size

        ## syntax

        ```json
          "residual OUTPUT_CHANNELS block=BLOCK_TYPE"
        ```

        ## examples

        ```json
          "residual 3"
        ```
    """


    def __init__(self, component, args, options):
        super(Residual, self).__init__(component, args, options)
        self.options = options
        self.style = options.style
        self.block = options.block or "default"
        layers = []
        shortcut = []
        self.scalar = Parameter(torch.zeros([]), requires_grad=True)
        output_channels = args[0]
        dims = list(component.current_size.dims)
        dims[0] = output_channels
        if self.block[0:4] == "down":
            dims[1] = dims[1] // 2
            dims[2] = dims[2] // 2
        if self.block == "default":
            current_size = component.current_size
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("conv " + str(component.current_size.channels) + " filter=3")[1]]
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            component.current_size = current_size
            shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0")[1]]
        if self.block == "down":
            current_size = component.current_size
            layers += [nn.ReLU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3 initializer=(orthogonal)")[1]]
            layers += [nn.ReLU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3 initializer=(orthogonal)")[1]]
            layers += [component.parse_layer("adaptive_avg_pool")[1]]
            component.current_size = current_size
            shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0 initializer=(orthogonal)")[1]]
            shortcut += [component.parse_layer("adaptive_avg_pool")[1]]
        if self.block == "down2":
            current_size = component.current_size
            layers += [nn.ReLU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            layers += [nn.ReLU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            layers += [component.parse_layer("avg_pool")[1]]
            component.current_size = current_size
            #shortcut += [component.parse_layer("avg_pool")[1]]
            #shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0")[1]]
            self.conv_id = component.parse_layer("conv " + str(output_channels - current_size.channels) + " filter=1 padding=0")[1]
            self.avg_pool = component.parse_layer("avg_pool")[1]
            component.current_size = LayerShape(output_channels, component.current_size.height, component.current_size.width)

        if self.block == "down3":
            current_size = component.current_size
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            layers += [nn.ReLU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            layers += [component.parse_layer("avg_pool")[1]]
            component.current_size = current_size
            #shortcut += [component.parse_layer("avg_pool")[1]]
            #shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0")[1]]
            self.conv_id = component.parse_layer("conv " + str(output_channels - current_size.channels) + " filter=3 stride=1")[1]
            self.avg_pool = component.parse_layer("avg_pool")[1]
            component.current_size = LayerShape(output_channels, component.current_size.height, component.current_size.width)

        if self.block == "down_cheap":
            current_size = component.current_size
            layers += [nn.ReLU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            layers += [component.parse_layer("avg_pool")[1]]
            component.current_size = current_size
            shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0")[1]]
            shortcut += [component.parse_layer("avg_pool")[1]]

        if self.block == "up":
            current_size = component.current_size
            upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
            component.current_size = upsample.output_size()
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [upsample]
            layers += [component.parse_layer("conv " + str(component.current_size.channels) + " filter=3")[1]]
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            component.current_size = current_size
            upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
            component.current_size = upsample.output_size()
            shortcut += [upsample]
            shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0")[1]]

        if self.block == "up_subpixel":
            current_size = component.current_size
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("conv " + str(output_channels*4) + " filter=3")[1]]
            component.current_size = LayerShape(output_channels, component.current_size.height * 2, component.current_size.width * 2)
            upsample = nn.PixelShuffle(2)
            component.current_size = current_size
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            upsample = nn.PixelShuffle(2)
            layers += [upsample]
            shortcut += [component.parse_layer("conv " + str(output_channels*4) + " filter=1 padding=0")[1]]
            shortcut += [upsample]
            component.current_size = LayerShape(output_channels, component.current_size.height * 2, component.current_size.width * 2)

        if self.block == "up_deconv":
            current_size = component.current_size
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("deconv " + str(output_channels) + " padding=0 filter=3")[1]]
            component.current_size = LayerShape(output_channels, component.current_size.height * 2, component.current_size.width * 2)
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3 padding=1")[1]]
            component.current_size = current_size
            shortcut += [component.parse_layer("deconv " + str(output_channels) + " padding=0 filter=3")[1]]
            component.current_size = LayerShape(output_channels, component.current_size.height * 2, component.current_size.width * 2)


        if self.block == "up_cheap":
            current_size = component.current_size
            upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
            component.current_size = upsample.output_size()
            layers += self.norm_layers(component)
            layers += [nn.SELU()]
            layers += [upsample]
            layers += [component.parse_layer("conv " + str(output_channels) + " filter=3")[1]]
            component.current_size = current_size
            upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
            component.current_size = upsample.output_size()
            shortcut += [upsample]
            shortcut += [component.parse_layer("conv " + str(output_channels) + " filter=1 padding=0")[1]]


        self.size = component.current_size
        self.layer_names = [None for x in layers]
        self.shortcut = nn.ModuleList(shortcut)
        self.shortcut_layer_names = [None for x in shortcut]
        self.layers = nn.ModuleList(layers)

    def output_size(self):
        return self.size

    def forward(self, input, context):
        if self.block == "down2" or self.block == "down3":
            avg_pool = self.avg_pool(input)
            shortcut = torch.cat([avg_pool, self.avg_pool(self.conv_id(input))], dim=1)
            rhs = self.forward_module_list(input, self.layer_names, self.layers, context)
            return shortcut + self.scalar * rhs
        return self.forward_module_list(input, self.shortcut_layer_names, self.shortcut, context) + self.scalar * self.forward_module_list(input, self.layer_names, self.layers, context)



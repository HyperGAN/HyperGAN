import torch.nn as nn
import hyperchamber as hc
import hypergan as hg
from hypergan.layer_shape import LayerShape

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
          "residual COUNT block=BLOCK_TYPE output_channels=OUTPUT_CHANNELS"
        ```

        ## examples

        ```json
          "residual 3"
        ```
    """


    def __init__(self, component, args, options):
        super(Residual, self).__init__(component, args, options)
        self.style = options.style
        self.block = options.block or "default"
        layers = []
        shortcut = []
        output_channels = options.output_channels or component.current_size.channels
        dims = list(component.current_size.dims)
        dims[0] = output_channels
        if self.block == "down":
            dims[1] = dims[1] // 2
            dims[2] = dims[2] // 2
        if self.block == "default":
            current_size = component.current_size
            for i in range(args[0] or 3):
                layers += [component.parse_layer("conv " + str(component.current_size.channels) + " initializer=xavier_normal filter=3")[1]]
                layers += self.norm_layers(component)
                layers += [nn.SELU()]
                layers += [component.parse_layer("conv " + str(output_channels) + " initializer=xavier_normal filter=3")[1]]
            component.current_size = current_size
            shortcut += [component.parse_layer("conv " + str(output_channels) + " initializer=xavier_normal filter=1 padding=0")[1]]
        if self.block == "down":
            current_size = component.current_size
            for i in range(args[0] or 3):
                layers += [nn.ReLU()]
                layers += [component.parse_layer("conv " + str(output_channels) + " initializer=xavier_normal filter=3")[1]]
                layers += [nn.ReLU()]
                layers += [component.parse_layer("conv " + str(output_channels) + " initializer=xavier_normal filter=3")[1]]
                layers += [component.parse_layer("avg_pool")[1]]
            component.current_size = current_size
            shortcut += [component.parse_layer("conv " + str(output_channels) + " initializer=orthogonal filter=1 padding=0")[1]]
            shortcut += [component.parse_layer("avg_pool")[1]]
        if self.block == "up":
            current_size = component.current_size
            for i in range(args[0] or 3):
                upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
                component.current_size = upsample.output_size()
                layers += self.norm_layers(component)
                layers += [nn.SELU()]
                layers += [upsample]
                layers += [component.parse_layer("conv " + str(component.current_size.channels) + " initializer=xavier_normal filter=3")[1]]
                layers += self.norm_layers(component)
                layers += [nn.SELU()]
                layers += [component.parse_layer("conv " + str(output_channels) + " initializer=xavier_normal filter=3")[1]]
            component.current_size = current_size
            upsample = hg.layers.Upsample(component, [], hc.Config({"w":component.current_size.width*2, "h": component.current_size.height*2}))
            component.current_size = upsample.output_size()
            shortcut += [upsample]
            shortcut += [component.parse_layer("conv " + str(output_channels) + " initializer=orthogonal filter=1 padding=0")[1]]

        self.size = component.current_size
        self.layer_names = [None for x in layers]
        self.shortcut = nn.ModuleList(shortcut)
        self.shortcut_layer_names = [None for x in shortcut]
        self.layers = nn.ModuleList(layers)

    def output_size(self):
        return self.size

    def forward(self, input, context):
        return self.forward_module_list(input, self.shortcut_layer_names, self.shortcut, context) + self.forward_module_list(input, self.layer_names, self.layers, context)


